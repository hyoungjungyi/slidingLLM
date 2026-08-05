"""QuaRot / SliceGPT-style orthogonal rotation of the residual stream.

The massive-activation problem: a handful of hidden dimensions in LLaMA-2 carry
activations orders of magnitude larger than the rest. QuaRot removes them by
exploiting *computational invariance*: RMSNorm without a learned scale commutes
with any orthogonal Q, so the whole residual stream can be re-expressed in a
rotated basis without changing the function the network computes.

  1. `fuse_layer_norms`  folds each RMSNorm's learned scale into the linears that
     read from it, leaving a bare (rotation-equivariant) RMSNorm.
  2. `rotate_model`      applies Q to the residual basis:
        embed_tokens   W <- W Q        (writes the residual)
        q/k/v/gate/up  W <- W Q        (read the residual)
        o_proj/down    W <- Q^T W      (write the residual, bias too)
        lm_head        W <- W Q        (reads the residual)

After this the outlier energy is spread over all `hidden_size` channels.

IMPORTANT — what this does and does not buy a whitening-based SVD method
-----------------------------------------------------------------------
Stage 0 whitens with the per-linear input covariance M = X^T X before
truncating. That whitening is invariant to *any* invertible reparameterisation
of the input space, so an orthogonal Q is, in exact arithmetic, a no-op for the
resulting factorisation:

  rotated input covariance  M' = Q^T M Q
  damping                   mean(diag(M')) = trace(M)/d  (trace is invariant)
                            and Q^T (M + lam I) Q  = M' + lam I
  optimal rank-k solution   argmin ||(W' - W_hat) X'||_F  =  (argmin for W) Q

The same holds for Stages 1-3: an MSE between two states rotated by the same Q
is unchanged (Frobenius norm is orthogonally invariant), so the loss surface and
the learned ranks are identical too.

What is left is *numerical*, and in bf16 it is not nothing: the un-rotated
covariance has a condition number driven by the massive channels, so the
Cholesky and the triangular solve in Stage 0 lose precision, and the bf16
forward in Stages 2-3 rounds badly on the outlier channels. Rotation fixes both.

Rotation becomes mathematically load-bearing (not just numerically) as soon as
the truncation is *not* whitened — plain SVD, Dobi-SVD's activation-space
truncation — or once per-channel quantisation enters the picture. Use
`--rotate` on `baselines.py --method dobisvd` to measure exactly that.
"""
import math

import torch

from llm_utils import (get_embed_tokens, get_final_norm, get_layers,
                       get_module, layer_linear_paths)

# Linears whose *input* is the residual stream -> W <- W Q
INPUT_SIDE = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "mlp.gate_proj", "mlp.up_proj"]
# Linears whose *output* is written back to the residual stream -> W <- Q^T W
OUTPUT_SIDE = ["self_attn.o_proj", "mlp.down_proj"]

# RMSNorm -> the linears that consume its output.
NORM_CONSUMERS = [
    ("input_layernorm", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
    ("post_attention_layernorm", ["mlp.gate_proj", "mlp.up_proj"]),
]


# --------------------------------------------------------------------------
# orthogonal matrices
# --------------------------------------------------------------------------
def _sylvester_hadamard(n, dtype=torch.float64):
    """Unnormalised Hadamard matrix of order `n`, `n` a power of two."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"Sylvester construction needs a power-of-two order, got {n}")
    H = torch.ones(1, 1, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H


def random_hadamard_matrix(n, generator, dtype=torch.float64):
    """QuaRot's randomised Hadamard: H diag(s) / sqrt(n), s uniform in {-1, +1}."""
    H = _sylvester_hadamard(n, dtype)
    s = torch.randint(0, 2, (n,), generator=generator, dtype=dtype) * 2 - 1
    return (H * s.unsqueeze(0)) / math.sqrt(n)


def random_orthogonal_matrix(n, generator, dtype=torch.float64):
    """Haar-distributed orthogonal matrix (QR of a Gaussian, with sign fix)."""
    A = torch.randn(n, n, generator=generator, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def get_rotation_matrix(mode, n, seed=0, device="cpu"):
    """`mode` in {hadamard, random}. Falls back to `random` when n is not 2^k."""
    g = torch.Generator().manual_seed(int(seed))
    if mode == "hadamard":
        if (n & (n - 1)) != 0:
            print(f"[rotate] hidden_size {n} is not a power of two — no Sylvester "
                  f"Hadamard exists and neither scipy nor fast_hadamard_transform "
                  f"is installed; falling back to a random orthogonal matrix.")
            Q = random_orthogonal_matrix(n, g)
        else:
            Q = random_hadamard_matrix(n, g)
    elif mode == "random":
        Q = random_orthogonal_matrix(n, g)
    else:
        raise ValueError(f"unknown rotation mode: {mode}")
    return Q.to(device)


# --------------------------------------------------------------------------
# step 1: fold the RMSNorm scales into the following linears
# --------------------------------------------------------------------------
@torch.no_grad()
def _fuse_norm_into(norm, linears):
    if getattr(norm, "bias", None) is not None:
        raise ValueError(
            "rotation requires RMSNorm (no bias). LayerNorm's mean subtraction is "
            "not rotation-equivariant and would need the SliceGPT mean-folding.")
    w = norm.weight.data.double()
    for fc in linears:
        fc.weight.data = (fc.weight.data.double() * w.unsqueeze(0)).to(fc.weight.dtype)
    norm.weight.data = torch.ones_like(norm.weight.data)


@torch.no_grad()
def fuse_layer_norms(model):
    """Make every RMSNorm scale-free by pushing its weight into its consumers."""
    if getattr(model.config, "tie_word_embeddings", False):
        # The final norm folds into lm_head only; with tied weights that would
        # corrupt the embedding, so untie first.
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.data.clone())
        model.config.tie_word_embeddings = False
        print("[rotate] untied lm_head from embed_tokens before fusing")

    for layer in get_layers(model):
        for norm_name, consumers in NORM_CONSUMERS:
            norm = getattr(layer, norm_name, None)
            if norm is None:
                continue
            fcs = [get_module(layer, p) for p in consumers
                   if p in layer_linear_paths(layer)]
            _fuse_norm_into(norm, fcs)

    final_norm = get_final_norm(model)
    if final_norm is not None and hasattr(model, "lm_head"):
        _fuse_norm_into(final_norm, [model.lm_head])
    print("[rotate] fused RMSNorm scales into the following linears")


# --------------------------------------------------------------------------
# step 2: rotate the residual basis
# --------------------------------------------------------------------------
@torch.no_grad()
def _right_rotate(fc, Q, device):
    """W <- W Q  (input lives in the residual basis)."""
    W = fc.weight.data
    home, dt = W.device, W.dtype
    fc.weight.data = (W.to(device=device, dtype=torch.float64) @ Q).to(dtype=dt, device=home)


@torch.no_grad()
def _left_rotate(fc, Q, device):
    """W <- Q^T W, b <- Q^T b  (output is written into the residual basis)."""
    W = fc.weight.data
    home, dt = W.device, W.dtype
    fc.weight.data = (Q.t() @ W.to(device=device, dtype=torch.float64)).to(dtype=dt, device=home)
    if fc.bias is not None:
        b = fc.bias.data
        fc.bias.data = (Q.t() @ b.to(device=device, dtype=torch.float64)).to(dtype=b.dtype, device=b.device)


@torch.no_grad()
def rotate_model(model, Q, device):
    from tqdm import tqdm

    embed = get_embed_tokens(model)
    W = embed.weight.data
    home, dt = W.device, W.dtype
    embed.weight.data = (W.to(device=device, dtype=torch.float64) @ Q).to(dtype=dt, device=home)

    for layer in tqdm(get_layers(model), desc="[rotate] rotating blocks"):
        paths = layer_linear_paths(layer)
        for p in INPUT_SIDE:
            if p in paths:
                _right_rotate(get_module(layer, p), Q, device)
        for p in OUTPUT_SIDE:
            if p in paths:
                _left_rotate(get_module(layer, p), Q, device)
        torch.cuda.empty_cache()

    if hasattr(model, "lm_head"):
        _right_rotate(model.lm_head, Q, device)
    torch.cuda.empty_cache()


@torch.no_grad()
def apply_rotation(model, mode, seed, device, Q=None):
    """Fuse the norms and rotate. Returns Q so callers can reuse it verbatim.

    The teacher and the student MUST be rotated with the *same* Q, otherwise the
    feature-MSE in Stages 1-3 compares states in two different bases.
    """
    if mode in (None, "none"):
        return None
    n = model.config.hidden_size
    if Q is None:
        Q = get_rotation_matrix(mode, n, seed=seed, device=device)
    print(f"[rotate] {mode} rotation of the residual stream (hidden_size={n}, seed={seed})")
    fuse_layer_norms(model)
    rotate_model(model, Q, device)
    return Q
