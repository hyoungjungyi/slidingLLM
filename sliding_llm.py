"""SlidingLLM — our ViT sliding-window SVD compression method, ported to LLMs.

Port of `svd_compression_test.py` (DeiT/Swin) to decoder-only LLMs. Same three
stages, same losses, same dual-ascent size penalty; only the plumbing changes so
that a 7B model fits: decoder blocks live on CPU and are streamed to the GPU a
window at a time, with the activations at each window boundary cached on CPU.

  Stage 0  whitened SVD init   Per-linear input covariance from a wikitext-2
                               calibration set -> GPTQ-damped whitening R (via
                               Cholesky) -> SVD of W R -> W_u diag(mask) W_v.
                               `min(in,out)` singular values are kept live and
                               gated; nothing is truncated yet.

  Stage 1  global rank search  Dobi-SVD-style continuous rank: one scalar `k`
                               per linear behind a tanh gate, trained end-to-end
                               against the dense model's final hidden states
                               plus lambda * |param_ratio - target| with dual
                               ascent on lambda. Produces the per-layer rank
                               initialisation for Stage 2.

  Stage 2  sliding-window rank Per-singular-value sigmoid gates Z_i, searched
           allocation          window-by-window against the dense model's
                               activations at the window exit, with a *local*
                               size penalty whose target is the Stage-1
                               allocation for that window. Windows are visited
                               in the staggered even/odd schedule (every other
                               window per epoch, alternating parity).

  Stage 3  sliding-window      Ranks are hard-sliced into a factored
           fine-tuning         `fc_v -> fc_u` pair, then W_u/W_v are fine-tuned
                               window-by-window on the same feature-MSE.

Example
-------
  python sliding_llm.py --ratio 0.8 --out_json runs/ours_r0.8.json
"""
import argparse
import gc
import json
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import data_utils
import rotation
from llm_utils import (HiddenCache, Timer, build_decoder_kwargs, evaluate_ppl,
                       get_embed_tokens, get_final_norm, get_layers, get_module,
                       layer_linear_paths, layer_rank_string, load_llm,
                       param_report, run_layer, save_result, seed_everything,
                       set_module)


# ==========================================================================
# 1. Gated low-rank linear
# ==========================================================================
class SVDGateLinear(nn.Module):
    """W ~= W_u diag(mask) W_v with a *learnable* rank gate.

    Two gate parameterisations, matching the two rank-search stages of the ViT
    script (`DobiSVDLinear` and `MaskedSVDLinear` respectively):

      mode "k"  a single continuous rank `k`, gate = 0.5*tanh(beta*(k-i))+0.5
      mode "z"  one logit per singular value, gate = sigmoid(Z_i / temperature)

    Both share the same W_u / W_v, so switching from Stage 1 to Stage 2 is a
    re-parameterisation of the gate only — the (expensive) whitened SVD is done
    once, in Stage 0.
    """

    def __init__(self, W_u, W_v, bias, init_k, beta=0.1, temperature=10.0, z_slope=0.1):
        super().__init__()
        # W_u: (out, r_max)   W_v: (r_max, in)
        self.W_u = nn.Parameter(W_u, requires_grad=False)
        self.W_v = nn.Parameter(W_v, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.out_features, self.r_max = W_u.shape
        self.in_features = W_v.shape[1]
        self.beta = beta
        self.temperature = temperature
        self.z_slope = z_slope
        self.mode = "k"

        self.register_buffer("idx", torch.arange(self.r_max, dtype=torch.float32))
        self.k = nn.Parameter(torch.tensor(float(init_k)), requires_grad=False)
        self.Z = None

    # -- gate ---------------------------------------------------------------
    def get_mask(self):
        if self.mode == "k":
            return 0.5 * torch.tanh(self.beta * (self.k - self.idx)) + 0.5
        return torch.sigmoid(self.Z / self.temperature)

    def hard_rank(self):
        if self.mode == "k":
            r = int(round(float(self.k.item())))
        else:
            r = int((self.Z.detach() > 0).sum().item())
        return max(1, min(r, self.r_max))

    def active_indices(self):
        if self.mode == "k":
            return torch.arange(self.hard_rank(), device=self.W_u.device)
        idx = torch.nonzero(self.Z.detach() > 0).squeeze(-1)
        if idx.numel() == 0:
            idx = torch.zeros(1, dtype=torch.long, device=self.W_u.device)
        return idx

    def to_z_mode(self, K=None):
        """Switch to per-singular-value gates, seeded from the Stage-1 rank K.

        Z_i = (K - i) * z_slope reproduces the ViT initialisation: a smooth
        boundary at K rather than a hard step, so Stage 2 can move it either way.
        """
        if K is None:
            K = self.hard_rank()
        z = (float(K) - self.idx) * self.z_slope
        self.Z = nn.Parameter(z.detach().clone())
        self.k.requires_grad_(False)
        self.mode = "z"

    def rank_params(self):
        return [self.k] if self.mode == "k" else [self.Z]

    # -- cost ---------------------------------------------------------------
    def dense_size(self):
        return self.in_features * self.out_features

    def soft_size(self):
        """Differentiable parameter count of the factored layer."""
        return self.get_mask().sum() * (self.in_features + self.out_features)

    def hard_size(self):
        return self.hard_rank() * (self.in_features + self.out_features)

    # -- forward ------------------------------------------------------------
    def forward(self, x):
        h = F.linear(x, self.W_v)
        mask = self.get_mask()
        if mask.requires_grad:
            # fp32 for the gate product so the gradient w.r.t. the gate is
            # accumulated at full precision.
            h = (h.float() * mask).to(self.W_u.dtype)
        else:
            h = h * mask.to(h.dtype)
        return F.linear(h, self.W_u, self.bias)


def iter_gates(model_or_layers):
    layers = model_or_layers if isinstance(model_or_layers, (list, nn.ModuleList)) \
        else get_layers(model_or_layers)
    for layer in layers:
        for m in layer.modules():
            if isinstance(m, SVDGateLinear):
                yield m


# ==========================================================================
# 2. Stage 0 — whitened SVD initialisation
# ==========================================================================
def _damped_cholesky(M):
    """Lower-triangular R with R R^T ~= M, SVD-LLM style microscopic shift.

    Any R satisfying R R^T = M gives the same optimal low-rank factorisation
    (the objective ||(W - W_hat) X||_F only sees M = X^T X), so Cholesky is used
    here instead of the eigendecomposition of the ViT script — same result,
    orders of magnitude faster at in_features = 11008.
    """
    d = M.shape[0]
    M = 0.5 * (M + M.t())
    try:
        return torch.linalg.cholesky(M)
    except Exception:
        eig = torch.linalg.eigvalsh(M)
        M = M + (-eig[0] + 1e-6) * torch.eye(d, dtype=M.dtype, device=M.device)
        return torch.linalg.cholesky(M)


@torch.no_grad()
def build_svd_student(model, calib_ids, device, batch_size=1,
                      init_ratio=0.5, beta=0.1, temperature=10.0, z_slope=0.1):
    """Replace every target Linear with an SVDGateLinear, block by block.

    A single dense sequential pass over the calibration set both (a) collects the
    per-linear input covariance and (b) produces the input activations for the
    next block, so the dense model is never fully resident on the GPU.
    """
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    seqlen = calib_ids.shape[1]

    kwargs_cache = {}

    def kwargs_fn(bs):
        if bs not in kwargs_cache:
            kwargs_cache[bs] = build_decoder_kwargs(seqlen, dtype, device, bs)
        return kwargs_cache[bs]

    print(f"[stage0] embedding {calib_ids.shape[0]} calibration sequences")
    cache = HiddenCache.from_embeddings(model, calib_ids, device)

    for li, layer in enumerate(tqdm(layers, desc="[stage0] whitened SVD")):
        layer.to(device)
        paths = layer_linear_paths(layer)

        cov = {}
        ntok = {}
        handles = []
        for p in paths:
            fc = get_module(layer, p)
            cov[p] = torch.zeros(fc.in_features, fc.in_features,
                                 dtype=torch.float32, device=device)
            ntok[p] = 0

        def make_hook(p):
            def hook(module, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                cov[p] += x.t() @ x
                ntok[p] += x.shape[0]
            return hook

        for p in paths:
            handles.append(get_module(layer, p).register_forward_pre_hook(make_hook(p)))

        outs = torch.empty_like(cache.data)
        for i in range(0, len(cache), batch_size):
            x = cache.data[i:i + batch_size].to(device)
            am, pid = kwargs_fn(x.shape[0])
            outs[i:i + batch_size] = run_layer(layer, x, am, pid).cpu()

        for h in handles:
            h.remove()

        for p in paths:
            fc = get_module(layer, p)
            W = fc.weight.data.float()
            bias = fc.bias.data.clone() if fc.bias is not None else None

            M = (cov[p] / max(1, ntok[p])).double()
            R = _damped_cholesky(M).float()
            del M
            cov[p] = None

            W_tilde = W @ R
            U, S, Vt = torch.linalg.svd(W_tilde, full_matrices=False)
            sqrt_s = S.clamp_min(0).sqrt()
            W_u = U * sqrt_s                                      # (out, r)
            # W_v = sqrt(S) Vt R^{-1}, solved instead of inverting R.
            Vt_un = torch.linalg.solve_triangular(R, Vt, upper=False, left=False)
            W_v = sqrt_s.unsqueeze(1) * Vt_un                     # (r, in)

            K = max(1, round(init_ratio * fc.in_features * fc.out_features
                             / (fc.in_features + fc.out_features)))
            new = SVDGateLinear(W_u.to(dtype), W_v.to(dtype),
                                None if bias is None else bias.to(dtype),
                                init_k=K, beta=beta, temperature=temperature,
                                z_slope=z_slope).to(device)
            set_module(layer, p, new)
            del W, W_tilde, U, S, Vt, sqrt_s, W_u, W_v, Vt_un, R
            torch.cuda.empty_cache()

        layer.to("cpu")
        cache = HiddenCache(outs)
        del cov, outs
        gc.collect()
        torch.cuda.empty_cache()

    return model


# ==========================================================================
# 3. Losses / size bookkeeping
# ==========================================================================
def feature_mse(x, target, rel_mse=True, eps=1e-8):
    """Feature reconstruction loss at a window exit.

    `rel_mse` divides by the teacher's feature energy. On LLMs this matters far
    more than on ViTs: activation magnitudes grow by orders of magnitude with
    depth (massive activations), so a plain MSE would make the deep windows
    dominate the size penalty and the shallow ones ignore it entirely.
    """
    x = x.float()
    target = target.float()
    mse = F.mse_loss(x, target)
    if rel_mse:
        return mse / (target.pow(2).mean() + eps)
    return mse


def soft_size_ratio(gates):
    cur = 0.0
    dense = 0.0
    for m in gates:
        cur = cur + m.soft_size()
        dense += m.dense_size()
    return cur / dense


def hard_size_ratio(gates):
    cur = sum(m.hard_size() for m in gates)
    dense = sum(m.dense_size() for m in gates)
    return cur / dense


# ==========================================================================
# 4. Stage 1 — global rank search
# ==========================================================================
@torch.no_grad()
def teacher_final_hidden(teacher, calib_ids, device, batch_size=1):
    """Dense model's final hidden states (post-norm), streamed layer by layer."""
    dtype = next(teacher.parameters()).dtype
    seqlen = calib_ids.shape[1]
    kwargs_cache = {}

    def kwargs_fn(bs):
        if bs not in kwargs_cache:
            kwargs_cache[bs] = build_decoder_kwargs(seqlen, dtype, device, bs)
        return kwargs_cache[bs]

    cache = HiddenCache.from_embeddings(teacher, calib_ids, device)
    for layer in tqdm(get_layers(teacher), desc="[stage1] teacher hidden", leave=False):
        layer.to(device)
        cache = cache.advance(layer, device, batch_size, kwargs_fn)
        layer.to("cpu")
    norm = get_final_norm(teacher)
    if norm is not None:
        norm.to(device)
        out = torch.empty_like(cache.data)
        for i in range(0, len(cache), batch_size):
            out[i:i + batch_size] = norm(cache.data[i:i + batch_size].to(device)).cpu()
        norm.to("cpu")
        cache = HiddenCache(out)
    return cache


def global_rank_search(student, calib_ids, target_hidden, device, ratio,
                       epochs=5, lr=5.0, lambda_rank=1.0, lambda_lr=10.0,
                       batch_size=1, rel_mse=True, grad_ckpt=True):
    """Stage 1: train the continuous rank `k` of every linear, end to end."""
    print(f"\n[stage1] global rank search -> target ratio {ratio}")
    layers = get_layers(student)
    gates = list(iter_gates(student))
    dtype = next(student.parameters()).dtype
    seqlen = calib_ids.shape[1]

    student.to(device)
    embed = get_embed_tokens(student)
    norm = get_final_norm(student)

    for p in student.parameters():
        p.requires_grad_(False)
    params = []
    for m in gates:
        m.k.requires_grad_(True)
        params.append(m.k)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    lam = float(lambda_rank)
    am, pid = build_decoder_kwargs(seqlen, dtype, device, batch_size)

    n = calib_ids.shape[0]
    for epoch in range(epochs):
        tot_loss = tot_mse = tot_size = 0.0
        nb = 0
        pbar = tqdm(range(0, n, batch_size), desc=f"[stage1] epoch {epoch+1}/{epochs}")
        for i in pbar:
            ids = calib_ids[i:i + batch_size].to(device)
            tgt = target_hidden.data[i:i + batch_size].to(device)
            bs = ids.shape[0]
            a, p_ = (am, pid) if bs == batch_size else build_decoder_kwargs(seqlen, dtype, device, bs)

            h = embed(ids)
            for layer in layers:
                if grad_ckpt:
                    h = torch.utils.checkpoint.checkpoint(
                        run_layer, layer, h, a, p_, use_reentrant=False)
                else:
                    h = run_layer(layer, h, a, p_)
            if norm is not None:
                h = norm(h)

            mse = feature_mse(h, tgt, rel_mse)
            size_ratio = soft_size_ratio(gates)
            size_loss = lam * torch.abs(size_ratio - ratio)
            loss = mse + size_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            lam = max(0.0, lam + lambda_lr * (size_ratio.item() - ratio))

            tot_loss += loss.item(); tot_mse += mse.item(); tot_size += size_loss.item(); nb += 1
            pbar.set_postfix(mse=f"{mse.item():.4f}", ratio=f"{size_ratio.item():.3f}", lam=f"{lam:.1f}")

        nb = max(1, nb)
        print(f"  epoch {epoch+1}: loss {tot_loss/nb:.6f} (mse {tot_mse/nb:.6f}, "
              f"size {tot_size/nb:.6f}, lambda {lam:.2f}) | hard ratio {hard_size_ratio(gates):.4f}")

    for m in gates:
        m.k.requires_grad_(False)
    ranks = {id(m): m.hard_rank() for m in gates}
    student.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()
    return ranks


# ==========================================================================
# 5. Sliding-window driver (shared by Stage 2 and Stage 3)
# ==========================================================================
def make_windows(num_layers, window, stride, progressive=False):
    if num_layers < window:
        raise ValueError(f"model has {num_layers} blocks, window is {window}")
    windows = []
    
    if not progressive:
        return [(i, window) for i in range(0, num_layers - window + 1, stride)]
        
    # 1. Progressive Expansion (Shallow)
    for w in range(1, window + 1):
        windows.append((0, w))
        
    # 2. Sliding (Intermediate)
    start_sliding = window - 1
    end_sliding = num_layers - 2 * window + 1
    
    if end_sliding >= start_sliding:
        sliding_starts = list(range(start_sliding, end_sliding, stride))
        if not sliding_starts or sliding_starts[-1] != end_sliding:
            sliding_starts.append(end_sliding)
            
        for i in sliding_starts:
            windows.append((i, window))
            
    # 3. Progressive Contraction (Deep)
    contract_start = num_layers - window
    for w in range(window, 0, -1):
        windows.append((contract_start, w))
        
    return windows


class WindowRunner:
    """Streams teacher/student activations forward one block at a time.

    Holds two CPU-resident caches — the dense model's activations and the
    student's — and walks them forward together. For a window starting at block
    `i` it yields (student input at i, dense output at i+w-1), which is exactly
    the pair the ViT script obtains with forward hooks on a full model pass.
    """

    def __init__(self, teacher, student, calib_ids, device, batch_size=1):
        self.teacher_layers = get_layers(teacher)
        self.student_layers = get_layers(student)
        self.device = device
        self.batch_size = batch_size
        self.dtype = next(teacher.parameters()).dtype
        self.seqlen = calib_ids.shape[1]
        self.calib_ids = calib_ids
        self.teacher = teacher
        self._kwargs = {}

    def kwargs_fn(self, bs):
        if bs not in self._kwargs:
            self._kwargs[bs] = build_decoder_kwargs(self.seqlen, self.dtype, self.device, bs)
        return self._kwargs[bs]

    def reset(self):
        """Re-seed both caches from the (shared, untouched) token embeddings."""
        emb = HiddenCache.from_embeddings(self.teacher, self.calib_ids, self.device)
        self.t_cache = emb
        self.s_cache = HiddenCache(emb.data.clone())
        self.pos = 0

    def advance_to(self, i):
        while self.pos < i:
            j = self.pos
            self.teacher_layers[j].to(self.device)
            self.t_cache = self.t_cache.advance(self.teacher_layers[j], self.device,
                                                self.batch_size, self.kwargs_fn)
            self.teacher_layers[j].to("cpu")
            self.student_layers[j].to(self.device)
            self.s_cache = self.s_cache.advance(self.student_layers[j], self.device,
                                                self.batch_size, self.kwargs_fn)
            self.student_layers[j].to("cpu")
            self.pos += 1
            torch.cuda.empty_cache()

    def teacher_window_out(self, i, w):
        """Dense activations at the exit of blocks [i, i+w)."""
        cache = self.t_cache
        for j in range(i, i + w):
            self.teacher_layers[j].to(self.device)
            cache = cache.advance(self.teacher_layers[j], self.device,
                                  self.batch_size, self.kwargs_fn,
                                  desc=f"teacher {j}")
            self.teacher_layers[j].to("cpu")
            torch.cuda.empty_cache()
        return cache

    def student_window(self, i, w, to_device=True):
        blocks = [self.student_layers[j] for j in range(i, i + w)]
        for b in blocks:
            b.to(self.device if to_device else "cpu")
        return blocks


# ==========================================================================
# 6. Stage 2 — sliding-window rank allocation
# ==========================================================================
def _save_stage2(path, student, epoch, lambdas):
    torch.save({"epoch": epoch,
                "Z": [m.Z.detach().cpu().clone() for m in iter_gates(student)],
                "lambdas": lambdas}, path + ".tmp")
    os.replace(path + ".tmp", path)          # atomic: a crash mid-write can't corrupt it


def _load_stage2(path, student):
    ck = torch.load(path, map_location="cpu")
    for m, z in zip(iter_gates(student), ck["Z"]):
        m.Z.data.copy_(z.to(m.Z.device))
    return ck["epoch"], ck["lambdas"]


def sliding_window_rank_search(teacher, student, calib_ids, device, ratio,
                               epochs=10, window=4, stride=1, stagger=2,
                               lr=5.0, lambda_rank=1.0, lambda_lr=10.0,
                               batch_size=1, rel_mse=True,
                               target_ratio_override=None, ckpt_path=None):
    print(f"\n[stage2] sliding-window rank search (window={window}, stride={stride}, "
          f"stagger={stagger}, epochs={epochs})")
    student_layers = get_layers(student)
    windows = make_windows(len(student_layers), window, stride)
    runner = WindowRunner(teacher, student, calib_ids, device, batch_size)

    # Each window keeps the parameter budget Stage 1 handed it, unless the caller
    # overrides with a flat target (the no-Stage-1 path).
    window_targets = []
    for (i, w) in windows:
        gates = list(iter_gates(student_layers[i:i + w]))
        window_targets.append(target_ratio_override if target_ratio_override is not None
                              else hard_size_ratio(gates))
    window_lambdas = [float(lambda_rank)] * len(windows)
    print("  per-window target ratios: " +
          ", ".join(f"{t:.3f}" for t in window_targets))

    # Stage 2 is the long pole (hours) and this is a shared GPU box, so an
    # epoch-granular checkpoint turns an OOM from "lose everything" into
    # "lose one epoch".
    start_epoch = 0
    if ckpt_path and os.path.exists(ckpt_path):
        start_epoch, window_lambdas = _load_stage2(ckpt_path, student)
        print(f"  [resume] {ckpt_path}: continuing from epoch {start_epoch + 1}/{epochs} "
              f"| hard ratio {hard_size_ratio(list(iter_gates(student))):.4f}")

    for epoch in range(start_epoch, epochs):
        offset = epoch % max(1, stagger)
        print(f"\n--- [stage2] epoch {epoch+1}/{epochs} (parity {offset}) ---")
        runner.reset()

        for w_idx, (i, w) in enumerate(windows):
            runner.advance_to(i)
            if (w_idx % max(1, stagger)) != offset:
                continue

            target = runner.teacher_window_out(i, w)
            blocks = runner.student_window(i, w)
            gates = list(iter_gates(blocks))
            params = []
            for m in gates:
                for p in m.rank_params():
                    p.requires_grad_(True)
                    params.append(p)

            opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
            lam = window_lambdas[w_idx]
            tgt_ratio = window_targets[w_idx]

            tot = tot_mse = tot_size = 0.0
            nb = 0
            pbar = tqdm(range(0, len(runner.s_cache), batch_size),
                        desc=f"  window {w_idx}/{len(windows)-1} [{i}:{i+w}]", leave=False)
            for b in pbar:
                x = runner.s_cache.data[b:b + batch_size].to(device)
                t = target.data[b:b + batch_size].to(device)
                am, pid = runner.kwargs_fn(x.shape[0])
                for blk in blocks:
                    x = run_layer(blk, x, am, pid)

                mse = feature_mse(x, t, rel_mse)
                size_ratio = soft_size_ratio(gates)
                size_loss = lam * torch.abs(size_ratio - tgt_ratio)
                loss = mse + size_loss

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                lam = max(0.0, lam + lambda_lr * (size_ratio.item() - tgt_ratio))

                tot += loss.item(); tot_mse += mse.item(); tot_size += size_loss.item(); nb += 1
                pbar.set_postfix(mse=f"{mse.item():.4f}", r=f"{size_ratio.item():.3f}")

            window_lambdas[w_idx] = lam
            nb = max(1, nb)
            kept = sum(m.hard_rank() for m in gates)
            full = sum(m.r_max for m in gates)
            print(f"  window {w_idx:>2} [{i}:{i+w}] loss {tot/nb:.6f} "
                  f"(mse {tot_mse/nb:.6f}, size {tot_size/nb:.6f}, lambda {lam:.2f}) | "
                  f"SVs {kept}/{full} | ratio {hard_size_ratio(gates):.4f} (target {tgt_ratio:.4f})")

            for m in gates:
                for p in m.rank_params():
                    p.requires_grad_(False)
            runner.student_window(i, w, to_device=False)
            del target, opt, params
            gc.collect()
            torch.cuda.empty_cache()

        if ckpt_path:
            _save_stage2(ckpt_path, student, epoch + 1, window_lambdas)
            print(f"  [ckpt] epoch {epoch+1}/{epochs} saved -> {ckpt_path}")

    print(f"[stage2] final global hard ratio: {hard_size_ratio(list(iter_gates(student))):.4f}")


# ==========================================================================
# 7. Slicing
# ==========================================================================
@torch.no_grad()
def slice_ranks(student):
    """Hard-truncate each gated layer into a factored `fc_v -> fc_u` pair."""
    print("\n[slice] materialising the learned ranks")
    for layer in tqdm(get_layers(student), desc="[slice]"):
        for path in layer_linear_paths(layer):
            m = get_module(layer, path)
            if not isinstance(m, SVDGateLinear):
                continue
            idx = m.active_indices().to(m.W_u.device)
            W_u = m.W_u.data[:, idx].clone()
            W_v = m.W_v.data[idx, :].clone()
            bias = m.bias.data.clone() if m.bias is not None else None
            r = W_u.shape[1]
            dtype, dev = W_u.dtype, W_u.device

            fc_v = nn.Linear(m.in_features, r, bias=False).to(device=dev, dtype=dtype)
            fc_u = nn.Linear(r, m.out_features, bias=bias is not None).to(device=dev, dtype=dtype)
            fc_v.weight.data.copy_(W_v)
            fc_u.weight.data.copy_(W_u)
            if bias is not None:
                fc_u.bias.data.copy_(bias)
            seq = nn.Sequential(OrderedDict([("fc_v", fc_v), ("fc_u", fc_u)]))
            seq.rank = r
            set_module(layer, path, seq)
            del m
        gc.collect()
    return student


def save_sliced(student, path):
    """Checkpoint the model as it stands right after `slice_ranks`.

    Stages 0-2 cost hours and do not depend on anything Stage 3 varies, so an
    experiment that only changes Stage 3 should pay for them once.
    """
    ranks = {}
    for li, layer in enumerate(get_layers(student)):
        for p in layer_linear_paths(layer):
            m = get_module(layer, p)
            ranks[f"{li}.{p}"] = int(getattr(m, "rank", None) or m[0].out_features)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"ranks": ranks, "state_dict": student.state_dict()}, path)
    n = sum(p.numel() for p in student.parameters())
    print(f"[save] sliced student -> {path} ({n/1e9:.2f}B params, "
          f"{os.path.getsize(path)/1e9:.1f} GB)")


def load_sliced(student, path):
    """Rebuild the factored `fc_v -> fc_u` pairs and load their weights."""
    ck = torch.load(path, map_location="cpu")
    dtype = next(student.parameters()).dtype
    for li, layer in enumerate(get_layers(student)):
        for p in layer_linear_paths(layer):
            fc = get_module(layer, p)
            r = ck["ranks"][f"{li}.{p}"]
            fc_v = nn.Linear(fc.in_features, r, bias=False)
            fc_u = nn.Linear(r, fc.out_features, bias=fc.bias is not None)
            seq = nn.Sequential(OrderedDict([("fc_v", fc_v), ("fc_u", fc_u)])).to(dtype)
            seq.rank = r
            set_module(layer, p, seq)
    student.load_state_dict(ck["state_dict"])
    del ck
    gc.collect()
    return student


def load_ranks_into_gates(student, path):
    """Adopt the per-linear ranks a previous run's result JSON recorded.

    `layer_rank_string` writes them in `layer_linear_paths` order, which is the
    order this walks, so the two stay in step.
    """
    learned = [int(x) for x in json.load(open(path))["layer_ranks"].split(",")]
    n = 0
    for layer in get_layers(student):
        for p in layer_linear_paths(layer):
            get_module(layer, p).k.data.fill_(float(learned[n]))
            n += 1
    if n != len(learned):
        raise ValueError(f"{path} has {len(learned)} ranks but the model has {n} linears")
    return n


# ==========================================================================
# 8. Stage 3 — sliding-window fine-tuning
# ==========================================================================
def build_ft_optimizer(params, lr_mode, lr, rel_lr, weight_decay):
    """AdamW for Stage 3, optionally with a per-tensor scale-relative step.

    Adam's update is ~`lr` per step no matter how big the gradient or the weight
    is, so one global `lr` means very different things to different tensors. The
    whitened SVD makes that gap extreme: `W_u = U sqrt(S)` while
    `W_v = sqrt(S) Vt R^-1`, so on Llama-2-7B layer 0 the two halves of the same
    factored linear differ in RMS by ~150x. At lr=1e-4 that is 16x the weight's
    own size for `down_proj.fc_u` and 0.1x for `down_proj.fc_v` over one window.

    In "relative" mode each tensor gets `lr = rel_lr * rms(W)`, so the *fraction*
    of its own size a tensor can move over a window is `rel_lr * steps` for every
    tensor alike — one interpretable knob instead of a scale lottery.
    """
    if lr_mode != "relative":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    groups = []
    for p in params:
        rms = float(p.detach().float().pow(2).mean().sqrt())
        groups.append({"params": [p], "lr": rel_lr * max(rms, 1e-8)})
    return torch.optim.AdamW(groups, lr=rel_lr, weight_decay=weight_decay)


class LoRALinear(nn.Module):
    """`nn.Linear` with a rank-`r` side path: `y = W x + (alpha/r) B A x`.

    The construction SVD-LLM inherits from PEFT: `A` gets the default
    `nn.Linear` init (kaiming uniform, a=sqrt(5)) and `B` starts at zero, so the
    wrapped module is output-identical to the bare linear at attach time and the
    fine-tuning resumes exactly where the previous stage left off. `merge()`
    folds `B A` back into `W`, which is what keeps the parameter count — and so
    the reported compression ratio — unchanged by the fine-tuning.

    Worth knowing before reading a LoRA arm's numbers: `dL/dA = B^T (dL/dW)` is
    exactly zero while `B` is zero, so the first step moves only `B`, inside the
    random row space `A` happened to be initialised with.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        dev, dt = base.weight.device, base.weight.dtype
        a = torch.empty(r, base.in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        self.lora_A = nn.Parameter(a.to(device=dev, dtype=dt), requires_grad=False)
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, r, device=dev, dtype=dt), requires_grad=False)
        self.base_rms = float(base.weight.detach().float().pow(2).mean().sqrt())
        base.weight.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)

    @torch.no_grad()
    def merge(self):
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        self.base.weight.data += delta.to(self.base.weight.dtype)
        return self.base


def attach_lora(layers, targets, r, alpha):
    """Wrap the `targets` half of every factored linear; returns the adapters."""
    if r <= 0:
        return []
    adapters = []
    for blk in layers:
        for mod in list(blk.modules()):
            for name in targets:
                child = getattr(mod, name, None)
                if isinstance(child, nn.Linear):
                    lin = LoRALinear(child, r, alpha)
                    setattr(mod, name, lin)
                    adapters.append(lin)
    return adapters


def merge_lora(layers):
    """Fold every adapter into its base weight and restore the plain `nn.Linear`."""
    n = 0
    for blk in layers:
        for mod in list(blk.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, LoRALinear):
                    setattr(mod, name, child.merge())
                    n += 1
    return n


def build_lora_optimizer(mods, lr_mode, lr, rel_lr, weight_decay):
    """AdamW for a LoRA arm, step-matched to the direct arm.

    `delta W = (alpha/r) B A` with `B` starting at zero, so an Adam step of size
    `s` in `B` moves `delta W` by about `(alpha/r) sqrt(r) s rms(A)` per entry.
    Setting that equal to the direct arm's `rel_lr * rms(W)` gives the lr below,
    so "each weight moves rel_lr * steps of its own RMS per window" means the
    same thing in both arms and they stay comparable at equal `--ft_rel_lr`.
    """
    params = [p for m in mods for p in (m.lora_A, m.lora_B)]
    if lr_mode != "relative":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    groups = []
    for m in mods:
        rms_a = float(m.lora_A.detach().float().pow(2).mean().sqrt())
        step = rel_lr * m.base_rms / max(m.scaling * math.sqrt(m.r) * rms_a, 1e-12)
        groups.append({"params": [m.lora_A, m.lora_B], "lr": step})
    return torch.optim.AdamW(groups, lr=rel_lr, weight_decay=weight_decay)


def stage3_schedule(num_layers, window, stride, progressive=False,
                    inner=False, innerpercent=False, skip_ends=0, edge="skip"):
    """The Stage-3 task list as `(start, width, mode)`, in forward order.

    `mode` decides where the *input* comes from:

      "sliding"     input = the student's own accumulated activation at `start`,
                    so the window has to absorb all upstream drift as well as
                    approximate its own blocks. This is the error-correcting
                    objective the sliding window exists for.

      "individual"  input = the *teacher's* activation at `start` — a pure
                    per-block reconstruction. No accumulated drift enters, so
                    the fit is well conditioned even where the block performs a
                    near-exact cancellation (blocks 29-31 on Llama-2-7B, where
                    the residual-stream massive activations are unwound and a
                    relative error in the block's output is amplified ~26x).

    `--stage3inner` / `--stage3innerpercent` carve out an inner band of blocks
    for the sliding windows; `edge` says what happens to the blocks outside it.
    """
    windows = make_windows(num_layers, window, stride, progressive=progressive)
    if skip_ends > 0:
        lo, hi = skip_ends, num_layers - skip_ends
    elif inner:
        lo, hi = 4, num_layers - 4
    elif innerpercent:
        lo, hi = int(num_layers * 0.20), num_layers - int(num_layers * 0.15)
    else:
        return [(i, w, "sliding") for i, w in windows]

    tasks = [(i, w, "sliding") for i, w in windows if i >= lo and (i + w - 1) < hi]
    if edge == "individual":
        tasks += [(b, 1, "individual")
                  for b in list(range(lo)) + list(range(hi, num_layers))]
    return sorted(tasks, key=lambda t: (t[0], t[1]))


def sliding_window_finetune(teacher, student, calib_ids, device, epochs=2,
                            window=4, stride=1, lr=1e-4, batch_size=1, rel_mse=True, weight_decay=0.01,
                            stage3inner=False, stage3innerpercent=False, stage3progressive=False,
                            skip_ends=0, edge="skip", edge_lr=None, clip_grad=0.0,
                            lr_mode="const", rel_lr=3e-5,
                            uv_mode="joint", lora_r=0, lora_alpha=16.0):
    if epochs <= 0:
        print("\n[stage3] skipped (epochs=0)")
        return 0.0
    print(f"\n[stage3] sliding-window fine-tuning (window={window}, stride={stride}, "
          f"epochs/window={epochs})")

    student_layers = get_layers(student)
    dtype = next(student.parameters()).dtype
    seqlen = calib_ids.shape[1]
    tasks = stage3_schedule(len(student_layers), window, stride,
                            progressive=stage3progressive, inner=stage3inner,
                            innerpercent=stage3innerpercent, skip_ends=skip_ends,
                            edge=edge)
    n_ind = sum(1 for _, _, m in tasks if m == "individual")
    touched = sorted({b for i, w, _ in tasks for b in range(i, i + w)})
    print(f"[stage3] {len(tasks)} tasks ({len(tasks) - n_ind} sliding, {n_ind} individual)"
          f" | blocks {touched[0]}-{touched[-1]}"
          f" | untouched {sorted(set(range(len(student_layers))) - set(touched))}")

    _kw: dict = {}

    def kwargs_fn(bs: int):
        if bs not in _kw:
            _kw[bs] = build_decoder_kwargs(seqlen, dtype, device, bs)
        return _kw[bs]

    # --- one pass per trainable half of the factored linears.
    #
    # "joint" is the original behaviour: fc_u and fc_v move together.
    #
    # "sequential" fits fc_u with fc_v frozen, then fc_v with the updated fc_u
    # frozen. The product W_u W_v carries an exact gauge freedom — (W_u M,
    # M^-1 W_v) is the same function for any invertible r x r M — so joint
    # training spends part of its step budget on ~r^2 directions per linear that
    # cannot change the loss at all. Freezing one side removes that subspace
    # outright, and with one side fixed a single-block window becomes a linear
    # least-squares problem rather than a bilinear one. SVD-LLM splits the same
    # way (U pass, merge, then V pass), there with LoRA and a global LM loss
    # instead of a window-local MSE.
    #
    # The cost is one extra teacher pass: the rolling exit buffer below is
    # consumed as the window advances, so a second sweep has to recompute it.
    phases = [("fc_u",), ("fc_v",)] if uv_mode == "sequential" else [("fc_v", "fc_u")]
    edge_lr = lr if edge_lr is None else edge_lr
    pass_avgs = []

    for ph, targets in enumerate(phases):
        if len(phases) > 1:
            frozen = "+".join(t for t in ("fc_v", "fc_u") if t not in targets)
            print(f"\n[stage3] ===== pass {ph + 1}/{len(phases)}: training "
                  f"{'+'.join(targets)} ({frozen} frozen) =====")

        adapters = attach_lora(student_layers, targets, lora_r, lora_alpha)
        if adapters:
            n_p = sum(a.lora_A.numel() + a.lora_B.numel() for a in adapters)
            n_b = sum(a.base.weight.numel() for a in adapters)
            print(f"[stage3] {len(adapters)} LoRA adapters | r={lora_r} alpha={lora_alpha} "
                  f"| {n_p / 1e6:.1f}M trainable = {100 * n_p / n_b:.2f}% of the "
                  f"matrices they sit on | merged back at the end of the pass")

        # --- teacher activations: computed once each, kept only while needed.
        # teacher_exits[j] = teacher hidden state after layer j (before layer j+1).
        # For window [i, i+w), the target is teacher_exits[i+w-1].
        # Teacher layers still run exactly once per pass (`tasks` is
        # non-decreasing in the start block, so the teacher only ever moves
        # forward), but holding all 32 exits at once costs
        # sw_nsamples*seqlen*hidden*2 bytes each — 132 GiB of host RAM at
        # sw_nsamples=256, which does not fit beside a second run on a shared
        # box. A task at block `i` never looks further back than exit i-1, so
        # everything below that is dropped as the window advances: peak RAM is
        # O(window) caches instead of O(num_layers).
        teacher_layers = get_layers(teacher)
        teacher_embed = HiddenCache.from_embeddings(teacher, calib_ids, device)
        teacher_exits: dict = {}
        t_state = {"pos": 0, "cache": teacher_embed}   # exits computed for blocks < pos

        def teacher_exit(j):
            """Teacher activation *leaving* block j, computing it if not cached."""
            while t_state["pos"] <= j:
                tlayer = teacher_layers[t_state["pos"]]
                tlayer.to(device)
                with torch.no_grad():
                    t_state["cache"] = t_state["cache"].advance(
                        tlayer, device, batch_size, kwargs_fn,
                        desc=f"  teacher {t_state['pos']}")
                tlayer.to("cpu")
                teacher_exits[t_state["pos"]] = t_state["cache"]
                t_state["pos"] += 1
                torch.cuda.empty_cache()
            return teacher_exits[j]

        def release_teacher_exits(keep_from):
            """Drop cached exits below `keep_from` — no later task asks for them."""
            for k in [k for k in teacher_exits if k < keep_from]:
                del teacher_exits[k]
            gc.collect()

        def teacher_in(j):
            """Teacher activation *entering* block j — the teacher-forced input."""
            return teacher_embed if j == 0 else teacher_exit(j - 1)

        # --- fine-tune the student window by window.
        # s_cache is advanced incrementally using already-fine-tuned student
        # layers, so later windows see corrected inputs from earlier blocks. It
        # is rebuilt per pass, so pass 2 starts from everything pass 1 changed.
        s_cache = HiddenCache.from_embeddings(student, calib_ids, device)
        s_pos = 0

        losses = []
        for w_idx, (i, w, mode) in enumerate(tasks):
            # Advance student cache to the start of this window.
            while s_pos < i:
                student_layers[s_pos].to(device)
                with torch.no_grad():
                    s_cache = s_cache.advance(student_layers[s_pos], device,
                                              batch_size, kwargs_fn)
                student_layers[s_pos].to("cpu")
                s_pos += 1
                torch.cuda.empty_cache()

            release_teacher_exits(i - 1)
            target = teacher_exit(i + w - 1)
            # "individual" feeds the teacher's own activation in, so the block is
            # fit locally with no upstream drift to absorb; "sliding" feeds the
            # student's accumulated activation, which is what makes the window
            # error-correcting.
            src = teacher_in(i) if mode == "individual" else s_cache
            tag = "block " if mode == "individual" else "window"

            blocks = [student_layers[j] for j in range(i, i + w)]
            for blk in blocks:
                blk.to(device)

            # With adapters attached they are the only trainable tensors, and
            # they only exist on this pass's half, so selecting them is enough.
            if lora_r > 0:
                lmods = [m for blk in blocks for m in blk.modules()
                         if isinstance(m, LoRALinear)]
                params = [p for m in lmods for p in (m.lora_A, m.lora_B)]
            else:
                lmods = []
                params = [p for blk in blocks
                          for name, p in blk.named_parameters()
                          if any(t in name for t in targets)]
            if not params:
                for blk in blocks:
                    blk.to("cpu")
                continue
            for p in params:
                p.requires_grad_(True)
            task_lr = edge_lr if mode == "individual" else lr
            opt = (build_lora_optimizer(lmods, lr_mode, task_lr, rel_lr, weight_decay)
                   if lora_r > 0 else
                   build_ft_optimizer(params, lr_mode, task_lr, rel_lr, weight_decay))

            last = 0.0
            for ep in range(epochs):
                tot = 0.0
                nb = 0
                pbar = tqdm(range(0, len(src), batch_size),
                            desc=f"  {tag} {w_idx}/{len(tasks)-1} [{i}:{i+w}] ep{ep+1}",
                            leave=False)
                for b in pbar:
                    x = src.data[b:b + batch_size].to(device)
                    t = target.data[b:b + batch_size].to(device)
                    am, pid = kwargs_fn(x.shape[0])
                    for blk in blocks:
                        x = run_layer(blk, x, am, pid)
                    loss = feature_mse(x, t, rel_mse)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(params, clip_grad)
                    opt.step()
                    tot += loss.item()
                    nb += 1
                    pbar.set_postfix(mse=f"{loss.item():.5f}")
                last = tot / max(1, nb)
                print(f"  {tag} {w_idx:>2} [{i}:{i+w}] ({mode[:4]}) "
                      f"pass {ph + 1} epoch {ep + 1}/{epochs} | mse {last:.6f}")
            losses.append(last)

            for blk in blocks:
                for p in blk.parameters():
                    p.requires_grad_(False)
                blk.to("cpu")
            del opt, params, lmods
            gc.collect()
            torch.cuda.empty_cache()

        # Fold this pass's adapters into the weights before the next pass, so
        # pass 2 sees pass 1's result as plain weights and the saved model keeps
        # exactly the parameter count Stage 2 sliced it down to.
        if adapters:
            n = merge_lora(student_layers)
            print(f"[stage3] merged {n} LoRA adapters into the base weights")
        del adapters

        pass_avgs.append(sum(losses) / max(1, len(losses)))
        if len(phases) > 1:
            print(f"[stage3] pass {ph + 1} mean window mse {pass_avgs[-1]:.6f}")
        del s_cache, teacher_exits, teacher_embed, t_state
        gc.collect()
        torch.cuda.empty_cache()

    # The last pass's mean is the training loss of the weights we ship.
    return pass_avgs[-1]


# ==========================================================================
# 9. Entry point
# ==========================================================================
def run(args):
    seed_everything(args.seed)
    device = torch.device(args.device)

    print(f"=== SlidingLLM | {args.model_id} | target ratio {args.ratio} ===")
    student, tokenizer = load_llm(args.model_id, args.dtype, args.seqlen)

    # QuaRot-style rotation, before any SVD. Q is kept so the teachers loaded
    # below land in exactly the same basis as the student.
    Q = rotation.apply_rotation(student, args.rotate, args.rotate_seed, device)

    calib_ids = data_utils.get_calib_input_ids(
        tokenizer, nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
        dataset=args.calib_dataset, data_root=args.data_root)
    sw_ids = calib_ids[:args.sw_nsamples]

    timings = {}
    teacher = None

    # --- resume straight from a cached sliced student ---------------------
    if args.load_sliced:
        print(f"\n[load] sliced student from {args.load_sliced} "
              f"— skipping Stage 0, 1, 2 and the slice")
        with Timer() as t:
            load_sliced(student, args.load_sliced)
        timings["load_sliced"] = t.elapsed
        gc.collect(); torch.cuda.empty_cache()
        return _finish(args, student, tokenizer, teacher, Q, sw_ids, timings, device)

    with Timer() as t:
        build_svd_student(student, calib_ids, device,
                          batch_size=args.calib_batch_size,
                          init_ratio=args.ratio, beta=args.beta,
                          temperature=args.temperature, z_slope=args.z_slope)
    timings["stage0_svd"] = t.elapsed

    gates = list(iter_gates(student))
    print(f"[stage0] {len(gates)} gated linears | init hard ratio {hard_size_ratio(gates):.4f}")

    # --- adopt a previous run's rank allocation ---------------------------
    if args.load_ranks:
        n = load_ranks_into_gates(student, args.load_ranks)
        print(f"[ranks] loaded {n} per-linear ranks from {args.load_ranks} "
              f"| hard ratio {hard_size_ratio(gates):.4f}")
        print("[ranks] Stages 1 and 2 skipped — they would overwrite these ranks")
        args.skip_stage1 = args.skip_stage2 = True

    # --- Stage 1 ---------------------------------------------------------
    if not args.skip_stage1 and args.stage1_epochs > 0:
        teacher, _ = load_llm(args.model_id, args.dtype, args.seqlen)
        rotation.apply_rotation(teacher, args.rotate, args.rotate_seed, device, Q=Q)
        with Timer() as t:
            tgt = teacher_final_hidden(teacher, sw_ids, device, args.sw_batch_size)
            del teacher
            gc.collect(); torch.cuda.empty_cache()
            global_rank_search(student, sw_ids, tgt, device, args.ratio,
                               epochs=args.stage1_epochs, lr=args.rank_lr,
                               lambda_rank=args.lambda_rank, lambda_lr=args.lambda_lr,
                               batch_size=args.stage1_batch_size, rel_mse=args.rel_mse,
                               grad_ckpt=not args.no_grad_ckpt)
            del tgt
            gc.collect(); torch.cuda.empty_cache()
        timings["stage1_global_rank"] = t.elapsed
    else:
        print("\n[stage1] skipped — seeding Stage 2 from the uniform rank init")

    # Hand the Stage-1 allocation to Stage 2's per-singular-value gates.
    for m in gates:
        m.to_z_mode()
    print(f"[stage1->2] hard ratio after switching to Z gates: {hard_size_ratio(gates):.4f}")

    teacher, _ = load_llm(args.model_id, args.dtype, args.seqlen)
    rotation.apply_rotation(teacher, args.rotate, args.rotate_seed, device, Q=Q)
    gc.collect(); torch.cuda.empty_cache()

    # --- Stage 2 ---------------------------------------------------------
    if not args.skip_stage2 and args.stage2_epochs > 0:
        with Timer() as t:
            sliding_window_rank_search(
                teacher, student, sw_ids, device, args.ratio,
                epochs=args.stage2_epochs, window=args.window, stride=args.stride,
                stagger=args.stagger, lr=args.rank_lr, lambda_rank=args.lambda_rank,
                lambda_lr=args.lambda_lr, batch_size=args.sw_batch_size,
                rel_mse=args.rel_mse,
                target_ratio_override=args.ratio if args.skip_stage1 else None,
                ckpt_path=args.stage2_ckpt or None)
        timings["stage2_window_rank"] = t.elapsed
    else:
        print("\n[stage2] skipped")

    # --- slice ------------------------------------------------------------
    slice_ranks(student)
    gc.collect(); torch.cuda.empty_cache()
    if args.save_sliced:
        save_sliced(student, args.save_sliced)

    return _finish(args, student, tokenizer, teacher, Q, sw_ids, timings, device)


def _finish(args, student, tokenizer, teacher, Q, sw_ids, timings, device):
    """Stage 3 + evaluation + reporting.

    Split out of `run()` so `--load_sliced` can jump straight here: Stages 0-2
    take hours and nothing Stage 3 varies feeds back into them.
    """
    avg_mse = 0.0
    if not args.skip_stage3 and args.stage3_epochs > 0:
        if teacher is None:
            teacher, _ = load_llm(args.model_id, args.dtype, args.seqlen)
            rotation.apply_rotation(teacher, args.rotate, args.rotate_seed, device, Q=Q)
        with Timer() as t:
            wd = 0.0 if args.disable_ft_weight_decay else 0.01
            avg_mse = sliding_window_finetune(
                teacher, student, sw_ids, device, epochs=args.stage3_epochs,
                window=args.window, stride=args.stride, lr=args.ft_lr,
                batch_size=args.sw_batch_size, rel_mse=args.rel_mse,
                weight_decay=wd, stage3inner=args.stage3inner,
                stage3innerpercent=args.stage3innerpercent,
                stage3progressive=args.stage3progressive,
                skip_ends=args.stage3_skip_ends,
                edge=args.stage3_edge, edge_lr=args.stage3_edge_lr,
                clip_grad=args.ft_clip_grad,
                lr_mode=args.ft_lr_mode, rel_lr=args.ft_rel_lr,
                uv_mode=args.stage3_uv_mode, lora_r=args.stage3_lora_r,
                lora_alpha=args.stage3_lora_alpha)
        timings["stage3_finetune"] = t.elapsed

    del teacher, Q
    gc.collect(); torch.cuda.empty_cache()

    # --- evaluate --------------------------------------------------------
    report = param_report(student)
    ranks = layer_rank_string(student)
    print(f"\n[eval] linear params {report['linear_params']/1e9:.3f}B / "
          f"{report['dense_linear_params']/1e9:.3f}B "
          f"= {report['linear_param_ratio']:.4f} | total {report['total_params']/1e9:.3f}B")

    ppl = {}
    for ds in args.eval_datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        test_ids = data_utils.get_test_input_ids(tokenizer, args.seqlen, ds, args.data_root)
        ppl[ds] = evaluate_ppl(student, test_ids, device, args.eval_batch_size, desc=f"ppl:{ds}")
        print(f"[eval] {ds} perplexity: {ppl[ds]:.4f}")

    payload = {
        "method": "ours",
        "model_id": args.model_id,
        "target_ratio": args.ratio,
        "ppl": ppl,
        "avg_window_mse": avg_mse,
        "timings_sec": timings,
        "config": vars(args),
        "layer_ranks": ranks,
        **report,
    }
    out_json = args.out_json
    if not out_json:
        model_name = args.model_id.split("/")[-1]
        out_json = f"results/{model_name}_r{args.ratio}.json"
        
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    save_result(out_json, payload)
    print(f"\n[save] Result JSON saved to {out_json}")
    if args.save_model:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)) or ".", exist_ok=True)
        torch.save({"model": student.cpu(), "tokenizer": tokenizer}, args.save_model)
        print(f"[save] {args.save_model}")
    return payload


def build_parser():
    p = argparse.ArgumentParser(description="SlidingLLM: sliding-window SVD compression for LLMs")
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--ratio", type=float, default=0.8,
                   help="target fraction of the original linear-layer parameters to KEEP")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    # data
    p.add_argument("--data_root", type=str, default=data_utils.DEFAULT_DATA_ROOT)
    p.add_argument("--calib_dataset", type=str, default="wikitext2")
    p.add_argument("--eval_datasets", type=str, default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128, help="sequences for the whitening covariance")
    p.add_argument("--sw_nsamples", type=int, default=128,
                   help="sequences for stages 1-3 (the first --sw_nsamples of the "
                        "--nsamples calibration set; all three stages see the same ones)")
    p.add_argument("--calib_batch_size", type=int, default=1)
    p.add_argument("--sw_batch_size", type=int, default=1)
    p.add_argument("--stage1_batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)

    # stage schedule
    p.add_argument("--stage1_epochs", type=int, default=5)
    p.add_argument("--stage2_epochs", type=int, default=10)
    p.add_argument("--stage3_epochs", type=int, default=2, help="epochs per window in Stage 3")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--stagger", type=int, default=2,
                   help="Stage-2 staggering: visit every Nth window per epoch, rotating parity")
    p.add_argument("--skip_stage1", action="store_true")
    p.add_argument("--skip_stage2", action="store_true")
    p.add_argument("--skip_stage3", action="store_true")
    p.add_argument("--stage3inner", action="store_true", help="Skip fine-tuning the first 4 and last 4 blocks in Stage 3")
    p.add_argument("--stage3innerpercent", action="store_true", help="Skip fine-tuning the first 20%% and last 15%% of blocks in Stage 3")
    p.add_argument("--stage3progressive", action="store_true", help="Progressively increase and decrease window size at the ends in Stage 3")
    p.add_argument("--stage3_skip_ends", type=int, default=0,
                   help="leave the first N and last N blocks out of the Stage-3 "
                        "sliding windows (1 = just blocks 0 and N-1; 4 reproduces "
                        "--stage3inner). Takes precedence over the two flags above")
    p.add_argument("--stage3_edge", type=str, default="skip", choices=["skip", "individual"],
                   help="What to do with the blocks that --stage3inner/--stage3innerpercent "
                        "leave outside the sliding band: 'skip' them (default), or fine-tune "
                        "each one 'individual'ly, teacher-forced (input = the dense model's "
                        "activation at that boundary, so no upstream drift enters the fit)")
    p.add_argument("--stage3_edge_lr", type=float, default=None,
                   help="learning rate for the individually fine-tuned edge blocks "
                        "(defaults to --ft_lr)")
    p.add_argument("--ft_lr_mode", type=str, default="const",
                   choices=["const", "relative"],
                   help="'const' gives every tensor --ft_lr. 'relative' gives each "
                        "tensor lr = --ft_rel_lr * rms(W), so every tensor moves the "
                        "same *fraction* of its own size per step")
    p.add_argument("--ft_rel_lr", type=float, default=3e-5,
                   help="only with --ft_lr_mode relative. Each weight may move about "
                        "rel_lr * (samples * epochs) times its own RMS per window; "
                        "3e-5 with 128 samples and 2 epochs is ~0.8%% per window")
    p.add_argument("--stage3_uv_mode", type=str, default="joint",
                   choices=["joint", "sequential"],
                   help="'joint' trains fc_u and fc_v together (one sweep). "
                        "'sequential' sweeps twice: fc_u with fc_v frozen, then "
                        "fc_v with the updated fc_u frozen. Removes the "
                        "(W_u M, M^-1 W_v) gauge freedom that joint training "
                        "wastes steps on; costs 2x the Stage-3 time")
    p.add_argument("--stage3_lora_r", type=int, default=0,
                   help="0 (default) updates W_u/W_v directly at full rank. "
                        ">0 attaches a rank-r LoRA adapter to each targeted "
                        "linear instead (B=0, A kaiming — SVD-LLM's setup) and "
                        "merges it back at the end of each pass, so the update "
                        "to each matrix is rank-limited but the parameter count "
                        "is unchanged")
    p.add_argument("--stage3_lora_alpha", type=float, default=16.0,
                   help="LoRA scaling is alpha/r; 16 with r=8 reproduces "
                        "SVD-LLM's 2.0")
    p.add_argument("--ft_clip_grad", type=float, default=0.0,
                   help="max grad-norm for Stage 3; 0 disables. Single-sample loss spikes "
                        "of 1e6 have been observed in the deep windows, so 1.0 is a "
                        "reasonable value")

    # optimisation
    p.add_argument("--rank_lr", type=float, default=5.0)
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--disable_ft_weight_decay", action="store_true", help="Disable AdamW weight decay in Stage 3")
    p.add_argument("--lambda_rank", type=float, default=1.0)
    p.add_argument("--lambda_lr", type=float, default=10.0)
    p.add_argument("--temperature", type=float, default=10.0, help="Stage-2 sigmoid gate temperature")
    p.add_argument("--beta", type=float, default=0.1, help="Stage-1 tanh gate sharpness")
    p.add_argument("--z_slope", type=float, default=0.1, help="Stage-2 Z initialisation slope")
    p.add_argument("--rotate", type=str, default="none", choices=["none", "hadamard", "random"],
                   help="QuaRot-style orthogonal rotation of the residual stream before "
                        "Stage 0. Spreads the massive activations across all channels. "
                        "Provably a no-op for the whitened SVD in exact arithmetic — the "
                        "gain is bf16 numerical conditioning. See rotation.py.")
    p.add_argument("--rotate_seed", type=int, default=0,
                   help="seed for the random Hadamard sign flips / random orthogonal Q")
    p.add_argument("--rel_mse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no_grad_ckpt", action="store_true")

    # checkpointing — Stages 0-2 cost hours and nothing in Stage 3 feeds back
    # into them, so a Stage-3 sweep should pay for them exactly once.
    p.add_argument("--stage2_ckpt", type=str, default="",
                   help="save the Stage-2 gate state after every epoch and resume "
                        "from it if the file already exists — Stage 2 runs for hours "
                        "on a shared box and an OOM otherwise loses all of it")
    p.add_argument("--load_ranks", type=str, default="",
                   help="adopt the per-linear `layer_ranks` recorded in a previous "
                        "run's result JSON and skip Stages 1 and 2 (Stage 0's SVD "
                        "still runs)")
    p.add_argument("--save_sliced", type=str, default="",
                   help="checkpoint the student right after the slice, so later "
                        "Stage-3 variants can start from it")
    p.add_argument("--load_sliced", type=str, default="",
                   help="resume from a --save_sliced checkpoint: skips Stage 0, 1, 2 "
                        "and the slice entirely")

    # output
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--save_model", type=str, default="")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
