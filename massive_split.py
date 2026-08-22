"""Massive-activation channel split: compress everything *except* the outlier channels.

The idea
--------
A handful of residual-stream channels in Llama-2-7B carry activations two orders
of magnitude larger than the rest (2533 at |h| ~ 2720, 1415 at ~1600, versus a
median of ~10 -- see `figs/acts_base.npz`). Those channels dominate the input
covariance `M = X^T X`, so the whitened SVD of Stage 0 spends essentially all of
its dynamic range reproducing them, and the singular directions that matter for
the other 4,094 channels get truncated first.

So take them out of the factorisation entirely. For a linear `W` (out, in) with
massive input channels `C` and the rest `K`:

    y = W x = W[:, K] x_K  +  W[:, C] x_C
              \__________/    \__________/
               compressed        exact

The split is *algebraically exact* -- no approximation is introduced by it. Only
the `W[:, K]` half is whitened (eigendecomposition of the reduced covariance
`M_KK`), factored by SVD, searched for optimal ranks (Stages 1 & 2), truncated,
and fine-tuned (Stage 3). `W[:, C]` is carried along at full precision as a
frozen `(out, |C|)` slab and is never touched.

Full multi-stage pipeline
-------------------------
  Stage 0  Isolate outlier channels C into W_mass; compute whitened SVD on
           the remaining normal channels K; wrap into `MassiveSplitGateLinear`.
  Stage 1  Global rank search (continuous rank k per linear).
  Stage 2  Sliding-window rank search (per-singular-value sigmoid gates Z_i).
  Slice    Hard-truncate W_u, W_v using exact learned singular-value masks (Z > 0)
           into `MassiveSplitLinear`.
  Stage 3  Sliding-window fine-tuning of the truncated W_u, W_v.

Example
-------
  # Full end-to-end pipeline with auto outlier detection:
  python massive_split.py --ratio 0.6 --device cuda:0 \\
      --mass_scope auto --mass_topk 2 --mass_ratio_thresh 50.0 \\
      --out_json results/r0.6_masssplit_auto.json
"""
import argparse
import gc
import json
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import data_utils
from llm_utils import (HiddenCache, Timer, build_decoder_kwargs, evaluate_ppl,
                       get_layers, get_module, layer_linear_paths, load_llm,
                       param_report, run_layer, save_result, seed_everything,
                       set_module)
from sliding_llm import (SVDGateLinear, global_rank_search, hard_size_ratio,
                         iter_gates, sliding_window_finetune,
                         sliding_window_rank_search, soft_size_ratio,
                         teacher_final_hidden)

# Linears whose input is the residual stream (through an RMSNorm, which is a
# per-channel rescale and so preserves the channel indexing). Only these can be
# given the residual-stream massive-channel indices; `o_proj` reads the
# attention output and `down_proj` the SwiGLU intermediate, both different
# spaces with different outliers -- those are handled by `--mass_topk`.
RESIDUAL_FED = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "mlp.gate_proj", "mlp.up_proj")


# ==========================================================================
# 1. The split linear modules
# ==========================================================================
class MassiveSplitLinear(nn.Module):
    """`fc_v -> fc_u` on the ordinary channels, plus an exact dense slab on `C`.

    The submodules are deliberately named `fc_v` / `fc_u` so that Stage 3's
    parameter filter (`"fc_v" in name or "fc_u" in name`) picks up exactly the
    low-rank half and leaves `W_mass` frozen, with no change to
    `sliding_window_finetune`.
    """

    def __init__(self, W_v, W_u, W_mass, bias, keep_idx, mass_idx, in_features):
        super().__init__()
        r = W_u.shape[1]
        out_features = W_u.shape[0]
        dtype, dev = W_u.dtype, W_u.device

        self.in_features = in_features
        self.out_features = out_features
        self.rank = r
        self.n_mass = 0 if mass_idx is None else int(mass_idx.numel())

        self.fc_v = nn.Linear(W_v.shape[1], r, bias=False).to(device=dev, dtype=dtype)
        self.fc_u = nn.Linear(r, out_features, bias=bias is not None).to(device=dev, dtype=dtype)
        self.fc_v.weight.data.copy_(W_v)
        self.fc_u.weight.data.copy_(W_u)
        if bias is not None:
            self.fc_u.bias.data.copy_(bias)

        if self.n_mass:
            # Frozen and exact: this is the whole point of the method.
            self.W_mass = nn.Parameter(W_mass.to(device=dev, dtype=dtype),
                                       requires_grad=False)
            self.register_buffer("mass_idx", mass_idx.to(dev))
            self.register_buffer("keep_idx", keep_idx.to(dev))
        else:
            self.register_parameter("W_mass", None)
            self.register_buffer("mass_idx", None)
            self.register_buffer("keep_idx", None)   # no gather needed

    def forward(self, x):
        if self.n_mass == 0:
            return self.fc_u(self.fc_v(x))
        y = self.fc_u(self.fc_v(x.index_select(-1, self.keep_idx)))
        return y + F.linear(x.index_select(-1, self.mass_idx), self.W_mass)


class MassiveSplitGateLinear(SVDGateLinear):
    """Gated low-rank SVD on normal channels K plus an exact dense slab on outlier channels C.

    Inherits from SVDGateLinear so all of Stage 1 (global_rank_search) and
    Stage 2 (sliding_window_rank_search) work directly without modifications.
    """

    def __init__(self, W_u, W_v, W_mass, bias, keep_idx, mass_idx, in_features,
                 init_k, beta=0.1, temperature=10.0, z_slope=0.1):
        super().__init__(W_u, W_v, bias, init_k, beta, temperature, z_slope)
        self.in_features = in_features
        self.n_mass = 0 if mass_idx is None else int(mass_idx.numel())

        if self.n_mass:
            self.W_mass = nn.Parameter(W_mass, requires_grad=False)
            self.register_buffer("mass_idx", mass_idx)
            self.register_buffer("keep_idx", keep_idx)
        else:
            self.register_parameter("W_mass", None)
            self.register_buffer("mass_idx", None)
            self.register_buffer("keep_idx", None)

    def dense_size(self):
        return self.in_features * self.out_features

    def soft_size(self):
        normal_in = self.in_features - self.n_mass
        return self.get_mask().sum() * (normal_in + self.out_features) + (self.out_features * self.n_mass)

    def hard_size(self):
        normal_in = self.in_features - self.n_mass
        return self.hard_rank() * (normal_in + self.out_features) + (self.out_features * self.n_mass)

    def forward(self, x):
        mask = self.get_mask()
        if self.n_mass == 0:
            h = F.linear(x, self.W_v)
            if mask.requires_grad:
                h = (h.float() * mask).to(self.W_u.dtype)
            else:
                h = h * mask.to(h.dtype)
            return F.linear(h, self.W_u, self.bias)
        else:
            x_k = x.index_select(-1, self.keep_idx)
            h = F.linear(x_k, self.W_v)
            if mask.requires_grad:
                h = (h.float() * mask).to(self.W_u.dtype)
            else:
                h = h * mask.to(h.dtype)
            y_k = F.linear(h, self.W_u, self.bias)
            x_c = x.index_select(-1, self.mass_idx)
            y_c = F.linear(x_c, self.W_mass)
            return y_k + y_c


def iter_split_linears(model):
    for li, layer in enumerate(get_layers(model)):
        for p in layer_linear_paths(layer):
            m = get_module(layer, p)
            if isinstance(m, MassiveSplitLinear):
                yield li, p, m


# ==========================================================================
# 2. Channel selection
# ==========================================================================
def pick_mass_channels(path, cov_diag, explicit, topk, ratio_thresh, scope):
    """Which input channels of `path` are held out of the compression.

    `cov_diag[c] = sum_t x_tc^2` is the per-channel input energy, which Stage 0
    already computes on the way to the covariance -- detection is free.
    """
    d = cov_diag.shape[0]
    if scope in ("residual", "all") and path in RESIDUAL_FED and explicit:
        return torch.tensor(sorted(c for c in explicit if 0 <= c < d), dtype=torch.long)
    if scope in ("auto", "all") and topk > 0:
        order = torch.argsort(cov_diag, descending=True)
        med = cov_diag.median().clamp_min(1e-30)
        sel = [int(c) for c in order[:topk] if float(cov_diag[c] / med) >= ratio_thresh]
        return torch.tensor(sorted(sel), dtype=torch.long)
    return torch.tensor([], dtype=torch.long)


# ==========================================================================
# 3. Whitening by eigendecomposition
# ==========================================================================
def whiten_eig(M, damp, eig_device, eig_dtype):
    """`R, R^-1` with `R R^T = M`, via `M = Q L Q^T` -> `R = Q L^{1/2}`.

    Any `R` with `R R^T = M` gives the same optimal factorisation (the objective
    `||(W - W_hat) X||_F` only sees `M`), so this is interchangeable with the
    Cholesky in `sliding_llm.build_svd_student`; the eigendecomposition is used
    here because it lets the spectrum be inspected and damped directly, which
    matters once the massive channels -- the things that made `M` ill
    conditioned in the first place -- have been removed.
    """
    dev = M.device if eig_device == "same" else torch.device(eig_device)
    dt = {"float64": torch.float64, "float32": torch.float32}[eig_dtype]
    M = M.to(device=dev, dtype=dt)
    M = 0.5 * (M + M.t())
    evals, evecs = torch.linalg.eigh(M)
    lo = damp * evals.max().clamp_min(1e-30)
    n_damped = int((evals < lo).sum())
    evals = evals.clamp_min(lo)
    sq = evals.sqrt()
    R = evecs * sq                     # Q L^{1/2}
    Rinv = (evecs / sq).t()            # L^{-1/2} Q^T
    cond = float(evals.max() / evals.min())
    return R, Rinv, cond, n_damped


# ==========================================================================
# 4. Build the gated split student (Stage 0)
# ==========================================================================
@torch.no_grad()
def build_split_gated_student(model, calib_ids, device, init_ratio=0.6, batch_size=1,
                              explicit=(), topk=0, ratio_thresh=50.0, scope="residual",
                              damp=1e-6, eig_device="same", eig_dtype="float64",
                              beta=0.1, temperature=10.0, z_slope=0.1, custom_ranks=None):
    """Replace every target Linear with a `MassiveSplitGateLinear`, block by block."""
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

    meta = {}
    gi = 0
    t0 = time.time()
    for li, layer in enumerate(layers):
        layer.to(device)
        paths = layer_linear_paths(layer)

        cov, ntok, handles = {}, {}, []
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
            W = fc.weight.data.float()                      # (out, in)
            bias = fc.bias.data.clone() if fc.bias is not None else None
            d_in, d_out = fc.in_features, fc.out_features

            C = pick_mass_channels(p, torch.diagonal(cov[p]).clone(),
                                   explicit, topk, ratio_thresh, scope).to(device)
            keep = torch.ones(d_in, dtype=torch.bool, device=device)
            keep[C] = False
            K = torch.nonzero(keep).squeeze(-1)

            # Reduced covariance & SVD on normal channels only
            M = (cov[p].index_select(0, K).index_select(1, K) / max(1, ntok[p])).double()
            R, Rinv, cond, n_damped = whiten_eig(M, damp, eig_device, eig_dtype)
            del M
            W_r = W.index_select(1, K)
            W_c = W.index_select(1, C) if C.numel() else None

            R = R.to(device=device, dtype=torch.float32)
            Rinv = Rinv.to(device=device, dtype=torch.float32)
            U, S, Vt = torch.linalg.svd(W_r @ R, full_matrices=False)

            sqrt_s = S.clamp_min(0).sqrt()
            W_u = U * sqrt_s                                      # (out, r_max)
            W_v = sqrt_s.unsqueeze(1) * (Vt @ Rinv)              # (r_max, in - |C|)

            n_mass = C.numel()
            normal_in = d_in - n_mass
            if custom_ranks is not None:
                k_val = custom_ranks[gi]
            else:
                k_val = max(1, round((init_ratio * d_in * d_out - d_out * n_mass) / (normal_in + d_out)))
            k_val = min(k_val, S.shape[0])

            new = MassiveSplitGateLinear(
                W_u.to(dtype), W_v.to(dtype),
                None if W_c is None else W_c.to(dtype),
                None if bias is None else bias.to(dtype),
                K, C if C.numel() else None, d_in,
                init_k=k_val, beta=beta, temperature=temperature,
                z_slope=z_slope).to(device)
            set_module(layer, p, new)
            meta[f"{li}.{p}"] = {"init_k": k_val, "mass": [int(c) for c in C.tolist()],
                                 "in": d_in, "out": d_out, "cond": cond, "damped": n_damped}
            del W, W_r, W_c, U, S, Vt, sqrt_s, W_u, W_v, R, Rinv
            torch.cuda.empty_cache()
            gi += 1

        layer.to("cpu")
        cache = HiddenCache(outs)
        del cov, outs
        gc.collect()
        torch.cuda.empty_cache()

        el = time.time() - t0
        print(f"  block {li:>2}/{len(layers)-1} whitened SVD + split "
              f"({el:.0f}s elapsed, ~{el/(li+1)*(len(layers)-li-1):.0f}s left)", flush=True)

    return model, meta


# ==========================================================================
# 5. Slicing with learned singular-value masks
# ==========================================================================
@torch.no_grad()
def slice_split_ranks(student):
    """Hard-truncate each MassiveSplitGateLinear into a MassiveSplitLinear using learned active indices."""
    print("\n[slice] materialising learned ranks into MassiveSplitLinear")
    meta = {}
    for li, layer in enumerate(get_layers(student)):
        for path in layer_linear_paths(layer):
            m = get_module(layer, path)
            if not isinstance(m, MassiveSplitGateLinear):
                continue
            idx = m.active_indices().to(m.W_u.device)
            W_u = m.W_u.data[:, idx].clone()
            W_v = m.W_v.data[idx, :].clone()
            W_mass = m.W_mass.data.clone() if m.n_mass else None
            bias = m.bias.data.clone() if m.bias is not None else None
            r = W_u.shape[1]
            dtype, dev = W_u.dtype, W_u.device

            new = MassiveSplitLinear(
                W_v, W_u, W_mass, bias,
                m.keep_idx, m.mass_idx, m.in_features).to(device=dev, dtype=dtype)
            set_module(layer, path, new)
            meta[f"{li}.{path}"] = {
                "rank": r,
                "mass": [int(c) for c in m.mass_idx.tolist()] if m.n_mass else [],
                "in": m.in_features,
                "out": m.out_features,
            }
            del m
        gc.collect()
    return student, meta


# ==========================================================================
# 6. Rank utilities & planning
# ==========================================================================
def ranks_from_stage2(path):
    """Per-linear rank implied by Stage 2's gate logits: `(Z > 0).sum()`."""
    ck = torch.load(path, map_location="cpu")
    if "Z" not in ck:
        raise ValueError(f"{path} has no 'Z' (keys: {list(ck)}) — not a Stage-2 resume ckpt")
    return [max(1, int((z > 0).sum())) for z in ck["Z"]], ck.get("epoch")


def uniform_ranks(model, ratio):
    out = []
    for layer in get_layers(model):
        for p in layer_linear_paths(layer):
            fc = get_module(layer, p)
            out.append(max(1, round(ratio * fc.in_features * fc.out_features
                                    / (fc.in_features + fc.out_features))))
    return out


def plan_shapes(model, explicit, topk, scope):
    """(in, out, |C|) per linear, using only shapes — for ratio arithmetic."""
    out = []
    for layer in get_layers(model):
        for p in layer_linear_paths(layer):
            fc = get_module(layer, p)
            if scope in ("residual", "all") and p in RESIDUAL_FED and explicit:
                m = len([c for c in explicit if 0 <= c < fc.in_features])
            elif scope in ("auto", "all"):
                m = topk
            else:
                m = 0
            out.append((fc.in_features, fc.out_features, m))
    return out


def realized_ratio(shapes, ranks):
    comp = sum(min(r, min(o, i - m)) * (i - m + o) + m * o
               for (i, o, m), r in zip(shapes, ranks))
    dense = sum(i * o for i, o, _ in shapes)
    return comp / dense


def match_ratio(shapes, ranks, target, tol=1e-5):
    """Scale every rank by a global alpha so the realized ratio hits `target`."""
    lo, hi = 0.0, 4.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        scaled = [max(1, round(mid * r)) for r in ranks]
        if realized_ratio(shapes, scaled) > target:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    alpha = 0.5 * (lo + hi)
    return [max(1, round(alpha * r)) for r in ranks], alpha


# ==========================================================================
# 7. Save / load
# ==========================================================================
def save_split(student, meta, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    torch.save({"meta": meta, "state_dict": student.state_dict()}, path)
    print(f"[save] split student -> {path} "
          f"({sum(p.numel() for p in student.parameters())/1e9:.2f}B params, "
          f"{os.path.getsize(path)/1e9:.1f} GB)")


def load_split(student, path):
    ck = torch.load(path, map_location="cpu")
    meta = ck["meta"]
    dtype = next(student.parameters()).dtype
    for li, layer in enumerate(get_layers(student)):
        for p in layer_linear_paths(layer):
            info = meta[f"{li}.{p}"]
            r, C = info["rank"], torch.tensor(info["mass"], dtype=torch.long)
            d_in, d_out = info["in"], info["out"]
            keep = torch.ones(d_in, dtype=torch.bool)
            keep[C] = False
            K = torch.nonzero(keep).squeeze(-1)
            fc = get_module(layer, p)
            new = MassiveSplitLinear(
                torch.zeros(r, K.numel()), torch.zeros(d_out, r),
                torch.zeros(d_out, C.numel()) if C.numel() else None,
                torch.zeros(d_out) if fc.bias is not None else None,
                K, C if C.numel() else None, d_in).to(dtype)
            set_module(layer, p, new)
    student.load_state_dict(ck["state_dict"])
    del ck
    gc.collect()
    return student, meta


# ==========================================================================
# 8. Entry point
# ==========================================================================
def run(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    print(f"=== massive-split | {args.model_id} | target ratio {args.ratio} ===")
    print("NOTE: no rotation — a Hadamard/orthogonal rotation would mix the "
          "massive channels into every other channel and make the split meaningless.")

    student, tokenizer = load_llm(args.model_id, args.dtype, args.seqlen)
    calib_ids = data_utils.get_calib_input_ids(
        tokenizer, nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
        dataset=args.calib_dataset, data_root=args.data_root)
    sw_ids = calib_ids

    timings = {}
    explicit = [int(c) for c in args.mass_channels.split(",") if c.strip() != ""]

    if args.load_split:
        print(f"\n[load] split student from {args.load_split} — skipping the build")
        with Timer() as t:
            student, meta = load_split(student, args.load_split)
        timings["load_split"] = t.elapsed
        rank_src = f"loaded:{args.load_split}"
        return _finish(args, student, tokenizer, None, sw_ids, timings, device, meta, rank_src, explicit)

    # --- Stage 0: Build Split Gated Student --------------------------------
    custom_ranks = None
    if args.stage2_ranks:
        ranks, ep = ranks_from_stage2(args.stage2_ranks)
        rank_src = f"stage2:{args.stage2_ranks}@ep{ep}"
        print(f"\n[ranks] loaded {len(ranks)} per-linear ranks from {args.stage2_ranks} (epoch {ep})")
        shapes = plan_shapes(student, explicit, args.mass_topk, args.mass_scope)
        if args.match_ratio:
            tgt = args.ratio if args.match_ratio_to < 0 else args.match_ratio_to
            ranks, alpha = match_ratio(shapes, ranks, tgt)
            print(f"[ranks] --match_ratio: scaled by alpha={alpha:.5f} -> {realized_ratio(shapes, ranks):.5f}")
        custom_ranks = ranks
        args.skip_stage1 = args.skip_stage2 = True
    else:
        rank_src = f"learned:r{args.ratio}"

    print(f"\n[split] scope={args.mass_scope} explicit={explicit} "
          f"topk={args.mass_topk} thresh={args.mass_ratio_thresh} | "
          f"eig {args.eig_dtype} on {args.eig_device}, damp={args.damp}")

    with Timer() as t:
        student, meta_pre = build_split_gated_student(
            student, calib_ids, device, init_ratio=args.ratio,
            batch_size=args.calib_batch_size, explicit=explicit,
            topk=args.mass_topk, ratio_thresh=args.mass_ratio_thresh,
            scope=args.mass_scope, damp=args.damp,
            eig_device=args.eig_device, eig_dtype=args.eig_dtype,
            beta=args.beta, temperature=args.temperature, z_slope=args.z_slope,
            custom_ranks=custom_ranks)
    timings["stage0_split_svd"] = t.elapsed

    gates = list(iter_gates(student))
    print(f"[stage0] {len(gates)} split gated linears | init hard ratio {hard_size_ratio(gates):.4f}")

    # --- Stage 1: Global Rank Search --------------------------------------
    if not args.skip_stage1 and args.stage1_epochs > 0:
        teacher, _ = load_llm(args.model_id, args.dtype, args.seqlen)
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
        print("\n[stage1] skipped")

    # Hand the Stage-1 allocation to Stage 2's per-singular-value gates
    for m in gates:
        m.to_z_mode()
    print(f"[stage1->2] hard ratio after switching to Z gates: {hard_size_ratio(gates):.4f}")

    # --- Stage 2: Sliding-Window Rank Search -------------------------------
    teacher = None
    if not args.skip_stage2 and args.stage2_epochs > 0:
        teacher, _ = load_llm(args.model_id, args.dtype, args.seqlen)
        gc.collect(); torch.cuda.empty_cache()
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

    # --- Slicing: Materialize MassiveSplitLinear --------------------------
    student, meta = slice_split_ranks(student)
    gc.collect(); torch.cuda.empty_cache()
    if args.save_split:
        save_split(student, meta, args.save_split)

    return _finish(args, student, tokenizer, teacher, sw_ids, timings, device, meta, rank_src, explicit)


def _finish(args, student, tokenizer, teacher, sw_ids, timings, device, meta, rank_src, explicit):
    """Stage 3 + evaluation + reporting."""
    n_mass = sum(len(v["mass"]) for v in meta.values())
    n_lin = sum(1 for v in meta.values() if v["mass"])
    print(f"\n[split] {n_lin}/{len(meta)} linears carry an exact branch "
          f"({n_mass} held-out channels in total)")

    # --- Stage 3: Sliding-Window Fine-Tuning -------------------------------
    avg_mse = 0.0
    if not args.skip_stage3 and args.stage3_epochs > 0:
        if teacher is None:
            teacher, _ = load_llm(args.model_id, args.dtype, args.seqlen)
        if args.ft_dataset == "alpaca":
            print(f"\n[stage3] loading alpaca dataset (seqlen 256) for fine-tuning")
            ft_ids = data_utils.get_alpaca_input_ids(tokenizer, seqlen=256, data_root=args.data_root)
        else:
            ft_ids = sw_ids

        gc.collect(); torch.cuda.empty_cache()
        with Timer() as t:
            wd = 0.0 if args.disable_ft_weight_decay else 0.01
            avg_mse = sliding_window_finetune(
                teacher, student, ft_ids, device, epochs=args.stage3_epochs,
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
                lora_alpha=args.stage3_lora_alpha,
                chunk_size=args.ft_chunk_size)
        timings["stage3_finetune"] = t.elapsed
        del teacher
        gc.collect(); torch.cuda.empty_cache()
    else:
        print("\n[stage3] skipped")

    # --- Evaluate & Report ------------------------------------------------
    rep = param_report(student)
    ranks_str = ",".join(str(v["rank"]) for v in meta.values())
    print(f"\n[eval] linear params {rep['linear_params']/1e9:.3f}B / "
          f"{rep['dense_linear_params']/1e9:.3f}B = {rep['linear_param_ratio']:.5f} | "
          f"total {rep['total_params']/1e9:.3f}B")

    ppl = {}
    for ds in args.eval_datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        test_ids = data_utils.get_test_input_ids(tokenizer, args.seqlen, ds, args.data_root)
        ppl[ds] = evaluate_ppl(student, test_ids, device, args.eval_batch_size, desc=f"ppl:{ds}")
        print(f"[eval] {ds} perplexity: {ppl[ds]:.4f}")

    payload = {
        "method": "massive_split",
        "model_id": args.model_id,
        "target_ratio": args.ratio,
        "rank_source": rank_src,
        "mass_channels": explicit,
        "mass_scope": args.mass_scope,
        "n_held_out_channels": n_mass,
        "ppl": ppl,
        "avg_window_mse": avg_mse,
        "timings_sec": timings,
        "config": vars(args),
        "layer_ranks": ranks_str,
        "per_linear_mass": {k: v["mass"] for k, v in meta.items() if v["mass"]},
        **rep,
    }
    out_json = args.out_json or f"results/{args.model_id.split('/')[-1]}_masssplit_r{args.ratio}.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    save_result(out_json, payload)
    print(f"\n[save] Result JSON saved to {out_json}")
    return payload


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--ratio", type=float, default=0.6)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    # massive channels
    p.add_argument("--mass_channels", type=str, default="2533,1415",
                   help="residual-stream massive channels, applied to q/k/v/gate/up")
    p.add_argument("--mass_scope", type=str, default="residual",
                   choices=["residual", "auto", "all", "none"],
                   help="residual: explicit list on q/k/v/gate/up only; "
                        "auto: top-k by input energy on every linear; "
                        "all: both; none: plain whitened SVD (ablation)")
    p.add_argument("--mass_topk", type=int, default=2,
                   help="channels to auto-detect per linear (auto/all scopes)")
    p.add_argument("--mass_ratio_thresh", type=float, default=50.0,
                   help="an auto-detected channel must carry >= this x the "
                        "median channel energy, else it is not an outlier")

    # whitening
    p.add_argument("--eig_device", type=str, default="same", choices=["same", "cpu"],
                   help="'same' runs eigh on --device. Measured on this box (3090): "
                        "d=11008 float64 takes 26s on GPU vs 91s on CPU, so the GPU "
                        "is the default despite the 1/64 FP64 rate; fall back to cpu "
                        "if the ~3 GB float64 workspace does not fit")
    p.add_argument("--eig_dtype", type=str, default="float64",
                   choices=["float64", "float32"])
    p.add_argument("--damp", type=float, default=1e-6,
                   help="eigenvalue floor as a fraction of the largest eigenvalue")

    # gating / rank search
    p.add_argument("--beta", type=float, default=0.1,
                   help="Stage 1 gate sharpness (tanh temperature)")
    p.add_argument("--temperature", type=float, default=10.0,
                   help="Stage 2 gate sharpness (sigmoid temperature)")
    p.add_argument("--z_slope", type=float, default=0.1,
                   help="slope for seeding Z_i = (K - i) * z_slope")
    p.add_argument("--skip_stage1", action="store_true",
                   help="skip global rank search and seed Stage 2 uniformly")
    p.add_argument("--stage1_epochs", type=int, default=5)
    p.add_argument("--stage1_batch_size", type=int, default=1)
    p.add_argument("--skip_stage2", action="store_true",
                   help="skip sliding-window rank search")
    p.add_argument("--stage2_epochs", type=int, default=10)
    p.add_argument("--stagger", type=int, default=2,
                   help="stride for the alternating even/odd window schedule")
    p.add_argument("--rank_lr", type=float, default=5.0,
                   help="learning rate for the gate parameters (k and Z)")
    p.add_argument("--lambda_rank", type=float, default=1.0,
                   help="initial Lagrange multiplier on the size penalty")
    p.add_argument("--lambda_lr", type=float, default=10.0,
                   help="dual ascent step size for lambda")
    p.add_argument("--no_grad_ckpt", action="store_true",
                   help="disable gradient checkpointing during Stage 1")
    p.add_argument("--stage2_ckpt", type=str, default="",
                   help="path to checkpoint/resume Stage 2 rank search")

    # legacy rank reuse
    p.add_argument("--stage2_ranks", type=str, default="",
                   help="Stage-2 resume ckpt to take per-linear ranks from (skips Stages 1 & 2)")
    p.add_argument("--match_ratio", action="store_true",
                   help="rescale all ranks so the realized param ratio matches "
                        "the no-split equivalent exactly")
    p.add_argument("--match_ratio_to", type=float, default=-1.0)

    # calibration
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--calib_dataset", type=str, default="wikitext2")
    p.add_argument("--data_root", type=str,
                   default="/home/kangeunjeon/geontack_kairi/data")
    p.add_argument("--calib_batch_size", type=int, default=1)
    p.add_argument("--sw_batch_size", type=int, default=1)

    # stage 3
    p.add_argument("--skip_stage3", action="store_true")
    p.add_argument("--stage3_epochs", type=int, default=2, help="epochs per window in Stage 3")
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--stage3inner", action="store_true",
                   help="Skip fine-tuning the first 4 and last 4 blocks in Stage 3")
    p.add_argument("--stage3innerpercent", action="store_true",
                   help="Skip fine-tuning the first 20%% and last 15%% of blocks in Stage 3")
    p.add_argument("--stage3progressive", action="store_true",
                   help="Progressively increase and decrease window size at the ends in Stage 3")
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
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--ft_lr_mode", type=str, default="const", choices=["const", "relative"],
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
    p.add_argument("--ft_dataset", type=str, default="wikitext2",
                   choices=["wikitext2", "alpaca"],
                   help="Dataset to use for Stage 3 fine-tuning. 'wikitext2' reuses "
                        "the Stage 0-2 calibration samples. 'alpaca' loads the "
                        "yahma/alpaca-cleaned dataset.")
    p.add_argument("--ft_chunk_size", type=int, default=1024,
                   help="Number of sequences to process at a time during Stage 3 "
                        "fine-tuning. Prevents Host RAM OOM when using large "
                        "datasets like alpaca.")
    p.add_argument("--ft_clip_grad", type=float, default=0.0,
                   help="max grad-norm for Stage 3; 0 disables. Single-sample loss spikes "
                        "of 1e6 have been observed in the deep windows, so 1.0 is a "
                        "reasonable value")
    p.add_argument("--disable_ft_weight_decay", action="store_true",
                   help="Disable AdamW weight decay in Stage 3")
    p.add_argument("--rel_mse", action="store_true", default=True)

    # io
    p.add_argument("--save_split", type=str, default="")
    p.add_argument("--load_split", type=str, default="")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--eval_datasets", type=str, default="wikitext2")
    p.add_argument("--eval_batch_size", type=int, default=1)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
