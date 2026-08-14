"""Massive-activation channel split: compress everything *except* the outlier channels.

The idea
--------
A handful of residual-stream channels in Llama-2-7B carry activations two orders
of magnitude larger than the rest (2533 at |h| ~ 2720, 1415 at ~1600, versus a
median of ~10 -- see `figs/acts_base.npz`). Those channels dominate the input
covariance `M = X^T X`, so the whitened SVD of Stage 0 spends essentially all of
its dynamic range reproducing them, and the singular directions that matter for
the other 4,094 channels get truncated first.

So take them out of the optimisation entirely. For a linear `W` (out, in) with
massive input channels `C` and the rest `K`:

    y = W x = W[:, K] x_K  +  W[:, C] x_C
              \__________/    \__________/
               compressed        exact

The split is *algebraically exact* -- no approximation is introduced by it. Only
the `W[:, K]` half is whitened (eigendecomposition of the reduced covariance
`M_KK`), factored by SVD, truncated, and fine-tuned. `W[:, C]` is carried along
at full precision as a frozen `(out, |C|)` slab and is never touched.

Note that "add the massive channels back in for evaluation" is *not* a separate
step at the end: the exact branch is part of the forward pass throughout, so the
Stage-3 windows are fit against the true layer output rather than against a
mutilated one. Dropping the branch during fine-tuning and re-adding it after
would train the low-rank half to compensate for a missing term and then
double-count it, which is strictly worse -- hence it is not offered here.

Cost of keeping `W[:, C]` dense: `|C| * (out - r)` extra parameters per linear,
about +0.04% of the dense linear budget at |C| = 2. `--match_ratio` shrinks the
ranks to absorb even that, if an exactly ratio-matched comparison is wanted.

Reusing Stage 2
---------------
`ckpt/r0.6_stage2_resume.pt` holds the per-singular-value gate logits `Z`, not
weights. Those logits index the singular values of `W R` in the *full-channel*
whitened basis; once `C` is removed the covariance, the whitening `R` and the
whole singular basis change, so `Z` cannot be copied over element-wise.

What does transfer is the number it implies: `r_i = (Z_i > 0).sum()`, the rank
Stage 2 allocated to linear `i`. Seeding the reduced-channel SVD with those
ranks reproduces Stage 2's *allocation* across layers -- which is the expensive
part, and the part being held fixed for the comparison -- without re-running it.
That is what `--stage2_ranks` does.

Example
-------
  python massive_split.py --ratio 0.6 --device cuda:6 \
      --stage2_ranks ckpt/r0.6_stage2_resume.pt \
      --mass_channels 2533,1415 \
      --out_json results/r0.6_masssplit.json
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
from sliding_llm import sliding_window_finetune

# Linears whose input is the residual stream (through an RMSNorm, which is a
# per-channel rescale and so preserves the channel indexing). Only these can be
# given the residual-stream massive-channel indices; `o_proj` reads the
# attention output and `down_proj` the SwiGLU intermediate, both different
# spaces with different outliers -- those are handled by `--mass_topk`.
RESIDUAL_FED = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "mlp.gate_proj", "mlp.up_proj")


# ==========================================================================
# 1. The split linear
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
# 4. Build the split student
# ==========================================================================
@torch.no_grad()
def build_split_student(model, calib_ids, device, ranks, batch_size=1,
                        explicit=(), topk=0, ratio_thresh=50.0, scope="residual",
                        damp=1e-6, eig_device="same", eig_dtype="float64"):
    """Replace every target Linear with a `MassiveSplitLinear`, block by block.

    One dense sequential pass: each block's forward both collects the per-linear
    input covariance and produces the input activations for the next block, so
    the dense model is never fully resident on the GPU.
    """
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    seqlen = calib_ids.shape[1]
    kwargs_cache = {}

    def kwargs_fn(bs):
        if bs not in kwargs_cache:
            kwargs_cache[bs] = build_decoder_kwargs(seqlen, dtype, device, bs)
        return kwargs_cache[bs]

    print(f"[split] embedding {calib_ids.shape[0]} calibration sequences")
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

            # Reduced covariance / weight: the massive columns leave the problem.
            M = (cov[p].index_select(0, K).index_select(1, K) / max(1, ntok[p])).double()
            R, Rinv, cond, n_damped = whiten_eig(M, damp, eig_device, eig_dtype)
            del M
            W_r = W.index_select(1, K)
            W_c = W.index_select(1, C) if C.numel() else None

            R = R.to(device=device, dtype=torch.float32)
            Rinv = Rinv.to(device=device, dtype=torch.float32)
            U, S, Vt = torch.linalg.svd(W_r @ R, full_matrices=False)

            r = int(min(ranks[gi], S.shape[0]))
            r = max(1, r)
            sq = S[:r].clamp_min(0).sqrt()
            W_u = U[:, :r] * sq                             # (out, r)
            W_v = sq.unsqueeze(1) * (Vt[:r] @ Rinv)         # (r, in - |C|)

            new = MassiveSplitLinear(W_v.to(dtype), W_u.to(dtype),
                                     None if W_c is None else W_c.to(dtype),
                                     None if bias is None else bias.to(dtype),
                                     K, C if C.numel() else None, d_in)
            set_module(layer, p, new)
            meta[f"{li}.{p}"] = {"rank": r, "mass": [int(c) for c in C.tolist()],
                                 "in": d_in, "out": d_out}
            if li == 0 or (li == len(layers) - 1):
                tail = float(S[r:].pow(2).sum() / S.pow(2).sum().clamp_min(1e-30))
                print(f"  L{li:>2} {p:<18} r={r:<5} |C|={C.numel()} "
                      f"cond={cond:.2e} damped={n_damped} tail_energy={tail:.4f} "
                      f"mass={C.tolist()}")

            cov[p] = None
            gi += 1
            del W, W_r, W_c, U, S, Vt, sq, W_u, W_v, R, Rinv
            torch.cuda.empty_cache()

        layer.to("cpu")
        cache = HiddenCache(outs)
        del cov, outs
        gc.collect()
        torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"[split] layer {li+1}/{len(layers)} done "
              f"({el:.0f}s elapsed, ~{el/(li+1)*(len(layers)-li-1):.0f}s left)", flush=True)

    if gi != len(ranks):
        raise ValueError(f"consumed {gi} ranks but was given {len(ranks)}")
    return model, meta


# ==========================================================================
# 5. Ranks
# ==========================================================================
def ranks_from_stage2(path):
    """Per-linear rank implied by Stage 2's gate logits: `(Z > 0).sum()`.

    `iter_gates` walks layers in forward order and, within a layer,
    `layer.modules()` order -- which for a Llama block is q, k, v, o, gate, up,
    down, exactly `layer_linear_paths` order. So the list lines up index for
    index with the walk in `build_split_student`.
    """
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
    """(in, out, |C|) per linear, using only shapes — for the ratio arithmetic.

    `|C|` is exact for the explicit residual channels and an upper bound for the
    auto-detected ones (a channel is dropped if it fails `--mass_ratio_thresh`),
    so `--match_ratio` may end up marginally *under* target rather than over.
    """
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
# 6. Save / load
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
# 7. Entry point
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
    sw_ids = calib_ids[:args.sw_nsamples]

    timings = {}
    explicit = [int(c) for c in args.mass_channels.split(",") if c.strip() != ""]

    if args.load_split:
        print(f"\n[load] split student from {args.load_split} — skipping the build")
        with Timer() as t:
            student, meta = load_split(student, args.load_split)
        timings["load_split"] = t.elapsed
        rank_src = f"loaded:{args.load_split}"
    else:
        # --- ranks -------------------------------------------------------
        if args.stage2_ranks:
            ranks, ep = ranks_from_stage2(args.stage2_ranks)
            rank_src = f"stage2:{args.stage2_ranks}@ep{ep}"
            print(f"\n[ranks] {len(ranks)} per-linear ranks from {args.stage2_ranks} "
                  f"(epoch {ep}) | sum {sum(ranks)}")
        else:
            ranks = uniform_ranks(student, args.ratio)
            rank_src = f"uniform:{args.ratio}"
            print(f"\n[ranks] uniform init at ratio {args.ratio} | sum {sum(ranks)}")

        shapes = plan_shapes(student, explicit, args.mass_topk, args.mass_scope)
        if len(shapes) != len(ranks):
            raise ValueError(f"{len(ranks)} ranks vs {len(shapes)} linears")
        pred = realized_ratio(shapes, ranks)
        base = sum(min(r, min(o, i)) * (i + o) for (i, o, _), r in zip(shapes, ranks)) \
            / sum(i * o for i, o, _ in shapes)
        print(f"[ranks] predicted ratio {pred:.5f} (no-split equivalent {base:.5f}, "
              f"delta {pred-base:+.5f} from the exact massive slabs)")
        if args.match_ratio:
            tgt = base if args.match_ratio_to < 0 else args.match_ratio_to
            ranks, alpha = match_ratio(shapes, ranks, tgt)
            print(f"[ranks] --match_ratio: scaled by alpha={alpha:.5f} -> "
                  f"{realized_ratio(shapes, ranks):.5f} (target {tgt:.5f})")
            rank_src += f"|match{alpha:.4f}"

        # --- build -------------------------------------------------------
        print(f"\n[split] scope={args.mass_scope} explicit={explicit} "
              f"topk={args.mass_topk} thresh={args.mass_ratio_thresh} | "
              f"eig {args.eig_dtype} on {args.eig_device}, damp={args.damp}")
        with Timer() as t:
            student, meta = build_split_student(
                student, calib_ids, device, ranks,
                batch_size=args.calib_batch_size, explicit=explicit,
                topk=args.mass_topk, ratio_thresh=args.mass_ratio_thresh,
                scope=args.mass_scope, damp=args.damp,
                eig_device=args.eig_device, eig_dtype=args.eig_dtype)
        timings["build_split"] = t.elapsed
        gc.collect(); torch.cuda.empty_cache()
        if args.save_split:
            save_split(student, meta, args.save_split)

    n_mass = sum(len(v["mass"]) for v in meta.values())
    n_lin = sum(1 for v in meta.values() if v["mass"])
    print(f"[split] {n_lin}/{len(meta)} linears carry an exact branch "
          f"({n_mass} held-out channels in total)")

    rep = param_report(student)
    print(f"[split] linear params {rep['linear_params']/1e9:.3f}B / "
          f"{rep['dense_linear_params']/1e9:.3f}B = {rep['linear_param_ratio']:.5f}")

    # --- Stage 3 ---------------------------------------------------------
    avg_mse = 0.0
    if not args.skip_stage3 and args.stage3_epochs > 0:
        teacher, _ = load_llm(args.model_id, args.dtype, args.seqlen)
        gc.collect(); torch.cuda.empty_cache()
        with Timer() as t:
            avg_mse = sliding_window_finetune(
                teacher, student, sw_ids, device, epochs=args.stage3_epochs,
                window=args.window, stride=args.stride, lr=args.ft_lr,
                batch_size=args.sw_batch_size, rel_mse=args.rel_mse,
                weight_decay=0.0 if args.disable_ft_weight_decay else 0.01,
                skip_ends=args.stage3_skip_ends, edge=args.stage3_edge,
                edge_lr=args.stage3_edge_lr, clip_grad=args.ft_clip_grad,
                lr_mode=args.ft_lr_mode, rel_lr=args.ft_rel_lr)
        timings["stage3_finetune"] = t.elapsed
        del teacher
        gc.collect(); torch.cuda.empty_cache()
    else:
        print("\n[stage3] skipped")

    # --- evaluate --------------------------------------------------------
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

    # ranks
    p.add_argument("--stage2_ranks", type=str, default="",
                   help="Stage-2 resume ckpt to take per-linear ranks from "
                        "(the gate logits themselves are basis-specific and are "
                        "NOT transferable — only the implied ranks are)")
    p.add_argument("--match_ratio", action="store_true",
                   help="rescale all ranks so the realized param ratio matches "
                        "the no-split equivalent exactly")
    p.add_argument("--match_ratio_to", type=float, default=-1.0)

    # calibration
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--sw_nsamples", type=int, default=128)
    p.add_argument("--calib_dataset", type=str, default="wikitext2")
    p.add_argument("--data_root", type=str,
                   default="/home/kangeunjeon/geontack_kairi/data")
    p.add_argument("--calib_batch_size", type=int, default=1)
    p.add_argument("--sw_batch_size", type=int, default=1)

    # stage 3
    p.add_argument("--skip_stage3", action="store_true")
    p.add_argument("--stage3_epochs", type=int, default=2)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--ft_lr", type=float, default=1e-4)
    p.add_argument("--ft_lr_mode", type=str, default="const", choices=["const", "relative"])
    p.add_argument("--ft_rel_lr", type=float, default=3e-5)
    p.add_argument("--ft_clip_grad", type=float, default=0.0)
    p.add_argument("--disable_ft_weight_decay", action="store_true")
    p.add_argument("--stage3_skip_ends", type=int, default=0)
    p.add_argument("--stage3_edge", type=str, default="skip", choices=["skip", "individual"])
    p.add_argument("--stage3_edge_lr", type=float, default=None)
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
