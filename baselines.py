"""Baselines for the LLM comparison: SVD-LLM and Dobi-SVD.

Both are re-implemented here against the same model/data plumbing our method
uses (`llm_utils` + `data_utils`), so the numbers differ only in the compression
algorithm — same calibration sequences, same dtype, same perplexity protocol.

  svdllm   SVD-LLM (Wang et al.). Cholesky whitening of the per-linear input
           covariance, SVD of `W L`, then a *uniform* truncation to
           `in*out*ratio/(in+out)` singular values in every layer. Follows
           `SVD-LLM/SVDLLM.py::profle_svdllm_low_resource` + `::whitening`.

  dobisvd  Dobi-SVD (Wang et al., ICLR'25). Differentiable *activation*
           truncation: one continuous rank `gamma` per linear, trained end to
           end on the LM loss plus a compression-ratio penalty, then an IPCA
           weight update that maps the truncated activation subspace back onto
           the weights. Follows `Dobi-SVD/svd_trainer.py` +
           `Dobi-SVD/weight_updater.py`; the numerically delicate SVD backward
           (`stable_lowrank_SVD`) and `IncrementalPCAonGPU` are imported from
           that repo verbatim rather than re-derived.

Example
-------
  python baselines.py --method svdllm  --ratio 0.8 --out_json runs/svdllm_r0.8.json
  python baselines.py --method dobisvd --ratio 0.8 --out_json runs/dobi_r0.8.json
"""
import argparse
import gc
import math
import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import tqdm

import data_utils
import rotation
from llm_utils import (HiddenCache, Timer, build_decoder_kwargs, evaluate_ppl,
                       get_layers, get_module, layer_linear_paths,
                       layer_rank_string, load_llm, param_report, run_layer,
                       save_result, seed_everything, set_module)

DOBI_REPO = os.environ.get("DOBI_SVD_REPO", "/home/kangeunjeon/hyunjung/Dobi-SVD")


def _import_dobi():
    """Pull `stable_lowrank_SVD` / `IncrementalPCAonGPU` from the Dobi-SVD repo."""
    if DOBI_REPO not in sys.path:
        sys.path.insert(0, DOBI_REPO)
    from modules.stable_svd import stable_lowrank_SVD
    from modules.IncrementalPCA import IncrementalPCAonGPU
    return stable_lowrank_SVD, IncrementalPCAonGPU


def _factored(W_u, W_v, bias, rank):
    """`fc_v -> fc_u` pair, the same container our method produces."""
    dtype, dev = W_u.dtype, W_u.device
    fc_v = nn.Linear(W_v.shape[1], rank, bias=False).to(device=dev, dtype=dtype)
    fc_u = nn.Linear(rank, W_u.shape[0], bias=bias is not None).to(device=dev, dtype=dtype)
    fc_v.weight.data.copy_(W_v)
    fc_u.weight.data.copy_(W_u)
    if bias is not None:
        fc_u.bias.data.copy_(bias)
    seq = nn.Sequential(OrderedDict([("fc_v", fc_v), ("fc_u", fc_u)]))
    seq.rank = int(rank)
    return seq


# ==========================================================================
# SVD-LLM
# ==========================================================================
@torch.no_grad()
def compress_svdllm(model, calib_ids, device, ratio, batch_size=1):
    """Whitening + uniform truncation, one decoder block at a time."""
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    seqlen = calib_ids.shape[1]
    kwargs_cache = {}

    def kwargs_fn(bs):
        if bs not in kwargs_cache:
            kwargs_cache[bs] = build_decoder_kwargs(seqlen, dtype, device, bs)
        return kwargs_cache[bs]

    cache = HiddenCache.from_embeddings(model, calib_ids, device)

    for layer in tqdm(layers, desc="[svdllm] whitening + SVD"):
        layer.to(device)
        paths = layer_linear_paths(layer)

        raw = {}
        handles = []
        for p in paths:
            fc = get_module(layer, p)
            raw[p] = torch.zeros(fc.in_features, fc.in_features,
                                 dtype=torch.float32, device=device)

        def make_hook(p):
            def hook(module, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
                raw[p] += x.t() @ x
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

            M = raw[p].double()
            try:
                L = torch.linalg.cholesky(M)
            except Exception:
                print("  [svdllm] scaling matrix not PD, shifting eigenvalues")
                eig = torch.linalg.eigvalsh(M)
                M = M + (-eig[0] + 1e-6) * torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
                L = torch.linalg.cholesky(M)
            raw[p] = None
            del M
            Lf = L.float()

            U, S, Vt = torch.linalg.svd(W @ Lf, full_matrices=False)
            k = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
            k = max(1, min(k, S.shape[0]))
            sqrt_s = S[:k].clamp_min(0).sqrt()
            W_u = U[:, :k] * sqrt_s
            Vt_un = torch.linalg.solve_triangular(Lf, Vt[:k, :], upper=False, left=False)
            W_v = sqrt_s.unsqueeze(1) * Vt_un

            set_module(layer, p, _factored(W_u.to(dtype), W_v.to(dtype),
                                           None if bias is None else bias.to(dtype), k))
            del W, U, S, Vt, W_u, W_v, Vt_un, L, Lf
            torch.cuda.empty_cache()

        layer.to("cpu")
        cache = HiddenCache(outs)
        del raw, outs
        gc.collect()
        torch.cuda.empty_cache()
    return model


# ==========================================================================
# Dobi-SVD
# ==========================================================================
class DobiActTruncLinear(nn.Module):
    """Differentiable activation truncation with a trainable rank `gamma`.

    Mirrors `Dobi-SVD/modules/stable_svd.py::SVDTransformLayer`: the *output
    activation* of the dense linear is low-rank truncated through a soft tanh
    gate on its singular values, and the gradient flows to `gamma` through the
    repo's numerically stabilised SVD backward.
    """

    def __init__(self, linear, gamma_init, seq_len, beta, svd_fn):
        super().__init__()
        self.ori = linear
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))
        self.seq_len = seq_len
        self.beta = beta
        self.svd_fn = svd_fn
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        out_dtype = x.dtype
        y = self.ori(x).float()
        squeezed = y.dim() == 3
        if squeezed:
            y = y.squeeze(0)
        q = int(self.gamma.detach().item()) + 5
        q = max(1, min(min(y.shape), q))
        U, S, V = self.svd_fn.apply(y, q)
        seq = torch.arange(1, S.shape[0] + 1, device=y.device, dtype=y.dtype)
        real_gamma = torch.clamp(self.gamma, min=1.0, max=float(S.shape[0]))
        trunc = 0.5 * torch.tanh(self.beta * (real_gamma - seq)) + 0.5
        y = (U * (S * trunc)) @ V.t()
        if squeezed:
            y = y.unsqueeze(0)
        return y.to(out_dtype)


def _dobi_rank_ratio(in_f, out_f, seq_len):
    """Dobi's `RANK_RATIO`: gamma is a rank *per seq_len-sized activation block*."""
    return min(in_f, out_f) / seq_len


def dobi_train_gamma(model, calib_ids, device, ratio, steps, lr, min_lr, beta,
                     lambda_reg, seq_len, log_every=8):
    """Stage 1 of Dobi-SVD: learn one continuous rank per linear."""
    stable_svd, _ = _import_dobi()
    layers = get_layers(model)

    dense_total = 0.0
    modules = []
    for layer in layers:
        for p in layer_linear_paths(layer):
            fc = get_module(layer, p)
            rr = _dobi_rank_ratio(fc.in_features, fc.out_features, seq_len)
            gamma_init = (1.0 / rr) * ratio * fc.in_features * fc.out_features \
                / (fc.in_features + fc.out_features)
            new = DobiActTruncLinear(fc, gamma_init, seq_len, beta, stable_svd)
            set_module(layer, p, new)
            modules.append(new)
            dense_total += fc.in_features * fc.out_features

    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    gammas = []
    for m in modules:
        m.gamma.requires_grad_(True)
        gammas.append(m.gamma)

    opt = torch.optim.Adam(gammas, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=min_lr)

    def compression_terms():
        size_new = 0.0
        penalty = 0.0
        for m in modules:
            rr = _dobi_rank_ratio(m.in_features, m.out_features, seq_len)
            size_now = (m.in_features + m.out_features) * m.gamma * rr
            size_ori = float(m.in_features * m.out_features)
            size_new = size_new + torch.clamp(size_now, max=size_ori)
            penalty = penalty + torch.relu(-m.gamma) ** 2 \
                + torch.relu(m.gamma - seq_len) ** 2
        return size_new / dense_total, penalty

    print(f"\n[dobisvd] gamma search: {len(modules)} linears, {steps} steps, "
          f"lr {lr}->{min_lr}, lambda_reg {lambda_reg}")
    n = calib_ids.shape[0]
    for step in tqdm(range(steps), desc="[dobisvd] gamma"):
        ids = calib_ids[step % n: step % n + 1].to(device)
        out = model(input_ids=ids, labels=ids)
        ppl = torch.exp(out.loss)
        comp_ratio, penalty = compression_terms()
        reg = lambda_reg * torch.abs(comp_ratio - ratio)
        loss = ppl + reg + penalty

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if (step + 1) % log_every == 0 or step == 0:
            print(f"  step {step+1:>4}/{steps} | ppl {ppl.item():.3f} | "
                  f"ratio {comp_ratio.item():.4f} | reg {reg.item():.3f} | "
                  f"lr {opt.param_groups[0]['lr']:.3f}")

    gamma_of = {}
    for layer_i, layer in enumerate(layers):
        for p in layer_linear_paths(layer):
            m = get_module(layer, p)
            gamma_of[(layer_i, p)] = float(m.gamma.detach().item())
            set_module(layer, p, m.ori)  # restore the dense linear
    model.to("cpu")
    for p in model.parameters():
        p.requires_grad_(False)
    gc.collect()
    torch.cuda.empty_cache()
    return gamma_of


@torch.no_grad()
def dobi_ipca_update(model, gamma_of, calib_ids, device, seq_len, beta,
                     n_update_samples, ipca_accum=None):
    """Stage 2 of Dobi-SVD: IPCA weight update, then factorise at rank gamma.

    Collects the right singular vectors of each linear's *output activation*
    across the calibration set with incremental PCA, projects the weight onto
    the resulting principal subspace, and decomposes the updated weight.
    """
    stable_svd, IncrementalPCAonGPU = _import_dobi()
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype

    state = {}
    for layer_i, layer in enumerate(layers):
        for p in layer_linear_paths(layer):
            fc = get_module(layer, p)
            gamma = gamma_of[(layer_i, p)]
            cut = _dobi_rank_ratio(fc.in_features, fc.out_features, seq_len)
            n_acc = max(1, int(cut)) if ipca_accum is None else ipca_accum
            real_gamma = int(min(fc.out_features, fc.in_features,
                                 max(1, math.ceil(gamma) * max(1, int(cut)))))
            state[(layer_i, p)] = {
                "gamma": gamma,
                "cut": cut,
                "n_acc": n_acc,
                "ipca": IncrementalPCAonGPU(n_components=real_gamma),
                "buf": [],
                "components": None,
            }

    # -- pass 1: accumulate the activation subspace, block by block ---------
    kwargs_cache = {}

    def kwargs_fn(bs):
        if bs not in kwargs_cache:
            kwargs_cache[bs] = build_decoder_kwargs(seq_len, dtype, device, bs)
        return kwargs_cache[bs]

    ids = calib_ids[:n_update_samples]
    cache = HiddenCache.from_embeddings(model, ids, device)

    for layer_i, layer in enumerate(tqdm(layers, desc="[dobisvd] IPCA pass")):
        layer.to(device)
        paths = layer_linear_paths(layer)
        handles = []

        def make_hook(key):
            st = state[key]

            def hook(module, inputs, output):
                y = output.detach().float()
                if y.dim() == 3:
                    y = y.reshape(-1, y.shape[-1])
                q = int(min(y.shape[0], y.shape[1], max(1, math.ceil(st["gamma"]))))
                _, _, V = torch.svd_lowrank(y, q=q, niter=2)
                st["buf"].append(V.cpu())
                if len(st["buf"]) >= st["n_acc"]:
                    full = torch.cat(st["buf"][:st["n_acc"]], dim=1).to(device)
                    st["ipca"].partial_fit(full.t())
                    st["components"] = torch.as_tensor(st["ipca"].components_).cpu()
                    st["buf"] = []
                    del full
                    torch.cuda.empty_cache()
            return hook

        for p in paths:
            handles.append(get_module(layer, p).register_forward_hook(make_hook((layer_i, p))))

        outs = torch.empty_like(cache.data)
        for i in range(len(cache)):
            x = cache.data[i:i + 1].to(device)
            am, pid = kwargs_fn(1)
            outs[i:i + 1] = run_layer(layer, x, am, pid).cpu()
        for h in handles:
            h.remove()

        # -- pass 2 (same block): update + decompose ------------------------
        for p in paths:
            key = (layer_i, p)
            st = state[key]
            fc = get_module(layer, p)
            W = fc.weight.data.float()
            bias = fc.bias.data.clone() if fc.bias is not None else None

            gamma_dec = st["cut"] * st["gamma"]
            gamma_dec = st["cut"] * math.ceil(gamma_dec / st["cut"])
            rank = int(max(1, min(min(W.shape), round(gamma_dec))))

            if st["components"] is not None:
                V_pca = st["components"].t().to(device).float()      # (out, n_comp)
                seq = torch.arange(1, V_pca.shape[1] + 1, device=device, dtype=torch.float32)
                real_gamma = min(min(W.shape), max(1.0, gamma_dec))
                trunc = 0.5 * torch.tanh(beta * (real_gamma - seq)) + 0.5
                W = (W.t() @ V_pca * trunc) @ V_pca.t()
                W = W.t().contiguous()
                del V_pca
            st["ipca"] = None
            st["components"] = None

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            sqrt_s = S[:rank].clamp_min(0).sqrt()
            W_u = U[:, :rank] * sqrt_s
            W_v = sqrt_s.unsqueeze(1) * Vt[:rank, :]
            set_module(layer, p, _factored(W_u.to(dtype), W_v.to(dtype),
                                           None if bias is None else bias.to(dtype), rank))
            del W, U, S, Vt, W_u, W_v
            torch.cuda.empty_cache()

        layer.to("cpu")
        cache = HiddenCache(outs)
        del outs
        gc.collect()
        torch.cuda.empty_cache()
    return model


def compress_dobisvd(model, calib_ids, device, ratio, args):
    gamma_of = dobi_train_gamma(
        model, calib_ids[:args.dobi_train_samples], device, ratio,
        steps=args.dobi_steps, lr=args.dobi_lr, min_lr=args.dobi_min_lr,
        beta=args.dobi_beta, lambda_reg=args.dobi_lambda_reg, seq_len=args.seqlen)
    return dobi_ipca_update(model, gamma_of, calib_ids, device, args.seqlen,
                            args.dobi_beta, args.dobi_update_samples)


# ==========================================================================
# Entry point
# ==========================================================================
def run(args):
    seed_everything(args.seed)
    device = torch.device(args.device)
    print(f"=== baseline {args.method} | {args.model_id} | target ratio {args.ratio} ===")

    model, tokenizer = load_llm(args.model_id, args.dtype, args.seqlen)
    rotation.apply_rotation(model, args.rotate, args.rotate_seed, device)
    calib_ids = data_utils.get_calib_input_ids(
        tokenizer, nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
        dataset=args.calib_dataset, data_root=args.data_root)

    with Timer() as t:
        if args.method == "svdllm":
            compress_svdllm(model, calib_ids, device, args.ratio, args.calib_batch_size)
        elif args.method == "dobisvd":
            compress_dobisvd(model, calib_ids, device, args.ratio, args)
        else:
            raise ValueError(args.method)
    elapsed = t.elapsed

    report = param_report(model)
    print(f"\n[eval] linear params {report['linear_params']/1e9:.3f}B / "
          f"{report['dense_linear_params']/1e9:.3f}B = {report['linear_param_ratio']:.4f}")

    ppl = {}
    for ds in args.eval_datasets.split(","):
        ds = ds.strip()
        if not ds:
            continue
        test_ids = data_utils.get_test_input_ids(tokenizer, args.seqlen, ds, args.data_root)
        ppl[ds] = evaluate_ppl(model, test_ids, device, args.eval_batch_size, desc=f"ppl:{ds}")
        print(f"[eval] {ds} perplexity: {ppl[ds]:.4f}")

    payload = {
        "method": args.method,
        "model_id": args.model_id,
        "target_ratio": args.ratio,
        "ppl": ppl,
        "timings_sec": {"compress": elapsed},
        "config": vars(args),
        "layer_ranks": layer_rank_string(model),
        **report,
    }
    if args.out_json:
        save_result(args.out_json, payload)
    if args.save_model:
        torch.save({"model": model.cpu(), "tokenizer": tokenizer}, args.save_model)
    return payload


def build_parser():
    p = argparse.ArgumentParser(description="SVD-LLM / Dobi-SVD baselines")
    p.add_argument("--method", type=str, required=True, choices=["svdllm", "dobisvd"])
    p.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--ratio", type=float, default=0.8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--data_root", type=str, default=data_utils.DEFAULT_DATA_ROOT)
    p.add_argument("--calib_dataset", type=str, default="wikitext2")
    p.add_argument("--eval_datasets", type=str, default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--calib_batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=1)

    # Dobi-SVD stage hyper-parameters (defaults follow Dobi-SVD/svd_trainer.py,
    # except the step count, which is reduced to keep a single run tractable).
    p.add_argument("--dobi_steps", type=int, default=64)
    p.add_argument("--dobi_train_samples", type=int, default=32)
    p.add_argument("--dobi_update_samples", type=int, default=8)
    p.add_argument("--dobi_lr", type=float, default=5.0)
    p.add_argument("--dobi_min_lr", type=float, default=4.0)
    p.add_argument("--dobi_beta", type=float, default=10.0)
    p.add_argument("--dobi_lambda_reg", type=float, default=200.0)

    # Off by default so both baselines stay in their published form; turn it on
    # for the rotation ablation (it is *not* a no-op for dobisvd, which truncates
    # in activation space without whitening).
    p.add_argument("--rotate", type=str, default="none", choices=["none", "hadamard", "random"])
    p.add_argument("--rotate_seed", type=int, default=0)

    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--save_model", type=str, default="")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
