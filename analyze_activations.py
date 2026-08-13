"""Measure per-channel activation magnitudes in the residual stream.

Answers "which channels carry the massive activations, and where do they appear
in depth / in the sequence" — the thing that makes an unnormalised feature-MSE
blow up in the deep sliding windows.

Two streaming passes over the decoder stack (blocks live on CPU, one on GPU at a
time), so a 7B model fits on a 24GB card:

  pass 1  per-boundary, per-channel |h| statistics over the whole calib set
          -> absmax_c, rms_c, plus per-token peak magnitude / argmax channel
  pass 2  full depth x token traces for the top-K channels found in pass 1

"Boundary b" = the residual stream *entering* block b; boundary `n_layers` is the
output of the last block (pre final-norm).

  python analyze_activations.py --nsamples 8 --out figs/acts_base.npz
  python analyze_activations.py --rotate hadamard --out figs/acts_hadamard.npz
"""
import argparse
import gc
import os

import numpy as np
import torch
from tqdm import tqdm

import data_utils
import rotation
from llm_utils import (build_decoder_kwargs, get_embed_tokens, get_layers,
                       load_llm, run_layer, seed_everything)


@torch.no_grad()
def _stream(model, calib_ids, device, boundary_fn, batch_size=1):
    """Push the calib set through the stack, calling `boundary_fn(b, h)` at each
    residual-stream boundary. `h` is (bs, seqlen, hidden) on GPU."""
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    seqlen = calib_ids.shape[1]
    kw = {}

    def kwargs_fn(bs):
        if bs not in kw:
            kw[bs] = build_decoder_kwargs(seqlen, dtype, device, bs)
        return kw[bs]

    embed = get_embed_tokens(model).to(device)
    cache = torch.cat([embed(calib_ids[i:i + 1].to(device)).cpu()
                       for i in range(calib_ids.shape[0])], dim=0)
    embed.to("cpu")

    for i in range(0, cache.shape[0], batch_size):
        boundary_fn(0, i, cache[i:i + batch_size].to(device))

    for b, layer in enumerate(tqdm(layers, desc="  stream", leave=False)):
        layer.to(device)
        out = torch.empty_like(cache)
        for i in range(0, cache.shape[0], batch_size):
            x = cache[i:i + batch_size].to(device)
            am, pid = kwargs_fn(x.shape[0])
            y = run_layer(layer, x, am, pid)
            boundary_fn(b + 1, i, y)
            out[i:i + batch_size] = y.cpu()
        layer.to("cpu")
        cache = out
        torch.cuda.empty_cache()
    del cache
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8, help="channels to trace in pass 2")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", type=str, default=data_utils.DEFAULT_DATA_ROOT)
    ap.add_argument("--rotate", type=str, default="none",
                    choices=["none", "hadamard", "random"])
    ap.add_argument("--rotate_seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="figs/acts_base.npz")
    # -- optional: measure a *compressed* model instead of the dense one -----
    ap.add_argument("--compress", type=float, default=0.0,
                    help="apply the Stage-0 whitened SVD and hard-truncate every "
                         "linear to this parameter ratio before measuring")
    ap.add_argument("--ranks_json", type=str, default="",
                    help="a results JSON whose learned `layer_ranks` to truncate to, "
                         "instead of the uniform rank implied by --compress")
    ap.add_argument("--calib_nsamples", type=int, default=128,
                    help="sequences for the Stage-0 whitening covariance")
    ap.add_argument("--channels", type=str, default="",
                    help="comma-separated channels to trace, instead of the top-k "
                         "(use to compare the same channels across runs)")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    model, tok = load_llm(args.model_id, args.dtype, args.seqlen)
    if args.rotate != "none":
        rotation.apply_rotation(model, args.rotate, args.rotate_seed, device)
    calib_ids = data_utils.get_calib_input_ids(
        tok, nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
        data_root=args.data_root)

    if args.compress > 0 or args.ranks_json:
        import json

        from llm_utils import get_module, layer_linear_paths
        from sliding_llm import build_svd_student, hard_size_ratio, iter_gates, slice_ranks

        ratio = args.compress if args.compress > 0 else 0.6
        svd_ids = data_utils.get_calib_input_ids(
            tok, nsamples=args.calib_nsamples, seqlen=args.seqlen, seed=args.seed,
            data_root=args.data_root)
        print(f"[compress] Stage-0 whitened SVD on {args.calib_nsamples} sequences")
        build_svd_student(model, svd_ids, device, batch_size=1, init_ratio=ratio)
        del svd_ids
        gc.collect()
        torch.cuda.empty_cache()

        if args.ranks_json:
            learned = [int(x) for x in
                       json.load(open(args.ranks_json))["layer_ranks"].split(",")]
            n = 0
            for layer in get_layers(model):
                for path in layer_linear_paths(layer):
                    get_module(layer, path).k.data.fill_(float(learned[n]))
                    n += 1
            print(f"[compress] loaded {n} learned ranks from {args.ranks_json}")

        print(f"[compress] hard ratio before slicing: "
              f"{hard_size_ratio(list(iter_gates(model))):.4f}")
        slice_ranks(model)
        gc.collect()
        torch.cuda.empty_cache()

    n_b = len(get_layers(model)) + 1
    hid = model.config.hidden_size
    ns, sl = calib_ids.shape

    absmax = torch.zeros(n_b, hid)                 # max_t |h[t, c]|
    sqsum = torch.zeros(n_b, hid, dtype=torch.float64)
    tok_peak = torch.zeros(n_b, ns, sl)            # max_c |h[t, c]|
    tok_argc = torch.zeros(n_b, ns, sl, dtype=torch.long)

    print(f"[pass 1/2] per-channel statistics ({ns} x {sl} tokens, {n_b} boundaries)")

    def stats(b, i, h):
        a = h.detach().float().abs()               # (bs, sl, hid)
        flat = a.reshape(-1, hid)
        absmax[b] = torch.maximum(absmax[b], flat.max(0).values.cpu())
        sqsum[b] += flat.pow(2).sum(0).double().cpu()
        pk, ac = a.max(-1)
        tok_peak[b, i:i + h.shape[0]] = pk.cpu()
        tok_argc[b, i:i + h.shape[0]] = ac.cpu()

    _stream(model, calib_ids, device, stats, args.batch_size)
    rms = (sqsum / (ns * sl)).sqrt().float()

    # Channels ranked by their peak magnitude anywhere in the stack.
    chan_peak = absmax.max(0).values
    if args.channels:
        top = torch.tensor(sorted(int(c) for c in args.channels.split(",")))
        args.topk = len(top)
    else:
        top = torch.topk(chan_peak, args.topk).indices.sort().values
    print(f"[pass 1/2] top-{args.topk} channels: {top.tolist()}")
    print(f"           peaks: {[round(float(chan_peak[c]), 1) for c in top]}")

    print(f"[pass 2/2] depth x token traces for {args.topk} channels")
    traces = torch.zeros(n_b, ns, sl, args.topk)
    top_dev = top.to(device)

    def trace(b, i, h):
        traces[b, i:i + h.shape[0]] = h.detach().float()[..., top_dev].cpu()

    _stream(model, calib_ids, device, trace, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        absmax=absmax.numpy(), rms=rms.numpy(),
        tok_peak=tok_peak.numpy(), tok_argc=tok_argc.numpy().astype(np.int32),
        traces=traces.numpy(), top_channels=top.numpy(),
        calib_ids=calib_ids.numpy().astype(np.int32),
        rotate=args.rotate, model_id=args.model_id,
    )
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
