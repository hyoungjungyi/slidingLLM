"""Open up the Stage-3 window loss: what is actually being subtracted from what.

For every window it reproduces exactly what `sliding_window_finetune` computes —

    x = student blocks [i, i+w) applied to the student's accumulated activation
    t = teacher blocks [i, i+w) applied to the *teacher's* activation
    loss = mean((x-t)^2) / mean(t^2)            # feature_mse(..., rel_mse=True)

— and then decomposes it, so you can see which channels and which token
positions the number is made of, and how that changes with depth.

No training, no gradients: one forward sweep.

  python inspect_window_mse.py --load_sliced ckpt/r0.6_sliced_LOADRANKS_no_stage2.pt \
      --nsamples 8 --device cuda:0 --out figs/window_mse.npz
"""
import argparse
import gc
import json
import os

import numpy as np
import torch

import data_utils
import rotation
from llm_utils import get_layers, load_llm, run_layer, seed_everything
from sliding_llm import (WindowRunner, load_sliced, make_windows, slice_ranks,
                         build_svd_student, load_ranks_into_gates, iter_gates,
                         hard_size_ratio)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="meta-llama/Llama-2-7b-hf")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--nsamples", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default=data_utils.DEFAULT_DATA_ROOT)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--load_sliced", default="", help="a --save_sliced checkpoint")
    ap.add_argument("--ratio", type=float, default=0.6, help="if building instead of loading")
    ap.add_argument("--ranks_json", default="", help="if building instead of loading")
    ap.add_argument("--topk", type=int, default=6, help="channels/tokens to report per window")
    ap.add_argument("--out", default="figs/window_mse.npz")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)

    student, tok = load_llm(args.model_id, args.dtype, args.seqlen)
    calib = data_utils.get_calib_input_ids(tok, nsamples=args.nsamples,
                                           seqlen=args.seqlen, seed=args.seed,
                                           data_root=args.data_root)
    if args.load_sliced:
        print(f"[load] {args.load_sliced}")
        load_sliced(student, args.load_sliced)
    else:
        print("[build] Stage-0 SVD, then slice (no checkpoint given)")
        build_svd_student(student, calib, device, batch_size=1, init_ratio=args.ratio)
        if args.ranks_json:
            load_ranks_into_gates(student, args.ranks_json)
        print(f"[build] hard ratio {hard_size_ratio(list(iter_gates(student))):.4f}")
        slice_ranks(student)
    gc.collect(); torch.cuda.empty_cache()

    teacher, _ = load_llm(args.model_id, args.dtype, args.seqlen)
    n_layers = len(get_layers(student))
    windows = make_windows(n_layers, args.window, args.stride)
    runner = WindowRunner(teacher, student, calib, device, batch_size=1)
    runner.reset()

    hid = student.config.hidden_size
    ns, sl = calib.shape
    rec = {k: [] for k in ("loss", "num", "den", "err_chan", "err_tok",
                           "tgt_chan", "start", "exit")}

    print(f"\n{'win':>3} {'blocks':>8} {'exit':>5} | {'rel mse':>10} {'raw mse':>11} "
          f"{'E[t^2]':>10} | {'top-3 channels by error share':<34} {'top tok':>8}")
    print("-" * 118)

    for w_idx, (i, w) in enumerate(windows):
        runner.advance_to(i)
        target = runner.teacher_window_out(i, w)
        blocks = runner.student_window(i, w)

        num = torch.zeros(hid, dtype=torch.float64)      # sum (x-t)^2 per channel
        den = torch.zeros(hid, dtype=torch.float64)      # sum t^2     per channel
        etok = torch.zeros(ns, sl, dtype=torch.float64)  # mean_c (x-t)^2 per token
        for b in range(ns):
            x = runner.s_cache.data[b:b + 1].to(device)
            t = target.data[b:b + 1].to(device)
            am, pid = runner.kwargs_fn(1)
            for blk in blocks:
                x = run_layer(blk, x, am, pid)
            d2 = (x.float() - t.float()).pow(2)[0]       # (seqlen, hid)
            num += d2.sum(0).double().cpu()
            den += t.float()[0].pow(2).sum(0).double().cpu()
            etok[b] = d2.mean(-1).double().cpu()

        n_elem = ns * sl
        raw = float(num.sum() / (n_elem * hid))
        tgt_e = float(den.sum() / (n_elem * hid))
        rel = raw / (tgt_e + 1e-8)

        share = (num / num.sum()).numpy()
        top_c = np.argsort(-share)[:args.topk]
        flat = etok.numpy().ravel()
        top_t = np.argsort(-flat)[:args.topk]
        s0, t0 = divmod(int(top_t[0]), sl)

        for k, v in [("loss", rel), ("num", num.numpy()), ("den", den.numpy()),
                     ("err_chan", share), ("err_tok", etok.numpy().astype(np.float32)),
                     ("tgt_chan", (den / den.sum()).numpy()),
                     ("start", i), ("exit", i + w)]:
            rec[k].append(v)

        chan_str = "  ".join(f"{c}:{share[c]*100:.0f}%" for c in top_c[:3])
        print(f"{w_idx:>3} {f'{i}-{i+w-1}':>8} {i+w:>5} | {rel:>10.5f} {raw:>11.2f} "
              f"{tgt_e:>10.2f} | {chan_str:<34} "
              f"s{s0}p{t0}={tok.decode([int(calib[s0, t0])])!r}")

        runner.student_window(i, w, to_device=False)
        del target
        gc.collect(); torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out,
        loss=np.array(rec["loss"]), start=np.array(rec["start"]),
        exit=np.array(rec["exit"]), err_chan=np.stack(rec["err_chan"]),
        tgt_chan=np.stack(rec["tgt_chan"]), num=np.stack(rec["num"]),
        den=np.stack(rec["den"]), err_tok=np.stack(rec["err_tok"]),
        calib_ids=calib.numpy().astype(np.int32))
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
