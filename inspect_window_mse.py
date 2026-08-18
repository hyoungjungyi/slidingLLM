"""Show the actual numbers the Stage-3 window loss is computed from.

For each window it reproduces exactly what `sliding_window_finetune` does —

    x = student blocks [i, i+w) applied to the student's accumulated activation
    t = teacher  blocks [i, i+w) applied to the teacher's activation
    loss = mean((x - t)^2)                 # feature_mse(..., rel_mse=False)

— and then prints the teacher value, the student value and their difference for
the entries that dominate that sum, so you can see what the MSE is made of
rather than just its value.

No training, no gradients: one forward sweep.

  python inspect_window_mse.py --load_sliced ckpt/r0.6_sliced_LOADRANKS_no_stage2.pt \
      --nsamples 8 --device cuda:6 --out figs/window_mse_r0.6.npz
"""
import argparse
import gc
import os

import numpy as np
import torch

import data_utils
from llm_utils import get_layers, load_llm, run_layer, seed_everything
from sliding_llm import (WindowRunner, build_svd_student, hard_size_ratio,
                         iter_gates, load_ranks_into_gates, load_sliced,
                         make_windows, slice_ranks)


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
    ap.add_argument("--load_sliced", default="")
    ap.add_argument("--ratio", type=float, default=0.6)
    ap.add_argument("--ranks_json", default="")
    ap.add_argument("--topk", type=int, default=8,
                    help="how many largest |x-t| entries to print per window")
    ap.add_argument("--watch", default="2533,1415,1512",
                    help="channels to always print teacher/student values for")
    ap.add_argument("--dump_window", type=int, default=-1,
                    help="also save the full teacher/student tensors for this window")
    ap.add_argument("--out", default="figs/window_mse.npz")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    watch = [int(c) for c in args.watch.split(",") if c.strip()]

    student, tok = load_llm(args.model_id, args.dtype, args.seqlen)
    calib = data_utils.get_calib_input_ids(tok, nsamples=args.nsamples,
                                           seqlen=args.seqlen, seed=args.seed,
                                           data_root=args.data_root)
    if args.load_sliced:
        print(f"[load] {args.load_sliced}")
        load_sliced(student, args.load_sliced)
    else:
        print("[build] Stage-0 SVD, then slice")
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
    n_elem = ns * sl * hid
    out = {k: [] for k in ("mse", "start", "exit", "t_absmean", "x_absmean",
                           "t_absmax", "x_absmax", "chan_sq", "chan_t_sq",
                           "watch_t", "watch_x")}

    for w_idx, (i, w) in enumerate(windows):
        runner.advance_to(i)
        target = runner.teacher_window_out(i, w)
        blocks = runner.student_window(i, w)

        chan_sq = torch.zeros(hid, dtype=torch.float64)
        chan_t_sq = torch.zeros(hid, dtype=torch.float64)
        t_abs = x_abs = 0.0
        t_max = x_max = 0.0
        # biggest |x-t| entries seen so far: (diff, sample, pos, chan, t, x)
        worst = []
        # teacher/student value on the token that carries the largest error,
        # for the watched channels
        wt = np.zeros((len(watch), ns), dtype=np.float32)
        wx = np.zeros((len(watch), ns), dtype=np.float32)

        for b in range(ns):
            x = runner.s_cache.data[b:b + 1].to(device)
            t = target.data[b:b + 1].to(device)
            am, pid = runner.kwargs_fn(1)
            for blk in blocks:
                x = run_layer(blk, x, am, pid)
            xf, tf = x.float()[0], t.float()[0]          # (seqlen, hid)
            d = xf - tf

            chan_sq += d.pow(2).sum(0).double().cpu()
            chan_t_sq += tf.pow(2).sum(0).double().cpu()
            t_abs += float(tf.abs().sum()); x_abs += float(xf.abs().sum())
            t_max = max(t_max, float(tf.abs().max()))
            x_max = max(x_max, float(xf.abs().max()))

            flat = d.abs().flatten()
            k = min(args.topk, flat.numel())
            v, idx = torch.topk(flat, k)
            for vv, ii in zip(v.tolist(), idx.tolist()):
                p, c = divmod(ii, hid)
                worst.append((vv, b, p, c, float(tf[p, c]), float(xf[p, c])))

            # the token where this sample's error is largest, watched channels
            tokerr = d.pow(2).mean(-1)
            p0 = int(tokerr.argmax())
            for j, c in enumerate(watch):
                wt[j, b] = float(tf[p0, c]); wx[j, b] = float(xf[p0, c])

            if w_idx == args.dump_window and b == 0:
                np.savez_compressed(args.out.replace(".npz", f"_win{w_idx}_s0.npz"),
                                    teacher=tf.cpu().numpy().astype(np.float32),
                                    student=xf.cpu().numpy().astype(np.float32))

        mse = float(chan_sq.sum()) / n_elem
        worst.sort(reverse=True)

        print(f"\n{'='*96}")
        print(f"window {w_idx}   blocks {i}-{i+w-1}   target = teacher at boundary {i+w}")
        print(f"{'='*96}")
        print(f"  raw MSE = mean((x-t)^2) = {mse:,.4f}        "
              f"[rel = {mse/(float(chan_t_sq.sum())/n_elem):.5f}]")
        print(f"  teacher : mean|t| {t_abs/n_elem:8.3f}   max|t| {t_max:10,.1f}")
        print(f"  student : mean|x| {x_abs/n_elem:8.3f}   max|x| {x_max:10,.1f}")
        print(f"\n  largest |x - t| entries:")
        print(f"    {'smp':>4} {'pos':>5} {'chan':>5} {'teacher':>12} {'student':>12} "
              f"{'diff':>12}  token")
        seen = set()
        for vv, b, p, c, tv, xv in worst:
            if (b, p, c) in seen:
                continue
            seen.add((b, p, c))
            print(f"    {b:>4} {p:>5} {c:>5} {tv:>+12,.1f} {xv:>+12,.1f} {xv-tv:>+12,.1f}"
                  f"  {tok.decode([int(calib[b, p])])!r}")
            if len(seen) >= args.topk:
                break
        print(f"\n  watched channels, on each sample's worst token:")
        print(f"    {'chan':>5} " + " ".join(f"{'s'+str(b):>9}" for b in range(ns)))
        for j, c in enumerate(watch):
            print(f"    {c:>5} t" + " ".join(f"{wt[j, b]:>+9,.0f}" for b in range(ns)))
            print(f"    {'':>5} x" + " ".join(f"{wx[j, b]:>+9,.0f}" for b in range(ns)))

        for k, v in [("mse", mse), ("start", i), ("exit", i + w),
                     ("t_absmean", t_abs / n_elem), ("x_absmean", x_abs / n_elem),
                     ("t_absmax", t_max), ("x_absmax", x_max),
                     ("chan_sq", chan_sq.numpy()), ("chan_t_sq", chan_t_sq.numpy()),
                     ("watch_t", wt), ("watch_x", wx)]:
            out[k].append(v)

        runner.student_window(i, w, to_device=False)
        del target
        gc.collect(); torch.cuda.empty_cache()

    print(f"\n{'='*96}\nSUMMARY — raw MSE per window (this is what Stage 3 minimises)\n{'='*96}")
    print(f"{'win':>4} {'blocks':>9} {'exit':>5} {'raw MSE':>12} {'mean|t|':>10} "
          f"{'mean|x|':>10} {'max|t|':>11} {'max|x|':>11}")
    for k in range(len(out["mse"])):
        blocks = f"{out['start'][k]}-{out['exit'][k] - 1}"
        print(f"{k:>4} {blocks:>9} {out['exit'][k]:>5} {out['mse'][k]:>12,.4f} "
              f"{out['t_absmean'][k]:>10.3f} {out['x_absmean'][k]:>10.3f} "
              f"{out['t_absmax'][k]:>11,.0f} {out['x_absmax'][k]:>11,.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, calib_ids=calib.numpy().astype(np.int32),
                        watch=np.array(watch),
                        **{k: np.array(v) if np.isscalar(v[0]) else np.stack(v)
                           for k, v in out.items()})
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
