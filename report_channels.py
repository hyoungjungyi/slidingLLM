"""Terminal report: compare per-channel activation stats across measurement runs.

  python report_channels.py figs/acts_base.npz figs/acts_seed0.npz figs/acts_seed7.npz
  python report_channels.py --labels dense,uniform,learned \
      figs/acts_base.npz figs/acts_svd0.6_uniform.npz figs/acts_svd0.6_learned.npz
"""
import argparse
import os

import numpy as np

WATCH = [2533, 1415, 1512]


def peak_trace(d, ch):
    """Signed value of `ch` on the token carrying the spike, per boundary."""
    top = list(d["top_channels"])
    if ch not in top:
        return None
    V = d["traces"][:, :, :, top.index(ch)]
    pk = np.abs(V[V.shape[0] // 2]).argmax(axis=1)
    return np.stack([[V[b, s, pk[s]] for s in range(V.shape[1])]
                     for b in range(V.shape[0])]).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--labels", default="", help="comma-separated, one per npz")
    ap.add_argument("--ref", type=int, default=0, help="index of the reference run")
    ap.add_argument("--topn", type=int, default=8)
    args = ap.parse_args()

    labs = args.labels.split(",") if args.labels else \
        [os.path.basename(p).replace("acts_", "").replace(".npz", "") for p in args.npz]
    runs = [(l, np.load(p)) for l, p in zip(labs, args.npz)]
    W = max(13, max(len(l) for l in labs) + 2)
    n_b = runs[0][1]["absmax"].shape[0]

    print("\n" + "=" * 78)
    print("TOP CHANNELS BY PEAK |h| OVER THE WHOLE STACK")
    print("=" * 78)
    for lab, d in runs:
        cp = d["absmax"].max(0)
        order = np.argsort(-cp)[:args.topn]
        print(f"  {lab:>{W}} : " + "  ".join(f"{c}({cp[c]:,.0f})" for c in order))

    print("\n" + "=" * 78)
    print("WATCHED CHANNELS — peak |h|")
    print("=" * 78)
    print(f"  {'run':>{W}} " + "".join(f"{'ch '+str(c):>14}" for c in WATCH))
    for lab, d in runs:
        cp = d["absmax"].max(0)
        print(f"  {lab:>{W}} " + "".join(f"{cp[c]:>14,.1f}" for c in WATCH))
    ref_cp = runs[args.ref][1]["absmax"].max(0)
    for lab, d in runs[1:] if args.ref == 0 else []:
        cp = d["absmax"].max(0)
        print(f"  {'vs ' + labs[args.ref]:>{W}} " +
              "".join(f"{(cp[c]/ref_cp[c]-1)*100:>+13.1f}%" for c in WATCH)
              + f"   ({lab})")

    print("\n" + "=" * 78)
    print(f"CHANNEL {WATCH[0]} THROUGH THE CANCELLATION  (signed, peak token)")
    print("=" * 78)
    bs = list(range(n_b - 5, n_b))
    print(f"  {'run':>{W}} " + "".join(f"{'bdry '+str(b):>12}" for b in bs))
    for lab, d in runs:
        v = peak_trace(d, WATCH[0])
        if v is None:
            continue
        print(f"  {lab:>{W}} " + "".join(f"{v[b]:>+12,.1f}" for b in bs))

    print("\n" + "=" * 78)
    print(f"PER-CHANNEL AGREEMENT WITH '{labs[args.ref]}'  (all 4,096 channels)")
    print("=" * 78)
    for lab, d in runs:
        if lab == labs[args.ref]:
            continue
        cp = d["absmax"].max(0)
        rel = np.abs(cp - ref_cp) / np.maximum(ref_cp, 1e-6)
        same_top3 = set(np.argsort(-cp)[:3]) == set(np.argsort(-ref_cp)[:3])
        new = np.where((cp > 200) & (ref_cp < 200))[0]
        print(f"  {lab:>{W}} : median {np.median(rel)*100:5.1f}%   p95 {np.quantile(rel,.95)*100:5.1f}%"
              f"   max {rel.max()*100:5.1f}%   moved>50%: {int((rel>0.5).sum()):>4}/4096"
              f"   top-3 identical: {'YES' if same_top3 else 'NO'}"
              + (f"   NEW>200: {new.tolist()[:6]}" if len(new) else ""))
    print()


if __name__ == "__main__":
    main()
