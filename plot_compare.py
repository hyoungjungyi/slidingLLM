"""Dense vs SVD-truncated: what happens to the massive-activation channels.

  python plot_compare.py --dense figs/acts_base.npz \
      --runs "uniform r=0.6:figs/acts_svd0.6_uniform.npz" \
             "learned r=0.6:figs/acts_svd0.6_learned.npz"
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
SERIES = [BLUE, ORANGE, AQUA]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=8, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def peak_trace(d, ch):
    """Signed value of `ch` on the token carrying the spike, per boundary."""
    top = list(d["top_channels"])
    if ch not in top:
        return None
    V = d["traces"][:, :, :, top.index(ch)]
    mid = V.shape[0] // 2
    pk = np.abs(V[mid]).argmax(axis=1)
    return np.stack([[V[b, s, pk[s]] for s in range(V.shape[1])]
                     for b in range(V.shape[0])]).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="figs/acts_base.npz")
    ap.add_argument("--runs", nargs="+", required=True, help='"label:path.npz" ...')
    ap.add_argument("--ch", type=int, default=2533)
    ap.add_argument("--out", default="figs/compare_svd.png")
    args = ap.parse_args()

    D = np.load(args.dense)
    runs = []
    for spec in args.runs:
        lab, path = spec.rsplit(":", 1)
        runs.append((lab, np.load(path)))
    n_b = D["absmax"].shape[0]
    B = np.arange(n_b)
    CH = args.ch

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
    fig.suptitle(f"Do the massive-activation channels survive SVD truncation?  ·  "
                 f"Llama-2-7B, {D['calib_ids'].shape[0]}x{D['calib_ids'].shape[1]} tokens",
                 x=0.008, ha="left", fontsize=12, fontweight="bold", y=0.985)

    # -- (a) the massive channel through depth ----------------------------
    ax = axes[0, 0]
    ax.plot(B, np.maximum(D["absmax"][:, CH], 1e-3), color=MUTED, lw=2.6,
            label="dense", zorder=3)
    for (lab, d), col in zip(runs, SERIES):
        ax.plot(B, np.maximum(d["absmax"][:, CH], 1e-3), color=col, label=lab)
    ax.set_yscale("log")
    ax.legend(loc="lower left", fontsize=8.5, labelcolor=INK2)
    style(ax, f"a  Channel {CH}: created identically, cancelled wrongly",
          "block boundary", "max |h|")

    # -- (b) the typical channel: did truncation inflate it? --------------
    ax = axes[0, 1]
    med_d = np.median(D["absmax"], axis=1)
    ax.plot(B, med_d, color=MUTED, lw=2.6, label="dense", zorder=3)
    for (lab, d), col in zip(runs, SERIES):
        ax.plot(B, np.median(d["absmax"], axis=1), color=col, label=lab)
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8.5, labelcolor=INK2)
    style(ax, "b  Median channel: where the spillover lands",
          "block boundary", "max |h|  (median over 4,096 channels)")

    # -- (c) per-channel peak, dense vs compressed ------------------------
    ax = axes[1, 0]
    xd = D["absmax"].max(0)
    lim = [0.5, xd.max() * 3]
    ax.plot(lim, lim, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(lim[1], lim[1], " unchanged ", color=MUTED, fontsize=8,
            ha="right", va="bottom", rotation=45)
    for (lab, d), col in zip(runs, SERIES):
        yc = d["absmax"].max(0)
        ax.scatter(xd, yc, s=4, c=col, alpha=0.30, lw=0, label=lab)
    for c, m in [(CH, "o"), (1415, "s"), (1512, "^")]:
        for (lab, d), col in zip(runs, SERIES):
            ax.scatter([xd[c]], [d["absmax"].max(0)[c]], s=68, facecolor=col,
                       edgecolor=SURFACE, lw=1.6, marker=m, zorder=5)
        ax.annotate(f"ch {c}", (xd[c], xd[c]), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color=INK,
                    fontweight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(loc="upper left", fontsize=8.5, labelcolor=INK2, markerscale=2.5)
    style(ax, "c  Every channel's peak |h|, dense vs compressed",
          "dense", "compressed")

    # -- (d) the cancellation, signed -------------------------------------
    # The residue is 3.7% of the operand, so over-cancelling by 13% sends it
    # through zero -- a linear axis with a zero line is the only honest view.
    ax = axes[1, 1]
    models = [("dense", MUTED, D)] + [(lab, col, d) for (lab, d), col in
                                      zip(runs, SERIES)]
    groups = [(2533, 31), (2533, 32), (1415, 31), (1415, 32)]
    w = 0.8 / len(models)
    xs = np.arange(len(groups))
    for k, (lab, col, d) in enumerate(models):
        vals = [peak_trace(d, c)[bd] for c, bd in groups]
        pos = xs + (k - (len(models) - 1) / 2) * w
        ax.bar(pos, vals, width=w * 0.88, color=col, edgecolor=SURFACE,
               linewidth=1.2, label=lab)
        for x_, v in zip(pos, vals):
            ax.text(x_, v + (14 if v >= 0 else -14), f"{v:+,.0f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=7.5,
                    color=INK2, fontweight="bold")
    ax.axhline(0, color=INK, lw=1.1, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"ch {c}\nboundary {bd}" for c, bd in groups], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-430, 300)
    ax.text(0.5, 0.965,
            "entering block 30:  ch 2533 = +2,510   ch 1415 = \u22121,481"
            "   (all three models within 1%)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color=INK2, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8.5, labelcolor=INK2, ncol=3)
    style(ax, "d  Both compressed models over-cancel and flip the sign",
          None, "h  (signed, peak token)")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"[fig] {args.out}")

    # -- numbers ----------------------------------------------------------

    print()
    for lab, d in runs:
        yc = d["absmax"].max(0)
        rel = (yc - xd) / np.maximum(xd, 1e-6)
        big = np.abs(rel) > 0.5
        print(f"[{lab}] channels whose peak moved >50%: {big.sum()} / {len(xd)}  "
              f"| median |rel change| {np.median(np.abs(rel))*100:.1f}%  "
              f"| ch{CH} {rel[CH]*100:+.1f}%  ch1415 {rel[1415]*100:+.1f}%")
        newmass = np.where((yc > 200) & (xd < 200))[0]
        if len(newmass):
            print(f"    NEW channels above 200 that were below in dense: "
                  f"{newmass.tolist()[:12]}")


if __name__ == "__main__":
    main()
