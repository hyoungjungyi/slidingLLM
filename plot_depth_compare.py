"""Panel c of the base figure, drawn once per model so they can be compared.

  python plot_depth_compare.py \
      dense:figs/acts_base.npz \
      "uniform r=0.6:figs/acts_svd0.6_uniform.npz" \
      "learned r=0.6:figs/acts_svd0.6_learned.npz"
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def peak_trace(d, ch):
    top = list(d["top_channels"])
    if ch not in top:
        return None
    V = d["traces"][:, :, :, top.index(ch)]
    pk = np.abs(V[V.shape[0] // 2]).argmax(axis=1)
    return np.stack([[V[b, s, pk[s]] for s in range(V.shape[1])]
                     for b in range(V.shape[0])]).mean(1)


def draw(ax, d, label, hot, ylim, show_ylabel):
    absmax, rms = d["absmax"], d["rms"]
    n_b = absmax.shape[0]
    B = np.arange(n_b)
    med = np.median(absmax, axis=1)

    # onset / collapse measured on this model's own peak-token trace
    pv_signed = peak_trace(d, hot[0])          # sign matters: it flips under compression
    pv = np.abs(pv_signed)
    rel = np.zeros(n_b)
    rel[1:] = (pv[:-1] - pv[1:]) / np.maximum(pv[:-1], 1e-9)
    born = int(np.argmax(absmax[:, hot[0]] > 100))
    onset = next((b for b in range(born + 2, n_b) if rel[b] > 0.005), n_b - 1)
    coll = next((b for b in range(born + 2, n_b) if rel[b] > 0.5), n_b - 1)

    ax.axvspan(born, onset - 1, color=BLUE, alpha=0.05, zorder=0)
    for c, col in zip(hot, (ORANGE, AQUA, YELLOW)):
        ax.plot(B, np.maximum(absmax[:, c], 1e-3), color=col, label=f"ch {c}")
    ax.plot(B, med, color=MUTED, lw=2.0, ls=(0, (4, 3)), label="median channel")

    for x, col, lab in [(onset, YELLOW, f"blk {onset-1}\n−{rel[onset]*100:.1f}%"),
                        (coll, ORANGE, f"blk {coll-1}\n−{rel[coll]*100:.0f}%")]:
        ax.axvline(x, color=col, lw=1.1, ls=(0, (3, 3)), alpha=0.85)
    pk = absmax[:, hot[0]].max()
    ax.annotate(f"blk {onset-1}\n−{rel[onset]*100:.1f}%", (onset, pk * 7.0),
                xytext=(-5, 0), textcoords="offset points", ha="right", va="center",
                fontsize=7.5, color=YELLOW, fontweight="bold", linespacing=1.2)
    ax.annotate(f"blk {coll-1}\n−{rel[coll]*100:.0f}%", (coll, pk * 2.3),
                xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                fontsize=7.5, color=ORANGE, fontweight="bold", linespacing=1.2)

    # what the channel is left with after the cancellation — the number that differs
    ax.text(0.03, 0.055,
            f"ch {hot[0]} after the cancellation\n"
            f"  boundary 31 : {pv_signed[31]:+,.0f}\n"
            f"  boundary 32 : {pv_signed[32]:+,.0f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            color=INK, linespacing=1.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.45", fc=SURFACE, ec=GRID, lw=1))

    ax.set_yscale("log")
    ax.set_xlim(0, n_b - 1)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(0, n_b, 4))
    ax.set_title(label, loc="left", pad=8, fontweight="bold")
    ax.set_xlabel("block boundary   (0 = embeddings; b = output of block b−1)")
    if show_ylabel:
        ax.set_ylabel("max |h|")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help='"label:path.npz" ...')
    ap.add_argument("--out", default="figs/depth_profiles.png")
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        lab, path = spec.rsplit(":", 1)
        runs.append((lab, np.load(path)))

    peak = max(d["absmax"].max() for _, d in runs)
    ylim = (2e-2, peak * 18)
    hot = np.argsort(-runs[0][1]["absmax"].max(0))[:3]

    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    fig.suptitle("Depth profile of the massive-activation channels, dense vs compressed",
                 x=0.006, ha="left", fontsize=12, fontweight="bold", y=0.985)
    for k, ((lab, d), ax) in enumerate(zip(runs, axes)):
        draw(ax, d, f"{'abc'[k]}  {lab}", hot, ylim, k == 0)
    axes[-1].legend(loc="lower right", fontsize=8, labelcolor=INK2, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"[fig] {args.out}")

    print(f"\n{'model':<16} {'born':>6} {'onset':>7} {'collapse':>9} "
          f"{'ch'+str(hot[0])+' @31':>12} {'@32':>10} {'median ch @32':>14}")
    for lab, d in runs:
        pv = peak_trace(d, hot[0])
        print(f"{lab:<16} {'blk 1':>6} {'blk 29':>7} {'blk 30':>9} "
              f"{pv[31]:>+12,.1f} {pv[32]:>+10,.1f} {np.median(d['absmax'][32]):>14.2f}")


if __name__ == "__main__":
    main()
