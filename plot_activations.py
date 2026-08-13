"""Figures for the residual-stream massive-activation analysis.

  python plot_activations.py --npz figs/acts_base.npz --tag base
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# --- palette -------------------------------------------------------------
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
GRID, SURFACE = "#e6e5e1", "#fcfcfb"
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="figs/acts_base.npz")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    absmax, rms = d["absmax"], d["rms"]          # (n_b, hid)
    traces, top = d["traces"], d["top_channels"]  # (n_b, ns, sl, k), (k,)
    tok_peak, tok_argc, ids = d["tok_peak"], d["tok_argc"], d["calib_ids"]
    n_b, hid = absmax.shape
    B = np.arange(n_b)

    chan_peak = absmax.max(0)
    rank = np.argsort(-chan_peak)
    hot = rank[:3]                                # the massive-activation channels
    med = np.median(absmax, axis=1)               # typical channel, per boundary

    os.makedirs(args.outdir, exist_ok=True)

    # ======================================================================
    # Figure 1 — which channels, and what happens to them with depth
    # ======================================================================
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.44, wspace=0.17)
    fig.suptitle(
        f"Massive activations in the Llama-2-7B residual stream  ·  "
        f"{ids.shape[0]}x{ids.shape[1]} wikitext-2 tokens"
        + ("" if str(d["rotate"]) == "none" else f"  ·  rotate={d['rotate']}"),
        x=0.008, ha="left", fontsize=12, fontweight="bold", y=0.985)

    # -- (a) heatmap: boundary x channel ----------------------------------
    ax = fig.add_subplot(gs[0, 0])
    # Ratio to the typical channel at the same depth: absolute magnitudes span
    # four decades over depth, which washes the outliers out of a raw ramp.
    # Only a couple of dozen of the 4,096 columns are ever interesting, so the
    # panel shows those rather than a 4,096-wide strip of noise.
    ratio = absmax / med[:, None]
    K = 24
    sel = np.sort(rank[:K])
    im = ax.pcolormesh(np.arange(K + 1), np.arange(n_b + 1) - 0.5,
                       np.maximum(ratio[:, sel], 1.0), cmap=CMAP,
                       norm=LogNorm(vmin=1, vmax=ratio.max()),
                       edgecolors=SURFACE, linewidth=0.5)
    ax.set_xticks(np.arange(K) + 0.5)
    ax.set_xticklabels(sel, rotation=90, fontsize=7)
    for lab, c in zip(ax.get_xticklabels(), sel):
        if c in hot:
            lab.set_color(dict(zip(hot, (ORANGE, AQUA, YELLOW)))[c])
            lab.set_fontweight("bold")
    ax.set_ylim(-0.5, n_b - 0.5)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045)
    cb.set_label("peak |h| / median channel", color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=INK2)
    style(ax, f"a  The {K} largest-|h| channels, at every depth",
          "residual-stream channel", "block boundary")

    # -- (b) which channels ------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.vlines(np.arange(hid), 1e-2, chan_peak, color=GRID, lw=0.5)
    for c, col in zip(hot, (ORANGE, AQUA, YELLOW)):
        ax.vlines(c, 1e-2, chan_peak[c], color=col, lw=2.0, zorder=3)
        ax.plot([c], [chan_peak[c]], "o", ms=5, color=col, zorder=4,
                mec=SURFACE, mew=1.5)
        ax.annotate(f"ch {c}\n{chan_peak[c]:,.0f}", (c, chan_peak[c]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=INK, fontweight="bold", linespacing=1.25)
    p999 = np.quantile(chan_peak, 0.999)
    ax.axhline(p999, color=MUTED, lw=1.0, ls=(0, (4, 3)))
    ax.text(hid, p999, f" 99.9th pct = {p999:.0f} ", color=MUTED, fontsize=8,
            va="center", ha="right", backgroundcolor=SURFACE)
    ax.set_yscale("log")
    ax.set_ylim(1, chan_peak.max() * 6)
    ax.set_xlim(-40, hid + 40)
    style(ax, "b  Two channels are ~30x above everything else",
          "residual-stream channel", "peak |h| over the whole stack")

    # -- (c) depth profile -------------------------------------------------
    # Boundary b holds the output of block b-1, so a change seen at boundary b
    # was written by block b-1. Labels below follow that convention.
    ax = fig.add_subplot(gs[1, :])
    trace = absmax[:, hot[0]]
    born_b = int(np.argmax(trace > 100))                 # boundary the spike appears at
    peak_val = trace.max()
    # Per-block relative change of the channel, measured on the token that
    # carries the spike (averaged over sequences) -- that is the quantity the
    # cancellation acts on. rel[b] is the fraction removed by block b-1.
    ki = {int(c): j for j, c in enumerate(top)}
    V = traces[:, :, :, ki[int(hot[0])]]
    pk_t = np.abs(V[n_b // 2]).argmax(axis=1)
    pv = np.abs(np.stack([[V[b, s, pk_t[s]] for s in range(V.shape[1])]
                          for b in range(n_b)])).mean(1)
    rel = np.zeros(n_b)
    rel[1:] = (pv[:-1] - pv[1:]) / np.maximum(pv[:-1], 1e-9)
    onset_b = next(b for b in range(born_b + 2, n_b) if rel[b] > 0.005)
    coll_b = next(b for b in range(born_b + 2, n_b) if rel[b] > 0.5)

    ends = sorted(range(len(hot)), key=lambda j: -absmax[-1, hot[j]])
    dy = {ends[0]: 0.0, ends[1]: 9.0, ends[2]: -9.0}
    for j, (c, col) in enumerate(zip(hot, (ORANGE, AQUA, YELLOW))):
        ax.plot(B, np.maximum(absmax[:, c], 1e-3), color=col, label=f"ch {c}")
        ax.annotate(f"ch {c}", (B[-1], max(absmax[-1, c], 1e-3)),
                    textcoords="offset points", xytext=(7, dy[j]), va="center",
                    fontsize=8.5, color=col, fontweight="bold")
    ax.plot(B, med, color=MUTED, lw=2.0, ls=(0, (4, 3)), label="median channel")
    ax.set_yscale("log")
    ax.set_xlim(0, n_b + 1.2)
    ax.set_ylim(2e-2, peak_val * 18)
    ax.set_xticks(np.arange(0, n_b + 1, 2))

    ax.axvspan(born_b, onset_b - 1, color=BLUE, alpha=0.05, zorder=0)
    ax.text((born_b + onset_b - 1) / 2, peak_val * 3.2,
            f"blocks {born_b}\u2013{onset_b - 2} write nothing into it "
            f"(every step below the bf16 ulp of 8)",
            ha="center", va="center", fontsize=8.5, color=BLUE, fontweight="bold")

    marks = [(born_b, BLUE, f"block {born_b - 1}\nwrites +{peak_val:,.0f}", "left", 8),
             (onset_b, YELLOW, f"block {onset_b - 1}\ndecline starts\n\u2212{rel[onset_b]*100:.1f}%",
              "right", -7),
             (coll_b, ORANGE, f"block {coll_b - 1}\ncollapse\n\u2212{rel[coll_b]*100:.0f}%",
              "left", 7)]
    for x, col, lab, ha, dx in marks:
        ax.axvline(x, color=col, lw=1.2, ls=(0, (3, 3)), alpha=0.85, zorder=1)
        ax.annotate(lab, (x, peak_val * 8), textcoords="offset points",
                    xytext=(dx, 0), ha=ha, va="center", fontsize=8,
                    color=col, fontweight="bold", linespacing=1.3)
    ax.annotate("median channel", (B[-1], med[-1]), textcoords="offset points",
                xytext=(7, -8), va="center", fontsize=8.5, color=MUTED,
                fontweight="bold")
    style(ax, f"c  Constant for {onset_b - 1 - born_b} blocks, then cancelled by "
              f"blocks {onset_b - 1}\u2013{coll_b}",
          "block boundary   (0 = token embeddings;  b = output of block b\u22121)",
          "max |h|")

    fig.subplots_adjust(left=0.065, right=0.963, top=0.923, bottom=0.072)
    p1 = os.path.join(args.outdir, f"massive_activations_{args.tag}.png")
    fig.savefig(p1, dpi=200)
    print(f"[fig] {p1}")

    # ======================================================================
    # Figure 2 — where in the sequence
    # ======================================================================
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    b_mid = n_b // 2
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.9),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    fig.suptitle(f"Where the massive activations sit in the sequence  ·  "
                 f"boundary {b_mid}", x=0.008, ha="left",
                 fontsize=12, fontweight="bold", y=0.99)

    # -- (e) one sequence, first 260 tokens --------------------------------
    ax = axes[0]
    s, NT = 0, 260
    ki = {int(c): i for i, c in enumerate(top)}
    for c, col in zip(hot[:2], (ORANGE, AQUA)):
        if int(c) in ki:
            ax.plot(np.arange(NT), np.abs(traces[b_mid, s, :NT, ki[int(c)]]),
                    color=col, label=f"ch {c}")
    others = [i for i, c in enumerate(top) if int(c) not in hot[:2]]
    if others:
        ax.plot(np.arange(NT), np.abs(traces[b_mid, s, :NT][:, others]).max(-1),
                color=MUTED, lw=1.2, ls=(0, (4, 3)), label="next 6 channels")
    pk = np.abs(traces[b_mid, s, :NT, ki[int(hot[0])]])
    for t in np.argsort(-pk)[:1]:
        ax.annotate(f"{tk.decode([int(ids[s, t])])!r} at pos {t}\n{pk[t]:,.0f}",
                    (t, pk[t]), textcoords="offset points", xytext=(14, -6),
                    fontsize=8, color=INK, fontweight="bold", linespacing=1.25,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set_yscale("log")
    ax.legend(loc="upper right", ncol=3, fontsize=8, labelcolor=INK2)
    style(ax, "e  A single calibration sequence", "token position", "|h|")

    # -- (f) which tokens carry them ---------------------------------------
    ax = axes[1]
    thr = 0.25 * tok_peak[b_mid].max()
    ss, tt = np.where(tok_peak[b_mid] > thr)
    from collections import Counter
    cnt = Counter(repr(tk.decode([int(ids[a, b])])) for a, b in zip(ss, tt))
    labs, vals = zip(*cnt.most_common(8))
    y = np.arange(len(labs))[::-1]
    ax.barh(y, vals, height=0.62, color=ORANGE, edgecolor=SURFACE, linewidth=2)
    for yy, v in zip(y, vals):
        ax.text(v + max(vals) * 0.02, yy, str(v), va="center", fontsize=8,
                color=INK2, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labs, fontsize=8, color=INK)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.grid(axis="y", visible=False)
    style(ax, f"f  Tokens with |h| > {thr:,.0f}  (n={len(ss)} of {tok_peak[b_mid].size:,})",
          "count", None)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = os.path.join(args.outdir, f"massive_activations_tokens_{args.tag}.png")
    fig.savefig(p2, dpi=200)
    print(f"[fig] {p2}")


if __name__ == "__main__":
    main()
