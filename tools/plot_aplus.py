#!/usr/bin/env python3
"""Figures for the A+ release (dark theme of assets/card.html).

Numbers are paired NLL deltas against the FP8 source on the frozen 64,859-token corpus
(receipts/nll/*.json), in nats; the plots show them as extra perplexity (exp(d) - 1).

    tools/plot_aplus.py                          # the three study figures
    tools/plot_aplus.py --aplus 0.226 0.032 0.105 0.170 [--aplus-se ...]   # + A+ bars and the A+ vs Vis figure
"""
import argparse, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, FG, DIM, LINE = "#0e1116", "#e6e9ef", "#8b93a3", "#252b36"
GREEN, GOLD, RED, BLUE, WHITE = "#7fd3a4", "#d9b46a", "#e07a6a", "#6fa8dc", "#f4f6fa"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
    "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM, "axes.edgecolor": LINE,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.family": ["DejaVu Sans Mono", "monospace"], "font.size": 11, "legend.frameon": False,
})
DOMAINS = ["prose (wikitext)", "math (gsm8k)", "code", "all 64,859 tokens"]
# nats above the FP8 source: wikitext, gsm8k, code, all
PACKS = [
    ("2-bit MixedK · 256 experts  (Vis, the served pack)", (0.271, 0.064, 0.162, 0.213), GREEN),
    ("3-bit REAP · 216 experts  (Vis3, our K3 build)", (0.562, 0.025, 0.067, 0.354), GOLD),
    ("3-bit REAP · 216  (0xSero text pack)", (0.647, 0.090, 0.063, 0.412), "#a88b4a"),
    ("2-bit pruned to 216  (pack E = pruning alone)", (0.649, 0.086, 0.171, 0.442), RED),
]
VIS3P = (0.566, 0.027, 0.065, 0.357)   # Vis3' (quantize-before-prune, 250 rows) = Vis3
PROXY_ERR = {0: 0.00771, 1: 0.00954, 2: 0.00615, 3: 0.00573, 4: 0.00455, 5: 0.00581, 6: 0.00632, 7: 0.00629, 8: 0.00884,
             9: 0.00774, 10: 0.00559, 11: 0.00791, 12: 0.00776, 13: 0.00734, 14: 0.00482, 15: 0.0061, 16: 0.00661,
             17: 0.00626, 18: 0.00443, 19: 0.00861, 20: 0.00513, 21: 0.00808, 22: 0.0087, 23: 0.01116, 24: 0.00821,
             25: 0.00939, 26: 0.0084, 27: 0.01163, 28: 0.00886, 29: 0.00899, 30: 0.00882, 31: 0.01094, 32: 0.01058,
             33: 0.00761, 34: 0.01013, 35: 0.01062, 36: 0.0071, 37: 0.00985, 38: 0.00616, 39: 0.00944, 40: 0.0096,
             41: 0.00842, 42: 0.0077}
VIS_K3 = {3, 13, 21, 22, 28, 41}
APLUS_K3 = [27, 23, 31, 35, 32, 34, 37, 40, 1, 39, 25, 29, 8, 30, 19, 26, 24, 11, 12, 9, 0, 42, 33, 36, 16, 6]


def pct(d):
    return (math.exp(d) - 1) * 100


def credit(fig, text):
    fig.text(0.99, 0.012, text, ha="right", va="bottom", color=DIM, fontsize=8.5)


def fig_ppl_vs_source(out, aplus=None, aplus_n=None):
    packs = list(PACKS)
    if aplus is not None:
        packs.append((f"A+ · 256 experts, 3-bit on {aplus_n} layers  (this release)", tuple(aplus), WHITE))
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(DOMAINS)); w = 0.8 / len(packs)
    for i, (name, vals, col) in enumerate(packs):
        v = [pct(d) for d in vals]
        bars = ax.bar(x + (i - (len(packs) - 1) / 2) * w, v, w * 0.92, label=name, color=col, edgecolor=BG, linewidth=0.5)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2, f"{val:+.0f}%", ha="center", va="bottom", fontsize=8, color=col)
    ax.set_xticks(x); ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("perplexity above the FP8 source  (%)")
    ax.set_title("What each single-Spark pack costs against the full-precision model (2×Spark FP8, KL reference)", fontsize=12, pad=12)
    ax.set_ylim(0, max(pct(v[0]) for _, v, _ in packs) * 1.18)
    ax.legend(loc="upper right", fontsize=8.6)
    credit(fig, "paired NLL on 64,859 frozen tokens (wikitext · gsm8k · code), receipts/nll/ · 2026-09-06/07")
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def fig_prune_vs_bits(out):
    vis, e, v3 = PACKS[0][1], PACKS[3][1], VIS3P
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=False)
    for ax, k, dom in zip(axes, range(3), DOMAINS[:3]):
        steps = [("2-bit\n256 experts", vis[k], GREEN), ("+ pruning\nto 216", e[k] - vis[k], RED), ("+ 3-bit on\nthe 216", v3[k] - e[k], BLUE), ("= 3-bit\n216 experts", v3[k], GOLD)]
        # waterfall
        base = 0.0
        for i, (lab, val, col) in enumerate(steps):
            if i in (0, 3):
                ax.bar(i, val, 0.62, color=col, edgecolor=BG); ax.text(i, val + 0.012, f"{val:+.3f}", ha="center", fontsize=9, color=col)
            else:
                start = vis[k] if i == 1 else e[k]
                ax.bar(i, val, 0.62, bottom=start, color=col, edgecolor=BG)
                ax.text(i, max(start, start + val) + 0.012, f"{val:+.3f}", ha="center", fontsize=9, color=col)
        ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
        ax.set_title(dom, fontsize=11); ax.set_ylim(0, 0.75)
        if k == 0: ax.set_ylabel("nats above the FP8 source")
    fig.suptitle("Experts versus bits, separated with the same weights: pruning 40 experts costs, 3-bit recovers part of it", fontsize=12, y=0.99)
    credit(fig, "pack E = Vis's own experts in a 216-slot pack (byte-identical, verified) · Vis3' = 3-bit K216 quantized from the unpruned source")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out, dpi=200); plt.close(fig)


def fig_layer_ranking(out, n_new=26):
    layers = list(range(43)); errs = [PROXY_ERR[L] for L in layers]
    chosen = set(APLUS_K3[:n_new])
    cols = [GREEN if L in VIS_K3 else (GOLD if L in chosen else DIM) for L in layers]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(layers, [e * 1000 for e in errs], 0.78, color=cols, edgecolor=BG)
    thr = sorted((PROXY_ERR[L] for L in APLUS_K3[:n_new]))[0] * 1000
    ax.axhline(thr, color=GOLD, linewidth=0.8, linestyle="--")
    ax.text(42.4, thr + 0.12, f"cut for the top {n_new}", ha="right", fontsize=8.5, color=GOLD)
    ax.set_xticks(layers); ax.set_xticklabels([str(L) for L in layers], fontsize=8)
    ax.set_xlabel("decoder layer"); ax.set_ylabel("mean proxy error of the 768 expert tensors at 3-bit  (×1000)")
    ax.set_title("Where 2-bit hurts most: the per-layer quantization error, and which layers A+ promotes to 3-bit", fontsize=12, pad=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GOLD, label=f"promoted to 3-bit in A+ ({n_new} layers, all 256 experts)"),
                       Patch(color=GREEN, label="already 3-bit in the MixedK pack (6 layers)"),
                       Patch(color=DIM, label="stays 2-bit (fits the memory budget)")], loc="upper left", fontsize=8.6)
    credit(fig, "proxy_err from the exllamav3 conversion log of the 216-expert 3-bit build (same source, same calibration)")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def fig_aplus_vs_vis(out, aplus, se, n):
    vis = PACKS[0][1]
    d = [a - v for a, v in zip(aplus, vis)]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    cols = [GREEN if x < 0 else RED for x in d]
    ax.bar(range(4), [pct(x) for x in d], 0.6, color=cols, edgecolor=BG,
           yerr=None if se is None else [abs(pct(x + s) - pct(x)) for x, s in zip(d, se)], ecolor=DIM, capsize=4)
    for i, x in enumerate(d):
        ax.text(i, pct(x) + (0.4 if x >= 0 else -0.4), f"{x:+.3f} nats\n{pct(x):+.1f}%", ha="center", va="bottom" if x >= 0 else "top", fontsize=9)
    ax.axhline(0, color=FG, linewidth=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("perplexity change vs the 2-bit MixedK pack  (%)")
    ax.set_title(f"A+ (3-bit on {n} layers, all 256 experts) against the served 2-bit pack — negative is better", fontsize=12, pad=12)
    lim = max(abs(pct(x)) for x in d) * 1.6 + 1
    ax.set_ylim(-lim, lim)
    credit(fig, "paired per-token deltas on the same 64,859 tokens; error bars = 1 SE")
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "aplus"))
    ap.add_argument("--aplus", type=float, nargs=4, help="A+ nats above the source: wikitext gsm8k code all")
    ap.add_argument("--aplus-se", type=float, nargs=4)
    ap.add_argument("--n", type=int, default=26, help="layers promoted to 3-bit in A+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_ppl_vs_source(os.path.join(a.out, "ppl_vs_source.png"), a.aplus, a.n)
    fig_prune_vs_bits(os.path.join(a.out, "prune_vs_bits.png"))
    fig_layer_ranking(os.path.join(a.out, "layer_ranking.png"), a.n)
    if a.aplus:
        fig_aplus_vs_vis(os.path.join(a.out, "aplus_vs_vis.png"), a.aplus, a.aplus_se, a.n)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
