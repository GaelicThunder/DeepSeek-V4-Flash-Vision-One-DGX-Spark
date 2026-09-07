#!/usr/bin/env python3
"""Figures for the MixedK+ release (dark theme of assets/card.html).

Every number is a paired per-token NLL difference against the ORIGINAL-precision release of the same model
(DeepSeek-V4-Flash-Vision-Exp, abliterated by drowzeys; FP8 attention, MXFP4 experts; served on two Sparks) on the
frozen 64,859-token corpus in receipts/nll/. The headline metric is what a quantized pack keeps of the original:

    retained = exp(-mean delta NLL) = geometric mean over tokens of p_pack(token) / p_original(token)

so 100 % is the original model itself. The 0xSero/MiaAI-Lab pack is a DIFFERENT base model (V4-Flash-0731, text);
its bar is measured against the Vision-Exp original and therefore mixes model difference with quantization. It is
drawn hatched and must not be read as a quantization comparison.

    tools/plot_aplus.py                 # all figures into assets/aplus/
"""
import argparse, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

BG, FG, DIM, LINE = "#0e1116", "#e6e9ef", "#8b93a3", "#252b36"
GREEN, GOLD, RED, BLUE, WHITE, GREY = "#7fd3a4", "#d9b46a", "#e07a6a", "#6fa8dc", "#f4f6fa", "#6b7280"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
    "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM, "axes.edgecolor": LINE,
    "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.family": ["DejaVu Sans Mono", "monospace"], "font.size": 11, "legend.frameon": False,
    "hatch.color": BG, "hatch.linewidth": 1.2,
})
DOMAINS = ["prose (wikitext)", "math (gsm8k)", "code", "all 64,859 tokens"]

# name, nats above the original (wikitext, gsm8k, code, all), colour, hatch (None = same model), note
PACKS = [
    ("MixedK+  (current: 256 experts, 3-bit on 28 layers)", (0.1374, 0.0023, 0.0812, 0.1034), WHITE, None),
    ("MixedK  (vcruz305: 256 experts, 3-bit on 6 layers) — served until 09-07", (0.271, 0.064, 0.162, 0.213), GREEN, None),
    ("REAP-216 3-bit  (this repo's build: 216 experts, all 3-bit)", (0.562, 0.025, 0.067, 0.354), GOLD, None),
    ("2-bit pruned to 216  (ablation: MixedK experts, REAP keep list)", (0.649, 0.086, 0.171, 0.442), RED, None),
    ("0731 REAP-216 3-bit  (0xSero / MiaAI-Lab) — DIFFERENT base model, text", (0.647, 0.090, 0.063, 0.412), GREY, "//"),
]
MIXEDK, MIXEDK_PLUS, REAP, PACK_E = PACKS[1][1], PACKS[0][1], PACKS[2][1], PACKS[3][1]
VIS3P = (0.566, 0.027, 0.065, 0.357)   # REAP-216 3-bit re-quantized from the unpruned source (= the REAP build)
SE_PLUS = (0.0051, 0.0078, 0.0053, 0.0151)  # SE of the MixedK+ vs MixedK paired deltas (tokens; chunks for "all")
PROXY_ERR = {0: 0.00771, 1: 0.00954, 2: 0.00615, 3: 0.00573, 4: 0.00455, 5: 0.00581, 6: 0.00632, 7: 0.00629, 8: 0.00884,
             9: 0.00774, 10: 0.00559, 11: 0.00791, 12: 0.00776, 13: 0.00734, 14: 0.00482, 15: 0.0061, 16: 0.00661,
             17: 0.00626, 18: 0.00443, 19: 0.00861, 20: 0.00513, 21: 0.00808, 22: 0.0087, 23: 0.01116, 24: 0.00821,
             25: 0.00939, 26: 0.0084, 27: 0.01163, 28: 0.00886, 29: 0.00899, 30: 0.00882, 31: 0.01094, 32: 0.01058,
             33: 0.00761, 34: 0.01013, 35: 0.01062, 36: 0.0071, 37: 0.00985, 38: 0.00616, 39: 0.00944, 40: 0.0096,
             41: 0.00842, 42: 0.0077}
MIXEDK_K3 = {3, 13, 21, 22, 28, 41}
PROMOTED = [27, 23, 31, 35, 32, 34, 37, 40, 1, 39, 25, 29, 8, 30, 19, 26, 24, 11, 12, 9, 0, 42]   # served set, ranking order
CONVERTED_NOT_FIT = [33, 36, 16, 6]


def retained(d):
    return math.exp(-d) * 100


def credit(fig, text):
    fig.text(0.99, 0.012, text, ha="right", va="bottom", color=DIM, fontsize=8.3)


def fig_retention(out):
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    x = np.arange(len(DOMAINS)); w = 0.8 / len(PACKS)
    for i, (name, vals, col, hatch) in enumerate(PACKS):
        v = [retained(d) for d in vals]
        bars = ax.bar(x + (i - (len(PACKS) - 1) / 2) * w, v, w * 0.92, label=name, color=col, edgecolor=BG, linewidth=0.5, hatch=hatch)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{val:.0f}", ha="center", va="bottom", fontsize=8.5, color=col if col != GREY else DIM)
    ax.axhline(100, color=FG, linewidth=0.9)
    ax.text(3.42, 101.0, "100 % = the original-precision model (FP8/MXFP4, two Sparks)", fontsize=8.8, color=FG, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("token probability kept, % of the original\n(geometric mean over tokens)")
    ax.set_ylim(40, 106)
    ax.set_title("How much of the original model each single-Spark pack keeps", fontsize=12.5, pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.4)
    credit(fig, "paired NLL on 64,859 frozen tokens, receipts/nll/ · hatched = different base model (V4-Flash-0731 text vs Vision-Exp): not a quantization comparison · 2026-09-07")
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def fig_plus_vs_mixedk(out):
    d = [a - v for a, v in zip(MIXEDK_PLUS, MIXEDK)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    kept_plus = [retained(a) for a in MIXEDK_PLUS]; kept_k = [retained(v) for v in MIXEDK]
    xs = np.arange(4)
    ax.bar(xs - 0.19, kept_k, 0.36, color=GREEN, edgecolor=BG, label="MixedK (2-bit, 6 layers at 3-bit)")
    ax.bar(xs + 0.19, kept_plus, 0.36, color=WHITE, edgecolor=BG, label="MixedK+ (28 layers at 3-bit)")
    for i in range(4):
        ax.text(xs[i] - 0.19, kept_k[i] + 0.6, f"{kept_k[i]:.1f}", ha="center", fontsize=9, color=GREEN)
        ax.text(xs[i] + 0.19, kept_plus[i] + 0.6, f"{kept_plus[i]:.1f}", ha="center", fontsize=9, color=WHITE)
        ax.text(xs[i], 44, f"paired Δ {d[i]:+.3f} nats\n(SE {SE_PLUS[i]:.3f})", ha="center", fontsize=8.3, color=DIM)
    ax.axhline(100, color=FG, linewidth=0.9)
    ax.set_xticks(xs); ax.set_xticklabels(DOMAINS)
    ax.set_ylim(40, 106); ax.set_ylabel("token probability kept, % of the original")
    ax.set_title("MixedK+ against the pack it replaces, on the same tokens", fontsize=12.5, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    credit(fig, "same 64,859 tokens, same engine, same everything but 22 layers of expert tensors · receipts/nll/")
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(out, dpi=200); plt.close(fig)


def fig_prune_vs_bits(out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=False)
    for ax, k, dom in zip(axes, range(3), DOMAINS[:3]):
        steps = [("MixedK\n2-bit, 256", MIXEDK[k], GREEN), ("prune 40\nexperts", PACK_E[k] - MIXEDK[k], RED),
                 ("3-bit on\nthe 216", VIS3P[k] - PACK_E[k], BLUE), ("REAP-216\n3-bit", VIS3P[k], GOLD)]
        for i, (lab, val, col) in enumerate(steps):
            if i in (0, 3):
                ax.bar(i, val, 0.62, color=col, edgecolor=BG); ax.text(i, val + 0.012, f"{val:+.3f}", ha="center", fontsize=9, color=col)
            else:
                start = MIXEDK[k] if i == 1 else PACK_E[k]
                ax.bar(i, val, 0.62, bottom=start, color=col, edgecolor=BG)
                ax.text(i, max(start, start + val) + 0.012, f"{val:+.3f}", ha="center", fontsize=9, color=col)
        ax.set_xticks(range(4)); ax.set_xticklabels([s[0] for s in steps], fontsize=8.5)
        ax.set_title(dom, fontsize=11); ax.set_ylim(0, 0.75)
        if k == 0: ax.set_ylabel("nats above the original (lower is better)")
    fig.suptitle("Experts versus bits, same model, same weights: pruning 40 experts costs, 3-bit gives part of it back", fontsize=12, y=0.99)
    credit(fig, "pruned pack = MixedK's own expert tensors in a 216-slot pack (byte-identical, verified) · REAP-216 3-bit re-quantized from the unpruned Vision-Exp source")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(out, dpi=200); plt.close(fig)


def fig_layer_ranking(out):
    layers = list(range(43)); errs = [PROXY_ERR[L] for L in layers]
    cols = [GREEN if L in MIXEDK_K3 else (GOLD if L in PROMOTED else (BLUE if L in CONVERTED_NOT_FIT else DIM)) for L in layers]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(layers, [e * 1000 for e in errs], 0.78, color=cols, edgecolor=BG)
    ax.set_xticks(layers); ax.set_xticklabels([str(L) for L in layers], fontsize=8)
    ax.set_xlabel("decoder layer"); ax.set_ylabel("mean quantization error of the layer's\n768 expert tensors at 3-bit  (×1000)")
    ax.set_title("Where 2-bit hurts most, and which layers MixedK+ lifts to 3-bit", fontsize=12.5, pad=12)
    ax.legend(handles=[Patch(color=GOLD, label="lifted to 3-bit in MixedK+ (22 layers, all 256 experts)"),
                       Patch(color=GREEN, label="already 3-bit in vcruz305's MixedK (6 layers)"),
                       Patch(color=BLUE, label="converted too, but the 26-layer pack leaves no room for the KV cache (4)"),
                       Patch(color=DIM, label="stays 2-bit (15 layers)")], loc="upper left", fontsize=8.4)
    credit(fig, "proxy_err from the exllamav3 conversion log of the REAP-216 3-bit build (same source model, same calibration)")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "aplus"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for stale in ("ppl_vs_source.png", "aplus_vs_vis.png"):
        p = os.path.join(a.out, stale)
        if os.path.exists(p): os.remove(p)
    fig_retention(os.path.join(a.out, "retention.png"))
    fig_plus_vs_mixedk(os.path.join(a.out, "mixedk_plus_vs_mixedk.png"))
    fig_prune_vs_bits(os.path.join(a.out, "prune_vs_bits.png"))
    fig_layer_ranking(os.path.join(a.out, "layer_ranking.png"))
    print("retained (%), all tokens:", {n.split("  ")[0]: round(retained(v[3]), 1) for n, v, _, _ in PACKS})
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
