#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-panel figures (Fig. 1-4, S1, S2) for the Arabidopsis sequence-to-expression manuscript.

Reads only the processed score tables in ../../results/ (override with FIG_TABLES)
and writes PDF+PNG to FIG_OUT (override with FIG_OUT). No invented scores; no GPU required.
Environment overrides: FIG_REPO_ROOT / FIG_TABLES / FIG_OUT.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(os.environ.get("FIG_REPO_ROOT", Path(__file__).resolve().parents[2]))
TABLES = Path(os.environ.get("FIG_TABLES", ROOT / "results"))
OUT_PAPER = Path(os.environ.get("FIG_OUT", ROOT / "figures"))
OUT_PACK = OUT_PAPER

sys.path.insert(0, str(Path(__file__).resolve().parent / "style"))
from house_flowchart import (  # noqa: E402
    BLACK,
    LIGHT_GRAY,
    WHITE,
    _arrow,
    _box,
    _text,
    apply_flowchart_style,
)
from publication_style import (  # noqa: E402
    FigureStyle,
    apply_publication_style,
    finalize_figure,
    qa_before_save,
)

C_CNN = "#4E79A7"
C_AGRONT = "#F28E2B"
INK = "#1A1A1A"
MUTED = "#5B5B5B"
EDGE = "#2F2F2F"
ACCENT = "#E15759"
LIGHT = "#F4F4F4"
REGION_ORDER = ["utr5", "utr3", "promoter", "terminator"]
# 面板 a / 面板 b 去灰：改用模型主色的极浅色调，区分度保留但不再是灰
TINT_CNN = "#E4EDF6"      # C_CNN 的极浅色调
TINT_CNN_STRONG = "#9DBBD8"  # 轨迹填充

REGION_COLORS = {
    "promoter": "#F28E2B",
    "utr5": "#4E79A7",
    "gap": "#D9D9D9",
    "utr3": "#59A14F",
    "terminator": "#B07AA1",
}
REGION_LAB = {
    "promoter": "Promoter",
    "utr5": "5′UTR",
    "utr3": "3′UTR",
    "terminator": "Terminator",
    "gap": "N-gap",
}
P, U5, GAP, U3, T = 1000, 500, 20, 500, 1000
FLANK_L = P + U5 + GAP + U3 + T
BOUNDS = {
    "promoter": (0, P),
    "utr5": (P, P + U5),
    "gap": (P + U5, P + U5 + GAP),
    "utr3": (P + U5 + GAP, P + U5 + GAP + U3),
    "terminator": (P + U5 + GAP + U3, FLANK_L),
}
HEAT = LinearSegmentedColormap.from_list(
    "paper_seq", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#4E79A7", "#1B3A5F"]
)
MCNEMAR_CNN_ONLY = 142
MCNEMAR_AGRONT_ONLY = 532
N_TEST = 3402


def assign_region(pos: int) -> str:
    if 0 <= pos < P:
        return "promoter"
    if P <= pos < P + U5:
        return "utr5"
    if P + U5 <= pos < P + U5 + GAP:
        return "gap"
    if P + U5 + GAP <= pos < P + U5 + GAP + U3:
        return "utr3"
    if P + U5 + GAP + U3 <= pos < FLANK_L:
        return "terminator"
    return "out"


def panel_letter(ax, letter: str, x: float = -0.02, y: float = 1.06) -> None:
    ax.text(
        x, y, letter, transform=ax.transAxes, fontsize=12.98, fontweight="bold",
        color=INK, ha="left", va="bottom", clip_on=False,
    )


def style_axis(ax, lw: float = 0.8) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color(INK)
    ax.tick_params(width=lw, length=2.8, labelsize=8.50, colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def apply_paper_style() -> None:
    apply_publication_style(FigureStyle(font_size=9.5, axes_linewidth=0.8, cjk=False))
    mpl.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_fig(fig, stem: str, tight: bool = True) -> None:
    qa = qa_before_save(fig, min_fontsize=5.5)
    defects = qa.get("defects") or []
    high = [d for d in defects if d.get("severity") == "high"]
    if high:
        print(f"[QA high] {stem}: {high}")
    elif defects:
        kinds = {}
        for d in defects:
            kinds[d.get("issue", "?")[:80]] = kinds.get(d.get("issue", "?")[:80], 0) + 1
        print(f"[QA] {stem}: {len(defects)} notes; {kinds}")
    for out_dir in (OUT_PAPER, OUT_PACK):
        out_dir.mkdir(parents=True, exist_ok=True)
        finalize_figure(
            fig, str(out_dir / stem), formats=["png", "pdf"], dpi=300,
            tight=tight, close=False, pad=0.04,
        )
    plt.close(fig)


def rounded(ax, x, y, w, h, fc="white", ec=EDGE, lw=0.9, rad=0.08, z=2):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={rad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, mutation_aspect=0.4, zorder=z,
    )
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, color=INK):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=9, lw=0.9,
        color=color, shrinkA=0, shrinkB=0, zorder=4,
    ))


def load_benchmark():
    df = pd.read_csv(TABLES / "benchmark_results.csv").set_index("model")
    cnn = df.loc["cnn"]
    agro = df.loc["agront"]
    assert abs(float(cnn["auroc"]) - 0.8423312743427458) < 1e-9
    assert abs(float(agro["auroc"]) - 0.9391213513543419) < 1e-9
    return cnn, agro


def mcnemar_counts():
    cnn, agro = load_benchmark()
    cnn_ok = int(round(float(cnn["accuracy"]) * N_TEST))
    agro_ok = int(round(float(agro["accuracy"]) * N_TEST))
    both_ok = cnn_ok - MCNEMAR_CNN_ONLY
    both_bad = N_TEST - agro_ok - MCNEMAR_CNN_ONLY
    assert both_ok + MCNEMAR_CNN_ONLY + MCNEMAR_AGRONT_ONLY + both_bad == N_TEST
    return np.array([[both_ok, MCNEMAR_CNN_ONLY], [MCNEMAR_AGRONT_ONLY, both_bad]], dtype=int)


def load_variant_frames():
    cnn = pd.read_csv(TABLES / "scores.tsv", sep="\t")
    cnn["region"] = cnn["flank_pos"].astype(int).map(assign_region)
    cnn = cnn[cnn["region"].isin(REGION_ORDER)].copy()
    agro = pd.read_csv(TABLES / "agront_scores_by_site.tsv", sep="\t")
    agro["abs_delta"] = agro["mean_abs_delta"]
    agro["region"] = agro["flank_pos"].astype(int).map(assign_region)
    agro = agro[agro["region"].isin(REGION_ORDER)].copy()
    return cnn, agro


def binned_mean(df, pos="flank_pos", val="abs_delta", bin_size=25, length=FLANK_L):
    pos_v = df[pos].to_numpy(dtype=int)
    val_v = df[val].to_numpy(dtype=float)
    nbin = int(np.ceil(length / bin_size))
    idx = np.clip(pos_v // bin_size, 0, nbin - 1)
    sums = np.bincount(idx, weights=val_v, minlength=nbin).astype(float)
    cnts = np.bincount(idx, minlength=nbin).astype(float)
    mean = np.divide(sums, cnts, out=np.full(nbin, np.nan), where=cnts > 0)
    centers = (np.arange(nbin) + 0.5) * bin_size
    return centers, mean


def shade_regions(ax, ymax=None):
    for name, (s, e) in BOUNDS.items():
        ax.axvspan(s, e, color=REGION_COLORS[name], alpha=0.13, lw=0, zorder=0)
    ax.axvline(P, color="white", lw=0.4, zorder=1)
    ax.axvline(P + U5, color="white", lw=0.4, zorder=1)
    ax.axvline(P + U5 + GAP + U3, color="white", lw=0.4, zorder=1)


def region_xticks(ax):
    mids = [(BOUNDS[r][0] + BOUNDS[r][1]) / 2 for r in ("promoter", "utr5", "utr3", "terminator")]
    ax.set_xticks(mids)
    ax.set_xticklabels(["Promoter", "5′UTR", "3′UTR", "Terminator"], fontsize=7.79)


def box_region(ax, scores, p_text=None):
    data = [scores.loc[scores["region"] == r, "abs_delta"].to_numpy() for r in REGION_ORDER]
    medians = [float(np.median(d)) for d in data]
    ns = [len(d) for d in data]
    ticks = [f"{REGION_LAB[r]}\n{m:.4f}" for r, m in zip(REGION_ORDER, medians)]
    bp = ax.boxplot(
        data, tick_labels=ticks,
        showfliers=False, patch_artist=True, widths=0.62,
        medianprops=dict(color=INK, linewidth=1.2),
        whiskerprops=dict(color="#4A4A4A", linewidth=0.8),
        capprops=dict(color="#4A4A4A", linewidth=0.8),
        boxprops=dict(linewidth=0.9),
    )
    for patch, reg in zip(bp["boxes"], REGION_ORDER):
        patch.set_facecolor(REGION_COLORS[reg])
        patch.set_alpha(0.38)
        patch.set_edgecolor(EDGE)
    whisker_tops = []
    for d in data:
        q1, q3 = np.percentile(d, [25, 75])
        hi = d[d <= q3 + 1.5 * (q3 - q1)]
        whisker_tops.append(float(hi.max()) if len(hi) else float(q3))
    wmax = max(whisker_tops)
    if p_text:
        y_br = wmax * 1.08
        ax.plot([1, 1, 2, 2], [y_br * 1.02, y_br, y_br, y_br * 1.02], color=INK, lw=0.8)
        ax.text(1.5, y_br * 1.03, p_text, ha="center", va="bottom", fontsize=7.79, color=INK)
        ax.set_ylim(0, y_br * 1.22)
    else:
        ax.set_ylim(0, wmax * 1.18)
    style_axis(ax)
    ax.tick_params(axis="x", length=0, labelsize=6.50)
    return ns


def _tss_marker(ax, x, y_top, label, *, y_text=None):
    ax.plot([x, x], [0, y_top], color=BLACK, lw=1.05, zorder=5, clip_on=False)
    ax.plot(x, y_top, marker="v", color=BLACK, ms=7.5, zorder=6, clip_on=False)
    ax.text(x, y_text if y_text is not None else y_top * 1.04, label,
            ha="center", va="bottom", fontsize=9.44, fontweight="bold", color=BLACK, clip_on=False)


def _join_down(ax, sources, dest, *, y_bar=None):
    """Orthogonal join: verticals from source bottoms to a bar, then one arrow into dest."""
    sx = [0.5 * (b[0] + b[2]) for b in sources]
    sy = [b[1] for b in sources]
    dx = 0.5 * (dest[0] + dest[2])
    if y_bar is None:
        y_bar = 0.5 * (min(sy) + dest[3])
    for x, y in zip(sx, sy):
        ax.plot([x, x], [y, y_bar], color=BLACK, lw=1.05, zorder=3, solid_capstyle="butt")
    ax.plot([min(sx), max(sx)], [y_bar, y_bar], color=BLACK, lw=1.05, zorder=3, solid_capstyle="butt")
    if abs(dx - min(sx)) > 0.02 and abs(dx - max(sx)) > 0.02:
        ax.plot([dx, dx], [y_bar, y_bar], color=BLACK, lw=1.05, zorder=3)
    _arrow(ax, (dx, y_bar), (dx, dest[3]))


def _split_down(ax, source, dests, *, y_bar=None):
    """Orthogonal split: one vertical from source, bar, then arrows into dest tops."""
    sx = 0.5 * (source[0] + source[2])
    dxs = [0.5 * (b[0] + b[2]) for b in dests]
    if y_bar is None:
        y_bar = 0.5 * (source[1] + dests[0][3])
    ax.plot([sx, sx], [source[1], y_bar], color=BLACK, lw=1.05, zorder=3, solid_capstyle="butt")
    ax.plot([min(dxs), max(dxs)], [y_bar, y_bar], color=BLACK, lw=1.05, zorder=3, solid_capstyle="butt")
    for x, b in zip(dxs, dests):
        _arrow(ax, (x, y_bar), (x, b[3]))


def plot_fig1() -> None:
    apply_flowchart_style(lang="en", base_size=9.5)
    fig = plt.figure(figsize=(7.28, 7.48), facecolor=WHITE)
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.28, 3.48], hspace=0.12,
        left=0.09, right=0.985, top=0.965, bottom=0.018,
    )
    gs_a = gs[0].subgridspec(2, 1, height_ratios=[0.70, 1.42], hspace=0.55)
    ax_pgb = fig.add_subplot(gs_a[0])
    ax_str = fig.add_subplot(gs_a[1])
    ax_b = fig.add_subplot(gs[1])
    ax_b.set_facecolor(WHITE)

    # (a) PGB 6 kb window — coordinate axis + TSS, not a color card
    ax_pgb.set_xlim(-5.45, 1.35)
    ax_pgb.set_ylim(0, 1.22)
    ax_pgb.add_patch(Rectangle((-5, 0.18), 5.0, 0.58, facecolor=WHITE, edgecolor=BLACK, lw=0.9, zorder=2))
    ax_pgb.add_patch(Rectangle((0, 0.18), 1.0, 0.58, facecolor=TINT_CNN, edgecolor=BLACK, lw=0.9, zorder=2))
    ax_pgb.text(-2.5, 0.47, "5 kb upstream", ha="center", va="center", fontsize=9.44, color=BLACK, zorder=3)
    ax_pgb.text(0.50, 0.47, "1 kb", ha="center", va="center", fontsize=9.44, color=BLACK, zorder=3)
    _tss_marker(ax_pgb, 0.0, 0.86, "TSS", y_text=0.92)
    ax_pgb.set_xticks([-5, -4, -3, -2, -1, 0, 1])
    ax_pgb.set_xticklabels(["-5", "-4", "-3", "-2", "-1", "0", "+1"])
    ax_pgb.set_xlabel("Position relative to TSS (kb)")
    ax_pgb.set_ylabel("PGB 6 kb")
    ax_pgb.set_yticks([])
    for side in ("top", "right", "left"):
        ax_pgb.spines[side].set_visible(False)
    ax_pgb.spines["bottom"].set_linewidth(0.8)
    ax_pgb.spines["bottom"].set_color(BLACK)
    ax_pgb.tick_params(axis="x", length=3.0, width=0.8, labelsize=8.50, colors=BLACK)
    ax_pgb.xaxis.label.set_size(8)
    ax_pgb.yaxis.label.set_size(8)
    panel_letter(ax_pgb, "a", -0.08, 1.14)

    # (a) structured 3,020-bp flanks: real CNN |Δ| track + gene-model bar
    cnn_sites, _ = load_variant_frames()
    xc, yc = binned_mean(cnn_sites, bin_size=25)
    ypeak = float(np.nanmax(yc))
    ymax = ypeak * 1.22
    bar0, bar1 = -0.30 * ypeak, -0.07 * ypeak
    tss_x = float(P)
    tts_x = float(P + U5 + GAP + U3)
    ax_str.set_xlim(-40, FLANK_L + 60)
    ax_str.set_ylim(bar0 - 0.06 * ypeak, ymax)
    ax_str.fill_between(xc, 0, yc, color=TINT_CNN_STRONG, lw=0, zorder=2)
    ax_str.plot(xc, yc, color=BLACK, lw=0.55, zorder=3, solid_capstyle="butt")
    ax_str.axhline(0, color=BLACK, lw=0.45, zorder=4)
    ax_str.plot([tss_x, tss_x], [bar0, ymax * 0.88], color=BLACK, lw=1.05, zorder=5)
    ax_str.plot([tts_x, tts_x], [bar0, ymax * 0.88], color=BLACK, lw=1.05, zorder=5)
    ax_str.plot(tss_x, ymax * 0.88, marker="v", color=BLACK, ms=7.5, zorder=6, clip_on=False)
    ax_str.plot(tts_x, ymax * 0.88, marker="v", color=BLACK, ms=7.5, zorder=6, clip_on=False)
    ax_str.text(tss_x, ymax * 0.93, "TSS", ha="center", va="bottom", fontsize=9.44, fontweight="bold")
    ax_str.text(tts_x, ymax * 0.93, "TTS", ha="center", va="bottom", fontsize=9.44, fontweight="bold")
    gene_bar = [
        (0, P, WHITE, "Promoter  1 kb"),
        (P, P + U5, "#E8E8E8", "5'UTR  500"),
        (P + U5, P + U5 + GAP, BLACK, ""),
        (P + U5 + GAP, P + U5 + GAP + U3, "#D0D0D0", "3'UTR  500"),
        (P + U5 + GAP + U3, FLANK_L, "#B5B5B5", "Terminator  1 kb"),
    ]
    for x0, x1, fc, lab in gene_bar:
        ax_str.add_patch(Rectangle(
            (x0, bar0), x1 - x0, bar1 - bar0, facecolor=fc, edgecolor=BLACK, lw=0.7, zorder=6,
        ))
        if lab:
            ax_str.text(0.5 * (x0 + x1), 0.5 * (bar0 + bar1), lab, ha="center", va="center",
                        fontsize=7.55, color=BLACK, zorder=7)
    ax_str.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3020])
    ax_str.set_yticks([0.0, 0.005, 0.010])
    ax_str.set_xlabel("Concatenated flank position (bp)")
    ax_str.set_ylabel(r"CNN $|\Delta|$")
    ax_str.spines["top"].set_visible(False)
    ax_str.spines["right"].set_visible(False)
    ax_str.spines["left"].set_linewidth(0.8)
    ax_str.spines["bottom"].set_linewidth(0.8)
    ax_str.spines["left"].set_bounds(0, 0.010)
    ax_str.spines["left"].set_color(BLACK)
    ax_str.spines["bottom"].set_color(BLACK)
    ax_str.tick_params(length=3.0, width=0.8, labelsize=8.50, colors=BLACK)
    ax_str.xaxis.label.set_size(8)
    ax_str.yaxis.label.set_size(8)

    # (b) PRISMA-style locked protocol
    ax_b.set_xlim(0, 10)
    ax_b.set_ylim(0, 10)
    ax_b.axis("off")
    panel_letter(ax_b, "b", -0.01, 1.01)

    seq = (0.18, 8.62, 4.72, 9.88)
    abund = (5.08, 8.62, 9.78, 9.88)
    paired = (0.18, 6.92, 6.42, 8.18)
    note_split = (6.72, 7.08, 9.78, 8.02)
    labels = (0.18, 5.18, 6.42, 6.48)
    note_lab = (6.72, 5.28, 9.78, 6.38)
    cnn = (0.18, 2.42, 3.22, 4.72)
    agro = (3.38, 2.42, 6.42, 4.72)
    note_mod = (6.72, 2.62, 9.78, 4.52)
    read = (0.18, 0.12, 6.42, 1.96)

    _box(ax_b, *seq)
    _text(ax_b, 0.5 * (seq[0] + seq[2]), 9.62, "PGB Arabidopsis sequence windows", size=8.0, weight="bold")
    _text(ax_b, 0.5 * (seq[0] + seq[2]), 9.12,
          "6 kb = 5 kb upstream + 1 kb downstream of TSS\nused for training and primary AUROC", size=6.8)

    _box(ax_b, *abund)
    _text(ax_b, 0.5 * (abund[0] + abund[2]), 9.62, "56-tissue / sample abundances", size=8.0, weight="bold")
    _text(ax_b, 0.5 * (abund[0] + abund[2]), 9.12,
          "gene-level matrix paired to the same genes\n56 tissues / samples per gene", size=6.8)

    _join_down(ax_b, [seq, abund], paired)
    _box(ax_b, *paired)
    _text(ax_b, 0.5 * (paired[0] + paired[2]), 7.92, "Official PGB gene split, no overlap", size=8.0, weight="bold")
    _text(ax_b, 0.5 * (paired[0] + paired[2]), 7.38,
          "Train 25,731   ·   Valid. 3,401   ·   Test 3,402\n32,534 genes after the public partition", size=6.8)
    _box(ax_b, *note_split, lw=1.0)
    _arrow(ax_b, (paired[2], 0.5 * (paired[1] + paired[3])), (note_split[0], 0.5 * (note_split[1] + note_split[3])))
    _text(ax_b, 0.5 * (note_split[0] + note_split[2]), 0.5 * (note_split[1] + note_split[3]),
          "Public split kept\nas published", size=6.8)

    _arrow(ax_b, (0.5 * (paired[0] + paired[2]), paired[1]), (0.5 * (labels[0] + labels[2]), labels[3]))
    _box(ax_b, *labels)
    _text(ax_b, 0.5 * (labels[0] + labels[2]), 6.22, "Tissue-averaged binary labels", size=8.0, weight="bold")
    _text(ax_b, 0.5 * (labels[0] + labels[2]), 5.68,
          r"$\bar{y}_g$ over 56 tissues;  $z_g=\mathbf{1}\{\bar{y}_g\geq\tau\}$"
          "\n"
          r"$\tau=$ train-set median $=5.605$", size=6.8)
    _box(ax_b, *note_lab, lw=1.0)
    _arrow(ax_b, (labels[2], 0.5 * (labels[1] + labels[3])), (note_lab[0], 0.5 * (note_lab[1] + note_lab[3])))
    _text(ax_b, 0.5 * (note_lab[0] + note_lab[2]), 0.5 * (note_lab[1] + note_lab[3]),
          "Derived binary task;\nnot the official PGB\nmulti-tissue regression", size=6.6)

    _split_down(ax_b, labels, [cnn, agro])
    _box(ax_b, *cnn, fc=WHITE)
    _text(ax_b, 0.5 * (cnn[0] + cnn[2]), 4.48, "1D-CNN  ·  DeepSEA-style", size=7.6, weight="bold")
    _text(ax_b, 0.5 * (cnn[0] + cnn[2]), 3.48,
          "One-hot 6 kb "
          r"$\times$ 4"
          "\n3 conv. blocks; k=8, 256; pool 4"
          "\nglobal max-pool + FC"
          "\n2.7 M  ·  30 epochs"
          "\n"
          r"AdamW $1\times10^{-3}$, 15 min"
          "\nseeds 42–46", size=6.4)
    _box(ax_b, *agro, fc=WHITE)
    _text(ax_b, 0.5 * (agro[0] + agro[2]), 4.48, "AgroNT-1B  ·  LoRA", size=7.6, weight="bold")
    _text(ax_b, 0.5 * (agro[0] + agro[2]), 3.48,
          "6-mer tokens; frozen body"
          "\nLoRA on Q/K/V/O"
          "\n"
          r"$W=W_0+(\alpha/r)BA$; $r=16$, $\alpha=32$"
          "\n16.2 M of 985 M (1.6%)"
          "\n"
          r"AdamW $2\times10^{-4}$, 3 epochs, 240 min"
          "\nseeds 42–46", size=6.4)
    _box(ax_b, *note_mod, lw=1.0)
    _arrow(ax_b, (agro[2], 0.5 * (agro[1] + agro[3])), (note_mod[0], 0.5 * (note_mod[1] + note_mod[3])))
    _text(ax_b, 0.5 * (note_mod[0] + note_mod[2]), 0.5 * (note_mod[1] + note_mod[3]),
          "Identical splits,\nlabels, and metrics\none RTX 4090", size=6.6)

    _join_down(ax_b, [cnn, agro], read)
    _box(ax_b, *read)
    _text(ax_b, 0.5 * (read[0] + read[2]), 1.78, "Four locked readouts", size=8.0, weight="bold")
    _text(ax_b, 0.5 * (read[0] + read[2]), 0.92,
          "1. Arabidopsis test n=3,402  —  AUROC 0.842 vs 0.939; McNemar 532 vs 142"
          "\n2. Region maps  —  exploratory; allele protocols differ by architecture"
          "\n3. Natural SNVs  —  250 genes; CNN 27,662 sites; AgroNT 26,518 sites"
          "\n4. Crop PGB transfer  —  rice, maize, tomato, soybean; AgroNT saw those genomes",
          size=6.5)

    save_fig(fig, "Figure1", tight=False)


def plot_fig2() -> None:
    apply_paper_style()
    cnn, agro = load_benchmark()
    seeds = pd.read_csv(TABLES / "cnn_multiseed_pgb.csv")
    seeds_ag = pd.read_csv(TABLES / "agront_multiseed_pgb.csv")
    mat = mcnemar_counts()
    fig = plt.figure(figsize=(7.35, 5.85))
    gs = fig.add_gridspec(
        2, 2, hspace=0.42, wspace=0.34, left=0.10, right=0.95, top=0.93, bottom=0.09,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    metrics = ["accuracy", "auroc", "auprc"]
    names = ["Accuracy", "AUROC", "AUPRC"]
    v0 = [float(cnn[m]) for m in metrics]
    v1 = [float(agro[m]) for m in metrics]
    x = np.arange(len(metrics), dtype=float)
    w = 0.36
    ax_a.bar(x - w / 2, v0, w, color=C_CNN, edgecolor=EDGE, linewidth=0.7, hatch="//", label="CNN")
    ax_a.bar(x + w / 2, v1, w, color=C_AGRONT, edgecolor=EDGE, linewidth=0.7, hatch="..", label="AgroNT")
    for xi, a, b in zip(x, v0, v1):
        ax_a.text(xi - w / 2, a + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=7.55, color=C_CNN)
        ax_a.text(xi + w / 2, b + 0.008, f"{b:.3f}", ha="center", va="bottom", fontsize=7.55, color=C_AGRONT)
    ax_a.set_xticks(x, names)
    ax_a.set_ylabel("Held-out score")
    ax_a.set_ylim(0.70, 1.00)
    ax_a.legend(frameon=False, loc="upper left", handlelength=1.3)
    ax_a.set_title("PGB Arabidopsis test  (n = 3,402)", fontsize=9.44, pad=3)
    style_axis(ax_a)
    panel_letter(ax_a, "a")

    im = ax_b.imshow(mat, cmap=LinearSegmentedColormap.from_list("mc", ["#F7FBFF", "#9ECAE1", "#2171B5"]),
                     vmin=100, vmax=2500)
    ax_b.set_xticks([0, 1], ["AgroNT correct", "AgroNT wrong"])
    ax_b.set_yticks([0, 1], ["CNN correct", "CNN wrong"])
    labels = [
        [f"{mat[0, 0]:,}\nboth correct", f"{mat[0, 1]:,}\nCNN only"],
        [f"{mat[1, 0]:,}\nAgroNT only", f"{mat[1, 1]:,}\nboth wrong"],
    ]
    for i in range(2):
        for j in range(2):
            col = "white" if mat[i, j] > 900 else INK
            ax_b.text(j, i, labels[i][j], ha="center", va="center", fontsize=7.79, color=col, linespacing=1.25)
    ax_b.tick_params(length=0, labelsize=7.79)
    for s in ax_b.spines.values():
        s.set_visible(False)
    ax_b.set_title("Paired classifications", fontsize=9.44, pad=3)
    cax = inset_axes(ax_b, width="4%", height="70%", loc="lower left",
                     bbox_to_anchor=(1.04, 0.05, 1, 1), bbox_transform=ax_b.transAxes, borderpad=0)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7.08, length=2)
    cb.set_label("Genes", fontsize=7.55)
    panel_letter(ax_b, "b", x=-0.08)

    vals = seeds["auroc"].to_numpy(dtype=float)
    acc = seeds["accuracy"].to_numpy(dtype=float)
    avals = seeds_ag["auroc"].to_numpy(dtype=float)
    aacc = seeds_ag["accuracy"].to_numpy(dtype=float)
    # Sep 16 版的 jitter 写死 3 个点而 seed 有 5 行，scatter 直接崩；改为按实际长度生成。
    jit = lambda n: np.linspace(-0.10, 0.10, num=n)
    ax_c.scatter(jit(len(vals)) - 0.30, vals, s=42, color=C_CNN,
                 edgecolors="white", lw=0.6, zorder=3)
    ax_c.scatter(jit(len(avals)) + 0.30, avals, s=42, color=C_AGRONT,
                 edgecolors="white", lw=0.6, zorder=3)
    mu, sd = float(vals.mean()), float(vals.std(ddof=1))
    amu, asd = float(avals.mean()), float(avals.std(ddof=1))
    for cxx, m_, s_, col in ((-0.30, mu, sd, C_CNN), (0.30, amu, asd, C_AGRONT)):
        ax_c.hlines(m_, cxx - 0.13, cxx + 0.13, color=col, lw=1.5, zorder=2)
        ax_c.plot([cxx, cxx], [m_ - s_, m_ + s_], color=col, lw=1.1, zorder=2)
    ax_c.text(-0.30, mu + 0.006, f"CNN {mu:.3f}±{sd:.3f}", ha="center",
              va="bottom", fontsize=7.55, color=C_CNN)
    ax_c.text(0.30, amu + 0.006, f"AgroNT {amu:.3f}±{asd:.3f}", ha="center",
              va="bottom", fontsize=7.55, color=C_AGRONT)
    ax_c.text(0.98, 0.04,
              f"accuracy  CNN {acc.mean():.3f}±{acc.std(ddof=1):.3f}"
              f"\n           AgroNT {aacc.mean():.3f}±{aacc.std(ddof=1):.3f}",
              transform=ax_c.transAxes, ha="right", va="bottom",
              fontsize=7.55, color=MUTED, linespacing=1.45)
    ax_c.set_xlim(-0.55, 0.55)
    ax_c.set_xticks([-0.30, 0.30], ["CNN", "AgroNT"])
    ax_c.set_ylabel("Test AUROC")
    ax_c.set_ylim(0.828, 0.968)
    ax_c.set_title("Five matched seeds (42–46)", fontsize=9.44, pad=3)
    style_axis(ax_c)
    panel_letter(ax_c, "c")

    ax_d.scatter([float(cnn["auroc"])], [2.7e6], s=90, color=C_CNN, edgecolors="white", lw=0.8, zorder=3)
    ax_d.scatter([float(agro["auroc"])], [16.2e6], s=90, color=C_AGRONT, edgecolors="white", lw=0.8, zorder=3)
    ax_d.annotate("CNN\n2.7 M  ·  15 min\nAUROC 0.842", (float(cnn["auroc"]), 2.7e6),
                  textcoords="offset points", xytext=(10, 12), fontsize=7.67, color=INK,
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax_d.annotate("AgroNT LoRA\n16.2 M of 985 M  ·  240 min\nAUROC 0.939", (float(agro["auroc"]), 16.2e6),
                  textcoords="offset points", xytext=(-8, -28), ha="right", fontsize=7.67, color=INK,
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax_d.set_yscale("log")
    ax_d.set_xlabel("AUROC")
    ax_d.set_ylabel("Trainable parameters")
    ax_d.set_xlim(0.80, 0.97)
    ax_d.set_ylim(1.3e6, 5.5e7)
    ax_d.set_title("Cost on one RTX 4090", fontsize=9.44, pad=3)
    style_axis(ax_d)
    panel_letter(ax_d, "d")

    save_fig(fig, "Figure2", tight=False)


def plot_fig3() -> None:
    apply_paper_style()
    cnn, agro = load_variant_frames()
    mm = pd.read_csv(TABLES / "same_method_region_importance.csv").set_index("region")
    desc = pd.read_csv(TABLES / "variant_region_descriptives.csv").set_index("region")
    desc_a = pd.read_csv(TABLES / "agront_variant_region_descriptives.csv").set_index("region")

    fig = plt.figure(figsize=(7.35, 7.55))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[1.12, 1.55], hspace=0.16,
        left=0.175, right=0.99, top=0.96, bottom=0.06,
    )
    gs_t = outer[0].subgridspec(2, 1, hspace=0.10)
    gs_b = outer[1].subgridspec(2, 2, hspace=0.48, wspace=0.32)
    ax_tr1 = fig.add_subplot(gs_t[0])
    ax_tr2 = fig.add_subplot(gs_t[1], sharex=ax_tr1)
    ax_mu = fig.add_subplot(gs_b[0, 0])
    ax_eq = fig.add_subplot(gs_b[0, 1])
    ax_b1 = fig.add_subplot(gs_b[1, 0])
    ax_b2 = fig.add_subplot(gs_b[1, 1])

    xc, yc = binned_mean(cnn)
    xa, ya = binned_mean(agro)
    shade_regions(ax_tr1)
    ax_tr1.plot(xc, yc, color=C_CNN, lw=1.15, zorder=3)
    ax_tr1.fill_between(xc, 0, yc, color=C_CNN, alpha=0.18, zorder=2)
    ax_tr1.set_xlim(0, FLANK_L)
    ax_tr1.set_ylabel("Mean |Δ|\nCNN")
    ax_tr1.tick_params(axis="x", length=0, labelbottom=False)
    ax_tr1.set_title("Natural flanking SNVs along the structured 3,020-bp representation", fontsize=9.44, pad=1)
    style_axis(ax_tr1)
    panel_letter(ax_tr1, "a", 0.0, 1.06)

    shade_regions(ax_tr2)
    ax_tr2.plot(xa, ya, color=C_AGRONT, lw=1.15, zorder=3)
    ax_tr2.fill_between(xa, 0, ya, color=C_AGRONT, alpha=0.18, zorder=2)
    ax_tr2.set_xlim(0, FLANK_L)
    ax_tr2.set_ylabel("Mean |Δ|\nAgroNT")
    region_xticks(ax_tr2)
    ax_tr2.tick_params(labelbottom=True)
    style_axis(ax_tr2)
    panel_letter(ax_tr2, "b", 0.012, 1.015)

    x = np.arange(len(REGION_ORDER), dtype=float)
    cnn_m = np.array([float(mm.loc[r, "cnn_mutagenesis"]) for r in REGION_ORDER])
    agro_m = np.array([float(mm.loc[r, "agront_mutagenesis"]) for r in REGION_ORDER])
    cnn_n = cnn_m / cnn_m.max()
    agro_n = agro_m / agro_m.max()
    ax_mu.bar(x - 0.18, cnn_n, 0.36, color=C_CNN, edgecolor=EDGE, lw=0.7, hatch="//", label="CNN")
    ax_mu.bar(x + 0.18, agro_n, 0.36, color=C_AGRONT, edgecolor=EDGE, lw=0.7, hatch="..", label="AgroNT")
    ax_mu.set_xticks(x, ["5′UTR", "3′UTR", "Prom.", "Term."], fontsize=7.55)
    ax_mu.set_ylabel("Mutagenesis / model max", fontsize=8.4)
    ax_mu.set_ylim(0, 1.32)
    ax_mu.legend(loc="upper center", ncol=2, fontsize=7.32, handlelength=1.1, frameon=False, borderpad=0, columnspacing=0.8)
    ax_mu.set_title("Matched 50-bp N windows", fontsize=9.44, pad=2)
    style_axis(ax_mu)
    panel_letter(ax_mu, "c", -0.04, 1.05)

    y = np.arange(len(REGION_ORDER), dtype=float)
    cnn_med = np.array([float(desc.loc[r, "median"]) for r in REGION_ORDER])
    cnn_lo = np.array([float(desc.loc[r, "median_ci95_lo"]) for r in REGION_ORDER])
    cnn_hi = np.array([float(desc.loc[r, "median_ci95_hi"]) for r in REGION_ORDER])
    agro_med = np.array([float(desc_a.loc[r, "median"]) for r in REGION_ORDER])
    agro_lo = np.array([float(desc_a.loc[r, "median_ci95_lo"]) for r in REGION_ORDER])
    agro_hi = np.array([float(desc_a.loc[r, "median_ci95_hi"]) for r in REGION_ORDER])
    ax_eq.errorbar(
        cnn_med, y + 0.12, xerr=[cnn_med - cnn_lo, cnn_hi - cnn_med],
        fmt="o", color=C_CNN, ms=5.5, lw=1.1, capsize=2.2, label="CNN",
    )
    ax_eq.errorbar(
        agro_med, y - 0.12, xerr=[agro_med - agro_lo, agro_hi - agro_med],
        fmt="o", color=C_AGRONT, ms=5.5, lw=1.1, capsize=2.2, label="AgroNT",
    )
    ax_eq.set_yticks(y, [REGION_LAB[r] for r in REGION_ORDER])
    ax_eq.set_xlabel("Median |Δ| (bootstrap 95% CI)")
    ax_eq.set_title("Region medians, not cross-model", fontsize=8.73, pad=2)
    ax_eq.legend(frameon=False, fontsize=7.08, loc="lower right", handlelength=1.2)
    style_axis(ax_eq)
    ax_eq.tick_params(axis="y", length=0)
    panel_letter(ax_eq, "d", -0.04, 1.05)

    box_region(ax_b1, cnn, None)
    ax_b1.set_ylabel("|Δ|  CNN", fontsize=8.4)
    ax_b1.set_title("Mapped alternate allele  (n=27,662)", fontsize=8.73, pad=2)
    panel_letter(ax_b1, "e", -0.04, 1.05)

    box_region(ax_b2, agro, None)
    ax_b2.set_ylabel("|Δ|  AgroNT", fontsize=8.4)
    ax_b2.set_title("Mean of three non-ref. bases  (n=26,518)", fontsize=8.73, pad=2)
    panel_letter(ax_b2, "f", -0.04, 1.05)


    save_fig(fig, "Figure3", tight=False)




def plot_fig4() -> None:
    apply_paper_style()
    prefer = ["Arabidopsis", "Rice", "Maize", "Tomato", "Soybean"]
    cnn = pd.read_csv(TABLES / "cross_species_results.csv")
    agro = pd.read_csv(TABLES / "cross_species_agront.csv")
    cnn = cnn.set_index("label").loc[prefer].reset_index()
    agro = agro.set_index("label").loc[prefer].reset_index()
    ab = pd.read_csv(TABLES / "input_length_ablation_both.csv")
    delta = agro["auroc"].to_numpy(dtype=float) - cnn["auroc"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(7.35, 6.55))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.05, 1.0], hspace=0.48, wspace=0.32,
        left=0.09, right=0.97, top=0.93, bottom=0.08,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    x = np.arange(len(prefer), dtype=float)
    w = 0.36
    ax_a.bar(x - w / 2, cnn["auroc"], w, color=C_CNN, edgecolor=EDGE, lw=0.7, hatch="//", label="CNN")
    ax_a.bar(x + w / 2, agro["auroc"], w, color=C_AGRONT, edgecolor=EDGE, lw=0.7, hatch="..", label="AgroNT")
    for xi, a, b, n in zip(x, cnn["auroc"], agro["auroc"], cnn["n_test"]):
        ax_a.text(xi - w / 2, a + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=7.32, color=C_CNN)
        ax_a.text(xi + w / 2, b + 0.008, f"{b:.3f}", ha="center", va="bottom", fontsize=7.32, color=C_AGRONT)
    ax_a.set_xticks(x, [f"{lab}\nn={int(n):,}" for lab, n in zip(prefer, cnn["n_test"])], fontsize=7.32)
    ax_a.set_ylabel("AUROC")
    ax_a.set_ylim(0.62, 1.05)
    ax_a.legend(frameon=False, loc="upper right", handlelength=1.2)
    ax_a.set_title("Cross-species transfer of Arabidopsis checkpoints", fontsize=8.97, pad=3)
    style_axis(ax_a)
    panel_letter(ax_a, "a")

    colors = [C_CNN if lab == "Arabidopsis" else "#59A14F" if lab in ("Rice", "Maize") else "#B07AA1" for lab in prefer]
    ax_b.barh(x[::-1], delta[::-1], color=colors[::-1], edgecolor=EDGE, lw=0.7, height=0.62)
    for yi, d in zip(x[::-1], delta[::-1]):
        ax_b.text(d + 0.004, yi, f"{d:.3f}", va="center", fontsize=7.55, color=INK)
    ax_b.set_yticks(x[::-1], prefer[::-1])
    ax_b.set_xlabel("ΔAUROC (AgroNT − CNN)")
    ax_b.set_xlim(0, max(delta) * 1.28)
    ax_b.axvline(0, color="#C0C0C0", lw=0.7)
    ax_b.set_title("ΔAUROC; not a species-unseen test", fontsize=8.97, pad=3)
    style_axis(ax_b)
    ax_b.tick_params(axis="y", length=0)
    panel_letter(ax_b, "b", -0.08)

    # combined heatmap: 2 models × 5 species for AUROC/AUPRC stacked as 2-row block? Keep CNN heat.
    # Use a 6-row matrix: CNN acc/auroc/auprc then AgroNT.
    mat = np.vstack([
        cnn[["accuracy", "auroc", "auprc"]].to_numpy(dtype=float).T,
        agro[["accuracy", "auroc", "auprc"]].to_numpy(dtype=float).T,
    ])
    im = ax_c.imshow(mat, cmap=HEAT, vmin=0.55, vmax=0.95, aspect="auto")
    ylabs = ["CNN acc.", "CNN AUROC", "CNN AUPRC", "AgroNT acc.", "AgroNT AUROC", "AgroNT AUPRC"]
    ax_c.set_yticks(range(6), ylabs, fontsize=7.32)
    ax_c.set_xticks(range(5), prefer, fontsize=7.55)
    ax_c.tick_params(length=0)
    ax_c.axhline(2.5, color="white", lw=1.4)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax_c.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7.32,
                      color="white" if val >= 0.80 else INK)
    ax_c.set_title("Accuracy / AUROC / AUPRC", fontsize=9.44, pad=3)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7.08, length=2.2)
    cbar.outline.set_linewidth(0.5)
    panel_letter(ax_c, "c", -0.12, 1.08)

    order = [6000, 3000, 1500]
    xx = np.arange(len(order))
    cnn_y = [float(ab[(ab["model"] == "cnn") & (ab["seq_len_bp"] == L)]["auroc"].iloc[0]) for L in order]
    agro_y = [float(ab[(ab["model"] == "agront") & (ab["seq_len_bp"] == L)]["auroc"].iloc[0]) for L in order]
    cnn_a = [float(ab[(ab["model"] == "cnn") & (ab["seq_len_bp"] == L)]["accuracy"].iloc[0]) for L in order]
    agro_a = [float(ab[(ab["model"] == "agront") & (ab["seq_len_bp"] == L)]["accuracy"].iloc[0]) for L in order]
    ax_d.plot(xx, cnn_y, "-o", color=C_CNN, lw=1.5, ms=6.5, label="CNN AUROC")
    ax_d.plot(xx, agro_y, "-o", color=C_AGRONT, lw=1.5, ms=6.5, label="AgroNT AUROC")
    ax_d.plot(xx, cnn_a, "--s", color=C_CNN, lw=1.0, ms=5, alpha=0.7, label="CNN acc.")
    ax_d.plot(xx, agro_a, "--s", color=C_AGRONT, lw=1.0, ms=5, alpha=0.7, label="AgroNT acc.")
    for xi, v in zip(xx, agro_y):
        ax_d.text(xi, v + 0.025, f"{v:.3f}", ha="center", fontsize=7.08, color=C_AGRONT)
    for xi, v in zip(xx, cnn_y):
        ax_d.text(xi, v - 0.045, f"{v:.3f}", ha="center", fontsize=7.08, color=C_CNN)
    ax_d.set_xticks(xx, ["6 kb", "3 kb", "1.5 kb"])
    ax_d.set_xlabel("Midpoint-cropped PGB window")
    ax_d.set_ylabel("Test score")
    ax_d.set_ylim(0.46, 1.05)
    ax_d.axhline(0.5, color="#C0C0C0", lw=0.7, ls=":")
    ax_d.legend(frameon=False, loc="upper right", fontsize=7.08, ncol=1, handlelength=1.4)
    ax_d.set_title("Retrained after midpoint crop", fontsize=9.44, pad=3)
    style_axis(ax_d)
    panel_letter(ax_d, "d")

    save_fig(fig, "Figure4", tight=False)


def plot_figs1() -> None:
    apply_paper_style()
    conc = pd.read_csv(TABLES / "tissue_label_concordance.csv")
    sp = pd.read_csv(TABLES / "tissue_pairwise_spearman.csv").to_numpy(dtype=float)
    agree = conc["agree_with_avg_label"].to_numpy(dtype=float)
    rho = conc["spearman_vs_gene_mean"].to_numpy(dtype=float)
    mean_agree = float(agree.mean())
    i_min = int(np.argmin(agree))

    fig = plt.figure(figsize=(7.35, 3.25))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 0.10], width_ratios=[1.05, 1.12, 1.18],
        hspace=0.10, wspace=0.30, left=0.07, right=0.98, top=0.90, bottom=0.14,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 1])
    ax_c = fig.add_subplot(gs[:, 2])

    order = np.argsort(agree)
    yy = np.arange(len(agree))
    ax_a.hlines(yy, 0.78, agree[order], color="#D0D0D0", lw=0.65, zorder=1)
    ax_a.scatter(agree[order], yy, s=16, color=C_AGRONT, edgecolors="white", lw=0.3, zorder=3)
    ax_a.axvline(mean_agree, color=ACCENT, ls=(0, (4, 2)), lw=1.15, zorder=4)
    ax_a.text(mean_agree - 0.006, len(agree) * 0.97, "mean 0.921\nCI 0.912–0.928",
              color=ACCENT, fontsize=7.32, ha="right", va="top")
    ax_a.set_xlabel("Agreement with mean label")
    ax_a.set_ylabel("Tissues / samples (sorted)")
    ax_a.set_yticks([])
    ax_a.set_xlim(0.775, 0.965)
    style_axis(ax_a)
    panel_letter(ax_a, "a")

    im = ax_b.imshow(sp, cmap=HEAT, vmin=0.5, vmax=1.0, aspect="auto")
    ax_b.set_xlabel("Tissue / sample index")
    ax_b.set_ylabel("Tissue / sample index")
    ax_b.tick_params(length=0, labelsize=7.08)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Spearman ρ", fontsize=7.79, labelpad=1)
    cbar.ax.tick_params(labelsize=7.08, length=2)
    cbar.outline.set_linewidth(0.5)
    style_axis(ax_b)
    panel_letter(ax_b, "b", -0.08)

    core = agree >= 0.85
    ax_c.scatter(rho[core], agree[core], s=22, color=C_CNN, edgecolors="white", lw=0.4, zorder=3, label="Other tissues")
    ax_c.scatter(
        [rho[i_min]], [agree[i_min]], s=38, color=ACCENT, edgecolors="white", lw=0.5, zorder=4, label="Lowest",
    )
    ax_c.axhline(mean_agree, color=ACCENT, ls=(0, (4, 2)), lw=1.0)
    ax_c.annotate(
        f"min {agree[i_min]:.3f}\nρ = {rho[i_min]:.2f}",
        xy=(rho[i_min], agree[i_min]),
        xytext=(0.72, 0.82),
        textcoords="data",
        fontsize=7.32,
        color=ACCENT,
        ha="left",
        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=0.7, mutation_scale=7),
    )
    ax_c.set_xlabel("Spearman vs gene mean")
    ax_c.set_ylabel("Agreement")
    ax_c.set_xlim(0.62, 0.98)
    ax_c.set_ylim(0.768, 0.965)
    ax_c.legend(frameon=False, loc="lower right", fontsize=7.08, handletextpad=0.3, borderaxespad=0.2)
    style_axis(ax_c)
    panel_letter(ax_c, "c")

    save_fig(fig, "FigureS1", tight=False)


def _ece(y, p, n_bins=15):
    """Expected calibration error with equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    tot, ece = 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        k = int(m.sum())
        if k == 0:
            continue
        tot += k
        ece += k / len(y) * abs(y[m].mean() - p[m].mean())
    return ece


def _reliability(ax, y, p, color, label, n_bins=15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    xs, ys, ns = np.array(xs), np.array(ys), np.array(ns)
    ece = _ece(y, p, n_bins)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls=(0, (4, 2)), zorder=1, label="_nolegend_")
    ax.plot(xs, ys, color=color, lw=1.4, marker="o", ms=4.2, mec="white", mew=0.7, zorder=3)
    ax.bar(xs, ys, width=1.0 / n_bins * 0.82, color=color, alpha=0.13, lw=0, zorder=2)
    ax.text(0.04, 0.95 if label == "CNN" else 0.72, f"{label}\nECE = {ece:.4f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.2,
            color=color, linespacing=1.4)
    return ece


def plot_figs2() -> None:
    """Reliability diagram and decision curve on the seed-42 PGB Arabidopsis test set.

    Replaces the previous eQTL-overlap panel, which did not match the caption in the
    manuscript. Inputs: pgb_binary_preds.npz (per-gene test probabilities) and
    decision_curve.csv (net benefit by threshold).
    """
    apply_paper_style()
    z = np.load(TABLES / "pgb_binary_preds.npz")
    y = z["y_true"].astype(float)
    p_cnn = z["p_cnn"].astype(float)
    p_ag = z["p_agront"].astype(float)
    dc = pd.read_csv(TABLES / "decision_curve.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.10, 3.05))
    ax_a, ax_b = axes

    _reliability(ax_a, y, p_cnn, C_CNN, "CNN")
    _reliability(ax_a, y, p_ag, C_AGRONT, "AgroNT")
    ax_a.set_xlabel("Mean predicted probability")
    ax_a.set_ylabel("Observed frequency")
    ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
    ax_a.set_title("Reliability diagram", fontsize=9.44, pad=3)
    style_axis(ax_a)
    panel_letter(ax_a, "a", -0.10, 1.06)

    ax_b.plot(dc["threshold"], dc["nb_cnn"], color=C_CNN, lw=1.5, marker="o", ms=3.4,
              mec="white", mew=0.5, label="CNN")
    ax_b.plot(dc["threshold"], dc["nb_agront"], color=C_AGRONT, lw=1.5, marker="s", ms=3.4,
              mec="white", mew=0.5, label="AgroNT")
    ax_b.plot(dc["threshold"], dc["nb_all"], color=MUTED, lw=1.0, ls=(0, (4, 2)),
              label="All genes high")
    ax_b.axhline(0.0, color=EDGE, lw=0.8)
    ax_b.set_xlabel("Threshold probability")
    ax_b.set_ylabel("Net benefit")
    ax_b.set_xlim(0.05, 0.95)
    ax_b.set_ylim(-0.10, 0.62)
    ax_b.set_title("Decision curve (high-expression call)", fontsize=9.44, pad=3)
    ax_b.legend(frameon=False, fontsize=7.55, loc="upper right", handlelength=1.5,
               bbox_to_anchor=(1.0, 0.90))
    ax_b.text(0.98, -0.235, "The all-genes-high curve falls below -8.5 beyond threshold 0.8 and is off-scale.",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=6.8, color=MUTED)
    style_axis(ax_b)
    panel_letter(ax_b, "b", -0.10, 1.06)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.24, wspace=0.34)
    save_fig(fig, "FigureS2", tight=False)



def main() -> None:
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_figs1()
    plot_figs2()
    print("wrote", OUT_PAPER)
    print("wrote", OUT_PACK)


if __name__ == "__main__":
    main()
