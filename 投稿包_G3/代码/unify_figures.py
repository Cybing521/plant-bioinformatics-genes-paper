#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manuscript figures — design language distilled from comparator papers
(Enformer Nat Methods 2021, AgroNT Commun Biol 2024, Peleke Nat Commun 2024,
Takou Genes 2025):

  * bold lowercase panel letters outside the top-left corner of every panel
  * no in-panel titles (panel content is described in the caption)
  * L-shaped axes (top/right spines removed), thin ticks, Arial/Helvetica
  * model contrast follows the AgroNT paper: CNN teal vs AgroNT purple
  * key statistics embedded inside panels (exact italic P, deltas)
  * blue-white annotated heatmaps with value cells and annotation strips

Chart plan (numbers come from 结果表/ CSVs; tables keep exact values):
  Fig1  a locked-protocol schematic | b metric dumbbell | c cost-effectiveness
  Fig2  a CV histogram | b concordance lollipop | c Spearman heatmap
  Fig3  a DeepLift per-base lollipop (CNN) | b matched mutagenesis, within-model
        normalized grouped dots (CNN vs AgroNT; ranks as in Table 2)
  Fig4  variant |Δ| boxplots, region-tinted, median-labelled, exact-P bracket
  Fig5  species×metric heatmap with dicot/monocot annotation strip
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.figure  # noqa: F401  (ensures mpl.figure is available)
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SKILL_SCRIPTS = Path.home() / ".agents/skills/scientific-figures/scripts"
sys.path.insert(0, str(SKILL_SCRIPTS))

from publication_style import (  # noqa: E402
    FigureStyle,
    apply_publication_style,
    create_subplots,
    finalize_figure,
    qa_before_save,
)

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
TABLES = ROOT / "结果表"
INTERP = ROOT / "可解释性"
PGB = REPO / "implementation" / "data" / "pgb"
OUT = ROOT / "图件"
SYNC = [
    REPO / "implementation" / "figures",
    ROOT / "论文" / "MDPI_格式" / "figures",
]

# ---- palette (model colors follow AgroNT Commun Biol 2024 model contrast) ----
C_CNN = "#00838F"      # teal   — compact CNN
C_AGRONT = "#7B1FA2"   # purple — foundation model
INK = "#1A1A1A"
MUTED = "#757575"
ACCENT_RED = "#B64342"

REGION_ORDER = ["utr5", "utr3", "promoter", "terminator"]
REGION_LABELS = ["5′UTR", "3′UTR", "Promoter", "Terminator"]
REGION_COLORS = {
    "utr5": "#3B6FB5",
    "utr3": "#45A5B8",
    "promoter": "#E8973A",
    "terminator": "#8C6BB8",
}

HEAT_BLUE = LinearSegmentedColormap.from_list(
    "nature_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#1A5276"]
)

P, U5, GAP, U3, T = 1000, 500, 20, 500, 1000


def assign_region(pos: int) -> str:
    if 0 <= pos < P:
        return "promoter"
    if P <= pos < P + U5:
        return "utr5"
    if P + U5 <= pos < P + U5 + GAP:
        return "gap"
    if P + U5 + GAP <= pos < P + U5 + GAP + U3:
        return "utr3"
    if P + U5 + GAP + U3 <= pos < P + U5 + GAP + U3 + T:
        return "terminator"
    return "out"


def sync_outputs(stem: str) -> None:
    for dest_dir in SYNC:
        dest_dir.mkdir(parents=True, exist_ok=True)
        for ext in (".png", ".pdf"):
            src = OUT / f"{stem}{ext}"
            if src.exists():
                shutil.copy2(src, dest_dir / src.name)


def panel_letter(ax, letter: str, x: float = -0.02, y: float = 1.04) -> None:
    """Bold lowercase panel letter outside the top-left corner (Nature style)."""
    ax.text(
        x, y, letter, transform=ax.transAxes,
        fontsize=13, fontweight="bold", color=INK,
        ha="right", va="bottom",
    )


def style_axis(ax, lw: float = 1.0) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(lw)
    ax.tick_params(width=lw, length=3.2, labelsize=mpl.rcParams["font.size"] - 1.5)


def rbox(ax, x, y, w, h, *, fc, ec, lw=1.2, radius=0.06) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
        )
    )


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.2) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=11,
            color=color, linewidth=lw, zorder=3,
            shrinkA=2, shrinkB=2,
        )
    )


# --------------------------------------------------------------------------
# Figure 1 — locked-protocol head-to-head benchmark (two data panels)
# --------------------------------------------------------------------------
def plot_fig1_benchmark() -> None:
    df = pd.read_csv(TABLES / "benchmark_results.csv")
    df = df.set_index("model").loc[["cnn", "agront"]].reset_index()
    cost = {"cnn": (2.7e6, 15), "agront": (16.2e6, 240)}
    cnn = df.loc[df["model"] == "cnn"].iloc[0]
    agr = df.loc[df["model"] == "agront"].iloc[0]

    apply_publication_style(FigureStyle(font_size=10.5, axes_linewidth=1.0))
    fig, axes = create_subplots(1, 2, figsize=(9.2, 4.0))
    fig.subplots_adjust(wspace=0.30)
    axb = axes[0]
    axc = axes[1]

    # --- (a) metric dumbbell ----------------------------------------------
    metrics = ["auprc", "auroc", "accuracy"]
    for i, m in enumerate(metrics):
        v0, v1 = float(cnn[m]), float(agr[m])
        axb.plot([v0, v1], [i, i], color="#BDBDBD", lw=1.6, zorder=1)
        axb.scatter([v0], [i], s=64, color=C_CNN, edgecolors="white",
                    linewidths=0.9, zorder=3,
                    label="CNN" if i == 0 else None)
        axb.scatter([v1], [i], s=64, color=C_AGRONT, edgecolors="white",
                    linewidths=0.9, zorder=3,
                    label="AgroNT" if i == 0 else None)
        axb.text(v0 - 0.010, i, f"{v0:.3f}", ha="right", va="center",
                 fontsize=7.6, color=C_CNN)
        axb.text(v1 + 0.010, i, f"{v1:.3f}", ha="left", va="center",
                 fontsize=7.6, color=C_AGRONT)
        axb.annotate(f"+{v1 - v0:.3f}", ((v0 + v1) / 2, i),
                     xytext=(0, 7), textcoords="offset points",
                     ha="center", fontsize=7.6, color=ACCENT_RED)
    axb.set_yticks(range(len(metrics)))
    axb.set_yticklabels([m.upper() for m in metrics])
    axb.set_xlabel("Score on held-out test set")
    axb.set_xlim(0.70, 1.03)
    axb.set_ylim(-0.6, len(metrics) - 0.3)
    axb.legend(loc="lower left", fontsize=8, borderaxespad=0.2)
    style_axis(axb)
    panel_letter(axb, "a")

    # --- (b) performance vs computational cost ----------------------------
    for _, r in df.iterrows():
        params, mins = cost[r["model"]]
        color = C_CNN if r["model"] == "cnn" else C_AGRONT
        axc.scatter(r["auroc"], params, s=170, color=color,
                    edgecolors="white", linewidths=1.0, zorder=3)
        n_m = params / 1e6
        param_txt = (
            f"{n_m:.1f}M params" if r["model"] == "cnn"
            else f"{n_m:.1f}M trainable"
        )
        label = (
            f"{('CNN' if r['model'] == 'cnn' else 'AgroNT')}\n"
            f"AUROC {r['auroc']:.3f}\n{mins} min · {param_txt}"
        )
        xytext, ha, va = ((14, 24), "left", "bottom") if r["model"] == "cnn" else (
            (-16, -30), "right", "top")
        axc.annotate(
            label, (r["auroc"], params),
            textcoords="offset points", xytext=xytext,
            fontsize=7.6, ha=ha, va=va, color=INK, linespacing=1.25,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                            shrinkA=1, shrinkB=4),
        )
    axc.set_yscale("log")
    axc.set_xlabel("AUROC")
    axc.set_ylabel("Trainable parameters (log scale)")
    axc.set_xlim(0.80, 0.98)
    axc.set_ylim(1.2e6, 4.0e7)
    style_axis(axc)
    panel_letter(axc, "b")

    qa_before_save(fig)
    finalize_figure(fig, str(OUT / "fig3_benchmark"), formats=["png", "pdf"], dpi=300, tight=True)
    sync_outputs("fig3_benchmark")


# --------------------------------------------------------------------------
# Figure 2 — tissue heterogeneity of expression labels
# --------------------------------------------------------------------------
def plot_fig2_tissue() -> None:
    apply_publication_style(FigureStyle(font_size=10.5, axes_linewidth=1.0))

    all_rows: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        path = PGB / f"arabidopsis_thaliana_{split}.fa"
        with open(path) as f:
            for line in f:
                if not line.startswith(">"):
                    continue
                fields = line[1:].strip().split("|")
                gid = fields[0].split(".")[0]
                if gid in all_rows:
                    continue
                all_rows[gid] = np.array([float(v) for v in fields[1:] if v != ""], dtype=float)
    n_t = max(len(r) for r in all_rows.values())
    X = np.full((len(all_rows), n_t), np.nan)
    for i, row in enumerate(all_rows.values()):
        X[i, : len(row)] = row
    gmean = np.nanmean(X, axis=1)
    gstd = np.nanstd(X, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(gmean > 0, gstd / gmean, np.nan)
    cv_ok = cv[np.isfinite(cv)]
    cv_q75 = float(np.nanpercentile(cv_ok, 75))

    conc = pd.read_csv(TABLES / "tissue_label_concordance.csv")
    sp = pd.read_csv(TABLES / "tissue_pairwise_spearman.csv").to_numpy(dtype=float)
    agree = conc["agree_with_avg_label"].sort_values().to_numpy()
    mean_agree = float(agree.mean())

    fig, axes = create_subplots(1, 3, figsize=(13.2, 3.7))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    ax.hist(cv_ok, bins=40, color="#3B6FB5", alpha=0.88,
            edgecolor="white", linewidth=0.3, zorder=2)
    ax.axvline(cv_q75, color=ACCENT_RED, ls=(0, (5, 3)), lw=1.6, zorder=3)
    ax.text(cv_q75 + 0.09, ax.get_ylim()[1] * 0.94,
            f"Q75 = {cv_q75:.2f}", color=ACCENT_RED, fontsize=8.6, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                      boxstyle="round,pad=0.15"))
    ax.set_xlabel("Cross-tissue coefficient of variation")
    ax.set_ylabel("Genes")
    ax.set_xlim(0, 4.2)
    style_axis(ax)
    panel_letter(ax, "a")

    ax = axes[1]
    yy = np.arange(len(agree))
    ax.hlines(yy, 0.75, agree, color="#D0D0D0", lw=0.8, zorder=1)
    ax.scatter(agree, yy, s=22, color=C_AGRONT, edgecolors="white",
               linewidths=0.4, zorder=3)
    ax.axvline(mean_agree, color=ACCENT_RED, ls=(0, (5, 3)), lw=1.6, zorder=4)
    ax.text(mean_agree - 0.004, len(agree) * 0.97,
            f"mean = {mean_agree:.3f}", color=ACCENT_RED, fontsize=8.6,
            ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                      boxstyle="round,pad=0.15"))
    ax.set_xlabel("Agreement with averaged high/low label")
    ax.set_ylabel("Tissues (sorted)")
    ax.set_yticks([])
    ax.set_xlim(0.75, 0.97)
    style_axis(ax)
    panel_letter(ax, "b")

    ax = axes[2]
    im = ax.imshow(sp, cmap=HEAT_BLUE, vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xlabel("Tissue/sample index")
    ax.set_ylabel("Tissue/sample index")
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Spearman ρ", fontsize=8.6)
    cbar.ax.tick_params(labelsize=7.6, width=0.8, length=2.5)
    cbar.outline.set_linewidth(0.6)
    style_axis(ax)
    panel_letter(ax, "c", x=-0.08)

    qa_before_save(fig)
    finalize_figure(fig, str(OUT / "fig7_tissue_heterogeneity"), formats=["png", "pdf"], dpi=300)
    sync_outputs("fig7_tissue_heterogeneity")


# --------------------------------------------------------------------------
# Figure 3 — region importance: DeepLift motif path + matched mutagenesis
# --------------------------------------------------------------------------
def plot_fig3_region_importance() -> None:
    dl = pd.read_csv(INTERP / "region_importance.csv")
    dl = dl[dl["region"] != "gap"].set_index("region").loc[REGION_ORDER].reset_index()
    dl = dl.sort_values("importance_per_bp", ascending=True).reset_index(drop=True)

    mm = pd.read_csv(TABLES / "same_method_region_importance.csv").set_index("region").loc[REGION_ORDER]
    cnn_norm = (mm["cnn_mutagenesis"] / mm["cnn_mutagenesis"].max()).to_numpy()
    agro_norm = (mm["agront_mutagenesis"] / mm["agront_mutagenesis"].max()).to_numpy()

    apply_publication_style(FigureStyle(font_size=10.5, axes_linewidth=1.0))
    fig, axes = create_subplots(1, 2, figsize=(11.6, 3.9))
    fig.subplots_adjust(wspace=0.42)

    # (a) DeepLift lollipop, CNN
    ax = axes[0]
    y = np.arange(len(dl))
    vals = dl["importance_per_bp"].to_numpy()
    colors = [REGION_COLORS[r] for r in dl["region"]]
    labels = [{"utr5": "5′UTR", "utr3": "3′UTR", "promoter": "Promoter",
               "terminator": "Terminator"}[r] for r in dl["region"]]
    ax.hlines(y, 0, vals, color="#D0D0D0", lw=1.4, zorder=1)
    ax.scatter(vals, y, s=86, c=colors, edgecolors="white", linewidths=0.9, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.04, yi, f"{v:.4f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean |DeepLift attribution| per bp")
    ax.set_xlim(0, max(vals) * 1.24)
    style_axis(ax)
    ax.tick_params(axis="y", length=0)
    panel_letter(ax, "a")

    # (b) matched sliding-window mutagenesis, normalized within model
    ax = axes[1]
    yy = np.arange(len(REGION_ORDER))[::-1]  # utr5 on top
    for yi, cv_, av_, reg, lab in zip(
        yy, cnn_norm, agro_norm, REGION_ORDER, REGION_LABELS
    ):
        ax.plot([min(cv_, av_), max(cv_, av_)], [yi, yi],
                color="#D0D0D0", lw=1.2, zorder=1)
        ax.scatter(cv_, yi + 0.10, s=74, color=C_CNN, edgecolors="white",
                   linewidths=1.1, zorder=3)
        ax.scatter(av_, yi - 0.14, s=74, color=C_AGRONT, edgecolors="white",
                   linewidths=1.1, zorder=3)
        raw_c = mm["cnn_mutagenesis"][reg]
        raw_a = mm["agront_mutagenesis"][reg]
        ax.annotate(f"{raw_c:.3f}", (cv_, yi + 0.10), xytext=(7, 4),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.4, color=C_CNN)
        ax.annotate(f"{raw_a:.3f}", (av_, yi - 0.14), xytext=(7, -5),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.4, color=C_AGRONT)
    ax.scatter([], [], s=58, color=C_CNN, label="CNN")
    ax.scatter([], [], s=58, color=C_AGRONT, label="AgroNT")
    ax.legend(loc="lower left", fontsize=8.4, handletextpad=0.2,
              borderaxespad=0.2)
    ax.set_yticks(yy)
    ax.set_yticklabels(REGION_LABELS)
    ax.set_xlabel("Mutagenesis importance per base\n(normalized to model maximum; raw values beside points)")
    ax.set_xlim(-0.05, 1.22)
    ax.set_ylim(-0.75, len(REGION_ORDER) - 0.25)
    style_axis(ax)
    ax.tick_params(axis="y", length=0)
    panel_letter(ax, "b")

    qa_before_save(fig)
    finalize_figure(fig, str(OUT / "fig4_region_importance"), formats=["png", "pdf"], dpi=300)
    sync_outputs("fig4_region_importance")


# --------------------------------------------------------------------------
# Figure 4 — predicted variant effects by flanking region
# --------------------------------------------------------------------------
def plot_fig4_variant_boxplot() -> None:
    cnn = pd.read_csv(TABLES / "scores.tsv", sep="\t")
    cnn["region"] = cnn["flank_pos"].astype(int).map(assign_region)
    cnn = cnn[cnn["region"].isin(REGION_ORDER)]

    agro_path = TABLES / "agront_scores_by_site.tsv"
    if not agro_path.exists():
        agro_path = TABLES / "agront_scores.tsv"
    has_agro = agro_path.exists()
    if has_agro:
        agro = pd.read_csv(agro_path, sep="\t")
        if "mean_abs_delta" in agro.columns and "abs_delta" not in agro.columns:
            agro["abs_delta"] = agro["mean_abs_delta"]
        agro["region"] = agro["flank_pos"].astype(int).map(assign_region)
        agro = agro[agro["region"].isin(REGION_ORDER)]

    apply_publication_style(FigureStyle(font_size=10.5, axes_linewidth=1.0))
    ncols = 2 if has_agro else 1
    fig, axes = create_subplots(1, ncols, figsize=(12.4 if has_agro else 7.2, 3.8))
    if ncols == 1:
        axes = [axes[0]]

    def _box(ax, scores, p_text=None):
        data = [scores.loc[scores["region"] == r, "abs_delta"].to_numpy()
                for r in REGION_ORDER]
        medians = [float(np.median(d)) for d in data]
        bp = ax.boxplot(
            data,
            tick_labels=REGION_LABELS,
            showfliers=False,
            patch_artist=True,
            widths=0.52,
            medianprops=dict(color=INK, linewidth=1.3),
            whiskerprops=dict(color="#4A4A4A", linewidth=0.9),
            capprops=dict(color="#4A4A4A", linewidth=0.9),
            boxprops=dict(linewidth=1.1),
        )
        for patch, reg in zip(bp["boxes"], REGION_ORDER):
            patch.set_facecolor(REGION_COLORS[reg])
            patch.set_alpha(0.30)
            patch.set_edgecolor(REGION_COLORS[reg])
        for xi, m in enumerate(medians, start=1):
            ax.annotate(
                f"{m:.4f}", (xi, m), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.8, color=INK,
            )
        whisker_tops = []
        for d in data:
            q1, q3 = np.percentile(d, [25, 75])
            hi = d[d <= q3 + 1.5 * (q3 - q1)]
            whisker_tops.append(float(hi.max()) if len(hi) else float(q3))
        wmax = max(whisker_tops) if whisker_tops else 0.01
        y_br = wmax * 1.12
        if p_text:
            x0, x1 = 1, 2
            ax.plot([x0, x0, x1, x1], [y_br + wmax * 0.05, y_br, y_br, y_br + wmax * 0.05],
                    color=INK, lw=0.9)
            ax.text((x0 + x1) / 2, y_br + wmax * 0.09, p_text,
                    ha="center", va="bottom", fontsize=8.2, color=INK)
            ax.set_ylim(0, y_br + wmax * 0.42)
        else:
            ax.set_ylim(0, y_br + wmax * 0.28)
        ax.set_ylabel("|Δ| predicted expression probability")
        style_axis(ax)
        ax.tick_params(axis="x", length=0)
        return ax

    _box(axes[0], cnn, "$P = 1.2\\times10^{-61}$")
    panel_letter(axes[0], "a")
    if has_agro:
        pair_p = TABLES / "agront_variant_region_pairwise.csv"
        p_text = None
        if pair_p.exists():
            pr = pd.read_csv(pair_p)
            hit = pr[pr["hypothesis"].astype(str).str.contains("utr5 > utr3", na=False)]
            if len(hit):
                pv = float(hit.iloc[0]["p"])
                if pv < 0.05:
                    p_text = f"$P = {pv:.1e}$"
        _box(axes[1], agro, p_text)
        panel_letter(axes[1], "b")

    qa_before_save(fig)
    finalize_figure(fig, str(OUT / "fig5b_variant_by_region"), formats=["png", "pdf"], dpi=300)
    sync_outputs("fig5b_variant_by_region")


# --------------------------------------------------------------------------
# Figure 5 — zero-shot cross-species transfer heatmap
# --------------------------------------------------------------------------
def plot_fig5_cross_species() -> None:
    cnn = pd.read_csv(TABLES / "cross_species_results.csv")
    prefer = ["Arabidopsis", "Rice", "Maize", "Tomato", "Soybean"]
    if "label" not in cnn.columns and "species" in cnn.columns:
        cnn["label"] = cnn["species"]
    cnn = cnn.set_index("label").loc[[x for x in prefer if x in set(cnn["label"])]].reset_index()

    agro_path = TABLES / "cross_species_agront.csv"
    has_agro = agro_path.exists()
    if has_agro:
        agro = pd.read_csv(agro_path)
        if "label" not in agro.columns and "species" in agro.columns:
            agro["label"] = agro["species"]
        agro = agro.set_index("label").loc[[x for x in prefer if x in set(agro["label"])]].reset_index()

    apply_publication_style(FigureStyle(font_size=10.5, axes_linewidth=1.0))
    nrows = 2 if has_agro else 1
    fig, axes = create_subplots(nrows, 1, figsize=(8.2, 6.2 if has_agro else 3.6))
    if nrows == 1:
        axes = [axes[0]]

    def _heat(ax, df, letter):
        mat = df[["accuracy", "auroc", "auprc"]].to_numpy(dtype=float).T
        im = ax.imshow(mat, cmap=HEAT_BLUE, vmin=0.55, vmax=0.95, aspect="auto")
        ax.set_yticks(range(3))
        ax.set_yticklabels(["Accuracy", "AUROC", "AUPRC"])
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["label"].tolist())
        ax.tick_params(length=0)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if val >= 0.80 else INK)
        groups = [("Self", 0, 0, "#8C6BB8"), ("Monocots", 1, 2, "#E8973A"),
                  ("Dicots", 3, 4, "#45A5B8")]
        for name, c0, c1, col in groups:
            if c1 >= len(df):
                continue
            xc0, xc1 = c0 - 0.5, c1 + 0.5
            ax.plot([xc0, xc1], [-0.68, -0.68], color=col, lw=2.6,
                    solid_capstyle="butt", clip_on=False)
            ax.text((xc0 + xc1) / 2, -0.88, name, ha="center", va="bottom",
                    fontsize=8.4, color=col)
        cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.03)
        cbar.set_label("Score", fontsize=8.6)
        cbar.ax.tick_params(labelsize=7.6, width=0.8, length=2.5)
        cbar.outline.set_linewidth(0.6)
        panel_letter(ax, letter, x=-0.08)
        return ax

    _heat(axes[0], cnn, "a")
    if has_agro:
        _heat(axes[1], agro, "b")

    qa_before_save(fig)
    finalize_figure(fig, str(OUT / "fig6_cross_species"), formats=["png", "pdf"], dpi=300)
    sync_outputs("fig6_cross_species")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_fig1_benchmark()
    plot_fig2_tissue()
    plot_fig3_region_importance()
    plot_fig4_variant_boxplot()
    plot_fig5_cross_species()
    print(f">>> unified figures written under {OUT}")


if __name__ == "__main__":
    main()
