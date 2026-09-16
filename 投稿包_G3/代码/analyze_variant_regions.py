#!/usr/bin/env python3
"""侧翼区变异效应细粒度统计（不重训）。

输入: scores.tsv (gene_id, flank_pos, abs_delta, ...)
区域映射与 build_dataset / prep_variants 一致:
  promoter [0,P), utr5 [P,P+U5), gap [...), utr3 [...), terminator [...]

产物:
  results/variant_region_stats.csv
  results/variant_region_pairwise.csv
  figures/fig5b_variant_by_region.png
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="投稿包_Genes/结果表/scores.tsv")
    ap.add_argument("--eqtl", default="投稿包_Genes/结果表/eqtl_scores.tsv")
    ap.add_argument("--out-dir", default="implementation/results")
    ap.add_argument("--fig-dir", default="implementation/figures")
    args = ap.parse_args()

    df = pd.read_csv(args.scores, sep="\t")
    df["region"] = df["flank_pos"].astype(int).map(assign_region)
    df = df[df["region"] != "out"].copy()
    # exclude artificial gap
    df_nongap = df[df["region"] != "gap"].copy()

    rows = []
    for region, g in df_nongap.groupby("region"):
        rows.append(
            {
                "region": region,
                "n": len(g),
                "median_abs_delta": float(g["abs_delta"].median()),
                "mean_abs_delta": float(g["abs_delta"].mean()),
                "q75_abs_delta": float(g["abs_delta"].quantile(0.75)),
                "q95_abs_delta": float(g["abs_delta"].quantile(0.95)),
            }
        )
    stats = pd.DataFrame(rows).sort_values("median_abs_delta", ascending=False)

    # pairwise Mann-Whitney (greater)
    order = ["utr5", "utr3", "promoter", "terminator"]
    pairs = []
    for i, a in enumerate(order):
        for b in order[i + 1 :]:
            xa = df_nongap.loc[df_nongap["region"] == a, "abs_delta"].values
            xb = df_nongap.loc[df_nongap["region"] == b, "abs_delta"].values
            u, p = mannwhitneyu(xa, xb, alternative="greater")
            pairs.append(
                {
                    "region_a": a,
                    "region_b": b,
                    "hypothesis": f"{a} > {b}",
                    "U": float(u),
                    "p": float(p),
                    "median_a": float(np.median(xa)),
                    "median_b": float(np.median(xb)),
                }
            )
    pair_df = pd.DataFrame(pairs)

    # UTR pooled vs terminator / promoter
    utr = df_nongap.loc[df_nongap["region"].isin(["utr5", "utr3"]), "abs_delta"].values
    ter = df_nongap.loc[df_nongap["region"] == "terminator", "abs_delta"].values
    pro = df_nongap.loc[df_nongap["region"] == "promoter", "abs_delta"].values
    u_ut, p_ut = mannwhitneyu(utr, ter, alternative="greater")
    u_up, p_up = mannwhitneyu(utr, pro, alternative="greater")

    # eQTL enrichment: among scored eQTL variants, region medians + compare to background
    eqtl_note = "eqtl_file_missing"
    if os.path.exists(args.eqtl):
        eq = pd.read_csv(args.eqtl, sep="\t")
        eq["region"] = eq["flank_pos"].astype(int).map(assign_region)
        eq = eq[eq["region"].isin(order)]
        # top eQTL (p < 1e-5) vs all variants median abs_delta
        top = eq[eq["eqtl_pvalue"] < 1e-5] if "eqtl_pvalue" in eq.columns else eq
        bg_med = float(df_nongap["abs_delta"].median())
        top_med = float(top["abs_delta"].median()) if len(top) else float("nan")
        if len(top) > 10:
            u_e, p_e = mannwhitneyu(
                top["abs_delta"].values, df_nongap["abs_delta"].values, alternative="greater"
            )
        else:
            u_e, p_e = float("nan"), float("nan")
        eqtl_note = (
            f"n_eqtl_mapped={len(eq)}; n_eqtl_p<1e-5={len(top)}; "
            f"median_|Δ|_eqtl={top_med:.6f} vs background={bg_med:.6f}; "
            f"MWU(eqtl>bg) p={p_e:.3e}"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    stats_path = os.path.join(args.out_dir, "variant_region_stats.csv")
    pair_path = os.path.join(args.out_dir, "variant_region_pairwise.csv")
    stats.to_csv(stats_path, index=False)
    pair_df.to_csv(pair_path, index=False)

    summary_path = os.path.join(args.out_dir, "variant_region_summary.txt")
    lines = [
        f"n_scored_nongap = {len(df_nongap)}",
        f"UTR_pooled median_|Δ| = {np.median(utr):.6f} (n={len(utr)})",
        f"promoter median_|Δ| = {np.median(pro):.6f} (n={len(pro)})",
        f"terminator median_|Δ| = {np.median(ter):.6f} (n={len(ter)})",
        f"MWU UTR>terminator: U={u_ut:.0f}, p={p_ut:.3e}",
        f"MWU UTR>promoter: U={u_up:.0f}, p={p_up:.3e}",
        eqtl_note,
    ]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    order_plot = ["utr5", "utr3", "promoter", "terminator"]
    data = [df_nongap.loc[df_nongap["region"] == r, "abs_delta"].values for r in order_plot]
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=300)
    bp = ax.boxplot(
        data,
        labels=["5′UTR", "3′UTR", "Promoter", "Terminator"],
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
    )
    colors = ["#4c72b0", "#55a868", "#dd8452", "#8172b3"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylabel("|Δ| predicted expression probability")
    ax.set_title("Cis-regulatory variant effect by flanking region")
    fig.tight_layout()
    fig_path = os.path.join(args.fig_dir, "fig5b_variant_by_region.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f">>> wrote {stats_path}")
    print(f">>> wrote {pair_path}")
    print(f">>> wrote {summary_path}")
    print(f">>> wrote {fig_path}")


if __name__ == "__main__":
    main()
