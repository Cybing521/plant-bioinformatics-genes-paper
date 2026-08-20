#!/usr/bin/env python3
"""PGB 组织异质性分析（不重训模型）。

从 FASTA 头解析每基因多组织表达值，量化：
  - 组织间 Spearman 相关
  - 组织平均标签 vs 单组织标签一致性
  - 高变异基因比例（跨组织 CV）

产物:
  results/tissue_heterogeneity_summary.txt
  results/tissue_label_concordance.csv
  results/tissue_pairwise_spearman.csv
  figures/fig7_tissue_heterogeneity.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_pgb_fasta(path: str) -> tuple[list[str], np.ndarray]:
    gene_ids, mats = [], []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            fields = line[1:].strip().split("|")
            gid = fields[0].split(".")[0]
            vals = [float(v) for v in fields[1:] if v != ""]
            gene_ids.append(gid)
            mats.append(vals)
    # pad to max tissues
    n_t = max(len(r) for r in mats)
    X = np.full((len(mats), n_t), np.nan, dtype=np.float64)
    for i, row in enumerate(mats):
        X[i, : len(row)] = row
    return gene_ids, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgb-dir", default="data/pgb")
    ap.add_argument("--species", default="arabidopsis_thaliana")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--fig-dir", default="figures")
    args = ap.parse_args()

    splits = {}
    for split in ("train", "validation", "test"):
        path = os.path.join(args.pgb_dir, f"{args.species}_{split}.fa")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        gids, X = parse_pgb_fasta(path)
        splits[split] = (gids, X)

    # use train median of gene-means for global binary threshold
    tr_X = splits["train"][1]
    tr_mean = np.nanmean(tr_X, axis=1)
    thr = float(np.median(tr_mean))

    # concatenate all genes for tissue stats (unique by gene; prefer train)
    all_rows = {}
    for split, (gids, X) in splits.items():
        for i, gid in enumerate(gids):
            if gid not in all_rows:
                all_rows[gid] = X[i]

    mat = np.vstack(list(all_rows.values()))
    n_genes, n_tissues = mat.shape
    gene_mean = np.nanmean(mat, axis=1)
    gene_std = np.nanstd(mat, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gene_cv = np.where(gene_mean > 0, gene_std / gene_mean, np.nan)

    # pairwise spearman among tissues (complete cases)
    from scipy.stats import spearmanr

    spearman = np.eye(n_tissues)
    for i in range(n_tissues):
        for j in range(i + 1, n_tissues):
            mask = np.isfinite(mat[:, i]) & np.isfinite(mat[:, j])
            if mask.sum() < 100:
                r = np.nan
            else:
                r, _ = spearmanr(mat[mask, i], mat[mask, j])
            spearman[i, j] = spearman[j, i] = r

    # label concordance: tissue-averaged binary vs each tissue binary (tissue median)
    avg_label = (gene_mean >= thr).astype(int)
    concordance = []
    for t in range(n_tissues):
        col = mat[:, t]
        ok = np.isfinite(col)
        t_thr = float(np.median(col[ok]))
        t_lab = (col >= t_thr).astype(int)
        agree = (t_lab[ok] == avg_label[ok]).mean()
        concordance.append(
            {
                "tissue_index": t,
                "n_genes": int(ok.sum()),
                "tissue_median": t_thr,
                "agree_with_avg_label": float(agree),
                "spearman_vs_gene_mean": float(
                    spearmanr(col[ok], gene_mean[ok]).correlation
                ),
            }
        )
    conc_df = pd.DataFrame(concordance)

    # high tissue variance genes: top CV quartile among expressed genes
    cv_ok = gene_cv[np.isfinite(gene_cv)]
    cv_q75 = float(np.nanpercentile(gene_cv, 75))
    high_cv_frac = float(np.nanmean(gene_cv >= cv_q75))

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    conc_path = os.path.join(args.out_dir, "tissue_label_concordance.csv")
    conc_df.to_csv(conc_path, index=False)
    sp_path = os.path.join(args.out_dir, "tissue_pairwise_spearman.csv")
    pd.DataFrame(spearman).to_csv(sp_path, index=False)

    offdiag = spearman[np.triu_indices(n_tissues, k=1)]
    offdiag = offdiag[np.isfinite(offdiag)]

    summary = [
        f"n_genes_unique = {n_genes}",
        f"n_tissues_or_samples = {n_tissues}",
        f"train_gene_mean_median_threshold = {thr:.6f}",
        f"gene_mean_expression: median={np.nanmedian(gene_mean):.4f}, IQR=[{np.nanpercentile(gene_mean,25):.4f}, {np.nanpercentile(gene_mean,75):.4f}]",
        f"cross_tissue_CV: median={np.nanmedian(gene_cv):.4f}, Q75={cv_q75:.4f}",
        f"fraction_genes_CV>=Q75 = {high_cv_frac:.4f}",
        f"pairwise_tissue_Spearman: median={np.nanmedian(offdiag):.4f}, IQR=[{np.nanpercentile(offdiag,25):.4f}, {np.nanpercentile(offdiag,75):.4f}]",
        f"avg_label_vs_tissue_label agreement: mean={conc_df['agree_with_avg_label'].mean():.4f}, min={conc_df['agree_with_avg_label'].min():.4f}, max={conc_df['agree_with_avg_label'].max():.4f}",
        f"tissue_vs_gene_mean Spearman: mean={conc_df['spearman_vs_gene_mean'].mean():.4f}, min={conc_df['spearman_vs_gene_mean'].min():.4f}",
    ]
    sum_path = os.path.join(args.out_dir, "tissue_heterogeneity_summary.txt")
    with open(sum_path, "w") as f:
        f.write("\n".join(summary) + "\n")
    print("\n".join(summary))

    # figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=300)

    # (a) CV distribution
    axes[0].hist(cv_ok, bins=40, color="#4c72b0", edgecolor="white", linewidth=0.4)
    axes[0].axvline(cv_q75, color="#c44e52", ls="--", lw=1.2, label=f"Q75={cv_q75:.2f}")
    axes[0].set_xlabel("Cross-tissue CV")
    axes[0].set_ylabel("Genes")
    axes[0].set_title("(a) Expression variability across tissues")
    axes[0].legend(fontsize=8)

    # (b) concordance
    axes[1].hist(
        conc_df["agree_with_avg_label"],
        bins=20,
        color="#55a868",
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].axvline(
        conc_df["agree_with_avg_label"].mean(),
        color="#c44e52",
        ls="--",
        lw=1.2,
        label=f"mean={conc_df['agree_with_avg_label'].mean():.3f}",
    )
    axes[1].set_xlabel("Agreement with tissue-averaged high/low label")
    axes[1].set_ylabel("Tissues/samples")
    axes[1].set_title("(b) Single-tissue vs averaged labels")
    axes[1].legend(fontsize=8)

    # (c) spearman heatmap (downsample labels)
    im = axes[2].imshow(spearman, cmap="viridis", vmin=0.2, vmax=1.0, aspect="auto")
    axes[2].set_title("(c) Tissue–tissue Spearman correlation")
    axes[2].set_xlabel("Tissue/sample index")
    axes[2].set_ylabel("Tissue/sample index")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig_path = os.path.join(args.fig_dir, "fig7_tissue_heterogeneity.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f">>> wrote {sum_path}")
    print(f">>> wrote {conc_path}")
    print(f">>> wrote {sp_path}")
    print(f">>> wrote {fig_path}")


if __name__ == "__main__":
    main()
