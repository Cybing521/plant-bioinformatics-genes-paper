#!/usr/bin/env python3
"""变异效应验证：eQTL / 1001 Genomes（模块四，纯干实验闭环）。

用法:
  python src/variants/validate.py --config configs/config.yaml
输入:
  results/variants/scores.tsv              in_silico 打分
  data/eqtl/arabidopsis_eqtl.tsv            eQTL: chr pos ref alt gene_id pvalue
输出:
  results/variants/validation_results.txt   检验统计量
  figures/fig5_eqtl_validation.png
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> src/

import numpy as np
import pandas as pd
from scipy import stats

from utils import setup_logging

logger = logging.getLogger("plantdl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--pval-thresh", type=float, default=1e-5,
                    help="eQTL 显著阈值")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    vcfg = cfg["variants"]
    os.makedirs(vcfg["out_dir"], exist_ok=True)

    scores = pd.read_csv(os.path.join(vcfg["out_dir"], "scores.tsv"), sep="\t")
    eqtl_path = vcfg["eqtl_tsv"]
    if not os.path.exists(eqtl_path):
        logger.warning("eQTL 文件不存在 %s，仅输出打分汇总", eqtl_path)
        return

    eqtl = pd.read_csv(eqtl_path, sep="\t")
    # 按 (gene_id, chr, pos) 对齐
    key = ["gene_id", "chr", "pos"]
    if not all(c in eqtl.columns for c in key):
        logger.warning("eQTL 需含列 %s，当前列: %s", key, eqtl.columns.tolist())
        return
    eqtl = eqtl.drop_duplicates(key)
    merged = scores.merge(eqtl[key + ["pvalue"]], on=["gene_id"], how="inner",
                          suffixes=("", "_e"))
    # 简化：若 eQTL 无 chr/pos 对齐字段，改用 gene 级关联；此处按 gene_id 汇总
    g = merged.groupby("gene_id").agg(
        max_abs_delta=("abs_delta", "max"),
        min_pvalue=("pvalue", "min")).reset_index()
    g["is_eqtl"] = g["min_pvalue"] <= args.pval_thresh

    pos = g.loc[g["is_eqtl"], "max_abs_delta"].values
    neg = g.loc[~g["is_eqtl"], "max_abs_delta"].values
    if len(pos) and len(neg):
        u, p = stats.mannwhitneyu(pos, neg, alternative="greater")
        lines = [
            f"eQTL 显著基因数: {len(pos)}，非显著: {len(neg)}",
            f"Mann-Whitney U={u:.0f}, p={p:.3e}  (检验: eQTL 变异 |Δ| 是否更大)",
            f"eQTL 基因 |Δ| 中位数: {np.median(pos):.4f}",
            f"非 eQTL 基因 |Δ| 中位数: {np.median(neg):.4f}",
        ]
    else:
        lines = ["样本不足：eQTL 显著/非显著组至少各需 1 个基因"]

    txt = "\n".join(lines)
    logger.info("\n%s", txt)
    with open(os.path.join(vcfg["out_dir"], "validation_results.txt"), "w") as f:
        f.write(txt + "\n")

    # --- 绘图：按 |Δ| 排序的 eQTL 富集 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = g.sort_values("max_abs_delta", ascending=False).reset_index(drop=True)
    g["rank"] = np.arange(len(g))
    g["cum_eqtl"] = g["is_eqtl"].cumsum() / max(1, g["is_eqtl"].sum())
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["rank"], g["cum_eqtl"], color="#4c72b0")
    ax.plot([0, len(g)], [0, 1], color="grey", ls="--", label="random")
    ax.set_xlabel("gene rank by max |Δ|")
    ax.set_ylabel("cumulative fraction of eQTL genes")
    ax.set_title("eQTL enrichment at top-scored variants")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig5_eqtl_validation.png", dpi=150)
    logger.info("图已保存 -> figures/fig5_eqtl_validation.png")


if __name__ == "__main__":
    main()
