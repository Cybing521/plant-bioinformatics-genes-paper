#!/usr/bin/env python3
"""变异效应区域富集验证：侧翼区不同区域的变异，其模型 |Δ| 应反映区域重要性。

验证逻辑：CNN 区域重要性是 5'UTR>3'UTR>启动子>终止子，
因此 UTR 区变异的 |Δ预测概率| 应显著高于终止子区变异。
产物: results/variants/region_validation.txt, figures/fig5_region_validation.png
"""
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy import stats

from utils import setup_logging

logger = logging.getLogger("plantdl")


def region_of(fp, bounds):
    for reg, (s, e) in bounds.items():
        if s <= fp < e:
            return reg
    return "unknown"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--scores", default="results/variants/scores.tsv")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    vcfg = cfg["variants"]
    os.makedirs(vcfg["out_dir"], exist_ok=True)

    scores = pd.read_csv(args.scores, sep="\t")
    bounds = json.load(open(os.path.join(cfg["data"]["out_dir"], "region_bounds.json")))
    scores["region"] = scores["flank_pos"].apply(lambda fp: region_of(int(fp), bounds))

    # 按区域统计 |Δ|
    regs = ["promoter", "utr5", "utr3", "terminator"]
    vals = {r: scores.loc[scores["region"] == r, "abs_delta"].values for r in regs}
    logger.info("各区域变异数: %s", {r: len(v) for r, v in vals.items()})

    # 关键检验：UTR (utr5+utr3) vs 终止子 |Δ|
    utr = np.concatenate([vals["utr5"], vals["utr3"]])
    ter = vals["terminator"]
    lines = []
    if len(utr) > 20 and len(ter) > 20:
        u, p = stats.mannwhitneyu(utr, ter, alternative="greater")
        lines += [
            f"n_UTR={len(utr)}, n_terminator={len(ter)}",
            f"UTR |Δ| 中位数={np.median(utr):.4f} vs 终止子={np.median(ter):.4f}",
            f"Mann-Whitney (UTR>终止子): U={u:.0f}, p={p:.3e}",
        ]
    else:
        lines.append("样本不足，跳过 UTR vs 终止子检验")

    # Spearman: 区域重要性顺序 vs 中位 |Δ|
    order = ["promoter", "utr5", "utr3", "terminator"]
    meds = [np.median(vals[r]) if len(vals[r]) else 0 for r in order]
    # 用模型区域重要性排序做秩相关
    import_rank = {"utr5": 1, "utr3": 2, "promoter": 3, "terminator": 4}  # 由低到高
    rank_imp = [import_rank[r] for r in order]
    rank_med = pd.Series(meds).rank().tolist()
    rho, prho = stats.spearmanr(rank_imp, rank_med)
    lines.append(f"区域重要性序 vs |Δ|序 Spearman: rho={rho:.3f}, p={prho:.3f}")

    txt = "\n".join(lines)
    logger.info("\n%s", txt)
    with open(os.path.join(vcfg["out_dir"], "region_validation.txt"), "w") as f:
        f.write(txt + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [vals[r] for r in order]
    ax.boxplot(data, tick_labels=order)
    ax.set_ylabel("|Δ predicted probability|")
    ax.set_title("Variant effect by flanking region")
    fig.tight_layout()
    fig.savefig("figures/fig5_region_validation.png", dpi=150)
    logger.info("图已保存 -> figures/fig5_region_validation.png")


if __name__ == "__main__":
    main()
