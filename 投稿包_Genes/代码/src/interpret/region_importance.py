#!/usr/bin/env python3
"""区域重要性分析（模块三）：把碱基归因按区域（启动子/5'UTR/3'UTR/终止子）聚合。

用法:
  python src/interpret/region_importance.py --config configs/config.yaml
产物:
  figures/fig4_region_importance.png
  results/interpret/region_importance.csv
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> src/

import numpy as np

from utils import setup_logging

logger = logging.getLogger("plantdl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    out = cfg["interpret"]["out_dir"]
    os.makedirs("figures", exist_ok=True)

    attr = np.load(os.path.join(out, "attributions.npz"))["arr_0"]  # (N,4,L)
    bounds = json.load(open(os.path.join(cfg["data"]["out_dir"],
                                         "region_bounds.json")))
    order = ["promoter", "utr5", "gap", "utr3", "terminator"]

    # 碱基级重要度 = 该位点 4 通道归因绝对值之和（对每个样本取均值）
    per_base = np.abs(attr).sum(axis=1).mean(axis=0)               # (L,)
    rows = {}
    for reg in order:
        s, e = bounds[reg]
        length = e - s
        if length <= 0:
            continue
        rows[reg] = per_base[s:e].sum() / length                   # 归一化到单位长度
    logger.info("区域重要性: %s", rows)

    import csv
    with open(os.path.join(out, "region_importance.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["region", "importance_per_bp"])
        for k, v in rows.items():
            w.writerow([k, v])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regs = [k for k in order if k in rows]
    vals = [rows[k] for k in regs]
    colors = ["#4c72b0", "#dd8452", "#cccccc", "#55a868", "#c44e52"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(regs, vals, color=colors)
    ax.set_ylabel("mean |attribution| per bp")
    ax.set_title("Region importance for expression prediction")
    fig.tight_layout()
    fig.savefig("figures/fig4_region_importance.png", dpi=150)
    logger.info("图已保存 -> figures/fig4_region_importance.png")


if __name__ == "__main__":
    main()
