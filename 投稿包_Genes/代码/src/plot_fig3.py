#!/usr/bin/env python3
"""快速重绘图3（读 benchmark_results.csv，不重跑模型评估）。
修正常用的模型显示名: cnn->CNN, agront->AgroNT
产物: figures/fig3_benchmark.png (300dpi)
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from utils import setup_logging

logger = logging.getLogger("plantdl")

NAME = {"cnn": "CNN", "agront": "AgroNT"}
COST = {  # 参数量 + 训练时长（来自训练日志，固定值）
    "cnn": {"params": 2.7e6, "train_min": 15},
    "agront": {"params": 985e6, "train_min": 240},
}


def main():
    setup_logging()
    csv_path = "results/benchmark_results.csv"
    if not os.path.exists(csv_path):
        sys.exit(f"缺 {csv_path}，先运行 evaluate.py")
    df = pd.read_csv(csv_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)

    # 左：分类指标
    x = np.arange(len(df))
    w = 0.25
    colors = {"accuracy": "#4c72b0", "auroc": "#55a868", "auprc": "#dd8452"}
    for j, met in enumerate(["accuracy", "auroc", "auprc"]):
        bars = ax[0].bar(x + (j - 1) * w, df[met], w, label=met.upper(),
                         color=colors[met], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, df[met]):
            ax[0].text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                       ha="center", va="bottom", fontsize=8.5)
    ax[0].set_xticks(x, [NAME.get(m, m) for m in df["model"]])
    ax[0].set_ylim(0.75, 1.0)
    ax[0].set_ylabel("Score")
    ax[0].set_title("(a) Benchmark on PGB test set")
    ax[0].legend(loc="lower right", fontsize=9)
    ax[0].grid(axis="y", alpha=0.3)

    # 右：性能-成本
    for _, r in df.iterrows():
        c = COST.get(r["model"], {})
        p = c.get("params", np.nan)
        t = c.get("train_min", np.nan)
        a = r["auroc"]
        color = "#4c72b0" if r["model"] == "cnn" else "#55a868"
        ax[1].scatter(a, p, s=180, zorder=3, color=color,
                      edgecolor="black", linewidth=0.8)
        ax[1].annotate(f"{NAME.get(r['model'], r['model'])}\nAUROC {a:.3f}\n"
                       f"{t} min · {p/1e6:.0f}M params",
                       (a, p), textcoords="offset points", xytext=(14, 6), fontsize=8.5)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("AUROC")
    ax[1].set_ylabel("Parameters (log)")
    ax[1].set_title("(b) Performance vs computational cost")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/fig3_benchmark.png", dpi=300, bbox_inches="tight")
    logger.info("图3 已重绘 -> figures/fig3_benchmark.png (300dpi)")


if __name__ == "__main__":
    main()
