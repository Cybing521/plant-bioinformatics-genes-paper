#!/usr/bin/env python3
"""跨物种迁移评估（模块四）：用拟南芥训练的 CNN 直接预测水稻/玉米的表达高低。

用法:
  先下载数据: python data/download_pgb.py --species oryza_sativa
              python data/download_pgb.py --species zea_mays
  再运行:    python src/cross_species.py
产物:
  results/cross_species_results.csv
  figures/fig6_cross_species.png
说明: 拟南芥模型参数固定（不微调），直接在异源物种测试集上评估，
      与拟南芥自身测试集对比，反映调控代码的跨物种保守性。
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))   # -> src/

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_device, set_seed, setup_logging, compute_metrics
from models import build_cnn

logger = logging.getLogger("plantdl")

SPECIES = [
    ("arabidopsis_thaliana", "Arabidopsis"),
    ("oryza_sativa", "Rice"),
    ("zea_mays", "Maize"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    base = cfg["data"].get("pgb_dir", "data/pgb")

    # --- 拟南芥训练的 CNN 模型 ---
    from dataloader import PGBFastaDataset
    ckpt = torch.load("results/models/cnn_pgb_binary.pt", map_location=device)
    model = build_cnn(cfg, num_classes=1, task="binary")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    logger.info("加载拟南芥 CNN 模型")

    rows = []
    for sp, label in SPECIES:
        try:
            te = PGBFastaDataset(base, "test", "binary", species=sp)
        except FileNotFoundError:
            logger.warning("缺少 %s 数据，请先运行 download_pgb.py --species %s", sp, sp)
            continue
        ld = DataLoader(te, batch_size=args.batch_size)
        ys, probs = [], []
        with torch.no_grad():
            for x, y in ld:
                p = torch.sigmoid(model(x.to(device))).squeeze(-1)
                probs.append(p.cpu().numpy())
                ys.append(y.numpy())
        ys = np.concatenate(ys)
        m = compute_metrics(ys, np.concatenate(probs))
        m.update({"species": sp, "label": label, "n_test": len(ys)})
        rows.append(m)
        logger.info("%-15s n=%d acc=%.3f auroc=%.3f auprc=%.3f",
                    label, len(ys), m["accuracy"], m["auroc"], m["auprc"])

    if not rows:
        sys.exit("没有任何物种数据，请先下载")

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv("results/cross_species_results.csv", index=False)
    logger.info("已保存 -> results/cross_species_results.csv")

    # --- 绘图 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(df))
    w = 0.25
    for j, met in enumerate(["accuracy", "auroc", "auprc"]):
        ax.bar(x + (j - 1) * w, df[met], w, label=met)
    ax.set_xticks(x, df["label"])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Cross-species transfer (Arabidopsis-trained CNN)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("figures/fig6_cross_species.png", dpi=150)
    logger.info("图已保存 -> figures/fig6_cross_species.png")


if __name__ == "__main__":
    main()
