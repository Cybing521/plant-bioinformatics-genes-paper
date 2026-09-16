#!/usr/bin/env python3
"""AgroNT 区域重要性（扰动法）：对结构化侧翼序列做滑窗突变，测量预测变化。

与 CNN 的 DeepLift 区域重要性可比。输入为自建 npz 数据集（3020bp 侧翼序列）。
产物:
  results/interpret/agront_region_importance.csv
  figures/fig4_agront_region_importance.png
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> src/

import numpy as np
import torch

from utils import get_device, set_seed, setup_logging

logger = logging.getLogger("plantdl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n-test", type=int, default=50, help="测试序列数")
    ap.add_argument("--window", type=int, default=50, help="突变窗口宽度(bp)；与 CNN 对齐")
    ap.add_argument("--step", type=int, default=25, help="窗口步长(bp)；与 CNN 对齐")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    out = cfg["interpret"]["out_dir"]
    os.makedirs(out, exist_ok=True)

    from models import AgroNTClassifier
    m = cfg["model"]["agront"]
    model = AgroNTClassifier(
        model_id=m["model_id"], num_classes=1, task="binary",
        lora_r=m["lora_r"], lora_alpha=m["lora_alpha"], lora_dropout=m["lora_dropout"],
        target_modules=tuple(m["target_modules"]),
        max_seq_len=cfg["agront_train"]["max_seq_len"])
    ckpt = torch.load("results/models/agront_pgb_binary.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # 自建数据集（结构化侧翼序列）
    from dataloader import load_text_splits
    (tr_seq, tr_y), _, (te_seq, te_y) = load_text_splits(cfg, "npz", "binary")
    n = min(args.n_test, len(te_seq))
    seqs = te_seq[:n]
    logger.info("对 %d 条侧翼序列做扰动归因（窗口 %d bp, 步长 %d bp）", n, args.window, args.step)

    L = len(seqs[0])
    profile = np.zeros(L)
    with torch.no_grad():
        for si, s in enumerate(seqs):
            base = model([s]).item()
            for start in range(0, L - args.window + 1, args.step):
                end = start + args.window
                mut = s[:start] + "N" * args.window + s[end:]
                p = model([mut]).item()
                profile[start:end] += abs(p - base)
            if (si + 1) % 10 == 0:
                logger.info("  已处理 %d/%d", si + 1, n)
    profile /= n

    # 按区域聚合
    bounds = json.load(open(os.path.join(cfg["data"]["out_dir"], "region_bounds.json")))
    order = ["promoter", "utr5", "gap", "utr3", "terminator"]
    rows = {}
    for reg in order:
        s, e = bounds[reg]
        if e - s <= 0:
            continue
        rows[reg] = float(profile[s:e].sum() / (e - s))
    logger.info("AgroNT 区域重要性: %s", rows)

    import csv
    with open(os.path.join(out, "agront_region_importance.csv"), "w") as f:
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
    ax.set_ylabel("mean perturbation effect per bp")
    ax.set_title("AgroNT region importance (in-silico mutation)")
    fig.tight_layout()
    fig.savefig("figures/fig4_agront_region_importance.png", dpi=150)
    logger.info("图已保存 -> figures/fig4_agront_region_importance.png")


if __name__ == "__main__":
    main()
