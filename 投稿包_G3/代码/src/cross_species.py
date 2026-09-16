#!/usr/bin/env python3
"""跨物种迁移评估：拟南芥训练的 CNN 或 AgroNT，零样本评估水稻/玉米/番茄/大豆。

标签协议与主实验对齐：各种物种用各自 train split 的表达中位数定义高/低，
再在 test 上评估（不用 test 自身中位数）。

AgroNT 骨干在 48 个可食用植物基因组上预训练，因此作物种评估是
「拟南芥 LoRA 检查点的跨物种应用」，不是与 CNN 同义的从未见过该物种。
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_device, set_seed, setup_logging, compute_metrics

logger = logging.getLogger("plantdl")

SPECIES = [
    ("arabidopsis_thaliana", "Arabidopsis"),
    ("oryza_sativa", "Rice"),
    ("zea_mays", "Maize"),
    ("solanum_lycopersicum", "Tomato"),
    ("glycine_max", "Soybean"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="cnn", choices=["cnn", "agront"])
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None,
                    help="输出 CSV；默认 results/cross_species_results.csv 或 *_agront.csv")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    base = cfg["data"].get("pgb_dir", "data/pgb")
    kind = args.model
    default_ckpt = ("results/models/cnn_pgb_binary.pt" if kind == "cnn"
                    else "results/models/agront_pgb_binary.pt")
    ckpt_path = args.ckpt or default_ckpt
    out_csv = args.out or (
        "results/cross_species_results.csv" if kind == "cnn"
        else "results/cross_species_agront.csv"
    )
    batch_size = args.batch_size or (
        cfg["train"]["batch_size"] if kind == "cnn" else max(4, int(cfg["agront_train"].get("batch_size", 4)))
    )

    from dataloader import PGBFastaDataset
    if kind == "cnn":
        from models import build_cnn
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_cnn(cfg, num_classes=1, task="binary")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()
    else:
        from models import load_finetuned_agront
        model = load_finetuned_agront(
            cfg, ckpt_path, device, task="binary",
            max_seq_len=cfg["agront_train"]["max_seq_len"],
        )
    logger.info("加载拟南芥 %s 模型: %s", kind.upper(), ckpt_path)

    rows = []
    for sp, label in SPECIES:
        try:
            tr = PGBFastaDataset(base, "train", "binary", species=sp)
            te = PGBFastaDataset(base, "test", "binary", median=tr.median, species=sp)
        except FileNotFoundError:
            logger.warning("缺少 %s 数据，请先运行 download_pgb.py --species %s", sp, sp)
            continue
        ys, probs = [], []
        with torch.no_grad():
            if kind == "cnn":
                ld = DataLoader(te, batch_size=batch_size)
                for x, y in ld:
                    p = torch.sigmoid(model(x.to(device))).squeeze(-1)
                    probs.append(p.cpu().numpy())
                    ys.append(y.numpy())
            else:
                labels = (te.expr >= te.median).astype(np.float32)
                from models import predict_seqs
                p = predict_seqs(model, te.seqs, batch_size=batch_size)
                probs.append(p)
                ys.append(labels)
        ys = np.concatenate(ys)
        m = compute_metrics(ys, np.concatenate(probs))
        m.update({
            "species": sp,
            "label": label,
            "n_test": len(ys),
            "train_median": float(tr.median),
            "label_protocol": "species_train_median",
            "model": kind,
        })
        rows.append(m)
        logger.info("%-10s %-15s n=%d median=%.3f acc=%.3f auroc=%.3f auprc=%.3f",
                    kind, label, len(ys), tr.median, m["accuracy"], m["auroc"], m["auprc"])

    if not rows:
        sys.exit("没有任何物种数据，请先下载")

    import pandas as pd
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info("已保存 -> %s", out_csv)

    if kind == "cnn":
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
        ax.set_title("Cross-species transfer (Arabidopsis CNN; train-median labels)")
        ax.legend()
        fig.tight_layout()
        fig.savefig("figures/fig6_cross_species.png", dpi=150)
        logger.info("图已保存 -> figures/fig6_cross_species.png")


if __name__ == "__main__":
    main()
