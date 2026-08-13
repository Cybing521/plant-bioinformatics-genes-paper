#!/usr/bin/env python3
"""基准对比的统计显著性（⭐4 稳健性）：
  - bootstrap 重采样：AUROC/AUPRC 的 95% CI 及差值 CI
  - McNemar 检验：配对分类准确率差异
产物: results/benchmark_statistics.txt
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from utils import get_device, set_seed, setup_logging

logger = logging.getLogger("plantdl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(args.seed)
    device = get_device(cfg["train"]["device"])
    os.makedirs("results", exist_ok=True)

    from dataloader import make_pgb_datasets
    _, _, te = make_pgb_datasets(cfg["data"].get("pgb_dir", "data/pgb"), "binary")
    y = np.array([te[i][1].item() for i in range(len(te))])

    def predict(kind):
        if kind == "cnn":
            from models import build_cnn
            model = build_cnn(cfg, num_classes=1, task="binary")
            ckpt = torch.load("results/models/cnn_pgb_binary.pt", map_location=device)
            model.load_state_dict(ckpt["model"])
            model.to(device).eval()
        else:
            from models import AgroNTClassifier
            m = cfg["model"]["agront"]
            model = AgroNTClassifier(model_id=m["model_id"], num_classes=1, task="binary",
                lora_r=m["lora_r"], lora_alpha=m["lora_alpha"], lora_dropout=m["lora_dropout"],
                target_modules=tuple(m["target_modules"]),
                max_seq_len=cfg["agront_train"]["max_seq_len"])
            ckpt = torch.load("results/models/agront_pgb_binary.pt", map_location=device)
            model.load_state_dict(ckpt["model"])
            model.to(device).eval()
        probs = np.zeros(len(te))
        from torch.utils.data import DataLoader
        ld = DataLoader(te, batch_size=cfg["agront_train"]["batch_size"] if kind == "agront" else 128)
        with torch.no_grad():
            for i, (x, _) in enumerate(ld):
                if kind == "agront":
                    from dataloader import onehot_to_seq
                    seqs = [onehot_to_seq(xj.numpy()) for xj in x]
                    p = torch.sigmoid(model(seqs)).squeeze(-1).cpu().numpy()
                else:
                    p = torch.sigmoid(model(x.to(device))).squeeze(-1).cpu().numpy()
                s = i * (cfg["agront_train"]["batch_size"] if kind == "agront" else 128)
                probs[s:s + len(p)] = p
        return probs

    logger.info("计算 CNN 预测...")
    p_cnn = predict("cnn")
    logger.info("计算 AgroNT 预测...")
    p_agr = predict("agront")

    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

    def auroc(p, yy): return roc_auc_score(yy, p)
    def auprc(p, yy): return average_precision_score(yy, p)

    rng = np.random.default_rng(args.seed)
    n = len(y)
    a_cnn, a_agr = auroc(p_cnn, y), auroc(p_agr, y)
    diffs, cnn_cis, agr_cis = [], [], []
    for _ in range(args.n_boot):
        idx = rng.choice(n, n, replace=True)
        a1 = auroc(p_cnn[idx], y[idx])
        a2 = auroc(p_agr[idx], y[idx])
        cnn_cis.append(a1); agr_cis.append(a2); diffs.append(a2 - a1)
    cnn_cis = np.percentile(cnn_cis, [2.5, 97.5])
    agr_cis = np.percentile(agr_cis, [2.5, 97.5])
    diff_ci = np.percentile(diffs, [2.5, 97.5])

    # McNemar
    c_correct = ((p_cnn >= .5).astype(int) == y)
    a_correct = ((p_agr >= .5).astype(int) == y)
    b = np.sum(c_correct & ~a_correct)   # CNN对 AgroNT错
    c = np.sum(~c_correct & a_correct)   # CNN错 AgroNT对
    from scipy.stats import binomtest
    mcnemar = binomtest(b, b + c) if (b + c) > 0 else None

    lines = [
        f"n_test = {n}",
        f"CNN    AUROC = {a_cnn:.4f} (95% CI {cnn_cis[0]:.4f}-{cnn_cis[1]:.4f})",
        f"AgroNT AUROC = {a_agr:.4f} (95% CI {agr_cis[0]:.4f}-{agr_cis[1]:.4f})",
        f"AUROC 差值 (AgroNT-CNN) = {a_agr-a_cnn:.4f} (95% CI {diff_ci[0]:.4f}-{diff_ci[1]:.4f})",
        f"AgroNT AUPRC = {auprc(p_agr, y):.4f}, CNN AUPRC = {auprc(p_cnn, y):.4f}",
        f"McNemar: CNN对/AgroNT错={b}, CNN错/AgroNT对={c}",
    ]
    if mcnemar is not None:
        lines.append(f"McNemar 双侧 p = {mcnemar.pvalue:.3e}")
    txt = "\n".join(lines)
    logger.info("\n%s", txt)
    with open("results/benchmark_statistics.txt", "w") as f:
        f.write(txt + "\n")
    logger.info("已保存 -> results/benchmark_statistics.txt")


if __name__ == "__main__":
    main()
