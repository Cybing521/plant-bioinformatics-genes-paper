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
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from utils import get_device, set_seed, setup_logging

logger = logging.getLogger("plantdl")


def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if not np.any(m):
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        w = float(m.mean())
        ece += w * abs(acc - conf)
        rows.append({"bin": i, "n": int(m.sum()), "confidence": conf, "accuracy": acc})
    return float(ece), pd.DataFrame(rows)


def delong_roc_test(y_true, p1, p2):
    """Two-sided DeLong test for paired AUROCs. Returns (auc1, auc2, z, p)."""
    from scipy.stats import norm
    y = np.asarray(y_true).astype(int)
    x1 = np.asarray(p1, dtype=float)
    x2 = np.asarray(p2, dtype=float)
    pos = x1[y == 1]
    neg = x1[y == 0]
    pos2 = x2[y == 1]
    neg2 = x2[y == 0]

    def midrank(x):
        order = np.argsort(x)
        ranks = np.empty(len(x), dtype=float)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return ranks

    def auc_and_v(pos, neg):
        m, n = len(pos), len(neg)
        allx = np.concatenate([neg, pos])
        r = midrank(allx)
        auc = (r[n:].sum() - m * (m + 1) / 2) / (m * n)
        v10 = (r[n:] - midrank(pos)) / n
        v01 = 1 - (r[:n] - midrank(neg)) / m
        return auc, v10, v01

    a1, v10_1, v01_1 = auc_and_v(pos, neg)
    a2, v10_2, v01_2 = auc_and_v(pos2, neg2)
    m, n = len(pos), len(neg)
    sx = np.cov(np.vstack([v10_1, v10_2]))
    sy = np.cov(np.vstack([v01_1, v01_2]))
    var = sx / m + sy / n
    se = np.sqrt(max(var[0, 0] + var[1, 1] - 2 * var[0, 1], 0.0))
    z = 0.0 if se == 0 else (a1 - a2) / se
    p = 2 * norm.sf(abs(z))
    return float(a1), float(a2), float(z), float(p)


def paired_bootstrap(y, p_cnn, p_agr, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = roc_auc_score(y[idx], p_agr[idx]) - roc_auc_score(y[idx], p_cnn[idx])
    return float(np.mean(diffs)), tuple(np.percentile(diffs, [2.5, 97.5]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preds", default="results/pgb_binary_preds.npz",
                    help="evaluate.py --save-preds 产物；缺则回退到现场推理")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(args.seed)
    os.makedirs("results", exist_ok=True)

    if os.path.exists(args.preds):
        blob = np.load(args.preds)
        y = blob["y_true"]
        p_cnn = blob["p_cnn"]
        p_agr = blob["p_agront"]
        logger.info("从 %s 读取预测", args.preds)
    else:
        device = get_device(cfg["train"]["device"])
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

    from sklearn.metrics import roc_auc_score, average_precision_score
    from utils import compute_metrics

    a_cnn, a_agr = roc_auc_score(y, p_cnn), roc_auc_score(y, p_agr)
    _, _, z, p_delong = delong_roc_test(y, p_agr, p_cnn)
    diff_mean, diff_ci = paired_bootstrap(y, p_cnn, p_agr, n_boot=args.n_boot, seed=args.seed)
    ece_cnn, rel_cnn = expected_calibration_error(y, p_cnn)
    ece_agr, rel_agr = expected_calibration_error(y, p_agr)
    m_cnn = compute_metrics(y, p_cnn)
    m_agr = compute_metrics(y, p_agr)

    rng = np.random.default_rng(args.seed)
    n = len(y)
    cnn_cis, agr_cis = [], []
    for _ in range(min(args.n_boot, 10000)):
        idx = rng.choice(n, n, replace=True)
        cnn_cis.append(roc_auc_score(y[idx], p_cnn[idx]))
        agr_cis.append(roc_auc_score(y[idx], p_agr[idx]))
    cnn_ci = np.percentile(cnn_cis, [2.5, 97.5])
    agr_ci = np.percentile(agr_cis, [2.5, 97.5])

    c_correct = ((p_cnn >= .5).astype(int) == y)
    a_correct = ((p_agr >= .5).astype(int) == y)
    b = np.sum(c_correct & ~a_correct)
    c = np.sum(~c_correct & a_correct)
    from scipy.stats import binomtest
    mcnemar = binomtest(int(b), int(b + c)) if (b + c) > 0 else None

    lines = [
        f"n_test = {n}",
        f"prevalence = {m_cnn['prevalence']:.4f} (AUPRC chance = prevalence)",
        f"CNN    AUROC = {a_cnn:.4f} (95% CI {cnn_ci[0]:.4f}-{cnn_ci[1]:.4f})  acc={m_cnn['accuracy']:.4f}  MCC={m_cnn['mcc']:.4f}  Brier={m_cnn['brier']:.4f}  ECE={ece_cnn:.4f}",
        f"AgroNT AUROC = {a_agr:.4f} (95% CI {agr_ci[0]:.4f}-{agr_ci[1]:.4f})  acc={m_agr['accuracy']:.4f}  MCC={m_agr['mcc']:.4f}  Brier={m_agr['brier']:.4f}  ECE={ece_agr:.4f}",
        f"AUROC 差值 (AgroNT-CNN) bootstrap mean = {diff_mean:.4f} (95% CI {diff_ci[0]:.4f}-{diff_ci[1]:.4f})",
        f"DeLong z (AgroNT vs CNN) = {z:.3f}, p = {p_delong:.3e}",
        f"AgroNT AUPRC = {average_precision_score(y, p_agr):.4f}, CNN AUPRC = {average_precision_score(y, p_cnn):.4f}",
        f"McNemar: CNN对/AgroNT错={b}, CNN错/AgroNT对={c}",
    ]
    if mcnemar is not None:
        lines.append(f"McNemar 双侧 p = {mcnemar.pvalue:.3e}")
    txt = "\n".join(lines)
    logger.info("\n%s", txt)
    with open("results/benchmark_statistics.txt", "w") as f:
        f.write(txt + "\n")
    rel_cnn.to_csv("results/reliability_cnn.csv", index=False)
    rel_agr.to_csv("results/reliability_agront.csv", index=False)
    logger.info("已保存 -> results/benchmark_statistics.txt")


if __name__ == "__main__":
    main()
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
