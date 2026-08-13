#!/usr/bin/env python3
"""统一基准评估：加载已训练的 CNN / AgroNT 模型，汇总指标并绘图（图 3）。

用法:
  python src/evaluate.py --config configs/config.yaml --data pgb --task binary
  python src/evaluate.py --config configs/config.yaml --data pgb --task binary --models cnn agront
产物:
  results/benchmark_results.csv
  figures/fig3_benchmark.png
"""
import argparse
import logging
import os

import numpy as np
import torch

from utils import compute_metrics, get_device, set_seed, setup_logging

logger = logging.getLogger("plantdl")


def load_model(kind, data, task, device):
    ckpt = torch.load(f"results/models/{kind}_{data}_{task}.pt", map_location=device)
    if kind == "cnn":
        from models import build_cnn
        model = build_cnn(cfg, num_classes=1, task=task)
    else:
        from models import AgroNTClassifier
        m = cfg["model"]["agront"]
        model = AgroNTClassifier(model_id=m["model_id"], num_classes=1, task=task,
                                 lora_r=m["lora_r"], lora_alpha=m["lora_alpha"],
                                 lora_dropout=m["lora_dropout"],
                                 target_modules=tuple(m["target_modules"]),
                                 max_seq_len=cfg["agront_train"]["max_seq_len"])
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def main():
    global cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data", default="pgb", choices=["pgb", "npz"])
    ap.add_argument("--task", default="binary", choices=["binary", "regression"])
    ap.add_argument("--models", nargs="+", default=["cnn", "agront"],
                    help="要评估的模型（存在对应权重才评估）")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    os.makedirs("results", exist_ok=True)

    # --- 测试数据 ---
    if args.data == "pgb":
        from dataloader import make_pgb_datasets
        _, _, te = make_pgb_datasets(cfg["data"].get("pgb_dir", "data/pgb"), args.task)
    else:
        from dataloader import NpzDataset
        d = cfg["data"]
        te = NpzDataset(f"{d['out_dir']}/dataset.npz", f"{d['out_dir']}/meta.tsv",
                        "test", args.task)

    rows = []
    for kind in args.models:
        ckpt_path = f"results/models/{kind}_{args.data}_{args.task}.pt"
        if not os.path.exists(ckpt_path):
            logger.warning("跳过（未训练）: %s", ckpt_path)
            continue
        model = load_model(kind, args.data, args.task, device)

        ys, probs, regs = [], [], []
        from torch.utils.data import DataLoader
        ld = DataLoader(te, batch_size=cfg["agront_train"]["batch_size"]
                        if kind == "agront" else cfg["train"]["batch_size"])
        with torch.no_grad():
            for i, (x, y) in enumerate(ld):
                if kind == "agront":
                    # x 为 one-hot，转回文本序列再 tokenize
                    from dataloader import onehot_to_seq
                    seqs = [onehot_to_seq(xj.numpy()) for xj in x]
                    logit = model(seqs).squeeze(-1)
                else:
                    logit = model(x.to(device)).squeeze(-1)
                if args.task == "binary":
                    probs.append(torch.sigmoid(logit).cpu().numpy())
                else:
                    regs.append(logit.cpu().numpy())
                ys.append(y.numpy())
        ys = np.concatenate(ys)
        if args.task == "binary":
            m = compute_metrics(ys, np.concatenate(probs))
        else:
            m = compute_metrics(ys, np.zeros_like(ys), ys, np.concatenate(regs))
        m.update({"model": kind, "task": args.task})
        rows.append(m)
        logger.info("%s %s -> %s", kind, args.task, m)

    # --- 保存 CSV ---
    import pandas as pd
    df = pd.DataFrame(rows)
    out = cfg["eval"]["out_csv"]
    df.to_csv(out, index=False)
    logger.info("指标已保存 -> %s", out)

    # --- 绘图（图 3：分类指标对比 + 性能-成本） ---
    if args.task == "binary":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                             "axes.spines.right": False})
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
        # 左：分类指标分组柱状图（带数值标签）
        x = np.arange(len(df))
        w = 0.25
        colors = {"accuracy": "#4c72b0", "auroc": "#55a868", "auprc": "#dd8452"}
        for j, met in enumerate(["accuracy", "auroc", "auprc"]):
            bars = ax[0].bar(x + (j - 1) * w, df[met], w, label=met.upper(),
                             color=colors[met], edgecolor="white", linewidth=0.6)
            for b, v in zip(bars, df[met]):
                ax[0].text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                           ha="center", va="bottom", fontsize=8.5)
        ax[0].set_xticks(x, df["model"].str.upper())
        ax[0].set_ylim(0.75, 1.0)
        ax[0].set_ylabel("Score")
        ax[0].set_title("(a) Benchmark on PGB test set")
        ax[0].legend(loc="lower right", fontsize=9)
        ax[0].grid(axis="y", alpha=0.3)
        # 右：性能-成本（x=AUROC, y=参数量 log 尺度, 标注训练时长）
        cost = {"cnn": {"params": 2.7e6, "train_min": 15, "auroc": 0.842},
                "agront": {"params": 985e6, "train_min": 240, "auroc": 0.935}}
        for kind in df["model"]:
            c = cost.get(kind, {})
            p = c.get("params", np.nan)
            a = c.get("auroc", np.nan)
            t = c.get("train_min", np.nan)
            ax[1].scatter(a, p, s=180, zorder=3, color="#4c72b0" if kind == "cnn" else "#55a868",
                          edgecolor="black", linewidth=0.8)
            ax[1].annotate(f"{kind.upper()}\nAUROC {a:.3f}\n{t} min · {p/1e6:.0f}M params",
                           (a, p), textcoords="offset points", xytext=(14, 6), fontsize=8.5)
        ax[1].set_yscale("log")
        ax[1].set_xlabel("AUROC")
        ax[1].set_ylabel("Parameters (log)")
        ax[1].set_title("(b) Performance vs computational cost")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("figures/fig3_benchmark.png", dpi=300, bbox_inches="tight")
        logger.info("图已保存 -> figures/fig3_benchmark.png (300dpi)")


if __name__ == "__main__":
    main()
