#!/usr/bin/env python3
"""1D-CNN 训练脚本（模块二核心）。

用法:
  python src/train.py --config configs/config.yaml --model cnn \
         --data pgb --task binary
  python src/train.py --config configs/config.yaml --model cnn \
         --data npz --task regression
产物:
  results/models/cnn_{data}_{task}.pt 最佳模型（含指标）
"""
import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import compute_metrics, get_device, save_checkpoint, set_seed, setup_logging
from models import build_cnn

logger = logging.getLogger("plantdl")


def evaluate(model, loader, device, task):
    model.eval()
    ys, probs, regs, preds = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logit = model(x)
            if task == "binary":
                p = torch.sigmoid(logit).squeeze(-1)
                probs.append(p.cpu().numpy())
            else:
                regs.append(logit.squeeze(-1).cpu().numpy())
            ys.append(y.numpy())
    ys = np.concatenate(ys)
    if task == "binary":
        return compute_metrics(ys, np.concatenate(probs))
    return compute_metrics(ys, np.zeros_like(ys), ys, np.concatenate(regs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default="cnn", choices=["cnn"])
    ap.add_argument("--data", default="pgb", choices=["pgb", "npz"])
    ap.add_argument("--task", default="binary", choices=["binary", "regression", "multitissue"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_seq_len", type=int, default=None,
                    help="覆盖 config 的序列截断长度（用于输入长度消融）")
    ap.add_argument("--crop_mode", default=None, choices=["center", "left"],
                    help="缩短窗口方式；默认 center（基因中心居中裁剪）")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 config seed；非 42 时检查点路径带 seed 后缀")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.max_seq_len:
        cfg["data"]["max_seq_len"] = args.max_seq_len
    if args.crop_mode:
        cfg["data"]["crop_mode"] = args.crop_mode
    if args.seed is not None:
        cfg["seed"] = args.seed
    tag = f"_{args.max_seq_len}bp" if args.max_seq_len else ""
    if args.crop_mode:
        tag += f"_{args.crop_mode}"
    if int(cfg["seed"]) != 42:
        tag += f"_seed{cfg['seed']}"
    tr = cfg["train"]
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(tr["device"])
    logger.info("设备: %s", device)

    # --- 数据 ---
    from dataloader import make_loaders
    train_ld, val_ld, test_ld = make_loaders(
        cfg, args.task, args.data,
        batch_size=args.batch_size or tr["batch_size"],
        num_workers=tr["num_workers"])

    # --- 模型 ---
    num_classes = int(getattr(train_ld.dataset, "n_tissues", 1) or 1) if args.task == "multitissue" else 1
    model = build_cnn(cfg, num_classes=num_classes, task=args.task)
    model.to(device)
    logger.info("CNN 参数量: %d  num_classes=%d", sum(p.numel() for p in model.parameters()), num_classes)

    loss_fn = nn.BCEWithLogitsLoss() if args.task == "binary" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)

    epochs = args.epochs or tr["epochs"]
    patience = tr["early_stop_patience"]
    best_val, best_epoch, bad = -1e9, 0, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logit = model(x).squeeze(-1)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        val_m = evaluate(model, val_ld, device, args.task)
        sched.step(-val_m.get("auroc", -val_m.get("r2", 0)))
        key = "auroc" if args.task == "binary" else "r2"
        logger.info("ep %02d train_loss=%.4f val_%s=%.4f (%.3fs)",
                    ep, tot / len(train_ld.dataset), key, val_m[key],
                    time.time() - t0)
        if val_m[key] > best_val:
            best_val, best_epoch, bad = val_m[key], ep, 0
            save_checkpoint({"model": model.state_dict(), "metrics": val_m},
                            f"results/models/cnn_{args.data}{tag}_{args.task}.pt")
        else:
            bad += 1
            if bad >= patience:
                logger.info("早停于 ep=%d", ep)
                break

    # --- 测试集最终评估 ---
    model.load_state_dict(torch.load(f"results/models/cnn_{args.data}{tag}_{args.task}.pt")["model"])
    test_m = evaluate(model, test_ld, device, args.task)
    logger.info("测试集指标: %s", test_m)


if __name__ == "__main__":
    main()
