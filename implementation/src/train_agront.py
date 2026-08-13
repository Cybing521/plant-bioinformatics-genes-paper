#!/usr/bin/env python3
"""AgroNT（1B DNA 大模型）+ LoRA 微调脚本（模块二）。

用法:
  python src/train_agront.py --config configs/config.yaml --data pgb --task binary
  python src/train_agront.py --config configs/config.yaml --data npz --task regression
提示:
  - 首次运行会下载 AgroNT 权重（约 4GB，需网络与磁盘）。
  - 若 LoRA target_modules 报错，运行 python -c "..." 打印模块名核对。
产物:
  results/models/agront_{data}_{task}.pt
"""
import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import compute_metrics, get_device, save_checkpoint, set_seed, setup_logging
from models import AgroNTClassifier

logger = logging.getLogger("plantdl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--data", default="pgb", choices=["pgb", "npz"])
    ap.add_argument("--task", default="binary", choices=["binary", "regression"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--print_modules", action="store_true",
                    help="打印 backbone 模块名并退出（核对 target_modules）")
    ap.add_argument("--resume", action="store_true",
                    help="从已有 checkpoint 续训（用于补跑更多 epochs）")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    at = cfg["agront_train"]
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    logger.info("设备: %s", device)

    # --- 数据（文本序列） ---
    from dataloader import load_text_splits, TextSeqDataset
    (tr_seq, tr_y), (va_seq, va_y), (te_seq, te_y) = load_text_splits(
        cfg, args.data, args.task, max_seq_len=at["max_seq_len"])
    train_ld = DataLoader(TextSeqDataset(tr_seq, tr_y, args.task),
                          batch_size=at["batch_size"], shuffle=True)
    val_ld = DataLoader(TextSeqDataset(va_seq, va_y, args.task),
                        batch_size=at["batch_size"])
    test_ld = DataLoader(TextSeqDataset(te_seq, te_y, args.task),
                         batch_size=at["batch_size"])

    # --- 模型 ---
    model = AgroNTClassifier(
        model_id=cfg["model"]["agront"]["model_id"],
        num_classes=1, task=args.task,
        lora_r=cfg["model"]["agront"]["lora_r"],
        lora_alpha=cfg["model"]["agront"]["lora_alpha"],
        lora_dropout=cfg["model"]["agront"]["lora_dropout"],
        target_modules=tuple(cfg["model"]["agront"]["target_modules"]),
        max_seq_len=at["max_seq_len"],
    )
    if args.resume:
        ckpt_path = f"results/models/agront_{args.data}_{args.task}.pt"
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            logger.info("已从 %s 续训（历史 val_auroc=%.4f）",
                        ckpt_path, ckpt["metrics"].get("auroc", float("nan")))
        else:
            logger.warning("--resume 但未找到 %s，从头训练", ckpt_path)
    if args.print_modules:
        model.print_modules()
        return
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss() if args.task == "binary" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=at["lr"], weight_decay=at["weight_decay"])
    grad_accum = at["grad_accum"]
    epochs = args.epochs or at["epochs"]
    patience = at["early_stop_patience"]
    best_val, best_epoch, bad = -1e9, 0, 0
    t0 = time.time()

    # 该机器 bf16 cublasLt matmul 报 CUBLAS_STATUS_INTERNAL_ERROR，改用 fp32
    autocast_dtype = torch.float32

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        tot = 0.0
        for i, (seqs, y) in enumerate(train_ld):
            y = y.to(device)
            logit = model(list(seqs)).squeeze(-1)
            loss = loss_fn(logit, y)
            loss = loss / grad_accum
            loss.backward()
            if (i + 1) % grad_accum == 0:
                opt.step()
                opt.zero_grad()
            tot += loss.item() * len(y) * grad_accum

        model.eval()
        ys, probs, regs = [], [], []
        with torch.no_grad():
            for seqs, y in val_ld:
                logit = model(list(seqs)).squeeze(-1)
                if args.task == "binary":
                    probs.append(torch.sigmoid(logit).cpu().numpy())
                else:
                    regs.append(logit.cpu().numpy())
                ys.append(y.numpy())
        ys = np.concatenate(ys)
        if args.task == "binary":
            val_m = compute_metrics(ys, np.concatenate(probs))
        else:
            val_m = compute_metrics(ys, np.zeros_like(ys), ys, np.concatenate(regs))
        key = "auroc" if args.task == "binary" else "r2"
        logger.info("ep %02d train_loss=%.4f val_%s=%.4f (%.3fs)",
                    ep, tot / len(train_ld.dataset), key, val_m[key],
                    time.time() - t0)
        if val_m[key] > best_val:
            best_val, best_epoch, bad = val_m[key], ep, 0
            save_checkpoint({"model": model.state_dict(), "metrics": val_m},
                            f"results/models/agront_{args.data}_{args.task}.pt")
        else:
            bad += 1
            if bad >= patience:
                logger.info("早停于 ep=%d", ep)
                break

    # --- 测试集最终评估 ---
    model.load_state_dict(torch.load(f"results/models/agront_{args.data}_{args.task}.pt")["model"])
    model.eval()
    ys, probs, regs = [], [], []
    with torch.no_grad():
        for seqs, y in test_ld:
            logit = model(list(seqs)).squeeze(-1)
            if args.task == "binary":
                probs.append(torch.sigmoid(logit).cpu().numpy())
            else:
                regs.append(logit.cpu().numpy())
            ys.append(y.numpy())
    ys = np.concatenate(ys)
    if args.task == "binary":
        test_m = compute_metrics(ys, np.concatenate(probs))
    else:
        test_m = compute_metrics(ys, np.zeros_like(ys), ys, np.concatenate(regs))
    logger.info("测试集指标: %s", test_m)


if __name__ == "__main__":
    main()
