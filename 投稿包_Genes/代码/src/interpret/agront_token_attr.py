#!/usr/bin/env python3
"""AgroNT token 级归因（captum IntegratedGradients on token embeddings）。

对结构化侧翼序列做 token 级归因，映射回碱基后聚合到区域（与 CNN DeepLift 可比）。
产物:
  results/interpret/agront_token_attr.npz
  results/interpret/agront_token_region.csv
  figures/fig4_agront_token_importance.png
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
from models import AgroNTClassifier

logger = logging.getLogger("plantdl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n-test", type=int, default=5, help="测试序列数（IG 较慢）")
    ap.add_argument("--n-steps", type=int, default=12, help="IG 步数")
    ap.add_argument("--internal-batch", type=int, default=2)
    ap.add_argument("--max-bp", type=int, default=0,
                    help="截断序列到该 bp 长度（默认全序列；小值大幅降显存）")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    out = cfg["interpret"]["out_dir"]
    os.makedirs(out, exist_ok=True)

    # --- 模型 ---
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
    # 保持梯度检查点开启（captum 归因时大幅降低激活显存）

    # --- 数据（结构化侧翼序列，3020bp） ---
    from dataloader import load_text_splits
    _, _, (te_seq, _) = load_text_splits(cfg, "npz", "binary")
    seqs = te_seq[: args.n_test]
    if args.max_bp > 0:
        seqs = [s[: args.max_bp] for s in seqs]
        logger.info("序列截断到 %d bp", args.max_bp)

    toks = model.tokenizer(seqs, return_tensors="pt", padding=True,
                           truncation=True, max_length=1020)
    input_ids = toks["input_ids"].to(device)          # (n, L)
    attn = (input_ids != model.tokenizer.pad_token_id).long()

    class Wrapper(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.m = mod
        def forward(self, ids, mask):
            out = self.m.backbone(input_ids=ids, attention_mask=mask)
            hs = out.last_hidden_state
            maskf = mask.unsqueeze(-1).float()
            pooled = (hs * maskf).sum(1) / maskf.sum(1).clamp(min=1)
            return self.m.head(pooled)

    wrapper = Wrapper(model).to(device)
    # LayerIntegratedGradients: 以 embedding 层为 target，正确处理整数 token 输入
    from captum.attr import LayerIntegratedGradients
    emb_layer = model.backbone.base_model.embeddings
    lig = LayerIntegratedGradients(wrapper, emb_layer)
    pad_id = model.tokenizer.pad_token_id or 0
    baseline = torch.full_like(input_ids, pad_id)
    logger.info("计算 AgroNT token 归因（%d 序列, %d 步）...", len(seqs), args.n_steps)
    attr = lig.attribute(input_ids, baselines=baseline, target=0,
                         additional_forward_args=(attn,),
                         internal_batch_size=args.internal_batch,
                         n_steps=args.n_steps)
    attr = attr.detach().cpu().numpy()                # (n, L)
    np.savez(os.path.join(out, "agront_token_attr.npz"), attr=attr, seq_len=input_ids.shape[1])
    logger.info("归因 shape: %s", attr.shape)

    # --- token -> 碱基映射（6-mer，token[0]=CLS 跳过） ---
    L_bp = len(seqs[0])
    base_imp = np.zeros(L_bp)
    n_tok = min(attr.shape[1] - 1, L_bp // 6)         # 去掉 CLS
    for i in range(n_tok):
        s, e = 6 * i, min(6 * (i + 1), L_bp)
        base_imp[s:e] += np.abs(attr[:, i + 1]).mean()
    base_imp /= max(6, (base_imp > 0).sum() / 6)

    # --- 区域聚合 ---
    bounds = json.load(open(os.path.join(cfg["data"]["out_dir"], "region_bounds.json")))
    order = ["promoter", "utr5", "gap", "utr3", "terminator"]
    rows = {}
    for reg in order:
        s, e = bounds[reg]
        if e - s <= 0:
            continue
        rows[reg] = float(base_imp[s:e].sum() / (e - s))
    logger.info("AgroNT token 区域重要性: %s", rows)

    import csv
    with open(os.path.join(out, "agront_token_region.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["region", "importance_per_bp"])
        for k, v in rows.items():
            w.writerow([k, v])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    regs = [k for k in order if k in rows]
    vals = [rows[k] for k in regs]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(regs, vals, color=["#4c72b0", "#dd8452", "#cccccc", "#55a868", "#c44e52"])
    ax.set_ylabel("mean |IG| per bp")
    ax.set_title("AgroNT token-level attribution by region")
    fig.tight_layout()
    fig.savefig("figures/fig4_agront_token_importance.png", dpi=150)
    logger.info("图已保存 -> figures/fig4_agront_token_importance.png")


if __name__ == "__main__":
    main()
