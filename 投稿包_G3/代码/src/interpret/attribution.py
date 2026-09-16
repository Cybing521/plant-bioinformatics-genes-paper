#!/usr/bin/env python3
"""DeepLift 碱基级归因（模块三第一步，用 captum 实现，替代有 bug 的 shap DeepExplainer）。

用法:
  python src/interpret/attribution.py --config configs/config.yaml \
         --task binary --n-test 500
产物:
  results/interpret/ohe.npz          one-hot (N,4,L)
  results/interpret/attributions.npz  归因 (N,4,L)
  results/interpret/labels.npz       标签
说明: 对 CNN 模型做归因；AgroNT 归因需 token 级处理，见 README 备选方案。
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # -> src/

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import get_device, set_seed, setup_logging

logger = logging.getLogger("plantdl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--task", default="binary")
    ap.add_argument("--n-test", type=int, default=200,
                    help="测试集取样数量（归因较慢）")
    ap.add_argument("--background", type=int, default=100)
    ap.add_argument("--internal-batch", type=int, default=32,
                    help="DeepLift 内部批次（显存控制）")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    out = cfg["interpret"]["out_dir"]
    os.makedirs(out, exist_ok=True)

    from models import build_cnn
    ckpt = torch.load(f"results/models/cnn_npz_{args.task}.pt", map_location=device)
    model = build_cnn(cfg, num_classes=1, task=args.task)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # --- 测试数据 ---
    from dataloader import NpzDataset
    d = cfg["data"]
    ds = NpzDataset(f"{d['out_dir']}/dataset.npz", f"{d['out_dir']}/meta.tsv",
                    "test", args.task)
    n = min(args.n_test, len(ds))
    X = torch.stack([ds[i][0] for i in range(n)])          # (N,4,L)
    y = torch.tensor([ds[i][1] for i in range(n)])

    # --- DeepLift 归因（captum） ---
    from captum.attr import DeepLift
    rng = np.random.default_rng(cfg["seed"])
    bg_idx = rng.choice(len(ds), min(args.background, len(ds)), replace=False)
    bg = torch.stack([ds[i][0] for i in bg_idx])

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)      # (N,1)，captum 用 target=0 取唯一输出

    wrapper = Wrapper(model).to(device)
    explainer = DeepLift(wrapper)
    # 基线 = 背景样本均值（全零 N 参考）
    baseline = bg.mean(0, keepdim=True).to(device)
    logger.info("计算 DeepLift 归因（%d 样本）...", n)
    # captum DeepLift 无 internal_batch_size，手动分块控显存
    attr_parts = []
    for i in range(0, n, args.internal_batch):
        xb = X[i: i + args.internal_batch].to(device)
        ab = explainer.attribute(xb, baselines=baseline, target=0)
        attr_parts.append(ab.detach().cpu())
    attr = torch.cat(attr_parts).numpy()                  # (N,4,L)
    logger.info("归因 shape: %s", attr.shape)

    np.savez(os.path.join(out, "ohe.npz"), arr_0=X.numpy())
    np.savez(os.path.join(out, "attributions.npz"), arr_0=attr)
    np.savez(os.path.join(out, "labels.npz"), y=y.numpy())
    logger.info("已保存 ohe / attributions / labels -> %s", out)


if __name__ == "__main__":
    main()
