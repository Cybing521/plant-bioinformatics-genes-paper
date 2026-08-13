"""数据集封装：自建 npz 数据集 + PGB 标准基准（FASTA 直接解析）。"""
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("plantdl")

BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def onehot_seq(seq: str) -> np.ndarray:
    """单条序列 -> (4, L) one-hot；N/其他碱基全零。"""
    seq = seq.upper()
    x = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        j = BASE_IDX.get(b)
        if j is not None:
            x[j, i] = 1.0
    return x


class NpzDataset(Dataset):
    """自建数据集（build_dataset.py 产物）。按 meta.tsv 的 split 列过滤。"""

    def __init__(self, npz_path, meta_path, split="train", task="binary"):
        data = np.load(npz_path)
        self.X = data["X"]                      # (N, 4, L)
        self.y = data["y"]                      # (N,)
        self.y_reg = data["y_reg"] if "y_reg" in data else None
        meta = pd.read_csv(meta_path, sep="\t")
        idx = meta.index[meta["split"] == split].values
        self.X = self.X[idx]
        self.y = self.y[idx]
        if self.y_reg is not None:
            self.y_reg = self.y_reg[idx]
        self.task = task
        logger.info("[npz] split=%s samples=%d seq_len=%d", split, len(self.X), self.X.shape[2])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])
        if self.task == "binary":
            return x, torch.tensor(self.y[i], dtype=torch.float32)
        return x, torch.tensor(self.y_reg[i], dtype=torch.float32)


class PGBFastaDataset(Dataset):
    """PGB 拟南芥基因表达任务（FASTA 直接解析）。

    FASTA 头: >gene_id|样本1值|样本2值|...   （多变量表达值，跨样本）
    - per-gene 表达 = 样本均值
    - binary 标签 = 表达 >= median（median 通常用 train 计算后传入）
    """

    def __init__(self, base_dir, split, task="binary", max_seq_len=None, median=None,
                 species="arabidopsis_thaliana"):
        path = os.path.join(base_dir, f"{species}_{split}.fa")
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少 PGB FASTA: {path}\n请先运行 python data/download_pgb.py")
        seqs, exprs = [], []
        cur = None
        with open(path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if cur is not None:
                        seqs.append(cur)
                    fields = line[1:].split("|")
                    vals = [float(v) for v in fields[1:]]
                    exprs.append(float(np.mean(vals)) if vals else 0.0)
                    cur = ""
                elif cur is not None:
                    cur += line.strip()
            if cur is not None:
                seqs.append(cur)
        self.seqs = seqs
        self.expr = np.asarray(exprs, dtype=np.float32)
        self.median = median if median is not None else float(np.median(self.expr))
        self.max_seq_len = max_seq_len
        self.task = task
        logger.info("[pgb-fasta] %s samples=%d median=%.3f", split, len(self.seqs), self.median)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        seq = self.seqs[i]
        if self.max_seq_len:
            seq = seq[: self.max_seq_len]
        x = torch.from_numpy(onehot_seq(seq))
        if self.task == "binary":
            y = torch.tensor(float(self.expr[i] >= self.median), dtype=torch.float32)
        else:
            y = torch.tensor(self.expr[i], dtype=torch.float32)
        return x, y


def make_pgb_datasets(base_dir, task="binary", max_seq_len=None, species="arabidopsis_thaliana"):
    """返回 (train, val, test) PGBFastaDataset；binary 的 median 以 train 为准。"""
    tr = PGBFastaDataset(base_dir, "train", task, max_seq_len, species=species)
    median = tr.median
    va = PGBFastaDataset(base_dir, "validation", task, max_seq_len, median=median, species=species)
    te = PGBFastaDataset(base_dir, "test", task, max_seq_len, median=median, species=species)
    return tr, va, te


def onehot_to_seq(x: np.ndarray) -> str:
    """(4,L) one-hot -> ACGTN 字符串（全零列 -> N）。"""
    base = "ACGT"
    out = []
    for i in range(x.shape[1]):
        idx = int(x[:, i].argmax())
        out.append(base[idx] if x[:, i].sum() > 0 else "N")
    return "".join(out)


class TextSeqDataset(Dataset):
    """AgroNT 用：直接返回序列字符串 + 标签。"""

    def __init__(self, seqs, labels, task="binary"):
        self.seqs = list(seqs)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.task = task

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], torch.tensor(self.labels[i], dtype=torch.float32)


def load_text_splits(cfg, mode="pgb", task="binary", hf_ds=None, max_seq_len=None):
    """返回 (train, val, test) 三个 (seqs, labels) 元组，供 AgroNT 文本输入。"""
    if mode == "pgb":
        base = cfg["data"].get("pgb_dir", "data/pgb")
        tr, va, te = make_pgb_datasets(base, task, max_seq_len)
        out = []
        for ds in (tr, va, te):
            if task == "binary":
                labels = (ds.expr >= ds.median).astype(np.float32)
            else:
                labels = ds.expr
            out.append((ds.seqs, labels))
        return out
    else:
        d = cfg["data"]
        data = np.load(f"{d['out_dir']}/dataset.npz")
        meta = pd.read_csv(f"{d['out_dir']}/meta.tsv", sep="\t")
        out = []
        for split in ("train", "val", "test"):
            idx = meta.index[meta["split"] == split].values
            seqs = [onehot_to_seq(x) for x in data["X"][idx]]
            if task == "binary":
                labels = data["y"][idx]
            else:
                labels = data["y_reg"][idx]
            out.append((seqs, labels))
        return out


def make_loaders(cfg, task="binary", mode="pgb", hf_ds=None, batch_size=64,
                 num_workers=4):
    """构造 train/val/test DataLoader。mode: pgb | npz"""
    from torch.utils.data import DataLoader
    if mode == "pgb":
        base = cfg["data"].get("pgb_dir", "data/pgb")
        msl = cfg["data"].get("max_seq_len")
        tr, va, te = make_pgb_datasets(base, task, max_seq_len=msl)
    else:
        d = cfg["data"]
        tr = NpzDataset(f"{d['out_dir']}/dataset.npz", f"{d['out_dir']}/meta.tsv", "train", task)
        va = NpzDataset(f"{d['out_dir']}/dataset.npz", f"{d['out_dir']}/meta.tsv", "val", task)
        te = NpzDataset(f"{d['out_dir']}/dataset.npz", f"{d['out_dir']}/meta.tsv", "test", task)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return (DataLoader(tr, shuffle=True, **kw),
            DataLoader(va, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw))
