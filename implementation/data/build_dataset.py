#!/usr/bin/env python3
"""构建拟南芥"侧翼序列→表达"数据集（染色体级防泄漏划分）。

输入:
  data/tair10/TAIR10_Chr_all.fas                    (基因组, pyfaidx)
  data/tair10/Araport11_GFF3_genes_transposons.gff3 (注释)
  data/tair10/expression_matrix.tsv                 (表达矩阵, gene × samples)
输出:
  data/dataset/dataset.npz       X: one-hot (N,4,L), y: 二分类(0/1), y_reg: 回归
  data/dataset/meta.tsv         gene_id / chr / strand / split / 区域边界
  data/dataset/region_bounds.json

序列结构(5'→3' 转录方向): [启动子 P][5'UTR][N-gap][3'UTR][终止子 T]

性能: 预建 GFF 索引（gene→mRNA→features），O(N)，25k 基因约几分钟。
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

try:
    from pyfaidx import Fasta
except ImportError:
    sys.exit("缺少 pyfaidx，请先运行 bash setup_env.sh")

GFF_COLS = ["seqid", "source", "type", "start", "end", "score",
            "strand", "phase", "attributes"]
GAP_N = 20  # 上/下游区块之间的 N 填充
FEAT_TYPES = ("exon", "CDS", "five_prime_UTR", "three_prime_UTR")


def parse_gff(path):
    """返回 DataFrame，字段标准化。"""
    df = pd.read_csv(path, sep="\t", comment="#", header=None, names=GFF_COLS,
                     low_memory=False)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df[df["type"].isin(["gene", "mRNA", "exon", "CDS",
                               "five_prime_UTR", "three_prime_UTR"])].copy()


def attr_to_dict(attr_str):
    d = {}
    for kv in str(attr_str).split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.strip()
            v = v.strip()
            # 去 Ensembl GFF 前缀: ID=gene:AT1G01010 / Parent=transcript:AT1G01010.1
            if k in ("ID", "Parent") and ":" in v:
                v = v.split(":")[-1]
            d[k] = v
    return d


def build_indexes(gff):
    """预建索引: gene→mRNA、mRNA→CDS 外显子数、mRNA→features(列表)。"""
    gff["attrs"] = gff["attributes"].apply(attr_to_dict)
    gff["ID"] = gff["attrs"].apply(lambda d: d.get("ID"))
    gff["Parent"] = gff["attrs"].apply(lambda d: d.get("Parent"))

    gene_to_mrna = {}
    mrna_rows = gff[gff["type"] == "mRNA"]
    for r in mrna_rows.itertuples(index=False):
        gene_to_mrna.setdefault(r.Parent, []).append(r.ID)

    cds_len = {}
    cds_rows = gff[gff["type"] == "CDS"]
    for r in cds_rows.itertuples(index=False):
        cds_len[r.Parent] = cds_len.get(r.Parent, 0) + 1

    feat_groups = {}
    feat_rows = gff[gff["type"].isin(FEAT_TYPES)]
    for r in feat_rows.itertuples(index=False):
        feat_groups.setdefault(r.Parent, []).append(
            (r.type, int(r.start), int(r.end), r.strand))
    return gene_to_mrna, cds_len, feat_groups


def get_intervals(feats):
    """features -> 合并后 0-indexed 半开区间 [(s,e), ...]"""
    if not feats:
        return []
    intervals = sorted((s, e) for _, s, e, _ in feats)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s - 1, e) for s, e in merged]


def fetch(genome, chrom, intervals):
    """按 0-indexed 区间从 FASTA 提取序列（正向）。"""
    if not intervals:
        return ""
    try:
        seq = genome[chrom]
    except KeyError:
        return ""
    parts = []
    for s, e in intervals:
        s = max(s, 0)
        e = min(e, len(seq))
        if e <= s:
            continue
        parts.append(str(seq[s:e]))
    return "".join(parts)


def rc(seq):
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq))


def trim_to_len(s, length):
    return s[:length] if len(s) >= length else s + "N" * (length - len(s))


def onehot(seq):
    """(4,L) one-hot；N/其他碱基全零。"""
    base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    x = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        j = base_idx.get(b)
        if j is not None:
            x[j, i] = 1.0
    return x


def build_regions(genome, chrom, strand, feats, P, U5, U3, T):
    """返回转录方向 5'->3' 的序列 + 区域边界。feats: (type,start,end,strand) 列表。"""
    exons = [f for f in feats if f[0] == "exon"]
    if not exons:
        return None, None
    u5 = [f for f in feats if f[0] == "five_prime_UTR"]
    u3 = [f for f in feats if f[0] == "three_prime_UTR"]
    cds = [f for f in feats if f[0] == "CDS"]

    tss = int(min(s for _, s, _, _ in exons) - 1) if strand == "+" else int(max(e for _, _, e, _ in exons))
    tts = int(max(e for _, _, e, _ in exons)) if strand == "+" else int(min(s for _, s, _, _ in exons) - 1)
    cds_start = int(min(s for _, s, _, _ in cds) - 1) if cds else None
    cds_end = int(max(e for _, _, e, _ in cds)) if cds else None

    if strand == "+":
        prom = [(tss - P, tss)]
        ter = [(tts, tts + T)]
        u5i = get_intervals(u5) or ([(tss, cds_start)] if cds_start is not None else [])
        u3i = get_intervals(u3) or ([(cds_end, tts)] if cds_end is not None else [])
    else:
        prom = [(tss, tss + P)]
        ter = [(tts - T, tts)]
        u5i = get_intervals(u5) or ([(cds_end, tss)] if cds_end is not None else [])
        u3i = get_intervals(u3) or ([(tts, cds_start)] if cds_start is not None else [])

    up = fetch(genome, chrom, prom) + fetch(genome, chrom, u5i)
    down = fetch(genome, chrom, u3i) + fetch(genome, chrom, ter)
    up = trim_to_len(up, P + U5)
    down = trim_to_len(down, U3 + T)
    if strand == "-":
        up, down = rc(up), rc(down)

    full = up + "N" * GAP_N + down
    bounds = {
        "promoter": [0, P],
        "utr5": [P, P + U5],
        "gap": [P + U5, P + U5 + GAP_N],
        "utr3": [P + U5 + GAP_N, P + U5 + GAP_N + U3],
        "terminator": [P + U5 + GAP_N + U3, P + U5 + GAP_N + U3 + T],
    }
    return full, bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)["data"]
    P = cfg["flanking"]["promoter"]
    U5 = cfg["flanking"]["utr5"]
    U3 = cfg["flanking"]["utr3"]
    T = cfg["flanking"]["terminator"]
    min_samples = cfg["min_expr_samples"]
    high_q = cfg["high_expr_quantile"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # --- 表达矩阵 ---
    expr = pd.read_csv(cfg["expression_tsv"], sep="\t")
    expr = expr.set_index(expr.columns[0])
    expr.index = expr.index.str.split(".").str[0]          # AT1G01010.1 -> AT1G01010
    valid = expr.notna().sum(axis=1) >= min_samples
    expr = expr[valid]
    mean_expr = expr.mean(axis=1)
    log_expr = np.log2(mean_expr + 1.0)
    print(f">>> 表达矩阵: {expr.shape[0]} 个基因通过最小样本数过滤")

    # --- GFF 索引 + 基因组 ---
    gff = parse_gff(cfg["gff3"])
    gene_to_mrna, cds_len, feat_groups = build_indexes(gff)
    genome = Fasta(cfg["genome_fasta"])
    gene_rows = gff[gff["type"] == "gene"]

    records, metas = [], []
    for gene in gene_rows.itertuples(index=False):
        gid = gene.ID
        if gid is None or gid not in mean_expr.index:
            continue
        mrnas = gene_to_mrna.get(gid)
        if not mrnas:
            continue
        tid = max(mrnas, key=lambda m: cds_len.get(m, 0))
        feats = feat_groups.get(tid)
        if not feats:
            continue
        strand = feats[0][3]
        seq, bounds = build_regions(genome, gene.seqid, strand, feats, P, U5, U3, T)
        if seq is None:
            continue
        records.append(onehot(seq))
        metas.append({
            "gene_id": gid, "chr": gene.seqid, "strand": strand,
            "mean_expr": float(mean_expr[gid]),
            "log_expr": float(log_expr[gid]),
        })

    if not records:
        sys.exit("错误: 未生成任何基因序列，请检查 GFF 基因ID与表达矩阵是否匹配")
    X = np.stack(records)   # (N,4,L)
    print(f">>> 样本数: {X.shape[0]}, 序列长: {X.shape[2]}, 通道: {X.shape[1]}")

    meta = pd.DataFrame(metas)
    y = (meta["log_expr"] >= meta["log_expr"].quantile(high_q)).astype(int).values
    y_reg = meta["log_expr"].values

    # --- 染色体级划分（防泄漏）---
    chrs = meta["chr"].unique()
    rng = np.random.default_rng(42)
    perm = rng.permutation(chrs)
    n = len(perm)
    tr, va, te = perm[: max(1, n // 2)], perm[max(1, n // 2): max(1, n // 2) + 1], perm[max(1, n // 2) + 1:]
    meta["split"] = np.select(
        [meta["chr"].isin(tr), meta["chr"].isin(va), meta["chr"].isin(te)],
        ["train", "val", "test"],
        default="train",
    )
    print("划分:", dict(meta["split"].value_counts()), "| 染色体 train/val/test:",
          tr.tolist(), va.tolist(), te.tolist())

    # --- 保存 ---
    np.savez_compressed(os.path.join(out_dir, "dataset.npz"), X=X, y=y, y_reg=y_reg)
    meta.to_csv(os.path.join(out_dir, "meta.tsv"), sep="\t", index=False)
    with open(os.path.join(out_dir, "region_bounds.json"), "w") as f:
        json.dump(bounds, f, indent=2)
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump({"promoter": P, "utr5": U5, "utr3": U3, "terminator": T,
                   "gap": GAP_N, "total_len": int(X.shape[2])}, f, indent=2)
    print(f">>> 数据集已保存到 {out_dir}")


if __name__ == "__main__":
    main()
