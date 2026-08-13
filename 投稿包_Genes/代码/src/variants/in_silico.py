#!/usr/bin/env python3
"""in silico 变异效应打分（模块四核心）。

用法:
  python src/variants/in_silico.py --config configs/config.yaml \
         --variants data/1001genomes/snps.tsv --n 1000
输入(变异 TSV): chr<TAB>pos<TAB>ref<TAB>alt<TAB>gene_id<TAB>flank_pos
  flank_pos: 变异在侧翼序列中的 0-indexed 位置；缺失时用 --gff 由 chr:pos 推算
             （仅支持启动子/终止子区域，UTR 内需显式给出）。
输出:
  results/variants/scores.tsv   每变异一行: gene_id, flank_pos, prob_ref, prob_alt, delta
  figures/fig5_variant_scores.png
"""
import argparse
import logging
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, SRC)   # -> src/
sys.path.insert(0, ROOT)  # -> implementation/（供 data 模块 import）

import numpy as np
import pandas as pd
import torch

from utils import get_device, set_seed, setup_logging

logger = logging.getLogger("plantdl")


def load_gene_seq(cfg):
    """gene_id -> one-hot(4,L) + meta。"""
    d = cfg["data"]
    data = np.load(f"{d['out_dir']}/dataset.npz")
    meta = pd.read_csv(f"{d['out_dir']}/meta.tsv", sep="\t")
    return {g: X for g, X in zip(meta["gene_id"], data["X"])}


def map_flank_pos(variants, gff_path, flanking):
    """由 chr:pos + gene TSS/TTS 推算侧翼序列内位置（启动子/终止子区）。"""
    sys.path.insert(0, os.path.join(ROOT, "data"))
    from build_dataset import parse_gff, pick_transcript, transcript_features, attr_to_dict
    gff = parse_gff(gff_path)
    P, U5, U3, T = (flanking[k] for k in ("promoter", "utr5", "utr3", "terminator"))
    for _, row in variants.iterrows():
        if not pd.isna(row.get("flank_pos")):
            continue
        gid = row["gene_id"]
        gene = gff[(gff["type"] == "gene") &
                   (gff["attributes"].apply(lambda a: attr_to_dict(a).get("ID") == gid))]
        if gene.empty:
            row["flank_pos"] = np.nan
            continue
        tid = pick_transcript(gff, gid)
        exons, cds, u5, u3, strand = transcript_features(gff, tid)
        tss = int(min(exons["start"]) - 1 if strand == "+" else max(exons["end"]))
        tts = int(max(exons["end"]) if strand == "+" else min(exons["start"]) - 1)
        chrom = str(row["chr"]).replace("Chr", "").replace("chr", "")
        pos = int(row["pos"]) - 1  # 0-indexed
        if chrom != str(gene["seqid"].iloc[0]).replace("Chr", "").replace("chr", ""):
            row["flank_pos"] = np.nan
            continue
        if strand == "+":
            if tss - P <= pos < tss:
                row["flank_pos"] = pos - (tss - P)
            elif tts < pos <= tts + T:
                row["flank_pos"] = P + U5 + 20 + U3 + (pos - tts - 1)
            else:
                row["flank_pos"] = np.nan
        else:
            if tss < pos <= tss + P:
                row["flank_pos"] = P - (pos - tss)   # 转录方向从后往前
            elif tts - T <= pos < tts:
                row["flank_pos"] = P + U5 + 20 + U3 + (tts - pos - 1)
            else:
                row["flank_pos"] = np.nan
    return variants


def mutate(x, pos, alt_base, base_idx):
    if not (0 <= pos < x.shape[1]):
        return None
    x = x.copy()
    x[:, pos] = 0.0
    j = base_idx.get(alt_base.upper())
    if j is None:
        return None
    x[j, pos] = 1.0
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--variants", default=None)
    ap.add_argument("--gff", default=None, help="提供后可由 chr:pos 推算 flank_pos")
    ap.add_argument("--n", type=int, default=None, help="只处理前 N 个变异")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    vcfg = cfg["variants"]
    os.makedirs(vcfg["out_dir"], exist_ok=True)

    vpath = args.variants or vcfg["variants_tsv"]
    if not os.path.exists(vpath):
        raise FileNotFoundError(f"变异文件不存在: {vpath}\n"
                                "请先用 data/download_1001.py 准备数据。")
    variants = pd.read_csv(vpath, sep="\t")
    if args.n:
        variants = variants.head(args.n)

    if args.gff or variants["flank_pos"].isna().any():
        gff = args.gff or cfg["data"]["gff3"]
        variants = map_flank_pos(variants, gff, cfg["data"]["flanking"])

    gene_seq = load_gene_seq(cfg)
    from models import build_cnn
    ckpt = torch.load("results/models/cnn_npz_binary.pt", map_location=device)
    model = build_cnn(cfg, num_classes=1, task="binary")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    rows = []
    sig = torch.sigmoid
    with torch.no_grad():
        for i, r in variants.iterrows():
            x = gene_seq.get(r["gene_id"])
            fp = r.get("flank_pos")
            if x is None or pd.isna(fp) or not (0 <= fp < x.shape[1]):
                continue
            x_alt = mutate(x, fp, r["alt"], base_idx)
            if x_alt is None:
                continue
            xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            xa = torch.tensor(x_alt, dtype=torch.float32).unsqueeze(0).to(device)
            p_ref = sig(model(xt)).item()
            p_alt = sig(model(xa)).item()
            rows.append({"gene_id": r["gene_id"], "flank_pos": int(fp),
                         "prob_ref": p_ref, "prob_alt": p_alt,
                         "delta": p_alt - p_ref})
            if len(rows) % 200 == 0:
                logger.info("已处理 %d 变异", len(rows))

    out_df = pd.DataFrame(rows)
    out_df["abs_delta"] = out_df["delta"].abs()
    out_csv = os.path.join(vcfg["out_dir"], "scores.tsv")
    out_df.to_csv(out_csv, sep="\t", index=False)
    logger.info("已保存 %d 条评分 -> %s", len(out_df), out_csv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(out_df["delta"], bins=60, color="#4c72b0")
    ax.axvline(0, color="grey", ls="--")
    ax.set_xlabel("Δ predicted expression probability (alt - ref)")
    ax.set_ylabel("variants")
    ax.set_title("In silico cis-regulatory variant effect")
    fig.tight_layout()
    fig.savefig("figures/fig5_variant_scores.png", dpi=150)
    logger.info("图已保存 -> figures/fig5_variant_scores.png")


if __name__ == "__main__":
    main()
