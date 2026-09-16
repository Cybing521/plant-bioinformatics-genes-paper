#!/usr/bin/env python3
"""用拟南芥 LoRA-AgroNT 对结构化侧翼上的自然 SNV 位点打分。

优先使用带 ref/alt 的 variants.tsv。若只有 CNN 的 scores.tsv（仅 gene_id+flank_pos），
则在每个位点对三种非参考碱基做替换，报告 mean |Δ|（位点匹配，非等位匹配）。
eQTL 的 17 个位点若含 ref/alt，则按精确等位打分。
"""
import argparse
import logging
import os
import sys

SRC = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import torch

from dataloader import onehot_to_seq
from utils import get_device, set_seed, setup_logging

logger = logging.getLogger("plantdl")
BASES = "ACGT"


def load_gene_seqs(cfg):
    d = cfg["data"]
    data = np.load(f"{d['out_dir']}/dataset.npz")
    meta = pd.read_csv(f"{d['out_dir']}/meta.tsv", sep="\t")
    return {g: onehot_to_seq(x) for g, x in zip(meta["gene_id"], data["X"])}


def mutate_str(seq: str, pos: int, alt: str):
    if not (0 <= pos < len(seq)):
        return None
    alt = alt.upper()
    if alt not in BASES:
        return None
    return seq[:pos] + alt + seq[pos + 1 :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="results/models/agront_pgb_binary.pt")
    ap.add_argument("--scores", default="results/variants/scores.tsv",
                    help="CNN 已打分位点表（至少 gene_id, flank_pos）")
    ap.add_argument("--variants", default=None,
                    help="可选 chr/pos/ref/alt/gene_id/flank_pos；有则按精确等位打分")
    ap.add_argument("--eqtl", default="results/variants/eqtl_scores.tsv")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="results/variants/agront_scores.tsv")
    ap.add_argument("--eqtl-out", default="results/variants/agront_eqtl_scores.tsv")
    args = ap.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    setup_logging()
    set_seed(cfg["seed"])
    device = get_device(cfg["train"]["device"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from models import load_finetuned_agront, predict_seqs
    model = load_finetuned_agront(
        cfg, args.ckpt, device, task="binary",
        max_seq_len=cfg["agront_train"]["max_seq_len"],
    )
    gene_seq = load_gene_seqs(cfg)
    logger.info("基因序列 %d 条", len(gene_seq))

    jobs = []  # dicts
    vpath = args.variants or cfg["variants"].get("variants_tsv")
    use_exact = bool(vpath and os.path.exists(vpath))
    if use_exact:
        vdf = pd.read_csv(vpath, sep="\t")
        need = {"gene_id", "flank_pos", "alt"}
        if not need.issubset(vdf.columns):
            logger.warning("%s 缺列 %s，改用 all-alts", vpath, need - set(vdf.columns))
            use_exact = False
    if use_exact:
        logger.info("精确等位打分 <- %s  n=%d", vpath, len(vdf))
        for r in vdf.itertuples(index=False):
            gid, fp, alt = r.gene_id, int(r.flank_pos), str(r.alt)
            seq = gene_seq.get(gid)
            mut = mutate_str(seq, fp, alt) if seq is not None else None
            if mut is None:
                continue
            jobs.append({"gene_id": gid, "flank_pos": fp, "alt": alt.upper(),
                         "mode": "exact", "mut": mut})
    else:
        if not os.path.exists(args.scores):
            raise FileNotFoundError(args.scores)
        sdf = pd.read_csv(args.scores, sep="\t")
        sites = sdf[["gene_id", "flank_pos"]].drop_duplicates()
        logger.info("all-alts 打分  unique sites=%d from %s", len(sites), args.scores)
        for r in sites.itertuples(index=False):
            gid, fp = r.gene_id, int(r.flank_pos)
            seq = gene_seq.get(gid)
            if seq is None or not (0 <= fp < len(seq)):
                continue
            ref = seq[fp].upper()
            for alt in BASES:
                if alt == ref:
                    continue
                mut = mutate_str(seq, fp, alt)
                if mut is None:
                    continue
                jobs.append({"gene_id": gid, "flank_pos": fp, "alt": alt,
                             "mode": "all_alts", "mut": mut})

    if not jobs:
        raise SystemExit("没有可打分的变异")

    jdf = pd.DataFrame(jobs)
    # 每个基因一条参考前向
    genes = [g for g in jdf["gene_id"].unique() if g in gene_seq]
    logger.info("参考序列 %d，突变序列 %d，batch=%d", len(genes), len(jdf), args.batch)
    ref_p = predict_seqs(model, [gene_seq[g] for g in genes], batch_size=args.batch)
    ref_map = dict(zip(genes, ref_p.tolist()))
    mut_p = predict_seqs(model, jdf["mut"].tolist(), batch_size=args.batch)
    jdf = jdf.drop(columns=["mut"])
    jdf["prob_ref"] = jdf["gene_id"].map(ref_map)
    jdf["prob_alt"] = mut_p
    jdf["delta"] = jdf["prob_alt"] - jdf["prob_ref"]
    jdf["abs_delta"] = jdf["delta"].abs()
    jdf.to_csv(args.out, sep="\t", index=False)
    logger.info("已保存 %d 条 -> %s", len(jdf), args.out)

    # 位点级 mean |Δ|（all_alts 时一行一个位点；exact 时一行一个等位）
    site = (jdf.groupby(["gene_id", "flank_pos"], as_index=False)
            .agg(n_alts=("alt", "nunique"),
                 mean_abs_delta=("abs_delta", "mean"),
                 max_abs_delta=("abs_delta", "max"),
                 mean_delta=("delta", "mean")))
    site_path = args.out.replace(".tsv", "_by_site.tsv")
    site.to_csv(site_path, sep="\t", index=False)
    logger.info("位点聚合 %d -> %s", len(site), site_path)

    if os.path.exists(args.eqtl):
        eq = pd.read_csv(args.eqtl, sep="\t")
        meta_rows, muts = [], []
        for r in eq.itertuples(index=False):
            gid, fp, alt = r.gene_id, int(r.flank_pos), str(r.alt).upper()
            seq = gene_seq.get(gid)
            mut = mutate_str(seq, fp, alt) if seq is not None else None
            if mut is None:
                continue
            muts.append(mut)
            meta_rows.append({c: getattr(r, c) for c in eq.columns})
        if muts:
            missing = [row["gene_id"] for row in meta_rows if row["gene_id"] not in ref_map]
            missing = list(dict.fromkeys(missing))
            if missing:
                extra = predict_seqs(model, [gene_seq[g] for g in missing], batch_size=args.batch)
                for g, p in zip(missing, extra):
                    ref_map[g] = float(p)
            p_alt = predict_seqs(model, muts, batch_size=args.batch)
            out_eq = pd.DataFrame(meta_rows)
            out_eq["agront_prob_ref"] = out_eq["gene_id"].map(ref_map)
            out_eq["agront_prob_alt"] = p_alt
            out_eq["agront_delta"] = out_eq["agront_prob_alt"] - out_eq["agront_prob_ref"]
            out_eq["agront_abs_delta"] = out_eq["agront_delta"].abs()
            out_eq.to_csv(args.eqtl_out, sep="\t", index=False)
            logger.info("eQTL 精确等位 %d -> %s", len(out_eq), args.eqtl_out)


if __name__ == "__main__":
    main()
