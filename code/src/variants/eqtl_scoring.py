#!/usr/bin/env python3
"""调控变异效应预测（模块四，图5）：用 cis-eQTL SNP 作为真实变异集打分并验证。

数据:
  data/eqtl/ciseqtl.xls   AtMAD 拟南芥 cis-eQTL（SNP/等位/基因/p-value）
  npz 数据集              基因侧翼序列 (data/dataset)
  GFF                     基因 TSS/TTS/strand（计算 flank_pos）
模型:
  cnn_npz_binary.pt       自建数据集 CNN（区域重要性来源）
流程:
  1. 解析 cis-eQTL SNP（chr/pos/ref/alt/gene/pvalue）
  2. 计算每个 SNP 在基因侧翼序列中的 flank_pos
  3. 用 CNN 对参考/变异序列打分，得 Δ 预测概率
  4. 验证：eQTL 显著性 vs |Δ| 关联（Mann-Whitney + 相关）
产物:
  results/variants/eqtl_scores.tsv
  results/variants/eqtl_validation.txt
  figures/fig5_variant_effect.png
"""
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import torch
from scipy import stats

from utils import get_device, set_seed, setup_logging

logger = logging.getLogger("plantdl")


def parse_ciseqtl(path):
    df = pd.read_excel(path, header=None)
    # 真实表头在第 10 行，数据从第 11 行开始
    df.columns = ["SNP", "Alleles", "Alteration", "Gene", "Beta", "tstat", "pvalue", "FDR"]
    df = df.iloc[11:].dropna(subset=["SNP"]).copy()
    df["chr"] = df["SNP"].str.split("_").str[0].str.replace("chr", "")
    df["pos"] = df["SNP"].str.split("_").str[1].astype(int)
    df["ref"] = df["Alleles"].str.split("_").str[0]
    df["alt"] = df["Alleles"].str.split("_").str[1]
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")
    return df


def get_gene_meta(gff_path):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(gff_path)), ".."))
    from build_dataset import parse_gff, build_indexes
    gff = parse_gff(gff_path)
    gene_to_mrna, cds_len, feat_groups = build_indexes(gff)
    meta = {}
    for gene in gff[gff["type"] == "gene"].itertuples(index=False):
        gid = gene.ID
        mrnas = gene_to_mrna.get(gid)
        if not mrnas:
            continue
        tid = max(mrnas, key=lambda m: cds_len.get(m, 0))
        feats = feat_groups.get(tid)
        if not feats:
            continue
        strand = feats[0][3]
        exons = [f for f in feats if f[0] == "exon"]
        cds = [f for f in feats if f[0] == "CDS"]
        if not exons:
            continue
        tss = int(min(s for _, s, _, _ in exons) - 1) if strand == "+" else int(max(e for _, _, e, _ in exons))
        tts = int(max(e for _, _, e, _ in exons)) if strand == "+" else int(min(s for _, s, _, _ in exons) - 1)
        if cds:
            cds_start = int(min(s for _, s, _, _ in cds) - 1) if strand == "+" else int(max(e for _, _, e, _ in cds))
            cds_end = int(max(e for _, _, e, _ in cds)) if strand == "+" else int(min(s for _, s, _, _ in cds) - 1)
        else:
            cds_start = tss
            cds_end = tts
        meta[gid] = (gene.seqid, strand, tss, tts, cds_start, cds_end)
    return meta


def flank_pos_for(gm, chrom, pos0, P=1000, U5=500, U3=500, T=1000, GAP=20):
    """返回侧翼序列[启动子][5'UTR][gap][3'UTR][终止子]中的 0-indexed 位置；不在侧翼区返回 None。"""
    chr_, strand, tss, tts, cds_start, cds_end = gm
    if str(chr_).replace("Chr", "") != str(chrom):
        return None
    if strand == "+":
        if tss - P <= pos0 < tss:                      # 启动子
            return pos0 - (tss - P)
        if tss <= pos0 < cds_start:                    # 5'UTR
            return P + (pos0 - tss)
        if cds_end < pos0 <= tts:                      # 3'UTR
            return P + U5 + GAP + (pos0 - cds_end - 1)
        if tts < pos0 <= tts + T:                      # 终止子
            return P + U5 + GAP + U3 + (pos0 - tts - 1)
    else:
        if tss < pos0 <= tss + P:                      # 启动子
            return P - (pos0 - tss)
        if cds_end < pos0 <= tss:                      # 5'UTR
            return P + (tss - pos0)
        if cds_start <= pos0 < cds_end:                # 3'UTR
            return P + U5 + GAP + (cds_end - pos0 - 1)
        if tts - T <= pos0 < tts:                      # 终止子
            return P + U5 + GAP + U3 + (tts - pos0 - 1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--eqtl", default="data/eqtl/ciseqtl.xls")
    ap.add_argument("--n", type=int, default=None, help="只处理前 N 个（测试）")
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

    # --- eQTL SNP ---
    eq = parse_ciseqtl(args.eqtl)
    if args.n:
        eq = eq.head(args.n)
    logger.info("cis-eQTL SNP 数: %d", len(eq))

    # --- 基因元数据 + npz 序列 ---
    gene_meta = get_gene_meta(cfg["data"]["gff3"])
    data = np.load(f"{cfg['data']['out_dir']}/dataset.npz")
    meta_df = pd.read_csv(f"{cfg['data']['out_dir']}/meta.tsv", sep="\t")
    gene_seq = {g: X for g, X in zip(meta_df["gene_id"], data["X"])}

    # --- CNN 模型 ---
    from models import build_cnn
    ckpt = torch.load("results/models/cnn_npz_binary.pt", map_location=device)
    model = build_cnn(cfg, num_classes=1, task="binary")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    sig = torch.sigmoid

    base_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    rows = []
    for _, r in eq.iterrows():
        gm = gene_meta.get(r["Gene"])
        x = gene_seq.get(r["Gene"])
        if gm is None or x is None:
            continue
        fp = flank_pos_for(gm, r["chr"], r["pos"] - 1)
        if fp is None or not (0 <= fp < x.shape[1]):
            continue
        j = base_idx.get(r["alt"].upper())
        if j is None:
            continue
        xa = x.copy()
        xa[:, fp] = 0.0
        xa[j, fp] = 1.0
        with torch.no_grad():
            p_ref = sig(model(torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device))).item()
            p_alt = sig(model(torch.tensor(xa, dtype=torch.float32).unsqueeze(0).to(device))).item()
        rows.append({"gene_id": r["Gene"], "chr": r["chr"], "pos": r["pos"],
                     "ref": r["ref"], "alt": r["alt"], "flank_pos": int(fp),
                     "eqtl_pvalue": r["pvalue"], "prob_ref": p_ref, "prob_alt": p_alt,
                     "delta": p_alt - p_ref})
        if len(rows) % 500 == 0:
            logger.info("已打分 %d/%d", len(rows), len(eq))

    out_df = pd.DataFrame(rows)
    out_df["abs_delta"] = out_df["delta"].abs()
    out_csv = os.path.join(vcfg["out_dir"], "eqtl_scores.tsv")
    out_df.to_csv(out_csv, sep="\t", index=False)
    logger.info("已打分 %d 个 SNP -> %s", len(out_df), out_csv)

    # --- 验证：eQTL 显著性 vs |Δ| ---
    out_df = out_df.dropna(subset=["eqtl_pvalue"])
    if len(out_df) < 100:
        logger.warning("有效样本过少，跳过统计验证")
        return
    neglog = -np.log10(out_df["eqtl_pvalue"].clip(lower=1e-300))
    corr, pcorr = stats.spearmanr(neglog, out_df["abs_delta"])
    # 强 vs 弱 eQTL（按 p 值中位数分组）
    strong = out_df["eqtl_pvalue"] <= out_df["eqtl_pvalue"].median()
    u, pw = stats.mannwhitneyu(out_df.loc[strong, "abs_delta"],
                               out_df.loc[~strong, "abs_delta"], alternative="greater")
    lines = [
        f"n_scored = {len(out_df)}",
        f"Spearman(-log10 p, |Δ|): rho={corr:.4f}, p={pcorr:.3e}",
        f"强eQTL |Δ| 中位数={out_df.loc[strong,'abs_delta'].median():.4f} vs 弱eQTL={out_df.loc[~strong,'abs_delta'].median():.4f}",
        f"Mann-Whitney (强>弱): U={u:.0f}, p={pw:.3e}",
    ]
    txt = "\n".join(lines)
    logger.info("\n%s", txt)
    with open(os.path.join(vcfg["out_dir"], "eqtl_validation.txt"), "w") as f:
        f.write(txt + "\n")

    # --- 图 ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(neglog, out_df["abs_delta"], s=3, alpha=0.4, color="#4c72b0")
    axes[0].set_xlabel("-log10(eQTL p)")
    axes[0].set_ylabel("|Δ predicted probability|")
    axes[0].set_title("eQTL significance vs variant effect")
    axes[1].hist([out_df.loc[strong, "abs_delta"], out_df.loc[~strong, "abs_delta"]],
                 bins=40, label=["strong eQTL", "weak eQTL"], alpha=0.7)
    axes[1].set_xlabel("|Δ|")
    axes[1].set_title("Distribution by eQTL strength")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("figures/fig5_variant_effect.png", dpi=150)
    logger.info("图已保存 -> figures/fig5_variant_effect.png")


if __name__ == "__main__":
    main()
