#!/usr/bin/env python3
"""从拟南芥变异 VCF 提取"基因侧翼区内的 SNP"，映射到侧翼序列坐标。

输入: data/1001genomes/arabidopsis_variants.vcf.gz (Ensembl Plants)
      data/tair10/Araport11_GFF3_genes_transposons.gff3
输出: data/1001genomes/variants.tsv
      chr<TAB>pos<TAB>ref<TAB>alt<TAB>gene_id<TAB>flank_pos
flank_pos 为变异在基因侧翼序列(启动子/UTR/终止子 串联)中的 0-indexed 位置。
"""
import bisect
import gzip
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from build_dataset import parse_gff, build_indexes

P, U5, U3, T = 1000, 500, 500, 1000
GAP = 20


def get_gene_meta(gff_path):
    """返回 gene_id -> (chr, strand, tss_0idx, tts_0idx, cds_start, cds_end)。"""
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
            cds_start, cds_end = tss, tts
        meta[gid] = (gene.seqid, strand, tss, tts, cds_start, cds_end)
    return meta


def flank_pos_for(gene_meta, chrom, pos0, P=P, U5=U5, U3=U3, T=T, GAP=GAP):
    """由 chr:pos(0-indexed) 计算侧翼序列内 flank_pos；不在侧翼区返回 None。"""
    chr_, strand, tss, tts, cds_start, cds_end = gene_meta
    if str(chr_) != str(chrom):
        return None
    if strand == "+":
        if tss - P <= pos0 < tss:                       # 启动子
            return pos0 - (tss - P)
        if tss <= pos0 < cds_start:                     # 5'UTR
            return P + (pos0 - tss)
        if cds_end < pos0 <= tts:                       # 3'UTR
            return P + U5 + GAP + (pos0 - cds_end - 1)
        if tts < pos0 <= tts + T:                       # 终止子
            return P + U5 + GAP + U3 + (pos0 - tts - 1)
    else:
        if tss < pos0 <= tss + P:                       # 启动子
            return P - (pos0 - tss)
        if cds_end < pos0 <= tss:                       # 5'UTR
            return P + (tss - pos0)
        if cds_start <= pos0 < cds_end:                 # 3'UTR
            return P + U5 + GAP + (cds_end - pos0 - 1)
        if tts - T <= pos0 < tts:                       # 终止子
            return P + U5 + GAP + U3 + (tts - pos0 - 1)
    return None


def main():
    vcf = "data/1001genomes/arabidopsis_variants.vcf.gz"
    gff = "data/tair10/Araport11_GFF3_genes_transposons.gff3"
    out = "data/1001genomes/variants.tsv"
    max_variants = int(os.environ.get("MAX_VARIANTS", 200000))

    print(">>> 构建基因元数据索引...")
    gene_meta = get_gene_meta(gff)
    # 按染色体构建排序的 (start, end, gid) 区间索引，配合 bisect 快速定位
    from collections import defaultdict
    by_chr = defaultdict(list)
    for gid, (chr_, strand, tss, tts, _, _) in gene_meta.items():
        by_chr[chr_].append((tss - P, tts + T, gid))
    for c in by_chr:
        by_chr[c].sort()
    starts = {c: [x[0] for x in by_chr[c]] for c in by_chr}

    print(">>> 解析 VCF（支持截断 gzip）...")
    rows = []
    try:
        fh = gzip.open(vcf, "rt")
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            chrom = cols[0].replace("Chr", "").replace("chr", "")
            if chrom not in by_chr:
                continue
            pos, ref = int(cols[1]), cols[3]
            alts = cols[4].split(",")
            pos0 = pos - 1
            slist = by_chr.get(chrom, [])
            i = bisect.bisect_right(starts[chrom], pos0) - 1
            for j in range(max(0, i - 2), min(len(slist), i + 3)):
                lo, hi, gid = slist[j]
                if lo <= pos0 < hi:
                    fp = flank_pos_for(gene_meta[gid], chrom, pos0)
                    if fp is not None:
                        for a in alts:
                            if len(ref) == 1 and len(a) == 1 and ref != a:
                                rows.append([chrom, pos, ref, a, gid, fp])
                    break
            if len(rows) >= max_variants:
                break
    except (EOFError, OSError, gzip.BadGzipFile) as e:
        print(f"[提示] VCF 为部分下载，读到截断点为止: {e}")

    df = pd.DataFrame(rows, columns=["chr", "pos", "ref", "alt", "gene_id", "flank_pos"])
    df = df.drop_duplicates()
    df.to_csv(out, sep="\t", index=False)
    print(f">>> 提取 {len(df)} 个侧翼区 SNP -> {out}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
