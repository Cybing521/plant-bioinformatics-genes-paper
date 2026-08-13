#!/usr/bin/env python3
"""下载拟南芥 TAIR10 参考基因组 + 基因注释（Ensembl Plants，稳定镜像）。

用法:  python data/download_tair10.py
产物:  data/tair10/
          TAIR10_Chr_all.fas      (基因组)
          Araport11_GFF3_genes_transposons.gff3  (基因注释)
说明:  表达矩阵 (expression_matrix.tsv) 需手动放置，格式见脚本末尾注释。
"""
import argparse
import gzip
import os
import shutil
import sys
import urllib.request

ENSEMBL_PLANTS_RELEASE = "59"
TAIR10_BASE = (
    "https://ftp.ebi.ac.uk/ensemblgenomes/pub/plants/release-"
    f"{ENSEMBL_PLANTS_RELEASE}/"
)

GENOME_URL = (TAIR10_BASE + "fasta/arabidopsis_thaliana/dna/"
              "Arabidopsis_thaliana.TAIR10.dna.toplevel.fa.gz")
GFF_URL = (TAIR10_BASE + "gff3/arabidopsis_thaliana/"
           "Arabidopsis_thaliana.TAIR10." + ENSEMBL_PLANTS_RELEASE + ".gff3.gz")

GENOME_GZ = "data/tair10/TAIR10_Chr_all.fa.gz"
GFF_GZ = "data/tair10/Araport11_GFF3_genes_transposons.gff3.gz"
GENOME_OUT = "data/tair10/TAIR10_Chr_all.fas"
GFF_OUT = "data/tair10/Araport11_GFF3_genes_transposons.gff3"


def download(url, dest):
    if os.path.exists(dest):
        print(f"已存在，跳过: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f">>> 下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"    -> {dest}")


def gunzip(src, dst):
    if os.path.exists(dst):
        print(f"已解压: {dst}")
        return
    print(f">>> 解压 {src}")
    with gzip.open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    print(f"    -> {dst}")


def main():
    if not os.path.exists(GENOME_OUT) or not os.path.exists(GFF_OUT):
        print(">>> 使用 Ensembl Plants "
              f"release-{ENSEMBL_PLANTS_RELEASE} 下载 TAIR10 数据")
        download(GENOME_URL, GENOME_GZ)
        download(GFF_URL, GFF_GZ)
        gunzip(GENOME_GZ, GENOME_OUT)
        gunzip(GFF_GZ, GFF_OUT)
    else:
        print(">>> 基因组与注释均已存在，跳过下载")

    print(f"""
>>> 完成。请手动准备表达矩阵:
    路径: data/tair10/expression_matrix.tsv
    格式: 制表符分隔，第一列为基因ID(如 AT1G01010.1)，其余列为一列一个样本/组织，
          值为 TPM/FPKM。示例:
            gene_id\\tsample_leaf\\tsample_root
            AT1G01010.1\\t120.5\\t45.2
    来源建议: TAIR(AtGenExpress)、Expression Atlas (E-GEOD-xxxx) 或自选公开 RNA-seq 矩阵。
""")


if __name__ == "__main__":
    main()
