#!/usr/bin/env python3
"""准备 1001 Genomes 自然变异 + eQTL 数据（用于变异效应验证，模块四）。

用法:  python data/download_1001.py
说明:
  - 1001 Genomes 全量变异文件很大（数百 MB），脚本支持从本地 VCF 转换，
    或按脚本内提示从官方 FTP 下载指定染色体 VCF 后手动处理。
  - eQTL 关联结果需从公开研究整理为 TSV（见脚本底部格式）。
"""
import argparse
import os


def vcf_to_tsv(vcf_path, out_tsv, max_variants=None, chroms=("1", "2", "3", "4", "5")):
    """将 VCF（每行一个位点的参考序列）转为轻量 TSV：chr pos ref alt gene_id(空)。
    供 in_silico.py 打分使用。gene_id 会在打分阶段按位置映射到最近基因。"""
    out = open(out_tsv, "w")
    out.write("chr\tpos\tref\talt\tgene_id\n")
    count = 0
    with open(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            chrom = cols[0].replace("Chr", "").replace("chr", "")
            if chrom not in chroms:
                continue
            pos, ref, alt = cols[1], cols[3], cols[4]
            for a in alt.split(","):
                out.write(f"{chrom}\t{pos}\t{ref}\t{a}\t\n")
                count += 1
                if max_variants and count >= max_variants:
                    out.close()
                    print(f">>> 已写出 {count} 个变异 -> {out_tsv}")
                    return
    out.close()
    print(f">>> 已写出 {count} 个变异 -> {out_tsv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", help="本地 1001 Genomes VCF 路径（可选）")
    ap.add_argument("--out", default="data/1001genomes/snps.tsv")
    ap.add_argument("--max", type=int, default=None, help="仅取前 N 个变异（测试用）")
    args = ap.parse_args()

    if args.vcf and os.path.exists(args.vcf):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        vcf_to_tsv(args.vcf, args.out, max_variants=args.max)
        print(">>> 变异数据就绪。若后续需要按基因打分，请补齐 gene_id 列。")
    else:
        print("""
>>> 未提供本地 VCF。请按以下任一路径准备 1001 Genomes 数据：

  1) 官方 FTP（按染色体下载，体量较大）:
     https://1001genomes.org/data/GMI-MPI/releases/v3.1/
     -> 选择 chr{1..5}.vcf.gz，解压后运行:
        python data/download_1001.py --vcf data/1001genomes/chr1.vcf --max 100000
  2) 或使用已有 VCF:  python data/download_1001.py --vcf <你的VCF>

>>> eQTL 数据（用于验证，可后置）:
  路径: data/eqtl/arabidopsis_eqtl.tsv
  格式(制表符):  chr<TAB>pos<TAB>ref<TAB>alt<TAB>gene_id<TAB>pvalue
  来源: 拟南芥公开 eQTL 研究（如 Kawakatsu et al. 2016, Cell）整理结果。
""")


if __name__ == "__main__":
    main()
