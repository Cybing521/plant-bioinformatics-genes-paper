#!/usr/bin/env python3
"""从 PGB 拟南芥表达任务 FASTA 提取表达标签 -> data/tair10/expression_matrix.tsv
【替换数据源】PGB 的 FASTA 头自带每基因约 53 个样本的表达值，
无需额外下载表达矩阵。build_dataset.py 会读取该 TSV 计算每基因平均表达。

用法:  python data/make_expression_from_pgb.py [--src data/pgb/arabidopsis_thaliana_train.fa]
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/pgb/arabidopsis_thaliana_train.fa",
                    help="PGB FASTA（默认 train，作为表达来源）")
    ap.add_argument("--out", default="data/tair10/expression_matrix.tsv")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        raise FileNotFoundError(f"缺少 PGB FASTA: {args.src}\n请先运行 python data/download_pgb.py")

    rows = []
    with open(args.src) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                fields = line[1:].split("|")
                gid = fields[0].split(".")[0]      # 去版本号
                vals = fields[1:]
                rows.append([gid] + vals)

    n_samples = max(len(r) - 1 for r in rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("gene_id\t" + "\t".join(f"sample{i}" for i in range(n_samples)) + "\n")
        for r in rows:
            r = r + [""] * (n_samples - (len(r) - 1))   # 补空列
            f.write("\t".join(r) + "\n")
    print(f">>> 已从 PGB 提取 {len(rows)} 个基因 x {n_samples} 样本 -> {args.out}")


if __name__ == "__main__":
    main()
