#!/usr/bin/env python3
"""下载 PGB 拟南芥基因表达任务（FASTA，直接解析，无需 datasets 库）。

PGB gene_exp 任务格式: FASTA 头为 >基因ID|样本1值|样本2值|...
标签编码在 FASTA 头中；本脚本只负责把 3 个 FASTA 文件拉到本地。

用法:  python data/download_pgb.py [--out data/pgb]
默认走 hf-mirror.com（国内镜像）；直连 hf.co 不可达时自动切换。
"""
import argparse
import os
import sys
import urllib.request

SPLITS = ["train", "validation", "test"]
# 可用物种: arabidopsis_thaliana / oryza_sativa / zea_mays / glycine_max / solanum_lycopersicum
SPECIES = "arabidopsis_thaliana"
REMOTE = ("datasets/InstaDeepAI/plant-genomic-benchmark/resolve/main/"
          "gene_exp/{species}_{split}.fa")
BASES = [
    "https://hf-mirror.com/",     # 国内镜像（优先）
    "https://huggingface.co/",    # 官方
]


def download(url, dest, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pgb")
    ap.add_argument("--species", default=SPECIES,
                    help="物种名，如 arabidopsis_thaliana / oryza_sativa / zea_mays")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ok, fail = 0, 0
    for split in SPLITS:
        dest = os.path.join(args.out, f"{args.species}_{split}.fa")
        if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
            print(f"[skip] {dest} 已存在")
            ok += 1
            continue
        for base in BASES:
            url = base + REMOTE.format(species=args.species, split=split)
            try:
                print(f">>> 下载 [{split}] <- {url}")
                download(url, dest)
                ok += 1
                break
            except Exception as e:
                print(f"    [{split}] {base} 失败: {e}")
                if os.path.exists(dest):
                    os.remove(dest)
        else:
            print(f"[FAIL] {split} 所有源都失败")
            fail += 1
    print(f">>> 完成: 成功 {ok}, 失败 {fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
