#!/usr/bin/env python3
"""下载替换数据源：

1. 表达矩阵   -> 无需额外下载：运行 make_expression_from_pgb.py 从 PGB 提取
                 （GSE78482 曾被误用，已弃用；TAIR 需登录被 403 拦截）
2. JASPAR     -> JASPAR2024 CORE plants non-redundant 模体库（TF-MoDISco TOMTOM 用）
3. (可选)1001 -> 仅测试可达性并打印下载说明

用法:  python data/download_aux.py [--with-1001]
"""
import argparse
import os
import urllib.request

JASPAR_URL = ("https://jaspar.elixir.no/download/data/2024/CORE/"
              "JASPAR2024_CORE_plants_non-redundant_pfms_jaspar.txt")
T1001_URL = "https://1001genomes.org/data/GMI-MPI/releases/v3.1/"


def _download(url, dest, timeout=300):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[skip] {dest}")
        return
    print(f">>> 下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"    -> {dest} ({os.path.getsize(dest)/1024:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-1001", action="store_true")
    args = ap.parse_args()

    _download(JASPAR_URL, "data/motifs/JASPAR2024_plants.jaspar")

    if args.with_1001:
        try:
            _download(T1001_URL, "data/1001genomes/_index.html", timeout=30)
            print("1001genomes.org 可达。完整 VCF 较大，建议按染色体下载后 "
                  "用 data/download_1001.py 转为 TSV。")
        except Exception as e:
            print(f"[FAIL] 1001genomes.org 不可达: {e}")


if __name__ == "__main__":
    main()
