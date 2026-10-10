"""区室 |Δ| 中位数的基因级 cluster bootstrap 重算。

背景
----
`results/variant_region_descriptives.csv` 与 `agront_variant_region_descriptives.csv`
中的 `median_ci95_lo/hi` 是**位点级**自助重采样（在 `locked_protocol_stats.py` 的
`bootstrap_median_ci()` 中以 `RNG.integers(0, n, n)` 重采样**位点**，10,000 次）。
同一基因内部的位点高度相关，位点级重采样会低估不确定性——这正是终审报告 C5
指出的伪重复问题。

正文 Table 2 报告的是**基因级 cluster bootstrap**：以基因为重采样单元整体抽出，
同一基因的所有位点同进同出。经验上该估计非常稳定（5 个随机种子的自助中位数
只在第 5 位小数变动），因此可作为 Table 2 的权威来源。

区室布局取自 `structured_dataset_params.json`：
promoter 0–999 / 5'UTR 1000–1499 / gap 20 nt / 3'UTR 1520–2019 / terminator 2020–3019。

用法
----
    python3 region_median_gene_level_bootstrap.py --tables ../results --out ../results
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

# 区室边界（bp，flank_pos 为 0 起始）：promoter 1000 / utr5 500 / gap 20 / utr3 500 / terminator 1000
BINS = [-1, 999, 1499, 1519, 2019, 10 ** 9]
LABELS = ["promoter", "utr5", "gap", "utr3", "terminator"]
KEEP = ["promoter", "utr5", "utr3", "terminator"]

# 四个区室的展示顺序
ORDER = ["promoter", "utr5", "utr3", "terminator"]


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """按 flank_pos 划入区室，并剔除 gap 区间（无生物学含义）。"""
    out = df.assign(region=pd.cut(df["flank_pos"], BINS, labels=LABELS))
    return out[out["region"].isin(KEEP)]


def cluster_bootstrap_median(
    x: np.ndarray, genes: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """基因级 cluster bootstrap 的中位数与 95% 百分位区间。

    重采样单元是**基因**：每次抽 len(genes) 个基因（可重复），取其全部位点合并后
    算中位数。同一基因的位点不独立，故必须整块进出。
    """
    rng = np.random.default_rng(seed)
    index: dict[object, np.ndarray] = {}
    for gene, rows in pd.Series(np.arange(len(x))).groupby(genes).groups.items():
        index[gene] = np.asarray(rows)
    keys = np.array(sorted(index))
    meds = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.choice(keys, size=len(keys), replace=True)
        meds[i] = float(np.median(x[np.concatenate([index[k] for k in picked])]))
    return float(np.median(x)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables", default="../results", type=Path)
    ap.add_argument("--out", default=None, type=Path,
                    help="输出 CSV 目录，缺省与 --tables 相同")
    ap.add_argument("--n-boot", default=2000, type=int)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    tdir: Path = args.tables
    odir: Path = args.out or tdir

    cnn = annotate(pd.read_csv(tdir / "scores.tsv", sep="\t"))
    agront = annotate(
        pd.read_csv(tdir / "agront_scores_by_site.tsv", sep="\t")
        .rename(columns={"mean_abs_delta": "abs_delta"})
    )

    # 站点数应与位点级统计表一致，否则说明区室边界或输入文件有变
    reference = {r["region"]: int(r["n"]) for r in
                 csv.DictReader(open(tdir / "variant_region_stats.csv"))}
    for reg in ORDER:
        n = int((cnn["region"] == reg).sum())
        if n != reference[reg]:
            raise SystemExit(f"区室 {reg} 站点数不一致：本次 {n}，原始 {reference[reg]}")

    rows = []
    for model, df in (("cnn", cnn), ("agront", agront)):
        for reg in ORDER:
            sub = df[df["region"] == reg]
            x = sub["abs_delta"].to_numpy(dtype=float)
            g = sub["gene_id"].to_numpy()
            med, lo, hi = cluster_bootstrap_median(x, g, args.n_boot, args.seed)
            rows.append({
                "model": model,
                "region": reg,
                "n_sites": len(x),
                "n_genes": int(pd.unique(g).size),
                "median_abs_delta": round(med, 8),
                "ci95_lo": round(lo, 8),
                "ci95_hi": round(hi, 8),
                "method": f"gene-level cluster bootstrap, {args.n_boot} resamples, seed={args.seed}",
            })
            print(f"{model:7s} {reg:11s} n={len(x):6d} genes={pd.unique(g).size:4d}  "
                  f"{med:.5f} ({lo:.5f}–{hi:.5f})")

    odir.mkdir(parents=True, exist_ok=True)
    out = odir / "region_median_gene_level_bootstrap.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
