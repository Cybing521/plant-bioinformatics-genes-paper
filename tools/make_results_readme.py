#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate results/README.md from the description table in make_excel_results.py.

The prose is kept in this script so that a new table can never ship without a written
explanation of what it is. Run it after adding a table to results/:

    python3 tools/make_results_readme.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from make_excel_results import META, TABLES

HEADER = """# results/ — 处理后结果表 / Processed result tables

本目录是论文正文图与表背后的**全部处理后表格**（PLOS ONE 稿件中的 *S1 Table*，共 {n_tables} 个文件）。
这些表是结果的权威记录。

This directory holds every processed table behind the manuscript's figures and tables
(the *S1 Table* Supporting Information of the PLOS ONE manuscript, {n_tables} files).

---

## 老师您好：怎么打开这些文件 / How to open these files

每个表格都有两个版本，**内容完全一致**：

| 版本 | 位置 | 用途 |
|---|---|---|
| 原始格式 | `results/*.csv` `*.tsv` `*.json` `*.npz` `*.txt` | 论文与期刊要求的机器可读版本，也是代码直接读取的版本 |
| **Excel 版** | `results/excel/*.xlsx` | **双击即可用 Excel 打开**，每个文件的第一个工作表 `说明` 写清楚了这是什么表、支持哪张图表 |

Excel 版与原始版是同一份数据，用 `tools/make_excel_results.py` 生成，可以随时重新生成：

```bash
python3 tools/make_excel_results.py      # 重新生成 results/excel/
```

| 版本 | Where | For |
|---|---|---|
| Original (`csv`/`tsv`/`json`/`npz`/`txt`) | `results/` | the machine-readable version the journal receives and the code reads |
| **Excel** | `results/excel/` | **double-click to open**; every file opens on a `说明` sheet describing the table and the figure it backs |

---

## 这 {n_tables} 个文件分别是什么 / What each file is

"""

FOOTER = """
---

## 文件名里没有的三个关键文件说明 / Notes on three files that are easy to misread

老师提到「只有 Excel 能打开，另外三个不知道是什么」，这三个是最容易被误解的，说明如下：

### 1. `scores.tsv` / `agront_scores_by_site.tsv` — 逐位点打分表（最大）

这两张表各有约 2.7 万行，是**逐个变异位点**的模型打分，不是摘要表。
每行一个「基因 + 变异位点」组合：

| 列 | 含义 |
|---|---|
| `gene_id` | 基因编号，例如 `AT1G01040` |
| `flank_pos` | 该基因到变异位点的距离（bp），`0` 表示变异就在基因所在位置 |
| `prob_ref` / `prob_alt` | 模型对参考等位基因 / 变异等位基因的高表达预测概率 |
| `delta` | `prob_alt − prob_ref`，即该变异的效应方向与大小 |
| `abs_delta` | `|delta|`，效应强度，Fig. 3 画的就是它 |

AgroNT 无法区分单个变异碱基，所以 `agront_scores_by_site.tsv` 记的是**每个位点上三个非参考碱基的平均 |Δ|**（`n_alts`、`mean_abs_delta`、`max_abs_delta` 三列）。两张表加起来才构成 Fig. 3e,f 的全部数据。

### 2. `structured_dataset_params.json` — 数据集构造参数（最短）

只有几十个字节，因为它**不是结果，是配置**。记录了结构化侧翼数据集怎么拼、区室多长、区室间空隙多少、
总长 3020 bp、标签阈值怎么定、以及染色体如何切成训练 / 验证 / 测试三份。正文 Fig. 1a 的序列示意图和
Methods 都以它为准。Excel 版把它拆成了「参数 / 取值」两列，`chr_split` 的染色体列表展开在下面。

### 3. `pgb_binary_preds.npz` — 稿件 Data Availability 点名的那个文件

这是 seed-42 测试集上**逐基因的预测概率**（n=3,402），三列：`y_true`（真实标签）、`p_cnn`（CNN 预测概率）、
`p_agront`（AgroNT 预测概率）。补充图 S2 的校准曲线和决策曲线就是从这张表算出来的，
稿件 Data Availability 里说的 "seed-42 test probabilities" 指的就是它。
`.npz` 是 NumPy 的压缩格式，双击打不开，所以给了 Excel 版（三列并排，更直观）。

---

## 与论文的对应关系 / How this maps onto the manuscript

| 稿件位置 | Manuscript location | 主要文件 / Key files |
|---|---|---|
| Table 1 | Table 1 | `benchmark_results.csv`, `benchmark_statistics.txt`, `seed42_extra_metrics.csv`, `agront_pgb_binary_seed42_test.json` |
| Table 2 | Table 2 | `region_median_gene_level_bootstrap.csv` |
| Fig. 1 | Fig. 1 | `structured_dataset_params.json`, `multiseed_acceptance.txt` |
| Fig. 2 | Fig. 2 | `benchmark_results.csv`, `cnn_multiseed_pgb.csv`, `agront_multiseed_pgb.csv`, `locked_protocol_stats.txt` |
| Fig. 3 | Fig. 3 | `scores.tsv`, `agront_scores_by_site.tsv`, `same_method_region_importance.csv`, `cnn_region_mutagenesis.csv`, `agront_region_importance.csv`, `variant_region_*.csv`, `region_median_gene_level_bootstrap.csv` |
| Fig. 4 | Fig. 4 | `cross_species_results.csv`, `cross_species_agront.csv`, `input_length_ablation*.csv` |
| Fig. S1 | Fig. S1 | `tissue_label_concordance.csv`, `tissue_pairwise_spearman.csv`, `tissue_concordance_ci.txt`, `tissue_heterogeneity_summary.txt` |
| Fig. S2 | Fig. S2 | `pgb_binary_preds.npz`, `reliability_cnn.csv`, `reliability_agront.csv`, `decision_curve.csv` |
| Data Availability | Data Availability | `pgb_binary_preds.npz` |

---

## 口径提醒 / Reading notes

- 表中的 `agront_` 前缀一律指 AgroNT-1B LoRA 模型，`cnn_` 前缀指本项目训练的 1D-CNN；
  不带前缀的文件是两者共有或历史对比用的表。
- 区域效应（`|Δ|`）的中位数都带**基因级 cluster bootstrap 95% CI**，两个模型的 CI **不合并**
  （见 `region_median_gene_level_bootstrap.csv` 的 `method` 列）。
- 区室两两比较以 `*_pairwise_fdr.csv` 为准，其中 `q_bh` 是 BH-FDR 校正后的 q 值；
  不带 `_fdr` 的同名文件是未校正的原始 p 值，正文不引用。
- eQTL 相关文件（`eqtl_*`, `agront_eqtl_scores.tsv`）是探索性分析，未做多重检验校正，
  正文明确标注为探索性，不作为主要结论。
"""

CN_HEADER = ""

EN_ROWS = """
---

## English description of every file

| File | Rows | What it is | Backs |
|---|---|---|---|
"""


def main() -> int:
    files = sorted(p for p in TABLES.iterdir() if p.is_file() and p.name in META)
    missing = sorted(
        p.name for p in TABLES.iterdir() if p.is_file() and p.name not in META and p.suffix.lower() in {".csv", ".tsv", ".json", ".npz", ".txt"}
    )
    if missing:
        print("WARNING: no description for:", ", ".join(missing), file=sys.stderr)

    n = len(files)
    parts = [HEADER.format(n_tables=n)]

    parts.append(CN_HEADER.format(n_tables=n))
    parts.append("| 文件 / File | 行数 / Rows | 是什么 / What it is | 对应图表 / Backs |\n|---|---|---|---|\n")
    for p in files:
        zh, _, fig = META[p.name]
        try:
            rows = max(sum(1 for _ in p.open(encoding="utf-8", errors="replace")) - 1, 0)
        except OSError:
            rows = 0
        parts.append(f"| `{p.name}` | {rows} | {zh} | {fig} |\n")

    parts.append(EN_ROWS)
    parts.append("\n".join(
        f"| `{p.name}` | {META[p.name][2]} | {META[p.name][1]} |" for p in files
    ) + "\n")

    parts.append(FOOTER)
    out = TABLES / "README.md"
    out.write_text("".join(parts), encoding="utf-8")
    print(f"wrote {out} ({n} tables described)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())