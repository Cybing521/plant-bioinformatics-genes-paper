# 中文说明（给作者与老师看的）

本仓库是投稿 PLOS ONE 的论文配套代码与数据仓库：
**<https://github.com/Cybing521/plant-bioinformatics-genes-paper>**

按 10 月 8 日的意见，代码与结果表都放在这个网站上了，文件名全部改过，仓库里不再出现 G3。

---

## 一、仓库里有什么

| 目录 | 内容 |
|---|---|
| `code/` | 全部代码：训练、评估、统计、可解释性、变异打分、画图 |
| `results/` | 52 个处理后结果表（稿件里的 S1 Table），即正文图表的全部底稿 |
| `results/excel/` | **同样内容的 Excel 版**，双击就能打开 |
| `figures/` | 6 张成图（Figure1–4、S1、S2），矢量 PDF |
| `tools/` | 两个维护脚本：生成 Excel 版、生成结果表说明 |

## 二、结果表怎么打开

`results/` 里的原始文件是 `.csv` `.tsv` `.json` `.npz` `.txt`，
这些是期刊要求和代码读取用的机器可读格式，**双击打不开**。

为此每个表都另存了一份 Excel 版，放在 `results/excel/`：

- 文件名与原始文件一一对应，例如 `scores.tsv` → `results/excel/scores.xlsx`
- 每个 Excel 的**第一个工作表叫 `说明`**，写清楚这张表是什么、支持哪张图
- 数据在第二个工作表里，列名、行列数与原始文件完全一致

汇总说明见 [`results/README.md`](results/README.md)，52 个文件的中英文逐条说明都在里面。

原来那 4 个文件对应关系：

| 原文件 | 是什么 | Excel 版 |
|---|---|---|
| `region_median_gene_level_bootstrap.csv` | Table 2 的核对表：四个区室的中位数 \|Δ\| 与基因级 bootstrap 95% CI | `results/excel/region_median_gene_level_bootstrap.xlsx` |
| `scores.tsv` | CNN 对 27,662 个变异位点的逐位点打分（2.7 万行） | `results/excel/scores.xlsx` |
| `agront_scores_by_site.tsv` | AgroNT 对同样位点的打分（2.6 万行） | `results/excel/agront_scores_by_site.xlsx` |
| `structured_dataset_params.json` | 数据集构造参数（区室长度、间隙、总长、标签阈值、染色体划分） | `results/excel/structured_dataset_params.xlsx` |

后三个文件的逐列解释见 `results/README.md` 的「三个关键文件说明」一节。

## 三、改了哪些文件名

| 原名 | 现名 |
|---|---|
| `04_代码/` | `code/` |
| `03_结果表/` | `results/` |
| `02_图件/` | `figures/` |
| `figures_src/plot_g3_figures.py` | `code/figures_src/plot_figures.py` |
| 环境变量 `G3_PKG_ROOT` / `G3_TABLES` / `G3_OUT` | `FIG_REPO_ROOT` / `FIG_TABLES` / `FIG_OUT` |

旧的 `投稿包_G3/` 整个目录已从仓库移除，其中重复的 `analyze_tissue_heterogeneity.py`
（与 `code/src/` 下完全相同的副本）也已删掉，只保留 `src/` 里的那一份。

## 四、代码能跑到哪一步

- **画图：可以。** `code/figures_src/plot_figures.py` 已实测能跑通，读 `results/` 里的表，
  重新生成 6 张 PDF，与仓库里的成品图逐字节一致（PDF 头里的生成时间戳除外）。
- **重训：没有端到端复跑验证。** 交付的训练脚本日期晚于服务器实际运行日志，
  从头重训不一定落回完全相同的数值。这一点稿件 Data Availability 里已如实声明。

## 五、本机有的、网上没有的

微调后的 CNN 权重与 LoRA adapter 不放在公开仓库，改为从公开的 AgroNT-1B 骨干重训复现；
第三方基因组数据同样不随仓库分发，用 `code/download_*.py` 现场拉取。
服务器地址、机器名、内部路径已全部清理过。