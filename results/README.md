# results/ — 处理后结果表 / Processed result tables

本目录是论文正文图与表背后的**全部处理后表格**（PLOS ONE 稿件中的 *S1 Table*，共 52 个文件）。
这些表是结果的权威记录。

This directory holds every processed table behind the manuscript's figures and tables
(the *S1 Table* Supporting Information of the PLOS ONE manuscript, 52 files).

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

## 这 52 个文件分别是什么 / What each file is

| 文件 / File | 行数 / Rows | 是什么 / What it is | 对应图表 / Backs |
|---|---|---|---|
| `agront_eqtl_scores.tsv` | 17 | AgroNT 在同样 17 个 eQTL 位点上的打分，字段同上并附 AgroNT 的三个概率列。 | Exploratory text |
| `agront_multiseed_pgb.csv` | 5 | AgroNT-1B LoRA 五个随机种子（42–46）在 PGB 拟南芥测试集上的 accuracy / AUROC / AUPRC。每个种子一行。 | Table 1; Fig. 2c |
| `agront_multiseed_pgb_summary.txt` | 4 | AgroNT 五种子的均值与标准差汇总（文本）。 | Fig. 2c |
| `agront_pgb_1500bp_center_binary_test.json` | 11 | 输入长度消融：AgroNT 1500 bp 中心裁剪。 | Fig. 4c |
| `agront_pgb_3000bp_center_binary_test.json` | 10 | 输入长度消融：AgroNT 3000 bp 中心裁剪。 | Fig. 4c |
| `agront_pgb_binary_r8_test.json` | 19 | LoRA rank 消融：rank=8 的检查点在同一测试集上的指标（主实验用 rank=16）。 | Supplementary |
| `agront_pgb_binary_seed42_test.json` | 16 | AgroNT 主对比检查点在 seed-42 测试集上的完整指标（accuracy、AUROC、AUPRC、MCC、Brier 与 LoRA 超参）。 | Table 1 |
| `agront_pgb_binary_seed43_test.json` | 19 | AgroNT seed-43 测试指标。 | Fig. 2c |
| `agront_pgb_binary_seed44_test.json` | 19 | AgroNT seed-44 测试指标。 | Fig. 2c |
| `agront_pgb_binary_seed45_test.json` | 19 | AgroNT seed-45 测试指标。 | Fig. 2c |
| `agront_pgb_binary_seed46_test.json` | 19 | AgroNT seed-46 测试指标。 | Fig. 2c |
| `agront_region_importance.csv` | 5 | AgroNT 各区室每 bp 的归一化重要性。 | Fig. 3a,b |
| `agront_scores_by_site.tsv` | 26574 | AgroNT 在同样变异位点的打分。与 scores.tsv 不同，AgroNT 无法区分单个变异碱基，这里记的是每个位点上三个非参考碱基的平均 |Δ| 与最大 |Δ|（n_alts 列），再按基因到变异位点的距离汇总。 | Fig. 3e,f |
| `agront_variant_region_descriptives.csv` | 4 | AgroNT 各区室 |Δ| 的完整描述统计。 | Fig. 3e |
| `agront_variant_region_pairwise.csv` | 6 | AgroNT 区室两两 Mann–Whitney 检验的原始结果。 | Text 3.2 |
| `agront_variant_region_pairwise_fdr.csv` | 6 | AgroNT 区室两两检验加 BH-FDR 校正。 | Text 3.2 |
| `agront_variant_region_stats.csv` | 4 | AgroNT 各区室变异效应 |Δ| 的分位数统计。 | Fig. 3e |
| `agront_variant_region_summary.txt` | 10 | AgroNT 区室统计的文本摘要。 | Fig. 3e |
| `benchmark_results.csv` | 2 | 主对比表：锁定协议、seed 42、同一测试集下 CNN 与 AgroNT 的 accuracy / AUROC / AUPRC。 | Table 1; Fig. 2a,b |
| `benchmark_statistics.txt` | 8 | 主对比的完整统计量：样本量、阳性率、AUROC 及 bootstrap 95% CI、accuracy、MCC、Brier、ECE。 | Table 1; Fig. S2 |
| `cnn_multiseed_pgb.csv` | 5 | 本项目 1D-CNN 五个随机种子（42–46）在同一测试集上的 accuracy / AUROC / AUPRC。 | Fig. 2c |
| `cnn_multiseed_pgb_summary.txt` | 4 | CNN 五种子的均值与标准差汇总（文本）。 | Fig. 2c |
| `cnn_region_mutagenesis.csv` | 5 | CNN 的 50 bp 滑窗 N 诱变实验结果：各区室每 bp 的重要性，滑窗 50 bp、步长 25 bp、n=200 个位点。 | Fig. 3c |
| `cross_species_agront.csv` | 5 | AgroNT 的跨物种迁移结果，字段同上。 | Fig. 4a,b |
| `cross_species_results.csv` | 5 | CNN 在各作物 PGB 测试集上的跨物种迁移结果，每行一个物种，含训练集中位阈值与标签协议。 | Fig. 4a,b |
| `decision_curve.csv` | 19 | seed-42 测试集上的决策曲线：各阈值下 CNN、AgroNT、全预测为阳与全预测为阴的净获益。 | Fig. S2b |
| `eqtl_exploratory.txt` | 7 | 17 个 eQTL 变异的探索性重叠分析（未做多重检验校正，正文明确标为探索性）。 | Exploratory text |
| `eqtl_scores.tsv` | 17 | CNN 在 17 个已知 eQTL 变异位点上的打分，含 eQTL p 值与参考/变异等位基因预测概率。 | Exploratory text |
| `input_length_ablation.csv` | 3 | CNN 输入长度消融（仅 CNN，中心裁剪）。 | Fig. 4c |
| `input_length_ablation_both.csv` | 6 | 输入长度消融，CNN 与 AgroNT 两个模型合并在同一张表里（表内 model 列区分）。 | Fig. 4c |
| `input_length_ablation_center.csv` | 3 | 中心裁剪模式下的输入长度消融。 | Fig. 4c |
| `locked_protocol_stats.txt` | 68 | 锁定协议的配对检验：DeLong、McNemar、bootstrap CI，以及 ΔAUROC 的配对结果。 | Text 3.1; Fig. 2 |
| `multiseed_acceptance.txt` | 20 | 多种子实验的验收记录（时间、切分、主对比设定）。 | Internal record |
| `pgb_binary_preds.npz` | 197 | seed-42 测试集逐基因的预测结果（n=3,402），三列分别是真实标签 y_true、CNN 预测概率 p_cnn、AgroNT 预测概率 p_agront。这是正文校准曲线与决策曲线的原始输入，也是 Data Availability 里所说的 "seed-42 test probabilities"。 | Fig. S2a,b; Data Availability |
| `region_median_gene_level_bootstrap.csv` | 8 | Table 2 的核对表：两模型四个区室的中位数 |Δ| 及其基因级 cluster bootstrap 95% CI（2000 次重采样，seed=42），两模型不合并。 | Table 2; Fig. 3d |
| `region_validation.txt` | 3 | UTR 与终止子区室的 Mann–Whitney 验证（正文中作为区室效应存在的独立复核）。 | Text 3.2 |
| `reliability_agront.csv` | 15 | AgroNT 校准曲线，字段同 reliability_cnn.csv。 | Fig. S2a |
| `reliability_cnn.csv` | 15 | CNN 校准曲线：15 个等宽概率分箱，每箱的样本数、平均预测概率（confidence）与实际阳性率（accuracy）。 | Fig. S2a |
| `same_method_region_importance.csv` | 4 | 同一份数据在两个模型各自的归一化下得到的区室重要性，用于 Fig. 3c 的可比化呈现。 | Fig. 3c |
| `scores.tsv` | 27719 | CNN 在 27,662 个非 gap 变异位点的逐位点打分：参考等位基因与变异等位基因的预测概率及其差值 Δ。gene_id 为基因编号，flank_pos 为该基因到变异位点的距离（bp）。这是 Fig. 3e,f 与区室统计的原始输入。 | Fig. 3e,f |
| `seed42_extra_metrics.csv` | 2 | seed 42 两模型的补充指标：balanced accuracy、F1、MCC、Brier、阳性率。 | Table 1 |
| `structured_cnn_test.txt` | 5 | 结构化侧翼数据集上 CNN 一次运行的记录：训练/验证/测试规模、序列长度、标签阈值。 | Methods |
| `structured_dataset_params.json` | 27 | 结构化侧翼数据集的构造参数：各区室长度、区室间间隙、总长、标签阈值、高表达分位点，以及训练/验证/测试的染色体划分。这是 Fig. 1a 序列几何与 Methods 的依据。 | Fig. 1a; Methods |
| `tissue_concordance_ci.txt` | 10 | 组织标签一致率的均值、标准差与 bootstrap 95% CI。 | Fig. S1a |
| `tissue_heterogeneity_summary.txt` | 8 | 组织异质性汇总：唯一基因数、组织/样本数、训练集中位阈值等。 | Fig. S1 |
| `tissue_label_concordance.csv` | 56 | PGB 的 56 个组织/样本列各自按中位数划分的高低标签，与组织平均标签的一致率，以及该组织表达谱与全基因平均表达谱的 Spearman 相关。 | Fig. S1a,c |
| `tissue_pairwise_spearman.csv` | 56 | 56×56 的组织表达谱两两 Spearman 相关矩阵。CSV 第一行与第一列是组织编号 0–55。 | Fig. S1b |
| `variant_region_descriptives.csv` | 4 | CNN 各区室 |Δ| 的完整描述统计，含中位数的 bootstrap 95% CI 与四分位距。 | Fig. 3e |
| `variant_region_pairwise.csv` | 6 | CNN 区室两两 Mann–Whitney U 检验的原始结果（未校正），含 U 值、单侧 p 与两区室中位数。 | Text 3.2 |
| `variant_region_pairwise_fdr.csv` | 6 | 同一批 CNN 区室两两检验加上 BH-FDR 校正（q_bh 列）与秩二列相关，是正文引用的口径。 | Text 3.2 |
| `variant_region_stats.csv` | 4 | CNN 各区室变异效应 |Δ| 的分位数统计：n、中位数、均值、75% 与 95% 分位。 | Fig. 3e |
| `variant_region_summary.txt` | 6 | CNN 区室统计的文本摘要。 | Fig. 3e |

---

## English description of every file

| File | Rows | What it is | Backs |
|---|---|---|---|
| `agront_eqtl_scores.tsv` | Exploratory text | AgroNT scores at the same 17 eQTL sites. |
| `agront_multiseed_pgb.csv` | Table 1; Fig. 2c | Per-seed test metrics for the AgroNT-1B LoRA model (seeds 42-46), one row per seed. |
| `agront_multiseed_pgb_summary.txt` | Fig. 2c | Mean and standard deviation across the five AgroNT seeds. |
| `agront_pgb_1500bp_center_binary_test.json` | Fig. 4c | Input-length ablation: AgroNT at 1500 bp, centre crop. |
| `agront_pgb_3000bp_center_binary_test.json` | Fig. 4c | Input-length ablation: AgroNT at 3000 bp, centre crop. |
| `agront_pgb_binary_r8_test.json` | Supplementary | LoRA rank ablation: test metrics for the rank-8 checkpoint (the primary run uses rank 16). |
| `agront_pgb_binary_seed42_test.json` | Table 1 | Full test metrics of the primary AgroNT checkpoint on the seed-42 test set, including LoRA hyperparameters. |
| `agront_pgb_binary_seed43_test.json` | Fig. 2c | AgroNT seed-43 test metrics. |
| `agront_pgb_binary_seed44_test.json` | Fig. 2c | AgroNT seed-44 test metrics. |
| `agront_pgb_binary_seed45_test.json` | Fig. 2c | AgroNT seed-45 test metrics. |
| `agront_pgb_binary_seed46_test.json` | Fig. 2c | AgroNT seed-46 test metrics. |
| `agront_region_importance.csv` | Fig. 3a,b | AgroNT per-bp region importance. |
| `agront_scores_by_site.tsv` | Fig. 3e,f | Per-site AgroNT scores for the same variants. AgroNT cannot resolve a single alternate base, so each site records the mean and max |delta| over the three non-reference bases (n_alts), aggregated by distance from the gene. |
| `agront_variant_region_descriptives.csv` | Fig. 3e | AgroNT region-wise descriptive statistics of |delta|. |
| `agront_variant_region_pairwise.csv` | Text 3.2 | Raw pairwise Mann-Whitney U tests between AgroNT regions. |
| `agront_variant_region_pairwise_fdr.csv` | Text 3.2 | AgroNT pairwise tests with Benjamini-Hochberg FDR correction. |
| `agront_variant_region_stats.csv` | Fig. 3e | AgroNT region-wise quantiles of variant effect |delta|. |
| `agront_variant_region_summary.txt` | Fig. 3e | Text summary of the AgroNT region statistics. |
| `benchmark_results.csv` | Table 1; Fig. 2a,b | Head-to-head contrast under the locked protocol (seed 42): accuracy / AUROC / AUPRC for both models. |
| `benchmark_statistics.txt` | Table 1; Fig. S2 | Full statistics for the head-to-head contrast: n, prevalence, AUROC with bootstrap CI, accuracy, MCC, Brier, ECE. |
| `cnn_multiseed_pgb.csv` | Fig. 2c | Per-seed test metrics for the in-house 1D-CNN (seeds 42-46), one row per seed. |
| `cnn_multiseed_pgb_summary.txt` | Fig. 2c | Mean and standard deviation across the five CNN seeds. |
| `cnn_region_mutagenesis.csv` | Fig. 3c | CNN 50 bp sliding-window N-mutagenesis: per-bp region importance, 50 bp window, 25 bp step, n=200 sites. |
| `cross_species_agront.csv` | Fig. 4a,b | AgroNT cross-species transfer, same fields. |
| `cross_species_results.csv` | Fig. 4a,b | CNN cross-species transfer to each crop PGB test set, one row per species, with the training-set median threshold and label protocol. |
| `decision_curve.csv` | Fig. S2b | Decision curves on the seed-42 test set: net benefit for CNN, AgroNT, treat-all and treat-none at each threshold. |
| `eqtl_exploratory.txt` | Exploratory text | Exploratory overlap analysis for the 17 eQTL variants (uncorrected; flagged as exploratory in the text). |
| `eqtl_scores.tsv` | Exploratory text | CNN scores at the 17 known eQTL variant sites, with the eQTL p-value and reference/alternate probabilities. |
| `input_length_ablation.csv` | Fig. 4c | CNN input-length ablation (CNN only, centre crop). |
| `input_length_ablation_both.csv` | Fig. 4c | Input-length ablation for both models in one table (distinguished by the model column). |
| `input_length_ablation_center.csv` | Fig. 4c | Input-length ablation under centre cropping. |
| `locked_protocol_stats.txt` | Text 3.1; Fig. 2 | Paired tests for the locked protocol: DeLong, McNemar, bootstrap CI, and the paired AUROC difference. |
| `multiseed_acceptance.txt` | Internal record | Acceptance record for the multi-seed runs. |
| `pgb_binary_preds.npz` | Fig. S2a,b; Data Availability | Per-gene predictions on the seed-42 test set (n=3,402): true label y_true, CNN probability p_cnn, AgroNT probability p_agront. Raw input for the calibration and decision-curve figures, and the "seed-42 test probabilities" named in the Data Availability statement. |
| `region_median_gene_level_bootstrap.csv` | Table 2; Fig. 3d | The Table 2 backing table: median |delta| per region for both models with gene-level cluster bootstrap 95% CIs (2000 resamples, seed 42); the two models are never pooled. |
| `region_validation.txt` | Text 3.2 | Mann-Whitney validation of UTR versus terminator effects. |
| `reliability_agront.csv` | Fig. S2a | AgroNT reliability diagram, same fields. |
| `reliability_cnn.csv` | Fig. S2a | CNN reliability diagram: 15 equal-width probability bins with n, mean predicted probability (confidence) and observed positive rate (accuracy). |
| `same_method_region_importance.csv` | Fig. 3c | Region importance from the same data under each model's own normalisation, used for the comparable presentation in Fig. 3c. |
| `scores.tsv` | Fig. 3e,f | Per-site CNN scores for 27,662 non-gap variants: predicted probability of the reference and alternate allele and their difference (delta). gene_id is the gene, flank_pos the distance in bp. Raw input for Fig. 3e,f and all region statistics. |
| `seed42_extra_metrics.csv` | Table 1 | Supplementary seed-42 metrics: balanced accuracy, F1, MCC, Brier, prevalence. |
| `structured_cnn_test.txt` | Methods | One CNN run on the structured-flank dataset: train/validation/test sizes, sequence length, label threshold. |
| `structured_dataset_params.json` | Fig. 1a; Methods | Construction parameters of the structured-flank dataset: region lengths, inter-region gap, total length, label threshold, high-expression quantile, and the train/validation/test chromosome split. Backs the sequence geometry in Fig. 1a and the Methods. |
| `tissue_concordance_ci.txt` | Fig. S1a | Mean, standard deviation and bootstrap 95% CI of the tissue label agreement. |
| `tissue_heterogeneity_summary.txt` | Fig. S1 | Tissue heterogeneity summary: unique gene count, number of tissues/samples, training-set median threshold. |
| `tissue_label_concordance.csv` | Fig. S1a,c | For each of the 56 PGB tissue/sample columns: agreement between that tissue's own median-based high/low label and the tissue-averaged label, plus Spearman correlation with the gene-mean profile. |
| `tissue_pairwise_spearman.csv` | Fig. S1b | 56x56 matrix of pairwise Spearman correlations among the 56 tissue expression profiles. The first row and column hold the tissue indices 0-55. |
| `variant_region_descriptives.csv` | Fig. 3e | CNN region-wise descriptive statistics of |delta|, including bootstrap 95% CI for the median and the IQR. |
| `variant_region_pairwise.csv` | Text 3.2 | Raw pairwise Mann-Whitney U tests between CNN regions (uncorrected): U, one-sided p, and both medians. |
| `variant_region_pairwise_fdr.csv` | Text 3.2 | The same CNN pairwise tests with Benjamini-Hochberg FDR correction (q_bh) and rank-biserial correlation; this is the version cited in the text. |
| `variant_region_stats.csv` | Fig. 3e | CNN region-wise quantiles of variant effect |delta|: n, median, mean, 75th and 95th percentile. |
| `variant_region_summary.txt` | Fig. 3e | Text summary of the CNN region statistics. |

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
