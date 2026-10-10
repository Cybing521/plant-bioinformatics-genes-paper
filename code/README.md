# 拟南芥序列→表达：CNN vs AgroNT 复现代码

本目录是随稿代码，对应投稿 PLOS ONE 的稿件。仓库根目录的 `README.md` 是总说明，
`results/` 是结果表的权威记录。

> **代码版本说明**：随包训练脚本的日期（2026-09-20）晚于服务器实际运行日志（2026-09-16–17），
> **未经端到端复现验证**。`../results/` 的表格是权威记录；脚本视为接近但未验证。详见 `AVAILABILITY.md`。
> 公开仓库只放代码、配置和处理后表格；**不要提交** `results/models/*.pt`、LoRA adapter
> 或 AgroNT 骨干拷贝（`.gitignore` 已拦截）。复现方式是下载公开 PGB 与 Hugging Face AgroNT-1B 后重训。

图件（`../figures/Figure1.pdf … FigureS2.pdf`）是成品 PDF，随稿交付。
**图件生成脚本已随包交付**：`figures_src/plot_figures.py`，只读取 `../results/`，不依赖 GPU。

```bash
cd figures_src
MPLCONFIGDIR=/tmp/mplcache FIG_OUT=<输出目录> python3 -c \
  "import plot_figures as P; [f() for f in (P.plot_fig1,P.plot_fig2,P.plot_fig3,P.plot_fig4,P.plot_figs1,P.plot_figs2)]"
```

路径可用环境变量覆盖（`FIG_REPO_ROOT` / `FIG_TABLES` / `FIG_OUT`）：

| 环境变量 | 默认值 |
|---|---|
| `FIG_REPO_ROOT` | 仓库根目录（脚本上溯两级） |
| `FIG_TABLES` | `<仓库根>/results` |
| `FIG_OUT` | `<仓库根>/figures` |

依赖 `matplotlib`、`pandas`、`numpy`。原先依赖本机 skill 目录的两个样式模块
（`house_flowchart.py`、`publication_style.py`）已随仓库内置到 `figures_src/style/`，
外部无需任何路径配置。

## 环境

```bash
bash setup_env.sh
# 或: pip install -r requirements.txt
```

AgroNT-1B LoRA 需要约 24 GB 显存（RTX 4090 / 3090）。仅 CNN 可用更小 GPU。

## 主流程

脚本在本目录根下，模型在 `src/`。

```bash
python download_pgb.py
python download_tair10.py          # 区域分析需要
python make_expression_from_pgb.py
python build_dataset.py

python src/train.py --data pgb --task binary
python src/train_agront.py --data pgb --task binary
python src/evaluate.py --data pgb --task binary

python src/interpret/attribution.py
python src/interpret/run_modisco.py
python src/variants/in_silico.py
python src/cross_species.py
```

补充分析（不必重训）：

```bash
python src/evaluate.py --data pgb --task binary --save-preds results/pgb_binary_preds.npz
python src/statistics.py --preds results/pgb_binary_preds.npz
python src/locked_protocol_stats.py --tables ../results
python src/analyze_tissue_heterogeneity.py --pgb-dir data/pgb --out-dir results
python src/analyze_variant_regions.py --scores ../results/scores.tsv --eqtl ../results/eqtl_scores.tsv --out-dir results
```

复现区室中位数表（Table 2 的核对表）：

```bash
python src/region_median_gene_level_bootstrap.py --tables ../results --out ../results
```

多种子 / LoRA rank / 官方多组织回归 / 统一可解释性（需本地 GPU 与本地 checkpoint）：

```bash
bash scripts/run_multiseed.sh
bash scripts/run_lora_ablation.sh
bash scripts/run_multitissue_regression.sh
bash scripts/run_unified_interpret.sh
```

超参见 `config.yaml`。PGB 走 Hugging Face（国内可用镜像）；TAIR10 来自 Ensembl Plants release 59。

## 产物对应

表内路径相对本目录。

| 稿中图表 | 主要输出 |
|---|---|
| Table 1 / Fig. 2 | `src/evaluate.py` → `../results/benchmark_results.csv` |
| Fig. S1 组织异质性 | `src/analyze_tissue_heterogeneity.py` |
| Table 2 / Fig. 3 区域 a–d | `src/interpret/`、`src/variants/`、`figures_src/plot_figures.py` |
| 变异 \|Δ\| Fig. 3e,f | `src/variants/` + `src/analyze_variant_regions.py` |
| Fig. 4 跨物种 | `src/cross_species.py` |
| 长度消融 Fig. 4c | `../results/input_length_ablation_both.csv` |
| Fig. S2 校准 / 决策曲线 | `src/statistics.py` |
| Table 2 区室中位数（基因级 bootstrap） | `src/region_median_gene_level_bootstrap.py` → `../results/region_median_gene_level_bootstrap.csv` |
| 全部 6 张图 | `figures_src/plot_figures.py` |

## 文件说明

| 路径 | 内容 |
|---|---|
| `download_pgb.py` / `download_tair10.py` / `download_aux.py` / `download_1001.py` | 第三方原始数据获取脚本，文件本身不入库 |
| `make_expression_from_pgb.py` | 从 PGB FASTA 头部提取表达矩阵 TSV |
| `build_dataset.py` | 结构化侧翼数据集构建 |
| `prep_variants.py` | 变异位点准备 |
| `src/models/` | CNN 与 AgroNT 模型定义 |
| `src/variants/` | 体内外变异打分、eQTL、区域验证 |
| `src/interpret/` | 归因、诱变、MoDISco、区室重要性 |
| `src/region_median_gene_level_bootstrap.py` | Table 2 区室中位数与基因级 bootstrap CI |
| `figures_src/plot_figures.py` | 生成全部 6 张图 |
| `figures_src/style/` | 内置绘图样式模块（matplotlib helpers） |
| `configs/config.yaml` | 供 17 个脚本的 `--config` 默认值解析使用 |

## 两处历史沿革

- 旧版 `analyze_variant_regions.py`（单侧 Mann--Whitney、无 FDR 校正）与正文口径矛盾，已删除；
  复现一律走 `src/analyze_variant_regions.py`。
- 原 Fig. S3（探索性 50 bp 诱变）与 Fig. 3c 是同一份数据，已移除；
  `src/interpret/cnn_region_mutagenesis.py` 仍然产出其底层表 `cnn_region_mutagenesis.csv`。