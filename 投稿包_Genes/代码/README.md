# 拟南芥"序列→表达"预测与调控变异效应分析 — 实现脚手架

对应研究方案：`../研究方案_定稿.md`（三区期刊目标 · 纯干实验 · 机器学习主体）

本仓库提供可一键运行的完整脚手架：数据下载 → 数据集构建 → **1D-CNN 与 AgroNT 大模型统一基准对比** → 可解释性（归因/区域重要性/TF-MoDISco）→ **in silico 变异效应预测与 eQTL 验证** → **组织异质性 / 变异分区补充分析**（无需重训）。

---

## 1. 服务器选型（AutoDL 推荐）

### 为什么需要 GPU 服务器
本机 Apple M4 无 NVIDIA GPU、磁盘紧张，**无法承担 AgroNT-1B 微调**。方案要求 24GB 显存（LoRA + bf16 + 梯度累积下 16GB 勉强可行）。

### AutoDL 实例选型（推荐）

| 配置项 | 推荐值 | 说明 |
|---|---|---|
| **GPU** | **RTX 4090 (24GB)**（优先）或 **RTX 3090 (24GB)**（性价比） | 24GB 满足 AgroNT LoRA 微调；4090 训练约快 2 倍 |
| **计费** | 4090 ≈ ¥2.5–3/h；3090 ≈ ¥1.5–2/h | 按小时计费，可关机不扣费（数据盘保留） |
| **地区** | 就近选择（华东/华南/华北） | 离你近延迟低 |
| **基础镜像** | **PyTorch 2.x + CUDA 12.x + Python 3.10** | 自带 torch+CUDA，无需重装 |
| **系统盘** | 50 GB | 存放环境与代码 |
| **数据盘** | **50–80 GB** | AgroNT 权重 ~4GB、PGB ~1GB、TAIR10 ~1GB、自建数据 ~2GB，留余量 |

### 开机步骤（约 5 分钟）
1. [autodl.com](https://www.autodl.com) 注册 → 控制台 → 「租用新实例」。
2. 地区任选 → GPU 选 RTX 4090/3090 → 基础镜像选 `PyTorch` 官方镜像（版本 2.x + cu121 及以上）。
3. 数据盘 50GB+ → 开机 → JupyterLab / SSH 进入。
4. 上传本仓库（可直接 `git init` 或用 JupyterLab 上传），运行：
   ```bash
   cd implementation
   bash setup_env.sh        # 装依赖（torch 已自带）
   bash scripts/run_pipeline.sh   # 端到端
   ```

> 备选：Google Colab 免费 T4 (16GB) 可跑 CNN 与小规模 AgroNT；阿里云/腾讯云 GPU 亦可，流程相同。

---

## 2. 目录结构

```
implementation/
├── README.md                  # 本文件
├── requirements.txt           # 依赖（不含 torch）
├── setup_env.sh               # 一键环境
├── configs/config.yaml        # 全部超参/路径
├── data/
│   ├── download_pgb.py        # PGB 拟南芥表达任务
│   ├── download_tair10.py     # TAIR10 基因组 + GFF
│   ├── build_dataset.py       # 侧翼序列+标签+防泄漏划分
│   └── download_1001.py       # 1001 Genomes 变异（模块四）
├── src/
│   ├── utils.py               # 设备/种子/指标
│   ├── dataloader.py          # npz + PGB 数据集
│   ├── models/
│   │   ├── cnn.py             # DeepSEA/Basenji 风格 1D-CNN
│   │   └── agront.py          # AgroNT-1B + LoRA
│   ├── train.py               # CNN 训练
│   ├── train_agront.py        # AgroNT 微调
│   ├── evaluate.py            # 统一基准评估（图 3）
│   ├── interpret/             # 归因/区域重要性/TF-MoDISco（图 4）
│   └── variants/              # in silico 打分/eQTL 验证（图 5）
├── scripts/run_pipeline.sh    # 端到端一键
├── results/                   # 模型权重/指标/验证结果
└── figures/                   # 论文图 3–5
```

---

## 3. 快速开始

```bash
cd implementation
bash setup_env.sh                      # 1) 环境
bash scripts/run_pipeline.sh           # 2) 端到端（数据→训练→评估→可解释性）
```

分步（对应研究方案模块）：

| 步骤 | 命令 | 对应模块/图 |
|---|---|---|
| 数据下载 | `python data/download_pgb.py` + `python data/download_tair10.py` | 模块一 |
| 建集 | `python data/build_dataset.py` | 模块一 |
| CNN 训练 | `python src/train.py --data pgb --task binary` | 模块二 |
| AgroNT 微调 | `python src/train_agront.py --data pgb --task binary` | 模块二 |
| 统一评估 | `python src/evaluate.py --data pgb --task binary` | 图 3 |
| 归因 | `python src/interpret/attribution.py` | 图 4 |
| 区域重要性 | `python src/interpret/region_importance.py` | 图 4 |
| TF-MoDISco | `python src/interpret/run_modisco.py` | 图 4 |
| 变异打分 | `python src/variants/in_silico.py` | 图 5 |
| eQTL 验证 | `python src/variants/validate.py` | 图 5 |

---

## 4. 数据准备细节（已实测可用）

### 4.0 数据可直接下载性核对（2026-08-05 实测）

| 数据 | 能否直接下载 | 方式 / 替换源 |
|---|---|---|
| PGB 拟南芥表达任务 | ✅ | `download_pgb.py` 走 **hf-mirror.com** 国内镜像直下 FASTA（hf.co 直连在国内不稳） |
| TAIR10 基因组 + GFF | ✅ | `download_tair10.py`（Ensembl Plants release-59，国内连接较慢，建议 curl `-C -` 续传） |
| 表达矩阵 | ✅ **已替换** | **无需额外下载**：`make_expression_from_pgb.py` 直接从 PGB FASTA 头提取每基因 56 样本表达值 → `expression_matrix.tsv` |
| JASPAR 植物模体库 | ✅ | `download_aux.py`（注意文件名是 `..._pfms_jaspar.txt`，复数 `pfms`） |
| 1001 Genomes 变异 | ⚠️ 体量大 | 1001genomes.org 可达；完整 VCF 大，建议按染色体下载后 `download_1001.py --vcf` 转 TSV |
| eQTL 数据 | ⚠️ 需整理 | `data/eqtl/arabidopsis_eqtl.tsv`（chr pos ref alt gene_id pvalue），从公开研究整理 |

### 4.1 一键下载数据（PGB + TAIR10 + 表达矩阵 + JASPAR）
```bash
bash setup_env.sh                        # 环境（torch 已自带则跳过）
python data/download_pgb.py              # PGB FASTA（hf-mirror）
python data/download_tair10.py           # TAIR10 基因组+GFF（慢，可用 curl -C - 续传）
python data/make_expression_from_pgb.py  # 表达矩阵（替换源，从 PGB 提取）
python data/download_aux.py              # JASPAR 模体库
```

### 4.2 1001 Genomes 变异（模块四，可选）
```bash
python data/download_1001.py --vcf <你的VCF> --max 100000
```

### 4.3 eQTL 数据（模块四验证，可选）
`data/eqtl/arabidopsis_eqtl.tsv`，列：`chr  pos  ref  alt  gene_id  pvalue`。

## 5. 服务器部署（AutoDL 实测记录）

| 服务器 | 用途 | 状态 |
|---|---|---|
| beijing_02 (RTX 4090) | 数据下载 | ✅ 全部数据已就位（PGB/TAIR10/表达/JASPAR） |
| shengxin_01 (RTX 4090) | 训练实例 | 新建 `plantdl` conda 环境装 torch+依赖 |

> 注意：beijing_02 的 GPU 被 Qwen vLLM 服务占用（`gpu-memory-utilization 0.9`），训练需用空闲实例。

---

## 6. 常见问题

### 5.1 AgroNT 报 `target_modules` 错误
AgroNT 结构属 InstaDeep NucleotideTransformer，模块名可能不是 `q_lin/k_lin/v_lin/out_lin`。排查：
```bash
python src/train_agront.py --print_modules
```
按输出的真实模块名更新 `configs/config.yaml` 的 `model.agront.target_modules`。

### 5.2 磁盘/显存不足
- 用 `torch.cuda.amp` 混合精度 + 梯度累积（`config.yaml` 已配 `grad_accum`）。
- 缩短 `max_seq_len` 或减小 `batch_size`。

### 5.3 PGB 数据集字段与脚本假设不符
`python data/download_pgb.py` 会打印实际列名；如字段名不同，按 `src/dataloader.py` 中 `PGBDataset` 的说明调整 `seq_col/label_col`。

### 5.4 归因只支持 CNN？
DeepSHAP 对文本输入（AgroNT）需 token 级归因，脚手架默认对 CNN 做碱基级归因；AgroNT 归因可参考 AgroNT 论文的 log-likelihood-ratio（LLR）法作为备选。

---

## 7. 结果到论文图件对应

| 论文图 | 生成命令 | 产物 |
|---|---|---|
| 图 2 数据基准 | `build_dataset.py` 日志 + meta.tsv | 数据统计 |
| 图 3 模型性能 | `evaluate.py` | `fig3_benchmark.png` + `benchmark_results.csv` |
| 图 4 可解释性 | `attribution.py` + `region_importance.py` + `run_modisco.py` | `fig4_region_importance.png` + modisco_report |
| 图 5 变异效应 | `in_silico.py` + `validate.py` | `fig5_variant_scores.png` / `fig5_eqtl_validation.png` |

> 性能-成本图（图 3 右）需从训练日志补充参数量/训练时长，`evaluate.py` 已留占位。

---

## 8. 后续建议

1. 先跑通 `--data pgb` 的 CNN 基线（最快），确认环境与指标，再上 AgroNT 微调。
2. 自建数据集（TAIR10 + 表达矩阵）用于「区域重要性 + 多组织」分析，PGB 用于统一基准。
3. 变异效应是差异化亮点：优先跑通 `in_silico.py` → `validate.py`，产出图 5。
4. 全部跑通后，进入论文撰写（可调用 ccf-paper-writer / 文献核验流程）。

## 补充分析（无需重训）
```bash
python analyze_tissue_heterogeneity.py --pgb-dir data/pgb --out-dir results --fig-dir figures
python analyze_variant_regions.py --scores ../结果表/scores.tsv --eqtl ../结果表/eqtl_scores.tsv --out-dir results --fig-dir figures
```
