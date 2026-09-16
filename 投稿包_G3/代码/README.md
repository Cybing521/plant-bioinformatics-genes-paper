# 拟南芥序列→表达：CNN vs AgroNT 复现代码

对应投稿稿 `../论文/g3_paper.pdf`（G3 Investigation）。本目录是随稿代码，完整数据与权重需按下方命令下载；录用前请把代码和权重存进带 DOI 的归档库。

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
python analyze_tissue_heterogeneity.py --pgb-dir data/pgb --out-dir results --fig-dir figures
python analyze_variant_regions.py --scores ../结果表/scores.tsv --eqtl ../结果表/eqtl_scores.tsv --out-dir results --fig-dir figures
```

超参见 `config.yaml`。PGB 走 Hugging Face（国内可用镜像）；TAIR10 来自 Ensembl Plants release 59。

## 产物对应

| 稿中图表 | 主要输出 |
|---|---|
| Table 1 / Fig. 1 | `src/evaluate.py` → `benchmark_results.csv` |
| Fig. 2 组织异质性 | `analyze_tissue_heterogeneity.py` |
| Table 2 / Fig. 3 区域 | `src/interpret/` |
| Table 3 / Fig. 4 变异 | `src/variants/` + `analyze_variant_regions.py` |
| Table 4 / Fig. 5 跨物种 | `src/cross_species.py` |
| Table 5 长度消融 | `input_length_ablation_both.csv` |
