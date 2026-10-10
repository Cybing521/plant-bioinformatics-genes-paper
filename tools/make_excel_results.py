#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert every table in results/ into an Excel-openable .xlsx under results/excel/.

Why: the deposited tables are machine-readable (csv/tsv/json/npz/txt), which is what
scripts and reviewers' pipelines want, but a plain .tsv or .json cannot be opened by
double-clicking it in Excel. This script produces one .xlsx per table with the same
name, plus a leading 说明 sheet that says what the table is and which figure/table it
backs. The original files are left untouched.

Usage:
    python3 tools/make_excel_results.py                # writes results/excel/
    python3 tools/make_excel_results.py --check        # verify existing .xlsx only
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results"
OUT = ROOT / "results" / "excel"

ROUND_DP = 10

# file stem -> (中文说明, English description, 对应图表)
META: dict[str, tuple[str, str, str]] = {
    "agront_multiseed_pgb.csv": (
        "AgroNT-1B LoRA 五个随机种子（42–46）在 PGB 拟南芥测试集上的 accuracy / AUROC / AUPRC。每个种子一行。",
        "Per-seed test metrics for the AgroNT-1B LoRA model (seeds 42-46), one row per seed.",
        "Table 1; Fig. 2c",
    ),
    "cnn_multiseed_pgb.csv": (
        "本项目 1D-CNN 五个随机种子（42–46）在同一测试集上的 accuracy / AUROC / AUPRC。",
        "Per-seed test metrics for the in-house 1D-CNN (seeds 42-46), one row per seed.",
        "Fig. 2c",
    ),
    "cnn_multiseed_pgb_summary.txt": ("CNN 五种子的均值与标准差汇总（文本）。", "Mean and standard deviation across the five CNN seeds.", "Fig. 2c"),
    "agront_multiseed_pgb_summary.txt": ("AgroNT 五种子的均值与标准差汇总（文本）。", "Mean and standard deviation across the five AgroNT seeds.", "Fig. 2c"),
    "benchmark_results.csv": (
        "主对比表：锁定协议、seed 42、同一测试集下 CNN 与 AgroNT 的 accuracy / AUROC / AUPRC。",
        "Head-to-head contrast under the locked protocol (seed 42): accuracy / AUROC / AUPRC for both models.",
        "Table 1; Fig. 2a,b",
    ),
    "benchmark_statistics.txt": (
        "主对比的完整统计量：样本量、阳性率、AUROC 及 bootstrap 95% CI、accuracy、MCC、Brier、ECE。",
        "Full statistics for the head-to-head contrast: n, prevalence, AUROC with bootstrap CI, accuracy, MCC, Brier, ECE.",
        "Table 1; Fig. S2",
    ),
    "locked_protocol_stats.txt": (
        "锁定协议的配对检验：DeLong、McNemar、bootstrap CI，以及 ΔAUROC 的配对结果。",
        "Paired tests for the locked protocol: DeLong, McNemar, bootstrap CI, and the paired AUROC difference.",
        "Text 3.1; Fig. 2",
    ),
    "seed42_extra_metrics.csv": (
        "seed 42 两模型的补充指标：balanced accuracy、F1、MCC、Brier、阳性率。",
        "Supplementary seed-42 metrics: balanced accuracy, F1, MCC, Brier, prevalence.",
        "Table 1",
    ),
    "multiseed_acceptance.txt": ("多种子实验的验收记录（时间、切分、主对比设定）。", "Acceptance record for the multi-seed runs.", "Internal record"),
    "scores.tsv": (
        "CNN 在 27,662 个非 gap 变异位点的逐位点打分：参考等位基因与变异等位基因的预测概率及其差值 Δ。"
        "gene_id 为基因编号，flank_pos 为该基因到变异位点的距离（bp）。这是 Fig. 3e,f 与区室统计的原始输入。",
        "Per-site CNN scores for 27,662 non-gap variants: predicted probability of the reference and "
        "alternate allele and their difference (delta). gene_id is the gene, flank_pos the distance in bp. "
        "Raw input for Fig. 3e,f and all region statistics.",
        "Fig. 3e,f",
    ),
    "agront_scores_by_site.tsv": (
        "AgroNT 在同样变异位点的打分。与 scores.tsv 不同，AgroNT 无法区分单个变异碱基，这里记的是"
        "每个位点上三个非参考碱基的平均 |Δ| 与最大 |Δ|（n_alts 列），再按基因到变异位点的距离汇总。",
        "Per-site AgroNT scores for the same variants. AgroNT cannot resolve a single alternate base, so each "
        "site records the mean and max |delta| over the three non-reference bases (n_alts), aggregated by distance "
        "from the gene.",
        "Fig. 3e,f",
    ),
    "same_method_region_importance.csv": (
        "同一份数据在两个模型各自的归一化下得到的区室重要性，用于 Fig. 3c 的可比化呈现。",
        "Region importance from the same data under each model's own normalisation, used for the comparable "
        "presentation in Fig. 3c.",
        "Fig. 3c",
    ),
    "cnn_region_mutagenesis.csv": (
        "CNN 的 50 bp 滑窗 N 诱变实验结果：各区室每 bp 的重要性，滑窗 50 bp、步长 25 bp、n=200 个位点。",
        "CNN 50 bp sliding-window N-mutagenesis: per-bp region importance, 50 bp window, 25 bp step, n=200 sites.",
        "Fig. 3c",
    ),
    "agront_region_importance.csv": ("AgroNT 各区室每 bp 的归一化重要性。", "AgroNT per-bp region importance.", "Fig. 3a,b"),
    "variant_region_stats.csv": (
        "CNN 各区室变异效应 |Δ| 的分位数统计：n、中位数、均值、75% 与 95% 分位。",
        "CNN region-wise quantiles of variant effect |delta|: n, median, mean, 75th and 95th percentile.",
        "Fig. 3e",
    ),
    "agront_variant_region_stats.csv": ("AgroNT 各区室变异效应 |Δ| 的分位数统计。", "AgroNT region-wise quantiles of variant effect |delta|.", "Fig. 3e"),
    "variant_region_descriptives.csv": (
        "CNN 各区室 |Δ| 的完整描述统计，含中位数的 bootstrap 95% CI 与四分位距。",
        "CNN region-wise descriptive statistics of |delta|, including bootstrap 95% CI for the median and the IQR.",
        "Fig. 3e",
    ),
    "agront_variant_region_descriptives.csv": ("AgroNT 各区室 |Δ| 的完整描述统计。", "AgroNT region-wise descriptive statistics of |delta|.", "Fig. 3e"),
    "variant_region_summary.txt": ("CNN 区室统计的文本摘要。", "Text summary of the CNN region statistics.", "Fig. 3e"),
    "agront_variant_region_summary.txt": ("AgroNT 区室统计的文本摘要。", "Text summary of the AgroNT region statistics.", "Fig. 3e"),
    "variant_region_pairwise.csv": (
        "CNN 区室两两 Mann–Whitney U 检验的原始结果（未校正），含 U 值、单侧 p 与两区室中位数。",
        "Raw pairwise Mann-Whitney U tests between CNN regions (uncorrected): U, one-sided p, and both medians.",
        "Text 3.2",
    ),
    "variant_region_pairwise_fdr.csv": (
        "同一批 CNN 区室两两检验加上 BH-FDR 校正（q_bh 列）与秩二列相关，是正文引用的口径。",
        "The same CNN pairwise tests with Benjamini-Hochberg FDR correction (q_bh) and rank-biserial correlation; "
        "this is the version cited in the text.",
        "Text 3.2",
    ),
    "agront_variant_region_pairwise.csv": ("AgroNT 区室两两 Mann–Whitney 检验的原始结果。", "Raw pairwise Mann-Whitney U tests between AgroNT regions.", "Text 3.2"),
    "agront_variant_region_pairwise_fdr.csv": ("AgroNT 区室两两检验加 BH-FDR 校正。", "AgroNT pairwise tests with Benjamini-Hochberg FDR correction.", "Text 3.2"),
    "region_validation.txt": (
        "UTR 与终止子区室的 Mann–Whitney 验证（正文中作为区室效应存在的独立复核）。",
        "Mann-Whitney validation of UTR versus terminator effects.",
        "Text 3.2",
    ),
    "region_median_gene_level_bootstrap.csv": (
        "Table 2 的核对表：两模型四个区室的中位数 |Δ| 及其基因级 cluster bootstrap 95% CI"
        "（2000 次重采样，seed=42），两模型不合并。",
        "The Table 2 backing table: median |delta| per region for both models with gene-level cluster bootstrap "
        "95% CIs (2000 resamples, seed 42); the two models are never pooled.",
        "Table 2; Fig. 3d",
    ),
    "cross_species_results.csv": (
        "CNN 在各作物 PGB 测试集上的跨物种迁移结果，每行一个物种，含训练集中位阈值与标签协议。",
        "CNN cross-species transfer to each crop PGB test set, one row per species, with the training-set median "
        "threshold and label protocol.",
        "Fig. 4a,b",
    ),
    "cross_species_agront.csv": ("AgroNT 的跨物种迁移结果，字段同上。", "AgroNT cross-species transfer, same fields.", "Fig. 4a,b"),
    "input_length_ablation.csv": ("CNN 输入长度消融（仅 CNN，中心裁剪）。", "CNN input-length ablation (CNN only, centre crop).", "Fig. 4c"),
    "input_length_ablation_center.csv": ("中心裁剪模式下的输入长度消融。", "Input-length ablation under centre cropping.", "Fig. 4c"),
    "input_length_ablation_both.csv": (
        "输入长度消融，CNN 与 AgroNT 两个模型合并在同一张表里（表内 model 列区分）。",
        "Input-length ablation for both models in one table (distinguished by the model column).",
        "Fig. 4c",
    ),
    "reliability_cnn.csv": (
        "CNN 校准曲线：15 个等宽概率分箱，每箱的样本数、平均预测概率（confidence）与实际阳性率（accuracy）。",
        "CNN reliability diagram: 15 equal-width probability bins with n, mean predicted probability (confidence) "
        "and observed positive rate (accuracy).",
        "Fig. S2a",
    ),
    "reliability_agront.csv": ("AgroNT 校准曲线，字段同 reliability_cnn.csv。", "AgroNT reliability diagram, same fields.", "Fig. S2a"),
    "decision_curve.csv": (
        "seed-42 测试集上的决策曲线：各阈值下 CNN、AgroNT、全预测为阳与全预测为阴的净获益。",
        "Decision curves on the seed-42 test set: net benefit for CNN, AgroNT, treat-all and treat-none at each "
        "threshold.",
        "Fig. S2b",
    ),
    "pgb_binary_preds.npz": (
        "seed-42 测试集逐基因的预测结果（n=3,402），三列分别是真实标签 y_true、CNN 预测概率 p_cnn、"
        "AgroNT 预测概率 p_agront。这是正文校准曲线与决策曲线的原始输入，也是 Data Availability 里所说的 "
        "\"seed-42 test probabilities\"。",
        "Per-gene predictions on the seed-42 test set (n=3,402): true label y_true, CNN probability p_cnn, "
        "AgroNT probability p_agront. Raw input for the calibration and decision-curve figures, and the "
        "\"seed-42 test probabilities\" named in the Data Availability statement.",
        "Fig. S2a,b; Data Availability",
    ),
    "tissue_label_concordance.csv": (
        "PGB 的 56 个组织/样本列各自按中位数划分的高低标签，与组织平均标签的一致率，以及该组织表达谱与"
        "全基因平均表达谱的 Spearman 相关。",
        "For each of the 56 PGB tissue/sample columns: agreement between that tissue's own median-based high/low "
        "label and the tissue-averaged label, plus Spearman correlation with the gene-mean profile.",
        "Fig. S1a,c",
    ),
    "tissue_pairwise_spearman.csv": (
        "56×56 的组织表达谱两两 Spearman 相关矩阵。CSV 第一行与第一列是组织编号 0–55。",
        "56x56 matrix of pairwise Spearman correlations among the 56 tissue expression profiles. The first row "
        "and column hold the tissue indices 0-55.",
        "Fig. S1b",
    ),
    "tissue_concordance_ci.txt": ("组织标签一致率的均值、标准差与 bootstrap 95% CI。", "Mean, standard deviation and bootstrap 95% CI of the tissue label agreement.", "Fig. S1a"),
    "tissue_heterogeneity_summary.txt": (
        "组织异质性汇总：唯一基因数、组织/样本数、训练集中位阈值等。",
        "Tissue heterogeneity summary: unique gene count, number of tissues/samples, training-set median threshold.",
        "Fig. S1",
    ),
    "eqtl_scores.tsv": (
        "CNN 在 17 个已知 eQTL 变异位点上的打分，含 eQTL p 值与参考/变异等位基因预测概率。",
        "CNN scores at the 17 known eQTL variant sites, with the eQTL p-value and reference/alternate probabilities.",
        "Exploratory text",
    ),
    "agront_eqtl_scores.tsv": ("AgroNT 在同样 17 个 eQTL 位点上的打分，字段同上并附 AgroNT 的三个概率列。", "AgroNT scores at the same 17 eQTL sites.", "Exploratory text"),
    "eqtl_exploratory.txt": (
        "17 个 eQTL 变异的探索性重叠分析（未做多重检验校正，正文明确标为探索性）。",
        "Exploratory overlap analysis for the 17 eQTL variants (uncorrected; flagged as exploratory in the text).",
        "Exploratory text",
    ),
    "structured_dataset_params.json": (
        "结构化侧翼数据集的构造参数：各区室长度、区室间间隙、总长、标签阈值、高表达分位点，以及训练/"
        "验证/测试的染色体划分。这是 Fig. 1a 序列几何与 Methods 的依据。",
        "Construction parameters of the structured-flank dataset: region lengths, inter-region gap, total length, "
        "label threshold, high-expression quantile, and the train/validation/test chromosome split. Backs the "
        "sequence geometry in Fig. 1a and the Methods.",
        "Fig. 1a; Methods",
    ),
    "structured_cnn_test.txt": (
        "结构化侧翼数据集上 CNN 一次运行的记录：训练/验证/测试规模、序列长度、标签阈值。",
        "One CNN run on the structured-flank dataset: train/validation/test sizes, sequence length, label threshold.",
        "Methods",
    ),
    "agront_pgb_binary_seed42_test.json": (
        "AgroNT 主对比检查点在 seed-42 测试集上的完整指标（accuracy、AUROC、AUPRC、MCC、Brier 与 LoRA 超参）。",
        "Full test metrics of the primary AgroNT checkpoint on the seed-42 test set, including LoRA hyperparameters.",
        "Table 1",
    ),
    "agront_pgb_binary_seed43_test.json": ("AgroNT seed-43 测试指标。", "AgroNT seed-43 test metrics.", "Fig. 2c"),
    "agront_pgb_binary_seed44_test.json": ("AgroNT seed-44 测试指标。", "AgroNT seed-44 test metrics.", "Fig. 2c"),
    "agront_pgb_binary_seed45_test.json": ("AgroNT seed-45 测试指标。", "AgroNT seed-45 test metrics.", "Fig. 2c"),
    "agront_pgb_binary_seed46_test.json": ("AgroNT seed-46 测试指标。", "AgroNT seed-46 test metrics.", "Fig. 2c"),
    "agront_pgb_binary_r8_test.json": (
        "LoRA rank 消融：rank=8 的检查点在同一测试集上的指标（主实验用 rank=16）。",
        "LoRA rank ablation: test metrics for the rank-8 checkpoint (the primary run uses rank 16).",
        "Supplementary",
    ),
    "agront_pgb_1500bp_center_binary_test.json": ("输入长度消融：AgroNT 1500 bp 中心裁剪。", "Input-length ablation: AgroNT at 1500 bp, centre crop.", "Fig. 4c"),
    "agront_pgb_3000bp_center_binary_test.json": ("输入长度消融：AgroNT 3000 bp 中心裁剪。", "Input-length ablation: AgroNT at 3000 bp, centre crop.", "Fig. 4c"),
}

INFO_ROWS = ("原始文件", "生成时间", "数据行数", "列数", "对应图表")


def _flatten(obj, prefix=""):
    rows = []
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            rows.extend(_flatten(v, key + "."))
        else:
            rows.append((key, ", ".join(map(str, v)) if isinstance(v, list) else v))
    return rows


def _round(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=["float64", "float32"]).columns
    if len(num):
        df = df.copy()
        df[num] = df[num].round(ROUND_DP)
    return df


def _info_sheet(stem: str, rows: int, cols: int, stamp: str) -> pd.DataFrame:
    zh, en, fig = META[stem]
    body = [INFO_ROWS, [stem, stamp, rows, cols, fig]]
    return pd.DataFrame(body[1:], columns=list(INFO_ROWS))


def _autofit(writer, df: pd.DataFrame, sheet: str, cap: int = 42) -> None:
    ws = writer.sheets[sheet]
    for i, col in enumerate(df.columns, start=1):
        sample = df[col].astype(str).head(200)
        width = max([len(str(col))] + [len(s) for s in sample]) + 2
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(width, cap)
    ws.freeze_panes = "A2"


def to_frames(path: Path) -> list[tuple[str, pd.DataFrame]]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        return [("data", _round(pd.read_csv(path, sep=sep)))]
    if suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [("text", pd.DataFrame({"原文": lines}))]
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            kv = pd.DataFrame(_flatten(raw), columns=["参数", "取值"])
            return [("key_value", kv), ("raw_json", pd.DataFrame({"原始 JSON": [json.dumps(raw, ensure_ascii=False, indent=2)]}))]
        return [("data", _round(pd.DataFrame(raw)))]
    if suffix == ".npz":
        z = np.load(path)
        arrays = [z[k] for k in z.files]
        if arrays and all(a.ndim == 1 for a in arrays) and len({a.shape[0] for a in arrays}) == 1:
            df = pd.DataFrame({k: z[k] for k in z.files})
            df.insert(0, "sample_index", np.arange(df.shape[0]))
            return [("predictions", _round(df))]
        return [(k, _round(pd.DataFrame({k: z[k].ravel()}))) for k in z.files]
    raise ValueError(f"unsupported file type: {path.name}")


def convert(check_only: bool = False) -> int:
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d")
    files = sorted(p for p in TABLES.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".json", ".npz", ".txt"})
    failures = []
    for path in files:
        try:
            frames = to_frames(path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{path.name}: {exc}")
            continue
        if check_only:
            continue
        target = OUT / f"{path.stem}.xlsx"
        main = frames[0][1]
        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            info = _info_sheet(path.name, len(main), len(main.columns), stamp)
            info.to_excel(writer, sheet_name="说明", index=False)
            _autofit(writer, info, "说明", cap=60)
            for sheet, df in frames:
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
                _autofit(writer, df, sheet[:31])
    if failures:
        print("FAILED:")
        for f in failures:
            print("  " + f)
    print(f"tables: {len(files)}  excel written: {len(files) - len(failures)}  failed: {len(failures)}")
    print(f"output: {OUT}")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only parse the tables, do not write .xlsx")
    raise SystemExit(convert(check_only=ap.parse_args().check))