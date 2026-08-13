#!/usr/bin/env bash
# =====================================================================
# run_full.sh — 完整流水线（shengxin_01 后台运行）
# 流程: 数据集构建 → CNN(PGB) → CNN(自建) → AgroNT微调 → 统一评估 → 可解释性
# 用法:  setsid nohup bash scripts/run_full.sh < /dev/null > /dev/null 2>&1 &
# 日志:  results/run_full.log
# =====================================================================
set -u
cd /root/autodl-tmp/plant_bioinfo/implementation
PY=/root/autodl-tmp/envs/plantdl/bin/python
export TMPDIR=/root/autodl-tmp/tmp
export HF_ENDPOINT=https://hf-mirror.com        # 国内镜像（AgroNT 权重下载）
export PIP_CACHE_DIR=/root/autodl-tmp/pipcache
LOG=results/run_full.log
mkdir -p results figures

echo "=== RUN START $(date) ===" > "$LOG"

echo ">> [1/6] build_dataset (自建 npz 数据集)" >> "$LOG"
$PY data/build_dataset.py >> "$LOG" 2>&1
echo ">> build_dataset exit=$?" >> "$LOG"

echo ">> [2/6] CNN on PGB (30 epochs)" >> "$LOG"
$PY src/train.py --data pgb --task binary >> "$LOG" 2>&1
echo ">> cnn_pgb exit=$?" >> "$LOG"

echo ">> [3/6] CNN on npz (30 epochs)" >> "$LOG"
$PY src/train.py --data npz --task binary >> "$LOG" 2>&1
echo ">> cnn_npz exit=$?" >> "$LOG"

echo ">> [4/6] AgroNT-1B LoRA 微调 (10 epochs, 首次下载权重~4GB)" >> "$LOG"
$PY src/train_agront.py --data pgb --task binary >> "$LOG" 2>&1
echo ">> agront exit=$?" >> "$LOG"

echo ">> [5/6] 统一基准评估 (图3)" >> "$LOG"
$PY src/evaluate.py --data pgb --task binary --models cnn agront >> "$LOG" 2>&1
echo ">> evaluate exit=$?" >> "$LOG"

echo ">> [6/6] 可解释性 (图4)" >> "$LOG"
$PY src/interpret/attribution.py >> "$LOG" 2>&1
echo ">> attribution exit=$?" >> "$LOG"
$PY src/interpret/region_importance.py >> "$LOG" 2>&1
echo ">> region_importance exit=$?" >> "$LOG"
$PY src/interpret/run_modisco.py >> "$LOG" 2>&1
echo ">> modisco exit=$?" >> "$LOG"

echo "=== RUN DONE $(date) ===" >> "$LOG"
