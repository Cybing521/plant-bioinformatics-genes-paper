#!/usr/bin/env bash
# AgroNT 剩余实验：跨物种推理 → 变异打分 → 3 kb / 1.5 kb 居中裁剪重训
# 在 shengxin_01 上: setsid nohup bash scripts/run_agront_remainder.sh < /dev/null > /dev/null 2>&1 &
set -u
cd /root/autodl-tmp/plant_bioinfo/implementation
PY=/root/autodl-tmp/envs/plantdl/bin/python
export TMPDIR=/root/autodl-tmp/tmp
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
LOG=results/agront_remainder.log
mkdir -p results figures results/variants results/models

echo "=== AGRONT REMAINDER START $(date) ===" >> "$LOG"

echo ">> [1/4] cross-species AgroNT" >> "$LOG"
$PY src/cross_species.py --model agront --batch-size 8 \
    --ckpt results/models/agront_pgb_binary.pt >> "$LOG" 2>&1
echo ">> cross-species exit=$?" >> "$LOG"

echo ">> [2/4] AgroNT variant scoring" >> "$LOG"
$PY src/variants/in_silico_agront.py --batch 8 \
    --scores results/variants/scores.tsv \
    --ckpt results/models/agront_pgb_binary.pt >> "$LOG" 2>&1
echo ">> in_silico_agront exit=$?" >> "$LOG"
$PY src/analyze_variant_regions.py \
    --scores results/variants/agront_scores_by_site.tsv \
    --abs-col mean_abs_delta \
    --eqtl results/variants/agront_eqtl_scores.tsv \
    --out-dir results --fig-dir figures --tag agront_ >> "$LOG" 2>&1
echo ">> analyze_variant_regions exit=$?" >> "$LOG"

echo ">> [3/4] AgroNT length ablation 3000bp center" >> "$LOG"
$PY src/train_agront.py --data pgb --task binary --max_seq_len 3000 --crop_mode center \
    --out-ckpt results/models/agront_pgb_3000bp_center_binary.pt >> "$LOG" 2>&1
echo ">> train 3000 exit=$?" >> "$LOG"

echo ">> [4/4] AgroNT length ablation 1500bp center" >> "$LOG"
$PY src/train_agront.py --data pgb --task binary --max_seq_len 1500 --crop_mode center \
    --out-ckpt results/models/agront_pgb_1500bp_center_binary.pt >> "$LOG" 2>&1
echo ">> train 1500 exit=$?" >> "$LOG"

$PY - <<'PY' >> "$LOG" 2>&1
import json, os
import pandas as pd
rows = []
cnn_path = "results/input_length_ablation_center.csv"
if os.path.exists(cnn_path):
    cnn = pd.read_csv(cnn_path)
    if "model" not in cnn.columns:
        cnn["model"] = "cnn"
    rows.append(cnn)
# 6 kb AgroNT from the locked benchmark
rows.append(pd.DataFrame([{
    "ckpt": "results/models/agront_pgb_binary.pt",
    "seq_len_bp": 6000,
    "crop_mode": "center",
    "accuracy": 0.8683127572016461,
    "auroc": 0.9391213513543419,
    "auprc": 0.9346074599131322,
    "model": "agront",
}]))
for n in (3000, 1500):
    p = f"results/models/agront_pgb_{n}bp_center_binary_test.json"
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    rows.append(pd.DataFrame([{
        "ckpt": d.get("ckpt"),
        "seq_len_bp": d.get("seq_len_bp"),
        "crop_mode": d.get("crop_mode", "center"),
        "accuracy": d.get("accuracy"),
        "auroc": d.get("auroc"),
        "auprc": d.get("auprc"),
        "model": "agront",
    }]))
out = pd.concat(rows, ignore_index=True)
out.to_csv("results/input_length_ablation_both.csv", index=False)
print("wrote results/input_length_ablation_both.csv", len(out))
PY

echo "=== AGRONT REMAINDER DONE $(date) ===" >> "$LOG"
