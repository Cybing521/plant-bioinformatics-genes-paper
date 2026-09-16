#!/usr/bin/env bash
# Remainder 结束后：AgroNT 多种子（43/44）。不覆盖 agront_pgb_binary.pt。
# 2026-09-02：用户确认 AgroNT 保持单种子（42），本脚本已在 seed 43 第 1 个 epoch 中途停止，不再跑 43/44。
# 在 shengxin_01 上: setsid nohup bash scripts/run_agront_followon.sh < /dev/null > /dev/null 2>&1 &
set -u
cd /root/autodl-tmp/plant_bioinfo/implementation
PY=/root/autodl-tmp/envs/plantdl/bin/python
export TMPDIR=/root/autodl-tmp/tmp
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1
LOG=results/agront_followon.log
mkdir -p results/models

echo "=== AGRONT FOLLOWON WAIT $(date) ===" >> "$LOG"
while pgrep -f 'run_agront_remainder.sh' >/dev/null 2>&1 \
   || pgrep -f 'src/train_agront.py' >/dev/null 2>&1; do
  echo "$(date '+%F %T') waiting for remainder/train_agront" >> "$LOG"
  sleep 120
done

echo "=== AGRONT FOLLOWON START $(date) ===" >> "$LOG"
for seed in 43 44; do
  ckpt="results/models/agront_pgb_binary_seed${seed}.pt"
  if [[ -f "$ckpt" ]]; then
    echo ">> skip seed $seed; $ckpt exists" >> "$LOG"
    continue
  fi
  echo ">> seed $seed -> $ckpt" >> "$LOG"
  $PY src/train_agront.py --data pgb --task binary --seed "$seed" \
    --out-ckpt "$ckpt" >> "$LOG" 2>&1
  echo ">> seed $seed exit=$?" >> "$LOG"
done
echo "=== AGRONT FOLLOWON DONE $(date) ===" >> "$LOG"
