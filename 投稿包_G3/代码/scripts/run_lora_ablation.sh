#!/usr/bin/env bash
# LoRA rank ablation on the locked PGB binary task. Weights are local-only.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
mkdir -p results/models
for r in 4 8 16 32 64; do
  echo "=== AgroNT LoRA r=$r ==="
  "$PY" src/train_agront.py --data pgb --task binary --lora-r "$r" --lora-alpha "$((2 * r))"
done
