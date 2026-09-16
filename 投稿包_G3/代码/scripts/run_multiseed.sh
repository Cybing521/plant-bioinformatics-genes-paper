#!/usr/bin/env bash
# CNN seeds 42-46 and AgroNT seeds 42-46. Checkpoints stay in results/models/ (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
mkdir -p results/models
for seed in 42 43 44 45 46; do
  echo "=== CNN seed $seed ==="
  "$PY" src/train.py --data pgb --task binary --seed "$seed"
done
for seed in 42 43 44 45 46; do
  echo "=== AgroNT seed $seed ==="
  "$PY" src/train_agront.py --data pgb --task binary --seed "$seed"
done
