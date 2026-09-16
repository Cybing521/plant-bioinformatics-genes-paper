#!/usr/bin/env bash
# Official PGB multi-tissue regression (56 outputs). Not the paper's primary binary task.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
mkdir -p results/models
"$PY" src/train.py --data pgb --task multitissue
"$PY" src/train_agront.py --data pgb --task multitissue
