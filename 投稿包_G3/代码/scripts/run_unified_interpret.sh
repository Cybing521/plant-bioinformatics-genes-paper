#!/usr/bin/env bash
# Unified 50-bp / 25-step mutagenesis and exact-allele variant scoring.
# Requires local checkpoints in results/models/ (not in git).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
"$PY" src/interpret/cnn_region_mutagenesis.py --window 50 --step 25
"$PY" src/interpret/agront_region_importance.py --window 50 --step 25
"$PY" src/variants/in_silico.py
"$PY" src/variants/in_silico_agront.py --allele-mode exact
"$PY" src/analyze_variant_regions.py --scores results/variants/scores.tsv --out-dir results --fig-dir figures
"$PY" src/analyze_variant_regions.py --scores results/variants/agront_scores.tsv --tag agront_ --abs-col abs_delta --out-dir results --fig-dir figures
