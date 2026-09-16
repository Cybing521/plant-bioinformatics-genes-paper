# Model card — Arabidopsis PGB binary benchmark

## Models

| Model | Public artifact | Local-only artifact |
|---|---|---|
| AgroNT-1B backbone | Hugging Face `InstaDeepAI/agro-nucleotide-transformer-1b` | none from this repo |
| AgroNT LoRA adapter + head | retrain with `src/train_agront.py` | `results/models/agront_*.pt` |
| DeepSEA-style 1D-CNN | architecture in `src/models/cnn.py` | `results/models/cnn_*.pt` |

Fine-tuned checkpoints are **not** deposited. Anyone can reproduce them from the public backbone, this repository's configs, and the reported seeds. Editors or reviewers may request weights privately.

## Locked binary task

- Data: Plants Genomic Benchmark Arabidopsis gene-expression FASTA (official gene split).
- Label: tissue-averaged abundance, thresholded at the **training-set** median (τ = 5.605). Not the official 56-tissue regression.
- Split: 25,731 / 3,401 / 3,402 genes.
- Primary seed: 42. CNN also reported for seeds 43 and 44.
- LoRA: r = 16, α = 32, targets query/key/value/dense, 3 epochs, early stop on validation AUROC.
- Hardware used for the reported runs: one NVIDIA RTX 4090.

## Known limitations

- AgroNT primary result is one seed.
- Crop PGB tests reuse a backbone pretrained on those species.
- Regional variant maps used different allele protocols unless `scripts/run_unified_interpret.sh` has been run.
- eQTL overlap (n = 17) is exploratory.

## Licence and third-party weights

Code in this repository is MIT. AgroNT weights follow InstaDeep / Hugging Face terms. PGB FASTA follows the Plants Genomic Benchmark deposit. Do not re-host those third-party weights in forks.
