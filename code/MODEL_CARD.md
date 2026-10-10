# Model card — Arabidopsis PGB binary benchmark

## Models

| Model | Public artifact | Local-only artifact |
|---|---|---|
| AgroNT-1B backbone | Hugging Face `InstaDeepAI/agro-nucleotide-transformer-1b` | none from this repo |
| AgroNT LoRA adapter + head | retrain with `src/train_agront.py` | `results/models/agront_*.pt` |
| DeepSEA-style 1D-CNN | architecture in `src/models/cnn.py` | `results/models/cnn_*.pt` |

Fine-tuned checkpoints are **not** deposited. They can in principle be retrained from the public backbone, this repository's configs and the reported seeds,
**but note that the deposited scripts have not been verified end-to-end against the reported runs** (see AVAILABILITY.md). Editors or reviewers may request weights privately.

## Locked binary task

- Data: Plants Genomic Benchmark Arabidopsis gene-expression FASTA (official gene split).
- Label: tissue-averaged abundance, thresholded at the **training-set** median (τ = 5.605). Not the official 56-tissue regression.
- Split: 25,731 / 3,401 / 3,402 genes.
- Both models reported for **five matched seeds (42–46)**; seed 42 is the primary paired contrast
  (Table 1, calibration, decision curve). Five-seed AUROC: CNN 0.847 ± 0.004, AgroNT 0.938 ± 0.001.
- LoRA: r = 16, α = 32, targets query/key/value/dense, 3 epochs, early stop on validation AUROC.
- Hardware used for the reported runs: one NVIDIA RTX 4090.

## Known limitations

- Crop PGB tests reuse a backbone pretrained on those species; the crop readout is a consistency check,
  not evidence of cross-species generalisation.
- The AgroNT backbone, the PGB benchmark and the expression matrix all originate in one published study,
  and the backbone's pretraining included the Arabidopsis genome supplying the test sequences.
- The transformer ran at most 3 epochs (compute-capped, not convergence-capped); the reported gap is a lower bound.
- Region comparisons are not adjusted for MAF, GC, distance to TSS or gene expression level.
- Regional variant maps used different allele protocols and different training representations; the two
  maps are reported separately and are not combined. The adapted backbone is insensitive at some sites on
  the representation it was not retrained for (|Δ| < 5e-9 on AT4G26560).
- eQTL overlap (n = 17) is exploratory.

## Licence and third-party weights

Code in this repository is MIT. AgroNT weights follow InstaDeep / Hugging Face terms. PGB FASTA follows the Plants Genomic Benchmark deposit. Do not re-host those third-party weights in forks.
