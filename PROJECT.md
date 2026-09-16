# Locked protocol — Arabidopsis AgroNT vs CNN

Target: mid-to-upper methods journal (~6.5/10; *Plant Methods* / *BMC Bioinformatics* / *G3* Investigation).
Competition this paper wins: **quantify LoRA-AgroNT gain over a DeepSEA-style CNN under one locked split, label rule, metric set, and early-stopping rule**. Not “foundation models are inherently stronger”.

## Claims that may appear in the manuscript

1. On the official PGB Arabidopsis gene split and a **tissue-averaged binary** high/low label, LoRA-AgroNT (seed 42) exceeds this CNN (seed 42) by ΔAUROC 0.097 (bootstrap 95% CI 0.086–0.108; McNemar *p* = 6.7×10⁻⁵⁴).
2. CNN three-seed AUROC is 0.844 ± 0.003; AgroNT remains a **single seed** until additional GPU runs finish. The primary table is therefore a locked-protocol contrast, not a mixed-effects multi-seed contest.
3. Tissue-averaged labels agree with single-tissue median labels for a mean of 92.1% of genes (bootstrap 95% CI over 56 tissues to be filled from `locked_protocol_stats.py`). This supports the derived binary task for architecture comparison; it does not license organ-specific conclusions.
4. Regional variant |\Delta| ranks are **architecture-specific exploratory maps**. CNN used mapped ref/alt on a structured-set checkpoint; AgroNT used mean |\Delta| over three non-reference bases with the PGB LoRA checkpoint. They are not a unified causal ranking.
5. Crop PGB tests are **cross-species transfer of Arabidopsis checkpoints**. AgroNT pretraining included rice, maize, tomato, and soybean genomes. This is not a species-unseen test.
6. The 17-variant eQTL overlap is **exploratory** (n = 17, uncorrected *p* = 0.041) and is not a main-text claim.

Out of scope until computed: official 56-tissue PGB regression, AgroNT 3–5 seeds, LoRA-rank ablation, unified allele mutagenesis, calibration/DCA, few-shot crop fine-tuning.

## Public repository policy (author lock)

Public GitHub: `https://github.com/Cybing521/plant-bioinformatics-genes-paper`

| In public repo | Not in public repo |
|---|---|
| Training/eval/plotting code, configs, seeds, environment files | Fine-tuned CNN `*.pt` / LoRA adapters / full AgroNT copies |
| Download scripts for PGB, TAIR10, Ensembl variation | HuggingFace cache, `results/models/` |
| Processed score tables used in figures (File S1) | Any file that lets a reader skip retraining |

Reproducibility route: public AgroNT-1B backbone (`InstaDeepAI/agro-nucleotide-transformer-1b`) + this repo’s scripts, configs, and seeds. Checkpoints are regenerated, not downloaded. Editors/reviewers may request weights privately; they are not a public artifact.
