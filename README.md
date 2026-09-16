# Arabidopsis sequence-to-expression (G3)

G3: Genes, Genomes, Genetics Investigation manuscript comparing a DeepSEA-style 1D-CNN with AgroNT-1B LoRA on Plants Genomic Benchmark Arabidopsis tissue-averaged expression labels.

The submission pack is `投稿包_G3/`. Details: `投稿包_G3/投稿说明.md`.

| Item | Path |
|---|---|
| Review PDF | `投稿包_G3/论文/g3_paper.pdf` |
| Official `gsag3jnl` source | `投稿包_G3/论文/g3_paper.tex` |
| Figures 1–4, S1 | `投稿包_G3/图件/` |
| Processed tables (File S1) | `投稿包_G3/结果表/` |
| Code | `投稿包_G3/代码/` |

PGB FASTA, TAIR10, and **fine-tuned model weights are not stored here**. Download commands are in `投稿包_G3/代码/README.md`. Public code lives at https://github.com/Cybing521/plant-bioinformatics-genes-paper. Checkpoints stay local (`results/models/`, gitignored) and are reproduced by retraining from the public AgroNT-1B backbone.
