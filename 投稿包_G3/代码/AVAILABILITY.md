# Data and code availability (author inventory)

Public repository: https://github.com/Cybing521/plant-bioinformatics-genes-paper

This file is the mapping used in the manuscript. No DOI is claimed until one is minted.

| Artifact | Role | Access |
|---|---|---|
| `投稿包_G3/代码/` | training, evaluation, plotting | public git |
| `投稿包_G3/结果表/` | File S1 processed scores | public git |
| `投稿包_G3/图件/` | Figures 1–4, S1 | public git |
| `download_pgb.py`, `download_tair10.py` | third-party primary data | scripts only; files not stored |
| `InstaDeepAI/agro-nucleotide-transformer-1b` | pretrained backbone | Hugging Face |
| `results/models/*.pt` | fine-tuned CNN / LoRA | **local only; gitignored** |
| Hugging Face / PEFT cache | downloaded backbone copies | **local only; gitignored** |

Reproducibility means: clone, install, download PGB, train. It does not mean downloading our fine-tuned weights.

Manuscript wording must stay: “code and processed tables are public; fine-tuned checkpoints are reproduced by retraining, not deposited.”
