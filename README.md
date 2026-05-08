# IWSLT 2026 Metrics Shared Task — Submission

Quality Estimation for Speech Translation (reference-free).

## Quick Start (GPU VM)

```bash
git pull
source env.sh
bash scripts/run.sh
```

## Pipeline

| Script | What it does | Time |
|--------|-------------|------|
| `01_download_data.py` | Download train/dev/test from HuggingFace → JSONL | ~2 min |
| `02_finetune_pairwise.py` | Fine-tune CometKiwi-23-XXL with pairwise ranking loss | ~2-3 hrs |
| `03_score_test.py` | Score test (48K) + dev with best model | ~30 min |
| `04_generate_submission.py` | Generate per-LP submission files + validate | ~1 min |

## Method

**Primary**: CometKiwi-23-XXL (10.7B params) fine-tuned with pairwise ranking loss.

The evaluation metric is per-document Kendall tau-b, so we directly optimize pairwise
ordering within documents using margin-based ranking loss combined with MSE calibration.

**Contrastive**: Pretrained CometKiwi-23-XXL (no fine-tuning).

## Submission Format

One file per language pair. One score per line (bare number). Same row order as HF dataset.

```
submission/primary_ende.txt     — 24,016 scores
submission/primary_enzh.txt     — 24,028 scores
submission/contrastive1_ende.txt
submission/contrastive1_enzh.txt
```

Email to: maike.zuefle@kit.edu AND vzouhar@ethz.ch

## Evaluation

Per-document Kendall tau-b averaged across documents, computed per LP then averaged.
Organizers' eval: https://github.com/zouharvi/iwslt26-metrics

## Important Dates

- Apr 21-30, 2026: Evaluation period
- May 10, 2026: System paper deadline
