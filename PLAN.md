# IWSLT 2026 Metrics — Fresh Pipeline Plan

> **Deadline**: May 7, 2026 AoE
> **Target**: Beat COMET baseline (34.6% segment-level Kendall tau)
> **Previous best**: Pairwise CK-22 fine-tune (35.12%), LLM debate (38.6% on dev only)
> **Hardware**: AWS Blackwell 96GB VRAM

## Strategy

**Primary submission**: CometKiwi-23-XXL (10.7B) fine-tuned with pairwise ranking loss.
- Rationale: CK-23-XXL is the largest COMET model (10.7B vs 580M for CK-22).
  Pairwise ranking loss directly optimizes Kendall tau (our eval metric).
  With 96GB VRAM, we can fine-tune this with LoRA or full with grad accum.

**Contrastive 1**: Pretrained CometKiwi-23-XXL (no fine-tuning) — strong baseline.

**Contrastive 2**: LLM-as-judge (gpt-4.1-mini via Webex proxy) — if time permits.

## Pipeline (5 scripts total)

```
scripts/
├── 01_download_data.py      # HF → JSONL (matches organizers' format exactly)
├── 02_finetune_pairwise.py  # Fine-tune CK-23-XXL with pairwise ranking loss
├── 03_score_test.py         # Score test set with fine-tuned model + pretrained
├── 04_generate_submission.py # Generate per-LP text files
├── run.sh                   # One command to run everything
└── ssl_fix.py               # SSL cert fix for pyenv
```

## Script Details

### 01_download_data.py
- Load `maikezu/iwslt2026-metrics-shared-train-dev` (train + dev splits)
- Load `maikezu/iwslt2026-metrics-shared-test` (test split)
- Write JSONL files matching organizers' format: `data/train.jsonl`, `data/dev.jsonl`, `data/test.jsonl`
- Verify row counts (train: ~33K, dev: ~5.5K, test: ~48K)

### 02_finetune_pairwise.py
- Load CometKiwi-23-XXL (`Unbabel/wmt23-cometkiwi-da-xxl`)
- Create within-doc pairwise training samples from train.jsonl
- Loss: 0.3*MSE + 0.7*margin ranking loss (same approach that got 35.12% with CK-22)
- Differential LR: encoder 1e-6, head 5e-5 (scaled for larger model)
- Gradient accumulation: 8 steps × batch 4 = effective batch 32
- Evaluate per-doc tau on dev after each epoch
- Save best checkpoint to `models/best.ckpt`
- Epochs: 5 (with early stopping patience=2)

### 03_score_test.py
- Load test.jsonl
- Score with:
  1. Fine-tuned CK-23-XXL (primary — best checkpoint)
  2. Pretrained CK-23-XXL (contrastive)
- Save scores to `outputs/test_finetuned.npy` and `outputs/test_pretrained.npy`

### 04_generate_submission.py
- Read test.jsonl to get row order and LP labels
- For each LP (ende, enzh):
  - Filter rows for that LP (preserving HF dataset order)
  - Write `submission/primary_ende.txt`, `submission/primary_enzh.txt` (fine-tuned scores)
  - Write `submission/contrastive1_ende.txt`, `submission/contrastive1_enzh.txt` (pretrained scores)
- Validate: run organizers' eval script on dev to confirm format is correct

### run.sh
```bash
#!/bin/bash
set -e
source env.sh  # SSL fix
poetry run python scripts/01_download_data.py
poetry run python scripts/02_finetune_pairwise.py --epochs 5 --batch-size 4 --grad-accum 8
poetry run python scripts/03_score_test.py --batch-size 32
poetry run python scripts/04_generate_submission.py
echo "Done! Submit: submission/primary_*.txt and submission/contrastive1_*.txt"
```

## Validation

Before submitting, run organizers' eval on dev:
```bash
python3 /tmp/iwslt26-metrics/evaluation -i data/dev.jsonl -m submission/dev_primary.jsonl
```
This confirms our format is correct AND shows our dev tau.

## Key Decisions

1. **Why CK-23-XXL over CK-22?** 10.7B vs 580M params — much stronger representations.
   CK-22 pairwise already beat the baseline; CK-23-XXL should do significantly better.

2. **Why pairwise loss?** Directly optimizes what we're measured on (within-doc ranking).
   MSE optimizes absolute scores which doesn't necessarily improve ranking.

3. **Why not ensemble?** Circularity problem: can't use fine-tuned scores as ensemble features
   on train data (model was trained on it). Single strong model > weak ensemble.

4. **Why not LLM?** Too slow for 48K test samples under deadline pressure.
   Keep as contrastive if time permits.

5. **One file per LP?** Confirmed from organizers' eval script: it reads ONE metric file with
   ALL rows, but evaluates per-LP. However, the website says per-LP files. We generate BOTH
   and verify with the eval script.

## Risk Mitigation

- If CK-23-XXL OOMs during fine-tuning: reduce batch to 2, increase grad accum to 16
- If fine-tuning diverges: fall back to pretrained CK-23-XXL (still better than CK-22 baseline)
- If test dataset format differs: 01_download_data.py prints schema and validates
- Format verification: run eval script on dev predictions before touching test
