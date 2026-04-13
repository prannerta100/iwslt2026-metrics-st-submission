# IWSLT 2026 Metrics Shared Task — Full Pipeline Flowchart

## How to Run (End to End from Scratch)

```bash
# FULL PIPELINE (everything from scratch, ~4-6 hours on Blackwell 96GB)
poetry run python scripts/run_all.py

# FULL PIPELINE (skip 71GB download if data already exists)
poetry run python scripts/run_all.py --skip-download

# RESUME FROM A SPECIFIC PHASE (e.g., after Phase 3 completes)
poetry run python scripts/run_all.py --phase 4 --skip-download

# GENERATE SUBMISSION (when test data is released)
poetry run python scripts/run_all.py --phase 6 --test-data path/to/test.parquet

# SKIP SPECIFIC MODELS (if they fail or take too long)
poetry run python scripts/run_all.py --skip-download --skip-metricx --skip-blaser
```

## Task

Predict human quality scores (0-100) for speech translation outputs.
Evaluated on **Kendall's Tau** (segment-level) and **Soft Pairwise Accuracy** (system-level).
Test language pairs: **en-de**, **en-zh**.

---

## Data Sources

```
HuggingFace Dataset (71 GB)
├── train split: 33,721 samples, 1,816 docs, 17 lang pairs, 90+ MT systems
│   Domains: iwslt23:ACL, wmt24, wmt25
│   Gold scores: 0-100 (mean=72.3, std=25.9)
│
├── dev split:   5,556 samples, 880 docs, 2 lang pairs (en-de, en-zh), 4 MT systems
│   Domain: iwslt23:ACL
│   Gold scores: 0-100 (mean=72.3, std=26.5)
│   877/880 docs contain BOTH de+zh translations (mixed-language ranking)
│
└── train_synthetic: Synthetic translation pairs (en-de only)
    Used by fine-tuning scripts 03 and 03b to augment training data
```

---

## Models Used (8 total signals)

There are 5 OFF-THE-SHELF pretrained models (no fine-tuning, used as-is):

| # | Model | Script | HuggingFace ID | Architecture | Output Column |
|---|-------|--------|----------------|-------------|---------------|
| 1 | CometKiwi-22 | 02 | Unbabel/wmt22-cometkiwi-da | XLM-R encoder + regression head | `cometkiwi22_score` |
| 2 | xCOMET-XL | 05 | Unbabel/XCOMET-XL | XLM-R + error span detection | `xcomet_score` |
| 3 | BLASER-2.0-QE | 06 | SONAR text encoder + BLASER QE head | SONAR embeddings | `blaser_score`, `sonar_cosine` |
| 4 | MetricX-24-Hybrid-XXL | 09 | google/metricx-24-hybrid-xxl-v2p6 | mT5-XXL encoder-decoder | `metricx_score` |
| 5 | CometKiwi-23-XXL | 10 | Unbabel/wmt23-cometkiwi-da-xxl | XLM-R XXL + regression head | `cometkiwi23xxl_score` |

And 2 FINE-TUNED models (both starting from CometKiwi-22 as base):

| # | Model | Script | Base Model | Loss | Training Data | Output Column |
|---|-------|--------|-----------|------|--------------|---------------|
| 6 | CometKiwi-22 fine-tuned (MSE) | 03 | Unbabel/wmt22-cometkiwi-da | MSE on gold score | train_text (en-de, en-zh rows only) + synthetic (en-de) | `finetuned_score` |
| 7 | CometKiwi-22 fine-tuned (Pairwise) | 03b | Unbabel/wmt22-cometkiwi-da | 0.7 * margin ranking + 0.3 * MSE | train_text (en-de, en-zh) + synthetic (en-de, en-zh) as 50K within-doc pairs | `pairwise_score` |

**Key facts about fine-tuning:**
- Only CometKiwi-22 is fine-tuned. xCOMET, BLASER, MetricX, CometKiwi-23-XXL are NOT fine-tuned.
- Both scripts train ONE multilingual model (not separate de/zh models).
- Synthetic data (`train_synthetic_text.parquet`) is used to augment training in both scripts.
- The pairwise script creates within-document translation pairs where one is better than the other, training the model to rank.

---

## Pipeline Phases

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │                                                                              │
 │  PHASE 1: DATA DOWNLOAD                                                     │
 │  Script: 01d_download_and_explore.py                                        │
 │                                                                              │
 │  HuggingFace dataset (71 GB)                                                │
 │       │                                                                      │
 │       ├──→ outputs/train_text.parquet  (33,721 rows: src_text, tgt_text,    │
 │       │                                 src_lang, tgt_lang, doc_id, score)   │
 │       │                                                                      │
 │       ├──→ outputs/dev_text.parquet    (5,556 rows: same columns)           │
 │       │                                                                      │
 │       └──→ outputs/train_synthetic_text.parquet (synthetic en-de pairs)     │
 │                                                                              │
 └──────────────────────────────────────────────────────────────────────────────┘
                    │                              │
                    │                              │
                    ▼                              ▼
 ┌─────────────────────────────────┐  ┌─────────────────────────────────────────┐
 │                                 │  │                                         │
 │  PHASE 2: SCORE DEV SET         │  │  (dev_text.parquet used here)           │
 │                                 │  │                                         │
 │  Input: dev_text.parquet        │  └─────────────────────────────────────────┘
 │  Output: dev_with_predictions   │
 │          .parquet               │
 │                                 │
 │  5 off-the-shelf pretrained     │
 │  metrics. NO fine-tuning here.  │
 │  Each scores dev independently, │
 │  results merge into one file:   │
 │                                 │
 │  ┌─ 02_cometkiwi_baseline.py ──────────────────────────────────────────┐
 │  │  Model: Unbabel/wmt22-cometkiwi-da (pretrained, NOT fine-tuned)    │
 │  │  What:  Score each (src_text, tgt_text) pair                       │
 │  │  → cometkiwi22_score                                               │
 │  │  Creates dev_with_predictions.parquet (first script to write it)   │
 │  └────────────────────────────────────────────────────────────────────┘
 │                                 │
 │  ┌─ 05_xcomet_inference.py ────────────────────────────────────────────┐
 │  │  Model: Unbabel/XCOMET-XL (pretrained, NOT fine-tuned)             │
 │  │  What:  Score each (src_text, tgt_text) pair                       │
 │  │  → xcomet_score                                                    │
 │  │  Merges into dev_with_predictions.parquet                          │
 │  └────────────────────────────────────────────────────────────────────┘
 │                                 │
 │  ┌─ 06_blaser_inference.py ────────────────────────────────────────────┐
 │  │  Models: SONAR text encoder + BLASER-2.0-QE head (pretrained)      │
 │  │  What:  Encode src & tgt into SONAR embeddings, compute QE score   │
 │  │         and cosine similarity between embeddings                   │
 │  │  → blaser_score, sonar_cosine                                      │
 │  │  Merges into dev_with_predictions.parquet                          │
 │  └────────────────────────────────────────────────────────────────────┘
 │                                 │
 │  ┌─ 09_metricx_inference.py ───────────────────────────────────────────┐
 │  │  Model: google/metricx-24-hybrid-xxl-v2p6 (pretrained)             │
 │  │  What:  mT5-XXL encoder-decoder. Input "source: ... candidate: ..."│
 │  │         Extract logit at vocab index 250089 = MQM error score      │
 │  │         Quality = 25 - error                                       │
 │  │  → metricx_score                                                   │
 │  │  Merges into dev_with_predictions.parquet                          │
 │  └────────────────────────────────────────────────────────────────────┘
 │                                 │
 │  ┌─ 10_cometkiwi23xxl_inference.py ────────────────────────────────────┐
 │  │  Model: Unbabel/wmt23-cometkiwi-da-xxl (pretrained)                │
 │  │  What:  Same as CometKiwi-22 but XXL-sized XLM-R encoder          │
 │  │  → cometkiwi23xxl_score                                            │
 │  │  Merges into dev_with_predictions.parquet                          │
 │  └────────────────────────────────────────────────────────────────────┘
 │                                 │
 │  Result: dev_with_predictions.parquet now has 6 columns:              │
 │    cometkiwi22_score, xcomet_score, blaser_score, sonar_cosine,       │
 │    metricx_score, cometkiwi23xxl_score                                │
 │                                 │
 └─────────────────────────────────┘
                    │
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │  PHASE 3: FINE-TUNE COMET MODELS ON TRAIN, SCORE DEV                        │
 │                                                                             │
 │  Three fine-tuning runs on two base models:                                 │
 │    - CometKiwi-22 (550M) fine-tuned with MSE loss (script 03)              │
 │    - CometKiwi-22 (550M) fine-tuned with pairwise ranking (script 03b)     │
 │    - CometKiwi-23-XXL (10.7B) fine-tuned with pairwise ranking (script 12) │
 │  All train ONE multilingual model (NOT separate models for de and zh).      │
 │                                                                             │
 │  ┌─ 03_finetune_cometkiwi.py ─────────────────────────────────────────┐    │
 │  │                                                                    │    │
 │  │  WHAT:   Fine-tune CometKiwi-22 with MSE loss (pointwise)         │    │
 │  │                                                                    │    │
 │  │  BASE:   Unbabel/wmt22-cometkiwi-da                               │    │
 │  │                                                                    │    │
 │  │  TRAIN DATA:                                                       │    │
 │  │    - train_text.parquet FILTERED to en→de and en→zh rows only      │    │
 │  │    - train_synthetic_text.parquet FILTERED to en→de only           │    │
 │  │    - Combined into data/train_focused.csv                          │    │
 │  │    - Format: (src, mt, score) where score is gold / 100            │    │
 │  │                                                                    │    │
 │  │  LOSS:   MSE between predicted score and gold score                │    │
 │  │                                                                    │    │
 │  │  TRAINING:                                                         │    │
 │  │    - PyTorch Lightning Trainer.fit() (COMET's built-in loop)       │    │
 │  │    - 10 epochs, batch_size=64, head_lr=1.5e-5, enc_lr=1e-6        │    │
 │  │    - Encoder frozen for first 30% of epoch 1, then unfrozen        │    │
 │  │    - Early stopping: patience=3 on val_kendall                     │    │
 │  │                                                                    │    │
 │  │  OUTPUT MODEL: models/cometkiwi_finetuned/best-*.ckpt             │    │
 │  │                                                                    │    │
 │  │  AFTER TRAINING: Scores dev → finetuned_score                     │    │
 │  │  Merges into dev_with_predictions.parquet                          │    │
 │  └────────────────────────────────────────────────────────────────────┘    │
 │                                                                             │
 │  ┌─ 03b_finetune_pairwise.py ─────────────────────────────────────────┐   │
 │  │                                                                    │    │
 │  │  WHAT:   Fine-tune CometKiwi-22 with pairwise ranking loss        │    │
 │  │                                                                    │    │
 │  │  BASE:   Unbabel/wmt22-cometkiwi-da (fresh copy, NOT from 03)     │    │
 │  │                                                                    │    │
 │  │  TRAIN DATA:                                                       │    │
 │  │    - train_text.parquet FILTERED to en→de and en→zh rows only      │    │
 │  │    - train_synthetic_text.parquet FILTERED to en→de and en→zh      │    │
 │  │    - Creates WITHIN-DOCUMENT PAIRS: for each doc, all (i,j) where  │    │
 │  │      translation i has higher gold score than translation j        │    │
 │  │    - Only pairs with score difference > 1.0                        │    │
 │  │    - Capped at 50,000 pairs (random subsample)                     │    │
 │  │    - Each pair: (src, mt_better, mt_worse, score_better,           │    │
 │  │                   score_worse, margin)                              │    │
 │  │                                                                    │    │
 │  │  LOSS:   0.7 * adaptive_margin_ranking + 0.3 * MSE                │    │
 │  │    - Ranking: hinge loss, model must score better > worse          │    │
 │  │    - MSE: calibration, predicted scores match gold                 │    │
 │  │                                                                    │    │
 │  │  TRAINING:                                                         │    │
 │  │    - Custom PyTorch loop (NOT Lightning)                           │    │
 │  │    - 10 epochs, batch_size=32 pairs, head_lr=1e-5, enc_lr=5e-7    │    │
 │  │    - AdamW, weight_decay=0.01, grad_clip=1.0                      │    │
 │  │    - Linear warmup 10% of steps, then linear decay to 5%          │    │
 │  │    - Encoder frozen for first 30% of epoch 1                       │    │
 │  │    - Early stopping: patience=3 on per-source Kendall Tau          │    │
 │  │                                                                    │    │
 │  │  OUTPUT MODEL: models/cometkiwi_pairwise/best-*.ckpt              │    │
 │  │                                                                    │    │
 │  │  AFTER TRAINING: Scores dev → pairwise_score                      │    │
 │  │  Merges into dev_with_predictions.parquet                          │    │
 │  └────────────────────────────────────────────────────────────────────┘    │
 │                                                                             │
 │  ┌─ 12_finetune_cometkiwi23xxl.py ────────────────────────────────────┐   │
 │  │                                                                    │    │
 │  │  WHAT:   Fine-tune CometKiwi-23-XXL with pairwise ranking loss    │    │
 │  │          10.7B params — much more capacity than CometKiwi-22       │    │
 │  │                                                                    │    │
 │  │  BASE:   Unbabel/wmt23-cometkiwi-da-xxl (XLM-R XXL, 48 layers)   │    │
 │  │                                                                    │    │
 │  │  TRAIN DATA:                                                       │    │
 │  │    - Same as 03b: train_text (en-de, en-zh) + synthetic, as pairs │    │
 │  │    - 50K within-document pairs, score diff > 1.0                   │    │
 │  │                                                                    │    │
 │  │  LOSS:   0.7 * adaptive_margin_ranking + 0.3 * MSE (same as 03b) │    │
 │  │                                                                    │    │
 │  │  TRAINING (adapted for 10.7B model on 102GB VRAM):                │    │
 │  │    - Custom PyTorch loop with bf16 autocast + GradScaler           │    │
 │  │    - Phase 1: Encoder fully frozen, train head only (35.7M params)│    │
 │  │      batch=8 pairs, 5 epochs, head_lr=5e-6                        │    │
 │  │    - Phase 2: Unfreeze top 4 encoder layers (~890M params)         │    │
 │  │      + gradient checkpointing, enc_lr=1e-7                         │    │
 │  │    - Gradient accumulation: 4 steps (effective batch=32)           │    │
 │  │    - Early stopping: patience=3 on per-source Kendall Tau          │    │
 │  │                                                                    │    │
 │  │  OUTPUT MODEL: models/cometkiwi23xxl_pairwise/best-*.ckpt         │    │
 │  │                                                                    │    │
 │  │  AFTER TRAINING: Scores dev → cometkiwi23xxl_finetuned_score      │    │
 │  │  Merges into dev_with_predictions.parquet                          │    │
 │  └────────────────────────────────────────────────────────────────────┘    │
 │                                                                             │
 │  Result: dev_with_predictions.parquet now has 6 + 3 = 9 signal columns:    │
 │    cometkiwi22_score, xcomet_score, blaser_score, sonar_cosine,             │
 │    metricx_score, cometkiwi23xxl_score,                                     │
 │    finetuned_score, pairwise_score, cometkiwi23xxl_finetuned_score          │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │  PHASE 4: SCORE TRAIN SET WITH 5 BASE METRICS                               │
 │  Script: 11_score_train.py                                                  │
 │                                                                             │
 │  Input:  outputs/train_text.parquet (33,721 samples, ALL 17 lang pairs)     │
 │  Output: outputs/train_scored.parquet (33,721 samples x 5 base scores)      │
 │                                                                             │
 │  WHY: The ensemble (Phase 5) was overfitting massively when trained via     │
 │  CV on only 5.5K dev samples (train tau=0.82, CV tau=0.36). Scoring the    │
 │  full 33K train set gives the ensemble enough data to generalize.           │
 │                                                                             │
 │  Runs the 5 OFF-THE-SHELF metrics on ALL 33K train rows:                    │
 │    CometKiwi-22        → cometkiwi22_score                                  │
 │    xCOMET-XL           → xcomet_score                                       │
 │    BLASER-2 QE         → blaser_score, sonar_cosine                         │
 │    MetricX-24-XXL      → metricx_score (25 - error)                         │
 │    CometKiwi-23-XXL    → cometkiwi23xxl_score                               │
 │                                                                             │
 │  WHY NOT finetuned_score and pairwise_score?                                │
 │    Those models were TRAINED on this same train data (Phase 3).             │
 │    Scoring train with a model trained on train = data leakage.              │
 │    They are only used for scoring dev and test.                             │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │  PHASE 5: ENSEMBLE                                                          │
 │  Script: 04b_ensemble_advanced.py                                           │
 │                                                                             │
 │  Train input:  outputs/train_scored.parquet (33K, 5 base scores + gold)     │
 │  Eval input:   outputs/dev_with_predictions.parquet (5.5K, 8 scores + gold) │
 │                                                                             │
 │  The ensemble learns: given metric scores, predict gold score.              │
 │  Trains on 33K train (5 base metrics only — no finetuned/pairwise).         │
 │  Evaluates on 5.5K dev (which has all 8 signals for comparison).            │
 │                                                                             │
 │  Methods tried:                                                             │
 │    1. LightGBM regression (train on 33K, predict dev)                       │
 │       Features: raw scores + cross-signal diffs/products + text length      │
 │       + doc-level mean/std/min/max + deviation from doc mean                │
 │    2. Isotonic calibration per language pair (CV on dev)                     │
 │    3. Stacked Ridge meta-learner (CV on dev)                                │
 │    4. Direct Kendall Tau weight optimization on dev:                         │
 │       10K random Dirichlet samples + Nelder-Mead refinement                 │
 │       Finds optimal linear combination of ALL 8 dev signals                 │
 │                                                                             │
 │  Output:                                                                    │
 │    outputs/dev_ensemble_advanced.parquet (dev predictions from all methods)  │
 │    outputs/ensemble_weights.json (optimized metric weights for submission)   │
 │    models/lgbm_model.txt (trained LightGBM model)                           │
 │                                                                             │
 │  Current best dev Kendall Tau: 0.3642 (Nelder-Mead weighted ensemble)       │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                                                                             │
 │  PHASE 6: GENERATE SUBMISSION                                               │
 │  Script: 08_generate_submission.py                                          │
 │                                                                             │
 │  Input:  test set parquet (provided via --test-data)                         │
 │  Models: All 8 metric models + ensemble weights from Phase 5                │
 │                                                                             │
 │  Steps:                                                                     │
 │    1. Score test with CometKiwi-22 (pretrained)                             │
 │    2. Score test with CometKiwi-22 fine-tuned MSE (from 03)                 │
 │    3. Score test with CometKiwi-22 fine-tuned pairwise (from 03b)           │
 │    4. Score test with xCOMET-XL (pretrained)                                │
 │    5. Score test with CometKiwi-23-XXL (pretrained)                         │
 │    6. Score test with CometKiwi-23-XXL fine-tuned pairwise (from 12)        │
 │    7. Score test with MetricX-24 (pretrained)                               │
 │    8. Score test with BLASER-2 (pretrained)                                 │
 │    9. Apply ensemble weights from outputs/ensemble_weights.json             │
 │   10. Write final_score per sample                                          │
 │                                                                             │
 │  Output:                                                                    │
 │    submission/scores.txt          (one score per line, 6 decimal places)    │
 │    submission/test_predictions.parquet (all individual + ensemble scores)   │
 │    submission/metadata.json       (team info, signals used, score stats)    │
 │                                                                             │
 │  NOTE: Test data released April 21-30, 2026                                 │
 │                                                                             │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
                         ┌─────────────────────────┐
                         │    HuggingFace (71GB)    │
                         └────────────┬────────────┘
                                      │
                         01d_download_and_explore.py
                                      │
              ┌───────────────────────┼────────────────────────┐
              ▼                       ▼                        ▼
   train_text.parquet        dev_text.parquet        train_synthetic_text
   33K rows, 17 LPs          5.5K rows               .parquet
   ALL have gold scores       en-de + en-zh           synthetic en-de pairs
              │               with gold scores
              │                       │
              │          ┌────────────┤
              │          │            │
              │          │   PHASE 2: Score dev with 5 PRETRAINED metrics
              │          │   (NO fine-tuning, just inference)
              │          │            │
              │          │   [02] CometKiwi-22 ──────→ cometkiwi22_score
              │          │   [05] xCOMET-XL ─────────→ xcomet_score
              │          │   [06] BLASER-2 QE ───────→ blaser_score, sonar_cosine
              │          │   [09] MetricX-24 ────────→ metricx_score
              │          │   [10] CometKiwi-23-XXL ──→ cometkiwi23xxl_score
              │          │            │
              │          │            ▼
              │          │   dev_with_predictions.parquet (6 score columns)
              │          │            │
              │          │            │
    ┌─────────┼──────────┘            │
    │         │                       │
    │   PHASE 3: Fine-tune COMET models (3 runs on 2 base models)
    │         │                       │
    │    train_text (en-de, en-zh)    │
    │    + synthetic data             │
    │         │                       │
    │    [03] CometKiwi-22 (550M)     │
    │    + MSE loss ─── score dev ────┤──→ finetuned_score
    │    ONE model, all LPs           │
    │         │                       │
    │    [03b] CometKiwi-22 (550M)    │
    │    + pairwise ranking ── score ─┤──→ pairwise_score
    │    50K pairs                    │
    │         │                       │
    │    [12] CometKiwi-23-XXL (10.7B)│
    │    + pairwise ranking ── score ─┤──→ cometkiwi23xxl_finetuned_score
    │    bf16, grad checkpointing     │
    │    freeze encoder → unfreeze 4  │
    │         │                       │
    │         │                       ▼
    │         │              dev_with_predictions.parquet (now 9 columns)
    │         │                       │
    │   PHASE 4: Score TRAIN          │
    │   with 5 base metrics           │
    │   (NOT finetuned models         │
    │    — that would be leakage)     │
    │         │                       │
    │    [11] CometKiwi-22            │
    │         xCOMET-XL               │
    │         BLASER-2                │
    │         MetricX-24              │
    │         CK-23-XXL              │
    │         │                       │
    │         ▼                       │
    │   train_scored.parquet          │
    │   (33K x 5 base scores)         │
    │         │                       │
    │         └─────────┬─────────────┘
    │                   │
    │         PHASE 5: Ensemble
    │         [04b] Train LightGBM on 33K (5 signals)
    │                Optimize weights on dev (9 signals)
    │                   │
    │                   ├──→ outputs/ensemble_weights.json
    │                   ├──→ models/lgbm_model.txt
    │                   │
    │                   │         ┌─────────────────┐
    │                   │         │  test.parquet    │
    │                   │         │  (April 21-30)   │
    │                   │         └────────┬────────┘
    │                   │                  │
    │                   └────────┬─────────┘
    │                            │
    │              PHASE 6: [08] Generate submission
    │              Score test with ALL 8 models
    │              Apply ensemble weights
    │                            │
    │                            ▼
    │                 submission/scores.txt
    │
    └─ NOTE: Fine-tuned models (03, 03b, 12) score TEST in Phase 6
       but do NOT score train in Phase 4 (circular dependency)
```

---

## Script Reference Table

| Script | Phase | What it does | Input | Output | GPU? |
|--------|-------|-------------|-------|--------|------|
| `01d_download_and_explore.py` | 1 | Download HF dataset, extract text columns | HuggingFace | train_text, dev_text, train_synthetic_text .parquet | No |
| `02_cometkiwi_baseline.py` | 2 | Run pretrained CometKiwi-22 on dev | dev_text.parquet | dev_with_predictions.parquet (creates it) | Yes |
| `05_xcomet_inference.py` | 2 | Run pretrained xCOMET-XL on dev | dev_text.parquet | merges xcomet_score into dev_with_predictions | Yes |
| `06_blaser_inference.py` | 2 | Run pretrained BLASER-2 on dev (text-text mode) | dev_text.parquet | merges blaser_score + sonar_cosine | Yes |
| `09_metricx_inference.py` | 2 | Run pretrained MetricX-24-XXL on dev | dev_text.parquet | merges metricx_score | Yes |
| `10_cometkiwi23xxl_inference.py` | 2 | Run pretrained CometKiwi-23-XXL on dev | dev_text.parquet | merges cometkiwi23xxl_score | Yes |
| `03_finetune_cometkiwi.py` | 3 | Fine-tune CometKiwi-22 with MSE loss, then score dev | train_text (en-de,zh) + synthetic (en-de) | model ckpt + finetuned_score on dev | Yes |
| `03b_finetune_pairwise.py` | 3 | Fine-tune CometKiwi-22 with pairwise ranking, then score dev | train_text (en-de,zh) + synthetic (en-de,zh) as 50K pairs | model ckpt + pairwise_score on dev | Yes |
| `12_finetune_cometkiwi23xxl.py` | 3 | Fine-tune CometKiwi-23-XXL (10.7B) with pairwise ranking | train_text (en-de,zh) + synthetic as 50K pairs | model ckpt + cometkiwi23xxl_finetuned_score on dev | Yes (40GB+) |
| `11_score_train.py` | 4 | Run 5 pretrained metrics on full 33K train set | train_text.parquet | train_scored.parquet (5 base scores) | Yes |
| `04b_ensemble_advanced.py` | 5 | Train LightGBM on 33K, optimize weights on dev | train_scored + dev_with_predictions | ensemble_weights.json + lgbm_model.txt | No |
| `08_generate_submission.py` | 6 | Score test with all 7 models, apply ensemble | test.parquet + models + weights | submission/scores.txt | Yes |

---

## All 8 Signal Columns in Detail

| Column | Model | Fine-tuned? | Present in train_scored? | Present in dev? | Score range | Notes |
|--------|-------|------------|-------------------------|-----------------|-------------|-------|
| `cometkiwi22_score` | CometKiwi-22 | No (pretrained) | Yes | Yes | ~0.3-0.95 | Baseline QE metric |
| `xcomet_score` | xCOMET-XL | No (pretrained) | Yes | Yes | ~0.3-0.95 | Includes error spans |
| `blaser_score` | BLASER-2.0-QE | No (pretrained) | Yes | Yes | ~1.0-5.0 | SONAR embedding-based |
| `sonar_cosine` | SONAR text encoder | No (pretrained) | Yes | Yes | ~0.5-1.0 | Cosine similarity of src/tgt embeddings |
| `metricx_score` | MetricX-24-Hybrid-XXL | No (pretrained) | Yes | Yes | 0-25 | 25 minus MQM error. mT5-XXL based |
| `cometkiwi23xxl_score` | CometKiwi-23-XXL | No (pretrained) | Yes | Yes | ~0.3-0.95 | Larger XLM-R encoder |
| `finetuned_score` | CometKiwi-22 + MSE | **Yes** (script 03) | **No** (leakage) | Yes | ~0.3-0.95 | Trained on train gold scores |
| `pairwise_score` | CometKiwi-22 + ranking | **Yes** (script 03b) | **No** (leakage) | Yes | ~0.3-0.95 | Trained on within-doc pairs |
| `cometkiwi23xxl_finetuned_score` | CometKiwi-23-XXL + ranking | **Yes** (script 12) | **No** (leakage) | Yes | ~0.3-0.95 | 10.7B params, pairwise ranking |

---

## Known Issues & Bottlenecks

1. **Within-document metric spread**: COMET-family metrics produce near-identical scores within a document (spread_ratio=0.006), making within-doc ranking nearly random. MetricX has 28x more spread (0.172).

2. **Mixed-language documents**: 877/880 dev docs contain BOTH de+zh translations. Ranking across languages with the same metric is inherently harder.

3. **Annotation noise**: 149 identical (doc_id, tgt_text) pairs have different gold scores (mean std=10.3 points). This sets a ceiling on achievable correlation.

4. **Ensemble overfitting**: With only 880 docs in dev, LightGBM overfits severely (train tau=0.82, CV tau=0.36). Phase 4 (scoring 33K train) is designed to fix this.

5. **MetricX requires manual setup**: `bash scripts/setup_metricx.sh` must run before MetricX inference. The `metricx24` package has no PyPI release — it's cloned to `/tmp/metricx` and added to PYTHONPATH.
