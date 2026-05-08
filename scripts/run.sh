#!/bin/bash
# IWSLT 2026 Metrics Shared Task — Full Pipeline
# Run on GPU VM: bash scripts/run.sh
set -e

echo "============================================================"
echo "  IWSLT 2026 METRICS — FULL PIPELINE"
echo "  $(date)"
echo "============================================================"

# SSL fix for pyenv
if [ -f env.sh ]; then
    source env.sh
fi

# Phase 1: Download data
echo ""
echo ">>> PHASE 1: Download data from HuggingFace"
poetry run python scripts/01_download_data.py

# Phase 2: Fine-tune CK-23-XXL with pairwise ranking loss
echo ""
echo ">>> PHASE 2: Fine-tune CometKiwi-23-XXL (pairwise ranking)"
poetry run python scripts/02_finetune_pairwise.py \
    --epochs 5 \
    --batch-size 8 \
    --grad-accum 4 \
    --patience 2

# Phase 3: Score test + dev with fine-tuned and pretrained models
echo ""
echo ">>> PHASE 3: Score test set"
poetry run python scripts/03_score_test.py --batch-size 32

# Phase 4: Generate submission files
echo ""
echo ">>> PHASE 4: Generate submission files"
poetry run python scripts/04_generate_submission.py

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Submission files: submission/primary_*.txt"
echo "============================================================"
