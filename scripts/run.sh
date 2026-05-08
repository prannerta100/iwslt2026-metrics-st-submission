#!/bin/bash
# IWSLT 2026 Metrics — Full Pipeline (CometKiwi-22 pairwise)
set -e

echo "============================================================"
echo "  IWSLT 2026 METRICS — CometKiwi-22 Pairwise Pipeline"
echo "  $(date)"
echo "============================================================"

if [ -f env.sh ]; then source env.sh; fi

echo ""
echo ">>> PHASE 1: Download data"
poetry run python scripts/01_download_data.py

echo ""
echo ">>> PHASE 2: Fine-tune CometKiwi-22 (pairwise ranking, ~30 min)"
poetry run python scripts/02_finetune_pairwise.py --epochs 10 --batch-size 32 --patience 3

echo ""
echo ">>> PHASE 3: Score test + dev"
poetry run python scripts/03_score_test.py --batch-size 128

echo ""
echo ">>> PHASE 4: Generate submission files"
poetry run python scripts/04_generate_submission.py

echo ""
echo "============================================================"
echo "  DONE. Submit: submission/primary_ende.txt + primary_enzh.txt"
echo "============================================================"
