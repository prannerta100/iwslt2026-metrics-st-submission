"""
Master pipeline: IWSLT 2026 Metrics Shared Task.

End-to-end: from data download → training → test scoring → final submission files.

  PHASE 1 — Data: Download train/dev from HF, extract text columns
  PHASE 2 — Score DEV: Run all QE metrics on dev set (5.5K samples)
  PHASE 3 — Fine-tune: Train CometKiwi models on train set, score dev
  PHASE 4 — Score TRAIN: Run all QE metrics on train set (33K samples)
  PHASE 5 — Ensemble: Train LightGBM on scored train, evaluate on dev
  PHASE 6 — Score TEST: Load test set from HF, score with all metrics
  PHASE 7 — Submission: Apply LightGBM to test, generate per-LP submission files

Submission format (from organizers):
  - One file per language pair (en-de, en-zh)
  - One score per line, same order as rows appear in the HF test dataset for that LP
  - Score is a single number (parseable by json.loads)

Run on GPU VM:
  poetry run python scripts/run_all.py                     # Full pipeline from scratch
  poetry run python scripts/run_all.py --phase 6           # Resume at test scoring
  poetry run python scripts/run_all.py --skip-download     # Skip data download
  poetry run python scripts/run_all.py --skip-metricx      # Skip MetricX (slow)
"""

import os
import sys
import time
import subprocess
import argparse

parser = argparse.ArgumentParser(
    description="IWSLT 2026 Metrics — full pipeline from scratch to submission"
)
parser.add_argument("--phase", type=int, default=1,
                    help="Start from this phase (1-7). Use to resume after failure.")
parser.add_argument("--skip-download", action="store_true",
                    help="Skip Phase 1 (data download)")
parser.add_argument("--skip-xcomet", action="store_true",
                    help="Skip xCOMET-XL inference")
parser.add_argument("--skip-blaser", action="store_true",
                    help="Skip BLASER-2 inference")
parser.add_argument("--skip-metricx", action="store_true",
                    help="Skip MetricX-24 inference")
parser.add_argument("--skip-cometkiwi23xxl", action="store_true",
                    help="Skip CometKiwi-23-XXL inference")
parser.add_argument("--skip-speech", action="store_true",
                    help="Skip Whisper speech features (requires audio files)")
parser.add_argument("--skip-finetune", action="store_true",
                    help="Skip CometKiwi fine-tuning")
parser.add_argument("--batch-size", type=int, default=128,
                    help="Default batch size for COMET models")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_step(name, cmd, critical=True):
    """Run a pipeline step."""
    print(f"\n{'─' * 70}")
    print(f"  {name}")
    print(f"{'─' * 70}")
    start = time.time()

    result = subprocess.run(cmd, shell=True)

    elapsed = time.time() - start
    if result.returncode != 0:
        if critical:
            print(f"\n  FAILED: {name} (exit code {result.returncode}) — aborting")
            sys.exit(1)
        else:
            print(f"\n  FAILED: {name} (non-critical, continuing)")
    else:
        print(f"\n  OK: {name} — {elapsed:.0f}s")

    return result.returncode == 0


def require_file(path, hint=""):
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        if hint:
            print(f"  Hint: {hint}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
pipeline_start = time.time()

print("=" * 70)
print("  IWSLT 2026 METRICS — FULL PIPELINE (PHASES 1-7)")
print("=" * 70)


# =========================================================================
# PHASE 1: Data Download & Preparation
# =========================================================================
if args.phase <= 1 and not args.skip_download:
    print("\n" + "=" * 70)
    print("  PHASE 1: DATA DOWNLOAD & PREPARATION")
    print("=" * 70)

    run_step(
        "Download HF dataset, extract text columns",
        "poetry run python scripts/01d_download_and_explore.py",
        critical=True,
    )

require_file("outputs/dev_text.parquet", "Run without --skip-download")
require_file("outputs/train_text.parquet", "Run without --skip-download")


# =========================================================================
# PHASE 2: Score DEV set with all base metrics
# =========================================================================
if args.phase <= 2:
    print("\n" + "=" * 70)
    print("  PHASE 2: SCORE DEV SET WITH BASE METRICS")
    print("=" * 70)

    run_step(
        "CometKiwi-22 baseline (creates dev_with_predictions.parquet)",
        "poetry run python scripts/02_cometkiwi_baseline.py",
        critical=True,
    )

    if not args.skip_xcomet:
        run_step(
            "xCOMET-XL inference (dev)",
            f"poetry run python scripts/05_xcomet_inference.py --dev-only --batch-size {args.batch_size}",
            critical=False,
        )

    if not args.skip_blaser:
        run_step(
            "BLASER-2 QE inference (dev, text-text)",
            "poetry run python scripts/06_blaser_inference.py --text-only --batch-size 256",
            critical=False,
        )

    if not args.skip_metricx:
        run_step(
            "MetricX-24 repo setup",
            "bash scripts/setup_metricx.sh",
            critical=False,
        )
        run_step(
            "MetricX-24-Hybrid-XXL inference (dev)",
            "PYTHONPATH=/tmp/metricx:$PYTHONPATH poetry run python scripts/09_metricx_inference.py --batch-size 8",
            critical=False,
        )

    if not args.skip_cometkiwi23xxl:
        run_step(
            "CometKiwi-23-XXL inference (dev)",
            f"poetry run python scripts/10_cometkiwi23xxl_inference.py --batch-size 32",
            critical=False,
        )

    require_file("outputs/dev_with_predictions.parquet")


# =========================================================================
# PHASE 3: Fine-tune CometKiwi on TRAIN set, score DEV
# =========================================================================
if args.phase <= 3 and not args.skip_finetune:
    print("\n" + "=" * 70)
    print("  PHASE 3: FINE-TUNE COMETKIWI ON TRAIN SET")
    print("=" * 70)

    run_step(
        "CometKiwi-22 fine-tune (MSE loss)",
        "poetry run python scripts/03_finetune_cometkiwi.py",
        critical=False,
    )

    run_step(
        "CometKiwi-22 fine-tune (pairwise ranking loss)",
        "poetry run python scripts/03b_finetune_pairwise.py --epochs 10 --batch-size 32",
        critical=False,
    )

    if not args.skip_cometkiwi23xxl:
        run_step(
            "CometKiwi-23-XXL fine-tune (pairwise ranking, 10.7B params)",
            "poetry run python scripts/12_finetune_cometkiwi23xxl.py --epochs 5 --batch-size 8 --grad-accum 4",
            critical=False,
        )


# =========================================================================
# PHASE 4: Score TRAIN set with all metrics
# =========================================================================
if args.phase <= 4:
    print("\n" + "=" * 70)
    print("  PHASE 4: SCORE TRAIN SET WITH ALL METRICS (33K samples)")
    print("=" * 70)

    skip_flags = []
    if args.skip_xcomet:
        skip_flags.append("--skip-xcomet")
    if args.skip_blaser:
        skip_flags.append("--skip-blaser")
    if args.skip_metricx:
        skip_flags.append("--skip-metricx")
    if args.skip_cometkiwi23xxl:
        skip_flags.append("--skip-cometkiwi23xxl")
    skip_str = " ".join(skip_flags)

    run_step(
        "Score train set with all metrics (33K samples)",
        f"PYTHONPATH=/tmp/metricx:$PYTHONPATH poetry run python scripts/11_score_train.py --batch-size {args.batch_size} {skip_str}",
        critical=True,
    )

    require_file("outputs/train_scored.parquet")


# =========================================================================
# PHASE 5: Ensemble (train on scored TRAIN, evaluate on DEV)
# =========================================================================
if args.phase <= 5:
    print("\n" + "=" * 70)
    print("  PHASE 5: TRAIN ENSEMBLE ON SCORED TRAIN, EVALUATE ON DEV")
    print("=" * 70)

    ensemble_cmd = "poetry run python scripts/04b_ensemble_advanced.py"
    if os.path.exists("outputs/train_scored.parquet"):
        ensemble_cmd += " --train-data outputs/train_scored.parquet"
    run_step(
        "Advanced ensemble (LightGBM + calibration + weight optimization)",
        ensemble_cmd,
        critical=True,
    )


# =========================================================================
# PHASE 6: Score TEST set with all metrics
# =========================================================================
if args.phase <= 6:
    print("\n" + "=" * 70)
    print("  PHASE 6: SCORE TEST SET WITH ALL METRICS (48K samples)")
    print("  Dataset: maikezu/iwslt2026-metrics-shared-test")
    print("=" * 70)

    skip_flags = []
    if args.skip_xcomet:
        skip_flags.append("--skip-xcomet")
    if args.skip_blaser:
        skip_flags.append("--skip-blaser")
    if args.skip_metricx:
        skip_flags.append("--skip-metricx")
    if args.skip_cometkiwi23xxl:
        skip_flags.append("--skip-cometkiwi23xxl")
    skip_str = " ".join(skip_flags)

    run_step(
        "Score test set (48K samples) with all available metrics",
        f"PYTHONPATH=/tmp/metricx:$PYTHONPATH poetry run python scripts/13_submit_test.py --batch-size {args.batch_size} {skip_str}",
        critical=True,
    )

    require_file("submission/test_predictions.parquet",
                 "13_submit_test.py should have created this")


# =========================================================================
# PHASE 7: Generate final submission (LightGBM + per-LP files)
# =========================================================================
if args.phase <= 7:
    print("\n" + "=" * 70)
    print("  PHASE 7: GENERATE FINAL SUBMISSION FILES")
    print("  Trains LightGBM on scored train, predicts on scored test,")
    print("  outputs one score file per language pair.")
    print("=" * 70)

    run_step(
        "Train LightGBM on train, predict on test, generate per-LP submission",
        "poetry run python scripts/15_final_submission.py",
        critical=True,
    )

    # Verify submission files exist
    for lp in ["ende", "enzh"]:
        require_file(f"submission/scores_{lp}.txt",
                     f"15_final_submission.py should have created submission/scores_{lp}.txt")


# =========================================================================
# Summary
# =========================================================================
total_time = time.time() - pipeline_start
print("\n" + "=" * 70)
print(f"  PIPELINE COMPLETE — {total_time/60:.1f} minutes")
print("=" * 70)

print("\nGenerated files:")
if os.path.isdir("outputs"):
    for f in sorted(os.listdir("outputs")):
        if f.endswith((".parquet", ".json", ".npy")):
            size = os.path.getsize(os.path.join("outputs", f))
            print(f"  outputs/{f:40s} {size/1024:>8.1f} KB")

if os.path.isdir("submission"):
    print("\nSUBMISSION FILES (send these to organizers):")
    for f in sorted(os.listdir("submission")):
        if f.endswith(".txt"):
            size = os.path.getsize(os.path.join("submission", f))
            n_lines = sum(1 for _ in open(os.path.join("submission", f)))
            print(f"  submission/{f:35s} {n_lines:>6} scores, {size/1024:>6.1f} KB")

print("\nTo submit:")
print("  1. submission/scores_ende.txt  →  en-de language pair")
print("  2. submission/scores_enzh.txt  →  en-zh language pair")
