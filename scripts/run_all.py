"""
Master pipeline: IWSLT 2026 Metrics Shared Task.

Runs the entire pipeline from scratch in logical phases:

  PHASE 1 — Data: Download HF dataset, extract text columns
  PHASE 2 — Score DEV: Run all QE metrics on dev set (5.5K samples)
  PHASE 3 — Fine-tune: Train CometKiwi on train set (33K), score dev
  PHASE 4 — Score TRAIN: Run all QE metrics on train set (33K samples)
  PHASE 5 — Ensemble: Train LightGBM on scored train, evaluate on dev
  PHASE 6 — Submission: Score test set, apply ensemble, write scores.txt

  ┌─────────────────────────────────────────────────────────────────────┐
  │  HF Dataset (71GB)                                                 │
  │       │                                                            │
  │       ▼                                                            │
  │  01d: Extract text ──→ train_text.parquet (33K)                    │
  │                    ──→ dev_text.parquet (5.5K)                     │
  │       │                      │                                     │
  │       │    ┌─────────────────┤                                     │
  │       │    │  PHASE 2: Score DEV with 5 metrics                    │
  │       │    │  02 → cometkiwi22_score                               │
  │       │    │  05 → xcomet_score                                    │
  │       │    │  06 → blaser_score, sonar_cosine                      │
  │       │    │  09 → metricx_score                                   │
  │       │    │  10 → cometkiwi23xxl_score                            │
  │       │    │         │                                             │
  │       │    │  PHASE 3: Fine-tune on TRAIN, score DEV               │
  │       │    │  03  → finetuned_score                                │
  │       │    │  03b → pairwise_score                                 │
  │       │    │         │                                             │
  │       │    │         ▼                                             │
  │       │    │  dev_with_predictions.parquet (5.5K × 8 scores)       │
  │       │    │                                                       │
  │       ▼    │  PHASE 4: Score TRAIN with all metrics                │
  │  11: Score train ──→ train_scored.parquet (33K × 8 scores)         │
  │       │    │                                                       │
  │       │    │  PHASE 5: Ensemble                                    │
  │       │    │  04b: Train LightGBM on 33K train, eval on 5.5K dev  │
  │       │    │         │                                             │
  │       ▼    ▼         ▼                                             │
  │  PHASE 6: 08 → submission/scores.txt                               │
  └─────────────────────────────────────────────────────────────────────┘

Run on GPU VM:
  poetry run python scripts/run_all.py
  poetry run python scripts/run_all.py --skip-download   # if data exists
  poetry run python scripts/run_all.py --phase 4         # resume from phase 4
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
                    help="Start from this phase (1-6). Use to resume after failure.")
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
parser.add_argument("--test-data", type=str, default=None,
                    help="Path to test set for Phase 6 submission generation")
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
        status = "FAILED"
        if critical:
            print(f"\n  ✗ {name} FAILED (exit code {result.returncode}) — aborting")
            sys.exit(1)
        else:
            print(f"\n  ✗ {name} FAILED (non-critical, continuing)")
    else:
        status = f"done in {elapsed:.0f}s"
        print(f"\n  ✓ {name} — {status}")

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
print("  IWSLT 2026 METRICS — FULL PIPELINE")
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

# Verify data exists
require_file("outputs/dev_text.parquet", "Run without --skip-download")
require_file("outputs/train_text.parquet", "Run without --skip-download")


# =========================================================================
# PHASE 2: Score DEV set with all base metrics
# =========================================================================
if args.phase <= 2:
    print("\n" + "=" * 70)
    print("  PHASE 2: SCORE DEV SET WITH BASE METRICS")
    print("  Each metric scores dev_text.parquet independently.")
    print("  Results merge into dev_with_predictions.parquet.")
    print("=" * 70)

    # 2a. CometKiwi-22 (baseline, creates dev_with_predictions.parquet)
    run_step(
        "CometKiwi-22 baseline (creates dev_with_predictions.parquet)",
        f"poetry run python scripts/02_cometkiwi_baseline.py",
        critical=True,
    )

    # 2b. xCOMET-XL
    if not args.skip_xcomet:
        run_step(
            "xCOMET-XL inference (dev)",
            f"poetry run python scripts/05_xcomet_inference.py --dev-only --batch-size {args.batch_size}",
            critical=False,
        )

    # 2c. BLASER-2 QE (text-text mode)
    if not args.skip_blaser:
        run_step(
            "BLASER-2 QE inference (dev, text-text)",
            "poetry run python scripts/06_blaser_inference.py --text-only --batch-size 256",
            critical=False,
        )

    # 2d. MetricX-24-Hybrid-XXL
    if not args.skip_metricx:
        # Setup metricx repo first
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

    # 2e. CometKiwi-23-XXL
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
    print("  Uses 33K train samples to fine-tune CometKiwi-22.")
    print("  Produces finetuned_score and pairwise_score on dev.")
    print("=" * 70)

    # 3a. MSE fine-tuning
    run_step(
        "CometKiwi fine-tune (MSE loss)",
        "poetry run python scripts/03_finetune_cometkiwi.py",
        critical=False,
    )

    # 3b. Pairwise ranking fine-tuning
    run_step(
        "CometKiwi fine-tune (pairwise ranking loss)",
        "poetry run python scripts/03b_finetune_pairwise.py --epochs 10 --batch-size 32",
        critical=False,
    )


# =========================================================================
# PHASE 4: Score TRAIN set with all metrics
# =========================================================================
if args.phase <= 4:
    print("\n" + "=" * 70)
    print("  PHASE 4: SCORE TRAIN SET WITH ALL METRICS")
    print("  Scores 33K train samples so the ensemble can train on them.")
    print("  This is critical to reduce overfitting (0.82 → 0.36 gap).")
    print("=" * 70)

    # Build skip flags from args
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
    print("  PHASE 5: ENSEMBLE")
    print("  Train LightGBM on 33K scored train samples.")
    print("  Evaluate on 5.5K dev samples.")
    print("=" * 70)

    # 5a. Advanced ensemble (train on scored train set if available)
    ensemble_cmd = "poetry run python scripts/04b_ensemble_advanced.py"
    if os.path.exists("outputs/train_scored.parquet"):
        ensemble_cmd += " --train-data outputs/train_scored.parquet"
    run_step(
        "Advanced ensemble (LightGBM + calibration + stacking)",
        ensemble_cmd,
        critical=True,
    )


# =========================================================================
# PHASE 6: Generate submission
# =========================================================================
if args.phase <= 6 and args.test_data:
    print("\n" + "=" * 70)
    print("  PHASE 6: GENERATE SUBMISSION")
    print(f"  Scoring test set: {args.test_data}")
    print("=" * 70)

    require_file(args.test_data, "Provide --test-data path")

    run_step(
        "Generate submission files",
        f"PYTHONPATH=/tmp/metricx:$PYTHONPATH poetry run python scripts/08_generate_submission.py --test-data {args.test_data}",
        critical=True,
    )
elif args.phase <= 6 and not args.test_data:
    print("\n  Phase 6 skipped: no --test-data provided.")
    print("  When test data is released (Apr 21-30), run:")
    print("    poetry run python scripts/run_all.py --phase 6 --test-data path/to/test.parquet")


# =========================================================================
# Summary
# =========================================================================
total_time = time.time() - pipeline_start
print("\n" + "=" * 70)
print(f"  PIPELINE COMPLETE — {total_time/60:.1f} minutes")
print("=" * 70)

# List outputs
print("\nGenerated files:")
if os.path.isdir("outputs"):
    for f in sorted(os.listdir("outputs")):
        if f.endswith((".parquet", ".json", ".npy")):
            size = os.path.getsize(os.path.join("outputs", f))
            print(f"  {f:45s} {size/1024:>8.1f} KB")

if os.path.isdir("submission"):
    print("\nSubmission files:")
    for f in sorted(os.listdir("submission")):
        size = os.path.getsize(os.path.join("submission", f))
        print(f"  {f:45s} {size/1024:>8.1f} KB")
