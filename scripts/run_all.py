"""
Master pipeline orchestrator for IWSLT 2026 Metrics submission.

Runs the full pipeline in order:
  1. Data preparation (download + explore)
  2. CometKiwi-22 baseline
  3. xCOMET-XL inference (if model access granted)
  4. SONAR/BLASER-2 QE inference
  5. Whisper speech feature extraction
  6. CometKiwi fine-tuning
  7. Speech QE model training
  8. Advanced ensemble
  9. Generate submission files

Run on GPU: python scripts/run_all.py [--skip-download] [--skip-finetune]
"""

import os
import sys
import time
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--skip-download", action="store_true",
                    help="Skip data download (assume already available)")
parser.add_argument("--skip-finetune", action="store_true",
                    help="Skip CometKiwi fine-tuning")
parser.add_argument("--skip-xcomet", action="store_true",
                    help="Skip xCOMET (if no access)")
parser.add_argument("--skip-blaser", action="store_true",
                    help="Skip BLASER (if fairseq2 not available)")
parser.add_argument("--skip-speech", action="store_true",
                    help="Skip speech feature extraction")
args = parser.parse_args()


def run_step(name, cmd, critical=True):
    """Run a pipeline step, handling errors."""
    print("\n" + "=" * 80)
    print(f"STEP: {name}")
    print("=" * 80)
    start = time.time()

    result = subprocess.run(cmd, shell=True)

    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\nERROR: {name} failed (exit code {result.returncode})")
        if critical:
            print("This is a critical step. Aborting pipeline.")
            sys.exit(1)
        else:
            print("Non-critical step. Continuing...")
    else:
        print(f"\n{name} completed in {elapsed:.0f}s")

    return result.returncode == 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
pipeline_start = time.time()

print("=" * 80)
print("IWSLT 2026 METRICS - FULL PIPELINE")
print("=" * 80)

# 1. Data preparation
if not args.skip_download:
    run_step(
        "Data Download & Exploration",
        "python scripts/01d_download_and_explore.py",
        critical=True,
    )
else:
    print("\nSkipping data download (--skip-download)")
    # Verify data exists
    for f in ["outputs/dev_text.parquet", "outputs/train_text.parquet"]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found. Run without --skip-download first.")
            sys.exit(1)

# 2. CometKiwi-22 baseline
run_step(
    "CometKiwi-22 Baseline",
    "python scripts/02_cometkiwi_baseline.py",
    critical=True,
)

# 3. xCOMET-XL inference
if not args.skip_xcomet:
    success = run_step(
        "xCOMET-XL Inference",
        "python scripts/05_xcomet_inference.py --batch-size 64",
        critical=False,
    )
    if not success:
        print("xCOMET-XL failed. This may be due to model access. Continuing...")
else:
    print("\nSkipping xCOMET-XL (--skip-xcomet)")

# 4. BLASER-2 QE
if not args.skip_blaser:
    success = run_step(
        "SONAR/BLASER-2 QE Inference",
        "python scripts/06_blaser_inference.py --batch-size 256",
        critical=False,
    )
    if not success:
        print("BLASER-2 failed. Continuing with text-only signals...")
else:
    print("\nSkipping BLASER-2 (--skip-blaser)")

# 5. Whisper speech features
if not args.skip_speech:
    success = run_step(
        "Whisper Speech Feature Extraction",
        "python scripts/07_speech_qe.py --mode extract_features --batch-size 32",
        critical=False,
    )
else:
    print("\nSkipping speech features (--skip-speech)")

# 6. CometKiwi fine-tuning
if not args.skip_finetune:
    run_step(
        "CometKiwi Fine-tuning",
        "python scripts/03_finetune_cometkiwi.py",
        critical=False,
    )
else:
    print("\nSkipping CometKiwi fine-tuning (--skip-finetune)")

# 7. Speech QE model training
if not args.skip_speech:
    run_step(
        "Speech QE Model Training",
        "python scripts/07_speech_qe.py --mode train --batch-size 64 --epochs 10",
        critical=False,
    )

# 8. Basic ensemble
run_step(
    "Basic Ensemble",
    "python scripts/04_ensemble.py",
    critical=False,
)

# 9. Advanced ensemble
run_step(
    "Advanced Ensemble (LightGBM + Calibration)",
    "python scripts/04b_ensemble_advanced.py",
    critical=True,
)

# Summary
total_time = time.time() - pipeline_start
print("\n" + "=" * 80)
print(f"PIPELINE COMPLETE - Total time: {total_time/60:.1f} minutes")
print("=" * 80)

# Check outputs
print("\nGenerated files:")
for f in sorted(os.listdir("outputs")):
    if f.endswith((".parquet", ".npy", ".md")):
        size = os.path.getsize(os.path.join("outputs", f))
        print(f"  {f} ({size/1024:.1f} KB)")
