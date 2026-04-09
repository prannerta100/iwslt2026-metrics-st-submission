"""
xCOMET-XL inference on the dev and train sets.

xCOMET is an error-aware metric that produces both sentence-level and
token-level quality scores. In QE mode (no reference), it uses src + mt.

The token-level error annotations provide valuable features for the
ensemble: error count, error severity distribution, etc.

Requires: HuggingFace access to Unbabel/XCOMET-XL (gated model).
Run on GPU: python scripts/05_xcomet_inference.py
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import numpy as np
import pandas as pd
import torch
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=16,
                    help="Batch size for inference (default: 16 for 96GB GPU)")
parser.add_argument("--model", type=str, default="Unbabel/XCOMET-XL",
                    choices=["Unbabel/XCOMET-XL", "Unbabel/XCOMET-XXL"],
                    help="xCOMET model variant")
parser.add_argument("--dev-only", action="store_true",
                    help="Only run on dev set (skip train)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"xCOMET INFERENCE ({args.model})")
print("=" * 80)

dev = pd.read_parquet("outputs/dev_text.parquet")
print(f"Dev set: {len(dev)} rows")

if not args.dev_only:
    train = pd.read_parquet("outputs/train_text.parquet")
    print(f"Train set: {len(train)} rows")

# ---------------------------------------------------------------------------
# 2. Load xCOMET model
# ---------------------------------------------------------------------------
from comet import download_model, load_from_checkpoint

print(f"\nLoading {args.model}...")
model_path = download_model(args.model)
model = load_from_checkpoint(model_path)
print(f"Model loaded: {type(model).__name__}")

# Move to GPU if available
if torch.cuda.is_available():
    gpus = 1
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    gpus = 0
    print("Using CPU")

# ---------------------------------------------------------------------------
# 3. Prepare samples - QE mode (src + mt, no ref)
# ---------------------------------------------------------------------------
def prepare_samples(df):
    """Prepare samples for xCOMET. QE mode uses src + mt only."""
    samples = []
    for _, row in df.iterrows():
        samples.append({
            "src": row["src_text"],
            "mt": row["tgt_text"],
        })
    return samples

dev_samples = prepare_samples(dev)
print(f"\nPrepared {len(dev_samples)} dev samples")

# ---------------------------------------------------------------------------
# 4. Run inference on dev set
# ---------------------------------------------------------------------------
print("\n--- Running xCOMET on dev set ---")
start = time.time()

# xCOMET predict() returns: scores (sentence-level), metadata with error spans
dev_output = model.predict(
    dev_samples,
    batch_size=args.batch_size,
    gpus=gpus,
    num_workers=2 if gpus == 0 else 4,
)
elapsed = time.time() - start
print(f"Dev inference: {elapsed:.1f}s ({len(dev_samples)/elapsed:.1f} samples/s)")

# Extract sentence-level scores
# CometOutput uses attribute access (.scores, .metadata), not dict access
dev["xcomet_score"] = dev_output.scores

# Extract token-level error information if available
# xCOMET returns error spans in output.metadata['error_spans']
# Each entry is a list of dicts: {'text': str, 'severity': str, 'start': int, 'end': int, 'confidence': float}
SEVERITY_WEIGHT = {"minor": 1.0, "major": 2.0, "critical": 3.0}

if hasattr(dev_output, 'metadata') and dev_output.metadata and "error_spans" in dev_output.metadata:
    print("\nExtracting token-level error features...")
    error_spans_list = dev_output.metadata["error_spans"]
    error_counts = []
    error_severity_scores = []
    error_confidences = []

    for spans in error_spans_list:
        error_counts.append(len(spans))
        if spans:
            # Weighted severity score
            sev_score = sum(SEVERITY_WEIGHT.get(s.get("severity", "minor"), 1.0) for s in spans)
            error_severity_scores.append(sev_score)
            # Average confidence of error spans
            confs = [s.get("confidence", 0.5) for s in spans]
            error_confidences.append(np.mean(confs))
        else:
            error_severity_scores.append(0.0)
            error_confidences.append(0.0)

    dev["xcomet_error_count"] = error_counts
    dev["xcomet_error_severity"] = error_severity_scores
    dev["xcomet_error_confidence"] = error_confidences
    print(f"  Mean error count: {np.mean(error_counts):.2f}")
    print(f"  Mean error severity: {np.mean(error_severity_scores):.4f}")
    print(f"  Mean error confidence: {np.mean(error_confidences):.4f}")

# ---------------------------------------------------------------------------
# 5. Evaluate on dev
# ---------------------------------------------------------------------------
print("\n--- xCOMET Dev Results ---")

# Overall metrics
overall_tau, _ = stats.kendalltau(dev["xcomet_score"].values, dev["score"].values)
pearson_r, _ = stats.pearsonr(dev["xcomet_score"].values, dev["score"].values)

# Per-source Kendall Tau
taus = []
for doc_id, group in dev.groupby("doc_id"):
    if len(group) < 2:
        continue
    tau, _ = stats.kendalltau(group["xcomet_score"].values, group["score"].values)
    if not np.isnan(tau):
        taus.append(tau)
per_source_tau = np.mean(taus) if taus else 0.0

print(f"  Overall Kendall Tau:     {overall_tau:.4f}")
print(f"  Per-source Kendall Tau:  {per_source_tau:.4f}")
print(f"  Pearson correlation:     {pearson_r:.4f}")

# Per language pair
for (src, tgt), group in dev.groupby(["src_lang", "tgt_lang"]):
    lp_taus = []
    for doc_id, doc_group in group.groupby("doc_id"):
        if len(doc_group) < 2:
            continue
        tau, _ = stats.kendalltau(doc_group["xcomet_score"].values, doc_group["score"].values)
        if not np.isnan(tau):
            lp_taus.append(tau)
    lp_tau = np.mean(lp_taus) if lp_taus else 0.0
    print(f"  {src}->{tgt}: per-source tau={lp_tau:.4f}")

# Score distribution
print(f"\n  xCOMET score range: [{dev['xcomet_score'].min():.4f}, {dev['xcomet_score'].max():.4f}]")
print(f"  xCOMET score mean:  {dev['xcomet_score'].mean():.4f}")
print(f"  xCOMET score std:   {dev['xcomet_score'].std():.4f}")

# ---------------------------------------------------------------------------
# 6. Run on train set (for fine-tuning features)
# ---------------------------------------------------------------------------
if not args.dev_only:
    print("\n--- Running xCOMET on train set ---")
    train_samples = prepare_samples(train)
    start = time.time()
    train_output = model.predict(
        train_samples,
        batch_size=args.batch_size,
        gpus=gpus,
        num_workers=4,
    )
    elapsed = time.time() - start
    print(f"Train inference: {elapsed:.1f}s ({len(train_samples)/elapsed:.1f} samples/s)")
    train["xcomet_score"] = train_output.scores
    train.to_parquet("outputs/train_with_xcomet.parquet", index=False)
    print(f"Saved train predictions to outputs/train_with_xcomet.parquet")

# ---------------------------------------------------------------------------
# 7. Save dev predictions
# ---------------------------------------------------------------------------
# Merge with existing predictions if available
existing_pred_file = "outputs/dev_with_predictions.parquet"
if os.path.exists(existing_pred_file):
    existing = pd.read_parquet(existing_pred_file)
    # Add xcomet columns to existing predictions
    for col in dev.columns:
        if col.startswith("xcomet_"):
            existing[col] = dev[col].values
    existing.to_parquet(existing_pred_file, index=False)
    print(f"\nMerged xCOMET scores into {existing_pred_file}")
else:
    dev.to_parquet("outputs/dev_with_xcomet.parquet", index=False)
    print(f"\nSaved to outputs/dev_with_xcomet.parquet")

print("\n" + "=" * 80)
print("xCOMET INFERENCE COMPLETE")
print("=" * 80)
