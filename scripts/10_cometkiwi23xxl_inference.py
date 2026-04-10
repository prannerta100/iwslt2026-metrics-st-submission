"""
CometKiwi-23-XXL inference for IWSLT 2026 QE submission.

CometKiwi-23-XXL is Unbabel's largest QE model (10.7B params, XLM-R-XXL backbone).
It's a significant upgrade from CometKiwi-22 (~565M).

Uses the same `unbabel-comet` package as our existing CometKiwi-22 scripts.

Model: Unbabel/wmt23-cometkiwi-da-xxl (~21GB at fp16)
Requires: ~25-30GB VRAM with batch inference

Run on GPU: python scripts/10_cometkiwi23xxl_inference.py [--batch-size 32]
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

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for inference")
parser.add_argument("--model", type=str, default="Unbabel/wmt23-cometkiwi-da-xxl",
                    help="COMET model name")
parser.add_argument("--score-train", action="store_true",
                    help="Also score training data")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print(f"COMETKIWI-23-XXL INFERENCE")
print("=" * 80)

dev = pd.read_parquet("outputs/dev_text.parquet")
print(f"Dev set: {len(dev)} rows")

# ---------------------------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------------------------
from comet import download_model, load_from_checkpoint

print(f"\nLoading model: {args.model}")
print("This is a 10.7B parameter model — download may take a while on first run.")

if not os.environ.get("HF_TOKEN"):
    print("WARNING: HF_TOKEN not set. This model is gated — run `huggingface-cli login` first.")

load_start = time.time()
model_path = download_model(args.model)
model = load_from_checkpoint(model_path)

print(f"Model loaded in {time.time() - load_start:.1f}s")

gpus = 1 if torch.cuda.is_available() else 0
if gpus:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
else:
    print("Using CPU (this will be VERY slow for XXL)")

# ---------------------------------------------------------------------------
# 3. Prepare samples and run inference
# ---------------------------------------------------------------------------
samples = [
    {"src": str(row["src_text"]) if pd.notna(row["src_text"]) else "",
     "mt": str(row["tgt_text"]) if pd.notna(row["tgt_text"]) else ""}
    for _, row in dev.iterrows()
]
print(f"\nPrepared {len(samples)} samples")

print(f"\n--- Running inference (batch_size={args.batch_size}) ---")
start = time.time()
output = model.predict(
    samples,
    batch_size=args.batch_size,
    gpus=gpus,
    num_workers=4 if gpus else 2,
)
elapsed = time.time() - start
print(f"Inference took {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/s)")

# Handle both attribute and dict access for output scores
scores = output.scores if hasattr(output, "scores") else output["scores"]
dev["cometkiwi23xxl_score"] = scores

# ---------------------------------------------------------------------------
# 4. Evaluate
# ---------------------------------------------------------------------------
def kendall_tau_per_source(df, pred_col, gold_col="score"):
    taus = []
    for doc_id, group in df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, _ = stats.kendalltau(group[pred_col].values, group[gold_col].values)
        if not np.isnan(tau):
            taus.append(tau)
    return np.mean(taus) if taus else 0.0


# Drop NaN gold scores before computing correlations
eval_dev = dev.dropna(subset=["score"])
pred_scores = eval_dev["cometkiwi23xxl_score"].values
per_source_tau = kendall_tau_per_source(eval_dev, "cometkiwi23xxl_score", "score")
overall_tau, _ = stats.kendalltau(pred_scores, eval_dev["score"].values)
pearson, _ = stats.pearsonr(pred_scores, eval_dev["score"].values)

print(f"\nCometKiwi-23-XXL Results:")
print(f"  Overall Kendall Tau:    {overall_tau:.4f}")
print(f"  Per-source Kendall Tau: {per_source_tau:.4f}")
print(f"  Pearson:                {pearson:.4f}")
print(f"  Score range: [{pred_scores.min():.4f}, {pred_scores.max():.4f}]")
print(f"  Score mean:  {pred_scores.mean():.4f} (std={pred_scores.std():.4f})")

# Per language pair
for (src, tgt), group in dev.groupby(["src_lang", "tgt_lang"]):
    lp_taus = []
    for doc_id, doc_group in group.groupby("doc_id"):
        if len(doc_group) < 2:
            continue
        tau, _ = stats.kendalltau(doc_group["cometkiwi23xxl_score"].values, doc_group["score"].values)
        if not np.isnan(tau):
            lp_taus.append(tau)
    lp_tau = np.mean(lp_taus) if lp_taus else 0.0
    print(f"  {src}->{tgt}: per-source tau={lp_tau:.4f}")


# ---------------------------------------------------------------------------
# 5. Save and merge
# ---------------------------------------------------------------------------
dev.to_parquet("outputs/dev_with_cometkiwi23xxl.parquet", index=False)
print(f"\nSaved to outputs/dev_with_cometkiwi23xxl.parquet")

# Merge into main predictions file
pred_file = "outputs/dev_with_predictions.parquet"
if os.path.exists(pred_file):
    existing = pd.read_parquet(pred_file)
    if len(existing) == len(dev):
        existing["cometkiwi23xxl_score"] = dev["cometkiwi23xxl_score"].values
        existing.to_parquet(pred_file, index=False)
        print(f"Merged cometkiwi23xxl_score into {pred_file}")
    else:
        print(f"WARNING: Row count mismatch ({len(existing)} vs {len(dev)}), skipping merge into {pred_file}")
        print(f"  Scores are saved separately in outputs/dev_with_cometkiwi23xxl.parquet")


# ---------------------------------------------------------------------------
# 6. Optionally score training data
# ---------------------------------------------------------------------------
if args.score_train:
    train_file = "outputs/train_text.parquet"
    if os.path.exists(train_file):
        train = pd.read_parquet(train_file)
        print(f"\n--- Scoring training data ({len(train)} samples) ---")
        train_samples = [
            {"src": str(row["src_text"]) if pd.notna(row["src_text"]) else "",
             "mt": str(row["tgt_text"]) if pd.notna(row["tgt_text"]) else ""}
            for _, row in train.iterrows()
        ]
        train_output = model.predict(
            train_samples, batch_size=args.batch_size,
            gpus=gpus, num_workers=4 if gpus else 2,
        )
        train_scores = train_output.scores if hasattr(train_output, "scores") else train_output["scores"]
        train["cometkiwi23xxl_score"] = train_scores
        train.to_parquet("outputs/train_with_cometkiwi23xxl.parquet", index=False)
        print(f"Saved to outputs/train_with_cometkiwi23xxl.parquet")


print("\n" + "=" * 80)
print("COMETKIWI-23-XXL INFERENCE COMPLETE")
print("=" * 80)
