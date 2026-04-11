"""
MetricX-24-Hybrid-XXL inference for IWSLT 2026 QE submission.

MetricX-24 is Google's WMT24-winning metric. Key details:
- Uses mT5-XXL backbone with regression head
- Requires the `metricx` package from github.com/google-research/metricx
- Input format: "source: {src} candidate: {mt}" for QE mode
- Outputs MQM error score in [0, 25] range (LOWER = better quality)
- Must remove EOS token before inference
- Tokenizer: google/mt5-xxl (separate from model weights)

Model: google/metricx-24-hybrid-xxl-v2p6-bfloat16 (~24GB in bfloat16)

Run on GPU: python scripts/09_metricx_inference.py [--batch-size 16]
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

# MetricX has no PyPI package — add cloned repo to path
METRICX_REPO = "/tmp/metricx"
if os.path.isdir(METRICX_REPO):
    sys.path.insert(0, METRICX_REPO)

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=16,
                    help="Batch size for inference (XXL model is large)")
parser.add_argument("--model", type=str,
                    default="google/metricx-24-hybrid-xxl-v2p6-bfloat16",
                    help="MetricX model ID on HuggingFace")
parser.add_argument("--max-length", type=int, default=1536,
                    help="Max input sequence length")
parser.add_argument("--score-train", action="store_true",
                    help="Also score training data")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("METRICX-24 INFERENCE")
print("=" * 80)

dev = pd.read_parquet("outputs/dev_text.parquet")
print(f"Dev set: {len(dev)} rows")

# ---------------------------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------------------------
print(f"\nLoading MetricX model: {args.model}")

# MetricX-24 uses a custom MT5ForRegression class from the metricx package.
# Try importing from metricx24 first; if not installed, fall back to loading
# the model with AutoModelForSeq2SeqLM and extracting logits manually.
try:
    from metricx24.models import MT5ForRegression
    print("Using metricx24.models.MT5ForRegression")
except ImportError:
    print("ERROR: metricx24 package not found.")
    print("MetricX requires the custom MT5ForRegression class — AutoModel fallback produces garbage.")
    print("Run: bash scripts/setup_metricx.sh")
    sys.exit(1)

from transformers import AutoTokenizer

# Determine which mt5 tokenizer to use based on model size
if "xxl" in args.model.lower():
    tokenizer_name = "google/mt5-xxl"
elif "xl" in args.model.lower():
    tokenizer_name = "google/mt5-xl"
else:
    tokenizer_name = "google/mt5-large"

print(f"Loading tokenizer: {tokenizer_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

print(f"Loading model weights...")
load_start = time.time()

model = MT5ForRegression.from_pretrained(args.model, torch_dtype="auto")

if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Model VRAM: {allocated:.1f} GB")
else:
    device = torch.device("cpu")
    print("Using CPU (this will be VERY slow for XXL)")

model.eval()
print(f"Model loaded in {time.time() - load_start:.1f}s")


# ---------------------------------------------------------------------------
# 3. Inference function
# ---------------------------------------------------------------------------
def score_metricx_batch(src_texts, mt_texts):
    """
    Score a batch of (source, candidate) pairs with MetricX-24.
    Returns MQM error scores in [0, 25] (lower = better).
    """
    # Format inputs for QE mode (no reference)
    input_texts = [
        f"source: {src} candidate: {mt}"
        for src, mt in zip(src_texts, mt_texts)
    ]

    # Tokenize each example, remove EOS per-example, then pad.
    # MetricX was trained without EOS — must remove BEFORE padding,
    # not after ([:, :-1] after padding removes PAD, not EOS).
    all_ids = []
    for text in input_texts:
        ids = tokenizer(text, max_length=args.max_length, truncation=True)["input_ids"]
        ids = ids[:-1]  # Remove EOS (always last token before padding)
        all_ids.append(ids)

    # Pad to max length in batch
    max_len = max(len(ids) for ids in all_ids)
    input_ids = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    for i, ids in enumerate(all_ids):
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :len(ids)] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        # MT5ForRegression returns an object with .predictions attribute
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # .predictions is shape [batch_size] with MQM error scores
        scores = outputs.predictions.cpu().numpy()

    # Clip to valid range [0, 25]
    scores = np.clip(scores, 0.0, 25.0)
    return scores


# ---------------------------------------------------------------------------
# 4. Run inference on dev set
# ---------------------------------------------------------------------------
print(f"\n--- Running MetricX-24 inference on dev set ({len(dev)} samples) ---")
print(f"Batch size: {args.batch_size}")

all_scores = []
start = time.time()

for i in tqdm(range(0, len(dev), args.batch_size), desc="MetricX-24"):
    batch = dev.iloc[i:i + args.batch_size]
    src_texts = batch["src_text"].tolist()
    mt_texts = batch["tgt_text"].tolist()

    # Ensure strings (not NaN/None)
    src_texts = [str(s) if pd.notna(s) else "" for s in src_texts]
    mt_texts = [str(m) if pd.notna(m) else "" for m in mt_texts]

    batch_scores = score_metricx_batch(src_texts, mt_texts)
    all_scores.extend(batch_scores.tolist())

elapsed = time.time() - start
print(f"\nInference took {elapsed:.1f}s ({len(dev)/elapsed:.1f} samples/s)")

# MetricX outputs ERROR scores (lower = better quality)
# Convert to QUALITY scores (higher = better) for ensemble compatibility
metricx_error = np.array(all_scores)
# Linear inversion: quality = 25 - error (maps [0,25] -> [25,0])
metricx_quality = 25.0 - metricx_error

dev["metricx_error"] = metricx_error
dev["metricx_score"] = metricx_quality  # Use this for ensemble (higher = better)

print(f"\nMetricX-24 Score Distribution (error, lower=better):")
print(f"  Min: {metricx_error.min():.4f}, Max: {metricx_error.max():.4f}")
print(f"  Mean: {metricx_error.mean():.4f}, Std: {metricx_error.std():.4f}")

print(f"\nMetricX-24 Score Distribution (quality, higher=better):")
print(f"  Min: {metricx_quality.min():.4f}, Max: {metricx_quality.max():.4f}")
print(f"  Mean: {metricx_quality.mean():.4f}, Std: {metricx_quality.std():.4f}")


# ---------------------------------------------------------------------------
# 5. Evaluate
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


# Note: gold scores are 0-100, metricx_score is 0-25 quality
# Kendall Tau is rank-based so scale doesn't matter
per_source_tau = kendall_tau_per_source(dev, "metricx_score", "score")
overall_tau, _ = stats.kendalltau(dev["metricx_score"].values, dev["score"].values)
pearson, _ = stats.pearsonr(dev["metricx_score"].values, dev["score"].values)

print(f"\nMetricX-24 Results:")
print(f"  Overall Kendall Tau:    {overall_tau:.4f}")
print(f"  Per-source Kendall Tau: {per_source_tau:.4f}")
print(f"  Pearson:                {pearson:.4f}")

# Per language pair
for (src, tgt), group in dev.groupby(["src_lang", "tgt_lang"]):
    lp_taus = []
    for doc_id, doc_group in group.groupby("doc_id"):
        if len(doc_group) < 2:
            continue
        tau, _ = stats.kendalltau(doc_group["metricx_score"].values, doc_group["score"].values)
        if not np.isnan(tau):
            lp_taus.append(tau)
    lp_tau = np.mean(lp_taus) if lp_taus else 0.0
    print(f"  {src}->{tgt}: per-source tau={lp_tau:.4f}")


# ---------------------------------------------------------------------------
# 6. Save and merge
# ---------------------------------------------------------------------------
dev.to_parquet("outputs/dev_with_metricx.parquet", index=False)
print(f"\nSaved to outputs/dev_with_metricx.parquet")

# Merge into main predictions file
pred_file = "outputs/dev_with_predictions.parquet"
if os.path.exists(pred_file):
    existing = pd.read_parquet(pred_file)
    if len(existing) == len(dev):
        existing["metricx_score"] = dev["metricx_score"].values
        existing["metricx_error"] = dev["metricx_error"].values
        existing.to_parquet(pred_file, index=False)
        print(f"Merged metricx_score into {pred_file}")
    else:
        print(f"WARNING: Row count mismatch ({len(existing)} vs {len(dev)}), skipping merge into {pred_file}")
        print(f"  Scores are saved separately in outputs/dev_with_metricx.parquet")


# ---------------------------------------------------------------------------
# 7. Optionally score training data
# ---------------------------------------------------------------------------
if args.score_train:
    train_file = "outputs/train_text.parquet"
    if os.path.exists(train_file):
        train = pd.read_parquet(train_file)
        print(f"\n--- Scoring training data ({len(train)} samples) ---")

        train_scores = []
        for i in tqdm(range(0, len(train), args.batch_size), desc="MetricX train"):
            batch = train.iloc[i:i + args.batch_size]
            src_texts = [str(s) if pd.notna(s) else "" for s in batch["src_text"].tolist()]
            mt_texts = [str(m) if pd.notna(m) else "" for m in batch["tgt_text"].tolist()]
            batch_scores = score_metricx_batch(src_texts, mt_texts)
            train_scores.extend(batch_scores.tolist())

        train["metricx_error"] = train_scores
        train["metricx_score"] = 25.0 - np.array(train_scores)
        train.to_parquet("outputs/train_with_metricx.parquet", index=False)
        print(f"Saved train scores to outputs/train_with_metricx.parquet")


print("\n" + "=" * 80)
print("METRICX-24 INFERENCE COMPLETE")
print("=" * 80)
