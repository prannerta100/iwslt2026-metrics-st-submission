"""
Generate submission files for the IWSLT 2026 Metrics Shared Task.

Takes the best ensemble model and generates predictions on the test set.
Output format follows the shared task requirements.

Run: python scripts/08_generate_submission.py --test-data path/to/test.parquet
"""

import os
import sys
import argparse
import json
import time

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--test-data", type=str, required=True,
                    help="Path to test set (parquet or directory)")
parser.add_argument("--model-dir", type=str, default="models/",
                    help="Directory with trained models")
parser.add_argument("--output-dir", type=str, default="submission/",
                    help="Output directory for submission files")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--method", type=str, default="auto",
                    choices=["auto", "cometkiwi", "finetuned", "ensemble"],
                    help="Scoring method (auto selects best available)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------------------------
print("=" * 80)
print("GENERATING SUBMISSION")
print("=" * 80)

if args.test_data.endswith(".parquet"):
    test = pd.read_parquet(args.test_data)
else:
    # Try loading from HuggingFace dataset
    from datasets import load_dataset
    ds = load_dataset(args.test_data)
    test_split = ds["test"] if "test" in ds else list(ds.values())[0]
    test = test_split.to_pandas()

print(f"Test set: {len(test)} rows")
if "src_lang" in test.columns and "tgt_lang" in test.columns:
    print(f"Language pairs: {test.groupby(['src_lang', 'tgt_lang']).size().to_dict()}")

# Ensure text columns exist
assert "src_text" in test.columns, "Missing src_text column"
assert "tgt_text" in test.columns, "Missing tgt_text column"


# ---------------------------------------------------------------------------
# 2. Score with all available models
# ---------------------------------------------------------------------------
from comet import download_model, load_from_checkpoint

samples = [
    {"src": row["src_text"], "mt": row["tgt_text"]}
    for _, row in test.iterrows()
]

gpus = 1 if torch.cuda.is_available() else 0
num_workers = 4 if gpus else 2

# --- CometKiwi-22 (always available) ---
print("\n--- CometKiwi-22 ---")
local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
if os.path.exists(local_ckpt):
    model_path = local_ckpt
else:
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)
output = model.predict(samples, batch_size=args.batch_size, gpus=gpus, num_workers=num_workers)
test["cometkiwi22_score"] = output["scores"]
print(f"  Score range: [{min(output['scores']):.4f}, {max(output['scores']):.4f}]")
del model

# --- Fine-tuned CometKiwi (if available) ---
finetuned_ckpt = None
if os.path.exists("models/cometkiwi_finetuned/"):
    import glob
    ckpts = glob.glob("models/cometkiwi_finetuned/*.ckpt")
    if ckpts:
        finetuned_ckpt = sorted(ckpts)[-1]  # Latest

if finetuned_ckpt:
    print(f"\n--- Fine-tuned CometKiwi ({finetuned_ckpt}) ---")
    model = load_from_checkpoint(finetuned_ckpt)
    output = model.predict(samples, batch_size=args.batch_size, gpus=gpus, num_workers=num_workers)
    test["finetuned_score"] = output["scores"]
    print(f"  Score range: [{min(output['scores']):.4f}, {max(output['scores']):.4f}]")
    del model

# --- xCOMET-XL (if available) ---
try:
    print("\n--- xCOMET-XL ---")
    model_path = download_model("Unbabel/XCOMET-XL")
    model = load_from_checkpoint(model_path)
    output = model.predict(samples, batch_size=args.batch_size, gpus=gpus, num_workers=num_workers)
    test["xcomet_score"] = output["scores"]
    print(f"  Score range: [{min(output['scores']):.4f}, {max(output['scores']):.4f}]")
    del model
except Exception as e:
    print(f"  xCOMET-XL not available: {e}")

# --- CometKiwi-23-XXL (if available) ---
try:
    print("\n--- CometKiwi-23-XXL ---")
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    model = load_from_checkpoint(model_path)
    output = model.predict(samples, batch_size=args.batch_size, gpus=gpus, num_workers=num_workers)
    ck23_scores = output.scores if hasattr(output, "scores") else output["scores"]
    test["cometkiwi23xxl_score"] = ck23_scores
    print(f"  Score range: [{min(ck23_scores):.4f}, {max(ck23_scores):.4f}]")
    del model
except Exception as e:
    print(f"  CometKiwi-23-XXL not available: {e}")

# --- MetricX-24 (if available) ---
try:
    print("\n--- MetricX-24-Hybrid-XXL ---")
    from transformers import AutoTokenizer
    try:
        from metricx24.models import MT5ForRegression
        metricx_model = MT5ForRegression.from_pretrained(
            "google/metricx-24-hybrid-xxl-v2p6-bfloat16", torch_dtype="auto"
        )
    except ImportError:
        from transformers import AutoModelForSeq2SeqLM
        metricx_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/metricx-24-hybrid-xxl-v2p6-bfloat16", torch_dtype="auto"
        )
    metricx_tokenizer = AutoTokenizer.from_pretrained("google/mt5-xxl")

    device_mx = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metricx_model = metricx_model.to(device_mx).eval()

    metricx_scores = []
    for i in range(0, len(test), 16):
        batch = test.iloc[i:i+16]
        input_texts = [f"source: {s} candidate: {m}" for s, m in zip(batch["src_text"], batch["tgt_text"])]
        inputs = metricx_tokenizer(input_texts, max_length=1536, truncation=True, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"][:, :-1].to(device_mx)
        attention_mask = inputs["attention_mask"][:, :-1].to(device_mx)
        with torch.no_grad():
            outputs = metricx_model(input_ids=input_ids, attention_mask=attention_mask)
            if hasattr(outputs, "predictions"):
                scores = outputs.predictions.cpu().numpy()
            else:
                logits = outputs.logits[:, 0, :]
                scores = logits[:, min(250089, logits.shape[-1]-1)].cpu().numpy()
        metricx_scores.extend(np.clip(scores, 0, 25).tolist())
    test["metricx_score"] = 25.0 - np.array(metricx_scores)  # Invert: quality = 25 - error
    print(f"  Score range: [{test['metricx_score'].min():.4f}, {test['metricx_score'].max():.4f}]")
    del metricx_model
except Exception as e:
    print(f"  MetricX-24 not available: {e}")

# --- BLASER-2 QE (if available) ---
try:
    print("\n--- BLASER-2 QE ---")
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    from sonar.models.blaser.loader import load_blaser_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    )
    blaser_qe = load_blaser_model("blaser_2_0_qe").to(device).eval()

    LANG_MAP = {
        "en": "eng_Latn", "de": "deu_Latn", "zh": "zho_Hans",
    }

    blaser_scores = np.zeros(len(test))
    for (src_lang, tgt_lang), group in test.groupby(["src_lang", "tgt_lang"]):
        indices = group.index.values
        src_code = LANG_MAP.get(src_lang, "eng_Latn")
        tgt_code = LANG_MAP.get(tgt_lang, "deu_Latn")

        for i in range(0, len(group), args.batch_size):
            batch = group.iloc[i:i+args.batch_size]
            with torch.no_grad():
                src_emb = text_encoder.predict(batch["src_text"].tolist(), source_lang=src_code)
                tgt_emb = text_encoder.predict(batch["tgt_text"].tolist(), source_lang=tgt_code)
                scores = blaser_qe(src=src_emb, mt=tgt_emb).squeeze(-1)
                blaser_scores[indices[i:i+len(batch)]] = scores.cpu().numpy()

    test["blaser_score"] = blaser_scores
    print(f"  Score range: [{blaser_scores.min():.4f}, {blaser_scores.max():.4f}]")
    del text_encoder, blaser_qe
except Exception as e:
    print(f"  BLASER-2 not available: {e}")

# ---------------------------------------------------------------------------
# 3. Ensemble
# ---------------------------------------------------------------------------
print("\n--- Computing ensemble score ---")

signal_cols = [c for c in test.columns if c.endswith("_score") and c != "score"]
print(f"Available signals: {signal_cols}")

if len(signal_cols) == 1:
    test["final_score"] = test[signal_cols[0]]
    print(f"Single signal mode: using {signal_cols[0]}")
else:
    # Load optimized weights from dev set (saved during training)
    weight_file = "outputs/ensemble_weights.json"
    if os.path.exists(weight_file):
        with open(weight_file) as f:
            saved_weights = json.load(f)
        weights = []
        for col in signal_cols:
            weights.append(saved_weights.get(col, 1.0 / len(signal_cols)))
        print(f"Loaded weights: {dict(zip(signal_cols, weights))}")
    else:
        # Equal weights as fallback
        weights = [1.0 / len(signal_cols)] * len(signal_cols)
        print(f"Using equal weights (no saved weights found)")

    # Weighted average
    ensemble = np.zeros(len(test))
    for col, w in zip(signal_cols, weights):
        ensemble += test[col].values * w
    ensemble /= sum(weights)
    test["final_score"] = ensemble

# ---------------------------------------------------------------------------
# 4. Generate submission file
# ---------------------------------------------------------------------------
print("\n--- Generating submission ---")

# Submission format: one score per line, matching test set order
submission_scores = test["final_score"].values

# Save scores
score_file = os.path.join(args.output_dir, "scores.txt")
with open(score_file, "w") as f:
    for score in submission_scores:
        f.write(f"{score:.6f}\n")
print(f"Saved {len(submission_scores)} scores to {score_file}")

# Save full predictions for analysis
test.to_parquet(os.path.join(args.output_dir, "test_predictions.parquet"), index=False)

# Save metadata
metadata = {
    "team": "pranav",
    "system": "ensemble-qe",
    "signals": signal_cols,
    "method": args.method,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "n_test_samples": len(test),
    "score_stats": {
        "mean": float(submission_scores.mean()),
        "std": float(submission_scores.std()),
        "min": float(submission_scores.min()),
        "max": float(submission_scores.max()),
    },
}
with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSubmission files saved to {args.output_dir}/")
print(f"Score stats: mean={submission_scores.mean():.4f}, std={submission_scores.std():.4f}")

print("\n" + "=" * 80)
print("SUBMISSION GENERATION COMPLETE")
print("=" * 80)
