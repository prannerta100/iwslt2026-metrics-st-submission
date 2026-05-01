"""
Generate final submission for IWSLT 2026 Metrics Shared Task (test set).

Loads maikezu/iwslt2026-metrics-shared-test, scores with all available metrics,
applies ensemble weights optimized on dev, and outputs submission scores.

Test set: 48,044 rows (en→de: 24,016, en→zh: 24,028), 58 docs, 8 systems.
No gold scores (blind evaluation).

Run on GPU VM:
  poetry run python scripts/13_submit_test.py
  poetry run python scripts/13_submit_test.py --batch-size 64 --skip-metricx
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

METRICX_REPO = "/tmp/metricx"
if os.path.isdir(METRICX_REPO):
    sys.path.insert(0, METRICX_REPO)

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--output-dir", type=str, default="submission/")
parser.add_argument("--skip-cometkiwi22", action="store_true")
parser.add_argument("--skip-xcomet", action="store_true")
parser.add_argument("--skip-blaser", action="store_true")
parser.add_argument("--skip-metricx", action="store_true")
parser.add_argument("--skip-cometkiwi23xxl", action="store_true")
parser.add_argument("--skip-finetuned", action="store_true")
parser.add_argument("--weights-file", type=str, default=None,
                    help="JSON file with ensemble weights (from 04b). If not provided, uses dev-optimized defaults.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------------------------
print("=" * 80)
print("IWSLT 2026 METRICS SHARED TASK — TEST SET SUBMISSION")
print("=" * 80)

from datasets import load_dataset
ds = load_dataset("maikezu/iwslt2026-metrics-shared-test")
# Remove audio column to avoid torchcodec decoding issues
if "audio" in ds["test"].column_names:
    ds["test"] = ds["test"].remove_columns(["audio"])
test = ds["test"].to_pandas()

print(f"Test set: {len(test)} rows")
print(f"Language pairs: {test.groupby(['src_lang', 'tgt_lang']).size().to_dict()}")
print(f"Domains: {test.groupby('domain').size().to_dict()}")
print(f"Docs: {test['doc_id'].nunique()}, Systems: {test['tgt_system'].nunique()}")

assert "src_text" in test.columns, "Missing src_text"
assert "tgt_text" in test.columns, "Missing tgt_text"
test["src_text"] = test["src_text"].fillna("").astype(str)
test["tgt_text"] = test["tgt_text"].fillna("").astype(str)

gpus = 1 if torch.cuda.is_available() else 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if gpus:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

comet_samples = [
    {"src": row["src_text"], "mt": row["tgt_text"]}
    for _, row in test.iterrows()
]

# ---------------------------------------------------------------------------
# 2. Score with all available metrics
# ---------------------------------------------------------------------------

# --- CometKiwi-22 ---
if not args.skip_cometkiwi22:
    print(f"\n--- CometKiwi-22 ({len(test)} samples) ---")
    from comet import download_model, load_from_checkpoint

    local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
    model_path = local_ckpt if os.path.exists(local_ckpt) else download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    start = time.time()
    output = model.predict(comet_samples, batch_size=args.batch_size, gpus=gpus, num_workers=4 if gpus else 2)
    test["cometkiwi22_score"] = output["scores"]
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(test)/elapsed:.1f} samples/s)")
    print(f"  Score range: [{min(output['scores']):.4f}, {max(output['scores']):.4f}]")
    del model
    if gpus:
        torch.cuda.empty_cache()

# --- xCOMET-XL ---
if not args.skip_xcomet:
    try:
        print(f"\n--- xCOMET-XL ({len(test)} samples) ---")
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)

        start = time.time()
        output = model.predict(comet_samples, batch_size=args.batch_size, gpus=gpus, num_workers=4 if gpus else 2)
        test["xcomet_score"] = output["scores"]
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(test)/elapsed:.1f} samples/s)")
        print(f"  Score range: [{min(output['scores']):.4f}, {max(output['scores']):.4f}]")
        del model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  xCOMET-XL not available: {e}")

# --- CometKiwi-23-XXL ---
if not args.skip_cometkiwi23xxl:
    try:
        print(f"\n--- CometKiwi-23-XXL ({len(test)} samples) ---")
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
        model = load_from_checkpoint(model_path)

        start = time.time()
        output = model.predict(comet_samples, batch_size=32, gpus=gpus, num_workers=4 if gpus else 2)
        ck23_scores = output.scores if hasattr(output, "scores") else output["scores"]
        test["cometkiwi23xxl_score"] = ck23_scores
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(test)/elapsed:.1f} samples/s)")
        print(f"  Score range: [{min(ck23_scores):.4f}, {max(ck23_scores):.4f}]")
        del model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CometKiwi-23-XXL not available: {e}")

# --- Fine-tuned CometKiwi (pairwise) ---
if not args.skip_finetuned:
    finetuned_ckpt = None
    for ckpt_dir in ["models/cometkiwi_finetuned/", "models/cometkiwi23xxl_pairwise/"]:
        if os.path.exists(ckpt_dir):
            import glob
            ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
            if ckpts:
                finetuned_ckpt = sorted(ckpts)[-1]
                break

    if finetuned_ckpt:
        try:
            print(f"\n--- Fine-tuned CometKiwi ({finetuned_ckpt}) ---")
            from comet import download_model, load_from_checkpoint

            # Determine base model
            if "23xxl" in finetuned_ckpt:
                base_model = "Unbabel/wmt23-cometkiwi-da-xxl"
                batch_sz = 32
            else:
                base_model = "Unbabel/wmt22-cometkiwi-da"
                batch_sz = args.batch_size

            model_path = download_model(base_model)
            model = load_from_checkpoint(model_path)
            state_dict = torch.load(finetuned_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)

            start = time.time()
            output = model.predict(comet_samples, batch_size=batch_sz, gpus=gpus, num_workers=4 if gpus else 2)
            ft_scores = output.scores if hasattr(output, "scores") else output["scores"]
            test["finetuned_score"] = ft_scores
            elapsed = time.time() - start
            print(f"  Done in {elapsed:.1f}s ({len(test)/elapsed:.1f} samples/s)")
            print(f"  Score range: [{min(ft_scores):.4f}, {max(ft_scores):.4f}]")
            del model
            if gpus:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Fine-tuned model failed: {e}")

# --- MetricX-24 ---
if not args.skip_metricx:
    try:
        print(f"\n--- MetricX-24-Hybrid-XXL ({len(test)} samples) ---")
        from transformers import AutoTokenizer
        from metricx24.models import MT5ForRegression

        try:
            metricx_model = MT5ForRegression.from_pretrained(
                "google/metricx-24-hybrid-xxl-v2p6-bfloat16", torch_dtype="auto",
                attn_implementation="eager"
            )
        except TypeError:
            metricx_model = MT5ForRegression.from_pretrained(
                "google/metricx-24-hybrid-xxl-v2p6-bfloat16", torch_dtype="auto"
            )

        for _, module in metricx_model.named_modules():
            if hasattr(module, 'config'):
                module.config._attn_implementation = "eager"
                if hasattr(module.config, '_attn_implementation_internal'):
                    module.config._attn_implementation_internal = "eager"

        metricx_model = metricx_model.to(device).eval()
        metricx_tokenizer = AutoTokenizer.from_pretrained("google/mt5-xxl")

        mx_batch = 16
        metricx_scores = []
        start = time.time()

        for i in tqdm(range(0, len(test), mx_batch), desc="MetricX"):
            batch = test.iloc[i:i + mx_batch]
            input_texts = [
                f"source: {s} candidate: {m}"
                for s, m in zip(batch["src_text"], batch["tgt_text"])
            ]
            all_ids = []
            for text in input_texts:
                ids = metricx_tokenizer(text, max_length=1536, truncation=True)["input_ids"]
                all_ids.append(ids[:-1])

            max_len = max(len(ids) for ids in all_ids)
            input_ids = torch.zeros(len(all_ids), max_len, dtype=torch.long, device=device)
            attention_mask = torch.zeros(len(all_ids), max_len, dtype=torch.long, device=device)
            for j, ids in enumerate(all_ids):
                input_ids[j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[j, :len(ids)] = 1

            with torch.no_grad():
                enc_out = metricx_model.encoder(
                    input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                )
                dec_ids = torch.zeros(input_ids.shape[0], 1, dtype=torch.long, device=device)
                dec_out = metricx_model.decoder(
                    input_ids=dec_ids,
                    encoder_hidden_states=enc_out.last_hidden_state,
                    encoder_attention_mask=attention_mask,
                    return_dict=True, use_cache=False,
                )
                lm_logits = metricx_model.lm_head(dec_out.last_hidden_state[:, 0, :])
                preds = lm_logits[:, 250089]
                scores = np.clip(preds.float().cpu().numpy(), 0.0, 25.0)

            metricx_scores.extend(scores.tolist())

        test["metricx_error"] = metricx_scores
        test["metricx_score"] = 25.0 - np.array(metricx_scores)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(test)/elapsed:.1f} samples/s)")
        print(f"  Score range: [{test['metricx_score'].min():.4f}, {test['metricx_score'].max():.4f}]")
        del metricx_model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  MetricX-24 not available: {e}")

# --- BLASER-2 QE ---
if not args.skip_blaser:
    try:
        print(f"\n--- BLASER-2 QE ({len(test)} samples) ---")
        from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        from sonar.models.blaser.loader import load_blaser_model

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
        start = time.time()

        for (src_lang, tgt_lang), group in test.groupby(["src_lang", "tgt_lang"]):
            indices = group.index.values
            src_code = LANG_MAP.get(src_lang, "eng_Latn")
            tgt_code = LANG_MAP.get(tgt_lang, "deu_Latn")
            print(f"  {src_lang}->{tgt_lang}: {len(group)} samples")

            for i in range(0, len(group), args.batch_size):
                batch = group.iloc[i:i + args.batch_size]
                batch_idx = indices[i:i + len(batch)]
                with torch.no_grad():
                    src_emb = text_encoder.predict(batch["src_text"].tolist(), source_lang=src_code)
                    tgt_emb = text_encoder.predict(batch["tgt_text"].tolist(), source_lang=tgt_code)
                    scores = blaser_qe(src=src_emb, mt=tgt_emb).squeeze(-1)
                    blaser_scores[batch_idx] = scores.cpu().numpy()

        test["blaser_score"] = blaser_scores
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(test)/elapsed:.1f} samples/s)")
        print(f"  Score range: [{blaser_scores.min():.4f}, {blaser_scores.max():.4f}]")
        del text_encoder, blaser_qe
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  BLASER-2 not available: {e}")


# ---------------------------------------------------------------------------
# 3. Ensemble
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("ENSEMBLE")
print("=" * 80)

signal_cols = [c for c in test.columns if c.endswith("_score") and c not in ("score",)]
print(f"Available signals ({len(signal_cols)}): {signal_cols}")

# Dev-optimized default weights (from pairwise_score dominance on dev)
# pairwise/finetuned: tau=0.35, cometkiwi22: 0.32, metricx: 0.30, xcomet: 0.29, ck23xxl: 0.30, blaser: 0.27
DEFAULT_WEIGHTS = {
    "finetuned_score": 0.30,
    "pairwise_score": 0.30,
    "cometkiwi22_score": 0.15,
    "cometkiwi23xxl_score": 0.10,
    "metricx_score": 0.08,
    "xcomet_score": 0.05,
    "blaser_score": 0.02,
}

if args.weights_file and os.path.exists(args.weights_file):
    with open(args.weights_file) as f:
        loaded_weights = json.load(f)
    print(f"Loaded weights from {args.weights_file}")
else:
    loaded_weights = DEFAULT_WEIGHTS
    print("Using default dev-optimized weights")

# Filter to only signals we actually have
active_weights = {k: v for k, v in loaded_weights.items() if k in signal_cols}

if not active_weights:
    # Fallback: equal weight all available signals
    active_weights = {c: 1.0 / len(signal_cols) for c in signal_cols}
    print("WARNING: No matching weights found, using equal weights")

# Normalize weights
total_w = sum(active_weights.values())
active_weights = {k: v / total_w for k, v in active_weights.items()}

print(f"\nEnsemble weights:")
for col, w in sorted(active_weights.items(), key=lambda x: -x[1]):
    print(f"  {col}: {w:.4f}")

# Compute weighted ensemble
ensemble_scores = np.zeros(len(test))
for col, w in active_weights.items():
    ensemble_scores += test[col].values * w

test["final_score"] = ensemble_scores

# ---------------------------------------------------------------------------
# 4. Generate submission files
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
# Merge pre-computed LLM debate scores if available
llm_test_cache = "outputs/llm_debate_test.parquet"
if os.path.exists(llm_test_cache) and "llm_debate_score" not in test.columns:
    llm_df = pd.read_parquet(llm_test_cache)
    if "llm_debate_score" in llm_df.columns and len(llm_df) == len(test):
        test["llm_debate_score"] = llm_df["llm_debate_score"].values
        print(f"  Merged llm_debate_score from {llm_test_cache}")

print("GENERATING SUBMISSION")
print("=" * 80)

# Primary submission: ensemble scores
score_file = os.path.join(args.output_dir, "scores.txt")
with open(score_file, "w") as f:
    for score in test["final_score"].values:
        f.write(f"{score:.6f}\n")
print(f"Saved {len(test)} scores to {score_file}")

# Also save individual signal scores as backup submissions
for col in signal_cols:
    backup_file = os.path.join(args.output_dir, f"scores_{col.replace('_score', '')}.txt")
    with open(backup_file, "w") as f:
        for score in test[col].values:
            f.write(f"{score:.6f}\n")
    print(f"  Backup: {backup_file}")

# Save full predictions for analysis
test.to_parquet(os.path.join(args.output_dir, "test_predictions.parquet"), index=False)

# Save metadata
metadata = {
    "team": "pranav",
    "system": "ensemble-qe",
    "task": "iwslt2026-metrics-shared-task",
    "signals": signal_cols,
    "weights": active_weights,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "n_test_samples": len(test),
    "language_pairs": test.groupby(["src_lang", "tgt_lang"]).size().to_dict(),
    "score_stats": {
        "mean": float(test["final_score"].mean()),
        "std": float(test["final_score"].std()),
        "min": float(test["final_score"].min()),
        "max": float(test["final_score"].max()),
    },
}
# Fix tuple keys for JSON serialization
metadata["language_pairs"] = {f"{k[0]}-{k[1]}": v for k, v in metadata["language_pairs"].items()}

with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nScore stats: mean={test['final_score'].mean():.4f}, std={test['final_score'].std():.4f}")
print(f"  min={test['final_score'].min():.4f}, max={test['final_score'].max():.4f}")

# Per-LP stats
for (src, tgt), group in test.groupby(["src_lang", "tgt_lang"]):
    print(f"  {src}->{tgt}: mean={group['final_score'].mean():.4f}, std={group['final_score'].std():.4f}")

print(f"\nAll files saved to {args.output_dir}/")
print("\n" + "=" * 80)
print("SUBMISSION COMPLETE")
print("=" * 80)
