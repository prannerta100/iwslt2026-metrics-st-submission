"""
SONAR/BLASER-2 QE inference for cross-modal speech-text quality estimation.

BLASER-2 QE computes quality scores by comparing:
  - Source speech embeddings (SONAR speech encoder)
  - Target text embeddings (SONAR text encoder)
in the shared SONAR embedding space.

This is our unique cross-modal signal: it directly measures whether the
target text is semantically aligned with the source SPEECH, not just
the ASR transcript. This can capture errors that text-only metrics miss
(e.g., ASR errors that propagate to incorrect metric scores).

Run on GPU: python scripts/06_blaser_inference.py
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--dev-only", action="store_true")
parser.add_argument("--text-only", action="store_true",
                    help="Use text-text mode (src_text + tgt_text) instead of speech-text")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("SONAR/BLASER-2 QE INFERENCE")
print("=" * 80)

dev = pd.read_parquet("outputs/dev_text.parquet")
print(f"Dev set: {len(dev)} rows")

# Check if we have audio paths
has_audio = "audio_path" in dev.columns or "audio" in dev.columns
if not has_audio:
    print("WARNING: No audio paths in dev data. Using text-text mode.")
    args.text_only = True


# ---------------------------------------------------------------------------
# 2. Load SONAR/BLASER models
# ---------------------------------------------------------------------------
print("\nLoading SONAR models...")

from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# SONAR text encoder
print("  Loading SONAR text encoder...")
text_encoder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    device=device,
)

# BLASER-2 QE model (reference-free)
print("  Loading BLASER-2 QE model...")
blaser_qe = load_blaser_model("blaser_2_0_qe", device=device).eval()

# Speech encoder (only if we have audio)
speech_encoder = None
if not args.text_only:
    try:
        from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
        print("  Loading SONAR speech encoder...")
        speech_encoder = SpeechToEmbeddingModelPipeline(
            encoder="sonar_speech_encoder_eng",
            device=device,
        )
        print("  Speech encoder loaded.")
    except Exception as e:
        print(f"  WARNING: Could not load speech encoder: {e}")
        print("  Falling back to text-text mode.")
        args.text_only = True


# ---------------------------------------------------------------------------
# 3. BLASER-2 QE scoring functions
# ---------------------------------------------------------------------------

# Language code mapping for SONAR (uses 3-letter codes)
LANG_MAP = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "cs": "ces_Latn",
    "uk": "ukr_Cyrl",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "is": "isl_Latn",
}


def compute_blaser_text_text(src_texts, tgt_texts, src_lang, tgt_lang, batch_size=32):
    """
    Compute BLASER-2 QE scores using text-text embeddings.
    Uses SONAR text encoder for both source and target.
    """
    src_lang_code = LANG_MAP.get(src_lang, "eng_Latn")
    tgt_lang_code = LANG_MAP.get(tgt_lang, "deu_Latn")

    scores = []
    n_batches = (len(src_texts) + batch_size - 1) // batch_size

    for i in range(0, len(src_texts), batch_size):
        batch_src = src_texts[i:i+batch_size]
        batch_tgt = tgt_texts[i:i+batch_size]

        # Encode source and target texts
        with torch.no_grad():
            src_emb = text_encoder.predict(batch_src, source_lang=src_lang_code)
            tgt_emb = text_encoder.predict(batch_tgt, source_lang=tgt_lang_code)

            # BLASER-2 QE: takes src and mt embeddings, no ref
            batch_scores = blaser_qe(src=src_emb, mt=tgt_emb).squeeze(-1)
            scores.extend(batch_scores.cpu().numpy().tolist())

        if (i // batch_size + 1) % 10 == 0:
            print(f"    Batch {i//batch_size + 1}/{n_batches}")

    return np.array(scores)


def compute_blaser_speech_text(audio_paths, tgt_texts, tgt_lang, batch_size=16):
    """
    Compute BLASER-2 QE scores using speech-text embeddings.
    Uses SONAR speech encoder for source audio and text encoder for target.
    """
    tgt_lang_code = LANG_MAP.get(tgt_lang, "deu_Latn")
    scores = []
    n_batches = (len(audio_paths) + batch_size - 1) // batch_size

    for i in range(0, len(audio_paths), batch_size):
        batch_audio = audio_paths[i:i+batch_size]
        batch_tgt = tgt_texts[i:i+batch_size]

        with torch.no_grad():
            # Load and encode audio
            # SONAR speech encoder expects 16kHz mono audio
            audio_tensors = []
            for path in batch_audio:
                waveform, sr = torchaudio.load(path)
                if sr != 16000:
                    waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                audio_tensors.append(waveform.squeeze(0))

            src_emb = speech_encoder.predict(audio_tensors)
            tgt_emb = text_encoder.predict(batch_tgt, source_lang=tgt_lang_code)

            batch_scores = blaser_qe(src=src_emb, mt=tgt_emb).squeeze(-1)
            scores.extend(batch_scores.cpu().numpy().tolist())

        if (i // batch_size + 1) % 10 == 0:
            print(f"    Batch {i//batch_size + 1}/{n_batches}")

    return np.array(scores)


# ---------------------------------------------------------------------------
# 4. Run inference on dev set
# ---------------------------------------------------------------------------
print("\n--- Running BLASER-2 on dev set ---")
start = time.time()

all_scores = np.zeros(len(dev))

# Process per language pair (SONAR needs language codes)
for (src_lang, tgt_lang), group in dev.groupby(["src_lang", "tgt_lang"]):
    print(f"\n  Processing {src_lang}->{tgt_lang} ({len(group)} samples)...")
    indices = group.index.values

    if args.text_only:
        lp_scores = compute_blaser_text_text(
            group["src_text"].tolist(),
            group["tgt_text"].tolist(),
            src_lang, tgt_lang,
            batch_size=args.batch_size,
        )
    else:
        audio_col = "audio_path" if "audio_path" in group.columns else "audio"
        lp_scores = compute_blaser_speech_text(
            group[audio_col].tolist(),
            group["tgt_text"].tolist(),
            tgt_lang,
            batch_size=args.batch_size // 2,  # Smaller batch for audio
        )

    all_scores[indices] = lp_scores

elapsed = time.time() - start
dev["blaser_score"] = all_scores
print(f"\nBLASER inference: {elapsed:.1f}s ({len(dev)/elapsed:.1f} samples/s)")

# ---------------------------------------------------------------------------
# 5. Evaluate on dev
# ---------------------------------------------------------------------------
print("\n--- BLASER-2 Dev Results ---")

overall_tau, _ = stats.kendalltau(dev["blaser_score"].values, dev["score"].values)
pearson_r, _ = stats.pearsonr(dev["blaser_score"].values, dev["score"].values)

taus = []
for doc_id, group in dev.groupby("doc_id"):
    if len(group) < 2:
        continue
    tau, _ = stats.kendalltau(group["blaser_score"].values, group["score"].values)
    if not np.isnan(tau):
        taus.append(tau)
per_source_tau = np.mean(taus) if taus else 0.0

print(f"  Overall Kendall Tau:     {overall_tau:.4f}")
print(f"  Per-source Kendall Tau:  {per_source_tau:.4f}")
print(f"  Pearson correlation:     {pearson_r:.4f}")

for (src, tgt), group in dev.groupby(["src_lang", "tgt_lang"]):
    lp_taus = []
    for doc_id, doc_group in group.groupby("doc_id"):
        if len(doc_group) < 2:
            continue
        tau, _ = stats.kendalltau(doc_group["blaser_score"].values, doc_group["score"].values)
        if not np.isnan(tau):
            lp_taus.append(tau)
    lp_tau = np.mean(lp_taus) if lp_taus else 0.0
    print(f"  {src}->{tgt}: per-source tau={lp_tau:.4f}")

print(f"\n  BLASER score range: [{dev['blaser_score'].min():.4f}, {dev['blaser_score'].max():.4f}]")
print(f"  BLASER score mean:  {dev['blaser_score'].mean():.4f}")

# ---------------------------------------------------------------------------
# 6. Compute SONAR embedding features (for ensemble)
# ---------------------------------------------------------------------------
print("\n--- Computing SONAR embedding features ---")

# Cosine similarity between src and tgt embeddings (raw, without BLASER head)
# This gives us an additional feature for the ensemble
cosine_scores = np.zeros(len(dev))
for (src_lang, tgt_lang), group in dev.groupby(["src_lang", "tgt_lang"]):
    indices = group.index.values
    src_lang_code = LANG_MAP.get(src_lang, "eng_Latn")
    tgt_lang_code = LANG_MAP.get(tgt_lang, "deu_Latn")

    with torch.no_grad():
        src_emb = text_encoder.predict(
            group["src_text"].tolist(), source_lang=src_lang_code
        )
        tgt_emb = text_encoder.predict(
            group["tgt_text"].tolist(), source_lang=tgt_lang_code
        )
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(src_emb, tgt_emb, dim=-1)
        cosine_scores[indices] = cos_sim.cpu().numpy()

dev["sonar_cosine"] = cosine_scores
cos_tau, _ = stats.kendalltau(cosine_scores, dev["score"].values)
print(f"  SONAR cosine similarity Kendall Tau: {cos_tau:.4f}")

# ---------------------------------------------------------------------------
# 7. Save predictions
# ---------------------------------------------------------------------------
existing_pred_file = "outputs/dev_with_predictions.parquet"
if os.path.exists(existing_pred_file):
    existing = pd.read_parquet(existing_pred_file)
    existing["blaser_score"] = dev["blaser_score"].values
    existing["sonar_cosine"] = dev["sonar_cosine"].values
    existing.to_parquet(existing_pred_file, index=False)
    print(f"\nMerged BLASER scores into {existing_pred_file}")
else:
    dev.to_parquet("outputs/dev_with_blaser.parquet", index=False)
    print(f"\nSaved to outputs/dev_with_blaser.parquet")

print("\n" + "=" * 80)
print("BLASER-2 INFERENCE COMPLETE")
print("=" * 80)
