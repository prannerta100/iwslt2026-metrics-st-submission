"""
Score the full training set (33K samples) with all available metrics.

This is critical: our LGBM ensemble overfits massively (0.82 train → 0.36 test)
when trained on only 5.5K dev samples via CV. Scoring 33K train samples and
training on those should dramatically reduce the generalization gap.

Run on GPU: poetry run python scripts/11_score_train.py [--batch-size 128]
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

METRICX_REPO = "/tmp/metricx"
if os.path.isdir(METRICX_REPO):
    sys.path.insert(0, METRICX_REPO)

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--input", type=str, default="outputs/train_text.parquet")
parser.add_argument("--output", type=str, default="outputs/train_scored.parquet")
parser.add_argument("--skip-cometkiwi22", action="store_true")
parser.add_argument("--skip-xcomet", action="store_true")
parser.add_argument("--skip-blaser", action="store_true")
parser.add_argument("--skip-metricx", action="store_true")
parser.add_argument("--skip-cometkiwi23xxl", action="store_true")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("SCORING TRAINING DATA WITH ALL METRICS")
print("=" * 80)

train = pd.read_parquet(args.input)
print(f"Training set: {len(train)} rows, {train['doc_id'].nunique()} docs")
print(f"Language pairs: {train.groupby(['src_lang', 'tgt_lang']).size().to_dict()}")

# Clean text columns
train["src_text"] = train["src_text"].fillna("").astype(str)
train["tgt_text"] = train["tgt_text"].fillna("").astype(str)

gpus = 1 if torch.cuda.is_available() else 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if gpus:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Prepare COMET samples
comet_samples = [
    {"src": row["src_text"], "mt": row["tgt_text"]}
    for _, row in train.iterrows()
]

# ---------------------------------------------------------------------------
# 2. CometKiwi-22
# ---------------------------------------------------------------------------
if not args.skip_cometkiwi22:
    print(f"\n--- CometKiwi-22 ({len(train)} samples) ---")
    from comet import download_model, load_from_checkpoint

    local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
    model_path = local_ckpt if os.path.exists(local_ckpt) else download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    start = time.time()
    output = model.predict(comet_samples, batch_size=args.batch_size, gpus=gpus, num_workers=4 if gpus else 2)
    train["cometkiwi22_score"] = output["scores"]
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({len(train)/elapsed:.1f} samples/s)")
    print(f"  Score range: [{min(output['scores']):.4f}, {max(output['scores']):.4f}]")
    del model
    if gpus:
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 3. xCOMET-XL
# ---------------------------------------------------------------------------
if not args.skip_xcomet:
    try:
        print(f"\n--- xCOMET-XL ({len(train)} samples) ---")
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)

        start = time.time()
        output = model.predict(comet_samples, batch_size=args.batch_size, gpus=gpus, num_workers=4 if gpus else 2)
        train["xcomet_score"] = output["scores"]
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(train)/elapsed:.1f} samples/s)")
        print(f"  Score range: [{min(output['scores']):.4f}, {max(output['scores']):.4f}]")
        del model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  xCOMET-XL failed: {e}")

# ---------------------------------------------------------------------------
# 4. BLASER-2 QE (text-text mode)
# ---------------------------------------------------------------------------
if not args.skip_blaser:
    try:
        print(f"\n--- BLASER-2 QE ({len(train)} samples) ---")
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
            "ja": "jpn_Jpan", "cs": "ces_Latn", "uk": "ukr_Cyrl",
            "ru": "rus_Cyrl", "ar": "arb_Arab", "is": "isl_Latn",
            "es": "spa_Latn", "et": "est_Latn", "hi": "hin_Deva",
            "it": "ita_Latn", "sr": "srp_Cyrl",
        }

        blaser_scores = np.zeros(len(train))
        sonar_cosine = np.zeros(len(train))
        start = time.time()

        for (src_lang, tgt_lang), group in train.groupby(["src_lang", "tgt_lang"]):
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
                    cos = torch.nn.functional.cosine_similarity(src_emb, tgt_emb, dim=-1)
                    sonar_cosine[batch_idx] = cos.cpu().numpy()

        train["blaser_score"] = blaser_scores
        train["sonar_cosine"] = sonar_cosine
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(train)/elapsed:.1f} samples/s)")
        del text_encoder, blaser_qe
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  BLASER failed: {e}")

# ---------------------------------------------------------------------------
# 5. MetricX-24
# ---------------------------------------------------------------------------
if not args.skip_metricx:
    try:
        print(f"\n--- MetricX-24-Hybrid-XXL ({len(train)} samples) ---")
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

        mx_batch = 16  # Smaller batch for XXL model
        metricx_scores = []
        start = time.time()

        for i in tqdm(range(0, len(train), mx_batch), desc="MetricX"):
            batch = train.iloc[i:i + mx_batch]
            input_texts = [
                f"source: {s} candidate: {m}"
                for s, m in zip(batch["src_text"], batch["tgt_text"])
            ]
            all_ids = []
            for text in input_texts:
                ids = metricx_tokenizer(text, max_length=1536, truncation=True)["input_ids"]
                all_ids.append(ids[:-1])  # Remove EOS

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

        train["metricx_error"] = metricx_scores
        train["metricx_score"] = 25.0 - np.array(metricx_scores)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(train)/elapsed:.1f} samples/s)")
        del metricx_model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  MetricX failed: {e}")

# ---------------------------------------------------------------------------
# 6. CometKiwi-23-XXL
# ---------------------------------------------------------------------------
if not args.skip_cometkiwi23xxl:
    try:
        print(f"\n--- CometKiwi-23-XXL ({len(train)} samples) ---")
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
        model = load_from_checkpoint(model_path)

        start = time.time()
        output = model.predict(
            comet_samples, batch_size=32, gpus=gpus, num_workers=4 if gpus else 2
        )
        ck23_scores = output.scores if hasattr(output, "scores") else output["scores"]
        train["cometkiwi23xxl_score"] = ck23_scores
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s ({len(train)/elapsed:.1f} samples/s)")
        del model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  CometKiwi-23-XXL failed: {e}")

# ---------------------------------------------------------------------------
# 7. Merge pre-computed LLM debate scores if available
# ---------------------------------------------------------------------------
llm_cache = "outputs/llm_debate_train.parquet"
if os.path.exists(llm_cache) and "llm_debate_score" not in train.columns:
    llm_df = pd.read_parquet(llm_cache)
    if "llm_debate_score" in llm_df.columns and len(llm_df) == len(train):
        train["llm_debate_score"] = llm_df["llm_debate_score"].values
        print(f"  Merged llm_debate_score from {llm_cache}")

# ---------------------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------------------
score_cols = [c for c in train.columns if c.endswith("_score") or c in ["metricx_error", "sonar_cosine"]]
print(f"\nScored columns: {[c for c in score_cols if c != 'score']}")
train.to_parquet(args.output, index=False)
print(f"Saved {len(train)} scored samples to {args.output}")

print("\n" + "=" * 80)
print("TRAINING DATA SCORING COMPLETE")
print("=" * 80)
