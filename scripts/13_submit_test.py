"""
Score test set with ALL available metrics for IWSLT 2026 Metrics Shared Task.

Loads maikezu/iwslt2026-metrics-shared-test, scores with every model:
  1. CometKiwi-22 (pretrained baseline)
  2. CometKiwi-22 fine-tuned MSE (models/cometkiwi_finetuned/*.ckpt)
  3. CometKiwi-22 fine-tuned pairwise (models/cometkiwi_pairwise/*.ckpt)
  4. CometKiwi-23-XXL (pretrained)
  5. CometKiwi-23-XXL fine-tuned pairwise (models/cometkiwi23xxl_pairwise/*.ckpt)
  6. xCOMET-XL (pretrained)
  7. MetricX-24-Hybrid-XXL
  8. BLASER-2 QE
  9. LLM debate scores (from cached parquet)

Outputs: submission/test_predictions.parquet with ALL score columns.
Then 15_final_submission.py trains LightGBM on these and picks best/2nd-best.

Run on GPU VM:
  poetry run python scripts/13_submit_test.py
  poetry run python scripts/13_submit_test.py --batch-size 64 --skip-metricx
"""

import os
import sys
import json
import glob
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
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------------------------
print("=" * 80)
print("IWSLT 2026 METRICS SHARED TASK — TEST SET SCORING")
print("=" * 80)

from datasets import load_dataset
ds = load_dataset("maikezu/iwslt2026-metrics-shared-test")
if "audio" in ds["test"].column_names:
    ds["test"] = ds["test"].remove_columns(["audio"])
test = ds["test"].to_pandas()

print(f"Test set: {len(test)} rows")
print(f"Language pairs: {test.groupby(['src_lang', 'tgt_lang']).size().to_dict()}")
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

scored = []


def report_score(name, scores):
    scored.append(name)
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}], "
          f"mean={np.mean(scores):.4f}")


# ---------------------------------------------------------------------------
# 2. CometKiwi-22 (pretrained baseline)
# ---------------------------------------------------------------------------
if not args.skip_cometkiwi22:
    print(f"\n{'─'*70}")
    print(f"  CometKiwi-22 (pretrained) — {len(test)} samples")
    print(f"{'─'*70}")
    from comet import download_model, load_from_checkpoint

    local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
    model_path = local_ckpt if os.path.exists(local_ckpt) else download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    start = time.time()
    output = model.predict(comet_samples, batch_size=args.batch_size, gpus=gpus, num_workers=4 if gpus else 2)
    test["cometkiwi22_score"] = output["scores"]
    print(f"  Done in {time.time()-start:.1f}s")
    report_score("cometkiwi22_score", output["scores"])
    del model
    if gpus:
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 3. CometKiwi-22 fine-tuned MSE (from 03_finetune_cometkiwi.py)
#    Checkpoint: models/cometkiwi_finetuned/best-*.ckpt (Lightning format)
# ---------------------------------------------------------------------------
if not args.skip_finetuned:
    ckpt_dir = "models/cometkiwi_finetuned/"
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt"))) if os.path.isdir(ckpt_dir) else []
    if ckpts:
        ckpt_path = ckpts[-1]
        print(f"\n{'─'*70}")
        print(f"  CometKiwi-22 fine-tuned MSE — {ckpt_path}")
        print(f"{'─'*70}")
        try:
            from comet import download_model, load_from_checkpoint

            local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
            model_path = local_ckpt if os.path.exists(local_ckpt) else download_model("Unbabel/wmt22-cometkiwi-da")
            model = load_from_checkpoint(model_path)

            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)

            start = time.time()
            output = model.predict(comet_samples, batch_size=args.batch_size, gpus=gpus, num_workers=4 if gpus else 2)
            ft_scores = output.scores if hasattr(output, "scores") else output["scores"]
            test["finetuned_score"] = ft_scores
            print(f"  Done in {time.time()-start:.1f}s")
            report_score("finetuned_score", ft_scores)
            del model
            if gpus:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print(f"\n  [SKIP] No fine-tuned MSE checkpoint in {ckpt_dir}")

# ---------------------------------------------------------------------------
# 4. CometKiwi-22 fine-tuned PAIRWISE (from 03b_finetune_pairwise.py)
#    Checkpoint: models/cometkiwi_pairwise/best-*.ckpt (raw state_dict)
# ---------------------------------------------------------------------------
if not args.skip_finetuned:
    ckpt_dir = "models/cometkiwi_pairwise/"
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt"))) if os.path.isdir(ckpt_dir) else []
    if ckpts:
        ckpt_path = ckpts[-1]
        print(f"\n{'─'*70}")
        print(f"  CometKiwi-22 fine-tuned PAIRWISE — {ckpt_path}")
        print(f"{'─'*70}")
        try:
            from comet import download_model, load_from_checkpoint

            local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
            model_path = local_ckpt if os.path.exists(local_ckpt) else download_model("Unbabel/wmt22-cometkiwi-da")
            model = load_from_checkpoint(model_path)

            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                model.load_state_dict(state_dict["state_dict"])
            else:
                model.load_state_dict(state_dict)

            start = time.time()
            output = model.predict(comet_samples, batch_size=args.batch_size, gpus=gpus, num_workers=4 if gpus else 2)
            pw_scores = output.scores if hasattr(output, "scores") else output["scores"]
            test["pairwise_score"] = pw_scores
            print(f"  Done in {time.time()-start:.1f}s")
            report_score("pairwise_score", pw_scores)
            del model
            if gpus:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print(f"\n  [SKIP] No pairwise checkpoint in {ckpt_dir}")

# ---------------------------------------------------------------------------
# 5. xCOMET-XL (pretrained)
# ---------------------------------------------------------------------------
if not args.skip_xcomet:
    print(f"\n{'─'*70}")
    print(f"  xCOMET-XL — {len(test)} samples")
    print(f"{'─'*70}")
    try:
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/XCOMET-XL")
        model = load_from_checkpoint(model_path)

        start = time.time()
        output = model.predict(comet_samples, batch_size=args.batch_size, gpus=gpus, num_workers=4 if gpus else 2)
        test["xcomet_score"] = output["scores"]
        print(f"  Done in {time.time()-start:.1f}s")
        report_score("xcomet_score", output["scores"])
        del model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")

# ---------------------------------------------------------------------------
# 6. CometKiwi-23-XXL (pretrained)
# ---------------------------------------------------------------------------
if not args.skip_cometkiwi23xxl:
    print(f"\n{'─'*70}")
    print(f"  CometKiwi-23-XXL (pretrained) — {len(test)} samples")
    print(f"{'─'*70}")
    try:
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
        model = load_from_checkpoint(model_path)

        start = time.time()
        output = model.predict(comet_samples, batch_size=32, gpus=gpus, num_workers=4 if gpus else 2)
        ck23_scores = output.scores if hasattr(output, "scores") else output["scores"]
        test["cometkiwi23xxl_score"] = ck23_scores
        print(f"  Done in {time.time()-start:.1f}s")
        report_score("cometkiwi23xxl_score", ck23_scores)
        del model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")

# ---------------------------------------------------------------------------
# 7. CometKiwi-23-XXL fine-tuned PAIRWISE (from 12_finetune_cometkiwi23xxl.py)
#    Checkpoint: models/cometkiwi23xxl_pairwise/best-*.ckpt (raw state_dict)
# ---------------------------------------------------------------------------
if not args.skip_cometkiwi23xxl and not args.skip_finetuned:
    ckpt_dir = "models/cometkiwi23xxl_pairwise/"
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt"))) if os.path.isdir(ckpt_dir) else []
    if ckpts:
        ckpt_path = ckpts[-1]
        print(f"\n{'─'*70}")
        print(f"  CometKiwi-23-XXL fine-tuned PAIRWISE — {ckpt_path}")
        print(f"{'─'*70}")
        try:
            from comet import download_model, load_from_checkpoint

            model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
            model = load_from_checkpoint(model_path)

            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "state_dict" in state_dict:
                model.load_state_dict(state_dict["state_dict"])
            else:
                model.load_state_dict(state_dict)

            start = time.time()
            output = model.predict(comet_samples, batch_size=16, gpus=gpus, num_workers=4 if gpus else 2)
            xxl_ft_scores = output.scores if hasattr(output, "scores") else output["scores"]
            test["cometkiwi23xxl_finetuned_score"] = xxl_ft_scores
            print(f"  Done in {time.time()-start:.1f}s")
            report_score("cometkiwi23xxl_finetuned_score", xxl_ft_scores)
            del model
            if gpus:
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  FAILED: {e}")
    else:
        print(f"\n  [SKIP] No CK-23-XXL pairwise checkpoint in {ckpt_dir}")

# ---------------------------------------------------------------------------
# 8. MetricX-24-Hybrid-XXL
# ---------------------------------------------------------------------------
if not args.skip_metricx:
    print(f"\n{'─'*70}")
    print(f"  MetricX-24-Hybrid-XXL — {len(test)} samples")
    print(f"{'─'*70}")
    try:
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
        print(f"  Done in {time.time()-start:.1f}s")
        report_score("metricx_score", test["metricx_score"].tolist())
        del metricx_model
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")

# ---------------------------------------------------------------------------
# 9. BLASER-2 QE (text-text)
# ---------------------------------------------------------------------------
if not args.skip_blaser:
    print(f"\n{'─'*70}")
    print(f"  BLASER-2 QE — {len(test)} samples")
    print(f"{'─'*70}")
    try:
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
        print(f"  Done in {time.time()-start:.1f}s")
        report_score("blaser_score", blaser_scores.tolist())
        del text_encoder, blaser_qe
        if gpus:
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAILED: {e}")

# ---------------------------------------------------------------------------
# 10. LLM debate scores (from cached parquet — no re-inference)
# ---------------------------------------------------------------------------
llm_test_cache = "outputs/llm_debate_test.parquet"
if os.path.exists(llm_test_cache) and "llm_debate_score" not in test.columns:
    llm_df = pd.read_parquet(llm_test_cache)
    if "llm_debate_score" in llm_df.columns and len(llm_df) == len(test):
        test["llm_debate_score"] = llm_df["llm_debate_score"].values
        print(f"\n  Merged llm_debate_score from {llm_test_cache}")
        report_score("llm_debate_score", test["llm_debate_score"].tolist())
    else:
        print(f"\n  WARNING: {llm_test_cache} exists but row count mismatch "
              f"({len(llm_df)} vs {len(test)})")
else:
    if "llm_debate_score" not in test.columns:
        print(f"\n  [SKIP] No cached LLM debate scores at {llm_test_cache}")
        print(f"         Run: WEBEX_TOKEN=... python scripts/16_llm_baseline.py --dataset test")

# ---------------------------------------------------------------------------
# 11. Summary & Save
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SCORING COMPLETE")
print("=" * 80)

signal_cols = [c for c in test.columns if c.endswith("_score") and c != "score"]
print(f"\nScored signals ({len(signal_cols)}): {signal_cols}")

for col in signal_cols:
    vals = test[col].values
    print(f"  {col:<35} mean={vals.mean():.4f}  std={vals.std():.4f}  "
          f"[{vals.min():.4f}, {vals.max():.4f}]")

# Save full predictions parquet (15_final_submission.py reads this)
output_file = os.path.join(args.output_dir, "test_predictions.parquet")
test.to_parquet(output_file, index=False)
print(f"\nSaved {len(test)} rows x {len(signal_cols)} signals to {output_file}")
print(f"\nNext step: poetry run python scripts/15_final_submission.py")
