"""
Score test set with fine-tuned and pretrained CometKiwi-23-XXL.

Reads:
  data/test.jsonl — 48K test samples
  models/best_pairwise_*.ckpt — fine-tuned checkpoint (if exists)

Outputs:
  outputs/test_finetuned_scores.json — scores from fine-tuned model
  outputs/test_pretrained_scores.json — scores from pretrained CK-23-XXL

Run:
  poetry run python scripts/03_score_test.py
  poetry run python scripts/03_score_test.py --batch-size 32
"""

import os
import sys
import json
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssl_fix

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--skip-pretrained", action="store_true")
args = parser.parse_args()

os.makedirs("outputs", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load test data
# ---------------------------------------------------------------------------
print("=" * 80)
print("SCORING TEST SET — CometKiwi-23-XXL")
print("=" * 80)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


test_data = load_jsonl("data/test.jsonl")
print(f"Test set: {len(test_data)} rows")

lp_counts = {}
for row in test_data:
    lp = f"{row['src_lang']}-{row['tgt_lang']}"
    lp_counts[lp] = lp_counts.get(lp, 0) + 1
print(f"LPs: {lp_counts}")

if not torch.cuda.is_available():
    print("ERROR: GPU required.")
    sys.exit(1)

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ---------------------------------------------------------------------------
# 2. Score with pretrained CK-23-XXL
# ---------------------------------------------------------------------------
from comet import download_model, load_from_checkpoint

if not args.skip_pretrained:
    print(f"\n{'─'*70}")
    print(f"  Pretrained CometKiwi-23-XXL — {len(test_data)} samples")
    print(f"{'─'*70}")

    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    model = load_from_checkpoint(model_path)
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    comet_samples = [{"src": r["src_text"], "mt": r["tgt_text"]} for r in test_data]
    pretrained_scores = []
    start = time.time()

    with torch.no_grad():
        for i in range(0, len(comet_samples), args.batch_size):
            batch_samples = comet_samples[i:i + args.batch_size]
            batch = model.prepare_sample(batch_samples, stage="predict")
            input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch[0].items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.forward(**input_dict)
            pretrained_scores.extend(prediction.score.float().cpu().tolist())

            if (i // args.batch_size) % 50 == 0:
                print(f"    {i}/{len(comet_samples)} ({i*100//len(comet_samples)}%)")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Score range: [{min(pretrained_scores):.4f}, {max(pretrained_scores):.4f}]")
    print(f"  Mean: {np.mean(pretrained_scores):.4f}")

    with open("outputs/test_pretrained_scores.json", "w") as f:
        json.dump(pretrained_scores, f)
    print(f"  Saved outputs/test_pretrained_scores.json")

else:
    print("\n  [SKIP] Pretrained scoring (--skip-pretrained)")


# ---------------------------------------------------------------------------
# 3. Score with fine-tuned model
# ---------------------------------------------------------------------------
ckpt_files = sorted(glob.glob("models/best_pairwise_*.ckpt"))
if ckpt_files:
    ckpt_path = ckpt_files[-1]
    print(f"\n{'─'*70}")
    print(f"  Fine-tuned CK-23-XXL — {ckpt_path}")
    print(f"{'─'*70}")

    # Reload fresh model if we already scored pretrained (model is still in memory)
    if not args.skip_pretrained:
        del model
        torch.cuda.empty_cache()

    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    model = load_from_checkpoint(model_path)
    model = model.to(dtype=torch.bfloat16, device=device)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")

    if hasattr(model.encoder.model, "gradient_checkpointing_disable"):
        model.encoder.model.gradient_checkpointing_disable()

    comet_samples = [{"src": r["src_text"], "mt": r["tgt_text"]} for r in test_data]
    finetuned_scores = []
    start = time.time()

    with torch.no_grad():
        for i in range(0, len(comet_samples), args.batch_size):
            batch_samples = comet_samples[i:i + args.batch_size]
            batch = model.prepare_sample(batch_samples, stage="predict")
            input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch[0].items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.forward(**input_dict)
            finetuned_scores.extend(prediction.score.float().cpu().tolist())

            if (i // args.batch_size) % 50 == 0:
                print(f"    {i}/{len(comet_samples)} ({i*100//len(comet_samples)}%)")

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Score range: [{min(finetuned_scores):.4f}, {max(finetuned_scores):.4f}]")
    print(f"  Mean: {np.mean(finetuned_scores):.4f}")

    with open("outputs/test_finetuned_scores.json", "w") as f:
        json.dump(finetuned_scores, f)
    print(f"  Saved outputs/test_finetuned_scores.json")

    del model
    torch.cuda.empty_cache()
else:
    print(f"\n  [SKIP] No fine-tuned checkpoint found in models/")
    print(f"         Run 02_finetune_pairwise.py first.")


# ---------------------------------------------------------------------------
# 4. Also score dev for validation
# ---------------------------------------------------------------------------
dev_data = load_jsonl("data/dev.jsonl")
print(f"\n{'─'*70}")
print(f"  Scoring DEV for validation — {len(dev_data)} samples")
print(f"{'─'*70}")

# Use whichever model is best — prefer fine-tuned
if ckpt_files:
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    model = load_from_checkpoint(model_path)
    model = model.to(dtype=torch.bfloat16, device=device)
    state_dict = torch.load(ckpt_files[-1], map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    label = "finetuned"
else:
    model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
    model = load_from_checkpoint(model_path)
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()
    label = "pretrained"

if hasattr(model.encoder.model, "gradient_checkpointing_disable"):
    model.encoder.model.gradient_checkpointing_disable()

comet_samples = [{"src": r["src_text"], "mt": r["tgt_text"]} for r in dev_data]
dev_scores = []

with torch.no_grad():
    for i in range(0, len(comet_samples), args.batch_size):
        batch_samples = comet_samples[i:i + args.batch_size]
        batch = model.prepare_sample(batch_samples, stage="predict")
        input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch[0].items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.forward(**input_dict)
        dev_scores.extend(prediction.score.float().cpu().tolist())

with open("outputs/dev_scores.json", "w") as f:
    json.dump(dev_scores, f)

# Compute dev tau for validation
from scipy import stats
from collections import defaultdict

doc_scores = defaultdict(lambda: {"pred": [], "gold": []})
for row, pred in zip(dev_data, dev_scores):
    doc_scores[row["doc_id"]]["pred"].append(pred)
    doc_scores[row["doc_id"]]["gold"].append(row["score"])

taus = []
for doc_id, vals in doc_scores.items():
    if len(vals["pred"]) < 2:
        continue
    tau, _ = stats.kendalltau(vals["pred"], vals["gold"])
    if not np.isnan(tau):
        taus.append(tau)

dev_tau = np.mean(taus) if taus else 0.0
print(f"  Dev per-source tau ({label}): {dev_tau:.4f}")
print(f"  (Organizers' COMET baseline: 0.346)")

del model
torch.cuda.empty_cache()

print("\nScoring complete. Run: poetry run python scripts/04_generate_submission.py")
