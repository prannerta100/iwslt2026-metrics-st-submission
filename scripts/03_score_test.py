"""
Score test and dev with fine-tuned CometKiwi-22 + pretrained baseline.

Reads:
  data/test.jsonl, data/dev.jsonl
  models/best_pairwise_*.ckpt (fine-tuned checkpoint)

Outputs:
  outputs/test_finetuned_scores.json
  outputs/test_pretrained_scores.json
  outputs/dev_scores.json

Run:
  poetry run python scripts/03_score_test.py
  poetry run python scripts/03_score_test.py --batch-size 128
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
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--skip-pretrained", action="store_true")
args = parser.parse_args()

os.makedirs("outputs", exist_ok=True)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def score_all(model, data, batch_size, device):
    """Score all rows using model.predict() — the standard COMET inference path."""
    comet_samples = [{"src": r["src_text"], "mt": r["tgt_text"]} for r in data]
    gpus = 1 if device.type == "cuda" else 0
    output = model.predict(comet_samples, batch_size=batch_size, gpus=gpus, num_workers=4 if gpus else 0)
    return output.scores


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
print("=" * 80)
print("SCORING — CometKiwi-22")
print("=" * 80)

test_data = load_jsonl("data/test.jsonl")
dev_data = load_jsonl("data/dev.jsonl")
print(f"Test: {len(test_data)} rows")
print(f"Dev: {len(dev_data)} rows")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ---------------------------------------------------------------------------
# 1. Pretrained CometKiwi-22 (baseline)
# ---------------------------------------------------------------------------
from comet import download_model, load_from_checkpoint

if not args.skip_pretrained:
    print(f"\n{'─'*70}")
    print(f"  Pretrained CometKiwi-22 — {len(test_data)} test samples")
    print(f"{'─'*70}")

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    start = time.time()
    pretrained_scores = score_all(model, test_data, args.batch_size, device)
    print(f"  Test done in {time.time()-start:.0f}s")
    print(f"  Score range: [{min(pretrained_scores):.4f}, {max(pretrained_scores):.4f}]")

    with open("outputs/test_pretrained_scores.json", "w") as f:
        json.dump(list(pretrained_scores), f)
    print(f"  Saved outputs/test_pretrained_scores.json")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 2. Fine-tuned CometKiwi-22 (primary)
# ---------------------------------------------------------------------------
ckpt_files = sorted(glob.glob("models/best_pairwise_*.ckpt"))
if ckpt_files:
    ckpt_path = ckpt_files[-1]
    print(f"\n{'─'*70}")
    print(f"  Fine-tuned CometKiwi-22 — {ckpt_path}")
    print(f"{'─'*70}")

    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)
    print(f"  Loaded checkpoint")

    # Score test
    start = time.time()
    finetuned_scores = score_all(model, test_data, args.batch_size, device)
    print(f"  Test done in {time.time()-start:.0f}s")
    print(f"  Score range: [{min(finetuned_scores):.4f}, {max(finetuned_scores):.4f}]")

    with open("outputs/test_finetuned_scores.json", "w") as f:
        json.dump(list(finetuned_scores), f)
    print(f"  Saved outputs/test_finetuned_scores.json")

    # Score dev for validation
    print(f"\n  Scoring dev ({len(dev_data)} samples)...")
    dev_scores = score_all(model, dev_data, args.batch_size, device)
    with open("outputs/dev_scores.json", "w") as f:
        json.dump(list(dev_scores), f)

    # Compute dev tau
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
    print(f"  Dev per-source tau (fine-tuned): {dev_tau:.4f}")
    print(f"  (Organizers' COMET baseline: 0.346)")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
else:
    print(f"\n  [SKIP] No fine-tuned checkpoint in models/")
    print(f"         Run 02_finetune_pairwise.py first.")

    # Fall back: score dev with pretrained
    if not args.skip_pretrained:
        print(f"\n  Scoring dev with pretrained for validation...")
        model_path = download_model("Unbabel/wmt22-cometkiwi-da")
        model = load_from_checkpoint(model_path)
        dev_scores = score_all(model, dev_data, args.batch_size, device)
        with open("outputs/dev_scores.json", "w") as f:
            json.dump(list(dev_scores), f)
        del model

print("\nScoring complete. Run: poetry run python scripts/04_generate_submission.py")
