"""
Download IWSLT 2026 Metrics data from HuggingFace and write JSONL files.

Output:
  data/train.jsonl  — ~33K rows (for fine-tuning)
  data/dev.jsonl    — ~5.5K rows (for evaluation, same format as organizers)
  data/test.jsonl   — ~48K rows (for submission scoring)

Format matches organizers exactly (one JSON dict per line):
  {"src_text": ..., "tgt_text": ..., "doc_id": ..., "score": ...,
   "src_lang": ..., "tgt_lang": ..., "tgt_system": ..., "audio_path": ...}
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssl_fix

os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset

os.makedirs("data", exist_ok=True)

FIELDS = ["audio_path", "src_text", "tgt_text", "tgt_system", "doc_id", "score", "src_lang", "tgt_lang"]


def write_jsonl(dataset, path, include_score=True):
    """Write dataset rows to JSONL, matching organizers' format."""
    fields = FIELDS if include_score else [f for f in FIELDS if f != "score"]
    count = 0
    with open(path, "w") as f:
        for row in dataset:
            obj = {}
            for field in fields:
                if field in row and row[field] is not None:
                    obj[field] = row[field]
                elif field == "audio_path":
                    obj[field] = row.get("audio_path", "")
            f.write(json.dumps(obj) + "\n")
            count += 1
    return count


# --- Train/Dev ---
print("Loading maikezu/iwslt2026-metrics-shared-train-dev...")
ds = load_dataset("maikezu/iwslt2026-metrics-shared-train-dev")

n_train = write_jsonl(ds["train"], "data/train.jsonl", include_score=True)
print(f"  data/train.jsonl: {n_train} rows")

n_dev = write_jsonl(ds["dev"], "data/dev.jsonl", include_score=True)
print(f"  data/dev.jsonl: {n_dev} rows")

# Print LP breakdown for train
train_lps = {}
for row in ds["train"]:
    lp = f"{row['src_lang']}-{row['tgt_lang']}"
    train_lps[lp] = train_lps.get(lp, 0) + 1
print(f"  Train LPs: {train_lps}")

dev_lps = {}
for row in ds["dev"]:
    lp = f"{row['src_lang']}-{row['tgt_lang']}"
    dev_lps[lp] = dev_lps.get(lp, 0) + 1
print(f"  Dev LPs: {dev_lps}")

# --- Test ---
print("\nLoading maikezu/iwslt2026-metrics-shared-test...")
ds_test = load_dataset("maikezu/iwslt2026-metrics-shared-test")
test_split = ds_test["test"] if "test" in ds_test else ds_test[list(ds_test.keys())[0]]

n_test = write_jsonl(test_split, "data/test.jsonl", include_score=False)
print(f"  data/test.jsonl: {n_test} rows")

test_lps = {}
for row in test_split:
    lp = f"{row['src_lang']}-{row['tgt_lang']}"
    test_lps[lp] = test_lps.get(lp, 0) + 1
print(f"  Test LPs: {test_lps}")

# --- Validate ---
print("\n--- Validation ---")
for path in ["data/train.jsonl", "data/dev.jsonl", "data/test.jsonl"]:
    with open(path) as f:
        lines = f.readlines()
    first = json.loads(lines[0])
    print(f"  {path}: {len(lines)} rows, keys={list(first.keys())}")
    assert "src_text" in first
    assert "tgt_text" in first
    assert "doc_id" in first
    assert "src_lang" in first
    assert "tgt_lang" in first

print("\nDone. All data files ready.")
