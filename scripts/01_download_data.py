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

# Remove audio column to avoid slow decoding during iteration
for split in ds:
    if "audio" in ds[split].column_names:
        ds[split] = ds[split].remove_columns(["audio"])

print(f"  Columns: {ds['train'].column_names}")

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

# Remove audio column
if "audio" in test_split.column_names:
    test_split = test_split.remove_columns(["audio"])

print(f"  Test columns: {test_split.column_names}")

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
    assert "src_text" in first, f"Missing src_text in {path}: {list(first.keys())}"
    assert "tgt_text" in first, f"Missing tgt_text in {path}: {list(first.keys())}"
    assert "doc_id" in first, f"Missing doc_id in {path}: {list(first.keys())}"
    assert "src_lang" in first, f"Missing src_lang in {path}: {list(first.keys())}"
    assert "tgt_lang" in first, f"Missing tgt_lang in {path}: {list(first.keys())}"
    # Check non-empty text
    assert len(first["src_text"]) > 0, f"Empty src_text in {path}"
    assert len(first["tgt_text"]) > 0, f"Empty tgt_text in {path}"

# Check train/dev have scores
with open("data/train.jsonl") as f:
    first_train = json.loads(f.readline())
assert "score" in first_train, "Training data missing 'score' field"

print("\nDone. All data files ready.")
