"""
Step 1: Explore the IWSLT 2026 Metrics Shared Task dataset.

This script loads the dataset (streaming first to check schema, then full for stats)
and produces a comprehensive summary of what we're working with.

Run: python scripts/01_explore_data.py
"""

import os
import json
import sys
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# 1. Streaming peek — verify schema without downloading everything
# ---------------------------------------------------------------------------
print("=" * 80)
print("STEP 1: Streaming peek at dataset schema")
print("=" * 80)

from datasets import load_dataset

DATASET_ID = "maikezu/iwslt2026-metrics-shared-train-dev"

# Stream just a few examples to verify the schema
# Use select_columns to avoid decoding audio (which requires torchcodec)
try:
    ds_stream = load_dataset(DATASET_ID, split="dev", streaming=True)
    first_example = next(iter(ds_stream))

    print("\nColumn names and types:")
    for key, value in first_example.items():
        if key == "audio":
            if isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
                if "sampling_rate" in value:
                    print(f"    sampling_rate: {value['sampling_rate']}")
                if "array" in value:
                    import numpy as np
                    arr = np.array(value["array"])
                    print(f"    array shape: {arr.shape}, dtype: {arr.dtype}")
                    print(f"    duration: {len(arr) / value['sampling_rate']:.2f}s")
            else:
                print(f"  {key}: {type(value).__name__}")
        else:
            print(f"  {key}: {type(value).__name__} = {repr(value)[:100]}")

    print("\n\nFirst example (without audio array):")
    for key, value in first_example.items():
        if key == "audio":
            print(f"  {key}: [audio data, sr={value.get('sampling_rate', '?')}]")
        else:
            print(f"  {key}: {repr(value)[:200]}")
except Exception as e:
    print(f"\nStreaming peek failed: {e}")
    print("Will proceed with non-audio columns...")


# ---------------------------------------------------------------------------
# 2. Load text-only columns for full stats (skip audio to save memory/time)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: Loading text columns for all splits")
print("=" * 80)

# We remove the audio column to avoid downloading all audio data
# Just load text + scores for exploration
splits_to_load = ["train", "train_synthetic", "dev"]
all_data = {}

for split_name in splits_to_load:
    print(f"\nLoading {split_name}...")
    ds = load_dataset(DATASET_ID, split=split_name)
    # Remove audio column to free memory
    ds_text = ds.remove_columns(["audio"])
    all_data[split_name] = ds_text
    print(f"  {split_name}: {len(ds_text)} examples")
    print(f"  Columns: {ds_text.column_names}")


# ---------------------------------------------------------------------------
# 3. Score distribution analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 3: Score distributions")
print("=" * 80)

for split_name, ds_text in all_data.items():
    scores = ds_text["score"]
    import numpy as np
    scores_arr = np.array(scores)

    print(f"\n--- {split_name} ({len(scores)} examples) ---")
    print(f"  Min:    {scores_arr.min():.4f}")
    print(f"  Max:    {scores_arr.max():.4f}")
    print(f"  Mean:   {scores_arr.mean():.4f}")
    print(f"  Median: {np.median(scores_arr):.4f}")
    print(f"  Std:    {scores_arr.std():.4f}")
    print(f"  25th:   {np.percentile(scores_arr, 25):.4f}")
    print(f"  75th:   {np.percentile(scores_arr, 75):.4f}")

    # Histogram bins
    if scores_arr.max() > 1.5:
        # Likely 0-100 scale
        bins = list(range(0, 110, 10))
        print(f"  Scale: 0-100")
    else:
        # Likely 0-1 scale
        bins = [i / 10 for i in range(11)]
        print(f"  Scale: 0-1")

    counts, edges = np.histogram(scores_arr, bins=bins)
    print(f"  Histogram:")
    for i in range(len(counts)):
        bar = "#" * (counts[i] * 40 // max(counts))
        print(f"    [{edges[i]:6.1f}, {edges[i+1]:6.1f}): {counts[i]:5d} {bar}")


# ---------------------------------------------------------------------------
# 4. Language pair analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 4: Language pairs")
print("=" * 80)

for split_name, ds_text in all_data.items():
    lang_pairs = Counter()
    for src, tgt in zip(ds_text["src_lang"], ds_text["tgt_lang"]):
        lang_pairs[f"{src} -> {tgt}"] += 1

    print(f"\n--- {split_name} ---")
    for pair, count in lang_pairs.most_common():
        print(f"  {pair}: {count}")


# ---------------------------------------------------------------------------
# 5. Domain analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 5: Domains")
print("=" * 80)

for split_name, ds_text in all_data.items():
    domains = Counter(ds_text["domain"])
    print(f"\n--- {split_name} ---")
    for domain, count in domains.most_common():
        print(f"  {domain}: {count}")


# ---------------------------------------------------------------------------
# 6. Translation system analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 6: Translation systems")
print("=" * 80)

for split_name, ds_text in all_data.items():
    systems = Counter(ds_text["tgt_system"])
    print(f"\n--- {split_name} ({len(systems)} systems) ---")
    for sys_name, count in systems.most_common():
        print(f"  {sys_name}: {count}")


# ---------------------------------------------------------------------------
# 7. Source text system (human vs ASR)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 7: Source text type (human vs ASR)")
print("=" * 80)

for split_name, ds_text in all_data.items():
    src_types = Counter(ds_text["src_text_system"])
    print(f"\n--- {split_name} ---")
    for src_type, count in src_types.most_common():
        print(f"  {src_type}: {count}")


# ---------------------------------------------------------------------------
# 8. Score distribution per language pair (for dev set)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 8: Score by language pair (dev set)")
print("=" * 80)

ds_dev = all_data["dev"]
lang_scores = defaultdict(list)
for src, tgt, score in zip(ds_dev["src_lang"], ds_dev["tgt_lang"], ds_dev["score"]):
    lang_scores[f"{src} -> {tgt}"].append(score)

for pair, scores in sorted(lang_scores.items()):
    scores_arr = np.array(scores)
    print(f"\n{pair} ({len(scores)} examples):")
    print(f"  Mean: {scores_arr.mean():.2f}, Median: {np.median(scores_arr):.2f}")
    print(f"  Std: {scores_arr.std():.2f}")
    print(f"  Range: [{scores_arr.min():.2f}, {scores_arr.max():.2f}]")


# ---------------------------------------------------------------------------
# 9. Score distribution per system (dev set) — key for understanding Kendall Tau
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 9: Score by system (dev set)")
print("=" * 80)

sys_scores = defaultdict(list)
for sys_name, score in zip(ds_dev["tgt_system"], ds_dev["score"]):
    sys_scores[sys_name].append(score)

for sys_name in sorted(sys_scores.keys()):
    scores_arr = np.array(sys_scores[sys_name])
    print(f"\n{sys_name} ({len(scores_arr)} examples):")
    print(f"  Mean: {scores_arr.mean():.2f}, Std: {scores_arr.std():.2f}")


# ---------------------------------------------------------------------------
# 10. doc_id analysis — can we extract document context?
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 10: Document ID structure (dev set, first 20)")
print("=" * 80)

doc_ids = ds_dev["doc_id"][:20]
for d in doc_ids:
    print(f"  {d}")

# Check how many unique doc_ids
unique_docs = set(ds_dev["doc_id"])
print(f"\nTotal unique doc_ids in dev: {len(unique_docs)}")
print(f"Total examples in dev: {len(ds_dev)}")
print(f"Avg segments per doc: {len(ds_dev) / len(unique_docs):.1f}")


# ---------------------------------------------------------------------------
# 11. Text length analysis
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 11: Text lengths (dev set)")
print("=" * 80)

src_lens = [len(t.split()) for t in ds_dev["src_text"]]
tgt_lens = [len(t.split()) for t in ds_dev["tgt_text"]]

src_arr = np.array(src_lens)
tgt_arr = np.array(tgt_lens)

print(f"Source text (words): mean={src_arr.mean():.1f}, median={np.median(src_arr):.1f}, "
      f"min={src_arr.min()}, max={src_arr.max()}")
print(f"Target text (words): mean={tgt_arr.mean():.1f}, median={np.median(tgt_arr):.1f}, "
      f"min={tgt_arr.min()}, max={tgt_arr.max()}")


# ---------------------------------------------------------------------------
# 12. Sample some low-scoring and high-scoring examples
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 12: Sample examples")
print("=" * 80)

# Sort dev by score
indices = np.argsort(np.array(ds_dev["score"]))

print("\n--- 5 LOWEST scoring examples (dev) ---")
for idx in indices[:5]:
    idx = int(idx)
    print(f"\n  Score: {ds_dev[idx]['score']:.2f}")
    print(f"  Lang: {ds_dev[idx]['src_lang']} -> {ds_dev[idx]['tgt_lang']}")
    print(f"  System: {ds_dev[idx]['tgt_system']}")
    print(f"  Source: {ds_dev[idx]['src_text'][:150]}...")
    print(f"  Target: {ds_dev[idx]['tgt_text'][:150]}...")

print("\n--- 5 HIGHEST scoring examples (dev) ---")
for idx in indices[-5:]:
    idx = int(idx)
    print(f"\n  Score: {ds_dev[idx]['score']:.2f}")
    print(f"  Lang: {ds_dev[idx]['src_lang']} -> {ds_dev[idx]['tgt_lang']}")
    print(f"  System: {ds_dev[idx]['tgt_system']}")
    print(f"  Source: {ds_dev[idx]['src_text'][:150]}...")
    print(f"  Target: {ds_dev[idx]['tgt_text'][:150]}...")

print("\n--- 5 MEDIAN scoring examples (dev) ---")
mid = len(indices) // 2
for idx in indices[mid-2:mid+3]:
    idx = int(idx)
    print(f"\n  Score: {ds_dev[idx]['score']:.2f}")
    print(f"  Lang: {ds_dev[idx]['src_lang']} -> {ds_dev[idx]['tgt_lang']}")
    print(f"  System: {ds_dev[idx]['tgt_system']}")
    print(f"  Source: {ds_dev[idx]['src_text'][:150]}...")
    print(f"  Target: {ds_dev[idx]['tgt_text'][:150]}...")


print("\n" + "=" * 80)
print("EXPLORATION COMPLETE")
print("=" * 80)
