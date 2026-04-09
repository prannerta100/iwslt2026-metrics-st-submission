"""
Lightweight data exploration: fetch a sample via the HuggingFace API
without downloading all 71GB of audio.

Uses the datasets viewer API (first 100 rows per split).
"""

import os
import json
import requests
from collections import Counter, defaultdict
import numpy as np

os.environ.setdefault("SSL_CERT_FILE", "/tmp/ca-bundle.crt")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/tmp/ca-bundle.crt")

DATASET_ID = "maikezu/iwslt2026-metrics-shared-train-dev"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def fetch_rows(split, offset=0, length=100, max_retries=5):
    """Fetch rows from the HuggingFace dataset viewer API with retries."""
    import time
    url = f"https://datasets-server.huggingface.co/rows?dataset={DATASET_ID}&config=default&split={split}&offset={offset}&length={length}"
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["rows"], data.get("num_rows_total", None)
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise

def fetch_all_text_rows(split, max_rows=None):
    """Fetch all rows from a split (text columns only, no audio)."""
    rows = []
    offset = 0
    batch_size = 100
    total = None
    while True:
        batch, total_reported = fetch_rows(split, offset=offset, length=batch_size)
        if total is None and total_reported:
            total = total_reported
            if max_rows:
                total = min(total, max_rows)
            print(f"  Total rows in {split}: {total_reported}" + (f" (fetching {total})" if max_rows else ""))

        for item in batch:
            row = item["row"]
            # Remove audio data (just keep path)
            if "audio" in row:
                del row["audio"]
            rows.append(row)

        offset += len(batch)
        if len(batch) < batch_size or (max_rows and offset >= max_rows):
            break

        if offset % 1000 == 0:
            print(f"    Fetched {offset}/{total}...")

    return rows

# ---------------------------------------------------------------------------
# Fetch data
# ---------------------------------------------------------------------------
print("=" * 80)
print("Fetching dataset rows via HuggingFace API")
print("=" * 80)

# Fetch all dev rows (5,556 — manageable)
print("\nFetching dev split...")
dev_rows = fetch_all_text_rows("dev")

# Fetch a large sample of train (first 5000 rows for stats)
print("\nFetching train split (sample)...")
train_rows = fetch_all_text_rows("train", max_rows=5000)

# Fetch all synthetic rows (7,000)
print("\nFetching train_synthetic split...")
synth_rows = fetch_all_text_rows("train_synthetic")

all_data = {"train (sample)": train_rows, "train_synthetic": synth_rows, "dev": dev_rows}

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

for split_name, rows in all_data.items():
    print(f"\n{'='*80}")
    print(f"SPLIT: {split_name} ({len(rows)} rows)")
    print(f"{'='*80}")

    # Columns
    print(f"\nColumns: {list(rows[0].keys())}")

    # Score distribution
    scores = np.array([r["score"] for r in rows])
    print(f"\n--- Score Distribution ---")
    print(f"  Min:    {scores.min():.4f}")
    print(f"  Max:    {scores.max():.4f}")
    print(f"  Mean:   {scores.mean():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    print(f"  Std:    {scores.std():.4f}")
    print(f"  25th:   {np.percentile(scores, 25):.4f}")
    print(f"  75th:   {np.percentile(scores, 75):.4f}")

    if scores.max() > 1.5:
        bins = list(range(0, 110, 10))
        scale = "0-100"
    else:
        bins = [i / 10 for i in range(11)]
        scale = "0-1"
    print(f"  Scale: {scale}")

    counts, edges = np.histogram(scores, bins=bins)
    print(f"  Histogram:")
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(len(counts)):
        bar = "#" * (counts[i] * 40 // max_count)
        print(f"    [{edges[i]:6.1f}, {edges[i+1]:6.1f}): {counts[i]:5d} {bar}")

    # Language pairs
    lang_pairs = Counter(f"{r['src_lang']} -> {r['tgt_lang']}" for r in rows)
    print(f"\n--- Language Pairs ---")
    for pair, count in lang_pairs.most_common():
        print(f"  {pair}: {count}")

    # Domains
    domains = Counter(r["domain"] for r in rows)
    print(f"\n--- Domains ---")
    for domain, count in domains.most_common():
        print(f"  {domain}: {count}")

    # Systems
    systems = Counter(r["tgt_system"] for r in rows)
    print(f"\n--- Translation Systems ({len(systems)} total) ---")
    for sys_name, count in systems.most_common():
        print(f"  {sys_name}: {count}")

    # Source text system
    src_types = Counter(r["src_text_system"] for r in rows)
    print(f"\n--- Source Text Type ---")
    for src_type, count in src_types.most_common():
        print(f"  {src_type}: {count}")

# ---------------------------------------------------------------------------
# Dev set deep dive
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print("DEV SET DEEP DIVE")
print(f"{'='*80}")

# Score per language pair
print("\n--- Score by Language Pair ---")
lang_scores = defaultdict(list)
for r in dev_rows:
    lang_scores[f"{r['src_lang']} -> {r['tgt_lang']}"].append(r["score"])

for pair, scores in sorted(lang_scores.items()):
    s = np.array(scores)
    print(f"\n  {pair} ({len(scores)} examples):")
    print(f"    Mean: {s.mean():.2f}, Median: {np.median(s):.2f}, Std: {s.std():.2f}")
    print(f"    Range: [{s.min():.2f}, {s.max():.2f}]")

# Score per system
print("\n--- Score by System ---")
sys_scores = defaultdict(list)
for r in dev_rows:
    sys_scores[r["tgt_system"]].append(r["score"])

for sys_name in sorted(sys_scores.keys()):
    s = np.array(sys_scores[sys_name])
    print(f"\n  {sys_name} ({len(s)} examples):")
    print(f"    Mean: {s.mean():.2f}, Median: {np.median(s):.2f}, Std: {s.std():.2f}")

# Score per system per language pair
print("\n--- Score by System x Language Pair ---")
sys_lang_scores = defaultdict(lambda: defaultdict(list))
for r in dev_rows:
    pair = f"{r['src_lang']} -> {r['tgt_lang']}"
    sys_lang_scores[pair][r["tgt_system"]].append(r["score"])

for pair in sorted(sys_lang_scores.keys()):
    print(f"\n  {pair}:")
    for sys_name in sorted(sys_lang_scores[pair].keys()):
        s = np.array(sys_lang_scores[pair][sys_name])
        print(f"    {sys_name}: mean={s.mean():.2f}, n={len(s)}")

# Document structure
print("\n--- Document IDs (first 20) ---")
for r in dev_rows[:20]:
    print(f"  doc_id={r['doc_id']}, system={r['tgt_system']}, lang={r['src_lang']}->{r['tgt_lang']}")

unique_docs = set(r["doc_id"] for r in dev_rows)
print(f"\nUnique doc_ids: {len(unique_docs)}")
print(f"Avg segments per doc: {len(dev_rows) / len(unique_docs):.1f}")

# Text lengths
src_lens = [len(r["src_text"].split()) for r in dev_rows]
tgt_lens = [len(r["tgt_text"].split()) for r in dev_rows]
print(f"\n--- Text Lengths (dev, words) ---")
print(f"  Source: mean={np.mean(src_lens):.1f}, median={np.median(src_lens):.1f}, max={max(src_lens)}")
print(f"  Target: mean={np.mean(tgt_lens):.1f}, median={np.median(tgt_lens):.1f}, max={max(tgt_lens)}")

# Sample examples
print("\n--- 5 LOWEST scoring examples (dev) ---")
sorted_dev = sorted(dev_rows, key=lambda r: r["score"])
for r in sorted_dev[:5]:
    print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
    print(f"  Source: {r['src_text'][:150]}...")
    print(f"  Target: {r['tgt_text'][:150]}...")

print("\n--- 5 HIGHEST scoring examples (dev) ---")
for r in sorted_dev[-5:]:
    print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
    print(f"  Source: {r['src_text'][:150]}...")
    print(f"  Target: {r['tgt_text'][:150]}...")

print("\n--- 5 MEDIAN scoring examples (dev) ---")
mid = len(sorted_dev) // 2
for r in sorted_dev[mid-2:mid+3]:
    print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
    print(f"  Source: {r['src_text'][:150]}...")
    print(f"  Target: {r['tgt_text'][:150]}...")

print(f"\n{'='*80}")
print("EXPLORATION COMPLETE")
print(f"{'='*80}")
