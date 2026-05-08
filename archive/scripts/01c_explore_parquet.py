"""
Data exploration using direct parquet download.
Downloads only the parquet files and reads text columns (skips audio bytes).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import numpy as np
from collections import Counter, defaultdict
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

DATASET_ID = "maikezu/iwslt2026-metrics-shared-train-dev"

# ---------------------------------------------------------------------------
# 1. List parquet files in the dataset
# ---------------------------------------------------------------------------
print("=" * 80)
print("STEP 1: Listing dataset files")
print("=" * 80)

api = HfApi()
files = api.list_repo_files(DATASET_ID, repo_type="dataset")
parquet_files = [f for f in files if f.endswith(".parquet")]
print(f"\nParquet files ({len(parquet_files)}):")
for f in sorted(parquet_files):
    print(f"  {f}")

# ---------------------------------------------------------------------------
# 2. Download and read parquet files (text columns only)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: Downloading and reading parquet files")
print("=" * 80)

TEXT_COLS = ["doc_id", "src_text", "src_text_system", "src_lang", "tgt_lang",
            "domain", "tgt_system", "tgt_text", "score", "audio_path"]

all_data = {}
for pf in sorted(parquet_files):
    split_name = pf.replace("data/", "").split("-")[0]

    print(f"\nDownloading {pf}...")
    local_path = hf_hub_download(
        DATASET_ID, pf, repo_type="dataset",
        token=os.environ.get("HF_TOKEN")
    )

    # Read only text columns (skip audio bytes)
    pf_obj = pq.ParquetFile(local_path)
    # Check available columns
    schema = pf_obj.schema_arrow
    available_cols = [c for c in TEXT_COLS if c in schema.names]
    table = pf_obj.read(columns=available_cols)
    df = table.to_pandas()

    if split_name not in all_data:
        all_data[split_name] = df
    else:
        import pandas as pd
        all_data[split_name] = pd.concat([all_data[split_name], df], ignore_index=True)

    print(f"  {split_name}: {len(df)} rows (cumulative: {len(all_data[split_name])})")

# ---------------------------------------------------------------------------
# 3. Analysis per split
# ---------------------------------------------------------------------------
for split_name in sorted(all_data.keys()):
    df = all_data[split_name]
    print(f"\n{'='*80}")
    print(f"SPLIT: {split_name} ({len(df)} rows)")
    print(f"{'='*80}")

    print(f"\nColumns: {list(df.columns)}")

    # Score distribution
    scores = df["score"].values
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
    max_count = max(counts) if max(counts) > 0 else 1
    print(f"  Histogram:")
    for i in range(len(counts)):
        bar = "#" * (counts[i] * 40 // max_count)
        print(f"    [{edges[i]:6.1f}, {edges[i+1]:6.1f}): {counts[i]:5d} {bar}")

    # Language pairs
    lang_pairs = Counter(df["src_lang"] + " -> " + df["tgt_lang"])
    print(f"\n--- Language Pairs ---")
    for pair, count in lang_pairs.most_common():
        print(f"  {pair}: {count}")

    # Domains
    domains = Counter(df["domain"])
    print(f"\n--- Domains ---")
    for domain, count in domains.most_common():
        print(f"  {domain}: {count}")

    # Systems
    systems = Counter(df["tgt_system"])
    print(f"\n--- Translation Systems ({len(systems)} total) ---")
    for sys_name, count in systems.most_common():
        print(f"  {sys_name}: {count}")

    # Source text system
    src_types = Counter(df["src_text_system"])
    print(f"\n--- Source Text Type ---")
    for src_type, count in src_types.most_common():
        print(f"  {src_type}: {count}")

# ---------------------------------------------------------------------------
# 4. Dev set deep dive
# ---------------------------------------------------------------------------
if "dev" in all_data:
    df_dev = all_data["dev"]
    print(f"\n{'='*80}")
    print("DEV SET DEEP DIVE")
    print(f"{'='*80}")

    # Score per language pair
    print("\n--- Score by Language Pair ---")
    for pair in sorted(df_dev.groupby(["src_lang", "tgt_lang"]).groups.keys()):
        mask = (df_dev["src_lang"] == pair[0]) & (df_dev["tgt_lang"] == pair[1])
        s = df_dev.loc[mask, "score"].values
        print(f"\n  {pair[0]} -> {pair[1]} ({len(s)} examples):")
        print(f"    Mean: {s.mean():.2f}, Median: {np.median(s):.2f}, Std: {s.std():.2f}")
        print(f"    Range: [{s.min():.2f}, {s.max():.2f}]")

    # Score per system
    print("\n--- Score by System ---")
    for sys_name in sorted(df_dev["tgt_system"].unique()):
        s = df_dev.loc[df_dev["tgt_system"] == sys_name, "score"].values
        print(f"  {sys_name} (n={len(s)}): mean={s.mean():.2f}, median={np.median(s):.2f}, std={s.std():.2f}")

    # Score per system x language pair
    print("\n--- Score by System x Language Pair ---")
    for pair in sorted(df_dev.groupby(["src_lang", "tgt_lang"]).groups.keys()):
        mask = (df_dev["src_lang"] == pair[0]) & (df_dev["tgt_lang"] == pair[1])
        sub = df_dev[mask]
        print(f"\n  {pair[0]} -> {pair[1]}:")
        for sys_name in sorted(sub["tgt_system"].unique()):
            s = sub.loc[sub["tgt_system"] == sys_name, "score"].values
            print(f"    {sys_name}: mean={s.mean():.2f}, n={len(s)}")

    # Document structure
    print("\n--- Document Structure ---")
    unique_docs = df_dev["doc_id"].nunique()
    print(f"  Unique doc_ids: {unique_docs}")
    print(f"  Total examples: {len(df_dev)}")
    print(f"  Avg segments per doc: {len(df_dev) / unique_docs:.1f}")
    print(f"\n  First 20 doc_ids:")
    for _, row in df_dev.head(20).iterrows():
        print(f"    doc_id={row['doc_id']}, system={row['tgt_system']}, lang={row['src_lang']}->{row['tgt_lang']}")

    # Text lengths
    src_lens = df_dev["src_text"].str.split().str.len()
    tgt_lens = df_dev["tgt_text"].str.split().str.len()
    print(f"\n--- Text Lengths (words) ---")
    print(f"  Source: mean={src_lens.mean():.1f}, median={src_lens.median():.1f}, max={src_lens.max()}")
    print(f"  Target: mean={tgt_lens.mean():.1f}, median={tgt_lens.median():.1f}, max={tgt_lens.max()}")

    # Sample examples at different score ranges
    df_sorted = df_dev.sort_values("score")

    print("\n--- 5 LOWEST scoring examples ---")
    for _, r in df_sorted.head(5).iterrows():
        print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
        print(f"  Source: {r['src_text'][:150]}...")
        print(f"  Target: {r['tgt_text'][:150]}...")

    print("\n--- 5 HIGHEST scoring examples ---")
    for _, r in df_sorted.tail(5).iterrows():
        print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
        print(f"  Source: {r['src_text'][:150]}...")
        print(f"  Target: {r['tgt_text'][:150]}...")

    print("\n--- 5 MEDIAN scoring examples ---")
    mid = len(df_sorted) // 2
    for _, r in df_sorted.iloc[mid-2:mid+3].iterrows():
        print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
        print(f"  Source: {r['src_text'][:150]}...")
        print(f"  Target: {r['tgt_text'][:150]}...")

# ---------------------------------------------------------------------------
# 5. Save text-only data for fast access later
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print("STEP 5: Saving text-only DataFrames")
print(f"{'='*80}")

output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
for split_name, df in all_data.items():
    path = os.path.join(output_dir, f"{split_name}_text.parquet")
    df.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(df)} rows)")

print(f"\n{'='*80}")
print("EXPLORATION COMPLETE")
print(f"{'='*80}")
