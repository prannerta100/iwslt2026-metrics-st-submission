"""
Download parquet files and extract text columns.
Downloads dev first (3 files), then train (17 files), then synthetic (1 file).
Reads only text columns (skips audio bytes) to save memory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

DATASET_ID = "maikezu/iwslt2026-metrics-shared-train-dev"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Text columns to extract (skip 'audio' which contains binary audio data)
TEXT_COLS = ["audio_path", "doc_id", "src_text", "src_text_system", "src_lang",
            "tgt_lang", "domain", "tgt_system", "tgt_text", "score"]

# File manifest from HuggingFace API
SPLITS = {
    "dev": [f"data/dev-{i:05d}-of-00003.parquet" for i in range(3)],
    "train": [f"data/train-{i:05d}-of-00017.parquet" for i in range(17)],
    "train_synthetic": ["data/train_synthetic-00000-of-00001.parquet"],
}


def download_and_read(split_name, files):
    """Download parquet files and extract text columns into a DataFrame."""
    dfs = []
    for f in files:
        print(f"  Downloading {f}...")
        try:
            local_path = hf_hub_download(
                DATASET_ID, f, repo_type="dataset", token=HF_TOKEN
            )
        except Exception as e:
            print(f"  ERROR downloading {f}: {e}")
            continue

        print(f"  Reading text columns...")
        pf = pq.ParquetFile(local_path)
        # Only read text columns that exist in this file
        available = [c for c in TEXT_COLS if c in pf.schema.names]
        table = pf.read(columns=available)
        df = table.to_pandas()
        dfs.append(df)
        print(f"  Got {len(df)} rows")

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        return result
    return pd.DataFrame()


def analyze_split(name, df):
    """Print comprehensive analysis of a data split."""
    print(f"\n{'='*80}")
    print(f"SPLIT: {name} ({len(df)} rows)")
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

    return scores


def dev_deep_dive(df):
    """Deep analysis of the dev set."""
    print(f"\n{'='*80}")
    print("DEV SET DEEP DIVE")
    print(f"{'='*80}")

    # Score per language pair
    print("\n--- Score by Language Pair ---")
    for (src, tgt), group in df.groupby(["src_lang", "tgt_lang"]):
        s = group["score"].values
        print(f"\n  {src} -> {tgt} ({len(s)} examples):")
        print(f"    Mean: {s.mean():.2f}, Median: {np.median(s):.2f}, Std: {s.std():.2f}")
        print(f"    Range: [{s.min():.2f}, {s.max():.2f}]")

    # Score per system
    print("\n--- Score by System ---")
    for sys_name, group in df.groupby("tgt_system"):
        s = group["score"].values
        print(f"  {sys_name} (n={len(s)}): mean={s.mean():.2f}, median={np.median(s):.2f}, std={s.std():.2f}")

    # Score per system x language pair
    print("\n--- Score by System x Language Pair ---")
    for (src, tgt), lang_group in df.groupby(["src_lang", "tgt_lang"]):
        print(f"\n  {src} -> {tgt}:")
        for sys_name, sys_group in lang_group.groupby("tgt_system"):
            s = sys_group["score"].values
            print(f"    {sys_name}: mean={s.mean():.2f}, n={len(s)}")

    # Document structure
    print("\n--- Document Structure ---")
    unique_docs = df["doc_id"].nunique()
    print(f"  Unique doc_ids: {unique_docs}")
    print(f"  Total examples: {len(df)}")
    print(f"  Avg segments per doc: {len(df) / unique_docs:.1f}")
    print(f"\n  First 20 doc_ids:")
    for _, row in df.head(20).iterrows():
        print(f"    doc_id={row['doc_id']}, system={row['tgt_system']}, lang={row['src_lang']}->{row['tgt_lang']}")

    # Text lengths
    src_lens = df["src_text"].str.split().str.len()
    tgt_lens = df["tgt_text"].str.split().str.len()
    print(f"\n--- Text Lengths (words) ---")
    print(f"  Source: mean={src_lens.mean():.1f}, median={src_lens.median():.1f}, max={src_lens.max()}")
    print(f"  Target: mean={tgt_lens.mean():.1f}, median={tgt_lens.median():.1f}, max={tgt_lens.max()}")

    # Sample examples at different score ranges
    df_sorted = df.sort_values("score")

    print("\n--- 5 LOWEST scoring examples ---")
    for _, r in df_sorted.head(5).iterrows():
        print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
        print(f"  Source: {str(r['src_text'])[:150]}")
        print(f"  Target: {str(r['tgt_text'])[:150]}")

    print("\n--- 5 HIGHEST scoring examples ---")
    for _, r in df_sorted.tail(5).iterrows():
        print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
        print(f"  Source: {str(r['src_text'])[:150]}")
        print(f"  Target: {str(r['tgt_text'])[:150]}")

    print("\n--- 5 MEDIAN scoring examples ---")
    mid = len(df_sorted) // 2
    for _, r in df_sorted.iloc[mid-2:mid+3].iterrows():
        print(f"\n  Score: {r['score']:.2f} | {r['src_lang']}->{r['tgt_lang']} | {r['tgt_system']}")
        print(f"  Source: {str(r['src_text'])[:150]}")
        print(f"  Target: {str(r['tgt_text'])[:150]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    all_data = {}

    # Download and analyze each split
    for split_name in ["dev", "train", "train_synthetic"]:
        print(f"\n{'='*80}")
        print(f"DOWNLOADING: {split_name}")
        print(f"{'='*80}")
        df = download_and_read(split_name, SPLITS[split_name])
        if len(df) == 0:
            print(f"  WARNING: No data for {split_name}")
            continue

        all_data[split_name] = df
        analyze_split(split_name, df)

        # Save text-only parquet for fast access later
        save_path = os.path.join(OUTPUT_DIR, f"{split_name}_text.parquet")
        df.to_parquet(save_path, index=False)
        print(f"\n  Saved to {save_path}")

    # Dev deep dive
    if "dev" in all_data:
        dev_deep_dive(all_data["dev"])

    print(f"\n{'='*80}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*80}")
