"""
Baseline 1: CometKiwi-22 and CometKiwi-23-XL on the dev set.
Evaluates using Kendall's Tau (segment-level) and Soft Pairwise Accuracy (system-level).

This script runs on CPU (local machine). For GPU, set gpus=1.
"""

import os
import sys
import time

# SSL fix must come before any HuggingFace imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ---------------------------------------------------------------------------
# 1. Load dev data
# ---------------------------------------------------------------------------
print("Loading dev data...")
dev = pd.read_parquet("outputs/dev_text.parquet")
print(f"Dev set: {len(dev)} rows, {dev['tgt_system'].nunique()} systems")
print(f"Language pairs: {dev.groupby(['src_lang', 'tgt_lang']).size().to_dict()}")

# ---------------------------------------------------------------------------
# 2. Prepare data for COMET
# ---------------------------------------------------------------------------
# COMET expects: {"src": ..., "mt": ...} for QE models
samples = [
    {"src": row["src_text"], "mt": row["tgt_text"]}
    for _, row in dev.iterrows()
]
print(f"Prepared {len(samples)} samples for COMET")

# ---------------------------------------------------------------------------
# 3. Run CometKiwi-22
# ---------------------------------------------------------------------------
from comet import download_model, load_from_checkpoint

print("\n" + "=" * 80)
print("Running CometKiwi-22...")
print("=" * 80)

# Load from local files (curl-downloaded to bypass SSL issues with hf_hub_download)
local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
if os.path.exists(local_ckpt):
    print(f"Loading from local checkpoint: {local_ckpt}")
    model_path = local_ckpt
else:
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

start = time.time()
# num_workers=2 needed on macOS to avoid multiprocessing_context bug with MPS detection
output = model.predict(samples, batch_size=32, gpus=0, num_workers=2)
elapsed = time.time() - start
print(f"Inference took {elapsed:.1f}s ({len(samples)/elapsed:.1f} samples/s)")

dev["cometkiwi22_score"] = output["scores"]

# ---------------------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------------------

def kendall_tau_per_source(df, pred_col, gold_col="score"):
    """
    Compute Kendall's Tau grouped by source segment (doc_id).
    For each source, compare all pairs of system translations.
    """
    taus = []
    # Group by doc_id (each doc_id is a unique source segment)
    for doc_id, group in df.groupby("doc_id"):
        if len(group) < 2:
            continue
        # Kendall Tau between predicted and gold scores for this source
        tau, pvalue = stats.kendalltau(group[pred_col].values, group[gold_col].values)
        if not np.isnan(tau):
            taus.append(tau)
    return np.mean(taus) if taus else 0.0


def soft_pairwise_accuracy(df, pred_col, gold_col="score", threshold=25.0):
    """
    Compute Soft Pairwise Accuracy at system level.
    For each pair of systems, check if the metric agrees with humans
    on which system is better. Use a soft threshold for ties.
    """
    # Compute system-level scores
    sys_pred = df.groupby("tgt_system")[pred_col].mean()
    sys_gold = df.groupby("tgt_system")[gold_col].mean()

    systems = list(sys_pred.index)
    if len(systems) < 2:
        return 0.0

    agreements = 0
    total = 0
    for sys_a, sys_b in combinations(systems, 2):
        gold_diff = sys_gold[sys_a] - sys_gold[sys_b]
        pred_diff = sys_pred[sys_a] - sys_pred[sys_b]

        # If gold difference is within threshold, either ordering is fine
        if abs(gold_diff) < threshold:
            agreements += 1
        elif (gold_diff > 0 and pred_diff > 0) or (gold_diff < 0 and pred_diff < 0):
            agreements += 1

        total += 1

    return agreements / total if total > 0 else 0.0


def evaluate_metric(df, pred_col, gold_col="score"):
    """Full evaluation of a metric prediction."""
    # Overall Kendall Tau
    overall_tau, _ = stats.kendalltau(df[pred_col].values, df[gold_col].values)

    # Per-source Kendall Tau
    per_source_tau = kendall_tau_per_source(df, pred_col, gold_col)

    # Pearson correlation
    pearson_r, _ = stats.pearsonr(df[pred_col].values, df[gold_col].values)

    # Per language pair
    lp_results = {}
    for (src, tgt), group in df.groupby(["src_lang", "tgt_lang"]):
        lp_key = f"{src}->{tgt}"
        lp_tau = kendall_tau_per_source(group, pred_col, gold_col)
        lp_pearson, _ = stats.pearsonr(group[pred_col].values, group[gold_col].values)
        lp_spa = soft_pairwise_accuracy(group, pred_col, gold_col)
        lp_results[lp_key] = {"kendall_tau": lp_tau, "pearson": lp_pearson, "spa": lp_spa}

    # Overall SPA
    overall_spa = soft_pairwise_accuracy(df, pred_col, gold_col)

    return {
        "overall_kendall_tau": overall_tau,
        "per_source_kendall_tau": per_source_tau,
        "pearson": pearson_r,
        "spa": overall_spa,
        "per_lp": lp_results,
    }


print("\n--- CometKiwi-22 Results ---")
results_kiwi22 = evaluate_metric(dev, "cometkiwi22_score")
print(f"  Overall Kendall Tau:     {results_kiwi22['overall_kendall_tau']:.4f}")
print(f"  Per-source Kendall Tau:  {results_kiwi22['per_source_kendall_tau']:.4f}")
print(f"  Pearson correlation:     {results_kiwi22['pearson']:.4f}")
print(f"  Soft Pairwise Accuracy:  {results_kiwi22['spa']:.4f}")
for lp, lp_res in results_kiwi22["per_lp"].items():
    print(f"  {lp}: tau={lp_res['kendall_tau']:.4f}, pearson={lp_res['pearson']:.4f}, spa={lp_res['spa']:.4f}")

# ---------------------------------------------------------------------------
# 5. Save predictions
# ---------------------------------------------------------------------------
dev.to_parquet("outputs/dev_with_predictions.parquet", index=False)
print(f"\nSaved predictions to outputs/dev_with_predictions.parquet")

# ---------------------------------------------------------------------------
# 6. Score distribution analysis
# ---------------------------------------------------------------------------
print("\n--- CometKiwi-22 Score Distribution ---")
pred_scores = dev["cometkiwi22_score"].values
print(f"  Min: {pred_scores.min():.4f}, Max: {pred_scores.max():.4f}")
print(f"  Mean: {pred_scores.mean():.4f}, Std: {pred_scores.std():.4f}")
print(f"  Correlation with gold: {np.corrcoef(pred_scores, dev['score'].values)[0,1]:.4f}")

# System-level comparison
print("\n--- System-level Scores ---")
print(f"{'System':<35} {'Gold Mean':>10} {'CometKiwi22 Mean':>18}")
for sys_name in sorted(dev["tgt_system"].unique()):
    mask = dev["tgt_system"] == sys_name
    gold_mean = dev.loc[mask, "score"].mean()
    pred_mean = dev.loc[mask, "cometkiwi22_score"].mean()
    print(f"  {sys_name:<33} {gold_mean:>10.2f} {pred_mean:>18.4f}")

print("\n" + "=" * 80)
print("BASELINE COMPLETE")
print("=" * 80)
