"""
Ensemble multiple metric signals to maximize Kendall Tau.

Signals:
1. CometKiwi-22 (base or fine-tuned)
2. xCOMET-XL (with error span features)
3. BLASER-2 QE (speech-text cross-modal)
4. Additional features (text length ratio, etc.)

Fusion methods:
- Simple average (baseline)
- Learned weights via scipy.optimize (optimize Kendall Tau directly)
- Gradient-boosted trees (LightGBM/sklearn)

Run: python scripts/04_ensemble.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, optimize
from itertools import combinations
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix


# ---------------------------------------------------------------------------
# Evaluation functions
# ---------------------------------------------------------------------------

def kendall_tau_per_source(df, pred_col, gold_col="score"):
    """Kendall Tau grouped by source (doc_id). Primary metric."""
    taus = []
    for doc_id, group in df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, _ = stats.kendalltau(group[pred_col].values, group[gold_col].values)
        if not np.isnan(tau):
            taus.append(tau)
    return np.mean(taus) if taus else 0.0


def soft_pairwise_accuracy(df, pred_col, gold_col="score", threshold=25.0):
    """System-level Soft Pairwise Accuracy."""
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
        if abs(gold_diff) < threshold:
            agreements += 1
        elif (gold_diff > 0 and pred_diff > 0) or (gold_diff < 0 and pred_diff < 0):
            agreements += 1
        total += 1
    return agreements / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(df):
    """Extract simple text-based features that can augment neural scores."""
    features = pd.DataFrame(index=df.index)

    # Text length features
    features["src_word_count"] = df["src_text"].str.split().str.len()
    features["tgt_word_count"] = df["tgt_text"].str.split().str.len()
    features["length_ratio"] = features["tgt_word_count"] / features["src_word_count"].clip(lower=1)

    # Character-level features
    features["src_char_count"] = df["src_text"].str.len()
    features["tgt_char_count"] = df["tgt_text"].str.len()
    features["char_ratio"] = features["tgt_char_count"] / features["src_char_count"].clip(lower=1)

    # Language pair indicator
    features["is_zh"] = (df["tgt_lang"] == "zh").astype(float)

    return features


# ---------------------------------------------------------------------------
# Ensemble methods
# ---------------------------------------------------------------------------

def weighted_average_ensemble(signals, weights):
    """Weighted average of multiple signal columns."""
    weighted = np.zeros(len(signals[0]))
    for signal, weight in zip(signals, weights):
        weighted += signal * weight
    return weighted / sum(weights)


def optimize_weights_kendall(df, signal_cols, gold_col="score"):
    """
    Find optimal weights by directly maximizing per-source Kendall Tau.
    Uses scipy.optimize.differential_evolution (gradient-free).
    """
    # Reset index so group.index matches positional indices into arrays
    df = df.reset_index(drop=True)
    signals = [df[col].values for col in signal_cols]
    n_signals = len(signals)

    def neg_kendall_tau(weights):
        """Negative Kendall Tau (for minimization)."""
        combined = np.zeros(len(signals[0]))
        for s, w in zip(signals, weights):
            combined += s * w
        combined /= sum(weights)

        # Compute per-source Kendall Tau
        taus = []
        for doc_id, group in df.groupby("doc_id"):
            if len(group) < 2:
                continue
            indices = group.index
            tau, _ = stats.kendalltau(combined[indices], group[gold_col].values)
            if not np.isnan(tau):
                taus.append(tau)
        return -np.mean(taus) if taus else 0.0

    # Bounds: all weights between 0 and 1
    bounds = [(0.0, 1.0)] * n_signals

    result = optimize.differential_evolution(
        neg_kendall_tau,
        bounds,
        seed=42,
        maxiter=100,
        tol=1e-6,
        workers=1,  # workers=-1 causes pickle error with nested functions
    )

    optimal_weights = result.x
    best_tau = -result.fun
    return optimal_weights, best_tau


def cross_validated_ensemble(df, signal_cols, gold_col="score", n_folds=5):
    """
    Cross-validated weight optimization to avoid overfitting on dev set.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_preds = np.zeros(len(df))
    fold_taus = []
    fold_weights = []

    # Group by doc_id to ensure same source goes to same fold
    unique_docs = df["doc_id"].unique()

    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_docs)):
        train_docs = set(unique_docs[train_idx])
        val_docs = set(unique_docs[val_idx])

        train_mask = df["doc_id"].isin(train_docs)
        val_mask = df["doc_id"].isin(val_docs)

        # Optimize on train fold
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        weights, train_tau = optimize_weights_kendall(train_df, signal_cols, gold_col)
        fold_weights.append(weights)

        # Apply to val fold
        signals = [val_df[col].values for col in signal_cols]
        combined = weighted_average_ensemble(signals, weights)
        all_preds[val_mask] = combined

        # Evaluate on val fold
        val_df["ensemble"] = combined
        val_tau = kendall_tau_per_source(val_df, "ensemble", gold_col)
        fold_taus.append(val_tau)
        print(f"  Fold {fold+1}: train_tau={train_tau:.4f}, val_tau={val_tau:.4f}, weights={weights}")

    avg_tau = np.mean(fold_taus)
    avg_weights = np.mean(fold_weights, axis=0)
    print(f"\n  CV average tau: {avg_tau:.4f}")
    print(f"  Average weights: {avg_weights}")

    return avg_weights, avg_tau, all_preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("ENSEMBLE PIPELINE")
    print("=" * 80)

    # Load dev data with predictions
    pred_file = "outputs/dev_with_predictions.parquet"
    if not os.path.exists(pred_file):
        print(f"ERROR: {pred_file} not found. Run baseline scripts first.")
        sys.exit(1)

    dev = pd.read_parquet(pred_file)
    print(f"Loaded {len(dev)} dev examples")

    # Check which signals are available
    signal_cols = []
    for col in ["cometkiwi22_score", "finetuned_score", "pairwise_score",
                "xcomet_score", "blaser_score",
                "metricx_score", "cometkiwi23xxl_score"]:
        if col in dev.columns:
            signal_cols.append(col)

    print(f"Available signals: {signal_cols}")

    if len(signal_cols) < 1:
        print("ERROR: No signals available for ensembling.")
        sys.exit(1)

    # Extract text features
    features = extract_features(dev)
    for col in features.columns:
        dev[f"feat_{col}"] = features[col]

    # Evaluate individual signals
    print("\n--- Individual Signal Performance ---")
    for col in signal_cols:
        tau = kendall_tau_per_source(dev, col, "score")
        spa = soft_pairwise_accuracy(dev, col, "score")
        overall_tau, _ = stats.kendalltau(dev[col].values, dev["score"].values)
        print(f"  {col}: per-source tau={tau:.4f}, overall tau={overall_tau:.4f}, spa={spa:.4f}")

    if len(signal_cols) >= 2:
        # Optimize ensemble weights
        print("\n--- Weight Optimization (direct Kendall Tau maximization) ---")
        opt_weights, opt_tau = optimize_weights_kendall(dev, signal_cols, "score")
        print(f"  Optimal weights: {dict(zip(signal_cols, opt_weights))}")
        print(f"  Optimized per-source Kendall Tau: {opt_tau:.4f}")

        # Apply optimal weights
        signals = [dev[col].values for col in signal_cols]
        dev["ensemble_score"] = weighted_average_ensemble(signals, opt_weights)

        # Evaluate ensemble
        ensemble_tau = kendall_tau_per_source(dev, "ensemble_score", "score")
        ensemble_spa = soft_pairwise_accuracy(dev, "ensemble_score", "score")
        print(f"\n  Ensemble per-source Kendall Tau: {ensemble_tau:.4f}")
        print(f"  Ensemble SPA: {ensemble_spa:.4f}")

        # Cross-validated ensemble
        print("\n--- Cross-Validated Ensemble ---")
        cv_weights, cv_tau, cv_preds = cross_validated_ensemble(
            dev, signal_cols, "score", n_folds=5
        )
        dev["cv_ensemble_score"] = cv_preds

    # Save results
    dev.to_parquet("outputs/dev_ensemble.parquet", index=False)
    print(f"\nSaved ensemble predictions to outputs/dev_ensemble.parquet")

    print("\n" + "=" * 80)
    print("ENSEMBLE COMPLETE")
    print("=" * 80)
