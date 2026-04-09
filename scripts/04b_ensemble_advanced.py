"""
Advanced ensemble: LightGBM + calibration for final submission.

This script goes beyond simple weighted averaging:
1. LightGBM gradient-boosted trees on all available signals + features
2. Isotonic regression calibration (per language pair)
3. Stacked generalization (meta-learner on top of base models)
4. Score clipping and normalization to match gold distribution

This is the final ensemble used for submission scoring.

Run: python scripts/04b_ensemble_advanced.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, optimize
from itertools import combinations
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def kendall_tau_per_source(df, pred_col, gold_col="score"):
    taus = []
    for doc_id, group in df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, _ = stats.kendalltau(group[pred_col].values, group[gold_col].values)
        if not np.isnan(tau):
            taus.append(tau)
    return np.mean(taus) if taus else 0.0


def soft_pairwise_accuracy(df, pred_col, gold_col="score", threshold=25.0):
    sys_pred = df.groupby("tgt_system")[pred_col].mean()
    sys_gold = df.groupby("tgt_system")[gold_col].mean()
    systems = list(sys_pred.index)
    if len(systems) < 2:
        return 0.0
    agreements = total = 0
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
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df, signal_cols):
    """Build full feature matrix from all available signals."""
    features = pd.DataFrame(index=df.index)

    # Neural metric scores
    for col in signal_cols:
        features[col] = df[col].values

    # Text features
    features["src_word_count"] = df["src_text"].str.split().str.len()
    features["tgt_word_count"] = df["tgt_text"].str.split().str.len()
    features["length_ratio"] = features["tgt_word_count"] / features["src_word_count"].clip(lower=1)
    features["src_char_count"] = df["src_text"].str.len()
    features["tgt_char_count"] = df["tgt_text"].str.len()
    features["char_ratio"] = features["tgt_char_count"] / features["src_char_count"].clip(lower=1)

    # Language pair indicator
    features["is_zh"] = (df["tgt_lang"] == "zh").astype(float)
    features["is_de"] = (df["tgt_lang"] == "de").astype(float)

    # Cross-signal features (if multiple signals available)
    if len(signal_cols) >= 2:
        for i, col_a in enumerate(signal_cols):
            for col_b in signal_cols[i+1:]:
                features[f"{col_a}_minus_{col_b}"] = df[col_a] - df[col_b]
                features[f"{col_a}_times_{col_b}"] = df[col_a] * df[col_b]

    # Score statistics per doc_id (context features)
    for col in signal_cols:
        doc_stats = df.groupby("doc_id")[col].agg(["mean", "std", "min", "max"])
        doc_stats.columns = [f"{col}_doc_{stat}" for stat in ["mean", "std", "min", "max"]]
        # Map doc_id stats back to each row via .map() instead of .join(on=...)
        for stat_col in doc_stats.columns:
            features[stat_col] = df["doc_id"].map(doc_stats[stat_col]).values
        # Deviation from doc mean
        features[f"{col}_doc_dev"] = df[col].values - features[f"{col}_doc_mean"].values

    # Speech features if available
    speech_feat_file = "outputs/dev_speech_features.parquet"
    if os.path.exists(speech_feat_file):
        speech_feats = pd.read_parquet(speech_feat_file)
        for col in speech_feats.columns:
            features[f"speech_{col}"] = speech_feats[col].values

    # Drop any NaN columns
    features = features.fillna(0)

    return features


# ---------------------------------------------------------------------------
# LightGBM ensemble
# ---------------------------------------------------------------------------

def lightgbm_ensemble(df, features, gold_col="score", n_folds=5):
    """
    Train LightGBM to predict quality scores from all features.
    Uses GroupKFold to keep same doc_id in same fold.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Install with: pip install lightgbm")
        return None, None

    X = features.values
    y = df[gold_col].values
    groups = df["doc_id"].values

    # GroupKFold ensures same source goes to same fold
    gkf = GroupKFold(n_splits=n_folds)
    all_preds = np.zeros(len(df))
    fold_taus = []
    feature_importance = np.zeros(X.shape[1])

    lgb_params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "early_stopping_rounds": 50,
        "seed": 42,
        # Regularization to prevent overfitting
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_child_samples": 20,
    }

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_X, val_X = X[train_idx], X[val_idx]
        train_y, val_y = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            train_X, train_y,
            eval_set=[(val_X, val_y)],
        )

        val_pred = model.predict(val_X)
        all_preds[val_idx] = val_pred

        # Evaluate fold
        fold_df = df.iloc[val_idx].copy()
        fold_df["lgbm_pred"] = val_pred
        fold_tau = kendall_tau_per_source(fold_df, "lgbm_pred", gold_col)
        fold_taus.append(fold_tau)

        feature_importance += model.feature_importances_

        print(f"  Fold {fold+1}: val_tau={fold_tau:.4f}, best_iter={model.best_iteration_}")

    avg_tau = np.mean(fold_taus)
    print(f"\n  LightGBM CV average tau: {avg_tau:.4f}")

    # Feature importance
    feature_importance /= n_folds
    feat_names = features.columns.tolist()
    imp_df = pd.DataFrame({"feature": feat_names, "importance": feature_importance})
    imp_df = imp_df.sort_values("importance", ascending=False)
    print("\n  Top 10 features:")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.1f}")

    return all_preds, avg_tau


# ---------------------------------------------------------------------------
# Isotonic calibration
# ---------------------------------------------------------------------------

def calibrate_predictions(df, pred_col, gold_col="score", n_folds=5):
    """
    Isotonic regression calibration per language pair.
    Ensures predictions are well-calibrated to the gold score distribution.
    """
    calibrated = np.zeros(len(df))
    groups = df["doc_id"].values
    gkf = GroupKFold(n_splits=n_folds)

    for (src_lang, tgt_lang), lp_group in df.groupby(["src_lang", "tgt_lang"]):
        lp_indices = lp_group.index.values

        for fold, (train_idx, val_idx) in enumerate(gkf.split(
            lp_group[pred_col].values,
            lp_group[gold_col].values,
            lp_group["doc_id"].values,
        )):
            train_pred = lp_group.iloc[train_idx][pred_col].values
            train_gold = lp_group.iloc[train_idx][gold_col].values
            val_pred = lp_group.iloc[val_idx][pred_col].values

            iso_reg = IsotonicRegression(out_of_bounds="clip")
            iso_reg.fit(train_pred, train_gold)
            calibrated[lp_indices[val_idx]] = iso_reg.transform(val_pred)

    return calibrated


# ---------------------------------------------------------------------------
# Stacking meta-learner
# ---------------------------------------------------------------------------

def stacked_ensemble(df, base_predictions, gold_col="score", n_folds=5):
    """
    Meta-learner: train a second-level model on base model predictions.
    Uses Ridge regression for simplicity and to avoid overfitting.
    """
    from sklearn.linear_model import Ridge

    X = np.column_stack(base_predictions)
    y = df[gold_col].values
    groups = df["doc_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    meta_preds = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        model = Ridge(alpha=1.0)
        model.fit(X[train_idx], y[train_idx])
        meta_preds[val_idx] = model.predict(X[val_idx])

    df_eval = df.copy()
    df_eval["meta_pred"] = meta_preds
    tau = kendall_tau_per_source(df_eval, "meta_pred", gold_col)
    print(f"  Stacked meta-learner per-source tau: {tau:.4f}")

    return meta_preds, tau


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED ENSEMBLE PIPELINE")
    print("=" * 80)

    # Load dev data with all predictions
    pred_file = "outputs/dev_with_predictions.parquet"
    if not os.path.exists(pred_file):
        print(f"ERROR: {pred_file} not found.")
        sys.exit(1)

    dev = pd.read_parquet(pred_file)
    print(f"Loaded {len(dev)} dev examples")
    print(f"Columns: {dev.columns.tolist()}")

    # Detect available signals
    signal_cols = []
    for col in ["cometkiwi22_score", "finetuned_score", "xcomet_score",
                 "blaser_score", "sonar_cosine", "speechqe_score"]:
        if col in dev.columns:
            signal_cols.append(col)

    print(f"\nAvailable signals: {signal_cols}")

    if len(signal_cols) < 1:
        print("ERROR: No signals available.")
        sys.exit(1)

    # Evaluate individual signals
    print("\n--- Individual Signal Performance ---")
    for col in signal_cols:
        tau = kendall_tau_per_source(dev, col, "score")
        spa = soft_pairwise_accuracy(dev, col, "score")
        print(f"  {col}: per-source tau={tau:.4f}, spa={spa:.4f}")

    # Build features
    print("\n--- Building Features ---")
    features = build_features(dev, signal_cols)
    print(f"Feature matrix: {features.shape}")

    # Method 1: LightGBM
    print("\n--- LightGBM Ensemble ---")
    lgbm_preds, lgbm_tau = lightgbm_ensemble(dev, features)
    if lgbm_preds is not None:
        dev["lgbm_score"] = lgbm_preds

    # Method 2: Calibrated predictions
    print("\n--- Isotonic Calibration ---")
    for col in signal_cols:
        cal_col = f"{col}_calibrated"
        dev[cal_col] = calibrate_predictions(dev, col, "score")
        cal_tau = kendall_tau_per_source(dev, cal_col, "score")
        print(f"  {col} calibrated tau: {cal_tau:.4f}")

    # Method 3: Stacked meta-learner
    if len(signal_cols) >= 2:
        print("\n--- Stacked Meta-Learner ---")
        base_preds = [dev[col].values for col in signal_cols]
        meta_preds, meta_tau = stacked_ensemble(dev, base_preds)
        dev["meta_score"] = meta_preds

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    eval_cols = signal_cols.copy()
    if "lgbm_score" in dev.columns:
        eval_cols.append("lgbm_score")
    if "meta_score" in dev.columns:
        eval_cols.append("meta_score")

    print(f"\n{'Method':<30} {'Per-src Tau':>12} {'Overall Tau':>12} {'SPA':>8}")
    print("-" * 62)
    for col in eval_cols:
        tau = kendall_tau_per_source(dev, col, "score")
        overall_tau, _ = stats.kendalltau(dev[col].values, dev["score"].values)
        spa = soft_pairwise_accuracy(dev, col, "score")
        print(f"  {col:<28} {tau:>12.4f} {overall_tau:>12.4f} {spa:>8.4f}")

    # Save final ensemble
    dev.to_parquet("outputs/dev_ensemble_advanced.parquet", index=False)
    print(f"\nSaved to outputs/dev_ensemble_advanced.parquet")

    # Determine best method for submission
    best_col = None
    best_tau = -1
    for col in eval_cols:
        tau = kendall_tau_per_source(dev, col, "score")
        if tau > best_tau:
            best_tau = tau
            best_col = col
    print(f"\nBest method: {best_col} (per-source tau={best_tau:.4f})")

    print("\n" + "=" * 80)
    print("ADVANCED ENSEMBLE COMPLETE")
    print("=" * 80)
