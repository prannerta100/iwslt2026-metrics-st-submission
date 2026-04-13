"""
Advanced ensemble: LightGBM + calibration for final submission.

This script goes beyond simple weighted averaging:
1. LightGBM gradient-boosted trees on all available signals + features
2. Isotonic regression calibration (per language pair)
3. Stacked generalization (meta-learner on top of base models)
4. Direct Kendall Tau weight optimization (random search + Nelder-Mead)
5. Score clipping and normalization to match gold distribution

Two modes:
  - Train/eval mode (--train-data): Train on scored train set, evaluate on dev
  - CV-only mode (no --train-data): 5-fold GroupKFold CV on dev set only

Run:
  python scripts/04b_ensemble_advanced.py --train-data outputs/train_scored.parquet
  python scripts/04b_ensemble_advanced.py   # CV-only fallback
"""

import os
import sys
import json
import argparse
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

def build_features(df, signal_cols, speech_feat_file=None):
    """Build full feature matrix from all available signals.

    Args:
        df: DataFrame with src_text, tgt_text, tgt_lang, doc_id, and signal columns
        signal_cols: list of metric score column names to use
        speech_feat_file: path to speech features parquet (must match df row count)
    """
    features = pd.DataFrame(index=df.index)

    # Neural metric scores
    for col in signal_cols:
        features[col] = df[col].values

    # Text features (guard against missing columns)
    if "src_text" in df.columns and "tgt_text" in df.columns:
        features["src_word_count"] = df["src_text"].str.split().str.len()
        features["tgt_word_count"] = df["tgt_text"].str.split().str.len()
        features["length_ratio"] = features["tgt_word_count"] / features["src_word_count"].clip(lower=1)
        features["src_char_count"] = df["src_text"].str.len()
        features["tgt_char_count"] = df["tgt_text"].str.len()
        features["char_ratio"] = features["tgt_char_count"] / features["src_char_count"].clip(lower=1)

    # Language pair indicator
    if "tgt_lang" in df.columns:
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
        for stat_col in doc_stats.columns:
            features[stat_col] = df["doc_id"].map(doc_stats[stat_col]).values
        features[f"{col}_doc_dev"] = df[col].values - features[f"{col}_doc_mean"].values

    # Speech features — only if explicitly passed and row count matches
    if speech_feat_file and os.path.exists(speech_feat_file):
        speech_feats = pd.read_parquet(speech_feat_file)
        if len(speech_feats) == len(df):
            for col in speech_feats.columns:
                features[f"speech_{col}"] = speech_feats[col].values
        else:
            print(f"  WARNING: Speech features row count mismatch ({len(speech_feats)} vs {len(df)}), skipping")

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
        print("LightGBM not installed. Install with: poetry add lightgbm")
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
            callbacks=[lgb.early_stopping(50, verbose=False)],
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
    # Reset index so positional and label indices align
    df_reset = df.reset_index(drop=True)
    calibrated = np.zeros(len(df_reset))

    for (src_lang, tgt_lang), lp_group in df_reset.groupby(["src_lang", "tgt_lang"]):
        lp_indices = lp_group.index.values
        n_unique_groups = lp_group["doc_id"].nunique()

        if n_unique_groups < 2:
            # Can't cross-validate, just copy raw predictions
            calibrated[lp_indices] = lp_group[pred_col].values
            continue

        lp_folds = min(n_folds, n_unique_groups)
        lp_gkf = GroupKFold(n_splits=lp_folds)

        for fold, (train_idx, val_idx) in enumerate(lp_gkf.split(
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
# LightGBM: train on full train set, predict on dev
# ---------------------------------------------------------------------------

def lightgbm_train_eval(train_df, dev_df, train_features, dev_features,
                        gold_col="score"):
    """
    Train LightGBM on full train set (33K), evaluate on dev (5.5K).
    This is the proper train/test split — no CV leakage.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Install with: poetry add lightgbm")
        return None, None, None

    X_train = train_features.values
    y_train = train_df[gold_col].values
    X_dev = dev_features.values
    y_dev = dev_df[gold_col].values

    # Use 10% of train as validation for early stopping
    n_val = int(len(X_train) * 0.1)
    rng = np.random.RandomState(42)
    val_mask = rng.choice(len(X_train), n_val, replace=False)
    train_mask = np.setdiff1d(np.arange(len(X_train)), val_mask)

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
        "n_estimators": 1000,
        "seed": 42,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "min_child_samples": 20,
    }

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_train[train_mask], y_train[train_mask],
        eval_set=[(X_train[val_mask], y_train[val_mask])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # Predict on dev
    dev_preds = model.predict(X_dev)
    dev_df_eval = dev_df.copy()
    dev_df_eval["lgbm_pred"] = dev_preds
    dev_tau = kendall_tau_per_source(dev_df_eval, "lgbm_pred", gold_col)

    # Train performance (sanity check)
    train_preds = model.predict(X_train)
    train_df_eval = train_df.copy()
    train_df_eval["lgbm_pred"] = train_preds
    train_tau = kendall_tau_per_source(train_df_eval, "lgbm_pred", gold_col)

    print(f"  Train tau: {train_tau:.4f}, Dev tau: {dev_tau:.4f}")
    print(f"  Gap: {train_tau - dev_tau:.4f} (target: < 0.10)")
    print(f"  Best iteration: {model.best_iteration_}")

    # Feature importance
    feat_names = train_features.columns.tolist()
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n  Top 10 features:")
    for _, row in imp_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.1f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.booster_.save_model("models/lgbm_model.txt")
    print(f"\n  Saved LightGBM model to models/lgbm_model.txt")

    return dev_preds, dev_tau, model


# ---------------------------------------------------------------------------
# Direct Kendall Tau weight optimization
# ---------------------------------------------------------------------------

def optimize_weights(dev_df, signal_cols, gold_col="score", n_random=10000):
    """
    Find metric weights that directly maximize per-source Kendall Tau on dev.
    Uses random search + Nelder-Mead refinement.
    """
    n = len(signal_cols)
    signal_matrix = np.column_stack([dev_df[c].values for c in signal_cols])

    def eval_weights(w):
        w = np.abs(w)
        w = w / w.sum()
        ensemble = signal_matrix @ w
        dev_df_tmp = dev_df.copy()
        dev_df_tmp["_ens"] = ensemble
        return -kendall_tau_per_source(dev_df_tmp, "_ens", gold_col)

    # Random search
    best_tau = -1
    best_w = np.ones(n) / n
    rng = np.random.RandomState(42)

    for trial in range(n_random):
        w = rng.dirichlet(np.ones(n))
        tau = -eval_weights(w)
        if tau > best_tau:
            best_tau = tau
            best_w = w
            if trial < 100 or trial % 1000 == 0:
                print(f"  Trial {trial}: tau={tau:.4f}")

    print(f"\n  Best random search tau: {best_tau:.4f}")
    print(f"  Weights: {dict(zip(signal_cols, [f'{w:.3f}' for w in best_w]))}")

    # Nelder-Mead refinement
    result = optimize.minimize(eval_weights, best_w, method="Nelder-Mead",
                               options={"maxiter": 5000, "xatol": 1e-6})
    final_w = np.abs(result.x)
    final_w = final_w / final_w.sum()
    final_tau = -result.fun

    print(f"\n  Nelder-Mead tau: {final_tau:.4f}")
    print(f"  Weights: {dict(zip(signal_cols, [f'{w:.3f}' for w in final_w]))}")

    return dict(zip(signal_cols, final_w.tolist())), final_tau


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, default=None,
                        help="Path to scored train set (e.g., outputs/train_scored.parquet). "
                             "If provided, trains on train and evaluates on dev. "
                             "Otherwise falls back to CV on dev only.")
    parser.add_argument("--dev-data", type=str, default="outputs/dev_with_predictions.parquet")
    cli_args = parser.parse_args()

    print("=" * 80)
    print("ADVANCED ENSEMBLE PIPELINE")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load dev data
    # ------------------------------------------------------------------
    if not os.path.exists(cli_args.dev_data):
        print(f"ERROR: {cli_args.dev_data} not found.")
        sys.exit(1)

    dev = pd.read_parquet(cli_args.dev_data)
    print(f"Dev set: {len(dev)} examples, {dev['doc_id'].nunique()} docs")

    # ------------------------------------------------------------------
    # Load train data (if provided)
    # ------------------------------------------------------------------
    train = None
    if cli_args.train_data and os.path.exists(cli_args.train_data):
        train = pd.read_parquet(cli_args.train_data)
        print(f"Train set: {len(train)} examples, {train['doc_id'].nunique()} docs")
        print(f"MODE: Train on {len(train)} samples, evaluate on {len(dev)} dev samples")
    else:
        if cli_args.train_data:
            print(f"WARNING: {cli_args.train_data} not found, falling back to CV mode")
        print(f"MODE: 5-fold GroupKFold CV on dev only ({len(dev)} samples)")

    # ------------------------------------------------------------------
    # Detect available signals (intersection of train + dev columns)
    # ------------------------------------------------------------------
    ALL_SIGNALS = [
        "cometkiwi22_score", "finetuned_score", "pairwise_score",
        "xcomet_score", "blaser_score", "sonar_cosine", "speechqe_score",
        "metricx_score", "cometkiwi23xxl_score", "cometkiwi23xxl_finetuned_score",
    ]

    dev_signal_cols = [c for c in ALL_SIGNALS if c in dev.columns]

    if train is not None:
        # For LGBM: use only signals present in BOTH train and dev
        # finetuned_score and pairwise_score are NOT in train (circular — trained on train gold)
        shared_signal_cols = [c for c in ALL_SIGNALS if c in dev.columns and c in train.columns]
        print(f"\nDev signals ({len(dev_signal_cols)}): {dev_signal_cols}")
        print(f"Shared train+dev signals ({len(shared_signal_cols)}): {shared_signal_cols}")
    else:
        shared_signal_cols = dev_signal_cols

    if len(dev_signal_cols) < 1:
        print("ERROR: No signals available.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Evaluate individual signals on dev
    # ------------------------------------------------------------------
    print("\n--- Individual Signal Performance (dev) ---")
    for col in dev_signal_cols:
        tau = kendall_tau_per_source(dev, col, "score")
        spa = soft_pairwise_accuracy(dev, col, "score")
        print(f"  {col}: per-source tau={tau:.4f}, spa={spa:.4f}")

    # ------------------------------------------------------------------
    # Method 1: LightGBM
    # ------------------------------------------------------------------
    if train is not None and len(shared_signal_cols) >= 1:
        # TRAIN/EVAL MODE: Train on 33K, evaluate on 5.5K dev
        # Use ONLY shared_signal_cols so train and dev have identical feature sets.
        # No speech_feat_file — we don't have speech features for train.
        # No xcomet error columns — build_features no longer adds them
        #   (they were conditional on df columns and caused train/dev mismatch).
        print("\n--- LightGBM (trained on full train set) ---")
        train_features = build_features(train, shared_signal_cols)
        dev_features = build_features(dev, shared_signal_cols)
        assert list(train_features.columns) == list(dev_features.columns), \
            f"Feature mismatch: train has {train_features.shape[1]}, dev has {dev_features.shape[1]}"
        print(f"  Train features: {train_features.shape}, Dev features: {dev_features.shape}")

        lgbm_preds, lgbm_tau, lgbm_model = lightgbm_train_eval(
            train, dev, train_features, dev_features
        )
        if lgbm_preds is not None:
            dev["lgbm_score"] = lgbm_preds
    else:
        # CV-ONLY MODE: GroupKFold on dev (can use all dev features including speech)
        print("\n--- LightGBM (5-fold CV on dev) ---")
        dev_features = build_features(dev, dev_signal_cols,
                                      speech_feat_file="outputs/dev_speech_features.parquet")
        print(f"  Feature matrix: {dev_features.shape}")
        lgbm_preds, lgbm_tau = lightgbm_ensemble(dev, dev_features)
        if lgbm_preds is not None:
            dev["lgbm_score"] = lgbm_preds

    # ------------------------------------------------------------------
    # Method 2: Isotonic calibration (always on dev via CV)
    # ------------------------------------------------------------------
    print("\n--- Isotonic Calibration ---")
    for col in dev_signal_cols:
        cal_col = f"{col}_calibrated"
        dev[cal_col] = calibrate_predictions(dev, col, "score")
        cal_tau = kendall_tau_per_source(dev, cal_col, "score")
        print(f"  {col} calibrated tau: {cal_tau:.4f}")

    # ------------------------------------------------------------------
    # Method 3: Stacked meta-learner (CV on dev)
    # ------------------------------------------------------------------
    if len(dev_signal_cols) >= 2:
        print("\n--- Stacked Meta-Learner ---")
        base_preds = [dev[col].values for col in dev_signal_cols]
        meta_preds, meta_tau = stacked_ensemble(dev, base_preds)
        dev["meta_score"] = meta_preds

    # ------------------------------------------------------------------
    # Method 4: Direct Kendall Tau weight optimization on dev
    # ------------------------------------------------------------------
    print("\n--- Direct Kendall Tau Weight Optimization (dev) ---")
    opt_weights, opt_tau = optimize_weights(dev, dev_signal_cols)

    # Apply optimized weights to dev
    ensemble_score = np.zeros(len(dev))
    for col, w in opt_weights.items():
        ensemble_score += dev[col].values * w
    dev["weighted_ensemble_score"] = ensemble_score

    # Save weights for submission
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/ensemble_weights.json", "w") as f:
        json.dump(opt_weights, f, indent=2)
    print(f"  Saved ensemble weights to outputs/ensemble_weights.json")

    # ------------------------------------------------------------------
    # Final comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL COMPARISON (all methods evaluated on dev)")
    print("=" * 80)

    eval_cols = dev_signal_cols.copy()
    if "lgbm_score" in dev.columns:
        eval_cols.append("lgbm_score")
    if "meta_score" in dev.columns:
        eval_cols.append("meta_score")
    if "weighted_ensemble_score" in dev.columns:
        eval_cols.append("weighted_ensemble_score")

    print(f"\n{'Method':<30} {'Per-src Tau':>12} {'Overall Tau':>12} {'SPA':>8}")
    print("-" * 66)
    best_col = None
    best_tau = -1
    for col in eval_cols:
        tau = kendall_tau_per_source(dev, col, "score")
        overall_tau, _ = stats.kendalltau(dev[col].values, dev["score"].values)
        spa = soft_pairwise_accuracy(dev, col, "score")
        marker = ""
        if tau > best_tau:
            best_tau = tau
            best_col = col
        print(f"  {col:<28} {tau:>12.4f} {overall_tau:>12.4f} {spa:>8.4f}")

    print(f"\n  >>> Best method: {best_col} (per-source tau={best_tau:.4f})")

    # Save final ensemble
    dev.to_parquet("outputs/dev_ensemble_advanced.parquet", index=False)
    print(f"\nSaved to outputs/dev_ensemble_advanced.parquet")

    print("\n" + "=" * 80)
    print("ADVANCED ENSEMBLE COMPLETE")
    print("=" * 80)
