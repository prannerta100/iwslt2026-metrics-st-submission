"""
Final submission: train LightGBM on scored train, predict on scored test,
produce submission file.

Submission format (from organizers' evaluation script at
github.com/zouharvi/iwslt26-metrics/blob/main/evaluation/__main__.py):
  - ONE file covering ALL language pairs (not separate per-LP files)
  - One score per line (bare number, parseable by json.loads)
  - Same row order as the input dataset (en-de and en-zh interleaved)
  - len(scores) must equal len(input_data)
  - The evaluation script splits by LP internally using src_lang+tgt_lang

This script:
  1. Loads scored train (outputs/train_scored.parquet) and scored test (submission/test_predictions.parquet)
  2. Finds shared signal columns between train and test
  3. Builds features (same as 04b_ensemble_advanced.py)
  4. Trains LightGBM on all train data (with 10% holdout for early stopping)
  5. Validates on dev if available
  6. Predicts on test
  7. Writes ONE submission file with all 48K scores in original dataset order
  8. Also produces per-signal backup files

Run:
  poetry run python scripts/15_final_submission.py
  poetry run python scripts/15_final_submission.py --train-data outputs/train_scored.parquet
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

parser = argparse.ArgumentParser()
parser.add_argument("--train-data", type=str, default="outputs/train_scored.parquet")
parser.add_argument("--dev-data", type=str, default="outputs/dev_with_predictions.parquet")
parser.add_argument("--test-data", type=str, default="submission/test_predictions.parquet")
parser.add_argument("--output-dir", type=str, default="submission/")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("FINAL SUBMISSION — LightGBM Ensemble + Per-LP Output")
print("=" * 80)

if not os.path.exists(args.test_data):
    print(f"ERROR: {args.test_data} not found.")
    print("  Run scripts/13_submit_test.py first to score the test set.")
    sys.exit(1)

test = pd.read_parquet(args.test_data)
print(f"Test: {len(test)} rows")
print(f"  en-de: {len(test[test['tgt_lang'] == 'de'])} rows")
print(f"  en-zh: {len(test[test['tgt_lang'] == 'zh'])} rows")

has_train = os.path.exists(args.train_data)
has_dev = os.path.exists(args.dev_data)

if has_train:
    train = pd.read_parquet(args.train_data)
    print(f"Train: {len(train)} rows")
else:
    train = None
    print("WARNING: No scored train data — will use weighted average instead of LightGBM")

if has_dev:
    dev = pd.read_parquet(args.dev_data)
    print(f"Dev: {len(dev)} rows (for validation)")
else:
    dev = None

# ---------------------------------------------------------------------------
# 2. Merge pre-computed LLM debate scores if available
# ---------------------------------------------------------------------------
llm_test_cache = "outputs/llm_debate_test.parquet"
if os.path.exists(llm_test_cache) and "llm_debate_score" not in test.columns:
    llm_df = pd.read_parquet(llm_test_cache)
    if "llm_debate_score" in llm_df.columns and len(llm_df) == len(test):
        test["llm_debate_score"] = llm_df["llm_debate_score"].values
        print(f"  Merged llm_debate_score into test from {llm_test_cache}")

if train is not None:
    llm_train_cache = "outputs/llm_debate_train.parquet"
    if os.path.exists(llm_train_cache) and "llm_debate_score" not in train.columns:
        llm_df = pd.read_parquet(llm_train_cache)
        if "llm_debate_score" in llm_df.columns and len(llm_df) == len(train):
            train["llm_debate_score"] = llm_df["llm_debate_score"].values
            print(f"  Merged llm_debate_score into train from {llm_train_cache}")

# ---------------------------------------------------------------------------
# 3. Identify signals
# ---------------------------------------------------------------------------
ALL_SIGNALS = [
    "cometkiwi22_score", "finetuned_score", "pairwise_score",
    "xcomet_score", "blaser_score", "metricx_score",
    "cometkiwi23xxl_score", "cometkiwi23xxl_finetuned_score",
    "sonar_cosine", "llm_debate_score",
]

test_signals = [c for c in ALL_SIGNALS if c in test.columns]
print(f"\nTest signals ({len(test_signals)}): {test_signals}")

if train is not None:
    train_signals = [c for c in ALL_SIGNALS if c in train.columns]
    shared_signals = [c for c in ALL_SIGNALS if c in train.columns and c in test.columns]
    print(f"Train signals ({len(train_signals)}): {train_signals}")
    print(f"Shared signals ({len(shared_signals)}): {shared_signals}")
else:
    shared_signals = test_signals


# ---------------------------------------------------------------------------
# 3. Feature engineering (mirrors 04b_ensemble_advanced.py exactly)
# ---------------------------------------------------------------------------
def build_features(df, signal_cols):
    features = pd.DataFrame(index=df.index)

    for col in signal_cols:
        features[col] = df[col].values

    if "src_text" in df.columns and "tgt_text" in df.columns:
        features["src_word_count"] = df["src_text"].str.split().str.len()
        features["tgt_word_count"] = df["tgt_text"].str.split().str.len()
        features["length_ratio"] = features["tgt_word_count"] / features["src_word_count"].clip(lower=1)
        features["src_char_count"] = df["src_text"].str.len()
        features["tgt_char_count"] = df["tgt_text"].str.len()
        features["char_ratio"] = features["tgt_char_count"] / features["src_char_count"].clip(lower=1)

    if "tgt_lang" in df.columns:
        features["is_zh"] = (df["tgt_lang"] == "zh").astype(float)
        features["is_de"] = (df["tgt_lang"] == "de").astype(float)

    if len(signal_cols) >= 2:
        for i, col_a in enumerate(signal_cols):
            for col_b in signal_cols[i+1:]:
                features[f"{col_a}_minus_{col_b}"] = df[col_a] - df[col_b]
                features[f"{col_a}_times_{col_b}"] = df[col_a] * df[col_b]

    if "doc_id" in df.columns:
        for col in signal_cols:
            doc_stats = df.groupby("doc_id")[col].agg(["mean", "std", "min", "max"])
            doc_stats.columns = [f"{col}_doc_{stat}" for stat in ["mean", "std", "min", "max"]]
            for stat_col in doc_stats.columns:
                features[stat_col] = df["doc_id"].map(doc_stats[stat_col]).values
            features[f"{col}_doc_dev"] = df[col].values - features[f"{col}_doc_mean"].values

    features = features.fillna(0)
    return features


# ---------------------------------------------------------------------------
# 4. Train LightGBM or fall back to weighted average
# ---------------------------------------------------------------------------
test_preds = None

if train is not None and len(shared_signals) >= 1:
    try:
        import lightgbm as lgb
    except ImportError:
        print("WARNING: LightGBM not installed. Falling back to weighted average.")
        train = None

if train is not None and len(shared_signals) >= 1:
    print("\n--- Training LightGBM ---")
    print(f"  Using {len(shared_signals)} shared signals as base features")

    train_features = build_features(train, shared_signals)
    test_features = build_features(test, shared_signals)

    # Verify feature alignment
    if list(train_features.columns) != list(test_features.columns):
        print("ERROR: Feature mismatch between train and test!")
        print(f"  Train: {train_features.shape[1]} features")
        print(f"  Test: {test_features.shape[1]} features")
        # Find mismatches
        train_only = set(train_features.columns) - set(test_features.columns)
        test_only = set(test_features.columns) - set(train_features.columns)
        if train_only:
            print(f"  Only in train: {train_only}")
        if test_only:
            print(f"  Only in test: {test_only}")
        sys.exit(1)

    print(f"  Train features: {train_features.shape}")
    print(f"  Test features: {test_features.shape}")

    X_train = train_features.values
    y_train = train["score"].values

    # Hold out 10% for early stopping
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

    # Validate on dev
    if dev is not None:
        dev_signals_ok = all(c in dev.columns for c in shared_signals)
        if dev_signals_ok:
            dev_features = build_features(dev, shared_signals)
            if list(dev_features.columns) == list(train_features.columns):
                dev_preds = model.predict(dev_features.values)
                dev_copy = dev.copy()
                dev_copy["lgbm_pred"] = dev_preds
                taus = []
                for doc_id, group in dev_copy.groupby("doc_id"):
                    if len(group) < 2:
                        continue
                    tau, _ = stats.kendalltau(group["lgbm_pred"].values, group["score"].values)
                    if not np.isnan(tau):
                        taus.append(tau)
                dev_tau = np.mean(taus) if taus else 0.0
                print(f"\n  Dev per-source Kendall Tau: {dev_tau:.4f}")

                # Per-LP dev tau
                for (src, tgt), lp_group in dev_copy.groupby(["src_lang", "tgt_lang"]):
                    lp_taus = []
                    for doc_id, doc_group in lp_group.groupby("doc_id"):
                        if len(doc_group) < 2:
                            continue
                        tau, _ = stats.kendalltau(doc_group["lgbm_pred"].values, doc_group["score"].values)
                        if not np.isnan(tau):
                            lp_taus.append(tau)
                    lp_tau = np.mean(lp_taus) if lp_taus else 0.0
                    print(f"    {src}->{tgt}: tau={lp_tau:.4f}")

    # Predict on test
    X_test = test_features.values
    test_preds = model.predict(X_test)
    print(f"\n  Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    # Save model
    model.booster_.save_model(os.path.join(args.output_dir, "lgbm_model.txt"))

# ---------------------------------------------------------------------------
# 5. Fallback: weighted average if no LightGBM
# ---------------------------------------------------------------------------
if test_preds is None:
    print("\n--- Weighted Average Fallback ---")
    # Weights from dev performance (tau values)
    DEFAULT_WEIGHTS = {
        "finetuned_score": 0.30,
        "pairwise_score": 0.30,
        "cometkiwi22_score": 0.15,
        "cometkiwi23xxl_score": 0.10,
        "cometkiwi23xxl_finetuned_score": 0.05,
        "metricx_score": 0.05,
        "xcomet_score": 0.03,
        "blaser_score": 0.02,
    }

    active_weights = {k: v for k, v in DEFAULT_WEIGHTS.items() if k in test_signals}
    if not active_weights:
        active_weights = {c: 1.0 / len(test_signals) for c in test_signals}

    total_w = sum(active_weights.values())
    active_weights = {k: v / total_w for k, v in active_weights.items()}

    print(f"  Weights: {active_weights}")
    test_preds = np.zeros(len(test))
    for col, w in active_weights.items():
        test_preds += test[col].values * w


# ---------------------------------------------------------------------------
# 6. Generate submission file (ONE file, all LPs, original row order)
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("GENERATING SUBMISSION FILE")
print("=" * 80)

test["final_score"] = test_preds

# PRIMARY SUBMISSION: one file with ALL scores in original dataset order.
# The evaluation script reads this and splits by LP internally.
submission_file = os.path.join(args.output_dir, "iwslt26test_lgbm_ensemble.jsonl")
with open(submission_file, "w") as f:
    for score in test["final_score"].values:
        f.write(f"{score}\n")
print(f"  {submission_file}: {len(test)} scores (ALL language pairs, original order)")
print(f"    mean={test_preds.mean():.4f}, std={test_preds.std():.4f}, "
      f"min={test_preds.min():.4f}, max={test_preds.max():.4f}")

# Per-LP stats (informational only)
for lp_name, tgt_lang in [("ende", "de"), ("enzh", "zh")]:
    lp_mask = test["tgt_lang"] == tgt_lang
    lp_scores = test.loc[lp_mask, "final_score"].values
    print(f"    {lp_name}: {len(lp_scores)} scores, "
          f"mean={lp_scores.mean():.4f}, std={lp_scores.std():.4f}")

# Save full test predictions for debugging
test.to_parquet(os.path.join(args.output_dir, "test_final_predictions.parquet"), index=False)

# Save metadata
metadata = {
    "team": "pranav",
    "system": "lgbm-ensemble-qe" if train is not None else "weighted-ensemble-qe",
    "task": "iwslt2026-metrics-shared-task",
    "signals_used": shared_signals if train is not None else test_signals,
    "method": "LightGBM trained on 33K scored train samples" if train is not None else "Weighted average",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S") if "time" in dir() else "unknown",
    "test_samples": len(test),
    "ende_samples": int((test["tgt_lang"] == "de").sum()),
    "enzh_samples": int((test["tgt_lang"] == "zh").sum()),
    "score_stats": {
        "ende": {
            "mean": float(test.loc[test["tgt_lang"] == "de", "final_score"].mean()),
            "std": float(test.loc[test["tgt_lang"] == "de", "final_score"].std()),
        },
        "enzh": {
            "mean": float(test.loc[test["tgt_lang"] == "zh", "final_score"].mean()),
            "std": float(test.loc[test["tgt_lang"] == "zh", "final_score"].std()),
        },
    },
}
with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ---------------------------------------------------------------------------
# 7. Also produce per-signal backup submissions (single file, all LPs)
# ---------------------------------------------------------------------------
print("\n--- Backup: per-signal submission files ---")
for col in test_signals:
    backup_file = os.path.join(args.output_dir, f"iwslt26test_{col.replace('_score', '')}.jsonl")
    with open(backup_file, "w") as f:
        for score in test[col].values:
            f.write(f"{score}\n")
    print(f"  {backup_file}: {len(test)} scores")

print("\n" + "=" * 80)
print("SUBMISSION COMPLETE")
print("=" * 80)
print("\nFILE TO SUBMIT:")
print(f"  {submission_file}")
print(f"  ({len(test)} scores — all language pairs in original dataset order)")
print("\nFormat: one bare number per line (json.loads parseable).")
print("The evaluation script splits by LP internally.")
