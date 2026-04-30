"""
Generate a compact results report (JSON) with dev scores for all methods.

Outputs: outputs/results_report.json

Fields per method:
  - name: method name
  - tau_overall: overall Kendall tau on dev
  - tau_per_source: per-source Kendall tau (primary metric)
  - tau_ende: per-source tau for en-de
  - tau_enzh: per-source tau for en-zh

Final field: "best_method" — the method with highest per-source tau,
which will be used for submission.

Run:
  poetry run python scripts/17_generate_report.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

# ---------------------------------------------------------------------------
# 1. Load dev predictions
# ---------------------------------------------------------------------------
dev_file = "outputs/dev_with_predictions.parquet"
if not os.path.exists(dev_file):
    print(f"ERROR: {dev_file} not found. Run pipeline phases 1-3 first.")
    sys.exit(1)

dev = pd.read_parquet(dev_file)
print(f"Dev set: {len(dev)} samples, {dev['doc_id'].nunique()} docs")

# Check for LLM debate scores
llm_file = "outputs/llm_debate_dev.parquet"
if os.path.exists(llm_file):
    llm_df = pd.read_parquet(llm_file)
    if "llm_debate_score" in llm_df.columns and len(llm_df) == len(dev):
        dev["llm_debate_score"] = llm_df["llm_debate_score"].values
        print("  Added llm_debate_score from llm_debate_dev.parquet")

# Check for LightGBM dev predictions
lgbm_file = "outputs/dev_ensemble_advanced.parquet"
if os.path.exists(lgbm_file):
    lgbm_df = pd.read_parquet(lgbm_file)
    if "lgbm_score" in lgbm_df.columns and len(lgbm_df) == len(dev):
        dev["lgbm_score"] = lgbm_df["lgbm_score"].values
        print("  Added lgbm_score from dev_ensemble_advanced.parquet")
    if "weighted_ensemble_score" in lgbm_df.columns and len(lgbm_df) == len(dev):
        dev["weighted_ensemble_score"] = lgbm_df["weighted_ensemble_score"].values

# ---------------------------------------------------------------------------
# 2. Evaluate all available methods
# ---------------------------------------------------------------------------
ALL_METHODS = [
    "cometkiwi22_score",
    "xcomet_score",
    "finetuned_score",
    "pairwise_score",
    "metricx_score",
    "blaser_score",
    "cometkiwi23xxl_score",
    "cometkiwi23xxl_finetuned_score",
    "llm_debate_score",
    "lgbm_score",
    "weighted_ensemble_score",
]

available_methods = [m for m in ALL_METHODS if m in dev.columns]
print(f"\nAvailable methods ({len(available_methods)}): {available_methods}")


def eval_method(df, pred_col, gold_col="score"):
    """Compute per-source tau (overall + per LP)."""
    result = {}

    # Overall tau
    overall_tau, _ = stats.kendalltau(df[pred_col].values, df[gold_col].values)
    result["tau_overall"] = round(float(overall_tau), 4) if not np.isnan(overall_tau) else 0.0

    # Per-source tau
    taus = []
    for doc_id, group in df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, _ = stats.kendalltau(group[pred_col].values, group[gold_col].values)
        if not np.isnan(tau):
            taus.append(tau)
    result["tau_per_source"] = round(np.mean(taus), 4) if taus else 0.0

    # Per-LP tau
    for (src, tgt), lp_group in df.groupby(["src_lang", "tgt_lang"]):
        lp_taus = []
        for doc_id, doc_group in lp_group.groupby("doc_id"):
            if len(doc_group) < 2:
                continue
            tau, _ = stats.kendalltau(doc_group[pred_col].values, doc_group[gold_col].values)
            if not np.isnan(tau):
                lp_taus.append(tau)
        lp_tau = round(np.mean(lp_taus), 4) if lp_taus else 0.0
        result[f"tau_{src}{tgt}"] = lp_tau

    return result


# ---------------------------------------------------------------------------
# 3. Build report
# ---------------------------------------------------------------------------
print("\n--- Results ---")
print(f"{'Method':<35} {'tau_per_source':>14} {'tau_ende':>10} {'tau_enzh':>10}")
print("-" * 72)

report = {"methods": [], "best_method": None, "best_tau": 0.0}

for method in available_methods:
    result = eval_method(dev, method)
    entry = {"name": method, **result}
    report["methods"].append(entry)

    tau = result["tau_per_source"]
    ende = result.get("tau_ende", 0.0)
    enzh = result.get("tau_enzh", 0.0)
    print(f"  {method:<33} {tau:>14.4f} {ende:>10.4f} {enzh:>10.4f}")

    if tau > report["best_tau"]:
        report["best_tau"] = tau
        report["best_method"] = method

# Sort methods by tau_per_source descending
report["methods"].sort(key=lambda x: x["tau_per_source"], reverse=True)

print(f"\n  BEST: {report['best_method']} (tau={report['best_tau']:.4f})")

# ---------------------------------------------------------------------------
# 4. Save report
# ---------------------------------------------------------------------------
report_file = os.path.join("outputs", "results_report.json")
with open(report_file, "w") as f:
    json.dump(report, f, indent=2)
print(f"\nSaved to {report_file}")
