"""
Deep error analysis: understand where and why the metric fails.

This script examines:
1. Which source segments have the worst ranking performance
2. Score distribution mismatches between prediction and gold
3. Language-pair-specific failure modes
4. Relationship between prediction confidence and accuracy
5. System-pair confusion analysis

The insights from this analysis directly inform:
- What kind of training data to emphasize
- Whether calibration/normalization helps
- Where ensemble diversity matters most

Run: python scripts/09_error_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def per_source_tau_detailed(df, pred_col, gold_col="score"):
    """Return per-source Kendall Tau with details."""
    results = []
    for doc_id, group in df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, pvalue = stats.kendalltau(group[pred_col].values, group[gold_col].values)
        if not np.isnan(tau):
            results.append({
                "doc_id": doc_id,
                "tau": tau,
                "pvalue": pvalue,
                "n_systems": len(group),
                "gold_range": group[gold_col].max() - group[gold_col].min(),
                "pred_range": group[pred_col].max() - group[pred_col].min(),
                "gold_mean": group[gold_col].mean(),
                "pred_mean": group[pred_col].mean(),
                "src_lang": group["src_lang"].iloc[0],
                "tgt_lang": group["tgt_lang"].iloc[0],
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    pred_file = "outputs/dev_with_predictions.parquet"
    if not os.path.exists(pred_file):
        print(f"ERROR: {pred_file} not found.")
        sys.exit(1)

    dev = pd.read_parquet(pred_file)
    print(f"Loaded {len(dev)} dev examples")

    # Determine which prediction columns are available
    pred_cols = [c for c in dev.columns if c.endswith("_score") and c != "score"]
    print(f"Prediction columns: {pred_cols}")

    for pred_col in pred_cols:
        print(f"\n{'='*60}")
        print(f"Analysis for: {pred_col}")
        print(f"{'='*60}")

        # 1. Per-source tau distribution
        tau_df = per_source_tau_detailed(dev, pred_col)
        print(f"\n--- Per-source Tau Distribution ---")
        print(f"  N sources: {len(tau_df)}")
        print(f"  Mean tau:  {tau_df['tau'].mean():.4f}")
        print(f"  Median tau: {tau_df['tau'].median():.4f}")
        print(f"  Std tau:   {tau_df['tau'].std():.4f}")
        print(f"  Min tau:   {tau_df['tau'].min():.4f}")
        print(f"  Max tau:   {tau_df['tau'].max():.4f}")

        # Tau distribution by quintile
        print(f"\n  Tau distribution:")
        for pct in [10, 25, 50, 75, 90]:
            print(f"    P{pct}: {tau_df['tau'].quantile(pct/100):.4f}")

        # 2. Worst-performing sources
        print(f"\n--- Worst 10 Sources (lowest tau) ---")
        worst = tau_df.nsmallest(10, "tau")
        for _, row in worst.iterrows():
            print(f"  doc_id={row['doc_id']}: tau={row['tau']:.4f}, "
                  f"n_sys={row['n_systems']}, gold_range={row['gold_range']:.1f}, "
                  f"pred_range={row['pred_range']:.4f}, {row['src_lang']}->{row['tgt_lang']}")

        # 3. What makes sources hard to rank?
        print(f"\n--- Correlation between source properties and tau ---")
        correlates = ["n_systems", "gold_range", "pred_range", "gold_mean"]
        for c in correlates:
            r, p = stats.spearmanr(tau_df[c], tau_df["tau"])
            print(f"  {c} vs tau: Spearman r={r:.4f}, p={p:.4f}")

        # 4. Per language pair breakdown
        print(f"\n--- Per Language Pair ---")
        for (src, tgt), lp_taus in tau_df.groupby(["src_lang", "tgt_lang"]):
            print(f"  {src}->{tgt}: mean_tau={lp_taus['tau'].mean():.4f}, "
                  f"n_sources={len(lp_taus)}, "
                  f"median_gold_range={lp_taus['gold_range'].median():.1f}")

        # 5. Score range analysis
        print(f"\n--- Prediction vs Gold Score Analysis ---")
        # Are we compressing the score range?
        gold_std = dev["score"].std()
        pred_std = dev[pred_col].std()
        print(f"  Gold score std:  {gold_std:.2f}")
        print(f"  Pred score std:  {pred_std:.4f}")
        print(f"  Compression ratio: {pred_std/gold_std:.4f}")

        # Score range by gold quintile
        dev["gold_quintile"] = pd.qcut(dev["score"], 5, labels=False, duplicates="drop")
        print(f"\n  Mean prediction by gold score quintile:")
        for q, group in dev.groupby("gold_quintile"):
            gold_range = f"[{group['score'].min():.0f}-{group['score'].max():.0f}]"
            print(f"    Q{int(q)}: gold_mean={group['score'].mean():.1f} {gold_range}, "
                  f"pred_mean={group[pred_col].mean():.4f}, "
                  f"pred_std={group[pred_col].std():.4f}")

        # 6. Pairwise agreement analysis
        print(f"\n--- Pairwise Agreement Analysis ---")
        agree_count = 0
        disagree_count = 0
        tie_count = 0
        disagree_examples = []

        for doc_id, group in dev.groupby("doc_id"):
            if len(group) < 2:
                continue
            systems = group.index.tolist()
            for i, j in combinations(range(len(systems)), 2):
                idx_i, idx_j = systems[i], systems[j]
                gold_diff = group.loc[idx_i, "score"] - group.loc[idx_j, "score"]
                pred_diff = group.loc[idx_i, pred_col] - group.loc[idx_j, pred_col]

                if abs(gold_diff) < 1.0:  # Effective tie in gold
                    tie_count += 1
                elif (gold_diff > 0 and pred_diff > 0) or (gold_diff < 0 and pred_diff < 0):
                    agree_count += 1
                else:
                    disagree_count += 1
                    if len(disagree_examples) < 20:
                        disagree_examples.append({
                            "doc_id": doc_id,
                            "gold_diff": gold_diff,
                            "pred_diff": pred_diff,
                            "sys_i": group.loc[idx_i, "tgt_system"],
                            "sys_j": group.loc[idx_j, "tgt_system"],
                        })

        total_pairs = agree_count + disagree_count + tie_count
        print(f"  Total pairs: {total_pairs}")
        print(f"  Agreements: {agree_count} ({agree_count/total_pairs*100:.1f}%)")
        print(f"  Disagreements: {disagree_count} ({disagree_count/total_pairs*100:.1f}%)")
        print(f"  Ties (gold diff < 1): {tie_count} ({tie_count/total_pairs*100:.1f}%)")

        if disagree_examples:
            print(f"\n  Sample disagreements (metric ranks wrong):")
            for ex in disagree_examples[:5]:
                print(f"    doc={ex['doc_id']}: gold_diff={ex['gold_diff']:.1f}, "
                      f"pred_diff={ex['pred_diff']:.4f}, "
                      f"{ex['sys_i']} vs {ex['sys_j']}")

        # 7. Examine specific failure cases in detail
        print(f"\n--- Detailed Failure Analysis (worst 3 sources) ---")
        for _, row in worst.head(3).iterrows():
            doc_group = dev[dev["doc_id"] == row["doc_id"]]
            print(f"\n  doc_id={row['doc_id']} (tau={row['tau']:.4f}):")
            print(f"  Source: '{doc_group['src_text'].iloc[0][:100]}...'")
            for _, seg in doc_group.sort_values("score", ascending=False).iterrows():
                print(f"    System={seg['tgt_system']}: gold={seg['score']:.1f}, "
                      f"pred={seg[pred_col]:.4f}")
                print(f"    Translation: '{seg['tgt_text'][:80]}...'")

    # 8. Cross-signal agreement (if multiple signals)
    if len(pred_cols) >= 2:
        print(f"\n{'='*60}")
        print("CROSS-SIGNAL ANALYSIS")
        print(f"{'='*60}")

        for col_a, col_b in combinations(pred_cols, 2):
            r, _ = stats.pearsonr(dev[col_a].values, dev[col_b].values)
            tau, _ = stats.kendalltau(dev[col_a].values, dev[col_b].values)
            print(f"\n  {col_a} vs {col_b}:")
            print(f"    Pearson: {r:.4f}")
            print(f"    Kendall Tau: {tau:.4f}")

            # Where do they disagree most?
            # Normalize both to same scale
            norm_a = (dev[col_a] - dev[col_a].mean()) / dev[col_a].std()
            norm_b = (dev[col_b] - dev[col_b].mean()) / dev[col_b].std()
            dev["_signal_diff"] = (norm_a - norm_b).abs()

            top_disagree = dev.nlargest(5, "_signal_diff")
            print(f"    Top disagreement examples:")
            for _, row in top_disagree.iterrows():
                print(f"      gold={row['score']:.1f}, "
                      f"{col_a}={row[col_a]:.4f}, {col_b}={row[col_b]:.4f}, "
                      f"diff={row['_signal_diff']:.4f}")
            dev.drop("_signal_diff", axis=1, inplace=True)

    print("\n" + "=" * 80)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 80)
