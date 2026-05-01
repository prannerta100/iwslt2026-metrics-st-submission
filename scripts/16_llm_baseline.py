"""
LLM-as-Judge baseline for translation quality estimation.

Uses a debate-style prompt (adapted from coherence research) where the LLM:
  1. Argues FOR high quality with evidence
  2. Argues AGAINST (for low quality) with evidence
  3. Concludes with a final score

Uses the Webex LLM proxy (WEBEX_TOKEN env var) with gpt-4.1-mini.

Run:
  WEBEX_TOKEN=your_token poetry run python scripts/16_llm_baseline.py
  WEBEX_TOKEN=your_token poetry run python scripts/16_llm_baseline.py --dataset test --max-workers 50
  WEBEX_TOKEN=your_token poetry run python scripts/16_llm_baseline.py --dataset dev --max-samples 500
"""

import os
import sys
import json
import time
import re
import argparse
import concurrent.futures

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="dev", choices=["dev", "test", "train"],
                    help="Which dataset to score")
parser.add_argument("--max-workers", type=int, default=30,
                    help="Concurrent LLM API calls")
parser.add_argument("--max-samples", type=int, default=None,
                    help="Limit number of samples (for testing)")
parser.add_argument("--model", type=str, default="gpt-4.1-mini")
parser.add_argument("--output-dir", type=str, default="outputs/")
parser.add_argument("--force", action="store_true",
                    help="Force re-run even if cached parquet exists")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 0. Check for cached results — skip expensive LLM calls if already scored
# ---------------------------------------------------------------------------
cache_path = os.path.join(args.output_dir, f"llm_debate_{args.dataset}.parquet")
if os.path.exists(cache_path) and not args.max_samples and not args.force:
    cached = pd.read_parquet(cache_path)
    if "llm_debate_score" in cached.columns:
        print(f"Found cached LLM debate scores at {cache_path} ({len(cached)} rows)")
        print(f"Score stats: mean={cached['llm_debate_score'].mean():.2f}, "
              f"std={cached['llm_debate_score'].std():.2f}")
        print("Skipping LLM inference (use --force to re-run)")
        # Still evaluate if gold scores available
        if "score" in cached.columns and args.dataset in ("dev", "train"):
            from scipy import stats
            taus = []
            for doc_id, group in cached.groupby("doc_id"):
                if len(group) < 2:
                    continue
                tau, _ = stats.kendalltau(group["llm_debate_score"].values, group["score"].values)
                if not np.isnan(tau):
                    taus.append(tau)
            per_source_tau = np.mean(taus) if taus else 0.0
            overall_tau, _ = stats.kendalltau(cached["llm_debate_score"].values, cached["score"].values)
            print(f"\n--- Cached Evaluation ---")
            print(f"  Per-source Kendall Tau: {per_source_tau:.4f}")
            print(f"  Overall Kendall Tau:    {overall_tau:.4f}")
        sys.exit(0)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("LLM-AS-JUDGE BASELINE (Debate Prompt)")
print("=" * 80)

if args.dataset == "dev":
    if os.path.exists("outputs/dev_text.parquet"):
        df = pd.read_parquet("outputs/dev_text.parquet")
    else:
        from datasets import load_dataset
        ds = load_dataset("maikezu/iwslt2026-metrics-shared-train-dev", split="dev")
        df = ds.to_pandas()
elif args.dataset == "train":
    if os.path.exists("outputs/train_text.parquet"):
        df = pd.read_parquet("outputs/train_text.parquet")
    else:
        from datasets import load_dataset
        ds = load_dataset("maikezu/iwslt2026-metrics-shared-train-dev", split="train")
        df = ds.to_pandas()
elif args.dataset == "test":
    if os.path.exists("submission/test_predictions.parquet"):
        df = pd.read_parquet("submission/test_predictions.parquet")
    else:
        from datasets import load_dataset
        ds = load_dataset("maikezu/iwslt2026-metrics-shared-test")
        if "audio" in ds["test"].column_names:
            ds["test"] = ds["test"].remove_columns(["audio"])
        df = ds["test"].to_pandas()

if args.max_samples:
    df = df.head(args.max_samples)

print(f"Dataset: {args.dataset}, {len(df)} samples")
print(f"Model: {args.model}, Workers: {args.max_workers}")

# ---------------------------------------------------------------------------
# 2. Debate prompt for translation QE
# ---------------------------------------------------------------------------
QE_DEBATE_PROMPT = """\
You are evaluating the quality of a machine translation. Given the source text and its translation, assess how well the translation conveys the meaning of the source.

Source language: {src_lang}
Target language: {tgt_lang}

Source text:
{src_text}

Translation:
{tgt_text}

Emulate a debate to reach your assessment:

1. Argue FOR high quality: List specific evidence that the translation is accurate, fluent, and complete. Quote relevant parts.
2. Argue AGAINST (for low quality): List specific errors — mistranslations, omissions, additions, grammatical errors, or unnatural phrasing. Quote relevant parts.
3. Conclude: Weighing both arguments, assign a final quality score from 0 to 100, where:
   - 90-100: Perfect or near-perfect translation
   - 70-89: Good translation with minor issues
   - 50-69: Adequate but with notable errors
   - 30-49: Poor translation with major errors
   - 0-29: Very poor, mostly wrong or incomprehensible

IMPORTANT: Be calibrated. Most machine translations from modern systems score 60-85. Reserve 90+ for truly flawless output and below 30 for gibberish.

Format your response as:

For: [your argument for high quality]

Against: [your argument for low quality]

Score: [0-100]
"""

# ---------------------------------------------------------------------------
# 3. LLM client
# ---------------------------------------------------------------------------
def get_client():
    from openai import OpenAI
    return OpenAI(
        base_url=os.environ.get(
            "LLM_PROXY_URL",
            "https://llm-proxy.us-east-2.int.infra.intelligence.webex.com/openai/v1",
        ),
        api_key=os.environ.get("WEBEX_TOKEN", "not-set"),
    )


def score_one(client, row) -> dict:
    src_lang_map = {"en": "English", "de": "German", "zh": "Chinese"}
    tgt_lang_map = {"de": "German", "zh": "Chinese", "en": "English"}

    prompt = QE_DEBATE_PROMPT.format(
        src_lang=src_lang_map.get(row.get("src_lang", "en"), row.get("src_lang", "en")),
        tgt_lang=tgt_lang_map.get(row.get("tgt_lang", "de"), row.get("tgt_lang", "de")),
        src_text=str(row["src_text"])[:2000],
        tgt_text=str(row["tgt_text"])[:2000],
    )

    try:
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000,
        )
        text = resp.choices[0].message.content.strip()
        m_score = re.search(r"Score:\s*(\d+)", text)
        if m_score:
            score = int(m_score.group(1))
            score = max(0, min(100, score))
        else:
            score = 50
        return {"score": score, "raw": text}
    except Exception as e:
        return {"score": 50, "raw": f"Error: {e}"}


# ---------------------------------------------------------------------------
# 4. Score all samples with thread pool
# ---------------------------------------------------------------------------
print(f"\nScoring {len(df)} samples with {args.max_workers} concurrent workers...")
start = time.time()

client = get_client()
results = [None] * len(df)

with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
    futures = {}
    for i, (_, row) in enumerate(df.iterrows()):
        future = executor.submit(score_one, client, row)
        futures[future] = i

    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="LLM scoring"):
        idx = futures[future]
        results[idx] = future.result()

elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s ({len(df)/elapsed:.1f} samples/s)")

# Extract scores
llm_scores = np.array([r["score"] for r in results], dtype=float)
df["llm_debate_score"] = llm_scores

print(f"Score stats: mean={llm_scores.mean():.2f}, std={llm_scores.std():.2f}, "
      f"min={llm_scores.min():.0f}, max={llm_scores.max():.0f}")

# ---------------------------------------------------------------------------
# 5. Evaluate on dev (if gold scores available)
# ---------------------------------------------------------------------------
if "score" in df.columns and args.dataset in ("dev", "train"):
    from scipy import stats

    taus = []
    for doc_id, group in df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, _ = stats.kendalltau(group["llm_debate_score"].values, group["score"].values)
        if not np.isnan(tau):
            taus.append(tau)
    per_source_tau = np.mean(taus) if taus else 0.0

    overall_tau, _ = stats.kendalltau(df["llm_debate_score"].values, df["score"].values)
    pearson_r, _ = stats.pearsonr(df["llm_debate_score"].values, df["score"].values)

    print(f"\n--- Dev Evaluation ---")
    print(f"  Per-source Kendall Tau: {per_source_tau:.4f}")
    print(f"  Overall Kendall Tau:    {overall_tau:.4f}")
    print(f"  Pearson r:              {pearson_r:.4f}")

    for (src, tgt), lp_group in df.groupby(["src_lang", "tgt_lang"]):
        lp_taus = []
        for doc_id, doc_group in lp_group.groupby("doc_id"):
            if len(doc_group) < 2:
                continue
            tau, _ = stats.kendalltau(doc_group["llm_debate_score"].values, doc_group["score"].values)
            if not np.isnan(tau):
                lp_taus.append(tau)
        lp_tau = np.mean(lp_taus) if lp_taus else 0.0
        print(f"    {src}->{tgt}: tau={lp_tau:.4f}")

# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------
output_file = os.path.join(args.output_dir, f"llm_debate_{args.dataset}.parquet")
df.to_parquet(output_file, index=False)
print(f"\nSaved to {output_file}")

# Save combined score file for test submission (all LPs, original row order)
if args.dataset == "test":
    os.makedirs("submission", exist_ok=True)
    score_file = "submission/iwslt26test_llm_debate.jsonl"
    with open(score_file, "w") as f:
        for s in df["llm_debate_score"].values:
            f.write(f"{s}\n")
    print(f"  {score_file}: {len(df)} scores (all LPs, original order)")

print("\nDone.")
