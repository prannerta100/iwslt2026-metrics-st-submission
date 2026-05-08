"""
Generate per-LP submission files from scored outputs.

Reads:
  data/test.jsonl — to know LP labels and row order
  outputs/test_finetuned_scores.json — primary submission (fine-tuned CK-23-XXL)
  outputs/test_pretrained_scores.json — contrastive (pretrained CK-23-XXL)
  outputs/dev_scores.json + data/dev.jsonl — for local validation

Outputs:
  submission/primary_ende.txt      — one score per line, en-de test rows
  submission/primary_enzh.txt      — one score per line, en-zh test rows
  submission/contrastive1_ende.txt — pretrained CK-23-XXL scores
  submission/contrastive1_enzh.txt — pretrained CK-23-XXL scores

Validation:
  Runs organizers' eval script on dev predictions to confirm format.

Submission format (from organizers):
  - One file per language pair
  - One score per line (parseable by json.loads)
  - Same order as rows in HF test dataset for that LP
"""

import os
import sys
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssl_fix

os.makedirs("submission", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load test data to know LP assignments and row order
# ---------------------------------------------------------------------------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


test_data = load_jsonl("data/test.jsonl")
print(f"Test data: {len(test_data)} rows")

# Map LP -> list of indices (in HF dataset order)
lp_indices = {}
for i, row in enumerate(test_data):
    lp = f"{row['src_lang']}{row['tgt_lang']}"
    if lp not in lp_indices:
        lp_indices[lp] = []
    lp_indices[lp].append(i)

print(f"LPs: {', '.join(f'{lp}={len(idxs)}' for lp, idxs in lp_indices.items())}")


# ---------------------------------------------------------------------------
# 2. Generate submission files
# ---------------------------------------------------------------------------
def write_submission(scores, name, description):
    """Write per-LP submission files from a full score list."""
    files_written = []
    for lp, indices in lp_indices.items():
        lp_scores = [scores[i] for i in indices]
        path = f"submission/{name}_{lp}.txt"
        with open(path, "w") as f:
            for s in lp_scores:
                f.write(f"{s}\n")
        files_written.append((path, len(lp_scores)))

    print(f"\n  {description}:")
    for path, n in files_written:
        print(f"    {path}: {n} scores")
    return files_written


# Primary: fine-tuned scores
finetuned_path = "outputs/test_finetuned_scores.json"
if os.path.exists(finetuned_path):
    with open(finetuned_path) as f:
        finetuned_scores = json.load(f)
    assert len(finetuned_scores) == len(test_data), \
        f"Score count mismatch: {len(finetuned_scores)} vs {len(test_data)} test rows"
    write_submission(finetuned_scores, "primary", "PRIMARY (fine-tuned CK-23-XXL pairwise)")
else:
    print(f"\n  WARNING: {finetuned_path} not found. No primary submission.")
    print(f"           Run 03_score_test.py first.")

# Contrastive: pretrained scores
pretrained_path = "outputs/test_pretrained_scores.json"
if os.path.exists(pretrained_path):
    with open(pretrained_path) as f:
        pretrained_scores = json.load(f)
    assert len(pretrained_scores) == len(test_data), \
        f"Score count mismatch: {len(pretrained_scores)} vs {len(test_data)} test rows"
    write_submission(pretrained_scores, "contrastive1", "CONTRASTIVE (pretrained CK-23-XXL)")
else:
    print(f"\n  WARNING: {pretrained_path} not found. No contrastive submission.")


# ---------------------------------------------------------------------------
# 3. Validate format on dev
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  FORMAT VALIDATION (on dev set)")
print("=" * 70)

dev_data = load_jsonl("data/dev.jsonl")
dev_scores_path = "outputs/dev_scores.json"

if os.path.exists(dev_scores_path):
    with open(dev_scores_path) as f:
        dev_scores = json.load(f)

    # Write dev predictions in organizers' format (combined file for eval script)
    dev_metric_path = "submission/dev_validation.jsonl"
    with open(dev_metric_path, "w") as f:
        for s in dev_scores:
            f.write(f"{s}\n")

    print(f"\n  Wrote {dev_metric_path} ({len(dev_scores)} scores)")

    # Try to run organizers' eval script
    eval_dir = "/tmp/iwslt26-metrics"
    if os.path.isdir(eval_dir):
        print(f"\n  Running organizers' eval script...")
        result = subprocess.run(
            ["python3", "evaluation",
             "-i", os.path.abspath("data/dev.jsonl"),
             "-m", os.path.abspath(dev_metric_path)],
            cwd=eval_dir,
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(result.stdout)
            print("  FORMAT VALIDATED — organizers' script ran successfully!")
        else:
            print(f"  Eval script failed: {result.stderr}")
    else:
        print(f"  Organizers' repo not found at {eval_dir}")
        print(f"  Clone it: git clone https://github.com/zouharvi/iwslt26-metrics /tmp/iwslt26-metrics")

    # Self-validate: check that each line is parseable as a number
    print("\n  Self-validation:")
    for path in sorted(os.listdir("submission")):
        if not path.endswith(".txt"):
            continue
        full_path = f"submission/{path}"
        with open(full_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            try:
                val = json.loads(line.strip())
                assert isinstance(val, (int, float)), f"Line {i}: not a number: {val}"
            except Exception as e:
                print(f"  FAIL: {full_path} line {i}: {e}")
                break
        else:
            print(f"  OK: {full_path} — {len(lines)} valid scores")
else:
    print(f"\n  No dev scores found at {dev_scores_path}. Skipping validation.")


# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  SUBMISSION FILES READY")
print("=" * 70)

print("\nFiles to submit (email to maike.zuefle@kit.edu AND vzouhar@ethz.ch):")
for f in sorted(os.listdir("submission")):
    if f.endswith(".txt") and not f.startswith("dev_"):
        full = f"submission/{f}"
        n_lines = sum(1 for _ in open(full))
        print(f"  {full}: {n_lines} scores")

print("\nFormat: one score per line, same order as HF test dataset rows for that LP.")
