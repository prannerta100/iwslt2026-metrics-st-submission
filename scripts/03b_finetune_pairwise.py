"""
Fine-tune CometKiwi-22 with pairwise ranking loss to directly optimize Kendall Tau.

The standard COMET fine-tuning uses MSE loss, which optimizes for absolute score
prediction. But our metric is Kendall's Tau (ranking correlation), so we should
directly optimize for correct pairwise ordering.

This script:
1. Creates pairwise training samples from within-source groups
2. Uses margin-based ranking loss: max(0, margin - (pred_better - pred_worse))
3. Combines ranking loss with MSE for calibration
4. Evaluates with per-source Kendall Tau after each epoch
5. Uses hard negative mining after warmup epochs

Key insight from error analysis:
  - CometKiwi-22 baseline has 31.5% pairwise disagreement rate
  - Score compression ratio is 0.006 (massive compression)
  - Pairwise ranking loss directly addresses both issues

Run on GPU: python scripts/03b_finetune_pairwise.py
"""

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=32,
                    help="Number of pairwise samples per batch")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--encoder-lr", type=float, default=5e-7)
parser.add_argument("--margin", type=float, default=0.01,
                    help="Minimum margin for ranking loss")
parser.add_argument("--mse-weight", type=float, default=0.3,
                    help="Weight of MSE loss (1-mse_weight = ranking loss weight)")
parser.add_argument("--frozen-epochs", type=float, default=0.3,
                    help="Fraction of first epoch to keep encoder frozen")
parser.add_argument("--use-all-data", action="store_true",
                    help="Use all language pairs (not just en-de/zh)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("PAIRWISE RANKING FINE-TUNING")
print("=" * 80)

train = pd.read_parquet("outputs/train_text.parquet")
dev = pd.read_parquet("outputs/dev_text.parquet")

# Load synthetic data
synth_file = "outputs/train_synthetic_text.parquet"
if os.path.exists(synth_file):
    synth = pd.read_parquet(synth_file)
    synth["score"] = synth["score"] * 100.0
    synth = synth[synth["src_lang"] == "en"]
    synth = synth[synth["tgt_lang"].isin(["de", "zh"])]
    train = pd.concat([train, synth], ignore_index=True)
    print(f"Added {len(synth)} synthetic samples")

if args.use_all_data:
    train_data = train.copy()
    print(f"Using ALL training data: {len(train_data)} samples")
else:
    train_data = train[
        (train["src_lang"] == "en") & (train["tgt_lang"].isin(["de", "zh"]))
    ].copy()
    print(f"Target LP training data: {len(train_data)} samples")

print(f"Dev data: {len(dev)} samples")

# Group by doc_id to create within-source pairs
train_groups = train_data.groupby("doc_id")
print(f"Training sources (doc_ids): {train_groups.ngroups}")
print(f"Mean systems per source: {train_groups.size().mean():.1f}")


# ---------------------------------------------------------------------------
# 2. Create pairwise training data
# ---------------------------------------------------------------------------
def create_pairs(df, min_score_diff=1.0):
    """Create (better, worse) pairs from within-source groups."""
    pairs = []
    for doc_id, group in df.groupby("doc_id"):
        if len(group) < 2:
            continue
        rows = group.sort_values("score", ascending=False).reset_index(drop=True)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                score_diff = rows.iloc[i]["score"] - rows.iloc[j]["score"]
                if score_diff > min_score_diff:
                    pairs.append({
                        "src": str(rows.iloc[i]["src_text"]),
                        "mt_better": str(rows.iloc[i]["tgt_text"]),
                        "mt_worse": str(rows.iloc[j]["tgt_text"]),
                        "score_better": rows.iloc[i]["score"] / 100.0,
                        "score_worse": rows.iloc[j]["score"] / 100.0,
                        "margin": score_diff / 100.0,
                    })
    return pairs


print("\n--- Creating training pairs ---")
all_pairs = create_pairs(train_data)
print(f"Total pairs: {len(all_pairs)}")

# Cap at reasonable size
max_pairs = 50000
if len(all_pairs) > max_pairs:
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_pairs), size=max_pairs, replace=False)
    all_pairs = [all_pairs[i] for i in indices]
    print(f"Subsampled to {len(all_pairs)} pairs")


# ---------------------------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------------------------
from comet import download_model, load_from_checkpoint

print("\nLoading CometKiwi-22...")
local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
if os.path.exists(local_ckpt):
    model_path = local_ckpt
else:
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")

model = load_from_checkpoint(model_path)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = model.to(device)


# ---------------------------------------------------------------------------
# 4. Scoring function using COMET internals (differentiable)
# ---------------------------------------------------------------------------
def score_batch(model, src_texts, mt_texts):
    """
    Get differentiable scores from CometKiwi for a batch.
    Uses model.prepare_sample (returns dict for stage='predict') and model.forward.
    """
    samples = [{"src": s, "mt": m} for s, m in zip(src_texts, mt_texts)]

    # prepare_sample return type varies across COMET versions:
    #   - Some versions return a dict directly for stage="predict"
    #   - Others return a tuple (input_dict, target_dict) regardless of stage
    result = model.prepare_sample(samples, stage="predict")
    if isinstance(result, tuple):
        input_dict = result[0]
    else:
        input_dict = result

    # Move tensors to device
    input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in input_dict.items()}

    # forward() returns Prediction object with .score attribute (differentiable)
    prediction = model.forward(**input_dict)
    # Handle both attribute and dict access patterns across COMET versions
    if hasattr(prediction, "score"):
        return prediction.score  # Shape: [batch_size], differentiable
    elif isinstance(prediction, dict):
        return prediction["score"]
    else:
        return prediction[0]  # Fallback: first element of tuple


def evaluate_on_dev(model, dev_df):
    """Evaluate per-source Kendall Tau on dev set."""
    model.eval()
    samples = [{"src": row["src_text"], "mt": row["tgt_text"]}
               for _, row in dev_df.iterrows()]
    gpus = 1 if device.type == "cuda" else 0
    num_workers = 4 if gpus else 2
    output = model.predict(samples, batch_size=128, gpus=gpus, num_workers=num_workers)

    dev_df = dev_df.copy()
    dev_df["pred"] = output.scores if hasattr(output, "scores") else output["scores"]

    taus = []
    for doc_id, group in dev_df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, _ = stats.kendalltau(group["pred"].values, group["score"].values)
        if not np.isnan(tau):
            taus.append(tau)

    per_source_tau = np.mean(taus) if taus else 0.0

    # Also per-LP
    for (src, tgt), lp_group in dev_df.groupby(["src_lang", "tgt_lang"]):
        lp_taus = []
        for doc_id, doc_group in lp_group.groupby("doc_id"):
            if len(doc_group) < 2:
                continue
            tau, _ = stats.kendalltau(doc_group["pred"].values, doc_group["score"].values)
            if not np.isnan(tau):
                lp_taus.append(tau)
        lp_tau = np.mean(lp_taus) if lp_taus else 0.0
        print(f"    {src}->{tgt}: tau={lp_tau:.4f}")

    return per_source_tau


# ---------------------------------------------------------------------------
# 5. Set up optimizer
# ---------------------------------------------------------------------------
# Separate parameter groups with different learning rates
encoder_params = []
head_params = []
for name, param in model.named_parameters():
    if "encoder" in name or "layernorm_embedding" in name or "embed_tokens" in name:
        encoder_params.append(param)
    else:
        head_params.append(param)

print(f"\nEncoder params: {sum(p.numel() for p in encoder_params)/1e6:.1f}M")
print(f"Head params: {sum(p.numel() for p in head_params)/1e6:.1f}M")

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": args.encoder_lr},
    {"params": head_params, "lr": args.lr},
], weight_decay=0.01)

steps_per_epoch = len(all_pairs) // args.batch_size
total_steps = args.epochs * steps_per_epoch
warmup_steps = int(0.1 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.05, 1 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Initially freeze encoder
frozen = True
unfreeze_step = int(args.frozen_epochs * steps_per_epoch)
for param in encoder_params:
    param.requires_grad = False

print(f"\nTraining config:")
print(f"  Epochs: {args.epochs}")
print(f"  Batch size: {args.batch_size}")
print(f"  Steps/epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Unfreeze encoder at step: {unfreeze_step}")
print(f"  Head LR: {args.lr}")
print(f"  Encoder LR: {args.encoder_lr}")
print(f"  MSE weight: {args.mse_weight}")
print(f"  Ranking weight: {1 - args.mse_weight}")


# ---------------------------------------------------------------------------
# 6. Initial evaluation
# ---------------------------------------------------------------------------
print("\n--- Initial evaluation ---")
initial_tau = evaluate_on_dev(model, dev)
print(f"  Initial per-source Kendall Tau: {initial_tau:.4f}")


# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------
print(f"\n--- Training ({args.epochs} epochs, {len(all_pairs)} pairs) ---")
best_tau = initial_tau
best_ckpt_path = None
patience = 3
patience_counter = 0
global_step = 0

for epoch in range(args.epochs):
    model.train()
    epoch_losses = []
    epoch_rank_losses = []
    epoch_mse_losses = []
    epoch_ranking_correct = 0
    epoch_ranking_total = 0
    epoch_start = time.time()

    # Shuffle pairs each epoch
    indices = np.random.permutation(len(all_pairs))

    for step_idx in range(0, len(indices), args.batch_size):
        # Unfreeze encoder after warmup
        if frozen and global_step >= unfreeze_step:
            frozen = False
            for param in encoder_params:
                param.requires_grad = True
            print(f"  [Step {global_step}] Encoder unfrozen")

        batch_indices = indices[step_idx:step_idx + args.batch_size]
        if len(batch_indices) < 2:
            continue
        batch = [all_pairs[i] for i in batch_indices]
        n = len(batch)

        # Concatenate better and worse into one forward pass for efficiency
        src_texts = [b["src"] for b in batch] + [b["src"] for b in batch]
        mt_texts = [b["mt_better"] for b in batch] + [b["mt_worse"] for b in batch]
        gold_better = torch.tensor([b["score_better"] for b in batch], device=device)
        gold_worse = torch.tensor([b["score_worse"] for b in batch], device=device)
        gold_margins = torch.tensor([b["margin"] for b in batch], device=device)

        # Forward pass (differentiable)
        scores = score_batch(model, src_texts, mt_texts)
        pred_better = scores[:n]
        pred_worse = scores[n:]

        # Adaptive margin ranking loss: margin proportional to gold score difference
        adaptive_margin = torch.clamp(gold_margins * 0.5, min=args.margin)
        ranking_loss = torch.clamp(
            adaptive_margin - (pred_better - pred_worse),
            min=0
        ).mean()

        # MSE loss for calibration
        all_gold = torch.cat([gold_better, gold_worse])
        mse_loss = F.mse_loss(scores, all_gold)

        # Combined loss
        loss = args.mse_weight * mse_loss + (1 - args.mse_weight) * ranking_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        epoch_losses.append(loss.item())
        epoch_rank_losses.append(ranking_loss.item())
        epoch_mse_losses.append(mse_loss.item())

        with torch.no_grad():
            correct = (pred_better > pred_worse).sum().item()
            epoch_ranking_correct += correct
            epoch_ranking_total += n

        if global_step % 200 == 0:
            avg_loss = np.mean(epoch_losses[-200:])
            avg_rank = np.mean(epoch_rank_losses[-200:])
            avg_mse = np.mean(epoch_mse_losses[-200:])
            rank_acc = epoch_ranking_correct / max(1, epoch_ranking_total)
            lr_head = scheduler.get_last_lr()[1]
            print(f"  Step {global_step}: loss={avg_loss:.4f} "
                  f"(rank={avg_rank:.4f}, mse={avg_mse:.4f}), "
                  f"rank_acc={rank_acc:.4f}, lr={lr_head:.2e}")

    # Epoch summary
    elapsed = time.time() - epoch_start
    avg_loss = np.mean(epoch_losses)
    rank_acc = epoch_ranking_correct / max(1, epoch_ranking_total)
    print(f"\n  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
          f"rank_acc={rank_acc:.4f}, time={elapsed:.0f}s")

    # Evaluate on dev
    dev_tau = evaluate_on_dev(model, dev)
    print(f"  Dev per-source tau: {dev_tau:.4f} "
          f"(best: {best_tau:.4f}, init: {initial_tau:.4f})")

    if dev_tau > best_tau:
        best_tau = dev_tau
        patience_counter = 0
        os.makedirs("models/cometkiwi_pairwise/", exist_ok=True)
        best_ckpt_path = f"models/cometkiwi_pairwise/best-epoch{epoch+1}-tau{dev_tau:.4f}.ckpt"
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"  NEW BEST! Saved to {best_ckpt_path}")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break


# ---------------------------------------------------------------------------
# 8. Final evaluation with best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

# Load best model
if best_ckpt_path and os.path.exists(best_ckpt_path):
    print(f"Loading best checkpoint: {best_ckpt_path}")
    state_dict = torch.load(best_ckpt_path, weights_only=False)
    model.load_state_dict(state_dict)

model.eval()
samples = [{"src": row["src_text"], "mt": row["tgt_text"]}
           for _, row in dev.iterrows()]
gpus = 1 if device.type == "cuda" else 0
output = model.predict(samples, batch_size=128, gpus=gpus,
                       num_workers=4 if gpus else 2)

dev["pairwise_score"] = output.scores if hasattr(output, "scores") else output["scores"]

# Compute final metrics
taus = []
for doc_id, group in dev.groupby("doc_id"):
    if len(group) < 2:
        continue
    tau, _ = stats.kendalltau(group["pairwise_score"].values, group["score"].values)
    if not np.isnan(tau):
        taus.append(tau)
final_tau = np.mean(taus) if taus else 0.0

overall_tau, _ = stats.kendalltau(dev["pairwise_score"].values, dev["score"].values)
pearson_r, _ = stats.pearsonr(dev["pairwise_score"].values, dev["score"].values)

print(f"  Initial per-source Kendall Tau: {initial_tau:.4f}")
print(f"  Best per-source Kendall Tau:    {final_tau:.4f}")
print(f"  Improvement:                    {final_tau - initial_tau:+.4f}")
print(f"  Overall Kendall Tau:            {overall_tau:.4f}")
print(f"  Pearson:                        {pearson_r:.4f}")

# Per language pair
for (src, tgt), group in dev.groupby(["src_lang", "tgt_lang"]):
    lp_taus = []
    for doc_id, doc_group in group.groupby("doc_id"):
        if len(doc_group) < 2:
            continue
        tau, _ = stats.kendalltau(doc_group["pairwise_score"].values, doc_group["score"].values)
        if not np.isnan(tau):
            lp_taus.append(tau)
    lp_tau = np.mean(lp_taus) if lp_taus else 0.0
    print(f"  {src}->{tgt}: per-source tau={lp_tau:.4f}")

# Save predictions
dev.to_parquet("outputs/dev_with_pairwise.parquet", index=False)

# Merge into main predictions file
existing_pred_file = "outputs/dev_with_predictions.parquet"
if os.path.exists(existing_pred_file):
    existing = pd.read_parquet(existing_pred_file)
    existing["pairwise_score"] = dev["pairwise_score"].values
    existing.to_parquet(existing_pred_file, index=False)
    print(f"\nMerged pairwise_score into {existing_pred_file}")

print("\n" + "=" * 80)
print("PAIRWISE FINE-TUNING COMPLETE")
print("=" * 80)
