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
from torch.utils.data import Dataset, DataLoader
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=8,
                    help="Number of SOURCE segments per batch (each with multiple systems)")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--encoder-lr", type=float, default=5e-7)
parser.add_argument("--margin", type=float, default=0.01,
                    help="Margin for ranking loss")
parser.add_argument("--mse-weight", type=float, default=0.3,
                    help="Weight of MSE loss (1-mse_weight = ranking loss weight)")
parser.add_argument("--frozen-epochs", type=float, default=0.3,
                    help="Fraction of first epoch to keep encoder frozen")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("PAIRWISE RANKING FINE-TUNING")
print("=" * 80)

train = pd.read_parquet("outputs/train_text.parquet")
dev = pd.read_parquet("outputs/dev_text.parquet")

# Also load synthetic data
synth_file = "outputs/train_synthetic_text.parquet"
if os.path.exists(synth_file):
    synth = pd.read_parquet(synth_file)
    synth["score"] = synth["score"] * 100.0
    # Only use target language pairs
    synth = synth[synth["src_lang"] == "en"]
    synth = synth[synth["tgt_lang"].isin(["de", "zh"])]
    train = pd.concat([train, synth], ignore_index=True)
    print(f"Added {len(synth)} synthetic samples")

# Filter to target language pairs for focused training
train_target = train[
    (train["src_lang"] == "en") & (train["tgt_lang"].isin(["de", "zh"]))
].copy()
print(f"Target LP training data: {len(train_target)} samples")
print(f"Dev data: {len(dev)} samples")

# Group by doc_id to create within-source pairs
train_groups = train_target.groupby("doc_id")
print(f"Training sources (doc_ids): {train_groups.ngroups}")
print(f"Mean systems per source: {train_groups.size().mean():.1f}")


# ---------------------------------------------------------------------------
# 2. Pairwise ranking dataset
# ---------------------------------------------------------------------------
class PairwiseRankingDataset(Dataset):
    """
    Samples pairs of translations for the same source segment.
    Each item returns (src, mt_better, mt_worse, margin).
    """

    def __init__(self, df, pairs_per_source=10):
        self.pairs = []
        for doc_id, group in df.groupby("doc_id"):
            if len(group) < 2:
                continue
            rows = group.sort_values("score", ascending=False).reset_index(drop=True)
            # Create all pairs (better, worse) ordered by gold score
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    score_diff = rows.iloc[i]["score"] - rows.iloc[j]["score"]
                    if score_diff > 1.0:  # Only clear preferences
                        self.pairs.append({
                            "src": rows.iloc[i]["src_text"],
                            "mt_better": rows.iloc[i]["tgt_text"],
                            "mt_worse": rows.iloc[j]["tgt_text"],
                            "score_better": rows.iloc[i]["score"] / 100.0,
                            "score_worse": rows.iloc[j]["score"] / 100.0,
                            "margin": score_diff / 100.0,
                        })

        # Subsample if too many pairs
        if len(self.pairs) > pairs_per_source * df["doc_id"].nunique():
            rng = np.random.RandomState(42)
            indices = rng.choice(len(self.pairs),
                                 size=pairs_per_source * df["doc_id"].nunique(),
                                 replace=False)
            self.pairs = [self.pairs[i] for i in indices]

        print(f"  Created {len(self.pairs)} training pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


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

# Detect device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = model.to(device)


# ---------------------------------------------------------------------------
# 4. Training loop with pairwise ranking loss
# ---------------------------------------------------------------------------
print("\n--- Creating training pairs ---")
train_dataset = PairwiseRankingDataset(train_target, pairs_per_source=15)

# Separate parameter groups with different learning rates
encoder_params = []
head_params = []
for name, param in model.named_parameters():
    if "encoder" in name or "layernorm_embedding" in name or "embed_tokens" in name:
        encoder_params.append(param)
    else:
        head_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": encoder_params, "lr": args.encoder_lr},
    {"params": head_params, "lr": args.lr},
], weight_decay=0.01)

# Learning rate scheduler
total_steps = args.epochs * len(train_dataset) // args.batch_size
warmup_steps = int(0.1 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    return max(0.1, 1 - (step - warmup_steps) / max(1, total_steps - warmup_steps))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Initially freeze encoder
frozen = True
unfreeze_step = int(args.frozen_epochs * len(train_dataset) / args.batch_size)
for param in encoder_params:
    param.requires_grad = False


def compute_score(model, src_texts, mt_texts):
    """Compute CometKiwi scores for a batch of (src, mt) pairs."""
    samples = [{"src": s, "mt": m} for s, m in zip(src_texts, mt_texts)]
    # Use model's internal encode + predict pipeline
    # We need to go through the model's forward pass properly
    output = model.predict(
        samples,
        batch_size=len(samples),
        gpus=1 if torch.cuda.is_available() else 0,
        num_workers=2,
    )
    return torch.tensor(output.scores, device=device, dtype=torch.float, requires_grad=False)


def evaluate_on_dev(model, dev_df):
    """Evaluate per-source Kendall Tau on dev set."""
    model.eval()
    samples = [{"src": row["src_text"], "mt": row["tgt_text"]}
               for _, row in dev_df.iterrows()]

    # Use model.predict for evaluation
    gpus = 1 if device.type == "cuda" else 0
    num_workers = 4 if gpus else 2
    output = model.predict(samples, batch_size=32, gpus=gpus, num_workers=num_workers)

    dev_df = dev_df.copy()
    dev_df["pred"] = output.scores

    taus = []
    for doc_id, group in dev_df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, _ = stats.kendalltau(group["pred"].values, group["score"].values)
        if not np.isnan(tau):
            taus.append(tau)
    return np.mean(taus) if taus else 0.0


# Initial evaluation
print("\n--- Initial evaluation ---")
initial_tau = evaluate_on_dev(model, dev)
print(f"  Initial per-source Kendall Tau: {initial_tau:.4f}")

# Training
print(f"\n--- Training ({args.epochs} epochs) ---")
best_tau = initial_tau
best_state = None
patience = 2
patience_counter = 0

for epoch in range(args.epochs):
    model.train()
    epoch_losses = []
    epoch_ranking_correct = 0
    epoch_ranking_total = 0

    # Shuffle pairs
    indices = np.random.permutation(len(train_dataset))

    for step_idx in range(0, len(indices), args.batch_size):
        global_step = epoch * (len(indices) // args.batch_size) + step_idx // args.batch_size

        # Unfreeze encoder after warmup
        if frozen and global_step >= unfreeze_step:
            frozen = False
            for param in encoder_params:
                param.requires_grad = True
            print(f"  [Step {global_step}] Encoder unfrozen")

        batch_indices = indices[step_idx:step_idx + args.batch_size]
        batch = [train_dataset[i] for i in batch_indices]

        # Get scores for better and worse translations
        src_texts = [b["src"] for b in batch]
        mt_better = [b["mt_better"] for b in batch]
        mt_worse = [b["mt_worse"] for b in batch]
        gold_better = torch.tensor([b["score_better"] for b in batch], device=device)
        gold_worse = torch.tensor([b["score_worse"] for b in batch], device=device)

        # Forward pass - get model scores with gradients
        # CometKiwi's prepare_sample returns (input_dict,) for predict stage
        all_src = src_texts + src_texts
        all_mt = mt_better + mt_worse
        all_samples = [{"src": s, "mt": m} for s, m in zip(all_src, all_mt)]

        # prepare_sample(stage='predict') returns tuple: (input_dict,)
        batch_tuple = model.prepare_sample(all_samples, stage="predict")
        batch_input = batch_tuple[0]
        # Move to device
        batch_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in batch_input.items()}

        # forward() returns Prediction with .score attribute (tensor with grad)
        model_output = model.forward(**batch_input)
        scores = model_output.score.squeeze(-1)

        n = len(batch)
        pred_better = scores[:n]
        pred_worse = scores[n:]

        # Ranking loss: we want pred_better > pred_worse by at least margin
        ranking_loss = torch.clamp(
            args.margin - (pred_better - pred_worse),
            min=0
        ).mean()

        # MSE loss for calibration
        all_gold = torch.cat([gold_better, gold_worse])
        mse_loss = F.mse_loss(scores, all_gold)

        # Combined loss
        loss = args.mse_weight * mse_loss + (1 - args.mse_weight) * ranking_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_losses.append(loss.item())

        # Track ranking accuracy
        with torch.no_grad():
            correct = (pred_better > pred_worse).sum().item()
            epoch_ranking_correct += correct
            epoch_ranking_total += n

        if (step_idx // args.batch_size) % 100 == 0 and step_idx > 0:
            avg_loss = np.mean(epoch_losses[-100:])
            rank_acc = epoch_ranking_correct / max(1, epoch_ranking_total)
            print(f"  Epoch {epoch+1}, Step {step_idx//args.batch_size}: "
                  f"loss={avg_loss:.4f}, rank_acc={rank_acc:.4f}")

    # Epoch evaluation
    avg_loss = np.mean(epoch_losses)
    rank_acc = epoch_ranking_correct / max(1, epoch_ranking_total)
    print(f"\n  Epoch {epoch+1} summary: avg_loss={avg_loss:.4f}, rank_acc={rank_acc:.4f}")

    dev_tau = evaluate_on_dev(model, dev)
    print(f"  Dev per-source Kendall Tau: {dev_tau:.4f} (initial: {initial_tau:.4f})")

    if dev_tau > best_tau:
        best_tau = dev_tau
        patience_counter = 0
        # Save best model
        os.makedirs("models/cometkiwi_pairwise/", exist_ok=True)
        ckpt_path = f"models/cometkiwi_pairwise/best-epoch{epoch+1}-tau{dev_tau:.4f}.ckpt"
        model.trainer = None  # Remove trainer reference for clean save
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved best model to {ckpt_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping (patience={patience})")
            break


# ---------------------------------------------------------------------------
# 5. Final evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

final_tau = evaluate_on_dev(model, dev)
print(f"  Initial per-source Kendall Tau: {initial_tau:.4f}")
print(f"  Best per-source Kendall Tau:    {best_tau:.4f}")
print(f"  Improvement:                    {best_tau - initial_tau:+.4f}")

# Save predictions with best model
samples = [{"src": row["src_text"], "mt": row["tgt_text"]}
           for _, row in dev.iterrows()]
gpus = 1 if device.type == "cuda" else 0
output = model.predict(samples, batch_size=32, gpus=gpus, num_workers=2 if not gpus else 4)
dev["pairwise_finetuned_score"] = output.scores

dev.to_parquet("outputs/dev_with_pairwise_finetuned.parquet", index=False)
print(f"\nSaved predictions to outputs/dev_with_pairwise_finetuned.parquet")

print("\n" + "=" * 80)
print("PAIRWISE FINE-TUNING COMPLETE")
print("=" * 80)
