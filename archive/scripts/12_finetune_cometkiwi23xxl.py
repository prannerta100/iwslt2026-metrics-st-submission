"""
Fine-tune CometKiwi-23-XXL (10.7B params) with pairwise ranking loss.

This is the same approach as 03b_finetune_pairwise.py but adapted for the
XXL model (facebook/xlm-roberta-xxl encoder, 48 layers, 4096 hidden).

Memory management for 102GB VRAM:
  - Model in bf16: ~21.5GB
  - Phase 1: Encoder frozen, train head only (35.7M params). Batch=8 pairs.
  - Phase 2: Unfreeze last 4 encoder layers (~890M params) + gradient
    checkpointing. Batch=4 pairs, grad_accum=4.
  - bf16 autocast throughout.

Key differences from 03b (CometKiwi-22 fine-tuning):
  - 10.7B vs 550M params
  - Only last 4 encoder layers unfrozen (vs all)
  - Gradient checkpointing when unfreezing
  - bf16 mixed precision
  - Smaller batch, gradient accumulation
  - Lower learning rates

Run on GPU (requires ~40GB VRAM minimum, 102GB recommended):
  poetry run python scripts/12_finetune_cometkiwi23xxl.py
  poetry run python scripts/12_finetune_cometkiwi23xxl.py --epochs 5 --batch-size 4
"""

import os
import sys
import time
import argparse
import gc

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
parser.add_argument("--epochs", type=int, default=5,
                    help="Fewer epochs than 03b — XXL converges faster")
parser.add_argument("--batch-size", type=int, default=8,
                    help="Pairs per step (smaller than 03b due to model size)")
parser.add_argument("--grad-accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
parser.add_argument("--lr", type=float, default=5e-6,
                    help="Head learning rate (lower than 03b's 1e-5)")
parser.add_argument("--encoder-lr", type=float, default=1e-7,
                    help="Encoder learning rate (lower than 03b's 5e-7)")
parser.add_argument("--margin", type=float, default=0.01)
parser.add_argument("--mse-weight", type=float, default=0.3)
parser.add_argument("--frozen-epochs", type=float, default=1.0,
                    help="Keep encoder frozen for this many epochs (default: full first epoch)")
parser.add_argument("--unfreeze-layers", type=int, default=4,
                    help="Number of top encoder layers to unfreeze (out of 48)")
parser.add_argument("--max-pairs", type=int, default=50000)
parser.add_argument("--eval-batch-size", type=int, default=32,
                    help="Batch size for dev evaluation (no gradients)")
parser.add_argument("--use-all-data", action="store_true")
args = parser.parse_args()

EFFECTIVE_BATCH = args.batch_size * args.grad_accum

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 80)
print("PAIRWISE RANKING FINE-TUNING — CometKiwi-23-XXL (10.7B params)")
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

if len(all_pairs) > args.max_pairs:
    rng = np.random.RandomState(42)
    indices = rng.choice(len(all_pairs), size=args.max_pairs, replace=False)
    all_pairs = [all_pairs[i] for i in indices]
    print(f"Subsampled to {len(all_pairs)} pairs")


# ---------------------------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------------------------
from comet import download_model, load_from_checkpoint

print("\nLoading CometKiwi-23-XXL (this downloads ~43GB on first run)...")
model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
model = load_from_checkpoint(model_path)

if not torch.cuda.is_available():
    print("ERROR: CometKiwi-23-XXL fine-tuning requires GPU.")
    sys.exit(1)

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {vram_gb:.1f} GB")

# Move model to GPU in bf16 to save memory
model = model.to(dtype=torch.bfloat16, device=device)
print(f"Model loaded in bf16 on GPU")

# Report memory after model load
allocated_gb = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory after model load: {allocated_gb:.1f} GB")


# ---------------------------------------------------------------------------
# 4. Parameter groups
# ---------------------------------------------------------------------------
# Separate into: encoder layers, layerwise attention, estimator head
encoder_layer_params = {}  # layer_idx -> list of params
other_encoder_params = []  # embeddings, etc.
head_params = []  # estimator + layerwise_attention

for name, param in model.named_parameters():
    if "encoder.model.encoder.layer." in name:
        # Extract layer number
        parts = name.split(".")
        layer_idx = int(parts[parts.index("layer") + 1])
        if layer_idx not in encoder_layer_params:
            encoder_layer_params[layer_idx] = []
        encoder_layer_params[layer_idx].append(param)
    elif "encoder" in name:
        other_encoder_params.append(param)
    else:
        head_params.append(param)

n_encoder_layers = max(encoder_layer_params.keys()) + 1
total_encoder = sum(p.numel() for ps in encoder_layer_params.values() for p in ps)
total_encoder += sum(p.numel() for p in other_encoder_params)
total_head = sum(p.numel() for p in head_params)

print(f"\nEncoder: {n_encoder_layers} layers, {total_encoder/1e9:.2f}B params")
print(f"Head: {total_head/1e6:.1f}M params")
print(f"Will unfreeze top {args.unfreeze_layers} encoder layers after {args.frozen_epochs} epochs")

# Freeze everything initially except head
for param in model.parameters():
    param.requires_grad = False
for param in head_params:
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Initially trainable: {trainable/1e6:.1f}M params (head only)")


# ---------------------------------------------------------------------------
# 5. Scoring function (same as 03b but with bf16 autocast)
# ---------------------------------------------------------------------------
def score_batch(model, src_texts, mt_texts):
    """Get differentiable scores from CometKiwi for a batch."""
    samples = [{"src": s, "mt": m} for s, m in zip(src_texts, mt_texts)]
    batch = model.prepare_sample(samples, stage="predict")
    input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch[0].items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.forward(**input_dict)
    return prediction.score


def evaluate_on_dev(model, dev_df):
    """Evaluate per-source Kendall Tau on dev set."""
    model.eval()
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(dev_df), args.eval_batch_size):
            batch_df = dev_df.iloc[i:i + args.eval_batch_size]
            src_texts = batch_df["src_text"].tolist()
            mt_texts = batch_df["tgt_text"].tolist()
            samples = [{"src": s, "mt": m} for s, m in zip(src_texts, mt_texts)]
            batch = model.prepare_sample(samples, stage="predict")
            input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch[0].items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.forward(**input_dict)
            all_scores.extend(prediction.score.float().cpu().tolist())

    dev_df = dev_df.copy()
    dev_df["pred"] = all_scores

    taus = []
    for doc_id, group in dev_df.groupby("doc_id"):
        if len(group) < 2:
            continue
        tau, _ = stats.kendalltau(group["pred"].values, group["score"].values)
        if not np.isnan(tau):
            taus.append(tau)

    per_source_tau = np.mean(taus) if taus else 0.0

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
# 6. Set up optimizer (head only initially)
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW([
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

unfreeze_step = int(args.frozen_epochs * steps_per_epoch)

print(f"\nTraining config:")
print(f"  Epochs: {args.epochs}")
print(f"  Batch size: {args.batch_size} pairs (effective: {EFFECTIVE_BATCH})")
print(f"  Gradient accumulation: {args.grad_accum} steps")
print(f"  Steps/epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Unfreeze top {args.unfreeze_layers} layers at step: {unfreeze_step}")
print(f"  Head LR: {args.lr}")
print(f"  Encoder LR (after unfreeze): {args.encoder_lr}")
print(f"  MSE weight: {args.mse_weight}, Ranking weight: {1 - args.mse_weight}")


# ---------------------------------------------------------------------------
# 7. Initial evaluation
# ---------------------------------------------------------------------------
print("\n--- Initial evaluation ---")
initial_tau = evaluate_on_dev(model, dev)
print(f"  Initial per-source Kendall Tau: {initial_tau:.4f}")


# ---------------------------------------------------------------------------
# 8. Training loop
# ---------------------------------------------------------------------------
print(f"\n--- Training ({args.epochs} epochs, {len(all_pairs)} pairs) ---")

# Pre-flight test
print("  [Pre-flight] Testing forward+backward on 2 samples...")
model.train()
_test_samples = [{"src": "test source", "mt": "test translation"}] * 2
_test_batch = model.prepare_sample(_test_samples, stage="predict")
_test_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v
               for k, v in _test_batch[0].items()}
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _test_pred = model.forward(**_test_input)
    _test_loss = _test_pred.score.mean()
_test_loss.backward()
_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters() if p.requires_grad)
print(f"  [Pre-flight] backward OK, head has gradients: {_has_grad}")
model.zero_grad()
print("  [Pre-flight] PASSED\n")

best_tau = initial_tau
best_ckpt_path = None
patience = 3
patience_counter = 0
global_step = 0
encoder_unfrozen = False

# GradScaler for mixed precision
scaler = torch.amp.GradScaler("cuda")

for epoch in range(args.epochs):
    model.train()
    epoch_losses = []
    epoch_rank_losses = []
    epoch_mse_losses = []
    epoch_ranking_correct = 0
    epoch_ranking_total = 0
    epoch_start = time.time()

    indices = np.random.permutation(len(all_pairs))

    for step_idx in range(0, len(indices), args.batch_size):
        # Unfreeze top encoder layers after warmup
        if not encoder_unfrozen and global_step >= unfreeze_step:
            encoder_unfrozen = True
            layers_to_unfreeze = list(range(
                n_encoder_layers - args.unfreeze_layers, n_encoder_layers
            ))
            for layer_idx in layers_to_unfreeze:
                for param in encoder_layer_params[layer_idx]:
                    param.requires_grad = True

            # Enable gradient checkpointing to manage memory
            model.encoder.model.gradient_checkpointing_enable()

            # Rebuild optimizer with encoder params
            unfrozen_encoder_params = []
            for layer_idx in layers_to_unfreeze:
                unfrozen_encoder_params.extend(encoder_layer_params[layer_idx])

            optimizer = torch.optim.AdamW([
                {"params": head_params, "lr": args.lr * lr_lambda(global_step)},
                {"params": unfrozen_encoder_params, "lr": args.encoder_lr},
            ], weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            # Advance scheduler to current step
            for _ in range(global_step):
                scheduler.step()

            unfrozen_count = sum(p.numel() for p in unfrozen_encoder_params)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  [Step {global_step}] Unfroze top {args.unfreeze_layers} encoder layers "
                  f"({unfrozen_count/1e6:.0f}M params). "
                  f"Total trainable: {trainable/1e6:.0f}M. "
                  f"Gradient checkpointing enabled.")
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  [Step {global_step}] GPU memory: {mem_gb:.1f} GB")

        batch_indices = indices[step_idx:step_idx + args.batch_size]
        if len(batch_indices) < 2:
            continue
        batch = [all_pairs[i] for i in batch_indices]
        n = len(batch)

        src_better = [b["src"] for b in batch]
        mt_better = [b["mt_better"] for b in batch]
        src_worse = [b["src"] for b in batch]
        mt_worse = [b["mt_worse"] for b in batch]

        gold_better = torch.tensor([b["score_better"] for b in batch],
                                   dtype=torch.float32, device=device)
        gold_worse = torch.tensor([b["score_worse"] for b in batch],
                                  dtype=torch.float32, device=device)
        gold_margins = torch.tensor([b["margin"] for b in batch],
                                    dtype=torch.float32, device=device)

        # Forward passes with autocast
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_better = score_batch(model, src_better, mt_better)
            pred_worse = score_batch(model, src_worse, mt_worse)

            # Cast predictions to float32 for loss computation
            pred_better = pred_better.float()
            pred_worse = pred_worse.float()

            # Adaptive margin ranking loss
            adaptive_margin = torch.clamp(gold_margins * 0.5, min=args.margin)
            ranking_loss = torch.clamp(
                adaptive_margin - (pred_better - pred_worse),
                min=0
            ).mean()

            # MSE loss for calibration
            mse_loss = (F.mse_loss(pred_better, gold_better)
                        + F.mse_loss(pred_worse, gold_worse)) / 2.0

            # Combined loss, scaled for gradient accumulation
            loss = (args.mse_weight * mse_loss + (1 - args.mse_weight) * ranking_loss)
            loss = loss / args.grad_accum

        # Backward with scaler
        scaler.scale(loss).backward()

        # Step every grad_accum steps
        if (global_step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        global_step += 1

        epoch_losses.append(loss.item() * args.grad_accum)  # un-scale for logging
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
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  Step {global_step}: loss={avg_loss:.4f} "
                  f"(rank={avg_rank:.4f}, mse={avg_mse:.4f}), "
                  f"rank_acc={rank_acc:.4f}, mem={mem_gb:.1f}GB")

    # Flush any remaining gradients
    if global_step % args.grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

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
        os.makedirs("models/cometkiwi23xxl_pairwise/", exist_ok=True)
        best_ckpt_path = f"models/cometkiwi23xxl_pairwise/best-epoch{epoch+1}-tau{dev_tau:.4f}.ckpt"
        # Save only head + unfrozen encoder state dict to save space
        save_dict = {k: v for k, v in model.state_dict().items()}
        torch.save(save_dict, best_ckpt_path)
        print(f"  NEW BEST! Saved to {best_ckpt_path}")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break


# ---------------------------------------------------------------------------
# 9. Final evaluation with best model
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

if best_ckpt_path and os.path.exists(best_ckpt_path):
    print(f"Loading best checkpoint: {best_ckpt_path}")
    state_dict = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)

# Disable gradient checkpointing for faster inference
if hasattr(model.encoder.model, "gradient_checkpointing_disable"):
    model.encoder.model.gradient_checkpointing_disable()

model.eval()

all_final_scores = []
with torch.no_grad():
    for i in range(0, len(dev), args.eval_batch_size):
        batch_df = dev.iloc[i:i + args.eval_batch_size]
        samples = [{"src": s, "mt": m}
                   for s, m in zip(batch_df["src_text"].tolist(), batch_df["tgt_text"].tolist())]
        batch = model.prepare_sample(samples, stage="predict")
        input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch[0].items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.forward(**input_dict)
        all_final_scores.extend(prediction.score.float().cpu().tolist())

dev["cometkiwi23xxl_finetuned_score"] = all_final_scores

# Compute final metrics
taus = []
for doc_id, group in dev.groupby("doc_id"):
    if len(group) < 2:
        continue
    tau, _ = stats.kendalltau(group["cometkiwi23xxl_finetuned_score"].values,
                              group["score"].values)
    if not np.isnan(tau):
        taus.append(tau)
final_tau = np.mean(taus) if taus else 0.0

overall_tau, _ = stats.kendalltau(dev["cometkiwi23xxl_finetuned_score"].values,
                                  dev["score"].values)
pearson_r, _ = stats.pearsonr(dev["cometkiwi23xxl_finetuned_score"].values,
                               dev["score"].values)

print(f"  Initial per-source Kendall Tau:  {initial_tau:.4f}")
print(f"  Best per-source Kendall Tau:     {final_tau:.4f}")
print(f"  Improvement:                     {final_tau - initial_tau:+.4f}")
print(f"  Overall Kendall Tau:             {overall_tau:.4f}")
print(f"  Pearson:                         {pearson_r:.4f}")

for (src, tgt), group in dev.groupby(["src_lang", "tgt_lang"]):
    lp_taus = []
    for doc_id, doc_group in group.groupby("doc_id"):
        if len(doc_group) < 2:
            continue
        tau, _ = stats.kendalltau(doc_group["cometkiwi23xxl_finetuned_score"].values,
                                  doc_group["score"].values)
        if not np.isnan(tau):
            lp_taus.append(tau)
    lp_tau = np.mean(lp_taus) if lp_taus else 0.0
    print(f"  {src}->{tgt}: per-source tau={lp_tau:.4f}")

# Save predictions
dev.to_parquet("outputs/dev_with_cometkiwi23xxl_finetuned.parquet", index=False)

# Merge into main predictions file
existing_pred_file = "outputs/dev_with_predictions.parquet"
if os.path.exists(existing_pred_file):
    existing = pd.read_parquet(existing_pred_file)
    existing["cometkiwi23xxl_finetuned_score"] = dev["cometkiwi23xxl_finetuned_score"].values
    existing.to_parquet(existing_pred_file, index=False)
    print(f"\nMerged cometkiwi23xxl_finetuned_score into {existing_pred_file}")

print("\n" + "=" * 80)
print("CometKiwi-23-XXL PAIRWISE FINE-TUNING COMPLETE")
print("=" * 80)
