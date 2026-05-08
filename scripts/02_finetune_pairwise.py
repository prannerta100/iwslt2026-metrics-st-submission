"""
Fine-tune CometKiwi-23-XXL (10.7B) with pairwise ranking loss.

Directly optimizes Kendall tau by training on (better, worse) pairs from within
the same document. Uses bf16, gradient checkpointing, and differential LR.

Memory budget (96-102GB VRAM):
  - Model bf16: ~21.5GB
  - Phase 1: Encoder frozen, head only (35.7M trainable). Batch=8 pairs.
  - Phase 2: Unfreeze top 4 encoder layers (~890M). Batch=4, grad_accum=8.
  - Gradient checkpointing enabled after unfreeze.

Run:
  poetry run python scripts/02_finetune_pairwise.py
  poetry run python scripts/02_finetune_pairwise.py --epochs 5 --batch-size 4 --grad-accum 8
"""

import os
import sys
import time
import json
import argparse
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssl_fix

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--grad-accum", type=int, default=4)
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--encoder-lr", type=float, default=1e-7)
parser.add_argument("--margin", type=float, default=0.01)
parser.add_argument("--mse-weight", type=float, default=0.3)
parser.add_argument("--frozen-epochs", type=float, default=1.0)
parser.add_argument("--unfreeze-layers", type=int, default=4)
parser.add_argument("--max-pairs", type=int, default=50000)
parser.add_argument("--eval-batch-size", type=int, default=32)
parser.add_argument("--min-score-diff", type=float, default=1.0)
parser.add_argument("--patience", type=int, default=2)
args = parser.parse_args()

EFFECTIVE_BATCH = args.batch_size * args.grad_accum


# ---------------------------------------------------------------------------
# 1. Load data from JSONL
# ---------------------------------------------------------------------------
print("=" * 80)
print("PAIRWISE RANKING FINE-TUNING — CometKiwi-23-XXL (10.7B)")
print("=" * 80)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


train_data = load_jsonl("data/train.jsonl")
dev_data = load_jsonl("data/dev.jsonl")

# Filter to target LPs only (en-de, en-zh)
train_data = [r for r in train_data
              if r["src_lang"] == "en" and r["tgt_lang"] in ("de", "zh")]
dev_data = [r for r in dev_data
            if r["src_lang"] == "en" and r["tgt_lang"] in ("de", "zh")]

print(f"Train: {len(train_data)} samples (en-de/zh only)")
print(f"Dev: {len(dev_data)} samples")

doc_ids = set(r["doc_id"] for r in train_data)
print(f"Training docs: {len(doc_ids)}")


# ---------------------------------------------------------------------------
# 2. Create pairwise training data
# ---------------------------------------------------------------------------
def create_pairs(data, min_score_diff):
    from collections import defaultdict
    groups = defaultdict(list)
    for row in data:
        groups[row["doc_id"]].append(row)

    pairs = []
    for doc_id, rows in groups.items():
        if len(rows) < 2:
            continue
        rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)
        for i in range(len(rows_sorted)):
            for j in range(i + 1, len(rows_sorted)):
                diff = rows_sorted[i]["score"] - rows_sorted[j]["score"]
                if diff > min_score_diff:
                    pairs.append({
                        "src": str(rows_sorted[i]["src_text"]),
                        "mt_better": str(rows_sorted[i]["tgt_text"]),
                        "mt_worse": str(rows_sorted[j]["tgt_text"]),
                        "score_better": rows_sorted[i]["score"] / 100.0,
                        "score_worse": rows_sorted[j]["score"] / 100.0,
                        "margin": diff / 100.0,
                    })
    return pairs


print("\nCreating training pairs...")
all_pairs = create_pairs(train_data, args.min_score_diff)
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

print("\nLoading CometKiwi-23-XXL...")
model_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")
model = load_from_checkpoint(model_path)

if not torch.cuda.is_available():
    print("ERROR: GPU required for CK-23-XXL fine-tuning.")
    sys.exit(1)

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"VRAM: {vram_gb:.1f} GB")

model = model.to(dtype=torch.bfloat16, device=device)
allocated_gb = torch.cuda.memory_allocated() / 1e9
print(f"Model loaded in bf16: {allocated_gb:.1f} GB")


# ---------------------------------------------------------------------------
# 4. Parameter groups
# ---------------------------------------------------------------------------
encoder_layer_params = {}
other_encoder_params = []
head_params = []

for name, param in model.named_parameters():
    if "encoder.model.encoder.layer." in name:
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
total_head = sum(p.numel() for p in head_params)
print(f"Encoder: {n_encoder_layers} layers")
print(f"Head: {total_head/1e6:.1f}M params")
print(f"Will unfreeze top {args.unfreeze_layers} layers after {args.frozen_epochs} epochs")

for param in model.parameters():
    param.requires_grad = False
for param in head_params:
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Initially trainable: {trainable/1e6:.1f}M params (head only)")


# ---------------------------------------------------------------------------
# 5. Scoring and evaluation
# ---------------------------------------------------------------------------
def score_batch(model, src_texts, mt_texts):
    samples = [{"src": s, "mt": m} for s, m in zip(src_texts, mt_texts)]
    batch = model.prepare_sample(samples, stage="predict")
    input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch[0].items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        prediction = model.forward(**input_dict)
    return prediction.score


def evaluate_on_dev(model, dev_rows):
    model.eval()
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(dev_rows), args.eval_batch_size):
            batch_rows = dev_rows[i:i + args.eval_batch_size]
            src_texts = [r["src_text"] for r in batch_rows]
            mt_texts = [r["tgt_text"] for r in batch_rows]
            samples = [{"src": s, "mt": m} for s, m in zip(src_texts, mt_texts)]
            batch = model.prepare_sample(samples, stage="predict")
            input_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch[0].items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prediction = model.forward(**input_dict)
            all_scores.extend(prediction.score.float().cpu().tolist())

    from collections import defaultdict
    doc_scores = defaultdict(lambda: {"pred": [], "gold": []})
    for row, pred in zip(dev_rows, all_scores):
        doc_scores[row["doc_id"]]["pred"].append(pred)
        doc_scores[row["doc_id"]]["gold"].append(row["score"])

    taus = []
    for doc_id, vals in doc_scores.items():
        if len(vals["pred"]) < 2:
            continue
        tau, _ = stats.kendalltau(vals["pred"], vals["gold"])
        if not np.isnan(tau):
            taus.append(tau)

    per_source_tau = np.mean(taus) if taus else 0.0

    # Per-LP breakdown
    lp_groups = defaultdict(list)
    for row, pred in zip(dev_rows, all_scores):
        lp_groups[f"{row['src_lang']}-{row['tgt_lang']}"].append((row, pred))

    for lp, items in sorted(lp_groups.items()):
        lp_doc_scores = defaultdict(lambda: {"pred": [], "gold": []})
        for row, pred in items:
            lp_doc_scores[row["doc_id"]]["pred"].append(pred)
            lp_doc_scores[row["doc_id"]]["gold"].append(row["score"])
        lp_taus = []
        for doc_id, vals in lp_doc_scores.items():
            if len(vals["pred"]) < 2:
                continue
            tau, _ = stats.kendalltau(vals["pred"], vals["gold"])
            if not np.isnan(tau):
                lp_taus.append(tau)
        lp_tau = np.mean(lp_taus) if lp_taus else 0.0
        print(f"    {lp}: tau={lp_tau:.4f}")

    return per_source_tau


# ---------------------------------------------------------------------------
# 6. Optimizer
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
print(f"  Epochs: {args.epochs}, Batch: {args.batch_size} (effective: {EFFECTIVE_BATCH})")
print(f"  Steps/epoch: {steps_per_epoch}, Total: {total_steps}")
print(f"  Unfreeze at step: {unfreeze_step}")
print(f"  Head LR: {args.lr}, Encoder LR: {args.encoder_lr}")


# ---------------------------------------------------------------------------
# 7. Initial evaluation
# ---------------------------------------------------------------------------
print("\n--- Initial evaluation ---")
initial_tau = evaluate_on_dev(model, dev_data)
print(f"  Initial per-source tau: {initial_tau:.4f}")


# ---------------------------------------------------------------------------
# 8. Pre-flight test
# ---------------------------------------------------------------------------
print("\n  [Pre-flight] Testing forward+backward...")
model.train()
_samples = [{"src": "test", "mt": "test"}] * 2
_batch = model.prepare_sample(_samples, stage="predict")
_input = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in _batch[0].items()}
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    _pred = model.forward(**_input)
    _pred.score.mean().backward()
_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters() if p.requires_grad)
print(f"  [Pre-flight] OK, gradients flowing: {_has_grad}")
model.zero_grad()


# ---------------------------------------------------------------------------
# 9. Training loop
# ---------------------------------------------------------------------------
print(f"\n--- Training ({args.epochs} epochs) ---")

best_tau = initial_tau
best_ckpt_path = None
patience_counter = 0
global_step = 0
encoder_unfrozen = False
scaler = torch.amp.GradScaler("cuda")

os.makedirs("models", exist_ok=True)

for epoch in range(args.epochs):
    model.train()
    epoch_losses = []
    epoch_ranking_correct = 0
    epoch_ranking_total = 0
    epoch_start = time.time()

    indices = np.random.permutation(len(all_pairs))

    for step_idx in range(0, len(indices), args.batch_size):
        # Unfreeze top encoder layers
        if not encoder_unfrozen and global_step >= unfreeze_step:
            encoder_unfrozen = True
            layers_to_unfreeze = list(range(
                n_encoder_layers - args.unfreeze_layers, n_encoder_layers
            ))
            for layer_idx in layers_to_unfreeze:
                for param in encoder_layer_params[layer_idx]:
                    param.requires_grad = True

            model.encoder.model.gradient_checkpointing_enable()

            unfrozen_encoder_params = []
            for layer_idx in layers_to_unfreeze:
                unfrozen_encoder_params.extend(encoder_layer_params[layer_idx])

            optimizer = torch.optim.AdamW([
                {"params": head_params, "lr": args.lr * lr_lambda(global_step)},
                {"params": unfrozen_encoder_params, "lr": args.encoder_lr},
            ], weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            for _ in range(global_step):
                scheduler.step()

            unfrozen_count = sum(p.numel() for p in unfrozen_encoder_params)
            print(f"  [Step {global_step}] Unfroze top {args.unfreeze_layers} layers "
                  f"({unfrozen_count/1e6:.0f}M params). Grad checkpointing ON.")

        batch_indices = indices[step_idx:step_idx + args.batch_size]
        if len(batch_indices) < 2:
            continue
        batch = [all_pairs[i] for i in batch_indices]
        n = len(batch)

        gold_better = torch.tensor([b["score_better"] for b in batch],
                                   dtype=torch.float32, device=device)
        gold_worse = torch.tensor([b["score_worse"] for b in batch],
                                  dtype=torch.float32, device=device)
        gold_margins = torch.tensor([b["margin"] for b in batch],
                                    dtype=torch.float32, device=device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_better = score_batch(model,
                                      [b["src"] for b in batch],
                                      [b["mt_better"] for b in batch])
            pred_worse = score_batch(model,
                                     [b["src"] for b in batch],
                                     [b["mt_worse"] for b in batch])

            pred_better_f = pred_better.float()
            pred_worse_f = pred_worse.float()

            adaptive_margin = torch.clamp(gold_margins * 0.5, min=args.margin)
            ranking_loss = torch.clamp(
                adaptive_margin - (pred_better_f - pred_worse_f), min=0
            ).mean()

            mse_loss = (F.mse_loss(pred_better_f, gold_better)
                        + F.mse_loss(pred_worse_f, gold_worse)) / 2.0

            loss = (args.mse_weight * mse_loss + (1 - args.mse_weight) * ranking_loss)
            loss = loss / args.grad_accum

        scaler.scale(loss).backward()

        if (global_step + 1) % args.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        global_step += 1
        epoch_losses.append(loss.item() * args.grad_accum)

        with torch.no_grad():
            epoch_ranking_correct += (pred_better_f > pred_worse_f).sum().item()
            epoch_ranking_total += n

        if global_step % 200 == 0:
            avg_loss = np.mean(epoch_losses[-200:])
            rank_acc = epoch_ranking_correct / max(1, epoch_ranking_total)
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  Step {global_step}: loss={avg_loss:.4f}, "
                  f"rank_acc={rank_acc:.4f}, mem={mem_gb:.1f}GB")

    # Flush remaining gradients
    if global_step % args.grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    elapsed = time.time() - epoch_start
    avg_loss = np.mean(epoch_losses) if epoch_losses else 0
    rank_acc = epoch_ranking_correct / max(1, epoch_ranking_total)
    print(f"\n  Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
          f"rank_acc={rank_acc:.4f}, time={elapsed:.0f}s")

    # Evaluate
    dev_tau = evaluate_on_dev(model, dev_data)
    print(f"  Dev tau: {dev_tau:.4f} (best: {best_tau:.4f}, init: {initial_tau:.4f})")

    if dev_tau > best_tau:
        best_tau = dev_tau
        patience_counter = 0
        best_ckpt_path = f"models/best_pairwise_epoch{epoch+1}_tau{dev_tau:.4f}.ckpt"
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"  NEW BEST! Saved to {best_ckpt_path}")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{args.patience})")
        if patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break


# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE")
print(f"  Initial tau: {initial_tau:.4f}")
print(f"  Best tau:    {best_tau:.4f}")
print(f"  Improvement: {best_tau - initial_tau:+.4f}")
if best_ckpt_path:
    print(f"  Checkpoint:  {best_ckpt_path}")
print("=" * 80)
