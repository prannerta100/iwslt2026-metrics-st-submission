"""
Fine-tune CometKiwi-22 on the IWSLT 2026 QE training data.

This script:
1. Prepares training CSV from our parquet data
2. Filters to en->de and en->zh (test language pairs)
3. Fine-tunes with the COMET trainer (PyTorch Lightning)
4. Uses val_kendall as the optimization metric

Run on GPU: python scripts/03_finetune_cometkiwi.py
"""

import os
import sys
import argparse

# SSL fix must come first
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import scripts.ssl_fix

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Prepare training data in COMET CSV format
# ---------------------------------------------------------------------------
print("=" * 80)
print("STEP 1: Preparing training data")
print("=" * 80)

# Load train and dev data
train = pd.read_parquet("outputs/train_text.parquet")
dev = pd.read_parquet("outputs/dev_text.parquet")

# Also load synthetic data and rescale scores from 0-1 to 0-100
synth = pd.read_parquet("outputs/train_synthetic_text.parquet")
synth["score"] = synth["score"] * 100.0

print(f"Train: {len(train)} rows")
print(f"Dev: {len(dev)} rows")
print(f"Synthetic: {len(synth)} rows")

# Filter to target language pairs (en->de, en->zh) for focused training
# But also include other pairs for cross-lingual transfer
train_target = train[
    ((train["src_lang"] == "en") & (train["tgt_lang"].isin(["de", "zh"])))
]
train_other = train[
    ~((train["src_lang"] == "en") & (train["tgt_lang"].isin(["de", "zh"])))
]
synth_target = synth[
    ((synth["src_lang"] == "en") & (synth["tgt_lang"].isin(["de"])))
]

print(f"\nTarget language pair train: {len(train_target)} rows")
print(f"Other language pair train: {len(train_other)} rows")
print(f"Target synthetic: {len(synth_target)} rows")

# Strategy: Use all target LP data + subset of other LPs for cross-lingual transfer
# Normalize scores to 0-1 range for COMET (which expects 0-1 scores)
def prepare_comet_csv(df, output_path):
    """Prepare CSV in COMET format: src, mt, score (normalized to 0-1)."""
    comet_df = pd.DataFrame({
        "src": df["src_text"].values,
        "mt": df["tgt_text"].values,
        "score": df["score"].values / 100.0,  # Normalize to 0-1
    })
    # Drop any rows with missing values
    comet_df = comet_df.dropna()
    comet_df.to_csv(output_path, index=False)
    print(f"  Saved {len(comet_df)} rows to {output_path}")
    return comet_df

os.makedirs("data", exist_ok=True)

# Full training set: target LPs + other LPs + synthetic
train_all = pd.concat([train_target, train_other, synth_target], ignore_index=True)
# Shuffle
train_all = train_all.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nPreparing training CSV...")
prepare_comet_csv(train_all, "data/train_all.csv")

# Target-only training set (for focused fine-tuning)
print("Preparing target-only training CSV...")
train_focused = pd.concat([train_target, synth_target], ignore_index=True)
train_focused = train_focused.sample(frac=1, random_state=42).reset_index(drop=True)
prepare_comet_csv(train_focused, "data/train_focused.csv")

# Validation set (dev)
print("Preparing validation CSV...")
dev_csv = prepare_comet_csv(dev, "data/dev.csv")

# Add system column for system-level accuracy in validation
dev_with_sys = pd.DataFrame({
    "src": dev["src_text"].values,
    "mt": dev["tgt_text"].values,
    "score": dev["score"].values / 100.0,
    "system": dev["tgt_system"].values,
})
dev_with_sys.to_csv("data/dev_with_system.csv", index=False)
print(f"  Saved {len(dev_with_sys)} rows to data/dev_with_system.csv")


# ---------------------------------------------------------------------------
# 2. Fine-tune CometKiwi
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 2: Fine-tuning CometKiwi-22")
print("=" * 80)

import torch
from comet import download_model, load_from_checkpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Detect hardware
if torch.cuda.is_available():
    accelerator = "gpu"
    devices = 1
    precision = "16-mixed"
    batch_size = 64  # 96GB VRAM: CometKiwi ~1.2GB model, plenty of room
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    accelerator = "cpu"
    devices = 1
    precision = 32
    batch_size = 8
    print("Using CPU")

# Load model
local_ckpt = "/tmp/cometkiwi22/checkpoints/model.ckpt"
if os.path.exists(local_ckpt):
    print(f"Loading from local checkpoint: {local_ckpt}")
    model_path = local_ckpt
else:
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
model = load_from_checkpoint(model_path)

# Override training parameters
model.hparams.train_data = ["data/train_focused.csv"]
model.hparams.validation_data = ["data/dev_with_system.csv"]
model.hparams.batch_size = batch_size
model.hparams.learning_rate = 1.5e-05
model.hparams.encoder_learning_rate = 1e-06
model.hparams.nr_frozen_epochs = 0.3
model.hparams.keep_embeddings_frozen = True

print(f"\nTraining config:")
print(f"  Train data: {model.hparams.train_data}")
print(f"  Val data: {model.hparams.validation_data}")
print(f"  Batch size: {model.hparams.batch_size}")
print(f"  Learning rate: {model.hparams.learning_rate}")
print(f"  Encoder LR: {model.hparams.encoder_learning_rate}")
print(f"  Accelerator: {accelerator}")
print(f"  Precision: {precision}")

# Set up callbacks
callbacks = [
    EarlyStopping(
        monitor="val_kendall",
        patience=3,
        mode="max",
        verbose=True,
    ),
    ModelCheckpoint(
        monitor="val_kendall",
        save_top_k=1,
        mode="max",
        dirpath="models/cometkiwi_finetuned/",
        filename="best-{epoch}-{val_kendall:.4f}",
        verbose=True,
    ),
]

# Create trainer
trainer = Trainer(
    max_epochs=5,
    accelerator=accelerator,
    devices=devices,
    precision=precision,
    callbacks=callbacks,
    log_every_n_steps=50,
    reload_dataloaders_every_n_epochs=1,
)

# Train!
print("\nStarting fine-tuning...")
trainer.fit(model)

print(f"\nBest model: {callbacks[1].best_model_path}")
print(f"Best val_kendall: {callbacks[1].best_model_score:.4f}")

# ---------------------------------------------------------------------------
# 3. Evaluate fine-tuned model on dev
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("STEP 3: Evaluating fine-tuned model")
print("=" * 80)

# Load best checkpoint
best_model = load_from_checkpoint(callbacks[1].best_model_path)

# Prepare samples
samples = [
    {"src": row["src_text"], "mt": row["tgt_text"]}
    for _, row in dev.iterrows()
]

# Run inference
if torch.cuda.is_available():
    output = best_model.predict(samples, batch_size=batch_size, gpus=1)
else:
    output = best_model.predict(samples, batch_size=batch_size, gpus=0, num_workers=2)

dev["finetuned_score"] = output["scores"]

# Compute metrics
from scipy import stats

# Overall Kendall Tau
overall_tau, _ = stats.kendalltau(dev["finetuned_score"].values, dev["score"].values / 100.0)
pearson_r, _ = stats.pearsonr(dev["finetuned_score"].values, dev["score"].values / 100.0)

# Per-source Kendall Tau
taus = []
for doc_id, group in dev.groupby("doc_id"):
    if len(group) < 2:
        continue
    tau, _ = stats.kendalltau(group["finetuned_score"].values, group["score"].values)
    if not np.isnan(tau):
        taus.append(tau)
per_source_tau = np.mean(taus) if taus else 0.0

print(f"\nFine-tuned CometKiwi Results:")
print(f"  Overall Kendall Tau:    {overall_tau:.4f}")
print(f"  Per-source Kendall Tau: {per_source_tau:.4f}")
print(f"  Pearson:                {pearson_r:.4f}")

# Per language pair
for (src, tgt), group in dev.groupby(["src_lang", "tgt_lang"]):
    lp_taus = []
    for doc_id, doc_group in group.groupby("doc_id"):
        if len(doc_group) < 2:
            continue
        tau, _ = stats.kendalltau(doc_group["finetuned_score"].values, doc_group["score"].values)
        if not np.isnan(tau):
            lp_taus.append(tau)
    lp_tau = np.mean(lp_taus) if lp_taus else 0.0
    print(f"  {src}->{tgt}: per-source tau={lp_tau:.4f}")

# Save predictions
dev.to_parquet("outputs/dev_with_finetuned.parquet", index=False)
print(f"\nSaved predictions to outputs/dev_with_finetuned.parquet")

print("\n" + "=" * 80)
print("FINE-TUNING COMPLETE")
print("=" * 80)
