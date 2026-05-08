"""
End-to-end Speech QE: Whisper encoder features + XLM-R text features → QE score.

Architecture:
  1. Whisper large-v3 encoder → speech embeddings (per-frame)
  2. XLM-R encoder (from CometKiwi) → text embeddings
  3. Cross-attention fusion layer
  4. Regression head → quality score

This model can capture speech-level quality cues that text-only metrics miss:
  - Disfluencies, hesitations, background noise
  - Prosodic alignment with translation
  - Speaker characteristics that affect translation difficulty

Training strategy:
  - Initialize text encoder from CometKiwi-22 checkpoint
  - Initialize speech encoder from Whisper large-v3
  - Train fusion layer + regression head from scratch
  - Use pairwise ranking loss (margin-based) to optimize Kendall Tau directly

Run on GPU: python scripts/07_speech_qe.py [--train | --predict]
"""

import os
import sys
import time
import argparse
import math

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

# Use soundfile for audio I/O (torchaudio may have ABI issues on some systems)
try:
    import torchaudio
    HAS_TORCHAUDIO = True
except (ImportError, OSError):
    HAS_TORCHAUDIO = False
import soundfile as sf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["train", "predict", "extract_features"],
                    default="extract_features",
                    help="Mode: train full model, predict, or just extract Whisper features")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--whisper-model", type=str, default="openai/whisper-large-v3")
parser.add_argument("--max-audio-len", type=float, default=30.0,
                    help="Maximum audio length in seconds (truncate longer)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Whisper Feature Extraction
# ---------------------------------------------------------------------------
class WhisperFeatureExtractor:
    """Extract encoder hidden states from Whisper as speech features."""

    def __init__(self, model_name="openai/whisper-large-v3", device="cuda"):
        from transformers import WhisperProcessor, WhisperModel

        self.device = torch.device(device)
        print(f"Loading Whisper model: {model_name}")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # We only need the encoder
        self.encoder = self.model.encoder
        # Freeze encoder weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        print(f"  Whisper encoder loaded ({sum(p.numel() for p in self.encoder.parameters())/1e6:.0f}M params)")

    @torch.no_grad()
    def extract(self, audio_paths, batch_size=32, max_len_sec=30.0):
        """
        Extract Whisper encoder features from audio files.
        Returns: dict with 'embeddings' (N, D) and 'frame_features' (N, T, D)
        """
        all_embeddings = []
        all_frame_features = []
        sample_rate = 16000
        max_samples = int(max_len_sec * sample_rate)

        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i+batch_size]
            waveforms = []

            for path in batch_paths:
                audio_array, sr = sf.read(path, dtype="float32")
                # Convert to mono if stereo
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                # Resample to 16kHz if needed
                if sr != sample_rate and HAS_TORCHAUDIO:
                    waveform = torch.from_numpy(audio_array).unsqueeze(0)
                    waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
                    audio_array = waveform.squeeze(0).numpy()
                # Truncate if too long
                if len(audio_array) > max_samples:
                    audio_array = audio_array[:max_samples]
                waveforms.append(audio_array)

            # Process with Whisper processor
            inputs = self.processor(
                waveforms,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            input_features = inputs.input_features.to(self.device)

            # Forward through encoder
            encoder_output = self.encoder(input_features)
            hidden_states = encoder_output.last_hidden_state  # (B, T, D)

            # Pool: mean over time dimension for sentence-level embedding
            embeddings = hidden_states.mean(dim=1)  # (B, D)
            all_embeddings.append(embeddings.cpu())
            all_frame_features.append(hidden_states.cpu())

            if (i // batch_size + 1) % 20 == 0:
                print(f"    Processed {i + len(batch_paths)}/{len(audio_paths)} audio files")

        return {
            "embeddings": torch.cat(all_embeddings, dim=0),  # (N, D)
            "frame_features": all_frame_features,  # List of (B, T, D)
        }


# ---------------------------------------------------------------------------
# 2. Speech QE Dataset
# ---------------------------------------------------------------------------
class SpeechQEDataset(Dataset):
    """Dataset for speech QE: (speech_embedding, text, score)."""

    def __init__(self, df, speech_embeddings, tokenizer, max_text_len=512):
        self.df = df.reset_index(drop=True)
        self.speech_embeddings = speech_embeddings  # (N, D)
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Speech features (pre-extracted)
        speech_emb = self.speech_embeddings[idx]

        # Text input: concatenate src_text and tgt_text
        src_text = str(row["src_text"])
        tgt_text = str(row["tgt_text"])
        text_input = f"{src_text} </s> {tgt_text}"

        encoding = self.tokenizer(
            text_input,
            max_length=self.max_text_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        score = torch.tensor(row["score"] / 100.0, dtype=torch.float)

        return {
            "speech_emb": speech_emb,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score": score,
            "doc_id": row["doc_id"],
        }


# ---------------------------------------------------------------------------
# 3. Speech QE Model
# ---------------------------------------------------------------------------
class SpeechQEModel(nn.Module):
    """
    Cross-modal QE model: fuses Whisper speech features with text features.
    """

    def __init__(self, speech_dim=1280, text_dim=1024, hidden_dim=512, dropout=0.1):
        super().__init__()

        # Project speech and text to shared dimension
        self.speech_proj = nn.Sequential(
            nn.Linear(speech_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-attention: speech attends to text
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # concat + element-wise product
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Regression head
        self.regressor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, speech_emb, text_emb):
        """
        speech_emb: (B, speech_dim)
        text_emb: (B, text_dim)
        """
        # Project
        speech_h = self.speech_proj(speech_emb)  # (B, hidden_dim)
        text_h = self.text_proj(text_emb)  # (B, hidden_dim)

        # Cross-attention (treat as single-token sequences)
        speech_h_unsq = speech_h.unsqueeze(1)  # (B, 1, hidden_dim)
        text_h_unsq = text_h.unsqueeze(1)  # (B, 1, hidden_dim)
        attn_out, _ = self.cross_attn(speech_h_unsq, text_h_unsq, text_h_unsq)
        speech_h_attn = self.cross_attn_norm(speech_h + attn_out.squeeze(1))

        # Fusion: [speech_attn, text, speech_attn * text]
        combined = torch.cat([
            speech_h_attn,
            text_h,
            speech_h_attn * text_h,  # element-wise interaction
        ], dim=-1)

        fused = self.fusion(combined)
        score = self.regressor(fused).squeeze(-1)
        return torch.sigmoid(score)  # Output in [0, 1]


# ---------------------------------------------------------------------------
# 4. Pairwise Ranking Loss (optimizes Kendall Tau)
# ---------------------------------------------------------------------------
class PairwiseRankingLoss(nn.Module):
    """
    Margin-based pairwise ranking loss.
    For each pair (i, j) where gold_i > gold_j, we want pred_i > pred_j.
    This directly optimizes for ranking correlation (Kendall Tau).
    """

    def __init__(self, margin=0.01, mse_weight=0.5):
        super().__init__()
        self.margin = margin
        self.mse_weight = mse_weight
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets, doc_ids=None):
        """
        predictions: (B,) predicted scores
        targets: (B,) gold scores
        doc_ids: optional, to only compare within same document
        """
        # MSE component (for calibration)
        mse_loss = self.mse(predictions, targets)

        # Pairwise ranking component
        n = len(predictions)
        if n < 2:
            return mse_loss

        # Create all pairs
        pred_diff = predictions.unsqueeze(0) - predictions.unsqueeze(1)  # (N, N)
        gold_diff = targets.unsqueeze(0) - targets.unsqueeze(1)  # (N, N)

        # Only consider pairs where gold scores differ
        valid_mask = gold_diff.abs() > 1e-6

        # If doc_ids provided, only compare within same document
        if doc_ids is not None:
            same_doc = torch.tensor(
                [[doc_ids[i] == doc_ids[j] for j in range(n)] for i in range(n)],
                device=predictions.device,
            )
            valid_mask = valid_mask & same_doc

        if valid_mask.sum() == 0:
            return mse_loss

        # Hinge loss: max(0, margin - sign(gold_diff) * pred_diff)
        signs = torch.sign(gold_diff)
        ranking_loss = torch.clamp(self.margin - signs * pred_diff, min=0)
        ranking_loss = (ranking_loss * valid_mask.float()).sum() / valid_mask.float().sum()

        return self.mse_weight * mse_loss + (1 - self.mse_weight) * ranking_loss


# ---------------------------------------------------------------------------
# 5. Feature extraction mode (most useful for ensemble)
# ---------------------------------------------------------------------------
def extract_whisper_features():
    """
    Extract Whisper encoder features and save them for ensemble use.
    This is the most practical mode: we extract speech embeddings and
    use them as additional features in the ensemble.
    """
    print("\n--- Extracting Whisper Features ---")

    # Check for audio paths in the dataset
    # Try loading the full dataset with audio
    from datasets import load_dataset

    print("Loading dataset with audio...")
    ds = load_dataset("maikezu/iwslt2026-metrics-shared-train-dev")

    # Save audio to temp files for Whisper processing
    import tempfile
    import soundfile as sf

    dev_text = pd.read_parquet("outputs/dev_text.parquet")

    # Match audio to our dev samples
    dev_split = ds["dev"] if "dev" in ds else ds["validation"]

    extractor = WhisperFeatureExtractor(
        model_name=args.whisper_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Save audio files temporarily and extract features
    audio_dir = "/tmp/iwslt_audio"
    os.makedirs(audio_dir, exist_ok=True)

    audio_paths = []
    print("Saving audio files...")
    for i, example in enumerate(dev_split):
        audio_path = os.path.join(audio_dir, f"dev_{i:05d}.wav")
        if not os.path.exists(audio_path):
            audio = example["audio"]
            sf.write(audio_path, audio["array"], audio["sampling_rate"])
        audio_paths.append(audio_path)
        if (i + 1) % 500 == 0:
            print(f"  Saved {i+1}/{len(dev_split)} audio files")

    print(f"\nExtracting Whisper features from {len(audio_paths)} audio files...")
    features = extractor.extract(
        audio_paths,
        batch_size=args.batch_size,
        max_len_sec=args.max_audio_len,
    )

    # Save embeddings
    embeddings = features["embeddings"].numpy()
    np.save("outputs/dev_whisper_embeddings.npy", embeddings)
    print(f"Saved Whisper embeddings: {embeddings.shape}")

    # Compute simple speech features for ensemble
    # These features capture speech characteristics
    if len(embeddings) != len(dev_text):
        print(f"WARNING: Row count mismatch: embeddings={len(embeddings)}, dev_text={len(dev_text)}")
        print("Using min length to align data")
        n = min(len(embeddings), len(dev_text))
        embeddings = embeddings[:n]
        audio_paths = audio_paths[:n]
        dev_text = dev_text.iloc[:n].reset_index(drop=True)

    speech_features = pd.DataFrame(index=dev_text.index)
    speech_features["whisper_emb_norm"] = np.linalg.norm(embeddings, axis=1)
    speech_features["whisper_emb_mean"] = embeddings.mean(axis=1)
    speech_features["whisper_emb_std"] = embeddings.std(axis=1)

    # Audio duration features
    durations = []
    for path in audio_paths:
        info = sf.info(path)
        durations.append(info.frames / info.samplerate)
    speech_features["audio_duration"] = durations

    # Words per second (speech rate proxy)
    speech_features["words_per_sec"] = (
        dev_text["src_text"].str.split().str.len() / speech_features["audio_duration"].clip(lower=0.1)
    )

    speech_features.to_parquet("outputs/dev_speech_features.parquet", index=False)
    print(f"Saved speech features: {speech_features.columns.tolist()}")

    # Quick evaluation: do speech features correlate with quality?
    for col in speech_features.columns:
        tau, _ = stats.kendalltau(speech_features[col].values, dev_text["score"].values)
        print(f"  {col} Kendall Tau: {tau:.4f}")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print(f"SPEECH QE MODEL ({args.mode})")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.mode == "extract_features":
        extract_whisper_features()

    elif args.mode == "train":
        print("\n--- Training Speech QE Model ---")

        # Load pre-extracted features
        if not os.path.exists("outputs/dev_whisper_embeddings.npy"):
            print("ERROR: Run with --mode extract_features first")
            sys.exit(1)

        dev = pd.read_parquet("outputs/dev_text.parquet")
        speech_embs = torch.from_numpy(np.load("outputs/dev_whisper_embeddings.npy"))

        # For now, use XLM-R to get text embeddings
        from transformers import AutoModel, AutoTokenizer

        print("Loading XLM-R for text embeddings...")
        xlmr_name = "xlm-roberta-large"
        tokenizer = AutoTokenizer.from_pretrained(xlmr_name)
        text_model = AutoModel.from_pretrained(xlmr_name)
        text_model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_model = text_model.to(device)

        # Extract text embeddings
        print("Extracting text embeddings...")
        text_embs = []
        for i in range(0, len(dev), 128):
            batch = dev.iloc[i:i+128]
            texts = [f"{r['src_text']} </s> {r['tgt_text']}" for _, r in batch.iterrows()]
            enc = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = text_model(**enc)
                # CLS token embedding
                text_embs.append(out.last_hidden_state[:, 0, :].cpu())
        text_embs = torch.cat(text_embs, dim=0)

        # Create model
        speech_dim = speech_embs.shape[1]  # 1280 for Whisper large
        text_dim = text_embs.shape[1]  # 1024 for XLM-R large
        model = SpeechQEModel(speech_dim=speech_dim, text_dim=text_dim).to(device)
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

        # Training with cross-validation
        from sklearn.model_selection import KFold

        unique_docs = dev["doc_id"].unique()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_preds = np.zeros(len(dev))

        for fold, (train_idx, val_idx) in enumerate(kf.split(unique_docs)):
            print(f"\n--- Fold {fold+1}/5 ---")
            train_docs = set(unique_docs[train_idx])
            val_docs = set(unique_docs[val_idx])

            train_mask = dev["doc_id"].isin(train_docs).values
            val_mask = dev["doc_id"].isin(val_docs).values

            train_speech = speech_embs[train_mask].to(device)
            train_text = text_embs[train_mask].to(device)
            train_scores = torch.tensor(dev.loc[train_mask, "score"].values / 100.0, dtype=torch.float).to(device)
            train_doc_ids = dev.loc[train_mask, "doc_id"].values

            val_speech = speech_embs[val_mask].to(device)
            val_text = text_embs[val_mask].to(device)
            val_scores = dev.loc[val_mask, "score"].values

            # Fresh model for each fold
            fold_model = SpeechQEModel(speech_dim=speech_dim, text_dim=text_dim).to(device)
            optimizer = torch.optim.AdamW(fold_model.parameters(), lr=args.lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            loss_fn = PairwiseRankingLoss(margin=0.01, mse_weight=0.3)

            best_val_tau = -1
            patience = 3
            patience_counter = 0

            for epoch in range(args.epochs):
                fold_model.train()
                # Mini-batch training
                perm = torch.randperm(train_speech.shape[0])
                epoch_loss = 0
                n_batches = 0

                for j in range(0, len(perm), args.batch_size):
                    idx = perm[j:j+args.batch_size]
                    pred = fold_model(train_speech[idx], train_text[idx])
                    batch_doc_ids = train_doc_ids[idx.cpu().numpy()]
                    loss = loss_fn(pred, train_scores[idx], batch_doc_ids)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()

                # Validation
                fold_model.eval()
                with torch.no_grad():
                    val_pred = fold_model(val_speech, val_text).cpu().numpy()

                val_df = dev[val_mask].copy()
                val_df["pred"] = val_pred
                val_taus = []
                for doc_id, group in val_df.groupby("doc_id"):
                    if len(group) < 2:
                        continue
                    tau, _ = stats.kendalltau(group["pred"].values, group["score"].values)
                    if not np.isnan(tau):
                        val_taus.append(tau)
                val_tau = np.mean(val_taus) if val_taus else 0.0

                print(f"  Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}, val_tau={val_tau:.4f}")

                if val_tau > best_val_tau:
                    best_val_tau = val_tau
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"  Early stopping at epoch {epoch+1}")
                        break

            # Load best model and predict on val
            fold_model.load_state_dict(best_state)
            fold_model.eval()
            with torch.no_grad():
                val_pred = fold_model(val_speech, val_text).cpu().numpy()
            all_preds[val_mask] = val_pred
            print(f"  Fold {fold+1} best val_tau: {best_val_tau:.4f}")

        # Final evaluation
        dev["speechqe_score"] = all_preds
        taus = []
        for doc_id, group in dev.groupby("doc_id"):
            if len(group) < 2:
                continue
            tau, _ = stats.kendalltau(group["speechqe_score"].values, group["score"].values)
            if not np.isnan(tau):
                taus.append(tau)
        per_source_tau = np.mean(taus) if taus else 0.0
        print(f"\nSpeech QE cross-validated per-source Kendall Tau: {per_source_tau:.4f}")

        # Save
        dev.to_parquet("outputs/dev_with_speechqe.parquet", index=False)
        print(f"Saved to outputs/dev_with_speechqe.parquet")

    elif args.mode == "predict":
        print("Predict mode - load trained model and run on test set")
        print("TODO: implement after training")

    print("\n" + "=" * 80)
    print("SPEECH QE COMPLETE")
    print("=" * 80)
