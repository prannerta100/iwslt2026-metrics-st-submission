#!/bin/bash
# =============================================================================
# GPU Instance Setup Script for IWSLT 2026 Metrics Submission
# Target: AWS Blackwell 96GB GPU
# =============================================================================
set -e

echo "============================================================"
echo "IWSLT 2026 Metrics - GPU Environment Setup"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/6] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg libsndfile1 sox git-lfs

# ---------------------------------------------------------------------------
# 2. Install Poetry
# ---------------------------------------------------------------------------
echo "[2/6] Installing Poetry..."
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi
poetry --version

# Configure poetry to create virtualenv in project directory
poetry config virtualenvs.in-project true

# ---------------------------------------------------------------------------
# 3. Install all dependencies
# ---------------------------------------------------------------------------
echo "[3/6] Installing Python dependencies via Poetry..."

# If DLAMI already has PyTorch+CUDA, use system packages
poetry install --no-root

# Verify CUDA
poetry run python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ---------------------------------------------------------------------------
# 4. HuggingFace authentication
# ---------------------------------------------------------------------------
echo "[4/6] Setting up HuggingFace auth..."
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Set it with: export HF_TOKEN=your_token"
    echo "Required for downloading gated models (xCOMET-XL, CometKiwi-23-XL)"
else
    poetry run huggingface-cli login --token "$HF_TOKEN"
    echo "HuggingFace authenticated."
fi

# ---------------------------------------------------------------------------
# 5. Download models
# ---------------------------------------------------------------------------
echo "[5/6] Downloading models..."

poetry run python3 -c "
import os
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

from comet import download_model

# CometKiwi-22 (open)
print('Downloading CometKiwi-22...')
try:
    path = download_model('Unbabel/wmt22-cometkiwi-da')
    print(f'  -> {path}')
except Exception as e:
    print(f'  WARN: {e}')

# CometKiwi-23-XL (gated - needs HF access)
print('Downloading CometKiwi-23-XL...')
try:
    path = download_model('Unbabel/wmt23-cometkiwi-da-xl')
    print(f'  -> {path}')
except Exception as e:
    print(f'  WARN: {e} (may need to accept license on HuggingFace)')

# xCOMET-XL (gated - needs HF access)
print('Downloading xCOMET-XL...')
try:
    path = download_model('Unbabel/XCOMET-XL')
    print(f'  -> {path}')
except Exception as e:
    print(f'  WARN: {e} (may need to accept license on HuggingFace)')
"

# Download SONAR models
poetry run python3 -c "
print('Downloading SONAR/BLASER models...')
try:
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    t2vec = TextToEmbeddingModelPipeline(encoder='text_sonar_basic_encoder', tokenizer='text_sonar_basic_encoder')
    print('  SONAR text encoder: OK')
except Exception as e:
    print(f'  WARN: {e}')

try:
    from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
    s2vec = SpeechToEmbeddingModelPipeline(encoder='sonar_speech_encoder_eng')
    print('  SONAR speech encoder: OK')
except Exception as e:
    print(f'  WARN: {e}')
"

# ---------------------------------------------------------------------------
# 6. Download training data (text columns only, skips 71GB audio)
# ---------------------------------------------------------------------------
echo "[6/6] Downloading IWSLT data (text-only parquet)..."
poetry run python scripts/01d_download_and_explore.py

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  poetry shell"
echo "  export HF_TOKEN=your_token"
echo "  python scripts/02_cometkiwi_baseline.py"
echo "  python scripts/03_finetune_cometkiwi.py"
echo "  python scripts/05_xcomet_inference.py"
echo "  python scripts/06_blaser_inference.py"
echo "  python scripts/07_speech_qe.py"
echo "  python scripts/04_ensemble.py"
echo "  python scripts/run_all.py"
