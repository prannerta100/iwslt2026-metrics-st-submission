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
echo "[1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg libsndfile1 sox git-lfs

# ---------------------------------------------------------------------------
# 2. Python environment
# ---------------------------------------------------------------------------
echo "[2/7] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. PyTorch (CUDA 12.x)
# ---------------------------------------------------------------------------
echo "[3/7] Installing PyTorch with CUDA..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ---------------------------------------------------------------------------
# 4. Core ML dependencies
# ---------------------------------------------------------------------------
echo "[4/7] Installing ML dependencies..."
pip install numpy scipy pandas scikit-learn tqdm matplotlib seaborn

# HuggingFace ecosystem
pip install transformers>=4.40.0 datasets huggingface-hub soundfile librosa

# COMET (MT evaluation) - pin setuptools for pkg_resources compatibility
pip install 'setuptools<82'
pip install 'unbabel-comet>=2.2.0'

# SONAR/BLASER (cross-modal QE)
pip install 'sonar-space>=0.4.0'

# Audio processing
pip install torchcodec

# LightGBM for ensemble
pip install lightgbm

# ---------------------------------------------------------------------------
# 5. HuggingFace authentication
# ---------------------------------------------------------------------------
echo "[5/7] Setting up HuggingFace auth..."
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Set it with: export HF_TOKEN=your_token"
    echo "Required for downloading gated models (xCOMET-XL, CometKiwi-23-XL)"
else
    huggingface-cli login --token "$HF_TOKEN"
    echo "HuggingFace authenticated."
fi

# ---------------------------------------------------------------------------
# 6. Download models
# ---------------------------------------------------------------------------
echo "[6/7] Downloading models..."

python3 -c "
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
python3 -c "
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
# 7. Download training data
# ---------------------------------------------------------------------------
echo "[7/7] Downloading IWSLT data..."
python3 -c "
from datasets import load_dataset
print('Loading IWSLT 2026 Metrics dataset...')
ds = load_dataset('maikezu/iwslt2026-metrics-shared-train-dev')
print(f'Splits: {list(ds.keys())}')
for split in ds:
    print(f'  {split}: {len(ds[split])} rows')
print('Dataset cached successfully.')
"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  source venv/bin/activate"
echo "  export HF_TOKEN=your_token"
echo "  python scripts/02_cometkiwi_baseline.py"
echo "  python scripts/03_finetune_cometkiwi.py"
echo "  python scripts/05_xcomet_inference.py"
echo "  python scripts/06_blaser_inference.py"
echo "  python scripts/07_speech_qe.py"
echo "  python scripts/04_ensemble.py"
echo "  python scripts/run_all.py"
