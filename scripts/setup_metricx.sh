#!/bin/bash
# Setup MetricX-24 on the VM
# Run this before running 09_metricx_inference.py

set -e

echo "=== Setting up MetricX-24 ==="

# Clone the metricx repo if not already present
if [ ! -d "/tmp/metricx" ]; then
    echo "Cloning metricx repo..."
    git clone https://github.com/google-research/metricx.git /tmp/metricx
else
    echo "metricx repo already cloned at /tmp/metricx"
fi

# Install metricx24 package
echo "Installing metricx24..."
cd /tmp/metricx
pip install -e . 2>/dev/null || {
    echo "pip install -e failed, trying manual install..."
    export PYTHONPATH="/tmp/metricx:$PYTHONPATH"
}

# Install sentencepiece if not present
pip install sentencepiece 2>/dev/null || poetry add sentencepiece 2>/dev/null || true

echo ""
echo "=== Testing MetricX-24 import ==="
python -c "
try:
    from metricx24.models import MT5ForRegression
    print('SUCCESS: metricx24.models.MT5ForRegression imported')
except ImportError as e:
    print(f'WARNING: metricx24 import failed: {e}')
    print('Will use transformers fallback mode')
    from transformers import AutoModelForSeq2SeqLM
    print('SUCCESS: transformers fallback available')
"

echo ""
echo "=== MetricX-24 setup complete ==="
echo "Now run: python scripts/09_metricx_inference.py --batch-size 16"
