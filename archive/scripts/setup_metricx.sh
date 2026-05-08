#!/bin/bash
# Setup MetricX-24 on the VM
# Run this before running 09_metricx_inference.py
#
# MetricX has no PyPI package. We clone the repo and add it to sys.path.
# All actual dependencies (transformers, sentencepiece, torch) are already
# in our poetry env via pyproject.toml.

set -e

METRICX_DIR="/tmp/metricx"

echo "=== Setting up MetricX-24 ==="

# Clone the metricx repo if not already present
if [ ! -d "$METRICX_DIR" ]; then
    echo "Cloning metricx repo..."
    git clone https://github.com/google-research/metricx.git "$METRICX_DIR"
else
    echo "metricx repo already cloned at $METRICX_DIR"
fi

echo ""
echo "=== Testing MetricX-24 import ==="
PYTHONPATH="$METRICX_DIR:$PYTHONPATH" poetry run python -c "
from metricx24.models import MT5ForRegression
print('SUCCESS: metricx24.models.MT5ForRegression imported')
print(f'Module location: {MT5ForRegression.__module__}')
"

echo ""
echo "=== MetricX-24 setup complete ==="
echo "Now run: poetry run python scripts/09_metricx_inference.py --batch-size 16"
