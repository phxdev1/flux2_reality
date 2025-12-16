#!/bin/bash
# Start script for RunPod GPU Pod
# Run this after connecting to your pod

set -e

echo "=========================================="
echo "  Flux Reality Engine - RunPod Setup"
echo "=========================================="

# Navigate to workspace
cd /workspace

# Clone repo if not present
if [ ! -d "flux2_reality" ]; then
    echo "Cloning repository..."
    git clone https://github.com/phxdev1/flux2_reality.git
fi

cd flux2_reality

# Install dependencies
echo "Installing dependencies..."
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu124 --quiet

# Set up environment
export PYTHONPATH=/workspace/flux2_reality/src
export PRIMITIVES_PATH=/workspace/flux2_reality/primitives

# Create model directory
mkdir -p /workspace/models

# Check for models
echo ""
echo "Checking for models..."
if [ ! -f "/workspace/models/flux2-dev.safetensors" ]; then
    echo "FLUX.2 model not found. It will be downloaded on first run."
    echo "Or manually download from: https://huggingface.co/black-forest-labs/FLUX.2-dev"
    echo "  Place at: /workspace/models/flux2-dev.safetensors"
fi

if [ ! -f "/workspace/models/ae.safetensors" ]; then
    echo "Autoencoder not found. It will be downloaded on first run."
fi

# Set model paths if they exist
if [ -f "/workspace/models/flux2-dev.safetensors" ]; then
    export FLUX2_MODEL_PATH=/workspace/models/flux2-dev.safetensors
fi
if [ -f "/workspace/models/ae.safetensors" ]; then
    export AE_MODEL_PATH=/workspace/models/ae.safetensors
fi

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Available commands:"
echo ""
echo "  1. Test primitives (no GPU needed):"
echo "     python scripts/test_primitives_lite.py"
echo ""
echo "  2. Interactive CLI (with FLUX.2):"
echo "     python scripts/cli.py --cpu_offloading True"
echo ""
echo "  3. Start serverless handler:"
echo "     python scripts/runpod_handler.py"
echo ""
echo "  4. Start Gradio UI:"
echo "     python scripts/gradio_app.py"
echo ""
echo "Environment:"
echo "  PYTHONPATH=$PYTHONPATH"
echo "  PRIMITIVES_PATH=$PRIMITIVES_PATH"
echo ""
