# RunPod Deployment Guide

Deploy the Flux Reality Engine on RunPod for GPU-accelerated image generation.

## Quick Start (GPU Pod)

### 1. Create a GPU Pod

- Go to [RunPod](https://runpod.io)
- Select a GPU: **H100 (80GB)** recommended, A100 or RTX 4090 also work
- Choose template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- Set volume: 100GB+ for models

### 2. Connect and Setup

```bash
# SSH into your pod, then:
cd /workspace

# Clone the repository
git clone https://github.com/phxdev1/flux2_reality.git
cd flux2_reality

# Run setup script
chmod +x scripts/start_runpod.sh
./scripts/start_runpod.sh
```

### 3. Download Models (First Time)

Models auto-download from HuggingFace, or manually:

```bash
# Create models directory
mkdir -p /workspace/models

# Download FLUX.2 (requires HF access)
huggingface-cli download black-forest-labs/FLUX.2-dev \
    flux2-dev.safetensors \
    --local-dir /workspace/models

# Download autoencoder
huggingface-cli download black-forest-labs/FLUX.2-dev \
    ae.safetensors \
    --local-dir /workspace/models
```

### 4. Test Generation

```bash
# Test primitives system (no GPU)
python scripts/test_primitives_lite.py

# Interactive CLI with FLUX.2
export PYTHONPATH=/workspace/flux2_reality/src
python scripts/cli.py --cpu_offloading True
```

---

## Serverless Deployment

For auto-scaling serverless inference:

### 1. Build Docker Image

```bash
# Build locally
docker build -t flux-reality-engine .

# Or use RunPod's container registry
```

### 2. Create Serverless Endpoint

1. Go to RunPod Serverless
2. Create new endpoint
3. Use your Docker image
4. Configure:
   - Min Workers: 0
   - Max Workers: 3
   - GPU: H100 or A100
   - Idle Timeout: 60s

### 3. API Usage

```python
import requests

response = requests.post(
    "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "input": {
            "prompt": "a dragon flying over a noir city at night",
            "width": 1360,
            "height": 768,
            "num_steps": 50,
            "guidance": 4.0,
            "use_primitives": True
        }
    }
)

result = response.json()
image_base64 = result["output"]["image"]
matched = result["output"]["matched_primitives"]
```

---

## GPU Requirements

| GPU | VRAM | Mode | Notes |
|-----|------|------|-------|
| H100 | 80GB | Full | Fastest, recommended |
| A100 | 80GB | Full | Great performance |
| A100 | 40GB | CPU Offload | Slower but works |
| RTX 4090 | 24GB | Quantized | Use diffusers quantized version |

### CPU Offloading

For GPUs with less VRAM:

```bash
python scripts/cli.py --cpu_offloading True
```

---

## Environment Variables

```bash
# Model paths (auto-download if not set)
export FLUX2_MODEL_PATH=/workspace/models/flux2-dev.safetensors
export AE_MODEL_PATH=/workspace/models/ae.safetensors

# Primitives
export PRIMITIVES_PATH=/workspace/flux2_reality/primitives

# Python path
export PYTHONPATH=/workspace/flux2_reality/src

# Optional: OpenRouter for prompt upsampling
export OPENROUTER_API_KEY=your-key-here
```

---

## Primitives on RunPod

The primitives directory is included in the Docker image. To add custom primitives:

```bash
# Copy to your RunPod volume
cp my_style.yaml /workspace/flux2_reality/primitives/styles/

# Or mount a volume with primitives
docker run -v /my/primitives:/workspace/primitives ...
```

---

## Troubleshooting

### Out of Memory

```bash
# Use CPU offloading
python scripts/cli.py --cpu_offloading True

# Or reduce resolution
prompt="..." width=1024 height=1024
```

### Model Download Fails

```bash
# Login to HuggingFace
huggingface-cli login

# Accept model license at:
# https://huggingface.co/black-forest-labs/FLUX.2-dev
```

### CUDA Errors

```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

---

## Costs

Approximate RunPod costs (as of 2024):

| GPU | $/hour | 1000 images (50 steps) |
|-----|--------|------------------------|
| H100 | ~$3.50 | ~$35 |
| A100 80GB | ~$2.00 | ~$25 |
| RTX 4090 | ~$0.70 | ~$15 |

*Estimates based on ~30s per image on H100*
