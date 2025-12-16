# CLAUDE.md - Flux Reality Engine

## Project Vision

This is a **Flux-based Reality Engine** - a system where words become reality. Describe anything and FLUX.2 manifests it as an image. The engine runs on RunPod GPU infrastructure for high-performance generation.

## Tech Stack

- **Model**: FLUX.2 [dev] - 32B parameter flow matching transformer
- **Runtime**: Python 3.10-3.12, PyTorch 2.8, CUDA 12.6+
- **Infrastructure**: RunPod (H100/A100 GPUs recommended)
- **Text Encoding**: Mistral-Small-3.2-24B-Instruct-2506
- **Prompt Enhancement**: Local Mistral or OpenRouter API

## Architecture

```
src/flux2/
├── model.py          # Core 32B flow matching transformer
├── autoencoder.py    # FLUX.2 VAE encoder/decoder
├── text_encoder.py   # Mistral text embeddings
├── sampling.py       # Denoising & scheduling
├── util.py           # Model loading utilities
└── openrouter_api_client.py  # API for prompt upsampling

scripts/
└── cli.py            # Interactive generation CLI
```

## Quick Start

```bash
# Setup environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu126 --no-cache-dir

# Set model paths (optional - auto-downloads if not set)
export FLUX2_MODEL_PATH="<path-to-flux2-weights>"
export AE_MODEL_PATH="<path-to-autoencoder>"

# Run the reality engine
export PYTHONPATH=src
python scripts/cli.py
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `[Enter]` or `run` | Generate image from current config |
| `show` | Display current configuration |
| `reset` | Reset to default settings |
| `prompt="..."` | Set the reality description |
| `width=N height=N` | Set output dimensions |
| `seed=N` | Set random seed for reproducibility |
| `num_steps=N` | Denoising steps (default: 50) |
| `guidance=N` | Guidance scale (default: 4.0) |
| `input_images="a.jpg,b.jpg"` | Reference images for editing |
| `upsample_prompt_mode="local"` | Enable prompt enhancement |

## RunPod Deployment

### GPU Requirements
- **Minimum**: RTX 4090 (24GB) with quantized model + remote text encoder
- **Recommended**: H100 (80GB) with CPU offloading
- **Optimal**: GB200 or multi-GPU setup

### RunPod Setup
```bash
# For H100 pods - use CPU offloading
python scripts/cli.py --cpu_offloading True

# Environment variables for RunPod
export FLUX2_MODEL_PATH="/workspace/models/flux2"
export AE_MODEL_PATH="/workspace/models/ae"
export OPENROUTER_API_KEY="<your-key>"  # For prompt upsampling
```

### Low-VRAM Mode (RTX 4090)
Use diffusers with remote text encoder:
```python
from diffusers import Flux2Pipeline
# See README.md for full example with remote_text_encoder()
```

## Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `width` | 1360 | Output width in pixels |
| `height` | 768 | Output height in pixels |
| `num_steps` | 50 | Denoising iterations (28 for speed) |
| `guidance` | 4.0 | Prompt adherence strength |
| `seed` | random | Reproducibility seed |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint code
ruff check src/ scripts/

# Format code
ruff format src/ scripts/
```

## Key Files

- `scripts/cli.py` - Main entry point, interactive REPL
- `src/flux2/model.py` - Transformer architecture
- `src/flux2/sampling.py` - Diffusion sampling logic
- `src/flux2/text_encoder.py` - Text-to-embedding pipeline

## Output

Generated images are saved to `output/sample_N.png` with EXIF metadata indicating AI generation.
