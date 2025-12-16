# CLAUDE.md - Flux Reality Engine

## Project Vision

This is a **Flux-based Reality Engine** - a system where words become reality. Describe anything using semantic triples and FLUX.2 manifests it as an image. The engine uses **emergent semantic structuring** with noun-predicate-noun abstractions to dynamically assemble generation pipelines.

## Core Innovation: Semantic Primitives

Instead of complex code, we use **semantic similarity** to dynamically build image generation pipelines:

```
"dragon → soars over → noir city"
        ↓ embed & match
[creatures: 0.87] [motion: 0.72] [noir: 0.91] [architecture: 0.68]
        ↓ load matched primitives
Dynamic pipeline with merged LoRAs, adjusted guidance, shaped schedule
        ↓
Generated image
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  SEMANTIC LAYER                                                 │
│  "dragon → soars over → noir city"                              │
│       ↓                                                         │
│  Triple Parser + Mistral Embedder                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓ cosine similarity
┌─────────────────────────────────────────────────────────────────┐
│  PRIMITIVE LIBRARY (primitives/)                                │
│                                                                 │
│  styles/     → noir.safetensors, ghibli.safetensors            │
│  concepts/   → creatures.safetensors, architecture.safetensors  │
│  controls/   → pose.safetensors, depth.safetensors             │
│  speed/      → fast.safetensors (LCM), quality.safetensors     │
└─────────────────────────────────────────────────────────────────┘
                              ↓ dynamic load & merge
┌─────────────────────────────────────────────────────────────────┐
│  FLUX.2 PIPELINE                                                │
│  - Primitives modify: guidance, steps, schedule shape           │
│  - LoRA weights merged by similarity score                      │
│  - Modulation vectors injected at transformer blocks            │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Base Model**: FLUX.2 [dev] - 32B parameter flow matching transformer
- **Runtime**: Python 3.10-3.12, PyTorch 2.8, CUDA 12.6+
- **Infrastructure**: RunPod (H100/A100 GPUs recommended)
- **Text Encoding**: Mistral-Small-3.2-24B-Instruct-2506
- **Primitive Format**: Safetensors + YAML manifests
- **Borrowed From**: IP-Adapter (styles), CtrLoRA (controls), LCM (speed)

## Directory Structure

```
src/
├── flux2/                    # Core FLUX.2 implementation
│   ├── model.py              # 32B flow matching transformer
│   ├── autoencoder.py        # VAE encoder/decoder
│   ├── text_encoder.py       # Mistral embeddings
│   ├── sampling.py           # Denoising & scheduling
│   └── util.py               # Model loading
│
├── primitives/               # Semantic primitive system
│   ├── schema.py             # Primitive manifest types
│   ├── matcher.py            # Cosine similarity matching
│   ├── loader.py             # Dynamic safetensors loading
│   └── parser.py             # Triple parser (noun→pred→noun)
│
└── reality/                  # Reality engine core
    ├── engine.py             # Main generation orchestrator
    └── pipeline.py           # Dynamic pipeline assembly

primitives/                   # Primitive library (safetensors + manifests)
├── styles/
│   ├── noir.safetensors
│   └── noir.yaml
├── concepts/
├── controls/
└── speed/

scripts/
└── cli.py                    # Interactive REPL
```

## Primitive Manifest Schema

```yaml
# primitives/styles/noir.yaml
name: noir
version: "1.0"
description: "Dark, high-contrast noir aesthetic"

# Semantic anchors for matching
embedding: "noir.embed.pt"  # Pre-computed embedding
triggers:
  - "noir"
  - "dark"
  - "shadow"
  - "detective"
  - "rain"
  - "contrast"

# Assets
weights: "noir.safetensors"  # LoRA weights
reference_images:            # Optional IP-Adapter style refs
  - "noir_ref_1.jpg"
  - "noir_ref_2.jpg"

# Pipeline modifications
params:
  guidance_bias: 2.0         # Add to base guidance
  contrast: 1.4              # Post-process adjustment
  num_steps_min: 40          # Minimum steps for quality

# Injection points
inject:
  modulation: true           # Inject into Modulation layers
  cross_attention: true      # Modify cross-attention
```

## Quick Start

```bash
# Setup environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu126 --no-cache-dir

# Run the reality engine
export PYTHONPATH=src
python scripts/cli.py
```

## Semantic Input Format

```bash
# Simple prompt (auto-parsed)
> a dragon flying over a neon city at night

# Explicit triples (more control)
> dragon → soars over → city
> neon lights → illuminate → streets
> rain → falls on → everything

# With primitive hints
> [noir] detective → walks through → rainy alley
> [fast] cat → sits on → windowsill
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `[Enter]` or `run` | Generate from current config |
| `show` | Display config + matched primitives |
| `reset` | Reset to defaults |
| `prompt="..."` | Set semantic description |
| `primitives` | List available primitives |
| `explain` | Show which primitives matched and why |

## RunPod Deployment

### GPU Requirements
- **Minimum**: RTX 4090 (24GB) - quantized + remote text encoder
- **Recommended**: H100 (80GB) - full model with CPU offloading
- **Optimal**: GB200 or multi-GPU

### Environment Variables
```bash
export FLUX2_MODEL_PATH="/workspace/models/flux2"
export AE_MODEL_PATH="/workspace/models/ae"
export PRIMITIVES_PATH="/workspace/primitives"
export OPENROUTER_API_KEY="<your-key>"
```

## Creating New Primitives

1. **Train a LoRA** (using CtrLoRA approach):
   ```bash
   # ~1000 images, ~1 hour on single GPU
   python scripts/train_primitive.py --name "cyberpunk" --data ./cyberpunk_images/
   ```

2. **Generate embedding**:
   ```bash
   python scripts/embed_primitive.py --name "cyberpunk" --triggers "neon,cyber,future,dystopia"
   ```

3. **Create manifest** (`primitives/styles/cyberpunk.yaml`)

4. **Drop in** - automatically indexed on next run

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Lint & format
ruff check src/ scripts/
ruff format src/ scripts/

# Test primitive matching
python -m pytest tests/test_matcher.py -v
```

## Key Concepts

### Semantic Similarity Matching
- Embed user input using Mistral
- Compare against pre-computed primitive embeddings
- Select primitives above threshold (default: 0.6)
- Weight influence by similarity score

### Dynamic Pipeline Assembly
- No hardcoded if/else for styles
- Pipeline emerges from matched primitives
- Multiple primitives compose (weighted merge)
- Only load what's needed (VRAM efficient)

### Primitive Composition
```
matched: [noir: 0.91, creatures: 0.87, motion: 0.72]
         ↓
guidance = base + (0.91 * noir.guidance_bias) + (0.87 * creatures.guidance_bias)
lora = merge(noir.weights * 0.91, creatures.weights * 0.87)
schedule = shape(base_schedule, motion.schedule_params * 0.72)
```

## Borrowed Innovations

| Source | What We Use |
|--------|-------------|
| FLUX.2 | Base 32B model, Mistral encoder |
| IP-Adapter | Reference image style injection |
| CtrLoRA | Efficient primitive training (~10% params) |
| LCM | Fast generation primitives (4-step) |
| Stable Cascade | Multi-stage pipeline concept |
| Diffusers | Modular pipeline architecture |
