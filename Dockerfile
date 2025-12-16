# Flux Reality Engine - RunPod Deployment
# Image: magickai/flux-reality-engine
# Base: PyTorch with CUDA 12.4 support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.title="Flux Reality Engine"
LABEL org.opencontainers.image.description="Semantic primitives-based image generation with FLUX.2"
LABEL org.opencontainers.image.vendor="magickai"
LABEL org.opencontainers.image.source="https://github.com/phxdev1/flux2_reality"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /workspace/flux2_reality

# Install Python dependencies
WORKDIR /workspace/flux2_reality
RUN pip install --no-cache-dir -e . \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install additional dependencies for RunPod
RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    gradio

# Create directories for models and outputs
RUN mkdir -p /workspace/models /workspace/outputs /workspace/primitives

# Copy primitives to workspace
RUN cp -r /workspace/flux2_reality/primitives/* /workspace/primitives/ 2>/dev/null || true

# Set environment variables for model paths
ENV PYTHONPATH=/workspace/flux2_reality/src
ENV FLUX2_MODEL_PATH=/workspace/models/flux2-dev.safetensors
ENV AE_MODEL_PATH=/workspace/models/ae.safetensors
ENV PRIMITIVES_PATH=/workspace/primitives
ENV HF_HOME=/workspace/hf_cache

# Expose port for Gradio UI (optional)
EXPOSE 7860

# Default command - can be overridden
CMD ["python", "-u", "/workspace/flux2_reality/scripts/runpod_handler.py"]
