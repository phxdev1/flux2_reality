#!/usr/bin/env python3
"""RunPod Serverless Handler for Flux Reality Engine.

This handler receives requests, processes them through the semantic
primitives system, and generates images using FLUX.2.

Environment Variables:
    FLUX2_MODEL_PATH: Path to FLUX.2 model weights
    AE_MODEL_PATH: Path to autoencoder weights
    PRIMITIVES_PATH: Path to primitives directory
"""

import os
import sys
import base64
import io
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import runpod
import torch
from PIL import Image

# Global model cache
MODELS = {
    "flux": None,
    "ae": None,
    "mistral": None,
    "engine": None,
}


def load_models():
    """Load FLUX.2 models into memory."""
    if MODELS["flux"] is not None:
        return  # Already loaded

    print("Loading models...")
    start = time.time()

    from flux2.util import load_flow_model, load_ae, load_mistral_small_embedder
    from primitives import SemanticMatcher, PrimitiveLoader, TripleParser

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_offload = device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 50 * 1024**3

    # Load FLUX.2 model
    print("Loading FLUX.2 model...")
    MODELS["flux"] = load_flow_model(
        "flux.2-dev",
        device="cpu" if cpu_offload else device
    )

    # Load autoencoder
    print("Loading autoencoder...")
    MODELS["ae"] = load_ae("flux.2-dev", device=device)
    MODELS["ae"].eval()

    # Load text encoder
    print("Loading Mistral encoder...")
    MODELS["mistral"] = load_mistral_small_embedder(device=device)
    MODELS["mistral"].eval()

    # Initialize primitives system
    primitives_path = Path(os.environ.get("PRIMITIVES_PATH", "/workspace/primitives"))
    print(f"Loading primitives from {primitives_path}...")

    MODELS["matcher"] = SemanticMatcher(primitives_dir=primitives_path, device=device)
    MODELS["loader"] = PrimitiveLoader(device=device)
    MODELS["parser"] = TripleParser()
    MODELS["cpu_offload"] = cpu_offload
    MODELS["device"] = device

    print(f"Models loaded in {time.time() - start:.1f}s")
    print(f"Loaded {len(MODELS['matcher'].primitives)} primitives")


def generate_image(
    prompt: str,
    width: int = 1360,
    height: int = 768,
    num_steps: int = 50,
    guidance: float = 4.0,
    seed: int = None,
    use_primitives: bool = True,
) -> dict:
    """Generate an image from a semantic prompt.

    Args:
        prompt: The semantic description or triple
        width: Output width
        height: Output height
        num_steps: Denoising steps
        guidance: Guidance scale
        seed: Random seed (None for random)
        use_primitives: Whether to use semantic primitive matching

    Returns:
        Dict with image (base64), parameters used, and matched primitives
    """
    import random
    from einops import rearrange
    from flux2.sampling import (
        batched_prc_img,
        batched_prc_txt,
        denoise,
        get_schedule,
        scatter_ids,
    )

    device = MODELS["device"]
    cpu_offload = MODELS["cpu_offload"]

    # Process through primitives system
    matched_primitives = []
    final_prompt = prompt
    negative_prompt = ""

    if use_primitives:
        # Parse and match
        triples = MODELS["parser"].parse(prompt)
        matches = MODELS["matcher"].match(prompt)

        if matches:
            # Compose prompt
            final_prompt, negative_prompt = MODELS["loader"].compose_prompt(matches, prompt)

            # Compose parameters
            params = MODELS["loader"].compose_params(matches, {
                "guidance": guidance,
                "num_steps": num_steps,
            })
            guidance = params.get("guidance", guidance)
            num_steps = int(params.get("num_steps", num_steps))

            matched_primitives = [
                {"name": m.primitive.name, "score": round(m.score, 2), "category": m.primitive.category}
                for m in matches
            ]

    # Generate seed
    if seed is None:
        seed = random.randrange(2**31)

    print(f"Generating: {final_prompt[:100]}...")
    print(f"  Steps: {num_steps}, Guidance: {guidance:.1f}, Seed: {seed}")

    # Run generation
    with torch.no_grad():
        # Encode prompt
        ctx = MODELS["mistral"]([final_prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        if cpu_offload:
            MODELS["mistral"] = MODELS["mistral"].cpu()
            torch.cuda.empty_cache()
            MODELS["flux"] = MODELS["flux"].to(device)

        # Create noise
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=device).manual_seed(seed)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device)
        x, x_ids = batched_prc_img(randn)

        # Get schedule and denoise
        timesteps = get_schedule(num_steps, x.shape[1])
        x = denoise(
            MODELS["flux"],
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=guidance,
        )

        # Decode
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = MODELS["ae"].decode(x).float()

        if cpu_offload:
            MODELS["flux"] = MODELS["flux"].cpu()
            torch.cuda.empty_cache()
            MODELS["mistral"] = MODELS["mistral"].to(device)

    # Convert to image
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    # Encode to base64
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", quality=95)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return {
        "image": img_base64,
        "prompt_original": prompt,
        "prompt_final": final_prompt,
        "negative_prompt": negative_prompt,
        "matched_primitives": matched_primitives,
        "parameters": {
            "width": width,
            "height": height,
            "num_steps": num_steps,
            "guidance": guidance,
            "seed": seed,
        }
    }


def handler(event):
    """RunPod serverless handler function."""
    try:
        # Ensure models are loaded
        load_models()

        # Parse input
        input_data = event.get("input", {})

        prompt = input_data.get("prompt", "a beautiful landscape")
        width = input_data.get("width", 1360)
        height = input_data.get("height", 768)
        num_steps = input_data.get("num_steps", 50)
        guidance = input_data.get("guidance", 4.0)
        seed = input_data.get("seed", None)
        use_primitives = input_data.get("use_primitives", True)

        # Validate dimensions
        width = max(256, min(2048, width))
        height = max(256, min(2048, height))
        width = (width // 16) * 16
        height = (height // 16) * 16

        # Generate
        result = generate_image(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
            use_primitives=use_primitives,
        )

        return {"output": result}

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# For local testing
if __name__ == "__main__":
    # Check if running as RunPod serverless or local test
    if os.environ.get("RUNPOD_POD_ID"):
        runpod.serverless.start({"handler": handler})
    else:
        print("Running local test...")

        # Test without full model loading
        print("\nTesting primitives system only (no GPU required):")

        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from primitives import SemanticMatcher, PrimitiveLoader, TripleParser

        primitives_path = Path(__file__).parent.parent / "primitives"
        matcher = SemanticMatcher(primitives_dir=primitives_path)
        loader = PrimitiveLoader()
        parser = TripleParser()

        test_prompt = "a dragon flying over a noir city at night"
        print(f"\nTest prompt: {test_prompt}")

        matches = matcher.match(test_prompt)
        if matches:
            print(f"Matched: {[(m.primitive.name, f'{m.score:.2f}') for m in matches]}")

            final, negative = loader.compose_prompt(matches, test_prompt)
            params = loader.compose_params(matches)

            print(f"Final prompt: {final[:100]}...")
            print(f"Parameters: guidance={params.get('guidance', 4.0):.1f}, steps={params.get('num_steps', 50)}")
        else:
            print("No primitives matched")

        print("\nTo run with full FLUX.2 model, deploy to RunPod or run with GPU.")
