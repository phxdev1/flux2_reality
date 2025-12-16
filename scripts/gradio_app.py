#!/usr/bin/env python3
"""Gradio Web UI for Flux Reality Engine.

Provides a simple web interface for testing semantic primitives
and generating images.

Run with: python scripts/gradio_app.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr

# Try to import full stack, fall back to lite mode
FULL_MODE = False
try:
    import torch
    if torch.cuda.is_available():
        FULL_MODE = True
except ImportError:
    pass

from primitives import SemanticMatcher, PrimitiveLoader, TripleParser

# Initialize primitives system
primitives_path = Path(__file__).parent.parent / "primitives"
matcher = SemanticMatcher(primitives_dir=primitives_path)
loader = PrimitiveLoader()
parser = TripleParser()

# Global model cache for full mode
MODELS = {}


def analyze_prompt(prompt: str) -> str:
    """Analyze a prompt and show matched primitives."""
    if not prompt.strip():
        return "Enter a prompt to analyze."

    lines = []
    lines.append("## Parsed Triples\n")

    triples = parser.parse(prompt)
    if triples:
        for t in triples:
            lines.append(f"- `{t.subject}` → `{t.predicate}` → `{t.object}`")
    else:
        lines.append("_(No explicit triples found, using raw text)_")

    lines.append("\n## Matched Primitives\n")

    matches = matcher.match(prompt)
    if matches:
        for m in matches:
            triggers = ", ".join(m.matched_triggers) if m.matched_triggers else "semantic"
            lines.append(f"- **{m.primitive.name}** [{m.primitive.category}]: {m.score:.2f} ({triggers})")
    else:
        lines.append("_(No primitives matched)_")

    lines.append("\n## Composed Prompt\n")

    if matches:
        final, negative = loader.compose_prompt(matches, prompt)
        lines.append(f"```\n{final}\n```")

        if negative:
            lines.append(f"\n**Negative:** `{negative}`")

        lines.append("\n## Composed Parameters\n")
        params = loader.compose_params(matches)
        lines.append(f"- Guidance: `{params.get('guidance', 4.0):.2f}`")
        lines.append(f"- Steps: `{params.get('num_steps', 50)}`")
        lines.append(f"- Contrast: `{params.get('contrast', 1.0):.2f}`")
    else:
        lines.append(f"```\n{prompt}\n```")
        lines.append("\n_(Using default parameters)_")

    return "\n".join(lines)


def list_primitives() -> str:
    """List all available primitives."""
    lines = ["## Available Primitives\n"]

    by_category = {}
    for p in matcher.primitives:
        by_category.setdefault(p.category, []).append(p)

    for category, prims in sorted(by_category.items()):
        lines.append(f"### {category.title()}\n")
        for p in sorted(prims, key=lambda x: x.name):
            triggers = ", ".join(p.triggers[:5])
            if len(p.triggers) > 5:
                triggers += "..."
            lines.append(f"- **{p.name}**: {p.description}")
            lines.append(f"  - Triggers: `{triggers}`")
            if p.prompt.prefix:
                lines.append(f"  - Prefix: _{p.prompt.prefix[:50]}..._")
        lines.append("")

    return "\n".join(lines)


def generate_image(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    use_primitives: bool,
):
    """Generate an image (requires GPU)."""
    if not FULL_MODE:
        return None, "GPU not available. Run on RunPod for image generation."

    # Import generation code
    from runpod_handler import generate_image as gen

    result = gen(
        prompt=prompt,
        width=width,
        height=height,
        num_steps=steps,
        guidance=guidance,
        seed=seed if seed > 0 else None,
        use_primitives=use_primitives,
    )

    # Decode base64 image
    import base64
    import io
    from PIL import Image

    img_data = base64.b64decode(result["image"])
    img = Image.open(io.BytesIO(img_data))

    info = f"**Matched:** {[m['name'] for m in result['matched_primitives']]}\n"
    info += f"**Final prompt:** {result['prompt_final'][:200]}..."

    return img, info


# Build Gradio interface
with gr.Blocks(title="Flux Reality Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🌌 Flux Reality Engine

    Semantic primitives-based image generation. Describe anything and
    primitives automatically compose the generation pipeline.

    **Try:** `a dragon flying over a noir city` or `[ghibli] magical forest with creatures`
    """)

    with gr.Tab("Analyze"):
        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a semantic description...",
                    lines=3,
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Column():
                analysis_output = gr.Markdown(label="Analysis")

        analyze_btn.click(
            analyze_prompt,
            inputs=[prompt_input],
            outputs=[analysis_output],
        )

        gr.Examples(
            examples=[
                ["a dragon flying over a dark city at night"],
                ["detective → walks through → rainy alley"],
                ["[noir] a cat sitting on a windowsill"],
                ["[fast] sunset over mountains"],
                ["neon lights illuminate the cyberpunk street"],
                ["a ghibli-style forest with magical creatures"],
            ],
            inputs=[prompt_input],
        )

    with gr.Tab("Generate"):
        if FULL_MODE:
            with gr.Row():
                with gr.Column():
                    gen_prompt = gr.Textbox(label="Prompt", lines=3)
                    with gr.Row():
                        gen_width = gr.Slider(256, 2048, 1360, step=16, label="Width")
                        gen_height = gr.Slider(256, 2048, 768, step=16, label="Height")
                    with gr.Row():
                        gen_steps = gr.Slider(4, 60, 50, step=1, label="Steps")
                        gen_guidance = gr.Slider(1, 10, 4, step=0.5, label="Guidance")
                    gen_seed = gr.Number(label="Seed (0 for random)", value=0)
                    gen_primitives = gr.Checkbox(label="Use Primitives", value=True)
                    gen_btn = gr.Button("Generate", variant="primary")

                with gr.Column():
                    gen_image = gr.Image(label="Generated Image")
                    gen_info = gr.Markdown()

            gen_btn.click(
                generate_image,
                inputs=[gen_prompt, gen_width, gen_height, gen_steps, gen_guidance, gen_seed, gen_primitives],
                outputs=[gen_image, gen_info],
            )
        else:
            gr.Markdown("""
            ## GPU Required

            Image generation requires a GPU. Deploy to RunPod:

            ```bash
            # On RunPod GPU pod:
            cd /workspace/flux2_reality
            python scripts/gradio_app.py
            ```

            Or use the CLI:
            ```bash
            python scripts/cli.py --cpu_offloading True
            ```
            """)

    with gr.Tab("Primitives"):
        gr.Markdown(list_primitives())


if __name__ == "__main__":
    # Get port from environment or default
    port = int(os.environ.get("GRADIO_PORT", 7860))

    print(f"Starting Gradio app on port {port}...")
    print(f"Mode: {'Full (GPU)' if FULL_MODE else 'Lite (CPU)'}")
    print(f"Loaded {len(matcher.primitives)} primitives")

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "false").lower() == "true",
    )
