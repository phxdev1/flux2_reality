"""Reality Engine - The core orchestrator for semantic image generation.

Transforms semantic input into images by:
1. Parsing input into semantic triples
2. Matching against primitive library
3. Dynamically assembling generation pipeline
4. Running FLUX.2 with composed parameters
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import torch

from primitives import (
    TripleParser,
    Triple,
    SemanticMatcher,
    PrimitiveLoader,
    PrimitiveMatch,
)


@dataclass
class GenerationConfig:
    """Configuration for a generation run."""

    prompt: str = ""
    triples: list[Triple] = field(default_factory=list)
    width: int = 1360
    height: int = 768
    seed: Optional[int] = None
    guidance: float = 4.0
    num_steps: int = 50

    # Primitive overrides
    force_primitives: list[str] = field(default_factory=list)  # Always include these
    exclude_primitives: list[str] = field(default_factory=list)  # Never include these
    similarity_threshold: float = 0.6


@dataclass
class GenerationResult:
    """Result of a generation run."""

    image: torch.Tensor  # Generated image tensor
    config: GenerationConfig
    matched_primitives: list[PrimitiveMatch]
    composed_params: dict
    final_prompt: str
    seed_used: int


class RealityEngine:
    """Main orchestrator for the semantic image generation pipeline.

    The Reality Engine provides a high-level interface for generating images
    from semantic descriptions. It handles:
    - Parsing natural language into semantic triples
    - Matching primitives based on semantic similarity
    - Composing pipeline parameters from matched primitives
    - Orchestrating the FLUX.2 generation pipeline
    """

    def __init__(
        self,
        primitives_dir: Path,
        embed_fn: Optional[Callable[[str], torch.Tensor]] = None,
        device: str = "cuda",
        verbose: bool = True,
    ):
        """Initialize the Reality Engine.

        Args:
            primitives_dir: Directory containing primitive manifests
            embed_fn: Function to embed text (from Mistral or other encoder)
            device: Device for computation
            verbose: Whether to print status messages
        """
        self.device = device
        self.verbose = verbose

        # Initialize components
        self.parser = TripleParser()
        self.matcher = SemanticMatcher(
            primitives_dir=primitives_dir,
            embed_fn=embed_fn,
            device=device,
        )
        self.loader = PrimitiveLoader(device=device)

        # FLUX.2 components (loaded lazily)
        self._flux_model = None
        self._ae = None
        self._mistral = None

    def set_embed_fn(self, embed_fn: Callable[[str], torch.Tensor]) -> None:
        """Set the embedding function for semantic matching."""
        self.matcher.set_embed_fn(embed_fn)

    def parse(self, text: str) -> list[Triple]:
        """Parse input text into semantic triples."""
        return self.parser.parse(text)

    def match(
        self,
        text: str,
        config: Optional[GenerationConfig] = None,
    ) -> list[PrimitiveMatch]:
        """Match primitives against input text.

        Args:
            text: Input text (raw or parsed into triples)
            config: Optional config for threshold and overrides

        Returns:
            List of matched primitives sorted by score
        """
        if config is None:
            config = GenerationConfig()

        # Get base matches
        matches = self.matcher.match(
            text,
            exclude=config.exclude_primitives,
        )

        # Add forced primitives
        for name in config.force_primitives:
            primitive = self.matcher.get_primitive(name)
            if primitive and not any(m.primitive.name == name for m in matches):
                matches.append(PrimitiveMatch(
                    primitive=primitive,
                    score=1.0,
                    matched_triggers=[f"[forced:{name}]"],
                ))

        # Filter by threshold
        matches = [m for m in matches if m.score >= config.similarity_threshold]

        # Re-sort
        matches.sort()

        return matches

    def compose(
        self,
        matches: list[PrimitiveMatch],
        base_config: Optional[GenerationConfig] = None,
    ) -> dict:
        """Compose pipeline parameters from matched primitives.

        Args:
            matches: List of primitive matches
            base_config: Base configuration to modify

        Returns:
            Composed parameter dictionary
        """
        if base_config is None:
            base_config = GenerationConfig()

        base_params = {
            "guidance": base_config.guidance,
            "num_steps": base_config.num_steps,
            "width": base_config.width,
            "height": base_config.height,
        }

        return self.loader.compose_params(matches, base_params)

    def explain(self, text: str, config: Optional[GenerationConfig] = None) -> str:
        """Explain what primitives would be matched and how.

        Args:
            text: Input text to analyze
            config: Optional generation config

        Returns:
            Human-readable explanation string
        """
        triples = self.parse(text)
        matches = self.match(text, config)
        composed = self.compose(matches, config)

        lines = ["=" * 60]
        lines.append("REALITY ENGINE ANALYSIS")
        lines.append("=" * 60)

        lines.append("\n📝 PARSED TRIPLES:")
        if triples:
            for t in triples:
                lines.append(f"  • {t}")
        else:
            lines.append("  (no triples parsed, using raw text)")

        lines.append("\n🎯 MATCHED PRIMITIVES:")
        if matches:
            for match in matches:
                lines.append(f"\n  [{match.primitive.category}] {match.primitive.name}")
                lines.append(f"    Score: {match.score:.2f}")
                if match.matched_triggers:
                    lines.append(f"    Triggers: {', '.join(match.matched_triggers)}")
                lines.append(f"    Description: {match.primitive.description}")
        else:
            lines.append("  (no primitives matched)")

        lines.append("\n⚙️  COMPOSED PARAMETERS:")
        for key, value in composed.items():
            lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def prepare_generation(
        self,
        text: str,
        config: Optional[GenerationConfig] = None,
    ) -> tuple[GenerationConfig, list[PrimitiveMatch], dict]:
        """Prepare for generation without running FLUX.2.

        Useful for previewing what will happen before committing to generation.

        Args:
            text: Input text
            config: Optional base configuration

        Returns:
            Tuple of (config, matches, composed_params)
        """
        if config is None:
            config = GenerationConfig()

        config.prompt = text
        config.triples = self.parse(text)

        matches = self.match(text, config)
        composed = self.compose(matches, config)

        # Update config with composed values
        config.guidance = composed.get("guidance", config.guidance)
        config.num_steps = int(composed.get("num_steps", config.num_steps))

        return config, matches, composed

    def generate(
        self,
        text: str,
        config: Optional[GenerationConfig] = None,
        flux_model=None,
        ae=None,
        mistral=None,
    ) -> GenerationResult:
        """Generate an image from semantic input.

        This is the main entry point for generation. It:
        1. Parses the input into triples
        2. Matches against primitives
        3. Composes parameters
        4. Runs FLUX.2 generation

        Args:
            text: Semantic input text
            config: Generation configuration
            flux_model: FLUX.2 model (or uses cached)
            ae: Autoencoder (or uses cached)
            mistral: Mistral encoder (or uses cached)

        Returns:
            GenerationResult with image and metadata
        """
        # Prepare generation
        config, matches, composed = self.prepare_generation(text, config)

        if self.verbose:
            print(self.explain(text, config))

        # Use provided models or cached
        model = flux_model or self._flux_model
        autoencoder = ae or self._ae
        encoder = mistral or self._mistral

        if model is None:
            raise RuntimeError(
                "No FLUX model available. Either pass flux_model parameter "
                "or call load_models() first."
            )

        # Merge primitive weights if any
        merged_weights = self.loader.merge_weights(matches)
        if merged_weights:
            if self.verbose:
                print(f"Applying merged weights from {len(matches)} primitives...")
            # Apply weights to model (implementation depends on FLUX architecture)
            # self.loader.apply_to_model(model, merged_weights)

        # Generate seed if not provided
        import random
        seed = config.seed if config.seed is not None else random.randrange(2**31)

        # Build final prompt from triples
        if config.triples:
            final_prompt = self.parser.to_prompt(config.triples)
        else:
            final_prompt = config.prompt

        if self.verbose:
            print(f"\n🎨 Generating with prompt: {final_prompt}")
            print(f"   Seed: {seed}, Steps: {config.num_steps}, Guidance: {config.guidance:.2f}")

        # Run generation (placeholder - actual FLUX.2 call goes here)
        # This would integrate with the existing cli.py generation logic
        image = self._run_flux_generation(
            model=model,
            ae=autoencoder,
            mistral=encoder,
            prompt=final_prompt,
            width=config.width,
            height=config.height,
            num_steps=config.num_steps,
            guidance=config.guidance,
            seed=seed,
        )

        return GenerationResult(
            image=image,
            config=config,
            matched_primitives=matches,
            composed_params=composed,
            final_prompt=final_prompt,
            seed_used=seed,
        )

    def _run_flux_generation(
        self,
        model,
        ae,
        mistral,
        prompt: str,
        width: int,
        height: int,
        num_steps: int,
        guidance: float,
        seed: int,
    ) -> torch.Tensor:
        """Run the actual FLUX.2 generation.

        This method contains the core generation logic adapted from cli.py.
        """
        from flux2.sampling import (
            batched_prc_img,
            batched_prc_txt,
            denoise,
            get_schedule,
            scatter_ids,
        )
        from einops import rearrange

        with torch.no_grad():
            # Encode prompt
            ctx = mistral([prompt]).to(torch.bfloat16)
            ctx, ctx_ids = batched_prc_txt(ctx)

            # Create noise
            shape = (1, 128, height // 16, width // 16)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            randn = torch.randn(
                shape,
                generator=generator,
                dtype=torch.bfloat16,
                device=self.device,
            )
            x, x_ids = batched_prc_img(randn)

            # Get schedule
            timesteps = get_schedule(num_steps, x.shape[1])

            # Denoise
            x = denoise(
                model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guidance,
            )

            # Decode
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            x = ae.decode(x).float()
            x = x.clamp(-1, 1)

        return x

    def list_primitives(self) -> dict[str, list[str]]:
        """List all available primitives by category."""
        result = {}
        for p in self.matcher.primitives:
            if p.category not in result:
                result[p.category] = []
            result[p.category].append(p.name)
        return result

    def reload_primitives(self) -> None:
        """Reload primitives from disk."""
        self.matcher.reload()
        self.loader.clear_cache()
        if self.verbose:
            print(f"Reloaded {len(self.matcher.primitives)} primitives")
