"""Primitive manifest schema definitions.

Defines the structure for semantic primitives that can be dynamically loaded
and composed based on cosine similarity matching.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml
import torch


@dataclass
class PrimitiveParams:
    """Pipeline parameter modifications applied by a primitive."""

    guidance_bias: float = 0.0  # Added to base guidance
    guidance_scale: float = 1.0  # Multiplied with base guidance
    num_steps_min: Optional[int] = None  # Minimum denoising steps
    num_steps_max: Optional[int] = None  # Maximum denoising steps
    num_steps_override: Optional[int] = None  # Force specific step count
    schedule_mu_bias: float = 0.0  # Adjust schedule mu parameter
    schedule_sigma: float = 1.0  # Schedule sigma for SNR shift
    contrast: float = 1.0  # Post-process contrast adjustment
    saturation: float = 1.0  # Post-process saturation
    lora_strength: float = 1.0  # LoRA merge strength multiplier


@dataclass
class PrimitiveInject:
    """Specifies where primitive weights should be injected."""

    modulation: bool = True  # Inject into Modulation layers
    cross_attention: bool = False  # Modify cross-attention weights
    self_attention: bool = False  # Modify self-attention weights
    mlp: bool = False  # Modify MLP layers
    time_embedding: bool = False  # Modify time embedding
    guidance_embedding: bool = False  # Modify guidance embedding


@dataclass
class Primitive:
    """A semantic primitive that modifies the generation pipeline.

    Primitives are the core building blocks of the Reality Engine. Each primitive
    encapsulates:
    - Semantic identity (name, triggers, embedding)
    - Visual assets (LoRA weights, reference images)
    - Pipeline modifications (guidance, steps, schedule)
    - Injection configuration (where to apply modifications)
    """

    name: str
    version: str = "1.0"
    description: str = ""
    category: str = "general"  # styles, concepts, controls, speed

    # Semantic anchors
    triggers: list[str] = field(default_factory=list)
    embedding_path: Optional[Path] = None
    _embedding: Optional[torch.Tensor] = field(default=None, repr=False)

    # Assets
    weights_path: Optional[Path] = None
    reference_images: list[Path] = field(default_factory=list)

    # Pipeline modifications
    params: PrimitiveParams = field(default_factory=PrimitiveParams)

    # Injection configuration
    inject: PrimitiveInject = field(default_factory=PrimitiveInject)

    @property
    def embedding(self) -> Optional[torch.Tensor]:
        """Lazy-load embedding from file."""
        if self._embedding is None and self.embedding_path is not None:
            if self.embedding_path.exists():
                self._embedding = torch.load(self.embedding_path, weights_only=True)
        return self._embedding

    @embedding.setter
    def embedding(self, value: torch.Tensor):
        self._embedding = value

    def has_weights(self) -> bool:
        """Check if this primitive has LoRA weights."""
        return self.weights_path is not None and self.weights_path.exists()

    def has_reference_images(self) -> bool:
        """Check if this primitive has reference images for IP-Adapter."""
        return len(self.reference_images) > 0

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Primitive":
        """Load a primitive from a YAML manifest file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        base_dir = yaml_path.parent

        # Parse params
        params_data = data.get("params", {})
        params = PrimitiveParams(**params_data)

        # Parse inject
        inject_data = data.get("inject", {})
        inject = PrimitiveInject(**inject_data)

        # Resolve paths relative to manifest location
        embedding_path = None
        if "embedding" in data:
            embedding_path = base_dir / data["embedding"]

        weights_path = None
        if "weights" in data:
            weights_path = base_dir / data["weights"]

        reference_images = []
        for img in data.get("reference_images", []):
            reference_images.append(base_dir / img)

        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            triggers=data.get("triggers", []),
            embedding_path=embedding_path,
            weights_path=weights_path,
            reference_images=reference_images,
            params=params,
            inject=inject,
        )

    def to_yaml(self, yaml_path: Path) -> None:
        """Save primitive manifest to YAML file."""
        base_dir = yaml_path.parent

        data = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "triggers": self.triggers,
        }

        if self.embedding_path:
            data["embedding"] = str(self.embedding_path.relative_to(base_dir))

        if self.weights_path:
            data["weights"] = str(self.weights_path.relative_to(base_dir))

        if self.reference_images:
            data["reference_images"] = [
                str(p.relative_to(base_dir)) for p in self.reference_images
            ]

        # Only include non-default params
        params_dict = {}
        default_params = PrimitiveParams()
        for field_name in vars(default_params):
            value = getattr(self.params, field_name)
            default = getattr(default_params, field_name)
            if value != default:
                params_dict[field_name] = value
        if params_dict:
            data["params"] = params_dict

        # Only include non-default inject
        inject_dict = {}
        default_inject = PrimitiveInject()
        for field_name in vars(default_inject):
            value = getattr(self.inject, field_name)
            default = getattr(default_inject, field_name)
            if value != default:
                inject_dict[field_name] = value
        if inject_dict:
            data["inject"] = inject_dict

        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@dataclass
class PrimitiveMatch:
    """Result of matching a primitive against user input."""

    primitive: Primitive
    score: float  # Cosine similarity score (0-1)
    matched_triggers: list[str] = field(default_factory=list)

    def __lt__(self, other: "PrimitiveMatch") -> bool:
        """Sort by score descending."""
        return self.score > other.score

    @property
    def weighted_params(self) -> PrimitiveParams:
        """Get params weighted by match score."""
        p = self.primitive.params
        return PrimitiveParams(
            guidance_bias=p.guidance_bias * self.score,
            guidance_scale=1.0 + (p.guidance_scale - 1.0) * self.score,
            num_steps_min=p.num_steps_min,
            num_steps_max=p.num_steps_max,
            num_steps_override=p.num_steps_override,
            schedule_mu_bias=p.schedule_mu_bias * self.score,
            schedule_sigma=1.0 + (p.schedule_sigma - 1.0) * self.score,
            contrast=1.0 + (p.contrast - 1.0) * self.score,
            saturation=1.0 + (p.saturation - 1.0) * self.score,
            lora_strength=p.lora_strength * self.score,
        )
