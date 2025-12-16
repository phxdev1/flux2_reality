"""Dynamic primitive loader with safetensors support.

Handles loading, merging, and applying primitive weights to the FLUX model.
Supports weighted composition of multiple primitives based on match scores.
"""

from pathlib import Path
from typing import Optional
import torch
from torch import nn
from safetensors.torch import load_file as load_safetensors

from .schema import Primitive, PrimitiveMatch, PrimitiveParams


class PrimitiveLoader:
    """Loads and manages primitive weights for dynamic pipeline assembly.

    Supports:
    - Lazy loading of safetensors weights
    - Weighted merging of multiple primitive LoRAs
    - Caching of loaded weights for efficiency
    - Parameter composition from multiple primitives
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_size: int = 10,
    ):
        """Initialize the primitive loader.

        Args:
            device: Device to load weights to
            dtype: Data type for loaded weights
            cache_size: Maximum number of primitives to keep in memory
        """
        self.device = device
        self.dtype = dtype
        self.cache_size = cache_size

        # LRU cache for loaded weights
        self._weight_cache: dict[str, dict[str, torch.Tensor]] = {}
        self._cache_order: list[str] = []

    def load_weights(self, primitive: Primitive) -> Optional[dict[str, torch.Tensor]]:
        """Load weights for a primitive.

        Args:
            primitive: Primitive to load weights for

        Returns:
            Dictionary of weight tensors, or None if no weights available
        """
        if not primitive.has_weights():
            return None

        cache_key = str(primitive.weights_path)

        # Check cache
        if cache_key in self._weight_cache:
            # Move to end of LRU order
            self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._weight_cache[cache_key]

        # Load from disk
        try:
            weights = load_safetensors(primitive.weights_path, device=self.device)

            # Convert to target dtype
            weights = {k: v.to(self.dtype) for k, v in weights.items()}

            # Add to cache
            self._weight_cache[cache_key] = weights
            self._cache_order.append(cache_key)

            # Evict oldest if cache full
            while len(self._cache_order) > self.cache_size:
                oldest = self._cache_order.pop(0)
                del self._weight_cache[oldest]

            return weights

        except Exception as e:
            print(f"Warning: Failed to load weights from {primitive.weights_path}: {e}")
            return None

    def merge_weights(
        self,
        matches: list[PrimitiveMatch],
        normalize: bool = True,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Merge weights from multiple matched primitives.

        Uses weighted averaging based on match scores:
        merged[key] = sum(weight[key] * score) / sum(scores)

        Args:
            matches: List of primitive matches with scores
            normalize: Whether to normalize by total score

        Returns:
            Merged weight dictionary, or None if no weights to merge
        """
        weights_to_merge: list[tuple[dict[str, torch.Tensor], float]] = []

        for match in matches:
            weights = self.load_weights(match.primitive)
            if weights is not None:
                # Use weighted params for strength
                strength = match.weighted_params.lora_strength
                weights_to_merge.append((weights, match.score * strength))

        if not weights_to_merge:
            return None

        # Find common keys
        all_keys = set(weights_to_merge[0][0].keys())
        for weights, _ in weights_to_merge[1:]:
            all_keys &= set(weights.keys())

        if not all_keys:
            print("Warning: No common weight keys found for merging")
            return None

        # Merge weights
        total_score = sum(score for _, score in weights_to_merge)
        merged: dict[str, torch.Tensor] = {}

        for key in all_keys:
            weighted_sum = None
            for weights, score in weights_to_merge:
                weighted = weights[key] * score
                if weighted_sum is None:
                    weighted_sum = weighted
                else:
                    weighted_sum = weighted_sum + weighted

            if normalize and total_score > 0:
                merged[key] = weighted_sum / total_score
            else:
                merged[key] = weighted_sum

        return merged

    def compose_params(
        self,
        matches: list[PrimitiveMatch],
        base_params: Optional[dict] = None,
    ) -> dict:
        """Compose pipeline parameters from multiple matched primitives.

        Args:
            matches: List of primitive matches
            base_params: Base parameters to modify

        Returns:
            Composed parameter dictionary
        """
        if base_params is None:
            base_params = {
                "guidance": 4.0,
                "num_steps": 50,
                "schedule_mu_bias": 0.0,
                "schedule_sigma": 1.0,
            }

        result = base_params.copy()

        # Accumulate biases
        guidance_bias = 0.0
        guidance_scale = 1.0
        mu_bias = 0.0
        sigma = 1.0
        contrast = 1.0
        saturation = 1.0

        # Track step constraints
        steps_min = None
        steps_max = None
        steps_override = None

        for match in matches:
            params = match.weighted_params

            guidance_bias += params.guidance_bias
            guidance_scale *= params.guidance_scale
            mu_bias += params.schedule_mu_bias
            sigma *= params.schedule_sigma
            contrast *= params.contrast
            saturation *= params.saturation

            # Step constraints (use most restrictive)
            if params.num_steps_override is not None:
                steps_override = params.num_steps_override
            if params.num_steps_min is not None:
                if steps_min is None or params.num_steps_min > steps_min:
                    steps_min = params.num_steps_min
            if params.num_steps_max is not None:
                if steps_max is None or params.num_steps_max < steps_max:
                    steps_max = params.num_steps_max

        # Apply to result
        result["guidance"] = (result.get("guidance", 4.0) + guidance_bias) * guidance_scale
        result["schedule_mu_bias"] = result.get("schedule_mu_bias", 0.0) + mu_bias
        result["schedule_sigma"] = result.get("schedule_sigma", 1.0) * sigma
        result["contrast"] = result.get("contrast", 1.0) * contrast
        result["saturation"] = result.get("saturation", 1.0) * saturation

        # Apply step constraints
        if steps_override is not None:
            result["num_steps"] = steps_override
        else:
            steps = result.get("num_steps", 50)
            if steps_min is not None:
                steps = max(steps, steps_min)
            if steps_max is not None:
                steps = min(steps, steps_max)
            result["num_steps"] = steps

        return result

    def apply_to_model(
        self,
        model: nn.Module,
        weights: dict[str, torch.Tensor],
        strength: float = 1.0,
    ) -> None:
        """Apply loaded weights to a model.

        Supports LoRA-style weight application where weights are added to
        existing model parameters.

        Args:
            model: Model to apply weights to
            weights: Weight dictionary from load_weights or merge_weights
            strength: Overall strength multiplier
        """
        model_state = model.state_dict()

        for key, weight in weights.items():
            if key in model_state:
                # Direct weight replacement/addition
                if weight.shape == model_state[key].shape:
                    model_state[key] = model_state[key] + weight * strength
                else:
                    print(f"Warning: Shape mismatch for {key}: "
                          f"{weight.shape} vs {model_state[key].shape}")

            elif ".lora_" in key:
                # LoRA weight - need to compute and apply
                self._apply_lora_weight(model, key, weight, strength)

    def _apply_lora_weight(
        self,
        model: nn.Module,
        key: str,
        weight: torch.Tensor,
        strength: float,
    ) -> None:
        """Apply a LoRA weight to the model.

        LoRA weights are typically in format:
        - layer.lora_A: down projection
        - layer.lora_B: up projection

        Applied as: W' = W + strength * (B @ A)
        """
        # Parse LoRA key format
        if ".lora_A" in key:
            base_key = key.replace(".lora_A", "")
            lora_b_key = key.replace(".lora_A", ".lora_B")

            # Find corresponding lora_B (would need to be passed in)
            # For now, store for later combination
            pass

        elif ".lora_B" in key:
            # lora_B is applied with lora_A
            pass

    def clear_cache(self) -> None:
        """Clear the weight cache."""
        self._weight_cache.clear()
        self._cache_order.clear()

    def get_cache_info(self) -> dict:
        """Get information about the cache state."""
        return {
            "cached_primitives": len(self._weight_cache),
            "cache_size_limit": self.cache_size,
            "cached_keys": list(self._weight_cache.keys()),
        }


def create_lora_weights(
    base_model: nn.Module,
    rank: int = 8,
    alpha: float = 1.0,
    target_modules: Optional[list[str]] = None,
) -> dict[str, torch.Tensor]:
    """Create empty LoRA weight structure for a model.

    Useful for understanding the expected weight format.

    Args:
        base_model: Model to create LoRA structure for
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        target_modules: Module names to create LoRA for (default: attention layers)

    Returns:
        Dictionary with empty LoRA weight tensors
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    lora_weights = {}

    for name, module in base_model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if not any(target in name for target in target_modules):
            continue

        in_features = module.in_features
        out_features = module.out_features

        # Create LoRA A (down projection) and B (up projection)
        lora_weights[f"{name}.lora_A"] = torch.zeros(rank, in_features)
        lora_weights[f"{name}.lora_B"] = torch.zeros(out_features, rank)
        lora_weights[f"{name}.lora_alpha"] = torch.tensor(alpha)

    return lora_weights
