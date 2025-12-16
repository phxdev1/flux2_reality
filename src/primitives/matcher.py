"""Semantic matcher for finding relevant primitives.

Uses cosine similarity between user input embeddings and pre-computed
primitive embeddings to dynamically select which primitives to apply.
"""

from pathlib import Path
from typing import Optional, Callable
import torch
import torch.nn.functional as F

from .schema import Primitive, PrimitiveMatch


class SemanticMatcher:
    """Matches user input against a library of semantic primitives.

    The matcher uses two strategies:
    1. Trigger matching: Direct keyword matching against primitive triggers
    2. Embedding similarity: Cosine similarity between input and primitive embeddings

    Both strategies contribute to the final match score.
    """

    def __init__(
        self,
        primitives_dir: Path,
        embed_fn: Optional[Callable[[str], torch.Tensor]] = None,
        similarity_threshold: float = 0.6,
        trigger_boost: float = 0.2,
        device: str = "cuda",
    ):
        """Initialize the semantic matcher.

        Args:
            primitives_dir: Root directory containing primitive subdirectories
            embed_fn: Function to embed text strings (e.g., from Mistral)
            similarity_threshold: Minimum similarity score to consider a match
            trigger_boost: Bonus added to score when trigger words match
            device: Device for tensor operations
        """
        self.primitives_dir = Path(primitives_dir)
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.trigger_boost = trigger_boost
        self.device = device

        self.primitives: list[Primitive] = []
        self.primitive_embeddings: Optional[torch.Tensor] = None
        self.trigger_index: dict[str, list[int]] = {}  # trigger -> primitive indices

        self._load_primitives()

    def _load_primitives(self) -> None:
        """Load all primitives from the primitives directory."""
        self.primitives = []
        self.trigger_index = {}

        if not self.primitives_dir.exists():
            return

        # Scan all subdirectories for YAML manifests
        for category_dir in self.primitives_dir.iterdir():
            if not category_dir.is_dir():
                continue

            for yaml_file in category_dir.glob("*.yaml"):
                try:
                    primitive = Primitive.from_yaml(yaml_file)
                    primitive.category = category_dir.name
                    idx = len(self.primitives)
                    self.primitives.append(primitive)

                    # Build trigger index
                    for trigger in primitive.triggers:
                        trigger_lower = trigger.lower()
                        if trigger_lower not in self.trigger_index:
                            self.trigger_index[trigger_lower] = []
                        self.trigger_index[trigger_lower].append(idx)

                except Exception as e:
                    print(f"Warning: Failed to load primitive from {yaml_file}: {e}")

        # Stack embeddings for batch similarity computation
        self._build_embedding_index()

    def _build_embedding_index(self) -> None:
        """Build the embedding index for fast similarity search."""
        embeddings = []
        for primitive in self.primitives:
            if primitive.embedding is not None:
                embeddings.append(primitive.embedding)
            else:
                # Create embedding from triggers if no pre-computed embedding
                if self.embed_fn and primitive.triggers:
                    trigger_text = " ".join(primitive.triggers)
                    emb = self.embed_fn(trigger_text)
                    primitive.embedding = emb
                    embeddings.append(emb)
                else:
                    # Placeholder zero embedding
                    embeddings.append(torch.zeros(1))

        if embeddings and all(e.shape == embeddings[0].shape for e in embeddings):
            self.primitive_embeddings = torch.stack(embeddings).to(self.device)
            # Normalize for cosine similarity
            self.primitive_embeddings = F.normalize(self.primitive_embeddings, dim=-1)

    def set_embed_fn(self, embed_fn: Callable[[str], torch.Tensor]) -> None:
        """Set the embedding function and rebuild index."""
        self.embed_fn = embed_fn
        self._build_embedding_index()

    def match(
        self,
        text: str,
        top_k: Optional[int] = None,
        categories: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> list[PrimitiveMatch]:
        """Match user input against all primitives.

        Args:
            text: User input text to match
            top_k: Maximum number of matches to return (None = all above threshold)
            categories: Only match primitives from these categories
            exclude: Primitive names to exclude from matching

        Returns:
            List of PrimitiveMatch objects sorted by score descending
        """
        if not self.primitives:
            return []

        text_lower = text.lower()
        matches: list[PrimitiveMatch] = []

        # Compute embedding similarity if available
        embedding_scores = self._compute_embedding_scores(text)

        for idx, primitive in enumerate(self.primitives):
            # Filter by category
            if categories and primitive.category not in categories:
                continue

            # Filter by exclusion
            if exclude and primitive.name in exclude:
                continue

            # Start with embedding similarity score
            score = embedding_scores[idx] if embedding_scores is not None else 0.0

            # Check for trigger matches
            matched_triggers = []
            for trigger in primitive.triggers:
                if trigger.lower() in text_lower:
                    matched_triggers.append(trigger)
                    score = min(1.0, score + self.trigger_boost)

            # Also check explicit primitive hints like [noir]
            hint_pattern = f"[{primitive.name.lower()}]"
            if hint_pattern in text_lower:
                matched_triggers.append(f"[{primitive.name}]")
                score = 1.0  # Explicit hint = max score

            if score >= self.similarity_threshold or matched_triggers:
                matches.append(
                    PrimitiveMatch(
                        primitive=primitive,
                        score=max(score, self.similarity_threshold if matched_triggers else 0),
                        matched_triggers=matched_triggers,
                    )
                )

        # Sort by score descending
        matches.sort()

        if top_k:
            matches = matches[:top_k]

        return matches

    def _compute_embedding_scores(self, text: str) -> Optional[torch.Tensor]:
        """Compute cosine similarity between text and all primitive embeddings."""
        if self.embed_fn is None or self.primitive_embeddings is None:
            return None

        try:
            # Embed the input text
            text_embedding = self.embed_fn(text).to(self.device)
            text_embedding = F.normalize(text_embedding, dim=-1)

            # Handle different embedding shapes
            if text_embedding.dim() == 1:
                text_embedding = text_embedding.unsqueeze(0)

            # Compute cosine similarity
            if self.primitive_embeddings.dim() == 2:
                # [num_primitives, embed_dim] @ [1, embed_dim].T -> [num_primitives, 1]
                scores = torch.mm(self.primitive_embeddings, text_embedding.T).squeeze(-1)
            else:
                # Fallback to individual comparisons
                scores = torch.tensor([
                    F.cosine_similarity(text_embedding, pe.unsqueeze(0)).item()
                    for pe in self.primitive_embeddings
                ])

            return scores.cpu()

        except Exception as e:
            print(f"Warning: Embedding similarity computation failed: {e}")
            return None

    def explain_match(self, text: str, match: PrimitiveMatch) -> str:
        """Generate a human-readable explanation of why a primitive matched."""
        lines = [f"Primitive: {match.primitive.name} (score: {match.score:.2f})"]

        if match.matched_triggers:
            lines.append(f"  Matched triggers: {', '.join(match.matched_triggers)}")

        if match.score >= self.similarity_threshold:
            lines.append(f"  Semantic similarity above threshold ({self.similarity_threshold})")

        params = match.weighted_params
        if params.guidance_bias != 0:
            lines.append(f"  Will adjust guidance by: {params.guidance_bias:+.2f}")
        if params.num_steps_override:
            lines.append(f"  Will override steps to: {params.num_steps_override}")
        if match.primitive.has_weights():
            lines.append(f"  Will apply LoRA with strength: {params.lora_strength:.2f}")

        return "\n".join(lines)

    def list_primitives(self, category: Optional[str] = None) -> list[str]:
        """List all available primitive names."""
        return [
            p.name for p in self.primitives
            if category is None or p.category == category
        ]

    def get_primitive(self, name: str) -> Optional[Primitive]:
        """Get a primitive by name."""
        for p in self.primitives:
            if p.name == name:
                return p
        return None

    def reload(self) -> None:
        """Reload all primitives from disk."""
        self._load_primitives()
