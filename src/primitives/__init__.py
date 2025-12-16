"""Semantic Primitives System for the Flux Reality Engine.

This module provides the core infrastructure for semantic-based dynamic pipeline assembly:
- Schema: Dataclasses for primitive manifests
- Matcher: Cosine similarity matching against primitive embeddings
- Loader: Dynamic safetensors loading and weight merging
- Parser: Triple parser for noun → predicate → noun extraction
"""

from .schema import Primitive, PrimitiveParams, PrimitiveInject, PrimitiveMatch
from .matcher import SemanticMatcher
from .loader import PrimitiveLoader
from .parser import TripleParser, Triple

__all__ = [
    "Primitive",
    "PrimitiveParams",
    "PrimitiveInject",
    "PrimitiveMatch",
    "SemanticMatcher",
    "PrimitiveLoader",
    "TripleParser",
    "Triple",
]
