#!/usr/bin/env python3
"""Lightweight test of semantic primitives (no torch required).

Tests parsing, trigger matching, and prompt composition without ML dependencies.

Run with: python scripts/test_primitives_lite.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import re

# Minimal implementations for testing without torch

@dataclass
class PromptModifier:
    prefix: str = ""
    suffix: str = ""
    keywords: list = field(default_factory=list)
    negative: list = field(default_factory=list)
    replace: dict = field(default_factory=dict)

@dataclass
class PrimitiveParams:
    guidance_bias: float = 0.0
    num_steps_min: int = None
    num_steps_override: int = None
    contrast: float = 1.0
    saturation: float = 1.0

@dataclass
class Primitive:
    name: str
    category: str = "general"
    description: str = ""
    triggers: list = field(default_factory=list)
    prompt: PromptModifier = field(default_factory=PromptModifier)
    params: PrimitiveParams = field(default_factory=PrimitiveParams)

    @classmethod
    def from_yaml(cls, path: Path):
        with open(path) as f:
            data = yaml.safe_load(f)

        prompt_data = data.get("prompt", {})
        prompt = PromptModifier(
            prefix=prompt_data.get("prefix", ""),
            suffix=prompt_data.get("suffix", ""),
            keywords=prompt_data.get("keywords", []),
            negative=prompt_data.get("negative", []),
            replace=prompt_data.get("replace", {}),
        )

        params_data = data.get("params", {})
        params = PrimitiveParams(
            guidance_bias=params_data.get("guidance_bias", 0.0),
            num_steps_min=params_data.get("num_steps_min"),
            num_steps_override=params_data.get("num_steps_override"),
            contrast=params_data.get("contrast", 1.0),
            saturation=params_data.get("saturation", 1.0),
        )

        return cls(
            name=data["name"],
            category=data.get("category", path.parent.name),
            description=data.get("description", ""),
            triggers=data.get("triggers", []),
            prompt=prompt,
            params=params,
        )

@dataclass
class Match:
    primitive: Primitive
    score: float
    matched_triggers: list = field(default_factory=list)


def load_primitives(primitives_dir: Path) -> list:
    """Load all primitives from directory."""
    primitives = []
    for yaml_file in primitives_dir.rglob("*.yaml"):
        try:
            p = Primitive.from_yaml(yaml_file)
            p.category = yaml_file.parent.name
            primitives.append(p)
        except Exception as e:
            print(f"Warning: {yaml_file}: {e}")
    return primitives


def match_triggers(text: str, primitives: list) -> list:
    """Match primitives based on trigger words."""
    text_lower = text.lower()
    matches = []

    for p in primitives:
        score = 0.0
        matched = []

        # Check explicit hints like [noir]
        hint_pattern = f"[{p.name.lower()}]"
        if hint_pattern in text_lower:
            score = 1.0
            matched.append(f"[{p.name}]")

        # Check trigger words
        for trigger in p.triggers:
            if trigger.lower() in text_lower:
                score = max(score, 0.7)
                matched.append(trigger)

        if score > 0:
            matches.append(Match(p, score, matched))

    return sorted(matches, key=lambda m: m.score, reverse=True)


def compose_prompt(matches: list, base_prompt: str) -> tuple:
    """Compose final prompt from matches."""
    if not matches:
        return base_prompt, ""

    prefixes = []
    suffixes = []
    keywords = []
    negatives = []

    for m in matches:
        if m.score < 0.5:
            continue
        pm = m.primitive.prompt
        if pm.prefix:
            prefixes.append(pm.prefix)
        if pm.suffix:
            suffixes.append(pm.suffix)
        keywords.extend(pm.keywords)
        negatives.extend(pm.negative)

    # Deduplicate
    prefixes = list(dict.fromkeys(prefixes))
    suffixes = list(dict.fromkeys(suffixes))
    keywords = list(dict.fromkeys(keywords))
    negatives = list(dict.fromkeys(negatives))

    parts = prefixes + [base_prompt]
    if keywords:
        parts.append(", ".join(keywords))
    parts.extend(suffixes)

    return ", ".join(parts), ", ".join(negatives)


def compose_params(matches: list) -> dict:
    """Compose parameters from matches."""
    guidance = 4.0
    steps = 50
    contrast = 1.0

    for m in matches:
        p = m.primitive.params
        guidance += p.guidance_bias * m.score
        if p.num_steps_override:
            steps = p.num_steps_override
        elif p.num_steps_min and p.num_steps_min > steps:
            steps = p.num_steps_min
        contrast *= 1.0 + (p.contrast - 1.0) * m.score

    return {"guidance": guidance, "num_steps": steps, "contrast": contrast}


# Triple parser (simplified)
TRIPLE_PATTERN = re.compile(r"^(?:\[([^\]]+)\]\s*)?(.+?)\s*[→\->]+\s*(.+?)\s*[→\->]+\s*(.+)$")

def parse_triple(text: str):
    m = TRIPLE_PATTERN.match(text)
    if m:
        hints, subj, pred, obj = m.groups()
        return {"subject": subj, "predicate": pred, "object": obj, "hints": hints}
    return None


def main():
    primitives_dir = Path(__file__).parent.parent / "primitives"
    primitives = load_primitives(primitives_dir)

    print("=" * 60)
    print("REALITY ENGINE - Primitives Test (No GPU Required)")
    print("=" * 60)

    print(f"\n📦 Loaded {len(primitives)} primitives:")
    by_cat = {}
    for p in primitives:
        by_cat.setdefault(p.category, []).append(p.name)
    for cat, names in sorted(by_cat.items()):
        print(f"   [{cat}] {', '.join(sorted(names))}")

    # Test cases
    tests = [
        "a dragon flying over a dark city at night",
        "detective → walks through → rainy alley",
        "[noir] a cat sitting on a windowsill",
        "[fast] sunset over mountains",
        "neon lights illuminate the cyberpunk street",
        "a ghibli-style forest with magical creatures",
    ]

    for prompt in tests:
        print("\n" + "-" * 60)
        print(f"📝 INPUT: {prompt}")

        # Check for triple syntax
        triple = parse_triple(prompt)
        if triple:
            print(f"   Triple: {triple['subject']} → {triple['predicate']} → {triple['object']}")

        matches = match_triggers(prompt, primitives)

        if matches:
            print(f"🎯 MATCHES:")
            for m in matches:
                print(f"   {m.primitive.name}: {m.score:.2f} ({', '.join(m.matched_triggers)})")

            final, negative = compose_prompt(matches, prompt)
            params = compose_params(matches)

            print(f"✨ FINAL: {final[:80]}..." if len(final) > 80 else f"✨ FINAL: {final}")
            if negative:
                print(f"🚫 NEGATIVE: {negative}")
            print(f"⚙️  PARAMS: guidance={params['guidance']:.1f}, steps={params['num_steps']}, contrast={params['contrast']:.2f}")
        else:
            print("   (no matches - base settings)")

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (Ctrl+C to exit)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
            if not prompt:
                continue

            matches = match_triggers(prompt, primitives)
            if matches:
                for m in matches:
                    print(f"  ✓ {m.primitive.name}: {m.score:.2f}")

                final, neg = compose_prompt(matches, prompt)
                params = compose_params(matches)
                print(f"  → {final}")
                print(f"  ⚙ guidance={params['guidance']:.1f}, steps={params['num_steps']}")
            else:
                print("  (no matches)")

        except KeyboardInterrupt:
            print("\nbye!")
            break


if __name__ == "__main__":
    main()
