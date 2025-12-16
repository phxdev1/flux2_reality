#!/usr/bin/env python3
"""Test the semantic primitives system without loading FLUX.

Run with: python scripts/test_primitives.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from primitives import TripleParser, SemanticMatcher, PrimitiveLoader


def main():
    print("=" * 60)
    print("REALITY ENGINE - Semantic Primitives Test")
    print("=" * 60)

    # Initialize components
    primitives_dir = Path(__file__).parent.parent / "primitives"

    parser = TripleParser()
    matcher = SemanticMatcher(primitives_dir=primitives_dir)
    loader = PrimitiveLoader()

    print(f"\n📦 Loaded {len(matcher.primitives)} primitives:")
    for category, names in sorted(matcher.list_primitives().items() if hasattr(matcher, 'list_primitives') else []):
        print(f"   [{category}] {', '.join(names) if isinstance(names, list) else names}")

    # Group by category manually
    by_category = {}
    for p in matcher.primitives:
        if p.category not in by_category:
            by_category[p.category] = []
        by_category[p.category].append(p.name)

    for cat, names in sorted(by_category.items()):
        print(f"   [{cat}] {', '.join(names)}")

    # Test cases
    test_prompts = [
        "a dragon flying over a dark city at night",
        "detective → walks through → rainy alley",
        "[noir] a cat sitting on a windowsill",
        "[fast] sunset over mountains",
        "neon lights illuminate the cyberpunk street",
        "a ghibli-style forest with magical creatures",
    ]

    for prompt in test_prompts:
        print("\n" + "=" * 60)
        print(f"📝 INPUT: {prompt}")
        print("=" * 60)

        # Parse into triples
        triples = parser.parse(prompt)
        if triples:
            print("\n🔷 PARSED TRIPLES:")
            for t in triples:
                print(f"   {t}")

        # Match primitives
        matches = matcher.match(prompt)

        if matches:
            print("\n🎯 MATCHED PRIMITIVES:")
            for m in matches:
                triggers = f" (triggers: {', '.join(m.matched_triggers)})" if m.matched_triggers else ""
                print(f"   [{m.primitive.category}] {m.primitive.name}: {m.score:.2f}{triggers}")

            # Compose parameters
            params = loader.compose_params(matches)
            print("\n⚙️  COMPOSED PARAMETERS:")
            for k, v in params.items():
                if isinstance(v, float):
                    print(f"   {k}: {v:.2f}")
                else:
                    print(f"   {k}: {v}")

            # Compose prompt
            final_prompt, negative = loader.compose_prompt(matches, prompt)
            print(f"\n✨ FINAL PROMPT:")
            print(f"   {final_prompt}")
            if negative:
                print(f"\n🚫 NEGATIVE:")
                print(f"   {negative}")
        else:
            print("\n   (no primitives matched)")

    # Interactive mode
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE - Enter prompts to test (Ctrl+C to exit)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n> ").strip()
            if not prompt:
                continue

            triples = parser.parse(prompt)
            matches = matcher.match(prompt)

            if triples:
                print(f"  Triples: {[str(t) for t in triples]}")

            if matches:
                print(f"  Matches: {[(m.primitive.name, f'{m.score:.2f}') for m in matches]}")

                final_prompt, negative = loader.compose_prompt(matches, prompt)
                print(f"  Final: {final_prompt}")

                params = loader.compose_params(matches)
                print(f"  Params: guidance={params.get('guidance', 4.0):.1f}, steps={params.get('num_steps', 50)}")
            else:
                print("  No matches - would use base FLUX settings")

        except KeyboardInterrupt:
            print("\n\nbye!")
            break
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
