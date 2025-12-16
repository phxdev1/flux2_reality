"""Triple parser for semantic input.

Parses user input into noun-predicate-noun triples for structured
semantic understanding and primitive matching.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Triple:
    """A semantic triple: subject → predicate → object."""

    subject: str
    predicate: str
    object: str
    modifiers: list[str] = field(default_factory=list)  # Adjectives, adverbs
    hints: list[str] = field(default_factory=list)  # Explicit primitive hints like [noir]

    def __str__(self) -> str:
        base = f"{self.subject} → {self.predicate} → {self.object}"
        if self.hints:
            return f"[{', '.join(self.hints)}] {base}"
        return base

    def to_prompt(self) -> str:
        """Convert triple back to natural language prompt."""
        parts = []

        # Add modifiers to subject
        subject_mods = [m for m in self.modifiers if m.endswith("_subj")]
        subject = self.subject
        if subject_mods:
            subject = " ".join(m.replace("_subj", "") for m in subject_mods) + " " + subject

        # Add modifiers to object
        object_mods = [m for m in self.modifiers if m.endswith("_obj")]
        obj = self.object
        if object_mods:
            obj = " ".join(m.replace("_obj", "") for m in object_mods) + " " + obj

        # Construct sentence
        return f"{subject} {self.predicate} {obj}"

    @property
    def components(self) -> list[str]:
        """Get all semantic components for embedding."""
        return [self.subject, self.predicate, self.object] + self.modifiers


class TripleParser:
    """Parses natural language into semantic triples.

    Supports multiple input formats:
    1. Explicit triples: "dragon → soars over → city"
    2. Natural language: "a dragon flying over a neon city"
    3. With hints: "[noir] detective → walks through → rainy alley"
    """

    # Common predicates for pattern matching
    PREDICATES = [
        # Spatial
        "over", "under", "above", "below", "beside", "near", "in", "on", "at",
        "through", "across", "between", "among", "around", "inside", "outside",
        # Motion
        "flies over", "soars over", "walks through", "runs across", "jumps over",
        "swims in", "floats above", "hovers over", "moves through", "travels across",
        # State/Action
        "illuminates", "lights up", "reflects on", "casts shadow on",
        "transforms into", "becomes", "turns into",
        "holds", "carries", "wears", "contains",
        "watches", "observes", "looks at", "gazes at",
        "falls on", "rains on", "covers", "surrounds",
        # Relational
        "is", "are", "has", "have", "with",
    ]

    # Pattern for explicit triple syntax
    TRIPLE_PATTERN = re.compile(
        r"^(?:\[([^\]]+)\]\s*)?(.+?)\s*[→\->]+\s*(.+?)\s*[→\->]+\s*(.+)$"
    )

    # Pattern for extracting hints
    HINT_PATTERN = re.compile(r"\[([^\]]+)\]")

    def __init__(self):
        # Sort predicates by length (longest first) for greedy matching
        self.predicates = sorted(self.PREDICATES, key=len, reverse=True)

    def parse(self, text: str) -> list[Triple]:
        """Parse input text into semantic triples.

        Args:
            text: User input text, may contain multiple lines/triples

        Returns:
            List of parsed Triple objects
        """
        triples = []

        # Split on newlines for multiple triples
        lines = text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try explicit triple syntax first
            triple = self._parse_explicit(line)
            if triple:
                triples.append(triple)
                continue

            # Fall back to natural language parsing
            parsed = self._parse_natural(line)
            triples.extend(parsed)

        return triples

    def _parse_explicit(self, text: str) -> Optional[Triple]:
        """Parse explicit triple syntax: subject → predicate → object."""
        match = self.TRIPLE_PATTERN.match(text)
        if not match:
            return None

        hints_str, subject, predicate, obj = match.groups()

        hints = []
        if hints_str:
            hints = [h.strip() for h in hints_str.split(",")]

        return Triple(
            subject=subject.strip(),
            predicate=predicate.strip(),
            object=obj.strip(),
            hints=hints,
        )

    def _parse_natural(self, text: str) -> list[Triple]:
        """Parse natural language into triples using pattern matching."""
        triples = []

        # Extract hints first
        hints = self.HINT_PATTERN.findall(text)
        text_clean = self.HINT_PATTERN.sub("", text).strip()

        # Try to find predicates in the text
        text_lower = text_clean.lower()

        for predicate in self.predicates:
            pred_lower = predicate.lower()

            # Look for predicate with word boundaries
            pattern = r"\b" + re.escape(pred_lower) + r"\b"
            match = re.search(pattern, text_lower)

            if match:
                start, end = match.span()

                # Extract subject (before predicate)
                subject_part = text_clean[:start].strip()

                # Extract object (after predicate)
                object_part = text_clean[end:].strip()

                # Clean up articles and common words
                subject = self._clean_noun_phrase(subject_part)
                obj = self._clean_noun_phrase(object_part)

                if subject and obj:
                    triples.append(Triple(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        hints=hints,
                    ))
                    break  # Use first matching predicate

        # If no predicate found, treat as simple description
        if not triples and text_clean:
            # Create a simple "exists" triple
            triples.append(Triple(
                subject=self._clean_noun_phrase(text_clean),
                predicate="exists as",
                object="scene",
                hints=hints,
            ))

        return triples

    def _clean_noun_phrase(self, phrase: str) -> str:
        """Clean a noun phrase by removing articles and extra whitespace."""
        # Remove leading articles
        articles = ["a", "an", "the", "some", "any"]
        words = phrase.split()

        while words and words[0].lower() in articles:
            words = words[1:]

        # Remove trailing punctuation
        result = " ".join(words)
        result = re.sub(r"[,;:.!?]+$", "", result)

        return result.strip()

    def to_prompt(self, triples: list[Triple]) -> str:
        """Convert triples back to a natural language prompt.

        Combines multiple triples into a coherent description.
        """
        if not triples:
            return ""

        if len(triples) == 1:
            return triples[0].to_prompt()

        # Combine multiple triples
        parts = []
        for i, triple in enumerate(triples):
            prompt = triple.to_prompt()
            if i == 0:
                parts.append(prompt)
            else:
                # Add conjunction
                parts.append(f", {prompt}")

        return "".join(parts)

    def extract_components(self, triples: list[Triple]) -> dict[str, list[str]]:
        """Extract all semantic components from triples for matching.

        Returns:
            Dict with keys: subjects, predicates, objects, hints, all
        """
        result = {
            "subjects": [],
            "predicates": [],
            "objects": [],
            "hints": [],
            "all": [],
        }

        for triple in triples:
            result["subjects"].append(triple.subject)
            result["predicates"].append(triple.predicate)
            result["objects"].append(triple.object)
            result["hints"].extend(triple.hints)
            result["all"].extend(triple.components)

        # Deduplicate
        for key in result:
            result[key] = list(dict.fromkeys(result[key]))

        return result
