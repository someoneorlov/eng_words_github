"""Data models for WSD Gold Dataset.

This module defines dataclasses for:
- GoldExample: one example in the gold dataset (target word + context + candidates)
- ModelOutput: LLM judge response
- GoldLabel: aggregated gold label from multiple judges
"""

from dataclasses import dataclass, field
from typing import Any

# Valid flags that LLM judges can return
VALID_FLAGS: frozenset[str] = frozenset(
    {
        "needs_more_context",
        "multiword",
        "metaphor",
        "none_of_the_above",
    }
)


@dataclass
class TargetWord:
    """Target word in a gold example.

    Attributes:
        surface: The actual word form as it appears in text (e.g., "running")
        lemma: The base form of the word (e.g., "run")
        pos: Part of speech tag (NOUN, VERB, ADJ, ADV)
        char_span: Character offsets (start, end) in context_window
    """

    surface: str
    lemma: str
    pos: str
    char_span: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "surface": self.surface,
            "lemma": self.lemma,
            "pos": self.pos,
            "char_span": list(self.char_span),  # JSON doesn't have tuples
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TargetWord":
        """Deserialize from dictionary."""
        return cls(
            surface=d["surface"],
            lemma=d["lemma"],
            pos=d["pos"],
            char_span=tuple(d["char_span"]),
        )


@dataclass
class Candidate:
    """A candidate synset for disambiguation.

    Attributes:
        synset_id: WordNet synset identifier (e.g., "bank.n.01")
        gloss: Definition of the synset
        examples: Example sentences from WordNet
    """

    synset_id: str
    gloss: str
    examples: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "synset_id": self.synset_id,
            "gloss": self.gloss,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Candidate":
        """Deserialize from dictionary."""
        return cls(
            synset_id=d["synset_id"],
            gloss=d["gloss"],
            examples=d.get("examples", []),
        )


@dataclass
class ExampleMetadata:
    """Metadata about a gold example for stratification.

    Attributes:
        wn_sense_count: Total number of senses in WordNet for this lemma+POS
        baseline_top1: The synset_id that baseline WSD predicted
        baseline_margin: Confidence margin of baseline prediction (0-1)
        is_multiword: Whether this is a multiword expression
    """

    wn_sense_count: int
    baseline_top1: str
    baseline_margin: float
    is_multiword: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "wn_sense_count": self.wn_sense_count,
            "baseline_top1": self.baseline_top1,
            "baseline_margin": self.baseline_margin,
            "is_multiword": self.is_multiword,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExampleMetadata":
        """Deserialize from dictionary."""
        return cls(
            wn_sense_count=d["wn_sense_count"],
            baseline_top1=d["baseline_top1"],
            baseline_margin=d["baseline_margin"],
            is_multiword=d["is_multiword"],
        )


@dataclass
class GoldExample:
    """One example in the gold dataset.

    Represents a single target word occurrence with its context and
    candidate synsets for disambiguation.

    Attributes:
        example_id: Unique identifier (format: "book:X|ch:Y|sent:Z|tok:W")
        source_id: Book/document identifier
        source_bucket: Category of source (classic_fiction, modern_nonfiction, etc.)
        year_bucket: Time period (pre_1950, 1950_2000, post_2000)
        genre_bucket: Genre category (fiction, nonfiction)
        text_left: Context before target word
        target: The target word with metadata
        text_right: Context after target word
        context_window: Full sentence (optionally with neighbors)
        candidates: List of candidate synsets
        metadata: Additional metadata for stratification
    """

    example_id: str
    source_id: str
    source_bucket: str
    year_bucket: str
    genre_bucket: str
    text_left: str
    target: TargetWord
    text_right: str
    context_window: str
    candidates: list[Candidate]
    metadata: ExampleMetadata

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "example_id": self.example_id,
            "source_id": self.source_id,
            "source_bucket": self.source_bucket,
            "year_bucket": self.year_bucket,
            "genre_bucket": self.genre_bucket,
            "text_left": self.text_left,
            "target": self.target.to_dict(),
            "text_right": self.text_right,
            "context_window": self.context_window,
            "candidates": [c.to_dict() for c in self.candidates],
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldExample":
        """Deserialize from dictionary."""
        return cls(
            example_id=d["example_id"],
            source_id=d["source_id"],
            source_bucket=d["source_bucket"],
            year_bucket=d["year_bucket"],
            genre_bucket=d["genre_bucket"],
            text_left=d["text_left"],
            target=TargetWord.from_dict(d["target"]),
            text_right=d["text_right"],
            context_window=d["context_window"],
            candidates=[Candidate.from_dict(c) for c in d["candidates"]],
            metadata=ExampleMetadata.from_dict(d["metadata"]),
        )

    def get_candidate_ids(self) -> list[str]:
        """Get list of candidate synset IDs."""
        return [c.synset_id for c in self.candidates]


@dataclass
class LLMUsage:
    """Token usage and cost information for an LLM call.

    Attributes:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (for cost calculation)
        cost_usd: Estimated cost in USD
    """

    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LLMUsage":
        """Deserialize from dictionary."""
        return cls(
            input_tokens=d["input_tokens"],
            output_tokens=d["output_tokens"],
            cached_tokens=d.get("cached_tokens", 0),
            cost_usd=d.get("cost_usd", 0.0),
        )


@dataclass
class ModelOutput:
    """Output from an LLM judge for one example.

    Attributes:
        chosen_synset_id: The synset ID chosen by the LLM
        confidence: LLM's confidence in the choice (0-1)
        flags: List of flags (needs_more_context, multiword, etc.)
        raw_text: Raw LLM response text (for debugging)
        usage: Token usage information
    """

    chosen_synset_id: str
    confidence: float
    flags: list[str]
    raw_text: str
    usage: LLMUsage

    def is_valid(self, candidates: list[str]) -> bool:
        """Check if the output is valid.

        Args:
            candidates: List of valid synset IDs

        Returns:
            True if chosen_synset_id is in candidates or none_of_the_above flag is set
        """
        # none_of_the_above is a valid response even with empty synset_id
        if "none_of_the_above" in self.flags:
            return True
        return self.chosen_synset_id in candidates

    def needs_referee(self, confidence_threshold: float = 0.6) -> bool:
        """Check if this output needs a referee (third judge).

        Args:
            confidence_threshold: Minimum confidence to not need referee

        Returns:
            True if confidence is low or problematic flags are present
        """
        # Flags that indicate uncertainty
        problematic_flags = {"needs_more_context", "none_of_the_above"}
        if any(f in problematic_flags for f in self.flags):
            return True

        # Low confidence
        if self.confidence < confidence_threshold:
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "chosen_synset_id": self.chosen_synset_id,
            "confidence": self.confidence,
            "flags": self.flags,
            "raw_text": self.raw_text,
            "usage": self.usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelOutput":
        """Deserialize from dictionary."""
        return cls(
            chosen_synset_id=d["chosen_synset_id"],
            confidence=d["confidence"],
            flags=d.get("flags", []),
            raw_text=d.get("raw_text", ""),
            usage=LLMUsage.from_dict(d["usage"]),
        )


@dataclass
class GoldLabel:
    """Aggregated gold label from multiple LLM judges.

    Attributes:
        synset_id: The final chosen synset ID (majority vote)
        confidence: Aggregated confidence
        agreement_ratio: Ratio of judges that agreed (0-1)
        flags: Aggregated flags from all judges
        needs_referee: Whether a third judge was needed
        judge_count: Number of judges used
    """

    synset_id: str
    confidence: float
    agreement_ratio: float
    flags: list[str] = field(default_factory=list)
    needs_referee: bool = False
    judge_count: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "synset_id": self.synset_id,
            "confidence": self.confidence,
            "agreement_ratio": self.agreement_ratio,
            "flags": self.flags,
            "needs_referee": self.needs_referee,
            "judge_count": self.judge_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GoldLabel":
        """Deserialize from dictionary."""
        return cls(
            synset_id=d["synset_id"],
            confidence=d["confidence"],
            agreement_ratio=d["agreement_ratio"],
            flags=d.get("flags", []),
            needs_referee=d.get("needs_referee", False),
            judge_count=d.get("judge_count", 2),
        )
