"""Data models for Word Family Pipeline v2.

Two-stage pipeline:
- Stage 1: MeaningExtractor → ExtractedMeaning
- Stage 2: CardGenerator → FinalCard
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SourceExample:
    """Example used to identify a meaning (may contain spoilers)."""

    index: int  # 1-based index in the examples list
    sentence_id: int  # Original sentence_id from tokens
    has_spoiler: bool = False
    spoiler_type: str | None = None  # "character_death", "plot_twist", etc.


@dataclass
class ExtractedMeaning:
    """Output of Stage 1: A single meaning identified from examples."""

    meaning_id: int
    definition_en: str
    part_of_speech: str  # "verb", "noun", "adj", "adv", "phrasal verb"
    is_phrasal: bool = False
    phrasal_form: str | None = None  # "go away", "look up", etc.
    source_examples: list[SourceExample] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Full output of Stage 1 for a lemma."""

    lemma: str
    all_sentence_ids: list[int]
    meanings: list[ExtractedMeaning]

    # Metadata
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CleanExample:
    """A clean example for the final card (no spoilers)."""

    sentence_id: int | None  # None if generated
    text: str
    source: Literal["book", "generated"]


@dataclass
class FinalCard:
    """Output of Stage 2: A complete flashcard ready for Anki."""

    card_id: str  # e.g., "go_1", "go_away_2"
    lemma: str  # Original lemma
    lemma_display: str  # Display name (may be phrasal: "go away")
    meaning_id: int

    definition_en: str
    definition_ru: str
    part_of_speech: str
    is_phrasal: bool = False
    phrasal_form: str | None = None

    # Traceability
    all_sentence_ids: list[int] = field(default_factory=list)
    source_sentence_ids: list[int] = field(default_factory=list)

    # Final examples
    clean_examples: list[CleanExample] = field(default_factory=list)

    # Metadata
    generation_cost_usd: float = 0.0

    @property
    def frequency(self) -> int:
        """Frequency of this specific meaning (for sorting cards)."""
        return len(self.source_sentence_ids)

    @property
    def lemma_frequency(self) -> int:
        """Frequency of the lemma overall (all meanings)."""
        return len(self.all_sentence_ids)


@dataclass
class GenerationResult:
    """Full output of Stage 2 for a lemma."""

    lemma: str
    cards: list[FinalCard]

    # Aggregated metadata
    total_meanings: int = 0
    total_cards: int = 0
    total_cost_usd: float = 0.0


@dataclass
class PipelineResult:
    """Complete result of the two-stage pipeline."""

    lemma: str

    # Stage 1 output
    extraction: ExtractionResult

    # Stage 2 output
    generation: GenerationResult

    # Combined stats
    @property
    def total_cost_usd(self) -> float:
        return self.extraction.cost_usd + self.generation.total_cost_usd

    @property
    def total_cards(self) -> int:
        return len(self.generation.cards)
