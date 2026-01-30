"""Base classes and types for Word Sense Disambiguation.

This module defines the abstract interface for sense disambiguation backends,
allowing different implementations (WordNet-based, clustering-based, etc.)
to be used interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class SenseAnnotation:
    """Result of word sense disambiguation for a single token.

    Attributes:
        lemma: The lemmatized form of the word
        sense_id: Unique identifier for the sense (e.g., "run.v.01" for WordNet,
                  "cluster_17" for clustering-based approaches)
        sense_label: Human-readable label for the sense category
                     (e.g., "verb.motion" for supersenses)
        confidence: Model confidence score (0.0-1.0)
        definition: Optional definition or description of the sense
        construction_tag: Optional tag if a grammatical construction was detected
                          (e.g., "BE_GOING_TO" for future tense)
    """

    lemma: str
    sense_id: str | None
    sense_label: str
    confidence: float
    definition: str | None = None
    construction_tag: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame operations."""
        return {
            "lemma": self.lemma,
            "synset_id": self.sense_id,
            "supersense": self.sense_label,
            "sense_confidence": self.confidence,
            "definition": self.definition,
            "construction_tag": self.construction_tag,
        }


class SenseBackend(ABC):
    """Abstract base class for sense disambiguation backends.

    This interface allows different WSD approaches to be used interchangeably:
    - WordNetSenseBackend: Uses Sentence-Transformers + WordNet definitions
    - ClusterSenseBackend: Uses embedding clustering (future)
    - FinetunedWSDBackend: Uses fine-tuned classifier (future)

    Usage:
        backend = WordNetSenseBackend()
        annotated_tokens = backend.annotate(tokens_df, sentences_df)
        stats = backend.aggregate(annotated_tokens)
    """

    @abstractmethod
    def annotate(
        self,
        tokens_df: pd.DataFrame,
        sentences_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add sense annotations to tokens DataFrame.

        This method processes each content word (NOUN, VERB, ADJ, ADV) in the
        tokens DataFrame and adds sense disambiguation columns.

        Args:
            tokens_df: DataFrame with token information. Required columns:
                - lemma: The lemmatized word
                - sentence_id: ID linking to sentences_df
                - pos: Part of speech tag
            sentences_df: DataFrame with sentence texts. Required columns:
                - sentence_id: Unique sentence identifier
                - sentence: The full sentence text

        Returns:
            DataFrame with original columns plus:
                - synset_id: The disambiguated sense ID (or None)
                - supersense: The supersense category (e.g., "verb.motion")
                - sense_confidence: Confidence score (0.0-1.0)

        Raises:
            ValueError: If required columns are missing
        """
        pass

    @abstractmethod
    def aggregate(
        self,
        annotated_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Aggregate statistics by (lemma, supersense).

        This method groups the annotated tokens by lemma and supersense,
        calculating frequency statistics for each sense.

        Args:
            annotated_df: DataFrame with sense annotations (output of annotate())
                Required columns:
                - lemma: The lemmatized word
                - supersense: The supersense category
                - sentence_id: For counting unique sentences

        Returns:
            DataFrame with columns:
                - lemma: The lemmatized word
                - supersense: The supersense category
                - book_freq: Total frequency of the lemma in the book
                - sense_freq: Frequency of this specific sense
                - sense_ratio: sense_freq / book_freq
                - doc_count: Number of unique sentences with this sense
                - sense_count: Number of different senses for this lemma
                - dominant_supersense: Most frequent supersense for this lemma

        Raises:
            ValueError: If required columns are missing
        """
        pass

    def disambiguate_word(
        self,
        sentence: str,
        lemma: str,
        pos: str | None = None,
    ) -> SenseAnnotation:
        """Disambiguate a single word in context.

        This is a convenience method for disambiguating individual words.
        For batch processing, use annotate() instead.

        Args:
            sentence: The sentence containing the target word
            lemma: The lemmatized form of the target word
            pos: Optional part of speech filter ('n', 'v', 'a', 'r')

        Returns:
            SenseAnnotation with the disambiguated sense

        Note:
            Default implementation raises NotImplementedError.
            Subclasses should override if they support single-word disambiguation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support single-word disambiguation. "
            "Use annotate() for batch processing."
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend for logging/display."""
        pass

    @property
    def supports_confidence(self) -> bool:
        """Whether this backend provides meaningful confidence scores."""
        return True

    @property
    def supports_definitions(self) -> bool:
        """Whether this backend provides sense definitions."""
        return True


def validate_tokens_df(df: pd.DataFrame) -> None:
    """Validate that tokens DataFrame has required columns.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required = {"lemma", "sentence_id", "pos"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"tokens_df missing required columns: {missing}")


def validate_sentences_df(df: pd.DataFrame) -> None:
    """Validate that sentences DataFrame has required columns.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required = {"sentence_id", "sentence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sentences_df missing required columns: {missing}")


def validate_annotated_df(df: pd.DataFrame) -> None:
    """Validate that annotated DataFrame has required columns for aggregation.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required = {"lemma", "supersense", "sentence_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"annotated_df missing required columns: {missing}")
