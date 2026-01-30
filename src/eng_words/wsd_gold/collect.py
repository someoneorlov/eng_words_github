"""Collection of examples for WSD Gold Dataset.

This module provides functions to extract gold examples from
annotated tokens DataFrames for WSD evaluation.
"""

from typing import Any

import pandas as pd
from nltk.corpus import wordnet as wn

from eng_words.wsd.wordnet_utils import map_spacy_pos_to_wordnet
from eng_words.wsd_gold.models import (
    Candidate,
    ExampleMetadata,
    GoldExample,
    TargetWord,
)


def build_example_id(source_id: str, sentence_id: int, token_position: int) -> str:
    """Build a unique example ID.

    Args:
        source_id: Book/document identifier
        sentence_id: Sentence number
        token_position: Token position in sentence

    Returns:
        Formatted example ID string
    """
    return f"book:{source_id}|sent:{sentence_id}|tok:{token_position}"


def calculate_char_span(sentence: str, surface: str, position: int) -> tuple[int, int]:
    """Calculate character span for a word in a sentence.

    Uses token position to find the word in the sentence.

    Args:
        sentence: The full sentence text
        surface: The surface form of the word
        position: Token position (0-indexed) in the sentence

    Returns:
        Tuple of (start, end) character indices
    """
    import re

    # Tokenize sentence into words with their positions
    # This pattern matches words and punctuation separately
    token_pattern = re.compile(r"\S+")
    tokens_with_spans = [(m.group(), m.start(), m.end()) for m in token_pattern.finditer(sentence)]

    # The position refers to the Nth token in the sentence
    # We need to find which token matches our surface form at that position
    if position < len(tokens_with_spans):
        token_text, start, end = tokens_with_spans[position]
        # Check if this token contains our surface (handle punctuation)
        if surface in token_text:
            # Find exact position of surface within the token
            offset = token_text.find(surface)
            return (start + offset, start + offset + len(surface))

    # Fallback: find the Nth occurrence of surface in sentence
    # counting only word-boundary matches
    current_pos = 0
    occurrence = 0

    while current_pos < len(sentence):
        idx = sentence.find(surface, current_pos)
        if idx == -1:
            break

        # Check word boundaries
        before_ok = idx == 0 or not sentence[idx - 1].isalnum()
        after_end = idx + len(surface)
        after_ok = after_end >= len(sentence) or not sentence[after_end].isalnum()

        if before_ok and after_ok:
            if occurrence == position:
                return (idx, idx + len(surface))
            occurrence += 1

        current_pos = idx + 1

    # Last fallback: just find the surface form
    idx = sentence.find(surface)
    if idx >= 0:
        return (idx, idx + len(surface))

    return (0, len(surface))


def get_candidates_for_lemma(lemma: str, pos: str) -> list[Candidate]:
    """Get candidate synsets for a lemma from WordNet.

    Args:
        lemma: The word lemma
        pos: Part of speech (NOUN, VERB, ADJ, ADV)

    Returns:
        List of Candidate objects with synset info
    """
    wn_pos = map_spacy_pos_to_wordnet(pos)
    if wn_pos is None:
        return []

    # Get synsets from WordNet
    synsets = wn.synsets(lemma, pos=wn_pos)
    if not synsets:
        # Try with underscore replacement for multi-word
        synsets = wn.synsets(lemma.replace(" ", "_"), pos=wn_pos)

    candidates = []
    for synset in synsets:
        candidate = Candidate(
            synset_id=synset.name(),
            gloss=synset.definition() or "",
            examples=synset.examples() or [],
        )
        candidates.append(candidate)

    return candidates


def assign_buckets(source_id: str, source_metadata: dict[str, dict[str, str]]) -> dict[str, str]:
    """Assign source/year/genre buckets based on metadata.

    Args:
        source_id: Book/document identifier
        source_metadata: Dictionary mapping source_id to bucket info

    Returns:
        Dictionary with source_bucket, year_bucket, genre_bucket
    """
    default_buckets = {
        "source_bucket": "unknown",
        "year_bucket": "unknown",
        "genre_bucket": "unknown",
    }

    if source_id not in source_metadata:
        return default_buckets

    meta = source_metadata[source_id]
    return {
        "source_bucket": meta.get("source_bucket", "unknown"),
        "year_bucket": meta.get("year_bucket", "unknown"),
        "genre_bucket": meta.get("genre_bucket", "unknown"),
    }


def _calculate_baseline_margin(
    tokens_df: pd.DataFrame,
    sentence_id: int,
    lemma: str,
    pos: str,
    assigned_synset: str,
) -> float:
    """Calculate baseline margin for WSD prediction.

    The margin is 1 - (second_best_score / best_score) if we had scores,
    but since we don't have scores in tokens_df, we use a heuristic
    based on sense count.

    Args:
        tokens_df: Tokens DataFrame
        sentence_id: Sentence ID
        lemma: Word lemma
        pos: Part of speech
        assigned_synset: The synset that was assigned

    Returns:
        Estimated margin (0-1), higher means more confident
    """
    # Get all candidates for this lemma
    candidates = get_candidates_for_lemma(lemma, pos)
    if len(candidates) <= 1:
        return 1.0  # Only one option, maximum confidence

    # Heuristic: fewer candidates = higher margin
    # This is a rough approximation
    margin = 1.0 / len(candidates)
    return min(margin * 2, 0.5)  # Cap at 0.5 to indicate uncertainty


def _is_multiword(lemma: str) -> bool:
    """Check if lemma is a multi-word expression."""
    return "_" in lemma or " " in lemma


def extract_examples_from_tokens(
    tokens_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    min_sense_count: int = 1,
    pos_filter: list[str] | None = None,
    source_metadata: dict[str, Any] | None = None,
) -> list[GoldExample]:
    """Extract gold examples from annotated tokens DataFrame.

    Args:
        tokens_df: Annotated tokens DataFrame with synset_id column
        sentences_df: Sentences DataFrame with sentence text
        min_sense_count: Minimum number of WordNet senses to include
        pos_filter: List of POS tags to include (None = all)
        source_metadata: Optional metadata for bucket assignment

    Returns:
        List of GoldExample objects
    """
    if tokens_df.empty:
        return []

    source_metadata = source_metadata or {}

    # Create sentence lookup
    sentence_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))

    examples = []

    for _, row in tokens_df.iterrows():
        # Skip tokens without synset annotation
        synset_id = row.get("synset_id")
        if pd.isna(synset_id) or synset_id is None:
            continue

        pos = row["pos"]
        lemma = row["lemma"]
        surface = row["surface"]
        sentence_id = row["sentence_id"]
        position = row["position"]
        book = row.get("book", "unknown")

        # Apply POS filter
        if pos_filter is not None and pos not in pos_filter:
            continue

        # Get sentence text
        sentence = sentence_lookup.get(sentence_id, "")
        if not sentence:
            continue

        # Get candidates from WordNet
        candidates = get_candidates_for_lemma(lemma, pos)
        if not candidates:
            continue

        # Apply min_sense_count filter
        if len(candidates) < min_sense_count:
            continue

        # Calculate character span
        char_span = calculate_char_span(sentence, surface, position)

        # Split sentence into left/right context
        text_left = sentence[: char_span[0]]
        text_right = sentence[char_span[1] :]

        # Build example ID
        example_id = build_example_id(book, sentence_id, position)

        # Assign buckets
        buckets = assign_buckets(book, source_metadata)

        # Calculate metadata
        baseline_margin = _calculate_baseline_margin(tokens_df, sentence_id, lemma, pos, synset_id)

        # Create target word
        target = TargetWord(
            surface=surface,
            lemma=lemma,
            pos=pos,
            char_span=char_span,
        )

        # Create metadata
        metadata = ExampleMetadata(
            wn_sense_count=len(candidates),
            baseline_top1=synset_id,
            baseline_margin=baseline_margin,
            is_multiword=_is_multiword(lemma),
        )

        # Create example
        example = GoldExample(
            example_id=example_id,
            source_id=book,
            source_bucket=buckets["source_bucket"],
            year_bucket=buckets["year_bucket"],
            genre_bucket=buckets["genre_bucket"],
            text_left=text_left,
            target=target,
            text_right=text_right,
            context_window=sentence,
            candidates=candidates,
            metadata=metadata,
        )

        examples.append(example)

    return examples
