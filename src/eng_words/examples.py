"""Utilities for attaching example sentences to lemmas and phrasal verbs."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .constants import (
    EXAMPLE,
    EXAMPLE_FALLBACK_MAX,
    EXAMPLE_FALLBACK_MIN,
    EXAMPLE_MAX_LENGTH,
    EXAMPLE_MIN_LENGTH,
    EXAMPLES_PER_ITEM_DEFAULT,
    LEMMA,
    PHRASAL,
    SENTENCE,
    SENTENCE_COLUMNS,
    SENTENCE_ID,
    TOP_N_DEFAULT,
)


def _validate_dataframe(df: pd.DataFrame, required_columns: Iterable[str], name: str) -> None:
    if df is None:
        raise ValueError(f"{name} must be provided")
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _normalize_quote_spacing(sentence: str) -> str:
    """Normalize spacing after opening quotes in English text.

    Removes space after opening quote if followed by a letter or digit.
    Example: '" That' -> '"That', but '" " (quote space quote) is preserved.
    Also removes spaces before closing quotes: 'word "' -> 'word"'.
    Handles straight and smart quotes.
    """
    if not sentence:
        return sentence

    normalized = sentence

    # Remove space after opening quotes (straight and smart)
    # " X -> "X,  “ X -> “X
    normalized = re.sub(r'"\s+([A-Za-z0-9])', r'"\1', normalized)
    normalized = re.sub(r"“\s+([A-Za-z0-9])", r"“\1", normalized)

    # Remove space before closing quotes (straight and smart)
    # X " -> X",  X ” -> X”
    normalized = re.sub(r'([A-Za-z0-9.,;:!?])\s+"', r'\1"', normalized)
    normalized = re.sub(r"([A-Za-z0-9.,;:!?])\s+”", r"\1”", normalized)

    # If quotes are unbalanced, trim stray leading/trailing double quotes.
    # Only target double quotes (straight or smart) to avoid apostrophes in contractions.
    def trim_unbalanced_quotes(text: str) -> str:
        # Count occurrences of double quotes (straight and smart)
        straight = text.count('"')
        left_smart = text.count("“")
        right_smart = text.count("”")
        # If odd number of total double-quote markers, consider trimming edge quotes
        total_markers = straight + left_smart + right_smart
        if total_markers % 2 == 1:
            # Trim leading quote if present and followed by non-space
            text = re.sub(r'^\s*["“]\s*', "", text)
            # Trim trailing quote if present and preceded by non-space
            text = re.sub(r'\s*["”]\s*$', "", text)
        return text

    normalized = trim_unbalanced_quotes(normalized)
    return normalized


def _build_sentence_lookup(sentences_df: pd.DataFrame) -> Dict[int, str]:
    sentences_dict = sentences_df.set_index(SENTENCE_ID)[SENTENCE].to_dict()
    # Normalize quote spacing in all sentences
    return {sid: _normalize_quote_spacing(sent) for sid, sent in sentences_dict.items()}


def _select_optimal_sentence(
    sentence_ids: List[int],
    sentence_lookup: Dict[int, str],
    *,
    min_length: int = EXAMPLE_MIN_LENGTH,
    max_length: int = EXAMPLE_MAX_LENGTH,
    fallback_min: int = EXAMPLE_FALLBACK_MIN,
    fallback_max: int = EXAMPLE_FALLBACK_MAX,
) -> Optional[str]:
    """Select sentence with optimal length (preferred: 50-150 chars).

    Args:
        sentence_ids: List of sentence IDs to choose from.
        sentence_lookup: Dictionary mapping sentence_id to sentence text.
        min_length: Preferred minimum length.
        max_length: Preferred maximum length.
        fallback_min: Minimum acceptable length if no sentence in preferred range.
        fallback_max: Maximum acceptable length if no sentence in preferred range.

    Returns:
        Selected sentence text, or None if no valid sentences found.
    """
    candidates = []
    for sentence_id in sentence_ids:
        sentence = sentence_lookup.get(sentence_id)
        if not sentence:
            continue
        length = len(sentence)
        candidates.append((sentence, length))

    if not candidates:
        return None

    # First, try to find sentence in preferred range
    in_range = [(s, length) for s, length in candidates if min_length <= length <= max_length]
    if in_range:
        # If multiple in range, prefer median length
        in_range.sort(key=lambda x: x[1])
        median_idx = len(in_range) // 2
        return in_range[median_idx][0]

    # If no sentence in preferred range, find closest to preferred range
    # Prefer sentences within fallback range
    in_fallback = [
        (s, length) for s, length in candidates if fallback_min <= length <= fallback_max
    ]
    if in_fallback:
        # Choose closest to preferred range center
        center = (min_length + max_length) / 2
        in_fallback.sort(key=lambda x: abs(x[1] - center))
        return in_fallback[0][0]

    # If all sentences are outside fallback range, return median
    candidates.sort(key=lambda x: x[1])
    median_idx = len(candidates) // 2
    return candidates[median_idx][0]


def _select_top_k_sentences(
    sentence_ids: List[int],
    sentence_lookup: Dict[int, str],
    *,
    k: int,
    min_length: int = EXAMPLE_MIN_LENGTH,
    max_length: int = EXAMPLE_MAX_LENGTH,
    fallback_min: int = EXAMPLE_FALLBACK_MIN,
    fallback_max: int = EXAMPLE_FALLBACK_MAX,
) -> List[str]:
    """Select up to K sentences closest to preferred length window.

    Strategy:
    1) Prefer sentences in [min_length, max_length], sorted by closeness to center.
    2) If not enough, take from [fallback_min, fallback_max] by closeness.
    3) If still not enough, take globally by closeness to center.
    """
    if k <= 0:
        return []

    scored: List[tuple[str, int, float]] = []
    center = (min_length + max_length) / 2
    for sentence_id in sentence_ids:
        sentence = sentence_lookup.get(sentence_id)
        if not sentence:
            continue
        length = len(sentence)
        distance = abs(length - center)
        scored.append((sentence, length, distance))

    if not scored:
        return []

    def pick_within(
        lo: int, hi: int, remaining: int, pool: List[tuple[str, int, float]]
    ) -> List[str]:
        if remaining <= 0:
            return []
        in_range = [(s, length, d) for (s, length, d) in pool if lo <= length <= hi]
        in_range.sort(key=lambda x: (x[2], abs(x[1] - center)))
        return [s for (s, _, _) in in_range[:remaining]]

    picked: List[str] = []
    # 1) Preferred window
    picked += pick_within(min_length, max_length, k - len(picked), scored)
    # 2) Fallback window
    if len(picked) < k:
        picked += pick_within(fallback_min, fallback_max, k - len(picked), scored)
    # 3) Global by closeness
    if len(picked) < k:
        remaining_pool = [(s, length, d) for (s, length, d) in scored if s not in set(picked)]
        remaining_pool.sort(key=lambda x: (x[2], abs(x[1] - center)))
        picked += [s for (s, _, _) in remaining_pool[: k - len(picked)]]

    # Deduplicate while preserving order (in case ids map to same text)
    seen = set()
    unique = []
    for s in picked:
        if s not in seen:
            unique.append(s)
            seen.add(s)
    return unique


def get_examples_for_lemmas(
    candidates_df: pd.DataFrame,
    tokens_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    *,
    top_n: int = TOP_N_DEFAULT,
    examples_per_item: int = EXAMPLES_PER_ITEM_DEFAULT,
    joiner: str = " <br><br> ",
) -> pd.DataFrame:
    """Attach up to N optimal-length example sentences for each lemma."""

    _validate_dataframe(candidates_df, {LEMMA}, "candidates_df")
    _validate_dataframe(tokens_df, {LEMMA, SENTENCE_ID}, "tokens_df")
    _validate_dataframe(sentences_df, SENTENCE_COLUMNS, "sentences_df")

    if top_n is not None and top_n > 0:
        target_candidates = candidates_df.head(top_n)
    else:
        target_candidates = candidates_df

    sentence_lookup = _build_sentence_lookup(sentences_df)
    lemma_sentences = (
        tokens_df.groupby(LEMMA)[SENTENCE_ID].agg(lambda ids: list(dict.fromkeys(ids))).to_dict()
    )

    examples: List[Optional[str]] = []
    for lemma in target_candidates[LEMMA]:
        sentence_ids = lemma_sentences.get(lemma, [])
        if sentence_ids:
            picked = _select_top_k_sentences(
                sentence_ids,
                sentence_lookup,
                k=max(1, examples_per_item),
            )
            examples.append(joiner.join(picked) if picked else None)
        else:
            examples.append(None)

    result = candidates_df.copy()
    result.loc[target_candidates.index, EXAMPLE] = examples
    if EXAMPLE not in result.columns:
        result[EXAMPLE] = None

    return result


def get_examples_for_phrasal_verbs(
    phrasal_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    *,
    examples_per_item: int = EXAMPLES_PER_ITEM_DEFAULT,
    joiner: str = " <br><br> ",
) -> pd.DataFrame:
    """Return dataframe mapping phrasal verbs to up to N optimal-length example sentences."""

    _validate_dataframe(phrasal_df, {PHRASAL, SENTENCE_ID}, "phrasal_df")
    _validate_dataframe(sentences_df, SENTENCE_COLUMNS, "sentences_df")

    sentence_lookup = _build_sentence_lookup(sentences_df)

    def optimal_sentences(sentence_ids: List[int]) -> Optional[str]:
        picked = _select_top_k_sentences(
            sentence_ids,
            sentence_lookup,
            k=max(1, examples_per_item),
        )
        return joiner.join(picked) if picked else None

    grouped = (
        phrasal_df.groupby(PHRASAL)[SENTENCE_ID]
        .agg(lambda ids: list(dict.fromkeys(ids)))
        .reset_index()
    )
    grouped[EXAMPLE] = grouped[SENTENCE_ID].apply(optimal_sentences)
    grouped = grouped.drop(columns=[SENTENCE_ID])
    return grouped
