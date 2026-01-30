"""Lemma frequency statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from wordfreq import zipf_frequency

from .constants import (
    BOOK_FREQ,
    DOC_COUNT,
    GLOBAL_ZIPF,
    IS_ALPHA,
    IS_STOP,
    LANGUAGE_EN,
    LEMMA,
    OTHER_POS_COUNT,
    POS,
    POS_PROPN,
    POS_VERB,
    REQUIRED_TOKEN_COLUMNS,
    SENTENCE_ID,
    STOPWORD_COUNT,
    VERB_COUNT,
)


def calculate_lemma_frequency(
    tokens_df: pd.DataFrame,
    *,
    include_stopwords: bool = False,
    include_proper_nouns: bool = False,
    require_alpha: bool = True,
) -> pd.DataFrame:
    """Aggregate lemma statistics with configurable filtering."""

    if tokens_df is None:
        raise ValueError("tokens_df must not be None")

    missing = [col for col in REQUIRED_TOKEN_COLUMNS if col not in tokens_df.columns]
    if missing:
        raise ValueError(f"tokens_df missing columns: {missing}")

    mask = pd.Series(True, index=tokens_df.index, dtype=bool)
    if not include_stopwords:
        mask &= ~tokens_df[IS_STOP].astype(bool)
    if not include_proper_nouns:
        mask &= tokens_df[POS] != POS_PROPN
    if require_alpha:
        mask &= tokens_df[IS_ALPHA].astype(bool)
    filtered = tokens_df.loc[mask, [LEMMA, SENTENCE_ID, POS, IS_STOP]]

    if filtered.empty:
        return pd.DataFrame(
            columns=[LEMMA, BOOK_FREQ, DOC_COUNT, VERB_COUNT, OTHER_POS_COUNT, STOPWORD_COUNT]
        )

    book_freq: Counter[str] = Counter()
    doc_ids: Dict[str, Set[int]] = defaultdict(set)
    pos_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    stopword_counts: Dict[str, int] = defaultdict(int)

    for lemma, sentence_id, pos, is_stop in filtered.itertuples(index=False, name=None):
        book_freq[lemma] += 1
        doc_ids[lemma].add(int(sentence_id))
        pos_counts[lemma][pos] += 1
        stopword_counts[lemma] += int(is_stop)

    rows = [
        {
            LEMMA: lemma,
            BOOK_FREQ: freq,
            DOC_COUNT: len(doc_ids[lemma]),
            VERB_COUNT: pos_counts[lemma].get(POS_VERB, 0),
            OTHER_POS_COUNT: freq - pos_counts[lemma].get(POS_VERB, 0),
            STOPWORD_COUNT: stopword_counts[lemma],
        }
        for lemma, freq in book_freq.items()
    ]

    return pd.DataFrame(rows).sort_values(BOOK_FREQ, ascending=False).reset_index(drop=True)


def add_global_frequency(lemma_df: pd.DataFrame, language: str = LANGUAGE_EN) -> pd.DataFrame:
    """Append global Zipf frequency using wordfreq."""

    if lemma_df is None:
        raise ValueError("lemma_df must not be None")
    if LEMMA not in lemma_df.columns:
        raise ValueError(f"lemma_df must contain '{LEMMA}' column")

    def _zipf_safe(lemma: str) -> float:
        try:
            return float(zipf_frequency(lemma, language))
        except Exception:  # noqa: BLE001
            return 0.0

    lemma_df = lemma_df.copy()
    lemma_df[GLOBAL_ZIPF] = lemma_df[LEMMA].astype(str).apply(_zipf_safe)
    return lemma_df


def save_lemma_stats_to_parquet(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Persist lemma statistics to parquet."""

    if stats_df is None:
        raise ValueError("stats_df must not be None")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_parquet(output_path, index=False)


def load_lemma_stats_from_parquet(file_path: Path) -> pd.DataFrame:
    """Load lemma stats from parquet."""

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    return pd.read_parquet(file_path)
