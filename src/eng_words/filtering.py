"""Known words filtering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .constants import (
    BOOK_FREQ,
    DIALECT_LEMMAS_FILTER,
    EXCLUDE_DEFAULT,
    GLOBAL_ZIPF,
    LEMMA,
    MAX_ZIPF_DEFAULT,
    MIN_BOOK_FREQ_DEFAULT,
    MIN_ZIPF_DEFAULT,
    SCORE,
    SENSE_COUNT,
    SENSE_FREQ,
    STATUS,
    SUPERSENSE,
    TARGET_ZIPF_DEFAULT,
)
from .storage import load_known_words


def load_known_words_from_csv(csv_path: Path) -> pd.DataFrame:
    """Load known words metadata from CSV.

    Deprecated: Use load_known_words() instead for support of multiple backends.
    This function is kept for backward compatibility.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with known words metadata.
    """
    return load_known_words(csv_path)


def filter_by_status(
    candidates_df: pd.DataFrame,
    known_df: pd.DataFrame,
    *,
    exclude_statuses: List[str] | None = None,
) -> pd.DataFrame:
    """Filter candidates by known status."""

    if candidates_df is None or known_df is None:
        raise ValueError("candidates_df and known_df must be provided")

    exclude = set(exclude_statuses or EXCLUDE_DEFAULT)

    merged = candidates_df.merge(
        known_df[[LEMMA, STATUS]],
        on=LEMMA,
        how="left",
        suffixes=("", "_known"),
    )

    status_col = f"{STATUS}_known" if f"{STATUS}_known" in merged.columns else STATUS
    mask = ~merged[status_col].astype(str).str.lower().isin(exclude)
    filtered = merged.loc[mask].drop(
        columns=[col for col in merged.columns if col.endswith("_known")]
    )
    return filtered.reset_index(drop=True)


def filter_known_words(candidates_df: pd.DataFrame, known_df: pd.DataFrame) -> pd.DataFrame:
    """Filter out known/ignored lemmas using default statuses."""

    return filter_by_status(candidates_df, known_df, exclude_statuses=list(EXCLUDE_DEFAULT))


def filter_dialect_lemmas(
    df: pd.DataFrame,
    *,
    lemma_column: str = LEMMA,
    dialect_lemmas: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Filter out dialect/tokenization error lemmas.

    Some books contain dialect speech (e.g., "ain't" → "ai", "de" for "the").
    spaCy may incorrectly tokenize/lemmatize these, creating invalid lemmas.
    This filter removes such lemmas from the DataFrame.

    Args:
        df: DataFrame containing lemmas to filter.
        lemma_column: Name of the lemma column (default: "lemma").
        dialect_lemmas: Set of lemmas to filter out. If None, uses DIALECT_LEMMAS_FILTER.

    Returns:
        DataFrame with dialect lemmas removed.

    Example:
        >>> df = pd.DataFrame({"lemma": ["run", "ai", "de", "walk"]})
        >>> filtered = filter_dialect_lemmas(df)
        >>> filtered["lemma"].tolist()
        ['run', 'walk']
    """
    if df is None or df.empty:
        return df

    if lemma_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{lemma_column}' column")

    filter_set = dialect_lemmas if dialect_lemmas is not None else DIALECT_LEMMAS_FILTER

    # Filter out rows where lemma is in the dialect set
    mask = ~df[lemma_column].str.lower().isin(filter_set)
    filtered = df[mask].reset_index(drop=True)

    removed_count = len(df) - len(filtered)
    if removed_count > 0:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Filtered out {removed_count} rows with dialect lemmas")

    return filtered


def filter_by_frequency(
    stats_df: pd.DataFrame,
    *,
    min_book_freq: int = MIN_BOOK_FREQ_DEFAULT,
    min_zipf: float | None = MIN_ZIPF_DEFAULT,
    max_zipf: float | None = MAX_ZIPF_DEFAULT,
) -> pd.DataFrame:
    """Filter lemma stats by local/global frequency thresholds."""

    if stats_df is None:
        raise ValueError("stats_df must be provided")

    required = {BOOK_FREQ, GLOBAL_ZIPF}
    missing = required - set(stats_df.columns)
    if missing:
        raise ValueError(f"stats_df is missing required columns: {missing}")

    df = stats_df.copy()

    if min_book_freq is not None:
        df = df[df[BOOK_FREQ] >= int(min_book_freq)]

    zipf_series = pd.to_numeric(df[GLOBAL_ZIPF], errors="coerce")
    if min_zipf is not None:
        df = df[zipf_series >= float(min_zipf)]
        zipf_series = pd.to_numeric(df[GLOBAL_ZIPF], errors="coerce")
    if max_zipf is not None:
        df = df[zipf_series <= float(max_zipf)]

    return df.reset_index(drop=True)


def calculate_rarity_score(
    zipf_freq: float | None, target_zipf: float = TARGET_ZIPF_DEFAULT
) -> float:
    """Compute rarity boost centered around target Zipf frequency."""

    if zipf_freq is None or pd.isna(zipf_freq):
        return 0.0

    diff = float(zipf_freq) - float(target_zipf)
    score = float(np.exp(-(diff**2) / 2))
    return score


def filter_by_supersense(
    supersense_stats_df: pd.DataFrame,
    *,
    min_sense_freq: int | None = None,
    max_senses: int | None = None,
) -> pd.DataFrame:
    """Filter supersense statistics by sense frequency and limit senses per lemma.

    Args:
        supersense_stats_df: DataFrame with supersense statistics. Must contain:
            - lemma: The lemmatized word
            - supersense: The supersense category
            - sense_freq: Frequency of this specific sense
            - sense_count: Number of different senses for this lemma
        min_sense_freq: Minimum frequency for a sense to be included
        max_senses: Maximum number of senses per lemma (keeps top N by frequency)

    Returns:
        Filtered DataFrame with same structure

    Example:
        >>> stats = pd.DataFrame({
        ...     "lemma": ["run", "run", "run", "bank", "bank"],
        ...     "supersense": ["verb.motion", "verb.social", "noun.act",
        ...                    "noun.group", "noun.object"],
        ...     "sense_freq": [17, 10, 8, 15, 5],
        ...     "sense_count": [3, 3, 3, 2, 2],
        ... })
        >>> filtered = filter_by_supersense(stats, min_sense_freq=10, max_senses=2)
        >>> # Only "run" with verb.motion and verb.social, "bank" with noun.group
    """
    if supersense_stats_df is None:
        raise ValueError("supersense_stats_df must be provided")

    required = {LEMMA, SUPERSENSE, SENSE_FREQ}
    missing = required - set(supersense_stats_df.columns)
    if missing:
        raise ValueError(f"supersense_stats_df is missing required columns: {missing}")

    df = supersense_stats_df.copy()

    # Filter by minimum sense frequency
    if min_sense_freq is not None:
        df = df[df[SENSE_FREQ] >= int(min_sense_freq)]

    # Limit number of senses per lemma (keep top N by frequency)
    if max_senses is not None and max_senses > 0:
        # Sort by sense_freq descending, then group by lemma and take top N
        df = df.sort_values(by=[LEMMA, SENSE_FREQ], ascending=[True, False])
        df = df.groupby(LEMMA).head(max_senses).reset_index(drop=True)

        # Recalculate sense_count after filtering
        if SENSE_COUNT in df.columns:
            sense_counts = df.groupby(LEMMA)[SUPERSENSE].nunique().reset_index(name=SENSE_COUNT)
            df = df.drop(columns=[SENSE_COUNT], errors="ignore")
            df = df.merge(sense_counts, on=LEMMA)

    return df.reset_index(drop=True)


def rank_candidates(
    candidates_df: pd.DataFrame,
    *,
    target_zipf: float = TARGET_ZIPF_DEFAULT,
) -> pd.DataFrame:
    """Add ranking score based on normalized frequency and rarity boost."""

    if candidates_df is None:
        raise ValueError("candidates_df must be provided")

    if candidates_df.empty:
        df_empty = candidates_df.copy()
        df_empty[SCORE] = pd.Series(dtype=float)
        return df_empty

    if BOOK_FREQ not in candidates_df.columns or GLOBAL_ZIPF not in candidates_df.columns:
        raise ValueError(f"candidates_df must contain '{BOOK_FREQ}' and '{GLOBAL_ZIPF}'")

    df = candidates_df.copy()
    max_freq = df[BOOK_FREQ].max()
    if max_freq and max_freq > 0:
        df["book_freq_norm"] = df[BOOK_FREQ] / max_freq
    else:
        df["book_freq_norm"] = 0.0

    df["rarity_boost"] = df[GLOBAL_ZIPF].apply(calculate_rarity_score, target_zipf=target_zipf)
    df[SCORE] = df["book_freq_norm"] * df["rarity_boost"]

    df = df.sort_values(by=SCORE, ascending=False).reset_index(drop=True)
    return df.drop(columns=["book_freq_norm", "rarity_boost"])
