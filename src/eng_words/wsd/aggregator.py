"""Aggregation utilities for sense statistics.

This module provides functions for aggregating sense annotations into
statistics by (lemma, supersense) pairs.
"""

import pandas as pd

from eng_words.wsd.base import validate_annotated_df


def aggregate_sense_statistics(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sense annotations into statistics by (lemma, supersense).

    This function groups annotated tokens by lemma and supersense, calculating:
    - sense_freq: Frequency of each (lemma, supersense) pair
    - book_freq: Total frequency of the lemma in the book
    - sense_ratio: sense_freq / book_freq
    - doc_count: Number of unique sentences with this sense
    - sense_count: Number of different senses for each lemma
    - dominant_supersense: Most frequent supersense for each lemma

    Args:
        annotated_df: DataFrame with sense annotations. Required columns:
            - lemma: The lemmatized word
            - supersense: The supersense category
            - sentence_id: For counting unique sentences

    Returns:
        DataFrame with columns:
            - lemma: The lemmatized word
            - supersense: The supersense category
            - sense_freq: Frequency of this specific sense
            - book_freq: Total frequency of the lemma
            - sense_ratio: sense_freq / book_freq
            - doc_count: Number of unique sentences with this sense
            - sense_count: Number of different senses for this lemma
            - dominant_supersense: Most frequent supersense for this lemma

    Raises:
        ValueError: If required columns are missing

    Example:
        >>> annotated = pd.DataFrame({
        ...     "lemma": ["run", "run", "run", "bank", "bank"],
        ...     "supersense": ["verb.motion", "verb.motion", "verb.social",
        ...                    "noun.group", "noun.object"],
        ...     "sentence_id": [1, 2, 3, 4, 5],
        ... })
        >>> stats = aggregate_sense_statistics(annotated)
        >>> stats[stats["lemma"] == "run"]["sense_count"].iloc[0]
        2
    """
    validate_annotated_df(annotated_df)

    if annotated_df.empty:
        return pd.DataFrame(
            columns=[
                "lemma",
                "supersense",
                "sense_freq",
                "book_freq",
                "sense_ratio",
                "doc_count",
                "sense_count",
                "dominant_supersense",
            ]
        )

    # Calculate book frequency per lemma
    book_freq = annotated_df.groupby("lemma").size().reset_index(name="book_freq")

    # Calculate sense frequency per (lemma, supersense)
    sense_stats = (
        annotated_df.groupby(["lemma", "supersense"])
        .agg(
            sense_freq=("supersense", "size"),
            doc_count=("sentence_id", "nunique"),
        )
        .reset_index()
    )

    # Merge book frequency
    sense_stats = sense_stats.merge(book_freq, on="lemma")

    # Calculate sense ratio
    sense_stats["sense_ratio"] = sense_stats["sense_freq"] / sense_stats["book_freq"]

    # Calculate sense_count (number of different senses per lemma)
    sense_count = (
        annotated_df.groupby("lemma")["supersense"].nunique().reset_index(name="sense_count")
    )
    sense_stats = sense_stats.merge(sense_count, on="lemma")

    # Find dominant supersense for each lemma
    dominant = sense_stats.loc[sense_stats.groupby("lemma")["sense_freq"].idxmax()][
        ["lemma", "supersense"]
    ].rename(columns={"supersense": "dominant_supersense"})

    # Merge dominant supersense
    sense_stats = sense_stats.merge(dominant, on="lemma")

    return sense_stats
