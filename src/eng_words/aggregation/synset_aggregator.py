"""
Synset-based aggregation for card generation.

Aggregates tokens by (lemma, synset_id) instead of (lemma, supersense)
to preserve precise word sense information.
"""

import logging
from dataclasses import dataclass

import pandas as pd
from nltk.corpus import wordnet as wn

from eng_words.constants import DIALECT_LEMMAS_FILTER

logger = logging.getLogger(__name__)


@dataclass
class SynsetStats:
    """Statistics for a single synset in the book."""

    lemma: str
    synset_id: str
    pos: str
    definition: str
    supersense: str
    freq: int
    sentence_ids: list[int]


def get_synset_info(synset_id: str | None) -> dict | None:
    """
    Get definition, POS, and supersense for a synset.

    Args:
        synset_id: WordNet synset ID (e.g., "dog.n.01")

    Returns:
        Dict with synset_id, definition, pos, supersense, or None if invalid.
    """
    if synset_id is None or pd.isna(synset_id):
        return None

    try:
        synset = wn.synset(synset_id)
        return {
            "synset_id": synset_id,
            "definition": synset.definition(),
            "pos": synset.pos(),
            "supersense": synset.lexname(),
        }
    except Exception:
        logger.debug(f"Invalid synset: {synset_id}")
        return None


def aggregate_by_synset(
    sense_tokens_df: pd.DataFrame,
    min_freq: int = 2,
    filter_dialect: bool = True,
) -> pd.DataFrame:
    """
    Aggregate tokens by (lemma, synset_id) instead of (lemma, supersense).

    This preserves precise word sense information, ensuring that each
    aggregated group contains tokens with the exact same meaning.

    Args:
        sense_tokens_df: DataFrame with columns: lemma, synset_id, sentence_id
        min_freq: Minimum frequency for a synset to be included
        filter_dialect: If True, filter out dialect/tokenization error lemmas

    Returns:
        DataFrame with columns:
        - lemma: the word lemma
        - synset_id: WordNet synset ID
        - pos: part of speech
        - definition: synset definition
        - supersense: coarse-grained category
        - freq: number of occurrences
        - sentence_ids: list of sentence IDs where this sense appears
    """
    if sense_tokens_df.empty:
        return pd.DataFrame(
            columns=[
                "lemma",
                "synset_id",
                "pos",
                "definition",
                "supersense",
                "freq",
                "sentence_ids",
            ]
        )

    # Filter out rows with missing synset_id
    df = sense_tokens_df[sense_tokens_df["synset_id"].notna()].copy()

    # Filter out dialect/tokenization error lemmas (e.g., "ai" from "ain't")
    if filter_dialect and "lemma" in df.columns:
        before_count = len(df)
        df = df[~df["lemma"].str.lower().isin(DIALECT_LEMMAS_FILTER)]
        filtered_count = before_count - len(df)
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} tokens with dialect lemmas")

    if df.empty:
        return pd.DataFrame(
            columns=[
                "lemma",
                "synset_id",
                "pos",
                "definition",
                "supersense",
                "freq",
                "sentence_ids",
            ]
        )

    # Aggregate by (lemma, synset_id)
    aggregated = (
        df.groupby(["lemma", "synset_id"])
        .agg(
            freq=("sentence_id", "count"),
            sentence_ids=("sentence_id", list),
        )
        .reset_index()
    )

    # Filter by minimum frequency
    aggregated = aggregated[aggregated["freq"] >= min_freq]

    if aggregated.empty:
        return pd.DataFrame(
            columns=[
                "lemma",
                "synset_id",
                "pos",
                "definition",
                "supersense",
                "freq",
                "sentence_ids",
            ]
        )

    # Add synset info (definition, pos, supersense)
    synset_info_cache = {}

    def get_cached_info(synset_id):
        if synset_id not in synset_info_cache:
            synset_info_cache[synset_id] = get_synset_info(synset_id)
        return synset_info_cache[synset_id]

    synset_data = []
    for _, row in aggregated.iterrows():
        info = get_cached_info(row["synset_id"])
        if info:
            synset_data.append(
                {
                    "lemma": row["lemma"],
                    "synset_id": row["synset_id"],
                    "pos": info["pos"],
                    "definition": info["definition"],
                    "supersense": info["supersense"],
                    "freq": row["freq"],
                    "sentence_ids": row["sentence_ids"],
                }
            )

    result = pd.DataFrame(synset_data)

    # Sort by lemma, then by frequency (descending)
    if not result.empty:
        result = result.sort_values(["lemma", "freq"], ascending=[True, False])

    logger.info(
        f"Aggregated {len(sense_tokens_df)} tokens into {len(result)} (lemma, synset) pairs"
    )

    return result
