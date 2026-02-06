"""MWE (multi-word expression) candidate list for Stage 1 (PIPELINE_B_FIXES_PLAN Stage 4).

Unified artifact: headword, type, count, sample_sentence_ids, source.
Phrasal verbs from Stage 1 detector; fixed_expression/adverbial_phrase can be added later.
"""

from __future__ import annotations

import pandas as pd

from eng_words.constants import (
    BOOK_FREQ,
    MWE_COUNT,
    MWE_HEADWORD,
    MWE_SAMPLE_SENTENCE_IDS,
    MWE_SOURCE,
    MWE_SOURCE_STAGE1_DETECTOR,
    MWE_TYPE,
    MWE_TYPE_PHRASAL_VERB,
    PHRASAL,
    SENTENCE_ID,
)

MWE_CANDIDATES_COLUMNS = [
    MWE_HEADWORD,
    MWE_TYPE,
    MWE_COUNT,
    MWE_SAMPLE_SENTENCE_IDS,
    MWE_SOURCE,
]


def build_mwe_candidates_from_phrasal(
    phrasal_df: pd.DataFrame,
    phrasal_stats_df: pd.DataFrame,
    *,
    max_sample_sentence_ids: int = 10,
) -> pd.DataFrame:
    """Build unified MWE candidates DataFrame from phrasal verbs (Stage 1 detector).

    phrasal_df: has PHRASAL, SENTENCE_ID.
    phrasal_stats_df: has PHRASAL, BOOK_FREQ (and optionally ITEM_TYPE).
    Returns DataFrame with headword, type=phrasal_verb, count, sample_sentence_ids, source=stage1_detector.
    """
    if phrasal_df is None or phrasal_df.empty:
        return pd.DataFrame(columns=MWE_CANDIDATES_COLUMNS)
    if phrasal_stats_df is None or phrasal_stats_df.empty:
        return pd.DataFrame(columns=MWE_CANDIDATES_COLUMNS)
    if PHRASAL not in phrasal_df.columns or SENTENCE_ID not in phrasal_df.columns:
        return pd.DataFrame(columns=MWE_CANDIDATES_COLUMNS)
    if PHRASAL not in phrasal_stats_df.columns or BOOK_FREQ not in phrasal_stats_df.columns:
        return pd.DataFrame(columns=MWE_CANDIDATES_COLUMNS)

    # sample_sentence_ids per phrasal: sorted, capped
    sid_agg = (
        phrasal_df.groupby(PHRASAL)[SENTENCE_ID]
        .apply(lambda s: list(s.unique())[:max_sample_sentence_ids])
        .reset_index()
    )
    sid_agg.columns = [MWE_HEADWORD, MWE_SAMPLE_SENTENCE_IDS]

    stats = phrasal_stats_df[[PHRASAL, BOOK_FREQ]].copy()
    stats.columns = [MWE_HEADWORD, MWE_COUNT]
    stats[MWE_TYPE] = MWE_TYPE_PHRASAL_VERB
    stats[MWE_SOURCE] = MWE_SOURCE_STAGE1_DETECTOR

    out = stats.merge(sid_agg, on=MWE_HEADWORD, how="left")
    out[MWE_SAMPLE_SENTENCE_IDS] = out[MWE_SAMPLE_SENTENCE_IDS].apply(
        lambda x: x if isinstance(x, list) else []
    )
    return out[MWE_CANDIDATES_COLUMNS].reset_index(drop=True)
