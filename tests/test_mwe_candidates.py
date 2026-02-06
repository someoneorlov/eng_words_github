"""Tests for MWE candidates (PIPELINE_B_FIXES_PLAN Stage 4)."""

from __future__ import annotations

import pandas as pd
import pytest

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
from eng_words.mwe_candidates import (
    MWE_CANDIDATES_COLUMNS,
    build_mwe_candidates_from_phrasal,
)


def test_build_mwe_candidates_from_phrasal_empty() -> None:
    """Empty phrasal inputs -> empty DataFrame with correct columns."""
    out = build_mwe_candidates_from_phrasal(
        pd.DataFrame(columns=[PHRASAL, SENTENCE_ID]),
        pd.DataFrame(columns=[PHRASAL, BOOK_FREQ]),
    )
    assert list(out.columns) == MWE_CANDIDATES_COLUMNS
    assert len(out) == 0


def test_build_mwe_candidates_from_phrasal_fixture() -> None:
    """Fixture: phrasal_df + stats -> headword, type, count, sample_sentence_ids, source."""
    phrasal_df = pd.DataFrame([
        {"phrasal": "give up", "sentence_id": 1},
        {"phrasal": "give up", "sentence_id": 2},
        {"phrasal": "look up", "sentence_id": 1},
    ])
    phrasal_stats_df = pd.DataFrame([
        {"phrasal": "give up", "book_freq": 2},
        {"phrasal": "look up", "book_freq": 1},
    ])
    out = build_mwe_candidates_from_phrasal(phrasal_df, phrasal_stats_df)
    assert list(out.columns) == MWE_CANDIDATES_COLUMNS
    assert len(out) == 2
    out = out.sort_values(MWE_HEADWORD).reset_index(drop=True)
    assert out.iloc[0][MWE_HEADWORD] == "give up"
    assert out.iloc[0][MWE_TYPE] == MWE_TYPE_PHRASAL_VERB
    assert out.iloc[0][MWE_COUNT] == 2
    assert set(out.iloc[0][MWE_SAMPLE_SENTENCE_IDS]) == {1, 2}
    assert out.iloc[0][MWE_SOURCE] == MWE_SOURCE_STAGE1_DETECTOR
    assert out.iloc[1][MWE_HEADWORD] == "look up"
    assert out.iloc[1][MWE_COUNT] == 1
    assert out.iloc[1][MWE_SAMPLE_SENTENCE_IDS] == [1]


def test_build_mwe_candidates_sample_sentence_ids_capped() -> None:
    """sample_sentence_ids are capped at max_sample_sentence_ids."""
    phrasal_df = pd.DataFrame([
        {"phrasal": "go on", "sentence_id": i} for i in range(15)
    ])
    phrasal_stats_df = pd.DataFrame([{"phrasal": "go on", "book_freq": 15}])
    out = build_mwe_candidates_from_phrasal(
        phrasal_df, phrasal_stats_df, max_sample_sentence_ids=5
    )
    assert len(out) == 1
    assert out.iloc[0][MWE_COUNT] == 15
    assert len(out.iloc[0][MWE_SAMPLE_SENTENCE_IDS]) == 5


def test_build_mwe_candidates_determinism() -> None:
    """Same input -> same output (deterministic order)."""
    phrasal_df = pd.DataFrame([
        {"phrasal": "give up", "sentence_id": 1},
        {"phrasal": "look up", "sentence_id": 2},
    ])
    phrasal_stats_df = pd.DataFrame([
        {"phrasal": "give up", "book_freq": 1},
        {"phrasal": "look up", "book_freq": 1},
    ])
    out1 = build_mwe_candidates_from_phrasal(phrasal_df, phrasal_stats_df)
    out2 = build_mwe_candidates_from_phrasal(phrasal_df, phrasal_stats_df)
    pd.testing.assert_frame_equal(out1.sort_values(MWE_HEADWORD).reset_index(drop=True),
                                  out2.sort_values(MWE_HEADWORD).reset_index(drop=True))


def test_build_mwe_candidates_none_empty_returns_empty() -> None:
    """None or empty stats -> empty DataFrame."""
    phrasal_df = pd.DataFrame([{"phrasal": "x", "sentence_id": 1}])
    assert len(build_mwe_candidates_from_phrasal(phrasal_df, None)) == 0
    assert len(build_mwe_candidates_from_phrasal(phrasal_df, pd.DataFrame())) == 0
    assert len(build_mwe_candidates_from_phrasal(None, pd.DataFrame())) == 0
