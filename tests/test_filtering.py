from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eng_words.filtering import (
    calculate_rarity_score,
    filter_by_frequency,
    filter_by_status,
    filter_dialect_lemmas,
    filter_known_words,
    load_known_words_from_csv,
    rank_candidates,
)


def test_load_known_words_from_csv_success(tmp_path: Path) -> None:
    csv_file = tmp_path / "known.csv"
    csv_file.write_text(
        "lemma,status,item_type,tags\n"
        "run,known,word,A2\n"
        "Run,learning,word,B1\n"
        "give up,ignore,phrasal_verb,B2\n"
    )

    df = load_known_words_from_csv(csv_file)

    assert len(df) == 2  # duplicate lemma/item_type collapsed
    assert set(df["lemma"]) == {"run", "give up"}
    assert df.loc[df["lemma"] == "run", "status"].iloc[0] == "learning"


def test_load_known_words_missing_columns(tmp_path: Path) -> None:
    csv_file = tmp_path / "known.csv"
    csv_file.write_text("lemma,status\nrun,known\n")

    with pytest.raises(ValueError, match="missing columns"):
        load_known_words_from_csv(csv_file)


def test_load_known_words_empty_file(tmp_path: Path) -> None:
    csv_file = tmp_path / "known.csv"
    csv_file.write_text("")

    df = load_known_words_from_csv(csv_file)
    assert df.empty


def test_filter_known_words_basic() -> None:
    candidates = pd.DataFrame({"lemma": ["run", "jump", "sleep"]})
    known = pd.DataFrame(
        {"lemma": ["run"], "status": ["known"], "item_type": ["word"], "tags": ["A2"]}
    )

    filtered = filter_known_words(candidates, known)

    assert list(filtered["lemma"]) == ["jump", "sleep"]


def test_filter_by_status_custom() -> None:
    candidates = pd.DataFrame({"lemma": ["run", "jump", "sleep"]})
    known = pd.DataFrame(
        {
            "lemma": ["run", "sleep"],
            "status": ["learning", "ignore"],
            "item_type": ["word", "word"],
            "tags": ["B1", "A2"],
        }
    )

    filtered = filter_by_status(candidates, known, exclude_statuses=["ignore"])

    assert list(filtered["lemma"]) == ["run", "jump"]


def test_filter_by_frequency_basic() -> None:
    stats = pd.DataFrame(
        {
            "lemma": ["rare", "medium", "common"],
            "book_freq": [2, 5, 20],
            "global_zipf": [1.5, 3.0, 6.0],
        }
    )

    result = filter_by_frequency(stats, min_book_freq=3, min_zipf=2.0, max_zipf=5.5)
    assert list(result["lemma"]) == ["medium"]


def test_filter_by_frequency_edge_cases() -> None:
    stats = pd.DataFrame({"lemma": ["a", "b"], "book_freq": [3, 3], "global_zipf": [2.0, 5.5]})
    result = filter_by_frequency(stats, min_book_freq=3, min_zipf=2.0, max_zipf=5.5)
    assert set(result["lemma"]) == {"a", "b"}


def test_calculate_rarity_score_behavior() -> None:
    center = calculate_rarity_score(4.0)
    far = calculate_rarity_score(1.0)
    none_value = calculate_rarity_score(None)

    assert center > far
    assert none_value == 0.0


def test_rank_candidates_orders_by_score() -> None:
    stats = pd.DataFrame(
        {
            "lemma": ["freq_high", "rarer_mid", "rare_low"],
            "book_freq": [50, 20, 5],
            "global_zipf": [6.0, 4.0, 2.5],
        }
    )

    ranked = rank_candidates(stats, target_zipf=4.0)

    assert ranked.iloc[0]["lemma"] == "rarer_mid"
    assert "score" in ranked.columns


# =============================================================================
# Dialect lemma filter tests
# =============================================================================


def test_filter_dialect_lemmas_removes_known_errors() -> None:
    """Test that known dialect/tokenization errors are filtered out."""
    df = pd.DataFrame({
        "lemma": ["run", "ai", "walk", "de", "jump", "dis"],
        "book_freq": [10, 5, 8, 3, 6, 2],
    })

    filtered = filter_dialect_lemmas(df)

    # Should remove ai, de, dis
    assert set(filtered["lemma"]) == {"run", "walk", "jump"}
    assert len(filtered) == 3


def test_filter_dialect_lemmas_case_insensitive() -> None:
    """Test that filtering is case insensitive."""
    df = pd.DataFrame({
        "lemma": ["AI", "De", "run", "DIS"],
    })

    filtered = filter_dialect_lemmas(df)

    assert list(filtered["lemma"]) == ["run"]


def test_filter_dialect_lemmas_empty_df() -> None:
    """Test filtering empty DataFrame."""
    df = pd.DataFrame({"lemma": []})

    filtered = filter_dialect_lemmas(df)

    assert len(filtered) == 0


def test_filter_dialect_lemmas_no_dialect() -> None:
    """Test DataFrame with no dialect lemmas stays unchanged."""
    df = pd.DataFrame({
        "lemma": ["run", "walk", "jump"],
        "book_freq": [10, 8, 6],
    })

    filtered = filter_dialect_lemmas(df)

    assert len(filtered) == 3
    assert list(filtered["lemma"]) == ["run", "walk", "jump"]


def test_filter_dialect_lemmas_custom_filter() -> None:
    """Test with custom dialect filter set."""
    df = pd.DataFrame({
        "lemma": ["foo", "bar", "baz"],
    })

    custom_filter = frozenset({"foo", "bar"})
    filtered = filter_dialect_lemmas(df, dialect_lemmas=custom_filter)

    assert list(filtered["lemma"]) == ["baz"]
