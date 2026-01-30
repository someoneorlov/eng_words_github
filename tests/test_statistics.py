from pathlib import Path

import pandas as pd
import pytest

from eng_words.statistics import (
    add_global_frequency,
    calculate_lemma_frequency,
    load_lemma_stats_from_parquet,
    save_lemma_stats_to_parquet,
)


def _sample_tokens_df():
    data = [
        {"lemma": "go", "sentence_id": 0, "is_stop": False, "pos": "VERB", "is_alpha": True},
        {"lemma": "go", "sentence_id": 1, "is_stop": False, "pos": "VERB", "is_alpha": True},
        {"lemma": "run", "sentence_id": 1, "is_stop": False, "pos": "VERB", "is_alpha": True},
        {"lemma": "the", "sentence_id": 2, "is_stop": True, "pos": "DET", "is_alpha": True},
        {"lemma": "london", "sentence_id": 3, "is_stop": False, "pos": "PROPN", "is_alpha": True},
        {"lemma": "42", "sentence_id": 4, "is_stop": False, "pos": "NUM", "is_alpha": False},
    ]
    return pd.DataFrame(data)


def test_calculate_lemma_frequency_basic():
    df = _sample_tokens_df()

    stats = calculate_lemma_frequency(df)

    assert list(stats.columns) == [
        "lemma",
        "book_freq",
        "doc_count",
        "verb_count",
        "other_pos_count",
        "stopword_count",
    ]
    assert stats.loc[stats["lemma"] == "go", "book_freq"].iloc[0] == 2
    assert stats.loc[stats["lemma"] == "go", "doc_count"].iloc[0] == 2
    assert stats.loc[stats["lemma"] == "go", "verb_count"].iloc[0] == 2
    assert stats.loc[stats["lemma"] == "go", "other_pos_count"].iloc[0] == 0
    assert stats.loc[stats["lemma"] == "go", "stopword_count"].iloc[0] == 0
    assert (stats["lemma"] == "the").sum() == 0
    assert (stats["lemma"] == "london").sum() == 0


def test_calculate_lemma_frequency_with_stopwords():
    df = _sample_tokens_df()

    stats = calculate_lemma_frequency(df, include_stopwords=True)

    assert "the" in set(stats["lemma"])
    the_row = stats.loc[stats["lemma"] == "the"].iloc[0]
    assert the_row["stopword_count"] == the_row["book_freq"]


def test_calculate_lemma_frequency_with_proper_nouns():
    df = _sample_tokens_df()

    stats = calculate_lemma_frequency(df, include_proper_nouns=True)

    assert "london" in set(stats["lemma"])


def test_calculate_lemma_frequency_empty_after_filter():
    tokens_df = pd.DataFrame(
        [
            {"lemma": "the", "sentence_id": 0, "is_stop": True, "pos": "DET", "is_alpha": True},
        ]
    )

    stats = calculate_lemma_frequency(tokens_df)

    assert stats.empty
    assert list(stats.columns) == [
        "lemma",
        "book_freq",
        "doc_count",
        "verb_count",
        "other_pos_count",
        "stopword_count",
    ]


def test_calculate_lemma_frequency_missing_columns():
    tokens_df = pd.DataFrame({"lemma": ["go"]})

    try:
        calculate_lemma_frequency(tokens_df)
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("ValueError expected")


def test_add_global_frequency_known_word():
    lemma_df = pd.DataFrame({"lemma": ["cat", "nonexistentwordzzz"]})

    result = add_global_frequency(lemma_df)

    assert "global_zipf" in result.columns
    cat_zipf = result.loc[result["lemma"] == "cat", "global_zipf"].iloc[0]
    assert cat_zipf > 3.0
    unknown_zipf = result.loc[result["lemma"] == "nonexistentwordzzz", "global_zipf"].iloc[0]
    assert unknown_zipf == 0.0


def test_save_and_load_lemma_stats(tmp_path: Path):
    stats = pd.DataFrame(
        {
            "lemma": ["run", "jump"],
            "book_freq": [5, 2],
            "doc_count": [3, 1],
            "global_zipf": [3.5, 2.0],
        }
    )
    file_path = tmp_path / "lemma_stats.parquet"

    save_lemma_stats_to_parquet(stats, file_path)
    loaded = load_lemma_stats_from_parquet(file_path)

    pd.testing.assert_frame_equal(stats, loaded)


def test_save_and_load_empty_lemma_stats(tmp_path: Path):
    stats = pd.DataFrame(columns=["lemma", "book_freq", "doc_count", "global_zipf"])
    file_path = tmp_path / "empty_lemma_stats.parquet"

    save_lemma_stats_to_parquet(stats, file_path)
    loaded = load_lemma_stats_from_parquet(file_path)

    assert loaded.empty
    assert list(loaded.columns) == list(stats.columns)


def test_load_lemma_stats_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        load_lemma_stats_from_parquet(missing)


def test_calculate_lemma_frequency_doc_count_unique():
    df = pd.DataFrame(
        [
            {"lemma": "run", "sentence_id": 0, "is_stop": False, "pos": "VERB", "is_alpha": True},
            {"lemma": "run", "sentence_id": 0, "is_stop": False, "pos": "VERB", "is_alpha": True},
            {"lemma": "run", "sentence_id": 1, "is_stop": False, "pos": "VERB", "is_alpha": True},
        ]
    )

    stats = calculate_lemma_frequency(df)
    run_row = stats.loc[stats["lemma"] == "run"].iloc[0]

    assert run_row["book_freq"] == 3
    assert run_row["doc_count"] == 2
    assert run_row["verb_count"] == 3
    assert run_row["other_pos_count"] == 0
    assert run_row["stopword_count"] == 0


def test_calculate_lemma_frequency_chunk_consistency():
    df_chunk1 = pd.DataFrame(
        [
            {
                "lemma": "adventure",
                "sentence_id": 0,
                "is_stop": False,
                "pos": "NOUN",
                "is_alpha": True,
            },
            {
                "lemma": "adventure",
                "sentence_id": 1,
                "is_stop": False,
                "pos": "NOUN",
                "is_alpha": True,
            },
        ]
    )
    df_chunk2 = pd.DataFrame(
        [
            {
                "lemma": "adventure",
                "sentence_id": 2,
                "is_stop": False,
                "pos": "NOUN",
                "is_alpha": True,
            },
            {"lemma": "hero", "sentence_id": 2, "is_stop": False, "pos": "NOUN", "is_alpha": True},
        ]
    )
    combined = pd.concat([df_chunk1, df_chunk2], ignore_index=True)

    stats = calculate_lemma_frequency(combined)

    adv_row = stats.loc[stats["lemma"] == "adventure"].iloc[0]
    assert adv_row["book_freq"] == 3
    assert adv_row["doc_count"] == 3
