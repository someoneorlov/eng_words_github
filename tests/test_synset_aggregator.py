"""Tests for synset aggregator."""

import pandas as pd
import pytest

from eng_words.aggregation.synset_aggregator import (
    SynsetStats,
    aggregate_by_synset,
    get_synset_info,
)


class TestGetSynsetInfo:
    """Tests for get_synset_info function."""

    def test_valid_synset(self):
        """Test getting info for a valid synset."""
        info = get_synset_info("dog.n.01")
        assert info is not None
        assert info["synset_id"] == "dog.n.01"
        assert "definition" in info
        assert info["pos"] == "n"
        assert "supersense" in info

    def test_invalid_synset(self):
        """Test getting info for an invalid synset returns None."""
        info = get_synset_info("not_a_synset.x.99")
        assert info is None

    def test_nan_synset(self):
        """Test handling NaN input."""
        info = get_synset_info(None)
        assert info is None


class TestAggregateBysynset:
    """Tests for aggregate_by_synset function."""

    @pytest.fixture
    def sample_tokens_df(self):
        """Create a sample tokens DataFrame for testing."""
        return pd.DataFrame(
            {
                "lemma": ["dog", "dog", "dog", "cat", "cat", "run", "run", "run", "run"],
                "synset_id": [
                    "dog.n.01",
                    "dog.n.01",
                    "dog.n.02",  # dog: 2x n.01, 1x n.02
                    "cat.n.01",
                    "cat.n.01",  # cat: 2x n.01
                    "run.v.01",
                    "run.v.01",
                    "run.v.01",
                    "run.v.02",  # run: 3x v.01, 1x v.02
                ],
                "sentence_id": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            }
        )

    def test_basic_aggregation(self, sample_tokens_df):
        """Test basic synset aggregation."""
        result = aggregate_by_synset(sample_tokens_df, min_freq=1)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # dog.n.01, dog.n.02, cat.n.01, run.v.01, run.v.02

        # Check columns
        expected_cols = [
            "lemma",
            "synset_id",
            "pos",
            "definition",
            "supersense",
            "freq",
            "sentence_ids",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_min_freq_filtering(self, sample_tokens_df):
        """Test that min_freq filtering works."""
        result = aggregate_by_synset(sample_tokens_df, min_freq=2)

        # Only synsets with freq >= 2: dog.n.01 (2), cat.n.01 (2), run.v.01 (3)
        assert len(result) == 3

        synset_ids = result["synset_id"].tolist()
        assert "dog.n.01" in synset_ids
        assert "cat.n.01" in synset_ids
        assert "run.v.01" in synset_ids
        assert "dog.n.02" not in synset_ids  # freq=1
        assert "run.v.02" not in synset_ids  # freq=1

    def test_freq_count_correct(self, sample_tokens_df):
        """Test that frequency counts are correct."""
        result = aggregate_by_synset(sample_tokens_df, min_freq=1)

        run_v01 = result[result["synset_id"] == "run.v.01"].iloc[0]
        assert run_v01["freq"] == 3

        dog_n01 = result[result["synset_id"] == "dog.n.01"].iloc[0]
        assert dog_n01["freq"] == 2

    def test_sentence_ids_collected(self, sample_tokens_df):
        """Test that sentence_ids are collected correctly."""
        result = aggregate_by_synset(sample_tokens_df, min_freq=1)

        run_v01 = result[result["synset_id"] == "run.v.01"].iloc[0]
        assert set(run_v01["sentence_ids"]) == {6, 7, 8}

    def test_handles_nan_synsets(self):
        """Test that NaN synset_ids are filtered out."""
        df = pd.DataFrame(
            {
                "lemma": ["dog", "dog", "cat"],
                "synset_id": ["dog.n.01", None, "cat.n.01"],
                "sentence_id": [1, 2, 3],
            }
        )

        result = aggregate_by_synset(df, min_freq=1)
        assert len(result) == 2  # Only valid synsets

    def test_empty_dataframe(self):
        """Test handling empty DataFrame."""
        df = pd.DataFrame(columns=["lemma", "synset_id", "sentence_id"])
        result = aggregate_by_synset(df, min_freq=1)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_definition_and_pos_populated(self, sample_tokens_df):
        """Test that definition and pos are populated from WordNet."""
        result = aggregate_by_synset(sample_tokens_df, min_freq=1)

        dog_row = result[result["synset_id"] == "dog.n.01"].iloc[0]
        assert dog_row["pos"] == "n"
        assert len(dog_row["definition"]) > 0
        assert len(dog_row["supersense"]) > 0


class TestSynsetStats:
    """Tests for SynsetStats dataclass."""

    def test_create_synset_stats(self):
        """Test creating SynsetStats."""
        stats = SynsetStats(
            lemma="dog",
            synset_id="dog.n.01",
            pos="n",
            definition="a domesticated carnivorous mammal",
            supersense="noun.animal",
            freq=10,
            sentence_ids=[1, 2, 3],
        )

        assert stats.lemma == "dog"
        assert stats.freq == 10
        assert len(stats.sentence_ids) == 3

    def test_synset_stats_to_dict(self):
        """Test converting SynsetStats to dict."""
        stats = SynsetStats(
            lemma="dog",
            synset_id="dog.n.01",
            pos="n",
            definition="test",
            supersense="noun.animal",
            freq=5,
            sentence_ids=[1, 2],
        )

        from dataclasses import asdict

        d = asdict(stats)

        assert d["lemma"] == "dog"
        assert d["freq"] == 5
