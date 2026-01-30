"""Tests for supersense filtering functionality."""

import pandas as pd
import pytest

from eng_words.filtering import filter_by_supersense


class TestFilterBySupersense:
    """Test filter_by_supersense function."""

    @pytest.fixture
    def sample_supersense_stats(self):
        """Create sample supersense statistics."""
        return pd.DataFrame(
            {
                "lemma": ["run", "run", "run", "bank", "bank", "bank", "cat"],
                "supersense": [
                    "verb.motion",
                    "verb.social",
                    "noun.act",
                    "noun.group",
                    "noun.object",
                    "noun.act",
                    "noun.animal",
                ],
                "sense_freq": [17, 10, 8, 15, 5, 2, 20],
                "book_freq": [35, 35, 35, 22, 22, 22, 20],
                "sense_ratio": [0.486, 0.286, 0.229, 0.682, 0.227, 0.091, 1.0],
                "sense_count": [3, 3, 3, 3, 3, 3, 1],
                "doc_count": [15, 8, 5, 12, 4, 2, 18],
            }
        )

    def test_no_filtering(self, sample_supersense_stats):
        """Should return all rows when no filters applied."""
        result = filter_by_supersense(sample_supersense_stats)
        assert len(result) == len(sample_supersense_stats)
        pd.testing.assert_frame_equal(result, sample_supersense_stats)

    def test_min_sense_freq_filter(self, sample_supersense_stats):
        """Should filter by minimum sense frequency."""
        result = filter_by_supersense(sample_supersense_stats, min_sense_freq=10)

        # Should keep: run/verb.motion (17), run/verb.social (10), bank/noun.group (15), cat (20)
        assert len(result) == 4
        assert all(result["sense_freq"] >= 10)

    def test_max_senses_filter(self, sample_supersense_stats):
        """Should limit number of senses per lemma."""
        result = filter_by_supersense(sample_supersense_stats, max_senses=2)

        # Should keep top 2 senses for each lemma
        run_senses = result[result["lemma"] == "run"]
        assert len(run_senses) == 2
        assert set(run_senses["supersense"]) == {"verb.motion", "verb.social"}

        bank_senses = result[result["lemma"] == "bank"]
        assert len(bank_senses) == 2
        assert set(bank_senses["supersense"]) == {"noun.group", "noun.object"}

        cat_senses = result[result["lemma"] == "cat"]
        assert len(cat_senses) == 1  # Only one sense

    def test_combined_filters(self, sample_supersense_stats):
        """Should apply both filters correctly."""
        result = filter_by_supersense(sample_supersense_stats, min_sense_freq=10, max_senses=2)

        # Should keep top 2 senses per lemma, but only those with freq >= 10
        # run: verb.motion (17), verb.social (10) - both kept
        # bank: noun.group (15) - kept, noun.object (5) - filtered out
        # cat: noun.animal (20) - kept

        run_senses = result[result["lemma"] == "run"]
        assert len(run_senses) == 2

        bank_senses = result[result["lemma"] == "bank"]
        assert len(bank_senses) == 1
        assert bank_senses.iloc[0]["supersense"] == "noun.group"

        cat_senses = result[result["lemma"] == "cat"]
        assert len(cat_senses) == 1

    def test_sense_count_recalculated(self, sample_supersense_stats):
        """Should recalculate sense_count after filtering."""
        result = filter_by_supersense(sample_supersense_stats, max_senses=2)

        # Check that sense_count is updated
        run_senses = result[result["lemma"] == "run"]
        assert all(run_senses["sense_count"] == 2)

        bank_senses = result[result["lemma"] == "bank"]
        assert all(bank_senses["sense_count"] == 2)

    def test_empty_dataframe(self):
        """Should handle empty DataFrame."""
        empty_df = pd.DataFrame(columns=["lemma", "supersense", "sense_freq", "sense_count"])
        result = filter_by_supersense(empty_df, min_sense_freq=5)
        assert len(result) == 0

    def test_missing_columns(self):
        """Should raise error for missing required columns."""
        bad_df = pd.DataFrame({"lemma": ["test"], "supersense": ["noun.animal"]})

        with pytest.raises(ValueError, match="missing required columns"):
            filter_by_supersense(bad_df)

    def test_max_senses_zero_or_negative(self, sample_supersense_stats):
        """Should handle max_senses <= 0."""
        result = filter_by_supersense(sample_supersense_stats, max_senses=0)
        # Should return all (max_senses <= 0 means no limit)
        assert len(result) == len(sample_supersense_stats)

        result = filter_by_supersense(sample_supersense_stats, max_senses=-1)
        assert len(result) == len(sample_supersense_stats)
