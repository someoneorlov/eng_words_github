"""Tests for pipeline Stage 1 and full pipeline (WSD removed from Stage 1)."""

from pathlib import Path

import pytest

try:
    import spacy
    spacy.load("en_core_web_sm")
    SPACY_MODEL_AVAILABLE = True
except OSError:
    SPACY_MODEL_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SPACY_MODEL_AVAILABLE,
    reason="spaCy model 'en_core_web_sm' not installed",
)


class TestPipelineStage1:
    """Test Stage 1 pipeline (no WSD)."""

    @pytest.fixture
    def sample_text(self):
        return """
        Chapter 1: The Bank
        I deposited money at the bank yesterday. The dog runs quickly.
        """

    @pytest.fixture
    def temp_book(self, tmp_path, sample_text):
        book_path = tmp_path / "test_book.txt"
        book_path.write_text(sample_text)
        return book_path

    def test_process_book_stage1_returns_expected_keys(self, temp_book, tmp_path):
        """Stage 1 returns tokens, lemma_stats, phrasal (no WSD outputs)."""
        from eng_words.pipeline import process_book_stage1

        results = process_book_stage1(
            book_path=temp_book,
            book_name="test_book",
            output_dir=tmp_path / "output",
        )

        assert results["tokens_df"] is not None
        assert results["tokens_path"].exists()
        assert results["lemma_stats_df"] is not None
        assert results["lemma_stats_path"].exists()
        assert "sense_tokens_df" not in results
        assert "sense_tokens_path" not in results
        assert "supersense_stats_df" not in results
        assert "supersense_stats_path" not in results

    def test_process_book_returns_expected_keys(self, temp_book, tmp_path):
        """Full pipeline returns paths (no sense_tokens/supersense_stats)."""
        from eng_words.pipeline import process_book

        outputs = process_book(
            book_path=temp_book,
            book_name="test_book",
            output_dir=tmp_path / "output",
            detect_phrasals=False,
        )

        assert outputs["tokens"] is not None
        assert outputs["lemma_stats"] is not None
        assert "sense_tokens" not in outputs
        assert "supersense_stats" not in outputs
