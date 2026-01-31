"""Integration tests for the main pipeline flow.

These tests verify the hot path works end-to-end.
LLM calls are mocked to avoid API costs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eng_words.pipeline import process_book, process_book_stage1
from eng_words.text_processing import (
    create_sentences_dataframe,
    create_tokens_dataframe,
    initialize_spacy_model,
    reconstruct_sentences_from_tokens,
    tokenize_text_in_chunks,
)


class TestPipelineIntegration:
    """Integration tests for the main pipeline hot path."""

    @pytest.fixture
    def sample_text(self) -> str:
        """Sample text for testing with sufficient repetition."""
        # Use repetitive text so words pass frequency filters (min_book_freq=3)
        return (
            "The quick brown fox jumps over the lazy dog. "
            "The quick brown fox jumps again. "
            "The quick brown fox jumps one more time. "
            "She gave up quickly. She gave up again. She gave up one more time. "
            "Turn the lights on now. Turn the lights on again. Turn the lights on once more. "
            "The cat sat on the mat. The cat sat on the mat again. The cat sat on the mat once more. "
            "Books open minds. Books open minds again. Books open minds once more."
        )

    @pytest.fixture
    def known_words_csv(self, tmp_path: Path) -> Path:
        """Create a sample known words CSV file."""
        csv_path = tmp_path / "known_words.csv"
        csv_path.write_text(
            "lemma,status,item_type,tags\n"
            "cat,known,word,A1\n"
            "mat,ignore,word,A2\n"  # Use 'ignore' instead of 'learning' (learning is not filtered by default)
            "give up,ignore,phrasal_verb,B1\n"
        )
        return csv_path

    def test_stage1_tokenization_smoke(self, tmp_path: Path, sample_text: str) -> None:
        """Stage 1: Text → Tokens works."""
        book_path = tmp_path / "test_book.txt"
        book_path.write_text(sample_text)
        output_dir = tmp_path / "outputs"

        try:
            result = process_book_stage1(
                book_path=book_path,
                book_name="test_book",
                output_dir=output_dir,
                min_book_freq=1,  # Lower threshold for test
                min_zipf=0.0,  # Lower threshold for test
            )
        except OSError as exc:
            if "Can't find model" in str(exc):
                pytest.skip("spaCy model en_core_web_sm is not installed")
            raise

        # Verify tokens are extracted
        tokens_df = result["tokens_df"]
        tokens_path = result["tokens_path"]

        assert tokens_path.exists(), "Tokens parquet file should be created"
        assert len(tokens_df) > 0, "Should extract tokens from text"
        assert "lemma" in tokens_df.columns, "Tokens should have lemma column"
        assert "sentence_id" in tokens_df.columns, "Tokens should have sentence_id column"
        assert tokens_df["book"].iloc[0] == "test_book", "Book name should be set"

        # Verify lemma stats are created
        lemma_stats_df = result["lemma_stats_df"]
        lemma_stats_path = result["lemma_stats_path"]

        assert lemma_stats_path.exists(), "Lemma stats parquet file should be created"
        assert len(lemma_stats_df) > 0, "Should have lemma statistics"
        assert "lemma" in lemma_stats_df.columns, "Stats should have lemma column"
        assert "book_freq" in lemma_stats_df.columns, "Stats should have book_freq column"
        assert "global_zipf" in lemma_stats_df.columns, "Stats should have global_zipf column"

    def test_known_words_filtering(self, tmp_path: Path, sample_text: str, known_words_csv: Path) -> None:
        """Known words are filtered correctly."""
        book_path = tmp_path / "test_book.txt"
        book_path.write_text(sample_text)
        output_dir = tmp_path / "outputs"

        try:
            result = process_book_stage1(
                book_path=book_path,
                book_name="test_book",
                output_dir=output_dir,
                known_words_path=known_words_csv,
                min_book_freq=1,  # Lower threshold for test
                min_zipf=0.0,  # Lower threshold for test
            )
        except OSError as exc:
            if "Can't find model" in str(exc):
                pytest.skip("spaCy model en_core_web_sm is not installed")
            raise

        lemma_stats_df = result["lemma_stats_df"]

        # Verify known words are filtered out
        # "cat" and "mat" should be filtered (known/ignore status)
        # "give up" is a phrasal verb, so it won't appear in lemma_stats
        lemmas = set(lemma_stats_df["lemma"].str.lower())
        assert "cat" not in lemmas, "Known word 'cat' should be filtered"
        assert "mat" not in lemmas, "Ignored word 'mat' should be filtered"

        # Other words should still be present
        assert len(lemma_stats_df) > 0, "Should have remaining lemmas after filtering"

    def test_full_pipeline_end_to_end(self, tmp_path: Path, sample_text: str) -> None:
        """Full pipeline: Text → Tokens → Filtering → Examples → Anki CSV."""
        book_path = tmp_path / "test_book.txt"
        book_path.write_text(sample_text)
        output_dir = tmp_path / "outputs"

        try:
            outputs = process_book(
                book_path=book_path,
                book_name="test_book",
                output_dir=output_dir,
                top_n=5,
                detect_phrasals=False,  # Disable phrasals for simpler test
                min_book_freq=1,  # Lower threshold for test
                min_zipf=0.0,  # Lower threshold for test
            )
        except OSError as exc:
            if "Can't find model" in str(exc):
                pytest.skip("spaCy model en_core_web_sm is not installed")
            raise

        # Verify all output files exist
        assert outputs["tokens"] is not None and Path(outputs["tokens"]).exists()
        assert outputs["lemma_stats"] is not None and Path(outputs["lemma_stats"]).exists()
        assert outputs["lemma_stats_full"] is not None and Path(outputs["lemma_stats_full"]).exists()

        # Verify Anki CSV is created if there are lemmas
        if outputs["anki_csv"]:
            anki_path = Path(outputs["anki_csv"])
            assert anki_path.exists(), "Anki CSV should be created"

            # Verify Anki CSV structure
            # Note: export_to_anki_csv uses default comma separator
            anki_df = pd.read_csv(anki_path)
            assert "front" in anki_df.columns, "Anki CSV should have front column"
            assert "back" in anki_df.columns, "Anki CSV should have back column"
            assert "tags" in anki_df.columns, "Anki CSV should have tags column"
            assert len(anki_df) > 0, "Anki CSV should have at least one row"

            # Verify tags contain book name
            assert all(anki_df["tags"] == "test_book"), "Tags should match book name"

    def test_pipeline_with_known_words_filtering(self, tmp_path: Path, sample_text: str, known_words_csv: Path) -> None:
        """Full pipeline with known words filtering."""
        book_path = tmp_path / "test_book.txt"
        book_path.write_text(sample_text)
        output_dir = tmp_path / "outputs"

        try:
            outputs = process_book(
                book_path=book_path,
                book_name="test_book",
                output_dir=output_dir,
                known_words_path=known_words_csv,
                top_n=10,
                detect_phrasals=False,
                min_book_freq=1,  # Lower threshold for test
                min_zipf=0.0,  # Lower threshold for test
            )
        except OSError as exc:
            if "Can't find model" in str(exc):
                pytest.skip("spaCy model en_core_web_sm is not installed")
            raise

        # Verify filtering worked
        lemma_stats_path = Path(outputs["lemma_stats"])
        lemma_stats_df = pd.read_parquet(lemma_stats_path)

        lemmas = set(lemma_stats_df["lemma"].str.lower())
        assert "cat" not in lemmas, "Known word should be filtered"
        assert "mat" not in lemmas, "Ignored word should be filtered"

        # Verify Anki CSV doesn't contain filtered words
        if outputs["anki_csv"]:
            # Note: export_to_anki_csv uses default comma separator
            anki_df = pd.read_csv(Path(outputs["anki_csv"]))
            anki_lemmas = set(anki_df["front"].str.lower())
            assert "cat" not in anki_lemmas, "Anki CSV should not contain known words"
            assert "mat" not in anki_lemmas, "Anki CSV should not contain ignored words"

    def test_tokenization_preserves_sentence_structure(self, tmp_path: Path) -> None:
        """Verify tokenization preserves sentence boundaries."""
        # Use text with clear sentence boundaries
        text = "First sentence. Second sentence! Third sentence?"
        book_path = tmp_path / "sentences.txt"
        book_path.write_text(text)
        output_dir = tmp_path / "outputs"

        try:
            result = process_book_stage1(
                book_path=book_path,
                book_name="sentences",
                output_dir=output_dir,
                min_book_freq=1,  # Lower threshold for test
                min_zipf=0.0,  # Lower threshold for test
            )
        except OSError as exc:
            if "Can't find model" in str(exc):
                pytest.skip("spaCy model en_core_web_sm is not installed")
            raise

        tokens_df = result["tokens_df"]

        # Verify sentence IDs are present and sequential
        sentence_ids = tokens_df["sentence_id"].unique()
        assert len(sentence_ids) >= 3, "Should have at least 3 sentences"

        # Verify we can reconstruct sentences
        sentences = reconstruct_sentences_from_tokens(tokens_df)
        sentences_df = create_sentences_dataframe(sentences)

        assert len(sentences_df) >= 3, "Should reconstruct at least 3 sentences"
        assert "sentence" in sentences_df.columns, "Should have sentence column"
        assert "sentence_id" in sentences_df.columns, "Should have sentence_id column"

        # Verify sentence text contains expected words
        all_sentences = " ".join(sentences_df["sentence"].str.lower())
        assert "first" in all_sentences or "sentence" in all_sentences
        assert "second" in all_sentences or "sentence" in all_sentences
        assert "third" in all_sentences or "sentence" in all_sentences
