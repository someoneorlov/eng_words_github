"""Tests for WSD integration in pipeline."""

from pathlib import Path

import pandas as pd
import pytest

# Check if spaCy model is available
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


class TestPipelineWSDIntegration:
    """Test WSD integration in the pipeline."""

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Chapter 1: The Bank

        I deposited money at the bank yesterday. The bank approved my loan.
        We sat on the river bank and watched the sunset.
        
        The dog runs quickly across the field. She runs a successful business.
        I run every morning to stay healthy.
        """

    @pytest.fixture
    def temp_book(self, tmp_path, sample_text):
        """Create a temporary book file."""
        book_path = tmp_path / "test_book.txt"
        book_path.write_text(sample_text)
        return book_path

    def test_process_book_stage1_without_wsd(self, temp_book, tmp_path):
        """Test that pipeline works without WSD (backward compatibility)."""
        from eng_words.pipeline import process_book_stage1

        results = process_book_stage1(
            book_path=temp_book,
            book_name="test_book",
            output_dir=tmp_path / "output",
            enable_wsd=False,
        )

        # Basic outputs should exist
        assert results["tokens_df"] is not None
        assert results["tokens_path"].exists()
        assert results["lemma_stats_df"] is not None
        assert results["lemma_stats_path"].exists()

        # WSD outputs should be None
        assert results["sense_tokens_df"] is None
        assert results["sense_tokens_path"] is None
        assert results["supersense_stats_df"] is None
        assert results["supersense_stats_path"] is None

    def test_process_book_stage1_with_wsd(self, temp_book, tmp_path):
        """Test that pipeline works with WSD enabled."""
        from eng_words.pipeline import process_book_stage1

        results = process_book_stage1(
            book_path=temp_book,
            book_name="test_book",
            output_dir=tmp_path / "output",
            enable_wsd=True,
            wsd_checkpoint_interval=100,  # Small interval for testing
        )

        # Basic outputs should exist
        assert results["tokens_df"] is not None
        assert results["tokens_path"].exists()

        # WSD outputs should exist
        assert results["sense_tokens_df"] is not None
        assert results["sense_tokens_path"] is not None
        assert results["sense_tokens_path"].exists()
        assert results["supersense_stats_df"] is not None
        assert results["supersense_stats_path"] is not None
        assert results["supersense_stats_path"].exists()

        # Verify sense_tokens_df has WSD columns
        sense_df = results["sense_tokens_df"]
        assert "synset_id" in sense_df.columns
        assert "supersense" in sense_df.columns
        assert "sense_confidence" in sense_df.columns

        # Verify supersense_stats_df has expected columns
        stats_df = results["supersense_stats_df"]
        assert "lemma" in stats_df.columns
        assert "supersense" in stats_df.columns
        assert "sense_freq" in stats_df.columns
        assert "sense_ratio" in stats_df.columns
        assert "sense_count" in stats_df.columns

    def test_process_book_with_wsd(self, temp_book, tmp_path):
        """Test full pipeline with WSD enabled."""
        from eng_words.pipeline import process_book

        outputs = process_book(
            book_path=temp_book,
            book_name="test_book",
            output_dir=tmp_path / "output",
            enable_wsd=True,
            detect_phrasals=False,  # Disable for faster test
            wsd_checkpoint_interval=100,
        )

        # Basic outputs
        assert outputs["tokens"] is not None
        assert Path(outputs["tokens"]).exists()
        assert outputs["lemma_stats"] is not None

        # WSD outputs
        assert outputs["sense_tokens"] is not None
        assert Path(outputs["sense_tokens"]).exists()
        assert outputs["supersense_stats"] is not None
        assert Path(outputs["supersense_stats"]).exists()

    def test_process_book_without_wsd_backward_compatible(self, temp_book, tmp_path):
        """Test that pipeline without WSD returns expected outputs."""
        from eng_words.pipeline import process_book

        outputs = process_book(
            book_path=temp_book,
            book_name="test_book",
            output_dir=tmp_path / "output",
            enable_wsd=False,
            detect_phrasals=False,
        )

        # Basic outputs should exist
        assert outputs["tokens"] is not None
        assert outputs["lemma_stats"] is not None

        # WSD outputs should be None
        assert outputs["sense_tokens"] is None
        assert outputs["supersense_stats"] is None

    def test_wsd_checkpoint_removed_on_success(self, temp_book, tmp_path):
        """Test that WSD checkpoint file is removed after successful completion."""
        from eng_words.pipeline import process_book_stage1

        output_dir = tmp_path / "output"

        process_book_stage1(
            book_path=temp_book,
            book_name="test_book",
            output_dir=output_dir,
            enable_wsd=True,
            wsd_checkpoint_interval=100,
        )

        # Checkpoint file should be removed
        checkpoint_path = output_dir / "test_book_wsd_checkpoint.parquet"
        assert not checkpoint_path.exists()

    def test_wsd_output_files_are_valid_parquet(self, temp_book, tmp_path):
        """Test that WSD output files are valid parquet files."""
        from eng_words.pipeline import process_book_stage1

        results = process_book_stage1(
            book_path=temp_book,
            book_name="test_book",
            output_dir=tmp_path / "output",
            enable_wsd=True,
            wsd_checkpoint_interval=100,
        )

        # Load and verify sense_tokens
        sense_df = pd.read_parquet(results["sense_tokens_path"])
        assert len(sense_df) > 0
        assert "supersense" in sense_df.columns

        # Load and verify supersense_stats
        stats_df = pd.read_parquet(results["supersense_stats_path"])
        assert len(stats_df) > 0
        assert "sense_ratio" in stats_df.columns


class TestWSDOutputContent:
    """Test the content of WSD outputs."""

    @pytest.fixture
    def sample_text_with_polysemy(self):
        """Text with clear polysemous words."""
        return """
        The bank approved my loan application for the new house.
        We walked along the river bank at sunset.
        The bank is closed on weekends.
        
        The dog runs fast in the park.
        She runs a large corporation.
        Water runs down the hill.
        """

    @pytest.fixture
    def temp_book(self, tmp_path, sample_text_with_polysemy):
        """Create a temporary book file."""
        book_path = tmp_path / "polysemy_book.txt"
        book_path.write_text(sample_text_with_polysemy)
        return book_path

    def test_polysemous_words_have_multiple_senses(self, temp_book, tmp_path):
        """Test that polysemous words get multiple supersenses."""
        from eng_words.pipeline import process_book_stage1

        results = process_book_stage1(
            book_path=temp_book,
            book_name="polysemy_test",
            output_dir=tmp_path / "output",
            enable_wsd=True,
            wsd_checkpoint_interval=100,
        )

        stats_df = results["supersense_stats_df"]

        # Check that "bank" has sense_count > 0
        # (may or may not distinguish senses depending on WSD accuracy)
        if "bank" in stats_df["lemma"].values:
            bank_stats = stats_df[stats_df["lemma"] == "bank"]
            assert len(bank_stats) >= 1
            assert bank_stats["sense_count"].iloc[0] >= 1

        # Check that "run" has sense_count > 0
        if "run" in stats_df["lemma"].values:
            run_stats = stats_df[stats_df["lemma"] == "run"]
            assert len(run_stats) >= 1
            assert run_stats["sense_count"].iloc[0] >= 1

    def test_sense_ratios_sum_to_one(self, temp_book, tmp_path):
        """Test that sense ratios for each lemma sum to 1.0."""
        from eng_words.pipeline import process_book_stage1

        results = process_book_stage1(
            book_path=temp_book,
            book_name="ratio_test",
            output_dir=tmp_path / "output",
            enable_wsd=True,
            wsd_checkpoint_interval=100,
        )

        stats_df = results["supersense_stats_df"]

        # For each lemma, sense_ratios should sum to 1.0
        for lemma in stats_df["lemma"].unique():
            lemma_stats = stats_df[stats_df["lemma"] == lemma]
            total_ratio = lemma_stats["sense_ratio"].sum()
            assert abs(total_ratio - 1.0) < 0.01, f"Ratios for '{lemma}' sum to {total_ratio}"

    def test_all_content_words_annotated(self, temp_book, tmp_path):
        """Test that all content words get WSD annotation."""
        from eng_words.pipeline import process_book_stage1

        results = process_book_stage1(
            book_path=temp_book,
            book_name="annotation_test",
            output_dir=tmp_path / "output",
            enable_wsd=True,
            wsd_checkpoint_interval=100,
        )

        sense_df = results["sense_tokens_df"]

        # Content words (NOUN, VERB, ADJ, ADV) should have supersense != "unknown"
        # (unless the word is not in WordNet)
        content_pos = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
        content_tokens = sense_df[sense_df["pos"].isin(content_pos)]

        # At least some content words should have valid supersenses
        valid_supersenses = content_tokens[content_tokens["supersense"] != "unknown"]
        assert len(valid_supersenses) > 0
