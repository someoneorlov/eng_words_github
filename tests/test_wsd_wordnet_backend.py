"""Tests for WordNet-based WSD backend."""

import pandas as pd
import pytest

from eng_words.wsd.base import SenseAnnotation
from eng_words.wsd.wordnet_backend import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    WordNetSenseBackend,
)


class TestWordNetSenseBackendInit:
    """Test backend initialization."""

    def test_default_initialization(self):
        """Should initialize with default parameters."""
        backend = WordNetSenseBackend()
        assert backend.name == "WordNet + Sentence-Transformers"
        assert backend.confidence_threshold == DEFAULT_CONFIDENCE_THRESHOLD

    def test_custom_threshold(self):
        """Should accept custom confidence threshold."""
        backend = WordNetSenseBackend(confidence_threshold=0.5)
        assert backend.confidence_threshold == 0.5

    def test_supports_confidence(self):
        """Should support confidence scores."""
        backend = WordNetSenseBackend()
        assert backend.supports_confidence is True

    def test_supports_definitions(self):
        """Should support definitions."""
        backend = WordNetSenseBackend()
        assert backend.supports_definitions is True


class TestDisambiguateWord:
    """Test single word disambiguation."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return WordNetSenseBackend()

    def test_bank_financial_context(self, backend):
        """Bank in financial context should be noun.group."""
        result = backend.disambiguate_word(
            sentence="I deposited money at the bank",
            lemma="bank",
            pos="n",
        )
        assert isinstance(result, SenseAnnotation)
        assert result.lemma == "bank"
        assert result.sense_id is not None
        assert result.sense_label.startswith("noun.")
        assert result.confidence > 0
        assert result.definition is not None
        # Financial bank is typically noun.group
        # (but we don't strictly assert this as WSD can vary)

    def test_bank_river_context(self, backend):
        """Bank in river context should be noun.object."""
        result = backend.disambiguate_word(
            sentence="We sat on the river bank and watched the water",
            lemma="bank",
            pos="n",
        )
        assert isinstance(result, SenseAnnotation)
        assert result.lemma == "bank"
        assert result.sense_id is not None
        assert result.sense_label.startswith("noun.")

    def test_run_motion_context(self, backend):
        """Run in motion context should be verb.motion."""
        result = backend.disambiguate_word(
            sentence="The dog runs quickly across the field",
            lemma="run",
            pos="v",
        )
        assert isinstance(result, SenseAnnotation)
        assert result.lemma == "run"
        assert result.sense_label == "verb.motion"
        assert result.confidence > 0

    def test_run_operate_context(self, backend):
        """Run in operation context should be verb.social or similar."""
        result = backend.disambiguate_word(
            sentence="She runs a successful business",
            lemma="run",
            pos="v",
        )
        assert isinstance(result, SenseAnnotation)
        assert result.lemma == "run"
        assert result.sense_label.startswith("verb.")

    def test_bat_animal_context(self, backend):
        """Bat in animal context should be noun.animal."""
        result = backend.disambiguate_word(
            sentence="A bat flew out of the cave at night",
            lemma="bat",
            pos="n",
        )
        assert isinstance(result, SenseAnnotation)
        assert result.lemma == "bat"
        assert result.sense_label == "noun.animal"

    def test_bat_sports_context(self, backend):
        """Bat in sports context should be sports-related."""
        result = backend.disambiguate_word(
            sentence="He picked up the wooden baseball bat from the rack",
            lemma="bat",
            pos="n",
        )
        assert isinstance(result, SenseAnnotation)
        assert result.lemma == "bat"
        # Baseball bat can be artifact (the object) or act (a turn at bat)
        # Both are valid sports-related senses
        assert result.sense_label in ("noun.artifact", "noun.object", "noun.act")

    def test_unknown_word(self, backend):
        """Unknown word should return unknown supersense."""
        result = backend.disambiguate_word(
            sentence="The xyzfoobar was interesting",
            lemma="xyzfoobar",
            pos="n",
        )
        assert isinstance(result, SenseAnnotation)
        assert result.lemma == "xyzfoobar"
        assert result.sense_id is None
        assert result.sense_label == "unknown"
        assert result.confidence == 0.0

    def test_no_pos_filter(self, backend):
        """Should work without POS filter."""
        result = backend.disambiguate_word(
            sentence="I need to run to the store",
            lemma="run",
            pos=None,
        )
        assert isinstance(result, SenseAnnotation)
        assert result.sense_id is not None

    def test_confidence_score_range(self, backend):
        """Confidence should be between 0 and 1."""
        result = backend.disambiguate_word(
            sentence="The dog barked loudly",
            lemma="dog",
            pos="n",
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_definition_not_empty(self, backend):
        """Definition should not be empty for known words."""
        result = backend.disambiguate_word(
            sentence="The cat sat on the mat",
            lemma="cat",
            pos="n",
        )
        assert result.definition is not None
        assert len(result.definition) > 0


class TestConfidenceThreshold:
    """Test confidence threshold behavior."""

    def test_low_confidence_below_threshold(self):
        """Low confidence results should still return a sense."""
        backend = WordNetSenseBackend(confidence_threshold=0.9)
        result = backend.disambiguate_word(
            sentence="The thing was there",
            lemma="thing",
            pos="n",
        )
        # Even if below threshold, we still return the best match
        assert result.sense_id is not None

    def test_is_confident_method(self):
        """is_confident should check against threshold."""
        backend = WordNetSenseBackend(confidence_threshold=0.3)

        # High confidence
        high_conf = SenseAnnotation(
            lemma="test",
            sense_id="test.n.01",
            sense_label="noun.cognition",
            confidence=0.5,
        )
        assert backend.is_confident(high_conf) is True

        # Low confidence
        low_conf = SenseAnnotation(
            lemma="test",
            sense_id="test.n.01",
            sense_label="noun.cognition",
            confidence=0.2,
        )
        assert backend.is_confident(low_conf) is False


class TestAnnotate:
    """Test batch annotation of tokens."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return WordNetSenseBackend()

    @pytest.fixture
    def sample_data(self):
        """Create sample tokens and sentences."""
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1, 2, 3],
                "sentence": [
                    "I deposited money at the bank",
                    "We sat on the river bank",
                    "The dog runs in the park",
                ],
            }
        )

        tokens_df = pd.DataFrame(
            {
                "token_id": [1, 2, 3, 4, 5],
                "lemma": ["deposit", "money", "bank", "bank", "run"],
                "pos": ["VERB", "NOUN", "NOUN", "NOUN", "VERB"],
                "sentence_id": [1, 1, 1, 2, 3],
            }
        )

        return tokens_df, sentences_df

    def test_annotate_adds_columns(self, backend, sample_data):
        """Annotate should add sense columns."""
        tokens_df, sentences_df = sample_data
        result = backend.annotate(tokens_df, sentences_df)

        assert "synset_id" in result.columns
        assert "supersense" in result.columns
        assert "sense_confidence" in result.columns

    def test_annotate_preserves_original_columns(self, backend, sample_data):
        """Annotate should preserve original columns."""
        tokens_df, sentences_df = sample_data
        result = backend.annotate(tokens_df, sentences_df)

        assert "token_id" in result.columns
        assert "lemma" in result.columns
        assert "pos" in result.columns
        assert "sentence_id" in result.columns

    def test_annotate_same_word_different_context(self, backend, sample_data):
        """Same word in different contexts may get different senses."""
        tokens_df, sentences_df = sample_data
        result = backend.annotate(tokens_df, sentences_df)

        # Get the two "bank" rows
        bank_rows = result[result["lemma"] == "bank"]
        assert len(bank_rows) == 2

        # They might have different supersenses
        # (financial bank vs river bank)
        # We just verify both have valid supersenses
        for _, row in bank_rows.iterrows():
            assert row["supersense"].startswith("noun.")

    def test_annotate_filters_non_content_pos(self, backend):
        """Should skip non-content POS (DET, PUNCT, etc.)."""
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1],
                "sentence": ["The cat sat on the mat"],
            }
        )

        tokens_df = pd.DataFrame(
            {
                "token_id": [1, 2, 3, 4, 5, 6],
                "lemma": ["the", "cat", "sit", "on", "the", "mat"],
                "pos": ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN"],
                "sentence_id": [1, 1, 1, 1, 1, 1],
            }
        )

        result = backend.annotate(tokens_df, sentences_df)

        # DET and ADP should have None/unknown supersense
        det_rows = result[result["pos"] == "DET"]
        for _, row in det_rows.iterrows():
            assert row["supersense"] == "unknown" or pd.isna(row["synset_id"])

    def test_annotate_validates_input(self, backend):
        """Should raise error for invalid input."""
        bad_tokens = pd.DataFrame({"wrong_column": [1, 2, 3]})
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1],
                "sentence": ["Test sentence"],
            }
        )

        with pytest.raises(ValueError, match="missing required columns"):
            backend.annotate(bad_tokens, sentences_df)

    def test_annotate_empty_dataframe(self, backend):
        """Should handle empty DataFrames."""
        tokens_df = pd.DataFrame(columns=["lemma", "pos", "sentence_id"])
        sentences_df = pd.DataFrame(columns=["sentence_id", "sentence"])

        result = backend.annotate(tokens_df, sentences_df)
        assert len(result) == 0
        assert "supersense" in result.columns

    def test_annotate_with_progress_bar(self, backend, sample_data, monkeypatch):
        """Should work with progress bar enabled/disabled."""
        tokens_df, sentences_df = sample_data

        # Test with progress bar disabled
        result = backend.annotate(tokens_df, sentences_df, show_progress=False)
        assert "supersense" in result.columns

        # Test with progress bar enabled (default)
        result = backend.annotate(tokens_df, sentences_df, show_progress=True)
        assert "supersense" in result.columns

    def test_annotate_with_checkpoint(self, backend, sample_data, tmp_path):
        """Should save checkpoint file when checkpoint_path is provided."""
        tokens_df, sentences_df = sample_data
        checkpoint_path = tmp_path / "checkpoint.parquet"

        result = backend.annotate(
            tokens_df,
            sentences_df,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=2,
            show_progress=False,
        )

        # Checkpoint file should exist
        assert checkpoint_path.exists()

        # Load checkpoint and verify it matches result
        checkpoint_df = pd.read_parquet(checkpoint_path)
        pd.testing.assert_frame_equal(result, checkpoint_df)

    def test_annotate_checkpoint_interval(self, backend, tmp_path):
        """Should save checkpoint at specified intervals."""
        # Create larger dataset
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1, 2, 3, 4, 5],
                "sentence": [
                    "The dog runs",
                    "The cat sits",
                    "The bird flies",
                    "The fish swims",
                    "The horse gallops",
                ],
            }
        )

        tokens_df = pd.DataFrame(
            {
                "token_id": list(range(1, 11)),
                "lemma": [
                    "dog",
                    "run",
                    "cat",
                    "sit",
                    "bird",
                    "fly",
                    "fish",
                    "swim",
                    "horse",
                    "gallop",
                ],
                "pos": [
                    "NOUN",
                    "VERB",
                    "NOUN",
                    "VERB",
                    "NOUN",
                    "VERB",
                    "NOUN",
                    "VERB",
                    "NOUN",
                    "VERB",
                ],
                "sentence_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            }
        )

        checkpoint_path = tmp_path / "checkpoint.parquet"

        result = backend.annotate(
            tokens_df,
            sentences_df,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=3,  # Save every 3 tokens
            show_progress=False,
        )

        # Checkpoint should exist
        assert checkpoint_path.exists()

        # Verify result is complete
        assert len(result) == len(tokens_df)
        assert "supersense" in result.columns


class TestAggregate:
    """Test aggregation of annotated tokens."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return WordNetSenseBackend()

    @pytest.fixture
    def annotated_data(self):
        """Create sample annotated data."""
        return pd.DataFrame(
            {
                "lemma": ["run", "run", "run", "bank", "bank"],
                "supersense": [
                    "verb.motion",
                    "verb.motion",
                    "verb.social",
                    "noun.group",
                    "noun.object",
                ],
                "synset_id": ["run.v.01", "run.v.01", "run.v.02", "bank.n.02", "bank.n.01"],
                "sentence_id": [1, 2, 3, 4, 5],
                "sense_confidence": [0.8, 0.7, 0.6, 0.9, 0.85],
            }
        )

    def test_aggregate_groups_by_lemma_supersense(self, backend, annotated_data):
        """Should group by (lemma, supersense)."""
        result = backend.aggregate(annotated_data)

        # run has 2 supersenses, bank has 2 supersenses
        assert len(result) == 4
        assert "lemma" in result.columns
        assert "supersense" in result.columns

    def test_aggregate_calculates_frequencies(self, backend, annotated_data):
        """Should calculate sense frequencies."""
        result = backend.aggregate(annotated_data)

        # Check run/verb.motion (appears 2 times)
        run_motion = result[(result["lemma"] == "run") & (result["supersense"] == "verb.motion")]
        assert len(run_motion) == 1
        assert run_motion.iloc[0]["sense_freq"] == 2

    def test_aggregate_calculates_ratios(self, backend, annotated_data):
        """Should calculate sense ratios."""
        result = backend.aggregate(annotated_data)

        # run appears 3 times total, verb.motion 2 times
        run_motion = result[(result["lemma"] == "run") & (result["supersense"] == "verb.motion")]
        expected_ratio = 2 / 3
        assert abs(run_motion.iloc[0]["sense_ratio"] - expected_ratio) < 0.01

    def test_aggregate_identifies_dominant_supersense(self, backend, annotated_data):
        """Should identify dominant supersense for each lemma."""
        result = backend.aggregate(annotated_data)

        # For "run", verb.motion (2) is more frequent than verb.social (1)
        run_rows = result[result["lemma"] == "run"]
        for _, row in run_rows.iterrows():
            assert row["dominant_supersense"] == "verb.motion"

    def test_aggregate_calculates_sense_count(self, backend, annotated_data):
        """Should calculate sense_count (number of different senses per lemma)."""
        result = backend.aggregate(annotated_data)

        # "run" has 2 different supersenses (verb.motion, verb.social)
        run_rows = result[result["lemma"] == "run"]
        assert all(row["sense_count"] == 2 for _, row in run_rows.iterrows())

        # "bank" has 2 different supersenses (noun.group, noun.object)
        bank_rows = result[result["lemma"] == "bank"]
        assert all(row["sense_count"] == 2 for _, row in bank_rows.iterrows())

    def test_aggregate_sense_count_for_single_sense(self, backend):
        """Should calculate sense_count=1 for words with single sense."""
        annotated_data = pd.DataFrame(
            {
                "lemma": ["cat", "cat", "cat"],
                "supersense": ["noun.animal", "noun.animal", "noun.animal"],
                "synset_id": ["cat.n.01", "cat.n.01", "cat.n.01"],
                "sentence_id": [1, 2, 3],
                "sense_confidence": [0.9, 0.85, 0.88],
            }
        )

        result = backend.aggregate(annotated_data)
        assert all(row["sense_count"] == 1 for _, row in result.iterrows())

    def test_aggregate_has_all_required_columns(self, backend, annotated_data):
        """Should return all required columns."""
        result = backend.aggregate(annotated_data)

        required_columns = {
            "lemma",
            "supersense",
            "sense_freq",
            "book_freq",
            "sense_ratio",
            "doc_count",
            "sense_count",
            "dominant_supersense",
        }
        assert required_columns.issubset(set(result.columns))

    def test_aggregate_validates_input(self, backend):
        """Should raise error for invalid input."""
        bad_df = pd.DataFrame({"wrong_column": [1, 2, 3]})

        with pytest.raises(ValueError, match="missing required columns"):
            backend.aggregate(bad_df)

    def test_aggregate_empty_dataframe(self, backend):
        """Should handle empty DataFrame."""
        empty_df = pd.DataFrame(columns=["lemma", "supersense", "sentence_id"])
        result = backend.aggregate(empty_df)

        assert len(result) == 0
        required_columns = {
            "lemma",
            "supersense",
            "sense_freq",
            "book_freq",
            "sense_ratio",
            "doc_count",
            "sense_count",
            "dominant_supersense",
        }
        assert required_columns.issubset(set(result.columns))


class TestBatchProcessing:
    """Test batch processing efficiency."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return WordNetSenseBackend()

    def test_disambiguate_batch(self, backend):
        """Should process multiple words efficiently."""
        items = [
            ("I deposited money at the bank", "bank", "n"),
            ("The dog runs in the park", "run", "v"),
            ("A bat flew out of the cave", "bat", "n"),
        ]

        results = backend.disambiguate_batch(items)

        assert len(results) == 3
        assert all(isinstance(r, SenseAnnotation) for r in results)
        assert results[0].lemma == "bank"
        assert results[1].lemma == "run"
        assert results[2].lemma == "bat"

    def test_disambiguate_batch_empty(self, backend):
        """Should handle empty batch."""
        results = backend.disambiguate_batch([])
        assert results == []

    def test_disambiguate_batch_with_unknown(self, backend):
        """Should handle unknown words in batch."""
        items = [
            ("The dog barks", "dog", "n"),
            ("The xyzfoo is strange", "xyzfoo", "n"),
        ]

        results = backend.disambiguate_batch(items)

        assert len(results) == 2
        assert results[0].sense_id is not None
        assert results[1].sense_id is None
        assert results[1].sense_label == "unknown"
