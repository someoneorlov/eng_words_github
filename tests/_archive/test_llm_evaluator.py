"""Tests for WSD Evaluator."""

import pandas as pd
import pytest

from eng_words.constants import LEMMA, POS, SENTENCE_ID, SUPERSENSE, SYNSET_ID


class TestSampleTokensForEvaluation:
    """Tests for sampling tokens for evaluation."""

    def test_import_function(self):
        """Test that function can be imported."""
        from eng_words.llm.evaluator import sample_tokens_for_evaluation

        assert callable(sample_tokens_for_evaluation)

    def test_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        from eng_words.llm.evaluator import sample_tokens_for_evaluation

        sense_tokens_df = pd.DataFrame(
            {
                LEMMA: ["run", "bank", "play"],
                POS: ["VERB", "NOUN", "VERB"],
                SYNSET_ID: ["run.v.01", "bank.n.01", "play.v.01"],
                SUPERSENSE: ["verb.motion", "noun.artifact", "verb.creation"],
                SENTENCE_ID: [1, 2, 3],
            }
        )
        sentences_df = pd.DataFrame(
            {
                SENTENCE_ID: [1, 2, 3],
                "text": [
                    "He runs quickly.",
                    "The bank is open.",
                    "They play music.",
                ],
            }
        )

        result = sample_tokens_for_evaluation(sense_tokens_df, sentences_df, n_samples=3)
        assert isinstance(result, pd.DataFrame)

    def test_returns_correct_columns(self):
        """Test that result has required columns."""
        from eng_words.llm.evaluator import sample_tokens_for_evaluation

        sense_tokens_df = pd.DataFrame(
            {
                LEMMA: ["run", "bank"],
                POS: ["VERB", "NOUN"],
                SYNSET_ID: ["run.v.01", "bank.n.01"],
                SUPERSENSE: ["verb.motion", "noun.artifact"],
                SENTENCE_ID: [1, 2],
            }
        )
        sentences_df = pd.DataFrame(
            {
                SENTENCE_ID: [1, 2],
                "text": ["He runs quickly.", "The bank is open."],
            }
        )

        result = sample_tokens_for_evaluation(sense_tokens_df, sentences_df, n_samples=2)
        required_columns = [LEMMA, POS, SYNSET_ID, SUPERSENSE, "sentence_text"]
        for col in required_columns:
            assert col in result.columns

    def test_respects_n_samples(self):
        """Test that function respects n_samples parameter."""
        from eng_words.llm.evaluator import sample_tokens_for_evaluation

        sense_tokens_df = pd.DataFrame(
            {
                LEMMA: [f"word{i}" for i in range(100)],
                POS: ["NOUN"] * 100,
                SYNSET_ID: [f"word{i}.n.01" for i in range(100)],
                SUPERSENSE: ["noun.artifact"] * 100,
                SENTENCE_ID: list(range(100)),
            }
        )
        sentences_df = pd.DataFrame(
            {
                SENTENCE_ID: list(range(100)),
                "text": [f"Sentence {i}." for i in range(100)],
            }
        )

        result = sample_tokens_for_evaluation(sense_tokens_df, sentences_df, n_samples=20)
        assert len(result) == 20

    def test_stratifies_by_supersense(self):
        """Test that sampling is stratified by supersense."""
        from eng_words.llm.evaluator import sample_tokens_for_evaluation

        # 50 verb.motion, 50 noun.artifact
        sense_tokens_df = pd.DataFrame(
            {
                LEMMA: [f"word{i}" for i in range(100)],
                POS: ["VERB"] * 50 + ["NOUN"] * 50,
                SYNSET_ID: [f"word{i}.v.01" for i in range(50)]
                + [f"word{i}.n.01" for i in range(50, 100)],
                SUPERSENSE: ["verb.motion"] * 50 + ["noun.artifact"] * 50,
                SENTENCE_ID: list(range(100)),
            }
        )
        sentences_df = pd.DataFrame(
            {
                SENTENCE_ID: list(range(100)),
                "text": [f"Sentence {i}." for i in range(100)],
            }
        )

        result = sample_tokens_for_evaluation(sense_tokens_df, sentences_df, n_samples=20)

        # Should have roughly equal representation
        verb_count = (result[SUPERSENSE] == "verb.motion").sum()
        noun_count = (result[SUPERSENSE] == "noun.artifact").sum()
        assert verb_count > 0
        assert noun_count > 0
        # Roughly equal (within 5)
        assert abs(verb_count - noun_count) <= 10

    def test_handles_less_samples_than_requested(self):
        """Test graceful handling when fewer samples available."""
        from eng_words.llm.evaluator import sample_tokens_for_evaluation

        sense_tokens_df = pd.DataFrame(
            {
                LEMMA: ["run", "bank"],
                POS: ["VERB", "NOUN"],
                SYNSET_ID: ["run.v.01", "bank.n.01"],
                SUPERSENSE: ["verb.motion", "noun.artifact"],
                SENTENCE_ID: [1, 2],
            }
        )
        sentences_df = pd.DataFrame(
            {
                SENTENCE_ID: [1, 2],
                "text": ["He runs quickly.", "The bank is open."],
            }
        )

        result = sample_tokens_for_evaluation(sense_tokens_df, sentences_df, n_samples=100)
        assert len(result) == 2  # Only 2 available

    def test_random_state_reproducibility(self):
        """Test that random_state produces reproducible results."""
        from eng_words.llm.evaluator import sample_tokens_for_evaluation

        sense_tokens_df = pd.DataFrame(
            {
                LEMMA: [f"word{i}" for i in range(50)],
                POS: ["NOUN"] * 50,
                SYNSET_ID: [f"word{i}.n.01" for i in range(50)],
                SUPERSENSE: ["noun.artifact"] * 50,
                SENTENCE_ID: list(range(50)),
            }
        )
        sentences_df = pd.DataFrame(
            {
                SENTENCE_ID: list(range(50)),
                "text": [f"Sentence {i}." for i in range(50)],
            }
        )

        result1 = sample_tokens_for_evaluation(
            sense_tokens_df, sentences_df, n_samples=10, random_state=42
        )
        result2 = sample_tokens_for_evaluation(
            sense_tokens_df, sentences_df, n_samples=10, random_state=42
        )

        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True), result2.reset_index(drop=True)
        )


class TestWSDEvaluator:
    """Tests for WSDEvaluator class."""

    def test_import_class(self):
        """Test that WSDEvaluator can be imported."""
        from eng_words.llm.evaluator import WSDEvaluator

        assert WSDEvaluator is not None

    def test_requires_providers(self):
        """Test that evaluator requires at least one provider."""
        from eng_words.llm.evaluator import WSDEvaluator

        with pytest.raises(ValueError, match="At least one provider"):
            WSDEvaluator(providers=[])

    def test_accepts_single_provider(self):
        """Test that evaluator accepts a single provider."""
        from unittest.mock import MagicMock

        from eng_words.llm.evaluator import WSDEvaluator

        mock_provider = MagicMock()
        evaluator = WSDEvaluator(providers=[mock_provider])
        assert len(evaluator.providers) == 1

    def test_accepts_multiple_providers(self):
        """Test that evaluator accepts multiple providers (jury)."""
        from unittest.mock import MagicMock

        from eng_words.llm.evaluator import WSDEvaluator

        mock_providers = [MagicMock(), MagicMock()]
        evaluator = WSDEvaluator(providers=mock_providers)
        assert len(evaluator.providers) == 2


class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_import_function(self):
        """Test that function can be imported."""
        from eng_words.llm.evaluator import compute_metrics

        assert callable(compute_metrics)

    def test_all_correct(self):
        """Test metrics when all results are correct."""
        from eng_words.llm.evaluator import EvaluationResult, compute_metrics

        results = [
            EvaluationResult(
                lemma="run",
                sentence_text="He runs.",
                assigned_synset="run.v.01",
                jury_votes=["correct", "correct"],
                jury_verdict="correct",
                confidence=0.9,
            ),
            EvaluationResult(
                lemma="bank",
                sentence_text="The bank.",
                assigned_synset="bank.n.01",
                jury_votes=["correct", "correct"],
                jury_verdict="correct",
                confidence=0.95,
            ),
        ]

        metrics = compute_metrics(results)

        assert metrics["total_samples"] == 2
        assert metrics["accuracy_strict"] == 1.0
        assert metrics["coverage"] == 1.0
        assert metrics["correct_count"] == 2
        assert metrics["incorrect_count"] == 0
        assert metrics["uncertain_count"] == 0

    def test_all_incorrect(self):
        """Test metrics when all results are incorrect."""
        from eng_words.llm.evaluator import EvaluationResult, compute_metrics

        results = [
            EvaluationResult(
                lemma="run",
                sentence_text="He runs.",
                assigned_synset="run.v.01",
                jury_votes=["incorrect", "incorrect"],
                jury_verdict="incorrect",
                confidence=0.8,
            ),
        ]

        metrics = compute_metrics(results)

        assert metrics["accuracy_strict"] == 0.0
        assert metrics["correct_count"] == 0
        assert metrics["incorrect_count"] == 1

    def test_mixed_results(self):
        """Test metrics with mixed results."""
        from eng_words.llm.evaluator import EvaluationResult, compute_metrics

        results = [
            EvaluationResult(
                lemma="a",
                sentence_text="A.",
                assigned_synset="a.n.01",
                jury_verdict="correct",
                confidence=0.9,
            ),
            EvaluationResult(
                lemma="b",
                sentence_text="B.",
                assigned_synset="b.n.01",
                jury_verdict="incorrect",
                confidence=0.8,
            ),
            EvaluationResult(
                lemma="c",
                sentence_text="C.",
                assigned_synset="c.n.01",
                jury_verdict="uncertain",
                confidence=0.5,
            ),
        ]

        metrics = compute_metrics(results)

        assert metrics["total_samples"] == 3
        assert metrics["correct_count"] == 1
        assert metrics["incorrect_count"] == 1
        assert metrics["uncertain_count"] == 1
        # accuracy_strict = correct / (correct + incorrect)
        assert metrics["accuracy_strict"] == 0.5
        # coverage = (correct + incorrect) / total
        assert metrics["coverage"] == 2 / 3

    def test_empty_results(self):
        """Test metrics with empty results."""
        from eng_words.llm.evaluator import compute_metrics

        metrics = compute_metrics([])

        assert metrics["total_samples"] == 0
        assert metrics["accuracy_strict"] == 0.0
        assert metrics["coverage"] == 0.0

    def test_average_confidence(self):
        """Test average confidence calculation."""
        from eng_words.llm.evaluator import EvaluationResult, compute_metrics

        results = [
            EvaluationResult(
                lemma="a",
                sentence_text="A.",
                assigned_synset="a.n.01",
                jury_verdict="correct",
                confidence=0.8,
            ),
            EvaluationResult(
                lemma="b",
                sentence_text="B.",
                assigned_synset="b.n.01",
                jury_verdict="correct",
                confidence=1.0,
            ),
        ]

        metrics = compute_metrics(results)

        assert metrics["avg_confidence"] == 0.9
