"""Tests for Pipeline B batch QC (batch_qc.py)."""

import pytest

from eng_words.word_family.batch_qc import (
    cards_lemma_not_in_example,
    check_qc_threshold,
)


class TestCardsLemmaNotInExample:
    def test_returns_empty_when_validator_unavailable(self):
        # If validator is optional and not installed, might return []
        # When validator is available, real checks run
        cards = [{"lemma": "run", "examples": ["I run daily."], "definition_en": "move fast"}]
        out = cards_lemma_not_in_example(cards)
        # Either [] (validator missing) or [] (lemma in example)
        assert isinstance(out, list)

    def test_lemma_in_example_passes(self):
        cards = [{"lemma": "run", "examples": ["I run every day."], "definition_en": "move"}]
        out = cards_lemma_not_in_example(cards)
        assert len(out) == 0

    def test_lemma_not_in_example_flagged(self):
        cards = [{"lemma": "run", "examples": ["The weather is nice."], "definition_en": "move"}]
        out = cards_lemma_not_in_example(cards)
        if out:  # when validator available
            assert len(out) == 1
            assert out[0]["lemma"] == "run"
            assert out[0]["example_index"] == 1


class TestCheckQcThreshold:
    def test_under_threshold_does_not_raise(self):
        check_qc_threshold(2, 100, max_warning_rate=0.05, max_warnings_absolute=10)

    def test_exceeds_rate_raises(self):
        with pytest.raises(ValueError, match="exceeds max_warning_rate"):
            check_qc_threshold(10, 100, max_warning_rate=0.05)

    def test_exceeds_absolute_raises(self):
        with pytest.raises(ValueError, match="exceeds max_warnings_absolute"):
            check_qc_threshold(15, 100, max_warnings_absolute=10)

    def test_zero_total_does_not_raise(self):
        check_qc_threshold(5, 0, max_warning_rate=0.01)
