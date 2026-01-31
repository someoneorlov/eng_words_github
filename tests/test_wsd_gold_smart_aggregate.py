"""Tests for smart aggregation with referee logic."""

import pytest

from eng_words.wsd_gold.models import GoldLabel, LLMUsage, ModelOutput
from eng_words.wsd_gold.smart_aggregate import (
    SmartAggregationResult,
    get_smart_aggregation_stats,
    needs_referee,
    smart_aggregate,
)


def make_output(synset_id: str = "bank.n.01", flags: list[str] | None = None) -> ModelOutput:
    """Helper to create test ModelOutput."""
    return ModelOutput(
        chosen_synset_id=synset_id,
        confidence=0.9,
        flags=flags or [],
        raw_text="{}",
        usage=LLMUsage(input_tokens=100, output_tokens=20),
    )


class TestSmartAggregate:
    """Tests for smart_aggregate function."""

    def test_full_agreement(self):
        """Anthropic + Gemini agree → use their answer."""
        a = make_output("bank.n.01")
        g = make_output("bank.n.01")

        result = smart_aggregate(a, g)

        assert result.label.synset_id == "bank.n.01"
        assert result.used_referee is False
        assert result.agreement_type == "full"
        assert result.label.agreement_ratio == 1.0

    def test_disagreement_needs_referee(self):
        """Anthropic ≠ Gemini without referee → pending."""
        a = make_output("bank.n.01")
        g = make_output("bank.n.02")

        result = smart_aggregate(a, g, openai_output=None)

        assert result.agreement_type == "pending_referee"
        assert result.label.needs_referee is True
        assert result.used_referee is False

    def test_majority_vote_anthropic_openai(self):
        """Anthropic + OpenAI agree → majority vote."""
        a = make_output("bank.n.01")
        g = make_output("bank.n.02")
        o = make_output("bank.n.01")

        result = smart_aggregate(a, g, o)

        assert result.label.synset_id == "bank.n.01"
        assert result.used_referee is True
        assert result.agreement_type == "majority"
        assert result.label.agreement_ratio == pytest.approx(2 / 3)

    def test_majority_vote_gemini_openai(self):
        """Gemini + OpenAI agree → majority vote."""
        a = make_output("bank.n.01")
        g = make_output("bank.n.02")
        o = make_output("bank.n.02")

        result = smart_aggregate(a, g, o)

        assert result.label.synset_id == "bank.n.02"
        assert result.agreement_type == "majority"

    def test_all_different_anthropic_fallback(self):
        """All 3 different → trust Anthropic."""
        a = make_output("bank.n.01")
        g = make_output("bank.n.02")
        o = make_output("bank.n.03")

        result = smart_aggregate(a, g, o)

        assert result.label.synset_id == "bank.n.01"  # Anthropic
        assert result.agreement_type == "anthropic_fallback"
        assert "all_disagree" in result.label.flags

    def test_flags_combined(self):
        """Flags from all judges are combined."""
        a = make_output("bank.n.01", ["metaphor"])
        g = make_output("bank.n.01", ["multiword"])

        result = smart_aggregate(a, g)

        assert "metaphor" in result.label.flags
        assert "multiword" in result.label.flags


class TestNeedsReferee:
    """Tests for needs_referee function."""

    def test_agree_no_referee(self):
        """Agreement → no referee needed."""
        a = make_output("bank.n.01")
        g = make_output("bank.n.01")

        assert needs_referee(a, g) is False

    def test_disagree_needs_referee(self):
        """Disagreement → referee needed."""
        a = make_output("bank.n.01")
        g = make_output("bank.n.02")

        assert needs_referee(a, g) is True


def make_label(synset_id: str = "bank.n.01") -> GoldLabel:
    """Helper to create test GoldLabel."""
    return GoldLabel(
        synset_id=synset_id,
        confidence=0.9,
        agreement_ratio=1.0,
        flags=[],
        needs_referee=False,
    )


class TestSmartAggregationStats:
    """Tests for aggregation statistics."""

    def test_stats_calculation(self):
        """Statistics are calculated correctly."""
        results = [
            SmartAggregationResult(
                label=make_label(),
                used_referee=False,
                agreement_type="full",
                primary_outputs={},
            ),
            SmartAggregationResult(
                label=make_label(),
                used_referee=False,
                agreement_type="full",
                primary_outputs={},
            ),
            SmartAggregationResult(
                label=make_label(),
                used_referee=True,
                agreement_type="majority",
                primary_outputs={},
                referee_output=make_output(),
            ),
        ]

        stats = get_smart_aggregation_stats(results)

        assert stats.total == 3
        assert stats.full_agreement == 2
        assert stats.majority_vote == 1
        assert stats.referee_calls == 1
        assert stats.referee_rate == pytest.approx(1 / 3)

    def test_empty_results(self):
        """Empty results handled correctly."""
        stats = get_smart_aggregation_stats([])

        assert stats.total == 0
        assert stats.referee_rate == 0.0
