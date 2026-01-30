"""Tests for WSD Gold Dataset label aggregation.

Following TDD: tests are written before implementation.
"""

import pytest

from eng_words.wsd_gold.models import GoldLabel, LLMUsage, ModelOutput


def make_output(
    synset_id: str = "bank.n.01",
    confidence: float = 0.9,
    flags: list[str] | None = None,
) -> ModelOutput:
    """Helper to create test ModelOutput."""
    return ModelOutput(
        chosen_synset_id=synset_id,
        confidence=confidence,
        flags=flags or [],
        raw_text="{}",
        usage=LLMUsage(input_tokens=100, output_tokens=20),
    )


class TestAggregateLables:
    """Tests for aggregate_labels function."""

    def test_import_function(self):
        """Can import aggregate_labels."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        assert aggregate_labels is not None

    def test_two_judges_agree(self):
        """2/2 judges agree → majority vote."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.9),
            make_output("bank.n.01", 0.85),
        ]

        result = aggregate_labels(outputs)

        assert isinstance(result, GoldLabel)
        assert result.synset_id == "bank.n.01"
        assert result.agreement_ratio == 1.0
        assert result.needs_referee is False

    def test_three_judges_majority(self):
        """2/3 judges agree → majority vote."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.9),
            make_output("bank.n.01", 0.8),
            make_output("bank.n.02", 0.7),
        ]

        result = aggregate_labels(outputs)

        assert result.synset_id == "bank.n.01"
        assert result.agreement_ratio == pytest.approx(2 / 3)
        assert result.needs_referee is False

    def test_three_judges_unanimous(self):
        """3/3 judges agree → unanimous."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.95),
            make_output("bank.n.01", 0.9),
            make_output("bank.n.01", 0.85),
        ]

        result = aggregate_labels(outputs)

        assert result.synset_id == "bank.n.01"
        assert result.agreement_ratio == 1.0
        assert result.needs_referee is False

    def test_all_different_needs_referee(self):
        """All judges disagree → needs_referee=True."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.8),
            make_output("bank.n.02", 0.7),
            make_output("bank.n.03", 0.6),
        ]

        result = aggregate_labels(outputs)

        assert result.needs_referee is True
        assert result.agreement_ratio == pytest.approx(1 / 3)

    def test_tie_break_by_confidence(self):
        """2 judges disagree → tie-break by max confidence."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.7),
            make_output("bank.n.02", 0.95),  # Higher confidence wins
        ]

        result = aggregate_labels(outputs)

        assert result.synset_id == "bank.n.02"
        assert result.needs_referee is True  # Disagreement

    def test_low_confidence_needs_referee(self):
        """All judges have low confidence → needs_referee."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.4),
            make_output("bank.n.01", 0.5),
        ]

        result = aggregate_labels(outputs)

        assert result.synset_id == "bank.n.01"
        assert result.needs_referee is True  # Low confidence

    def test_problematic_flags_needs_referee(self):
        """Problematic flags → needs_referee."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.9, ["needs_more_context"]),
            make_output("bank.n.01", 0.85),
        ]

        result = aggregate_labels(outputs)

        assert result.synset_id == "bank.n.01"
        assert result.needs_referee is True

    def test_none_of_the_above_needs_referee(self):
        """none_of_the_above flag → needs_referee."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("", 0.8, ["none_of_the_above"]),
            make_output("bank.n.01", 0.75),
        ]

        result = aggregate_labels(outputs)

        assert result.needs_referee is True

    def test_aggregated_confidence(self):
        """Aggregated confidence is average of agreeing judges."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.9),
            make_output("bank.n.01", 0.8),
            make_output("bank.n.02", 0.7),
        ]

        result = aggregate_labels(outputs)

        # Average of agreeing judges (0.9 + 0.8) / 2
        assert result.confidence == pytest.approx(0.85)

    def test_aggregated_flags(self):
        """Aggregated flags contain all unique flags."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [
            make_output("bank.n.01", 0.9, ["metaphor"]),
            make_output("bank.n.01", 0.85, ["multiword"]),
        ]

        result = aggregate_labels(outputs)

        assert "metaphor" in result.flags
        assert "multiword" in result.flags

    def test_empty_list_raises(self):
        """Empty list raises ValueError."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        with pytest.raises(ValueError):
            aggregate_labels([])

    def test_single_judge(self):
        """Single judge returns that result."""
        from eng_words.wsd_gold.aggregate import aggregate_labels

        outputs = [make_output("bank.n.01", 0.9)]

        result = aggregate_labels(outputs)

        assert result.synset_id == "bank.n.01"
        assert result.agreement_ratio == 1.0
        assert result.needs_referee is True  # Single judge is uncertain


class TestNeedsThirdJudge:
    """Tests for needs_third_judge function."""

    def test_import_function(self):
        """Can import needs_third_judge."""
        from eng_words.wsd_gold.aggregate import needs_third_judge

        assert needs_third_judge is not None

    def test_disagreement_needs_third(self):
        """Disagreement → needs third judge."""
        from eng_words.wsd_gold.aggregate import needs_third_judge

        output_a = make_output("bank.n.01", 0.9)
        output_b = make_output("bank.n.02", 0.85)

        assert needs_third_judge(output_a, output_b) is True

    def test_agreement_high_confidence_no_third(self):
        """Agreement with high confidence → no third judge."""
        from eng_words.wsd_gold.aggregate import needs_third_judge

        output_a = make_output("bank.n.01", 0.9)
        output_b = make_output("bank.n.01", 0.85)

        assert needs_third_judge(output_a, output_b) is False

    def test_agreement_low_confidence_needs_third(self):
        """Agreement but low confidence → needs third judge."""
        from eng_words.wsd_gold.aggregate import needs_third_judge

        output_a = make_output("bank.n.01", 0.5)
        output_b = make_output("bank.n.01", 0.55)

        assert needs_third_judge(output_a, output_b) is True

    def test_needs_more_context_flag(self):
        """needs_more_context flag → needs third judge."""
        from eng_words.wsd_gold.aggregate import needs_third_judge

        output_a = make_output("bank.n.01", 0.9, ["needs_more_context"])
        output_b = make_output("bank.n.01", 0.85)

        assert needs_third_judge(output_a, output_b) is True

    def test_none_of_the_above_flag(self):
        """none_of_the_above flag → needs third judge."""
        from eng_words.wsd_gold.aggregate import needs_third_judge

        output_a = make_output("", 0.8, ["none_of_the_above"])
        output_b = make_output("bank.n.01", 0.85)

        assert needs_third_judge(output_a, output_b) is True

    def test_metaphor_flag_no_third_if_agree(self):
        """metaphor flag alone doesn't require third if agree."""
        from eng_words.wsd_gold.aggregate import needs_third_judge

        output_a = make_output("bank.n.01", 0.9, ["metaphor"])
        output_b = make_output("bank.n.01", 0.85)

        assert needs_third_judge(output_a, output_b) is False

    def test_configurable_threshold(self):
        """Confidence threshold is configurable."""
        from eng_words.wsd_gold.aggregate import needs_third_judge

        output_a = make_output("bank.n.01", 0.65)
        output_b = make_output("bank.n.01", 0.7)

        # Default threshold 0.6 → both above → no third
        assert needs_third_judge(output_a, output_b) is False

        # Higher threshold → needs third
        assert needs_third_judge(output_a, output_b, confidence_threshold=0.75) is True


class TestAggregationStats:
    """Tests for aggregation statistics."""

    def test_import_function(self):
        """Can import get_aggregation_stats."""
        from eng_words.wsd_gold.aggregate import get_aggregation_stats

        assert get_aggregation_stats is not None

    def test_stats_for_labels(self):
        """Get statistics for a list of GoldLabels."""
        from eng_words.wsd_gold.aggregate import aggregate_labels, get_aggregation_stats

        # Create various cases
        outputs_list = [
            # Unanimous
            [make_output("bank.n.01", 0.9), make_output("bank.n.01", 0.85)],
            # Majority
            [
                make_output("bank.n.01", 0.9),
                make_output("bank.n.01", 0.8),
                make_output("bank.n.02", 0.7),
            ],
            # Disagreement
            [make_output("bank.n.01", 0.7), make_output("bank.n.02", 0.8)],
        ]

        labels = [aggregate_labels(outputs) for outputs in outputs_list]
        stats = get_aggregation_stats(labels)

        assert "total" in stats
        assert stats["total"] == 3
        assert "needs_referee_count" in stats
        assert "needs_referee_ratio" in stats
        assert "avg_agreement" in stats
