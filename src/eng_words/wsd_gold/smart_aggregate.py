"""Smart aggregation with referee logic for WSD Gold Dataset.

This module implements cost-efficient aggregation:
1. Primary judges: Anthropic + Gemini
2. If they agree → use their answer
3. If they disagree → call OpenAI as referee
4. 2/3 agree → majority vote
5. All different → trust Anthropic (most reliable from pilot)
"""

from dataclasses import dataclass

from eng_words.wsd_gold.models import GoldLabel, ModelOutput


@dataclass
class SmartAggregationResult:
    """Result of smart aggregation.

    Attributes:
        label: The final GoldLabel
        used_referee: Whether OpenAI referee was called
        agreement_type: 'full' | 'majority' | 'anthropic_fallback'
        primary_outputs: Dict of primary judge outputs
        referee_output: Optional referee output
    """

    label: GoldLabel
    used_referee: bool
    agreement_type: str
    primary_outputs: dict[str, ModelOutput]
    referee_output: ModelOutput | None = None


def smart_aggregate(
    anthropic_output: ModelOutput,
    gemini_output: ModelOutput,
    openai_output: ModelOutput | None = None,
) -> SmartAggregationResult:
    """Aggregate labels using smart referee logic.

    Args:
        anthropic_output: Output from Anthropic (primary, most trusted)
        gemini_output: Output from Gemini (primary, cheapest)
        openai_output: Output from OpenAI (referee, only if needed)

    Returns:
        SmartAggregationResult with final label and metadata
    """
    a_synset = anthropic_output.chosen_synset_id
    g_synset = gemini_output.chosen_synset_id

    primary_outputs = {
        "anthropic": anthropic_output,
        "gemini": gemini_output,
    }

    # Case 1: Primary judges agree
    if a_synset == g_synset:
        # Combine flags
        all_flags = list(set(anthropic_output.flags + gemini_output.flags))

        label = GoldLabel(
            synset_id=a_synset,
            confidence=1.0,  # Full agreement = max confidence
            agreement_ratio=1.0,
            flags=all_flags,
            needs_referee=False,
        )

        return SmartAggregationResult(
            label=label,
            used_referee=False,
            agreement_type="full",
            primary_outputs=primary_outputs,
            referee_output=None,
        )

    # Case 2: Primary judges disagree - need referee
    if openai_output is None:
        # Return preliminary result indicating referee is needed
        label = GoldLabel(
            synset_id=a_synset,  # Anthropic as preliminary
            confidence=0.5,
            agreement_ratio=0.5,
            flags=["needs_referee"],
            needs_referee=True,
        )

        return SmartAggregationResult(
            label=label,
            used_referee=False,
            agreement_type="pending_referee",
            primary_outputs=primary_outputs,
            referee_output=None,
        )

    # Case 3: Have referee - do majority vote
    o_synset = openai_output.chosen_synset_id

    # Count votes
    votes = {a_synset: 1, g_synset: 1}
    votes[o_synset] = votes.get(o_synset, 0) + 1

    # Find winner
    max_votes = max(votes.values())

    if max_votes >= 2:
        # Majority vote
        winner = [s for s, v in votes.items() if v == max_votes][0]
        all_flags = list(set(anthropic_output.flags + gemini_output.flags + openai_output.flags))

        label = GoldLabel(
            synset_id=winner,
            confidence=max_votes / 3,  # 2/3 = 0.67, 3/3 = 1.0
            agreement_ratio=max_votes / 3,
            flags=all_flags,
            needs_referee=False,
        )

        return SmartAggregationResult(
            label=label,
            used_referee=True,
            agreement_type="majority",
            primary_outputs=primary_outputs,
            referee_output=openai_output,
        )

    else:
        # All different - trust Anthropic
        all_flags = list(
            set(
                anthropic_output.flags
                + gemini_output.flags
                + openai_output.flags
                + ["all_disagree"]
            )
        )

        label = GoldLabel(
            synset_id=a_synset,
            confidence=1 / 3,  # Low confidence
            agreement_ratio=1 / 3,
            flags=all_flags,
            needs_referee=False,
        )

        return SmartAggregationResult(
            label=label,
            used_referee=True,
            agreement_type="anthropic_fallback",
            primary_outputs=primary_outputs,
            referee_output=openai_output,
        )


def needs_referee(
    anthropic_output: ModelOutput,
    gemini_output: ModelOutput,
) -> bool:
    """Check if referee is needed.

    Args:
        anthropic_output: Output from Anthropic
        gemini_output: Output from Gemini

    Returns:
        True if outputs disagree
    """
    return anthropic_output.chosen_synset_id != gemini_output.chosen_synset_id


@dataclass
class SmartAggregationStats:
    """Statistics for smart aggregation run."""

    total: int
    full_agreement: int
    majority_vote: int
    anthropic_fallback: int
    referee_calls: int
    referee_rate: float

    @property
    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Total: {self.total}, "
            f"Full agreement: {self.full_agreement} ({100*self.full_agreement/self.total:.0f}%), "
            f"Referee calls: {self.referee_calls} ({100*self.referee_rate:.0f}%)"
        )


def get_smart_aggregation_stats(results: list[SmartAggregationResult]) -> SmartAggregationStats:
    """Calculate statistics for smart aggregation results.

    Args:
        results: List of SmartAggregationResult

    Returns:
        SmartAggregationStats with counts and rates
    """
    total = len(results)
    if total == 0:
        return SmartAggregationStats(
            total=0,
            full_agreement=0,
            majority_vote=0,
            anthropic_fallback=0,
            referee_calls=0,
            referee_rate=0.0,
        )

    full_agreement = sum(1 for r in results if r.agreement_type == "full")
    majority_vote = sum(1 for r in results if r.agreement_type == "majority")
    anthropic_fallback = sum(1 for r in results if r.agreement_type == "anthropic_fallback")
    referee_calls = sum(1 for r in results if r.used_referee)

    return SmartAggregationStats(
        total=total,
        full_agreement=full_agreement,
        majority_vote=majority_vote,
        anthropic_fallback=anthropic_fallback,
        referee_calls=referee_calls,
        referee_rate=referee_calls / total,
    )
