"""Aggregation of multiple LLM labels for WSD Gold Dataset.

This module provides functions to combine labels from multiple
LLM judges into a single gold label using majority voting.
"""

from collections import Counter

from eng_words.wsd_gold.models import GoldLabel, ModelOutput

# Default confidence threshold for determining low confidence
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Flags that indicate uncertainty and require referee
PROBLEMATIC_FLAGS = {"needs_more_context", "none_of_the_above"}


def aggregate_labels(outputs: list[ModelOutput]) -> GoldLabel:
    """Aggregate labels from multiple LLM judges into one gold label.

    Uses majority voting with tie-break by confidence.

    Args:
        outputs: List of ModelOutput from different LLM judges

    Returns:
        GoldLabel with aggregated result

    Raises:
        ValueError: If outputs list is empty
    """
    if not outputs:
        raise ValueError("Cannot aggregate empty list of outputs")

    # Single judge case
    if len(outputs) == 1:
        output = outputs[0]
        return GoldLabel(
            synset_id=output.chosen_synset_id,
            confidence=output.confidence,
            agreement_ratio=1.0,
            flags=output.flags.copy(),
            needs_referee=True,  # Single judge is uncertain
        )

    # Count votes for each synset
    vote_counts = Counter(o.chosen_synset_id for o in outputs)
    most_common = vote_counts.most_common()

    # Check for majority
    top_synset, top_count = most_common[0]
    agreement_ratio = top_count / len(outputs)

    # Check if there's a tie at the top
    has_tie = len(most_common) > 1 and most_common[0][1] == most_common[1][1]

    # Collect all flags
    all_flags: list[str] = []
    for o in outputs:
        for flag in o.flags:
            if flag not in all_flags:
                all_flags.append(flag)

    # Determine needs_referee
    needs_referee = _check_needs_referee(outputs, agreement_ratio, has_tie, all_flags)

    # Determine final synset
    if has_tie:
        # Tie-break by confidence
        tied_synsets = {s for s, c in most_common if c == top_count}
        final_synset, final_confidence = _tie_break_by_confidence(outputs, tied_synsets)
    else:
        final_synset = top_synset
        # Calculate average confidence of agreeing judges
        agreeing_confidences = [o.confidence for o in outputs if o.chosen_synset_id == final_synset]
        final_confidence = sum(agreeing_confidences) / len(agreeing_confidences)

    return GoldLabel(
        synset_id=final_synset,
        confidence=final_confidence,
        agreement_ratio=agreement_ratio,
        flags=all_flags,
        needs_referee=needs_referee,
    )


def _check_needs_referee(
    outputs: list[ModelOutput],
    agreement_ratio: float,
    has_tie: bool,
    all_flags: list[str],
) -> bool:
    """Check if the aggregation needs a referee.

    Args:
        outputs: List of outputs
        agreement_ratio: Ratio of judges agreeing
        has_tie: Whether there's a tie
        all_flags: All collected flags

    Returns:
        True if referee is needed
    """
    # No majority (all different for 3 judges, or tie for 2)
    if agreement_ratio < 0.5 or has_tie:
        return True

    # Problematic flags
    for flag in all_flags:
        if flag in PROBLEMATIC_FLAGS:
            return True

    # All low confidence
    if all(o.confidence < DEFAULT_CONFIDENCE_THRESHOLD for o in outputs):
        return True

    return False


def _tie_break_by_confidence(
    outputs: list[ModelOutput],
    tied_synsets: set[str],
) -> tuple[str, float]:
    """Break tie by selecting highest confidence.

    Args:
        outputs: List of outputs
        tied_synsets: Set of synsets with same vote count

    Returns:
        Tuple of (selected synset, confidence)
    """
    best_synset = ""
    best_confidence = -1.0

    for output in outputs:
        if output.chosen_synset_id in tied_synsets:
            if output.confidence > best_confidence:
                best_synset = output.chosen_synset_id
                best_confidence = output.confidence

    return best_synset, best_confidence


def needs_third_judge(
    output_a: ModelOutput,
    output_b: ModelOutput,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> bool:
    """Determine if a third judge is needed.

    Args:
        output_a: First judge's output
        output_b: Second judge's output
        confidence_threshold: Minimum confidence for certainty (default 0.6)

    Returns:
        True if third judge is needed
    """
    # Disagreement
    if output_a.chosen_synset_id != output_b.chosen_synset_id:
        return True

    # Low confidence for either judge
    if output_a.confidence < confidence_threshold or output_b.confidence < confidence_threshold:
        return True

    # Problematic flags
    for output in [output_a, output_b]:
        for flag in output.flags:
            if flag in PROBLEMATIC_FLAGS:
                return True

    return False


def get_aggregation_stats(labels: list[GoldLabel]) -> dict:
    """Get statistics for aggregated labels.

    Args:
        labels: List of GoldLabel instances

    Returns:
        Dictionary with statistics
    """
    if not labels:
        return {
            "total": 0,
            "needs_referee_count": 0,
            "needs_referee_ratio": 0.0,
            "avg_agreement": 0.0,
            "avg_confidence": 0.0,
        }

    total = len(labels)
    needs_referee_count = sum(1 for label in labels if label.needs_referee)
    avg_agreement = sum(label.agreement_ratio for label in labels) / total
    avg_confidence = sum(label.confidence for label in labels) / total

    return {
        "total": total,
        "needs_referee_count": needs_referee_count,
        "needs_referee_ratio": needs_referee_count / total,
        "avg_agreement": avg_agreement,
        "avg_confidence": avg_confidence,
    }
