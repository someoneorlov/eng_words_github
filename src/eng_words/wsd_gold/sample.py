"""Stratified sampling for WSD Gold Dataset.

This module provides functions to:
- Calculate difficulty features for examples
- Perform stratified sampling by difficulty, POS, and source
- Split examples into dev and test_locked sets
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from eng_words.wsd_gold.models import GoldExample

# Thresholds for difficulty classification
EASY_MAX_SENSE_COUNT = 3
EASY_MIN_MARGIN = 0.3
HARD_MIN_SENSE_COUNT = 7
HARD_MAX_MARGIN = 0.15


@dataclass
class DifficultyFeatures:
    """Features used to determine example difficulty.

    Attributes:
        wn_sense_count: Number of WordNet senses for this lemma+POS
        baseline_margin: Confidence margin of baseline WSD prediction
        pos: Part of speech
        is_multiword: Whether this is a multiword expression
        is_phrasal_verb: Whether this is a phrasal verb
        difficulty_level: Classified difficulty (easy/medium/hard)
    """

    wn_sense_count: int
    baseline_margin: float
    pos: str
    is_multiword: bool
    is_phrasal_verb: bool
    difficulty_level: Literal["easy", "medium", "hard"]


def classify_difficulty(
    sense_count: int,
    margin: float,
    easy_max_sense: int = EASY_MAX_SENSE_COUNT,
    easy_min_margin: float = EASY_MIN_MARGIN,
    hard_min_sense: int = HARD_MIN_SENSE_COUNT,
    hard_max_margin: float = HARD_MAX_MARGIN,
) -> Literal["easy", "medium", "hard"]:
    """Classify difficulty based on sense count and margin.

    Args:
        sense_count: Number of WordNet senses
        margin: Baseline confidence margin (0-1)
        easy_max_sense: Max sense count for easy classification
        easy_min_margin: Min margin for easy classification
        hard_min_sense: Min sense count for hard classification
        hard_max_margin: Max margin for hard classification

    Returns:
        Difficulty level: "easy", "medium", or "hard"
    """
    # Hard if many senses OR very low confidence
    if sense_count >= hard_min_sense or margin < hard_max_margin:
        return "hard"

    # Easy if few senses AND high confidence
    if sense_count <= easy_max_sense and margin >= easy_min_margin:
        return "easy"

    # Everything else is medium
    return "medium"


def _is_phrasal_verb(lemma: str, pos: str) -> bool:
    """Check if lemma is a phrasal verb."""
    if pos.upper() != "VERB":
        return False
    # Phrasal verbs typically have underscore or space
    return "_" in lemma or " " in lemma


def calculate_difficulty_features(example: GoldExample) -> DifficultyFeatures:
    """Calculate difficulty features for an example.

    Args:
        example: GoldExample to analyze

    Returns:
        DifficultyFeatures with calculated values
    """
    metadata = example.metadata
    target = example.target

    is_multiword = metadata.is_multiword
    is_phrasal_verb = _is_phrasal_verb(target.lemma, target.pos)

    difficulty_level = classify_difficulty(
        sense_count=metadata.wn_sense_count,
        margin=metadata.baseline_margin,
    )

    return DifficultyFeatures(
        wn_sense_count=metadata.wn_sense_count,
        baseline_margin=metadata.baseline_margin,
        pos=target.pos,
        is_multiword=is_multiword,
        is_phrasal_verb=is_phrasal_verb,
        difficulty_level=difficulty_level,
    )


def stratified_sample(
    examples: list[GoldExample],
    n: int,
    random_state: int = 42,
) -> list[GoldExample]:
    """Perform stratified sampling of examples.

    Stratifies by difficulty level to ensure representation.

    Args:
        examples: List of examples to sample from
        n: Target number of examples
        random_state: Random seed for reproducibility

    Returns:
        List of sampled examples
    """
    if not examples:
        return []

    if n >= len(examples):
        return examples.copy()

    rng = random.Random(random_state)

    # Group examples by difficulty
    by_difficulty: dict[str, list[GoldExample]] = defaultdict(list)
    for ex in examples:
        features = calculate_difficulty_features(ex)
        by_difficulty[features.difficulty_level].append(ex)

    # Calculate target counts per difficulty
    # Default: proportional to available, but ensure representation
    difficulty_levels = ["easy", "medium", "hard"]
    available = {d: len(by_difficulty[d]) for d in difficulty_levels}
    total_available = sum(available.values())

    if total_available == 0:
        return []

    # Start with proportional allocation
    target_counts: dict[str, int] = {}
    remaining = n

    for difficulty in difficulty_levels:
        if available[difficulty] == 0:
            target_counts[difficulty] = 0
        else:
            # Proportional allocation
            proportion = available[difficulty] / total_available
            count = min(int(n * proportion), available[difficulty])
            target_counts[difficulty] = count
            remaining -= count

    # Distribute remaining to levels that have capacity
    for difficulty in difficulty_levels:
        if remaining <= 0:
            break
        capacity = available[difficulty] - target_counts[difficulty]
        if capacity > 0:
            add = min(remaining, capacity)
            target_counts[difficulty] += add
            remaining -= add

    # Sample from each difficulty level
    sampled: list[GoldExample] = []
    for difficulty in difficulty_levels:
        pool = by_difficulty[difficulty]
        count = target_counts[difficulty]
        if count > 0 and pool:
            rng.shuffle(pool)
            sampled.extend(pool[:count])

    # Shuffle final result
    rng.shuffle(sampled)

    return sampled


def split_by_source(
    examples: list[GoldExample],
    dev_ratio: float = 0.25,
    random_state: int = 42,
) -> tuple[list[GoldExample], list[GoldExample]]:
    """Split examples by source_id to avoid leakage.

    Ensures no source_id appears in both dev and test_locked.

    Args:
        examples: List of examples to split
        dev_ratio: Fraction of sources to put in dev set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (dev, test_locked) lists
    """
    if not examples:
        return [], []

    rng = random.Random(random_state)

    # Group examples by source_id
    by_source: dict[str, list[GoldExample]] = defaultdict(list)
    for ex in examples:
        by_source[ex.source_id].append(ex)

    # Get list of sources and shuffle
    sources = list(by_source.keys())
    rng.shuffle(sources)

    # Calculate split point
    # Use example count to approximate ratio
    total_examples = len(examples)
    target_dev_count = int(total_examples * dev_ratio)

    dev_sources: set[str] = set()
    dev_count = 0

    for source in sources:
        source_count = len(by_source[source])
        if dev_count + source_count <= target_dev_count:
            dev_sources.add(source)
            dev_count += source_count
        elif dev_count == 0:
            # If we haven't added anything yet and this would exceed,
            # add it anyway if it's less than half the target
            if source_count < target_dev_count * 0.5:
                dev_sources.add(source)
                dev_count += source_count

    # Build split lists
    dev: list[GoldExample] = []
    test_locked: list[GoldExample] = []

    for source, exs in by_source.items():
        if source in dev_sources:
            dev.extend(exs)
        else:
            test_locked.extend(exs)

    return dev, test_locked


def get_sampling_stats(examples: list[GoldExample]) -> dict[str, Any]:
    """Calculate statistics for a set of examples.

    Args:
        examples: List of examples

    Returns:
        Dictionary with statistics
    """
    if not examples:
        return {
            "total": 0,
            "by_pos": {},
            "by_difficulty": {},
            "by_source_bucket": {},
        }

    by_pos: dict[str, int] = defaultdict(int)
    by_difficulty: dict[str, int] = defaultdict(int)
    by_source_bucket: dict[str, int] = defaultdict(int)

    for ex in examples:
        by_pos[ex.target.pos] += 1
        by_source_bucket[ex.source_bucket] += 1

        features = calculate_difficulty_features(ex)
        by_difficulty[features.difficulty_level] += 1

    return {
        "total": len(examples),
        "by_pos": dict(by_pos),
        "by_difficulty": dict(by_difficulty),
        "by_source_bucket": dict(by_source_bucket),
    }
