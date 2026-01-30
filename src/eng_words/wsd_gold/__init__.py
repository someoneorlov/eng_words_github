"""WSD Gold Dataset module.

This module provides tools for creating and managing a gold test set
for Word Sense Disambiguation evaluation.
"""

from eng_words.wsd_gold.aggregate import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    PROBLEMATIC_FLAGS,
    aggregate_labels,
    get_aggregation_stats,
    needs_third_judge,
)
from eng_words.wsd_gold.collect import (
    assign_buckets,
    build_example_id,
    calculate_char_span,
    extract_examples_from_tokens,
    get_candidates_for_lemma,
)
from eng_words.wsd_gold.models import (
    VALID_FLAGS,
    Candidate,
    ExampleMetadata,
    GoldExample,
    GoldLabel,
    LLMUsage,
    ModelOutput,
    TargetWord,
)
from eng_words.wsd_gold.sample import (
    DifficultyFeatures,
    calculate_difficulty_features,
    classify_difficulty,
    get_sampling_stats,
    split_by_source,
    stratified_sample,
)
from eng_words.wsd_gold.smart_aggregate import (
    SmartAggregationResult,
    SmartAggregationStats,
    get_smart_aggregation_stats,
    needs_referee,
    smart_aggregate,
)
from eng_words.wsd_gold.eval import (
    classify_example_difficulty,
    compute_metrics,
    compute_metrics_by_segment,
    evaluate_single,
    evaluate_wsd_on_gold,
    get_supersense,
    load_gold_examples,
    run_wsd_prediction,
)
from eng_words.wsd_gold.cache import (
    LLMCache,
    get_cache_key,
)

__all__ = [
    # Models
    "VALID_FLAGS",
    "Candidate",
    "ExampleMetadata",
    "GoldExample",
    "GoldLabel",
    "LLMUsage",
    "ModelOutput",
    "TargetWord",
    # Collect
    "assign_buckets",
    "build_example_id",
    "calculate_char_span",
    "extract_examples_from_tokens",
    "get_candidates_for_lemma",
    # Sample
    "DifficultyFeatures",
    "calculate_difficulty_features",
    "classify_difficulty",
    "get_sampling_stats",
    "split_by_source",
    "stratified_sample",
    # Aggregate
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "PROBLEMATIC_FLAGS",
    "aggregate_labels",
    "get_aggregation_stats",
    "needs_third_judge",
    # Smart Aggregate
    "SmartAggregationResult",
    "SmartAggregationStats",
    "get_smart_aggregation_stats",
    "needs_referee",
    "smart_aggregate",
    # Eval
    "classify_example_difficulty",
    "compute_metrics",
    "compute_metrics_by_segment",
    "evaluate_single",
    "evaluate_wsd_on_gold",
    "get_supersense",
    "load_gold_examples",
    "run_wsd_prediction",
    # Cache
    "LLMCache",
    "get_cache_key",
]
