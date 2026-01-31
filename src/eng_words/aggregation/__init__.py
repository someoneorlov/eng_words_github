"""Aggregation module for synset-based card generation.

Note: fallback.py has been moved to _archive/ folder (fallback logic removed from pipeline).
"""

from .llm_aggregator import (
    AggregationResult,
    LLMAggregator,
    SynsetGroup,
    build_aggregation_prompt,
    parse_aggregation_response,
)
from .synset_aggregator import SynsetStats, aggregate_by_synset, get_synset_info

__all__ = [
    "SynsetStats",
    "aggregate_by_synset",
    "get_synset_info",
    "SynsetGroup",
    "AggregationResult",
    "LLMAggregator",
    "build_aggregation_prompt",
    "parse_aggregation_response",
]
