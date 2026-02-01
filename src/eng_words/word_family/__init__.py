"""Word Family pipeline (Pipeline B): LLM-based clustering by lemma.

Used by run_pipeline_b_batch.py (Batch API). Groups examples by lemma,
clusters by meaning, produces 1–3 cards per lemma.
"""

from eng_words.word_family.clusterer import (
    CLUSTER_PROMPT_TEMPLATE,
    ClusterResult,
    WordFamilyClusterer,
    group_examples_by_lemma,
)

__all__ = [
    "CLUSTER_PROMPT_TEMPLATE",
    "ClusterResult",
    "WordFamilyClusterer",
    "group_examples_by_lemma",
]
