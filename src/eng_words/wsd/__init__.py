"""Word Sense Disambiguation (WSD) module.

This module provides functionality for disambiguating word senses using
Sentence-Transformers and WordNet supersenses.

Main components:
- SenseBackend: Abstract base class for sense disambiguation backends
- SenseAnnotation: Data class for sense annotation results
- EmbeddingModel: Wrapper for sentence-transformers model
- WordNet utilities: Functions for synset lookup and supersense mapping
- WordNetSenseBackend: WSD using Sentence-Transformers + WordNet definitions
"""

from eng_words.wsd.aggregator import aggregate_sense_statistics
from eng_words.wsd.base import SenseAnnotation, SenseBackend
from eng_words.wsd.embeddings import (
    WSD_BATCH_SIZE,
    WSD_EMBEDDING_DIM,
    WSD_MODEL_NAME,
    DefinitionEmbeddingCache,
    EmbeddingModel,
    compute_cosine_similarity,
    get_batch_embeddings,
    get_definition_cache,
    get_embedding_model,
    get_sentence_embedding,
)
from eng_words.wsd.llm_wsd import llm_wsd_sentence, redistribute_empty_cards
from eng_words.wsd.wordnet_backend import (
    CONTENT_POS_TAGS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    WordNetSenseBackend,
)
from eng_words.wsd.wordnet_utils import (
    SPACY_TO_WORDNET_POS,
    get_all_supersenses_for_lemma,
    get_definition,
    get_most_common_synset,
    get_synsets,
    get_synsets_with_definitions,
    has_multiple_senses,
    map_spacy_pos_to_wordnet,
    synset_to_supersense,
)

__all__ = [
    # Base
    "SenseAnnotation",
    "SenseBackend",
    # Backend implementation
    "WordNetSenseBackend",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "CONTENT_POS_TAGS",
    # Aggregation
    "aggregate_sense_statistics",
    # Embeddings
    "WSD_MODEL_NAME",
    "WSD_BATCH_SIZE",
    "WSD_EMBEDDING_DIM",
    "EmbeddingModel",
    "get_embedding_model",
    "get_sentence_embedding",
    "get_batch_embeddings",
    "compute_cosine_similarity",
    "DefinitionEmbeddingCache",
    "get_definition_cache",
    # WordNet utilities
    "SPACY_TO_WORDNET_POS",
    "get_synsets",
    "get_synsets_with_definitions",
    "synset_to_supersense",
    "get_definition",
    "map_spacy_pos_to_wordnet",
    "get_all_supersenses_for_lemma",
    "has_multiple_senses",
    "get_most_common_synset",
    # LLM WSD
    "llm_wsd_sentence",
    "redistribute_empty_cards",
]
