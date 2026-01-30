"""Embedding model for Word Sense Disambiguation.

This module provides functions for loading and using sentence-transformers
models to generate embeddings for WSD.

Usage:
    from eng_words.wsd.embeddings import get_sentence_embedding, get_batch_embeddings

    embedding = get_sentence_embedding("The cat sat on the mat.")
    embeddings = get_batch_embeddings(["Sentence 1", "Sentence 2"])
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

WSD_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
WSD_BATCH_SIZE = 32
WSD_EMBEDDING_DIM = 768  # Dimension for all-mpnet-base-v2


# =============================================================================
# Embedding Model Wrapper
# =============================================================================


class EmbeddingModel:
    """Wrapper around sentence-transformers model with device management.

    This class provides a singleton pattern for loading the model once
    and reusing it across the application.

    Attributes:
        model: The underlying SentenceTransformer model
        device: The device the model is running on (cpu, cuda, mps)
    """

    _instance: EmbeddingModel | None = None

    def __init__(self, model_name: str = WSD_MODEL_NAME):
        """Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to load
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for WSD. "
                "Install with: pip install sentence-transformers"
            ) from e

        logger.info(f"Loading embedding model: {model_name}")

        # Detect best available device
        device = self._detect_device()
        logger.info(f"Using device: {device}")

        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self._model_name = model_name

        logger.info(f"Model loaded successfully on {device}")

    @staticmethod
    def _detect_device() -> str:
        """Detect the best available device for inference.

        Returns:
            Device string: 'cuda', 'mps', or 'cpu'
        """
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                # Apple Silicon (M1/M2/M3)
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def encode(
        self,
        sentences: str | Sequence[str],
        batch_size: int = WSD_BATCH_SIZE,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode sentences into embeddings.

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings to unit length

        Returns:
            Numpy array of embeddings. Shape (embedding_dim,) for single sentence,
            (n_sentences, embedding_dim) for multiple sentences.
        """
        if isinstance(sentences, str):
            sentences = [sentences]
            single = True
        else:
            single = False

        if len(sentences) == 0:
            return np.zeros((0, WSD_EMBEDDING_DIM), dtype=np.float32)

        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if single:
            return embeddings[0]
        return embeddings


# =============================================================================
# Singleton Access
# =============================================================================


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = WSD_MODEL_NAME) -> EmbeddingModel:
    """Get the singleton embedding model instance.

    This function ensures the model is loaded only once and reused
    across all calls.

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        EmbeddingModel instance
    """
    return EmbeddingModel(model_name)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_sentence_embedding(sentence: str) -> np.ndarray:
    """Get embedding for a single sentence.

    Args:
        sentence: The sentence to encode

    Returns:
        Numpy array of shape (embedding_dim,)
    """
    model = get_embedding_model()
    return model.encode(sentence)


def get_batch_embeddings(sentences: List[str]) -> np.ndarray:
    """Get embeddings for multiple sentences.

    Args:
        sentences: List of sentences to encode

    Returns:
        Numpy array of shape (n_sentences, embedding_dim)
    """
    model = get_embedding_model()
    return model.encode(sentences)


# =============================================================================
# Similarity Functions
# =============================================================================


def compute_cosine_similarity(
    query: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray | float:
    """Compute cosine similarity between query and candidate embeddings.

    Args:
        query: Query embedding of shape (embedding_dim,)
        candidates: Either a single embedding (embedding_dim,) or
                   matrix of embeddings (n_candidates, embedding_dim)

    Returns:
        If candidates is 1D: float similarity score
        If candidates is 2D: array of similarity scores (n_candidates,)
    """
    # Normalize vectors (in case they aren't already)
    query_norm = query / (np.linalg.norm(query) + 1e-10)

    if candidates.ndim == 1:
        # Single candidate
        cand_norm = candidates / (np.linalg.norm(candidates) + 1e-10)
        return float(np.dot(query_norm, cand_norm))
    else:
        # Multiple candidates
        cand_norms = candidates / (np.linalg.norm(candidates, axis=1, keepdims=True) + 1e-10)
        return np.dot(cand_norms, query_norm)


# =============================================================================
# Definition Embedding Cache
# =============================================================================


class DefinitionEmbeddingCache:
    """Cache for WordNet definition embeddings.

    This cache stores embeddings for WordNet synset definitions to avoid
    recomputing them for each disambiguation.

    Usage:
        cache = DefinitionEmbeddingCache()
        embedding = cache.get_or_compute("run.v.01", "move fast by using legs")
    """

    def __init__(self, max_size: int = 10000):
        """Initialize the cache.

        Args:
            max_size: Maximum number of definitions to cache
        """
        self._cache: dict[str, np.ndarray] = {}
        self._max_size = max_size

    def get_or_compute(self, synset_id: str, definition: str) -> np.ndarray:
        """Get embedding from cache or compute it.

        Args:
            synset_id: WordNet synset ID (used as cache key)
            definition: The definition text

        Returns:
            Embedding vector for the definition
        """
        if synset_id in self._cache:
            return self._cache[synset_id]

        embedding = get_sentence_embedding(definition)

        # Simple LRU-like behavior: if cache is full, clear half of it
        if len(self._cache) >= self._max_size:
            keys_to_remove = list(self._cache.keys())[: self._max_size // 2]
            for key in keys_to_remove:
                del self._cache[key]

        self._cache[synset_id] = embedding
        return embedding

    def get(self, synset_id: str) -> np.ndarray | None:
        """Get embedding from cache by synset_id.

        Args:
            synset_id: WordNet synset ID

        Returns:
            Embedding vector if cached, None otherwise
        """
        return self._cache.get(synset_id)

    def get_batch(self, synset_definitions: List[tuple[str, str]]) -> dict[str, np.ndarray]:
        """Get embeddings for multiple definitions, using cache where possible.

        Args:
            synset_definitions: List of (synset_id, definition) tuples

        Returns:
            Dictionary mapping synset_id to embedding
        """
        result = {}
        to_compute = []

        for synset_id, definition in synset_definitions:
            if synset_id in self._cache:
                result[synset_id] = self._cache[synset_id]
            else:
                to_compute.append((synset_id, definition))

        if to_compute:
            definitions = [d for _, d in to_compute]
            embeddings = get_batch_embeddings(definitions)

            for (synset_id, _), embedding in zip(to_compute, embeddings):
                self._cache[synset_id] = embedding
                result[synset_id] = embedding

        return result

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached embeddings."""
        return len(self._cache)


# Global cache instance
_definition_cache: DefinitionEmbeddingCache | None = None


def get_definition_cache() -> DefinitionEmbeddingCache:
    """Get the global definition embedding cache.

    Returns:
        DefinitionEmbeddingCache instance
    """
    global _definition_cache
    if _definition_cache is None:
        _definition_cache = DefinitionEmbeddingCache()
    return _definition_cache
