"""Tests for WSD embeddings module."""

import numpy as np

from eng_words.wsd.embeddings import (
    WSD_BATCH_SIZE,
    WSD_MODEL_NAME,
    EmbeddingModel,
    compute_cosine_similarity,
    get_batch_embeddings,
    get_embedding_model,
    get_sentence_embedding,
)


class TestEmbeddingModelConfig:
    """Test embedding model configuration."""

    def test_model_name_is_defined(self):
        """WSD_MODEL_NAME should be a valid sentence-transformers model."""
        assert WSD_MODEL_NAME == "sentence-transformers/all-mpnet-base-v2"

    def test_batch_size_is_reasonable(self):
        """WSD_BATCH_SIZE should be a reasonable value."""
        assert 1 <= WSD_BATCH_SIZE <= 128
        assert WSD_BATCH_SIZE == 32


class TestEmbeddingModel:
    """Test EmbeddingModel class."""

    def test_singleton_pattern(self):
        """get_embedding_model should return the same instance."""
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        assert model1 is model2

    def test_model_loads_successfully(self):
        """Model should load without errors."""
        model = get_embedding_model()
        assert model is not None
        assert isinstance(model, EmbeddingModel)

    def test_model_has_encode_method(self):
        """Model should have encode capability."""
        model = get_embedding_model()
        assert hasattr(model, "encode")

    def test_model_embedding_dimension(self):
        """Model should produce embeddings of expected dimension."""
        model = get_embedding_model()
        embedding = model.encode("test sentence")
        # all-mpnet-base-v2 produces 768-dimensional embeddings
        assert embedding.shape == (768,)


class TestGetSentenceEmbedding:
    """Test get_sentence_embedding function."""

    def test_returns_numpy_array(self):
        """Should return a numpy array."""
        embedding = get_sentence_embedding("Hello world")
        assert isinstance(embedding, np.ndarray)

    def test_embedding_dimension(self):
        """Should return 768-dimensional embedding for all-mpnet-base-v2."""
        embedding = get_sentence_embedding("This is a test sentence.")
        assert embedding.shape == (768,)

    def test_different_sentences_different_embeddings(self):
        """Different sentences should produce different embeddings."""
        emb1 = get_sentence_embedding("The cat sat on the mat.")
        emb2 = get_sentence_embedding("Financial markets are volatile.")
        # Embeddings should be different
        assert not np.allclose(emb1, emb2)

    def test_similar_sentences_similar_embeddings(self):
        """Similar sentences should produce similar embeddings."""
        emb1 = get_sentence_embedding("The cat is sleeping.")
        emb2 = get_sentence_embedding("The cat is resting.")
        # Cosine similarity should be high
        similarity = compute_cosine_similarity(emb1, emb2)
        assert similarity > 0.7

    def test_empty_string(self):
        """Should handle empty string gracefully."""
        embedding = get_sentence_embedding("")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

    def test_normalized_embeddings(self):
        """Embeddings should be normalized (unit length)."""
        embedding = get_sentence_embedding("Test sentence for normalization.")
        norm = np.linalg.norm(embedding)
        assert np.isclose(norm, 1.0, atol=0.01)


class TestGetBatchEmbeddings:
    """Test get_batch_embeddings function."""

    def test_returns_2d_numpy_array(self):
        """Should return a 2D numpy array."""
        sentences = ["Hello", "World", "Test"]
        embeddings = get_batch_embeddings(sentences)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2

    def test_correct_shape(self):
        """Should return (n_sentences, embedding_dim) shape."""
        sentences = ["One", "Two", "Three", "Four"]
        embeddings = get_batch_embeddings(sentences)
        assert embeddings.shape == (4, 768)

    def test_empty_list(self):
        """Should handle empty list gracefully."""
        embeddings = get_batch_embeddings([])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0, 768)

    def test_single_sentence(self):
        """Should handle single sentence."""
        embeddings = get_batch_embeddings(["Single sentence"])
        assert embeddings.shape == (1, 768)

    def test_batch_matches_individual(self):
        """Batch embeddings should match individual embeddings."""
        sentences = ["First sentence.", "Second sentence."]
        batch_embs = get_batch_embeddings(sentences)
        individual_embs = [get_sentence_embedding(s) for s in sentences]

        for i, (batch, individual) in enumerate(zip(batch_embs, individual_embs)):
            assert np.allclose(batch, individual, atol=1e-5), f"Mismatch at index {i}"

    def test_large_batch(self):
        """Should handle batches larger than WSD_BATCH_SIZE."""
        sentences = [f"Sentence number {i}" for i in range(50)]
        embeddings = get_batch_embeddings(sentences)
        assert embeddings.shape == (50, 768)


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vec = np.array([1.0, 2.0, 3.0])
        similarity = compute_cosine_similarity(vec, vec)
        assert np.isclose(similarity, 1.0)

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert np.isclose(similarity, 0.0)

    def test_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([-1.0, -2.0, -3.0])
        similarity = compute_cosine_similarity(vec1, vec2)
        assert np.isclose(similarity, -1.0)

    def test_similarity_range(self):
        """Similarity should be in [-1, 1] range."""
        vec1 = np.random.randn(768)
        vec2 = np.random.randn(768)
        similarity = compute_cosine_similarity(vec1, vec2)
        assert -1.0 <= similarity <= 1.0

    def test_batch_similarity(self):
        """Should compute similarity between vector and matrix."""
        query = np.array([1.0, 0.0, 0.0])
        candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        similarities = compute_cosine_similarity(query, candidates)
        expected = np.array([1.0, 0.0, -1.0])
        assert np.allclose(similarities, expected)


class TestDeviceSelection:
    """Test device selection for model."""

    def test_model_device_is_set(self):
        """Model should have a device attribute."""
        model = get_embedding_model()
        assert hasattr(model, "device")

    def test_device_is_valid(self):
        """Device should be cpu, cuda, or mps."""
        model = get_embedding_model()
        device = str(model.device)
        assert device in ["cpu", "cuda", "mps"] or device.startswith("cuda:")


class TestDefinitionEmbeddingCache:
    """Test definition embedding cache."""

    def test_cache_stores_embeddings(self):
        """Cache should store and retrieve embeddings."""
        from eng_words.wsd.embeddings import DefinitionEmbeddingCache

        cache = DefinitionEmbeddingCache()
        embedding = cache.get_or_compute("test.n.01", "a test definition")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (768,)

        # Second call should return cached value
        embedding2 = cache.get_or_compute("test.n.01", "different text")
        assert np.allclose(embedding, embedding2)

    def test_cache_length(self):
        """Cache should track number of entries."""
        from eng_words.wsd.embeddings import DefinitionEmbeddingCache

        cache = DefinitionEmbeddingCache()
        assert len(cache) == 0

        cache.get_or_compute("a.n.01", "definition a")
        assert len(cache) == 1

        cache.get_or_compute("b.n.01", "definition b")
        assert len(cache) == 2

        # Same key shouldn't increase count
        cache.get_or_compute("a.n.01", "definition a")
        assert len(cache) == 2

    def test_cache_clear(self):
        """Cache should be clearable."""
        from eng_words.wsd.embeddings import DefinitionEmbeddingCache

        cache = DefinitionEmbeddingCache()
        cache.get_or_compute("test.n.01", "test")
        assert len(cache) > 0

        cache.clear()
        assert len(cache) == 0

    def test_cache_batch_retrieval(self):
        """Cache should support batch retrieval."""
        from eng_words.wsd.embeddings import DefinitionEmbeddingCache

        cache = DefinitionEmbeddingCache()

        # Pre-populate one entry
        cache.get_or_compute("cached.n.01", "cached definition")

        # Batch request with mix of cached and new
        synset_defs = [
            ("cached.n.01", "cached definition"),
            ("new.n.01", "new definition"),
            ("another.n.01", "another definition"),
        ]

        result = cache.get_batch(synset_defs)

        assert len(result) == 3
        assert "cached.n.01" in result
        assert "new.n.01" in result
        assert "another.n.01" in result

        # All should now be cached
        assert len(cache) == 3

    def test_cache_max_size(self):
        """Cache should respect max size limit."""
        from eng_words.wsd.embeddings import DefinitionEmbeddingCache

        cache = DefinitionEmbeddingCache(max_size=5)

        # Add more than max_size entries
        for i in range(10):
            cache.get_or_compute(f"entry.n.{i:02d}", f"definition {i}")

        # Cache should have been trimmed
        assert len(cache) <= 5
