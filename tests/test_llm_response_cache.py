"""Tests for LLM response caching."""

import json
import tempfile
from pathlib import Path

import pytest

from eng_words.llm.base import LLMResponse


class TestResponseCache:
    """Tests for ResponseCache."""

    def test_import_cache(self):
        """Test that ResponseCache can be imported."""
        from eng_words.llm.response_cache import ResponseCache

        assert ResponseCache is not None

    def test_create_cache(self):
        """Test creating cache instance."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))
            assert cache is not None
            assert cache.enabled is True

    def test_cache_disabled(self):
        """Test that cache can be disabled."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)
            assert cache.enabled is False

    def test_generate_key(self):
        """Test cache key generation."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))
            key1 = cache.generate_key("model1", "prompt1", 0.0)
            key2 = cache.generate_key("model1", "prompt1", 0.0)
            key3 = cache.generate_key("model2", "prompt1", 0.0)
            key4 = cache.generate_key("model1", "prompt2", 0.0)
            key5 = cache.generate_key("model1", "prompt1", 0.5)

            # Same inputs should give same key
            assert key1 == key2
            # Different inputs should give different keys
            assert key1 != key3
            assert key1 != key4
            assert key1 != key5

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))
            result = cache.get("nonexistent_key")
            assert result is None

    def test_cache_set_and_get(self):
        """Test storing and retrieving from cache."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="test response",
                model="test-model",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )

            key = cache.generate_key("test-model", "test prompt", 0.0)
            cache.set(key, response)

            # Retrieve
            cached = cache.get(key)
            assert cached is not None
            assert cached.content == "test response"
            assert cached.model == "test-model"
            assert cached.input_tokens == 10
            assert cached.output_tokens == 5
            assert cached.cost_usd == 0.001

    def test_cache_persistence(self):
        """Test that cache persists to disk."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # First cache instance
            cache1 = ResponseCache(cache_dir=cache_dir)
            response = LLMResponse(
                content="persistent response",
                model="test-model",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )
            key = cache1.generate_key("test-model", "test prompt", 0.0)
            cache1.set(key, response)

            # New cache instance should load persisted data
            cache2 = ResponseCache(cache_dir=cache_dir)
            cached = cache2.get(key)
            assert cached is not None
            assert cached.content == "persistent response"

    def test_cache_disabled_no_store(self):
        """Test that disabled cache doesn't store."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)

            response = LLMResponse(
                content="test response",
                model="test-model",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )

            key = cache.generate_key("test-model", "test prompt", 0.0)
            cache.set(key, response)

            # Should return None even after set
            assert cache.get(key) is None

    def test_stats_initial(self):
        """Test initial statistics."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))
            stats = cache.stats()

            assert stats["hits"] == 0
            assert stats["misses"] == 0
            assert stats["hit_rate"] == 0.0
            assert stats["tokens_saved"] == 0
            assert stats["cost_saved"] == 0.0

    def test_stats_tracking(self):
        """Test statistics tracking."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="test response",
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.01,
            )

            key = cache.generate_key("test-model", "test prompt", 0.0)
            cache.set(key, response)

            # Miss
            cache.get("nonexistent")

            # Hit
            cache.get(key)
            cache.get(key)

            stats = cache.stats()
            assert stats["hits"] == 2
            assert stats["misses"] == 1
            assert stats["hit_rate"] == 2 / 3
            assert stats["tokens_saved"] == 300  # 2 hits * (100 + 50) tokens
            assert stats["cost_saved"] == 0.02  # 2 hits * 0.01

    def test_get_or_compute_cached(self):
        """Test get_or_compute returns cached value."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))

            response = LLMResponse(
                content="cached response",
                model="test-model",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )

            key = cache.generate_key("test-model", "test prompt", 0.0)
            cache.set(key, response)

            compute_called = False

            def compute_fn():
                nonlocal compute_called
                compute_called = True
                return LLMResponse(
                    content="computed response",
                    model="test-model",
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.001,
                )

            result = cache.get_or_compute(key, compute_fn)

            assert result.content == "cached response"
            assert compute_called is False

    def test_get_or_compute_not_cached(self):
        """Test get_or_compute computes when not cached."""
        from eng_words.llm.response_cache import ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))

            compute_called = False

            def compute_fn():
                nonlocal compute_called
                compute_called = True
                return LLMResponse(
                    content="computed response",
                    model="test-model",
                    input_tokens=10,
                    output_tokens=5,
                    cost_usd=0.001,
                )

            key = cache.generate_key("test-model", "test prompt", 0.0)
            result = cache.get_or_compute(key, compute_fn)

            assert result.content == "computed response"
            assert compute_called is True

            # Should be cached now
            cached = cache.get(key)
            assert cached.content == "computed response"


class TestCachedProvider:
    """Tests for CachedProvider wrapper."""

    def test_import_cached_provider(self):
        """Test that CachedProvider can be imported."""
        from eng_words.llm.response_cache import CachedProvider

        assert CachedProvider is not None

    def test_cached_provider_wraps_calls(self):
        """Test that CachedProvider wraps and caches calls."""
        from unittest.mock import MagicMock

        from eng_words.llm.response_cache import CachedProvider, ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))

            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0
            mock_provider.complete.return_value = LLMResponse(
                content="response",
                model="test-model",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )

            cached_provider = CachedProvider(mock_provider, cache)

            # First call - should call underlying provider
            result1 = cached_provider.complete("test prompt")
            assert mock_provider.complete.call_count == 1
            assert result1.content == "response"

            # Second call - should return cached
            result2 = cached_provider.complete("test prompt")
            assert mock_provider.complete.call_count == 1  # Still 1
            assert result2.content == "response"

    def test_cached_provider_different_prompts(self):
        """Test that different prompts are cached separately."""
        from unittest.mock import MagicMock

        from eng_words.llm.response_cache import CachedProvider, ResponseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(cache_dir=Path(tmpdir))

            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0
            mock_provider.complete.return_value = LLMResponse(
                content="response",
                model="test-model",
                input_tokens=10,
                output_tokens=5,
                cost_usd=0.001,
            )

            cached_provider = CachedProvider(mock_provider, cache)

            cached_provider.complete("prompt 1")
            cached_provider.complete("prompt 2")
            cached_provider.complete("prompt 1")  # Cached
            cached_provider.complete("prompt 2")  # Cached

            assert mock_provider.complete.call_count == 2

