"""Tests for LLM response cache."""

from pathlib import Path

import pytest

from eng_words.wsd_gold.cache import LLMCache, get_cache_key
from eng_words.wsd_gold.models import LLMUsage, ModelOutput


class TestGetCacheKey:
    """Tests for cache key generation."""

    def test_different_examples_different_keys(self):
        """Different examples produce different keys."""
        key1 = get_cache_key("book:test|sent:1|tok:1", "gpt-5.2", "v1.0")
        key2 = get_cache_key("book:test|sent:2|tok:1", "gpt-5.2", "v1.0")
        assert key1 != key2

    def test_different_models_different_keys(self):
        """Different models produce different keys."""
        key1 = get_cache_key("book:test|sent:1|tok:1", "gpt-5.2", "v1.0")
        key2 = get_cache_key("book:test|sent:1|tok:1", "claude-opus", "v1.0")
        assert key1 != key2

    def test_different_versions_different_keys(self):
        """Different prompt versions produce different keys."""
        key1 = get_cache_key("book:test|sent:1|tok:1", "gpt-5.2", "v1.0")
        key2 = get_cache_key("book:test|sent:1|tok:1", "gpt-5.2", "v2.0")
        assert key1 != key2

    def test_key_is_filename_safe(self):
        """Key is safe for use as filename."""
        key = get_cache_key("book:test|sent:1|tok:1", "gpt-5.2", "v1.0")
        # No problematic characters
        assert ":" not in key
        assert "|" not in key
        assert "/" not in key


class TestLLMCache:
    """Tests for LLMCache class."""

    @pytest.fixture
    def temp_cache(self, tmp_path: Path) -> LLMCache:
        """Create a cache in a temp directory."""
        return LLMCache(cache_dir=tmp_path / "cache", prompt_version="v1.0")

    @pytest.fixture
    def sample_output(self) -> ModelOutput:
        """Create a sample ModelOutput."""
        return ModelOutput(
            chosen_synset_id="bank.n.01",
            confidence=0.95,
            flags=[],
            raw_text='{"synset": "bank.n.01", "confidence": 0.95}',
            usage=LLMUsage(input_tokens=100, output_tokens=50, cost_usd=0.001),
        )

    def test_cache_miss_returns_none(self, temp_cache: LLMCache):
        """Cache miss returns None."""
        result = temp_cache.get("nonexistent", "gpt-5.2")
        assert result is None

    def test_set_and_get(self, temp_cache: LLMCache, sample_output: ModelOutput):
        """Can set and get a cached value."""
        temp_cache.set("test|1", "gpt-5.2", sample_output)
        result = temp_cache.get("test|1", "gpt-5.2")

        assert result is not None
        assert result.chosen_synset_id == "bank.n.01"
        assert result.confidence == 0.95

    def test_different_model_is_miss(self, temp_cache: LLMCache, sample_output: ModelOutput):
        """Different model is a cache miss."""
        temp_cache.set("test|1", "gpt-5.2", sample_output)
        result = temp_cache.get("test|1", "claude-opus")

        assert result is None

    def test_stats_tracking(self, temp_cache: LLMCache, sample_output: ModelOutput):
        """Stats are tracked correctly."""
        temp_cache.set("test|1", "gpt-5.2", sample_output)

        temp_cache.get("test|1", "gpt-5.2")  # Hit
        temp_cache.get("test|1", "gpt-5.2")  # Hit
        temp_cache.get("nonexistent", "gpt-5.2")  # Miss

        stats = temp_cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["writes"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_disabled_cache(self, tmp_path: Path, sample_output: ModelOutput):
        """Disabled cache doesn't store or retrieve."""
        cache = LLMCache(cache_dir=tmp_path, enabled=False)

        cache.set("test|1", "gpt-5.2", sample_output)
        result = cache.get("test|1", "gpt-5.2")

        assert result is None

    def test_clear_cache(self, temp_cache: LLMCache, sample_output: ModelOutput):
        """Can clear all cached files."""
        temp_cache.set("test|1", "gpt-5.2", sample_output)
        temp_cache.set("test|2", "gpt-5.2", sample_output)

        count = temp_cache.clear_cache()

        assert count == 2
        assert temp_cache.get("test|1", "gpt-5.2") is None

    def test_get_or_compute_uses_cache(self, temp_cache: LLMCache, sample_output: ModelOutput):
        """get_or_compute uses cached value if available."""
        from eng_words.wsd_gold.models import (
            Candidate,
            ExampleMetadata,
            GoldExample,
            TargetWord,
        )

        example = GoldExample(
            example_id="test|1",
            source_id="test",
            source_bucket="fiction",
            year_bucket="2020s",
            genre_bucket="novel",
            text_left="",
            target=TargetWord(surface="bank", lemma="bank", pos="NOUN", char_span=(0, 4)),
            text_right="",
            context_window="I went to the bank.",
            candidates=[Candidate(synset_id="bank.n.01", gloss="financial", examples=[])],
            metadata=ExampleMetadata(
                wn_sense_count=2, baseline_top1="bank.n.01", baseline_margin=0.5, is_multiword=False
            ),
        )

        # Pre-populate cache
        temp_cache.set("test|1", "gpt-5.2", sample_output)

        # Counter to check if compute_fn is called
        call_count = [0]

        def compute_fn(ex):
            call_count[0] += 1
            return sample_output

        result = temp_cache.get_or_compute(example, "gpt-5.2", compute_fn)

        assert result is not None
        assert result.chosen_synset_id == "bank.n.01"
        assert call_count[0] == 0  # compute_fn was NOT called
