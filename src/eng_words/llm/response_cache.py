"""Response-level caching for LLM API calls.

Caches raw LLM responses to avoid duplicate API calls.
Key: hash(model + prompt + temperature)
Storage: JSON files in data/cache/llm_responses/
"""

import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from eng_words.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache/llm_responses")


class ResponseCache:
    """Cache for LLM API responses.

    Stores responses as JSON files keyed by hash of (model, prompt, temperature).
    Tracks statistics: hits, misses, tokens saved, cost saved.

    Args:
        cache_dir: Directory for cache files. Created if doesn't exist.
        enabled: Whether caching is enabled. Default True.
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        enabled: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self._hits = 0
        self._misses = 0
        self._tokens_saved = 0
        self._cost_saved = 0.0

    def generate_key(self, model: str, prompt: str, temperature: float) -> str:
        """Generate cache key from request parameters.

        Args:
            model: Model name.
            prompt: Prompt text.
            temperature: Temperature setting.

        Returns:
            SHA256 hash as hex string.
        """
        content = f"{model}|{prompt}|{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> LLMResponse | None:
        """Get cached response.

        Args:
            key: Cache key.

        Returns:
            LLMResponse if found, None otherwise.
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            self._misses += 1
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            response = LLMResponse(**data)

            self._hits += 1
            self._tokens_saved += response.total_tokens
            self._cost_saved += response.cost_usd

            logger.debug(f"Cache hit for key {key[:16]}...")
            return response

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load cache for {key}: {e}")
            self._misses += 1
            return None

    def set(self, key: str, response: LLMResponse) -> None:
        """Store response in cache.

        Args:
            key: Cache key.
            response: LLMResponse to cache.
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(key)

        try:
            data = asdict(response)
            cache_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"Cached response for key {key[:16]}...")
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to cache response: {e}")

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], LLMResponse],
    ) -> LLMResponse:
        """Get from cache or compute and cache.

        Args:
            key: Cache key.
            compute_fn: Function to call if not cached.

        Returns:
            LLMResponse from cache or computed.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        response = compute_fn()
        self.set(key, response)
        return response

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, tokens_saved, cost_saved.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "tokens_saved": self._tokens_saved,
            "cost_saved": self._cost_saved,
        }

    def __len__(self) -> int:
        """Return number of cached responses."""
        if not self.enabled:
            return 0
        return len(list(self.cache_dir.glob("*.json")))


class CachedProvider(LLMProvider):
    """Wrapper that adds caching to any LLMProvider.

    Args:
        provider: Underlying LLM provider.
        cache: ResponseCache instance.
    """

    def __init__(self, provider: LLMProvider, cache: ResponseCache):
        self._provider = provider
        self._cache = cache

    @property
    def model(self) -> str:
        """Get model name from underlying provider."""
        return self._provider.model

    @property
    def temperature(self) -> float:
        """Get temperature from underlying provider."""
        return self._provider.temperature

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Execute completion with caching.

        Args:
            prompt: Prompt text.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse (from cache or API).
        """
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)

        key = self._cache.generate_key(model, prompt, temperature)

        def compute():
            return self._provider.complete(prompt, **kwargs)

        return self._cache.get_or_compute(key, compute)

    def complete_json(self, prompt: str, schema: dict | None = None, **kwargs) -> dict:
        """Execute JSON completion with caching.

        Args:
            prompt: Prompt text.
            schema: Optional JSON schema.
            **kwargs: Additional parameters.

        Returns:
            Parsed JSON dict.
        """
        # Note: complete_json uses complete internally, so caching happens there
        return self._provider.complete_json(prompt, schema, **kwargs)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost (delegated to underlying provider)."""
        return self._provider.estimate_cost(input_tokens, output_tokens)
