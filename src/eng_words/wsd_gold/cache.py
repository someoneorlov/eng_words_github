"""LLM response cache for WSD Gold Dataset labeling.

This module provides simple JSON-based caching for LLM responses
to avoid duplicate API calls and save costs.

Cache key format: {example_id}:{model}:{prompt_version}
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from eng_words.wsd_gold.models import GoldExample, LLMUsage, ModelOutput
from eng_words.wsd_gold.providers.prompts import PROMPT_VERSION_GOLD_LABELING

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/wsd_gold/cache")


def get_cache_key(example_id: str, model: str, prompt_version: str) -> str:
    """Generate cache key for an example.

    Args:
        example_id: Unique example identifier
        model: LLM model name
        prompt_version: Prompt version string

    Returns:
        Cache key string (safe for filenames)
    """
    # Create a hash-based key to handle special characters
    raw_key = f"{example_id}|{model}|{prompt_version}"
    key_hash = hashlib.md5(raw_key.encode()).hexdigest()[:16]

    # Also keep a readable prefix
    safe_id = example_id.replace(":", "_").replace("|", "_")[:40]
    safe_model = model.replace("/", "_").replace("-", "_")[:20]

    return f"{safe_id}_{safe_model}_{key_hash}"


class LLMCache:
    """Simple JSON-based cache for LLM responses.

    Attributes:
        cache_dir: Directory for cache files
        prompt_version: Current prompt version
        enabled: Whether caching is enabled
    """

    def __init__(
        self,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        prompt_version: str = PROMPT_VERSION_GOLD_LABELING,
        enabled: bool = True,
    ):
        """Initialize the cache.

        Args:
            cache_dir: Directory for cache files
            prompt_version: Prompt version for cache keys
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.prompt_version = prompt_version
        self.enabled = enabled
        self._stats = {"hits": 0, "misses": 0, "writes": 0}

        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, example_id: str, model: str) -> Path:
        """Get cache file path for an example."""
        key = get_cache_key(example_id, model, self.prompt_version)
        return self.cache_dir / f"{key}.json"

    def get(self, example_id: str, model: str) -> ModelOutput | None:
        """Get cached result for an example.

        Args:
            example_id: Unique example identifier
            model: LLM model name

        Returns:
            Cached ModelOutput or None if not found
        """
        if not self.enabled:
            return None

        cache_path = self._get_cache_path(example_id, model)

        if not cache_path.exists():
            self._stats["misses"] += 1
            return None

        try:
            with open(cache_path) as f:
                data = json.load(f)

            # Verify prompt version matches
            if data.get("prompt_version") != self.prompt_version:
                logger.debug(f"Cache version mismatch for {example_id}")
                self._stats["misses"] += 1
                return None

            # Reconstruct ModelOutput
            output_data = data["output"]
            usage = LLMUsage(
                input_tokens=output_data["usage"]["input_tokens"],
                output_tokens=output_data["usage"]["output_tokens"],
                cached_tokens=output_data["usage"].get("cached_tokens", 0),
                cost_usd=output_data["usage"].get("cost_usd", 0),
            )

            output = ModelOutput(
                chosen_synset_id=output_data["chosen_synset_id"],
                confidence=output_data["confidence"],
                flags=output_data["flags"],
                raw_text=output_data["raw_text"],
                usage=usage,
            )

            self._stats["hits"] += 1
            return output

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache read error for {example_id}: {e}")
            self._stats["misses"] += 1
            return None

    def set(self, example_id: str, model: str, output: ModelOutput) -> None:
        """Cache result for an example.

        Args:
            example_id: Unique example identifier
            model: LLM model name
            output: ModelOutput to cache
        """
        if not self.enabled:
            return

        cache_path = self._get_cache_path(example_id, model)

        data = {
            "example_id": example_id,
            "model": model,
            "prompt_version": self.prompt_version,
            "output": output.to_dict(),
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            self._stats["writes"] += 1
        except OSError as e:
            logger.warning(f"Cache write error for {example_id}: {e}")

    def get_or_compute(
        self,
        example: GoldExample,
        model: str,
        compute_fn: Any,
    ) -> ModelOutput | None:
        """Get from cache or compute and cache.

        Args:
            example: GoldExample to process
            model: LLM model name
            compute_fn: Function to call if not cached (takes example, returns ModelOutput)

        Returns:
            ModelOutput from cache or computation
        """
        # Try cache first
        cached = self.get(example.example_id, model)
        if cached is not None:
            return cached

        # Compute
        result = compute_fn(example)

        # Cache result
        if result is not None:
            self.set(example.example_id, model, result)

        return result

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0.0

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "writes": self._stats["writes"],
            "total_requests": total,
            "hit_rate": hit_rate,
        }

    def clear_stats(self) -> None:
        """Reset statistics."""
        self._stats = {"hits": 0, "misses": 0, "writes": 0}

    def clear_cache(self) -> int:
        """Clear all cached files.

        Returns:
            Number of files deleted
        """
        if not self.cache_dir.exists():
            return 0

        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1

        logger.info(f"Cleared {count} cache files")
        return count

