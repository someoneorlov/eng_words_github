"""LLM integration module.

Provides:
- LLMProvider abstract base class for different LLM backends
- Concrete providers for OpenAI, Anthropic, Gemini
- ResponseCache for caching LLM responses
- Retry utilities for robust LLM calls

Used by word_family/clusterer when running in-process clustering.
Batch script (run_pipeline_b_batch.py) calls Gemini API directly and does not use this module.
Legacy card_generator/cache/evaluator removed; see docs/REMOVED_ARCHIVE.md.
"""

from eng_words.llm.base import LLMProvider, LLMResponse
from eng_words.llm.providers import AnthropicProvider, OpenAIProvider
from eng_words.llm.retry import call_llm_json, call_llm_with_retry

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "call_llm_with_retry",
    "call_llm_json",
]
