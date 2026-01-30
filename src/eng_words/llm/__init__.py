"""LLM integration module for card generation.

This module provides:
- LLMProvider abstract base class for different LLM backends
- Concrete providers for OpenAI, Anthropic, Gemini
- SmartCardGenerator for creating Anki flashcards
- ResponseCache for caching LLM responses
- Retry utilities for robust LLM calls

Note: Legacy modules (cache.py, card_generator.py, evaluator.py, prompts.py)
have been moved to _archive/ folder.
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
