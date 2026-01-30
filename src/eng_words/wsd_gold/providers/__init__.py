"""LLM providers for WSD Gold labeling.

This module provides providers for different LLM APIs:
- OpenAIGoldProvider: GPT models
- AnthropicGoldProvider: Claude models
- GeminiGoldProvider: Google Gemini models
"""

from eng_words.wsd_gold.providers.anthropic_gold import AnthropicGoldProvider
from eng_words.wsd_gold.providers.base import GoldLabelProvider
from eng_words.wsd_gold.providers.gemini_gold import GeminiGoldProvider
from eng_words.wsd_gold.providers.openai_gold import OpenAIGoldProvider

__all__ = [
    "GoldLabelProvider",
    "OpenAIGoldProvider",
    "AnthropicGoldProvider",
    "GeminiGoldProvider",
]
