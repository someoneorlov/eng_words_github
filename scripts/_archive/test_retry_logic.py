#!/usr/bin/env python3
"""Тест логики retry."""

import json
from unittest.mock import MagicMock

from eng_words.llm.base import LLMProvider, LLMResponse
from eng_words.llm.retry import call_llm_with_retry


class MockProvider(LLMProvider):
    def __init__(self):
        self.call_count = 0
        self.model = "test"
        self.temperature = 0.0
    
    def complete(self, prompt, **kwargs):
        self.call_count += 1
        return LLMResponse(
            content="Invalid JSON",
            model=self.model,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
    
    def complete_json(self, prompt, schema=None, **kwargs):
        return json.loads(self.complete(prompt).content)
    
    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001


provider = MockProvider()

print("Testing retry logic with invalid JSON...")
try:
    result = call_llm_with_retry(
        provider=provider,
        prompt="test",
        cache=None,
        max_retries=2,
        validate_json=True,
    )
    print(f"ERROR: No exception raised! Result: {result}")
except ValueError as e:
    print(f"✅ Exception raised correctly: {e}")
    print(f"   Call count: {provider.call_count} (expected 3: initial + 2 retries)")
