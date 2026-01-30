#!/usr/bin/env python3
"""Прямой тест retry логики."""

import json
from unittest.mock import MagicMock

from eng_words.llm.base import LLMProvider, LLMResponse
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.retry import call_llm_with_retry


class MockProvider(LLMProvider):
    def __init__(self):
        self.call_count = 0
        self.model = "test"
        self.temperature = 0.0
    
    def complete(self, prompt, **kwargs):
        self.call_count += 1
        print(f"  complete() called (count: {self.call_count})")
        return LLMResponse(
            content="Invalid JSON",
            model=self.model,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
    
    def complete_json(self, prompt, schema=None, **kwargs):
        print(f"  complete_json() called")
        try:
            return json.loads(self.complete(prompt).content)
        except:
            raise
    
    def estimate_cost(self, input_tokens, output_tokens):
        return 0.001


provider = MockProvider()
cache = MagicMock(spec=ResponseCache)
cache.get.return_value = None
cache.generate_key.return_value = "test-key"

print("Testing retry with max_retries=2 (should be 3 attempts total)...")
try:
    result = call_llm_with_retry(
        provider=provider,
        prompt="test",
        cache=cache,
        max_retries=2,
        validate_json=True,
    )
    print(f"❌ ERROR: No exception raised! Result: {result}")
except ValueError as e:
    print(f"✅ Exception raised correctly: {e}")
    print(f"   Total calls: {provider.call_count} (expected 3-6)")
except Exception as e:
    print(f"❌ Unexpected exception: {type(e).__name__}: {e}")
