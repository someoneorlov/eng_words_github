"""Tests for synset_validator module."""

import json
from unittest.mock import MagicMock

import pytest

from eng_words.llm.base import LLMProvider, LLMResponse
from eng_words.llm.response_cache import ResponseCache
from eng_words.validation.synset_validator import (
    validate_examples_for_synset_group,
)


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[LLMResponse | dict] | None = None):
        self.responses = responses or []
        self.call_count = 0
        self.model = "test-model"

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """Return mock response."""
        if self.call_count < len(self.responses):
            response_data = self.responses[self.call_count]
            self.call_count += 1

            # Handle both LLMResponse and dict
            if isinstance(response_data, LLMResponse):
                return response_data
            else:
                return LLMResponse(
                    content=json.dumps(response_data),
                    model=self.model,
                    input_tokens=100,
                    output_tokens=50,
                    cost_usd=0.001,
                )
        # Default response
        return LLMResponse(
            content=json.dumps({"valid_indices": [], "invalid_indices": []}),
            model=self.model,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )

    def complete_json(self, prompt: str, schema: dict | None = None, **kwargs) -> dict:
        """Return mock JSON response."""
        response = self.complete(prompt, **kwargs)
        return json.loads(response.content)

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost."""
        return 0.001


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_cache():
    """Create a mock cache."""
    cache = MagicMock(spec=ResponseCache)
    cache.get.return_value = None
    cache.set.return_value = None

    # Mock generate_key to return a hash-like string
    def mock_generate_key(model, prompt, temperature):
        import hashlib

        content = f"{model}|{prompt}|{temperature}"
        return hashlib.sha256(content.encode()).hexdigest()

    cache.generate_key.side_effect = mock_generate_key
    return cache


@pytest.fixture
def sample_examples():
    """Sample examples for testing."""
    return [
        (12008, "I waited long for the bus."),
        (12009, "The long road stretched ahead."),
        (15000, "He longed for home."),  # Different meaning
    ]


class TestValidateExamplesForSynsetGroup:
    """Tests for validate_examples_for_synset_group function."""

    def test_valid_examples_found(self, mock_provider, mock_cache, sample_examples):
        """Test when valid examples are found."""
        # Mock LLM response
        mock_provider.responses = [
            {
                "valid_indices": [1, 2],
                "invalid_indices": [3],
                "validation_details": {
                    "1": {"valid": True, "reason": "Matches long.r.01 meaning"},
                    "2": {"valid": True, "reason": "Matches long.s.01 meaning"},
                    "3": {"valid": False, "reason": "Different meaning (long.v.01)"},
                },
            }
        ]

        result = validate_examples_for_synset_group(
            lemma="long",
            synset_group=["long.r.01", "long.s.01"],
            primary_synset="long.r.01",
            examples=sample_examples,
            provider=mock_provider,
            cache=mock_cache,
        )

        assert result["has_valid"] is True
        assert result["valid_sentence_ids"] == [12008, 12009]
        assert result["invalid_sentence_ids"] == [15000]
        assert mock_provider.call_count == 1

    def test_no_valid_examples(self, mock_provider, mock_cache, sample_examples):
        """Test when no valid examples are found."""
        # Mock LLM response - all invalid
        mock_provider.responses = [
            {
                "valid_indices": [],
                "invalid_indices": [1, 2, 3],
                "validation_details": {
                    "1": {"valid": False, "reason": "Different meaning"},
                    "2": {"valid": False, "reason": "Different meaning"},
                    "3": {"valid": False, "reason": "Different meaning"},
                },
            }
        ]

        result = validate_examples_for_synset_group(
            lemma="bank",
            synset_group=["bank.n.01"],  # Financial institution
            primary_synset="bank.n.01",
            examples=[(20000, "They sat on the river bank.")],  # bank.n.02
            provider=mock_provider,
            cache=mock_cache,
        )

        assert result["has_valid"] is False
        assert result["valid_sentence_ids"] == []
        assert len(result["invalid_sentence_ids"]) == 1

    def test_empty_examples(self, mock_provider, mock_cache):
        """Test with empty examples list."""
        result = validate_examples_for_synset_group(
            lemma="test",
            synset_group=["test.n.01"],
            primary_synset="test.n.01",
            examples=[],
            provider=mock_provider,
            cache=mock_cache,
        )

        assert result["has_valid"] is False
        assert result["valid_sentence_ids"] == []
        assert result["invalid_sentence_ids"] == []

    def test_caching(self, mock_provider, mock_cache, sample_examples):
        """Test that results are cached."""
        # First call - cache miss
        mock_provider.responses = [
            {
                "valid_indices": [1, 2],
                "invalid_indices": [3],
            }
        ]

        result1 = validate_examples_for_synset_group(
            lemma="long",
            synset_group=["long.r.01"],
            primary_synset="long.r.01",
            examples=sample_examples,
            provider=mock_provider,
            cache=mock_cache,
        )

        assert mock_provider.call_count == 1
        assert mock_cache.set.called

        # Second call - should use cache
        mock_cache.get.return_value = LLMResponse(
            content=json.dumps({"valid_indices": [1, 2], "invalid_indices": [3]}),
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )

        result2 = validate_examples_for_synset_group(
            lemma="long",
            synset_group=["long.r.01"],
            primary_synset="long.r.01",
            examples=sample_examples,
            provider=mock_provider,
            cache=mock_cache,
        )

        # Should not call provider again
        assert mock_provider.call_count == 1
        assert result1 == result2

    def test_invalid_json_retry(self, mock_provider, mock_cache, sample_examples):
        """Test retry on invalid JSON response."""
        # First call - invalid JSON
        # Second call - valid JSON
        mock_provider.responses = [
            LLMResponse(
                content="Invalid JSON response",
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            ),
            {
                "valid_indices": [1],
                "invalid_indices": [2, 3],
            },
        ]

        result = validate_examples_for_synset_group(
            lemma="long",
            synset_group=["long.r.01"],
            primary_synset="long.r.01",
            examples=sample_examples,
            provider=mock_provider,
            cache=mock_cache,
        )

        # Should retry and succeed
        assert mock_provider.call_count == 2
        assert result["has_valid"] is True

    def test_all_retries_fail_conservative(self, mock_provider, mock_cache, sample_examples):
        """Test conservative approach when all retries fail."""
        # All calls return invalid JSON
        # Need enough responses for all retry attempts
        # max_retries=2 means 3 total attempts (initial + 2 retries)
        # But complete_json calls complete() twice, so we need 6 responses
        mock_provider.responses = [
            LLMResponse(
                content="Invalid JSON",
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
        ] * 10  # Enough for all attempts

        result = validate_examples_for_synset_group(
            lemma="long",
            synset_group=["long.r.01"],
            primary_synset="long.r.01",
            examples=sample_examples,
            provider=mock_provider,
            cache=mock_cache,
            max_retries=2,  # Explicitly set
        )

        # Conservative: all invalid
        assert result["has_valid"] is False
        assert result["valid_sentence_ids"] == []
        assert len(result["invalid_sentence_ids"]) == len(sample_examples)

    def test_single_synset_group(self, mock_provider, mock_cache):
        """Test with single synset in group."""
        mock_provider.responses = [
            {
                "valid_indices": [1],
                "invalid_indices": [],
            }
        ]

        result = validate_examples_for_synset_group(
            lemma="bank",
            synset_group=["bank.n.01"],
            primary_synset="bank.n.01",
            examples=[(100, "I went to the bank to deposit money.")],
            provider=mock_provider,
            cache=mock_cache,
        )

        assert result["has_valid"] is True
        assert result["valid_sentence_ids"] == [100]

    def test_multiple_synsets_in_group(self, mock_provider, mock_cache):
        """Test with multiple synsets in group."""
        mock_provider.responses = [
            {
                "valid_indices": [1, 2],
                "invalid_indices": [],
            }
        ]

        examples = [
            (1, "I waited long for the bus."),  # long.r.01
            (2, "The long road stretched ahead."),  # long.s.01
        ]

        result = validate_examples_for_synset_group(
            lemma="long",
            synset_group=["long.r.01", "long.s.01"],
            primary_synset="long.r.01",
            examples=examples,
            provider=mock_provider,
            cache=mock_cache,
        )

        assert result["has_valid"] is True
        assert result["valid_sentence_ids"] == [1, 2]
