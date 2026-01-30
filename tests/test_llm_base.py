"""Tests for LLM base module: providers, types, and configuration."""

from dataclasses import asdict

import pytest

from eng_words.constants.llm_config import (
    CARD_BATCH_SIZE,
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_ANTHROPIC,
    DEFAULT_MODEL_OPENAI,
    DEFAULT_TEMPERATURE,
    EVALUATION_BATCH_SIZE,
    MAX_CANDIDATES_PER_ITEM,
    MAX_EXAMPLE_LENGTH,
    MAX_EXAMPLES_PER_SENSE,
    MIN_JURY_CONFIDENCE,
    SPOILER_RISK_THRESHOLD,
)
from eng_words.llm.base import LLMProvider, LLMResponse


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_create_response(self):
        """Test creating an LLM response."""
        response = LLMResponse(
            content='{"result": "test"}',
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        assert response.content == '{"result": "test"}'
        assert response.model == "gpt-4o-mini"
        assert response.input_tokens == 100
        assert response.output_tokens == 50
        assert response.cost_usd == 0.001

    def test_response_to_dict(self):
        """Test converting response to dictionary."""
        response = LLMResponse(
            content="test",
            model="gpt-4o-mini",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.0001,
        )
        data = asdict(response)
        assert data["content"] == "test"
        assert data["model"] == "gpt-4o-mini"

    def test_response_total_tokens(self):
        """Test total tokens calculation."""
        response = LLMResponse(
            content="test",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        assert response.total_tokens == 150


class TestLLMProviderABC:
    """Tests for LLMProvider abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LLMProvider()

    def test_concrete_implementation_required_methods(self):
        """Test that concrete implementations must have required methods."""

        class IncompleteProvider(LLMProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_concrete_implementation_works(self):
        """Test that a proper concrete implementation works."""

        class MockProvider(LLMProvider):
            def complete(self, prompt: str, **kwargs) -> LLMResponse:
                return LLMResponse(
                    content="mock response",
                    model="mock-model",
                    input_tokens=len(prompt),
                    output_tokens=10,
                    cost_usd=0.0,
                )

            def complete_json(self, prompt: str, schema: dict | None = None, **kwargs) -> dict:
                return {"mock": "response"}

            def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
                return 0.0

        provider = MockProvider()
        response = provider.complete("test prompt")
        assert response.content == "mock response"

        json_response = provider.complete_json("test prompt")
        assert json_response == {"mock": "response"}


class TestLLMConfig:
    """Tests for LLM configuration constants."""

    def test_temperature_is_zero(self):
        """Test that default temperature is 0 for determinism."""
        assert DEFAULT_TEMPERATURE == 0.0

    def test_models_are_defined(self):
        """Test that default models are defined."""
        assert DEFAULT_MODEL_OPENAI == "gpt-4.1-mini"
        assert DEFAULT_MODEL_ANTHROPIC == "claude-3-5-haiku-20241022"

    def test_evaluation_config(self):
        """Test evaluation configuration."""
        assert MIN_JURY_CONFIDENCE == 0.8
        assert MAX_CANDIDATES_PER_ITEM == 5
        assert EVALUATION_BATCH_SIZE == 20

    def test_card_generation_config(self):
        """Test card generation configuration."""
        assert MAX_EXAMPLE_LENGTH == 300
        assert MAX_EXAMPLES_PER_SENSE == 15
        assert CARD_BATCH_SIZE == 10

    def test_spoiler_policy(self):
        """Test spoiler policy configuration."""
        assert SPOILER_RISK_THRESHOLD == "none"

    def test_cache_dir_is_path(self):
        """Test that cache dir is a Path object."""
        from pathlib import Path

        assert isinstance(DEFAULT_CACHE_DIR, Path)
        assert "llm_cache" in str(DEFAULT_CACHE_DIR)
