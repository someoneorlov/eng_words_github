"""Tests for LLM providers (OpenAI, Anthropic)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from eng_words.llm.base import LLMProvider, LLMResponse


class TestOpenAIProvider:
    """Tests for OpenAI provider."""

    def test_provider_requires_api_key(self):
        """Test that provider requires API key."""
        from eng_words.llm.providers.openai import OpenAIProvider

        # Clear env var if set
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIProvider()

    def test_provider_accepts_api_key_from_env(self):
        """Test that provider accepts API key from environment."""
        from eng_words.llm.providers.openai import OpenAIProvider

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("eng_words.llm.providers.openai.OpenAI"):
                provider = OpenAIProvider()
                assert provider is not None

    def test_provider_accepts_explicit_api_key(self):
        """Test that provider accepts explicit API key."""
        from eng_words.llm.providers.openai import OpenAIProvider

        with patch("eng_words.llm.providers.openai.OpenAI"):
            provider = OpenAIProvider(api_key="explicit-key")
            assert provider is not None

    def test_complete_returns_llm_response(self):
        """Test that complete returns LLMResponse."""
        from eng_words.llm.providers.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test response"))]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("eng_words.llm.providers.openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")
            response = provider.complete("test prompt")

            assert isinstance(response, LLMResponse)
            assert response.content == "test response"
            assert response.model == "gpt-4o-mini"
            assert response.input_tokens == 10
            assert response.output_tokens == 5

    def test_complete_json_returns_dict(self):
        """Test that complete_json returns parsed JSON."""
        from eng_words.llm.providers.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "test"}'))]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("eng_words.llm.providers.openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")
            result = provider.complete_json("test prompt")

            assert isinstance(result, dict)
            assert result == {"result": "test"}

    def test_complete_json_handles_invalid_json(self):
        """Test that complete_json raises on invalid JSON."""
        from eng_words.llm.providers.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="not valid json"))]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("eng_words.llm.providers.openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")
            with pytest.raises(ValueError, match="Invalid JSON"):
                provider.complete_json("test prompt")

    def test_uses_default_temperature(self):
        """Test that provider uses temperature=0 by default."""
        from eng_words.llm.providers.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]
        mock_response.model = "gpt-4o-mini"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("eng_words.llm.providers.openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")
            provider.complete("test")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["temperature"] == 0.0

    def test_uses_default_model(self):
        """Test that provider uses gpt-4.1-mini by default."""
        from eng_words.llm.providers.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]
        mock_response.model = "gpt-4.1-mini"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("eng_words.llm.providers.openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")
            provider.complete("test")

            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4.1-mini"

    def test_cost_calculation(self):
        """Test cost calculation for GPT-4.1-mini."""
        from eng_words.llm.providers.openai import OpenAIProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="test"))]
        mock_response.model = "gpt-4.1-mini"
        # 1000 input tokens, 500 output tokens
        mock_response.usage = MagicMock(prompt_tokens=1000, completion_tokens=500)
        mock_client.chat.completions.create.return_value = mock_response

        with patch("eng_words.llm.providers.openai.OpenAI", return_value=mock_client):
            provider = OpenAIProvider(api_key="test-key")
            response = provider.complete("test")

            # gpt-4.1-mini (Standard): $0.40/1M input, $1.60/1M output
            expected_cost = (1000 * 0.40 + 500 * 1.60) / 1_000_000
            assert abs(response.cost_usd - expected_cost) < 0.0001


class TestAnthropicProvider:
    """Tests for Anthropic provider."""

    def test_provider_requires_api_key(self):
        """Test that provider requires API key."""
        from eng_words.llm.providers.anthropic import AnthropicProvider

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AnthropicProvider()

    def test_provider_accepts_api_key_from_env(self):
        """Test that provider accepts API key from environment."""
        from eng_words.llm.providers.anthropic import AnthropicProvider

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("eng_words.llm.providers.anthropic.Anthropic"):
                provider = AnthropicProvider()
                assert provider is not None

    def test_complete_returns_llm_response(self):
        """Test that complete returns LLMResponse."""
        from eng_words.llm.providers.anthropic import AnthropicProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="test response")]
        mock_response.model = "claude-3-5-haiku-20241022"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        with patch("eng_words.llm.providers.anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="test-key")
            response = provider.complete("test prompt")

            assert isinstance(response, LLMResponse)
            assert response.content == "test response"
            assert response.model == "claude-3-5-haiku-20241022"

    def test_uses_default_model(self):
        """Test that provider uses claude-3-5-haiku by default."""
        from eng_words.llm.providers.anthropic import AnthropicProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="test")]
        mock_response.model = "claude-3-5-haiku-20241022"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)
        mock_client.messages.create.return_value = mock_response

        with patch("eng_words.llm.providers.anthropic.Anthropic", return_value=mock_client):
            provider = AnthropicProvider(api_key="test-key")
            provider.complete("test")

            call_kwargs = mock_client.messages.create.call_args.kwargs
            assert call_kwargs["model"] == "claude-3-5-haiku-20241022"


class TestGeminiProvider:
    """Tests for Gemini provider."""

    def test_provider_requires_api_key(self):
        """Test that provider requires API key."""
        from eng_words.llm.providers.gemini import GeminiProvider

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                GeminiProvider()

    def test_provider_accepts_api_key_from_env(self):
        """Test that provider accepts API key from environment."""
        from eng_words.llm.providers.gemini import GeminiProvider

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            with patch("eng_words.llm.providers.gemini.genai"):
                provider = GeminiProvider()
                assert provider is not None

    def test_provider_accepts_explicit_api_key(self):
        """Test that provider accepts explicit API key."""
        from eng_words.llm.providers.gemini import GeminiProvider

        with patch("eng_words.llm.providers.gemini.genai"):
            provider = GeminiProvider(api_key="explicit-key")
            assert provider is not None

    def test_complete_returns_llm_response(self):
        """Test that complete returns LLMResponse."""
        from eng_words.llm.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test response"
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
        mock_client.models.generate_content.return_value = mock_response

        with patch("eng_words.llm.providers.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(api_key="test-key", model="gemini-3-flash-preview")
            response = provider.complete("test prompt")

            assert isinstance(response, LLMResponse)
            assert response.content == "test response"
            assert response.input_tokens == 10
            assert response.output_tokens == 5

    def test_complete_json_returns_dict(self):
        """Test that complete_json returns parsed JSON."""
        from eng_words.llm.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"result": "test"}'
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
        mock_client.models.generate_content.return_value = mock_response

        with patch("eng_words.llm.providers.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(api_key="test-key", model="gemini-3-flash-preview")
            result = provider.complete_json("test prompt")

            assert isinstance(result, dict)
            assert result == {"result": "test"}

    def test_complete_json_handles_invalid_json(self):
        """Test that complete_json raises on invalid JSON."""
        from eng_words.llm.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "not valid json"
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
        mock_client.models.generate_content.return_value = mock_response

        with patch("eng_words.llm.providers.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(api_key="test-key", model="gemini-3-flash-preview")
            with pytest.raises(ValueError, match="Invalid JSON"):
                provider.complete_json("test prompt")

    def test_uses_default_temperature(self):
        """Test that provider uses temperature=0 by default."""
        from eng_words.llm.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=5)
        mock_client.models.generate_content.return_value = mock_response

        with patch("eng_words.llm.providers.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(api_key="test-key", model="gemini-3-flash-preview")
            provider.complete("test")

            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            assert call_kwargs["config"].temperature == 0.0

    def test_cost_calculation(self):
        """Test cost calculation for gemini-3-flash-preview."""
        from eng_words.llm.providers.gemini import GeminiProvider

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "test"
        # 1000 input tokens, 500 output tokens
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=1000, candidates_token_count=500
        )
        mock_client.models.generate_content.return_value = mock_response

        with patch("eng_words.llm.providers.gemini.genai") as mock_genai:
            mock_genai.Client.return_value = mock_client
            provider = GeminiProvider(api_key="test-key", model="gemini-3-flash-preview")
            response = provider.complete("test")

            # gemini-3-flash-preview: $0.50/1M input, $3.00/1M output
            expected_cost = (1000 * 0.50 + 500 * 3.00) / 1_000_000
            assert abs(response.cost_usd - expected_cost) < 0.0001

    def test_estimate_cost_method(self):
        """Test estimate_cost method."""
        from eng_words.llm.providers.gemini import GeminiProvider

        with patch("eng_words.llm.providers.gemini.genai"):
            provider = GeminiProvider(api_key="test-key", model="gemini-3-flash-preview")
            # gemini-3-flash-preview: $0.50/1M input, $3.00/1M output
            cost = provider.estimate_cost(input_tokens=1000, output_tokens=500)
            expected_cost = (1000 * 0.50 + 500 * 3.00) / 1_000_000
            assert abs(cost - expected_cost) < 0.0001
