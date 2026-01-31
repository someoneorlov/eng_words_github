"""Tests for WSD Gold Dataset LLM providers.

Following TDD: tests are written before implementation.
"""

from unittest.mock import MagicMock, patch

from eng_words.wsd_gold.models import (
    Candidate,
    ExampleMetadata,
    GoldExample,
    LLMUsage,
    ModelOutput,
    TargetWord,
)


def make_example(
    example_id: str = "test",
    lemma: str = "bank",
    pos: str = "NOUN",
) -> GoldExample:
    """Helper to create test examples."""
    return GoldExample(
        example_id=example_id,
        source_id="test_book",
        source_bucket="classic_fiction",
        year_bucket="pre_1950",
        genre_bucket="fiction",
        text_left="The river ",
        target=TargetWord(surface=lemma, lemma=lemma, pos=pos, char_span=(11, 11 + len(lemma))),
        text_right=" was muddy.",
        context_window=f"The river {lemma} was muddy.",
        candidates=[
            Candidate(
                synset_id=f"{lemma}.n.01",
                gloss="sloping land beside water",
                examples=["river bank"],
            ),
            Candidate(
                synset_id=f"{lemma}.n.02",
                gloss="financial institution",
                examples=["deposit at bank"],
            ),
        ],
        metadata=ExampleMetadata(
            wn_sense_count=2,
            baseline_top1=f"{lemma}.n.01",
            baseline_margin=0.3,
            is_multiword=False,
        ),
    )


class TestGoldLabelProvider:
    """Tests for GoldLabelProvider ABC."""

    def test_cannot_instantiate_directly(self):
        """Cannot instantiate ABC directly."""
        import pytest

        from eng_words.wsd_gold.providers import GoldLabelProvider

        with pytest.raises(TypeError):
            GoldLabelProvider()

    def test_concrete_implementation_works(self):
        """Concrete implementation can be instantiated."""
        from eng_words.wsd_gold.providers import GoldLabelProvider

        class ConcreteProvider(GoldLabelProvider):
            @property
            def name(self) -> str:
                return "test"

            def label_one(self, example):
                return ModelOutput(
                    chosen_synset_id="test.n.01",
                    confidence=0.9,
                    flags=[],
                    raw_text="{}",
                    usage=LLMUsage(input_tokens=100, output_tokens=20),
                )

            def label_batch(self, examples):
                return [self.label_one(ex) for ex in examples]

            def estimate_cost(self, examples):
                return len(examples) * 0.001

        provider = ConcreteProvider()
        assert provider.name == "test"


class TestOpenAIGoldProvider:
    """Tests for OpenAI Gold labeling provider."""

    def test_provider_has_name(self):
        """Provider has name property."""
        from eng_words.wsd_gold.providers import OpenAIGoldProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIGoldProvider()
            assert provider.name == "openai"

    def test_label_one_returns_model_output(self):
        """label_one returns ModelOutput."""
        from eng_words.wsd_gold.providers import OpenAIGoldProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIGoldProvider()

            # Mock the OpenAI client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = (
                '{"chosen_synset_id": "bank.n.01", "confidence": 0.9, "flags": []}'
            )
            mock_response.usage.prompt_tokens = 500
            mock_response.usage.completion_tokens = 30

            with patch.object(
                provider.client.chat.completions, "create", return_value=mock_response
            ):
                example = make_example()
                output = provider.label_one(example)

                assert isinstance(output, ModelOutput)
                assert output.chosen_synset_id == "bank.n.01"
                assert output.confidence == 0.9

    def test_label_batch_calls_label_one(self):
        """label_batch processes each example."""
        from eng_words.wsd_gold.providers import OpenAIGoldProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIGoldProvider()

            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = (
                '{"chosen_synset_id": "bank.n.01", "confidence": 0.9, "flags": []}'
            )
            mock_response.usage.prompt_tokens = 500
            mock_response.usage.completion_tokens = 30

            with patch.object(
                provider.client.chat.completions, "create", return_value=mock_response
            ):
                examples = [make_example(example_id=f"ex{i}") for i in range(3)]
                outputs = provider.label_batch(examples)

                assert len(outputs) == 3
                assert all(isinstance(o, ModelOutput) for o in outputs)

    def test_estimate_cost(self):
        """estimate_cost returns float."""
        from eng_words.wsd_gold.providers import OpenAIGoldProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIGoldProvider()
            examples = [make_example() for _ in range(10)]

            cost = provider.estimate_cost(examples)

            assert isinstance(cost, float)
            assert cost > 0

    def test_uses_temperature_zero(self):
        """Uses temperature=0 for determinism."""
        from eng_words.wsd_gold.providers import OpenAIGoldProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIGoldProvider()
            assert provider.temperature == 0.0

    def test_handles_invalid_json(self):
        """Handles invalid JSON response gracefully."""
        from eng_words.wsd_gold.providers import OpenAIGoldProvider

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            provider = OpenAIGoldProvider()

            # First call returns invalid JSON, second returns valid
            mock_invalid = MagicMock()
            mock_invalid.choices = [MagicMock()]
            mock_invalid.choices[0].message.content = "not valid json"
            mock_invalid.usage.prompt_tokens = 500
            mock_invalid.usage.completion_tokens = 30

            mock_valid = MagicMock()
            mock_valid.choices = [MagicMock()]
            mock_valid.choices[0].message.content = (
                '{"chosen_synset_id": "bank.n.01", "confidence": 0.9, "flags": []}'
            )
            mock_valid.usage.prompt_tokens = 500
            mock_valid.usage.completion_tokens = 30

            with patch.object(
                provider.client.chat.completions,
                "create",
                side_effect=[mock_invalid, mock_valid],
            ):
                example = make_example()
                output = provider.label_one(example)

                # Should retry and succeed
                assert output.chosen_synset_id == "bank.n.01"


class TestAnthropicGoldProvider:
    """Tests for Anthropic Gold labeling provider."""

    def test_provider_has_name(self):
        """Provider has name property."""
        from eng_words.wsd_gold.providers import AnthropicGoldProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicGoldProvider()
            assert provider.name == "anthropic"

    def test_label_one_returns_model_output(self):
        """label_one returns ModelOutput."""
        from eng_words.wsd_gold.providers import AnthropicGoldProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicGoldProvider()

            mock_response = MagicMock()
            mock_response.content = [MagicMock()]
            mock_response.content[0].text = (
                '{"chosen_synset_id": "bank.n.01", "confidence": 0.85, "flags": []}'
            )
            mock_response.usage.input_tokens = 500
            mock_response.usage.output_tokens = 30

            with patch.object(provider.client.messages, "create", return_value=mock_response):
                example = make_example()
                output = provider.label_one(example)

                assert isinstance(output, ModelOutput)
                assert output.chosen_synset_id == "bank.n.01"

    def test_estimate_cost(self):
        """estimate_cost returns float."""
        from eng_words.wsd_gold.providers import AnthropicGoldProvider

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            provider = AnthropicGoldProvider()
            examples = [make_example() for _ in range(10)]

            cost = provider.estimate_cost(examples)

            assert isinstance(cost, float)
            assert cost > 0


class TestGeminiGoldProvider:
    """Tests for Gemini Gold labeling provider."""

    def test_provider_has_name(self):
        """Provider has name property."""
        from eng_words.wsd_gold.providers import GeminiGoldProvider

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiGoldProvider()
            assert provider.name == "gemini"

    def test_estimate_cost(self):
        """estimate_cost returns float."""
        from eng_words.wsd_gold.providers import GeminiGoldProvider

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            provider = GeminiGoldProvider()
            examples = [make_example() for _ in range(10)]

            cost = provider.estimate_cost(examples)

            assert isinstance(cost, float)
            assert cost > 0


class TestBuildGoldLabelingPrompt:
    """Tests for prompt building function."""

    def test_prompt_includes_context(self):
        """Prompt includes context window."""
        from eng_words.wsd_gold.providers.prompts import build_gold_labeling_prompt

        example = make_example()
        prompt = build_gold_labeling_prompt(example)

        assert example.context_window in prompt
        assert example.target.lemma in prompt

    def test_prompt_includes_all_candidates(self):
        """Prompt includes all candidates."""
        from eng_words.wsd_gold.providers.prompts import build_gold_labeling_prompt

        example = make_example()
        prompt = build_gold_labeling_prompt(example)

        for candidate in example.candidates:
            assert candidate.synset_id in prompt
            assert candidate.gloss in prompt

    def test_prompt_includes_json_format(self):
        """Prompt includes JSON format instructions."""
        from eng_words.wsd_gold.providers.prompts import build_gold_labeling_prompt

        example = make_example()
        prompt = build_gold_labeling_prompt(example)

        assert "chosen_synset_id" in prompt
        assert "confidence" in prompt
        assert "flags" in prompt

    def test_static_prefix_meets_caching_threshold(self):
        """Static prefix is >= 1024 tokens for prompt caching."""
        from eng_words.wsd_gold.providers.prompts import GOLD_LABELING_SYSTEM_PROMPT

        # Rough estimate: 4 chars = 1 token
        estimated_tokens = len(GOLD_LABELING_SYSTEM_PROMPT) // 4

        assert (
            estimated_tokens >= 1024
        ), f"Static prefix should be >= 1024 tokens for caching, got {estimated_tokens}"


class TestParseGoldLabelResponse:
    """Tests for response parsing function."""

    def test_parses_valid_json(self):
        """Parses valid JSON response."""
        from eng_words.wsd_gold.providers.prompts import parse_gold_label_response

        raw = '{"chosen_synset_id": "bank.n.01", "confidence": 0.9, "flags": []}'
        candidates = ["bank.n.01", "bank.n.02"]
        usage = LLMUsage(input_tokens=100, output_tokens=20)

        output = parse_gold_label_response(raw, candidates, usage)

        assert output is not None
        assert output.chosen_synset_id == "bank.n.01"
        assert output.confidence == 0.9
        assert output.flags == []

    def test_returns_none_for_invalid_json(self):
        """Returns None for invalid JSON."""
        from eng_words.wsd_gold.providers.prompts import parse_gold_label_response

        raw = "not valid json"
        candidates = ["bank.n.01", "bank.n.02"]
        usage = LLMUsage(input_tokens=100, output_tokens=20)

        output = parse_gold_label_response(raw, candidates, usage)

        assert output is None

    def test_returns_none_for_invalid_synset(self):
        """Returns None when synset not in candidates."""
        from eng_words.wsd_gold.providers.prompts import parse_gold_label_response

        raw = '{"chosen_synset_id": "bank.n.99", "confidence": 0.9, "flags": []}'
        candidates = ["bank.n.01", "bank.n.02"]
        usage = LLMUsage(input_tokens=100, output_tokens=20)

        output = parse_gold_label_response(raw, candidates, usage)

        assert output is None

    def test_handles_none_of_the_above(self):
        """Handles none_of_the_above flag."""
        from eng_words.wsd_gold.providers.prompts import parse_gold_label_response

        raw = '{"chosen_synset_id": "", "confidence": 0.8, "flags": ["none_of_the_above"]}'
        candidates = ["bank.n.01", "bank.n.02"]
        usage = LLMUsage(input_tokens=100, output_tokens=20)

        output = parse_gold_label_response(raw, candidates, usage)

        assert output is not None
        assert "none_of_the_above" in output.flags

    def test_returns_none_for_confidence_out_of_range(self):
        """Returns None when confidence is out of [0, 1] range."""
        from eng_words.wsd_gold.providers.prompts import parse_gold_label_response

        # Confidence > 1
        raw = '{"chosen_synset_id": "bank.n.01", "confidence": 1.5, "flags": []}'
        candidates = ["bank.n.01", "bank.n.02"]
        usage = LLMUsage(input_tokens=100, output_tokens=20)

        output = parse_gold_label_response(raw, candidates, usage)
        assert output is None

        # Confidence < 0
        raw = '{"chosen_synset_id": "bank.n.01", "confidence": -0.5, "flags": []}'
        output = parse_gold_label_response(raw, candidates, usage)
        assert output is None

    def test_filters_invalid_flags(self):
        """Filters out invalid flags but keeps valid ones."""
        from eng_words.wsd_gold.providers.prompts import parse_gold_label_response

        raw = '{"chosen_synset_id": "bank.n.01", "confidence": 0.9, "flags": ["metaphor", "invalid_flag"]}'
        candidates = ["bank.n.01", "bank.n.02"]
        usage = LLMUsage(input_tokens=100, output_tokens=20)

        output = parse_gold_label_response(raw, candidates, usage)

        assert output is not None
        assert "metaphor" in output.flags
        assert "invalid_flag" not in output.flags
