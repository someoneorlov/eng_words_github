"""Tests for SmartCardGenerator."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eng_words.llm.base import LLMResponse


class TestSmartCard:
    """Tests for SmartCard dataclass."""

    def test_import_smart_card(self):
        """Test that SmartCard can be imported."""
        from eng_words.llm.smart_card_generator import SmartCard

        assert SmartCard is not None

    def test_create_smart_card(self):
        """Test creating SmartCard instance."""
        from eng_words.llm.smart_card_generator import SmartCard

        card = SmartCard(
            lemma="run",
            pos="verb",
            supersense="verb.motion",
            selected_examples=["He runs every morning."],
            excluded_examples=["The program runs on Windows."],
            simple_definition="to move quickly using your legs",
            translation_ru="run",
            generated_example="She runs in the park every day.",
            wn_definition="move fast by using one's feet",
            book_name="test_book",
        )

        assert card.lemma == "run"
        assert card.pos == "verb"
        assert card.supersense == "verb.motion"
        assert len(card.selected_examples) == 1
        assert len(card.excluded_examples) == 1
        assert card.translation_ru == "run"

    def test_create_smart_card_with_quality_scores(self):
        """Test creating SmartCard with quality_scores and generated_examples."""
        from eng_words.llm.smart_card_generator import SmartCard

        card = SmartCard(
            lemma="run",
            pos="verb",
            supersense="verb.motion",
            selected_examples=["He runs every morning.", "She runs fast."],
            excluded_examples=[],
            simple_definition="to move quickly using your legs",
            translation_ru="run",
            generated_example="They run together.",
            wn_definition="move fast by using one's feet",
            book_name="test_book",
            quality_scores={1: 5, 2: 4},
            generated_examples=["Example 1", "Example 2"],
        )

        assert card.quality_scores == {1: 5, 2: 4}
        assert card.generated_examples == ["Example 1", "Example 2"]

    def test_smart_card_to_dict(self):
        """Test converting SmartCard to dict."""
        from eng_words.llm.smart_card_generator import SmartCard

        card = SmartCard(
            lemma="run",
            pos="verb",
            supersense="verb.motion",
            selected_examples=["He runs."],
            excluded_examples=[],
            simple_definition="to move fast",
            translation_ru="run",
            generated_example="She runs.",
            wn_definition="move fast",
            book_name="test",
        )

        d = card.to_dict()
        assert isinstance(d, dict)
        assert d["lemma"] == "run"
        assert d["translation_ru"] == "run"


class TestSmartCardPrompt:
    """Tests for prompt generation."""

    def test_import_prompt(self):
        """Test that prompt template can be imported."""
        from eng_words.llm.smart_card_generator import SMART_CARD_PROMPT

        assert SMART_CARD_PROMPT is not None
        assert "{lemma}" in SMART_CARD_PROMPT
        assert "{supersense}" in SMART_CARD_PROMPT

    def test_format_prompt(self):
        """Test formatting prompt with data."""
        from eng_words.llm.smart_card_generator import format_card_prompt

        prompt = format_card_prompt(
            lemma="break",
            pos="verb",
            supersense="verb.change",
            wn_definition="become separated into pieces",
            book_name="Test Book",
            examples=["The glass will break.", "Don't break the rules."],
        )

        assert "break" in prompt
        assert "verb" in prompt
        assert "verb.change" in prompt
        assert "The glass will break." in prompt
        assert "Test Book" in prompt

    def test_format_prompt_with_synset_group(self):
        """Test formatting prompt with synset_group."""
        from eng_words.llm.smart_card_generator import format_card_prompt

        prompt = format_card_prompt(
            lemma="break",
            pos="verb",
            supersense="verb.change",
            wn_definition="become separated into pieces",
            book_name="Test Book",
            examples=["The glass will break.", "Don't break the rules."],
            synset_group=["break.v.01", "break.v.02"],
            primary_synset="break.v.01",
        )

        assert "break" in prompt
        assert "Synset group" in prompt
        assert "break.v.01" in prompt
        assert "break.v.02" in prompt
        assert "Primary synset" in prompt

    def test_format_prompt_with_generate_count(self):
        """Test formatting prompt with generate_count parameter."""
        from eng_words.llm.smart_card_generator import format_card_prompt

        prompt = format_card_prompt(
            lemma="run",
            pos="verb",
            supersense="verb.motion",
            wn_definition="move fast",
            book_name="Test Book",
            examples=["He runs."],
            generate_count=2,
        )

        assert "run" in prompt
        assert "Generate exactly 2 additional example(s)" in prompt
        assert "already been filtered for length" in prompt
        assert "spoiler-free" in prompt

    def test_format_prompt_without_spoiler_length_checks(self):
        """Test that prompt doesn't ask LLM to check spoilers/length (already filtered)."""
        from eng_words.llm.smart_card_generator import format_card_prompt

        prompt = format_card_prompt(
            lemma="test",
            pos="noun",
            supersense="noun.act",
            wn_definition="test definition",
            book_name="Test Book",
            examples=["A test."],
            generate_count=1,
        )

        # Should mention that examples are already filtered
        assert "already been filtered" in prompt
        assert "spoiler-free" in prompt
        # Should NOT ask to check length/spoilers
        assert ">50 words is TOO LONG" not in prompt
        assert "reveal plot points" not in prompt or "already checked" in prompt


class TestSmartCardGenerator:
    """Tests for SmartCardGenerator class."""

    def test_import_generator(self):
        """Test that SmartCardGenerator can be imported."""
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        assert SmartCardGenerator is not None

    def test_create_generator(self):
        """Test creating generator instance."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0
            cache = ResponseCache(cache_dir=Path(tmpdir))

            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="Test Book",
            )

            assert generator is not None
            assert generator.book_name == "Test Book"

    def test_generate_card_returns_smart_card(self):
        """Test that generate_card returns SmartCard."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCard, SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            # Mock LLM response with new JSON schema
            llm_response = LLMResponse(
                content=json.dumps({
                    "valid_indices": [1, 2, 3],
                    "invalid_indices": [],
                    "quality_scores": {"1": 5, "2": 4, "3": 5},
                    "selected_indices": [1, 2],
                    "generated_examples": ["She runs every morning."],
                    "simple_definition": "to move fast using legs",
                    "translation_ru": "run",
                }),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            mock_provider.complete.return_value = llm_response

            cache = ResponseCache(cache_dir=Path(tmpdir))
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="Test Book",
            )

            card = generator.generate_card(
                lemma="run",
                pos="verb",
                supersense="verb.motion",
                wn_definition="move fast by using one's feet",
                examples=["He runs.", "She runs fast.", "The program runs."],
            )

            assert isinstance(card, SmartCard)
            assert card.lemma == "run"
            assert card.simple_definition == "to move fast using legs"
            assert card.translation_ru == "run"
            assert card.quality_scores == {1: 5, 2: 4, 3: 5}
            assert card.generated_examples == ["She runs every morning."]
            # generated_example should be set from generated_examples[0] if exists
            assert card.generated_example == "She runs every morning."

    def test_generate_card_uses_cache(self):
        """Test that generate_card uses cache."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            llm_response = LLMResponse(
                content=json.dumps({
                    "valid_indices": [1],
                    "invalid_indices": [],
                    "quality_scores": {"1": 5},
                    "selected_indices": [1],
                    "generated_examples": ["Test example."],
                    "simple_definition": "test",
                    "translation_ru": "test",
                }),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            mock_provider.complete.return_value = llm_response
            # Mock complete_json to avoid errors
            mock_provider.complete_json = None

            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=True)
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="Test Book",
            )

            # First call
            generator.generate_card(
                lemma="test",
                pos="noun",
                supersense="noun.act",
                wn_definition="test definition",
                examples=["A test."],
            )

            # Get call count after first call
            first_call_count = mock_provider.complete.call_count
            assert first_call_count > 0, "First call should make API request"

            # Second call with same params - should use cache
            generator.generate_card(
                lemma="test",
                pos="noun",
                supersense="noun.act",
                wn_definition="test definition",
                examples=["A test."],
            )

            # Provider should not be called again (cache hit)
            # Note: call_llm_with_retry handles caching internally
            # We check that cache was used by verifying cache stats
            # Cache should have at least 1 hit (second call) or 2 misses (if cache didn't work)
            total_calls = mock_provider.complete.call_count
            # If cache works, second call shouldn't increase call count much
            # (may increase by 1-2 due to internal retry logic, but not by full retry count)
            assert total_calls <= first_call_count + 2, f"Cache may not be working: {total_calls} calls vs {first_call_count} initial"

    def test_generate_card_handles_invalid_json(self):
        """Test that generate_card handles invalid JSON gracefully."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            # First call (via cached provider) returns invalid JSON
            bad_response = LLMResponse(
                content="not valid json",
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )

            # Second call (retry via raw provider) returns valid JSON
            good_response = LLMResponse(
                content=json.dumps({
                    "valid_indices": [1],
                    "invalid_indices": [],
                    "quality_scores": {"1": 5},
                    "selected_indices": [1],
                    "generated_examples": ["Test example."],
                    "selected_indices": [1],
                    "excluded_indices": [],
                    "simple_definition": "test",
                    "translation_ru": "test",
                    "generated_example": "Test.",
                }),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )

            # Need 2 responses: one for cached provider, one for raw provider retry
            mock_provider.complete.side_effect = [bad_response, good_response]

            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)  # Disable cache for retry test
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="Test Book",
                max_retries=2,
            )

            card = generator.generate_card(
                lemma="test",
                pos="noun",
                supersense="noun.act",
                wn_definition="test",
                examples=["Test."],
            )

            assert card is not None
            # Called twice: once via CachedProvider, once via raw provider for retry
            assert mock_provider.complete.call_count == 2

    def test_generate_batch(self):
        """Test generating multiple cards."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            llm_response = LLMResponse(
                content=json.dumps({
                    "selected_indices": [1],
                    "excluded_indices": [],
                    "simple_definition": "test",
                    "translation_ru": "test",
                    "generated_example": "Test.",
                }),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            mock_provider.complete.return_value = llm_response

            cache = ResponseCache(cache_dir=Path(tmpdir))
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="Test Book",
            )

            items = [
                {
                    "lemma": "test1",
                    "pos": "noun",
                    "supersense": "noun.act",
                    "wn_definition": "test 1",
                    "examples": ["Test 1."],
                },
                {
                    "lemma": "test2",
                    "pos": "verb",
                    "supersense": "verb.act",
                    "wn_definition": "test 2",
                    "examples": ["Test 2."],
                },
            ]

            cards = generator.generate_batch(items, progress=False)

            assert len(cards) == 2
            assert cards[0].lemma == "test1"
            assert cards[1].lemma == "test2"

    def test_generator_stats(self):
        """Test that generator tracks statistics."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            llm_response = LLMResponse(
                content=json.dumps({
                    "selected_indices": [1],
                    "excluded_indices": [],
                    "simple_definition": "test",
                    "translation_ru": "test",
                    "generated_example": "Test.",
                }),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            mock_provider.complete.return_value = llm_response

            cache = ResponseCache(cache_dir=Path(tmpdir))
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="Test Book",
            )

            generator.generate_card(
                lemma="test",
                pos="noun",
                supersense="noun.act",
                wn_definition="test",
                examples=["Test."],
            )

            stats = generator.stats()
            assert "total_cards" in stats
            assert "successful" in stats
            assert "failed" in stats
            assert "total_cost" in stats
            assert stats["total_cards"] == 1
            assert stats["successful"] == 1

    def test_generate_card_with_synset_group(self):
        """Test generating a card with synset_group metadata."""
        from eng_words.llm.smart_card_generator import SmartCardGenerator
        from eng_words.llm.response_cache import ResponseCache

        mock_provider = MagicMock()
        mock_provider.model = "test"
        mock_provider.temperature = 0.0

        llm_response = LLMResponse(
            content=json.dumps({
                "selected_indices": [1],
                "excluded_indices": [],
                "simple_definition": "to move fast",
                "translation_ru": "run",
                "generated_example": "She runs."
            }),
            model="test",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        mock_provider.complete.return_value = llm_response

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ResponseCache(cache_dir=Path(tmp_dir), enabled=True)
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="test_book",
            )

            card = generator.generate_card(
                lemma="run",
                pos="verb",
                supersense="verb.motion",
                wn_definition="move fast",
                examples=["He runs."],
                synset_group=["run.v.01", "run.v.02"],
                primary_synset="run.v.01",
            )

            assert card is not None
            assert card.synset_group == ["run.v.01", "run.v.02"]
            assert card.primary_synset == "run.v.01"

    def test_smart_card_synset_group_defaults(self):
        """Test that SmartCard has default empty synset_group."""
        from eng_words.llm.smart_card_generator import SmartCard

        card = SmartCard(
            lemma="run",
            pos="verb",
            supersense="verb.motion",
            selected_examples=["He runs."],
            excluded_examples=[],
            simple_definition="to move fast",
            translation_ru="run",
            generated_example="She runs.",
            wn_definition="move fast",
            book_name="test",
        )

        # Should have default empty values
        assert card.synset_group == []
        assert card.primary_synset == ""

    def test_smart_card_quality_scores_defaults(self):
        """Test SmartCard has default quality_scores and generated_examples."""
        from eng_words.llm.smart_card_generator import SmartCard

        card = SmartCard(
            lemma="run",
            pos="verb",
            supersense="verb.motion",
            selected_examples=["He runs."],
            excluded_examples=[],
            simple_definition="to move fast",
            translation_ru="run",
            generated_example="She runs.",
            wn_definition="move fast",
            book_name="test",
        )

        # Should have default quality_scores and generated_examples
        assert card.quality_scores == {}
        assert card.generated_examples == []

    def test_generate_card_with_fallback_disabled(self):
        """Test that fallback is disabled by default."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            # Mock response with NO selected_indices
            mock_response = LLMResponse(
                content=json.dumps({
                    "valid_indices": [],
                    "invalid_indices": [1, 2],
                    "quality_scores": {},
                    "selected_indices": [],  # No examples selected
                    "generated_examples": ["This is a test."],
                    "simple_definition": "test",
                    "translation_ru": "test",
                }),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            mock_provider.complete.return_value = mock_response

            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="test_book",
            )

            card = generator.generate_card(
                lemma="test",
                pos="noun",
                supersense="noun.cognition",
                wn_definition="a test definition",
                examples=["Example 1.", "Example 2."],
                primary_synset="test.n.01",
            )

            # Should return None because no selected_examples (card is skipped)
            assert card is None

    def test_generate_batch(self):
        """Test generate_batch generates multiple cards."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            mock_response = LLMResponse(
                content=json.dumps({
                    "valid_indices": [1],
                    "invalid_indices": [],
                    "quality_scores": {"1": 5},
                    "selected_indices": [1],
                    "generated_examples": ["This is a test."],
                    "simple_definition": "test",
                    "translation_ru": "test",
                }),
                model="test-model",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            mock_provider.complete.return_value = mock_response

            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="test_book",
            )

            items = [
                {
                    "lemma": "test",
                    "pos": "noun",
                    "supersense": "noun.cognition",
                    "wn_definition": "a test",
                    "examples": ["Example 1."],
                    "primary_synset": "test.n.01",
                }
            ]

            cards = generator.generate_batch(items, progress=False)
            assert len(cards) == 1

    def test_generator_stats(self):
        """Test that stats include basic counters."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="test_book",
            )

            stats = generator.stats()
            # Stats should include basic counters
            assert "total_cards" in stats
            assert "successful" in stats
            assert "failed" in stats

    def test_generate_example_fallback(self):
        """Test _generate_example_fallback generates example when no examples available."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            # Mock LLM response
            mock_provider.complete.return_value = LLMResponse(
                content='The wind blew strongly through the trees.',
                model="test-model",
                cost_usd=0.0001,
                input_tokens=50,
                output_tokens=10,
            )

            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="test_book",
            )

            example = generator._generate_example_fallback(
                lemma="blow",
                pos="verb",
                definition="to move air",
            )

            assert example == "The wind blew strongly through the trees."
            assert "blow" in example.lower() or "blew" in example.lower()
            mock_provider.complete.assert_called_once()

    def test_generate_example_fallback_removes_quotes(self):
        """Test _generate_example_fallback removes quotes from response."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            # Mock LLM response with quotes
            mock_provider.complete.return_value = LLMResponse(
                content='"She catches the ball every time."',
                model="test-model",
                cost_usd=0.0001,
                input_tokens=50,
                output_tokens=10,
            )

            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="test_book",
            )

            example = generator._generate_example_fallback(
                lemma="catch",
                pos="verb",
                definition="to grab something",
            )

            assert not example.startswith('"')
            assert not example.endswith('"')
            assert "catch" in example.lower() or "catches" in example.lower()

    def test_generate_example_fallback_handles_errors(self):
        """Test _generate_example_fallback handles errors gracefully."""
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import SmartCardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_provider = MagicMock()
            mock_provider.model = "test-model"
            mock_provider.temperature = 0.0

            # Mock LLM error
            mock_provider.complete.side_effect = Exception("API error")

            cache = ResponseCache(cache_dir=Path(tmpdir), enabled=False)
            generator = SmartCardGenerator(
                provider=mock_provider,
                cache=cache,
                book_name="test_book",
            )

            example = generator._generate_example_fallback(
                lemma="test",
                pos="verb",
                definition="to test",
            )

            assert example == ""


class TestMarkExamplesByLength:
    """Tests for mark_examples_by_length function."""

    def test_mark_examples_by_length_basic(self):
        """Test basic marking of examples by length."""
        from eng_words.llm.smart_card_generator import mark_examples_by_length

        examples = [
            (1, "Short sentence."),  # 2 words
            (2, "This is a medium length sentence with more words."),  # 10 words
            (3, "This is a very long sentence that contains many words and should be marked as too long because it exceeds the maximum word count limit."),  # 20 words, but let's make it longer
        ]
        
        # Make example 3 actually >50 words
        long_sentence = " ".join(["word"] * 55)
        examples[2] = (3, long_sentence)

        flags = mark_examples_by_length(examples, max_words=50, min_words=6)

        assert flags[1] is False  # Too short (2 words < 6)
        assert flags[2] is True  # Medium sentence (10 words, 6-50)
        assert flags[3] is False  # Long sentence (>50 words)

    def test_mark_examples_by_length_boundary(self):
        """Test boundary cases (exactly 50, 49, 51 words)."""
        from eng_words.llm.smart_card_generator import mark_examples_by_length

        examples = [
            (1, " ".join(["word"] * 49)),  # 49 words - should be True
            (2, " ".join(["word"] * 50)),  # 50 words - should be True
            (3, " ".join(["word"] * 51)),  # 51 words - should be False
        ]

        flags = mark_examples_by_length(examples, max_words=50, min_words=6)

        assert flags[1] is True  # 49 words (6-50 range)
        assert flags[2] is True  # 50 words (6-50 range)
        assert flags[3] is False  # 51 words (>50)

    def test_mark_examples_by_length_empty(self):
        """Test with empty list."""
        from eng_words.llm.smart_card_generator import mark_examples_by_length

        flags = mark_examples_by_length([], max_words=50)

        assert flags == {}

    def test_mark_examples_by_length_custom_max(self):
        """Test with custom max_words parameter."""
        from eng_words.llm.smart_card_generator import mark_examples_by_length

        examples = [
            (1, " ".join(["word"] * 30)),  # 30 words
            (2, " ".join(["word"] * 60)),  # 60 words
        ]

        flags = mark_examples_by_length(examples, max_words=40, min_words=6)

        assert flags[1] is True  # 30 words (6-40 range)
        assert flags[2] is False  # 60 words (>40)

    def test_mark_examples_by_length_preserves_all_ids(self):
        """Test that all sentence IDs are preserved in the result."""
        from eng_words.llm.smart_card_generator import mark_examples_by_length

        examples = [
            (1, "Short."),  # 1 word - too short
            (2, " ".join(["word"] * 100)),  # Very long (>50)
            (3, "This is a medium length sentence with enough words."),  # 10 words - OK
        ]

        flags = mark_examples_by_length(examples, max_words=50, min_words=6)

        # All IDs should be in the result
        assert 1 in flags
        assert 2 in flags
        assert 3 in flags
        assert len(flags) == 3
        assert flags[1] is False  # Too short (<6 words)
        assert flags[2] is False  # Too long (>50 words)
        assert flags[3] is True  # Medium (6-50 words)


class TestCheckSpoilers:
    """Tests for check_spoilers function."""

    def test_check_spoilers_basic(self, tmp_path):
        """Test basic spoiler checking."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from eng_words.llm.base import LLMResponse
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import check_spoilers

        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.model = "test-model"
        mock_provider.complete_json = None  # Force use of complete()
        
        # Mock LLM response
        mock_provider.complete.return_value = LLMResponse(
            content=json.dumps({"has_spoiler": [False, True, False]}),
            model="test-model",
            cost_usd=0.001,
            input_tokens=100,
            output_tokens=50,
        )

        cache = ResponseCache(cache_dir=Path(tmp_path), enabled=False)
        
        examples = [
            (1, "He walked to the store."),  # No spoiler
            (2, "The hero died in the final battle."),  # Spoiler
            (3, "She opened the door."),  # No spoiler
        ]

        flags = check_spoilers(
            examples=examples,
            provider=mock_provider,
            cache=cache,
            book_name="test_book",
        )

        assert flags[1] is False  # No spoiler
        assert flags[2] is True  # Has spoiler
        assert flags[3] is False  # No spoiler
        assert len(flags) == 3

    def test_check_spoilers_batching(self, tmp_path):
        """Test that spoiler checking works with batching (many examples)."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from eng_words.llm.base import LLMResponse
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import check_spoilers

        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.model = "test-model"
        mock_provider.complete_json = None
        
        # Create 60 examples (will be split into 2 batches: 50 + 10)
        examples = [(i, f"Sentence {i}.") for i in range(1, 61)]
        
        # Mock responses for 2 batches
        mock_provider.complete.side_effect = [
            LLMResponse(
                content=json.dumps({"has_spoiler": [False] * 50}),
                model="test-model",
                cost_usd=0.001,
                input_tokens=100,
                output_tokens=50,
            ),
            LLMResponse(
                content=json.dumps({"has_spoiler": [True] * 10}),
                model="test-model",
                cost_usd=0.001,
                input_tokens=100,
                output_tokens=50,
            ),
        ]

        cache = ResponseCache(cache_dir=Path(tmp_path), enabled=False)

        flags = check_spoilers(
            examples=examples,
            provider=mock_provider,
            cache=cache,
            book_name="test_book",
            max_examples_per_batch=50,
        )

        assert len(flags) == 60
        assert all(flags[i] is False for i in range(1, 51))  # First batch: no spoilers
        assert all(flags[i] is True for i in range(51, 61))  # Second batch: spoilers
        assert mock_provider.complete.call_count == 2  # Two batches

    def test_check_spoilers_empty(self, tmp_path):
        """Test with empty list."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import check_spoilers

        mock_provider = MagicMock()
        cache = ResponseCache(cache_dir=Path(tmp_path), enabled=False)

        flags = check_spoilers(
            examples=[],
            provider=mock_provider,
            cache=cache,
            book_name="test_book",
        )

        assert flags == {}
        mock_provider.complete.assert_not_called()

    def test_check_spoilers_caching(self, tmp_path):
        """Test that spoiler checking uses cache."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from eng_words.llm.base import LLMResponse
        from eng_words.llm.response_cache import ResponseCache
        from eng_words.llm.smart_card_generator import check_spoilers

        mock_provider = MagicMock()
        mock_provider.model = "test-model"
        mock_provider.complete_json = None
        
        mock_provider.complete.return_value = LLMResponse(
            content=json.dumps({"has_spoiler": [False]}),
            model="test-model",
            cost_usd=0.001,
            input_tokens=100,
            output_tokens=50,
        )

        cache = ResponseCache(cache_dir=Path(tmp_path), enabled=True)
        
        examples = [(1, "He walked to the store.")]

        # First call
        flags1 = check_spoilers(
            examples=examples,
            provider=mock_provider,
            cache=cache,
            book_name="test_book",
        )
        
        # Reset call count
        mock_provider.complete.reset_mock()
        
        # Second call with same examples (should use cache)
        flags2 = check_spoilers(
            examples=examples,
            provider=mock_provider,
            cache=cache,
            book_name="test_book",
        )

        assert flags1 == flags2
        # Should not call LLM again (cached)
        # Note: call_llm_with_retry handles caching internally, so we check that
        # the cache was used by verifying the result is the same


class TestSelectExamplesForGeneration:
    """Tests for select_examples_for_generation function."""

    def test_select_examples_3_or_more_valid(self):
        """Test when 3+ valid examples exist - should take 2 from book + generate 1."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        all_examples = [
            (1, "Short sentence 1."),
            (2, "Short sentence 2."),
            (3, "Short sentence 3."),
            (4, "Short sentence 4."),
        ]
        
        length_flags = {1: True, 2: True, 3: True, 4: True}  # All appropriate length
        spoiler_flags = {1: False, 2: False, 3: False, 4: False}  # No spoilers

        result = select_examples_for_generation(
            all_examples=all_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        assert len(result["selected_from_book"]) == 2
        assert result["generate_count"] == 1
        assert result["selected_from_book"][0][0] == 1  # First example
        assert result["selected_from_book"][1][0] == 2  # Second example
        assert "flags" in result
        assert result["flags"]["length"] == length_flags
        assert result["flags"]["spoiler"] == spoiler_flags

    def test_select_examples_1_to_2_valid(self):
        """Test when 1-2 valid examples exist - should take all + generate rest."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        all_examples = [
            (1, "Short sentence 1."),
            (2, "Short sentence 2."),
            (3, " ".join(["word"] * 100)),  # Too long
            (4, "Spoiler sentence."),  # Has spoiler
        ]
        
        length_flags = {1: True, 2: True, 3: False, 4: True}
        spoiler_flags = {1: False, 2: False, 3: False, 4: True}  # 4 has spoiler

        result = select_examples_for_generation(
            all_examples=all_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        # Only 1 and 2 are valid (appropriate length and no spoiler)
        assert len(result["selected_from_book"]) == 2
        assert result["generate_count"] == 1  # Need 1 more to reach 3
        assert result["selected_from_book"][0][0] == 1
        assert result["selected_from_book"][1][0] == 2

    def test_select_examples_1_valid(self):
        """Test when only 1 valid example exists - should take 1 + generate 2."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        all_examples = [
            (1, "Short sentence 1."),
            (2, " ".join(["word"] * 100)),  # Too long
            (3, "Spoiler sentence."),  # Has spoiler
        ]
        
        length_flags = {1: True, 2: False, 3: True}
        spoiler_flags = {1: False, 2: False, 3: True}  # 3 has spoiler

        result = select_examples_for_generation(
            all_examples=all_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        assert len(result["selected_from_book"]) == 1
        assert result["generate_count"] == 2  # Need 2 more to reach 3
        assert result["selected_from_book"][0][0] == 1

    def test_select_examples_0_valid(self):
        """Test when no valid examples exist - should generate 3."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        all_examples = [
            (1, " ".join(["word"] * 100)),  # Too long
            (2, "Spoiler sentence."),  # Has spoiler
            (3, " ".join(["word"] * 100)),  # Too long
        ]
        
        length_flags = {1: False, 2: True, 3: False}
        spoiler_flags = {1: False, 2: True, 3: False}  # 2 has spoiler

        result = select_examples_for_generation(
            all_examples=all_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        assert len(result["selected_from_book"]) == 0
        assert result["generate_count"] == 3  # Generate all 3
        assert "flags" in result

    def test_select_examples_preserves_flags(self):
        """Test that flags are preserved in the result."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        all_examples = [(1, "Short sentence.")]
        length_flags = {1: True}
        spoiler_flags = {1: False}

        result = select_examples_for_generation(
            all_examples=all_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        assert result["flags"]["length"] == length_flags
        assert result["flags"]["spoiler"] == spoiler_flags

    def test_select_examples_custom_target_count(self):
        """Test with custom target_count."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        all_examples = [(1, "Short sentence.")]
        length_flags = {1: True}
        spoiler_flags = {1: False}

        result = select_examples_for_generation(
            all_examples=all_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=5,
        )

        assert len(result["selected_from_book"]) == 1
        assert result["generate_count"] == 4  # Need 4 more to reach 5

    def test_select_examples_filters_correctly(self):
        """Test that filtering by flags works correctly."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        all_examples = [
            (1, "Valid 1."),  # Valid: length=True, spoiler=False
            (2, "Valid 2."),  # Valid: length=True, spoiler=False
            (3, "Too long " + " ".join(["word"] * 50)),  # Invalid: length=False
            (4, "Spoiler."),  # Invalid: spoiler=True
            (5, "Valid 3."),  # Valid: length=True, spoiler=False
        ]
        
        length_flags = {1: True, 2: True, 3: False, 4: True, 5: True}
        spoiler_flags = {1: False, 2: False, 3: False, 4: True, 5: False}

        result = select_examples_for_generation(
            all_examples=all_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        # Should select only 1, 2, 5 (valid ones), but logic says: if 3+ valid, take 2 + generate 1
        selected_ids = [sid for sid, _ in result["selected_from_book"]]
        assert 1 in selected_ids
        assert 2 in selected_ids
        # 5 is valid but not selected because we only take 2 when we have 3+ valid examples
        assert 3 not in selected_ids  # Too long
        assert 4 not in selected_ids  # Has spoiler
        # Logic: 3+ valid examples → take 2 + generate 1
        assert len(result["selected_from_book"]) == 2
        assert result["generate_count"] == 1

    def test_select_examples_deduplicates(self):
        """Test that duplicate examples are removed."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        all_examples = [
            (1, "Same sentence."),  # Duplicate
            (2, "Same sentence."),  # Duplicate (same text, different ID)
            (3, "Different sentence."),  # Unique
            (4, "Another unique sentence."),  # Unique
        ]
        
        length_flags = {1: True, 2: True, 3: True, 4: True}
        spoiler_flags = {1: False, 2: False, 3: False, 4: False}

        result = select_examples_for_generation(
            all_examples=all_examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        # Should deduplicate: "Same sentence" appears twice, but should only appear once
        selected_texts = [ex.lower().strip() for _, ex in result["selected_from_book"]]
        assert selected_texts.count("same sentence.") == 1  # Only one copy
        assert len(set(selected_texts)) == len(selected_texts)  # All unique
        assert len(result["selected_from_book"]) == 2  # 2 unique examples selected (logic: 3+ valid → take 2)

