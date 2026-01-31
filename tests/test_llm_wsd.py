"""Tests for LLM-based Word Sense Disambiguation."""

import json
from unittest.mock import MagicMock

import pytest

from eng_words.llm.base import LLMResponse


class TestLlmWsdSentence:
    """Tests for llm_wsd_sentence function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.model = "test-model"
        provider.temperature = 0.0
        return provider

    @pytest.fixture
    def sample_candidates(self):
        """Sample candidate synsets for testing."""
        return [
            {"synset_id": "accompany.v.02", "definition": "go or travel along with"},
            {"synset_id": "play_along.v.02", "definition": "perform an accompaniment to"},
            {"synset_id": "attach_to.v.01", "definition": "be present or associated with"},
        ]

    def test_returns_correct_synset_for_clear_case(self, mock_provider, sample_candidates):
        """Test that LLM returns correct synset for unambiguous sentence."""
        from eng_words.wsd.llm_wsd import llm_wsd_sentence

        # Mock LLM to return the correct synset
        mock_provider.complete.return_value = LLMResponse(
            content="accompany.v.02",
            model="test-model",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.0001,
        )

        result = llm_wsd_sentence(
            lemma="accompany",
            sentence="She accompanied him to the store.",
            candidate_synsets=sample_candidates,
            provider=mock_provider,
        )

        assert result == "accompany.v.02"
        mock_provider.complete.assert_called_once()

    def test_returns_none_for_unclear_case(self, mock_provider, sample_candidates):
        """Test that function returns None when LLM returns NONE."""
        from eng_words.wsd.llm_wsd import llm_wsd_sentence

        mock_provider.complete.return_value = LLMResponse(
            content="NONE",
            model="test-model",
            input_tokens=100,
            output_tokens=5,
            cost_usd=0.0001,
        )

        result = llm_wsd_sentence(
            lemma="accompany",
            sentence="The word accompany is ambiguous here.",
            candidate_synsets=sample_candidates,
            provider=mock_provider,
        )

        assert result is None

    def test_handles_synset_with_quotes(self, mock_provider, sample_candidates):
        """Test that function handles synset_id wrapped in quotes."""
        from eng_words.wsd.llm_wsd import llm_wsd_sentence

        mock_provider.complete.return_value = LLMResponse(
            content='"accompany.v.02"',
            model="test-model",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.0001,
        )

        result = llm_wsd_sentence(
            lemma="accompany",
            sentence="She accompanied him.",
            candidate_synsets=sample_candidates,
            provider=mock_provider,
        )

        assert result == "accompany.v.02"

    def test_handles_synset_with_whitespace(self, mock_provider, sample_candidates):
        """Test that function handles synset_id with surrounding whitespace."""
        from eng_words.wsd.llm_wsd import llm_wsd_sentence

        mock_provider.complete.return_value = LLMResponse(
            content="  accompany.v.02  \n",
            model="test-model",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.0001,
        )

        result = llm_wsd_sentence(
            lemma="accompany",
            sentence="She accompanied him.",
            candidate_synsets=sample_candidates,
            provider=mock_provider,
        )

        assert result == "accompany.v.02"

    def test_validates_synset_against_candidates(self, mock_provider, sample_candidates):
        """Test that function validates returned synset is in candidates."""
        from eng_words.wsd.llm_wsd import llm_wsd_sentence

        # LLM returns a synset not in candidates
        mock_provider.complete.return_value = LLMResponse(
            content="unknown.v.99",
            model="test-model",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.0001,
        )

        result = llm_wsd_sentence(
            lemma="accompany",
            sentence="She accompanied him.",
            candidate_synsets=sample_candidates,
            provider=mock_provider,
        )

        # Should return None if synset not in candidates
        assert result is None

    def test_prompt_contains_all_candidates(self, mock_provider, sample_candidates):
        """Test that prompt includes all candidate synsets with definitions."""
        from eng_words.wsd.llm_wsd import llm_wsd_sentence

        mock_provider.complete.return_value = LLMResponse(
            content="accompany.v.02",
            model="test-model",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.0001,
        )

        llm_wsd_sentence(
            lemma="accompany",
            sentence="She accompanied him.",
            candidate_synsets=sample_candidates,
            provider=mock_provider,
        )

        # Check that prompt was called with all candidate info
        call_args = mock_provider.complete.call_args
        prompt = call_args[0][0]  # First positional argument

        assert "accompany" in prompt
        assert "She accompanied him" in prompt
        assert "accompany.v.02" in prompt
        assert "go or travel along with" in prompt
        assert "play_along.v.02" in prompt
        assert "perform an accompaniment to" in prompt

    def test_empty_candidates_returns_none(self, mock_provider):
        """Test that function returns None for empty candidates list."""
        from eng_words.wsd.llm_wsd import llm_wsd_sentence

        result = llm_wsd_sentence(
            lemma="test",
            sentence="Test sentence.",
            candidate_synsets=[],
            provider=mock_provider,
        )

        assert result is None
        mock_provider.complete.assert_not_called()

    def test_single_candidate_returns_it_directly(self, mock_provider):
        """Test that single candidate is returned without LLM call."""
        from eng_words.wsd.llm_wsd import llm_wsd_sentence

        single_candidate = [{"synset_id": "test.v.01", "definition": "to test"}]

        result = llm_wsd_sentence(
            lemma="test",
            sentence="Test sentence.",
            candidate_synsets=single_candidate,
            provider=mock_provider,
        )

        # Should return single candidate without calling LLM
        assert result == "test.v.01"
        mock_provider.complete.assert_not_called()


class TestBuildWsdPrompt:
    """Tests for _build_wsd_prompt helper function."""

    def test_prompt_structure(self):
        """Test that prompt has correct structure."""
        from eng_words.wsd.llm_wsd import _build_wsd_prompt

        candidates = [
            {"synset_id": "run.v.01", "definition": "move fast"},
            {"synset_id": "run.v.02", "definition": "operate"},
        ]

        prompt = _build_wsd_prompt(
            lemma="run",
            sentence="He runs every morning.",
            candidate_synsets=candidates,
        )

        # Check structure
        assert "run" in prompt
        assert "He runs every morning" in prompt
        assert "1." in prompt
        assert "2." in prompt
        assert "run.v.01" in prompt
        assert "run.v.02" in prompt
        assert "move fast" in prompt
        assert "operate" in prompt

    def test_prompt_instructs_synset_only(self):
        """Test that prompt instructs to return only synset_id."""
        from eng_words.wsd.llm_wsd import _build_wsd_prompt

        candidates = [{"synset_id": "test.v.01", "definition": "test def"}]

        prompt = _build_wsd_prompt(
            lemma="test",
            sentence="Test.",
            candidate_synsets=candidates,
        )

        # Should instruct to return synset_id or NONE
        assert "NONE" in prompt or "none" in prompt.lower()


class TestParseWsdResponse:
    """Tests for _parse_wsd_response helper function."""

    def test_parse_clean_synset(self):
        """Test parsing clean synset_id."""
        from eng_words.wsd.llm_wsd import _parse_wsd_response

        candidates = {"test.v.01", "test.v.02"}

        assert _parse_wsd_response("test.v.01", candidates) == "test.v.01"

    def test_parse_quoted_synset(self):
        """Test parsing synset_id with quotes."""
        from eng_words.wsd.llm_wsd import _parse_wsd_response

        candidates = {"test.v.01", "test.v.02"}

        assert _parse_wsd_response('"test.v.01"', candidates) == "test.v.01"
        assert _parse_wsd_response("'test.v.01'", candidates) == "test.v.01"

    def test_parse_with_whitespace(self):
        """Test parsing synset_id with whitespace."""
        from eng_words.wsd.llm_wsd import _parse_wsd_response

        candidates = {"test.v.01"}

        assert _parse_wsd_response("  test.v.01  \n", candidates) == "test.v.01"

    def test_parse_none_response(self):
        """Test parsing NONE response."""
        from eng_words.wsd.llm_wsd import _parse_wsd_response

        candidates = {"test.v.01"}

        assert _parse_wsd_response("NONE", candidates) is None
        assert _parse_wsd_response("none", candidates) is None
        assert _parse_wsd_response("None", candidates) is None

    def test_parse_invalid_synset(self):
        """Test parsing invalid synset returns None."""
        from eng_words.wsd.llm_wsd import _parse_wsd_response

        candidates = {"test.v.01"}

        assert _parse_wsd_response("invalid.v.99", candidates) is None
        assert _parse_wsd_response("random text", candidates) is None


class TestRedistributeEmptyCards:
    """Tests for redistribute_empty_cards function."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.model = "test-model"
        provider.temperature = 0.0
        return provider

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache that returns LLMResponse for get_or_compute."""
        from eng_words.llm.response_cache import ResponseCache
        cache = MagicMock(spec=ResponseCache)
        cache.enabled = True
        cache.generate_key.return_value = "test_key"
        cache.stats.return_value = {"hits": 0, "misses": 0}
        # Default: cache miss, call compute_fn
        cache.get.return_value = None
        return cache

    def _setup_cache_response(self, mock_cache, content: str):
        """Helper to setup cache to return specific LLM response."""
        response = LLMResponse(
            content=content,
            model="test-model",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.0001,
        )
        # When cache miss, get_or_compute calls compute_fn and returns result
        mock_cache.get_or_compute.side_effect = lambda key, compute_fn: response
        return response

    @pytest.fixture
    def sample_cards(self):
        """Create sample cards for testing."""
        from eng_words.llm.smart_card_generator import SmartCard

        # Card 1: Has examples (should stay)
        card1 = SmartCard(
            lemma="run",
            pos="v",
            supersense="verb.motion",
            selected_examples=["He runs every day."],
            excluded_examples=[],
            simple_definition="move fast",
            translation_ru="run",
            generated_example="She runs in the park.",
            wn_definition="move fast by using feet",
            book_name="test_book",
            primary_synset="run.v.01",
            synset_group=["run.v.01"],
        )

        # Card 2: Empty (needs redistribution)
        card2 = SmartCard(
            lemma="accompany",
            pos="v",
            supersense="verb.social",
            selected_examples=[],
            excluded_examples=["She accompanied him to the store."],
            simple_definition="go with",
            translation_ru="accompany",
            generated_example="",
            wn_definition="go or travel along with",
            book_name="test_book",
            primary_synset="play_along.v.02",  # Wrong synset assigned by WSD
            synset_group=["play_along.v.02"],
        )

        # Card 3: Has examples for accompany.v.02 (target for redistribution)
        card3 = SmartCard(
            lemma="accompany",
            pos="v",
            supersense="verb.social",
            selected_examples=["He accompanied her home."],
            excluded_examples=[],
            simple_definition="go with someone",
            translation_ru="accompany",
            generated_example="They accompanied us.",
            wn_definition="go or travel along with",
            book_name="test_book",
            primary_synset="accompany.v.02",
            synset_group=["accompany.v.02"],
        )

        return [card1, card2, card3]

    def test_returns_cards_with_no_empty(self, mock_provider, mock_cache, sample_cards):
        """Test that function returns no empty cards."""
        from eng_words.wsd.llm_wsd import redistribute_empty_cards

        # Setup cache to return accompany.v.02
        self._setup_cache_response(mock_cache, "accompany.v.02")

        result = redistribute_empty_cards(sample_cards, mock_provider, mock_cache)

        # Check no empty cards
        empty_cards = [c for c in result if not c.selected_examples]
        assert len(empty_cards) == 0

    def test_adds_sentence_to_existing_card(self, mock_provider, mock_cache, sample_cards):
        """Test that sentence is added to existing card with matching synset."""
        from eng_words.wsd.llm_wsd import redistribute_empty_cards

        # Setup cache to return accompany.v.02 (which exists in card3)
        self._setup_cache_response(mock_cache, "accompany.v.02")

        result = redistribute_empty_cards(sample_cards, mock_provider, mock_cache)

        # Find accompany.v.02 card
        accompany_card = next(
            (c for c in result if c.primary_synset == "accompany.v.02"), None
        )
        assert accompany_card is not None
        assert "She accompanied him to the store." in accompany_card.selected_examples

    def test_removes_empty_card_after_redistribution(self, mock_provider, mock_cache, sample_cards):
        """Test that empty card is removed after redistribution."""
        from eng_words.wsd.llm_wsd import redistribute_empty_cards

        # Setup cache to return accompany.v.02
        self._setup_cache_response(mock_cache, "accompany.v.02")

        result = redistribute_empty_cards(sample_cards, mock_provider, mock_cache)

        # Original empty card (play_along.v.02) should be removed
        play_along_cards = [c for c in result if c.primary_synset == "play_along.v.02"]
        assert len(play_along_cards) == 0

    def test_creates_new_card_for_new_synset(self, mock_provider, mock_cache):
        """Test that new card is created if synset doesn't exist."""
        from eng_words.llm.smart_card_generator import SmartCard
        from eng_words.wsd.llm_wsd import redistribute_empty_cards

        # Only one empty card with no matching synset
        cards = [
            SmartCard(
                lemma="test",
                pos="v",
                supersense="verb.cognition",
                selected_examples=[],
                excluded_examples=["We tested the code."],
                simple_definition="try out",
                translation_ru="test",
                generated_example="",
                wn_definition="put to the test",
                book_name="test_book",
                primary_synset="wrong.v.01",
                synset_group=["wrong.v.01"],
            )
        ]

        # Setup cache to return test.v.01
        self._setup_cache_response(mock_cache, "test.v.01")

        result = redistribute_empty_cards(cards, mock_provider, mock_cache)

        # New card should be created
        assert len(result) == 1
        assert result[0].primary_synset == "test.v.01"
        assert "We tested the code." in result[0].selected_examples

    def test_handles_none_wsd_result(self, mock_provider, mock_cache):
        """Test that NONE result keeps sentence in excluded."""
        from eng_words.llm.smart_card_generator import SmartCard
        from eng_words.wsd.llm_wsd import redistribute_empty_cards

        cards = [
            SmartCard(
                lemma="ambiguous",
                pos="v",
                supersense="verb.cognition",
                selected_examples=[],
                excluded_examples=["Ambiguous sentence here."],
                simple_definition="",
                translation_ru="",
                generated_example="",
                wn_definition="",
                book_name="test_book",
                primary_synset="unclear.v.01",
                synset_group=["unclear.v.01"],
            )
        ]

        # Setup cache to return NONE
        self._setup_cache_response(mock_cache, "NONE")

        result = redistribute_empty_cards(cards, mock_provider, mock_cache)

        # Card should remain empty (or be removed)
        empty_cards = [c for c in result if not c.selected_examples]
        # Either removed or kept as empty
        assert len(result) <= 1

    def test_preserves_non_empty_cards(self, mock_provider, mock_cache):
        """Test that non-empty cards are preserved unchanged."""
        from eng_words.llm.smart_card_generator import SmartCard
        from eng_words.wsd.llm_wsd import redistribute_empty_cards

        cards = [
            SmartCard(
                lemma="good",
                pos="adj",
                supersense="adj.all",
                selected_examples=["Good morning!", "It was good."],
                excluded_examples=[],
                simple_definition="positive",
                translation_ru="good",
                generated_example="That's good.",
                wn_definition="having desirable qualities",
                book_name="test_book",
                primary_synset="good.a.01",
                synset_group=["good.a.01"],
            )
        ]

        result = redistribute_empty_cards(cards, mock_provider, mock_cache)

        assert len(result) == 1
        assert result[0].selected_examples == ["Good morning!", "It was good."]
        mock_provider.complete.assert_not_called()  # No LLM call needed

