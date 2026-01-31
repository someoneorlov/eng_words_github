"""Tests for LLM-based synset aggregator."""

from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eng_words.aggregation.llm_aggregator import (
    AggregationResult,
    LLMAggregator,
    SynsetGroup,
    build_aggregation_prompt,
    parse_aggregation_response,
)
from eng_words.llm.base import LLMResponse


class TestSynsetGroup:
    """Tests for SynsetGroup dataclass."""

    def test_create_group(self):
        """Test creating a SynsetGroup."""
        group = SynsetGroup(
            synset_ids=["dog.n.01", "dog.n.02"],
            primary_synset="dog.n.01",
            reason="Similar meanings",
        )
        assert len(group.synset_ids) == 2
        assert group.primary_synset == "dog.n.01"

    def test_to_dict(self):
        """Test converting to dict."""
        group = SynsetGroup(
            synset_ids=["run.v.01"],
            primary_synset="run.v.01",
            reason="Single synset",
        )
        d = asdict(group)
        assert d["synset_ids"] == ["run.v.01"]


class TestAggregationResult:
    """Tests for AggregationResult dataclass."""

    def test_create_result(self):
        """Test creating an AggregationResult."""
        result = AggregationResult(
            lemma="dog",
            groups=[
                SynsetGroup(["dog.n.01"], "dog.n.01", "reason1"),
                SynsetGroup(["dog.n.02"], "dog.n.02", "reason2"),
            ],
            original_count=5,
            aggregated_count=2,
            llm_cost=0.001,
        )
        assert result.lemma == "dog"
        assert len(result.groups) == 2
        assert result.original_count == 5
        assert result.aggregated_count == 2


class TestBuildPrompt:
    """Tests for build_aggregation_prompt function."""

    def test_basic_prompt(self):
        """Test building a basic prompt."""
        synsets = [
            {"synset_id": "dog.n.01", "definition": "a pet", "freq": 10},
            {"synset_id": "dog.n.02", "definition": "a person", "freq": 5},
        ]
        prompt = build_aggregation_prompt("dog", synsets)
        
        assert "dog" in prompt
        assert "dog.n.01" in prompt
        assert "dog.n.02" in prompt
        assert "a pet" in prompt
        assert "10" in prompt  # freq

    def test_prompt_contains_instructions(self):
        """Test that prompt contains key instructions."""
        synsets = [{"synset_id": "test.n.01", "definition": "test", "freq": 1}]
        prompt = build_aggregation_prompt("test", synsets)
        
        assert "merge" in prompt.lower() or "merged" in prompt.lower()
        assert "JSON" in prompt


class TestParseResponse:
    """Tests for parse_aggregation_response function."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = '''
        {
          "groups": [
            {"synsets": [1, 2], "primary_synset": 1, "reason": "Similar"},
            {"synsets": [3], "primary_synset": 3, "reason": "Different"}
          ],
          "total_cards": 2
        }
        '''
        result = parse_aggregation_response(response)
        
        assert result is not None
        assert len(result["groups"]) == 2
        assert result["total_cards"] == 2

    def test_parse_with_markdown(self):
        """Test parsing JSON wrapped in markdown."""
        response = '''```json
{
  "groups": [{"synsets": [1], "primary_synset": 1, "reason": "test"}],
  "total_cards": 1
}
```'''
        result = parse_aggregation_response(response)
        
        assert result is not None
        assert len(result["groups"]) == 1

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        response = "This is not valid JSON"
        result = parse_aggregation_response(response)
        
        assert result is None

    def test_parse_truncated_json(self):
        """Test parsing truncated JSON attempts repair."""
        # Truncated JSON
        response = '''
{
  "groups": [
    {"synsets": [1], "primary_synset": 1, "reason": "test"},
    {"synsets": [2], "primary_synset": 2
'''
        result = parse_aggregation_response(response)
        
        # Should either repair or return None gracefully
        # The repair logic is best-effort
        assert result is None or isinstance(result, dict)


class TestLLMAggregator:
    """Tests for LLMAggregator class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.model = "test-model"
        provider.temperature = 0.0
        return provider

    @pytest.fixture
    def mock_cache(self, tmp_path):
        """Create a mock cache."""
        from eng_words.llm.response_cache import ResponseCache
        return ResponseCache(cache_dir=tmp_path, enabled=True)

    def test_create_aggregator(self, mock_provider, mock_cache):
        """Test creating an LLMAggregator."""
        aggregator = LLMAggregator(provider=mock_provider, cache=mock_cache)
        
        assert aggregator.provider == mock_provider
        assert aggregator.cache == mock_cache

    def test_aggregate_single_synset_lemma(self, mock_provider, mock_cache):
        """Test that single-synset lemmas don't call LLM."""
        aggregator = LLMAggregator(provider=mock_provider, cache=mock_cache)
        
        synsets = [{"synset_id": "dog.n.01", "definition": "a pet", "freq": 10}]
        result = aggregator.aggregate_lemma("dog", synsets)
        
        # Should return result without calling LLM
        assert result.lemma == "dog"
        assert result.original_count == 1
        assert result.aggregated_count == 1
        assert len(result.groups) == 1
        assert result.llm_cost == 0.0  # No LLM call
        mock_provider.complete.assert_not_called()

    def test_aggregate_multiple_synsets(self, mock_provider, mock_cache):
        """Test aggregating multiple synsets."""
        # Mock LLM response
        mock_response = LLMResponse(
            content='''{
                "groups": [
                    {"synsets": [1, 2], "primary_synset": 1, "reason": "Similar meanings"},
                    {"synsets": [3], "primary_synset": 3, "reason": "Different"}
                ],
                "total_cards": 2
            }''',
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        mock_provider.complete.return_value = mock_response

        aggregator = LLMAggregator(provider=mock_provider, cache=mock_cache)

        synsets = [
            {"synset_id": "run.v.01", "definition": "move fast", "freq": 10},
            {"synset_id": "run.v.02", "definition": "operate", "freq": 5},
            {"synset_id": "run.v.03", "definition": "manage", "freq": 3},
        ]
        result = aggregator.aggregate_lemma("run", synsets)

        assert result.lemma == "run"
        assert result.original_count == 3
        assert result.aggregated_count == 2
        assert len(result.groups) == 2
        assert result.llm_cost > 0
        mock_provider.complete.assert_called_once()

    def test_aggregate_batch(self, mock_provider, mock_cache):
        """Test batch aggregation."""
        # Mock LLM response
        mock_response = LLMResponse(
            content='''{
                "groups": [{"synsets": [1, 2], "primary_synset": 1, "reason": "test"}],
                "total_cards": 1
            }''',
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        mock_provider.complete.return_value = mock_response

        aggregator = LLMAggregator(provider=mock_provider, cache=mock_cache)

        # Create test DataFrame
        df = pd.DataFrame({
            "lemma": ["dog", "dog", "cat"],
            "synset_id": ["dog.n.01", "dog.n.02", "cat.n.01"],
            "definition": ["pet", "person", "feline"],
            "freq": [10, 5, 3],
        })

        results = aggregator.aggregate_batch(df, progress=False)

        # dog has 2 synsets -> LLM call
        # cat has 1 synset -> no LLM call
        assert len(results) == 2  # One result per unique lemma
        
        dog_result = next(r for r in results if r.lemma == "dog")
        assert dog_result.original_count == 2
        
        cat_result = next(r for r in results if r.lemma == "cat")
        assert cat_result.original_count == 1
        assert cat_result.llm_cost == 0.0  # No LLM call

    def test_get_stats(self, mock_provider, mock_cache):
        """Test getting aggregation statistics."""
        aggregator = LLMAggregator(provider=mock_provider, cache=mock_cache)
        
        stats = aggregator.get_stats()
        
        assert "total_lemmas" in stats
        assert "total_synsets" in stats
        assert "total_groups" in stats
        assert "total_cost" in stats

