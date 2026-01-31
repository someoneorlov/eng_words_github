"""Tests for Word Family Clusterer."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from eng_words.experiment.word_family_clusterer import (
    CLUSTER_PROMPT_TEMPLATE,
    ClusterResult,
    WordFamilyClusterer,
    group_examples_by_lemma,
)
from eng_words.llm.base import LLMResponse


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock()
    provider.model = "test-model"
    provider.temperature = 0.0
    return provider


@pytest.fixture
def sample_response():
    """Sample valid LLM response."""
    return LLMResponse(
        content=json.dumps({
            "cards": [
                {
                    "meaning_id": 1,
                    "definition_en": "to have an opinion",
                    "definition_ru": "have an opinion",
                    "part_of_speech": "verb",
                    "selected_example_indices": [1, 3],
                    "generated_example": "I think this is correct."
                },
                {
                    "meaning_id": 2,
                    "definition_en": "to consider carefully",
                    "definition_ru": "consider",
                    "part_of_speech": "verb",
                    "selected_example_indices": [2, 4],
                    "generated_example": "Let me think about it."
                }
            ],
            "ignored_indices": [5],
            "ignore_reasons": {"5": "unclear context"}
        }),
        model="test-model",
        input_tokens=1000,
        output_tokens=200,
        cost_usd=0.001,
    )


class TestWordFamilyClusterer:
    """Tests for WordFamilyClusterer class."""
    
    def test_cluster_single_batch(self, mock_provider, sample_response):
        """Test clustering with single batch (<=100 examples)."""
        mock_provider.complete.return_value = sample_response
        
        clusterer = WordFamilyClusterer(provider=mock_provider, cache=None)
        
        examples = [
            "I think you are right.",
            "Let me think about this problem.",
            "She thinks he is handsome.",
            "Think carefully before deciding.",
            "Random unclear sentence.",
        ]
        
        result = clusterer.cluster_lemma(
            lemma="think",
            examples=examples,
            sentence_ids=[1, 2, 3, 4, 5],
        )
        
        assert isinstance(result, ClusterResult)
        assert result.lemma == "think"
        assert len(result.cards) == 2
        assert result.ignored_indices == [5]
        assert result.batches_processed == 1
        
        # Check card content
        card1 = result.cards[0]
        assert card1['lemma'] == "think"
        assert card1['definition_en'] == "to have an opinion"
        assert 'examples' in card1
    
    def test_cluster_multi_pass(self, mock_provider, sample_response):
        """Test clustering with multi-pass (>100 examples)."""
        mock_provider.complete.return_value = sample_response
        
        # Disable merge for this test
        with patch(
            'eng_words.experiment.word_family_clusterer.WordFamilyClusterer._merge_similar_cards',
            lambda self, cards: cards
        ):
            clusterer = WordFamilyClusterer(provider=mock_provider, cache=None)
            
            # Create 150 examples (should use 2 batches)
            examples = [f"Example sentence {i}." for i in range(150)]
            sentence_ids = list(range(150))
            
            result = clusterer.cluster_lemma(
                lemma="test",
                examples=examples,
                sentence_ids=sentence_ids,
            )
            
            assert result.batches_processed == 2
            # Each batch produces 2 cards, so 4 total (before merge)
            assert len(result.cards) == 4
            assert mock_provider.complete.call_count == 2
    
    def test_parse_json_with_markdown(self, mock_provider):
        """Test parsing JSON wrapped in markdown code blocks."""
        response = LLMResponse(
            content='```json\n{"cards": [], "ignored_indices": [], "ignore_reasons": {}}\n```',
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0001,
        )
        mock_provider.complete.return_value = response
        
        clusterer = WordFamilyClusterer(provider=mock_provider, cache=None)
        
        result = clusterer.cluster_lemma(
            lemma="test",
            examples=["Example 1.", "Example 2."],
        )
        
        assert result.cards == []
        assert result.ignored_indices == []
    
    def test_invalid_json_handling(self, mock_provider):
        """Test handling of invalid JSON response."""
        response = LLMResponse(
            content="This is not valid JSON",
            model="test-model",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0001,
        )
        mock_provider.complete.return_value = response
        
        clusterer = WordFamilyClusterer(provider=mock_provider, cache=None)
        
        result = clusterer.cluster_lemma(
            lemma="test",
            examples=["Example 1."],
        )
        
        # Should return empty cards with error in reasons
        assert result.cards == []
        assert 'error' in result.ignore_reasons
    
    def test_stats_tracking(self, mock_provider, sample_response):
        """Test that statistics are tracked correctly."""
        mock_provider.complete.return_value = sample_response
        
        clusterer = WordFamilyClusterer(provider=mock_provider, cache=None)
        
        clusterer.cluster_lemma(lemma="test1", examples=["Ex 1."])
        clusterer.cluster_lemma(lemma="test2", examples=["Ex 2."])
        
        stats = clusterer.stats()
        
        assert stats['total_api_calls'] == 2
        assert stats['total_input_tokens'] == 2000
        assert stats['total_output_tokens'] == 400
        assert stats['total_cost_usd'] == 0.002


class TestGroupExamplesByLemma:
    """Tests for group_examples_by_lemma function."""
    
    def test_basic_grouping(self):
        """Test basic lemma grouping."""
        tokens = pd.DataFrame({
            'lemma': ['think', 'think', 'know', 'know', 'know'],
            'sentence_id': [1, 2, 1, 2, 3],
            'pos': ['VERB', 'VERB', 'VERB', 'VERB', 'VERB'],
            'is_alpha': [True, True, True, True, True],
            'is_stop': [False, False, False, False, False],
        })
        
        sentences = pd.DataFrame({
            'sentence_id': [1, 2, 3],
            'text': ['Sentence one.', 'Sentence two.', 'Sentence three.'],
        })
        
        result = group_examples_by_lemma(tokens, sentences)
        
        assert len(result) == 2
        
        # know has more examples
        know_row = result[result['lemma'] == 'know'].iloc[0]
        assert know_row['example_count'] == 3
        
        think_row = result[result['lemma'] == 'think'].iloc[0]
        assert think_row['example_count'] == 2
    
    def test_filters_non_content_words(self):
        """Test that non-content words are filtered."""
        tokens = pd.DataFrame({
            'lemma': ['think', 'the', 'is', 'think'],
            'sentence_id': [1, 1, 1, 2],
            'pos': ['VERB', 'DET', 'AUX', 'VERB'],
            'is_alpha': [True, True, True, True],
            'is_stop': [False, True, True, False],
        })
        
        sentences = pd.DataFrame({
            'sentence_id': [1, 2],
            'text': ['Sentence one.', 'Sentence two.'],
        })
        
        result = group_examples_by_lemma(tokens, sentences)
        
        # Only 'think' should remain
        assert len(result) == 1
        assert result.iloc[0]['lemma'] == 'think'
    
    def test_empty_input(self):
        """Test with empty input."""
        tokens = pd.DataFrame({
            'lemma': [],
            'sentence_id': [],
            'pos': [],
            'is_alpha': [],
            'is_stop': [],
        })
        
        sentences = pd.DataFrame({
            'sentence_id': [],
            'text': [],
        })
        
        result = group_examples_by_lemma(tokens, sentences)
        
        assert len(result) == 0


class TestPromptTemplate:
    """Tests for prompt templates."""
    
    def test_prompt_template_formatting(self):
        """Test that prompt template formats correctly."""
        prompt = CLUSTER_PROMPT_TEMPLATE.format(
            lemma="test",
            numbered_examples="1. Example one.\n2. Example two.",
        )
        
        assert "LEMMA: test" in prompt
        assert "1. Example one." in prompt
        assert "2. Example two." in prompt
        assert "DISTINCT MEANINGS" in prompt
        assert "strict JSON" in prompt
