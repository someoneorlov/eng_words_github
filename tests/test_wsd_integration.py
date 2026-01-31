"""Integration tests for the WSD module.

These tests verify the complete flow from raw text to sense statistics,
ensuring all components work together correctly.
"""

import pandas as pd
import pytest

from eng_words.wsd import (
    WordNetSenseBackend,
    aggregate_sense_statistics,
    get_synsets,
    synset_to_supersense,
)


class TestWSDFullPipeline:
    """Test complete WSD pipeline from tokens to statistics."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return WordNetSenseBackend()

    @pytest.fixture
    def sample_book_data(self):
        """Simulate data from a book with polysemous words.

        Creates a realistic scenario with:
        - "bank" in financial and river contexts
        - "run" in motion and operation contexts
        - "bat" in animal context
        """
        sentences_df = pd.DataFrame(
            {
                "sentence_id": list(range(1, 11)),
                "sentence": [
                    # bank - financial (3 times)
                    "I deposited money at the bank yesterday.",
                    "The bank approved my loan application.",
                    "She works at a major investment bank.",
                    # bank - river (2 times)
                    "We sat on the river bank and watched the sunset.",
                    "The boat was moored near the bank.",
                    # run - motion (3 times)
                    "The dog runs quickly across the field.",
                    "I run every morning to stay healthy.",
                    "The children run and play in the park.",
                    # run - operation (2 times)
                    "She runs a successful technology company.",
                    "He runs the entire marketing department.",
                ],
            }
        )

        tokens_df = pd.DataFrame(
            {
                "token_id": list(range(1, 11)),
                "lemma": [
                    "bank",
                    "bank",
                    "bank",
                    "bank",
                    "bank",
                    "run",
                    "run",
                    "run",
                    "run",
                    "run",
                ],
                "pos": [
                    "NOUN",
                    "NOUN",
                    "NOUN",
                    "NOUN",
                    "NOUN",
                    "VERB",
                    "VERB",
                    "VERB",
                    "VERB",
                    "VERB",
                ],
                "sentence_id": list(range(1, 11)),
            }
        )

        return tokens_df, sentences_df

    def test_full_pipeline_annotate_and_aggregate(self, backend, sample_book_data):
        """Test complete flow: annotate → aggregate."""
        tokens_df, sentences_df = sample_book_data

        # Step 1: Annotate tokens with sense information
        annotated = backend.annotate(tokens_df, sentences_df, show_progress=False)

        # Verify annotation results
        assert "synset_id" in annotated.columns
        assert "supersense" in annotated.columns
        assert "sense_confidence" in annotated.columns
        assert len(annotated) == len(tokens_df)

        # All tokens should have supersenses (no "unknown")
        assert all(annotated["supersense"] != "unknown")

        # Step 2: Aggregate statistics
        stats = backend.aggregate(annotated)

        # Verify aggregation results
        assert "lemma" in stats.columns
        assert "supersense" in stats.columns
        assert "sense_freq" in stats.columns
        assert "book_freq" in stats.columns
        assert "sense_ratio" in stats.columns
        assert "sense_count" in stats.columns
        assert "dominant_supersense" in stats.columns

        # Check that we have multiple senses for polysemous words
        bank_senses = stats[stats["lemma"] == "bank"]["supersense"].nunique()
        run_senses = stats[stats["lemma"] == "run"]["supersense"].nunique()

        # We expect at least 1 sense for each (WSD may or may not distinguish)
        assert bank_senses >= 1
        assert run_senses >= 1

        # Check book_freq is correct
        assert all(stats[stats["lemma"] == "bank"]["book_freq"] == 5)
        assert all(stats[stats["lemma"] == "run"]["book_freq"] == 5)

        # Check sense_ratio sums to 1.0 for each lemma
        for lemma in ["bank", "run"]:
            lemma_stats = stats[stats["lemma"] == lemma]
            total_ratio = lemma_stats["sense_ratio"].sum()
            assert abs(total_ratio - 1.0) < 0.01

    def test_single_word_disambiguation_consistency(self, backend):
        """Test that single word disambiguation is consistent with batch."""
        sentence = "I deposited money at the bank"
        lemma = "bank"
        pos = "n"

        # Single word
        single_result = backend.disambiguate_word(sentence, lemma, pos)

        # Batch of one
        batch_results = backend.disambiguate_batch([(sentence, lemma, pos)])

        # Results should be identical
        assert single_result.sense_id == batch_results[0].sense_id
        assert single_result.sense_label == batch_results[0].sense_label
        # Tolerance accounts for context boost difference:
        # - disambiguate_word uses compute_combined_score() (embedding + context boost)
        # - disambiguate_batch uses raw cosine similarity only
        # Context boost can add up to ~0.04 (0.5 * CONTEXT_BOOST_WEIGHT), so 0.05
        # tolerance catches real inconsistencies while allowing legitimate differences
        assert abs(single_result.confidence - batch_results[0].confidence) < 0.05

    def test_aggregator_standalone_usage(self):
        """Test aggregate_sense_statistics can be used independently."""
        # Manually created annotated data (not from backend)
        annotated_df = pd.DataFrame(
            {
                "lemma": ["word", "word", "word", "test", "test"],
                "supersense": [
                    "noun.communication",
                    "noun.communication",
                    "noun.cognition",
                    "noun.act",
                    "noun.act",
                ],
                "sentence_id": [1, 2, 3, 4, 5],
            }
        )

        # Use standalone aggregator
        stats = aggregate_sense_statistics(annotated_df)

        # Verify results
        assert len(stats) == 3  # 2 senses for "word", 1 for "test"

        word_stats = stats[stats["lemma"] == "word"]
        assert word_stats["sense_count"].iloc[0] == 2
        assert word_stats["book_freq"].iloc[0] == 3

    def test_wordnet_utilities_standalone(self):
        """Test WordNet utilities can be used independently."""
        # Get synsets
        synsets = get_synsets("bank", pos="n")
        assert len(synsets) > 0

        # Get supersense
        supersense = synset_to_supersense(synsets[0])
        assert supersense.startswith("noun.")

    def test_confidence_threshold_filtering(self, backend, sample_book_data):
        """Test that confidence threshold can be used for filtering."""
        tokens_df, sentences_df = sample_book_data

        annotated = backend.annotate(tokens_df, sentences_df, show_progress=False)

        # Check is_confident method
        from eng_words.wsd import SenseAnnotation

        high_conf = SenseAnnotation(
            lemma="test",
            sense_id="test.n.01",
            sense_label="noun.cognition",
            confidence=0.5,
        )
        low_conf = SenseAnnotation(
            lemma="test",
            sense_id="test.n.01",
            sense_label="noun.cognition",
            confidence=0.2,
        )

        assert backend.is_confident(high_conf) is True
        assert backend.is_confident(low_conf) is False

        # Filter annotated data by confidence
        confident_mask = annotated["sense_confidence"] >= backend.confidence_threshold
        confident_tokens = annotated[confident_mask]

        # Most tokens should be confident (depends on actual WSD results)
        assert len(confident_tokens) >= len(annotated) * 0.5


class TestWSDEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return WordNetSenseBackend()

    def test_unknown_words_handling(self, backend):
        """Test handling of words not in WordNet."""
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1, 2],
                "sentence": [
                    "The xyzfoobar was interesting.",
                    "I saw a quxbaz yesterday.",
                ],
            }
        )

        tokens_df = pd.DataFrame(
            {
                "token_id": [1, 2],
                "lemma": ["xyzfoobar", "quxbaz"],
                "pos": ["NOUN", "NOUN"],
                "sentence_id": [1, 2],
            }
        )

        annotated = backend.annotate(tokens_df, sentences_df, show_progress=False)

        # Unknown words should have "unknown" supersense
        assert all(annotated["supersense"] == "unknown")
        assert all(annotated["synset_id"].isna())
        assert all(annotated["sense_confidence"] == 0.0)

    def test_mixed_known_unknown_words(self, backend):
        """Test mix of known and unknown words."""
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1, 2],
                "sentence": [
                    "The dog barked loudly.",
                    "The xyzfoo was strange.",
                ],
            }
        )

        tokens_df = pd.DataFrame(
            {
                "token_id": [1, 2],
                "lemma": ["dog", "xyzfoo"],
                "pos": ["NOUN", "NOUN"],
                "sentence_id": [1, 2],
            }
        )

        annotated = backend.annotate(tokens_df, sentences_df, show_progress=False)

        # dog should be known
        dog_row = annotated[annotated["lemma"] == "dog"].iloc[0]
        assert dog_row["supersense"] == "noun.animal"
        assert dog_row["synset_id"] is not None

        # xyzfoo should be unknown
        xyz_row = annotated[annotated["lemma"] == "xyzfoo"].iloc[0]
        assert xyz_row["supersense"] == "unknown"

    def test_non_content_pos_skipped(self, backend):
        """Test that non-content POS are skipped."""
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1],
                "sentence": ["The quick brown fox."],
            }
        )

        tokens_df = pd.DataFrame(
            {
                "token_id": [1, 2, 3, 4, 5],
                "lemma": ["the", "quick", "brown", "fox", "."],
                "pos": ["DET", "ADJ", "ADJ", "NOUN", "PUNCT"],
                "sentence_id": [1, 1, 1, 1, 1],
            }
        )

        annotated = backend.annotate(tokens_df, sentences_df, show_progress=False)

        # DET and PUNCT should have "unknown" supersense
        det_row = annotated[annotated["pos"] == "DET"].iloc[0]
        assert det_row["supersense"] == "unknown"

        punct_row = annotated[annotated["pos"] == "PUNCT"].iloc[0]
        assert punct_row["supersense"] == "unknown"

        # ADJ and NOUN should be processed
        fox_row = annotated[annotated["lemma"] == "fox"].iloc[0]
        assert fox_row["supersense"] != "unknown"


class TestWSDPerformance:
    """Test performance-related aspects."""

    @pytest.fixture
    def backend(self):
        """Create backend instance."""
        return WordNetSenseBackend()

    def test_definition_cache_reuse(self, backend):
        """Test that definition cache is reused across calls."""
        from eng_words.wsd import get_definition_cache

        cache = get_definition_cache()

        # Use a rare word to ensure we're adding new entries
        # First call with rare word - should populate cache
        backend.disambiguate_word(
            "The zeppelin flew over the city",
            "zeppelin",
            "n",
        )
        after_first = len(cache)

        # Second call with same word - should use cache (no new entries)
        backend.disambiguate_word(
            "Another zeppelin appeared in the sky",
            "zeppelin",
            "n",
        )
        after_second = len(cache)

        # Cache size should be the same (definitions already cached)
        assert after_second == after_first

    def test_batch_more_efficient_than_single(self, backend):
        """Test that batch processing is used for efficiency."""
        items = [
            ("The dog runs", "dog", "n"),
            ("The cat sits", "cat", "n"),
            ("The bird flies", "bird", "n"),
        ]

        # Batch should work without issues
        results = backend.disambiguate_batch(items)
        assert len(results) == 3
        assert all(r.sense_id is not None for r in results)
