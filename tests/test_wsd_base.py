"""Tests for WSD base module and supersenses constants."""

import pandas as pd
import pytest

from eng_words.constants.supersenses import (
    ADJ_ALL,
    ADJ_PPL,
    ADJ_SUPERSENSES,
    ADV_ALL,
    ADV_SUPERSENSES,
    ALL_SUPERSENSES,
    LEXNAME_TO_SUPERSENSE,
    NOUN_SUPERSENSES,
    POS_TO_SUPERSENSES,
    SUPERSENSE_DESCRIPTIONS,
    SUPERSENSE_UNKNOWN,
    VERB_SUPERSENSES,
    get_supersense,
    is_valid_supersense,
)
from eng_words.wsd.base import (
    SenseAnnotation,
    SenseBackend,
    validate_annotated_df,
    validate_sentences_df,
    validate_tokens_df,
)


class TestSupersensesCompleteness:
    """Test that all supersenses are properly defined."""

    def test_noun_supersenses_count(self):
        """There should be 25 noun supersenses in WordNet."""
        assert len(NOUN_SUPERSENSES) == 25

    def test_verb_supersenses_count(self):
        """There should be 15 verb supersenses in WordNet."""
        assert len(VERB_SUPERSENSES) == 15

    def test_adj_supersenses_count(self):
        """There should be 2 adjective supersenses."""
        assert len(ADJ_SUPERSENSES) == 2
        assert ADJ_ALL in ADJ_SUPERSENSES
        assert ADJ_PPL in ADJ_SUPERSENSES

    def test_adv_supersenses_count(self):
        """There should be 1 adverb supersense."""
        assert len(ADV_SUPERSENSES) == 1
        assert ADV_ALL in ADV_SUPERSENSES

    def test_all_supersenses_is_union(self):
        """ALL_SUPERSENSES should be union of all category sets."""
        expected = NOUN_SUPERSENSES | VERB_SUPERSENSES | ADJ_SUPERSENSES | ADV_SUPERSENSES
        assert ALL_SUPERSENSES == expected

    def test_all_supersenses_total_count(self):
        """Total supersenses should be 43 (25 noun + 15 verb + 2 adj + 1 adv)."""
        assert len(ALL_SUPERSENSES) == 43

    def test_noun_supersenses_all_start_with_noun(self):
        """All noun supersenses should start with 'noun.'."""
        for ss in NOUN_SUPERSENSES:
            assert ss.startswith("noun."), f"{ss} doesn't start with 'noun.'"

    def test_verb_supersenses_all_start_with_verb(self):
        """All verb supersenses should start with 'verb.'."""
        for ss in VERB_SUPERSENSES:
            assert ss.startswith("verb."), f"{ss} doesn't start with 'verb.'"


class TestSupersensesMappings:
    """Test supersense mapping functions and dictionaries."""

    def test_lexname_to_supersense_identity(self):
        """Lexname to supersense should be identity mapping for valid supersenses."""
        for ss in ALL_SUPERSENSES:
            assert LEXNAME_TO_SUPERSENSE[ss] == ss

    def test_lexname_to_supersense_unknown(self):
        """Unknown should map to itself."""
        assert LEXNAME_TO_SUPERSENSE[SUPERSENSE_UNKNOWN] == SUPERSENSE_UNKNOWN

    def test_get_supersense_valid(self):
        """get_supersense should return the supersense for valid lexnames."""
        assert get_supersense("noun.person") == "noun.person"
        assert get_supersense("verb.motion") == "verb.motion"
        assert get_supersense("adj.all") == "adj.all"

    def test_get_supersense_invalid(self):
        """get_supersense should return SUPERSENSE_UNKNOWN for invalid lexnames."""
        assert get_supersense("invalid.supersense") == SUPERSENSE_UNKNOWN
        assert get_supersense("") == SUPERSENSE_UNKNOWN
        assert get_supersense("noun") == SUPERSENSE_UNKNOWN

    def test_is_valid_supersense_true(self):
        """is_valid_supersense should return True for valid supersenses."""
        assert is_valid_supersense("noun.person") is True
        assert is_valid_supersense("verb.motion") is True
        assert is_valid_supersense(SUPERSENSE_UNKNOWN) is True

    def test_is_valid_supersense_false(self):
        """is_valid_supersense should return False for invalid supersenses."""
        assert is_valid_supersense("invalid") is False
        assert is_valid_supersense("") is False
        assert is_valid_supersense("noun") is False

    def test_pos_to_supersenses_mapping(self):
        """POS tags should map to correct supersense sets."""
        assert POS_TO_SUPERSENSES["n"] == NOUN_SUPERSENSES
        assert POS_TO_SUPERSENSES["v"] == VERB_SUPERSENSES
        assert POS_TO_SUPERSENSES["a"] == ADJ_SUPERSENSES
        assert POS_TO_SUPERSENSES["s"] == ADJ_SUPERSENSES  # satellite adjectives
        assert POS_TO_SUPERSENSES["r"] == ADV_SUPERSENSES

    def test_all_supersenses_have_descriptions(self):
        """All supersenses should have descriptions."""
        for ss in ALL_SUPERSENSES:
            assert ss in SUPERSENSE_DESCRIPTIONS, f"Missing description for {ss}"
            assert len(SUPERSENSE_DESCRIPTIONS[ss]) > 0


class TestSenseAnnotation:
    """Test SenseAnnotation dataclass."""

    def test_sense_annotation_creation(self):
        """SenseAnnotation should be created with all fields."""
        annotation = SenseAnnotation(
            lemma="run",
            sense_id="run.v.01",
            sense_label="verb.motion",
            confidence=0.85,
            definition="move fast by using one's feet",
        )
        assert annotation.lemma == "run"
        assert annotation.sense_id == "run.v.01"
        assert annotation.sense_label == "verb.motion"
        assert annotation.confidence == 0.85
        assert annotation.definition == "move fast by using one's feet"

    def test_sense_annotation_optional_definition(self):
        """SenseAnnotation definition should be optional."""
        annotation = SenseAnnotation(
            lemma="run",
            sense_id="run.v.01",
            sense_label="verb.motion",
            confidence=0.85,
        )
        assert annotation.definition is None

    def test_sense_annotation_to_dict(self):
        """to_dict should return dictionary with correct keys."""
        annotation = SenseAnnotation(
            lemma="run",
            sense_id="run.v.01",
            sense_label="verb.motion",
            confidence=0.85,
            definition="move fast",
        )
        d = annotation.to_dict()
        assert d["lemma"] == "run"
        assert d["synset_id"] == "run.v.01"
        assert d["supersense"] == "verb.motion"
        assert d["sense_confidence"] == 0.85
        assert d["definition"] == "move fast"

    def test_sense_annotation_unknown(self):
        """SenseAnnotation should handle unknown senses."""
        annotation = SenseAnnotation(
            lemma="xyz",
            sense_id=None,
            sense_label=SUPERSENSE_UNKNOWN,
            confidence=0.0,
        )
        assert annotation.sense_id is None
        assert annotation.sense_label == SUPERSENSE_UNKNOWN


class TestValidationFunctions:
    """Test DataFrame validation functions."""

    def test_validate_tokens_df_missing_column(self):
        """validate_tokens_df should raise for missing columns."""
        df = pd.DataFrame(
            {
                "lemma": ["run"],
                "sentence_id": [1],
                # missing 'pos'
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            validate_tokens_df(df)

    def test_validate_sentences_df_missing_column(self):
        """validate_sentences_df should raise for missing columns."""
        df = pd.DataFrame(
            {
                "sentence_id": [1],
                # missing 'sentence'
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            validate_sentences_df(df)

    def test_validate_annotated_df_missing_column(self):
        """validate_annotated_df should raise for missing columns."""
        df = pd.DataFrame(
            {
                "lemma": ["run"],
                # missing 'supersense' and 'sentence_id'
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            validate_annotated_df(df)


class TestSenseBackendABC:
    """Test SenseBackend abstract base class."""

    def test_concrete_implementation_works(self):
        """Concrete implementation with all methods should work."""

        class MockSenseBackend(SenseBackend):
            def annotate(self, tokens_df, sentences_df):
                return tokens_df

            def aggregate(self, annotated_df):
                return annotated_df

            @property
            def name(self):
                return "mock"

        backend = MockSenseBackend()
        assert backend.name == "mock"
        assert backend.supports_confidence is True
        assert backend.supports_definitions is True

    def test_disambiguate_word_default_raises(self):
        """Default disambiguate_word should raise NotImplementedError."""

        class MockSenseBackend(SenseBackend):
            def annotate(self, tokens_df, sentences_df):
                return tokens_df

            def aggregate(self, annotated_df):
                return annotated_df

            @property
            def name(self):
                return "mock"

        backend = MockSenseBackend()
        with pytest.raises(NotImplementedError):
            backend.disambiguate_word("test sentence", "test", "n")
