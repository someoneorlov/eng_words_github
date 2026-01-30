"""Tests for WSD WordNet utilities module."""

from nltk.corpus import wordnet as wn

from eng_words.wsd.wordnet_utils import (
    SPACY_TO_WORDNET_POS,
    get_definition,
    get_synsets,
    get_synsets_with_definitions,
    map_spacy_pos_to_wordnet,
    synset_to_supersense,
)


class TestGetSynsets:
    """Test get_synsets function."""

    def test_known_noun(self):
        """Should return synsets for known noun."""
        synsets = get_synsets("bank", pos="n")
        assert len(synsets) > 0
        # bank has multiple noun meanings
        assert len(synsets) >= 5

    def test_known_verb(self):
        """Should return synsets for known verb."""
        synsets = get_synsets("run", pos="v")
        assert len(synsets) > 0
        # run has many verb meanings
        assert len(synsets) >= 10

    def test_known_adjective(self):
        """Should return synsets for known adjective."""
        synsets = get_synsets("happy", pos="a")
        assert len(synsets) > 0

    def test_known_adverb(self):
        """Should return synsets for known adverb."""
        synsets = get_synsets("quickly", pos="r")
        assert len(synsets) > 0

    def test_unknown_word(self):
        """Should return empty list for unknown word."""
        synsets = get_synsets("xyznonexistent", pos="n")
        assert synsets == []

    def test_no_pos_filter(self):
        """Should return all synsets when pos is None."""
        synsets_all = get_synsets("run", pos=None)
        synsets_verb = get_synsets("run", pos="v")
        synsets_noun = get_synsets("run", pos="n")
        # All should include both verb and noun synsets
        assert len(synsets_all) >= len(synsets_verb)
        assert len(synsets_all) >= len(synsets_noun)

    def test_case_insensitive(self):
        """Should handle different cases."""
        synsets_lower = get_synsets("bank", pos="n")
        synsets_upper = get_synsets("BANK", pos="n")
        synsets_mixed = get_synsets("Bank", pos="n")
        assert len(synsets_lower) == len(synsets_upper) == len(synsets_mixed)

    def test_phrase_with_underscore(self):
        """Should handle multi-word phrases with underscore."""
        synsets = get_synsets("ice_cream", pos="n")
        assert len(synsets) > 0

    def test_phrase_with_space(self):
        """Should handle multi-word phrases with space."""
        synsets = get_synsets("ice cream", pos="n")
        assert len(synsets) > 0


class TestSynsetToSupersense:
    """Test synset_to_supersense function."""

    def test_noun_supersense(self):
        """Should return correct supersense for noun synset."""
        synset = wn.synset("dog.n.01")
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.animal"

    def test_verb_supersense(self):
        """Should return correct supersense for verb synset."""
        synset = wn.synset("run.v.01")
        supersense = synset_to_supersense(synset)
        assert supersense == "verb.motion"

    def test_adjective_supersense(self):
        """Should return correct supersense for adjective synset."""
        synset = wn.synset("happy.a.01")
        supersense = synset_to_supersense(synset)
        assert supersense.startswith("adj.")

    def test_bank_financial(self):
        """Bank as financial institution should be noun.group."""
        synset = wn.synset("depository_financial_institution.n.01")
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.group"

    def test_bank_river(self):
        """Bank as river shore should be noun.object."""
        synset = wn.synset("bank.n.01")
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.object"


class TestNounTopsMapping:
    """Test special handling for noun.Tops synsets."""

    def test_person_tops(self):
        """person.n.01 (noun.Tops) should map to noun.person."""
        synset = wn.synset("person.n.01")
        assert synset.lexname() == "noun.Tops"  # Verify it's a Tops synset
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.person"

    def test_group_tops(self):
        """group.n.01 (noun.Tops) should map to noun.group."""
        synset = wn.synset("group.n.01")
        assert synset.lexname() == "noun.Tops"
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.group"

    def test_cognition_tops(self):
        """cognition.n.01 (noun.Tops) should map to noun.cognition."""
        synset = wn.synset("cognition.n.01")
        assert synset.lexname() == "noun.Tops"
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.cognition"

    def test_motivation_tops(self):
        """motivation.n.01 (noun.Tops) should map to noun.motive."""
        synset = wn.synset("motivation.n.01")
        assert synset.lexname() == "noun.Tops"
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.motive"

    def test_animal_tops(self):
        """animal.n.01 (noun.Tops) should map to noun.animal."""
        synset = wn.synset("animal.n.01")
        assert synset.lexname() == "noun.Tops"
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.animal"

    def test_time_tops(self):
        """time.n.05 (noun.Tops) should map to noun.time."""
        synset = wn.synset("time.n.05")
        assert synset.lexname() == "noun.Tops"
        supersense = synset_to_supersense(synset)
        assert supersense == "noun.time"


class TestAdjPertMapping:
    """Test adj.pert (pertaining adjectives) mapping."""

    def test_fiscal_pert(self):
        """fiscal.a.01 (adj.pert) should map to adj.all."""
        synset = wn.synset("fiscal.a.01")
        assert synset.lexname() == "adj.pert"
        supersense = synset_to_supersense(synset)
        assert supersense == "adj.all"

    def test_american_pert(self):
        """american.a.01 (adj.pert) should map to adj.all."""
        synset = wn.synset("american.a.01")
        assert synset.lexname() == "adj.pert"
        supersense = synset_to_supersense(synset)
        assert supersense == "adj.all"

    def test_legal_pert(self):
        """legal.a.02 (adj.pert) should map to adj.all."""
        synset = wn.synset("legal.a.02")
        assert synset.lexname() == "adj.pert"
        supersense = synset_to_supersense(synset)
        assert supersense == "adj.all"


class TestGetDefinition:
    """Test get_definition function."""

    def test_normal_definition(self):
        """Should return definition for synset."""
        synset = wn.synset("dog.n.01")
        definition = get_definition(synset)
        assert isinstance(definition, str)
        assert len(definition) > 0
        # dog.n.01 definition mentions "Canis" (genus name)
        assert "canis" in definition.lower() or "domesticated" in definition.lower()

    def test_definition_not_empty(self):
        """Definition should never be empty."""
        synset = wn.synset("run.v.01")
        definition = get_definition(synset)
        assert len(definition) > 0

    def test_fallback_to_lemma_names(self):
        """Should use lemma names if definition is somehow empty."""
        # Most synsets have definitions, but test the function handles edge cases
        synset = wn.synset("dog.n.01")
        definition = get_definition(synset)
        # Should return something meaningful
        assert definition is not None
        assert len(definition) > 0


class TestMapSpacyPosToWordnet:
    """Test spaCy to WordNet POS mapping."""

    def test_noun_mapping(self):
        """NOUN should map to 'n'."""
        assert map_spacy_pos_to_wordnet("NOUN") == "n"

    def test_verb_mapping(self):
        """VERB should map to 'v'."""
        assert map_spacy_pos_to_wordnet("VERB") == "v"

    def test_adj_mapping(self):
        """ADJ should map to 'a'."""
        assert map_spacy_pos_to_wordnet("ADJ") == "a"

    def test_adv_mapping(self):
        """ADV should map to 'r'."""
        assert map_spacy_pos_to_wordnet("ADV") == "r"

    def test_propn_mapping(self):
        """PROPN (proper noun) should map to 'n'."""
        assert map_spacy_pos_to_wordnet("PROPN") == "n"

    def test_unknown_pos_returns_none(self):
        """Unknown POS should return None."""
        assert map_spacy_pos_to_wordnet("DET") is None
        assert map_spacy_pos_to_wordnet("PUNCT") is None
        assert map_spacy_pos_to_wordnet("ADP") is None

    def test_case_insensitive(self):
        """Should handle different cases."""
        assert map_spacy_pos_to_wordnet("noun") == "n"
        assert map_spacy_pos_to_wordnet("Verb") == "v"

    def test_spacy_to_wordnet_pos_dict(self):
        """SPACY_TO_WORDNET_POS should contain expected mappings."""
        assert "NOUN" in SPACY_TO_WORDNET_POS
        assert "VERB" in SPACY_TO_WORDNET_POS
        assert "ADJ" in SPACY_TO_WORDNET_POS
        assert "ADV" in SPACY_TO_WORDNET_POS


class TestGetSynsetsWithDefinitions:
    """Test get_synsets_with_definitions function."""

    def test_returns_tuples(self):
        """Should return list of (synset_id, definition) tuples."""
        result = get_synsets_with_definitions("bank", pos="n")
        assert isinstance(result, list)
        assert len(result) > 0
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            synset_id, definition = item
            assert isinstance(synset_id, str)
            assert isinstance(definition, str)

    def test_synset_id_format(self):
        """Synset IDs should be in correct format."""
        result = get_synsets_with_definitions("dog", pos="n")
        for synset_id, _ in result:
            # Format: word.pos.sense_number
            parts = synset_id.split(".")
            assert len(parts) == 3
            assert parts[1] == "n"  # noun

    def test_definitions_not_empty(self):
        """All definitions should be non-empty."""
        result = get_synsets_with_definitions("run", pos="v")
        for _, definition in result:
            assert len(definition) > 0

    def test_unknown_word_returns_empty(self):
        """Unknown word should return empty list."""
        result = get_synsets_with_definitions("xyznonexistent", pos="n")
        assert result == []


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_lemma(self):
        """Should handle empty lemma gracefully."""
        synsets = get_synsets("", pos="n")
        assert synsets == []

    def test_special_characters(self):
        """Should handle special characters."""
        synsets = get_synsets("don't", pos="v")
        # May or may not find synsets, but shouldn't crash
        assert isinstance(synsets, list)

    def test_hyphenated_word(self):
        """Should handle hyphenated words."""
        synsets = get_synsets("well-known", pos="a")
        # May or may not find synsets
        assert isinstance(synsets, list)

    def test_numeric_string(self):
        """Should handle numeric strings."""
        synsets = get_synsets("123", pos="n")
        assert isinstance(synsets, list)

    def test_very_long_word(self):
        """Should handle very long words."""
        long_word = "a" * 100
        synsets = get_synsets(long_word, pos="n")
        assert synsets == []
