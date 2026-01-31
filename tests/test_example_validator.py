"""Tests for example validation."""

import pytest

from eng_words.llm.smart_card_generator import SmartCard
from eng_words.validation.example_validator import (
    _get_synset_synonyms,
    _get_word_forms,
    _word_in_text,
    fix_invalid_cards,
    validate_card_examples,
)


class TestGetWordForms:
    """Tests for _get_word_forms function."""

    def test_base_form_included(self):
        """Base form should always be included."""
        forms = _get_word_forms("run")
        assert "run" in forms

    def test_regular_verb_forms(self):
        """Regular verb forms (+s, +ed, +ing)."""
        forms = _get_word_forms("walk")
        assert "walks" in forms
        assert "walked" in forms
        assert "walking" in forms

    def test_e_ending_verb(self):
        """Verbs ending in -e (make -> making)."""
        forms = _get_word_forms("make")
        assert "making" in forms
        assert "makes" in forms
        assert "maked" in forms or "made" in forms  # irregular

    def test_y_ending_verb(self):
        """Verbs ending in -y (try -> tried, tries)."""
        forms = _get_word_forms("try")
        assert "tried" in forms
        assert "tries" in forms

    def test_irregular_forms(self):
        """Irregular forms from dictionary."""
        forms = _get_word_forms("go")
        assert "went" in forms
        assert "gone" in forms
        assert "goes" in forms

    def test_irregular_adjective_bad(self):
        """Irregular adjective 'bad' -> 'worse', 'worst'."""
        forms = _get_word_forms("bad")
        assert "worse" in forms
        assert "worst" in forms

    def test_irregular_become(self):
        """Irregular verb 'become' -> 'became'."""
        forms = _get_word_forms("become")
        assert "became" in forms
        assert "becomes" in forms


class TestGetSynsetSynonyms:
    """Tests for _get_synset_synonyms function."""

    def test_valid_synset(self):
        """Should return synonyms for valid synset."""
        synonyms = _get_synset_synonyms("run.v.01")
        assert "run" in synonyms
        # run.v.01 has synonyms like 'run'
        assert len(synonyms) > 0

    def test_invalid_synset(self):
        """Should return empty set for invalid synset."""
        synonyms = _get_synset_synonyms("invalid.x.99")
        assert synonyms == set()

    def test_synonyms_include_forms(self):
        """Synonyms should include morphological forms."""
        # think.v.01 includes 'believe'
        synonyms = _get_synset_synonyms("think.v.01")
        assert "think" in synonyms or "thinks" in synonyms or "thought" in synonyms


class TestWordInText:
    """Tests for _word_in_text function."""

    def test_exact_match(self):
        """Should find exact word match."""
        assert _word_in_text("run", "I like to run every day")

    def test_case_insensitive(self):
        """Should be case insensitive."""
        assert _word_in_text("run", "RUN is fun")
        assert _word_in_text("RUN", "I like to run")

    def test_word_boundary(self):
        """Should respect word boundaries."""
        assert not _word_in_text("run", "I was running yesterday")
        assert _word_in_text("run", "I run daily")

    def test_punctuation(self):
        """Should find word before punctuation."""
        assert _word_in_text("run", "Let's run!")
        assert _word_in_text("run", "Can you run?")


class TestValidateCardExamples:
    """Tests for validate_card_examples function."""

    @pytest.fixture
    def valid_card(self):
        """Card with valid examples containing the lemma."""
        return SmartCard(
            lemma="run",
            pos="verb",
            supersense="verb.motion",
            selected_examples=[
                "I like to run every morning.",
                "She runs faster than me.",
            ],
            excluded_examples=[],
            simple_definition="move fast on foot",
            translation_ru="run",
            generated_example="He runs in the park.",
            wn_definition="move fast by using one's feet",
            book_name="test_book",
            primary_synset="run.v.01",
            synset_group=["run.v.01"],
        )

    @pytest.fixture
    def invalid_card(self):
        """Card with example NOT containing the lemma."""
        return SmartCard(
            lemma="brood",
            pos="verb",
            supersense="verb.cognition",
            selected_examples=[
                "Hortense, because of the hovering floorwalker, was pretending.",
            ],
            excluded_examples=[],
            simple_definition="think moodily",
            translation_ru="brood",
            generated_example="She brooded over her mistakes.",
            wn_definition="think moodily or anxiously",
            book_name="test_book",
            primary_synset="brood.v.01",
            synset_group=["brood.v.01"],
        )

    def test_valid_card(self, valid_card):
        """Card with lemma in examples should be valid."""
        result = validate_card_examples(valid_card)
        assert result.is_valid
        assert len(result.valid_examples) == 2
        assert len(result.invalid_examples) == 0

    def test_invalid_card(self, invalid_card):
        """Card without lemma in examples should be invalid."""
        result = validate_card_examples(invalid_card)
        assert not result.is_valid
        assert len(result.valid_examples) == 0
        assert len(result.invalid_examples) == 1

    def test_mixed_examples(self):
        """Card with some valid and some invalid examples."""
        card = SmartCard(
            lemma="walk",
            pos="verb",
            supersense="verb.motion",
            selected_examples=[
                "I walk to school.",  # valid
                "The floorwalker was hovering.",  # invalid (partial match)
            ],
            excluded_examples=[],
            simple_definition="move on foot",
            translation_ru="walk",
            generated_example="They walked home.",
            wn_definition="use one's feet to advance",
            book_name="test_book",
            primary_synset="walk.v.01",
            synset_group=["walk.v.01"],
        )
        result = validate_card_examples(card)
        assert result.is_valid
        assert len(result.valid_examples) == 1
        assert len(result.invalid_examples) == 1

    def test_synonym_match(self):
        """Card where synonym appears instead of lemma."""
        # think.v.01 has "believe" as synonym
        card = SmartCard(
            lemma="believe",
            pos="verb",
            supersense="verb.cognition",
            selected_examples=[
                "She thought him better than the others.",  # 'thought' is form of 'think'
            ],
            excluded_examples=[],
            simple_definition="judge or regard",
            translation_ru="count",
            generated_example="I believe you.",
            wn_definition="judge or regard",
            book_name="test_book",
            primary_synset="think.v.01",
            synset_group=["think.v.01"],
        )
        result = validate_card_examples(card)
        # Should be valid because 'thought' is a form of 'think' which is in the synset
        assert result.is_valid

    def test_morphological_form_match(self):
        """Card where morphological form appears."""
        card = SmartCard(
            lemma="bad",
            pos="adj",
            supersense="adj.all",
            selected_examples=[
                "She felt worse than before.",  # 'worse' is form of 'bad'
            ],
            excluded_examples=[],
            simple_definition="capable of harming",
            translation_ru="bad",
            generated_example="This is bad.",
            wn_definition="capable of harming",
            book_name="test_book",
            primary_synset="bad.s.09",
            synset_group=["bad.s.09"],
        )
        result = validate_card_examples(card)
        assert result.is_valid
        assert "worse" in result.found_forms

    def test_empty_examples(self):
        """Card with no examples should be invalid."""
        card = SmartCard(
            lemma="test",
            pos="noun",
            supersense="noun.act",
            selected_examples=[],
            excluded_examples=[],
            simple_definition="a test",
            translation_ru="test",
            generated_example="This is a test.",
            wn_definition="examination",
            book_name="test_book",
            primary_synset="test.n.01",
            synset_group=["test.n.01"],
        )
        result = validate_card_examples(card)
        assert not result.is_valid


class TestFixInvalidCards:
    """Tests for fix_invalid_cards function."""

    @pytest.fixture
    def valid_card(self):
        """Card with valid examples."""
        return SmartCard(
            lemma="run",
            pos="verb",
            supersense="verb.motion",
            selected_examples=["I run every day."],
            excluded_examples=[],
            simple_definition="move fast",
            translation_ru="run",
            generated_example="He runs in the park.",
            wn_definition="move fast",
            book_name="test_book",
            primary_synset="run.v.01",
            synset_group=["run.v.01"],
        )

    @pytest.fixture
    def invalid_card_with_generated(self):
        """Card with invalid examples but has generated_example."""
        return SmartCard(
            lemma="brood",
            pos="verb",
            supersense="verb.cognition",
            selected_examples=["The floorwalker was hovering."],
            excluded_examples=[],
            simple_definition="think moodily",
            translation_ru="consider",
            generated_example="She brooded over her mistakes.",
            wn_definition="think moodily",
            book_name="test_book",
            primary_synset="brood.v.01",
            synset_group=["brood.v.01"],
        )

    @pytest.fixture
    def invalid_card_no_generated(self):
        """Card with invalid examples and no generated_example."""
        return SmartCard(
            lemma="obscure",
            pos="verb",
            supersense="verb.perception",
            selected_examples=["Random sentence without the word."],
            excluded_examples=[],
            simple_definition="make unclear",
            translation_ru="obscure",
            generated_example="",  # No generated example
            wn_definition="make unclear",
            book_name="test_book",
            primary_synset="obscure.v.01",
            synset_group=["obscure.v.01"],
        )

    def test_valid_cards_unchanged(self, valid_card):
        """Valid cards should remain unchanged."""
        fixed, review = fix_invalid_cards([valid_card])
        assert len(fixed) == 1
        assert len(review) == 0
        assert fixed[0].selected_examples == ["I run every day."]

    def test_invalid_card_uses_generated(self, invalid_card_with_generated):
        """Invalid card should use generated_example."""
        fixed, review = fix_invalid_cards([invalid_card_with_generated])
        assert len(fixed) == 1
        assert len(review) == 0
        assert fixed[0].selected_examples == ["She brooded over her mistakes."]

    def test_invalid_card_no_generated_goes_to_review(self, invalid_card_no_generated):
        """Invalid card without generated_example goes to review."""
        fixed, review = fix_invalid_cards([invalid_card_no_generated])
        assert len(fixed) == 0
        assert len(review) == 1
        assert review[0].lemma == "obscure"

    def test_mixed_examples_keeps_valid(self):
        """Card with mixed examples keeps only valid ones."""
        card = SmartCard(
            lemma="walk",
            pos="verb",
            supersense="verb.motion",
            selected_examples=[
                "I walk to school.",  # valid
                "Random sentence.",  # invalid
            ],
            excluded_examples=[],
            simple_definition="move on foot",
            translation_ru="walk",
            generated_example="They walked home.",
            wn_definition="move on foot",
            book_name="test_book",
            primary_synset="walk.v.01",
            synset_group=["walk.v.01"],
        )
        fixed, review = fix_invalid_cards([card])
        assert len(fixed) == 1
        assert len(review) == 0
        assert len(fixed[0].selected_examples) == 1
        assert fixed[0].selected_examples[0] == "I walk to school."
        # Invalid example should be moved to excluded
        assert "Random sentence." in fixed[0].excluded_examples

    def test_disable_generated_example_fallback(self, invalid_card_with_generated):
        """Should not use generated_example when disabled."""
        fixed, review = fix_invalid_cards(
            [invalid_card_with_generated], use_generated_example=False
        )
        assert len(fixed) == 0
        assert len(review) == 1


class TestExtendedIrregularForms:
    """Tests for extended IRREGULAR_FORMS dictionary."""

    def test_blow_recognizes_blew(self):
        """Test that 'blew' is recognized as form of 'blow'."""
        from eng_words.validation.example_validator import _get_word_forms

        forms = _get_word_forms("blow")
        assert "blew" in forms
        assert "blown" in forms
        assert "blows" in forms

    def test_catch_recognizes_caught(self):
        """Test that 'caught' is recognized as form of 'catch'."""
        from eng_words.validation.example_validator import _get_word_forms

        forms = _get_word_forms("catch")
        assert "caught" in forms
        assert "catches" in forms

    def test_cling_recognizes_clung(self):
        """Test that 'clung' is recognized as form of 'cling'."""
        from eng_words.validation.example_validator import _get_word_forms

        forms = _get_word_forms("cling")
        assert "clung" in forms
        assert "clings" in forms

    def test_hang_recognizes_hung(self):
        """Test that 'hung' is recognized as form of 'hang'."""
        from eng_words.validation.example_validator import _get_word_forms

        forms = _get_word_forms("hang")
        assert "hung" in forms
        assert "hanged" in forms
        assert "hangs" in forms

    def test_throw_recognizes_threw(self):
        """Test that 'threw' is recognized as form of 'throw'."""
        from eng_words.validation.example_validator import _get_word_forms

        forms = _get_word_forms("throw")
        assert "threw" in forms
        assert "thrown" in forms
        assert "throws" in forms

    def test_foot_recognizes_feet(self):
        """Test that 'feet' is recognized as plural of 'foot'."""
        from eng_words.validation.example_validator import _get_word_forms

        forms = _get_word_forms("foot")
        assert "feet" in forms

    def test_tooth_recognizes_teeth(self):
        """Test that 'teeth' is recognized as plural of 'tooth'."""
        from eng_words.validation.example_validator import _get_word_forms

        forms = _get_word_forms("tooth")
        assert "teeth" in forms

    def test_validation_with_extended_forms(self):
        """Test validation with extended irregular forms."""
        from eng_words.llm.smart_card_generator import SmartCard
        from eng_words.validation.example_validator import validate_card_examples

        # Test card with 'blew' for 'blow'
        card = SmartCard(
            lemma="blow",
            pos="v",
            supersense="verb",
            selected_examples=["The wind blew strongly through the trees."],
            excluded_examples=[],
            simple_definition="to move air",
            translation_ru="blow",
            generated_example="",
            wn_definition="to move air",
            book_name="test",
            primary_synset="blow.v.01",
            synset_group=["blow.v.01"],
        )

        result = validate_card_examples(card)
        assert result.is_valid
        assert "blew" in result.found_forms
        assert len(result.invalid_examples) == 0

        # Test card with 'caught' for 'catch'
        card2 = SmartCard(
            lemma="catch",
            pos="v",
            supersense="verb",
            selected_examples=["She caught the ball."],
            excluded_examples=[],
            simple_definition="to grab something",
            translation_ru="catch",
            generated_example="",
            wn_definition="to grab something",
            book_name="test",
            primary_synset="catch.v.01",
            synset_group=["catch.v.01"],
        )

        result2 = validate_card_examples(card2)
        assert result2.is_valid
        assert "caught" in result2.found_forms
        assert len(result2.invalid_examples) == 0

    def test_extended_forms_coverage(self):
        """Test that extended IRREGULAR_FORMS includes standard verbs."""
        from eng_words.validation.example_validator import IRREGULAR_FORMS

        # Standard list of common irregular verbs
        standard_verbs = [
            "beat",
            "bend",
            "bet",
            "bid",
            "bite",
            "bleed",
            "breed",
            "burn",
            "burst",
            "cast",
            "cost",
            "creep",
            "deal",
            "dig",
            "dream",
            "drink",
            "eat",
            "feed",
            "fight",
            "flee",
            "fling",
            "fly",
            "forbid",
            "forecast",
            "forgive",
            "freeze",
            "grind",
            "hide",
            "hit",
            "kneel",
            "knit",
            "lend",
            "light",
            "mislay",
            "mislead",
            "prove",
            "quit",
            "rid",
            "ride",
            "ring",
            "saw",
            "seek",
            "sew",
            "shear",
            "shed",
            "shine",
            "shoe",
            "shoot",
            "show",
            "shrink",
            "shut",
            "sing",
            "sink",
            "slay",
            "sleep",
            "slide",
            "sling",
            "slink",
            "slit",
            "smell",
            "smite",
            "sow",
            "spell",
            "spill",
            "spit",
            "split",
            "spoil",
            "spread",
            "steal",
            "stick",
            "sting",
            "stink",
            "strew",
            "stride",
            "string",
            "strive",
            "swear",
            "swim",
            "teach",
            "tear",
            "thrive",
            "thrust",
            "tread",
            "undergo",
            "undertake",
            "undo",
            "upset",
            "wake",
            "weave",
            "wed",
            "weep",
            "wet",
            "win",
            "wind",
            "withdraw",
            "withstand",
            "wring",
        ]

        missing = [v for v in standard_verbs if v not in IRREGULAR_FORMS]
        if missing:
            pytest.fail(f"Missing verbs in IRREGULAR_FORMS: {missing[:10]}")

        # Check that we have at least 150 verbs (expanded from ~100)
        assert len(IRREGULAR_FORMS) >= 150, f"Only {len(IRREGULAR_FORMS)} verbs in dictionary"
