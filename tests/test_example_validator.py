"""Tests for example validation."""

from eng_words.validation.example_validator import (
    _get_word_forms,
    _word_in_text,
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
