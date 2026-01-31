"""Tests for WSD construction detection.

Tests cover:
- ConstructionMatch dataclass
- CONSTRUCTION_PATTERNS validation
- detect_constructions function
"""

import pytest


class TestConstructionMatch:
    """Tests for ConstructionMatch dataclass."""

    def test_create_skip_policy(self) -> None:
        """Test creating a ConstructionMatch with SKIP policy."""
        from eng_words.wsd.constructions import ConstructionMatch

        match = ConstructionMatch(
            construction_id="BE_GOING_TO",
            lemma="go",
            matched_tokens=["is", "going", "to"],
            span=(5, 17),
            policy="SKIP",
            forbid_supersenses=set(),
            prefer_synsets=[],
            reason="Future tense construction",
        )

        assert match.construction_id == "BE_GOING_TO"
        assert match.lemma == "go"
        assert match.policy == "SKIP"
        assert match.reason == "Future tense construction"

    def test_create_constrain_policy(self) -> None:
        """Test creating a ConstructionMatch with CONSTRAIN policy."""
        from eng_words.wsd.constructions import ConstructionMatch

        match = ConstructionMatch(
            construction_id="COME_ON",
            lemma="come",
            matched_tokens=["come", "on"],
            span=(0, 7),
            policy="CONSTRAIN",
            forbid_supersenses={"verb.motion"},
            prefer_synsets=[],
            reason="Phrasal verb: encouragement",
        )

        assert match.policy == "CONSTRAIN"
        assert "verb.motion" in match.forbid_supersenses

    def test_create_override_policy(self) -> None:
        """Test creating a ConstructionMatch with OVERRIDE policy."""
        from eng_words.wsd.constructions import ConstructionMatch

        match = ConstructionMatch(
            construction_id="OVER_THERE",
            lemma="over",
            matched_tokens=["over", "there"],
            span=(10, 20),
            policy="OVERRIDE",
            forbid_supersenses=set(),
            prefer_synsets=["over.r.01"],
            reason="Spatial reference",
        )

        assert match.policy == "OVERRIDE"
        assert "over.r.01" in match.prefer_synsets

    def test_invalid_policy_raises(self) -> None:
        """Test that invalid policy raises ValueError."""
        from eng_words.wsd.constructions import ConstructionMatch

        with pytest.raises(ValueError, match="Invalid policy"):
            ConstructionMatch(
                construction_id="TEST",
                lemma="test",
                matched_tokens=["test"],
                span=(0, 4),
                policy="INVALID",  # type: ignore
                forbid_supersenses=set(),
                prefer_synsets=[],
                reason="Test",
            )


class TestConstructionPatterns:
    """Tests for CONSTRUCTION_PATTERNS dictionary."""

    def test_all_patterns_have_required_fields(self) -> None:
        """Test that all patterns have required fields."""
        from eng_words.wsd.constructions import CONSTRUCTION_PATTERNS

        required_fields = {"pattern", "target_lemma", "policy", "reason"}

        for pattern_id, pattern in CONSTRUCTION_PATTERNS.items():
            for field in required_fields:
                assert field in pattern, f"{pattern_id} missing {field}"

    def test_all_patterns_valid_regex(self) -> None:
        """Test that all pattern regexes are valid."""
        import re

        from eng_words.wsd.constructions import CONSTRUCTION_PATTERNS

        for pattern_id, pattern in CONSTRUCTION_PATTERNS.items():
            try:
                re.compile(pattern["pattern"], re.IGNORECASE)
            except re.error as e:
                pytest.fail(f"{pattern_id} has invalid regex: {e}")


class TestDetectConstructions:
    """Tests for detect_constructions function."""

    def test_be_going_to_future_tense(self) -> None:
        """Test detection of 'be going to' future tense."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "I am going to finish this today."
        matches = detect_constructions(sentence, lemma="go", pos="VERB")

        assert len(matches) >= 1
        assert matches[0].construction_id == "BE_GOING_TO"
        assert matches[0].policy == "SKIP"

    def test_gonna_informal(self) -> None:
        """Test detection of 'gonna' informal future."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "I'm gonna do it later."
        matches = detect_constructions(sentence, lemma="go", pos="VERB")

        assert len(matches) >= 1
        assert matches[0].construction_id == "GONNA"
        assert matches[0].policy == "SKIP"

    def test_come_on_encouragement(self) -> None:
        """Test detection of 'come on' phrasal verb."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "Come on, let's go!"
        matches = detect_constructions(sentence, lemma="come", pos="VERB")

        assert len(matches) >= 1
        assert matches[0].construction_id == "COME_ON"
        assert matches[0].policy == "CONSTRAIN"

    def test_go_on_continue(self) -> None:
        """Test detection of 'go on' phrasal verb."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "Please go on with your story."
        matches = detect_constructions(sentence, lemma="go", pos="VERB")

        assert len(matches) >= 1
        assert matches[0].construction_id == "GO_ON"
        assert matches[0].policy == "CONSTRAIN"

    def test_make_sure_expression(self) -> None:
        """Test detection of 'make sure' expression."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "Make sure you lock the door."
        matches = detect_constructions(sentence, lemma="make", pos="VERB")

        assert len(matches) >= 1
        assert matches[0].construction_id == "MAKE_SURE"

    def test_call_up_telephone(self) -> None:
        """Test detection of 'call up' phrasal verb."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "I called up my friend yesterday."
        matches = detect_constructions(sentence, lemma="call", pos="VERB")

        assert len(matches) >= 1
        assert matches[0].construction_id == "CALL_UP"

    def test_no_match_for_regular_verb(self) -> None:
        """Test that regular verbs don't match constructions."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "I go to the store every day."
        matches = detect_constructions(sentence, lemma="go", pos="VERB")

        # Should not match "going to" (future tense) pattern
        assert all(m.construction_id != "BE_GOING_TO" for m in matches)

    def test_no_match_wrong_lemma(self) -> None:
        """Test that patterns only match for target lemma."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "I am going to the store."
        # Looking for "run" but sentence has "go"
        matches = detect_constructions(sentence, lemma="run", pos="VERB")

        assert len(matches) == 0

    def test_case_insensitive(self) -> None:
        """Test that matching is case-insensitive."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "COME ON, let's hurry!"
        matches = detect_constructions(sentence, lemma="come", pos="VERB")

        assert len(matches) >= 1
        assert matches[0].construction_id == "COME_ON"

    def test_span_is_correct(self) -> None:
        """Test that character span is correctly calculated."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "He is going to win."
        matches = detect_constructions(sentence, lemma="go", pos="VERB")

        assert len(matches) >= 1
        span = matches[0].span
        assert sentence[span[0] : span[1]].lower() == "is going to"

    def test_point_of_view(self) -> None:
        """Test detection of 'point of view' multiword expression."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "From my point of view, this is correct."
        matches = detect_constructions(sentence, lemma="point", pos="NOUN")

        assert len(matches) >= 1
        assert matches[0].construction_id == "POINT_OF_VIEW"
        assert matches[0].policy == "SKIP"

    def test_at_the_time(self) -> None:
        """Test detection of 'at the time' temporal expression."""
        from eng_words.wsd.constructions import detect_constructions

        sentence = "At the time, I didn't know."
        matches = detect_constructions(sentence, lemma="time", pos="NOUN")

        assert len(matches) >= 1
        assert matches[0].construction_id == "AT_THE_TIME"


class TestConstructionIntegration:
    """Integration tests for construction detection with WSD."""

    def test_skip_policy_returns_none_sense(self) -> None:
        """Test that SKIP policy results in None sense_id."""
        from eng_words.wsd.constructions import apply_construction_policy, detect_constructions

        sentence = "I am going to finish this."
        matches = detect_constructions(sentence, lemma="go", pos="VERB")

        assert len(matches) >= 1
        result = apply_construction_policy(matches[0], candidates=[])

        assert result is not None
        assert result.get("skip") is True

    def test_constrain_policy_returns_guidance(self) -> None:
        """Test that CONSTRAIN policy returns guidance for WSD."""
        from eng_words.wsd.constructions import apply_construction_policy, detect_constructions

        sentence = "Come on, let's go!"
        matches = detect_constructions(sentence, lemma="come", pos="VERB")

        assert len(matches) >= 1
        result = apply_construction_policy(
            matches[0],
            candidates=["come.v.01", "come.v.02"],  # Simulated
        )

        assert result is not None
        assert result.get("skip") is False
        assert "construction_id" in result
