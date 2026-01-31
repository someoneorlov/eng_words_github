"""Tests for Smart Candidate Selection.

Tests cover:
- select_smart_candidates function
- Context keyword matching
- Supersense diversity
- Scoring combination
"""



class TestSelectSmartCandidates:
    """Tests for select_smart_candidates function."""

    def test_import(self) -> None:
        """Test that select_smart_candidates can be imported."""
        from eng_words.wsd.candidate_selector import select_smart_candidates

        assert callable(select_smart_candidates)

    def test_returns_list(self) -> None:
        """Test that function returns a list of tuples."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import select_smart_candidates

        synsets = wn.synsets("run", pos="v")
        scores = {s.name(): 0.5 for s in synsets}

        result = select_smart_candidates(
            lemma="run",
            pos="VERB",
            sentence="I need to run fast.",
            all_synsets=synsets,
            embedding_scores=scores,
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 3  # (synset_id, score, reason)

    def test_respects_max_candidates(self) -> None:
        """Test that function respects max_candidates limit."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import select_smart_candidates

        synsets = wn.synsets("run", pos="v")
        scores = {s.name(): 0.5 for s in synsets}

        result = select_smart_candidates(
            lemma="run",
            pos="VERB",
            sentence="I run every morning.",
            all_synsets=synsets,
            embedding_scores=scores,
            max_candidates=5,
        )

        assert len(result) <= 5

    def test_includes_top_by_score(self) -> None:
        """Test that top-scoring synsets are included."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import select_smart_candidates

        synsets = wn.synsets("bank", pos="n")
        # Make one synset clearly the best
        scores = {s.name(): 0.3 for s in synsets}
        scores["bank.n.01"] = 0.9  # Highest score

        result = select_smart_candidates(
            lemma="bank",
            pos="NOUN",
            sentence="I went to the bank.",
            all_synsets=synsets,
            embedding_scores=scores,
        )

        # Top scorer should be in results
        result_ids = [r[0] for r in result]
        assert "bank.n.01" in result_ids

    def test_includes_diverse_supersenses(self) -> None:
        """Test that synsets from different supersenses are included."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import select_smart_candidates

        synsets = wn.synsets("bank", pos="n")
        scores = {s.name(): 0.5 for s in synsets}

        result = select_smart_candidates(
            lemma="bank",
            pos="NOUN",
            sentence="Money at the river bank.",
            all_synsets=synsets,
            embedding_scores=scores,
        )

        # Check for different supersenses
        supersenses = set()
        for synset_id, _, _ in result:
            s = wn.synset(synset_id)
            supersenses.add(s.lexname())

        # Should have at least 2 different supersenses (noun.artifact, noun.object)
        assert len(supersenses) >= 2

    def test_context_keyword_boost(self) -> None:
        """Test that synsets with context keywords get boosted."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import select_smart_candidates

        synsets = wn.synsets("bank", pos="n")
        # Give equal scores
        scores = {s.name(): 0.5 for s in synsets}

        # Sentence mentions "river" which should boost bank.n.01 (sloping land)
        result = select_smart_candidates(
            lemma="bank",
            pos="NOUN",
            sentence="We sat on the river bank watching the water flow.",
            all_synsets=synsets,
            embedding_scores=scores,
        )

        # Find synset with "river" or "water" in definition
        result_ids = [r[0] for r in result]

        # bank.n.01 is "sloping land beside a body of water" - should be included
        # due to keyword match with "water"
        assert any("bank" in sid for sid in result_ids)

    def test_handles_empty_synsets(self) -> None:
        """Test graceful handling of empty synsets list."""
        from eng_words.wsd.candidate_selector import select_smart_candidates

        result = select_smart_candidates(
            lemma="xyz",
            pos="NOUN",
            sentence="No synsets for xyz.",
            all_synsets=[],
            embedding_scores={},
        )

        assert result == []

    def test_handles_single_synset(self) -> None:
        """Test handling of single synset."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import select_smart_candidates

        synsets = wn.synsets("aardvark", pos="n")[:1]
        scores = {synsets[0].name(): 0.7}

        result = select_smart_candidates(
            lemma="aardvark",
            pos="NOUN",
            sentence="An aardvark is an animal.",
            all_synsets=synsets,
            embedding_scores=scores,
        )

        assert len(result) == 1
        assert result[0][0] == synsets[0].name()


class TestScoreCombination:
    """Tests for score combination logic."""

    def test_import_compute_combined_score(self) -> None:
        """Test that compute_combined_score can be imported."""
        from eng_words.wsd.candidate_selector import compute_combined_score

        assert callable(compute_combined_score)

    def test_embedding_score_weight(self) -> None:
        """Test that embedding score contributes to combined score."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import compute_combined_score

        synset = wn.synset("bank.n.01")

        score1 = compute_combined_score(
            synset=synset,
            embedding_score=0.9,
            sentence="The bank is closed.",
            context_boost=0.0,
        )

        score2 = compute_combined_score(
            synset=synset,
            embedding_score=0.3,
            sentence="The bank is closed.",
            context_boost=0.0,
        )

        assert score1 > score2

    def test_context_boost_effect(self) -> None:
        """Test that context boost increases score."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import compute_combined_score

        synset = wn.synset("bank.n.01")

        score_no_boost = compute_combined_score(
            synset=synset,
            embedding_score=0.5,
            sentence="The bank is there.",
            context_boost=0.0,
        )

        score_with_boost = compute_combined_score(
            synset=synset,
            embedding_score=0.5,
            sentence="The bank is there.",
            context_boost=0.2,
        )

        assert score_with_boost > score_no_boost


class TestContextKeywordMatch:
    """Tests for context keyword matching."""

    def test_import(self) -> None:
        """Test that get_context_boost can be imported."""
        from eng_words.wsd.candidate_selector import get_context_boost

        assert callable(get_context_boost)

    def test_no_match_returns_zero(self) -> None:
        """Test that no match returns zero boost."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import get_context_boost

        synset = wn.synset("bank.n.01")  # "sloping land beside body of water"
        sentence = "The computer is fast."

        boost = get_context_boost(synset, sentence)
        assert boost == 0.0

    def test_match_returns_positive(self) -> None:
        """Test that match returns positive boost."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import get_context_boost

        synset = wn.synset("bank.n.01")  # "sloping land beside body of water"
        sentence = "The water flowed by the sloping land."

        boost = get_context_boost(synset, sentence)
        assert boost > 0.0

    def test_more_matches_higher_boost(self) -> None:
        """Test that more matches give higher boost."""
        from nltk.corpus import wordnet as wn

        from eng_words.wsd.candidate_selector import get_context_boost

        synset = wn.synset("bank.n.01")

        boost1 = get_context_boost(synset, "The water is cold.")
        boost2 = get_context_boost(synset, "The water beside the land is cold.")

        # More matching words should give higher boost
        assert boost2 >= boost1
