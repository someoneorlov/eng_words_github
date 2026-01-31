"""Tests for WSD Gold Dataset collection module.

Following TDD: tests are written before implementation.
"""

import pandas as pd


class TestExtractExamplesFromTokens:
    """Tests for extract_examples_from_tokens function."""

    def test_returns_list_of_gold_examples(self):
        """Function returns a list of GoldExample objects."""
        from eng_words.wsd_gold.collect import extract_examples_from_tokens
        from eng_words.wsd_gold.models import GoldExample

        tokens_df = pd.DataFrame(
            {
                "sentence_id": [0, 0, 0, 0, 0],
                "position": [0, 1, 2, 3, 4],
                "surface": ["The", "river", "bank", "was", "muddy"],
                "lemma": ["the", "river", "bank", "be", "muddy"],
                "pos": ["DET", "NOUN", "NOUN", "AUX", "ADJ"],
                "is_stop": [True, False, False, True, False],
                "is_alpha": [True, True, True, True, True],
                "whitespace": [" ", " ", " ", " ", ""],
                "book": ["test_book"] * 5,
                "synset_id": [None, "river.n.01", "bank.n.01", None, "muddy.a.01"],
                "supersense": [
                    "unknown",
                    "noun.object",
                    "noun.object",
                    "unknown",
                    "adj.all",
                ],
                "sense_confidence": [0.0, 0.85, 0.72, 0.0, 0.90],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [0],
                "sentence": ["The river bank was muddy."],
            }
        )

        examples = extract_examples_from_tokens(tokens_df, sentences_df)

        assert isinstance(examples, list)
        assert len(examples) > 0
        assert all(isinstance(e, GoldExample) for e in examples)

    def test_extracts_only_annotated_tokens(self):
        """Only tokens with synset_id are extracted."""
        from eng_words.wsd_gold.collect import extract_examples_from_tokens

        tokens_df = pd.DataFrame(
            {
                "sentence_id": [0, 0, 0],
                "position": [0, 1, 2],
                "surface": ["The", "bank", "collapsed"],
                "lemma": ["the", "bank", "collapse"],
                "pos": ["DET", "NOUN", "VERB"],
                "is_stop": [True, False, False],
                "is_alpha": [True, True, True],
                "whitespace": [" ", " ", ""],
                "book": ["test_book"] * 3,
                "synset_id": [None, "bank.n.01", "collapse.v.01"],
                "supersense": ["unknown", "noun.object", "verb.change"],
                "sense_confidence": [0.0, 0.75, 0.80],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [0],
                "sentence": ["The bank collapsed."],
            }
        )

        examples = extract_examples_from_tokens(tokens_df, sentences_df)

        # Only bank and collapsed have synset_id
        assert len(examples) == 2
        lemmas = [e.target.lemma for e in examples]
        assert "bank" in lemmas
        assert "collapse" in lemmas
        assert "the" not in lemmas

    def test_example_has_correct_structure(self):
        """Extracted example has all required fields."""
        from eng_words.wsd_gold.collect import extract_examples_from_tokens

        tokens_df = pd.DataFrame(
            {
                "sentence_id": [0, 0, 0],
                "position": [0, 1, 2],
                "surface": ["Big", "bank", "here"],
                "lemma": ["big", "bank", "here"],
                "pos": ["ADJ", "NOUN", "ADV"],
                "is_stop": [False, False, False],
                "is_alpha": [True, True, True],
                "whitespace": [" ", " ", ""],
                "book": ["test_book"] * 3,
                "synset_id": [None, "bank.n.02", None],
                "supersense": ["unknown", "noun.artifact", "unknown"],
                "sense_confidence": [0.0, 0.88, 0.0],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [0],
                "sentence": ["Big bank here."],
            }
        )

        examples = extract_examples_from_tokens(tokens_df, sentences_df)

        assert len(examples) == 1
        example = examples[0]

        # Check example_id format
        assert "test_book" in example.example_id
        assert "sent:0" in example.example_id

        # Check target
        assert example.target.surface == "bank"
        assert example.target.lemma == "bank"
        assert example.target.pos == "NOUN"

        # Check context
        assert example.context_window == "Big bank here."
        assert "Big" in example.text_left or example.text_left == "Big "
        assert "here" in example.text_right or example.text_right == " here."

        # Check candidates are loaded
        assert len(example.candidates) > 0

        # Check metadata
        assert example.metadata.baseline_top1 == "bank.n.02"
        assert example.metadata.baseline_margin >= 0

    def test_filters_by_min_sense_count(self):
        """Can filter examples by minimum WordNet sense count."""
        from eng_words.wsd_gold.collect import extract_examples_from_tokens

        tokens_df = pd.DataFrame(
            {
                "sentence_id": [0, 0],
                "position": [0, 1],
                "surface": ["bank", "runs"],
                "lemma": ["bank", "run"],
                "pos": ["NOUN", "VERB"],
                "is_stop": [False, False],
                "is_alpha": [True, True],
                "whitespace": [" ", ""],
                "book": ["test_book"] * 2,
                "synset_id": ["bank.n.01", "run.v.01"],
                "supersense": ["noun.object", "verb.motion"],
                "sense_confidence": [0.75, 0.80],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [0],
                "sentence": ["bank runs."],
            }
        )

        # bank has many senses, let's filter for sense_count >= 5
        examples = extract_examples_from_tokens(tokens_df, sentences_df, min_sense_count=5)

        # Both bank (noun) and run (verb) have many senses
        for ex in examples:
            assert ex.metadata.wn_sense_count >= 5

    def test_filters_by_pos(self):
        """Can filter examples by POS."""
        from eng_words.wsd_gold.collect import extract_examples_from_tokens

        tokens_df = pd.DataFrame(
            {
                "sentence_id": [0, 0, 0],
                "position": [0, 1, 2],
                "surface": ["quickly", "ran", "home"],
                "lemma": ["quickly", "run", "home"],
                "pos": ["ADV", "VERB", "NOUN"],
                "is_stop": [False, False, False],
                "is_alpha": [True, True, True],
                "whitespace": [" ", " ", ""],
                "book": ["test_book"] * 3,
                "synset_id": ["quickly.r.01", "run.v.01", "home.n.01"],
                "supersense": ["adv.all", "verb.motion", "noun.location"],
                "sense_confidence": [0.85, 0.80, 0.75],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [0],
                "sentence": ["quickly ran home."],
            }
        )

        examples = extract_examples_from_tokens(
            tokens_df, sentences_df, pos_filter=["NOUN", "VERB"]
        )

        poses = [e.target.pos for e in examples]
        assert "NOUN" in poses
        assert "VERB" in poses
        assert "ADV" not in poses

    def test_source_id_from_book(self):
        """source_id is taken from book column."""
        from eng_words.wsd_gold.collect import extract_examples_from_tokens

        tokens_df = pd.DataFrame(
            {
                "sentence_id": [0],
                "position": [0],
                "surface": ["bank"],
                "lemma": ["bank"],
                "pos": ["NOUN"],
                "is_stop": [False],
                "is_alpha": [True],
                "whitespace": [""],
                "book": ["my_great_book"],
                "synset_id": ["bank.n.01"],
                "supersense": ["noun.object"],
                "sense_confidence": [0.80],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [0],
                "sentence": ["bank"],
            }
        )

        examples = extract_examples_from_tokens(tokens_df, sentences_df)

        assert len(examples) == 1
        assert examples[0].source_id == "my_great_book"

    def test_empty_tokens_returns_empty_list(self):
        """Empty tokens_df returns empty list."""
        from eng_words.wsd_gold.collect import extract_examples_from_tokens

        tokens_df = pd.DataFrame(
            columns=[
                "sentence_id",
                "position",
                "surface",
                "lemma",
                "pos",
                "is_stop",
                "is_alpha",
                "whitespace",
                "book",
                "synset_id",
                "supersense",
                "sense_confidence",
            ]
        )
        sentences_df = pd.DataFrame(columns=["sentence_id", "sentence"])

        examples = extract_examples_from_tokens(tokens_df, sentences_df)

        assert examples == []


class TestGetCandidatesForLemma:
    """Tests for get_candidates_for_lemma function."""

    def test_returns_list_of_candidates(self):
        """Function returns a list of Candidate objects."""
        from eng_words.wsd_gold.collect import get_candidates_for_lemma
        from eng_words.wsd_gold.models import Candidate

        candidates = get_candidates_for_lemma("bank", "NOUN")

        assert isinstance(candidates, list)
        assert len(candidates) > 0
        assert all(isinstance(c, Candidate) for c in candidates)

    def test_candidate_has_synset_id(self):
        """Each candidate has a valid synset_id."""
        from eng_words.wsd_gold.collect import get_candidates_for_lemma

        candidates = get_candidates_for_lemma("run", "VERB")

        for c in candidates:
            assert c.synset_id.endswith(".v.01") or ".v." in c.synset_id
            assert len(c.synset_id) > 0

    def test_candidate_has_gloss(self):
        """Each candidate has a gloss (definition)."""
        from eng_words.wsd_gold.collect import get_candidates_for_lemma

        candidates = get_candidates_for_lemma("bank", "NOUN")

        for c in candidates:
            assert len(c.gloss) > 0

    def test_candidate_has_examples(self):
        """Candidates can have examples."""
        from eng_words.wsd_gold.collect import get_candidates_for_lemma

        candidates = get_candidates_for_lemma("bank", "NOUN")

        # At least some candidates should have examples
        has_examples = any(len(c.examples) > 0 for c in candidates)
        assert has_examples

    def test_unknown_word_returns_empty(self):
        """Unknown word returns empty list."""
        from eng_words.wsd_gold.collect import get_candidates_for_lemma

        candidates = get_candidates_for_lemma("xyznonexistent", "NOUN")

        assert candidates == []

    def test_pos_filter_works(self):
        """POS filter returns only matching synsets."""
        from eng_words.wsd_gold.collect import get_candidates_for_lemma

        noun_candidates = get_candidates_for_lemma("bank", "NOUN")
        verb_candidates = get_candidates_for_lemma("bank", "VERB")

        # All noun candidates should have .n. in synset_id
        for c in noun_candidates:
            assert ".n." in c.synset_id

        # All verb candidates should have .v. in synset_id
        for c in verb_candidates:
            assert ".v." in c.synset_id


class TestCalculateCharSpan:
    """Tests for calculate_char_span function."""

    def test_finds_word_position(self):
        """Correctly calculates character span for a word."""
        from eng_words.wsd_gold.collect import calculate_char_span

        sentence = "The river bank was muddy."
        span = calculate_char_span(sentence, "bank", position=2)

        # "bank" starts at index 10
        assert span == (10, 14)
        assert sentence[span[0] : span[1]] == "bank"

    def test_handles_first_word(self):
        """Handles word at the beginning."""
        from eng_words.wsd_gold.collect import calculate_char_span

        sentence = "Bank was open."
        span = calculate_char_span(sentence, "Bank", position=0)

        assert span == (0, 4)
        assert sentence[span[0] : span[1]] == "Bank"

    def test_handles_last_word(self):
        """Handles word at the end."""
        from eng_words.wsd_gold.collect import calculate_char_span

        sentence = "Visit the bank."
        span = calculate_char_span(sentence, "bank", position=2)

        assert span == (10, 14)
        assert sentence[span[0] : span[1]] == "bank"

    def test_handles_duplicate_words(self):
        """Correctly handles sentences with duplicate words."""
        from eng_words.wsd_gold.collect import calculate_char_span

        sentence = "bank to bank transfer"
        # First "bank" at position 0
        span1 = calculate_char_span(sentence, "bank", position=0)
        # Second "bank" at position 2
        span2 = calculate_char_span(sentence, "bank", position=2)

        assert span1 == (0, 4)
        assert span2 == (8, 12)
        assert sentence[span1[0] : span1[1]] == "bank"
        assert sentence[span2[0] : span2[1]] == "bank"


class TestBuildExampleId:
    """Tests for build_example_id function."""

    def test_builds_correct_format(self):
        """Builds example_id in correct format."""
        from eng_words.wsd_gold.collect import build_example_id

        example_id = build_example_id(source_id="my_book", sentence_id=42, token_position=7)

        assert example_id == "book:my_book|sent:42|tok:7"

    def test_handles_special_characters(self):
        """Handles special characters in source_id."""
        from eng_words.wsd_gold.collect import build_example_id

        example_id = build_example_id(
            source_id="an_american_tragedy", sentence_id=100, token_position=15
        )

        assert "an_american_tragedy" in example_id
        assert "sent:100" in example_id
        assert "tok:15" in example_id


class TestAssignBuckets:
    """Tests for assign_buckets function."""

    def test_assigns_buckets_from_metadata(self):
        """Assigns buckets based on source metadata."""
        from eng_words.wsd_gold.collect import assign_buckets

        source_metadata = {
            "test_book": {
                "source_bucket": "classic_fiction",
                "year_bucket": "pre_1950",
                "genre_bucket": "fiction",
            }
        }

        buckets = assign_buckets("test_book", source_metadata)

        assert buckets["source_bucket"] == "classic_fiction"
        assert buckets["year_bucket"] == "pre_1950"
        assert buckets["genre_bucket"] == "fiction"

    def test_default_buckets_for_unknown_source(self):
        """Uses default buckets for unknown source."""
        from eng_words.wsd_gold.collect import assign_buckets

        source_metadata = {}

        buckets = assign_buckets("unknown_book", source_metadata)

        assert buckets["source_bucket"] == "unknown"
        assert buckets["year_bucket"] == "unknown"
        assert buckets["genre_bucket"] == "unknown"

    def test_partial_metadata(self):
        """Handles partial metadata gracefully."""
        from eng_words.wsd_gold.collect import assign_buckets

        source_metadata = {
            "partial_book": {
                "source_bucket": "modern_nonfiction",
                # Missing year_bucket and genre_bucket
            }
        }

        buckets = assign_buckets("partial_book", source_metadata)

        assert buckets["source_bucket"] == "modern_nonfiction"
        assert buckets["year_bucket"] == "unknown"
        assert buckets["genre_bucket"] == "unknown"
