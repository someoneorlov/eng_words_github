"""Tests for LLM prompts."""


class TestFormatCandidates:
    """Tests for format_candidates function."""

    def test_import_function(self):
        """Test that function can be imported."""
        from eng_words.llm.prompts import format_candidates

        assert callable(format_candidates)

    def test_formats_single_candidate(self):
        """Test formatting a single candidate."""
        from eng_words.llm.prompts import format_candidates

        candidates = [{"synset_id": "bank.n.01", "definition": "a financial institution"}]
        result = format_candidates(candidates)

        assert "**A.**" in result
        assert "a financial institution" in result

    def test_formats_multiple_candidates(self):
        """Test formatting multiple candidates."""
        from eng_words.llm.prompts import format_candidates

        candidates = [
            {"synset_id": "bank.n.01", "definition": "a financial institution"},
            {"synset_id": "bank.n.02", "definition": "sloping land beside water"},
            {"synset_id": "bank.n.03", "definition": "a supply of something"},
        ]
        result = format_candidates(candidates)

        assert "**A.**" in result
        assert "**B.**" in result
        assert "**C.**" in result
        assert "financial institution" in result
        assert "sloping land" in result
        assert "supply of something" in result

    def test_handles_missing_definition(self):
        """Test handling candidate without definition."""
        from eng_words.llm.prompts import format_candidates

        candidates = [{"synset_id": "bank.n.01"}]
        result = format_candidates(candidates)

        assert "No definition available" in result


class TestBuildEvaluationPrompt:
    """Tests for build_evaluation_prompt function."""

    def test_import_function(self):
        """Test that function can be imported."""
        from eng_words.llm.prompts import build_evaluation_prompt

        assert callable(build_evaluation_prompt)

    def test_includes_sentence(self):
        """Test that prompt includes the sentence."""
        from eng_words.llm.prompts import build_evaluation_prompt

        candidates = [{"synset_id": "run.v.01", "definition": "move fast using legs"}]
        result = build_evaluation_prompt(
            sentence="He runs every morning.",
            lemma="run",
            pos="VERB",
            candidates=candidates,
        )

        assert "He runs every morning." in result

    def test_includes_lemma_and_pos(self):
        """Test that prompt includes lemma and POS."""
        from eng_words.llm.prompts import build_evaluation_prompt

        candidates = [{"synset_id": "run.v.01", "definition": "move fast using legs"}]
        result = build_evaluation_prompt(
            sentence="He runs every morning.",
            lemma="run",
            pos="VERB",
            candidates=candidates,
        )

        assert "**run**" in result
        assert "VERB" in result

    def test_includes_all_candidates(self):
        """Test that prompt includes all candidates."""
        from eng_words.llm.prompts import build_evaluation_prompt

        candidates = [
            {"synset_id": "bank.n.01", "definition": "financial institution"},
            {"synset_id": "bank.n.02", "definition": "land beside water"},
        ]
        result = build_evaluation_prompt(
            sentence="I went to the bank.",
            lemma="bank",
            pos="NOUN",
            candidates=candidates,
        )

        assert "**A.**" in result
        assert "**B.**" in result
        assert "financial institution" in result
        assert "land beside water" in result

    def test_includes_json_format_instructions(self):
        """Test that prompt includes JSON format instructions."""
        from eng_words.llm.prompts import build_evaluation_prompt

        candidates = [{"synset_id": "run.v.01", "definition": "move fast"}]
        result = build_evaluation_prompt(
            sentence="He runs.",
            lemma="run",
            pos="VERB",
            candidates=candidates,
        )

        assert '"choice"' in result
        assert '"reasoning"' in result

    def test_includes_prompt_version(self):
        """Test that prompt includes version for reproducibility."""
        from eng_words.llm.prompts import build_evaluation_prompt

        candidates = [{"synset_id": "run.v.01", "definition": "move fast"}]
        result = build_evaluation_prompt(
            sentence="He runs.",
            lemma="run",
            pos="VERB",
            candidates=candidates,
        )

        assert "Prompt Version:" in result


class TestGetCandidateIndex:
    """Tests for get_candidate_index function."""

    def test_import_function(self):
        """Test that function can be imported."""
        from eng_words.llm.prompts import get_candidate_index

        assert callable(get_candidate_index)

    def test_returns_correct_index_for_a(self):
        """Test returns 0 for choice A."""
        from eng_words.llm.prompts import get_candidate_index

        candidates = [{"synset_id": "a"}, {"synset_id": "b"}]
        result = get_candidate_index("A", candidates)
        assert result == 0

    def test_returns_correct_index_for_b(self):
        """Test returns 1 for choice B."""
        from eng_words.llm.prompts import get_candidate_index

        candidates = [{"synset_id": "a"}, {"synset_id": "b"}]
        result = get_candidate_index("B", candidates)
        assert result == 1

    def test_handles_lowercase(self):
        """Test handles lowercase letters."""
        from eng_words.llm.prompts import get_candidate_index

        candidates = [{"synset_id": "a"}, {"synset_id": "b"}]
        result = get_candidate_index("b", candidates)
        assert result == 1

    def test_returns_none_for_uncertain(self):
        """Test returns None for uncertain."""
        from eng_words.llm.prompts import get_candidate_index

        candidates = [{"synset_id": "a"}]
        result = get_candidate_index("uncertain", candidates)
        assert result is None

    def test_returns_none_for_out_of_range(self):
        """Test returns None for out-of-range letter."""
        from eng_words.llm.prompts import get_candidate_index

        candidates = [{"synset_id": "a"}]  # Only 1 candidate
        result = get_candidate_index("C", candidates)  # C would be index 2
        assert result is None

    def test_returns_none_for_invalid_choice(self):
        """Test returns None for invalid choice."""
        from eng_words.llm.prompts import get_candidate_index

        candidates = [{"synset_id": "a"}]
        result = get_candidate_index("invalid", candidates)
        assert result is None


class TestFormatSenseForPrompt:
    """Tests for format_sense_for_prompt function."""

    def test_import_function(self):
        """Function can be imported."""
        from eng_words.llm.prompts import format_sense_for_prompt

        assert callable(format_sense_for_prompt)

    def test_includes_basic_info(self):
        """Includes lemma, pos, synset_id."""
        from eng_words.llm.prompts import format_sense_for_prompt

        sense = {
            "synset_id": "bank.n.01",
            "lemma": "bank",
            "pos": "NOUN",
            "supersense": "noun.group",
            "definition": "a financial institution",
            "book_name": "Test Book",
            "book_examples": [],
        }

        result = format_sense_for_prompt(sense)

        assert "bank" in result
        assert "NOUN" in result
        assert "bank.n.01" in result
        assert "noun.group" in result

    def test_includes_book_examples(self):
        """Includes book examples when present."""
        from eng_words.llm.prompts import format_sense_for_prompt

        sense = {
            "synset_id": "bank.n.01",
            "lemma": "bank",
            "pos": "NOUN",
            "supersense": "noun.group",
            "definition": "a financial institution",
            "book_name": "Test Book",
            "book_examples": ["He went to the bank.", "The bank was closed."],
        }

        result = format_sense_for_prompt(sense)

        assert "He went to the bank." in result
        assert "The bank was closed." in result

    def test_handles_no_examples(self):
        """Handles sense with no book examples."""
        from eng_words.llm.prompts import format_sense_for_prompt

        sense = {
            "synset_id": "bank.n.01",
            "lemma": "bank",
            "pos": "NOUN",
            "supersense": "noun.group",
            "definition": "a financial institution",
            "book_name": "Test Book",
            "book_examples": [],
        }

        result = format_sense_for_prompt(sense)

        assert "none available" in result


class TestBuildCardGenerationPrompt:
    """Tests for build_card_generation_prompt function."""

    def test_import_function(self):
        """Function can be imported."""
        from eng_words.llm.prompts import build_card_generation_prompt

        assert callable(build_card_generation_prompt)

    def test_includes_all_senses(self):
        """Includes all senses in the prompt."""
        from eng_words.llm.prompts import build_card_generation_prompt

        senses = [
            {
                "synset_id": "bank.n.01",
                "lemma": "bank",
                "pos": "NOUN",
                "supersense": "noun.group",
                "definition": "a financial institution",
                "book_name": "Test Book",
                "book_examples": ["He went to the bank."],
            },
            {
                "synset_id": "run.v.01",
                "lemma": "run",
                "pos": "VERB",
                "supersense": "verb.motion",
                "definition": "move fast using legs",
                "book_name": "Test Book",
                "book_examples": ["He runs fast."],
            },
        ]

        result = build_card_generation_prompt(senses)

        assert "bank.n.01" in result
        assert "run.v.01" in result
        assert "He went to the bank." in result
        assert "He runs fast." in result

    def test_includes_prompt_version(self):
        """Includes prompt version for reproducibility."""
        from eng_words.llm.prompts import build_card_generation_prompt

        senses = [
            {
                "synset_id": "bank.n.01",
                "lemma": "bank",
                "pos": "NOUN",
                "supersense": "noun.group",
                "definition": "a financial institution",
                "book_name": "Test Book",
                "book_examples": [],
            },
        ]

        result = build_card_generation_prompt(senses)

        assert "Prompt Version:" in result

    def test_includes_spoiler_risk_instructions(self):
        """Includes spoiler_risk rating instructions."""
        from eng_words.llm.prompts import build_card_generation_prompt

        senses = [
            {
                "synset_id": "bank.n.01",
                "lemma": "bank",
                "pos": "NOUN",
                "supersense": "noun.group",
                "definition": "a financial institution",
                "book_name": "Test Book",
                "book_examples": [],
            },
        ]

        result = build_card_generation_prompt(senses)

        assert "spoiler_risk" in result
        assert '"none"' in result or "'none'" in result

    def test_includes_json_example(self):
        """Includes JSON response example."""
        from eng_words.llm.prompts import build_card_generation_prompt

        senses = [
            {
                "synset_id": "bank.n.01",
                "lemma": "bank",
                "pos": "NOUN",
                "supersense": "noun.group",
                "definition": "a financial institution",
                "book_name": "Test Book",
                "book_examples": [],
            },
        ]

        result = build_card_generation_prompt(senses)

        assert "definition_simple" in result
        assert "translation_ru" in result
        assert "book_examples_selected" in result
        assert "generic_examples" in result
