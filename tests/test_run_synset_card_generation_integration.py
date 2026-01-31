"""Tests for integration of Stage 2.5 filtering in run_synset_card_generation.py"""





class TestStage25Integration:
    """Test that Stage 2.5 filtering is properly integrated."""

    def test_imports_available(self):
        """Test that required functions can be imported."""
        from eng_words.llm.smart_card_generator import (
            check_spoilers,
            mark_examples_by_length,
            select_examples_for_generation,
        )

        assert mark_examples_by_length is not None
        assert check_spoilers is not None
        assert select_examples_for_generation is not None

    def test_mark_examples_by_length_with_min_words(self):
        """Test that mark_examples_by_length filters out too short examples."""
        from eng_words.llm.smart_card_generator import mark_examples_by_length

        examples = [
            (1, "Short."),  # 1 word - too short
            (2, "This is a good example sentence."),  # 8 words - OK
            (3, "This is another good example sentence with more words."),  # 10 words - OK
            (4, " ".join(["word"] * 60)),  # 60 words - too long
        ]

        flags = mark_examples_by_length(examples, max_words=50, min_words=6)

        assert flags[1] is False  # Too short
        assert flags[2] is True  # OK
        assert flags[3] is True  # OK
        assert flags[4] is False  # Too long

    def test_select_examples_deduplicates(self):
        """Test that select_examples_for_generation deduplicates examples."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        # Create examples with duplicates
        examples = [
            (1, "This is a good example."),
            (2, "This is a good example."),  # Duplicate
            (3, "Another good example here."),
            (4, "Yet another example sentence."),
        ]

        length_flags = {1: True, 2: True, 3: True, 4: True}
        spoiler_flags = {1: False, 2: False, 3: False, 4: False}

        result = select_examples_for_generation(
            all_examples=examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        # Should deduplicate and take 2 unique examples
        selected_texts = [ex for _, ex in result["selected_from_book"]]
        assert len(selected_texts) == 2
        assert len(set(selected_texts)) == 2  # All unique
        assert result["generate_count"] == 1

    def test_selection_logic_3_plus_examples(self):
        """Test selection logic when 3+ valid examples available."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        examples = [
            (1, "First good example."),
            (2, "Second good example."),
            (3, "Third good example."),
            (4, "Fourth good example."),
        ]

        length_flags = {1: True, 2: True, 3: True, 4: True}
        spoiler_flags = {1: False, 2: False, 3: False, 4: False}

        result = select_examples_for_generation(
            all_examples=examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        assert len(result["selected_from_book"]) == 2
        assert result["generate_count"] == 1

    def test_selection_logic_1_2_examples(self):
        """Test selection logic when 1-2 valid examples available."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        examples = [
            (1, "Only one good example."),
        ]

        length_flags = {1: True}
        spoiler_flags = {1: False}

        result = select_examples_for_generation(
            all_examples=examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        assert len(result["selected_from_book"]) == 1
        assert result["generate_count"] == 2

    def test_selection_logic_0_examples(self):
        """Test selection logic when 0 valid examples available."""
        from eng_words.llm.smart_card_generator import select_examples_for_generation

        examples = [
            (1, "Too long example " * 20),  # Too long
            (2, "Spoiler example"),  # Has spoiler
        ]

        length_flags = {1: False, 2: True}
        spoiler_flags = {1: False, 2: True}

        result = select_examples_for_generation(
            all_examples=examples,
            length_flags=length_flags,
            spoiler_flags=spoiler_flags,
            target_count=3,
        )

        assert len(result["selected_from_book"]) == 0
        assert result["generate_count"] == 3
