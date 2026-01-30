"""Tests for LLM CardGenerator module.

TDD approach: tests first, implementation second.
"""

import tempfile
from pathlib import Path

import pandas as pd


class TestCollectBookExamples:
    """Tests for collect_book_examples function."""

    def test_import_function(self):
        """Function can be imported."""
        from eng_words.llm.card_generator import collect_book_examples

        assert collect_book_examples is not None

    def test_returns_dict(self):
        """Returns dict mapping synset_id to list of examples."""
        from eng_words.llm.card_generator import collect_book_examples

        sense_tokens_df = pd.DataFrame(
            {
                "sentence_id": [1, 2, 3],
                "synset_id": ["bank.n.01", "bank.n.01", "run.v.01"],
                "lemma": ["bank", "bank", "run"],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1, 2, 3],
                "text": ["He went to the bank.", "The bank was closed.", "He runs fast."],
            }
        )

        result = collect_book_examples(sense_tokens_df, sentences_df)

        assert isinstance(result, dict)
        assert "bank.n.01" in result
        assert "run.v.01" in result

    def test_groups_by_synset(self):
        """Groups examples by synset_id."""
        from eng_words.llm.card_generator import collect_book_examples

        sense_tokens_df = pd.DataFrame(
            {
                "sentence_id": [1, 2, 3, 4],
                "synset_id": ["bank.n.01", "bank.n.01", "bank.n.02", "bank.n.01"],
                "lemma": ["bank", "bank", "bank", "bank"],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1, 2, 3, 4],
                "text": ["Sentence 1", "Sentence 2", "Sentence 3", "Sentence 4"],
            }
        )

        result = collect_book_examples(sense_tokens_df, sentences_df)

        assert len(result["bank.n.01"]) == 3
        assert len(result["bank.n.02"]) == 1

    def test_deduplicates_sentences(self):
        """Removes duplicate sentences for same synset."""
        from eng_words.llm.card_generator import collect_book_examples

        sense_tokens_df = pd.DataFrame(
            {
                "sentence_id": [1, 1, 1],  # Same sentence, multiple tokens
                "synset_id": ["bank.n.01", "bank.n.01", "bank.n.01"],
                "lemma": ["bank", "bank", "bank"],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1],
                "text": ["He went to the bank."],
            }
        )

        result = collect_book_examples(sense_tokens_df, sentences_df)

        assert len(result["bank.n.01"]) == 1

    def test_skips_null_synsets(self):
        """Skips tokens without synset_id."""
        from eng_words.llm.card_generator import collect_book_examples

        sense_tokens_df = pd.DataFrame(
            {
                "sentence_id": [1, 2, 3],
                "synset_id": ["bank.n.01", None, "run.v.01"],
                "lemma": ["bank", "the", "run"],
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": [1, 2, 3],
                "text": ["Sentence 1", "Sentence 2", "Sentence 3"],
            }
        )

        result = collect_book_examples(sense_tokens_df, sentences_df)

        assert len(result) == 2  # Only bank.n.01 and run.v.01

    def test_respects_max_examples(self):
        """Limits examples per synset to MAX_EXAMPLES_PER_SENSE."""
        from eng_words.constants.llm_config import MAX_EXAMPLES_PER_SENSE
        from eng_words.llm.card_generator import collect_book_examples

        # Create many examples for one synset
        n_examples = MAX_EXAMPLES_PER_SENSE + 10
        sense_tokens_df = pd.DataFrame(
            {
                "sentence_id": list(range(n_examples)),
                "synset_id": ["bank.n.01"] * n_examples,
                "lemma": ["bank"] * n_examples,
            }
        )
        sentences_df = pd.DataFrame(
            {
                "sentence_id": list(range(n_examples)),
                "text": [f"Sentence {i}" for i in range(n_examples)],
            }
        )

        result = collect_book_examples(sense_tokens_df, sentences_df)

        assert len(result["bank.n.01"]) <= MAX_EXAMPLES_PER_SENSE


class TestPrepareCardBatch:
    """Tests for prepare_card_batch function."""

    def test_import_function(self):
        """Function can be imported."""
        from eng_words.llm.card_generator import prepare_card_batch

        assert prepare_card_batch is not None

    def test_returns_list_of_dicts(self):
        """Returns list of sense info dicts ready for prompt."""
        from eng_words.llm.card_generator import prepare_card_batch

        synset_infos = [
            {
                "synset_id": "bank.n.01",
                "lemma": "bank",
                "pos": "NOUN",
                "supersense": "noun.group",
                "definition": "a financial institution",
            }
        ]
        book_examples = {"bank.n.01": ["He went to the bank."]}

        result = prepare_card_batch(synset_infos, book_examples, book_name="Test Book")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["synset_id"] == "bank.n.01"
        assert "book_examples" in result[0]

    def test_truncates_long_examples(self):
        """Truncates examples longer than MAX_EXAMPLE_LENGTH."""
        from eng_words.constants.llm_config import MAX_EXAMPLE_LENGTH
        from eng_words.llm.card_generator import prepare_card_batch

        synset_infos = [
            {
                "synset_id": "bank.n.01",
                "lemma": "bank",
                "pos": "NOUN",
                "supersense": "noun.group",
                "definition": "a financial institution",
            }
        ]
        long_example = "A" * (MAX_EXAMPLE_LENGTH + 100)
        book_examples = {"bank.n.01": [long_example]}

        result = prepare_card_batch(synset_infos, book_examples, book_name="Test Book")

        assert len(result[0]["book_examples"][0]) <= MAX_EXAMPLE_LENGTH + 3  # +3 for "..."


class TestCardGenerator:
    """Tests for CardGenerator class."""

    def test_import_class(self):
        """CardGenerator can be imported."""
        from eng_words.llm.card_generator import CardGenerator

        assert CardGenerator is not None

    def test_create_generator(self):
        """CardGenerator can be created with provider and cache."""
        from unittest.mock import MagicMock

        from eng_words.llm.cache import SenseCache
        from eng_words.llm.card_generator import CardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))
            provider = MagicMock()

            generator = CardGenerator(provider=provider, cache=cache)

            assert generator is not None

    def test_generator_has_generate_batch_method(self):
        """CardGenerator has generate_batch method."""
        from unittest.mock import MagicMock

        from eng_words.llm.cache import SenseCache
        from eng_words.llm.card_generator import CardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))
            provider = MagicMock()

            generator = CardGenerator(provider=provider, cache=cache)

            assert hasattr(generator, "generate_batch")
            assert callable(generator.generate_batch)

    def test_generator_has_generate_all_method(self):
        """CardGenerator has generate_all method for batching."""
        from unittest.mock import MagicMock

        from eng_words.llm.cache import SenseCache
        from eng_words.llm.card_generator import CardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))
            provider = MagicMock()

            generator = CardGenerator(provider=provider, cache=cache)

            assert hasattr(generator, "generate_all")
            assert callable(generator.generate_all)


class TestCardGeneratorLLMIntegration:
    """Tests for CardGenerator LLM integration."""

    def test_generate_batch_calls_llm(self):
        """generate_batch calls LLM for uncached synsets."""
        import json
        from unittest.mock import MagicMock

        from eng_words.llm.base import LLMResponse
        from eng_words.llm.cache import SenseCache
        from eng_words.llm.card_generator import CardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))
            provider = MagicMock()

            # Mock complete to return LLMResponse with JSON content
            response_data = [
                {
                    "synset_id": "bank.n.01",
                    "definition_simple": "a place to keep money",
                    "translation_ru": "банк",
                    "book_examples_selected": [
                        {"text": "He went to the bank.", "spoiler_risk": "none"}
                    ],
                    "generic_examples": ["I deposit money at the bank."],
                }
            ]
            provider.complete.return_value = LLMResponse(
                content=json.dumps(response_data),
                model="gpt-4o-mini",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            provider.model = "gpt-4o-mini"

            generator = CardGenerator(provider=provider, cache=cache)

            synset_infos = [
                {
                    "synset_id": "bank.n.01",
                    "lemma": "bank",
                    "pos": "NOUN",
                    "supersense": "noun.group",
                    "definition": "a financial institution",
                }
            ]
            book_examples = {"bank.n.01": ["He went to the bank."]}

            cards = generator.generate_batch(synset_infos, book_examples, "Test Book")

            assert provider.complete.called
            assert len(cards) == 1
            assert cards[0].definition_simple == "a place to keep money"
            assert cards[0].translation_ru == "банк"

    def test_generate_batch_filters_spoilers(self):
        """generate_batch filters out examples with spoiler_risk != none."""
        import json
        from unittest.mock import MagicMock

        from eng_words.llm.base import LLMResponse
        from eng_words.llm.cache import SenseCache
        from eng_words.llm.card_generator import CardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))
            provider = MagicMock()

            response_data = [
                {
                    "synset_id": "bank.n.01",
                    "definition_simple": "a place to keep money",
                    "translation_ru": "банк",
                    "book_examples_selected": [
                        {"text": "He went to the bank.", "spoiler_risk": "none"},
                        {"text": "The bank was robbed!", "spoiler_risk": "high"},
                        {"text": "The bank closed early.", "spoiler_risk": "medium"},
                    ],
                    "generic_examples": [],
                }
            ]
            provider.complete.return_value = LLMResponse(
                content=json.dumps(response_data),
                model="gpt-4o-mini",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            provider.model = "gpt-4o-mini"

            generator = CardGenerator(provider=provider, cache=cache)

            synset_infos = [
                {
                    "synset_id": "bank.n.01",
                    "lemma": "bank",
                    "pos": "NOUN",
                    "supersense": "noun.group",
                    "definition": "a financial institution",
                }
            ]

            cards = generator.generate_batch(
                synset_infos,
                {"bank.n.01": ["He went to the bank.", "The bank was robbed!"]},
                "Test Book",
            )

            # Only "none" spoiler_risk should be included
            assert len(cards[0].book_examples.get("Test Book", [])) == 1
            assert cards[0].book_examples["Test Book"][0] == "He went to the bank."

    def test_generate_batch_uses_cache_for_existing(self):
        """generate_batch skips LLM for cached synsets."""
        from datetime import datetime
        from unittest.mock import MagicMock

        from eng_words.llm.cache import SenseCache, SenseCard
        from eng_words.llm.card_generator import CardGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))

            # Pre-cache a card
            cached_card = SenseCard(
                synset_id="bank.n.01",
                lemma="bank",
                pos="NOUN",
                supersense="noun.group",
                definition_simple="cached definition",
                translation_ru="кэшированный перевод",
                generic_examples=[],
                book_examples={},
                generated_at=datetime.now(),
                model="cached-model",
                prompt_version="v1.0",
            )
            cache.store(cached_card)

            provider = MagicMock()
            generator = CardGenerator(provider=provider, cache=cache)

            synset_infos = [
                {
                    "synset_id": "bank.n.01",
                    "lemma": "bank",
                    "pos": "NOUN",
                    "supersense": "noun.group",
                    "definition": "a financial institution",
                }
            ]

            cards = generator.generate_batch(synset_infos, {}, "Test Book")

            # LLM should NOT be called for cached synset
            assert not provider.complete_json.called
            assert len(cards) == 1
            assert cards[0].definition_simple == "cached definition"


class TestExportToAnki:
    """Tests for export_to_anki function."""

    def test_import_function(self):
        """Function can be imported."""
        from eng_words.llm.card_generator import export_to_anki

        assert callable(export_to_anki)

    def test_exports_cards_to_csv(self):
        """Exports cards to tab-separated CSV."""
        from datetime import datetime

        from eng_words.llm.cache import SenseCard
        from eng_words.llm.card_generator import export_to_anki

        with tempfile.TemporaryDirectory() as tmpdir:
            cards = [
                SenseCard(
                    synset_id="bank.n.01",
                    lemma="bank",
                    pos="NOUN",
                    supersense="noun.group",
                    definition_simple="a place to keep money",
                    translation_ru="банк",
                    generic_examples=["I deposit money at the bank."],
                    book_examples={"Test Book": ["He went to the bank."]},
                    generated_at=datetime.now(),
                    model="test",
                    prompt_version="v1.0",
                )
            ]

            out_path = Path(tmpdir) / "test_anki.csv"
            exported = export_to_anki(cards, out_path, book_name="Test Book")

            assert exported == 1
            assert out_path.exists()

            content = out_path.read_text(encoding="utf-8")
            assert "bank" in content
            assert "банк" in content
            assert "He went to the bank" in content

    def test_skips_cards_without_examples(self):
        """Skips cards with no examples."""
        from datetime import datetime

        from eng_words.llm.cache import SenseCard
        from eng_words.llm.card_generator import export_to_anki

        with tempfile.TemporaryDirectory() as tmpdir:
            cards = [
                SenseCard(
                    synset_id="bank.n.01",
                    lemma="bank",
                    pos="NOUN",
                    supersense="noun.group",
                    definition_simple="a place to keep money",
                    translation_ru="банк",
                    generic_examples=[],  # No examples
                    book_examples={},
                    generated_at=datetime.now(),
                    model="test",
                    prompt_version="v1.0",
                )
            ]

            out_path = Path(tmpdir) / "test_anki.csv"
            exported = export_to_anki(cards, out_path)

            assert exported == 0

    def test_uses_generic_example_if_no_book_example(self):
        """Uses generic example on front if no book examples."""
        from datetime import datetime

        from eng_words.llm.cache import SenseCard
        from eng_words.llm.card_generator import export_to_anki

        with tempfile.TemporaryDirectory() as tmpdir:
            cards = [
                SenseCard(
                    synset_id="bank.n.01",
                    lemma="bank",
                    pos="NOUN",
                    supersense="noun.group",
                    definition_simple="a place to keep money",
                    translation_ru="банк",
                    generic_examples=["Generic example here."],
                    book_examples={},  # No book examples
                    generated_at=datetime.now(),
                    model="test",
                    prompt_version="v1.0",
                )
            ]

            out_path = Path(tmpdir) / "test_anki.csv"
            exported = export_to_anki(cards, out_path)

            assert exported == 1
            content = out_path.read_text(encoding="utf-8")
            assert "Generic example here" in content


class TestContainsLemma:
    """Tests for _contains_lemma function."""

    def test_import_function(self):
        """Function can be imported."""
        from eng_words.llm.card_generator import _contains_lemma

        assert _contains_lemma is not None

    def test_finds_base_form(self):
        """Finds base form of word."""
        from eng_words.llm.card_generator import _contains_lemma

        assert _contains_lemma("The bank is closed", "bank")
        assert _contains_lemma("He runs fast", "run")

    def test_finds_regular_inflections(self):
        """Finds regular inflections (s, ed, ing)."""
        from eng_words.llm.card_generator import _contains_lemma

        assert _contains_lemma("He runs fast", "run")
        assert _contains_lemma("He ran yesterday", "run")  # irregular
        assert _contains_lemma("He is running", "run")
        assert _contains_lemma("He walked home", "walk")

    def test_finds_irregular_verbs(self):
        """Finds irregular verb forms."""
        from eng_words.llm.card_generator import _contains_lemma

        assert _contains_lemma("He took the dollar", "take")
        assert _contains_lemma("The car sped by", "speed")
        assert _contains_lemma("He drew a picture", "draw")
        assert _contains_lemma("It was heralded", "herald")

    def test_returns_false_when_not_found(self):
        """Returns False when lemma not found."""
        from eng_words.llm.card_generator import _contains_lemma

        assert not _contains_lemma("The bank is closed", "run")
        assert not _contains_lemma("He runs fast", "walk")


class TestTruncateAroundLemma:
    """Tests for _truncate_around_lemma function."""

    def test_import_function(self):
        """Function can be imported."""
        from eng_words.llm.card_generator import _truncate_around_lemma

        assert _truncate_around_lemma is not None

    def test_keeps_lemma_visible(self):
        """Truncated text contains the lemma."""
        from eng_words.llm.card_generator import _truncate_around_lemma

        long_text = (
            "This is a very long sentence that contains the word bank in the middle somewhere."
        )
        result = _truncate_around_lemma(long_text, "bank", 50)

        assert "bank" in result.lower()
        assert len(result) <= 53  # 50 + 3 for "..."

    def test_centers_around_lemma(self):
        """Truncates centered around lemma position."""
        from eng_words.llm.card_generator import _truncate_around_lemma

        # Lemma at position ~100
        long_text = "A" * 100 + "bank" + "B" * 100
        result = _truncate_around_lemma(long_text, "bank", 50)

        assert "bank" in result.lower()
        # Should have context before and after
        assert result.count("A") > 0 or result.startswith("...")

    def test_handles_irregular_verbs(self):
        """Finds and centers around irregular verb forms."""
        from eng_words.llm.card_generator import _truncate_around_lemma

        long_text = "A" * 50 + "took" + "B" * 50
        result = _truncate_around_lemma(long_text, "take", 40)

        assert "took" in result.lower()

    def test_handles_short_text(self):
        """Returns original text if shorter than max_length."""
        from eng_words.llm.card_generator import _truncate_around_lemma

        short_text = "The bank is closed."
        result = _truncate_around_lemma(short_text, "bank", 100)

        assert result == short_text

    def test_handles_lemma_not_found(self):
        """Truncates from start if lemma not found."""
        from eng_words.llm.card_generator import _truncate_around_lemma

        long_text = "A" * 200
        result = _truncate_around_lemma(long_text, "bank", 50)

        assert len(result) <= 53
        assert result.endswith("...")
