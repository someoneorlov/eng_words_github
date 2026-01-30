"""Tests for LLM cache module (SenseCache).

TDD approach: tests first, implementation second.
"""

import tempfile
from datetime import datetime
from pathlib import Path


class TestSenseCardDataclass:
    """Tests for SenseCard dataclass."""

    def test_import_sense_card(self):
        """SenseCard can be imported."""
        from eng_words.llm.cache import SenseCard

        assert SenseCard is not None

    def test_create_sense_card(self):
        """SenseCard can be created with required fields."""
        from eng_words.llm.cache import SenseCard

        card = SenseCard(
            synset_id="bank.n.01",
            lemma="bank",
            pos="NOUN",
            supersense="noun.group",
            definition_simple="a company where you keep your money safe",
            translation_ru="банк",
            generic_examples=["I need to go to the bank."],
            book_examples={},
            generated_at=datetime.now(),
            model="gpt-4o-mini",
            prompt_version="v1.0",
        )

        assert card.synset_id == "bank.n.01"
        assert card.lemma == "bank"
        assert card.translation_ru == "банк"

    def test_sense_card_to_dict(self):
        """SenseCard can be converted to dict for JSON serialization."""
        from eng_words.llm.cache import SenseCard

        card = SenseCard(
            synset_id="bank.n.01",
            lemma="bank",
            pos="NOUN",
            supersense="noun.group",
            definition_simple="a company where you keep your money safe",
            translation_ru="банк",
            generic_examples=["I need to go to the bank."],
            book_examples={"American Tragedy": ["He went to the bank."]},
            generated_at=datetime(2024, 1, 1, 12, 0, 0),
            model="gpt-4o-mini",
            prompt_version="v1.0",
        )

        d = card.to_dict()

        assert d["synset_id"] == "bank.n.01"
        assert d["book_examples"] == {"American Tragedy": ["He went to the bank."]}
        assert "generated_at" in d

    def test_sense_card_from_dict(self):
        """SenseCard can be created from dict (JSON deserialization)."""
        from eng_words.llm.cache import SenseCard

        d = {
            "synset_id": "bank.n.01",
            "lemma": "bank",
            "pos": "NOUN",
            "supersense": "noun.group",
            "definition_simple": "a company where you keep your money safe",
            "translation_ru": "банк",
            "generic_examples": ["I need to go to the bank."],
            "book_examples": {},
            "generated_at": "2024-01-01T12:00:00",
            "model": "gpt-4o-mini",
            "prompt_version": "v1.0",
        }

        card = SenseCard.from_dict(d)

        assert card.synset_id == "bank.n.01"
        assert card.lemma == "bank"
        assert isinstance(card.generated_at, datetime)


class TestSenseCache:
    """Tests for SenseCache class."""

    def test_import_sense_cache(self):
        """SenseCache can be imported."""
        from eng_words.llm.cache import SenseCache

        assert SenseCache is not None

    def test_create_cache_with_path(self):
        """SenseCache can be created with a cache directory."""
        from eng_words.llm.cache import SenseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))
            assert cache is not None

    def test_has_returns_false_for_missing(self):
        """has() returns False for non-existent synset."""
        from eng_words.llm.cache import SenseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))
            assert cache.has("nonexistent.n.01") is False

    def test_get_returns_none_for_missing(self):
        """get() returns None for non-existent synset."""
        from eng_words.llm.cache import SenseCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))
            assert cache.get("nonexistent.n.01") is None

    def test_store_and_get(self):
        """store() saves card and get() retrieves it."""
        from eng_words.llm.cache import SenseCache, SenseCard

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))

            card = SenseCard(
                synset_id="bank.n.01",
                lemma="bank",
                pos="NOUN",
                supersense="noun.group",
                definition_simple="a company where you keep your money safe",
                translation_ru="банк",
                generic_examples=["I need to go to the bank."],
                book_examples={},
                generated_at=datetime.now(),
                model="gpt-4o-mini",
                prompt_version="v1.0",
            )

            cache.store(card)

            assert cache.has("bank.n.01") is True
            retrieved = cache.get("bank.n.01")
            assert retrieved is not None
            assert retrieved.synset_id == "bank.n.01"
            assert retrieved.translation_ru == "банк"

    def test_store_persists_to_disk(self):
        """store() persists data to disk (survives cache recreation)."""
        from eng_words.llm.cache import SenseCache, SenseCard

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create and store
            cache1 = SenseCache(cache_dir=cache_dir)
            card = SenseCard(
                synset_id="run.v.01",
                lemma="run",
                pos="VERB",
                supersense="verb.motion",
                definition_simple="to move fast using your legs",
                translation_ru="бежать",
                generic_examples=["He runs every morning."],
                book_examples={},
                generated_at=datetime.now(),
                model="gpt-4o-mini",
                prompt_version="v1.0",
            )
            cache1.store(card)

            # Create new cache instance and retrieve
            cache2 = SenseCache(cache_dir=cache_dir)
            retrieved = cache2.get("run.v.01")

            assert retrieved is not None
            assert retrieved.translation_ru == "бежать"

    def test_add_book_examples(self):
        """add_book_examples() adds examples to existing card."""
        from eng_words.llm.cache import SenseCache, SenseCard

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))

            card = SenseCard(
                synset_id="bank.n.01",
                lemma="bank",
                pos="NOUN",
                supersense="noun.group",
                definition_simple="a company where you keep your money safe",
                translation_ru="банк",
                generic_examples=["I need to go to the bank."],
                book_examples={},
                generated_at=datetime.now(),
                model="gpt-4o-mini",
                prompt_version="v1.0",
            )
            cache.store(card)

            # Add examples from a book
            cache.add_book_examples(
                synset_id="bank.n.01",
                book_name="American Tragedy",
                examples=["He walked into the bank.", "The bank was closed."],
            )

            retrieved = cache.get("bank.n.01")
            assert "American Tragedy" in retrieved.book_examples
            assert len(retrieved.book_examples["American Tragedy"]) == 2

    def test_add_book_examples_accumulates(self):
        """add_book_examples() accumulates examples from multiple books."""
        from eng_words.llm.cache import SenseCache, SenseCard

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))

            card = SenseCard(
                synset_id="bank.n.01",
                lemma="bank",
                pos="NOUN",
                supersense="noun.group",
                definition_simple="a company where you keep your money safe",
                translation_ru="банк",
                generic_examples=[],
                book_examples={},
                generated_at=datetime.now(),
                model="gpt-4o-mini",
                prompt_version="v1.0",
            )
            cache.store(card)

            # Add from first book
            cache.add_book_examples(
                synset_id="bank.n.01",
                book_name="Book A",
                examples=["Example from A."],
            )

            # Add from second book
            cache.add_book_examples(
                synset_id="bank.n.01",
                book_name="Book B",
                examples=["Example from B."],
            )

            retrieved = cache.get("bank.n.01")
            assert "Book A" in retrieved.book_examples
            assert "Book B" in retrieved.book_examples

    def test_get_all_synset_ids(self):
        """get_all_synset_ids() returns list of cached synset IDs."""
        from eng_words.llm.cache import SenseCache, SenseCard

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))

            for synset_id in ["bank.n.01", "run.v.01", "happy.a.01"]:
                card = SenseCard(
                    synset_id=synset_id,
                    lemma=synset_id.split(".")[0],
                    pos="NOUN",
                    supersense="noun.group",
                    definition_simple="test",
                    translation_ru="тест",
                    generic_examples=[],
                    book_examples={},
                    generated_at=datetime.now(),
                    model="gpt-4o-mini",
                    prompt_version="v1.0",
                )
                cache.store(card)

            synset_ids = cache.get_all_synset_ids()
            assert len(synset_ids) == 3
            assert "bank.n.01" in synset_ids
            assert "run.v.01" in synset_ids

    def test_get_uncached_synsets(self):
        """get_uncached_synsets() returns synsets not in cache."""
        from eng_words.llm.cache import SenseCache, SenseCard

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))

            # Store one card
            card = SenseCard(
                synset_id="bank.n.01",
                lemma="bank",
                pos="NOUN",
                supersense="noun.group",
                definition_simple="test",
                translation_ru="тест",
                generic_examples=[],
                book_examples={},
                generated_at=datetime.now(),
                model="gpt-4o-mini",
                prompt_version="v1.0",
            )
            cache.store(card)

            # Check which are uncached
            requested = ["bank.n.01", "run.v.01", "happy.a.01"]
            uncached = cache.get_uncached_synsets(requested)

            assert "bank.n.01" not in uncached
            assert "run.v.01" in uncached
            assert "happy.a.01" in uncached

    def test_store_batch(self):
        """store_batch() stores multiple cards efficiently."""
        from eng_words.llm.cache import SenseCache, SenseCard

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SenseCache(cache_dir=Path(tmpdir))

            cards = []
            for i, synset_id in enumerate(["a.n.01", "b.n.02", "c.v.01"]):
                cards.append(
                    SenseCard(
                        synset_id=synset_id,
                        lemma=synset_id[0],
                        pos="NOUN",
                        supersense="noun.group",
                        definition_simple=f"definition {i}",
                        translation_ru=f"перевод {i}",
                        generic_examples=[],
                        book_examples={},
                        generated_at=datetime.now(),
                        model="gpt-4o-mini",
                        prompt_version="v1.0",
                    )
                )

            cache.store_batch(cards)

            assert cache.has("a.n.01")
            assert cache.has("b.n.02")
            assert cache.has("c.v.01")
