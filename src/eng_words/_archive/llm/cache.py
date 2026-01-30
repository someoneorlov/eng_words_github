"""LLM Cache module for storing sense cards.

Provides SenseCard dataclass and SenseCache for persisting LLM-generated
definitions, translations, and examples.

POC implementation uses JSON files for simplicity.
Can be migrated to SQLite when needed (> 5000 cards, concurrent access).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path("data/llm_cache")
SENSE_CARDS_FILE = "sense_cards.json"


@dataclass
class SenseCard:
    """Card for a single word sense.

    Contains LLM-generated definition, translation, examples,
    and metadata for cache invalidation.

    Attributes:
        synset_id: WordNet synset ID (e.g., "bank.n.01")
        lemma: Base form of the word
        pos: Part of speech (NOUN, VERB, ADJ, ADV)
        supersense: WordNet supersense category
        definition_simple: Simple English definition (B1 level)
        translation_ru: Russian translation
        generic_examples: LLM-generated examples (not from book)
        book_examples: Examples from books {book_name: [examples]}
        generated_at: When the card was generated
        model: LLM model used
        prompt_version: Version of the prompt used
    """

    synset_id: str
    lemma: str
    pos: str
    supersense: str
    definition_simple: str
    translation_ru: str
    generic_examples: list[str]
    book_examples: dict[str, list[str]] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    model: str = ""
    prompt_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert datetime to ISO string
        d["generated_at"] = self.generated_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SenseCard:
        """Create SenseCard from dictionary (JSON deserialization)."""
        # Parse datetime from ISO string
        if isinstance(d.get("generated_at"), str):
            d["generated_at"] = datetime.fromisoformat(d["generated_at"])
        return cls(**d)


class SenseCache:
    """Cache for storing and retrieving SenseCards.

    POC implementation uses JSON file storage.
    Thread-safe for single-process usage.

    Args:
        cache_dir: Directory for cache files. Created if doesn't exist.
    """

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / SENSE_CARDS_FILE
        self._cache: dict[str, SenseCard] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self._cache_file.exists():
            try:
                data = json.loads(self._cache_file.read_text(encoding="utf-8"))
                for synset_id, card_dict in data.items():
                    self._cache[synset_id] = SenseCard.from_dict(card_dict)
                logger.info(f"Loaded {len(self._cache)} cards from cache")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to load cache: {e}. Starting fresh.")
                self._cache = {}
        else:
            self._cache = {}

    def _save(self) -> None:
        """Save cache to disk."""
        data = {synset_id: card.to_dict() for synset_id, card in self._cache.items()}
        self._cache_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def has(self, synset_id: str) -> bool:
        """Check if synset is in cache.

        Args:
            synset_id: WordNet synset ID

        Returns:
            True if synset is cached, False otherwise
        """
        return synset_id in self._cache

    def get(self, synset_id: str) -> SenseCard | None:
        """Get SenseCard for synset.

        Args:
            synset_id: WordNet synset ID

        Returns:
            SenseCard if found, None otherwise
        """
        return self._cache.get(synset_id)

    def store(self, card: SenseCard) -> None:
        """Store a SenseCard in cache.

        Overwrites existing card for same synset_id.

        Args:
            card: SenseCard to store
        """
        self._cache[card.synset_id] = card
        self._save()

    def store_batch(self, cards: list[SenseCard]) -> None:
        """Store multiple SenseCards efficiently.

        Saves to disk once after all cards are added.

        Args:
            cards: List of SenseCards to store
        """
        for card in cards:
            self._cache[card.synset_id] = card
        self._save()

    def add_book_examples(self, synset_id: str, book_name: str, examples: list[str]) -> None:
        """Add book examples to existing card.

        Creates book entry if doesn't exist, appends examples.

        Args:
            synset_id: WordNet synset ID
            book_name: Name of the book
            examples: List of example sentences

        Raises:
            KeyError: If synset_id not in cache
        """
        if synset_id not in self._cache:
            raise KeyError(f"Synset {synset_id} not in cache")

        card = self._cache[synset_id]

        if book_name not in card.book_examples:
            card.book_examples[book_name] = []

        # Add new examples (avoid duplicates)
        existing = set(card.book_examples[book_name])
        for example in examples:
            if example not in existing:
                card.book_examples[book_name].append(example)
                existing.add(example)

        self._save()

    def get_all_synset_ids(self) -> list[str]:
        """Get list of all cached synset IDs.

        Returns:
            List of synset IDs in cache
        """
        return list(self._cache.keys())

    def get_uncached_synsets(self, synset_ids: list[str]) -> list[str]:
        """Get synsets not in cache.

        Args:
            synset_ids: List of synset IDs to check

        Returns:
            List of synset IDs not in cache
        """
        return [sid for sid in synset_ids if sid not in self._cache]

    def __len__(self) -> int:
        """Return number of cached cards."""
        return len(self._cache)
