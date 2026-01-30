"""Card Generator for creating Anki flashcards from WSD results.

This module provides functions to:
- Collect book examples for each sense
- Prepare batches for LLM processing
- Generate SenseCards with definitions, translations, and examples
- Export to Anki-compatible CSV format
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from eng_words.constants import SENTENCE_ID, SYNSET_ID
from eng_words.constants.llm_config import (
    CARD_BATCH_SIZE,
    MAX_EXAMPLE_LENGTH,
    MAX_EXAMPLES_PER_SENSE,
    PROMPT_VERSION_CARD_GENERATION,
)
from eng_words.llm.base import LLMProvider
from eng_words.llm.cache import SenseCache, SenseCard
from eng_words.llm.prompts import build_card_generation_prompt

logger = logging.getLogger(__name__)


def collect_book_examples(
    sense_tokens_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Collect unique examples from book for each synset.

    Groups sentences by synset_id, deduplicates, and limits to MAX_EXAMPLES_PER_SENSE.

    Args:
        sense_tokens_df: DataFrame with WSD annotations.
            Required: sentence_id, synset_id
        sentences_df: DataFrame with sentences.
            Required: sentence_id, text (or sentence_text)

    Returns:
        Dict mapping synset_id to list of example sentences.
    """
    # Determine text column name
    text_col = "text" if "text" in sentences_df.columns else "sentence_text"

    # Filter out null synsets
    valid_tokens = sense_tokens_df[sense_tokens_df[SYNSET_ID].notna()].copy()

    # Merge to get sentence text
    merged = valid_tokens.merge(
        sentences_df[[SENTENCE_ID, text_col]],
        on=SENTENCE_ID,
        how="left",
    )

    # Rename to consistent column
    if text_col != "text":
        merged = merged.rename(columns={text_col: "text"})

    # Drop rows without text
    merged = merged.dropna(subset=["text"])

    # Group by synset_id and collect unique sentences
    result: dict[str, list[str]] = {}

    for synset_id, group in merged.groupby(SYNSET_ID):
        # Get unique sentences
        unique_sentences = group["text"].drop_duplicates().tolist()

        # Limit to MAX_EXAMPLES_PER_SENSE
        result[synset_id] = unique_sentences[:MAX_EXAMPLES_PER_SENSE]

    return result


def _contains_lemma(text: str, lemma: str) -> bool:
    """Check if text contains lemma using spaCy lemmatization.

    Uses spaCy to lemmatize all tokens in text and checks if any token's
    lemma matches the target lemma. This automatically handles irregular
    verbs and all inflections.

    Args:
        text: Text to search in.
        lemma: Base form of the word (lemma).

    Returns:
        True if any token in text has the target lemma.
    """
    try:
        from eng_words.text_processing import initialize_spacy_model

        # Lazy load spaCy model (cached by initialize_spacy_model)
        nlp = initialize_spacy_model()
        doc = nlp(text)

        lemma_lower = lemma.lower()

        # Check if any token's lemma matches
        for token in doc:
            if token.lemma_.lower() == lemma_lower:
                return True

        return False
    except Exception as e:
        # Fallback: simple string search + common irregular verbs
        # This is less accurate but works if spaCy model is not available
        logger.warning(f"spaCy lemmatization failed, using fallback: {e}")
        text_lower = text.lower()
        lemma_lower = lemma.lower()

        # Check base form
        if lemma_lower in text_lower:
            return True

        # Common irregular verb mappings (minimal set for fallback)
        irregular_map = {
            "take": ["took", "taken", "takes", "taking"],
            "go": ["went", "gone", "goes", "going"],
            "be": ["is", "am", "are", "was", "were", "been", "being"],
            "have": ["has", "had", "having"],
            "run": ["ran", "runs", "running"],
            "speed": ["sped", "speeds", "speeding"],
            "draw": ["drew", "drawn", "draws", "drawing"],
            "herald": ["heralds", "heralded", "heralding"],
        }

        if lemma_lower in irregular_map:
            return any(form in text_lower for form in irregular_map[lemma_lower])

        return False


def _truncate_around_lemma(text: str, lemma: str, max_length: int) -> str:
    """Truncate text centered around the lemma occurrence.

    Uses spaCy to find tokens with matching lemma, then centers truncation
    around the first occurrence. Automatically handles irregular verbs.

    Args:
        text: Full example text.
        lemma: Lemma to center around.
        max_length: Maximum length of result.

    Returns:
        Truncated text with lemma visible (if found).
    """
    try:
        from eng_words.text_processing import initialize_spacy_model

        # Lazy load spaCy model
        nlp = initialize_spacy_model()
        doc = nlp(text)

        lemma_lower = lemma.lower()

        # Find first token with matching lemma
        target_token = None
        for token in doc:
            if token.lemma_.lower() == lemma_lower:
                target_token = token
                break

        if target_token is None:
            # Fallback: try to find lemma as substring (for edge cases like "A"*100+"bank")
            # Also check for irregular verb forms
            text_lower = text.lower()
            pos = text_lower.find(lemma_lower)

            # Try irregular verbs if base form not found
            if pos < 0:
                irregular_map = {
                    "take": ["took", "taken", "takes", "taking"],
                    "go": ["went", "gone", "goes", "going"],
                    "run": ["ran", "runs", "running"],
                    "speed": ["sped", "speeds", "speeding"],
                    "draw": ["drew", "drawn", "draws", "drawing"],
                }
                if lemma_lower in irregular_map:
                    for form in irregular_map[lemma_lower]:
                        pos = text_lower.find(form)
                        if pos >= 0:
                            lemma_lower = form  # Use found form for length calculation
                            break

            if pos >= 0:
                # Found as substring - use simple truncation
                needs_start_ellipsis = pos > 0
                needs_end_ellipsis = pos + len(lemma_lower) < len(text)
                ellipsis_chars = (3 if needs_start_ellipsis else 0) + (
                    3 if needs_end_ellipsis else 0
                )
                available_length = max(max_length - ellipsis_chars, 10)  # Min 10 chars

                half_window = available_length // 2
                start = max(0, pos - half_window)
                end = min(len(text), pos + len(lemma_lower) + half_window)
                if start == 0:
                    end = min(len(text), start + available_length)
                elif end == len(text):
                    start = max(0, end - available_length)

                result = text[start:end]
                if start > 0:
                    result = "..." + result
                if end < len(text):
                    result = result + "..."

                if len(result) > max_length:
                    excess = len(result) - max_length
                    if result.endswith("..."):
                        result = result[: -3 - excess] + "..."
                    else:
                        result = result[:-excess]

                return result

            # Lemma not found - truncate from start
            return text[: max_length - 3] + "..."

        # Get character position of token start
        pos = target_token.idx
        token_end = pos + len(target_token.text)

        # Calculate available length for text (accounting for ellipsis)
        # We may add "..." at start (3 chars) and/or end (3 chars)
        needs_start_ellipsis = pos > 0
        needs_end_ellipsis = token_end < len(text)

        ellipsis_chars = (3 if needs_start_ellipsis else 0) + (3 if needs_end_ellipsis else 0)
        available_length = max_length - ellipsis_chars

        # Ensure available_length is positive
        if available_length <= 0:
            available_length = max_length - 6  # Worst case: both ellipsis

        # Calculate window centered on token
        half_window = available_length // 2
        start = max(0, pos - half_window)
        end = min(len(text), token_end + half_window)

        # Adjust if near boundaries
        if start == 0:
            end = min(len(text), start + available_length)
        elif end == len(text):
            start = max(0, end - available_length)

        result = text[start:end]

        # Add ellipsis where truncated
        if start > 0:
            result = "..." + result
        if end < len(text):
            result = result + "..."

        # Ensure result doesn't exceed max_length (safety check)
        if len(result) > max_length:
            # Trim from end if too long
            excess = len(result) - max_length
            if result.endswith("..."):
                result = result[: -3 - excess] + "..."
            else:
                result = result[:-excess]

        return result
    except Exception as e:
        # Fallback: simple truncation with basic lemma search
        logger.warning(f"spaCy truncation failed, using fallback: {e}")
        text_lower = text.lower()
        lemma_lower = lemma.lower()

        # Try to find lemma or common forms
        pos = text_lower.find(lemma_lower)
        if pos < 0:
            # Try irregular verbs
            irregular_map = {
                "take": ["took", "taken", "takes", "taking"],
                "go": ["went", "gone", "goes", "going"],
                "run": ["ran", "runs", "running"],
                "speed": ["sped", "speeds", "speeding"],
                "draw": ["drew", "drawn", "draws", "drawing"],
            }
            if lemma_lower in irregular_map:
                for form in irregular_map[lemma_lower]:
                    pos = text_lower.find(form)
                    if pos >= 0:
                        break

        if pos >= 0:
            # Center around found position
            # Account for ellipsis
            needs_start_ellipsis = pos > 0
            needs_end_ellipsis = pos < len(text)
            ellipsis_chars = (3 if needs_start_ellipsis else 0) + (3 if needs_end_ellipsis else 0)
            available_length = max_length - ellipsis_chars

            half_window = available_length // 2
            start = max(0, pos - half_window)
            end = min(len(text), pos + half_window)
            if start == 0:
                end = min(len(text), start + available_length)
            elif end == len(text):
                start = max(0, end - available_length)
            result = text[start:end]
            if start > 0:
                result = "..." + result
            if end < len(text):
                result = result + "..."
            return result

        # No match found - truncate from start
        return text[: max_length - 3] + "..."


def prepare_card_batch(
    synset_infos: list[dict[str, Any]],
    book_examples: dict[str, list[str]],
    book_name: str,
) -> list[dict[str, Any]]:
    """Prepare batch of senses for LLM card generation.

    Truncates long examples and adds book metadata.

    Args:
        synset_infos: List of dicts with synset info.
            Required: synset_id, lemma, pos, supersense, definition
        book_examples: Dict mapping synset_id to examples.
        book_name: Name of the book for context.

    Returns:
        List of dicts ready for prompt template.
    """
    batch = []

    for info in synset_infos:
        synset_id = info["synset_id"]
        lemma = info["lemma"].lower()

        # Get examples and truncate if needed (centered around lemma)
        examples = book_examples.get(synset_id, [])
        truncated_examples = []
        for ex in examples:
            if len(ex) > MAX_EXAMPLE_LENGTH:
                truncated_examples.append(_truncate_around_lemma(ex, lemma, MAX_EXAMPLE_LENGTH))
            else:
                truncated_examples.append(ex)

        batch.append(
            {
                "synset_id": synset_id,
                "lemma": info["lemma"],
                "pos": info["pos"],
                "supersense": info["supersense"],
                "definition": info.get("definition", ""),
                "book_name": book_name,
                "book_examples": truncated_examples,
            }
        )

    return batch


class CardGenerator:
    """Generator for creating SenseCards using LLM.

    Uses LLM to generate definitions, translations, and select best examples.

    Args:
        provider: LLM provider for generating content.
        cache: SenseCache for storing/retrieving cards.
    """

    def __init__(self, provider: LLMProvider, cache: SenseCache):
        self.provider = provider
        self.cache = cache

    def generate_batch(
        self,
        synset_infos: list[dict[str, Any]],
        book_examples: dict[str, list[str]],
        book_name: str,
    ) -> list[SenseCard]:
        """Generate SenseCards for a batch of synsets.

        For cached synsets, only adds new book examples.
        For uncached synsets, generates full card via LLM.

        Args:
            synset_infos: List of dicts with synset info.
            book_examples: Dict mapping synset_id to examples.
            book_name: Name of the book.

        Returns:
            List of generated/updated SenseCards.
        """
        # Separate cached vs uncached
        synset_ids = [info["synset_id"] for info in synset_infos]
        uncached_ids = set(self.cache.get_uncached_synsets(synset_ids))

        uncached_infos = [info for info in synset_infos if info["synset_id"] in uncached_ids]
        cached_ids = [sid for sid in synset_ids if sid not in uncached_ids]

        results: list[SenseCard] = []

        # For cached: just add book examples
        for synset_id in cached_ids:
            examples = book_examples.get(synset_id, [])
            if examples:
                try:
                    self.cache.add_book_examples(synset_id, book_name, examples)
                except KeyError:
                    pass
            card = self.cache.get(synset_id)
            if card:
                results.append(card)

        # For uncached: generate via LLM
        if uncached_infos:
            batch = prepare_card_batch(uncached_infos, book_examples, book_name)
            new_cards = self._generate_from_llm(batch, book_name)
            self.cache.store_batch(new_cards)
            results.extend(new_cards)

        return results

    def _generate_from_llm(
        self,
        batch: list[dict[str, Any]],
        book_name: str,
        max_retries: int = 2,
    ) -> list[SenseCard]:
        """Generate SenseCards via LLM call with retry logic.

        Calls LLM with card generation prompt and parses JSON response.
        Filters book examples by spoiler_risk (only "none" accepted).
        Retries for missing synsets if needed.

        Args:
            batch: Prepared batch of sense infos.
            book_name: Name of the book.
            max_retries: Maximum retry attempts for missing synsets.

        Returns:
            List of generated SenseCards.
        """
        if not batch:
            return []

        # Build prompt
        prompt = build_card_generation_prompt(batch)

        # Call LLM with full response to get token counts
        try:
            llm_response = self.provider.complete(prompt)
            logger.info(
                f"LLM call: {llm_response.input_tokens} input, "
                f"{llm_response.output_tokens} output tokens, "
                f"${llm_response.cost_usd:.4f}"
            )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._create_placeholder_cards(batch, book_name)

        # Parse JSON from response content
        try:
            content = llm_response.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            response = json.loads(content.strip())
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}")
            return self._create_placeholder_cards(batch, book_name)

        # Parse response - expect list of card dicts
        if not isinstance(response, list):
            logger.warning(f"Expected list from LLM, got {type(response)}")
            return self._create_placeholder_cards(batch, book_name)

        # Log coverage and identify missing
        response_by_synset = {item.get("synset_id"): item for item in response}
        matched = sum(1 for item in batch if item["synset_id"] in response_by_synset)
        missing_items = [item for item in batch if item["synset_id"] not in response_by_synset]

        if matched < len(batch):
            requested = [item["synset_id"] for item in batch]
            returned = [item.get("synset_id") for item in response]
            logger.warning(
                f"LLM returned {len(response)} items for {len(batch)} synsets. "
                f"Matched: {matched}/{len(batch)}, Missing: {len(missing_items)}"
            )
            logger.debug(f"Requested synsets: {requested}")
            logger.debug(f"Returned synsets: {returned}")

        # Retry for missing synsets (if any and retries left)
        if missing_items and max_retries > 0:
            logger.info(f"Retrying for {len(missing_items)} missing synsets...")
            retry_cards = self._generate_from_llm(
                missing_items, book_name, max_retries=max_retries - 1
            )
            for card in retry_cards:
                if card.definition_simple != "[no definition]":
                    response_by_synset[card.synset_id] = {
                        "synset_id": card.synset_id,
                        "definition_simple": card.definition_simple,
                        "translation_ru": card.translation_ru,
                        "generic_examples": card.generic_examples,
                        "book_examples_selected": (
                            [
                                {"text": ex, "spoiler_risk": "none"}
                                for ex in list(card.book_examples.values())[0]
                            ]
                            if card.book_examples
                            else []
                        ),
                    }

        # Create SenseCards from response
        cards = []

        for item in batch:
            synset_id = item["synset_id"]
            llm_result = response_by_synset.get(synset_id, {})

            # Extract and filter book examples (only spoiler_risk="none")
            # Also validate that examples contain lemma or its forms
            book_examples_selected = []
            lemma_lower = item["lemma"].lower()
            original_examples = item.get("book_examples", [])

            for ex in llm_result.get("book_examples_selected", []):
                if isinstance(ex, dict):
                    if ex.get("spoiler_risk", "unknown") == "none":
                        ex_text = ex.get("text", "")
                        # Validate lemma is in example
                        if _contains_lemma(ex_text, lemma_lower):
                            book_examples_selected.append(ex_text)
                        else:
                            logger.warning(
                                f"LLM example missing lemma '{lemma_lower}': {ex_text[:50]}..."
                            )
                elif isinstance(ex, str):
                    if _contains_lemma(ex, lemma_lower):
                        book_examples_selected.append(ex)

            # Fallback: if no valid examples, use originals (up to 2)
            if not book_examples_selected and original_examples:
                book_examples_selected = [
                    ex for ex in original_examples[:2] if _contains_lemma(ex, lemma_lower)
                ]

            card = SenseCard(
                synset_id=synset_id,
                lemma=item["lemma"],
                pos=item["pos"],
                supersense=item["supersense"],
                definition_simple=llm_result.get("definition_simple", "[no definition]"),
                translation_ru=llm_result.get("translation_ru", "[no translation]"),
                generic_examples=llm_result.get("generic_examples", []),
                book_examples={book_name: book_examples_selected} if book_examples_selected else {},
                generated_at=datetime.now(),
                model=getattr(self.provider, "model", "unknown"),
                prompt_version=PROMPT_VERSION_CARD_GENERATION,
            )
            cards.append(card)

        return cards

    def _create_placeholder_cards(
        self,
        batch: list[dict[str, Any]],
        book_name: str,
    ) -> list[SenseCard]:
        """Create placeholder cards when LLM fails.

        Args:
            batch: Prepared batch of sense infos.
            book_name: Name of the book.

        Returns:
            List of placeholder SenseCards.
        """
        cards = []
        for item in batch:
            card = SenseCard(
                synset_id=item["synset_id"],
                lemma=item["lemma"],
                pos=item["pos"],
                supersense=item["supersense"],
                definition_simple="[LLM error - no definition]",
                translation_ru="[LLM error - no translation]",
                generic_examples=[],
                book_examples=(
                    {book_name: item["book_examples"][:3]} if item.get("book_examples") else {}
                ),
                generated_at=datetime.now(),
                model=getattr(self.provider, "model", "unknown"),
                prompt_version=PROMPT_VERSION_CARD_GENERATION,
            )
            cards.append(card)
        return cards

    def generate_all(
        self,
        synset_infos: list[dict[str, Any]],
        book_examples: dict[str, list[str]],
        book_name: str,
        batch_size: int = CARD_BATCH_SIZE,
    ) -> list[SenseCard]:
        """Generate cards for all synsets with batching.

        Processes synsets in batches of batch_size.

        Args:
            synset_infos: List of all synset infos.
            book_examples: Dict mapping synset_id to examples.
            book_name: Name of the book.
            batch_size: Number of senses per LLM batch.

        Returns:
            List of all generated SenseCards.
        """
        all_cards = []

        for i in range(0, len(synset_infos), batch_size):
            batch = synset_infos[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} ({len(batch)} senses)")
            cards = self.generate_batch(batch, book_examples, book_name)
            all_cards.extend(cards)

        return all_cards


def export_to_anki(
    cards: list[SenseCard],
    output_path: Path | str,
    book_name: str | None = None,
) -> int:
    """Export SenseCards to Anki-compatible CSV.

    Creates a tab-separated CSV file with Front and Back columns.

    Front format:
        word (POS) — supersense
        📖 "Book example sentence."
           — Book Name

    Back format:
        🔤 Simple definition
        🇷🇺 Russian translation

        📝 Examples:
        • Generic example 1
        • Generic example 2

        📖 Also from book:
        • "Other book example"

    Args:
        cards: List of SenseCards to export.
        output_path: Path to output CSV file.
        book_name: Optional book name for front display.

    Returns:
        Number of cards exported.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")

        for card in cards:
            # Get first book example for front
            front_example = ""
            front_book = book_name or ""
            all_book_examples = []

            for bname, examples in card.book_examples.items():
                all_book_examples.extend(examples)
                if not front_example and examples:
                    front_example = examples[0]
                    front_book = bname

            # Skip cards without any examples
            if not front_example and not card.generic_examples:
                continue

            # If no book example, use first generic
            if not front_example and card.generic_examples:
                front_example = card.generic_examples[0]
                front_book = ""

            # Build Front
            front_lines = [
                f"<b>{card.lemma}</b> ({card.pos.lower()}) — {card.supersense}",
                "",
            ]
            if front_book:
                front_lines.append(f'📖 "{front_example}"')
                front_lines.append(f"   — <i>{front_book}</i>")
            else:
                front_lines.append(f"📝 {front_example}")

            front = "<br>".join(front_lines)

            # Build Back
            back_lines = [
                f"🔤 {card.definition_simple}",
                f"🇷🇺 {card.translation_ru}",
            ]

            # Generic examples
            if card.generic_examples:
                back_lines.append("")
                back_lines.append("📝 Examples:")
                for ex in card.generic_examples:
                    back_lines.append(f"• {ex}")

            # Other book examples (excluding front one)
            other_book_examples = [ex for ex in all_book_examples if ex != front_example]
            if other_book_examples:
                back_lines.append("")
                back_lines.append("📖 Also from book:")
                for ex in other_book_examples[:3]:  # Limit to 3
                    back_lines.append(f'• "{ex}"')

            back = "<br>".join(back_lines)

            writer.writerow([front, back])
            exported += 1

    return exported
