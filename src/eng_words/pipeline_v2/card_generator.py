"""Stage 2: Generate flashcards from extracted meanings.

Takes meanings from Stage 1 and:
- Selects clean examples (no spoilers)
- Generates examples if needed
- Translates definitions to Russian

Optimized: Batches meanings (default 5) with retry on failure.
"""

import json
import logging
from typing import Any

from eng_words.llm.base import LLMProvider
from eng_words.llm.response_cache import ResponseCache

from .data_models import (
    CleanExample,
    ExtractedMeaning,
    ExtractionResult,
    FinalCard,
    GenerationResult,
)

logger = logging.getLogger(__name__)

# Default batch size for Stage 2
DEFAULT_BATCH_SIZE = 5
MIN_BATCH_SIZE = 1  # Fallback to per-meaning on repeated failures


# Single meaning prompt (for small batches or fallback)
GENERATION_PROMPT_SINGLE = """You are creating an Anki flashcard for English learners (B1-B2 level).

LEMMA: {lemma}
{phrasal_info}

MEANING:
- Definition: {definition_en}
- Part of speech: {pos}

SOURCE EXAMPLES (from the book):
{source_examples}

TASK:
Create a flashcard for this meaning.

RULES:
1. SELECT 2-3 CLEAN EXAMPLES from source:
   - Skip examples marked [SPOILER]
   - Length: 10-30 words preferred

2. GENERATE if needed:
   - If fewer than 2 clean examples
   - Create simple, natural sentences
   - Mark as source="generated"

3. TRANSLATE DEFINITION to Russian (B1-B2 level)

OUTPUT (strict JSON, no markdown):
{{
  "definition_ru": "translation",
  "clean_examples": [
    {{"sentence_id": 5, "text": "Example from book.", "source": "book"}},
    {{"sentence_id": null, "text": "Generated example.", "source": "generated"}}
  ]
}}
"""

# Batched prompt (multiple meanings at once)
GENERATION_PROMPT_BATCHED = """You are creating Anki flashcards for English learners (B1-B2 level).

LEMMA: {lemma}

MEANINGS TO CREATE CARDS FOR:
{meanings_list}

TASK:
Create a flashcard for EACH meaning.

RULES FOR EACH CARD:
1. SELECT 2-3 CLEAN EXAMPLES from the provided source examples:
   - Skip examples marked [SPOILER]
   - Length: 10-30 words preferred

2. GENERATE if needed:
   - If fewer than 2 clean examples available
   - Create simple, natural sentences
   - Mark as source="generated"

3. TRANSLATE DEFINITION to Russian (B1-B2 level)

OUTPUT (strict JSON array, no markdown):
{{
  "cards": [
    {{
      "meaning_id": 1,
      "definition_ru": "translation",
      "clean_examples": [
        {{"sentence_id": 5, "text": "Example.", "source": "book"}},
        {{"sentence_id": null, "text": "Generated.", "source": "generated"}}
      ]
    }}
  ]
}}
"""


class CardGenerator:
    """Stage 2: Generate flashcards from extracted meanings.
    
    Optimized approach:
    - Batches meanings (default 5) for efficiency
    - Uses only source_examples (not all) to reduce tokens
    - Retry with smaller batch on parse errors
    
    Args:
        provider: LLM provider for API calls
        cache: Response cache (optional)
        batch_size: Number of meanings per batch (default 5)
    """
    
    def __init__(
        self,
        provider: LLMProvider,
        cache: ResponseCache | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.provider = provider
        self.cache = cache
        self.batch_size = batch_size
        
        # Stats
        self.total_api_calls = 0
        self.cache_hits = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.retry_count = 0
    
    def generate(
        self,
        extraction: ExtractionResult,
        examples: list[str],
        sentence_ids: list[int],
    ) -> GenerationResult:
        """Generate cards for all meanings (batched with retry).
        
        Args:
            extraction: Output from Stage 1
            examples: All example sentences for this lemma
            sentence_ids: Corresponding sentence IDs
            
        Returns:
            GenerationResult with all cards
        """
        if not extraction.meanings:
            return GenerationResult(
                lemma=extraction.lemma,
                cards=[],
                total_meanings=0,
                total_cards=0,
            )
        
        cards = []
        total_cost = 0.0
        
        # Process meanings in batches
        meanings = extraction.meanings
        i = 0
        current_batch_size = self.batch_size
        
        while i < len(meanings):
            batch = meanings[i:i + current_batch_size]
            
            if len(batch) == 1:
                # Single meaning - use simple prompt
                result = self._generate_single(
                    extraction.lemma, batch[0], examples,
                    extraction.all_sentence_ids,
                )
                if result:
                    cards.append(result)
                    total_cost += result.generation_cost_usd
                i += 1
            else:
                # Multiple meanings - use batched prompt
                result = self._generate_batch(
                    extraction.lemma, batch, examples,
                    extraction.all_sentence_ids,
                )
                if result is not None:
                    cards.extend(result)
                    total_cost += sum(c.generation_cost_usd for c in result)
                    i += len(batch)
                else:
                    # Batch failed - reduce batch size and retry
                    self.retry_count += 1
                    current_batch_size = max(MIN_BATCH_SIZE, current_batch_size // 2)
                    logger.warning(
                        f"Batch failed for '{extraction.lemma}', "
                        f"reducing batch size to {current_batch_size}"
                    )
                    if current_batch_size == MIN_BATCH_SIZE:
                        # Already at minimum, skip this batch
                        i += len(batch)
        
        return GenerationResult(
            lemma=extraction.lemma,
            cards=cards,
            total_meanings=len(extraction.meanings),
            total_cards=len(cards),
            total_cost_usd=total_cost,
        )
    
    def _generate_single(
        self,
        lemma: str,
        meaning: ExtractedMeaning,
        examples: list[str],
        all_sentence_ids: list[int],
    ) -> FinalCard | None:
        """Generate a single card (for batch=1 or fallback)."""
        
        # Build phrasal info
        phrasal_info = ""
        if meaning.is_phrasal and meaning.phrasal_form:
            phrasal_info = f"PHRASAL VERB: {meaning.phrasal_form}"
        
        # Build source examples only (no all_examples!)
        source_lines = []
        for se in meaning.source_examples:
            if 0 < se.index <= len(examples):
                text = examples[se.index - 1]
                spoiler_mark = " [SPOILER]" if se.has_spoiler else ""
                source_lines.append(f"{se.index}. {text}{spoiler_mark}")
        source_examples_str = "\n".join(source_lines) or "No source examples"
        
        prompt = GENERATION_PROMPT_SINGLE.format(
            lemma=lemma,
            phrasal_info=phrasal_info,
            definition_en=meaning.definition_en,
            pos=meaning.part_of_speech,
            source_examples=source_examples_str,
        )
        
        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                self.provider.model, prompt, self.provider.temperature
            )
            cached = self.cache.get(cache_key)
            if cached:
                self.cache_hits += 1
                return self._parse_single_response(
                    lemma, meaning, cached.content, all_sentence_ids
                )
        
        # Call LLM
        response = self.provider.complete(prompt)
        self.total_api_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost += response.cost_usd
        
        # Cache response
        if self.cache:
            self.cache.set(cache_key, response)
        
        return self._parse_single_response(
            lemma, meaning, response.content, all_sentence_ids,
            cost_usd=response.cost_usd,
        )
    
    def _generate_batch(
        self,
        lemma: str,
        meanings: list[ExtractedMeaning],
        examples: list[str],
        all_sentence_ids: list[int],
    ) -> list[FinalCard] | None:
        """Generate multiple cards in one call."""
        
        # Build meanings list with source examples
        meanings_lines = []
        for m in meanings:
            phrasal_note = f" (PHRASAL: {m.phrasal_form})" if m.is_phrasal else ""
            
            # Source examples for this meaning
            source_lines = []
            for se in m.source_examples:
                if 0 < se.index <= len(examples):
                    text = examples[se.index - 1]
                    spoiler_mark = " [SPOILER]" if se.has_spoiler else ""
                    source_lines.append(f"    - {text}{spoiler_mark}")
            sources_str = "\n".join(source_lines) or "    - No examples"
            
            meanings_lines.append(
                f"{m.meaning_id}. [{m.part_of_speech}]{phrasal_note} {m.definition_en}\n"
                f"  Source examples:\n{sources_str}"
            )
        meanings_str = "\n\n".join(meanings_lines)
        
        prompt = GENERATION_PROMPT_BATCHED.format(
            lemma=lemma,
            meanings_list=meanings_str,
        )
        
        # Check cache
        if self.cache:
            cache_key = self.cache.generate_key(
                self.provider.model, prompt, self.provider.temperature
            )
            cached = self.cache.get(cache_key)
            if cached:
                self.cache_hits += 1
                return self._parse_batch_response(
                    lemma, meanings, cached.content, all_sentence_ids
                )
        
        # Call LLM
        response = self.provider.complete(prompt)
        self.total_api_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost += response.cost_usd
        
        # Cache response
        if self.cache:
            self.cache.set(cache_key, response)
        
        return self._parse_batch_response(
            lemma, meanings, response.content, all_sentence_ids,
            cost_usd=response.cost_usd,
        )
    
    def _parse_single_response(
        self,
        lemma: str,
        meaning: ExtractedMeaning,
        content: str,
        all_sentence_ids: list[int],
        cost_usd: float = 0.0,
    ) -> FinalCard | None:
        """Parse single card LLM response."""
        try:
            data = self._parse_json(content)
            return self._build_card(lemma, meaning, data, all_sentence_ids, cost_usd)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse response for '{lemma}' meaning {meaning.meaning_id}: {e}")
            return None
    
    def _parse_batch_response(
        self,
        lemma: str,
        meanings: list[ExtractedMeaning],
        content: str,
        all_sentence_ids: list[int],
        cost_usd: float = 0.0,
    ) -> list[FinalCard] | None:
        """Parse batched LLM response."""
        try:
            data = self._parse_json(content)
            meaning_by_id = {m.meaning_id: m for m in meanings}
            
            cards = []
            cost_per_card = cost_usd / max(len(data.get("cards", [])), 1)
            
            for card_data in data.get("cards", []):
                meaning_id = card_data.get("meaning_id")
                meaning = meaning_by_id.get(meaning_id)
                if meaning:
                    card = self._build_card(
                        lemma, meaning, card_data, all_sentence_ids, cost_per_card
                    )
                    if card:
                        cards.append(card)
            
            return cards if cards else None
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse batch response for '{lemma}': {e}")
            return None
    
    def _parse_json(self, content: str) -> dict:
        """Clean and parse JSON from LLM response."""
        text = content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    
    def _build_card(
        self,
        lemma: str,
        meaning: ExtractedMeaning,
        data: dict,
        all_sentence_ids: list[int],
        cost_usd: float,
    ) -> FinalCard:
        """Build FinalCard from parsed data."""
        # Parse clean examples
        clean_examples = []
        for ex in data.get("clean_examples", []):
            clean_examples.append(CleanExample(
                sentence_id=ex.get("sentence_id"),
                text=ex.get("text", ""),
                source=ex.get("source", "generated"),
            ))
        
        # Build card ID and display name
        if meaning.is_phrasal and meaning.phrasal_form:
            card_id = f"{meaning.phrasal_form.replace(' ', '_')}_{meaning.meaning_id}"
            lemma_display = meaning.phrasal_form
        else:
            card_id = f"{lemma}_{meaning.meaning_id}"
            lemma_display = lemma
        
        # Get source sentence IDs
        source_sids = [
            se.sentence_id for se in meaning.source_examples
            if se.sentence_id >= 0
        ]
        
        return FinalCard(
            card_id=card_id,
            lemma=lemma,
            lemma_display=lemma_display,
            meaning_id=meaning.meaning_id,
            definition_en=meaning.definition_en,
            definition_ru=data.get("definition_ru", ""),
            part_of_speech=meaning.part_of_speech,
            is_phrasal=meaning.is_phrasal,
            phrasal_form=meaning.phrasal_form,
            all_sentence_ids=all_sentence_ids,
            source_sentence_ids=source_sids,
            clean_examples=clean_examples,
            generation_cost_usd=cost_usd,
        )
    
    def stats(self) -> dict[str, Any]:
        """Return generation statistics."""
        return {
            "total_api_calls": self.total_api_calls,
            "cache_hits": self.cache_hits,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "retry_count": self.retry_count,
            "batch_size": self.batch_size,
        }
