"""Smart Card Generator using LLM for quality control.

Generates high-quality Anki flashcards by:
1. Selecting best examples from book
2. Detecting WSD errors (wrong sense examples)
3. Creating simple definitions and translations
4. Generating additional example sentences
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from nltk.corpus import wordnet as wn

from eng_words.llm.base import LLMProvider, LLMResponse
from eng_words.llm.response_cache import ResponseCache
from eng_words.llm.retry import call_llm_with_retry

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Template
# =============================================================================

SMART_CARD_PROMPT = """You are helping create Anki flashcards for an English language learner (B1-B2 level).

**Note**: The example sentences provided have already been filtered for length (>50 words removed) and spoilers. You will receive only quality examples that are appropriate length and spoiler-free.

## Word Information
- Word: "{lemma}" ({pos})
- Semantic category: {supersense}
- WordNet definition: {wn_definition}
{synset_group_info}

## Example sentences from the book "{book_name}"
{examples_numbered}

## Your Task
1. **Quality assessment**: Assess quality of provided examples:
   - Good: clear, natural, appropriate length (10-30 words)
   - Bad: unclear context, awkward phrasing
2. **Select best examples**: Choose the BEST quality examples from the provided list
   - You will receive pre-selected examples based on availability
   - Select the best ones from what is provided
3. **Generate additional examples**: Generate exactly {generate_count} additional example(s) to reach 3 total
   - Generated examples must be simple, clear, 10-20 words
   - Must contain the word "{lemma}" (or its grammatical forms)
   - Do NOT include plot spoilers or story-specific details
4. **Simple definition**: Write a clear, simple definition (avoid jargon, max 15 words)
5. **Translation**: Provide Russian translation for THIS specific meaning

## Quality Criteria
- **Clarity**: Examples should be self-contained and clear
- **Relevance**: Must match the exact meaning of the synset group
- **Length**: All examples are already filtered (10-30 words ideal)
- **Spoilers**: All examples are already checked (no spoilers)

## Response Format (JSON only, no markdown)
{{
  "valid_indices": [1, 2, 3, 4],
  "invalid_indices": [5, 6],  // Only for wrong sense (different meaning), NOT for length/spoilers (already filtered)
  "quality_scores": {{"1": 5, "2": 4, "3": 5, "4": 3}},
  "selected_indices": [1, 2, 3],
  "generated_examples": ["Example 1", "Example 2"],
  "simple_definition": "to move quickly using your legs",
  "translation_ru": "run"
}}

**Note**: invalid_indices should only be used for examples with wrong sense (different meaning), NOT for length or spoilers (these are already filtered before this step).

Return ONLY valid JSON, no explanations."""


EXAMPLE_GENERATION_PROMPT = """You are helping create example sentences for English learning flashcards.

Word: "{lemma}" ({pos})
Definition: {definition}

Generate a simple, clear example sentence (10-20 words) that demonstrates this word in natural context.

CRITICAL REQUIREMENT:
- The sentence MUST contain the exact word "{lemma}" (or its correct grammatical form: past tense, -ing, -s, etc.)
- DO NOT use synonyms or similar words - use "{lemma}" itself
- The word "{lemma}" must be clearly visible in the sentence

Additional requirements:
- Keep it simple and suitable for B1-B2 level learners
- Make it a standalone, clear example
- Do NOT include plot spoilers or story-specific details
- Use natural, everyday English

Example for "run" (verb): "She runs in the park every morning." ✅ (contains "runs")
Bad example: "She exercises daily." ❌ (no "run" or its forms)

Return ONLY the example sentence, nothing else."""


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def mark_examples_by_length(
    examples: list[tuple[int, str]],  # (sentence_id, sentence)
    max_words: int = 50,
    min_words: int = 6,  # Minimum length for a proper context
) -> dict[int, bool]:  # sentence_id -> is_appropriate_length
    """Mark examples by length (don't filter, just mark).
    
    Args:
        examples: List of (sentence_id, sentence) tuples
        max_words: Maximum allowed words (default: 50)
        min_words: Minimum allowed words (default: 6)
        
    Returns:
        Dictionary mapping sentence_id to is_appropriate_length (True/False)
        True = appropriate length (min_words <= length <= max_words), 
        False = too short (<min_words) or too long (>max_words)
    """
    length_flags = {}
    for sid, sentence in examples:
        word_count = count_words(sentence)
        length_flags[sid] = min_words <= word_count <= max_words
    return length_flags


SPOILER_CHECK_PROMPT = """You are helping filter example sentences for language learning flashcards.

Book: "{book_name}"

## Example sentences
{examples_numbered}

## Your Task
For each sentence, determine if it contains plot spoilers (reveals story events, character deaths, plot twists, or story endings).

Return JSON:
{{
  "has_spoiler": [true, false, false, true, ...]  // One boolean per sentence (same order)
}}

Return ONLY valid JSON, no explanations."""


def check_spoilers(
    examples: list[tuple[int, str]],  # (sentence_id, sentence)
    provider: LLMProvider,
    cache: ResponseCache,
    book_name: str,
    max_examples_per_batch: int = 50,
) -> dict[int, bool]:  # sentence_id -> has_spoiler
    """Check examples for spoilers.
    
    Args:
        examples: List of (sentence_id, sentence) tuples
        provider: LLM provider
        cache: Response cache
        book_name: Name of the book
        max_examples_per_batch: Maximum examples per batch (default: 50)
        
    Returns:
        Dictionary mapping sentence_id to has_spoiler (True/False)
    """
    if not examples:
        return {}
    
    # Process in batches if needed
    if len(examples) <= max_examples_per_batch:
        return _check_spoilers_batch(
            examples=examples,
            provider=provider,
            cache=cache,
            book_name=book_name,
        )
    
    # Process in batches
    logger.info(f"Processing {len(examples)} examples for spoiler check in batches of {max_examples_per_batch}")
    
    all_flags = {}
    num_batches = (len(examples) + max_examples_per_batch - 1) // max_examples_per_batch
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_examples_per_batch
        end_idx = min(start_idx + max_examples_per_batch, len(examples))
        batch_examples = examples[start_idx:end_idx]
        
        logger.debug(f"Processing spoiler check batch {batch_idx + 1}/{num_batches} ({len(batch_examples)} examples)")
        
        batch_flags = _check_spoilers_batch(
            examples=batch_examples,
            provider=provider,
            cache=cache,
            book_name=book_name,
        )
        
        all_flags.update(batch_flags)
    
    return all_flags


def _check_spoilers_batch(
    examples: list[tuple[int, str]],
    provider: LLMProvider,
    cache: ResponseCache,
    book_name: str,
) -> dict[int, bool]:
    """Check spoilers for a batch of examples.
    
    Args:
        examples: List of (sentence_id, sentence) tuples
        provider: LLM provider
        cache: Response cache
        book_name: Name of the book
        
    Returns:
        Dictionary mapping sentence_id to has_spoiler (True/False)
    """
    # Format examples for prompt
    examples_numbered = "\n".join(
        f"{idx + 1}. {sentence}" for idx, (_, sentence) in enumerate(examples)
    )
    
    prompt = SPOILER_CHECK_PROMPT.format(
        book_name=book_name,
        examples_numbered=examples_numbered,
    )
    
    # Check cache first
    model_name = getattr(provider, "model", "unknown-model")
    temperature = getattr(provider, "temperature", 0.0)
    cache_key = cache.generate_key(model_name, prompt, temperature)
    cached_response = cache.get(cache_key)
    
    if cached_response:
        logger.debug(f"Cache hit for spoiler check ({len(examples)} examples)")
        try:
            response_data = json.loads(cached_response.content)
            return _parse_spoiler_response(response_data, examples)
        except Exception as e:
            logger.warning(f"Failed to parse cached spoiler response: {e}, retrying")
    
    # Use universal retry utility
    try:
        response = call_llm_with_retry(
            provider=provider,
            prompt=prompt,
            cache=None,  # Don't use cache here, we handle it above
            max_retries=3,
            retry_delay=1.0,
            validate_json=True,
            on_retry=lambda attempt, error: logger.debug(
                f"Retry {attempt} for spoiler check: {error}"
            ),
        )
        
        # Cache the response manually
        try:
            cache.set(cache_key, response)
        except Exception as e:
            logger.warning(f"Failed to cache spoiler response: {e}")
        
        # Parse response
        response_data = json.loads(response.content)
        return _parse_spoiler_response(response_data, examples)
        
    except (ValueError, json.JSONDecodeError, Exception) as e:
        # All retries failed - use conservative approach (mark all as spoilers)
        logger.error(f"All retries failed for spoiler check: {e}, marking all as spoilers")
        return {sid: True for sid, _ in examples}


def _parse_spoiler_response(
    response_data: dict,
    examples: list[tuple[int, str]],
) -> dict[int, bool]:
    """Parse spoiler check response from LLM.
    
    Args:
        response_data: JSON response from LLM with "has_spoiler" list
        examples: List of (sentence_id, sentence) tuples
        
    Returns:
        Dictionary mapping sentence_id to has_spoiler (True/False)
    """
    has_spoiler_list = response_data.get("has_spoiler", [])
    
    if len(has_spoiler_list) != len(examples):
        logger.warning(
            f"Spoiler response length mismatch: expected {len(examples)}, got {len(has_spoiler_list)}"
        )
        # Use conservative approach: mark all as spoilers if mismatch
        return {sid: True for sid, _ in examples}
    
    flags = {}
    for idx, (sid, _) in enumerate(examples):
        flags[sid] = bool(has_spoiler_list[idx])
    
    return flags


def select_examples_for_generation(
    all_examples: list[tuple[int, str]],  # All examples (unfiltered)
    length_flags: dict[int, bool],  # sentence_id -> is_appropriate_length
    spoiler_flags: dict[int, bool],  # sentence_id -> has_spoiler
    target_count: int = 3,
) -> dict[str, Any]:
    """Select examples for generation based on simple logic and flags.
    
    Logic:
    - Use only examples with is_appropriate_length=True and has_spoiler=False
    - Deduplicate: drop duplicate sentences
    - If 3+ such examples: take 2 from book + generate 1
    - If 1-2 such examples: take all + generate the rest up to 3
    - If none: generate 3
    
    Args:
        all_examples: List of all (sentence_id, sentence) tuples (not filtered)
        length_flags: Dictionary mapping sentence_id to is_appropriate_length (True/False)
        spoiler_flags: Dictionary mapping sentence_id to has_spoiler (True/False)
        target_count: Target number of examples (default: 3)
        
    Returns:
        Dictionary with:
        - selected_from_book: List of (sentence_id, sentence) to use from book
        - generate_count: Number of examples to generate (always 1, 2, or 3)
        - flags: Dictionary with all flags for future reference
    """
    # Keep only examples with correct flags
    valid_examples = [
        (sid, ex) for sid, ex in all_examples
        if length_flags.get(sid, False) and not spoiler_flags.get(sid, True)
    ]
    
    # Deduplicate: drop duplicate sentences (same text, different sentence_id)
    seen_sentences = set()
    deduplicated_examples = []
    for sid, ex in valid_examples:
        # Normalize for comparison (lowercase, strip)
        normalized = ex.lower().strip()
        if normalized not in seen_sentences:
            seen_sentences.add(normalized)
            deduplicated_examples.append((sid, ex))
    
    valid_examples = deduplicated_examples
    
    count = len(valid_examples)
    
    if count >= 3:
        # 3+ good examples: take 2 from book + generate 1
        return {
            "selected_from_book": valid_examples[:2],
            "generate_count": 1,
            "flags": {
                "length": length_flags,
                "spoiler": spoiler_flags,
            },
        }
    elif count >= 1:
        # 1-2 good examples: take all + generate rest up to 3
        return {
            "selected_from_book": valid_examples,
            "generate_count": target_count - count,  # 1 or 2
            "flags": {
                "length": length_flags,
                "spoiler": spoiler_flags,
            },
        }
    else:
        # No good examples: generate 3
        return {
            "selected_from_book": [],
            "generate_count": target_count,  # 3
            "flags": {
                "length": length_flags,
                "spoiler": spoiler_flags,
            },
        }


def validate_example_length(example: str, max_words: int = 50) -> tuple[bool, str]:
    """Validate example length.
    
    Args:
        example: Example sentence.
        max_words: Maximum allowed words (default: 50).
        
    Returns:
        Tuple of (is_valid, reason). reason is empty if valid.
    """
    word_count = count_words(example)
    if word_count > max_words:
        return False, f"too_long_{word_count}_words"
    return True, ""


def format_card_prompt(
    lemma: str,
    pos: str,
    supersense: str,
    wn_definition: str,
    book_name: str,
    examples: list[str],
    synset_group: list[str] | None = None,
    primary_synset: str = "",
    additional_instruction: str = "",
    generate_count: int = 0,
) -> str:
    """Format the prompt template with actual data.

    Args:
        lemma: Word lemma.
        pos: Part of speech.
        supersense: WordNet supersense category.
        wn_definition: WordNet definition.
        book_name: Name of the source book.
        examples: List of example sentences (already filtered for length and spoilers).
        synset_group: Optional list of synset_ids merged into this card.
        primary_synset: Main synset_id for definition.
        additional_instruction: Optional additional instruction for retry.
        generate_count: Number of examples to generate (0, 1, 2, or 3).

    Returns:
        Formatted prompt string.
    """
    examples_numbered = "\n".join(
        f"{i+1}. \"{ex}\"" for i, ex in enumerate(examples)
    )

    # Format synset_group_info
    if synset_group and len(synset_group) > 1:
        synset_group_info = f"- Synset group: {', '.join(synset_group)}\n- Primary synset: {primary_synset or synset_group[0]}"
    elif primary_synset:
        synset_group_info = f"- Synset: {primary_synset}"
    else:
        synset_group_info = ""

    prompt = SMART_CARD_PROMPT.format(
        lemma=lemma,
        pos=pos.lower(),
        supersense=supersense,
        wn_definition=wn_definition,
        book_name=book_name,
        examples_numbered=examples_numbered,
        synset_group_info=synset_group_info,
        generate_count=generate_count,
    )
    
    # Add additional instruction if provided (for retry with specific issue)
    if additional_instruction:
        prompt += f"\n\n## CRITICAL: {additional_instruction}"
    
    return prompt


# =============================================================================
# SmartCard Dataclass
# =============================================================================


@dataclass
class SmartCard:
    """Generated flashcard with LLM-verified quality.

    Attributes:
        lemma: Base form of the word.
        pos: Part of speech (noun, verb, adj, adv).
        supersense: WordNet supersense category.
        selected_examples: Best example sentences from book.
        excluded_examples: Examples with wrong sense (WSD errors).
        simple_definition: Simple English definition (B1-B2 level).
        translation_ru: Russian translation for this meaning.
        generated_example: LLM-generated example sentence (first from generated_examples).
        generated_examples: List of LLM-generated example sentences.
        quality_scores: Quality scores for examples (1-based index -> score 1-5).
        wn_definition: Original WordNet definition.
        book_name: Source book name.
        synset_group: List of synset_ids that were merged into this card.
        primary_synset: Main synset_id for definition.
    """

    lemma: str
    pos: str
    supersense: str
    selected_examples: list[str]
    excluded_examples: list[str]
    simple_definition: str
    translation_ru: str
    generated_example: str
    wn_definition: str
    book_name: str
    synset_group: list[str] = field(default_factory=list)
    primary_synset: str = ""
    quality_scores: dict[int, int] = field(default_factory=dict)
    generated_examples: list[str] = field(default_factory=list)
    skip_reason: str = ""  # Reason for skipping card (e.g., "no_examples_from_book")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_anki_row(self) -> dict[str, str]:
        """Convert to Anki CSV row format.

        Returns:
            Dict with 'front', 'back', and 'tags' fields for Anki.
        """
        # Front: word with example
        examples_text = " | ".join(self.selected_examples[:2])
        front = f"<b>{self.lemma}</b> ({self.pos})<br><br><i>{examples_text}</i>"

        # Back: definition, translation, generated example
        back = (
            f"<b>{self.simple_definition}</b><br><br>"
            f"🇷🇺 {self.translation_ru}<br><br>"
            f"💡 {self.generated_example}"
        )

        # Tags: book name + supersense
        tags = f"{self.book_name} {self.supersense.replace('.', '_')}"

        return {"front": front, "back": back, "tags": tags}


# =============================================================================
# SmartCardGenerator
# =============================================================================


@dataclass
class GeneratorStats:
    """Statistics for card generation."""

    total_cards: int = 0
    successful: int = 0
    failed: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    cache_hits: int = 0


class SmartCardGenerator:
    """LLM-powered flashcard generator with quality control.

    Uses LLM to:
    - Select best examples from book
    - Detect WSD errors
    - Generate simple definitions and translations
    - Create additional examples

    Args:
        provider: LLM provider instance.
        cache: Response cache instance.
        book_name: Name of the source book.
        max_retries: Max retries on JSON parse errors.
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache: ResponseCache,
        book_name: str,
        max_retries: int = 2,
    ):
        self._provider = provider
        self._cache = cache
        self.book_name = book_name
        self.max_retries = max_retries
        self._stats = GeneratorStats()

        # Store cache for use with call_llm_with_retry
        self._cache = cache

    def generate_card(
        self,
        lemma: str,
        pos: str,
        supersense: str,
        wn_definition: str,
        examples: list[str],
        synset_group: list[str] | None = None,
        primary_synset: str = "",
        existing_synsets: set[str] | None = None,
        generate_count: int = 0,
    ) -> SmartCard | None:
        """Generate a single flashcard.

        Args:
            lemma: Word lemma.
            pos: Part of speech.
            supersense: WordNet supersense category.
            wn_definition: WordNet definition.
            examples: Example sentences from book (already filtered for length and spoilers).
            synset_group: Optional list of synset_ids merged into this card.
            primary_synset: Main synset_id for definition.
            existing_synsets: Synsets already in deck (to avoid duplicates, not used for fallback).
            generate_count: Number of examples to generate (0, 1, 2, or 3).

        Returns:
            SmartCard if successful, None if failed after retries.
        """
        self._stats.total_cards += 1

        prompt = format_card_prompt(
            lemma=lemma,
            pos=pos,
            supersense=supersense,
            wn_definition=wn_definition,
            book_name=self.book_name,
            examples=examples[:10],  # Limit to 10 examples
            synset_group=synset_group,
            primary_synset=primary_synset,
            generate_count=generate_count,
        )

        # Try to generate with retries and validation
        card = self._try_generate_with_validation(
            prompt=prompt,
            lemma=lemma,
            pos=pos,
            supersense=supersense,
            wn_definition=wn_definition,
            examples=examples,
            synset_group=synset_group,
            primary_synset=primary_synset,
            generate_count=generate_count,
        )

        if card is None:
            # LLM failed completely (invalid JSON etc)
            self._stats.failed += 1
            return None

        # Check if card should be skipped (no examples from book)
        if not card.selected_examples:
            card.skip_reason = "no_examples_from_book"
            logger.debug(f"Skipping card for '{lemma}': no examples from book")
            self._stats.failed += 1
            return None

        # Card generated successfully with examples from book
        self._stats.successful += 1
        return card

    def _validate_examples(self, examples: list[str]) -> tuple[list[str], list[str]]:
        """Validate examples for length and spoilers.
        
        Args:
            examples: List of example sentences.
            
        Returns:
            Tuple of (valid_examples, issues). issues is list of issue descriptions.
        """
        valid_examples = []
        issues = []
        
        for ex in examples:
            # Check length
            is_valid, reason = validate_example_length(ex, max_words=50)
            if not is_valid:
                issues.append(f"Example too long: {reason}")
                continue
            
            # TODO: Add spoiler detection (can use LLM or keyword-based)
            # For now, we trust LLM's spoiler detection from prompt
            
            valid_examples.append(ex)
        
        return valid_examples, issues
    
    def _try_generate_with_validation(
        self,
        prompt: str,
        lemma: str,
        pos: str,
        supersense: str,
        wn_definition: str,
        examples: list[str],
        synset_group: list[str] | None,
        primary_synset: str,
        generate_count: int = 0,
    ) -> SmartCard | None:
        """Try to generate a card with retries and validation.
        
        Uses call_llm_with_retry for API error handling.
        Validates examples after generation and retries with improved prompt if needed.

        Returns SmartCard if LLM returns valid response, None if all retries failed.
        """
        max_validation_retries = 2  # Additional retries for validation issues
        current_prompt = prompt
        
        for validation_attempt in range(max_validation_retries + 1):
            try:
                # Use call_llm_with_retry for robust API error handling
                response = call_llm_with_retry(
                    provider=self._provider,
                    prompt=current_prompt,
                    cache=self._cache,
                    max_retries=self.max_retries,
                    retry_delay=1.0,
                    validate_json=True,
                )

                self._stats.total_tokens += response.total_tokens
                self._stats.total_cost += response.cost_usd

                # Parse JSON response
                result = self._parse_response(response.content)

                # Build SmartCard
                card = self._build_card(
                    lemma=lemma,
                    pos=pos,
                    supersense=supersense,
                    wn_definition=wn_definition,
                    examples=examples,
                    llm_result=result,
                    synset_group=synset_group,
                    primary_synset=primary_synset,
                )
                
                # Validate examples after generation
                all_examples = card.selected_examples + card.generated_examples
                valid_examples, issues = self._validate_examples(all_examples)
                
                # Check if we have validation issues
                if issues and validation_attempt < max_validation_retries:
                    # Build improved prompt with specific issue
                    issue_text = "; ".join(issues[:3])  # Limit to first 3 issues
                    additional_instruction = ""
                    
                    if "too_long" in issue_text:
                        additional_instruction = (
                            "CRITICAL: Some examples are TOO LONG (>50 words). "
                            "You MUST mark them as invalid in invalid_indices. "
                            "DO NOT select examples longer than 50 words."
                        )
                    elif "spoiler" in issue_text.lower():
                        additional_instruction = (
                            "CRITICAL: Some examples contain SPOILERS. "
                            "You MUST mark them as invalid in invalid_indices. "
                            "DO NOT select examples that reveal plot points or story endings."
                        )
                    
                    if additional_instruction:
                        # Retry with improved prompt
                        current_prompt = format_card_prompt(
                            lemma=lemma,
                            pos=pos,
                            supersense=supersense,
                            wn_definition=wn_definition,
                            book_name=self.book_name,
                            examples=examples[:10],
                            synset_group=synset_group,
                            primary_synset=primary_synset,
                            additional_instruction=additional_instruction,
                            generate_count=generate_count,
                        )
                        logger.warning(
                            f"Validation issues for '{lemma}': {issue_text}. Retrying with improved prompt."
                        )
                        continue
                
                # Card is valid or we've exhausted retries
                return card

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(
                    f"Attempt {validation_attempt + 1}/{max_validation_retries + 1} failed for '{lemma}': {e}"
                )
                if validation_attempt >= max_validation_retries:
                    break
            except Exception as e:
                # API errors, network errors, etc. are handled by call_llm_with_retry
                # But if it still fails after all retries, we catch it here
                logger.error(f"Failed to generate card for '{lemma}': {e}")
                if validation_attempt >= max_validation_retries:
                    break

        logger.error(f"Failed to generate card for '{lemma}' after {max_validation_retries + 1} validation attempts")
        return None

    def _parse_response(self, content: str) -> dict:
        """Parse LLM response as JSON.

        Args:
            content: Raw response content.

        Returns:
            Parsed JSON dict.

        Raises:
            json.JSONDecodeError: If not valid JSON.
            ValueError: If missing required fields.
        """
        # Handle markdown code blocks
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        result = json.loads(content.strip())

        # Validate required fields (new JSON schema)
        required = ["selected_indices", "simple_definition", "translation_ru"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Ensure optional fields have defaults
        if "valid_indices" not in result:
            result["valid_indices"] = result.get("selected_indices", [])
        if "invalid_indices" not in result:
            result["invalid_indices"] = []
        if "quality_scores" not in result:
            result["quality_scores"] = {}
        if "generated_examples" not in result:
            # Backward compatibility: check for old "generated_example" field
            if "generated_example" in result:
                result["generated_examples"] = [result["generated_example"]]
            else:
                result["generated_examples"] = []

        return result

    def _build_card(
        self,
        lemma: str,
        pos: str,
        supersense: str,
        wn_definition: str,
        examples: list[str],
        llm_result: dict,
        synset_group: list[str] | None = None,
        primary_synset: str = "",
    ) -> SmartCard:
        """Build SmartCard from LLM result.

        Args:
            lemma: Word lemma.
            pos: Part of speech.
            supersense: Supersense category.
            wn_definition: WordNet definition.
            examples: Original examples list.
            llm_result: Parsed LLM JSON response.
            synset_group: List of synset_ids merged into this card.
            primary_synset: Main synset_id for definition.

        Returns:
            SmartCard instance.
        """
        # Convert 1-based indices to examples
        selected_indices = llm_result.get("selected_indices", [])
        invalid_indices = llm_result.get("invalid_indices", [])

        selected_examples = [
            examples[i - 1]
            for i in selected_indices
            if 1 <= i <= len(examples)
        ]

        excluded_examples = [
            examples[i - 1]
            for i in invalid_indices
            if 1 <= i <= len(examples)
        ]

        # Parse quality_scores (convert string keys to int if needed)
        quality_scores = llm_result.get("quality_scores", {})
        quality_scores_int = {}
        for k, v in quality_scores.items():
            try:
                quality_scores_int[int(k)] = int(v)
            except (ValueError, TypeError):
                pass

        # Get generated_examples
        generated_examples = llm_result.get("generated_examples", [])
        if not generated_examples and "generated_example" in llm_result:
            # Backward compatibility
            generated_examples = [llm_result["generated_example"]]

        # Set generated_example to first from generated_examples (for backward compatibility)
        generated_example = generated_examples[0] if generated_examples else ""

        return SmartCard(
            lemma=lemma,
            pos=pos,
            supersense=supersense,
            selected_examples=selected_examples,
            excluded_examples=excluded_examples,
            simple_definition=llm_result["simple_definition"],
            translation_ru=llm_result["translation_ru"],
            generated_example=generated_example,
            generated_examples=generated_examples,
            quality_scores=quality_scores_int,
            wn_definition=wn_definition,
            book_name=self.book_name,
            synset_group=list(synset_group) if synset_group is not None else [],
            primary_synset=primary_synset,
            skip_reason="",
        )

    def _generate_example_fallback(
        self,
        lemma: str,
        pos: str,
        definition: str,
    ) -> str:
        """Generate an example sentence via LLM when no examples available.

        This is used as a fallback when:
        - No selected_examples
        - No excluded_examples (nothing to redistribute)
        - No generated_example (from initial LLM call)

        Args:
            lemma: Word lemma
            pos: Part of speech
            definition: Word definition

        Returns:
            Generated example sentence, or empty string if failed
        """
        prompt = EXAMPLE_GENERATION_PROMPT.format(
            lemma=lemma,
            pos=pos.lower(),
            definition=definition,
        )

        try:
            response = call_llm_with_retry(
                provider=self._provider,
                prompt=prompt,
                cache=self._cache,
                max_retries=self.max_retries,
                retry_delay=1.0,
            )

            # Extract example from response (should be plain text, not JSON)
            example = response.content.strip()

            # Remove quotes if present
            if example.startswith('"') and example.endswith('"'):
                example = example[1:-1]
            if example.startswith("'") and example.endswith("'"):
                example = example[1:-1]

            # Validate: check if lemma or its forms are in the example
            from eng_words.validation.example_validator import _get_word_forms

            lemma_forms = _get_word_forms(lemma)
            example_lower = example.lower()

            # Check if any form of the lemma appears in the example
            form_found = any(form.lower() in example_lower for form in lemma_forms)

            if form_found:
                return example

            logger.warning(
                f"Generated example for '{lemma}' doesn't contain the lemma or its forms: {example[:50]}..."
            )
            # Return anyway - validation will catch it, but at least we have something
            return example

        except Exception as e:
            logger.warning(f"Failed to generate fallback example for '{lemma}': {e}")
            return ""

    def generate_batch(
        self,
        items: list[dict],
        progress: bool = True,
    ) -> list[SmartCard]:
        """Generate multiple flashcards.

        Args:
            items: List of dicts with keys: lemma, pos, supersense, wn_definition, examples.
            progress: Whether to show progress.

        Returns:
            List of successfully generated SmartCards.
        """
        cards = []
        total = len(items)

        # Track synsets already in the deck to avoid duplicates
        existing_synsets: set[str] = set()

        for i, item in enumerate(items):
            if progress and (i + 1) % 10 == 0:
                logger.info(f"Generating cards: {i + 1}/{total}")

            card = self.generate_card(
                lemma=item["lemma"],
                pos=item["pos"],
                supersense=item["supersense"],
                wn_definition=item["wn_definition"],
                examples=item["examples"],
                synset_group=item.get("synset_group"),
                primary_synset=item.get("primary_synset", ""),
                existing_synsets=existing_synsets,
            )

            if card is not None:
                cards.append(card)
                # Track synset for duplicate avoidance
                if card.primary_synset:
                    existing_synsets.add(card.primary_synset)

        if progress:
            logger.info(f"Generated {len(cards)}/{total} cards")

        return cards

    def stats(self) -> dict:
        """Get generation statistics.

        Returns:
            Dict with total_cards, successful, failed, total_tokens, total_cost.
        """
        return {
            "total_cards": self._stats.total_cards,
            "successful": self._stats.successful,
            "failed": self._stats.failed,
            "total_tokens": self._stats.total_tokens,
            "total_cost": self._stats.total_cost,
            "cache_stats": self._cache.stats(),
        }

