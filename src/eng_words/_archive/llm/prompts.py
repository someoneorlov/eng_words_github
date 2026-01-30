"""Prompt templates for LLM integration.

All prompts are versioned for reproducibility and cache invalidation.
"""

from typing import Any

from eng_words.constants.llm_config import (
    PROMPT_VERSION_CARD_GENERATION,
    PROMPT_VERSION_EVALUATION,
)

# =============================================================================
# WSD Evaluation Prompts
# =============================================================================

WSD_EVALUATION_SYSTEM = """You are a linguistics expert evaluating word sense disambiguation.
Your task is to determine which sense (A, B, C, etc.) best fits the target word in context.

CRITICAL RULES:
1. Focus on the EXACT meaning of the word in THIS specific context.
2. Don't be fooled by similar-sounding definitions - check if the meaning actually applies.
3. Watch for idioms and phrasal verbs (e.g., "go through" vs "go").
4. Consider the grammatical role (is "air" used as atmosphere or as manner/demeanor?).
5. If the assigned sense is clearly WRONG for the context, say so confidently.
6. Only use "uncertain" if two senses are genuinely indistinguishable in context.
"""

WSD_EVALUATION_PROMPT_TEMPLATE = """# Word Sense Disambiguation Evaluation

**Prompt Version:** {prompt_version}

## Task
Given a sentence and multiple candidate senses for the target word, identify which sense (A, B, C, ...) best matches the usage in the sentence.

## Examples

### Example 1 - CORRECT match
Sentence: "She drove her **car** to work."
Target: car (NOUN)
Candidates:
A. a motor vehicle with four wheels
B. a conveyance for passengers on a cable railway
C. a wheeled vehicle adapted to rails of railroad

Answer: {{"choice": "A", "reasoning": "drove indicates a motor vehicle, not cable car or train"}}

### Example 2 - INCORRECT match (should detect error)
Sentence: "He assumed a jocular **air** despite his worries."
Target: air (NOUN)
Candidates:
A. a slight wind (usually refreshing)
B. a distinctive but intangible quality surrounding a person
C. a mixture of gases required for breathing

Answer: {{"choice": "B", "reasoning": "'assumed an air' means adopting a manner/demeanor, not wind"}}

### Example 3 - Phrasal verb trap
Sentence: "Are you **going** to open your present?"
Target: go (VERB)
Candidates:
A. go through in search of something; search belongings
B. change location; move, travel
C. follow a procedure or take a course

Answer: {{"choice": "B", "reasoning": "'going to' is future tense auxiliary, closest to movement sense"}}

---

## Your Evaluation

## Sentence
"{sentence}"

## Target Word
**{lemma}** (POS: {pos})

## Candidate Senses
{candidates}

## Instructions
1. Read the sentence and identify the EXACT meaning of the target word.
2. Match it to the BEST candidate sense.
3. Be critical - if a sense doesn't fit the context, don't choose it just because words overlap.

## Response Format
Respond with ONLY this JSON (no markdown, no explanation):
{{"choice": "A", "reasoning": "brief one-line reasoning"}}

## Your Response:
"""


def format_candidates(candidates: list[dict]) -> str:
    """Format candidate senses for the prompt.

    Args:
        candidates: List of dicts with "synset_id" and "definition" keys.

    Returns:
        Formatted string with labeled candidates (A, B, C, ...).
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for i, candidate in enumerate(candidates):
        letter = letters[i] if i < len(letters) else f"({i+1})"
        definition = candidate.get("definition", "No definition available")
        lines.append(f"**{letter}.** {definition}")
    return "\n".join(lines)


def build_evaluation_prompt(
    sentence: str,
    lemma: str,
    pos: str,
    candidates: list[dict],
) -> str:
    """Build the evaluation prompt for a single WSD sample.

    Args:
        sentence: The sentence containing the target word.
        lemma: The target word (lemma form).
        pos: Part of speech (NOUN, VERB, ADJ, ADV).
        candidates: List of candidate senses with definitions.

    Returns:
        Complete prompt string for LLM.
    """
    candidates_text = format_candidates(candidates)

    return WSD_EVALUATION_PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION_EVALUATION,
        sentence=sentence,
        lemma=lemma,
        pos=pos,
        candidates=candidates_text,
    )


def get_candidate_index(choice: str, candidates: list[dict]) -> int | None:
    """Convert letter choice to candidate index.

    Args:
        choice: Letter choice (A, B, C, ...) or "uncertain".
        candidates: List of candidates.

    Returns:
        Index of the chosen candidate, or None if uncertain or invalid.
    """
    if choice.lower() == "uncertain":
        return None

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_upper = choice.upper()

    if choice_upper in letters:
        index = letters.index(choice_upper)
        if index < len(candidates):
            return index

    return None


# =============================================================================
# Card Generation Prompts
# =============================================================================

CARD_GENERATION_SYSTEM = """You are helping create English learning flashcards for a Russian speaker (B1-B2 level).

Your task is to:
1. Create simple, clear definitions (not dictionary-style)
2. Provide accurate Russian translations
3. Select best examples from the book (avoiding spoilers!)
4. Generate simple generic examples

SPOILER POLICY (CRITICAL):
- "spoiler" = deaths, plot twists, reveals, character fates, relationship secrets
- For each book example, rate spoiler_risk: "none" | "low" | "medium" | "high"
- When in doubt, mark as "medium" or "high"
- We ONLY use "none" in final flashcards
"""

CARD_GENERATION_PROMPT_TEMPLATE = """# Flashcard Generation

**Prompt Version:** {prompt_version}

## Task
For each word sense below, provide:
1. **Simple definition** (B1 level English, 10-15 words max, NOT dictionary-style)
2. **Russian translation** (most common usage)
3. **Best 2-3 book examples** with spoiler_risk rating
4. **1-2 generic examples** (simple, universal, not from book)

---

## Word Senses

{senses_block}

---

## Response Format
Return a JSON array with one object per sense:

```json
[
  {{
    "synset_id": "bank.n.01",
    "definition_simple": "a company where you keep your money safe",
    "translation_ru": "банк",
    "book_examples_selected": [
      {{"text": "He walked into the bank.", "spoiler_risk": "none"}},
      {{"text": "The bank was closed.", "spoiler_risk": "none"}}
    ],
    "generic_examples": [
      "I need to go to the bank to deposit money."
    ]
  }}
]
```

CRITICAL REQUIREMENTS:
- **synset_id**: COPY EXACTLY as provided (e.g., "abbreviated.s.01" NOT "brief.adj.01")
- definition_simple should be SIMPLE and CLEAR, not a dictionary definition
- translation_ru should be the MOST COMMON Russian word for this meaning
- Only include book_examples that clearly demonstrate THIS specific meaning
- Rate spoiler_risk honestly - we filter out anything except "none"
- generic_examples should be simple B1-level sentences

WARNING: synset_id must match EXACTLY what is provided. Do NOT:
- Change the lemma (abbreviated.s.01 stays abbreviated.s.01, not brief.adj.01)
- Expand POS abbreviations (n stays n, not noun)
- Modify the format in any way

## Your Response (JSON only, no markdown):
"""


def format_sense_for_prompt(sense: dict[str, Any]) -> str:
    """Format a single sense for the card generation prompt.

    Args:
        sense: Dict with synset_id, lemma, pos, supersense, definition, book_examples.

    Returns:
        Formatted string for the prompt.
    """
    lines = [
        f"### {sense['lemma']} ({sense['pos']}) — {sense['synset_id']}",
        f"- Supersense: {sense['supersense']}",
        f"- WordNet definition: \"{sense['definition']}\"",
        f"- Book: \"{sense['book_name']}\"",
    ]

    if sense.get("book_examples"):
        lines.append("- Book examples:")
        for i, ex in enumerate(sense["book_examples"], 1):
            lines.append(f'  {i}. "{ex}"')
    else:
        lines.append("- Book examples: (none available)")

    return "\n".join(lines)


def build_card_generation_prompt(senses: list[dict[str, Any]]) -> str:
    """Build the card generation prompt for a batch of senses.

    Args:
        senses: List of sense dicts prepared by prepare_card_batch.

    Returns:
        Complete prompt string for LLM.
    """
    senses_block = "\n\n".join(format_sense_for_prompt(s) for s in senses)

    return CARD_GENERATION_PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION_CARD_GENERATION,
        senses_block=senses_block,
    )
