"""Prompts for WSD Gold labeling.

This module provides prompt templates and parsing functions
for LLM-based WSD gold labeling.
"""

import json
import logging

from eng_words.wsd_gold.models import GoldExample, LLMUsage, ModelOutput

logger = logging.getLogger(__name__)

# Prompt version for tracking
PROMPT_VERSION_GOLD_LABELING = "v1.1"

# Static prefix (should be >= 1024 tokens for prompt caching)
# This prefix is identical for all examples to enable caching
GOLD_LABELING_SYSTEM_PROMPT = """You are an expert linguist and lexicographer performing Word Sense Disambiguation (WSD).

Your task is to analyze English sentences and determine which WordNet synset (word sense) best matches how a target word is used in context. This is a critical task for building high-quality linguistic resources.

## Background on WordNet

WordNet is a lexical database that groups English words into sets of synonyms called synsets. Each synset represents a distinct concept and has:
- A unique identifier (e.g., "bank.n.01")
- A definition (gloss) explaining the meaning
- Example sentences showing usage
- Relationships to other synsets (hypernyms, hyponyms, etc.)

The synset ID format is: lemma.pos.sense_number
- lemma: the base form of the word
- pos: part of speech (n=noun, v=verb, a=adjective, r=adverb)
- sense_number: distinguishes different meanings (01, 02, etc.)

## Your Task

Given a sentence with a highlighted target word, you must:
1. Carefully read and understand the full context
2. Analyze all candidate synsets provided
3. Select the synset whose definition BEST matches the word's meaning in context
4. Assign a confidence score based on how certain you are
5. Add appropriate flags if needed

## Decision Rules

### Rule 1: Context is King
The surrounding words, the topic of discussion, and the overall meaning of the sentence determine which sense is correct. Always prioritize contextual evidence.

### Rule 2: Consider All Candidates
Do not jump to conclusions. Read ALL candidate definitions before making a choice. Sometimes the first candidate seems right but a later one is actually better.

### Rule 3: Literal vs. Figurative
Pay attention to metaphorical, idiomatic, or figurative uses:
- "He's drowning in debt" → figurative use of "drown"
- "She broke the ice" → could be literal or idiomatic
- "Time flies" → figurative use of "fly"
Use the "metaphor" flag when the word is used figuratively.

### Rule 4: Multiword Expressions
Some words form part of a larger expression (phrasal verbs, idioms, collocations):
- "give up" (phrasal verb) vs "give" (simple verb)
- "break down" vs "break"
- "look after" vs "look"
Use the "multiword" flag when the target is part of a multiword expression.

### Rule 5: Insufficient Context
Sometimes the context doesn't provide enough information:
- "The bank was closed." (which bank? river bank or financial institution?)
- "She saw a bat." (animal or sports equipment?)
Use "needs_more_context" flag when you cannot be highly confident.

### Rule 6: None of the Above
If NONE of the candidate synsets match the word's meaning in context, use "none_of_the_above" flag with an empty synset_id. This is rare but important to identify.

## Output Format

You MUST return a valid JSON object with exactly these fields:

```json
{
  "chosen_synset_id": "word.pos.NN",
  "confidence": 0.85,
  "flags": []
}
```

### Field Descriptions:

**chosen_synset_id** (string, required):
- Must be one of the synset IDs from the candidates list
- Use empty string "" only when using "none_of_the_above" flag

**confidence** (float, required):
- A number between 0.0 and 1.0
- 0.9-1.0: Very confident, context clearly indicates this sense
- 0.7-0.9: Confident, strong contextual evidence
- 0.5-0.7: Somewhat confident, reasonable guess
- 0.3-0.5: Low confidence, context is ambiguous
- 0.0-0.3: Very uncertain, essentially guessing

**flags** (array of strings):
- Valid flags: "needs_more_context", "multiword", "metaphor", "none_of_the_above"
- Use empty array [] when no flags apply
- Multiple flags can be used together

## Examples

### Example 1: Clear Financial Context
Sentence: "After receiving her paycheck, she went to the bank to deposit the money into her savings account."
Target: "bank" (NOUN)
Candidates:
A) bank.n.01 - sloping land (especially the slope beside a body of water)
B) depository_financial_institution.n.01 - a financial institution that accepts deposits and channels the money into lending activities
C) bank.n.03 - a long ridge or pile

Analysis: The context mentions "paycheck", "deposit", and "savings account" - all financial terms. This clearly indicates a financial institution.

Answer:
```json
{"chosen_synset_id": "depository_financial_institution.n.01", "confidence": 0.98, "flags": []}
```

### Example 2: Metaphorical Usage
Sentence: "The startup was drowning in technical debt after years of rushed development."
Target: "drowning" (VERB)
Candidates:
A) drown.v.01 - cover completely or make imperceptible
B) drown.v.02 - die from being submerged in water, get killed by lack of oxygen in water
C) drown.v.03 - kill by submerging in water

Analysis: A startup cannot literally drown. "Drowning in technical debt" is a metaphor meaning overwhelmed by accumulated problems. drown.v.01 "cover completely" captures this figurative sense.

Answer:
```json
{"chosen_synset_id": "drown.v.01", "confidence": 0.92, "flags": ["metaphor"]}
```

### Example 3: Ambiguous Context
Sentence: "I saw the bat near the old building."
Target: "bat" (NOUN)
Candidates:
A) bat.n.01 - nocturnal mouselike mammal with forelimbs modified to form membranous wings
B) bat.n.02 - a club used for hitting a ball in various games
C) bat.n.05 - a turn at batting in baseball

Analysis: "Near the old building" could apply to either an animal bat (flying around) or a sports bat (left outside). Without more context, I'll choose the animal interpretation as it's more natural to say an animal is "near" something, but confidence is reduced.

Answer:
```json
{"chosen_synset_id": "bat.n.01", "confidence": 0.55, "flags": ["needs_more_context"]}
```

### Example 4: Phrasal Verb
Sentence: "After struggling for months, she finally gave up smoking."
Target: "gave" (VERB)
Candidates:
A) give.v.01 - cause to have, in the abstract sense or physical sense
B) give.v.03 - transfer possession of something concrete or abstract to somebody
C) give.v.10 - give up (in phrases)

Analysis: "Gave up" is a phrasal verb meaning "to stop doing something". The target word "gave" is part of this multiword expression. give.v.10 specifically mentions "give up" phrases.

Answer:
```json
{"chosen_synset_id": "give.v.10", "confidence": 0.95, "flags": ["multiword"]}
```

### Example 5: Technical Domain
Sentence: "The kernel module failed to load because of a memory allocation error in the heap."
Target: "heap" (NOUN)
Candidates:
A) heap.n.01 - a collection of objects laid on top of each other
B) heap.n.02 - (often followed by of) a large number or amount or extent
C) heap.n.03 - a car that is old and unreliable

Analysis: Technical context with "kernel module", "memory allocation" indicates computer science domain. In computing, "heap" refers to dynamically allocated memory. However, if no computing-specific synset is available, heap.n.01 (collection) is the closest general meaning.

Answer:
```json
{"chosen_synset_id": "heap.n.01", "confidence": 0.70, "flags": []}
```

## Important Notes

1. ALWAYS return valid JSON - no additional text before or after
2. The synset_id must EXACTLY match one from the candidates list
3. Be consistent in your reasoning across similar examples
4. When uncertain, lower your confidence score rather than picking randomly
5. Flags help us understand your reasoning - use them appropriately

Now analyze the following:
"""


def build_gold_labeling_prompt(example: GoldExample) -> str:
    """Build a prompt for gold labeling.

    Args:
        example: GoldExample to label

    Returns:
        Full prompt string
    """
    # Format candidates
    candidate_lines = []
    for i, candidate in enumerate(example.candidates):
        letter = chr(ord("A") + i)
        examples_str = ""
        if candidate.examples:
            examples_str = f' (e.g., "{candidate.examples[0]}")'
        candidate_lines.append(f"{letter}) {candidate.synset_id} - {candidate.gloss}{examples_str}")
    candidates_text = "\n".join(candidate_lines)

    # Build dynamic part
    dynamic_prompt = f"""
## Input

**Sentence:** "{example.context_window}"

**Target word:** "{example.target.lemma}" ({example.target.pos})

**Candidates:**
{candidates_text}

## Your Answer (JSON only):
"""

    return GOLD_LABELING_SYSTEM_PROMPT + dynamic_prompt


def parse_gold_label_response(
    raw_text: str,
    candidates: list[str],
    usage: LLMUsage,
) -> ModelOutput | None:
    """Parse LLM response into ModelOutput.

    Args:
        raw_text: Raw LLM response text
        candidates: List of valid synset IDs
        usage: Token usage information

    Returns:
        ModelOutput if valid, None otherwise
    """
    try:
        # Try to extract JSON from response
        text = raw_text.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()

        data = json.loads(text)

        chosen_synset_id = data.get("chosen_synset_id", "")
        confidence = float(data.get("confidence", 0.0))
        flags = data.get("flags", [])

        # Validate flags
        valid_flags = {"needs_more_context", "multiword", "metaphor", "none_of_the_above"}
        invalid_flags = [f for f in flags if f not in valid_flags]
        if invalid_flags:
            logger.debug(f"Invalid flags removed: {invalid_flags}")
        flags = [f for f in flags if f in valid_flags]

        # Validate confidence range
        if confidence < 0.0 or confidence > 1.0:
            logger.warning(f"Confidence {confidence} out of range [0, 1], raw: {raw_text[:100]}")
            return None

        # Create output
        output = ModelOutput(
            chosen_synset_id=chosen_synset_id,
            confidence=confidence,
            flags=flags,
            raw_text=raw_text,
            usage=usage,
        )

        # Validate synset_id
        if not output.is_valid(candidates):
            logger.debug(f"Invalid synset_id '{chosen_synset_id}' not in candidates: {candidates}")
            return None

        return output

    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e}, raw: {raw_text[:200]}")
        return None
    except ValueError as e:
        logger.debug(f"Value error: {e}, raw: {raw_text[:200]}")
        return None
    except (KeyError, TypeError) as e:
        logger.debug(f"Data extraction error: {e}, raw: {raw_text[:200]}")
        return None
