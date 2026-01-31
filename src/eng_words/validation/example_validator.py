"""Example validation for SmartCards.

Validates that card examples contain the lemma or its synonyms,
ensuring users can see the word in context.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nltk.corpus import wordnet as wn

if TYPE_CHECKING:
    from eng_words.llm.smart_card_generator import SmartCard

logger = logging.getLogger(__name__)


# Irregular forms that aren't captured by simple suffix rules
IRREGULAR_FORMS: dict[str, list[str]] = {
    "bad": ["worse", "worst"],
    "good": ["better", "best"],
    "go": ["went", "gone", "goes"],
    "be": ["was", "were", "been", "am", "is", "are"],
    "have": ["had", "has"],
    "do": ["did", "does", "done"],
    "say": ["said", "says"],
    "get": ["got", "gotten", "gets"],
    "make": ["made", "makes"],
    "see": ["saw", "seen", "sees"],
    "come": ["came", "comes"],
    "take": ["took", "taken", "takes"],
    "know": ["knew", "known", "knows"],
    "think": ["thought", "thinks"],
    "give": ["gave", "given", "gives"],
    "find": ["found", "finds"],
    "tell": ["told", "tells"],
    "become": ["became", "becomes"],
    "leave": ["left", "leaves"],
    "feel": ["felt", "feels"],
    "put": ["puts"],
    "bring": ["brought", "brings"],
    "begin": ["began", "begun", "begins"],
    "keep": ["kept", "keeps"],
    "hold": ["held", "holds"],
    "write": ["wrote", "written", "writes"],
    "stand": ["stood", "stands"],
    "hear": ["heard", "hears"],
    "let": ["lets"],
    "mean": ["meant", "means"],
    "set": ["sets"],
    "meet": ["met", "meets"],
    "run": ["ran", "runs"],
    "pay": ["paid", "pays"],
    "sit": ["sat", "sits"],
    "speak": ["spoke", "spoken", "speaks"],
    "lie": ["lay", "lain", "lies", "lying"],
    "lead": ["led", "leads"],
    "read": ["reads"],
    "grow": ["grew", "grown", "grows"],
    "lose": ["lost", "loses"],
    "fall": ["fell", "fallen", "falls"],
    "send": ["sent", "sends"],
    "build": ["built", "builds"],
    "understand": ["understood", "understands"],
    "draw": ["drew", "drawn", "draws"],
    "break": ["broke", "broken", "breaks"],
    "spend": ["spent", "spends"],
    "cut": ["cuts"],
    "rise": ["rose", "risen", "rises"],
    "arise": ["arose", "arisen", "arises"],
    "drive": ["drove", "driven", "drives"],
    "buy": ["bought", "buys"],
    "wear": ["wore", "worn", "wears"],
    "choose": ["chose", "chosen", "chooses"],
    # Missing irregular forms that caused validation errors
    "blow": ["blew", "blown", "blows"],
    "catch": ["caught", "catches"],
    "cling": ["clung", "clings"],
    "hang": ["hung", "hanged", "hangs"],
    "foot": ["feet"],  # Irregular plural
    "tooth": ["teeth"],  # Irregular plural
    "throw": ["threw", "thrown", "throws"],
    "strike": ["struck", "strikes"],
    "sweep": ["swept", "sweeps"],
    "swell": ["swelled", "swollen", "swells"],
    "swing": ["swung", "swings"],
    "spring": ["sprang", "sprung", "springs"],
    "spin": ["spun", "spins"],
    "speed": ["sped", "speeds"],
    "shake": ["shook", "shaken", "shakes"],
    "sell": ["sold", "sells"],
    "lay": ["laid", "lays"],
    "leaf": ["leaves"],  # Irregular plural
    "knife": ["knives"],  # Irregular plural
    "partake": ["partook", "partaken", "partakes"],
    "gentleman": ["gentlemen"],  # Irregular plural
    "free": ["freed", "freeing", "frees"],
    "forget": ["forgot", "forgotten", "forgets"],
    "direct": ["directed", "directs"],
    "deliver": ["delivered", "delivers"],
    "count": ["counted", "counts"],
    "convey": ["conveyed", "conveys"],
    "contrive": ["contrived", "contrives"],
    "consider": ["considered", "considers"],
    "check": ["checked", "checks"],
    "child": ["children"],  # Irregular plural
    "carefully": ["careful"],  # Adjective form
    "brood": ["brooded", "broods"],
    "bind": ["bound", "binds"],
    "befall": ["befell", "befallen", "befalls"],
    "bear": ["bore", "borne", "bears"],
    "base": ["based", "bases"],
    "awake": ["awoke", "awakened", "awakes"],
    "assert": ["asserted", "asserts"],
    "afford": ["afforded", "affords"],
    "affair": ["affairs"],
    "admit": ["admitted", "admits"],
    "address": ["addressed", "addresses"],
    "act": ["acted", "acts"],
    "about": [],  # Preposition/adverb, no forms
    # Additional irregular verbs from standard list
    "beat": ["beat", "beaten", "beats"],
    "bend": ["bent", "bends"],
    "bet": ["bet", "bets"],
    "bid": ["bid", "bade", "bids"],
    "bite": ["bit", "bitten", "bites"],
    "bleed": ["bled", "bleeds"],
    "breed": ["bred", "breeds"],
    "burn": ["burnt", "burned", "burns"],
    "burst": ["burst", "bursts"],
    "cast": ["cast", "casts"],
    "cost": ["cost", "costs"],
    "creep": ["crept", "creeps"],
    "deal": ["dealt", "deals"],
    "dig": ["dug", "digs"],
    "dream": ["dreamt", "dreamed", "dreams"],
    "drink": ["drank", "drunk", "drinks"],
    "eat": ["ate", "eaten", "eats"],
    "feed": ["fed", "feeds"],
    "fight": ["fought", "fights"],
    "flee": ["fled", "flees"],
    "fling": ["flung", "flings"],
    "fly": ["flew", "flown", "flies"],
    "forbid": ["forbade", "forbidden", "forbids"],
    "forecast": ["forecast", "forecasts"],
    "forgive": ["forgave", "forgiven", "forgives"],
    "freeze": ["froze", "frozen", "freezes"],
    "grind": ["ground", "grinds"],
    "hide": ["hid", "hidden", "hides"],
    "hit": ["hit", "hits"],
    "kneel": ["knelt", "kneels"],
    "knit": ["knit", "knitted", "knits"],
    "lend": ["lent", "lends"],
    "light": ["lit", "lighted", "lights"],
    "mislay": ["mislaid", "mislays"],
    "mislead": ["misled", "misleads"],
    "prove": ["proved", "proven", "proves"],
    "quit": ["quit", "quits"],
    "rid": ["rid", "rids"],
    "ride": ["rode", "ridden", "rides"],
    "ring": ["rang", "rung", "rings"],
    "saw": ["sawed", "sawn", "saws"],
    "seek": ["sought", "seeks"],
    "sew": ["sewed", "sewn", "sews"],
    "shear": ["shore", "shorn", "shears"],
    "shed": ["shed", "sheds"],
    "shine": ["shone", "shined", "shines"],
    "shoe": ["shod", "shoes"],
    "shoot": ["shot", "shoots"],
    "show": ["showed", "shown", "shows"],
    "shrink": ["shrank", "shrunk", "shrinks"],
    "shut": ["shut", "shuts"],
    "sing": ["sang", "sung", "sings"],
    "sink": ["sank", "sunk", "sinks"],
    "slay": ["slew", "slain", "slays"],
    "sleep": ["slept", "sleeps"],
    "slide": ["slid", "slides"],
    "sling": ["slung", "slings"],
    "slink": ["slunk", "slinks"],
    "slit": ["slit", "slits"],
    "smell": ["smelt", "smelled", "smells"],
    "smite": ["smote", "smitten", "smites"],
    "sow": ["sowed", "sown", "sows"],
    "spell": ["spelt", "spelled", "spells"],
    "spill": ["spilt", "spilled", "spills"],
    "spit": ["spat", "spit", "spits"],
    "split": ["split", "splits"],
    "spoil": ["spoilt", "spoiled", "spoils"],
    "spread": ["spread", "spreads"],
    "steal": ["stole", "stolen", "steals"],
    "stick": ["stuck", "sticks"],
    "sting": ["stung", "stings"],
    "stink": ["stank", "stunk", "stinks"],
    "strew": ["strewed", "strewn", "strews"],
    "stride": ["strode", "stridden", "strides"],
    "string": ["strung", "strings"],
    "strive": ["strove", "striven", "strives"],
    "swear": ["swore", "sworn", "swears"],
    "swim": ["swam", "swum", "swims"],
    "teach": ["taught", "teaches"],
    "tear": ["tore", "torn", "tears"],
    "thrive": ["throve", "thriven", "thrives"],
    "thrust": ["thrust", "thrusts"],
    "tread": ["trod", "trodden", "treads"],
    "undergo": ["underwent", "undergone", "undergoes"],
    "undertake": ["undertook", "undertaken", "undertakes"],
    "undo": ["undid", "undone", "undoes"],
    "upset": ["upset", "upsets"],
    "wake": ["woke", "woken", "wakes"],
    "weave": ["wove", "woven", "weaves"],
    "wed": ["wed", "weds"],
    "weep": ["wept", "weeps"],
    "wet": ["wet", "wetted", "wets"],
    "win": ["won", "wins"],
    "wind": ["wound", "winds"],
    "withdraw": ["withdrew", "withdrawn", "withdraws"],
    "withstand": ["withstood", "withstands"],
    "wring": ["wrung", "wrings"],
    # Irregular plurals (additional)
    "man": ["men"],
    "woman": ["women"],
    "person": ["people", "persons"],
    "mouse": ["mice"],
    "louse": ["lice"],
    "goose": ["geese"],
    "die": ["dies", "dice"],  # As in gaming dice
    "ox": ["oxen"],
}


@dataclass
class ValidationResult:
    """Result of validating card examples."""

    is_valid: bool
    valid_examples: list[str] = field(default_factory=list)
    invalid_examples: list[str] = field(default_factory=list)
    found_forms: list[str] = field(default_factory=list)


def _get_word_forms(lemma: str) -> set[str]:
    """Get all possible forms of a lemma.

    Includes:
    - Base lemma
    - Regular morphological forms (+ed, +ing, +s, etc.)
    - Irregular forms from dictionary
    """
    forms = {lemma.lower()}

    # Add irregular forms
    if lemma.lower() in IRREGULAR_FORMS:
        forms.update(IRREGULAR_FORMS[lemma.lower()])

    # Add regular morphological forms
    base = lemma.lower()

    # Verbs: +s, +ed, +ing
    forms.add(base + "s")
    forms.add(base + "ed")
    forms.add(base + "ing")

    # Handle -e ending (make -> making, made)
    if base.endswith("e"):
        forms.add(base[:-1] + "ing")  # make -> making
        forms.add(base + "d")  # move -> moved

    # Handle -y ending (try -> tried, tries)
    if base.endswith("y") and len(base) > 2:
        forms.add(base[:-1] + "ied")  # try -> tried
        forms.add(base[:-1] + "ies")  # try -> tries

    # Handle consonant doubling (run -> running, stop -> stopped)
    if len(base) >= 3 and base[-1] not in "aeiouwy" and base[-2] in "aeiou":
        forms.add(base + base[-1] + "ing")  # run -> running
        forms.add(base + base[-1] + "ed")  # stop -> stopped

    # Nouns: +s, +es
    forms.add(base + "es")

    # Adjectives: +er, +est
    forms.add(base + "er")
    forms.add(base + "est")
    if base.endswith("e"):
        forms.add(base + "r")  # large -> larger
        forms.add(base + "st")  # large -> largest
    if base.endswith("y") and len(base) > 2:
        forms.add(base[:-1] + "ier")  # happy -> happier
        forms.add(base[:-1] + "iest")  # happy -> happiest

    return forms


def _get_synset_synonyms(synset_id: str) -> set[str]:
    """Get all synonyms (lemma names) from a WordNet synset."""
    try:
        synset = wn.synset(synset_id)
        synonyms = set()
        for lemma in synset.lemmas():
            name = lemma.name().lower().replace("_", " ")
            synonyms.add(name)
            # Also add forms of synonyms
            synonyms.update(_get_word_forms(name))
        return synonyms
    except Exception:
        return set()


def _word_in_text(word: str, text: str) -> bool:
    """Check if a word appears in text as a whole word."""
    # Use word boundary regex to avoid partial matches
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


def validate_card_examples(card: SmartCard) -> ValidationResult:
    """Validate that card examples contain the lemma or its synonyms.

    Checks:
    1. Exact lemma match (case insensitive)
    2. Morphological forms of the lemma
    3. WordNet synonyms from the synset
    4. Morphological forms of synonyms

    Args:
        card: SmartCard to validate

    Returns:
        ValidationResult with valid/invalid examples and found forms
    """
    if not card.selected_examples:
        return ValidationResult(
            is_valid=False,
            valid_examples=[],
            invalid_examples=[],
            found_forms=[],
        )

    # Get all valid forms
    lemma_forms = _get_word_forms(card.lemma)
    synonym_forms = _get_synset_synonyms(card.primary_synset)
    all_forms = lemma_forms | synonym_forms

    valid_examples = []
    invalid_examples = []
    found_forms = []

    for example in card.selected_examples:
        example_lower = example.lower()

        # Check if any form appears in the example
        form_found = None
        for form in all_forms:
            if _word_in_text(form, example_lower):
                form_found = form
                break

        if form_found:
            valid_examples.append(example)
            if form_found not in found_forms:
                found_forms.append(form_found)
        else:
            invalid_examples.append(example)

    is_valid = len(valid_examples) > 0

    return ValidationResult(
        is_valid=is_valid,
        valid_examples=valid_examples,
        invalid_examples=invalid_examples,
        found_forms=found_forms,
    )


def fix_invalid_cards(
    cards: list[SmartCard],
    use_generated_example: bool = True,
    remove_unfixable: bool = False,
) -> tuple[list[SmartCard], list[SmartCard]]:
    """Fix cards with invalid examples.

    Strategy:
    1. Validate each card
    2. If invalid:
       a) Keep only valid_examples (if any)
       b) If no valid examples and use_generated_example:
          - Use generated_example as the only example
       c) Otherwise: mark for review or remove (based on remove_unfixable)

    Args:
        cards: List of SmartCard objects
        use_generated_example: Whether to use generated_example as fallback
        remove_unfixable: If True, remove cards that can't be fixed; if False, return them

    Returns:
        Tuple of (fixed_cards, cards_for_review_or_removed)
    """
    from copy import deepcopy

    fixed_cards = []
    cards_for_review = []

    stats = {
        "total": len(cards),
        "already_valid": 0,
        "fixed_partial": 0,  # Had some valid examples
        "fixed_generated": 0,  # Used generated_example
        "for_review": 0,
    }

    for card in cards:
        result = validate_card_examples(card)

        if result.is_valid and not result.invalid_examples:
            # All examples are valid
            fixed_cards.append(card)
            stats["already_valid"] += 1

        elif result.is_valid:
            # Some examples valid, some invalid - keep only valid
            fixed_card = deepcopy(card)
            fixed_card.selected_examples = result.valid_examples
            fixed_card.excluded_examples = card.excluded_examples + result.invalid_examples
            fixed_cards.append(fixed_card)
            stats["fixed_partial"] += 1
            logger.debug(
                f"Card '{card.lemma}': kept {len(result.valid_examples)} valid, "
                f"removed {len(result.invalid_examples)} invalid examples"
            )

        elif use_generated_example and card.generated_example:
            # No valid examples, but has generated_example
            fixed_card = deepcopy(card)
            fixed_card.selected_examples = [card.generated_example]
            fixed_card.excluded_examples = card.excluded_examples + result.invalid_examples
            fixed_cards.append(fixed_card)
            stats["fixed_generated"] += 1
            logger.info(
                f"Card '{card.lemma}': using generated_example "
                f"(removed {len(result.invalid_examples)} invalid)"
            )

        else:
            # No valid examples and no generated_example
            cards_for_review.append(card)
            stats["for_review"] += 1
            action = "removed" if remove_unfixable else "marked for review"
            logger.warning(
                f"Card '{card.lemma}' ({card.primary_synset}): " f"no valid examples, {action}"
            )

    action_str = "removed" if remove_unfixable else "for review"
    logger.info(
        f"Validation complete: {stats['already_valid']} valid, "
        f"{stats['fixed_partial']} fixed (partial), "
        f"{stats['fixed_generated']} fixed (generated), "
        f"{stats['for_review']} {action_str}"
    )

    return fixed_cards, cards_for_review
