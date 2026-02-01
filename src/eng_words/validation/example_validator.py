"""Example validation for SmartCards.

Validates that card examples contain the lemma or its synonyms,
ensuring users can see the word in context.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

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


def _word_in_text(word: str, text: str) -> bool:
    """Check if a word appears in text as a whole word."""
    # Use word boundary regex to avoid partial matches
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))
