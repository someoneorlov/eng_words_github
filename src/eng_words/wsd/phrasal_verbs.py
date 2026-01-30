"""
Phrasal Verb detection using spaCy dependency parsing.

Phrasal verbs are multi-word verbs consisting of a verb + particle.
Examples: look up, take off, give up, break down.

This module:
1. Maintains a dictionary of common phrasal verbs with meanings
2. Uses spaCy to detect particle dependencies
3. Integrates with the construction detector

Note: Based on findings from Stage 1, we use CONSTRAIN policy conservatively
(recording the tag) but don't filter synsets, as gold labels often use
literal meanings.
"""

from typing import Any

import spacy

from eng_words.wsd.constructions import ConstructionMatch, detect_constructions

# Load spaCy model (cached after first load)
_nlp = None


def get_nlp():
    """Lazily load spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# Dictionary of phrasal verbs organized by:
# verb -> particle -> {meaning_category: [synset_hints]}
PHRASAL_VERBS: dict[str, dict[str, dict[str, list[str]]]] = {
    # ========== LOOK ==========
    "look": {
        "up": {
            "research": ["consult.v.01", "look_up.v.01"],
            "raise_eyes": ["look.v.01"],
        },
        "after": {"care": ["care.v.01", "look_after.v.01"]},
        "forward": {"anticipate": ["anticipate.v.01", "look_forward.v.01"]},
        "into": {"investigate": ["investigate.v.01", "look_into.v.01"]},
        "out": {"beware": ["watch.v.01", "look_out.v.01"]},
        "over": {"review": ["review.v.01", "examine.v.01"]},
    },
    # ========== TAKE ==========
    "take": {
        "off": {
            "remove": ["remove.v.01", "take_off.v.01"],
            "depart": ["depart.v.01", "take_off.v.02"],
        },
        "on": {
            "accept": ["accept.v.02", "take_on.v.01"],
            "employ": ["hire.v.01"],
        },
        "up": {
            "begin": ["begin.v.01", "take_up.v.01"],
            "occupy": ["occupy.v.02"],
        },
        "over": {"assume_control": ["take_over.v.01"]},
        "out": {"remove": ["remove.v.01", "extract.v.01"]},
        "in": {"absorb": ["absorb.v.01"], "deceive": ["deceive.v.01"]},
    },
    # ========== GIVE ==========
    "give": {
        "up": {
            "surrender": ["surrender.v.01", "give_up.v.01"],
            "quit": ["discontinue.v.01", "abandon.v.01"],
        },
        "in": {"yield": ["yield.v.01", "give_in.v.01"]},
        "away": {
            "reveal": ["reveal.v.01", "give_away.v.01"],
            "donate": ["donate.v.01"],
        },
        "out": {"distribute": ["distribute.v.01"]},
        "back": {"return": ["return.v.01"]},
    },
    # ========== PUT ==========
    "put": {
        "off": {"postpone": ["postpone.v.01", "put_off.v.01"]},
        "up": {
            "accommodate": ["accommodate.v.01"],
            "tolerate": ["tolerate.v.01", "put_up.v.01"],
        },
        "on": {
            "wear": ["wear.v.01", "put_on.v.01"],
            "pretend": ["pretend.v.01"],
        },
        "down": {"criticize": ["disparage.v.01"], "write": ["write.v.01"]},
        "away": {"store": ["store.v.01"]},
    },
    # ========== GET ==========
    "get": {
        "up": {"rise": ["rise.v.01", "get_up.v.01"]},
        "over": {"recover": ["recover.v.01", "get_over.v.01"]},
        "along": {
            "progress": ["progress.v.01"],
            "relate": ["relate.v.01", "get_along.v.01"],
        },
        "away": {"escape": ["escape.v.01", "get_away.v.01"]},
        "by": {"survive": ["survive.v.01", "get_by.v.01"]},
        "through": {"endure": ["endure.v.01"]},
        "out": {"exit": ["exit.v.01"]},
    },
    # ========== SET ==========
    "set": {
        "up": {"establish": ["establish.v.01", "set_up.v.01"]},
        "off": {
            "trigger": ["trigger.v.01", "set_off.v.01"],
            "depart": ["depart.v.01"],
        },
        "out": {"begin_journey": ["depart.v.01", "set_out.v.01"]},
        "aside": {"reserve": ["reserve.v.01"]},
    },
    # ========== BREAK ==========
    "break": {
        "down": {
            "malfunction": ["fail.v.02", "break_down.v.01"],
            "decompose": ["decompose.v.01"],
        },
        "up": {"end_relationship": ["separate.v.01", "break_up.v.01"]},
        "out": {
            "escape": ["escape.v.01", "break_out.v.01"],
            "erupt": ["erupt.v.01"],
        },
        "in": {"enter_forcibly": ["intrude.v.01"]},
        "off": {"discontinue": ["discontinue.v.01"]},
    },
    # ========== TURN ==========
    "turn": {
        "out": {
            "result": ["result.v.01", "turn_out.v.01"],
            "appear": ["appear.v.01"],
        },
        "up": {
            "arrive": ["arrive.v.01", "turn_up.v.01"],
            "increase": ["increase.v.01"],
        },
        "down": {
            "reject": ["reject.v.01", "turn_down.v.01"],
            "decrease": ["decrease.v.01"],
        },
        "in": {"submit": ["submit.v.01"]},
        "off": {"deactivate": ["deactivate.v.01"]},
        "on": {"activate": ["activate.v.01"]},
    },
    # ========== COME ==========
    "come": {
        "up": {"arise": ["arise.v.01", "come_up.v.01"]},
        "across": {"encounter": ["encounter.v.01", "come_across.v.01"]},
        "out": {"emerge": ["emerge.v.01", "come_out.v.01"]},
        "back": {"return": ["return.v.01"]},
        "in": {"enter": ["enter.v.01"]},
    },
    # ========== GO ==========
    "go": {
        "through": {"experience": ["experience.v.01", "go_through.v.01"]},
        "over": {"review": ["review.v.01", "go_over.v.01"]},
        "ahead": {"proceed": ["proceed.v.01", "go_ahead.v.01"]},
        "off": {"explode": ["explode.v.01"]},
        "back": {"return": ["return.v.01"]},
    },
    # ========== RUN ==========
    "run": {
        "out": {"exhaust": ["exhaust.v.01", "run_out.v.01"]},
        "into": {"encounter": ["encounter.v.01"]},
        "over": {"review": ["review.v.01"]},
        "away": {"flee": ["flee.v.01"]},
    },
    # ========== WORK ==========
    "work": {
        "out": {
            "exercise": ["exercise.v.01"],
            "solve": ["solve.v.01", "work_out.v.01"],
        },
        "on": {"improve": ["improve.v.01"]},
        "up": {"develop": ["develop.v.01"]},
    },
    # ========== CARRY ==========
    "carry": {
        "out": {"execute": ["execute.v.01", "carry_out.v.01"]},
        "on": {"continue": ["continue.v.01"]},
    },
    # ========== HOLD ==========
    "hold": {
        "on": {"wait": ["wait.v.01", "hold_on.v.01"]},
        "up": {"delay": ["delay.v.01"]},
        "back": {"restrain": ["restrain.v.01"]},
    },
    # ========== PICK ==========
    "pick": {
        "up": {
            "collect": ["collect.v.01", "pick_up.v.01"],
            "learn": ["learn.v.01"],
        },
        "out": {"select": ["select.v.01"]},
    },
    # ========== BRING ==========
    "bring": {
        "up": {
            "raise": ["raise.v.01", "bring_up.v.01"],
            "mention": ["mention.v.01"],
        },
        "about": {"cause": ["cause.v.01"]},
        "back": {"restore": ["restore.v.01"]},
    },
    # ========== KEEP ==========
    "keep": {
        "up": {"maintain": ["maintain.v.01", "keep_up.v.01"]},
        "on": {"continue": ["continue.v.01"]},
    },
    # ========== FIGURE ==========
    "figure": {
        "out": {"understand": ["understand.v.01", "figure_out.v.01"]},
    },
    # ========== FIND ==========
    "find": {
        "out": {"discover": ["discover.v.01", "find_out.v.01"]},
    },
    # ========== THROW ==========
    "throw": {
        "away": {"discard": ["discard.v.01"]},
        "out": {"eject": ["eject.v.01"]},
        "up": {"vomit": ["vomit.v.01"]},
    },
    # ========== STAND ==========
    "stand": {
        "up": {"rise": ["rise.v.01"]},
        "out": {"distinguish": ["distinguish.v.01"]},
        "for": {"represent": ["represent.v.01"]},
    },
    # ========== FILL ==========
    "fill": {
        "out": {"complete_form": ["complete.v.01"]},
        "in": {"substitute": ["substitute.v.01"]},
        "up": {"make_full": ["fill.v.01"]},
    },
}


def detect_phrasal_verb(
    sentence: str,
    lemma: str,
    pos: str,
    token_idx: int | None = None,
) -> ConstructionMatch | None:
    """
    Detect phrasal verb using spaCy dependency parsing.

    Args:
        sentence: Full sentence text
        lemma: Verb lemma (e.g., "look")
        pos: Part of speech (should be "VERB")
        token_idx: Optional token index for disambiguation

    Returns:
        ConstructionMatch if phrasal verb found, else None
    """
    if pos != "VERB":
        return None

    if lemma not in PHRASAL_VERBS:
        return None

    nlp = get_nlp()
    doc = nlp(sentence)

    for token in doc:
        # Match by lemma
        if token.lemma_.lower() != lemma.lower():
            continue

        if token.pos_ != "VERB":
            continue

        # Find particle child
        for child in token.children:
            if child.dep_ == "prt":  # particle dependency
                particle = child.text.lower()

                if particle in PHRASAL_VERBS[lemma]:
                    # Found phrasal verb!
                    meanings = PHRASAL_VERBS[lemma][particle]
                    # Get first meaning's synset hints
                    first_meaning = list(meanings.keys())[0]
                    synset_hints = meanings[first_meaning]

                    return ConstructionMatch(
                        construction_id=f"PHRASAL_{lemma.upper()}_{particle.upper()}",
                        lemma=lemma,
                        matched_tokens=[token.text, child.text],
                        span=(token.idx, child.idx + len(child.text)),
                        policy="CONSTRAIN",
                        forbid_supersenses=set(),  # Don't forbid (lesson from Stage 1)
                        prefer_synsets=synset_hints,
                        reason=f"Phrasal verb: {lemma} {particle} ({first_meaning})",
                    )

    return None


def detect_all_constructions(
    sentence: str,
    lemma: str,
    pos: str,
    token_idx: int | None = None,
) -> list[ConstructionMatch]:
    """
    Detect all constructions using multiple methods.

    Priority:
    1. Regex patterns (faster, more precise for fixed expressions)
    2. Phrasal verb detection (dependency-based, for verb particles)

    Args:
        sentence: Full sentence text
        lemma: Target word lemma
        pos: Part of speech
        token_idx: Optional token index for disambiguation

    Returns:
        List of matched constructions (may be empty)
    """
    matches: list[ConstructionMatch] = []

    # 1. Regex patterns first (from constructions.py)
    regex_matches = detect_constructions(sentence, lemma, pos)
    matches.extend(regex_matches)

    # 2. Phrasal verbs if no regex match and it's a verb
    if not matches and pos == "VERB":
        pv_match = detect_phrasal_verb(sentence, lemma, pos, token_idx)
        if pv_match:
            matches.append(pv_match)

    return matches


def get_phrasal_verb_info(lemma: str, particle: str) -> dict[str, Any] | None:
    """
    Get information about a specific phrasal verb.

    Args:
        lemma: Verb lemma (e.g., "look")
        particle: Particle (e.g., "up")

    Returns:
        Dictionary with meanings and synset hints, or None if not found
    """
    if lemma not in PHRASAL_VERBS:
        return None

    if particle not in PHRASAL_VERBS[lemma]:
        return None

    return {
        "lemma": lemma,
        "particle": particle,
        "meanings": PHRASAL_VERBS[lemma][particle],
    }

