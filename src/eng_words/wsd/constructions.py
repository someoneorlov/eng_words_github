"""
Construction detection for WSD.

Detects grammatical constructions where a word is used in a grammatical
rather than lexical sense, such as:
- "going to" = future tense, not verb.motion
- "come on" = encouragement, not verb.motion
- "make up" = fabricate, not verb.creation

These constructions can be used to:
- SKIP: Don't assign any synset (e.g., "gonna")
- CONSTRAIN: Filter out certain supersenses (e.g., "go on" not verb.motion)
- OVERRIDE: Force a specific synset (e.g., "over there" = over.r.01)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# Valid policies for construction handling
Policy = Literal["SKIP", "CONSTRAIN", "OVERRIDE"]
VALID_POLICIES = {"SKIP", "CONSTRAIN", "OVERRIDE"}


@dataclass
class ConstructionMatch:
    """Result of construction detection."""

    construction_id: str  # e.g., "BE_GOING_TO", "COME_ON"
    lemma: str  # matched lemma
    matched_tokens: list[str]  # tokens that form the construction
    span: tuple[int, int]  # character span in sentence

    # WSD guidance
    policy: Policy
    forbid_supersenses: set[str] = field(default_factory=set)
    prefer_synsets: list[str] = field(default_factory=list)

    reason: str = ""  # human-readable explanation

    def __post_init__(self) -> None:
        """Validate policy value."""
        if self.policy not in VALID_POLICIES:
            raise ValueError(f"Invalid policy: {self.policy}. Must be one of {VALID_POLICIES}")


# Construction patterns based on error analysis
# Each pattern has:
# - pattern: regex to match (case-insensitive)
# - target_lemma: the lemma this pattern applies to
# - policy: SKIP, CONSTRAIN, or OVERRIDE
# - reason: human-readable explanation
# Optional:
# - forbid_supersenses: set of supersenses to exclude (for CONSTRAIN)
# - prefer_synsets: list of synsets to prefer (for OVERRIDE)
# - prefer_supersenses: set of supersenses to prefer (for CONSTRAIN)

CONSTRUCTION_PATTERNS: dict[str, dict[str, Any]] = {
    # ========== Future tense constructions ==========
    "BE_GOING_TO": {
        "pattern": r"\b(am|is|are|was|were|'m|'s|'re)\s+going\s+to\b",
        "target_lemma": "go",
        "policy": "SKIP",
        "reason": "Future tense construction, not movement",
    },
    "GONNA": {
        "pattern": r"\bgonna\b",
        "target_lemma": "go",
        "policy": "SKIP",
        "reason": "Informal future tense",
    },
    # ========== GO phrasal verbs ==========
    "GO_ON": {
        "pattern": r"\b(go|goes|went|gone|going)\s+on\b",
        "target_lemma": "go",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.stative", "verb.communication"},
        "reason": "Phrasal verb: continue",
    },
    "GO_AHEAD": {
        "pattern": r"\b(go|goes|went|gone|going)\s+ahead\b",
        "target_lemma": "go",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.stative", "verb.communication"},
        "reason": "Phrasal verb: proceed/permission",
    },
    "GO_THROUGH": {
        "pattern": r"\b(go|goes|went|gone|going)\s+through\b",
        "target_lemma": "go",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.cognition", "verb.change"},
        "reason": "Phrasal verb: experience/examine",
    },
    "GO_OVER": {
        "pattern": r"\b(go|goes|went|gone|going)\s+over\b",
        "target_lemma": "go",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.cognition"},
        "reason": "Phrasal verb: review",
    },
    # ========== COME phrasal verbs ==========
    "COME_ON": {
        "pattern": r"\bcome\s+on\b",
        "target_lemma": "come",
        "policy": "CONSTRAIN",
        # Note: Don't forbid verb.motion as "Come on!" can still be motion-based invitation
        "prefer_supersenses": {"verb.stative", "verb.change"},
        "reason": "Phrasal verb: encouragement or progress",
    },
    "COME_UP": {
        "pattern": r"\b(come|comes|came|coming)\s+up\b",
        "target_lemma": "come",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.stative", "verb.change"},
        "reason": "Phrasal verb: arise/occur",
    },
    "COME_ACROSS": {
        "pattern": r"\b(come|comes|came|coming)\s+across\b",
        "target_lemma": "come",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.perception", "verb.cognition"},
        "reason": "Phrasal verb: encounter/appear",
    },
    # ========== MAKE expressions ==========
    "MAKE_SURE": {
        "pattern": r"\bmake\s+sure\b",
        "target_lemma": "make",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.cognition"},
        "reason": "Fixed expression: ensure",
    },
    "MAKE_UP": {
        "pattern": r"\b(make|makes|made|making)\s+(up|it\s+up)\b",
        "target_lemma": "make",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.creation", "verb.cognition", "verb.social"},
        "reason": "Phrasal verb: fabricate/reconcile/compose",
    },
    "MAKE_OUT": {
        "pattern": r"\b(make|makes|made|making)\s+out\b",
        "target_lemma": "make",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.perception", "verb.cognition"},
        "reason": "Phrasal verb: discern/claim",
    },
    # ========== CALL expressions ==========
    "CALL_UP": {
        "pattern": r"\b(call|calls|called|calling)\s+up\b",
        "target_lemma": "call",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.communication"},
        "reason": "Phrasal verb: telephone",
    },
    "CALL_OFF": {
        "pattern": r"\b(call|calls|called|calling)\s+off\b",
        "target_lemma": "call",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.communication"},
        "reason": "Phrasal verb: cancel",
    },
    # ========== TIME expressions ==========
    "AT_THE_TIME": {
        "pattern": r"\bat\s+(the|that|this)\s+time\b",
        "target_lemma": "time",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"noun.time"},
        "reason": "Temporal expression",
    },
    "IN_TIME": {
        "pattern": r"\bin\s+time\b",
        "target_lemma": "time",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"noun.time"},
        "reason": "Temporal expression: soon enough",
    },
    "ON_TIME": {
        "pattern": r"\bon\s+time\b",
        "target_lemma": "time",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"noun.time"},
        "reason": "Temporal expression: punctual",
    },
    # ========== POINT expressions ==========
    "POINT_OF_VIEW": {
        "pattern": r"\bpoint\s+of\s+view\b",
        "target_lemma": "point",
        "policy": "SKIP",
        "reason": "Fixed expression: perspective (multiword)",
    },
    "TO_THE_POINT": {
        "pattern": r"\bto\s+the\s+point\b",
        "target_lemma": "point",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"noun.communication"},
        "reason": "Idiomatic: relevance",
    },
    "AT_THIS_POINT": {
        "pattern": r"\bat\s+(this|that)\s+point\b",
        "target_lemma": "point",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"noun.time"},
        "reason": "Temporal expression: now/then",
    },
    # ========== LET expressions ==========
    "LET_GO": {
        "pattern": r"\blet\s+(go|it\s+go)\b",
        "target_lemma": "let",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.social"},
        "reason": "Fixed expression: release",
    },
    "LET_ALONE": {
        "pattern": r"\blet\s+alone\b",
        "target_lemma": "let",
        "policy": "SKIP",
        "reason": "Conjunction: not to mention (multiword)",
    },
    # ========== SEE expressions ==========
    "SEE_TO": {
        "pattern": r"\b(see|sees|saw|seen|seeing)\s+to\b",
        "target_lemma": "see",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.social", "verb.cognition"},
        "reason": "Phrasal verb: attend to",
    },
    # ========== LOOK expressions ==========
    "LOOK_FORWARD": {
        "pattern": r"\b(look|looks|looked|looking)\s+forward\s+to\b",
        "target_lemma": "look",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.emotion"},
        "reason": "Fixed expression: anticipate",
    },
    "LOOK_UP_TO": {
        "pattern": r"\b(look|looks|looked|looking)\s+up\s+to\b",
        "target_lemma": "look",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"verb.emotion"},
        "reason": "Phrasal verb: admire",
    },
    # ========== OVER expressions ==========
    "OVER_THERE": {
        "pattern": r"\bover\s+there\b",
        "target_lemma": "over",
        "policy": "OVERRIDE",
        "prefer_synsets": ["over.r.01"],
        "reason": "Spatial reference",
    },
    # ========== WAY expressions ==========
    "BY_THE_WAY": {
        "pattern": r"\bby\s+the\s+way\b",
        "target_lemma": "way",
        "policy": "SKIP",
        "reason": "Fixed expression: incidentally (multiword)",
    },
    "IN_A_WAY": {
        "pattern": r"\bin\s+a\s+way\b",
        "target_lemma": "way",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"noun.attribute"},
        "reason": "Fixed expression: manner",
    },
    # ========== LIFE expressions ==========
    "FOR_LIFE": {
        "pattern": r"\bfor\s+life\b",
        "target_lemma": "life",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"noun.time"},
        "reason": "Fixed expression: permanently",
    },
    # ========== PLACE expressions ==========
    "TAKE_PLACE": {
        "pattern": r"\btake\s+place\b",
        "target_lemma": "place",
        "policy": "SKIP",
        "reason": "Fixed expression: occur (multiword verb)",
    },
    "IN_PLACE": {
        "pattern": r"\bin\s+place\b",
        "target_lemma": "place",
        "policy": "CONSTRAIN",
        "prefer_supersenses": {"noun.location", "noun.state"},
        "reason": "Fixed expression: established",
    },
}


def detect_constructions(
    sentence: str,
    lemma: str,
    pos: str,
) -> list[ConstructionMatch]:
    """
    Detect grammatical constructions in sentence.

    Args:
        sentence: Full sentence text
        lemma: Target word lemma (e.g., "go")
        pos: Part of speech (e.g., "VERB")

    Returns:
        List of matched constructions (may be empty), sorted by match length (longest first)
    """
    matches: list[ConstructionMatch] = []

    for construction_id, pattern_data in CONSTRUCTION_PATTERNS.items():
        # Only match patterns for the target lemma
        if pattern_data["target_lemma"] != lemma:
            continue

        regex = pattern_data["pattern"]
        match = re.search(regex, sentence, re.IGNORECASE)

        if match:
            matches.append(
                ConstructionMatch(
                    construction_id=construction_id,
                    lemma=lemma,
                    matched_tokens=match.group().split(),
                    span=(match.start(), match.end()),
                    policy=pattern_data["policy"],
                    forbid_supersenses=set(pattern_data.get("forbid_supersenses", [])),
                    prefer_synsets=list(pattern_data.get("prefer_synsets", [])),
                    reason=pattern_data["reason"],
                )
            )

    # Sort by match length (longest first) for priority
    matches.sort(key=lambda m: m.span[1] - m.span[0], reverse=True)

    return matches


def apply_construction_policy(
    match: ConstructionMatch,
    candidates: list[str],
) -> dict[str, Any]:
    """
    Apply construction policy to get WSD guidance.

    Args:
        match: ConstructionMatch from detect_constructions
        candidates: List of candidate synset IDs

    Returns:
        Dictionary with policy guidance:
        - For SKIP: {"skip": True}
        - For CONSTRAIN: {"forbid_supersenses": set, "prefer_supersenses": set}
        - For OVERRIDE: {"override_synset": str} if found in candidates
    """
    if match.policy == "SKIP":
        return {"skip": True, "construction_id": match.construction_id}

    elif match.policy == "CONSTRAIN":
        return {
            "skip": False,
            "construction_id": match.construction_id,
            "forbid_supersenses": match.forbid_supersenses,
            "prefer_synsets": match.prefer_synsets,
        }

    elif match.policy == "OVERRIDE":
        # Check if preferred synset is in candidates
        for preferred in match.prefer_synsets:
            if preferred in candidates:
                return {
                    "skip": False,
                    "construction_id": match.construction_id,
                    "override_synset": preferred,
                }
        # If preferred synset not found, don't override
        return {
            "skip": False,
            "construction_id": match.construction_id,
            "prefer_synsets": match.prefer_synsets,
        }

    return {"skip": False}
