"""Text normalization for lemma-in-example matching and QC (Stage 2).

Used only for matching/QC; does not modify card text unless explicitly requested.
Deterministic, pure, no network.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_for_matching(text: str) -> str:
    """Normalize text for matching: NFKC, unify apostrophes/dashes, collapse whitespace.

    Does not change the original card example text; use only when comparing lemma to examples.
    """
    if not text:
        return ""
    s = unicodedata.normalize("NFKC", text)
    # Unify apostrophes (U+2019, etc.) -> ASCII
    s = s.replace("\u2019", "'").replace("\u2018", "'").replace("\u2032", "'")
    # Unify en/em dash -> hyphen
    s = s.replace("\u2014", "-").replace("\u2013", "-")
    # Collapse whitespace and strip
    s = " ".join(s.split())
    return s


# Common English contractions for matching only (small, no hallucination)
_CONTRACTIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdon't\b", re.I), "do not"),
    (re.compile(r"\bdoesn't\b", re.I), "does not"),
    (re.compile(r"\bdidn't\b", re.I), "did not"),
    (re.compile(r"\bwon't\b", re.I), "will not"),
    (re.compile(r"\bwouldn't\b", re.I), "would not"),
    (re.compile(r"\bcan't\b", re.I), "can not"),
    (re.compile(r"\bcouldn't\b", re.I), "could not"),
    (re.compile(r"\bshan't\b", re.I), "shall not"),
    (re.compile(r"\bshouldn't\b", re.I), "should not"),
    (re.compile(r"\bisn't\b", re.I), "is not"),
    (re.compile(r"\baren't\b", re.I), "are not"),
    (re.compile(r"\bwasn't\b", re.I), "was not"),
    (re.compile(r"\bweren't\b", re.I), "were not"),
    (re.compile(r"\bhaven't\b", re.I), "have not"),
    (re.compile(r"\bhasn't\b", re.I), "has not"),
    (re.compile(r"\bhadn't\b", re.I), "had not"),
    (re.compile(r"\bI'm\b", re.I), "I am"),
    (re.compile(r"\bit's\b", re.I), "it is"),
    (re.compile(r"\bhe's\b", re.I), "he is"),
    (re.compile(r"\bshe's\b", re.I), "she is"),
    (re.compile(r"\bwe're\b", re.I), "we are"),
    (re.compile(r"\bthey're\b", re.I), "they are"),
    (re.compile(r"\bI've\b", re.I), "I have"),
    (re.compile(r"\bwe've\b", re.I), "we have"),
    (re.compile(r"\bthey've\b", re.I), "they have"),
    (re.compile(r"\byou've\b", re.I), "you have"),
    (re.compile(r"\bI'll\b", re.I), "I will"),
    (re.compile(r"\bwe'll\b", re.I), "we will"),
    (re.compile(r"\bthey'll\b", re.I), "they will"),
    (re.compile(r"\byou'll\b", re.I), "you will"),
    (re.compile(r"\bain't\b", re.I), "am not"),
]


def expand_contractions_for_matching(text: str) -> str:
    """Expand common contractions for matching only. Returns new string; original unchanged."""
    s = text
    for pat, repl in _CONTRACTIONS:
        s = pat.sub(repl, s)
    return s


def _whole_word_pattern(word: str) -> re.Pattern:
    return re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)


def word_in_text_for_matching(word: str, text: str) -> bool:
    """Check if word appears as whole word in text, using normalized and expanded text for matching.

    Use for lemma-in-example QC: reduces false negatives from apostrophes/contractions.
    """
    if not word or not text:
        return False
    norm = normalize_for_matching(text)
    expanded = expand_contractions_for_matching(norm)
    pat = _whole_word_pattern(word)
    return bool(pat.search(norm)) or bool(pat.search(expanded))


def match_target_in_text(target: str, text: str) -> bool:
    """Check if target (word or phrase) appears in text. Stage 4: single entry for lemma/headword QC.

    - Word: whole-word match with normalization and contraction expansion (apostrophe, hyphen-safe).
    - Phrase (target contains space, e.g. 'look up'): normalized substring match.
    Rule: target must be found in every example when used for QC (caller enforces 'each example').
    """
    if not target or not text:
        return False
    target = target.strip()
    if not target:
        return False
    if " " in target:
        norm_text = normalize_for_matching(text).lower()
        norm_target = normalize_for_matching(target).lower()
        return norm_target in norm_text
    return word_in_text_for_matching(target, text)
