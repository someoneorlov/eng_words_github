"""Function words (ADP/SCONJ) configuration.

These are prepositions and subordinating conjunctions that can be included
or excluded from vocabulary learning based on their rarity and usefulness.

ignore=True: excluded by default (common function words like "of", "in", "to")
ignore=False: included by default (rare/interesting words like "despite", "throughout")

Users can override these defaults via CLI/config.
"""

FUNCTION_WORDS_CONFIG: dict[str, dict[str, bool | float | int]] = {
    "about": {
        "ignore": True,  # zipf=6.40, in_wn=True, freq=824
    },
    "above": {
        "ignore": False,  # zipf=5.20, in_wn=True, freq=70
    },
    "across": {
        "ignore": False,  # zipf=5.25, in_wn=True, freq=43
    },
    "after": {
        "ignore": True,  # zipf=6.11, in_wn=True, freq=698
    },
    "against": {
        "ignore": True,  # zipf=5.68, in_wn=False, freq=114
    },
    "albeit": {
        "ignore": True,  # zipf=3.91, in_wn=False, freq=1
    },
    "along": {
        "ignore": False,  # zipf=5.38, in_wn=True, freq=148
    },
    "alongside": {
        "ignore": False,  # zipf=4.45, in_wn=True, freq=1
    },
    "although": {
        "ignore": True,  # zipf=5.30, in_wn=False, freq=215
    },
    "amid": {
        "ignore": True,  # zipf=3.95, in_wn=False, freq=11
    },
    "among": {
        "ignore": True,  # zipf=5.31, in_wn=False, freq=52
    },
    "around": {
        "ignore": True,  # zipf=5.77, in_wn=True, freq=87
    },
    "as": {
        "ignore": True,  # zipf=6.77, in_wn=True, freq=3580
    },
    "aside": {
        "ignore": False,  # zipf=4.59, in_wn=True, freq=1
    },
    "at": {
        "ignore": True,  # zipf=6.70, in_wn=True, freq=2877
    },
    "away": {
        "ignore": True,  # zipf=5.62, in_wn=True, freq=12
    },
    "back": {
        "ignore": True,  # zipf=6.04, in_wn=True, freq=14
    },
    "because": {
        "ignore": True,  # zipf=6.03, in_wn=False, freq=560
    },
    "before": {
        "ignore": True,  # zipf=5.93, in_wn=True, freq=567
    },
    "behind": {
        "ignore": False,  # zipf=5.29, in_wn=True, freq=47
    },
    "below": {
        "ignore": False,  # zipf=5.05, in_wn=True, freq=31
    },
    "beneath": {
        "ignore": False,  # zipf=4.24, in_wn=True, freq=13
    },
    "beside": {
        "ignore": True,  # zipf=4.19, in_wn=False, freq=42
    },
    "besides": {
        "ignore": False,  # zipf=4.63, in_wn=True, freq=47
    },
    "between": {
        "ignore": True,  # zipf=5.77, in_wn=True, freq=156
    },
    "beyond": {
        "ignore": False,  # zipf=4.98, in_wn=True, freq=54
    },
    "but": {
        "ignore": True,  # zipf=6.63, in_wn=True, freq=30
    },
    "by": {
        "ignore": True,  # zipf=6.66, in_wn=True, freq=1628
    },
    "cause": {
        "ignore": False,  # zipf=5.35, in_wn=True, freq=2
    },
    "de": {
        "ignore": False,  # zipf=5.23, in_wn=True, freq=15
    },
    "despite": {
        "ignore": False,  # zipf=5.00, in_wn=True, freq=33
    },
    "dey're": {
        "ignore": True,  # zipf=0.00, in_wn=False, freq=1
    },
    "down": {
        "ignore": True,  # zipf=5.88, in_wn=True, freq=217
    },
    "due": {
        "ignore": False,  # zipf=5.35, in_wn=True, freq=7
    },
    "during": {
        "ignore": True,  # zipf=5.72, in_wn=False, freq=53
    },
    "en": {
        "ignore": False,  # zipf=4.50, in_wn=True, freq=9
    },
    "except": {
        "ignore": False,  # zipf=5.02, in_wn=True, freq=39
    },
    "for": {
        "ignore": True,  # zipf=7.01, in_wn=False, freq=3098
    },
    "forth": {
        "ignore": False,  # zipf=4.40, in_wn=True, freq=19
    },
    "forthwith": {
        "ignore": False,  # zipf=2.65, in_wn=True, freq=5
    },
    "forward": {
        "ignore": False,  # zipf=5.12, in_wn=True, freq=2
    },
    "from": {
        "ignore": True,  # zipf=6.63, in_wn=False, freq=1283
    },
    "gainst": {
        "ignore": True,  # zipf=2.07, in_wn=False, freq=1
    },
    "how": {
        "ignore": True,  # zipf=6.24, in_wn=False, freq=818
    },
    "if": {
        "ignore": True,  # zipf=6.47, in_wn=False, freq=1451
    },
    "in": {
        "ignore": True,  # zipf=7.27, in_wn=True, freq=5781
    },
    "inside": {
        "ignore": False,  # zipf=5.19, in_wn=True, freq=4
    },
    "into": {
        "ignore": True,  # zipf=6.11, in_wn=False, freq=347
    },
    "leadeth": {
        "ignore": True,  # zipf=1.93, in_wn=False, freq=3
    },
    "lest": {
        "ignore": False,  # zipf=3.54, in_wn=True, freq=9
    },
    "like": {
        "ignore": True,  # zipf=6.41, in_wn=True, freq=460
    },
    "near": {
        "ignore": False,  # zipf=5.29, in_wn=True, freq=121
    },
    "nearest": {
        "ignore": False,  # zipf=4.11, in_wn=True, freq=4
    },
    "next": {
        "ignore": True,  # zipf=5.70, in_wn=True, freq=7
    },
    "notwithstanding": {
        "ignore": False,  # zipf=3.48, in_wn=True, freq=2
    },
    "of": {
        "ignore": True,  # zipf=7.40, in_wn=False, freq=8799
    },
    "off": {
        "ignore": True,  # zipf=5.93, in_wn=True, freq=135
    },
    "on": {
        "ignore": True,  # zipf=6.91, in_wn=True, freq=1855
    },
    "once": {
        "ignore": True,  # zipf=5.53, in_wn=True, freq=69
    },
    "onto": {
        "ignore": True,  # zipf=4.77, in_wn=False, freq=3
    },
    "opposite": {
        "ignore": False,  # zipf=4.62, in_wn=True, freq=3
    },
    "orblike": {
        "ignore": True,  # zipf=0.00, in_wn=False, freq=1
    },
    "out": {
        "ignore": True,  # zipf=6.38, in_wn=True, freq=631
    },
    "outside": {
        "ignore": False,  # zipf=5.25, in_wn=True, freq=25
    },
    "over": {
        "ignore": True,  # zipf=6.08, in_wn=True, freq=375
    },
    "past": {
        "ignore": False,  # zipf=5.36, in_wn=True, freq=15
    },
    "per": {
        "ignore": True,  # zipf=5.45, in_wn=False, freq=5
    },
    "saith": {
        "ignore": True,  # zipf=2.94, in_wn=False, freq=1
    },
    "since": {
        "ignore": True,  # zipf=5.75, in_wn=False, freq=310
    },
    "so": {
        "ignore": True,  # zipf=6.52, in_wn=True, freq=103
    },
    "supposing": {
        "ignore": False,  # zipf=2.98, in_wn=True, freq=1
    },
    "than": {
        "ignore": True,  # zipf=6.13, in_wn=False, freq=581
    },
    "that": {
        "ignore": True,  # zipf=7.01, in_wn=False, freq=3395
    },
    "though": {
        "ignore": True,  # zipf=5.50, in_wn=True, freq=236
    },
    "through": {
        "ignore": True,  # zipf=5.87, in_wn=True, freq=174
    },
    "throughout": {
        "ignore": False,  # zipf=4.99, in_wn=True, freq=21
    },
    "till": {
        "ignore": False,  # zipf=4.79, in_wn=True, freq=6
    },
    "to": {
        "ignore": True,  # zipf=7.43, in_wn=False, freq=4409
    },
    "toward": {
        "ignore": True,  # zipf=4.72, in_wn=False, freq=164
    },
    "towards": {
        "ignore": True,  # zipf=5.06, in_wn=False, freq=2
    },
    "tryout": {
        "ignore": False,  # zipf=2.86, in_wn=True, freq=1
    },
    "under": {
        "ignore": True,  # zipf=5.73, in_wn=True, freq=148
    },
    "underneath": {
        "ignore": False,  # zipf=4.08, in_wn=True, freq=2
    },
    "unless": {
        "ignore": True,  # zipf=4.98, in_wn=False, freq=114
    },
    "unlike": {
        "ignore": False,  # zipf=4.57, in_wn=True, freq=19
    },
    "until": {
        "ignore": True,  # zipf=5.61, in_wn=False, freq=217
    },
    "unto": {
        "ignore": True,  # zipf=3.96, in_wn=False, freq=16
    },
    "up": {
        "ignore": True,  # zipf=6.39, in_wn=True, freq=672
    },
    "upon": {
        "ignore": True,  # zipf=5.12, in_wn=False, freq=357
    },
    "via": {
        "ignore": True,  # zipf=5.02, in_wn=False, freq=13
    },
    "when": {
        "ignore": True,  # zipf=6.37, in_wn=False, freq=552
    },
    "whence": {
        "ignore": False,  # zipf=3.10, in_wn=True, freq=1
    },
    "whenever": {
        "ignore": True,  # zipf=4.63, in_wn=False, freq=26
    },
    "where": {
        "ignore": True,  # zipf=6.00, in_wn=False, freq=465
    },
    "whereas": {
        "ignore": True,  # zipf=4.41, in_wn=False, freq=24
    },
    "whereby": {
        "ignore": True,  # zipf=3.74, in_wn=False, freq=3
    },
    "wherein": {
        "ignore": True,  # zipf=3.78, in_wn=False, freq=1
    },
    "whereupon": {
        "ignore": True,  # zipf=3.02, in_wn=False, freq=5
    },
    "wherever": {
        "ignore": False,  # zipf=4.27, in_wn=True, freq=16
    },
    "wherewith": {
        "ignore": True,  # zipf=2.49, in_wn=False, freq=11
    },
    "whether": {
        "ignore": True,  # zipf=5.32, in_wn=False, freq=174
    },
    "while": {
        "ignore": True,  # zipf=5.86, in_wn=True, freq=281
    },
    "why": {
        "ignore": True,  # zipf=5.93, in_wn=True, freq=379
    },
    "wis": {
        "ignore": False,  # zipf=3.24, in_wn=True, freq=1
    },
    "with": {
        "ignore": True,  # zipf=6.85, in_wn=False, freq=2849
    },
    "within": {
        "ignore": False,  # zipf=5.44, in_wn=True, freq=79
    },
    "without": {
        "ignore": True,  # zipf=5.69, in_wn=False, freq=299
    },
    "'cause": {
        "ignore": True,  # zipf=5.35, in_wn=False, freq=4
    },
}


def should_ignore_function_word(lemma: str, *, override: dict[str, bool] | None = None) -> bool:
    """Check if a function word should be ignored.

    Args:
        lemma: The lemma to check
        override: Optional dict to override defaults (e.g., {"despite": True})

    Returns:
        True if should be ignored, False if should be included
    """
    lemma_lower = lemma.lower()

    # Check override first
    if override and lemma_lower in override:
        return override[lemma_lower]

    # Check config
    if lemma_lower in FUNCTION_WORDS_CONFIG:
        return FUNCTION_WORDS_CONFIG[lemma_lower]["ignore"]

    # Default: ignore unknown function words
    return True


def get_included_function_words() -> set[str]:
    """Get set of function words that are included by default (ignore=False)."""
    return {lemma for lemma, config in FUNCTION_WORDS_CONFIG.items() if not config["ignore"]}


def get_ignored_function_words() -> set[str]:
    """Get set of function words that are ignored by default (ignore=True)."""
    return {lemma for lemma, config in FUNCTION_WORDS_CONFIG.items() if config["ignore"]}
