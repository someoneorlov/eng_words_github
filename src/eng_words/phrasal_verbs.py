"""Phrasal verb detection utilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from eng_words.constants import (
    BOOK,
    BOOK_FREQ,
    COMPONENT_PARSER,
    DEFAULT_MODEL_NAME,
    DEP_PRT,
    ITEM_TYPE,
    ITEM_TYPE_PHRASAL_VERB,
    LEMMA,
    PARTICLE,
    PHRASAL,
    PHRASAL_DISABLED,
    PHRASAL_MIN_FREQ_DEFAULT,
    POS_VERB,
    POSITION,
    SCORE,
    SENTENCE_ID,
    SENTENCE_TEXT,
    SPACY_MAX_LENGTH,
    SURFACE,
    VERB,
    WHITESPACE,
)

REQUIRED_TOKEN_COLUMNS = {BOOK, SENTENCE_ID, SURFACE}
PHRASAL_STATS_COLUMNS = [PHRASAL, BOOK_FREQ, ITEM_TYPE]


@lru_cache
def initialize_phrasal_model(model_name: str = DEFAULT_MODEL_NAME) -> Language:
    """Load spaCy model with parser enabled for phrasal verb detection."""

    nlp = spacy.load(model_name, disable=PHRASAL_DISABLED)
    nlp.max_length = max(nlp.max_length, SPACY_MAX_LENGTH)
    if not nlp.has_pipe(COMPONENT_PARSER):
        nlp.add_pipe(COMPONENT_PARSER)
    return nlp


def _sentence_from_group(group: pd.DataFrame) -> str:
    """Reconstruct sentence from token group, using whitespace if available."""
    ordered = group.sort_values(POSITION) if POSITION in group.columns else group

    # If whitespace column is available, use it for accurate reconstruction
    if WHITESPACE in ordered.columns:
        return "".join(
            str(surface) + str(whitespace)
            for surface, whitespace in zip(ordered[SURFACE], ordered[WHITESPACE])
        ).strip()

    # Fallback to space-separated tokens
    return " ".join(str(token) for token in ordered[SURFACE].tolist()).strip()


def _iter_phrasal_candidates(doc: Doc) -> Iterable[tuple[str, str, str]]:
    for sent in doc.sents:
        for token in sent:
            if token.pos_ != POS_VERB:
                continue
            for child in token.children:
                if child.dep_ == DEP_PRT:
                    yield (
                        token.lemma_.lower(),
                        child.text.lower(),
                        sent.text.strip(),
                    )


def detect_phrasal_verbs(tokens_df: pd.DataFrame, nlp: Language) -> pd.DataFrame:
    """Detect phrasal verbs via dependency parsing."""

    if tokens_df is None:
        raise ValueError("tokens_df must be provided")
    if tokens_df.empty:
        return pd.DataFrame(columns=[BOOK, SENTENCE_ID, PHRASAL, VERB, PARTICLE, SENTENCE_TEXT])
    missing = REQUIRED_TOKEN_COLUMNS - set(tokens_df.columns)
    if missing:
        raise ValueError(f"tokens_df is missing required columns: {missing}")
    if not nlp.has_pipe(COMPONENT_PARSER):
        raise ValueError("Provided spaCy model must include a parser for phrasal detection")

    records: List[dict] = []
    for sentence_id, group in tokens_df.groupby(SENTENCE_ID):
        sentence_text = _sentence_from_group(group)
        if not sentence_text:
            continue
        doc = nlp(sentence_text)
        book_name = str(group[BOOK].iloc[0])
        for verb_lemma, particle, sent_text in _iter_phrasal_candidates(doc):
            phrasal = f"{verb_lemma} {particle}".strip()
            records.append(
                {
                    BOOK: book_name,
                    SENTENCE_ID: sentence_id,
                    PHRASAL: phrasal,
                    VERB: verb_lemma,
                    PARTICLE: particle,
                    SENTENCE_TEXT: sent_text,
                }
            )

    if not records:
        return pd.DataFrame(columns=[BOOK, SENTENCE_ID, PHRASAL, VERB, PARTICLE, SENTENCE_TEXT])

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=[BOOK, SENTENCE_ID, PHRASAL]).reset_index(drop=True)
    return df


def calculate_phrasal_verb_stats(phrasal_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate phrasal verb frequencies."""

    if phrasal_df is None:
        raise ValueError("phrasal_df must be provided")

    if phrasal_df.empty:
        return pd.DataFrame(columns=PHRASAL_STATS_COLUMNS)

    if PHRASAL not in phrasal_df.columns:
        raise ValueError(f"phrasal_df is missing required column '{PHRASAL}'")

    stats = (
        phrasal_df.groupby(PHRASAL)
        .size()
        .reset_index(name=BOOK_FREQ)
        .sort_values(BOOK_FREQ, ascending=False)
    )
    stats[ITEM_TYPE] = ITEM_TYPE_PHRASAL_VERB
    return stats[PHRASAL_STATS_COLUMNS].reset_index(drop=True)


def filter_phrasal_verbs(
    phrasal_stats_df: pd.DataFrame,
    known_df: pd.DataFrame | None,
    *,
    min_freq: int = PHRASAL_MIN_FREQ_DEFAULT,
) -> pd.DataFrame:
    """Filter phrasal verbs by known list and frequency."""

    if phrasal_stats_df is None:
        raise ValueError("phrasal_stats_df must be provided")

    if phrasal_stats_df.empty:
        return pd.DataFrame(columns=PHRASAL_STATS_COLUMNS)

    df = phrasal_stats_df.copy()
    df = df[df[BOOK_FREQ] >= max(min_freq, 1)]

    if known_df is not None and not known_df.empty:
        if PHRASAL not in known_df.columns and LEMMA in known_df.columns:
            known_phrasals = known_df[LEMMA].str.lower()
        else:
            known_phrasals = known_df[PHRASAL].astype(str).str.lower()
        df = df[~df[PHRASAL].str.lower().isin(known_phrasals)]

    return df.reset_index(drop=True)


def rank_phrasal_verbs(phrasal_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Rank phrasal verbs by book frequency."""

    if phrasal_stats_df is None:
        raise ValueError("phrasal_stats_df must be provided")

    if phrasal_stats_df.empty:
        return pd.DataFrame(columns=PHRASAL_STATS_COLUMNS + [SCORE])

    if BOOK_FREQ not in phrasal_stats_df.columns:
        raise ValueError(f"phrasal_stats_df must contain '{BOOK_FREQ}'")

    df = phrasal_stats_df.copy()
    max_freq = df[BOOK_FREQ].max()
    df[SCORE] = df[BOOK_FREQ] / max_freq if max_freq else 0.0
    df = df.sort_values(SCORE, ascending=False).reset_index(drop=True)
    return df
