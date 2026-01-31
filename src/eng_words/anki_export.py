"""Utilities for preparing Anki exports."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from eng_words.constants import (
    ANKI_COLUMNS,
    BACK,
    EXAMPLE,
    FRONT,
    LEMMA,
    MSG_NO_EXAMPLE,
    PHRASAL,
    TAGS,
)


def _validate_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    if df is None:
        raise ValueError(f"{name} must be provided")
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def prepare_anki_export(candidates_df: pd.DataFrame, book_name: str) -> pd.DataFrame:
    """Convert candidates dataframe to Anki-friendly structure."""

    if not isinstance(book_name, str) or not book_name:
        raise ValueError("book_name must be a non-empty string")

    if candidates_df is None or candidates_df.empty:
        return pd.DataFrame(columns=ANKI_COLUMNS)

    front_column = LEMMA if LEMMA in candidates_df.columns else PHRASAL
    _validate_columns(candidates_df, {front_column, EXAMPLE}, "candidates_df")

    df = candidates_df.copy()
    df[FRONT] = df[front_column]
    df[BACK] = df[EXAMPLE].fillna(MSG_NO_EXAMPLE)
    df[TAGS] = book_name
    df = df[ANKI_COLUMNS].copy()
    return df


def export_to_anki_csv(anki_df: pd.DataFrame, output_path: Path) -> None:
    """Export Anki dataframe to CSV."""

    _validate_columns(anki_df, ANKI_COLUMNS, "anki_df")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    anki_df.to_csv(output_path, index=False)
