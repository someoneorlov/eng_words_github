"""Universal loader for known words from various backends."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from eng_words.storage.backends import CSVBackend, GoogleSheetsBackend, KnownWordsBackend

# Pattern for Google Sheets URL: gsheets://spreadsheet_id/worksheet_name
GSHEETS_PATTERN = re.compile(r"^gsheets://([^/]+)(?:/(.+))?$")


def load_known_words(source: str | Path) -> pd.DataFrame:
    """Load known words from various backends.

    Supports:
    - CSV files: Path or string ending in .csv
    - Google Sheets: gsheets://spreadsheet_id/worksheet_name format

    Args:
        source: Source identifier (file path or Google Sheets URL).

    Returns:
        DataFrame with known words metadata.

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If source format is invalid or data format is invalid.

    Examples:
        >>> # Load from CSV
        >>> df = load_known_words("data/known_words.csv")
        >>> df = load_known_words(Path("data/known_words.csv"))

        >>> # Load from Google Sheets
        >>> df = load_known_words("gsheets://abc123/Sheet1")
    """
    backend = _get_backend(source)
    return backend.load()


def save_known_words(df: pd.DataFrame, source: str | Path) -> None:
    """Save known words to various backends.

    Args:
        df: DataFrame with known words metadata to save.
        source: Destination identifier (file path or Google Sheets URL).

    Raises:
        ValueError: If source format is invalid or data format is invalid.
    """
    backend = _get_backend(source)
    backend.save(df)


def _get_backend(source: str | Path) -> KnownWordsBackend:
    """Get appropriate backend for the given source.

    Args:
        source: Source identifier.

    Returns:
        Backend instance.

    Raises:
        ValueError: If source format is not recognized.
    """
    source_str = str(source)

    # Check for Google Sheets URL
    match = GSHEETS_PATTERN.match(source_str)
    if match:
        spreadsheet_id = match.group(1)
        worksheet_name = match.group(2) or "Sheet1"
        return GoogleSheetsBackend(spreadsheet_id, worksheet_name)

    # Default to CSV
    return CSVBackend(source)
