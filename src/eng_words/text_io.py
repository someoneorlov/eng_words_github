"""Text input utilities: load from file and basic preprocessing."""

from __future__ import annotations

import re
from pathlib import Path

from eng_words.constants import ENCODING_UTF8
from eng_words.epub_reader import extract_epub_text  # Backward compatibility
from eng_words.readers.epub_reader import EpubReader
from eng_words.readers.text_reader import TextReader


class TextLoadError(Exception):
    """Raised when text cannot be loaded from a file."""


def load_text_from_file(file_path: Path, encoding: str = ENCODING_UTF8) -> str:
    """Load raw text from a file.

    Args:
        file_path: Path to the text file.
        encoding: File encoding (default UTF-8).

    Returns:
        The file contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        TextLoadError: For empty files or decoding errors.
    """

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    try:
        text = file_path.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:
        raise TextLoadError(f"Failed to decode file {file_path} with encoding {encoding}") from exc

    if not text:
        raise TextLoadError(f"File {file_path} is empty")

    return text


def normalize_headers(text: str) -> str:
    """Normalize book headers (Book I, Chapter 1, Part II, etc.) by adding periods.

    Args:
        text: Text string potentially containing headers.

    Returns:
        Text with normalized headers.
    """

    if not text:
        return text

    # Pattern for common header formats:
    # - "Book I", "Book II", "Book III", etc.
    # - "Chapter 1", "Chapter 2", etc.
    # - "Part I", "Part II", etc.
    # - Standalone roman numerals (I, II, III, IV, V, etc.)
    # - Standalone numbers (1, 2, 3, etc.) on their own line
    header_patterns = [
        # Book/Chapter/Part with roman numerals or numbers
        (r"^(Book|Chapter|Part)\s+([IVXLCDM]+|\d+)\s*$", r"\1 \2."),
        # Standalone roman numerals on their own line (common in books)
        (r"^([IVXLCDM]+)\s*$", r"\1."),
        # Standalone numbers on their own line (if they look like chapter numbers)
        (r"^(\d+)\s*$", r"\1."),
    ]

    lines = text.split("\n")
    normalized_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            normalized_lines.append(line)
            continue

        # Check if line matches header patterns
        matched = False
        for pattern, replacement in header_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                normalized_lines.append(re.sub(pattern, replacement, stripped, flags=re.IGNORECASE))
                matched = True
                break

        if not matched:
            normalized_lines.append(line)

    return "\n".join(normalized_lines)


def preprocess_text(text: str, normalize_newlines: bool = True) -> str:
    """Basic preprocessing: strip BOM, normalize whitespace/newlines.

    Args:
        text: Raw text string.
        normalize_newlines: If True replace CRLF/CR with LF.

    Returns:
        Cleaned text string.
    """

    if text is None:
        raise ValueError("text must not be None")

    cleaned = text

    # Remove BOM if present
    if cleaned.startswith("\ufeff"):
        cleaned = cleaned.lstrip("\ufeff")

    # Remove zero-width and soft hyphen characters
    for invisible in ("\ufeff", "\u200b", "\u00ad"):
        cleaned = cleaned.replace(invisible, "")

    # Normalize punctuation
    replacements = {
        "—": " - ",
        "–": " - ",
        """: '"',  # Left double quotation mark
        """: '"',  # Right double quotation mark
        "\u201c": '"',  # Left double quotation mark (alternative)
        "\u201d": '"',  # Right double quotation mark (alternative)
        "„": '"',
        "«": '"',
        "»": '"',
        "'": "'",
        "´": "'",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)

    # Normalize headers (Book I, Chapter 1, etc.)
    cleaned = normalize_headers(cleaned)

    if normalize_newlines:
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        # Normalize multiple \n\n to single \n\n
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
        return cleaned.strip()

    return cleaned.rstrip()


def _get_reader_for_path(path: Path):
    """Get the appropriate BookReader for the given file path.

    Args:
        path: Path to the book file.

    Returns:
        BookReader instance that supports the file format.

    Raises:
        TextLoadError: If no reader supports the file format.
    """
    readers = [EpubReader(), TextReader()]

    for reader in readers:
        if reader.supports(path):
            return reader

    raise TextLoadError(f"Unsupported book format: {path.suffix}")


def load_book_text(file_path: Path, *, normalize_newlines: bool = True) -> str:
    """Load and preprocess book text based on file extension.

    Uses the BookReader interface to support multiple formats (EPUB, TXT, etc.).
    """

    reader = _get_reader_for_path(file_path)
    raw_text = reader.read(file_path)
    return preprocess_text(raw_text, normalize_newlines=normalize_newlines)
