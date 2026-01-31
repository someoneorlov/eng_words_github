"""Utilities for extracting text from EPUB files.

This module is maintained for backward compatibility.
New code should use `eng_words.readers.epub_reader.EpubReader` instead.
"""

from __future__ import annotations

# Re-export from the new location for backward compatibility
from eng_words.readers.epub_reader import EpubExtractionError, extract_epub_text

__all__ = ["EpubExtractionError", "extract_epub_text"]
