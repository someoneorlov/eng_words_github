"""EPUB book reader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import List

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from . import BookReader


class EpubExtractionError(Exception):
    """Raised when EPUB parsing fails."""


def _extract_document_text(html_bytes: bytes) -> str:
    """Extract visible text from a single EPUB HTML document."""

    soup = BeautifulSoup(html_bytes, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text


class EpubReader(BookReader):
    """Reader for EPUB format books."""

    def read(self, path: Path, *, block_separator: str = "\n\n") -> str:
        """Read an EPUB file and return concatenated plain text.

        Args:
            path: Path to the `.epub` file.
            block_separator: Separator inserted between document blocks.

        Returns:
            Combined text content extracted from the EPUB.

        Raises:
            FileNotFoundError: If the EPUB file does not exist.
            EpubExtractionError: If parsing fails or no text content is found.
        """

        if not path.exists():
            raise FileNotFoundError(path)

        try:
            book = epub.read_epub(str(path))
        except Exception as exc:  # noqa: BLE001
            raise EpubExtractionError(f"Failed to read EPUB file: {path}") from exc

        documents: List[str] = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            text = _extract_document_text(item.get_content())
            if text:
                documents.append(text)

        if not documents:
            raise EpubExtractionError(f"No textual content extracted from {path}")

        return block_separator.join(documents)

    def supports(self, path: Path) -> bool:
        """Check if this reader supports the given file.

        Args:
            path: Path to the book file.

        Returns:
            True if the file has a `.epub` extension, False otherwise.
        """

        return path.suffix.lower() == ".epub"


# Backward compatibility: export the function for existing code
def extract_epub_text(epub_path: Path, block_separator: str = "\n\n") -> str:
    """Read an EPUB file and return concatenated plain text.

    This function is maintained for backward compatibility.
    New code should use EpubReader class instead.

    Args:
        epub_path: Path to the `.epub` file.
        block_separator: Separator inserted between document blocks.

    Returns:
        Combined text content extracted from the EPUB.

    Raises:
        FileNotFoundError: If the EPUB file does not exist.
        EpubExtractionError: If parsing fails or no text content is found.
    """

    reader = EpubReader()
    return reader.read(epub_path, block_separator=block_separator)
