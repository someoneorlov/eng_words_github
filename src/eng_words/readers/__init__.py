"""Book readers for various formats (EPUB, PDF, TXT, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

__all__ = ["BookReader"]


class BookReader(ABC):
    """Abstract interface for reading books in various formats."""

    @abstractmethod
    def read(self, path: Path) -> str:
        """Read book and return full text content.

        Args:
            path: Path to the book file.

        Returns:
            Full text content extracted from the book.

        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: Format-specific errors (e.g., EpubExtractionError).
        """
        pass

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Check if this reader supports the given file.

        Args:
            path: Path to the book file.

        Returns:
            True if this reader can handle the file format, False otherwise.
        """
        pass
