"""Plain text book reader implementation."""

from __future__ import annotations

from pathlib import Path

from . import BookReader
from ..constants import ENCODING_UTF8


class TextLoadError(Exception):
    """Raised when text cannot be loaded from a file."""


class TextReader(BookReader):
    """Reader for plain text format books."""

    def read(self, path: Path, *, encoding: str = ENCODING_UTF8) -> str:
        """Load raw text from a file.

        Args:
            path: Path to the text file.
            encoding: File encoding (default UTF-8).

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
            TextLoadError: For empty files or decoding errors.
        """

        if not path.exists():
            raise FileNotFoundError(path)

        try:
            text = path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            raise TextLoadError(f"Failed to decode file {path} with encoding {encoding}") from exc

        if not text:
            raise TextLoadError(f"File {path} is empty")

        return text

    def supports(self, path: Path) -> bool:
        """Check if this reader supports the given file.

        Args:
            path: Path to the book file.

        Returns:
            True if the file has a `.txt` or `.text` extension, False otherwise.
        """

        return path.suffix.lower() in {".txt", ".text"}
