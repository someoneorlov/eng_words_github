from pathlib import Path

import pytest
from ebooklib import epub

from eng_words.readers.epub_reader import extract_epub_text
from eng_words.text_io import load_book_text


def _create_test_epub(tmp_path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title("Test Book")
    book.set_language("en")
    book.add_author("Author")

    chapter1 = epub.EpubHtml(title="Chapter 1", file_name="chap1.xhtml", lang="en")
    chapter1.content = "<h1>Chapter 1</h1><p>Hello world.</p>"

    chapter2 = epub.EpubHtml(title="Chapter 2", file_name="chap2.xhtml", lang="en")
    chapter2.content = "<h1>Chapter 2</h1><p>Second chapter text.</p>"

    book.add_item(chapter1)
    book.add_item(chapter2)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    book.spine = ["nav", chapter1, chapter2]

    epub_path = tmp_path / "test.epub"
    epub.write_epub(str(epub_path), book)
    return epub_path


def test_extract_epub_text_returns_concatenated_text(tmp_path: Path) -> None:
    epub_path = _create_test_epub(tmp_path)

    text = extract_epub_text(epub_path)

    assert "Hello world." in text
    assert "Second chapter text." in text
    assert "\n\n" in text  # separator between documents


def test_extract_epub_text_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        extract_epub_text(Path("missing.epub"))


def test_load_book_text_normalizes_epub(tmp_path: Path) -> None:
    epub_path = _create_test_epub(tmp_path)

    text = load_book_text(epub_path)

    assert "Hello world." in text
    assert "Second chapter text." in text
    assert "\r" not in text
