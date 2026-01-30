from __future__ import annotations

from pathlib import Path

import ebooklib
import pytest
from bs4 import BeautifulSoup
from ebooklib import epub

from eng_words.text_io import load_book_text

SOURCE_EPUb = Path("data/raw/theodore-dreiser_an-american-tragedy.epub")


def _create_excerpt_from_epub(tmp_path: Path, max_documents: int = 2) -> Path:
    if not SOURCE_EPUb.exists():
        pytest.skip(f"Source EPUB not found at {SOURCE_EPUb}")

    book = epub.read_epub(str(SOURCE_EPUb))

    excerpt = epub.EpubBook()
    excerpt.set_identifier("an-american-tragedy-excerpt")
    excerpt.set_title("An American Tragedy (Excerpt)")
    excerpt.set_language("en")
    excerpt.add_author("Theodore Dreiser")

    documents = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        text_preview = BeautifulSoup(item.get_content(), "html.parser").get_text(" ", strip=True)
        if len(text_preview) < 200:
            continue
        documents.append(item)
        if len(documents) == max_documents:
            break

    if not documents:
        raise AssertionError("Source EPUB contains no document items")

    new_docs = []
    for idx, item in enumerate(documents, start=1):
        chapter = epub.EpubHtml(
            title=f"Excerpt Chapter {idx}",
            file_name=f"excerpt_{idx}.xhtml",
            lang="en",
        )
        chapter.content = item.get_content()
        excerpt.add_item(chapter)
        new_docs.append(chapter)

    excerpt.add_item(epub.EpubNcx())
    excerpt.add_item(epub.EpubNav())
    excerpt.spine = ["nav", *new_docs]

    excerpt_path = tmp_path / "an-american-tragedy-excerpt.epub"
    epub.write_epub(str(excerpt_path), excerpt)
    return excerpt_path


def test_load_book_text_on_real_excerpt(tmp_path: Path) -> None:
    excerpt_path = _create_excerpt_from_epub(tmp_path)

    text = load_book_text(excerpt_path)

    assert "Dusk" in text
    assert "American city" in text
    assert 5000 < len(text) < 100000  # excerpt should be limited but non-trivial
