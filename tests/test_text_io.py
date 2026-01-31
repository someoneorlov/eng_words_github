from pathlib import Path

import pytest

from eng_words.text_io import TextLoadError, load_text_from_file, normalize_headers, preprocess_text


def test_load_text_from_file_success(tmp_path: Path) -> None:
    text_file = tmp_path / "book.txt"
    sample_text = "Hello world!"
    text_file.write_text(sample_text, encoding="utf-8")

    loaded = load_text_from_file(text_file)

    assert loaded == sample_text


def test_load_text_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_text_from_file(missing_file)


def test_load_text_empty_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    with pytest.raises(TextLoadError, match="empty"):
        load_text_from_file(empty_file)


def test_load_text_wrong_encoding(tmp_path: Path) -> None:
    """Loading cp1251-encoded file (non-UTF-8) must raise TextLoadError."""
    encoded_file = tmp_path / "cp1251.txt"
    # Cyrillic in cp1251 — not valid UTF-8
    encoded_file.write_bytes("привет".encode("cp1251"))

    with pytest.raises(TextLoadError):
        load_text_from_file(encoded_file)


def test_preprocess_text_normalizes_newlines_and_bom() -> None:
    raw_text = "\ufeffLine 1\r\nLine 2  \rLine 3\n"

    cleaned = preprocess_text(raw_text)

    assert cleaned == "Line 1\nLine 2\nLine 3"


def test_preprocess_text_without_normalization() -> None:
    raw_text = "Line 1\r\nLine 2  \rLine 3\n"

    cleaned = preprocess_text(raw_text, normalize_newlines=False)

    assert "\r\n" in cleaned


def test_preprocess_text_none_input() -> None:
    with pytest.raises(ValueError):
        preprocess_text(None)  # type: ignore[arg-type]


def test_preprocess_text_removes_invisible_and_normalizes_dash() -> None:
    # Using Unicode escapes for smart quotes to avoid syntax errors
    raw_text = "Dusk\ufeff—of a \u201csummer\u201d night\u200b — and soft\u00adhyphen's 'tests'."

    cleaned = preprocess_text(raw_text)

    # Check that invisible chars are removed, dashes normalized, and smart quotes replaced
    assert "\ufeff" not in cleaned
    assert "\u200b" not in cleaned
    assert "\u00ad" not in cleaned
    assert "—" not in cleaned or " - " in cleaned
    assert "\u201c" not in cleaned  # Smart quote should be replaced
    assert "\u201d" not in cleaned  # Smart quote should be replaced
    assert "summer" in cleaned


def test_normalize_headers_book_roman() -> None:
    text = "Book I\n\nBook II\n\nSome content here."

    normalized = normalize_headers(text)

    assert "Book I." in normalized
    assert "Book II." in normalized
    assert "Some content here." in normalized


def test_normalize_headers_chapter_numbers() -> None:
    text = "Chapter 1\n\nChapter 2\n\nText content."

    normalized = normalize_headers(text)

    assert "Chapter 1." in normalized
    assert "Chapter 2." in normalized
    assert "Text content." in normalized


def test_normalize_headers_standalone_roman() -> None:
    text = "I\n\nII\n\nIII\n\nContent starts here."

    normalized = normalize_headers(text)

    assert "I." in normalized
    assert "II." in normalized
    assert "III." in normalized


def test_normalize_headers_preserves_content() -> None:
    text = "Book I\n\nThis is actual content.\n\nBook II\n\nMore content."

    normalized = normalize_headers(text)

    assert "Book I." in normalized
    assert "This is actual content." in normalized
    assert "Book II." in normalized
    assert "More content." in normalized


def test_preprocess_text_normalizes_headers() -> None:
    text = "Book I\n\nBook II\n\nContent here."

    cleaned = preprocess_text(text)

    assert "Book I." in cleaned
    assert "Book II." in cleaned
    assert "Content here." in cleaned


def test_preprocess_text_normalizes_multiple_newlines() -> None:
    text = "Paragraph one.\n\n\n\nParagraph two."

    cleaned = preprocess_text(text)

    # Should normalize multiple \n\n to single \n\n
    assert "\n\n\n" not in cleaned
    assert "Paragraph one.\n\nParagraph two." in cleaned
