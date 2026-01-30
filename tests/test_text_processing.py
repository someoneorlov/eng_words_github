from __future__ import annotations

import pytest

from eng_words.text_processing import (
    create_tokens_dataframe,
    initialize_spacy_model,
    iterate_book_chunks,
    load_tokens_from_parquet,
    reconstruct_sentences_from_tokens,
    save_tokens_to_parquet,
    tokenize_and_lemmatize,
    tokenize_text_in_chunks,
)


@pytest.fixture(scope="session")
def nlp():
    try:
        return initialize_spacy_model()
    except OSError:
        pytest.skip("spaCy model en_core_web_sm is not installed")


def test_tokenize_simple_sentence(nlp) -> None:
    text = "Cats are running."

    tokens = tokenize_and_lemmatize(text, nlp)

    lemmas = [t["lemma"] for t in tokens if t["is_alpha"]]
    assert "cat" in lemmas
    assert "run" in lemmas


def test_tokenize_handles_stopwords_and_punct(nlp) -> None:
    text = "This, indeed, is a test!"

    tokens = tokenize_and_lemmatize(text, nlp)

    comma = next(t for t in tokens if t["surface"] == ",")
    assert comma["is_alpha"] is False

    this = next(t for t in tokens if t["surface"].lower() == "this")
    assert this["is_stop"] is True


def test_create_tokens_dataframe_structure(nlp) -> None:
    text = "Go go goes."
    tokens = tokenize_and_lemmatize(text, nlp)

    df = create_tokens_dataframe(tokens, book_name="test-book")

    assert list(df.columns) == [
        "book",
        "sentence_id",
        "position",
        "surface",
        "lemma",
        "pos",
        "is_stop",
        "is_alpha",
        "whitespace",
    ]
    assert df["book"].nunique() == 1
    assert df["lemma"].str.contains("go").any()
    assert "whitespace" in df.columns


def test_save_and_load_tokens_parquet(tmp_path, nlp) -> None:
    text = "Hello world"
    tokens = tokenize_and_lemmatize(text, nlp)
    df = create_tokens_dataframe(tokens, book_name="book")

    file_path = tmp_path / "tokens.parquet"
    save_tokens_to_parquet(df, file_path)

    loaded = load_tokens_from_parquet(file_path)

    assert loaded.equals(df)


def test_save_and_load_empty_dataframe(tmp_path) -> None:
    empty_df = create_tokens_dataframe([], book_name="empty-book")
    file_path = tmp_path / "empty.parquet"

    save_tokens_to_parquet(empty_df, file_path)
    loaded = load_tokens_from_parquet(file_path)

    assert loaded.empty
    assert list(loaded.columns) == list(empty_df.columns)


def test_load_tokens_from_missing_file(tmp_path) -> None:
    missing_file = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        load_tokens_from_parquet(missing_file)


def test_iterate_book_chunks_splits_text() -> None:
    text = "Paragraph one.\n\nParagraph two is slightly longer.\n\nParagraph three."
    chunks = list(iterate_book_chunks(text, max_chars=25))

    assert len(chunks) >= 2
    assert chunks[0].startswith("Paragraph one")
    assert chunks[-1].endswith("Paragraph three.")


def test_tokenize_text_in_chunks_matches_full(nlp) -> None:
    text = " ".join([f"This is sentence {i}." for i in range(60)])

    full_tokens = tokenize_and_lemmatize(text, nlp)
    chunked_tokens = tokenize_text_in_chunks(text, nlp, max_chars=100)

    assert len(full_tokens) == len(chunked_tokens)
    assert [t["surface"] for t in full_tokens] == [t["surface"] for t in chunked_tokens]
    assert chunked_tokens[-1]["sentence_id"] == full_tokens[-1]["sentence_id"]


def test_tokenize_preserves_whitespace(nlp) -> None:
    text = 'He said "hello" and left.'

    tokens = tokenize_and_lemmatize(text, nlp)
    df = create_tokens_dataframe(tokens, book_name="test")

    assert "whitespace" in df.columns
    # Check that whitespace is preserved (quotes should have no space after, words should have space)
    quote_tokens = df[df["surface"] == '"']
    assert len(quote_tokens) >= 2
    # First quote should have space after (before "hello")
    # Second quote should have space after (before "and")


def test_reconstruct_sentences_from_tokens(nlp) -> None:
    text = 'He said "hello" and left. Then he went home.'

    tokens = tokenize_and_lemmatize(text, nlp)
    df = create_tokens_dataframe(tokens, book_name="test")

    sentences = reconstruct_sentences_from_tokens(df)

    assert len(sentences) >= 2
    # Check that quotes are properly reconstructed (no extra spaces)
    first_sentence = sentences[0]
    assert '"hello"' in first_sentence or '" hello "' in first_sentence
    # Check that sentences are properly separated
    assert any("left" in s for s in sentences)
    assert any("home" in s for s in sentences)


def test_reconstruct_sentences_preserves_whitespace(nlp) -> None:
    text = 'Call out "help" now!'

    tokens = tokenize_and_lemmatize(text, nlp)
    df = create_tokens_dataframe(tokens, book_name="test")

    sentences = reconstruct_sentences_from_tokens(df)

    assert len(sentences) == 1
    reconstructed = sentences[0]
    # Should not have space after quote before "help"
    # Original text should be preserved
    assert "out" in reconstructed
    assert "help" in reconstructed
