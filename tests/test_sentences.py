from __future__ import annotations

import pytest

from eng_words.text_processing import (
    create_sentences_dataframe,
    extract_sentences,
    initialize_spacy_model,
)


@pytest.fixture(scope="module")
def sent_nlp():
    try:
        return initialize_spacy_model()
    except OSError:
        pytest.skip("spaCy model en_core_web_sm is not installed")


def test_extract_sentences_basic(sent_nlp):
    text = "First sentence. Second sentence! Third one?"

    sentences = extract_sentences(text, sent_nlp)

    assert sentences == ["First sentence.", "Second sentence!", "Third one?"]


def test_extract_sentences_empty_string(sent_nlp):
    sentences = extract_sentences("   ", sent_nlp)
    assert sentences == []


def test_create_sentences_dataframe_structure():
    sentences = ["One.", "Two."]

    df = create_sentences_dataframe(sentences)

    assert list(df.columns) == ["sentence_id", "sentence"]
    assert list(df["sentence_id"]) == [0, 1]
    assert list(df["sentence"]) == sentences


def test_create_sentences_dataframe_empty():
    df = create_sentences_dataframe([])
    assert df.empty
    assert list(df.columns) == ["sentence_id", "sentence"]
