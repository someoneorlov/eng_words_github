"""spaCy-based tokenization and lemmatization utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd
import spacy
from spacy.language import Language

from .constants import (
    BOOK,
    COMPONENT_SENTER,
    DEFAULT_MODEL_NAME,
    IS_ALPHA,
    IS_STOP,
    LEMMA,
    MAX_CHARS_DEFAULT,
    POS,
    POSITION,
    SENTENCE,
    SENTENCE_ID,
    SPACY_MAX_LENGTH,
    SURFACE,
    TOKEN_COLUMNS,
    TOKENIZATION_DISABLED,
    WHITESPACE,
)

TokenDict = Dict[str, object]


@lru_cache
def initialize_spacy_model(model_name: str = DEFAULT_MODEL_NAME) -> Language:
    """Load and cache spaCy model."""

    nlp = spacy.load(model_name, disable=TOKENIZATION_DISABLED)
    nlp.enable_pipe(COMPONENT_SENTER)
    nlp.max_length = max(nlp.max_length, SPACY_MAX_LENGTH)
    return nlp


def tokenize_and_lemmatize(
    text: str, nlp: Language, *, sentence_offset: int = 0
) -> List[TokenDict]:
    """Tokenize text and return per-token metadata."""

    if text is None:
        raise ValueError("text must not be None")

    doc = nlp(text)
    records: List[TokenDict] = []
    for sent_id, sent in enumerate(doc.sents, start=sentence_offset):
        for position, token in enumerate(sent):
            records.append(
                {
                    SENTENCE_ID: sent_id,
                    POSITION: position,
                    SURFACE: token.text,
                    LEMMA: token.lemma_.lower(),
                    POS: token.pos_,
                    IS_STOP: bool(token.is_stop),
                    IS_ALPHA: bool(token.is_alpha),
                    WHITESPACE: token.whitespace_,
                }
            )
    return records


def create_tokens_dataframe(tokens: List[TokenDict], book_name: str) -> pd.DataFrame:
    """Create DataFrame with canonical column order."""

    if not book_name:
        raise ValueError("book_name must be provided")

    records = []
    for token in tokens:
        record = {
            **token,
            BOOK: book_name,
        }
        records.append(record)

    df = pd.DataFrame(records, columns=TOKEN_COLUMNS)
    return df


def iterate_book_chunks(text: str, max_chars: int = MAX_CHARS_DEFAULT) -> Iterator[str]:
    """Yield chunks of text not exceeding max_chars, split on paragraph boundaries."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    buffer: List[str] = []
    current_len = 0

    def flush_buffer() -> str | None:
        nonlocal buffer, current_len
        if buffer:
            chunk = "\n\n".join(buffer).strip()
            buffer = []
            current_len = 0
            return chunk if chunk else None
        return None

    def split_large_segment(segment: str) -> Iterator[str]:
        start = 0
        length = len(segment)
        while start < length:
            end = min(start + max_chars, length)
            if end < length:
                split = segment.rfind("\n\n", start, end)
                if split != -1:
                    end = split + 2
                else:
                    for sep in (". ", "! ", "? "):
                        idx = segment.rfind(sep, start, end)
                        if idx != -1:
                            end = idx + len(sep)
                            break
                    else:
                        split = segment.rfind(" ", start, end)
                        if split <= start:
                            split = segment.find(" ", end)
                            if split == -1:
                                split = length
                        end = split
            chunk = segment[start:end].strip()
            if chunk:
                yield chunk
            start = end

    paragraphs = text.split("\n\n")
    for raw_para in paragraphs:
        para = raw_para.strip()
        if not para:
            continue
        part = para
        part_len = len(part)
        # If the paragraph alone is longer than max_chars, split within paragraph
        if part_len > max_chars:
            chunk = flush_buffer()
            if chunk:
                yield chunk
            for segment in split_large_segment(part):
                yield segment
            continue

        if current_len + part_len + 2 > max_chars:
            chunk = flush_buffer()
            if chunk:
                yield chunk

        buffer.append(para)
        current_len += part_len + 2

    chunk = flush_buffer()
    if chunk:
        yield chunk


def tokenize_text_in_chunks(
    text: str, nlp: Language, *, max_chars: int = MAX_CHARS_DEFAULT
) -> List[TokenDict]:
    """Tokenize entire text by processing chunks sequentially."""

    all_tokens: List[TokenDict] = []
    sentence_offset = 0

    for chunk in iterate_book_chunks(text, max_chars=max_chars):
        chunk_tokens = tokenize_and_lemmatize(chunk, nlp, sentence_offset=sentence_offset)
        if chunk_tokens:
            sentence_offset = chunk_tokens[-1][SENTENCE_ID] + 1
            all_tokens.extend(chunk_tokens)

    return all_tokens


def save_tokens_to_parquet(tokens_df: pd.DataFrame, output_path: Path) -> None:
    """Persist tokens dataframe to parquet."""

    if tokens_df is None:
        raise ValueError("tokens_df must not be None")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokens_df.to_parquet(output_path, index=False)


def load_tokens_from_parquet(file_path: Path) -> pd.DataFrame:
    """Load tokens dataframe from parquet."""

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    return pd.read_parquet(file_path)


def reconstruct_sentences_from_tokens(tokens_df: pd.DataFrame) -> List[str]:
    """Reconstruct sentences from tokens DataFrame using whitespace information."""

    if tokens_df is None or tokens_df.empty:
        return []

    required_columns = {SENTENCE_ID, SURFACE, WHITESPACE}
    missing = required_columns - set(tokens_df.columns)
    if missing:
        raise ValueError(f"tokens_df is missing required columns: {missing}")

    sentences: List[str] = []
    for sentence_id, group in tokens_df.groupby(SENTENCE_ID, sort=True):
        # Sort by position to maintain token order
        group = group.sort_values(POSITION)
        # Reconstruct sentence: surface + whitespace for each token
        sentence = "".join(
            str(surface) + str(whitespace)
            for surface, whitespace in zip(group[SURFACE], group[WHITESPACE])
        )
        sentences.append(sentence.strip())

    return sentences


def extract_sentences(text: str, nlp: Language) -> List[str]:
    """Extract normalized sentences from text using spaCy."""

    if text is None:
        raise ValueError("text must not be None")
    if nlp is None:
        raise ValueError("nlp must be provided")

    doc = nlp(str(text))
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def create_sentences_dataframe(sentences: List[str]) -> pd.DataFrame:
    """Build dataframe with sentence ids."""

    if sentences is None:
        raise ValueError("sentences must be provided")

    records = [{SENTENCE_ID: idx, SENTENCE: sentence} for idx, sentence in enumerate(sentences)]
    columns = [SENTENCE_ID, SENTENCE]
    df = pd.DataFrame(records, columns=columns)
    if df.empty:
        df = pd.DataFrame(columns=columns)
    return df
