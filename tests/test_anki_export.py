from __future__ import annotations

from pathlib import Path

import pandas as pd

from eng_words.anki_export import export_to_anki_csv, prepare_anki_export


def test_prepare_anki_export_basic():
    candidates = pd.DataFrame(
        {"lemma": ["light", "door"], "example": ["Light it up.", "Open the door."]}
    )

    anki_df = prepare_anki_export(candidates, book_name="test_book")

    assert list(anki_df.columns) == ["front", "back", "tags"]
    assert list(anki_df["front"]) == ["light", "door"]
    assert list(anki_df["back"]) == ["Light it up.", "Open the door."]
    assert all(tag == "test_book" for tag in anki_df["tags"])


def test_prepare_anki_export_empty():
    anki_df = prepare_anki_export(pd.DataFrame(columns=["lemma", "example"]), book_name="book")
    assert anki_df.empty
    assert list(anki_df.columns) == ["front", "back", "tags"]


def test_export_to_anki_csv(tmp_path: Path):
    anki_df = pd.DataFrame({"front": ["word"], "back": ["example sentence"], "tags": ["book_tag"]})
    output_file = tmp_path / "anki.csv"

    export_to_anki_csv(anki_df, output_file)

    loaded = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(anki_df, loaded)
