"""Tests for storage backends."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from eng_words.constants import (
    ITEM_TYPE,
    ITEM_TYPE_WORD,
    LEMMA,
    REQUIRED_KNOWN_WORDS_COLUMNS,
    STATUS,
    STATUS_KNOWN,
    TAGS,
)
from eng_words.storage import CSVBackend, GoogleSheetsBackend, load_known_words, save_known_words


def test_csv_backend_load_success(tmp_path: Path) -> None:
    """Test CSV backend loads data correctly."""
    csv_path = tmp_path / "known_words.csv"
    df = pd.DataFrame(
        {
            LEMMA: ["run", "jump"],
            STATUS: [STATUS_KNOWN, STATUS_KNOWN],
            ITEM_TYPE: [ITEM_TYPE_WORD, ITEM_TYPE_WORD],
            TAGS: ["A2", "A1"],
        }
    )
    df.to_csv(csv_path, index=False)

    backend = CSVBackend(csv_path)
    loaded = backend.load()

    assert len(loaded) == 2
    assert list(loaded[LEMMA]) == ["run", "jump"]


def test_csv_backend_load_empty_file(tmp_path: Path) -> None:
    """Test CSV backend handles empty file."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("")

    backend = CSVBackend(csv_path)
    loaded = backend.load()

    assert loaded.empty
    assert list(loaded.columns) == REQUIRED_KNOWN_WORDS_COLUMNS


def test_csv_backend_load_missing_file(tmp_path: Path) -> None:
    """Test CSV backend raises FileNotFoundError for missing file."""
    csv_path = tmp_path / "missing.csv"

    backend = CSVBackend(csv_path)
    with pytest.raises(FileNotFoundError):
        backend.load()


def test_csv_backend_save(tmp_path: Path) -> None:
    """Test CSV backend saves data correctly."""
    csv_path = tmp_path / "known_words.csv"
    df = pd.DataFrame(
        {
            LEMMA: ["run", "jump"],
            STATUS: [STATUS_KNOWN, STATUS_KNOWN],
            ITEM_TYPE: [ITEM_TYPE_WORD, ITEM_TYPE_WORD],
            TAGS: ["A2", "A1"],
        }
    )

    backend = CSVBackend(csv_path)
    backend.save(df)

    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == 2
    assert list(loaded[LEMMA]) == ["run", "jump"]


def test_csv_backend_exists(tmp_path: Path) -> None:
    """Test CSV backend exists() method."""
    csv_path = tmp_path / "known_words.csv"

    backend = CSVBackend(csv_path)
    assert not backend.exists()

    csv_path.write_text("test")
    assert backend.exists()


def test_csv_backend_normalizes_data(tmp_path: Path) -> None:
    """Test CSV backend normalizes data (lowercase, strip, deduplicate)."""
    csv_path = tmp_path / "known_words.csv"
    df = pd.DataFrame(
        {
            LEMMA: ["  RUN  ", "run", "Jump"],
            STATUS: ["KNOWN", " known ", "KNOWN"],
            ITEM_TYPE: [ITEM_TYPE_WORD, ITEM_TYPE_WORD, ITEM_TYPE_WORD],
            TAGS: ["A2", "A2", "A1"],
        }
    )
    df.to_csv(csv_path, index=False)

    backend = CSVBackend(csv_path)
    loaded = backend.load()

    # Should normalize and deduplicate
    assert len(loaded) == 2  # "run" appears twice, should be deduplicated
    assert set(loaded[LEMMA]) == {"run", "jump"}
    assert all(loaded[STATUS] == STATUS_KNOWN)


def test_load_known_words_csv(tmp_path: Path) -> None:
    """Test universal loader with CSV file."""
    csv_path = tmp_path / "known_words.csv"
    df = pd.DataFrame(
        {
            LEMMA: ["run"],
            STATUS: [STATUS_KNOWN],
            ITEM_TYPE: [ITEM_TYPE_WORD],
            TAGS: ["A2"],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = load_known_words(csv_path)
    assert len(loaded) == 1
    assert loaded.iloc[0][LEMMA] == "run"


def test_load_known_words_gsheets_url() -> None:
    """Test universal loader detects Google Sheets URL."""
    from eng_words.storage.loader import _get_backend

    backend = _get_backend("gsheets://abc123/Sheet1")
    assert isinstance(backend, GoogleSheetsBackend)
    assert backend.spreadsheet_id == "abc123"
    assert backend.worksheet_name == "Sheet1"

    # Default worksheet name
    backend = _get_backend("gsheets://abc123")
    assert isinstance(backend, GoogleSheetsBackend)
    assert backend.worksheet_name == "Sheet1"


def test_save_known_words_csv(tmp_path: Path) -> None:
    """Test universal saver with CSV file."""
    csv_path = tmp_path / "known_words.csv"
    df = pd.DataFrame(
        {
            LEMMA: ["run"],
            STATUS: [STATUS_KNOWN],
            ITEM_TYPE: [ITEM_TYPE_WORD],
            TAGS: ["A2"],
        }
    )

    save_known_words(df, csv_path)

    assert csv_path.exists()
    loaded = pd.read_csv(csv_path)
    assert len(loaded) == 1


def test_google_sheets_backend_load_mock() -> None:
    """Test Google Sheets backend load with mocked gspread."""
    # Mock gspread
    mock_client = Mock()
    mock_worksheet = Mock()
    mock_worksheet.get_all_records.return_value = [
        {LEMMA: "run", STATUS: STATUS_KNOWN, ITEM_TYPE: ITEM_TYPE_WORD, TAGS: "A2"},
        {LEMMA: "jump", STATUS: STATUS_KNOWN, ITEM_TYPE: ITEM_TYPE_WORD, TAGS: "A1"},
    ]
    mock_spreadsheet = Mock()
    mock_spreadsheet.worksheet.return_value = mock_worksheet
    mock_client.open_by_key.return_value = mock_spreadsheet

    with patch("eng_words.storage.backends.gspread.authorize", return_value=mock_client):
        with patch(
            "eng_words.storage.backends.Credentials.from_service_account_file",
            return_value=Mock(),
        ):
            with patch("pathlib.Path.exists", return_value=True):
                backend = GoogleSheetsBackend(
                    "test_id", "Sheet1", credentials_path=Path("/fake/path.json")
                )
                loaded = backend.load()

                assert len(loaded) == 2
                assert list(loaded[LEMMA]) == ["run", "jump"]


def test_google_sheets_backend_save_mock() -> None:
    """Test Google Sheets backend save with mocked gspread."""
    # Mock gspread
    mock_client = Mock()
    mock_worksheet = Mock()
    mock_spreadsheet = Mock()
    mock_spreadsheet.worksheet.return_value = mock_worksheet
    mock_client.open_by_key.return_value = mock_spreadsheet

    df = pd.DataFrame(
        {
            LEMMA: ["run"],
            STATUS: [STATUS_KNOWN],
            ITEM_TYPE: [ITEM_TYPE_WORD],
            TAGS: ["A2"],
        }
    )

    with patch("eng_words.storage.backends.gspread.authorize", return_value=mock_client):
        with patch(
            "eng_words.storage.backends.Credentials.from_service_account_file",
            return_value=Mock(),
        ):
            with patch("pathlib.Path.exists", return_value=True):
                backend = GoogleSheetsBackend(
                    "test_id", "Sheet1", credentials_path=Path("/fake/path.json")
                )
                backend.save(df)

                # Verify worksheet methods were called
                mock_worksheet.clear.assert_called_once()
                assert mock_worksheet.append_row.call_count >= 1  # Header + data


def test_google_sheets_backend_creates_worksheet_if_missing() -> None:
    """Test Google Sheets backend creates worksheet if it doesn't exist."""
    import gspread.exceptions

    # Mock gspread
    mock_client = Mock()
    mock_worksheet = Mock()
    mock_spreadsheet = Mock()
    mock_spreadsheet.worksheet.side_effect = gspread.exceptions.WorksheetNotFound("Not found")
    mock_spreadsheet.add_worksheet.return_value = mock_worksheet
    mock_client.open_by_key.return_value = mock_spreadsheet

    with patch("eng_words.storage.backends.gspread.authorize", return_value=mock_client):
        with patch(
            "eng_words.storage.backends.Credentials.from_service_account_file",
            return_value=Mock(),
        ):
            with patch("pathlib.Path.exists", return_value=True):
                backend = GoogleSheetsBackend(
                    "test_id", "NewSheet", credentials_path=Path("/fake/path.json")
                )
                worksheet = backend._get_worksheet()

                mock_spreadsheet.add_worksheet.assert_called_once()
                assert worksheet == mock_worksheet
