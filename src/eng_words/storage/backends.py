"""Storage backends for known words data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

try:
    import gspread
    from google.oauth2.service_account import Credentials

    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

from ..constants import (
    ITEM_TYPE,
    LEMMA,
    REQUIRED_KNOWN_WORDS_COLUMNS,
    STATUS,
    STATUS_KNOWN,
    TAGS,
)


class KnownWordsBackend(ABC):
    """Abstract base class for known words storage backends.

    This interface defines the contract for storing and retrieving known words data.
    All backends must implement the core abstract methods: `load()`, `save()`, and `exists()`.

    **Data Format:**
        All backends work with DataFrames containing the following required columns:
        - `lemma` (str): Lowercased word lemma
        - `status` (str): One of "known", "learning", "ignore"
        - `item_type` (str): One of "word", "phrasal_verb"
        - `tags` (str): Optional tags/metadata

    **Core Methods (Required):**
        - `load()`: Load all known words as a DataFrame
        - `save(df)`: Save/replace all known words from a DataFrame
        - `exists()`: Check if backend is accessible

    **Optional Methods (Default Implementations):**
        The following methods have default implementations that use `load()`/`save()`
        as fallback. Backends can override these for better performance:
        - `is_known(lemma, item_type)`: Check if a word is marked as known
        - `mark_as_learned(lemma, item_type, tags)`: Mark a word as learned
        - `get_learning_progress()`: Get counts by status
        - `update_words(updates)`: Incrementally update multiple words
        - `sync(other_backend)`: Bidirectional sync with another backend

    **Future Backend Support:**
        This interface is designed to support:
        - Anki sync (export learned cards, import known words)
        - Database backends (SQLite, PostgreSQL) with efficient queries
        - Remote API backends with incremental updates
    """

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load known words data.

        Returns:
            DataFrame with known words metadata. Must contain columns:
            `lemma`, `status`, `item_type`, `tags`. Returns empty DataFrame
            with correct columns if no data exists.

        Raises:
            FileNotFoundError: If the backend source doesn't exist.
            ValueError: If the data format is invalid.
        """
        pass

    @abstractmethod
    def save(self, df: pd.DataFrame) -> None:
        """Save known words data.

        This method replaces all existing data with the provided DataFrame.
        For incremental updates, use `update_words()` if available, or override
        this method in your backend implementation.

        Args:
            df: DataFrame with known words metadata to save. Must contain
                columns: `lemma`, `status`, `item_type`, `tags`.

        Raises:
            ValueError: If the DataFrame format is invalid.
        """
        pass

    @abstractmethod
    def exists(self) -> bool:
        """Check if the backend source exists/is accessible.

        Returns:
            True if the backend is accessible, False otherwise.
        """
        pass

    def is_known(self, lemma: str, item_type: str) -> bool:
        """Check if a word is marked as known.

        Default implementation loads all data and checks. Backends can override
        for better performance (e.g., database queries).

        Args:
            lemma: Word lemma (will be normalized to lowercase).
            item_type: Item type (e.g., "word", "phrasal_verb").

        Returns:
            True if the word exists with status "known", False otherwise.
        """
        df = self.load()
        if df.empty:
            return False

        lemma_normalized = str(lemma).strip().lower()
        item_type_normalized = str(item_type).strip()

        mask = (df[LEMMA] == lemma_normalized) & (df[ITEM_TYPE] == item_type_normalized)
        matching = df[mask]

        if matching.empty:
            return False

        # Return True if any matching entry has status "known"
        return bool((matching[STATUS] == STATUS_KNOWN).any())

    def mark_as_learned(self, lemma: str, item_type: str, tags: str = "") -> None:
        """Mark a word as learned (status="known").

        Default implementation loads all data, updates the entry, and saves.
        Backends can override for better performance (e.g., single-row updates).

        Args:
            lemma: Word lemma (will be normalized to lowercase).
            item_type: Item type (e.g., "word", "phrasal_verb").
            tags: Optional tags/metadata.

        Raises:
            ValueError: If backend is not accessible.
        """
        df = self.load()

        lemma_normalized = str(lemma).strip().lower()
        item_type_normalized = str(item_type).strip()
        tags_normalized = str(tags).strip() if tags else ""

        # Create new entry
        new_entry = pd.DataFrame(
            {
                LEMMA: [lemma_normalized],
                STATUS: [STATUS_KNOWN],
                ITEM_TYPE: [item_type_normalized],
                TAGS: [tags_normalized],
            }
        )

        # Remove existing entry if present (deduplication happens in _normalize_dataframe)
        if not df.empty:
            mask = (df[LEMMA] == lemma_normalized) & (df[ITEM_TYPE] == item_type_normalized)
            df = df[~mask]

        # Combine and save
        df = pd.concat([df, new_entry], ignore_index=True)
        self.save(df)

    def get_learning_progress(self) -> dict[str, int]:
        """Get learning progress statistics.

        Returns a dictionary with counts of words by status.

        Default implementation loads all data and counts. Backends can override
        for better performance (e.g., database aggregation).

        Returns:
            Dictionary with status as keys and counts as values.
            Example: {"known": 100, "learning": 5, "ignore": 10}

        Raises:
            ValueError: If backend is not accessible.
        """
        df = self.load()
        if df.empty:
            return {}

        counts = df[STATUS].value_counts().to_dict()
        return dict(counts)

    def update_words(self, updates: pd.DataFrame) -> None:
        """Incrementally update multiple words.

        Merges the provided updates with existing data. Updates override existing
        entries for the same (lemma, item_type) combination.

        Default implementation loads all data, merges, and saves. Backends can
        override for better performance (e.g., batch updates).

        Args:
            updates: DataFrame with updates. Must contain columns:
                `lemma`, `status`, `item_type`, `tags`. Only rows present
                will be updated/added.

        Raises:
            ValueError: If the DataFrame format is invalid or backend is not accessible.
        """
        if updates is None or updates.empty:
            return

        # Validate updates
        missing = [col for col in REQUIRED_KNOWN_WORDS_COLUMNS if col not in updates.columns]
        if missing:
            raise ValueError(f"Updates DataFrame missing columns: {missing}")

        # Normalize updates
        updates_normalized = self._normalize_dataframe(updates)

        # Load existing data (handle case where backend doesn't exist yet)
        try:
            df = self.load()
        except FileNotFoundError:
            # Backend doesn't exist yet, just save updates
            self.save(updates_normalized)
            return

        if df.empty:
            # No existing data, just save updates
            self.save(updates_normalized)
            return

        # Remove entries that will be updated
        # Create a set of (lemma, item_type) tuples from updates for efficient lookup
        update_keys = set(zip(updates_normalized[LEMMA], updates_normalized[ITEM_TYPE]))
        mask = df.apply(
            lambda row: (row[LEMMA], row[ITEM_TYPE]) in update_keys,
            axis=1,
        )
        df = df[~mask]

        # Combine and save
        df = pd.concat([df, updates_normalized], ignore_index=True)
        self.save(df)

    def sync(self, other_backend: KnownWordsBackend) -> None:
        """Synchronize data bidirectionally with another backend.

        This is an advanced operation that merges data from both backends,
        resolving conflicts (typically keeping the most recent entry).

        Default implementation raises NotImplementedError. Backends that support
        sync (e.g., Anki, remote APIs) should override this method.

        Args:
            other_backend: Another KnownWordsBackend instance to sync with.

        Raises:
            NotImplementedError: If sync is not supported by this backend.
        """
        raise NotImplementedError(
            f"Bidirectional sync is not supported by {self.__class__.__name__}. "
            "Override sync() method to implement sync functionality."
        )

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and validate known words DataFrame.

        Args:
            df: Raw DataFrame to normalize.

        Returns:
            Normalized DataFrame.

        Raises:
            ValueError: If required columns are missing.
        """
        if df.empty:
            return pd.DataFrame(columns=REQUIRED_KNOWN_WORDS_COLUMNS)

        missing = [col for col in REQUIRED_KNOWN_WORDS_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Known words data missing columns: {missing}")

        df = df[REQUIRED_KNOWN_WORDS_COLUMNS].copy()
        df[LEMMA] = df[LEMMA].astype(str).str.strip().str.lower()
        df[STATUS] = df[STATUS].astype(str).str.strip().str.lower()
        df[ITEM_TYPE] = df[ITEM_TYPE].astype(str).str.strip()
        df[TAGS] = df[TAGS].astype(str).fillna("")

        df = df.drop_duplicates(subset=[LEMMA, ITEM_TYPE], keep="last").reset_index(drop=True)
        return df


class CSVBackend(KnownWordsBackend):
    """CSV file backend for known words storage."""

    def __init__(self, csv_path: Path | str) -> None:
        """Initialize CSV backend.

        Args:
            csv_path: Path to the CSV file.
        """
        self.path = Path(csv_path)

    def load(self) -> pd.DataFrame:
        """Load known words from CSV file."""
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        try:
            df = pd.read_csv(self.path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=REQUIRED_KNOWN_WORDS_COLUMNS)

        return self._normalize_dataframe(df)

    def save(self, df: pd.DataFrame) -> None:
        """Save known words to CSV file."""
        if df is None:
            raise ValueError("DataFrame must not be None")

        # Validate before saving
        normalized = self._normalize_dataframe(df)

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        normalized.to_csv(self.path, index=False)

    def exists(self) -> bool:
        """Check if CSV file exists."""
        return self.path.exists()


class GoogleSheetsBackend(KnownWordsBackend):
    """Google Sheets backend for known words storage."""

    def __init__(
        self,
        spreadsheet_id: str,
        worksheet_name: str = "Sheet1",
        credentials_path: Path | str | None = None,
    ) -> None:
        """Initialize Google Sheets backend.

        Args:
            spreadsheet_id: Google Sheets spreadsheet ID.
            worksheet_name: Name of the worksheet to use (default: "Sheet1").
            credentials_path: Path to Google service account JSON credentials file.
                If None, uses default credentials from environment.
        """
        self.spreadsheet_id = spreadsheet_id
        self.worksheet_name = worksheet_name
        self.credentials_path = Path(credentials_path) if credentials_path else None
        self._client = None
        self._worksheet = None

    def _get_client(self):
        """Get or create gspread client."""
        if not GSPREAD_AVAILABLE:
            raise ImportError(
                "gspread and google-auth are required for Google Sheets backend. "
                "Install with: pip install gspread google-auth"
            )

        if self._client is None:
            if self.credentials_path:
                if not self.credentials_path.exists():
                    raise FileNotFoundError(
                        f"Google credentials file not found: {self.credentials_path}"
                    )
                creds = Credentials.from_service_account_file(
                    str(self.credentials_path),
                    scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive.readonly",
                    ],
                )
            else:
                # Try to use default credentials from environment
                creds = Credentials.from_service_account_info(
                    self._get_credentials_from_env(),
                    scopes=[
                        "https://www.googleapis.com/auth/spreadsheets",
                        "https://www.googleapis.com/auth/drive.readonly",
                    ],
                )

            self._client = gspread.authorize(creds)

        return self._client

    def _get_credentials_from_env(self) -> dict:
        """Get credentials from environment variable."""
        import json
        import os

        creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        if creds_json:
            return json.loads(creds_json)

        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            path = Path(creds_path)
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)

        raise ValueError(
            "Google credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS "
            "or GOOGLE_APPLICATION_CREDENTIALS_JSON environment variable, "
            "or provide credentials_path parameter."
        )

    def _get_worksheet(self):
        """Get or create worksheet object."""
        if self._worksheet is None:
            client = self._get_client()
            spreadsheet = client.open_by_key(self.spreadsheet_id)

            try:
                self._worksheet = spreadsheet.worksheet(self.worksheet_name)
            except gspread.exceptions.WorksheetNotFound:  # type: ignore[attr-defined]
                # Create worksheet if it doesn't exist
                self._worksheet = spreadsheet.add_worksheet(
                    title=self.worksheet_name, rows=1000, cols=10
                )
                # Set header row
                self._worksheet.append_row(REQUIRED_KNOWN_WORDS_COLUMNS)

        return self._worksheet

    def load(self) -> pd.DataFrame:
        """Load known words from Google Sheets."""
        try:
            worksheet = self._get_worksheet()
            records = worksheet.get_all_records()
        except gspread.exceptions.APIError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"Failed to access Google Sheets: {exc}") from exc

        if not records:
            return pd.DataFrame(columns=REQUIRED_KNOWN_WORDS_COLUMNS)

        df = pd.DataFrame(records)
        return self._normalize_dataframe(df)

    def save(self, df: pd.DataFrame) -> None:
        """Save known words to Google Sheets."""
        if df is None:
            raise ValueError("DataFrame must not be None")

        # Validate and normalize before saving
        normalized = self._normalize_dataframe(df)

        try:
            worksheet = self._get_worksheet()

            # Clear existing data (except header)
            worksheet.clear()

            # Write header
            worksheet.append_row(REQUIRED_KNOWN_WORDS_COLUMNS)

            # Write data rows
            if not normalized.empty:
                values = normalized.values.tolist()
                worksheet.append_rows(values)
        except gspread.exceptions.APIError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"Failed to save to Google Sheets: {exc}") from exc

    def exists(self) -> bool:
        """Check if Google Sheets spreadsheet is accessible."""
        try:
            self._get_client().open_by_key(self.spreadsheet_id)
            return True
        except Exception:  # noqa: BLE001
            return False
