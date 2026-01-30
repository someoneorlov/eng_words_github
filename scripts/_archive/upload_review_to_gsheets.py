#!/usr/bin/env python3
"""Upload review CSV to Google Sheets for manual marking.

Reads a review CSV file and uploads it to Google Sheets for easy manual review.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from eng_words.storage import save_known_words  # noqa: E402


def main() -> None:
    """Upload review CSV to Google Sheets."""
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(description="Upload review CSV to Google Sheets")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to review CSV file (from export_for_review.py)",
    )
    parser.add_argument(
        "--gsheets-url",
        type=str,
        help="Google Sheets URL (gsheets://spreadsheet_id/worksheet_name). "
        "If not provided, uses GOOGLE_SHEETS_REVIEW_URL from .env or GOOGLE_SHEETS_URL",
    )

    args = parser.parse_args()

    if not args.csv.exists():
        print(f"❌ File not found: {args.csv}")
        sys.exit(1)

    # Determine Google Sheets URL
    gsheets_url = args.gsheets_url
    if not gsheets_url:
        gsheets_url = os.getenv("GOOGLE_SHEETS_REVIEW_URL") or os.getenv("GOOGLE_SHEETS_URL")
        if gsheets_url:
            print(f"Using Google Sheets URL from .env: {gsheets_url}")
        else:
            print("❌ Google Sheets URL not provided and not found in .env")
            print("   Set GOOGLE_SHEETS_REVIEW_URL or GOOGLE_SHEETS_URL in .env")
            sys.exit(1)

    # Read CSV
    print(f"Reading {args.csv}...")
    df = pd.read_csv(args.csv)

    # Rename 'item' column to 'lemma' for compatibility with known words format
    if "item" in df.columns:
        df = df.rename(columns={"item": "lemma"})

    # Ensure required columns exist
    required_cols = ["lemma", "status", "item_type", "tags"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        sys.exit(1)

    # Upload to Google Sheets
    print(f"Uploading {len(df)} rows to Google Sheets...")
    try:
        save_known_words(df[required_cols], gsheets_url)
        print(f"✅ Successfully uploaded to {gsheets_url}")
        print("\nNow you can:")
        print("1. Open the Google Sheets in your browser")
        print("2. Mark words as 'known', 'learning', 'ignore', or 'maybe' in the 'status' column")
        print("3. Add tags if needed")
        print("4. The sheet is already sorted by global_zipf and book_freq (descending)")
    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
