#!/usr/bin/env python3
"""Test script for Google Sheets integration.

⚠️  WARNING: This script will OVERWRITE data in the specified worksheet!
Use a separate test worksheet (e.g., "TestSheet") to avoid losing real data.
"""

import os
import sys

import pandas as pd
from dotenv import load_dotenv

from eng_words.storage import load_known_words, save_known_words


def main() -> None:
    """Test Google Sheets integration."""
    # Load environment variables from .env file
    load_dotenv()

    # Try to get from .env, otherwise use default or require manual setting
    # Priority:
    # 1. GOOGLE_SHEETS_TEST_ID (specific for test script)
    # 2. Extract from GOOGLE_SHEETS_URL (main config)
    # 3. Default placeholder (will show error)

    SPREADSHEET_ID = os.getenv("GOOGLE_SHEETS_TEST_ID")
    WORKSHEET_NAME = os.getenv("GOOGLE_SHEETS_TEST_WORKSHEET", "TestSheet")

    # If not set, try to extract from GOOGLE_SHEETS_URL
    if not SPREADSHEET_ID:
        gsheets_url = os.getenv("GOOGLE_SHEETS_URL", "")
        if gsheets_url.startswith("gsheets://"):
            # Extract spreadsheet_id and worksheet_name from URL
            parts = gsheets_url.replace("gsheets://", "").split("/")
            if len(parts) >= 1 and parts[0]:
                SPREADSHEET_ID = parts[0]
            if len(parts) >= 2 and parts[1] and WORKSHEET_NAME == "TestSheet":
                # Only override if using default TestSheet
                WORKSHEET_NAME = parts[1]

    if not SPREADSHEET_ID or SPREADSHEET_ID == "YOUR_SPREADSHEET_ID_HERE":
        print("❌ Error: SPREADSHEET_ID not found!")
        print("\nOptions to set it:")
        print("1. Set GOOGLE_SHEETS_TEST_ID in .env file")
        print("2. Set GOOGLE_SHEETS_URL in .env file (will extract ID from URL)")
        print("3. Edit scripts/test_gsheets.py and set SPREADSHEET_ID directly")
        sys.exit(1)

    # URL формат
    gsheets_url = f"gsheets://{SPREADSHEET_ID}/{WORKSHEET_NAME}"

    print("=" * 60)
    print("Testing Google Sheets integration")
    print("=" * 60)
    print(f"URL: {gsheets_url}")
    print(f"⚠️  WARNING: This will OVERWRITE data in '{WORKSHEET_NAME}' worksheet!")
    print("=" * 60)

    # Запрос подтверждения
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ("yes", "y"):
        print("Cancelled.")
        return

    # Тест 1: Загрузка данных
    print("\nTest 1: Loading data from Google Sheets...")
    try:
        df = load_known_words(gsheets_url)
        print(f"✅ Load successful! Found {len(df)} rows")
        if not df.empty:
            print("\nFirst few rows:")
            print(df.head().to_string())
        else:
            print("⚠️  Sheet is empty (this is OK for first run)")
    except Exception as e:
        print(f"❌ Load failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that GOOGLE_APPLICATION_CREDENTIALS is set")
        print("2. Check that Service Account has access to the sheet")
        print("3. Check that Spreadsheet ID is correct")
        return

    # Тест 2: Сохранение тестовых данных
    print("\n" + "=" * 60)
    print("Test 2: Saving test data to Google Sheets...")
    print("⚠️  This will OVERWRITE all existing data in the worksheet!")
    try:
        test_df = pd.DataFrame(
            {
                "lemma": ["test_word", "test_phrase"],
                "status": ["learning", "known"],
                "item_type": ["word", "phrasal_verb"],
                "tags": ["test", "test"],
            }
        )
        save_known_words(test_df, gsheets_url)
        print("✅ Save successful!")
    except Exception as e:
        print(f"❌ Save failed: {e}")
        return

    # Тест 3: Повторная загрузка для проверки
    print("\n" + "=" * 60)
    print("Test 3: Reloading data to verify save...")
    try:
        df = load_known_words(gsheets_url)
        print(f"✅ Reload successful! Found {len(df)} rows")
        if "test_word" in df["lemma"].values:
            print("✅ Test data found in sheet!")
            print("\nAll test data:")
            print(df[df["lemma"].str.startswith("test")].to_string())
        else:
            print("⚠️  Test data not found (might have been filtered)")
    except Exception as e:
        print(f"❌ Reload failed: {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
