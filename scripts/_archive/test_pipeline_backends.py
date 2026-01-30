#!/usr/bin/env python3
"""Test script to verify both CSV and Google Sheets backends work with the pipeline.

This script runs a minimal pipeline test with both backends to ensure everything works.
"""

import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from eng_words.pipeline import process_book  # noqa: E402


def test_csv_backend(book_path: Path, book_name: str, output_dir: Path) -> bool:
    """Test pipeline with CSV backend."""
    print("=" * 60)
    print("Testing CSV Backend")
    print("=" * 60)

    # Create test CSV file
    csv_path = output_dir / "test_known_words.csv"
    csv_content = """lemma,status,item_type,tags
the,ignore,word,stopword
a,ignore,word,stopword
be,known,word,A1
have,known,word,A1
"""
    csv_path.write_text(csv_content)
    print(f"✅ Created test CSV: {csv_path}")

    try:
        print("\nRunning pipeline with CSV backend...")
        print(f"Book: {book_path}")
        print(f"Known words: {csv_path}")

        outputs = process_book(
            book_path=book_path,
            book_name=book_name,
            output_dir=output_dir,
            known_words_path=str(csv_path),
            min_book_freq=1,  # Lower threshold for testing
            min_zipf=0.0,
            max_zipf=10.0,
            top_n=10,  # Small number for quick test
            detect_phrasals=False,  # Skip phrasals for faster test
        )

        print("\n✅ CSV backend test PASSED!")
        print("\nOutputs:")
        for key, value in outputs.items():
            if value:
                print(f"  - {key}: {value}")

        return True
    except Exception as e:
        print(f"\n❌ CSV backend test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gsheets_backend(book_path: Path, book_name: str, output_dir: Path) -> bool:
    """Test pipeline with Google Sheets backend."""
    print("\n" + "=" * 60)
    print("Testing Google Sheets Backend")
    print("=" * 60)

    # Load .env to get Google Sheets URL
    load_dotenv()
    gsheets_url = os.getenv("GOOGLE_SHEETS_URL")

    if not gsheets_url:
        print("⚠️  GOOGLE_SHEETS_URL not set in .env file")
        print("   Skipping Google Sheets backend test")
        return None

    if not gsheets_url.startswith("gsheets://"):
        print(f"⚠️  Invalid GOOGLE_SHEETS_URL format: {gsheets_url}")
        print("   Expected format: gsheets://SPREADSHEET_ID/WORKSHEET_NAME")
        return None

    print(f"Using Google Sheets URL: {gsheets_url}")

    try:
        print("\nRunning pipeline with Google Sheets backend...")
        print(f"Book: {book_path}")
        print(f"Known words: {gsheets_url}")

        outputs = process_book(
            book_path=book_path,
            book_name=book_name,
            output_dir=output_dir,
            known_words_path=gsheets_url,
            min_book_freq=1,  # Lower threshold for testing
            min_zipf=0.0,
            max_zipf=10.0,
            top_n=10,  # Small number for quick test
            detect_phrasals=False,  # Skip phrasals for faster test
        )

        print("\n✅ Google Sheets backend test PASSED!")
        print("\nOutputs:")
        for key, value in outputs.items():
            if value:
                print(f"  - {key}: {value}")

        return True
    except Exception as e:
        print(f"\n❌ Google Sheets backend test FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> None:
    """Run tests for both backends."""
    print("=" * 60)
    print("Pipeline Backend Testing")
    print("=" * 60)

    # Configuration
    project_root = Path(__file__).parent.parent

    # Try to find a test EPUB file (prefer smaller excerpt for faster testing)
    excerpt_path = project_root / "data/raw/american_tragedy_excerpt.epub"
    full_path = project_root / "data/raw/theodore-dreiser_an-american-tragedy.epub"

    if excerpt_path.exists():
        book_path = excerpt_path
        print(f"Using excerpt file for faster testing: {book_path.name}")
    elif full_path.exists():
        book_path = full_path
        print(f"Using full book file: {book_path.name}")
    else:
        book_path = None

    book_name = "test_backend"
    output_dir = project_root / "data/processed/test_backends"

    # Check if book exists
    if book_path is None or not book_path.exists():
        print("❌ No test EPUB file found!")
        print("\nPlease ensure one of these files exists:")
        print("  - data/raw/american_tragedy_excerpt.epub (preferred for testing)")
        print("  - data/raw/theodore-dreiser_an-american-tragedy.epub")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    results = {}

    # Test CSV backend
    csv_result = test_csv_backend(book_path, book_name, output_dir)
    results["CSV"] = csv_result

    # Test Google Sheets backend
    gsheets_result = test_gsheets_backend(book_path, book_name, output_dir)
    results["Google Sheets"] = gsheets_result

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for backend, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"{backend:20} {status}")

    # Cleanup option
    print("\n" + "=" * 60)
    response = input("Delete test output files? (yes/no): ").strip().lower()
    if response in ("yes", "y"):
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"✅ Deleted {output_dir}")
        csv_path = project_root / "data/processed/test_known_words.csv"
        if csv_path.exists():
            csv_path.unlink()
            print(f"✅ Deleted {csv_path}")
    else:
        print(f"Test files kept in: {output_dir}")

    # Exit code
    if all(r for r in results.values() if r is not None):
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed or were skipped")
        sys.exit(1)


if __name__ == "__main__":
    main()
