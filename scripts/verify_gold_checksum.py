#!/usr/bin/env python3
"""Verify gold_test_locked.jsonl checksum hasn't changed.

This script is meant to be run in CI to ensure the locked test set
hasn't been modified accidentally.

Usage:
    uv run python scripts/verify_gold_checksum.py
"""

import hashlib
import sys
from pathlib import Path

from eng_words.constants import get_gold_checksum_path, get_gold_test_locked_path

TEST_LOCKED_PATH = get_gold_test_locked_path()
CHECKSUM_PATH = get_gold_checksum_path()


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main() -> int:
    """Verify checksum."""
    if not TEST_LOCKED_PATH.exists():
        print(f"❌ File not found: {TEST_LOCKED_PATH}")
        return 1

    if not CHECKSUM_PATH.exists():
        print(f"❌ Checksum file not found: {CHECKSUM_PATH}")
        return 1

    # Read expected checksum
    with open(CHECKSUM_PATH) as f:
        expected = f.read().strip().split()[0]

    # Compute actual checksum
    actual = compute_sha256(TEST_LOCKED_PATH)

    if actual == expected:
        print(f"✅ Checksum verified: {actual[:16]}...")
        return 0
    else:
        print("❌ CHECKSUM MISMATCH!")
        print(f"   Expected: {expected}")
        print(f"   Actual:   {actual}")
        print("\n⚠️  gold_test_locked.jsonl has been modified!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
