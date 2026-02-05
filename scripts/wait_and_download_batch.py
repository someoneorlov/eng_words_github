#!/usr/bin/env python3
"""Poll batch status every poll_interval_sec until SUCCEEDED, then run download.
Use when 'wait' takes too long and you want to run in background or with custom interval.

Usage:
  uv run python scripts/wait_and_download_batch.py [--poll-interval 120]
  # Then run manual QC when cards_B_batch.json has enough cards.
"""

import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eng_words.word_family.batch import BATCH_DIR
from eng_words.word_family.batch_io import download_batch, wait_for_batch
from eng_words.word_family.batch_schemas import BatchConfig
from eng_words.word_family.batch import (
    TOKENS_PATH,
    SENTENCES_PATH,
    OUTPUT_CARDS_PATH,
)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Wait for batch then download")
    p.add_argument("--poll-interval", type=int, default=60, help="Seconds between status polls")
    p.add_argument("--timeout", type=int, default=None, help="Max seconds to wait (default: none)")
    args = p.parse_args()

    print("Waiting for batch (poll every %ss)... Ctrl+C to stop." % args.poll_interval)
    try:
        wait_for_batch(BATCH_DIR, poll_interval_sec=args.poll_interval, timeout_sec=args.timeout)
    except (FileNotFoundError, ValueError) as e:
        print("Error:", e)
        sys.exit(1)
    except TimeoutError as e:
        print("Timeout:", e)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        sys.exit(130)

    print("\nBatch complete. Running download...")
    config = BatchConfig(
        tokens_path=TOKENS_PATH,
        sentences_path=SENTENCES_PATH,
        batch_dir=BATCH_DIR,
        output_cards_path=OUTPUT_CARDS_PATH,
    )
    download_batch(config, from_file=False, retry_empty=True, retry_thinking=False)
    n = len(__import__("json").loads(OUTPUT_CARDS_PATH.read_text(encoding="utf-8")).get("cards", []))
    print("Done. Cards written: %d → %s" % (n, OUTPUT_CARDS_PATH))


if __name__ == "__main__":
    main()
