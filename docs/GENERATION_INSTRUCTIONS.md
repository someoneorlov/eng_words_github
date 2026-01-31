# Full card generation instructions

## Current status

- **Total cards to generate**: ~7,872
- **Already generated**: check via commands below
- **Generation script**: `scripts/run_synset_card_generation.py`
- **Checkpoint support**: ✅ Yes (save every 100 cards)
- **Resume support**: ✅ Yes (automatic continuation from checkpoint)

## Quick start

### 1. Start generation

```bash
./scripts/run_full_generation.sh
```

Or manually:
```bash
nohup uv run python scripts/run_synset_card_generation.py \
  > data/synset_cards/full_generation.log 2>&1 &
```

### 2. Monitor progress (in another terminal)

```bash
./scripts/monitor_generation.sh
```

The script shows:
- Current progress (X / 7,872 cards)
- Completion percentage
- Checkpoint file size
- Latest log entries

### 3. Check status

```bash
# Number of generated cards
python3 -c "import json; f='data/synset_cards/synset_smart_cards_partial.json'; print(len(json.load(open(f))) if __import__('os').path.exists(f) else 0)"

# View last log lines
tail -20 data/synset_cards/full_generation.log

# Search for progress in log
tail -f data/synset_cards/full_generation.log | grep "Generating cards"
```

## Detailed commands

### View logs

```bash
# Follow log in real time
tail -f data/synset_cards/full_generation.log

# Progress only
tail -f data/synset_cards/full_generation.log | grep "Generating cards"

# Last 50 lines
tail -50 data/synset_cards/full_generation.log
```

### Check checkpoint

```bash
# Number of cards in checkpoint
python3 -c "import json; print(len(json.load(open('data/synset_cards/synset_smart_cards_partial.json'))))"

# File size
ls -lh data/synset_cards/synset_smart_cards_partial.json

# Last update time
stat -f "%Sm" data/synset_cards/synset_smart_cards_partial.json
```

### Stop process

```bash
# Find process
ps aux | grep run_synset_card_generation.py

# Stop (replace PID with actual)
kill <PID>

# Or more aggressive
killall -9 python
```

**Important**: On stop, progress is saved in the checkpoint. You can simply restart the script — it will continue from the last checkpoint.

### Restart after stop

Just run the script again:
```bash
./scripts/run_full_generation.sh
```

The script will detect the checkpoint and continue from where it left off.

## After generation completes

When all cards are generated, run final processing:

```bash
uv run python scripts/complete_card_generation.py
```

This script:
1. Runs `redistribute_empty_cards` for cards without examples
2. Runs `fix_invalid_cards` for example validation
3. Saves the final result to `data/synset_cards/synset_smart_cards_final.json`

## Expected runtime

- **Speed**: ~2–3 seconds per card
- **For 7,872 cards**: ~4–6 hours
- **With retries and API delays**: may be longer

## Troubleshooting

### 503 Server Error (UNAVAILABLE)

The script handles this automatically:
- Saves checkpoint
- Waits 10 seconds
- Resumes generation

### Interrupted generation

If generation was interrupted:
1. Check checkpoint: `ls -lh data/synset_cards/synset_smart_cards_partial.json`
2. Restart the script — it will continue automatically

### Verify checkpoint

```bash
# Check that JSON is valid
python3 -c "import json; json.load(open('data/synset_cards/synset_smart_cards_partial.json')); print('OK')"
```

## Resource monitoring

```bash
# Memory usage of process
ps aux | grep run_synset_card_generation.py | awk '{print $6/1024 " MB"}'

# Log file size
ls -lh data/synset_cards/full_generation.log

# Cache size (can be large)
du -sh data/synset_cards/llm_cache/
```

## Outputs

After generation and processing complete:

- **Final JSON**: `data/synset_cards/synset_smart_cards_final.json`
- **Anki CSV**: `data/synset_cards/synset_anki.csv` (if export enabled)
- **Logs**: `data/synset_cards/full_generation.log`
- **Checkpoint**: removed automatically after successful completion

## Support

If something goes wrong, check:
1. Logs: `tail -100 data/synset_cards/full_generation.log`
2. Checkpoint: does it exist and is it valid
3. Process: is it running (`ps aux | grep python`)
