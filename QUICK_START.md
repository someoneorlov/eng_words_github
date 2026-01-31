# Quick start — full card generation

## 1. Run

```bash
./scripts/run_full_generation.sh
```

## 2. Monitor (in another terminal)

```bash
./scripts/monitor_generation.sh
```

## 3. Check status

```bash
# How many cards generated
python3 -c "import json; f='data/synset_cards/synset_smart_cards_partial.json'; print(len(json.load(open(f))) if __import__('os').path.exists(f) else 0)"

# Logs
tail -f data/synset_cards/full_generation.log
```

---

Full docs: `docs/GENERATION_INSTRUCTIONS.md`
