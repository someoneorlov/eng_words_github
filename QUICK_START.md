# Быстрый старт — полная генерация карточек

## 1. Запуск

```bash
./scripts/run_full_generation.sh
```

## 2. Мониторинг (в другом терминале)

```bash
./scripts/monitor_generation.sh
```

## 3. Проверка статуса

```bash
# Сколько карточек сгенерировано
python3 -c "import json; f='data/synset_cards/synset_smart_cards_partial.json'; print(len(json.load(open(f))) if __import__('os').path.exists(f) else 0)"

# Логи
tail -f data/synset_cards/full_generation.log
```

---

Подробная документация: `docs/GENERATION_INSTRUCTIONS.md`
