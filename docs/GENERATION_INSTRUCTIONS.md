# Инструкции по полной генерации карточек

## Текущий статус

- **Всего карточек к генерации**: ~7 872
- **Уже сгенерировано**: проверьте командами ниже
- **Скрипт генерации**: `scripts/run_synset_card_generation.py`
- **Чекпоинты**: ✅ да (сохранение каждые 100 карточек)
- **Продолжение с места остановки**: ✅ да (автоматически с последнего чекпоинта)

## Быстрый старт

### 1. Запуск генерации

```bash
./scripts/run_full_generation.sh
```

Или вручную:
```bash
nohup uv run python scripts/run_synset_card_generation.py \
  > data/synset_cards/full_generation.log 2>&1 &
```

### 2. Мониторинг (в другом терминале)

```bash
./scripts/monitor_generation.sh
```

Скрипт показывает:
- Текущий прогресс (X / 7 872 карточек)
- Процент выполнения
- Размер файла чекпоинта
- Последние строки лога

### 3. Проверка статуса

```bash
# Количество сгенерированных карточек
python3 -c "import json; f='data/synset_cards/synset_smart_cards_partial.json'; print(len(json.load(open(f))) if __import__('os').path.exists(f) else 0)"

# Последние строки лога
tail -20 data/synset_cards/full_generation.log

# Поиск прогресса в логе
tail -f data/synset_cards/full_generation.log | grep "Generating cards"
```

## Подробные команды

### Просмотр логов

```bash
# Лог в реальном времени
tail -f data/synset_cards/full_generation.log

# Только прогресс
tail -f data/synset_cards/full_generation.log | grep "Generating cards"

# Последние 50 строк
tail -50 data/synset_cards/full_generation.log
```

### Проверка чекпоинта

```bash
# Количество карточек в чекпоинте
python3 -c "import json; print(len(json.load(open('data/synset_cards/synset_smart_cards_partial.json'))))"

# Размер файла
ls -lh data/synset_cards/synset_smart_cards_partial.json

# Время последнего обновления
stat -f "%Sm" data/synset_cards/synset_smart_cards_partial.json
```

### Остановка процесса

```bash
# Найти процесс
ps aux | grep run_synset_card_generation.py

# Остановить (подставьте реальный PID)
kill <PID>

# Или жёстче
killall -9 python
```

**Важно:** при остановке прогресс сохраняется в чекпоинте. Можно просто перезапустить скрипт — он продолжит с последнего чекпоинта.

### Перезапуск после остановки

Просто запустите скрипт снова:
```bash
./scripts/run_full_generation.sh
```

Скрипт обнаружит чекпоинт и продолжит с места остановки.

## После завершения генерации

Когда все карточки сгенерированы, запустите финальную обработку:

```bash
uv run python scripts/complete_card_generation.py
```

Скрипт:
1. Запускает `redistribute_empty_cards` для карточек без примеров
2. Запускает `fix_invalid_cards` для проверки примеров
3. Сохраняет итог в `data/synset_cards/synset_smart_cards_final.json`

## Ожидаемое время

- **Скорость**: ~2–3 секунды на карточку
- **Для 7 872 карточек**: ~4–6 часов
- **С учётом повторов и задержек API**: может быть дольше

## Устранение неполадок

### 503 Server Error (UNAVAILABLE)

Скрипт обрабатывает это автоматически:
- Сохраняет чекпоинт
- Ждёт 10 секунд
- Продолжает генерацию

### Прерванная генерация

Если генерация прервалась:
1. Проверьте чекпоинт: `ls -lh data/synset_cards/synset_smart_cards_partial.json`
2. Перезапустите скрипт — он продолжит автоматически

### Проверка чекпоинта

```bash
# Проверить, что JSON валиден
python3 -c "import json; json.load(open('data/synset_cards/synset_smart_cards_partial.json')); print('OK')"
```

## Мониторинг ресурсов

```bash
# Память процесса
ps aux | grep run_synset_card_generation.py | awk '{print $6/1024 " MB"}'

# Размер лог-файла
ls -lh data/synset_cards/full_generation.log

# Размер кэша (может быть большим)
du -sh data/synset_cards/llm_cache/
```

## Результаты

После завершения генерации и обработки:

- **Итоговый JSON**: `data/synset_cards/synset_smart_cards_final.json`
- **Anki CSV**: `data/synset_cards/synset_anki.csv` (если экспорт включён)
- **Логи**: `data/synset_cards/full_generation.log`
- **Чекпоинт**: удаляется автоматически после успешного завершения

## Поддержка

Если что-то пошло не так, проверьте:
1. Логи: `tail -100 data/synset_cards/full_generation.log`
2. Чекпоинт: существует ли файл и валиден ли он
3. Процесс: запущен ли (`ps aux | grep python`)
