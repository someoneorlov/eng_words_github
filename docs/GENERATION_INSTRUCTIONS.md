# Инструкции по запуску полной генерации карточек

## Текущий статус

- **Всего карточек для генерации**: ~7,872
- **Уже сгенерировано**: проверьте через команды ниже
- **Скрипт генерации**: `scripts/run_synset_card_generation.py`
- **Поддержка checkpoint**: ✅ Да (сохранение каждые 100 карточек)
- **Поддержка resume**: ✅ Да (автоматическое продолжение с checkpoint)

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

### 2. Мониторинг прогресса (в отдельном терминале)

```bash
./scripts/monitor_generation.sh
```

Скрипт показывает:
- Текущий прогресс (X / 7,872 карточек)
- Процент завершения
- Размер checkpoint файла
- Последние события из лога

### 3. Проверка статуса

```bash
# Количество сгенерированных карточек
python3 -c "import json; f='data/synset_cards/synset_smart_cards_partial.json'; print(len(json.load(open(f))) if __import__('os').path.exists(f) else 0)"

# Просмотр последних строк лога
tail -20 data/synset_cards/full_generation.log

# Поиск прогресса в логе
tail -f data/synset_cards/full_generation.log | grep "Generating cards"
```

## Детальные команды

### Просмотр логов

```bash
# Следить за логом в реальном времени
tail -f data/synset_cards/full_generation.log

# Только прогресс
tail -f data/synset_cards/full_generation.log | grep "Generating cards"

# Последние 50 строк
tail -50 data/synset_cards/full_generation.log
```

### Проверка checkpoint

```bash
# Количество карточек в checkpoint
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

# Остановить (замените PID на реальный)
kill <PID>

# Или более агрессивно
killall -9 python
```

**Важно**: При остановке прогресс сохраняется в checkpoint. Можно просто перезапустить скрипт - он продолжит с последнего checkpoint.

### Перезапуск после остановки

Просто запустите скрипт снова:
```bash
./scripts/run_full_generation.sh
```

Скрипт автоматически обнаружит checkpoint и продолжит генерацию с последнего места.

## После завершения генерации

Когда генерация всех карточек завершится, выполните финальную обработку:

```bash
uv run python scripts/complete_card_generation.py
```

Этот скрипт:
1. Выполнит `redistribute_empty_cards` для карточек без примеров
2. Выполнит `fix_invalid_cards` для валидации примеров
3. Сохранит финальный результат в `data/synset_cards/synset_smart_cards_final.json`

## Ожидаемое время выполнения

- **Скорость**: ~2-3 секунды на карточку
- **Для 7,872 карточек**: ~4-6 часов
- **С учетом retry и API delays**: может быть больше

## Возможные проблемы

### 503 Server Error (UNAVAILABLE)

Скрипт автоматически обрабатывает эту ошибку:
- Сохраняет checkpoint
- Ждет 10 секунд
- Продолжает генерацию

### Прерванная генерация

Если генерация была прервана:
1. Проверьте checkpoint: `ls -lh data/synset_cards/synset_smart_cards_partial.json`
2. Перезапустите скрипт - он автоматически продолжит

### Проверка корректности checkpoint

```bash
# Проверить, что JSON валидный
python3 -c "import json; json.load(open('data/synset_cards/synset_smart_cards_partial.json')); print('OK')"
```

## Мониторинг ресурсов

```bash
# Использование памяти процессом
ps aux | grep run_synset_card_generation.py | awk '{print $6/1024 " MB"}'

# Размер лог файла
ls -lh data/synset_cards/full_generation.log

# Размер cache (может быть большой)
du -sh data/synset_cards/llm_cache/
```

## Результаты

После завершения генерации и обработки:

- **Финальный JSON**: `data/synset_cards/synset_smart_cards_final.json`
- **Anki CSV**: `data/synset_cards/synset_anki.csv` (если экспорт включен)
- **Логи**: `data/synset_cards/full_generation.log`
- **Checkpoint**: автоматически удаляется после успешного завершения

## Контакты и поддержка

При возникновении проблем проверьте:
1. Логи: `tail -100 data/synset_cards/full_generation.log`
2. Checkpoint: существует ли и валиден ли
3. Процесс: запущен ли (`ps aux | grep python`)

