#!/bin/bash
# Скрипт для проверки прогресса теста

LOG_FILE="logs/stage25_integration_test_100.log"
FINAL_FILE="data/synset_cards/synset_smart_cards_final.json"

echo "Проверка прогресса теста на 100 карточках..."
echo "=========================================="

if [ -f "$FINAL_FILE" ]; then
    echo "✅ Процесс завершен!"
    echo ""
    echo "Статистика:"
    tail -80 "$LOG_FILE" | grep -A 30 "Validation Statistics"
    echo ""
    echo "Финальные результаты:"
    tail -30 "$LOG_FILE" | grep -A 20 "SUMMARY"
else
    echo "⏳ Процесс продолжается..."
    echo ""
    echo "Текущий прогресс:"
    tail -3 "$LOG_FILE" | grep "Generating cards" || tail -5 "$LOG_FILE"
    echo ""
    if [ -f "data/synset_cards/synset_smart_cards_partial.json" ]; then
        echo "Промежуточные результаты:"
        python3 -c "import json; data = json.load(open('data/synset_cards/synset_smart_cards_partial.json')); print(f'  Сгенерировано карточек: {len(data)}')" 2>/dev/null || echo "  Не удалось прочитать промежуточный файл"
    fi
fi
