#!/bin/bash
# Real-time monitoring script for card generation

LOG_FILE="data/synset_cards/full_generation.log"
CHECKPOINT_FILE="data/synset_cards/synset_smart_cards_partial.json"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     МОНИТОРИНГ ГЕНЕРАЦИИ КАРТОЧЕК (real-time)                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Нажмите Ctrl+C для выхода"
echo ""
echo "Обновление каждые 5 секунд..."
echo ""

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║     МОНИТОРИНГ ГЕНЕРАЦИИ КАРТОЧЕК                             ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Get last progress line
    LAST_PROGRESS=$(tail -100 "$LOG_FILE" 2>/dev/null | grep "Generating cards:" | tail -1)
    
    if [ -n "$LAST_PROGRESS" ]; then
        echo "📊 ПОСЛЕДНИЙ ПРОГРЕСС:"
        echo "─" | head -c 60 && echo ""
        echo "$LAST_PROGRESS" | sed 's/.*Generating cards:/  /'
        echo ""
    fi
    
    # Check checkpoint
    if [ -f "$CHECKPOINT_FILE" ]; then
        CARD_COUNT=$(python3 -c "import json; print(len(json.load(open('$CHECKPOINT_FILE'))))" 2>/dev/null || echo "?")
        FILE_SIZE=$(ls -lh "$CHECKPOINT_FILE" 2>/dev/null | awk '{print $5}')
        FILE_TIME=$(ls -lT "$CHECKPOINT_FILE" 2>/dev/null | awk '{print $6, $7, $8}' || stat -f "%Sm" "$CHECKPOINT_FILE" 2>/dev/null)
        
        echo "💾 ЧЕКПОИНТ:"
        echo "─" | head -c 60 && echo ""
        echo "  Карточек: $CARD_COUNT"
        echo "  Размер:   $FILE_SIZE"
        echo "  Время:    $FILE_TIME"
        echo ""
    fi
    
    # Get last 5 log lines
    echo "📝 ПОСЛЕДНИЕ СОБЫТИЯ:"
    echo "─" | head -c 60 && echo ""
    tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/  /' | tail -5
    echo ""
    
    # Calculate progress
    if [ -n "$CARD_COUNT" ] && [ "$CARD_COUNT" != "?" ]; then
        TOTAL=7872
        PROGRESS=$(echo "scale=1; $CARD_COUNT * 100 / $TOTAL" | bc 2>/dev/null || echo "0")
        REMAINING=$((TOTAL - CARD_COUNT))
        echo "═" | head -c 60 && echo ""
        echo "  Прогресс: $PROGRESS% ($CARD_COUNT / $TOTAL)"
        echo "  Осталось: $REMAINING карточек"
    fi
    
    echo ""
    echo "Обновление через 5 секунд... (Ctrl+C для выхода)"
    sleep 5
done

