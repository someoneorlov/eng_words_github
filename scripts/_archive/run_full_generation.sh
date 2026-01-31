#!/bin/bash
# Скрипт для полной генерации всех карточек с мониторингом

set -e

LOG_FILE="data/synset_cards/full_generation.log"
SCRIPT="scripts/run_synset_card_generation.py"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     ЗАПУСК ПОЛНОЙ ГЕНЕРАЦИИ КАРТОЧЕК                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Логи будут сохранены в: $LOG_FILE"
echo "📊 Мониторинг: ./scripts/monitor_generation.sh"
echo ""
echo "🚀 Запуск генерации..."
echo ""

# Запускаем в фоне с перенаправлением вывода
nohup uv run python "$SCRIPT" > "$LOG_FILE" 2>&1 &
PID=$!

echo "✅ Генерация запущена (PID: $PID)"
echo ""
echo "📊 ДЛЯ МОНИТОРИНГА:"
echo "   ./scripts/monitor_generation.sh"
echo ""
echo "📝 ДЛЯ ПРОСМОТРА ЛОГОВ:"
echo "   tail -f $LOG_FILE"
echo ""
echo "🛑 ДЛЯ ОСТАНОВКИ:"
echo "   kill $PID"
echo ""
echo "══════════════════════════════════════════════════════════════"

