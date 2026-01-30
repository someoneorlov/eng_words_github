#!/bin/bash
# Простой скрипт для запуска генерации всех оставшихся карточек

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     ЗАПУСК ГЕНЕРАЦИИ ВСЕХ КАРТОЧЕК                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Проверяем статус
echo "📊 Проверка текущего статуса..."
python3 << 'PYTHON_SCRIPT'
import json
import pandas as pd
from pathlib import Path

total = len(pd.read_parquet('data/synset_aggregation_full/aggregated_cards.parquet'))
partial = Path('data/synset_cards/synset_smart_cards_partial.json')
final = Path('data/synset_cards/synset_smart_cards_final.json')

generated = 0
if partial.exists():
    generated = len(json.load(open(partial)))
elif final.exists():
    generated = len(json.load(open(final)))

remaining = total - generated
progress = (generated / total * 100) if total > 0 else 0

print(f'  Всего карточек: {total:,}')
print(f'  Сгенерировано:  {generated:,} ({progress:.1f}%)')
print(f'  Осталось:       {remaining:,}')
PYTHON_SCRIPT

echo ""
echo "🚀 Запуск генерации..."
echo "   Логи: data/synset_cards/full_generation.log"
echo ""

# Запускаем в фоне
nohup uv run python scripts/run_synset_card_generation.py \
    > data/synset_cards/full_generation.log 2>&1 &
PID=$!

echo "✅ Генерация запущена (PID: $PID)"
echo ""
echo "📊 Для мониторинга в другом терминале:"
echo "   ./scripts/monitor_generation.sh"
echo ""
echo "📝 Для просмотра логов:"
echo "   tail -f data/synset_cards/full_generation.log"
echo ""
echo "🛑 Для остановки:"
echo "   kill $PID"
echo ""
echo "══════════════════════════════════════════════════════════════"
