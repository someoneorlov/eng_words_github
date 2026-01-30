#!/bin/bash
# Скрипт для быстрого просмотра Mermaid диаграммы

if [ $# -eq 0 ]; then
    echo "Использование: $0 <файл_с_диаграммой.md>"
    echo ""
    echo "Примеры:"
    echo "  $0 docs/PIPELINE_PRE_AGGREGATION.md"
    echo "  $0 docs/PIPELINE_ANALYSIS.md"
    exit 1
fi

FILE="$1"

if [ ! -f "$FILE" ]; then
    echo "❌ Файл не найден: $FILE"
    exit 1
fi

echo "📊 Извлечение диаграммы из $FILE..."
echo ""

# Извлечь диаграмму Mermaid
DIAGRAM=$(grep -A 1000 '```mermaid' "$FILE" | grep -B 1000 '```' | sed '1d;$d')

if [ -z "$DIAGRAM" ]; then
    echo "❌ Диаграмма Mermaid не найдена в файле"
    exit 1
fi

echo "✅ Диаграмма найдена!"
echo ""
echo "📋 Варианты просмотра:"
echo ""
echo "1. Онлайн редактор (рекомендуется):"
echo "   → Откройте https://mermaid.live"
echo "   → Вставьте код ниже"
echo ""
echo "2. VS Code Preview:"
echo "   → Откройте $FILE в VS Code"
echo "   → Нажмите Cmd+Shift+V для preview"
echo ""
echo "3. GitHub:"
echo "   → Откройте файл на GitHub"
echo "   → GitHub автоматически отобразит диаграмму"
echo ""
echo "══════════════════════════════════════════════════════════════"
echo "КОД ДИАГРАММЫ:"
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "$DIAGRAM"
echo ""
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "💡 Совет: Скопируйте код выше и вставьте в https://mermaid.live"
echo ""
