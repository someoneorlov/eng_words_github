#!/usr/bin/env python3
"""
Анализ структуры WSD Golden Dataset для Этапа 0.1.5 плана QUALITY_FILTERING_PLAN.

Изучает:
- Структуру gold_dev.jsonl и gold_test_locked.jsonl
- Как использовать для проверки precision/recall валидации примеров
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.wsd_gold.eval import load_gold_examples


def analyze_gold_dataset():
    """Анализирует структуру Golden Dataset."""
    print("=" * 70)
    print("АНАЛИЗ СТРУКТУРЫ WSD GOLDEN DATASET")
    print("=" * 70)

    # Проверяем наличие файлов
    dev_path = Path("data/wsd_gold/gold_dev.jsonl")
    test_path = Path("data/wsd_gold/gold_test_locked.jsonl")

    if not dev_path.exists():
        print(f"❌ Файл не найден: {dev_path}")
        return

    # Загружаем dev set
    print(f"\n📚 Загрузка dev set: {dev_path}")
    dev_examples = load_gold_examples(dev_path)
    print(f"  Загружено примеров: {len(dev_examples):,}")

    # Анализ структуры первого примера
    if dev_examples:
        print("\n📝 Структура примера:")
        example = dev_examples[0]
        for key, value in example.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list) and len(subvalue) > 0:
                        print(
                            f"    {subkey}: list[{len(subvalue)}] (первый элемент: {subvalue[0] if isinstance(subvalue[0], str) else type(subvalue[0]).__name__})"
                        )
                    else:
                        print(f"    {subkey}: {subvalue}")
            elif isinstance(value, list):
                print(f"  {key}: list[{len(value)}]")
                if len(value) > 0:
                    print(
                        f"    Первый элемент: {value[0] if isinstance(value[0], str) else value[0]}"
                    )
            else:
                print(f"  {key}: {value}")

    # Статистика по полям
    print("\n📊 Статистика по полям:")

    # POS distribution
    pos_counts = {}
    synset_counts = {}
    has_gold = 0

    for ex in dev_examples:
        if "target" in ex and "pos" in ex["target"]:
            pos = ex["target"]["pos"]
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        if "gold_synset_id" in ex:
            has_gold += 1
            synset = ex["gold_synset_id"]
            synset_counts[synset] = synset_counts.get(synset, 0) + 1

    print(f"  Примеров с gold_synset_id: {has_gold} ({has_gold/len(dev_examples)*100:.1f}%)")
    print("  Распределение по POS:")
    for pos, count in sorted(pos_counts.items(), key=lambda x: -x[1]):
        print(f"    {pos}: {count} ({count/len(dev_examples)*100:.1f}%)")

    # Примеры для валидации
    print("\n🔍 Примеры для использования в валидации:")
    print("  Каждый пример содержит:")
    print("    - context_window: предложение с целевым словом")
    print("    - target.lemma: лемма слова")
    print("    - target.pos: часть речи")
    print("    - gold_synset_id: правильный synset (эталон)")
    print("    - candidates: список возможных synsets")

    # Показываем несколько примеров
    print("\n📋 Примеры из dev set:")
    for i, ex in enumerate(dev_examples[:3], 1):
        print(f"\n  Пример {i}:")
        print(f"    Lemma: {ex.get('target', {}).get('lemma', 'N/A')}")
        print(f"    POS: {ex.get('target', {}).get('pos', 'N/A')}")
        print(f"    Gold synset: {ex.get('gold_synset_id', 'N/A')}")
        print(f"    Context: {ex.get('context_window', 'N/A')[:80]}...")
        print(f"    Candidates: {len(ex.get('candidates', []))} synsets")

    # Проверка test_locked
    if test_path.exists():
        print("\n🔒 Test locked set:")
        test_examples = load_gold_examples(test_path)
        print(f"  Загружено примеров: {len(test_examples):,}")
        print("  ⚠️  ВАЖНО: Использовать только для финального сравнения!")
        print("  Не смотреть во время разработки!")
    else:
        print(f"\n⚠️  Test locked set не найден: {test_path}")

    print("\n" + "=" * 70)
    print("✅ Анализ завершен")
    print("=" * 70)

    print("\n💡 Стратегия использования:")
    print(f"  1. Dev set ({len(dev_examples):,} примеров) - использовать для разработки")
    print("  2. Test locked - только для финального сравнения")
    print("  3. Для валидации: проверять, правильно ли определяется соответствие")
    print("     примера synset_group (precision/recall)")


if __name__ == "__main__":
    analyze_gold_dataset()
