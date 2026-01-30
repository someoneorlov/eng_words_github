#!/usr/bin/env python3
"""
Тестирование интеграции Stage 2.5 в run_synset_card_generation.py на выборке.

Проверяет:
- Корректность работы фильтрации длины и спойлеров
- Правильность выбора примеров
- Статистику
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_synset_card_generation import run_card_generation

if __name__ == "__main__":
    # Тестируем на выборке из 50 карточек
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ИНТЕГРАЦИИ STAGE 2.5")
    print("=" * 70)
    print("\nЗапуск на выборке из 50 карточек...")
    print("Проверяем:")
    print("  - Корректность работы mark_examples_by_length")
    print("  - Корректность работы check_spoilers")
    print("  - Корректность работы select_examples_for_generation")
    print("  - Правильность передачи generate_count")
    print("  - Статистику фильтрации")
    print()
    
    run_card_generation(limit=50)
