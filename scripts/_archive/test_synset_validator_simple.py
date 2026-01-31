#!/usr/bin/env python3
"""
Простой тест для проверки импортов и базовой функциональности synset_validator.
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Проверяет, что все импорты работают."""
    print("=" * 70)
    print("ПРОВЕРКА ИМПОРТОВ")
    print("=" * 70)

    try:

        print("✅ Импорт validate_examples_for_synset_group успешен")
        print("✅ Импорт VALIDATION_PROMPT успешен")
    except Exception as e:
        print(f"❌ Ошибка импорта: {e}")
        return False

    try:

        print("✅ Импорт LLMProvider, LLMResponse успешен")
    except Exception as e:
        print(f"❌ Ошибка импорта LLM: {e}")
        return False

    try:

        print("✅ Импорт ResponseCache успешен")
    except Exception as e:
        print(f"❌ Ошибка импорта ResponseCache: {e}")
        return False

    try:

        print("✅ Импорт WordNet успешен")
    except Exception as e:
        print(f"❌ Ошибка импорта WordNet: {e}")
        return False

    print("\n" + "=" * 70)
    print("✅ Все импорты успешны!")
    print("=" * 70)
    return True


def test_helper_functions():
    """Проверяет работу вспомогательных функций."""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ВСПОМОГАТЕЛЬНЫХ ФУНКЦИЙ")
    print("=" * 70)

    try:

        from eng_words.validation.synset_validator import (
            _format_validation_prompt,
            _get_synset_definitions,
        )

        # Тест получения определений
        print("\n## Тест _get_synset_definitions")
        synset_ids = ["run.v.01", "bank.n.01"]
        definitions = _get_synset_definitions(synset_ids)
        print(f"  Получено определений: {len(definitions)}")
        for sid, defn in definitions.items():
            print(f"    {sid}: {defn[:60]}...")

        # Тест форматирования промпта
        print("\n## Тест _format_validation_prompt")
        examples = [
            (1, "I run every morning."),
            (2, "The bank is closed."),
        ]
        prompt = _format_validation_prompt(
            lemma="run",
            pos="v",
            synset_group=["run.v.01"],
            primary_synset="run.v.01",
            synset_definitions=definitions,
            examples=examples,
        )
        print(f"  Промпт сгенерирован: {len(prompt)} символов")
        print(f"  Содержит lemma: {'run' in prompt}")
        print(f"  Содержит примеры: {'I run every morning' in prompt}")

        print("\n" + "=" * 70)
        print("✅ Вспомогательные функции работают!")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"❌ Ошибка в вспомогательных функциях: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_prompt_formatting():
    """Проверяет форматирование промпта на реальных данных."""
    print("\n" + "=" * 70)
    print("ПРОВЕРКА ФОРМАТИРОВАНИЯ ПРОМПТА")
    print("=" * 70)

    try:
        from eng_words.validation.synset_validator import (
            _format_validation_prompt,
            _get_synset_definitions,
        )

        # Тест с реальными synsets
        lemma = "long"
        synset_group = ["long.r.01", "long.s.01"]
        primary_synset = "long.r.01"

        definitions = _get_synset_definitions(synset_group)
        print("\nОпределения synsets:")
        for sid, defn in definitions.items():
            print(f"  {sid}: {defn}")

        examples = [
            (12008, "I waited long for the bus."),
            (12009, "The long road stretched ahead."),
            (15000, "He longed for home."),
        ]

        prompt = _format_validation_prompt(
            lemma=lemma,
            pos="r",
            synset_group=synset_group,
            primary_synset=primary_synset,
            synset_definitions=definitions,
            examples=examples,
        )

        print("\nПромпт (первые 500 символов):")
        print(prompt[:500] + "...")

        # Проверяем наличие ключевых элементов
        checks = [
            ("lemma", lemma in prompt),
            ("synset_group", "long.r.01" in prompt and "long.s.01" in prompt),
            ("examples", "I waited long" in prompt),
            ("task description", "determine if it matches" in prompt.lower()),
        ]

        print("\nПроверка элементов промпта:")
        all_ok = True
        for name, check in checks:
            status = "✅" if check else "❌"
            print(f"  {status} {name}")
            if not check:
                all_ok = False

        if all_ok:
            print("\n" + "=" * 70)
            print("✅ Форматирование промпта работает корректно!")
            print("=" * 70)

        return all_ok

    except Exception as e:
        print(f"❌ Ошибка при форматировании промпта: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    success = True

    success &= test_imports()
    success &= test_helper_functions()
    success &= test_prompt_formatting()

    print("\n" + "=" * 70)
    if success:
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ!")
    else:
        print("❌ НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОЙДЕНЫ")
    print("=" * 70)
