#!/usr/bin/env python3
"""Проверка статуса регенерации карточек."""

import json
import time
from pathlib import Path

OUTPUT_FILE = Path("data/stage2_5_test/test_results_with_full_cards.json")
EXPECTED_COUNT = 115


def check_status():
    """Проверяет статус регенерации."""
    print("=" * 70)
    print("СТАТУС РЕГЕНЕРАЦИИ КАРТОЧЕК")
    print("=" * 70)

    if OUTPUT_FILE.exists():
        print(f"\n✅ Файл существует: {OUTPUT_FILE}")
        print(f"   Размер: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            print("\n📊 Статистика:")
            print(f"   Карточек в файле: {len(data)} / {EXPECTED_COUNT}")
            print(f"   Прогресс: {len(data)/EXPECTED_COUNT*100:.1f}%")

            if len(data) > 0:
                first = data[0]
                has_card_full = "card_full" in first and first.get("card_full")
                print("\n✅ Проверка данных:")
                print(f"   Есть card_full: {has_card_full}")

                if has_card_full:
                    cf = first["card_full"]
                    print(f"   - selected_examples: {len(cf.get('selected_examples', []))}")
                    print(f"   - generated_examples: {len(cf.get('generated_examples', []))}")
                    print(f"   - simple_definition: {bool(cf.get('simple_definition'))}")
                    print(f"   - translation_ru: {bool(cf.get('translation_ru'))}")

                    # Показываем примеры
                    if cf.get("selected_examples"):
                        print("\n📚 Пример selected_examples:")
                        print(f"   \"{cf['selected_examples'][0][:80]}...\"")
                    if cf.get("generated_examples"):
                        print("\n✨ Пример generated_examples:")
                        print(f"   \"{cf['generated_examples'][0][:80]}...\"")

                    # Показываем последнюю карточку
                    if len(data) == EXPECTED_COUNT:
                        last = data[-1]
                        print("\n✅ Регенерация завершена!")
                        print(
                            f"   Последняя карточка: {last.get('lemma', 'N/A')} ({last.get('pos', 'N/A')})"
                        )
                    else:
                        last = data[-1]
                        print("\n⏳ Регенерация в процессе...")
                        print(
                            f"   Последняя обработанная: {last.get('lemma', 'N/A')} ({last.get('pos', 'N/A')})"
                        )

        except json.JSONDecodeError as e:
            print(f"\n⚠️  Файл поврежден (невалидный JSON): {e}")
        except Exception as e:
            print(f"\n❌ Ошибка при чтении файла: {e}")
    else:
        print("\n⏳ Файл еще не создан")
        print(f"   Ожидаемый путь: {OUTPUT_FILE.absolute()}")
        print("   Процесс может еще не начать запись в файл")

    # Проверяем логи
    log_files = sorted(Path("logs").glob("regenerate_cards_*.log"), reverse=True)
    if log_files:
        print(f"\n📝 Последний лог: {log_files[0].name}")
        print(f"   Размер: {log_files[0].stat().st_size / 1024:.1f} KB")
        print(f"   Модифицирован: {time.ctime(log_files[0].stat().st_mtime)}")

        # Показываем последние строки лога
        try:
            with open(log_files[0], "r", encoding="utf-8") as f:
                lines = f.readlines()
                print("\n   Последние строки лога:")
                for line in lines[-5:]:
                    print(f"   {line.rstrip()}")
        except:
            pass

    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_status()
