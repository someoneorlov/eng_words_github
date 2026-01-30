#!/usr/bin/env python3
"""
Анализ структуры aggregated_cards.parquet для Этапа 0.1 плана QUALITY_FILTERING_PLAN.

Изучает:
- Структуру колонок
- Типы данных
- Связь между sentence_ids и synset_group
- Примеры данных
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def analyze_aggregated_cards():
    """Анализирует структуру aggregated_cards.parquet."""
    path = Path("data/synset_aggregation_full/aggregated_cards.parquet")
    
    if not path.exists():
        print(f"❌ Файл не найден: {path}")
        return
    
    print("=" * 70)
    print("АНАЛИЗ СТРУКТУРЫ AGGREGATED CARDS")
    print("=" * 70)
    
    # Загрузка данных
    df = pd.read_parquet(path)
    
    print(f"\n📊 Общая информация:")
    print(f"  Shape: {df.shape[0]:,} строк × {df.shape[1]} колонок")
    
    print(f"\n📋 Колонки:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\n🔍 Типы данных:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    
    # Анализ первой строки
    print(f"\n📝 Пример первой строки:")
    first_row = df.iloc[0]
    for col in df.columns:
        value = first_row[col]
        if isinstance(value, (list, dict)):
            print(f"  {col}: {type(value).__name__} (len={len(value) if hasattr(value, '__len__') else 'N/A'})")
            if isinstance(value, list) and len(value) > 0:
                print(f"    Первый элемент: {value[0]}")
            elif isinstance(value, dict):
                print(f"    Ключи: {list(value.keys())[:5]}")
        else:
            print(f"  {col}: {value}")
    
    # Анализ synset_group
    if 'synset_group' in df.columns:
        print(f"\n🔗 Анализ synset_group:")
        sample_groups = df['synset_group'].head(5)
        for idx, group in enumerate(sample_groups, 1):
            print(f"  Пример {idx}: {group} (type: {type(group).__name__})")
            if isinstance(group, list):
                print(f"    Количество synsets в группе: {len(group)}")
                print(f"    Synsets: {group[:3]}...")
    
    # Анализ sentence_ids
    if 'sentence_ids' in df.columns:
        print(f"\n📚 Анализ sentence_ids:")
        sample_sentences = df['sentence_ids'].head(5)
        for idx, sids in enumerate(sample_sentences, 1):
            print(f"  Пример {idx}: {type(sids).__name__}")
            if isinstance(sids, list):
                print(f"    Количество sentence_ids: {len(sids)}")
                print(f"    Первые 5: {sids[:5]}")
            elif pd.notna(sids):
                print(f"    Значение: {sids}")
    
    # Статистика по synset_group
    if 'synset_group' in df.columns:
        print(f"\n📊 Статистика по synset_group:")
        # Подсчет количества synsets в каждой группе
        group_sizes = []
        for group in df['synset_group']:
            if isinstance(group, list):
                group_sizes.append(len(group))
            elif isinstance(group, str):
                # Попробуем распарсить JSON
                try:
                    parsed = json.loads(group)
                    if isinstance(parsed, list):
                        group_sizes.append(len(parsed))
                    else:
                        group_sizes.append(1)
                except:
                    group_sizes.append(1)
            else:
                group_sizes.append(1)
        
        if group_sizes:
            print(f"  Средний размер группы: {sum(group_sizes) / len(group_sizes):.2f}")
            print(f"  Минимальный размер: {min(group_sizes)}")
            print(f"  Максимальный размер: {max(group_sizes)}")
            print(f"  Группы с 1 synset: {sum(1 for s in group_sizes if s == 1)} ({sum(1 for s in group_sizes if s == 1)/len(group_sizes)*100:.1f}%)")
            print(f"  Группы с 2+ synsets: {sum(1 for s in group_sizes if s > 1)} ({sum(1 for s in group_sizes if s > 1)/len(group_sizes)*100:.1f}%)")
    
    # Статистика по sentence_ids
    if 'sentence_ids' in df.columns:
        print(f"\n📊 Статистика по sentence_ids:")
        sentence_counts = []
        for sids in df['sentence_ids']:
            if isinstance(sids, list):
                sentence_counts.append(len(sids))
            elif pd.notna(sids):
                sentence_counts.append(1)
            else:
                sentence_counts.append(0)
        
        if sentence_counts:
            print(f"  Среднее количество примеров: {sum(sentence_counts) / len(sentence_counts):.2f}")
            print(f"  Минимум: {min(sentence_counts)}")
            print(f"  Максимум: {max(sentence_counts)}")
            print(f"  Карточки без примеров: {sum(1 for c in sentence_counts if c == 0)} ({sum(1 for c in sentence_counts if c == 0)/len(sentence_counts)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("✅ Анализ завершен")
    print("=" * 70)


if __name__ == "__main__":
    analyze_aggregated_cards()
