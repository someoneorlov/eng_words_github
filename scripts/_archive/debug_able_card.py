#!/usr/bin/env python3
"""
Детальная проверка проблемной карточки "able (a)".
"""

import json
import sys
from pathlib import Path

import nltk
import pandas as pd
from dotenv import load_dotenv
from nltk.corpus import wordnet as wn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from eng_words.llm.base import get_provider
from eng_words.llm.response_cache import ResponseCache
from eng_words.text_processing import create_sentences_dataframe, reconstruct_sentences_from_tokens
from eng_words.validation.synset_validator import validate_examples_for_synset_group

load_dotenv()

# Загрузка WordNet
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


def check_wordnet_definitions():
    """Проверяет определения WordNet для able.a.01, able.s.02, able.s.03."""
    print("=" * 80)
    print("WORDNET ОПРЕДЕЛЕНИЯ ДЛЯ ABLE")
    print("=" * 80)

    synsets = ["able.a.01", "able.s.02", "able.s.03"]

    for synset_id in synsets:
        try:
            synset = wn.synset(synset_id)
            print(f"\n{synset_id}:")
            print(f"  Definition: {synset.definition()}")
            print(f"  Examples: {synset.examples()}")
            print(f"  Lemmas: {[lemma.name() for lemma in synset.lemmas()]}")
        except Exception as e:
            print(f"\n{synset_id}: ERROR - {e}")


def check_able_examples():
    """Проверяет примеры для able (a) и валидацию."""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ПРИМЕРОВ ДЛЯ ABLE (A)")
    print("=" * 80)

    # Загрузка данных
    results_path = Path("data/quality_analysis/validation_100_results.parquet")
    tokens_path = Path("data/processed/american_tragedy_tokens.parquet")
    aggregated_path = Path("data/synset_aggregation_full/aggregated_cards.parquet")

    results_df = pd.read_parquet(results_path)
    tokens_df = pd.read_parquet(tokens_path)
    cards_df = pd.read_parquet(aggregated_path)

    # Находим карточку able (a)
    able_row = results_df[
        (results_df["lemma"] == "able")
        & (results_df["pos"] == "a")
        & (results_df["primary_synset"] == "able.a.01")
    ]

    if able_row.empty:
        print("❌ Карточка able (a) не найдена в результатах")
        return

    able_row = able_row.iloc[0]
    print("\nКарточка: able (a)")
    print(f"Primary synset: {able_row['primary_synset']}")
    print(f"Synset group: {able_row['synset_group']}")
    print(f"Has valid: {able_row['has_valid']}")
    print(f"Valid count: {able_row['valid_count']}")
    print(f"Invalid count: {able_row['invalid_count']}")
    print(f"Total examples: {able_row['total_examples']}")

    # Находим в aggregated_cards
    card_row = cards_df[(cards_df["lemma"] == "able") & (cards_df["primary_synset"] == "able.a.01")]

    if card_row.empty:
        print("❌ Карточка не найдена в aggregated_cards")
        return

    card_row = card_row.iloc[0]
    sentence_ids = card_row.get("sentence_ids", [])

    # Восстановление предложений
    sentences = reconstruct_sentences_from_tokens(tokens_df)
    sentences_df = create_sentences_dataframe(sentences)
    sentences_lookup = dict(zip(sentences_df["sentence_id"], sentences_df["sentence"]))

    # Получаем примеры
    examples = [
        (sid, sentences_lookup.get(sid, ""))
        for sid in sentence_ids[:20]  # Первые 20
        if sid in sentences_lookup and sentences_lookup.get(sid, "")
    ]

    print(f"\nПримеры (первые {len(examples)}):")
    for i, (sid, sentence) in enumerate(examples, 1):
        # Проверяем наличие слова "able"
        has_able = "able" in sentence.lower()
        print(f"\n  {i}. [{sid}] Has 'able': {has_able}")
        print(f"     {sentence[:200]}...")

    # Валидируем снова и смотрим детали
    print("\n" + "=" * 80)
    print("ВАЛИДАЦИЯ С ДЕТАЛЯМИ")
    print("=" * 80)

    provider = get_provider("gemini", "gemini-3-flash-preview")
    cache = ResponseCache(cache_dir=Path("data/test_validation_cache"), enabled=True)

    import numpy as np

    synset_group = able_row["synset_group"]
    if isinstance(synset_group, (np.ndarray, list)):
        synset_group = list(synset_group) if len(synset_group) > 0 else []
    elif isinstance(synset_group, str):
        try:
            synset_group = json.loads(synset_group)
            if not isinstance(synset_group, list):
                synset_group = [synset_group]
        except (json.JSONDecodeError, TypeError):
            synset_group = [synset_group] if synset_group else []
    elif synset_group is None:
        synset_group = []
    else:
        synset_group = [synset_group]

    print(f"\nSynset group: {synset_group}")
    print(f"Examples count: {len(examples)}")

    # Валидируем
    validation = validate_examples_for_synset_group(
        lemma="able",
        synset_group=synset_group,
        primary_synset="able.a.01",
        examples=examples,
        provider=provider,
        cache=cache,
    )

    print("\nРезультаты валидации:")
    print(f"  Has valid: {validation['has_valid']}")
    print(f"  Valid sentence_ids: {validation['valid_sentence_ids']}")
    print(f"  Invalid sentence_ids: {validation['invalid_sentence_ids']}")

    # Показываем детали валидации если есть
    if validation.get("validation_details"):
        print("\nДетали валидации:")
        for ex_idx, details in list(validation["validation_details"].items())[:10]:
            if isinstance(details, dict):
                reason = details.get("reason", "N/A")
                valid = details.get("valid", "N/A")
                print(f"    Пример {ex_idx}: valid={valid}")
                print(f"      Reason: {reason[:150]}...")

    # Проверяем кэш - возможно там есть ответ LLM
    print("\n" + "=" * 80)
    print("ПРОВЕРКА КЭША")
    print("=" * 80)

    # Пытаемся найти кэшированный ответ
    cache_key = cache.generate_key(
        prompt_type="validation",
        lemma="able",
        synset_group=synset_group,
        primary_synset="able.a.01",
        examples=examples,
    )

    print(f"Cache key: {cache_key[:50]}...")
    # Проверяем есть ли в кэше
    cache_file = cache.cache_dir / f"{cache_key}.json"
    if cache_file.exists():
        print(f"✅ Кэш найден: {cache_file}")
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
                print(
                    f"  Response preview: {str(cached_data.get('response', {}).get('content', ''))[:500]}..."
                )
        except Exception as e:
            print(f"  Ошибка чтения кэша: {e}")
    else:
        print("❌ Кэш не найден")


if __name__ == "__main__":
    check_wordnet_definitions()
    check_able_examples()
