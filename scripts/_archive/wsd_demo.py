#!/usr/bin/env python3
"""
WSD Demo Script - Word Sense Disambiguation using Sentence-Transformers + WordNet.

This script demonstrates the WSD approach:
1. Get sentence embedding using sentence-transformers
2. Get all candidate synsets from WordNet for a word
3. Compare sentence embedding with definition embeddings
4. Select the synset with highest cosine similarity

Usage:
    python scripts/wsd_demo.py

Requirements:
    pip install sentence-transformers
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

import time
from dataclasses import dataclass

import numpy as np
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class WSDResult:
    """Result of word sense disambiguation."""

    lemma: str
    sentence: str
    synset_name: str | None
    supersense: str | None
    definition: str | None
    confidence: float
    all_synsets: list[str]
    processing_time_ms: float


@dataclass
class TestCase:
    """A test case for WSD evaluation."""

    sentence: str
    target_word: str
    pos: str  # 'n' for noun, 'v' for verb, 'a' for adjective, 'r' for adverb
    expected_supersense: str
    description: str


# ============================================================================
# Test Cases
# ============================================================================

TEST_CASES = [
    # Bank - financial vs river
    TestCase(
        sentence="I need to deposit money at the bank before it closes.",
        target_word="bank",
        pos="n",
        expected_supersense="noun.artifact",
        description="bank (financial institution)",
    ),
    TestCase(
        sentence="We had a picnic on the river bank.",
        target_word="bank",
        pos="n",
        expected_supersense="noun.object",
        description="bank (river shore)",
    ),
    # Run - motion vs manage
    TestCase(
        sentence="He ran quickly to catch the bus.",
        target_word="run",
        pos="v",
        expected_supersense="verb.motion",
        description="run (move fast)",
    ),
    TestCase(
        sentence="She runs a successful company with 50 employees.",
        target_word="run",
        pos="v",
        expected_supersense="verb.social",
        description="run (manage/operate)",
    ),
    # Bat - animal vs sports equipment
    TestCase(
        sentence="The bat flew out of the cave at dusk.",
        target_word="bat",
        pos="n",
        expected_supersense="noun.animal",
        description="bat (flying mammal)",
    ),
    TestCase(
        sentence="He swung the baseball bat and hit a home run.",
        target_word="bat",
        pos="n",
        expected_supersense="noun.artifact",
        description="bat (sports equipment)",
    ),
    # Light - illumination vs weight
    TestCase(
        sentence="The light from the lamp illuminated the room.",
        target_word="light",
        pos="n",
        expected_supersense="noun.phenomenon",
        description="light (illumination)",
    ),
    TestCase(
        sentence="This suitcase is very light, easy to carry.",
        target_word="light",
        pos="a",
        expected_supersense="adj.all",
        description="light (not heavy)",
    ),
    # Play - performance vs activity
    TestCase(
        sentence="We went to see a play at the theater.",
        target_word="play",
        pos="n",
        expected_supersense="noun.communication",
        description="play (theatrical performance)",
    ),
    TestCase(
        sentence="The children play in the park every afternoon.",
        target_word="play",
        pos="v",
        expected_supersense="verb.competition",
        description="play (engage in activity)",
    ),
    # Book - object vs reserve
    TestCase(
        sentence="I'm reading an interesting book about history.",
        target_word="book",
        pos="n",
        expected_supersense="noun.communication",
        description="book (written work)",
    ),
    TestCase(
        sentence="I need to book a hotel room for next week.",
        target_word="book",
        pos="v",
        expected_supersense="verb.communication",
        description="book (make reservation)",
    ),
    # Spring - season vs water source vs jump
    TestCase(
        sentence="The flowers bloom in spring.",
        target_word="spring",
        pos="n",
        expected_supersense="noun.time",
        description="spring (season)",
    ),
    TestCase(
        sentence="Fresh water flows from the mountain spring.",
        target_word="spring",
        pos="n",
        expected_supersense="noun.object",
        description="spring (water source)",
    ),
    # Leaves - departs vs foliage
    TestCase(
        sentence="The train leaves at 3 PM.",
        target_word="leave",
        pos="v",
        expected_supersense="verb.motion",
        description="leave (depart)",
    ),
    TestCase(
        sentence="The autumn leaves are beautiful.",
        target_word="leaf",
        pos="n",
        expected_supersense="noun.plant",
        description="leaf (foliage)",
    ),
    # Match - competition vs fire starter
    TestCase(
        sentence="We watched the football match on TV.",
        target_word="match",
        pos="n",
        expected_supersense="noun.event",
        description="match (competition)",
    ),
    TestCase(
        sentence="He lit the candle with a match.",
        target_word="match",
        pos="n",
        expected_supersense="noun.artifact",
        description="match (fire starter)",
    ),
    # Crane - bird vs machine
    TestCase(
        sentence="A crane stood in the shallow water looking for fish.",
        target_word="crane",
        pos="n",
        expected_supersense="noun.animal",
        description="crane (bird)",
    ),
    TestCase(
        sentence="The construction crane lifted the heavy beams.",
        target_word="crane",
        pos="n",
        expected_supersense="noun.artifact",
        description="crane (machine)",
    ),
]


# ============================================================================
# WSD Functions
# ============================================================================


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the sentence transformer model."""
    print(f"Loading model: {model_name}")
    start = time.time()
    model = SentenceTransformer(model_name)
    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")
    return model


def get_synsets(lemma: str, pos: str | None = None) -> list:
    """Get all WordNet synsets for a lemma."""
    return wn.synsets(lemma, pos=pos)


def synset_to_supersense(synset) -> str:
    """Convert a WordNet synset to its supersense (lexname)."""
    return synset.lexname()


def disambiguate_word(
    model: SentenceTransformer,
    sentence: str,
    target_word: str,
    pos: str | None = None,
) -> WSDResult:
    """
    Disambiguate a word in context using sentence embeddings.

    Args:
        model: Sentence transformer model
        sentence: The sentence containing the target word
        target_word: The word to disambiguate
        pos: Part of speech filter ('n', 'v', 'a', 'r') or None for all

    Returns:
        WSDResult with the best matching synset
    """
    start_time = time.time()

    # Get all candidate synsets
    synsets = get_synsets(target_word, pos)

    if not synsets:
        return WSDResult(
            lemma=target_word,
            sentence=sentence,
            synset_name=None,
            supersense="unknown",
            definition=None,
            confidence=0.0,
            all_synsets=[],
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    # Get definitions for all synsets
    definitions = [s.definition() for s in synsets]

    # Get embeddings
    sentence_emb = model.encode([sentence])[0]
    definition_embs = model.encode(definitions)

    # Calculate similarities
    similarities = cosine_similarity([sentence_emb], definition_embs)[0]

    # Find best match
    best_idx = int(np.argmax(similarities))
    best_synset = synsets[best_idx]

    elapsed_ms = (time.time() - start_time) * 1000

    return WSDResult(
        lemma=target_word,
        sentence=sentence,
        synset_name=best_synset.name(),
        supersense=synset_to_supersense(best_synset),
        definition=definitions[best_idx],
        confidence=float(similarities[best_idx]),
        all_synsets=[s.name() for s in synsets],
        processing_time_ms=elapsed_ms,
    )


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_test_cases(model: SentenceTransformer, test_cases: list[TestCase]) -> dict:
    """
    Evaluate WSD on test cases.

    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = len(test_cases)
    results = []
    total_time_ms = 0.0

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    for i, tc in enumerate(test_cases, 1):
        result = disambiguate_word(model, tc.sentence, tc.target_word, tc.pos)
        total_time_ms += result.processing_time_ms

        is_correct = result.supersense == tc.expected_supersense

        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"

        print(f"\n{i}. {tc.description}")
        print(f'   Sentence: "{tc.sentence}"')
        print(f"   Expected: {tc.expected_supersense}")
        print(f"   Got:      {result.supersense} ({result.synset_name})")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Time: {result.processing_time_ms:.1f}ms")
        print(f"   Status: {status}")

        if not is_correct:
            print(f"   Definition: {result.definition}")
            print(f"   All synsets: {result.all_synsets[:5]}...")

        results.append(
            {
                "test_case": tc,
                "result": result,
                "correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0
    avg_time_ms = total_time_ms / total if total > 0 else 0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")
    print(f"Average time per word: {avg_time_ms:.1f}ms")
    print(f"Total time: {total_time_ms:.1f}ms")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_time_ms": avg_time_ms,
        "total_time_ms": total_time_ms,
        "results": results,
    }


def benchmark_speed(model: SentenceTransformer, n_sentences: int = 100) -> dict:
    """
    Benchmark WSD speed on synthetic sentences.

    Args:
        model: Sentence transformer model
        n_sentences: Number of sentences to process

    Returns:
        Dictionary with timing statistics
    """
    print(f"\n{'=' * 80}")
    print(f"SPEED BENCHMARK ({n_sentences} sentences)")
    print("=" * 80)

    # Generate test sentences
    test_words = [
        ("bank", "n"),
        ("run", "v"),
        ("play", "v"),
        ("light", "n"),
        ("book", "n"),
    ]

    sentences = []
    for i in range(n_sentences):
        word, pos = test_words[i % len(test_words)]
        sentences.append((f"This is test sentence {i} with the word {word}.", word, pos))

    # Benchmark
    start = time.time()
    times = []

    for sentence, word, pos in sentences:
        result = disambiguate_word(model, sentence, word, pos)
        times.append(result.processing_time_ms)

    total_time = time.time() - start

    times_arr = np.array(times)
    stats = {
        "n_sentences": n_sentences,
        "total_time_s": total_time,
        "avg_time_ms": float(np.mean(times_arr)),
        "median_time_ms": float(np.median(times_arr)),
        "min_time_ms": float(np.min(times_arr)),
        "max_time_ms": float(np.max(times_arr)),
        "std_time_ms": float(np.std(times_arr)),
        "throughput_per_s": n_sentences / total_time,
    }

    print(f"Total time: {stats['total_time_s']:.2f}s")
    print(f"Average time per word: {stats['avg_time_ms']:.1f}ms")
    print(f"Median time: {stats['median_time_ms']:.1f}ms")
    print(f"Min/Max: {stats['min_time_ms']:.1f}ms / {stats['max_time_ms']:.1f}ms")
    print(f"Std dev: {stats['std_time_ms']:.1f}ms")
    print(f"Throughput: {stats['throughput_per_s']:.1f} words/second")

    return stats


def check_memory_usage() -> dict:
    """Check current memory usage."""
    import os

    try:
        import psutil

        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            "rss_mb": mem_info.rss / 1024 / 1024,
            "vms_mb": mem_info.vms / 1024 / 1024,
        }
    except ImportError:
        return {"error": "psutil not installed"}


# ============================================================================
# Main
# ============================================================================


def main():
    """Run WSD demo and evaluation."""
    print("=" * 80)
    print("WSD DEMO - Word Sense Disambiguation")
    print("Using: Sentence-Transformers + WordNet")
    print("=" * 80)

    # Check memory before loading model
    mem_before = check_memory_usage()
    print(f"\nMemory before model load: {mem_before}")

    # Load model
    model = load_model()

    # Check memory after loading model
    mem_after = check_memory_usage()
    print(f"Memory after model load: {mem_after}")

    if "rss_mb" in mem_before and "rss_mb" in mem_after:
        model_mem = mem_after["rss_mb"] - mem_before["rss_mb"]
        print(f"Model memory footprint: ~{model_mem:.0f}MB")

    # Run evaluation
    eval_results = evaluate_test_cases(model, TEST_CASES)

    # Run speed benchmark
    speed_results = benchmark_speed(model, n_sentences=100)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Accuracy on test cases: {eval_results['accuracy']:.1%}")
    print(f"Average WSD time: {speed_results['avg_time_ms']:.1f}ms per word")
    print(f"Throughput: {speed_results['throughput_per_s']:.1f} words/second")

    # Check if criteria are met
    print("\n" + "-" * 40)
    print("CRITERIA CHECK:")

    accuracy_ok = eval_results["accuracy"] >= 0.70
    time_ok = speed_results["avg_time_ms"] < 100
    mem_ok = mem_after.get("rss_mb", 0) < 2000

    print(f"  Accuracy >= 70%: {'✓' if accuracy_ok else '✗'} ({eval_results['accuracy']:.1%})")
    print(f"  Time < 100ms/word: {'✓' if time_ok else '✗'} ({speed_results['avg_time_ms']:.1f}ms)")
    print(f"  Memory < 2GB: {'✓' if mem_ok else '✗'} ({mem_after.get('rss_mb', 'N/A')}MB)")

    if accuracy_ok and time_ok:
        print("\n✓ All criteria met! Ready for Этап 2.")
    else:
        print("\n✗ Some criteria not met. Consider Этап 1.1 (fine-tuning).")

    return {
        "evaluation": eval_results,
        "speed": speed_results,
        "memory": mem_after,
    }


if __name__ == "__main__":
    main()
