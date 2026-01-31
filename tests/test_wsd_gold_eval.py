"""Tests for WSD evaluation on gold dataset."""

import json
from pathlib import Path


from eng_words.wsd_gold.models import Candidate, ExampleMetadata, GoldExample, TargetWord

# =============================================================================
# TEST DATA FIXTURES
# =============================================================================


def make_gold_example(
    lemma: str = "bank",
    pos: str = "NOUN",
    context: str = "I went to the bank to deposit money.",
    gold_synset_id: str = "bank.n.01",
    baseline_top1: str = "bank.n.01",
) -> GoldExample:
    """Create a test GoldExample."""
    return GoldExample(
        example_id=f"test|{lemma}|1",
        source_id="test_book",
        source_bucket="fiction",
        year_bucket="2020s",
        genre_bucket="novel",
        text_left="",
        target=TargetWord(
            surface=lemma,
            lemma=lemma,
            pos=pos,
            char_span=(0, len(lemma)),
        ),
        text_right="",
        context_window=context,
        candidates=[
            Candidate(synset_id="bank.n.01", gloss="financial institution", examples=[]),
            Candidate(synset_id="bank.n.02", gloss="river bank", examples=[]),
        ],
        metadata=ExampleMetadata(
            wn_sense_count=2,
            baseline_top1=baseline_top1,
            baseline_margin=0.5,
            is_multiword=False,
        ),
    )


# =============================================================================
# TESTS FOR WSD EVALUATION FUNCTIONS
# =============================================================================


class TestLoadGoldExamples:
    """Tests for loading gold examples from JSONL."""

    def test_load_from_file(self, tmp_path: Path):
        """Load examples from JSONL file."""
        from eng_words.wsd_gold.eval import load_gold_examples

        # Create test file
        test_file = tmp_path / "gold.jsonl"
        ex1 = make_gold_example(lemma="bank", gold_synset_id="bank.n.01")
        ex2 = make_gold_example(lemma="run", gold_synset_id="run.v.01")

        with open(test_file, "w") as f:
            f.write(json.dumps({**ex1.to_dict(), "gold_synset_id": "bank.n.01"}) + "\n")
            f.write(json.dumps({**ex2.to_dict(), "gold_synset_id": "run.v.01"}) + "\n")

        examples = load_gold_examples(test_file)

        assert len(examples) == 2
        assert examples[0]["gold_synset_id"] == "bank.n.01"
        assert examples[1]["gold_synset_id"] == "run.v.01"


class TestEvaluateSingleExample:
    """Tests for evaluating a single WSD prediction."""

    def test_correct_prediction(self):
        """Correct prediction returns is_correct=True."""
        from eng_words.wsd_gold.eval import evaluate_single

        result = evaluate_single(
            predicted_synset="bank.n.01",
            gold_synset="bank.n.01",
        )

        assert result["is_correct"] is True
        assert result["predicted"] == "bank.n.01"
        assert result["gold"] == "bank.n.01"

    def test_incorrect_prediction(self):
        """Incorrect prediction returns is_correct=False."""
        from eng_words.wsd_gold.eval import evaluate_single

        result = evaluate_single(
            predicted_synset="bank.n.02",
            gold_synset="bank.n.01",
        )

        assert result["is_correct"] is False

    def test_none_prediction(self):
        """None prediction is handled."""
        from eng_words.wsd_gold.eval import evaluate_single

        result = evaluate_single(
            predicted_synset=None,
            gold_synset="bank.n.01",
        )

        assert result["is_correct"] is False
        assert result["predicted"] is None


class TestComputeMetrics:
    """Tests for computing evaluation metrics."""

    def test_perfect_accuracy(self):
        """100% accuracy with all correct."""
        from eng_words.wsd_gold.eval import compute_metrics

        results = [
            {"is_correct": True, "predicted": "a", "gold": "a"},
            {"is_correct": True, "predicted": "b", "gold": "b"},
            {"is_correct": True, "predicted": "c", "gold": "c"},
        ]

        metrics = compute_metrics(results)

        assert metrics["accuracy"] == 1.0
        assert metrics["total"] == 3
        assert metrics["correct"] == 3

    def test_partial_accuracy(self):
        """Partial accuracy calculated correctly."""
        from eng_words.wsd_gold.eval import compute_metrics

        results = [
            {"is_correct": True, "predicted": "a", "gold": "a"},
            {"is_correct": False, "predicted": "x", "gold": "b"},
            {"is_correct": True, "predicted": "c", "gold": "c"},
            {"is_correct": False, "predicted": "y", "gold": "d"},
        ]

        metrics = compute_metrics(results)

        assert metrics["accuracy"] == 0.5
        assert metrics["total"] == 4
        assert metrics["correct"] == 2

    def test_empty_results(self):
        """Empty results handled."""
        from eng_words.wsd_gold.eval import compute_metrics

        metrics = compute_metrics([])

        assert metrics["accuracy"] == 0.0
        assert metrics["total"] == 0


class TestComputeMetricsBySegment:
    """Tests for computing metrics broken down by segment."""

    def test_by_pos(self):
        """Metrics computed by POS tag."""
        from eng_words.wsd_gold.eval import compute_metrics_by_segment

        results = [
            {"is_correct": True, "pos": "NOUN"},
            {"is_correct": True, "pos": "NOUN"},
            {"is_correct": False, "pos": "VERB"},
            {"is_correct": True, "pos": "VERB"},
        ]

        by_pos = compute_metrics_by_segment(results, "pos")

        assert by_pos["NOUN"]["accuracy"] == 1.0
        assert by_pos["NOUN"]["total"] == 2
        assert by_pos["VERB"]["accuracy"] == 0.5
        assert by_pos["VERB"]["total"] == 2

    def test_by_difficulty(self):
        """Metrics computed by difficulty."""
        from eng_words.wsd_gold.eval import compute_metrics_by_segment

        results = [
            {"is_correct": True, "difficulty": "easy"},
            {"is_correct": True, "difficulty": "easy"},
            {"is_correct": False, "difficulty": "hard"},
            {"is_correct": False, "difficulty": "hard"},
        ]

        by_diff = compute_metrics_by_segment(results, "difficulty")

        assert by_diff["easy"]["accuracy"] == 1.0
        assert by_diff["hard"]["accuracy"] == 0.0


class TestRunWSDPrediction:
    """Tests for running WSD prediction on examples."""

    def test_returns_synset_id(self):
        """Returns synset ID from WSD backend."""
        from eng_words.wsd_gold.eval import run_wsd_prediction

        # This test requires the actual backend, so we'll mock it
        example = {
            "context_window": "I went to the bank to deposit money.",
            "target": {"lemma": "bank", "pos": "NOUN"},
        }

        # Mock WSD backend
        class MockBackend:
            def disambiguate_word(self, sentence, lemma, pos):
                class Result:
                    sense_id = "bank.n.01"
                    confidence = 0.8

                return Result()

        result = run_wsd_prediction(example, MockBackend())

        assert result["predicted_synset"] == "bank.n.01"
        assert result["confidence"] == 0.8


class TestFullEvaluation:
    """Tests for full evaluation pipeline."""

