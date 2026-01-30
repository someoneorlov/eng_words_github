"""Tests for WSD Gold Dataset sampling module.

Following TDD: tests are written before implementation.
"""

from eng_words.wsd_gold.models import (
    Candidate,
    ExampleMetadata,
    GoldExample,
    TargetWord,
)


def make_example(
    example_id: str = "test",
    source_id: str = "test_book",
    source_bucket: str = "classic_fiction",
    lemma: str = "bank",
    pos: str = "NOUN",
    sense_count: int = 5,
    baseline_margin: float = 0.2,
    is_multiword: bool = False,
) -> GoldExample:
    """Helper to create test examples."""
    return GoldExample(
        example_id=example_id,
        source_id=source_id,
        source_bucket=source_bucket,
        year_bucket="pre_1950",
        genre_bucket="fiction",
        text_left="The ",
        target=TargetWord(surface=lemma, lemma=lemma, pos=pos, char_span=(4, 4 + len(lemma))),
        text_right=" was there.",
        context_window=f"The {lemma} was there.",
        candidates=[
            Candidate(synset_id=f"{lemma}.n.0{i}", gloss=f"gloss {i}", examples=[])
            for i in range(1, sense_count + 1)
        ],
        metadata=ExampleMetadata(
            wn_sense_count=sense_count,
            baseline_top1=f"{lemma}.n.01",
            baseline_margin=baseline_margin,
            is_multiword=is_multiword,
        ),
    )


class TestDifficultyFeatures:
    """Tests for DifficultyFeatures dataclass."""

    def test_import_difficulty_features(self):
        """Can import DifficultyFeatures."""
        from eng_words.wsd_gold.sample import DifficultyFeatures

        assert DifficultyFeatures is not None

    def test_create_difficulty_features(self):
        """Can create DifficultyFeatures."""
        from eng_words.wsd_gold.sample import DifficultyFeatures

        features = DifficultyFeatures(
            wn_sense_count=5,
            baseline_margin=0.2,
            pos="NOUN",
            is_multiword=False,
            is_phrasal_verb=False,
            difficulty_level="medium",
        )
        assert features.wn_sense_count == 5
        assert features.difficulty_level == "medium"


class TestCalculateDifficultyFeatures:
    """Tests for calculate_difficulty_features function."""

    def test_import_function(self):
        """Can import calculate_difficulty_features."""
        from eng_words.wsd_gold.sample import calculate_difficulty_features

        assert calculate_difficulty_features is not None

    def test_returns_difficulty_features(self):
        """Function returns DifficultyFeatures object."""
        from eng_words.wsd_gold.sample import (
            DifficultyFeatures,
            calculate_difficulty_features,
        )

        example = make_example(sense_count=5, baseline_margin=0.2)
        features = calculate_difficulty_features(example)

        assert isinstance(features, DifficultyFeatures)

    def test_easy_for_low_sense_count_high_margin(self):
        """Easy difficulty for low sense count and high margin."""
        from eng_words.wsd_gold.sample import calculate_difficulty_features

        example = make_example(sense_count=2, baseline_margin=0.5)
        features = calculate_difficulty_features(example)

        assert features.difficulty_level == "easy"

    def test_hard_for_high_sense_count_low_margin(self):
        """Hard difficulty for high sense count and low margin."""
        from eng_words.wsd_gold.sample import calculate_difficulty_features

        example = make_example(sense_count=8, baseline_margin=0.1)
        features = calculate_difficulty_features(example)

        assert features.difficulty_level == "hard"

    def test_medium_for_moderate_values(self):
        """Medium difficulty for moderate sense count and margin."""
        from eng_words.wsd_gold.sample import calculate_difficulty_features

        example = make_example(sense_count=4, baseline_margin=0.25)
        features = calculate_difficulty_features(example)

        assert features.difficulty_level == "medium"

    def test_detects_multiword(self):
        """Correctly detects multiword expressions."""
        from eng_words.wsd_gold.sample import calculate_difficulty_features

        example = make_example(lemma="break_down", is_multiword=True)
        features = calculate_difficulty_features(example)

        assert features.is_multiword is True

    def test_detects_phrasal_verb(self):
        """Correctly detects phrasal verbs."""
        from eng_words.wsd_gold.sample import calculate_difficulty_features

        example = make_example(lemma="give_up", pos="VERB", is_multiword=True)
        features = calculate_difficulty_features(example)

        assert features.is_phrasal_verb is True

    def test_extracts_pos(self):
        """Extracts POS from example."""
        from eng_words.wsd_gold.sample import calculate_difficulty_features

        example = make_example(pos="VERB")
        features = calculate_difficulty_features(example)

        assert features.pos == "VERB"


class TestClassifyDifficulty:
    """Tests for classify_difficulty function."""

    def test_import_function(self):
        """Can import classify_difficulty."""
        from eng_words.wsd_gold.sample import classify_difficulty

        assert classify_difficulty is not None

    def test_easy_thresholds(self):
        """Easy when sense_count <= 3 and margin >= 0.3."""
        from eng_words.wsd_gold.sample import classify_difficulty

        assert classify_difficulty(sense_count=2, margin=0.4) == "easy"
        assert classify_difficulty(sense_count=3, margin=0.3) == "easy"

    def test_hard_thresholds(self):
        """Hard when sense_count >= 7 or margin < 0.15."""
        from eng_words.wsd_gold.sample import classify_difficulty

        assert classify_difficulty(sense_count=8, margin=0.2) == "hard"
        assert classify_difficulty(sense_count=4, margin=0.1) == "hard"

    def test_medium_default(self):
        """Medium for everything else."""
        from eng_words.wsd_gold.sample import classify_difficulty

        assert classify_difficulty(sense_count=5, margin=0.2) == "medium"
        assert classify_difficulty(sense_count=4, margin=0.25) == "medium"


class TestStratifiedSample:
    """Tests for stratified_sample function."""

    def test_import_function(self):
        """Can import stratified_sample."""
        from eng_words.wsd_gold.sample import stratified_sample

        assert stratified_sample is not None

    def test_returns_correct_size(self):
        """Returns exactly n examples."""
        from eng_words.wsd_gold.sample import stratified_sample

        examples = [make_example(example_id=f"ex{i}") for i in range(100)]
        sampled = stratified_sample(examples, n=50, random_state=42)

        assert len(sampled) == 50

    def test_respects_random_state(self):
        """Same random_state produces same sample."""
        from eng_words.wsd_gold.sample import stratified_sample

        examples = [make_example(example_id=f"ex{i}") for i in range(100)]
        sample1 = stratified_sample(examples, n=50, random_state=42)
        sample2 = stratified_sample(examples, n=50, random_state=42)

        ids1 = [e.example_id for e in sample1]
        ids2 = [e.example_id for e in sample2]
        assert ids1 == ids2

    def test_different_random_state_different_sample(self):
        """Different random_state produces different sample."""
        from eng_words.wsd_gold.sample import stratified_sample

        examples = [make_example(example_id=f"ex{i}") for i in range(100)]
        sample1 = stratified_sample(examples, n=50, random_state=42)
        sample2 = stratified_sample(examples, n=50, random_state=123)

        ids1 = set(e.example_id for e in sample1)
        ids2 = set(e.example_id for e in sample2)
        # Should have different samples (with high probability)
        assert ids1 != ids2

    def test_stratifies_by_difficulty(self):
        """Stratifies by difficulty level."""
        from eng_words.wsd_gold.sample import stratified_sample

        # Create examples with different difficulties
        examples = []
        for i in range(30):
            examples.append(make_example(example_id=f"easy{i}", sense_count=2, baseline_margin=0.5))
        for i in range(30):
            examples.append(
                make_example(example_id=f"medium{i}", sense_count=5, baseline_margin=0.2)
            )
        for i in range(30):
            examples.append(
                make_example(example_id=f"hard{i}", sense_count=10, baseline_margin=0.05)
            )

        sampled = stratified_sample(examples, n=60, random_state=42)

        # Should have examples from all difficulty levels
        easy_count = sum(1 for e in sampled if "easy" in e.example_id)
        medium_count = sum(1 for e in sampled if "medium" in e.example_id)
        hard_count = sum(1 for e in sampled if "hard" in e.example_id)

        # Should have representation from all levels
        assert easy_count > 0
        assert medium_count > 0
        assert hard_count > 0

    def test_handles_small_n(self):
        """Handles when n is smaller than available examples."""
        from eng_words.wsd_gold.sample import stratified_sample

        examples = [make_example(example_id=f"ex{i}") for i in range(10)]
        sampled = stratified_sample(examples, n=5, random_state=42)

        assert len(sampled) == 5

    def test_handles_n_larger_than_available(self):
        """Handles when n is larger than available examples."""
        from eng_words.wsd_gold.sample import stratified_sample

        examples = [make_example(example_id=f"ex{i}") for i in range(5)]
        sampled = stratified_sample(examples, n=10, random_state=42)

        # Should return all available examples
        assert len(sampled) == 5


class TestSplitBySource:
    """Tests for split_by_source function."""

    def test_import_function(self):
        """Can import split_by_source."""
        from eng_words.wsd_gold.sample import split_by_source

        assert split_by_source is not None

    def test_returns_two_lists(self):
        """Returns tuple of (dev, test_locked) lists."""
        from eng_words.wsd_gold.sample import split_by_source

        examples = [make_example(example_id=f"ex{i}", source_id=f"book{i % 5}") for i in range(50)]
        dev, test_locked = split_by_source(examples, dev_ratio=0.25, random_state=42)

        assert isinstance(dev, list)
        assert isinstance(test_locked, list)

    def test_no_source_overlap(self):
        """No source_id appears in both dev and test_locked."""
        from eng_words.wsd_gold.sample import split_by_source

        examples = [
            make_example(example_id=f"ex{i}", source_id=f"book{i % 10}") for i in range(100)
        ]
        dev, test_locked = split_by_source(examples, dev_ratio=0.25, random_state=42)

        dev_sources = set(e.source_id for e in dev)
        test_sources = set(e.source_id for e in test_locked)

        assert dev_sources.isdisjoint(test_sources)

    def test_approximate_ratio(self):
        """Split ratio is approximately correct."""
        from eng_words.wsd_gold.sample import split_by_source

        examples = [
            make_example(example_id=f"ex{i}", source_id=f"book{i % 20}") for i in range(200)
        ]
        dev, test_locked = split_by_source(examples, dev_ratio=0.25, random_state=42)

        total = len(dev) + len(test_locked)
        dev_ratio = len(dev) / total

        # Should be approximately 25% (allow some variance due to source grouping)
        assert 0.15 <= dev_ratio <= 0.35

    def test_reproducible_with_random_state(self):
        """Same random_state produces same split."""
        from eng_words.wsd_gold.sample import split_by_source

        examples = [
            make_example(example_id=f"ex{i}", source_id=f"book{i % 10}") for i in range(100)
        ]

        dev1, test1 = split_by_source(examples, dev_ratio=0.25, random_state=42)
        dev2, test2 = split_by_source(examples, dev_ratio=0.25, random_state=42)

        dev_ids1 = set(e.example_id for e in dev1)
        dev_ids2 = set(e.example_id for e in dev2)

        assert dev_ids1 == dev_ids2

    def test_all_examples_included(self):
        """All examples are in either dev or test_locked."""
        from eng_words.wsd_gold.sample import split_by_source

        examples = [make_example(example_id=f"ex{i}", source_id=f"book{i % 5}") for i in range(50)]
        dev, test_locked = split_by_source(examples, dev_ratio=0.25, random_state=42)

        all_ids = set(e.example_id for e in examples)
        split_ids = set(e.example_id for e in dev) | set(e.example_id for e in test_locked)

        assert all_ids == split_ids

    def test_handles_single_source(self):
        """Handles case where all examples from single source."""
        from eng_words.wsd_gold.sample import split_by_source

        examples = [make_example(example_id=f"ex{i}", source_id="single_book") for i in range(20)]
        dev, test_locked = split_by_source(examples, dev_ratio=0.25, random_state=42)

        # With single source, all should go to one split
        assert len(dev) == 0 or len(test_locked) == 0
        assert len(dev) + len(test_locked) == 20


class TestGetSamplingStats:
    """Tests for get_sampling_stats function."""

    def test_import_function(self):
        """Can import get_sampling_stats."""
        from eng_words.wsd_gold.sample import get_sampling_stats

        assert get_sampling_stats is not None

    def test_returns_stats_dict(self):
        """Returns dictionary with stats."""
        from eng_words.wsd_gold.sample import get_sampling_stats

        examples = [
            make_example(example_id=f"ex{i}", pos=["NOUN", "VERB", "ADJ"][i % 3]) for i in range(30)
        ]
        stats = get_sampling_stats(examples)

        assert isinstance(stats, dict)
        assert "total" in stats
        assert "by_pos" in stats
        assert "by_difficulty" in stats
        assert "by_source_bucket" in stats

    def test_counts_total(self):
        """Correctly counts total examples."""
        from eng_words.wsd_gold.sample import get_sampling_stats

        examples = [make_example(example_id=f"ex{i}") for i in range(25)]
        stats = get_sampling_stats(examples)

        assert stats["total"] == 25

    def test_counts_by_pos(self):
        """Correctly counts by POS."""
        from eng_words.wsd_gold.sample import get_sampling_stats

        examples = [make_example(example_id=f"noun{i}", pos="NOUN") for i in range(10)] + [
            make_example(example_id=f"verb{i}", pos="VERB") for i in range(5)
        ]

        stats = get_sampling_stats(examples)

        assert stats["by_pos"]["NOUN"] == 10
        assert stats["by_pos"]["VERB"] == 5
