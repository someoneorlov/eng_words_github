"""Tests for WSD Gold Dataset data models.

Following TDD: tests are written before implementation.
"""


class TestTargetWord:
    """Tests for TargetWord dataclass."""

    def test_import_target_word(self):
        """Can import TargetWord from models."""
        from eng_words.wsd_gold.models import TargetWord

        assert TargetWord is not None

    def test_create_target_word(self):
        """Can create TargetWord with required fields."""
        from eng_words.wsd_gold.models import TargetWord

        target = TargetWord(
            surface="bank",
            lemma="bank",
            pos="NOUN",
            char_span=(123, 127),
        )
        assert target.surface == "bank"
        assert target.lemma == "bank"
        assert target.pos == "NOUN"
        assert target.char_span == (123, 127)

    def test_target_word_to_dict(self):
        """TargetWord can be serialized to dict."""
        from eng_words.wsd_gold.models import TargetWord

        target = TargetWord(
            surface="running",
            lemma="run",
            pos="VERB",
            char_span=(10, 17),
        )
        d = target.to_dict()
        assert d == {
            "surface": "running",
            "lemma": "run",
            "pos": "VERB",
            "char_span": [10, 17],
        }

    def test_target_word_from_dict(self):
        """TargetWord can be deserialized from dict."""
        from eng_words.wsd_gold.models import TargetWord

        d = {
            "surface": "bank",
            "lemma": "bank",
            "pos": "NOUN",
            "char_span": [123, 127],
        }
        target = TargetWord.from_dict(d)
        assert target.surface == "bank"
        assert target.lemma == "bank"
        assert target.pos == "NOUN"
        assert target.char_span == (123, 127)


class TestCandidate:
    """Tests for Candidate dataclass."""

    def test_import_candidate(self):
        """Can import Candidate from models."""
        from eng_words.wsd_gold.models import Candidate

        assert Candidate is not None

    def test_create_candidate(self):
        """Can create Candidate with required fields."""
        from eng_words.wsd_gold.models import Candidate

        candidate = Candidate(
            synset_id="bank.n.01",
            gloss="sloping land beside a body of water",
            examples=["they pulled the canoe up on the bank"],
        )
        assert candidate.synset_id == "bank.n.01"
        assert candidate.gloss == "sloping land beside a body of water"
        assert candidate.examples == ["they pulled the canoe up on the bank"]

    def test_candidate_empty_examples(self):
        """Candidate can have empty examples list."""
        from eng_words.wsd_gold.models import Candidate

        candidate = Candidate(
            synset_id="bank.n.02",
            gloss="a financial institution",
            examples=[],
        )
        assert candidate.examples == []

    def test_candidate_to_dict(self):
        """Candidate can be serialized to dict."""
        from eng_words.wsd_gold.models import Candidate

        candidate = Candidate(
            synset_id="bank.n.01",
            gloss="sloping land",
            examples=["example 1", "example 2"],
        )
        d = candidate.to_dict()
        assert d == {
            "synset_id": "bank.n.01",
            "gloss": "sloping land",
            "examples": ["example 1", "example 2"],
        }

    def test_candidate_from_dict(self):
        """Candidate can be deserialized from dict."""
        from eng_words.wsd_gold.models import Candidate

        d = {
            "synset_id": "run.v.01",
            "gloss": "move fast",
            "examples": ["he ran to the store"],
        }
        candidate = Candidate.from_dict(d)
        assert candidate.synset_id == "run.v.01"
        assert candidate.gloss == "move fast"
        assert candidate.examples == ["he ran to the store"]


class TestExampleMetadata:
    """Tests for ExampleMetadata dataclass."""

    def test_import_example_metadata(self):
        """Can import ExampleMetadata from models."""
        from eng_words.wsd_gold.models import ExampleMetadata

        assert ExampleMetadata is not None

    def test_create_example_metadata(self):
        """Can create ExampleMetadata with required fields."""
        from eng_words.wsd_gold.models import ExampleMetadata

        meta = ExampleMetadata(
            wn_sense_count=8,
            baseline_top1="bank.n.02",
            baseline_margin=0.07,
            is_multiword=False,
        )
        assert meta.wn_sense_count == 8
        assert meta.baseline_top1 == "bank.n.02"
        assert meta.baseline_margin == 0.07
        assert meta.is_multiword is False

    def test_example_metadata_to_dict(self):
        """ExampleMetadata can be serialized to dict."""
        from eng_words.wsd_gold.models import ExampleMetadata

        meta = ExampleMetadata(
            wn_sense_count=5,
            baseline_top1="run.v.01",
            baseline_margin=0.15,
            is_multiword=True,
        )
        d = meta.to_dict()
        assert d == {
            "wn_sense_count": 5,
            "baseline_top1": "run.v.01",
            "baseline_margin": 0.15,
            "is_multiword": True,
        }

    def test_example_metadata_from_dict(self):
        """ExampleMetadata can be deserialized from dict."""
        from eng_words.wsd_gold.models import ExampleMetadata

        d = {
            "wn_sense_count": 3,
            "baseline_top1": "set.v.01",
            "baseline_margin": 0.25,
            "is_multiword": False,
        }
        meta = ExampleMetadata.from_dict(d)
        assert meta.wn_sense_count == 3
        assert meta.baseline_top1 == "set.v.01"
        assert meta.baseline_margin == 0.25
        assert meta.is_multiword is False


class TestGoldExample:
    """Tests for GoldExample dataclass."""

    def test_import_gold_example(self):
        """Can import GoldExample from models."""
        from eng_words.wsd_gold.models import GoldExample

        assert GoldExample is not None

    def test_create_gold_example(self):
        """Can create GoldExample with all fields."""
        from eng_words.wsd_gold.models import (
            Candidate,
            ExampleMetadata,
            GoldExample,
            TargetWord,
        )

        example = GoldExample(
            example_id="book:test|ch:1|sent:0001|tok:5",
            source_id="test_book",
            source_bucket="classic_fiction",
            year_bucket="pre_1950",
            genre_bucket="fiction",
            text_left="He walked to the",
            target=TargetWord(
                surface="bank",
                lemma="bank",
                pos="NOUN",
                char_span=(17, 21),
            ),
            text_right="of the river.",
            context_window="He walked to the bank of the river.",
            candidates=[
                Candidate(
                    synset_id="bank.n.01",
                    gloss="sloping land",
                    examples=["by the river bank"],
                ),
                Candidate(
                    synset_id="bank.n.02",
                    gloss="financial institution",
                    examples=["deposit at the bank"],
                ),
            ],
            metadata=ExampleMetadata(
                wn_sense_count=8,
                baseline_top1="bank.n.01",
                baseline_margin=0.12,
                is_multiword=False,
            ),
        )
        assert example.example_id == "book:test|ch:1|sent:0001|tok:5"
        assert example.source_bucket == "classic_fiction"
        assert len(example.candidates) == 2
        assert example.target.lemma == "bank"

    def test_gold_example_to_dict(self):
        """GoldExample can be serialized to dict."""
        from eng_words.wsd_gold.models import (
            Candidate,
            ExampleMetadata,
            GoldExample,
            TargetWord,
        )

        example = GoldExample(
            example_id="book:test|ch:1|sent:0001|tok:5",
            source_id="test_book",
            source_bucket="classic_fiction",
            year_bucket="pre_1950",
            genre_bucket="fiction",
            text_left="left context",
            target=TargetWord(surface="bank", lemma="bank", pos="NOUN", char_span=(10, 14)),
            text_right="right context",
            context_window="full sentence",
            candidates=[Candidate(synset_id="bank.n.01", gloss="gloss1", examples=[])],
            metadata=ExampleMetadata(
                wn_sense_count=2,
                baseline_top1="bank.n.01",
                baseline_margin=0.5,
                is_multiword=False,
            ),
        )
        d = example.to_dict()
        assert d["example_id"] == "book:test|ch:1|sent:0001|tok:5"
        assert d["target"]["lemma"] == "bank"
        assert len(d["candidates"]) == 1
        assert d["metadata"]["wn_sense_count"] == 2

    def test_gold_example_from_dict(self):
        """GoldExample can be deserialized from dict."""
        from eng_words.wsd_gold.models import GoldExample

        d = {
            "example_id": "book:test|ch:1|sent:0001|tok:5",
            "source_id": "test_book",
            "source_bucket": "classic_fiction",
            "year_bucket": "pre_1950",
            "genre_bucket": "fiction",
            "text_left": "left",
            "target": {
                "surface": "bank",
                "lemma": "bank",
                "pos": "NOUN",
                "char_span": [10, 14],
            },
            "text_right": "right",
            "context_window": "sentence",
            "candidates": [{"synset_id": "bank.n.01", "gloss": "gloss", "examples": []}],
            "metadata": {
                "wn_sense_count": 2,
                "baseline_top1": "bank.n.01",
                "baseline_margin": 0.5,
                "is_multiword": False,
            },
        }
        example = GoldExample.from_dict(d)
        assert example.example_id == "book:test|ch:1|sent:0001|tok:5"
        assert example.target.lemma == "bank"
        assert len(example.candidates) == 1

    def test_gold_example_round_trip(self):
        """GoldExample survives to_dict -> from_dict round trip."""
        from eng_words.wsd_gold.models import (
            Candidate,
            ExampleMetadata,
            GoldExample,
            TargetWord,
        )

        original = GoldExample(
            example_id="book:test|ch:1|sent:0001|tok:5",
            source_id="test_book",
            source_bucket="modern_nonfiction",
            year_bucket="post_2000",
            genre_bucket="nonfiction",
            text_left="The company's",
            target=TargetWord(surface="record", lemma="record", pos="NOUN", char_span=(14, 20)),
            text_right="was impressive.",
            context_window="The company's record was impressive.",
            candidates=[
                Candidate(
                    synset_id="record.n.01",
                    gloss="written account",
                    examples=["keep a record"],
                ),
                Candidate(
                    synset_id="record.n.02",
                    gloss="disk",
                    examples=["vinyl record"],
                ),
            ],
            metadata=ExampleMetadata(
                wn_sense_count=5,
                baseline_top1="record.n.01",
                baseline_margin=0.08,
                is_multiword=False,
            ),
        )
        d = original.to_dict()
        restored = GoldExample.from_dict(d)

        assert restored.example_id == original.example_id
        assert restored.source_bucket == original.source_bucket
        assert restored.target.lemma == original.target.lemma
        assert restored.target.char_span == original.target.char_span
        assert len(restored.candidates) == len(original.candidates)
        assert restored.candidates[0].synset_id == original.candidates[0].synset_id
        assert restored.metadata.baseline_margin == original.metadata.baseline_margin

    def test_gold_example_get_candidate_ids(self):
        """GoldExample has helper to get candidate synset_ids."""
        from eng_words.wsd_gold.models import (
            Candidate,
            ExampleMetadata,
            GoldExample,
            TargetWord,
        )

        example = GoldExample(
            example_id="test",
            source_id="test",
            source_bucket="classic_fiction",
            year_bucket="pre_1950",
            genre_bucket="fiction",
            text_left="",
            target=TargetWord(surface="x", lemma="x", pos="NOUN", char_span=(0, 1)),
            text_right="",
            context_window="x",
            candidates=[
                Candidate(synset_id="x.n.01", gloss="g1", examples=[]),
                Candidate(synset_id="x.n.02", gloss="g2", examples=[]),
                Candidate(synset_id="x.n.03", gloss="g3", examples=[]),
            ],
            metadata=ExampleMetadata(
                wn_sense_count=3,
                baseline_top1="x.n.01",
                baseline_margin=0.3,
                is_multiword=False,
            ),
        )
        ids = example.get_candidate_ids()
        assert ids == ["x.n.01", "x.n.02", "x.n.03"]


class TestLLMUsage:
    """Tests for LLMUsage dataclass."""

    def test_import_llm_usage(self):
        """Can import LLMUsage from models."""
        from eng_words.wsd_gold.models import LLMUsage

        assert LLMUsage is not None

    def test_create_llm_usage(self):
        """Can create LLMUsage with all fields."""
        from eng_words.wsd_gold.models import LLMUsage

        usage = LLMUsage(
            input_tokens=500,
            output_tokens=50,
            cached_tokens=400,
            cost_usd=0.001,
        )
        assert usage.input_tokens == 500
        assert usage.output_tokens == 50
        assert usage.cached_tokens == 400
        assert usage.cost_usd == 0.001

    def test_llm_usage_defaults(self):
        """LLMUsage has sensible defaults."""
        from eng_words.wsd_gold.models import LLMUsage

        usage = LLMUsage(input_tokens=100, output_tokens=20)
        assert usage.cached_tokens == 0
        assert usage.cost_usd == 0.0

    def test_llm_usage_total_tokens(self):
        """LLMUsage has total_tokens property."""
        from eng_words.wsd_gold.models import LLMUsage

        usage = LLMUsage(input_tokens=500, output_tokens=50)
        assert usage.total_tokens == 550

    def test_llm_usage_to_dict(self):
        """LLMUsage can be serialized to dict."""
        from eng_words.wsd_gold.models import LLMUsage

        usage = LLMUsage(
            input_tokens=100,
            output_tokens=20,
            cached_tokens=80,
            cost_usd=0.0005,
        )
        d = usage.to_dict()
        assert d == {
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_tokens": 80,
            "cost_usd": 0.0005,
        }


class TestModelOutput:
    """Tests for ModelOutput dataclass."""

    def test_import_model_output(self):
        """Can import ModelOutput from models."""
        from eng_words.wsd_gold.models import ModelOutput

        assert ModelOutput is not None

    def test_create_model_output(self):
        """Can create ModelOutput with required fields."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="bank.n.01",
            confidence=0.85,
            flags=[],
            raw_text='{"chosen_synset_id": "bank.n.01", "confidence": 0.85}',
            usage=LLMUsage(input_tokens=500, output_tokens=30),
        )
        assert output.chosen_synset_id == "bank.n.01"
        assert output.confidence == 0.85
        assert output.flags == []

    def test_model_output_with_flags(self):
        """ModelOutput can have flags."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="bank.n.01",
            confidence=0.5,
            flags=["needs_more_context", "metaphor"],
            raw_text="{}",
            usage=LLMUsage(input_tokens=100, output_tokens=20),
        )
        assert "needs_more_context" in output.flags
        assert "metaphor" in output.flags

    def test_model_output_is_valid_true(self):
        """ModelOutput.is_valid() returns True for valid synset in candidates."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="bank.n.01",
            confidence=0.9,
            flags=[],
            raw_text="{}",
            usage=LLMUsage(input_tokens=100, output_tokens=20),
        )
        candidates = ["bank.n.01", "bank.n.02", "bank.n.03"]
        assert output.is_valid(candidates) is True

    def test_model_output_is_valid_false_not_in_candidates(self):
        """ModelOutput.is_valid() returns False if synset not in candidates."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="bank.n.99",
            confidence=0.9,
            flags=[],
            raw_text="{}",
            usage=LLMUsage(input_tokens=100, output_tokens=20),
        )
        candidates = ["bank.n.01", "bank.n.02"]
        assert output.is_valid(candidates) is False

    def test_model_output_is_valid_false_none_of_above(self):
        """ModelOutput with none_of_the_above flag is valid (special case)."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="",
            confidence=0.8,
            flags=["none_of_the_above"],
            raw_text="{}",
            usage=LLMUsage(input_tokens=100, output_tokens=20),
        )
        candidates = ["bank.n.01", "bank.n.02"]
        # none_of_the_above is a valid response even if synset_id is empty
        assert output.is_valid(candidates) is True

    def test_model_output_needs_referee_disagreement(self):
        """ModelOutput.needs_referee() based on flags."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="bank.n.01",
            confidence=0.3,
            flags=["needs_more_context"],
            raw_text="{}",
            usage=LLMUsage(input_tokens=100, output_tokens=20),
        )
        assert output.needs_referee() is True

    def test_model_output_needs_referee_low_confidence(self):
        """ModelOutput.needs_referee() True for low confidence."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="bank.n.01",
            confidence=0.4,
            flags=[],
            raw_text="{}",
            usage=LLMUsage(input_tokens=100, output_tokens=20),
        )
        assert output.needs_referee(confidence_threshold=0.6) is True

    def test_model_output_needs_referee_false(self):
        """ModelOutput.needs_referee() False for high confidence, no flags."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="bank.n.01",
            confidence=0.9,
            flags=[],
            raw_text="{}",
            usage=LLMUsage(input_tokens=100, output_tokens=20),
        )
        assert output.needs_referee() is False

    def test_model_output_to_dict(self):
        """ModelOutput can be serialized to dict."""
        from eng_words.wsd_gold.models import LLMUsage, ModelOutput

        output = ModelOutput(
            chosen_synset_id="bank.n.01",
            confidence=0.85,
            flags=["metaphor"],
            raw_text='{"test": 1}',
            usage=LLMUsage(input_tokens=100, output_tokens=20),
        )
        d = output.to_dict()
        assert d["chosen_synset_id"] == "bank.n.01"
        assert d["confidence"] == 0.85
        assert d["flags"] == ["metaphor"]
        assert "usage" in d

    def test_model_output_from_dict(self):
        """ModelOutput can be deserialized from dict."""
        from eng_words.wsd_gold.models import ModelOutput

        d = {
            "chosen_synset_id": "run.v.01",
            "confidence": 0.75,
            "flags": [],
            "raw_text": "{}",
            "usage": {
                "input_tokens": 200,
                "output_tokens": 30,
                "cached_tokens": 150,
                "cost_usd": 0.001,
            },
        }
        output = ModelOutput.from_dict(d)
        assert output.chosen_synset_id == "run.v.01"
        assert output.confidence == 0.75
        assert output.usage.input_tokens == 200


class TestGoldLabel:
    """Tests for GoldLabel dataclass (aggregated label)."""

    def test_import_gold_label(self):
        """Can import GoldLabel from models."""
        from eng_words.wsd_gold.models import GoldLabel

        assert GoldLabel is not None

    def test_create_gold_label(self):
        """Can create GoldLabel with all fields."""
        from eng_words.wsd_gold.models import GoldLabel

        label = GoldLabel(
            synset_id="bank.n.01",
            confidence=0.9,
            agreement_ratio=1.0,
            flags=[],
            needs_referee=False,
            judge_count=2,
        )
        assert label.synset_id == "bank.n.01"
        assert label.agreement_ratio == 1.0
        assert label.judge_count == 2

    def test_gold_label_to_dict(self):
        """GoldLabel can be serialized to dict."""
        from eng_words.wsd_gold.models import GoldLabel

        label = GoldLabel(
            synset_id="run.v.01",
            confidence=0.8,
            agreement_ratio=0.67,
            flags=["metaphor"],
            needs_referee=True,
            judge_count=3,
        )
        d = label.to_dict()
        assert d["synset_id"] == "run.v.01"
        assert d["agreement_ratio"] == 0.67
        assert d["needs_referee"] is True

    def test_gold_label_from_dict(self):
        """GoldLabel can be deserialized from dict."""
        from eng_words.wsd_gold.models import GoldLabel

        d = {
            "synset_id": "bank.n.02",
            "confidence": 0.95,
            "agreement_ratio": 1.0,
            "flags": [],
            "needs_referee": False,
            "judge_count": 2,
        }
        label = GoldLabel.from_dict(d)
        assert label.synset_id == "bank.n.02"
        assert label.agreement_ratio == 1.0


class TestValidFlags:
    """Tests for flag validation."""

    def test_valid_flags_constant(self):
        """VALID_FLAGS constant is defined."""
        from eng_words.wsd_gold.models import VALID_FLAGS

        assert "needs_more_context" in VALID_FLAGS
        assert "multiword" in VALID_FLAGS
        assert "metaphor" in VALID_FLAGS
        assert "none_of_the_above" in VALID_FLAGS
