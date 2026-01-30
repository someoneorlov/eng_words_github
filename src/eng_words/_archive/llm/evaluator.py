"""WSD Evaluator for assessing Word Sense Disambiguation quality.

Uses LLM-based "blind" evaluation where the LLM sees the sentence and
candidate senses (A, B, C, ...) without knowing which one was assigned.
"""

from dataclasses import dataclass, field

import pandas as pd

from eng_words.constants import (
    LEMMA,
    POS,
    SENTENCE_ID,
    SUPERSENSE,
    SYNSET_ID,
)
from eng_words.llm.base import LLMProvider
from eng_words.llm.prompts import build_evaluation_prompt, get_candidate_index


def sample_tokens_for_evaluation(
    sense_tokens_df: pd.DataFrame,
    sentences_df: pd.DataFrame,
    n_samples: int = 200,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Sample tokens for WSD evaluation with stratification by supersense.

    Args:
        sense_tokens_df: DataFrame with WSD annotations.
            Required columns: lemma, pos, synset_id, supersense, sentence_id.
        sentences_df: DataFrame with sentences.
            Required columns: sentence_id, text.
        n_samples: Number of samples to select. Defaults to 200.
        random_state: Random state for reproducibility.

    Returns:
        DataFrame with sampled tokens including sentence text.
        Columns: lemma, pos, synset_id, supersense, sentence_id, sentence_text.
    """
    # Filter only tokens with actual WSD results (non-null synset_id, non-unknown supersense)
    valid_mask = (
        sense_tokens_df[SYNSET_ID].notna()
        & (sense_tokens_df[SYNSET_ID] != "")
        & (sense_tokens_df[SUPERSENSE] != "unknown")
    )
    filtered_df = sense_tokens_df[valid_mask].copy()

    # Merge with sentences to get text
    text_col = "text" if "text" in sentences_df.columns else "sentence_text"
    merged = filtered_df.merge(
        sentences_df[[SENTENCE_ID, text_col]],
        on=SENTENCE_ID,
        how="left",
    )

    # Rename to consistent column name
    if text_col != "sentence_text":
        merged = merged.rename(columns={text_col: "sentence_text"})

    # Remove rows without sentence text
    merged = merged.dropna(subset=["sentence_text"])

    # If fewer rows than requested, return all
    if len(merged) <= n_samples:
        return merged[[LEMMA, POS, SYNSET_ID, SUPERSENSE, SENTENCE_ID, "sentence_text"]]

    # Stratified sampling by supersense
    # Group by supersense and sample proportionally
    supersense_counts = merged[SUPERSENSE].value_counts()
    total_count = len(merged)

    samples_per_supersense = {}
    remaining = n_samples

    for supersense in supersense_counts.index:
        count = supersense_counts[supersense]
        # Proportional allocation
        allocated = max(1, int(n_samples * count / total_count))
        samples_per_supersense[supersense] = min(allocated, count, remaining)
        remaining -= samples_per_supersense[supersense]
        if remaining <= 0:
            break

    # Sample from each supersense
    sampled_dfs = []
    for supersense, n_to_sample in samples_per_supersense.items():
        group = merged[merged[SUPERSENSE] == supersense]
        if len(group) <= n_to_sample:
            sampled_dfs.append(group)
        else:
            sampled_dfs.append(group.sample(n=n_to_sample, random_state=random_state))

    result = pd.concat(sampled_dfs, ignore_index=True)

    return result[[LEMMA, POS, SYNSET_ID, SUPERSENSE, SENTENCE_ID, "sentence_text"]]


@dataclass
class EvaluationResult:
    """Result of evaluating a single WSD sample.

    Attributes:
        lemma: The word being evaluated.
        sentence_text: The sentence containing the word.
        assigned_synset: The synset assigned by WSD.
        jury_votes: List of votes from each provider.
        jury_verdict: Final verdict (correct/incorrect/uncertain).
        confidence: Confidence score based on jury agreement.
    """

    lemma: str
    sentence_text: str
    assigned_synset: str
    jury_votes: list[str] = field(default_factory=list)
    jury_verdict: str = "uncertain"
    confidence: float = 0.0


class WSDEvaluator:
    """Evaluator for WSD quality using LLM jury.

    Uses multiple LLM providers (jury) to evaluate WSD results.
    Each provider votes independently, and results are aggregated.

    Args:
        providers: List of LLMProvider instances to use as jury.
            Recommended: 2 providers (e.g., GPT-4o-mini + Claude Haiku).

    Raises:
        ValueError: If no providers are given.
    """

    def __init__(self, providers: list[LLMProvider]):
        if not providers:
            raise ValueError("At least one provider is required for evaluation.")
        self.providers = providers

    def evaluate_sample(
        self,
        lemma: str,
        sentence_text: str,
        assigned_synset: str,
        candidate_synsets: list[dict],
        pos: str = "NOUN",
    ) -> EvaluationResult:
        """Evaluate a single WSD sample using the jury.

        This uses "blind" evaluation: the LLM sees candidate senses labeled
        A, B, C, ... without knowing which one was assigned by our WSD.

        Args:
            lemma: The target word.
            sentence_text: The sentence containing the word.
            assigned_synset: The synset assigned by WSD pipeline.
            candidate_synsets: List of candidate synsets with definitions.
                Each dict: {"synset_id": str, "definition": str}
            pos: Part of speech (NOUN, VERB, ADJ, ADV).

        Returns:
            EvaluationResult with jury votes and verdict.
        """
        # Find the index of the assigned synset in candidates
        assigned_index = None
        for i, candidate in enumerate(candidate_synsets):
            if candidate.get("synset_id") == assigned_synset:
                assigned_index = i
                break

        if assigned_index is None:
            # Assigned synset not in candidates - shouldn't happen
            return EvaluationResult(
                lemma=lemma,
                sentence_text=sentence_text,
                assigned_synset=assigned_synset,
                jury_votes=["error"],
                jury_verdict="uncertain",
                confidence=0.0,
            )

        # Build the prompt
        prompt = build_evaluation_prompt(
            sentence=sentence_text,
            lemma=lemma,
            pos=pos,
            candidates=candidate_synsets,
        )

        # Get votes from each provider
        votes = []

        for provider in self.providers:
            try:
                response = provider.complete_json(prompt)
                choice = response.get("choice", "uncertain")

                # Convert letter choice to index
                chosen_index = get_candidate_index(choice, candidate_synsets)

                if chosen_index is not None:
                    votes.append("correct" if chosen_index == assigned_index else "incorrect")
                else:
                    votes.append("uncertain")

            except Exception:
                votes.append("error")

        # Aggregate jury votes - confidence is now based on agreement, not LLM self-report
        verdict, confidence = self._aggregate_votes(votes)

        return EvaluationResult(
            lemma=lemma,
            sentence_text=sentence_text,
            assigned_synset=assigned_synset,
            jury_votes=votes,
            jury_verdict=verdict,
            confidence=confidence,
        )

    def _aggregate_votes(self, votes: list[str]) -> tuple[str, float]:
        """Aggregate jury votes into final verdict.

        Confidence is computed as agreement ratio (not LLM self-reported confidence).
        - 3/3 agree = 1.0 confidence
        - 2/3 agree = 0.67 confidence
        - No majority = uncertain with 0.0 confidence

        Args:
            votes: List of votes from providers ("correct", "incorrect", "uncertain", "error").

        Returns:
            Tuple of (verdict, confidence).
            Confidence = proportion of providers agreeing on verdict.
        """
        # Filter out errors
        valid_votes = [v for v in votes if v not in ["error"]]

        if not valid_votes:
            return "uncertain", 0.0

        total_valid = len(valid_votes)

        # Count votes
        correct_count = sum(1 for v in valid_votes if v == "correct")
        incorrect_count = sum(1 for v in valid_votes if v == "incorrect")
        uncertain_count = sum(1 for v in valid_votes if v == "uncertain")

        # Find majority vote
        max_count = max(correct_count, incorrect_count, uncertain_count)

        # Need strict majority for verdict (more than half)
        if correct_count == max_count and correct_count > total_valid / 2:
            return "correct", correct_count / total_valid
        elif incorrect_count == max_count and incorrect_count > total_valid / 2:
            return "incorrect", incorrect_count / total_valid
        elif uncertain_count == max_count and uncertain_count > total_valid / 2:
            return "uncertain", uncertain_count / total_valid
        else:
            # No majority - mark as uncertain with low confidence
            return "uncertain", max_count / total_valid


def compute_metrics(results: list[EvaluationResult]) -> dict:
    """Compute evaluation metrics from results.

    Metrics:
    - accuracy_strict: correct / (correct + incorrect). Excludes uncertain.
    - coverage: (correct + incorrect) / total. How many samples were classified.
    - avg_confidence: Average confidence across all samples.

    Args:
        results: List of EvaluationResult objects.

    Returns:
        Dictionary with computed metrics.
    """
    if not results:
        return {
            "total_samples": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "uncertain_count": 0,
            "accuracy_strict": 0.0,
            "coverage": 0.0,
            "avg_confidence": 0.0,
        }

    total = len(results)
    correct = sum(1 for r in results if r.jury_verdict == "correct")
    incorrect = sum(1 for r in results if r.jury_verdict == "incorrect")
    uncertain = sum(1 for r in results if r.jury_verdict == "uncertain")

    # accuracy_strict excludes uncertain samples
    classified = correct + incorrect
    accuracy_strict = correct / classified if classified > 0 else 0.0

    # coverage = how many samples were confidently classified
    coverage = classified / total if total > 0 else 0.0

    # average confidence
    avg_confidence = sum(r.confidence for r in results) / total if total > 0 else 0.0

    return {
        "total_samples": total,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "uncertain_count": uncertain,
        "accuracy_strict": accuracy_strict,
        "coverage": coverage,
        "avg_confidence": avg_confidence,
    }
