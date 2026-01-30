"""Tests for Smart Fallback logic."""

import pytest

from eng_words.aggregation.fallback import get_fallback_synset, get_synsets_by_frequency


class TestGetSynsetsByFrequency:
    """Tests for get_synsets_by_frequency function."""

    def test_returns_synsets_for_valid_word(self):
        """Test getting synsets for a common word."""
        synsets = get_synsets_by_frequency("run", "v")
        assert len(synsets) > 0
        assert all(s.endswith(".v.") or ".v." in s for s in synsets)

    def test_returns_synsets_sorted_by_frequency(self):
        """Test that synsets are returned in frequency order."""
        synsets = get_synsets_by_frequency("run", "v")
        # First should be run.v.01 (most common)
        assert synsets[0] == "run.v.01"
        # Should have multiple synsets
        assert len(synsets) > 5

    def test_returns_empty_for_unknown_word(self):
        """Test returns empty list for unknown word."""
        synsets = get_synsets_by_frequency("xyznonexistent", "n")
        assert synsets == []

    def test_filters_by_pos(self):
        """Test that only synsets with matching POS are returned."""
        noun_synsets = get_synsets_by_frequency("run", "n")
        verb_synsets = get_synsets_by_frequency("run", "v")
        
        # All noun synsets should have .n.
        for s in noun_synsets:
            assert ".n." in s
        
        # All verb synsets should have .v.
        for s in verb_synsets:
            assert ".v." in s


class TestGetFallbackSynset:
    """Tests for get_fallback_synset function."""

    def test_returns_next_available_synset(self):
        """Test getting next synset when first is taken."""
        # about.r.01 exists, should return about.r.02
        result = get_fallback_synset(
            lemma="about",
            pos="r",
            failed_synset="about.r.05",
            existing_synsets={"about.r.01"},
        )
        assert result == "about.r.02"

    def test_skips_failed_synset(self):
        """Test that failed synset is skipped."""
        result = get_fallback_synset(
            lemma="run",
            pos="v",
            failed_synset="run.v.01",
            existing_synsets=set(),
        )
        # Should return next available, not run.v.01
        assert result is not None
        assert result != "run.v.01"
        assert result.startswith("run.v.")

    def test_skips_existing_synsets(self):
        """Test that existing synsets are skipped."""
        # Get actual synsets for run.v
        all_synsets = get_synsets_by_frequency("run", "v")
        # Use first 3 as existing
        existing = set(all_synsets[:3])
        failed = all_synsets[4] if len(all_synsets) > 4 else "run.v.99"
        
        result = get_fallback_synset(
            lemma="run",
            pos="v",
            failed_synset=failed,
            existing_synsets=existing,
        )
        # Should return a synset not in existing
        assert result is not None
        assert result not in existing
        assert result != failed

    def test_returns_none_when_all_exhausted(self):
        """Test returns None when all synsets are used."""
        # Get all synsets for a word with few synsets
        synsets = get_synsets_by_frequency("hello", "n")
        
        if synsets:
            # Mark all as existing
            result = get_fallback_synset(
                lemma="hello",
                pos="n",
                failed_synset=synsets[0],
                existing_synsets=set(synsets),
            )
            assert result is None

    def test_returns_first_available_when_none_exist(self):
        """Test returns first synset when nothing exists yet."""
        result = get_fallback_synset(
            lemma="run",
            pos="v",
            failed_synset="run.v.99",  # Non-existent
            existing_synsets=set(),
        )
        # Should return run.v.01 (first)
        assert result == "run.v.01"

    def test_handles_unknown_word(self):
        """Test gracefully handles unknown word."""
        result = get_fallback_synset(
            lemma="xyznonexistent",
            pos="n",
            failed_synset="xyznonexistent.n.01",
            existing_synsets=set(),
        )
        assert result is None

    def test_real_case_about(self):
        """Test real fallback case for 'about'."""
        # Scenario: WSD picked about.r.05 (rare), we have about.r.01 and about.r.03
        result = get_fallback_synset(
            lemma="about",
            pos="r",
            failed_synset="about.r.05",
            existing_synsets={"about.r.01", "about.r.03"},
        )
        # Should return about.r.02 (next available)
        assert result == "about.r.02"

    def test_real_case_add(self):
        """Test real fallback case for 'add'."""
        # Scenario: WSD picked add.v.04 (math), we have nothing
        result = get_fallback_synset(
            lemma="add",
            pos="v",
            failed_synset="add.v.04",
            existing_synsets=set(),
        )
        # Should return add.v.01 (most common)
        assert result == "add.v.01"

