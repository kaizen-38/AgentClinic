"""
Unit tests for knowledge/info_gain.py

Tests the Shannon entropy and information gain computation
against known analytical values.
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medical_agent.knowledge.info_gain import InformationGainCalculator
from medical_agent.knowledge.knowledge_graph import KnowledgeGraph


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_mock_kg(disease_symptoms: dict) -> KnowledgeGraph:
    """Build a KnowledgeGraph mock backed by a simple dict."""
    kg = MagicMock(spec=KnowledgeGraph)
    kg.get_symptoms.side_effect = lambda d: disease_symptoms.get(d, [])
    kg.get_tests.side_effect = lambda d: []
    kg.is_loaded.return_value = True
    kg.all_diseases.return_value = list(disease_symptoms.keys())
    return kg


# ── entropy tests ─────────────────────────────────────────────────────────────

class TestEntropy:
    def test_uniform_two(self):
        """Uniform distribution over 2 diseases: H = 1.0 bit."""
        calc = InformationGainCalculator(MagicMock())
        h = calc.entropy([0.5, 0.5])
        assert abs(h - 1.0) < 1e-9

    def test_uniform_four(self):
        """Uniform distribution over 4 diseases: H = 2.0 bits."""
        calc = InformationGainCalculator(MagicMock())
        h = calc.entropy([0.25, 0.25, 0.25, 0.25])
        assert abs(h - 2.0) < 1e-9

    def test_certain(self):
        """Certain distribution: H = 0."""
        calc = InformationGainCalculator(MagicMock())
        h = calc.entropy([1.0, 0.0, 0.0])
        assert abs(h) < 1e-9

    def test_zeros_ignored(self):
        """Zero-probability entries do not contribute to entropy."""
        calc = InformationGainCalculator(MagicMock())
        h_with_zeros = calc.entropy([0.5, 0.5, 0.0, 0.0])
        h_without = calc.entropy([0.5, 0.5])
        assert abs(h_with_zeros - h_without) < 1e-9

    def test_single_disease(self):
        """Single disease with p=1: entropy is 0."""
        calc = InformationGainCalculator(MagicMock())
        assert calc.entropy([1.0]) == 0.0


# ── p_symptom_given_disease tests ─────────────────────────────────────────────

class TestPSymptomGivenDisease:
    def setup_method(self):
        symptoms = {
            "Disease A": ["fever", "cough", "fatigue"],  # 3 symptoms
            "Disease B": ["rash", "fever"],              # 2 symptoms
        }
        self.kg = make_mock_kg(symptoms)
        self.calc = InformationGainCalculator(self.kg)

    def test_present_symptom(self):
        p = self.calc.p_symptom_given_disease("fever", "Disease A")
        assert abs(p - 1/3) < 1e-9

    def test_absent_symptom(self):
        p = self.calc.p_symptom_given_disease("rash", "Disease A")
        assert p == 0.0

    def test_disease_with_two_symptoms(self):
        p = self.calc.p_symptom_given_disease("rash", "Disease B")
        assert abs(p - 0.5) < 1e-9

    def test_unknown_disease(self):
        p = self.calc.p_symptom_given_disease("fever", "Unknown Disease")
        assert p == 0.0


# ── information_gain tests ────────────────────────────────────────────────────

class TestInformationGain:
    def setup_method(self):
        """
        Minimal two-disease scenario:
          Disease A: symptoms = [fever, cough]    → P(s|A) = 0.5 each
          Disease B: symptoms = [rash, fatigue]   → P(s|B) = 0.5 each
          (disjoint symptom sets)
        """
        symptoms = {
            "Disease A": ["fever", "cough"],
            "Disease B": ["rash", "fatigue"],
        }
        self.kg = make_mock_kg(symptoms)
        self.calc = InformationGainCalculator(self.kg)

    def test_perfectly_discriminating_symptom(self):
        """
        When P(s|A)=1.0 and P(s|B)=0 with uniform prior [0.5, 0.5],
        observing the symptom resolves all uncertainty → IG = 1.0 bit.

        This requires single-symptom diseases so that P(s|Di) = 1/1 = 1.0.
        """
        # Single-symptom diseases: P(fever|A)=1.0, P(fever|B)=0
        kg = make_mock_kg({"Disease A": ["fever"], "Disease B": ["rash"]})
        calc = InformationGainCalculator(kg)
        differential = {"Disease A": 0.5, "Disease B": 0.5}
        ig = calc.information_gain("fever", differential)
        assert abs(ig - 1.0) < 1e-9, f"Expected IG = 1.0, got {ig}"

    def test_partial_discriminating_symptom(self):
        """
        With P(s|A)=0.5 (one of two symptoms) and P(s|B)=0,
        IG is positive but less than 1.0.
        Analytical value ≈ 0.311 bits.
        """
        differential = {"Disease A": 0.5, "Disease B": 0.5}
        ig = self.calc.information_gain("fever", differential)
        assert 0.2 < ig < 0.5, f"Expected IG ≈ 0.31, got {ig}"

    def test_non_discriminating_symptom(self):
        """
        'shared_symptom' appears equally in both diseases.
        IG should be near 0.
        """
        symptoms = {
            "Disease A": ["shared_symptom", "cough"],
            "Disease B": ["shared_symptom", "rash"],
        }
        kg = make_mock_kg(symptoms)
        calc = InformationGainCalculator(kg)
        differential = {"Disease A": 0.5, "Disease B": 0.5}
        ig = calc.information_gain("shared_symptom", differential)
        # P(s|A) = P(s|B) = 0.5 → posterior equals prior → IG = 0
        assert ig < 0.1, f"Expected near-zero IG, got {ig}"

    def test_ig_non_negative(self):
        """Information gain is always non-negative."""
        differential = {"Disease A": 0.5, "Disease B": 0.5}
        for symptom in ["fever", "rash", "unknown_symptom"]:
            ig = self.calc.information_gain(symptom, differential)
            assert ig >= 0.0

    def test_empty_differential(self):
        """Empty differential returns 0."""
        ig = self.calc.information_gain("fever", {})
        assert ig == 0.0

    def test_certain_differential(self):
        """If one disease has probability 1.0, IG is 0 (no uncertainty to resolve)."""
        differential = {"Disease A": 1.0, "Disease B": 0.0}
        ig = self.calc.information_gain("fever", differential)
        assert ig < 1e-9


# ── rank_symptoms tests ───────────────────────────────────────────────────────

class TestRankSymptoms:
    def setup_method(self):
        symptoms = {
            "Disease A": ["unique_a", "common"],
            "Disease B": ["unique_b", "common"],
        }
        self.kg = make_mock_kg(symptoms)
        self.calc = InformationGainCalculator(self.kg)
        self.differential = {"Disease A": 0.5, "Disease B": 0.5}

    def test_discriminating_ranked_higher(self):
        """Unique symptoms ranked above shared symptoms."""
        ranked = self.calc.rank_symptoms(
            ["unique_a", "unique_b", "common"], self.differential
        )
        assert len(ranked) == 3
        names = [s for s, _ in ranked]
        # Both unique symptoms should rank above the shared one
        common_rank = names.index("common")
        assert common_rank > 0

    def test_already_asked_excluded(self):
        """Already-asked symptoms are excluded from ranking."""
        ranked = self.calc.rank_symptoms(
            ["unique_a", "common"], self.differential,
            already_asked=["unique_a"]
        )
        names = [s for s, _ in ranked]
        assert "unique_a" not in names

    def test_min_ig_threshold(self):
        """Symptoms below min_ig threshold are filtered out."""
        ranked = self.calc.rank_symptoms(
            ["common"], self.differential, min_ig=0.5
        )
        # 'common' has low IG; should be filtered
        assert len(ranked) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
