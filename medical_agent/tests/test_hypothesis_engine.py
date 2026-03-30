"""
Unit tests for state/hypothesis_register.py and agents/hypothesis_engine.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medical_agent.config import PipelineConfig
from medical_agent.state.clinical_state import ClinicalState
from medical_agent.state.hypothesis_register import Hypothesis, HypothesisRegister


# ── HypothesisRegister tests ──────────────────────────────────────────────────

class TestHypothesisRegister:
    def _make_register(self, diseases_and_confidences: list) -> HypothesisRegister:
        hyps = [
            Hypothesis(
                disease=d,
                confidence=c,
                supporting_evidence=[],
                contradicting_evidence=[],
                kg_matched=True,
                first_mentioned_turn=1,
                last_updated_turn=1,
            )
            for d, c in diseases_and_confidences
        ]
        return HypothesisRegister(hypotheses=hyps)

    def test_top_returns_highest_confidence(self):
        reg = self._make_register([("SLE", 0.3), ("RA", 0.5), ("Gout", 0.2)])
        top = reg.top()
        assert top.disease == "RA"
        assert top.confidence == 0.5

    def test_ranked_descending(self):
        reg = self._make_register([("A", 0.1), ("B", 0.4), ("C", 0.3)])
        ranked = reg.ranked()
        confidences = [h.confidence for h in ranked]
        assert confidences == sorted(confidences, reverse=True)

    def test_top_k_names(self):
        reg = self._make_register([("A", 0.5), ("B", 0.3), ("C", 0.15), ("D", 0.05)])
        top3 = reg.top_k_names(3)
        assert top3 == ["A", "B", "C"]

    def test_stagnation_detection_true(self):
        reg = self._make_register([("A", 0.5), ("B", 0.3), ("C", 0.2)])
        # Record same top-3 for 3 consecutive turns
        for t in range(3):
            reg.record_snapshot(t)
        assert reg.is_stagnant(window=3) is True

    def test_stagnation_detection_false_when_differential_changes(self):
        reg = self._make_register([("A", 0.5), ("B", 0.3), ("C", 0.2)])
        reg.record_snapshot(0)
        # Change differential
        reg.hypotheses[0].disease = "X"
        reg.record_snapshot(1)
        reg.record_snapshot(2)
        assert reg.is_stagnant(window=3) is False

    def test_stagnation_false_when_not_enough_history(self):
        reg = self._make_register([("A", 0.5), ("B", 0.3)])
        reg.record_snapshot(0)
        assert reg.is_stagnant(window=3) is False

    def test_normalize_confidences_sums_to_one(self):
        reg = self._make_register([("A", 0.6), ("B", 0.6), ("C", 0.6)])
        reg.normalize_confidences()
        total = sum(h.confidence for h in reg.hypotheses)
        assert abs(total - 1.0) < 1e-9

    def test_normalize_leaves_valid_untouched(self):
        reg = self._make_register([("A", 0.5), ("B", 0.3), ("C", 0.2)])
        reg.normalize_confidences()
        total = sum(h.confidence for h in reg.hypotheses)
        assert abs(total - 1.0) < 1e-9

    def test_top_confidence_returns_float(self):
        reg = self._make_register([("A", 0.4)])
        assert reg.top_confidence() == 0.4

    def test_top_confidence_empty_register(self):
        reg = HypothesisRegister()
        assert reg.top_confidence() == 0.0

    def test_get_existing_hypothesis(self):
        reg = self._make_register([("SLE", 0.6)])
        h = reg.get("SLE")
        assert h is not None
        assert h.confidence == 0.6

    def test_get_case_insensitive(self):
        reg = self._make_register([("Systemic Lupus Erythematosus", 0.6)])
        h = reg.get("systemic lupus erythematosus")
        assert h is not None

    def test_get_nonexistent_returns_none(self):
        reg = self._make_register([("SLE", 0.6)])
        h = reg.get("Nonexistent Disease")
        assert h is None

    def test_to_dict_list_format(self):
        reg = self._make_register([("SLE", 0.6)])
        d_list = reg.to_dict_list()
        assert isinstance(d_list, list)
        assert len(d_list) == 1
        assert "disease" in d_list[0]
        assert "confidence" in d_list[0]
        assert "supporting_evidence" in d_list[0]

    def test_snapshot_copy_is_independent(self):
        reg = self._make_register([("SLE", 0.6)])
        copy = reg.snapshot_copy()
        copy.hypotheses[0].confidence = 0.9
        assert reg.hypotheses[0].confidence == 0.6


# ── HypothesisEngine tests ────────────────────────────────────────────────────

class TestHypothesisEngine:
    def _make_engine(self):
        from medical_agent.agents.hypothesis_engine import HypothesisEngine
        from medical_agent.knowledge.knowledge_graph import KnowledgeGraph
        config = PipelineConfig(log_format="text", doctor_model="test-model")
        kg = MagicMock(spec=KnowledgeGraph)
        kg.is_loaded.return_value = True
        kg.find_matching_diseases.return_value = [("SLE", 2), ("RA", 1)]
        kg.triples_as_text.return_value = "SLE → malar rash\nSLE → fever"
        return HypothesisEngine(config, kg=kg), config

    def test_update_applies_llm_output(self):
        """Hypothesis engine should apply valid LLM JSON output."""
        engine, _ = self._make_engine()
        state = ClinicalState(turn_number=2, phase="investigation")
        register = HypothesisRegister()

        mock_response = """{
  "hypotheses": [
    {"disease": "SLE", "confidence": 0.55, "supporting": ["malar rash"], "contradicting": [], "kg_matched": true},
    {"disease": "RA", "confidence": 0.25, "supporting": ["arthritis"], "contradicting": [], "kg_matched": true},
    {"disease": "Gout", "confidence": 0.10, "supporting": [], "contradicting": [], "kg_matched": false},
    {"disease": "Psoriatic Arthritis", "confidence": 0.07, "supporting": [], "contradicting": [], "kg_matched": false},
    {"disease": "Sarcoidosis", "confidence": 0.03, "supporting": [], "contradicting": [], "kg_matched": false}
  ],
  "reasoning": "SLE most likely given malar rash and positive ANA"
}"""

        with patch.object(engine, 'query', return_value=mock_response):
            register = engine.update(state, register, new_findings="malar rash noted")

        assert len(register.hypotheses) == 5
        top = register.top()
        assert top.disease == "SLE"
        assert abs(top.confidence - 0.55) < 0.01

    def test_graceful_degradation_on_parse_failure(self):
        """If LLM returns invalid JSON, register should be preserved."""
        engine, _ = self._make_engine()
        state = ClinicalState(turn_number=3)
        from medical_agent.state.hypothesis_register import Hypothesis
        existing = Hypothesis("SLE", 0.4, [], [], True, 1, 1)
        register = HypothesisRegister(hypotheses=[existing])

        with patch.object(engine, 'query', return_value="This is not JSON at all"):
            register = engine.update(state, register)

        # Should still have the original hypothesis
        assert any(h.disease == "SLE" for h in register.hypotheses)

    def test_confidences_normalized(self):
        """Confidences returned by engine must sum to <= 1.0."""
        engine, _ = self._make_engine()
        state = ClinicalState(turn_number=1)
        register = HypothesisRegister()

        # Over-sum confidences from LLM
        mock_response = """{
  "hypotheses": [
    {"disease": "A", "confidence": 0.9, "supporting": [], "contradicting": [], "kg_matched": false},
    {"disease": "B", "confidence": 0.8, "supporting": [], "contradicting": [], "kg_matched": false}
  ],
  "reasoning": ""
}"""

        with patch.object(engine, 'query', return_value=mock_response):
            register = engine.update(state, register)

        total = sum(h.confidence for h in register.hypotheses)
        assert total <= 1.0 + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
