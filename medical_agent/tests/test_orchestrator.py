"""
Unit tests for orchestrator.py — phase transitions and turn lifecycle.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medical_agent.config import PipelineConfig
from medical_agent.orchestrator import Orchestrator
from medical_agent.state.clinical_state import ClinicalState
from medical_agent.state.hypothesis_register import Hypothesis, HypothesisRegister


def make_config(**kwargs) -> PipelineConfig:
    defaults = dict(
        max_turns=10,
        intake_turns=2,
        min_turns_before_diagnosis=3,
        confidence_threshold=0.75,
        stagnation_window=3,
        log_format="text",
        log_level="WARNING",
        enable_kg_hypothesis=False,
        enable_info_gain=False,
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def make_orchestrator(config=None, **kwargs) -> Orchestrator:
    if config is None:
        config = make_config(**kwargs)
    orch = Orchestrator(config, case_id="test_001", dataset="MedQA")
    return orch


# ── Phase transition tests ────────────────────────────────────────────────────

class TestPhaseTransitions:
    def test_starts_in_intake(self):
        orch = make_orchestrator()
        assert orch.state.phase == "intake"

    def test_intake_phase_for_first_two_turns(self):
        orch = make_orchestrator(intake_turns=2)
        with patch.object(orch.intake_parser, 'parse_turn', return_value=orch.state), \
             patch.object(orch.hypothesis_engine, 'initialize', return_value=HypothesisRegister()), \
             patch.object(orch.hypothesis_engine, 'update', return_value=HypothesisRegister()), \
             patch.object(orch.verifier, 'verify', return_value=MagicMock(flagged=False, recommendation="continue", to_dict=lambda: {})), \
             patch.object(orch.inquiry_strategist, 'select_next_action', return_value=("question", "test question", 0.0)), \
             patch.object(orch.doctor_dialogue, 'generate', return_value="Question for patient?"):

            result = orch.step("Patient response 1")
            assert result.phase == "intake"

            result = orch.step("Patient response 2")
            assert result.phase == "intake"

    def test_transitions_to_investigation_after_intake(self):
        orch = make_orchestrator(intake_turns=1)
        with patch.object(orch.intake_parser, 'parse_turn', return_value=orch.state), \
             patch.object(orch.hypothesis_engine, 'initialize', return_value=HypothesisRegister()), \
             patch.object(orch.hypothesis_engine, 'update', return_value=HypothesisRegister()), \
             patch.object(orch.verifier, 'verify', return_value=MagicMock(flagged=False, recommendation="continue", to_dict=lambda: {})), \
             patch.object(orch.inquiry_strategist, 'select_next_action', return_value=("question", "test question", 0.0)), \
             patch.object(orch.doctor_dialogue, 'generate', return_value="Question?"):

            orch.step("Turn 1")  # intake
            result = orch.step("Turn 2")  # should be investigation
            assert result.phase in ("investigation", "convergence")

    def test_convergence_when_high_confidence(self):
        orch = make_orchestrator(intake_turns=1)
        # Set high confidence hypothesis
        hyp = Hypothesis("SLE", 0.7, [], [], True, 1, 1)
        register = HypothesisRegister(hypotheses=[hyp])

        with patch.object(orch.intake_parser, 'parse_turn', return_value=orch.state), \
             patch.object(orch.hypothesis_engine, 'initialize', return_value=register), \
             patch.object(orch.hypothesis_engine, 'update', return_value=register), \
             patch.object(orch.verifier, 'verify', return_value=MagicMock(flagged=False, recommendation="continue", to_dict=lambda: {})), \
             patch.object(orch.inquiry_strategist, 'select_next_action', return_value=("question", "test question", 0.0)), \
             patch.object(orch.doctor_dialogue, 'generate', return_value="Question?"):

            orch.step("Turn 1")
            result = orch.step("Turn 2")
            assert result.phase == "convergence"


# ── Termination condition tests ───────────────────────────────────────────────

class TestTerminationConditions:
    def test_confidence_threshold_triggers_diagnosis(self):
        """When confidence >= threshold after min_turns, diagnosis should be issued."""
        orch = make_orchestrator(
            min_turns_before_diagnosis=1,
            confidence_threshold=0.75,
        )
        hyp = Hypothesis("Myasthenia Gravis", 0.8, [], [], True, 1, 1)
        register = HypothesisRegister(hypotheses=[hyp])

        mock_diagnosis = "DIAGNOSIS READY: Myasthenia Gravis"

        with patch.object(orch.state_tracker, 'update', return_value=orch.state), \
             patch.object(orch.hypothesis_engine, 'initialize', return_value=register), \
             patch.object(orch.hypothesis_engine, 'update', return_value=register), \
             patch.object(orch.verifier, 'verify', return_value=MagicMock(flagged=False, recommendation="continue", to_dict=lambda: {})), \
             patch.object(orch.diagnosis_resolver, 'resolve', return_value=mock_diagnosis):

            result = orch.step("Patient response")

        assert result.is_diagnosis
        assert "DIAGNOSIS READY" in result.utterance

    def test_turn_limit_triggers_diagnosis(self):
        """At max_turns, a diagnosis should always be issued."""
        config = make_config(max_turns=3, min_turns_before_diagnosis=1)
        orch = make_orchestrator(config=config)

        mock_diagnosis = "DIAGNOSIS READY: Unknown"

        with patch.object(orch.state_tracker, 'update', return_value=orch.state), \
             patch.object(orch.hypothesis_engine, 'initialize', return_value=HypothesisRegister()), \
             patch.object(orch.hypothesis_engine, 'update', return_value=HypothesisRegister()), \
             patch.object(orch.verifier, 'verify', return_value=MagicMock(flagged=False, recommendation="continue", to_dict=lambda: {})), \
             patch.object(orch.inquiry_strategist, 'select_next_action', return_value=("question", "?", 0.0)), \
             patch.object(orch.doctor_dialogue, 'generate', return_value="Question?"), \
             patch.object(orch.diagnosis_resolver, 'resolve', return_value=mock_diagnosis):

            # Run to max turns
            for _ in range(3):
                result = orch.step("Patient response")

        assert result.is_diagnosis

    def test_no_early_diagnosis(self):
        """Diagnosis should NOT be issued before min_turns_before_diagnosis."""
        config = make_config(
            min_turns_before_diagnosis=5,
            confidence_threshold=0.75,
        )
        orch = make_orchestrator(config=config)
        hyp = Hypothesis("SLE", 0.9, [], [], True, 1, 1)
        register = HypothesisRegister(hypotheses=[hyp])

        with patch.object(orch.state_tracker, 'update', return_value=orch.state), \
             patch.object(orch.hypothesis_engine, 'initialize', return_value=register), \
             patch.object(orch.hypothesis_engine, 'update', return_value=register), \
             patch.object(orch.verifier, 'verify', return_value=MagicMock(flagged=False, recommendation="continue", to_dict=lambda: {})), \
             patch.object(orch.inquiry_strategist, 'select_next_action', return_value=("question", "?", 0.0)), \
             patch.object(orch.doctor_dialogue, 'generate', return_value="Question?"):

            result = orch.step("Patient response turn 1")

        assert not result.is_diagnosis

    def test_stagnation_triggers_diagnosis(self):
        """Stagnant differential (unchanged for stagnation_window turns) should trigger diagnosis."""
        config = make_config(
            min_turns_before_diagnosis=1,
            stagnation_window=3,
            confidence_threshold=0.99,  # Disable confidence trigger
        )
        orch = make_orchestrator(config=config)
        hyp = Hypothesis("Gout", 0.5, [], [], True, 1, 1)
        register = HypothesisRegister(hypotheses=[hyp])

        # Pre-populate history with same top-3 to simulate stagnation
        for i in range(3):
            register.record_snapshot(i)

        mock_diagnosis = "DIAGNOSIS READY: Gout"

        with patch.object(orch.state_tracker, 'update', return_value=orch.state), \
             patch.object(orch.hypothesis_engine, 'initialize', return_value=register), \
             patch.object(orch.hypothesis_engine, 'update', return_value=register), \
             patch.object(orch.verifier, 'verify', return_value=MagicMock(flagged=False, recommendation="continue", to_dict=lambda: {})), \
             patch.object(orch.diagnosis_resolver, 'resolve', return_value=mock_diagnosis):

            result = orch.step("No new information")

        assert result.is_diagnosis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
