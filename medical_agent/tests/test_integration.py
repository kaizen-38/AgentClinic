"""
End-to-end integration test with a mock patient.

Tests the full pipeline against a simulated myasthenia gravis case:
- Mock patient agent returns scripted responses
- Mock measurement agent returns scripted test results
- Pipeline should eventually diagnose Myasthenia Gravis
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medical_agent.config import PipelineConfig
from medical_agent.interface.agentclinic_adapter import SDRPDoctorAgent
from medical_agent.state.hypothesis_register import Hypothesis, HypothesisRegister


# ── Mock scenario ─────────────────────────────────────────────────────────────

MYASTHENIA_GRAVIS_DIALOGUE = [
    "I'm a 35-year-old woman. I've been having double vision and drooping of my eyelids for the past month. It gets worse in the evening.",
    "The weakness affects my eyelids and when I chew food. I get very tired when I try to do things repeatedly.",
    "I haven't noticed any sensory changes. No pain. I did have a bout of weakness in my arms last week.",
    "RESULTS: AChR antibody: POSITIVE (elevated at 5.2 nmol/L). Normal range: <0.4 nmol/L.",
    "The symptoms are definitely worse after activity and better in the morning after rest.",
    "Yes, I've noticed my voice becomes nasal when I talk for a long time.",
]

MYASTHENIA_GRAVIS_CORRECT = "Myasthenia Gravis"


def make_test_config() -> PipelineConfig:
    return PipelineConfig(
        doctor_model="test-model",
        support_model="test-model",
        model_backend="voyager",
        max_turns=8,
        intake_turns=2,
        min_turns_before_diagnosis=3,
        confidence_threshold=0.75,
        stagnation_window=5,
        log_format="text",
        log_level="WARNING",
        enable_kg_hypothesis=False,  # Don't need KG for mock test
        enable_info_gain=False,
        enable_osce_state=False,  # Disable LLM parsing to avoid real calls
        enable_verifier=False,
    )


# ── Integration test ──────────────────────────────────────────────────────────

class TestEndToEndIntegration:
    """
    Full pipeline test with all LLM calls mocked.
    Verifies that the pipeline:
    1. Runs through multiple turns
    2. Updates state and hypotheses
    3. Eventually issues DIAGNOSIS READY
    4. Produces a trajectory
    """

    def _setup_mocks(self, doctor_agent: SDRPDoctorAgent, target_disease: str):
        """Configure all LLM calls to return deterministic responses."""
        orch = doctor_agent.orchestrator

        # StateTracker: do nothing (OSCE disabled)
        # HypothesisEngine: gradually increase confidence
        turn_counter = {"n": 0}

        def mock_hyp_update(state, register, new_findings=""):
            n = turn_counter["n"]
            turn_counter["n"] += 1
            confidence = min(0.2 + n * 0.15, 0.85)
            hyp = Hypothesis(
                disease=target_disease,
                confidence=confidence,
                supporting_evidence=["ptosis", "diplopia", "fatigable weakness"],
                contradicting_evidence=[],
                kg_matched=False,
                first_mentioned_turn=1,
                last_updated_turn=state.turn_number,
            )
            register.hypotheses = [hyp]
            register.record_snapshot(state.turn_number)
            return register

        def mock_hyp_initialize(state, register):
            return mock_hyp_update(state, register)

        # InquiryStrategist: alternate between question and test
        def mock_select(state, register):
            if state.turn_number % 3 == 0:
                return ("test", "AChR antibody", 0.3)
            return ("question", "Have you noticed fatigable weakness?", 0.25)

        # DoctorDialogue: passthrough
        def mock_dialogue(state, register, action_type, action_content, **kwargs):
            if action_type == "test":
                return f"REQUEST TEST: {action_content.replace(' ', '_')}"
            return action_content

        # DiagnosisResolver
        def mock_resolve(state, register, **kwargs):
            return f"DIAGNOSIS READY: {target_disease}"

        # Verifier
        mock_verifier = MagicMock()
        mock_verifier.return_value = MagicMock(
            is_consistent=True,
            flagged=False,
            recommendation="continue",
            to_dict=lambda: {},
        )

        orch.hypothesis_engine.initialize = mock_hyp_initialize
        orch.hypothesis_engine.update = mock_hyp_update
        orch.inquiry_strategist.select_next_action = mock_select
        orch.doctor_dialogue.generate = mock_dialogue
        orch.diagnosis_resolver.resolve = mock_resolve
        orch.verifier.verify = mock_verifier

    def test_full_pipeline_issues_diagnosis(self, tmp_path):
        """Pipeline must issue DIAGNOSIS READY within max_turns."""
        config = make_test_config()
        config.trajectory_output_dir = str(tmp_path)
        config.max_turns = 8

        doctor_agent = SDRPDoctorAgent(
            config=config,
            scenario=None,
            case_id="test_mg_001",
            dataset="MedQA",
            correct_diagnosis=MYASTHENIA_GRAVIS_CORRECT,
        )
        self._setup_mocks(doctor_agent, MYASTHENIA_GRAVIS_CORRECT)

        diagnosis_issued = False
        final_utterance = ""
        for patient_text in MYASTHENIA_GRAVIS_DIALOGUE:
            utterance = doctor_agent.inference(patient_text)
            if "DIAGNOSIS READY" in utterance:
                diagnosis_issued = True
                final_utterance = utterance
                break

        assert diagnosis_issued, "Pipeline failed to issue DIAGNOSIS READY"
        assert MYASTHENIA_GRAVIS_CORRECT in final_utterance

    def test_trajectory_saved(self, tmp_path):
        """Pipeline should save a trajectory file after finalize."""
        config = make_test_config()
        config.trajectory_output_dir = str(tmp_path)
        config.max_turns = 6

        doctor_agent = SDRPDoctorAgent(
            config=config,
            scenario=None,
            case_id="test_mg_002",
            dataset="MedQA",
            correct_diagnosis=MYASTHENIA_GRAVIS_CORRECT,
        )
        self._setup_mocks(doctor_agent, MYASTHENIA_GRAVIS_CORRECT)

        # Run a few turns
        for patient_text in MYASTHENIA_GRAVIS_DIALOGUE[:4]:
            doctor_agent.inference(patient_text)

        traj_path = doctor_agent.finalize_case(is_correct=True)
        assert Path(traj_path).exists(), f"Trajectory not saved at {traj_path}"

    def test_agent_hist_accumulates(self):
        """agent_hist should grow with each turn."""
        config = make_test_config()
        config.max_turns = 5

        doctor_agent = SDRPDoctorAgent(
            config=config,
            scenario=None,
            case_id="test_mg_003",
            dataset="MedQA",
        )
        self._setup_mocks(doctor_agent, MYASTHENIA_GRAVIS_CORRECT)

        assert doctor_agent.agent_hist == ""
        doctor_agent.inference("First patient message")
        assert len(doctor_agent.agent_hist) > 0

        prev_len = len(doctor_agent.agent_hist)
        doctor_agent.inference("Second patient message")
        assert len(doctor_agent.agent_hist) > prev_len

    def test_inference_count_tracked(self):
        """infs counter should increment each turn."""
        config = make_test_config()
        config.max_turns = 5

        doctor_agent = SDRPDoctorAgent(
            config=config,
            scenario=None,
            case_id="test_mg_004",
            dataset="MedQA",
        )
        self._setup_mocks(doctor_agent, MYASTHENIA_GRAVIS_CORRECT)

        assert doctor_agent.infs == 0
        for i, text in enumerate(MYASTHENIA_GRAVIS_DIALOGUE[:3]):
            doctor_agent.inference(text)
            if "DIAGNOSIS READY" not in doctor_agent.agent_hist:
                pass
        assert doctor_agent.infs >= 1

    def test_adapter_system_prompt(self):
        """system_prompt() should return a non-empty string."""
        config = make_test_config()
        agent = SDRPDoctorAgent(config=config)
        prompt = agent.system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 20
        assert "DIAGNOSIS READY" in prompt or "diagnosis" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
