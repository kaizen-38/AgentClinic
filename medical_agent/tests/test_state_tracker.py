"""
Unit tests for state/clinical_state.py and agents/state_tracker.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from medical_agent.config import PipelineConfig
from medical_agent.state.clinical_state import ClinicalState


# ── ClinicalState tests ───────────────────────────────────────────────────────

class TestClinicalState:
    def test_default_initialization(self):
        state = ClinicalState()
        assert state.phase == "intake"
        assert state.turn_number == 0
        assert state.symptoms == {}
        assert state.chief_complaint == ""

    def test_snapshot_is_deep_copy(self):
        state = ClinicalState()
        state.symptoms["fever"] = "positive"
        state.key_positives.append("fever")

        snap = state.snapshot()
        snap.symptoms["cough"] = "positive"
        snap.key_positives.append("cough")

        # Original should be unmodified
        assert "cough" not in state.symptoms
        assert "cough" not in state.key_positives

    def test_to_dict_contains_required_keys(self):
        state = ClinicalState(
            age="45",
            gender="female",
            chief_complaint="chest pain",
        )
        d = state.to_dict()
        required_keys = [
            "age", "gender", "chief_complaint", "hpi", "symptoms",
            "past_medical_history", "medications", "lab_results",
            "key_positives", "key_negatives", "turn_number", "phase",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_positive_symptoms_filter(self):
        state = ClinicalState()
        state.symptoms = {
            "fever": "positive",
            "cough": "negative",
            "dyspnea": "positive",
            "rash": "unknown",
        }
        positives = state.positive_symptoms()
        assert set(positives) == {"fever", "dyspnea"}

    def test_negative_symptoms_filter(self):
        state = ClinicalState()
        state.symptoms = {
            "fever": "positive",
            "cough": "negative",
            "nausea": "negative",
        }
        negatives = state.negative_symptoms()
        assert set(negatives) == {"cough", "nausea"}

    def test_snapshot_preserves_all_fields(self):
        state = ClinicalState(
            age="35",
            gender="male",
            chief_complaint="headache",
            phase="investigation",
            turn_number=5,
        )
        state.symptoms = {"headache": "positive"}
        state.medications = ["aspirin"]
        state.lab_results = {"CBC": "normal"}

        snap = state.snapshot()
        assert snap.age == "35"
        assert snap.gender == "male"
        assert snap.phase == "investigation"
        assert snap.turn_number == 5
        assert snap.symptoms == {"headache": "positive"}
        assert snap.medications == ["aspirin"]
        assert snap.lab_results == {"CBC": "normal"}


# ── StateTracker integration tests ────────────────────────────────────────────

class TestStateTrackerWithMockLLM:
    """Test StateTracker with a mocked LLM response."""

    def _make_config(self):
        return PipelineConfig(
            enable_osce_state=True,
            log_format="text",
        )

    def test_lab_result_recorded(self):
        """Lab results should update state.lab_results and flag abnormals."""
        from medical_agent.agents.state_tracker import StateTracker
        config = self._make_config()
        tracker = StateTracker(config)

        state = ClinicalState()
        tracker.update(
            state,
            "RESULTS: Hemoglobin: 7.2 g/dL (low), WBC: 12000 (elevated)",
            is_lab_result=True,
            test_name="CBC",
        )
        assert "CBC" in state.lab_results
        assert len(state.abnormal_results) > 0  # Not normal

    def test_normal_lab_result_not_flagged(self):
        """Normal lab results should not be added to abnormal_results."""
        from medical_agent.agents.state_tracker import StateTracker
        config = self._make_config()
        tracker = StateTracker(config)

        state = ClinicalState()
        tracker.update(
            state,
            "RESULTS: All values within normal limits",
            is_lab_result=True,
            test_name="Chemistry panel",
        )
        assert "Chemistry panel" in state.lab_results
        assert len(state.abnormal_results) == 0

    def test_raw_dialogue_always_appended(self):
        """Raw dialogue is always appended regardless of OSCE parsing."""
        from medical_agent.agents.state_tracker import StateTracker
        config = PipelineConfig(enable_osce_state=False, log_format="text")
        tracker = StateTracker(config)

        state = ClinicalState()
        tracker.update(state, "I have been feeling tired for weeks.")
        assert len(state.raw_dialogue_history) == 1
        assert "tired" in state.raw_dialogue_history[0]

    def test_llm_parse_applied(self):
        """When LLM returns valid JSON, fields should be applied to state."""
        from medical_agent.agents.state_tracker import StateTracker

        config = self._make_config()
        tracker = StateTracker(config)

        mock_json_response = """{
  "age": "42",
  "gender": "female",
  "chief_complaint": "chest pain",
  "hpi_additions": "sharp chest pain for 2 hours",
  "new_positive_symptoms": ["chest pain", "shortness of breath"],
  "new_negative_symptoms": ["fever"],
  "medications": ["atorvastatin"],
  "allergies": [],
  "past_medical_history": ["hypertension"],
  "family_history": ["coronary artery disease"],
  "social_history": ["non-smoker"],
  "physical_exam_findings": {}
}"""

        with patch.object(tracker, 'query', return_value=mock_json_response):
            state = ClinicalState()
            tracker.update(state, "I'm a 42-year-old woman with chest pain.")

        assert state.age == "42"
        assert state.gender == "female"
        assert state.chief_complaint == "chest pain"
        assert state.symptoms.get("chest pain") == "positive"
        assert state.symptoms.get("fever") == "negative"
        assert "atorvastatin" in state.medications
        assert "hypertension" in state.past_medical_history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
