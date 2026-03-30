"""
Module 4: OSCE Clinical State Manager.
Parses new patient/measurement responses and updates the ClinicalState.
"""

import json
from typing import Optional

from ..agents.base_agent import BaseAgent
from ..config import PipelineConfig
from ..prompts.intake_prompt import INTAKE_SYSTEM, intake_extract_prompt
from ..state.clinical_state import ClinicalState


class StateTracker(BaseAgent):
    """
    Updates the shared ClinicalState after each turn.

    When enable_osce_state=False (ablation), skips LLM parsing and only
    appends raw text to raw_dialogue_history.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config, "state_tracker")

    def update(
        self,
        state: ClinicalState,
        new_text: str,
        is_lab_result: bool = False,
        test_name: Optional[str] = None,
    ) -> ClinicalState:
        """
        Parse new_text and update state in-place.
        Returns the mutated state (same object — caller is responsible for snapshots).
        """
        # Append to raw history regardless
        state.raw_dialogue_history.append(new_text)

        if not self.config.enable_osce_state:
            return state

        # Handle lab/measurement results separately
        if is_lab_result and test_name:
            return self._update_lab_result(state, new_text, test_name)

        return self._update_from_dialogue(state, new_text)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _update_lab_result(
        self, state: ClinicalState, result_text: str, test_name: str
    ) -> ClinicalState:
        """Parse a measurement agent response and update lab_results."""
        state.lab_results[test_name] = result_text
        # Determine if result is abnormal via simple heuristic
        normal_keywords = ["normal", "within normal", "negative", "unremarkable", "wn limits"]
        is_normal = any(kw in result_text.lower() for kw in normal_keywords)
        if not is_normal and test_name not in state.abnormal_results:
            state.abnormal_results.append(f"{test_name}: {result_text[:100]}")
        self.logger.info(
            "lab_result_recorded",
            turn=state.turn_number,
            data={"test": test_name, "normal": is_normal},
        )
        return state

    def _update_from_dialogue(
        self, state: ClinicalState, patient_text: str
    ) -> ClinicalState:
        """Call LLM to extract structured fields from patient dialogue."""
        current_state_json = json.dumps(state.to_dict(), indent=2)
        prompt = intake_extract_prompt(patient_text, current_state_json)

        try:
            raw = self.query(
                prompt=prompt,
                system_prompt=INTAKE_SYSTEM,
                model=self.config.support_model,
                max_tokens=512,
            )
            parsed = self.parse_json(raw)
            if parsed is None:
                self.logger.warning(
                    "state_parse_failed",
                    turn=state.turn_number,
                    data={"raw_output": raw[:200]},
                )
                return state
            self._apply_parsed(state, parsed)
        except Exception as exc:
            self.logger.error(
                "state_tracker_error",
                turn=state.turn_number,
                data={"error": str(exc)},
            )
        return state

    def _apply_parsed(self, state: ClinicalState, parsed: dict) -> None:
        """Apply extracted fields to the state object."""
        if parsed.get("age"):
            state.age = parsed["age"]
        if parsed.get("gender"):
            state.gender = parsed["gender"]
        if parsed.get("chief_complaint"):
            state.chief_complaint = parsed["chief_complaint"]
        if parsed.get("hpi_additions"):
            state.hpi = (state.hpi + " " + parsed["hpi_additions"]).strip()

        for sym in parsed.get("new_positive_symptoms", []):
            sym = sym.lower().strip()
            if sym:
                state.symptoms[sym] = "positive"
                if sym not in state.key_positives:
                    state.key_positives.append(sym)

        for sym in parsed.get("new_negative_symptoms", []):
            sym = sym.lower().strip()
            if sym:
                state.symptoms[sym] = "negative"
                if sym not in state.key_negatives:
                    state.key_negatives.append(sym)

        for item in parsed.get("medications", []):
            if item and item not in state.medications:
                state.medications.append(item)

        for item in parsed.get("allergies", []):
            if item and item not in state.allergies:
                state.allergies.append(item)

        for item in parsed.get("past_medical_history", []):
            if item and item not in state.past_medical_history:
                state.past_medical_history.append(item)

        for item in parsed.get("family_history", []):
            if item and item not in state.family_history:
                state.family_history.append(item)

        for item in parsed.get("social_history", []):
            if item and item not in state.social_history:
                state.social_history.append(item)

        for area, finding in parsed.get("physical_exam_findings", {}).items():
            state.physical_exam[area] = finding
