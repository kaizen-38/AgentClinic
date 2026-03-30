"""
Module 1: Structured Intake Parser.
Handles the first 1-2 turns, focusing on gathering a complete initial history
(CC, HPI, PMH, medications, allergies, family/social history).
"""

import json
from typing import Optional

from ..agents.base_agent import BaseAgent
from ..config import PipelineConfig
from ..prompts.intake_prompt import INTAKE_SYSTEM, intake_extract_prompt
from ..state.clinical_state import ClinicalState


INTAKE_SYSTEM_EXTENDED = (
    "You are a clinical intake specialist. During the initial assessment, your goal is to "
    "systematically gather a complete history in OSCE format. Extract ALL available structured "
    "information from the patient presentation, including demographic data, presenting complaint, "
    "HPI, review of systems, PMH, medications, allergies, family history, and social history."
)


def initial_intake_prompt(presentation_text: str) -> str:
    return f"""A patient has presented with the following information. Extract ALL available structured clinical data.

Patient Presentation:
\"\"\"{presentation_text}\"\"\"

Output a comprehensive JSON object:
{{
  "age": "patient age or null",
  "gender": "patient gender or null",
  "chief_complaint": "primary reason for visit",
  "hpi": "history of present illness narrative",
  "positive_symptoms": ["confirmed symptoms"],
  "negative_symptoms": ["denied symptoms"],
  "past_medical_history": ["past conditions"],
  "medications": ["current medications with doses if available"],
  "allergies": ["known allergies"],
  "family_history": ["relevant family history"],
  "social_history": ["occupation, smoking, alcohol, drugs, living situation"],
  "physical_exam": {{"system": "finding"}},
  "review_of_systems": {{"system": "positive/negative findings"}},
  "reasoning": "brief note on confidence of extraction"
}}

Rules:
- Extract EVERYTHING mentioned, even if only briefly.
- Use lowercase for symptoms.
- If a field is not mentioned, use null (not an empty string).
- Output ONLY valid JSON.
"""


class IntakeParser(BaseAgent):
    """
    Module 1: Handles initial structured intake.
    Used during the intake phase (turns 1-2) to build the initial ClinicalState.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config, "intake_parser")

    def parse_initial_presentation(
        self, presentation_text: str, state: ClinicalState
    ) -> ClinicalState:
        """
        Parse the initial scenario presentation text into a structured ClinicalState.
        Called once at case start.
        """
        if not self.config.enable_osce_state:
            state.raw_dialogue_history.append(presentation_text)
            state.chief_complaint = presentation_text[:200]
            return state

        try:
            raw = self.query(
                prompt=initial_intake_prompt(presentation_text),
                system_prompt=INTAKE_SYSTEM_EXTENDED,
                model=self.config.support_model,
                max_tokens=800,
            )
            parsed = self.parse_json(raw)
            if parsed:
                self._apply_full_intake(state, parsed)
                self.logger.info(
                    "initial_intake_complete",
                    turn=0,
                    data={
                        "chief_complaint": state.chief_complaint,
                        "n_symptoms": len(state.symptoms),
                        "age": state.age,
                        "gender": state.gender,
                    },
                )
            else:
                self.logger.warning(
                    "initial_intake_parse_failed",
                    turn=0,
                    data={"raw": raw[:300]},
                )
                state.chief_complaint = presentation_text[:200]
        except Exception as exc:
            self.logger.error(
                "initial_intake_error",
                turn=0,
                data={"error": str(exc)},
            )
            state.chief_complaint = presentation_text[:200]

        state.raw_dialogue_history.append(presentation_text)
        return state

    def parse_turn(
        self, patient_text: str, state: ClinicalState
    ) -> ClinicalState:
        """
        Parse a patient dialogue turn during the intake phase.
        Delegates to the standard intake_extract_prompt.
        """
        state.raw_dialogue_history.append(patient_text)
        if not self.config.enable_osce_state:
            return state

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
            if parsed:
                self._apply_incremental(state, parsed)
        except Exception as exc:
            self.logger.warning(
                "intake_turn_error",
                turn=state.turn_number,
                data={"error": str(exc)},
            )
        return state

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_full_intake(self, state: ClinicalState, parsed: dict) -> None:
        if parsed.get("age"):
            state.age = parsed["age"]
        if parsed.get("gender"):
            state.gender = parsed["gender"]
        if parsed.get("chief_complaint"):
            state.chief_complaint = parsed["chief_complaint"]
        if parsed.get("hpi"):
            state.hpi = parsed["hpi"]
        for sym in parsed.get("positive_symptoms", []):
            sym = sym.lower().strip()
            if sym:
                state.symptoms[sym] = "positive"
                if sym not in state.key_positives:
                    state.key_positives.append(sym)
        for sym in parsed.get("negative_symptoms", []):
            sym = sym.lower().strip()
            if sym:
                state.symptoms[sym] = "negative"
                if sym not in state.key_negatives:
                    state.key_negatives.append(sym)
        for item in parsed.get("past_medical_history", []) or []:
            if item and item not in state.past_medical_history:
                state.past_medical_history.append(item)
        for item in parsed.get("medications", []) or []:
            if item and item not in state.medications:
                state.medications.append(item)
        for item in parsed.get("allergies", []) or []:
            if item and item not in state.allergies:
                state.allergies.append(item)
        for item in parsed.get("family_history", []) or []:
            if item and item not in state.family_history:
                state.family_history.append(item)
        for item in parsed.get("social_history", []) or []:
            if item and item not in state.social_history:
                state.social_history.append(item)
        for area, finding in (parsed.get("physical_exam") or {}).items():
            state.physical_exam[area] = finding

    def _apply_incremental(self, state: ClinicalState, parsed: dict) -> None:
        if parsed.get("age") and not state.age:
            state.age = parsed["age"]
        if parsed.get("gender") and not state.gender:
            state.gender = parsed["gender"]
        if parsed.get("chief_complaint") and not state.chief_complaint:
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
        for item in parsed.get("medications", []):
            if item and item not in state.medications:
                state.medications.append(item)
        for item in parsed.get("allergies", []):
            if item and item not in state.allergies:
                state.allergies.append(item)
