"""
Module 5: LLM Dialogue Generator.
Generates natural language doctor utterances incorporating the strategist's
chosen question/test, the current clinical context, and verifier guidance.
"""

import json
from typing import Optional, Tuple

from ..agents.base_agent import BaseAgent
from ..config import PipelineConfig
from ..prompts.doctor_prompt import DIALOGUE_SYSTEM, doctor_dialogue_prompt
from ..state.clinical_state import ClinicalState
from ..state.hypothesis_register import HypothesisRegister


class DoctorDialogue(BaseAgent):
    """
    Module 5: Generates natural language from structured inputs.

    Given:
        - The inquiry strategist's chosen action (question or test)
        - Current OSCE state
        - Top hypothesis and phase
        - Verifier recommendation

    Produces:
        Natural language utterance for the patient
        OR "REQUEST TEST: [test_name]"
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config, "doctor_dialogue")

    def generate(
        self,
        state: ClinicalState,
        register: HypothesisRegister,
        action_type: str,
        action_content: str,
        verifier_recommendation: str = "continue",
    ) -> str:
        """
        Generate doctor utterance.

        Args:
            state:                   Current clinical state.
            register:                Current hypothesis register.
            action_type:             "question" or "test"
            action_content:          The symptom question or test name.
            verifier_recommendation: Guidance from the verifier module.

        Returns:
            Doctor utterance string (ready to pass to AgentClinic).
        """
        # Test requests are formatted directly without LLM (avoids wasted tokens)
        if action_type == "test":
            test_name = action_content.replace(" ", "_")
            utterance = f"REQUEST TEST: {test_name}"
            self.logger.info(
                "test_request_generated",
                turn=state.turn_number,
                data={"test": test_name},
            )
            return utterance

        # For questions, use LLM to generate natural conversational phrasing
        top = register.top()
        top_hypothesis = top.disease if top else "unknown"
        top_confidence = top.confidence if top else 0.0

        # Build a concise OSCE summary (avoid full JSON for dialogue to save tokens)
        osce_summary = self._build_osce_summary(state)

        prompt = doctor_dialogue_prompt(
            osce_state_json=osce_summary,
            top_question=action_content,
            phase=state.phase,
            turn=state.turn_number,
            max_turns=self.config.max_turns,
            top_hypothesis=top_hypothesis,
            top_confidence=top_confidence,
            verifier_recommendation=verifier_recommendation,
        )

        try:
            utterance = self.query(
                prompt=prompt,
                system_prompt=DIALOGUE_SYSTEM,
                model=self.config.doctor_model,
                max_tokens=200,
            ).strip()

            # Safety: strip any accidental JSON that leaked out
            if utterance.startswith("{"):
                # Try to extract a plain string from the JSON
                parsed = self.parse_json(utterance)
                if parsed and "response" in parsed:
                    utterance = parsed["response"]
                elif parsed and "utterance" in parsed:
                    utterance = parsed["utterance"]
                else:
                    utterance = action_content  # fallback to raw question

            self.logger.info(
                "dialogue_generated",
                turn=state.turn_number,
                data={"utterance": utterance[:100], "phase": state.phase},
            )
            # Track question asked
            state.questions_asked.append(action_content)
            return utterance

        except Exception as exc:
            self.logger.error(
                "dialogue_generation_error",
                turn=state.turn_number,
                data={"error": str(exc)},
            )
            # Graceful degradation: return the raw question
            state.questions_asked.append(action_content)
            return action_content

    @staticmethod
    def _build_osce_summary(state: ClinicalState) -> str:
        """Build a compact clinical summary for the dialogue prompt."""
        parts = []
        if state.chief_complaint:
            parts.append(f"CC: {state.chief_complaint}")
        if state.age or state.gender:
            parts.append(f"Patient: {state.age or '?'}-year-old {state.gender or '?'}")
        if state.key_positives:
            parts.append(f"Positives: {', '.join(state.key_positives[:5])}")
        if state.key_negatives:
            parts.append(f"Negatives: {', '.join(state.key_negatives[:3])}")
        if state.lab_results:
            recent = list(state.lab_results.items())[-3:]
            parts.append("Labs: " + "; ".join(f"{k}: {v[:40]}" for k, v in recent))
        if state.unexplained_findings:
            parts.append(f"Unexplained: {', '.join(state.unexplained_findings[:2])}")
        return "\n".join(parts) if parts else "No structured data yet."
