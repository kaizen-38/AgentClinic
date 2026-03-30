"""
Module 7: Confidence-Calibrated Diagnosis Resolver.
Produces a final diagnosis with chain-of-thought reasoning and
normalizes the label to dataset-expected format.
"""

import json
from typing import List, Optional

from ..agents.base_agent import BaseAgent
from ..config import PipelineConfig
from ..prompts.resolver_prompt import RESOLVER_SYSTEM, resolver_prompt
from ..state.clinical_state import ClinicalState
from ..state.hypothesis_register import HypothesisRegister


class DiagnosisResolver(BaseAgent):
    """
    Module 7: Final diagnosis generator.

    Triggered by:
    - Confidence threshold: top hypothesis confidence > config.confidence_threshold
    - Stagnation: top-3 differential unchanged for config.stagnation_window turns
    - Turn limit: max_turns reached

    Produces a dataset-normalized diagnosis string for AgentClinic.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config, "diagnosis_resolver")

    def resolve(
        self,
        state: ClinicalState,
        register: HypothesisRegister,
        dataset: str = "MedQA",
        answer_choices: Optional[List[str]] = None,
        trigger_reason: str = "confidence_threshold",
    ) -> str:
        """
        Generate a final diagnosis.

        Returns:
            AgentClinic-formatted string: "DIAGNOSIS READY: [diagnosis]"
        """
        osce_json = json.dumps(state.to_dict(), indent=2)
        hyps_json = json.dumps(register.to_dict_list(), indent=2)

        prompt = resolver_prompt(
            osce_state_json=osce_json,
            hypotheses_json=hyps_json,
            dataset=dataset,
            answer_choices=answer_choices,
        )

        diagnosis = None
        try:
            raw = self.query(
                prompt=prompt,
                system_prompt=RESOLVER_SYSTEM,
                model=self.config.doctor_model,
                max_tokens=700,
            )
            parsed = self.parse_json(raw)
            if parsed and "diagnosis" in parsed:
                diagnosis = parsed["diagnosis"].strip()
                self.logger.info(
                    "diagnosis_resolved",
                    turn=state.turn_number,
                    data={
                        "diagnosis": diagnosis,
                        "confidence": parsed.get("confidence", 0),
                        "trigger": trigger_reason,
                        "reasoning_preview": parsed.get("chain_of_thought", "")[:200],
                    },
                )
            else:
                # Fallback: use top hypothesis from register
                self.logger.warning(
                    "resolver_parse_failed",
                    turn=state.turn_number,
                    data={"raw": raw[:300]},
                )
                diagnosis = self._fallback_diagnosis(register, answer_choices)

        except Exception as exc:
            self.logger.error(
                "resolver_error",
                turn=state.turn_number,
                data={"error": str(exc)},
            )
            diagnosis = self._fallback_diagnosis(register, answer_choices)

        if not diagnosis:
            diagnosis = self._fallback_diagnosis(register, answer_choices)

        # Normalize to dataset format
        diagnosis = self._normalize_label(diagnosis, dataset, answer_choices)
        state.phase = "diagnosis"

        return f"DIAGNOSIS READY: {diagnosis}"

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fallback_diagnosis(
        self,
        register: HypothesisRegister,
        answer_choices: Optional[List[str]] = None,
    ) -> str:
        """Use the top-confidence hypothesis as fallback."""
        top = register.top()
        if not top:
            return "Unknown"
        if answer_choices:
            # Find the best matching choice
            best = self._best_choice_match(top.disease, answer_choices)
            return best or top.disease
        return top.disease

    @staticmethod
    def _normalize_label(
        diagnosis: str,
        dataset: str,
        answer_choices: Optional[List[str]] = None,
    ) -> str:
        """
        Normalize diagnosis label to dataset format.

        - NEJM/NEJM_Ext: match to exact answer choice string
        - MedQA/MedQA_Ext: ensure USMLE-level specificity
        """
        if answer_choices:
            # Try exact match first
            for choice in answer_choices:
                if choice.lower() == diagnosis.lower():
                    return choice
            # Try partial match
            best = DiagnosisResolver._best_choice_match(diagnosis, answer_choices)
            if best:
                return best

        # For MedQA: basic capitalization normalization
        if dataset in ("MedQA", "MedQA_Ext"):
            # Title-case the diagnosis if it's all lowercase
            if diagnosis == diagnosis.lower():
                diagnosis = diagnosis.title()

        return diagnosis

    @staticmethod
    def _best_choice_match(diagnosis: str, choices: List[str]) -> Optional[str]:
        """Find the best matching answer choice using substring overlap."""
        diag_lower = diagnosis.lower()
        best_choice = None
        best_overlap = 0

        for choice in choices:
            choice_lower = choice.lower()
            # Count common words
            diag_words = set(diag_lower.split())
            choice_words = set(choice_lower.split())
            overlap = len(diag_words & choice_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_choice = choice

        return best_choice if best_overlap > 0 else None
