"""
Module 6: Consistency Verification & Reflection.
Runs after each turn to check for reasoning inconsistencies, dropped hypotheses,
and unexplained findings.
"""

import json
from dataclasses import dataclass, field
from typing import List

from ..agents.base_agent import BaseAgent
from ..config import PipelineConfig
from ..prompts.verifier_prompt import VERIFIER_SYSTEM, verifier_check_prompt
from ..state.clinical_state import ClinicalState
from ..state.hypothesis_register import HypothesisRegister


@dataclass
class VerificationResult:
    is_consistent: bool = True
    unexplained_positives: List[str] = field(default_factory=list)
    ignored_abnormals: List[str] = field(default_factory=list)
    dropped_hypotheses_warning: List[str] = field(default_factory=list)
    recommendation: str = "continue"   # "continue" | "reconsider_dropped" | "seek_new_evidence"
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "is_consistent": self.is_consistent,
            "unexplained_positives": self.unexplained_positives,
            "ignored_abnormals": self.ignored_abnormals,
            "dropped_hypotheses_warning": self.dropped_hypotheses_warning,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
        }

    @property
    def flagged(self) -> bool:
        return not self.is_consistent or bool(self.dropped_hypotheses_warning)


class Verifier(BaseAgent):
    """
    Module 6: Per-turn consistency verification.

    Checks:
    1. Does the top hypothesis explain all confirmed positives?
    2. Are abnormal results integrated?
    3. Were any strong hypotheses dropped prematurely?

    Runs every `verification_frequency` turns.
    When enable_verifier=False, returns a trivially consistent result.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__(config, "verifier")
        self._previous_top3: List[str] = []

    def verify(
        self,
        state: ClinicalState,
        register: HypothesisRegister,
    ) -> VerificationResult:
        """
        Run a consistency check for the current turn.
        Returns VerificationResult (trivially OK if verifier is disabled).
        """
        if not self.config.enable_verifier:
            return VerificationResult()

        # Only run at specified frequency
        if state.turn_number % self.config.verification_frequency != 0:
            return VerificationResult()

        top = register.top()
        if not top:
            return VerificationResult()

        current_top3 = register.top_k_names(3)
        result = self._run_check(state, register, top.disease, top.confidence, current_top3)

        # Update state unexplained findings
        state.unexplained_findings = result.unexplained_positives

        self.logger.info(
            "verification_complete",
            turn=state.turn_number,
            data={
                "is_consistent": result.is_consistent,
                "recommendation": result.recommendation,
                "dropped_warnings": result.dropped_hypotheses_warning,
                "unexplained": result.unexplained_positives,
            },
        )

        self._previous_top3 = current_top3
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _run_check(
        self,
        state: ClinicalState,
        register: HypothesisRegister,
        top_hypothesis: str,
        top_confidence: float,
        current_top3: List[str],
    ) -> VerificationResult:
        hyp_json = json.dumps(register.to_dict_list(), indent=2)
        osce_json = json.dumps(state.to_dict(), indent=2)

        prompt = verifier_check_prompt(
            osce_state_json=osce_json,
            top_hypothesis=top_hypothesis,
            top_confidence=top_confidence,
            hypotheses_json=hyp_json,
            previous_top3=self._previous_top3,
            current_top3=current_top3,
        )

        try:
            raw = self.query(
                prompt=prompt,
                system_prompt=VERIFIER_SYSTEM,
                model=self.config.support_model,
                max_tokens=512,
            )
            parsed = self.parse_json(raw)
            if parsed:
                return VerificationResult(
                    is_consistent=bool(parsed.get("is_consistent", True)),
                    unexplained_positives=parsed.get("unexplained_positives", []),
                    ignored_abnormals=parsed.get("ignored_abnormals", []),
                    dropped_hypotheses_warning=parsed.get("dropped_hypotheses_warning", []),
                    recommendation=parsed.get("recommendation", "continue"),
                    reasoning=parsed.get("reasoning", ""),
                )
            else:
                self.logger.warning(
                    "verifier_parse_failed",
                    turn=state.turn_number,
                    data={"raw": raw[:200]},
                )
                return VerificationResult()

        except Exception as exc:
            self.logger.error(
                "verifier_error",
                turn=state.turn_number,
                data={"error": str(exc)},
            )
            return VerificationResult()
