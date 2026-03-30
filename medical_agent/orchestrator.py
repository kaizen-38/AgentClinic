"""
Pipeline Orchestrator.

Manages the full diagnostic pipeline turn-by-turn.

Turn lifecycle:
1. Receive patient response (or initial presentation)
2. StateTracker: update OSCE state with new information
3. HypothesisEngine: update differential based on new evidence
4. Verifier: check consistency, flag issues
5. IF termination → DiagnosisResolver → "DIAGNOSIS READY: ..."
6. ELSE → InquiryStrategist: select highest-IG question/test
7. DoctorDialogue: generate natural language
8. Output utterance to AgentClinic
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

from .agents.diagnosis_resolver import DiagnosisResolver
from .agents.doctor_dialogue import DoctorDialogue
from .agents.hypothesis_engine import HypothesisEngine
from .agents.inquiry_strategist import InquiryStrategist
from .agents.intake_parser import IntakeParser
from .agents.state_tracker import StateTracker
from .agents.verifier import Verifier, VerificationResult
from .config import PipelineConfig
from .knowledge.knowledge_graph import KnowledgeGraph
from .monitoring.logger import PipelineLogger
from .monitoring.metrics import CaseMetrics, TurnMetrics
from .monitoring.trajectory_recorder import TrajectoryRecorder
from .state.clinical_state import ClinicalState
from .state.hypothesis_register import HypothesisRegister


@dataclass
class StepResult:
    utterance: str
    action_type: str        # "question" | "test" | "diagnosis"
    top_hypothesis: str
    top_confidence: float
    info_gain: float
    phase: str
    verifier_result: Optional[VerificationResult] = None
    is_diagnosis: bool = False


class Orchestrator:
    """
    Manages the complete SDRP diagnostic pipeline.

    Instantiate once per case; call step() for each dialogue turn.
    After the case ends, call finalize() to flush metrics and trajectory.
    """

    def __init__(
        self,
        config: PipelineConfig,
        case_id: str = "unknown",
        dataset: str = "MedQA",
        correct_diagnosis: str = "",
        answer_choices: Optional[List[str]] = None,
        presentation_text: str = "",
    ):
        self.config = config
        self.dataset = dataset
        self.answer_choices = answer_choices
        self.logger = PipelineLogger("orchestrator", config.log_level, config.log_format)

        # ── Shared state ──────────────────────────────────────────────────────
        self.state = ClinicalState()
        self.register = HypothesisRegister()

        # ── Knowledge graph (shared across modules) ───────────────────────────
        self._kg = KnowledgeGraph(config.kg_path)

        # ── Pipeline modules ──────────────────────────────────────────────────
        self.intake_parser = IntakeParser(config)
        self.state_tracker = StateTracker(config)
        self.hypothesis_engine = HypothesisEngine(config, kg=self._kg)
        self.inquiry_strategist = InquiryStrategist(config, kg=self._kg)
        self.verifier = Verifier(config)
        self.doctor_dialogue = DoctorDialogue(config)
        self.diagnosis_resolver = DiagnosisResolver(config)

        # ── Monitoring ────────────────────────────────────────────────────────
        self.metrics = CaseMetrics(
            case_id=case_id,
            dataset=dataset,
            correct_diagnosis=correct_diagnosis,
            max_turns=config.max_turns,
        )
        self.trajectory = TrajectoryRecorder(
            case_id=case_id,
            dataset=dataset,
            correct_diagnosis=correct_diagnosis,
            output_dir=config.trajectory_output_dir,
        )

        # ── Runtime state ─────────────────────────────────────────────────────
        self._turn = 0
        self._last_verifier_rec = "continue"
        self._diagnosis_issued = False

        # Parse initial presentation if provided
        if presentation_text:
            self._initialize(presentation_text)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, patient_response: str) -> StepResult:
        """
        Process one dialogue turn.

        Args:
            patient_response: Text from patient agent or measurement agent.

        Returns:
            StepResult with the doctor's utterance and metadata.
        """
        self._turn += 1
        self.state.turn_number = self._turn
        self._update_phase()

        self.logger.info(
            "turn_start",
            turn=self._turn,
            data={"phase": self.state.phase, "patient_response": patient_response[:100]},
        )

        # ── Detect if this is a measurement result ────────────────────────────
        is_lab = "RESULTS:" in patient_response.upper()
        test_name = None
        if is_lab:
            test_name = self._extract_test_name_from_result(patient_response)

        # ── Step 2: State update ──────────────────────────────────────────────
        if self.state.phase == "intake" and not is_lab:
            self.intake_parser.parse_turn(patient_response, self.state)
        else:
            self.state_tracker.update(
                self.state,
                patient_response,
                is_lab_result=is_lab,
                test_name=test_name,
            )
        if is_lab and test_name:
            self.state.tests_ordered.append(test_name)
            self.trajectory.record_test(test_name)
            self.metrics.tests_ordered.append(test_name)
            # Check if normal
            normal_kw = ["normal", "negative", "unremarkable", "within normal"]
            if any(kw in patient_response.lower() for kw in normal_kw):
                self.metrics.normal_results_count += 1

        # ── Step 3: Hypothesis update ─────────────────────────────────────────
        if self._turn == 1:
            self.register = self.hypothesis_engine.initialize(self.state, self.register)
        else:
            self.register = self.hypothesis_engine.update(
                self.state, self.register, new_findings=patient_response[:300]
            )

        # ── Step 4: Verification ──────────────────────────────────────────────
        verifier_result = self.verifier.verify(self.state, self.register)
        self._last_verifier_rec = verifier_result.recommendation

        # ── Step 5: Termination check ─────────────────────────────────────────
        trigger = self._check_termination()
        if trigger:
            utterance = self.diagnosis_resolver.resolve(
                self.state, self.register,
                dataset=self.dataset,
                answer_choices=self.answer_choices,
                trigger_reason=trigger,
            )
            self._diagnosis_issued = True
            self.metrics.diagnosis_turn = self._turn
            result = StepResult(
                utterance=utterance,
                action_type="diagnosis",
                top_hypothesis=self.register.top().disease if self.register.top() else "",
                top_confidence=self.register.top_confidence(),
                info_gain=0.0,
                phase="diagnosis",
                verifier_result=verifier_result,
                is_diagnosis=True,
            )
            self._record_turn(result, verifier_result)
            return result

        # ── Step 6: Inquiry strategy ──────────────────────────────────────────
        action_type, action_content, ig = self.inquiry_strategist.select_next_action(
            self.state, self.register
        )

        # ── Step 7: Dialogue generation ───────────────────────────────────────
        utterance = self.doctor_dialogue.generate(
            state=self.state,
            register=self.register,
            action_type=action_type,
            action_content=action_content,
            verifier_recommendation=self._last_verifier_rec,
        )

        # ── Step 8: Build result ──────────────────────────────────────────────
        top = self.register.top()
        result = StepResult(
            utterance=utterance,
            action_type=action_type,
            top_hypothesis=top.disease if top else "",
            top_confidence=self.register.top_confidence(),
            info_gain=ig,
            phase=self.state.phase,
            verifier_result=verifier_result,
            is_diagnosis=False,
        )
        self._record_turn(result, verifier_result)
        return result

    def finalize(self, predicted_diagnosis: str, is_correct: bool) -> str:
        """
        Flush trajectory and metrics after case completion.
        Returns the trajectory file path.
        """
        self.metrics.predicted_diagnosis = predicted_diagnosis
        self.metrics.accuracy = is_correct
        self.metrics.finalize()

        self.trajectory.finalize(predicted=predicted_diagnosis, correct=is_correct)
        path = self.trajectory.save()
        self.logger.info(
            "case_finalized",
            turn=self._turn,
            data={
                "correct": is_correct,
                "predicted": predicted_diagnosis,
                "turns_used": self._turn,
                "trajectory_path": path,
            },
        )
        return path

    # ── Private helpers ───────────────────────────────────────────────────────

    def _initialize(self, presentation_text: str) -> None:
        """Parse the initial scenario presentation before turn 1."""
        self.state = self.intake_parser.parse_initial_presentation(
            presentation_text, self.state
        )
        self.logger.info(
            "case_initialized",
            turn=0,
            data={
                "chief_complaint": self.state.chief_complaint,
                "age": self.state.age,
                "gender": self.state.gender,
            },
        )

    def _update_phase(self) -> None:
        """Transition between pipeline phases based on turn and confidence."""
        if self._turn <= self.config.intake_turns:
            self.state.phase = "intake"
        elif self.register.top_confidence() >= 0.5:
            self.state.phase = "convergence"
        else:
            self.state.phase = "investigation"

    def _check_termination(self) -> Optional[str]:
        """
        Check all termination conditions. Returns trigger reason or None.
        """
        if self._turn < self.config.min_turns_before_diagnosis:
            return None

        # Confidence threshold
        if (
            self.config.enable_confidence_termination
            and self.register.top_confidence() >= self.config.confidence_threshold
        ):
            self.logger.info(
                "termination_confidence",
                turn=self._turn,
                data={"confidence": self.register.top_confidence()},
            )
            return "confidence_threshold"

        # Stagnation — only fire if top confidence is meaningful (≥ 0.45)
        # Prevents early termination when the model hasn't built enough evidence
        if (
            self.register.is_stagnant(self.config.stagnation_window)
            and self.register.top_confidence() >= 0.45
        ):
            self.logger.info("termination_stagnation", turn=self._turn)
            return "stagnation"

        # Turn limit
        if self._turn >= self.config.max_turns:
            self.logger.info("termination_turn_limit", turn=self._turn)
            return "turn_limit"

        return None

    def _record_turn(
        self, result: StepResult, verifier_result: VerificationResult
    ) -> None:
        """Record turn to metrics and trajectory."""
        tm = TurnMetrics(
            turn=self._turn,
            phase=result.phase,
            action=result.action_type,
            top_hypothesis=result.top_hypothesis,
            top_confidence=result.top_confidence,
            info_gain=result.info_gain,
            verifier_flagged=verifier_result.flagged,
        )
        self.metrics.record_turn(tm)

        state_snapshot = self.state.to_dict()
        hyp_snapshot = self.register.to_dict_list()
        verifier_dict = verifier_result.to_dict() if verifier_result.flagged else None

        self.trajectory.record_turn(
            turn=self._turn,
            clinical_state_dict=state_snapshot,
            hypotheses_list=hyp_snapshot,
            top_confidence=result.top_confidence,
            info_gain=result.info_gain,
            action=result.action_type,
            utterance=result.utterance,
            verifier_result=verifier_dict,
        )

    @staticmethod
    def _extract_test_name_from_result(result_text: str) -> Optional[str]:
        """
        Try to extract the test name from a measurement agent response.
        Falls back to "unknown_test" if not parseable.
        """
        # Common patterns: "RESULTS: CBC: ..." or "RESULTS: [CBC] ..."
        patterns = [
            r"RESULTS:\s*([A-Za-z0-9_\-\s]+?):",
            r"RESULTS:\s*\[([^\]]+)\]",
            r"RESULTS:\s*([A-Za-z0-9_\-\s]{3,40})\n",
        ]
        for pat in patterns:
            m = re.search(pat, result_text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return "unknown_test"
