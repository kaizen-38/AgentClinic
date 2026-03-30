"""
AgentClinic Adapter.

SDRPDoctorAgent is a drop-in replacement for AgentClinic's DoctorAgent.
It exposes the same interface:
  - system_prompt()    → str
  - inference(question) → str
  - agent_hist          → str
  - MAX_INFS            → int
  - infs                → int
"""

import re
from typing import List, Optional

from ..config import PipelineConfig
from ..monitoring.logger import PipelineLogger
from ..orchestrator import Orchestrator


class SDRPDoctorAgent:
    """
    Drop-in replacement for AgentClinic's DoctorAgent.

    Usage in agentclinic.py — replace:
        doctor_agent = DoctorAgent(scenario, backend_str=doctor_llm, ...)

    With:
        from medical_agent.interface import SDRPDoctorAgent
        from medical_agent.config import PipelineConfig
        cfg = PipelineConfig(
            doctor_model="qwen3-235b",
            support_model="qwen3-30b",
            model_backend="voyager",
            api_base_url=voyager_api_base,
            api_key=voyager_api_key,
            max_turns=total_inferences,
        )
        doctor_agent = SDRPDoctorAgent(
            config=cfg,
            scenario=scenario,
            case_id=str(_scenario_id),
            dataset=dataset,
        )
    """

    def __init__(
        self,
        config: PipelineConfig,
        scenario=None,
        case_id: str = "unknown",
        dataset: str = "MedQA",
        correct_diagnosis: str = "",
        answer_choices: Optional[List[str]] = None,
    ):
        self.config = config
        self.MAX_INFS = config.max_turns
        self.infs = 0
        self.agent_hist = ""
        self._dataset = dataset
        self._case_id = case_id
        self._correct_diagnosis = correct_diagnosis
        self._answer_choices = answer_choices
        self._logger = PipelineLogger("adapter", config.log_level, config.log_format)
        self._final_diagnosis: Optional[str] = None

        # Extract presentation text and correct diagnosis from scenario
        presentation_text = ""
        if scenario is not None:
            try:
                presentation_text = str(scenario.examiner_information())
            except Exception:
                pass
            if not correct_diagnosis:
                try:
                    self._correct_diagnosis = str(scenario.diagnosis_information())
                except Exception:
                    pass
            # For NEJM_Ext, extract answer choices
            if answer_choices is None and dataset in ("NEJM", "NEJM_Ext"):
                try:
                    raw = scenario.scenario_dict.get("answers", [])
                    self._answer_choices = [a["text"] for a in raw if "text" in a]
                except Exception:
                    pass

        self.orchestrator = Orchestrator(
            config=config,
            case_id=case_id,
            dataset=dataset,
            correct_diagnosis=self._correct_diagnosis,
            answer_choices=self._answer_choices,
            presentation_text=presentation_text,
        )

    # ── AgentClinic interface ─────────────────────────────────────────────────

    def system_prompt(self) -> str:
        """
        Returns an AgentClinic-compatible system prompt.
        The internal pipeline handles reasoning; this is primarily for
        compatibility with AgentClinic's agent initialization.
        """
        return (
            "You are Dr. Agent, an AI diagnostic reasoning system. "
            f"You have a maximum of {self.MAX_INFS} turns to arrive at a diagnosis. "
            "You gather patient history, order targeted tests, and issue a final diagnosis "
            "using 'DIAGNOSIS READY: [diagnosis]'. "
            "You can request tests with 'REQUEST TEST: [test_name]'."
        )

    def inference(self, patient_response: str) -> str:
        """
        Main entry point called by AgentClinic each turn.

        Args:
            patient_response: Text from the patient agent or measurement agent.

        Returns:
            Doctor's response string (AgentClinic-compatible).
        """
        if self.infs >= self.MAX_INFS:
            # Force diagnosis at turn limit
            if not self._final_diagnosis:
                result = self.orchestrator.step(patient_response)
                self._final_diagnosis = result.utterance
                if not result.is_diagnosis:
                    # Force a diagnosis
                    from ..agents.diagnosis_resolver import DiagnosisResolver
                    resolver = DiagnosisResolver(self.config)
                    self._final_diagnosis = resolver.resolve(
                        self.orchestrator.state,
                        self.orchestrator.register,
                        dataset=self._dataset,
                        answer_choices=self._answer_choices,
                        trigger_reason="turn_limit_forced",
                    )
            return self._final_diagnosis

        result = self.orchestrator.step(patient_response)
        self.agent_hist += patient_response + "\n" + result.utterance + "\n"
        self.infs += 1

        if result.is_diagnosis:
            self._final_diagnosis = result.utterance
            self._logger.info(
                "diagnosis_issued",
                turn=self.infs,
                data={"utterance": result.utterance, "case_id": self._case_id},
            )

        return result.utterance

    def inference_doctor(self, patient_response: str, image_requested: bool = False) -> str:
        """
        Alias matching AgentClinic's DoctorAgent.inference_doctor() signature.
        """
        return self.inference(patient_response)

    def finalize_case(self, is_correct: bool) -> str:
        """
        Call after the case ends to flush trajectory and metrics.
        Returns trajectory file path.
        """
        predicted = ""
        if self._final_diagnosis:
            m = re.search(r"DIAGNOSIS READY:\s*(.+)", self._final_diagnosis, re.IGNORECASE)
            if m:
                predicted = m.group(1).strip()
        return self.orchestrator.finalize(
            predicted_diagnosis=predicted, is_correct=is_correct
        )

    def reset(self) -> None:
        """Reset for a new case (not typically used — create a new instance instead)."""
        self.infs = 0
        self.agent_hist = ""
        self._final_diagnosis = None
