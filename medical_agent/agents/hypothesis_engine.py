"""
Module 2: KG-Grounded Hypothesis Engine.
Maintains a ranked differential diagnosis, grounded in the knowledge graph
and updated each turn with new evidence via LLM reasoning.
"""

from typing import Dict, List, Optional

from ..agents.base_agent import BaseAgent
from ..config import PipelineConfig
from ..knowledge.knowledge_graph import KnowledgeGraph
from ..prompts.doctor_prompt import HYPOTHESIS_SYSTEM, hypothesis_update_prompt
from ..state.clinical_state import ClinicalState
from ..state.hypothesis_register import Hypothesis, HypothesisRegister


class HypothesisEngine(BaseAgent):
    """
    Module 2: Updates the differential diagnosis each turn.

    When enable_kg_hypothesis=False (ablation), falls back to LLM-only
    hypothesis generation without knowledge graph grounding.
    """

    def __init__(self, config: PipelineConfig, kg: Optional[KnowledgeGraph] = None):
        super().__init__(config, "hypothesis_engine")
        self._kg = kg if kg is not None else KnowledgeGraph(config.kg_path)

    def initialize(
        self, state: ClinicalState, register: HypothesisRegister
    ) -> HypothesisRegister:
        """
        Generate an initial differential from the first clinical state.
        Called after intake parsing.
        """
        return self.update(state, register, new_findings="Initial clinical presentation")

    def update(
        self,
        state: ClinicalState,
        register: HypothesisRegister,
        new_findings: str = "",
    ) -> HypothesisRegister:
        """
        Update the hypothesis register given the current clinical state.
        """
        if not new_findings:
            new_findings = self._summarize_new_findings(state)

        # KG subgraph context — capped to keep input tokens manageable
        kg_text = "(KG disabled)"
        if self.config.enable_kg_hypothesis and self._kg.is_loaded():
            existing_diseases = [h.disease for h in register.hypotheses]
            kg_matches = self._kg.find_matching_diseases(
                state.key_positives, threshold=1
            )
            kg_diseases = existing_diseases + [d for d, _ in kg_matches[:10]]
            raw_kg = self._kg.triples_as_text(list(set(kg_diseases)))
            kg_text = raw_kg[:800] + ("…" if len(raw_kg) > 800 else "")

        compact_state = self._compact_state_summary(state)
        compact_diff  = self._compact_differential(register.to_dict_list())

        prompt = hypothesis_update_prompt(
            compact_state=compact_state,
            kg_text=kg_text,
            new_findings=new_findings,
            compact_differential=compact_diff,
            max_hypotheses=self.config.max_hypotheses,
        )

        try:
            raw = self.query(
                prompt=prompt,
                system_prompt=HYPOTHESIS_SYSTEM,
                model=self.config.doctor_model,
                max_tokens=2500,
            )
            parsed = self.parse_json_with_salvage(raw)
            if parsed and "hypotheses" in parsed:
                register = self._apply_update(register, parsed, state.turn_number)
                self.logger.info(
                    "hypothesis_update",
                    turn=state.turn_number,
                    data={
                        "top_hypothesis": register.top().disease if register.top() else "none",
                        "confidence": register.top_confidence(),
                        "num_hypotheses": len(register.hypotheses),
                        "new_findings": new_findings[:100],
                        "reasoning": parsed.get("reasoning", "")[:200],
                    },
                )
            else:
                self.logger.warning(
                    "hypothesis_parse_failed",
                    turn=state.turn_number,
                    data={"raw": raw[:300]},
                )
                # Graceful degradation: keep existing register
        except Exception as exc:
            self.logger.error(
                "hypothesis_engine_error",
                turn=state.turn_number,
                data={"error": str(exc)},
            )

        # Record snapshot for stagnation detection
        register.record_snapshot(state.turn_number)
        return register

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_update(
        self,
        register: HypothesisRegister,
        parsed: dict,
        turn: int,
    ) -> HypothesisRegister:
        """Rebuild the hypothesis register from the LLM's output."""
        new_hyps = []
        for h_data in parsed["hypotheses"][: self.config.max_hypotheses]:
            disease = h_data.get("disease", "Unknown")
            confidence = float(h_data.get("confidence", 0.1))
            kg_matched = bool(h_data.get("kg_matched", False))

            # Preserve first_mentioned_turn from existing hypothesis if possible
            existing = register.get(disease)
            first_turn = existing.first_mentioned_turn if existing else turn

            hyp = Hypothesis(
                disease=disease,
                confidence=max(0.0, min(1.0, confidence)),
                supporting_evidence=h_data.get("supporting", []),
                contradicting_evidence=h_data.get("contradicting", []),
                kg_matched=kg_matched,
                first_mentioned_turn=first_turn,
                last_updated_turn=turn,
            )
            new_hyps.append(hyp)

        register.hypotheses = new_hyps
        register.normalize_confidences()
        return register

    def _summarize_new_findings(self, state: ClinicalState) -> str:
        """Summarize new evidence from the current state (fallback if not provided)."""
        parts = []
        if state.key_positives:
            parts.append("Positives: " + ", ".join(state.key_positives[-3:]))
        if state.abnormal_results:
            parts.append("Abnormal: " + ", ".join(state.abnormal_results[-2:]))
        return "; ".join(parts) if parts else "No new findings"

    @staticmethod
    def _compact_state_summary(state: ClinicalState) -> str:
        """
        Build a compact, human-readable state summary for the hypothesis prompt.
        Much smaller than the full JSON dump — reduces input token count ~60%.
        """
        parts = []
        demo = " ".join(filter(None, [state.age, state.gender]))
        if demo:
            parts.append(f"Patient: {demo}")
        if state.chief_complaint:
            parts.append(f"CC: {state.chief_complaint}")
        if state.hpi:
            parts.append(f"HPI: {state.hpi[:200]}")
        pos = state.key_positives[:6]
        if pos:
            parts.append(f"+findings: {', '.join(pos)}")
        neg = state.key_negatives[:4]
        if neg:
            parts.append(f"-findings: {', '.join(neg)}")
        if state.abnormal_results:
            parts.append(f"Abnormal labs: {', '.join(state.abnormal_results[:4])}")
        if state.lab_results:
            labs = [f"{k}: {v}" for k, v in list(state.lab_results.items())[:4]]
            parts.append(f"Labs: {', '.join(labs)}")
        if state.past_medical_history:
            parts.append(f"PMH: {', '.join(state.past_medical_history[:3])}")
        if state.medications:
            parts.append(f"Meds: {', '.join(state.medications[:3])}")
        return "\n".join(parts) if parts else "No state captured yet"

    @staticmethod
    def _compact_differential(hypotheses_list: list) -> str:
        """
        Render the current differential as a compact numbered list.
        Example: '1. Myasthenia Gravis (0.45) | +: ptosis, fatigable weakness | -: —'
        """
        if not hypotheses_list:
            return "(none)"
        lines = []
        for i, h in enumerate(hypotheses_list, 1):
            sup = ", ".join(h.get("supporting", [])[:2]) or "—"
            con = ", ".join(h.get("contradicting", [])[:1]) or "—"
            lines.append(
                f"{i}. {h['disease']} ({h.get('confidence', 0):.2f})"
                f" | +: {sup} | -: {con}"
            )
        return "\n".join(lines)

    def differential_as_dict(self, register: HypothesisRegister) -> Dict[str, float]:
        """Return {disease: confidence} dict for info-gain computation."""
        return {h.disease: h.confidence for h in register.hypotheses}
