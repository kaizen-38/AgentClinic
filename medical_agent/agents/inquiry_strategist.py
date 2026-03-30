"""
Module 3: Information-Gain Question Strategist.
Selects the highest-information-gain question or test to ask next,
using KG-grounded Shannon entropy computation.
"""

import random
from typing import List, Optional, Tuple

from ..agents.base_agent import BaseAgent
from ..config import PipelineConfig
from ..knowledge.info_gain import InformationGainCalculator
from ..knowledge.knowledge_graph import KnowledgeGraph
from ..state.clinical_state import ClinicalState
from ..state.hypothesis_register import HypothesisRegister


# Fallback generic intake questions used in intake phase or when IG is too low
INTAKE_QUESTION_POOL = [
    "Can you describe your symptoms in more detail?",
    "When did your symptoms first start?",
    "Have your symptoms been getting better, worse, or staying the same?",
    "Do you have any significant past medical history or chronic conditions?",
    "What medications are you currently taking?",
    "Do you have any known allergies?",
    "Does anyone in your family have a similar condition?",
    "Can you describe your occupation and daily routine?",
    "Do you smoke, drink alcohol, or use any recreational substances?",
    "Have you recently traveled or had any exposures to sick contacts?",
]


class InquiryStrategist(BaseAgent):
    """
    Module 3: Selects the next question or test using information gain.

    Outputs one of:
        ("question", "natural language question to ask patient")
        ("test",     "TEST_NAME")
    """

    def __init__(
        self,
        config: PipelineConfig,
        kg: Optional[KnowledgeGraph] = None,
    ):
        super().__init__(config, "inquiry_strategist")
        if kg is None:
            kg = KnowledgeGraph(config.kg_path)
        self._ig_calc = InformationGainCalculator(kg)
        self._kg = kg

    def select_next_action(
        self,
        state: ClinicalState,
        register: HypothesisRegister,
    ) -> Tuple[str, str, float]:
        """
        Choose the highest-IG next action.

        Returns:
            (action_type, content, ig_value)
            action_type: "question" or "test"
            content:     Natural language question or test name
            ig_value:    Information gain of this action
        """
        # During intake phase, use structured intake questions
        if state.phase == "intake":
            return self._intake_question(state)

        if not self.config.enable_info_gain:
            return self._random_fallback(state, register)

        differential = {h.disease: h.confidence for h in register.hypotheses}

        if not differential:
            return self._intake_question(state)

        # In convergence phase, prioritize confirmatory tests for the top hypothesis
        if state.phase == "convergence":
            test_action = self._select_confirmatory_test(state, register, differential)
            if test_action:
                return test_action

        # Rank symptom questions by information gain
        symptom_ranking = self._ig_calc.top_k_symptoms(
            differential=differential,
            k=self.config.top_k_questions,
            already_asked=state.questions_asked + list(state.symptoms.keys()),
            min_ig=self.config.min_info_gain_threshold,
        )

        # Rank tests by information gain
        test_ranking = self._ig_calc.top_k_tests(
            differential=differential,
            k=self.config.top_k_questions,
            already_ordered=state.tests_ordered,
        )

        # Decide between question and test
        best_sym_ig = symptom_ranking[0][1] if symptom_ranking else 0.0
        best_test_ig = test_ranking[0][1] if test_ranking else 0.0

        if best_test_ig > best_sym_ig * 1.2:  # Prefer tests when clearly better
            test_name, ig = test_ranking[0]
            self.logger.info(
                "selected_test",
                turn=state.turn_number,
                data={"test": test_name, "ig": round(ig, 4)},
            )
            return ("test", test_name, ig)

        if symptom_ranking:
            sym, ig = symptom_ranking[0]
            question = self._symptom_to_question(sym, state)
            self.logger.info(
                "selected_question",
                turn=state.turn_number,
                data={"symptom": sym, "ig": round(ig, 4)},
            )
            return ("question", question, ig)

        # Fallback: pick a test if available
        if test_ranking:
            test_name, ig = test_ranking[0]
            return ("test", test_name, ig)

        return self._random_fallback(state, register)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select_confirmatory_test(
        self,
        state: ClinicalState,
        register: HypothesisRegister,
        differential: dict,
    ) -> Optional[Tuple[str, str, float]]:
        """In convergence phase, look for the most discriminating test for top hypothesis."""
        top = register.top()
        if not top:
            return None
        top_tests = self._kg.get_tests(top.disease)
        already = set(t.lower() for t in state.tests_ordered)
        for test in top_tests:
            if test.lower() not in already:
                ig = self._ig_calc.information_gain(test, differential)
                if ig >= self.config.min_info_gain_threshold:
                    self.logger.info(
                        "selected_confirmatory_test",
                        turn=state.turn_number,
                        data={"test": test, "disease": top.disease, "ig": round(ig, 4)},
                    )
                    return ("test", test, ig)
        return None

    def _intake_question(self, state: ClinicalState) -> Tuple[str, str, float]:
        """Return an appropriate structured intake question."""
        asked = set(q.lower() for q in state.questions_asked)
        for q in INTAKE_QUESTION_POOL:
            if q.lower() not in asked:
                return ("question", q, 0.0)
        return ("question", "Is there anything else about your symptoms you'd like to share?", 0.0)

    def _random_fallback(
        self, state: ClinicalState, register: HypothesisRegister
    ) -> Tuple[str, str, float]:
        """Fallback when IG-based selection yields nothing."""
        top = register.top()
        if top and self._kg.is_loaded():
            tests = [t for t in self._kg.get_tests(top.disease) if t not in state.tests_ordered]
            if tests:
                return ("test", random.choice(tests), 0.0)
        questions = [q for q in INTAKE_QUESTION_POOL if q not in state.questions_asked]
        if questions:
            return ("question", random.choice(questions), 0.0)
        return ("question", "Can you tell me more about how you've been feeling?", 0.0)

    @staticmethod
    def _symptom_to_question(symptom: str, state: ClinicalState) -> str:
        """Convert a symptom name to a natural language yes/no question."""
        templates = [
            f"Have you experienced any {symptom}?",
            f"Do you have {symptom}?",
            f"Can you tell me if you've had any {symptom}?",
            f"Have you noticed any {symptom} recently?",
        ]
        return random.choice(templates)
