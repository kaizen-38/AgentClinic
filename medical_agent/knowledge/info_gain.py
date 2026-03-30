"""
Shannon entropy and information gain computation.
Follows the MedKGI (Wang et al. 2025) formulation:

    H(D) = -Σ P(Di) * log2(P(Di))

    P(s) = Σ P(s|Di) * P(Di)
    P(Di|s)  = P(s|Di) * P(Di) / P(s)
    P(Di|¬s) = P(¬s|Di) * P(Di) / P(¬s)

    IG(s) = H(D) - H(D|s)
    H(D|s) = P(s) * H(D|s+) + P(¬s) * H(D|s-)

Where:
    P(s|Di)  = 1 / |N(Di)|  if symptom s ∈ N(Di), else 0
    N(Di)    = set of symptom nodes connected to disease Di in KG
"""

import math
from typing import Dict, List, Optional, Tuple

from .knowledge_graph import KnowledgeGraph


class InformationGainCalculator:
    """
    Computes per-symptom information gain given:
        - A current differential (dict of disease → posterior probability)
        - A KnowledgeGraph for P(s|Di) look-ups
    """

    def __init__(self, kg: KnowledgeGraph):
        self._kg = kg

    # ── Core entropy / IG ─────────────────────────────────────────────────────

    @staticmethod
    def entropy(probs: List[float]) -> float:
        """Shannon entropy H = -Σ p * log2(p)  (ignores zeros)."""
        h = 0.0
        for p in probs:
            if p > 1e-12:
                h -= p * math.log2(p)
        return h

    def prior_entropy(self, differential: Dict[str, float]) -> float:
        return self.entropy(list(differential.values()))

    def p_symptom_given_disease(self, symptom: str, disease: str) -> float:
        """P(s|Di) = 1/|N(Di)| if s ∈ N(Di), else 0."""
        symptoms = self._kg.get_symptoms(disease)
        if not symptoms:
            return 0.0
        sym_lower = symptom.lower()
        if sym_lower in symptoms:
            return 1.0 / len(symptoms)
        return 0.0

    def information_gain(
        self,
        symptom: str,
        differential: Dict[str, float],
    ) -> float:
        """
        Compute IG(s) for a candidate symptom query given the current differential.

        Args:
            symptom:      Candidate symptom / question.
            differential: {disease: posterior_probability} dict (must sum to ~1).

        Returns:
            Information gain (float >= 0).
        """
        diseases = list(differential.keys())
        priors = [differential[d] for d in diseases]

        # Normalize if needed
        total = sum(priors)
        if total < 1e-12:
            return 0.0
        priors = [p / total for p in priors]

        # P(s) = Σ P(s|Di) * P(Di)
        p_s = sum(
            self.p_symptom_given_disease(symptom, d) * priors[i]
            for i, d in enumerate(diseases)
        )
        p_not_s = 1.0 - p_s

        # Prior entropy H(D)
        h_d = self.entropy(priors)

        # Posterior given symptom positive  — P(Di|s+) ∝ P(s|Di)*P(Di)
        if p_s > 1e-12:
            post_s = [
                self.p_symptom_given_disease(symptom, d) * priors[i] / p_s
                for i, d in enumerate(diseases)
            ]
        else:
            post_s = priors[:]

        # Posterior given symptom negative  — P(Di|¬s) ∝ P(¬s|Di)*P(Di)
        if p_not_s > 1e-12:
            post_not_s = []
            for i, d in enumerate(diseases):
                p_not_s_given_d = 1.0 - self.p_symptom_given_disease(symptom, d)
                post_not_s.append(p_not_s_given_d * priors[i] / p_not_s)
        else:
            post_not_s = priors[:]

        h_d_given_s = p_s * self.entropy(post_s) + p_not_s * self.entropy(post_not_s)
        ig = max(0.0, h_d - h_d_given_s)
        return ig

    # ── Batch ranking ─────────────────────────────────────────────────────────

    def rank_symptoms(
        self,
        candidate_symptoms: List[str],
        differential: Dict[str, float],
        already_asked: Optional[List[str]] = None,
        min_ig: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate symptoms by information gain (descending).

        Args:
            candidate_symptoms: Symptoms/questions to evaluate.
            differential:       Current {disease: probability} posterior.
            already_asked:      Symptoms already asked (excluded from ranking).
            min_ig:             Minimum IG threshold (filter out low-value symptoms).

        Returns:
            List of (symptom, ig_value) tuples sorted by descending IG.
        """
        asked = set(s.lower() for s in (already_asked or []))
        results = []
        for sym in candidate_symptoms:
            if sym.lower() in asked:
                continue
            ig = self.information_gain(sym, differential)
            if ig >= min_ig:
                results.append((sym, ig))
        results.sort(key=lambda x: -x[1])
        return results

    def top_k_symptoms(
        self,
        differential: Dict[str, float],
        k: int = 3,
        already_asked: Optional[List[str]] = None,
        min_ig: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """
        Automatically generate candidate symptoms from KG for all diseases in
        the differential, then rank and return the top-k.
        """
        candidates: set = set()
        for disease in differential:
            for sym in self._kg.get_symptoms(disease):
                candidates.add(sym)
        ranked = self.rank_symptoms(
            list(candidates), differential,
            already_asked=already_asked, min_ig=min_ig
        )
        return ranked[:k]

    def top_k_tests(
        self,
        differential: Dict[str, float],
        k: int = 3,
        already_ordered: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Rank KG-suggested tests by the information gain of their associated symptom.
        Returns (test_name, ig_proxy) tuples.
        """
        ordered = set(t.lower() for t in (already_ordered or []))
        test_scores: Dict[str, float] = {}
        for disease in differential:
            for test in self._kg.get_tests(disease):
                if test.lower() in ordered:
                    continue
                # Approximate IG by treating the test result as a symptom query
                ig = self.information_gain(test, differential)
                if test not in test_scores or ig > test_scores[test]:
                    test_scores[test] = ig
        ranked = sorted(test_scores.items(), key=lambda x: -x[1])
        return ranked[:k]
