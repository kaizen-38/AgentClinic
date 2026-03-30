"""
Knowledge Graph — loads the disease-symptom knowledge base JSON and exposes
fast query methods used by the HypothesisEngine and InformationGainCalculator.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class KnowledgeGraph:
    """
    Thin wrapper around the disease_symptom_kb.json knowledge base.

    Node types:
        - Disease nodes  (e.g., "Systemic Lupus Erythematosus")
        - Symptom nodes  (e.g., "malar rash")

    Edges: disease → symptom (bidirectional lookups supported).
    """

    def __init__(self, kb_path: Optional[str] = None):
        if kb_path is None:
            kb_path = str(
                Path(__file__).parent / "disease_symptom_kb.json"
            )
        self._kb_path = kb_path
        self._diseases: Dict[str, Dict] = {}
        self._symptom_to_diseases: Dict[str, List[str]] = {}
        self._loaded = False
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            with open(self._kb_path, "r") as f:
                data = json.load(f)
            self._diseases = data.get("diseases", {})
            # Build reverse index: symptom → [disease, ...]
            for disease, info in self._diseases.items():
                for symptom in info.get("symptoms", []):
                    s_lower = symptom.lower()
                    self._symptom_to_diseases.setdefault(s_lower, [])
                    self._symptom_to_diseases[s_lower].append(disease)
            self._loaded = True
        except Exception as e:
            print(f"[KnowledgeGraph] Failed to load KB from {self._kb_path}: {e}")
            self._loaded = False

    # ── Query API ─────────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._loaded

    def get_disease_info(self, disease: str) -> Optional[Dict]:
        return self._diseases.get(disease)

    def get_symptoms(self, disease: str) -> List[str]:
        info = self._diseases.get(disease, {})
        return [s.lower() for s in info.get("symptoms", [])]

    def get_tests(self, disease: str) -> List[str]:
        info = self._diseases.get(disease, {})
        return info.get("tests", [])

    def get_prevalence(self, disease: str) -> str:
        info = self._diseases.get(disease, {})
        return info.get("prevalence", "unknown")

    def diseases_for_symptom(self, symptom: str) -> List[str]:
        return self._symptom_to_diseases.get(symptom.lower(), [])

    def all_diseases(self) -> List[str]:
        return list(self._diseases.keys())

    def all_symptoms(self) -> List[str]:
        return list(self._symptom_to_diseases.keys())

    def symptom_count(self, disease: str) -> int:
        return len(self.get_symptoms(disease))

    # ── Subgraph extraction ───────────────────────────────────────────────────

    def subgraph_for_diseases(self, diseases: List[str]) -> List[Tuple[str, str]]:
        """
        Return (disease, symptom) triples for the given disease list.
        Used to inject KG context into LLM prompts.
        """
        triples = []
        for d in diseases:
            for s in self.get_symptoms(d):
                triples.append((d, s))
        return triples

    def triples_as_text(self, diseases: List[str]) -> str:
        triples = self.subgraph_for_diseases(diseases)
        lines = [f"  {d} → {s}" for d, s in triples]
        return "\n".join(lines) if lines else "(no KG data available)"

    def find_matching_diseases(
        self, positive_symptoms: List[str], threshold: int = 1
    ) -> List[Tuple[str, int]]:
        """
        Return (disease, overlap_count) for diseases sharing >= threshold
        symptoms with the given positive symptom list.
        """
        counts: Dict[str, int] = {}
        for sym in positive_symptoms:
            for disease in self.diseases_for_symptom(sym):
                counts[disease] = counts.get(disease, 0) + 1
        results = [(d, cnt) for d, cnt in counts.items() if cnt >= threshold]
        return sorted(results, key=lambda x: -x[1])

    def normalize_symptom(self, raw: str) -> str:
        """
        Simple fuzzy match: return the KG symptom key that best matches `raw`.
        Falls back to the lowercased raw string if no match found.
        """
        raw_lower = raw.lower().strip()
        # Exact match first
        if raw_lower in self._symptom_to_diseases:
            return raw_lower
        # Substring match
        for kg_sym in self._symptom_to_diseases:
            if kg_sym in raw_lower or raw_lower in kg_sym:
                return kg_sym
        return raw_lower
