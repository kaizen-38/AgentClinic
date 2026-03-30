import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Hypothesis:
    disease: str
    confidence: float           # 0.0 – 1.0
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    kg_matched: bool = False    # Grounded in knowledge graph?
    first_mentioned_turn: int = 0
    last_updated_turn: int = 0

    def to_dict(self) -> dict:
        return {
            "disease": self.disease,
            "confidence": round(self.confidence, 4),
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "kg_matched": self.kg_matched,
            "first_mentioned_turn": self.first_mentioned_turn,
            "last_updated_turn": self.last_updated_turn,
        }


@dataclass
class HypothesisRegister:
    hypotheses: List[Hypothesis] = field(default_factory=list)
    # Snapshot of top-k hypothesis names at each turn (for stagnation detection)
    history: List[Dict] = field(default_factory=list)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def top(self) -> Optional[Hypothesis]:
        if not self.hypotheses:
            return None
        return sorted(self.hypotheses, key=lambda h: h.confidence, reverse=True)[0]

    def ranked(self) -> List[Hypothesis]:
        return sorted(self.hypotheses, key=lambda h: h.confidence, reverse=True)

    def top_k_names(self, k: int = 3) -> List[str]:
        return [h.disease for h in self.ranked()[:k]]

    def top_confidence(self) -> float:
        h = self.top()
        return h.confidence if h else 0.0

    def get(self, disease_name: str) -> Optional[Hypothesis]:
        for h in self.hypotheses:
            if h.disease.lower() == disease_name.lower():
                return h
        return None

    # ── Mutation ──────────────────────────────────────────────────────────────

    def record_snapshot(self, turn: int) -> None:
        snapshot = {
            "turn": turn,
            "hypotheses": [h.to_dict() for h in self.ranked()],
        }
        self.history.append(snapshot)

    def is_stagnant(self, window: int = 3) -> bool:
        """Return True if the top-3 differential is unchanged for `window` turns."""
        if len(self.history) < window:
            return False
        recent = self.history[-window:]
        top3_sets = [
            frozenset(h["disease"] for h in snap["hypotheses"][:3])
            for snap in recent
        ]
        return len(set(top3_sets)) == 1

    def normalize_confidences(self) -> None:
        """Ensure confidences sum to <= 1.0 by renormalizing if they exceed it."""
        total = sum(h.confidence for h in self.hypotheses)
        if total > 1.0:
            for h in self.hypotheses:
                h.confidence = h.confidence / total

    def to_dict_list(self) -> List[dict]:
        return [h.to_dict() for h in self.ranked()]

    def snapshot_copy(self) -> "HypothesisRegister":
        return copy.deepcopy(self)
