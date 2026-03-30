"""
Turn-level and case-level metrics collection.
Matches the error analysis framework used for trajectory post-hoc analysis.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TurnMetrics:
    turn: int
    phase: str
    action: str                         # "question" | "test_request" | "diagnosis"
    top_hypothesis: str = ""
    top_confidence: float = 0.0
    info_gain: float = 0.0              # IG of the selected question/test
    verifier_flagged: bool = False
    new_positives: List[str] = field(default_factory=list)
    new_negatives: List[str] = field(default_factory=list)


@dataclass
class CaseMetrics:
    case_id: str
    dataset: str
    correct_diagnosis: str = ""
    predicted_diagnosis: str = ""

    # ── Outcome ───────────────────────────────────────────────────────────────
    accuracy: bool = False

    # ── Turn usage ────────────────────────────────────────────────────────────
    turns_used: int = 0
    max_turns: int = 20
    diagnosis_turn: int = 0
    late_diagnosis: bool = False        # diagnosed at >= 90% of max_turns

    # ── Testing behaviour ─────────────────────────────────────────────────────
    tests_ordered: List[str] = field(default_factory=list)
    normal_results_count: int = 0       # tests returning normal/uninformative results

    # ── Reasoning quality ─────────────────────────────────────────────────────
    hypothesis_drift_detected: bool = False
    info_gain_per_turn: List[float] = field(default_factory=list)
    confidence_trajectory: List[float] = field(default_factory=list)

    # ── Error taxonomy ────────────────────────────────────────────────────────
    error_category: Optional[str] = None  # e.g. "knowledge_gap", "context_drift", ...

    # ── Turn-level detail ─────────────────────────────────────────────────────
    turn_metrics: List[TurnMetrics] = field(default_factory=list)

    def record_turn(self, tm: TurnMetrics) -> None:
        self.turn_metrics.append(tm)
        self.confidence_trajectory.append(tm.top_confidence)
        self.info_gain_per_turn.append(tm.info_gain)
        if tm.verifier_flagged:
            self.hypothesis_drift_detected = True

    def finalize(self) -> None:
        self.turns_used = len(self.turn_metrics)
        if self.turns_used > 0 and self.max_turns > 0:
            self.late_diagnosis = self.diagnosis_turn >= int(0.9 * self.max_turns)

    def avg_info_gain(self) -> float:
        vals = [v for v in self.info_gain_per_turn if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "correct_diagnosis": self.correct_diagnosis,
            "predicted_diagnosis": self.predicted_diagnosis,
            "accuracy": self.accuracy,
            "turns_used": self.turns_used,
            "max_turns": self.max_turns,
            "diagnosis_turn": self.diagnosis_turn,
            "late_diagnosis": self.late_diagnosis,
            "tests_ordered": self.tests_ordered,
            "normal_results_count": self.normal_results_count,
            "hypothesis_drift_detected": self.hypothesis_drift_detected,
            "avg_info_gain": self.avg_info_gain(),
            "info_gain_per_turn": self.info_gain_per_turn,
            "confidence_trajectory": self.confidence_trajectory,
            "error_category": self.error_category,
        }


class MetricsCollector:
    """Aggregates CaseMetrics across multiple scenarios and writes summary JSON."""

    def __init__(self, output_dir: str = "metrics/"):
        self._cases: List[CaseMetrics] = []
        self._output_dir = Path(output_dir)

    def add(self, case: CaseMetrics) -> None:
        self._cases.append(case)

    def summary(self) -> Dict:
        if not self._cases:
            return {}
        n = len(self._cases)
        correct = sum(1 for c in self._cases if c.accuracy)
        avg_turns = sum(c.turns_used for c in self._cases) / n
        avg_ig = sum(c.avg_info_gain() for c in self._cases) / n
        late = sum(1 for c in self._cases if c.late_diagnosis)
        drift = sum(1 for c in self._cases if c.hypothesis_drift_detected)

        # Error category breakdown
        error_counts: Dict[str, int] = {}
        for c in self._cases:
            cat = c.error_category or "none"
            error_counts[cat] = error_counts.get(cat, 0) + 1

        return {
            "n_cases": n,
            "accuracy": round(correct / n, 4),
            "avg_turns_used": round(avg_turns, 2),
            "avg_info_gain": round(avg_ig, 4),
            "late_diagnosis_rate": round(late / n, 4),
            "hypothesis_drift_rate": round(drift / n, 4),
            "error_categories": error_counts,
        }

    def save(self, filename: str = "metrics_summary.json") -> str:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / filename
        payload = {
            "summary": self.summary(),
            "cases": [c.to_dict() for c in self._cases],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return str(path)
