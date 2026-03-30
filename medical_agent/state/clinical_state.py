import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClinicalState:
    """
    OSCE-aligned clinical state.  Updated (via snapshot) after every turn.
    This is the single source of truth shared across all pipeline modules.
    """
    # ── Demographics ──────────────────────────────────────────────────────────
    age: Optional[str] = None
    gender: Optional[str] = None

    # ── Chief complaint ───────────────────────────────────────────────────────
    chief_complaint: str = ""

    # ── History of Present Illness ────────────────────────────────────────────
    hpi: str = ""

    # ── Symptoms: name → "positive" | "negative" | "unknown" ─────────────────
    symptoms: Dict[str, str] = field(default_factory=dict)

    # ── Social / medical history ──────────────────────────────────────────────
    past_medical_history: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    family_history: List[str] = field(default_factory=list)
    social_history: List[str] = field(default_factory=list)

    # ── Physical examination ──────────────────────────────────────────────────
    physical_exam: Dict[str, str] = field(default_factory=dict)

    # ── Lab / imaging results ─────────────────────────────────────────────────
    lab_results: Dict[str, str] = field(default_factory=dict)

    # ── Curated evidence lists (maintained by verifier) ───────────────────────
    key_positives: List[str] = field(default_factory=list)
    key_negatives: List[str] = field(default_factory=list)
    abnormal_results: List[str] = field(default_factory=list)
    unexplained_findings: List[str] = field(default_factory=list)

    # ── Metadata ──────────────────────────────────────────────────────────────
    turn_number: int = 0
    phase: str = "intake"  # intake | investigation | convergence | diagnosis
    tests_ordered: List[str] = field(default_factory=list)
    questions_asked: List[str] = field(default_factory=list)

    # ── Raw dialogue history (fallback when OSCE parsing is disabled) ─────────
    raw_dialogue_history: List[str] = field(default_factory=list)

    def snapshot(self) -> "ClinicalState":
        """Return a deep copy of the current state (immutable per turn)."""
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {
            "age": self.age,
            "gender": self.gender,
            "chief_complaint": self.chief_complaint,
            "hpi": self.hpi,
            "symptoms": self.symptoms,
            "past_medical_history": self.past_medical_history,
            "medications": self.medications,
            "allergies": self.allergies,
            "family_history": self.family_history,
            "social_history": self.social_history,
            "physical_exam": self.physical_exam,
            "lab_results": self.lab_results,
            "key_positives": self.key_positives,
            "key_negatives": self.key_negatives,
            "abnormal_results": self.abnormal_results,
            "unexplained_findings": self.unexplained_findings,
            "turn_number": self.turn_number,
            "phase": self.phase,
            "tests_ordered": self.tests_ordered,
            "questions_asked": self.questions_asked,
        }

    def positive_symptoms(self) -> List[str]:
        return [s for s, v in self.symptoms.items() if v == "positive"]

    def negative_symptoms(self) -> List[str]:
        return [s for s, v in self.symptoms.items() if v == "negative"]
