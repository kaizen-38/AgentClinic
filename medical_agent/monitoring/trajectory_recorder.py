"""
Full trajectory recorder — saves per-case JSON files for post-hoc analysis.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrajectoryRecorder:
    case_id: str
    dataset: str
    correct_diagnosis: str
    output_dir: str = "trajectories/"

    # ── Fields populated during the run ──────────────────────────────────────
    predicted_diagnosis: str = ""
    correct: bool = False
    turns_used: int = 0
    tests_ordered: List[str] = field(default_factory=list)
    confidence_trajectory: List[float] = field(default_factory=list)
    hypothesis_history: List[List[Dict]] = field(default_factory=list)
    clinical_state_history: List[Dict] = field(default_factory=list)
    verifier_flags: List[Dict] = field(default_factory=list)
    info_gain_per_turn: List[float] = field(default_factory=list)
    dialogue_turns: List[Dict] = field(default_factory=list)
    error_category: Optional[str] = None

    def record_turn(
        self,
        *,
        turn: int,
        clinical_state_dict: Dict,
        hypotheses_list: List[Dict],
        top_confidence: float,
        info_gain: float,
        action: str,
        utterance: str,
        verifier_result: Optional[Dict] = None,
    ) -> None:
        self.clinical_state_history.append(clinical_state_dict)
        self.hypothesis_history.append(hypotheses_list)
        self.confidence_trajectory.append(round(top_confidence, 4))
        self.info_gain_per_turn.append(round(info_gain, 4))
        self.dialogue_turns.append({
            "turn": turn,
            "action": action,
            "utterance": utterance,
        })
        if verifier_result:
            self.verifier_flags.append({"turn": turn, **verifier_result})

    def record_test(self, test_name: str) -> None:
        if test_name not in self.tests_ordered:
            self.tests_ordered.append(test_name)

    def finalize(self, predicted: str, correct: bool) -> None:
        self.predicted_diagnosis = predicted
        self.correct = correct
        self.turns_used = len(self.dialogue_turns)

    def save(self) -> str:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Sanitize case_id for filename
        safe_id = str(self.case_id).replace("/", "_").replace(" ", "_")
        filename = f"trajectory_{safe_id}.json"
        full_path = output_path / filename

        payload = {
            "case_id": self.case_id,
            "dataset": self.dataset,
            "correct_diagnosis": self.correct_diagnosis,
            "predicted_diagnosis": self.predicted_diagnosis,
            "correct": self.correct,
            "turns_used": self.turns_used,
            "tests_ordered": self.tests_ordered,
            "confidence_trajectory": self.confidence_trajectory,
            "hypothesis_history": self.hypothesis_history,
            "clinical_state_history": self.clinical_state_history,
            "verifier_flags": self.verifier_flags,
            "info_gain_per_turn": self.info_gain_per_turn,
            "dialogue_turns": self.dialogue_turns,
            "error_category": self.error_category,
        }
        with open(full_path, "w") as f:
            json.dump(payload, f, indent=2)
        return str(full_path)
