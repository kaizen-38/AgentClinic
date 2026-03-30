from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DEFAULT_KB_PATH = str(Path(__file__).parent / "knowledge" / "disease_symptom_kb.json")


@dataclass
class PipelineConfig:
    # ── Model configuration ───────────────────────────────────────────────────
    doctor_model: str = "qwen3-235b"        # Primary reasoning model
    support_model: str = "qwen3-30b"        # Lighter model for parsing/verification
    model_backend: str = "voyager"          # "voyager", "vllm", "openai"
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None

    # ── Turn budget ───────────────────────────────────────────────────────────
    max_turns: int = 20
    intake_turns: int = 2                   # Turns allocated to intake phase
    min_turns_before_diagnosis: int = 5     # Don't allow diagnosis before this

    # ── Hypothesis engine ─────────────────────────────────────────────────────
    max_hypotheses: int = 5
    kg_path: str = _DEFAULT_KB_PATH
    semantic_similarity_threshold: float = 0.85

    # ── Inquiry strategist ────────────────────────────────────────────────────
    top_k_questions: int = 3
    min_info_gain_threshold: float = 0.1

    # ── Termination ───────────────────────────────────────────────────────────
    confidence_threshold: float = 0.75
    stagnation_window: int = 5

    # ── Verifier ──────────────────────────────────────────────────────────────
    max_unexplained_findings: int = 2
    verification_frequency: int = 1

    # ── Ablation flags ────────────────────────────────────────────────────────
    enable_kg_hypothesis: bool = True
    enable_info_gain: bool = True
    enable_osce_state: bool = True
    enable_verifier: bool = True
    enable_confidence_termination: bool = True

    # ── API timeouts ──────────────────────────────────────────────────────────
    request_timeout: float = 600.0          # 10 min — needed for 235b model

    # ── Logging / output ──────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"                # "json" or "text"
    trajectory_output_dir: str = "trajectories/"
    metrics_output_dir: str = "metrics/"
