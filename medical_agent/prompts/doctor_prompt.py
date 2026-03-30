"""Prompt templates for the HypothesisEngine and DoctorDialogue modules."""

import json
from typing import Dict, List, Optional


HYPOTHESIS_SYSTEM = """\
You are a clinical differential diagnosis engine.

## Output format
Respond with ONLY valid JSON — no preamble, no markdown fences, nothing after the closing brace:
{
  "hypotheses": [
    {
      "disease": "<string>",
      "confidence": <float 0.0–1.0>,
      "supporting": ["<≤5-word phrase>", "<≤5-word phrase>"],
      "contradicting": ["<≤5-word phrase>"],
      "kg_matched": <bool>
    }
  ]
}

## Hard constraints
1. Output exactly the number of hypotheses requested, ordered highest→lowest confidence.
2. confidence values: each in [0.0, 1.0], sum across all hypotheses ≤ 1.0.
3. supporting: at most 2 items; contradicting: at most 1 item; each item ≤ 5 words.
4. Never drop a hypothesis solely because one test was negative (consider sensitivity/specificity).
5. Ambiguous or partial evidence → keep confidence 0.20–0.40 rather than dropping entirely.
6. No additional JSON keys. No text outside the JSON object.\
"""


def hypothesis_update_prompt(
    compact_state: str,
    kg_text: str,
    new_findings: str,
    compact_differential: str,
    max_hypotheses: int,
) -> str:
    return f"""\
<patient_state>
{compact_state}
</patient_state>

<knowledge_graph>
{kg_text}
</knowledge_graph>

<new_evidence>
{new_findings}
</new_evidence>

<current_differential>
{compact_differential}
</current_differential>

Update the differential to exactly {max_hypotheses} hypotheses.\
"""


DIALOGUE_SYSTEM = (
    "You are Dr. Agent, a thoughtful physician conducting a diagnostic interview. "
    "You ask clear, focused questions to gather clinical information. "
    "You are professional, empathetic, and concise. "
    "Your responses are 1-3 sentences maximum."
)


def doctor_dialogue_prompt(
    osce_state_json: str,
    top_question: str,
    phase: str,
    turn: int,
    max_turns: int,
    top_hypothesis: str,
    top_confidence: float,
    verifier_recommendation: str = "continue",
) -> str:
    phase_instruction = {
        "intake": "Focus on gathering the full history: chief complaint, onset, duration, severity, associated symptoms, PMH, medications, allergies, family and social history.",
        "investigation": f"Your leading hypothesis is {top_hypothesis} (confidence: {top_confidence:.0%}). Ask targeted questions or request tests to confirm or refute this.",
        "convergence": f"You are close to a diagnosis ({top_hypothesis}, {top_confidence:.0%} confidence). Seek key confirmatory or disconfirmatory evidence.",
        "diagnosis": "You are ready to make your diagnosis.",
    }.get(phase, "Gather clinical information systematically.")

    verifier_note = ""
    if verifier_recommendation == "reconsider_dropped":
        verifier_note = "\nIMPORTANT: A previously strong hypothesis may have been prematurely dropped. Consider revisiting it.\n"
    elif verifier_recommendation == "seek_new_evidence":
        verifier_note = "\nIMPORTANT: There are unexplained positive findings. Seek evidence to explain them.\n"

    return f"""You are conducting a diagnostic interview. Generate a natural, concise doctor response.

Current Phase: {phase} (Turn {turn}/{max_turns})
Phase Guidance: {phase_instruction}
{verifier_note}
Clinical State Summary:
{osce_state_json}

Highest Information-Gain Question to Address:
{top_question}

Generate a doctor's response that naturally incorporates the above question.
Your response should be 1-3 sentences. It may include ONE of:
  1. A conversational question to the patient
  2. A test request in format: "REQUEST TEST: [test_name]"
  3. A diagnosis (only when confident): "DIAGNOSIS READY: [diagnosis]"

Do NOT over-explain. Be direct and professional.
Output only the doctor's spoken words — no JSON, no labels.
"""
