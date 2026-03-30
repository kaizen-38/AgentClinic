"""Prompt templates for the DiagnosisResolver module."""


RESOLVER_SYSTEM = (
    "You are a senior physician making a final diagnosis. "
    "You have completed your clinical assessment and must now commit to a diagnosis. "
    "Reason carefully through all the evidence, then provide your final answer."
)


def resolver_prompt(
    osce_state_json: str,
    hypotheses_json: str,
    dataset: str,
    answer_choices: list = None,
) -> str:
    choices_block = ""
    if answer_choices:
        choices_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(answer_choices))
        choices_block = f"""
Answer Choices (select EXACTLY one of these):
{choices_text}

You MUST output the diagnosis as one of the exact strings above.
"""

    dataset_note = {
        "MedQA": "Use USMLE-appropriate diagnostic specificity (e.g., 'Type 2 Diabetes Mellitus', not just 'Diabetes').",
        "MedQA_Ext": "Use USMLE-appropriate diagnostic specificity.",
        "NEJM": "Select the exact answer choice string provided.",
        "NEJM_Ext": "Select the exact answer choice string provided.",
    }.get(dataset, "Use specific, clinically accurate diagnostic terminology.")

    return f"""You are making a final diagnosis after completing a clinical assessment.

Clinical State:
{osce_state_json}

Differential Diagnosis (ranked by confidence):
{hypotheses_json}
{choices_block}
Dataset: {dataset}
Label Format: {dataset_note}

Reason through the evidence step by step, then commit to your final diagnosis.

Output a JSON object:
{{
  "chain_of_thought": "Step-by-step reasoning through the evidence...",
  "diagnosis": "Final diagnosis label (normalized to dataset format)",
  "confidence": 0.XX,
  "key_supporting_evidence": ["evidence 1", "evidence 2", "evidence 3"],
  "alternative_considered": "The next most likely diagnosis and why it was ruled out"
}}

Rules:
- The "diagnosis" field must be specific and match dataset label format.
- Do not hedge — commit to one diagnosis.
- Output ONLY valid JSON.
"""
