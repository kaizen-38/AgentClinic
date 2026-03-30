"""Prompt templates for the Verifier module."""


VERIFIER_SYSTEM = (
    "You are a clinical quality-assurance agent reviewing a diagnostic workup. "
    "Your job is to identify inconsistencies, gaps, and errors in the reasoning. "
    "Be critical but fair — flag real problems, not hypothetical ones."
)


def verifier_check_prompt(
    osce_state_json: str,
    top_hypothesis: str,
    top_confidence: float,
    hypotheses_json: str,
    previous_top3: list,
    current_top3: list,
) -> str:
    dropped = [d for d in previous_top3 if d not in current_top3]
    dropped_text = ", ".join(dropped) if dropped else "none"

    return f"""You are reviewing the consistency of a diagnostic workup. Check for reasoning errors.

Current Clinical State:
{osce_state_json}

Top Diagnosis: {top_hypothesis} (confidence: {top_confidence:.2f})

Full Differential:
{hypotheses_json}

Hypotheses Dropped Since Last Turn: {dropped_text}

Answer the following questions:

1. Does {top_hypothesis} explain ALL confirmed positive findings?
2. Are there abnormal lab/imaging results that haven't been incorporated into the differential?
3. Were any previously strong hypotheses dropped? If so, was the evidence sufficient to rule them out?
4. Are there any positive findings that remain unexplained by the current differential?

Output a JSON object:
{{
  "is_consistent": true or false,
  "unexplained_positives": ["findings not explained by top hypothesis"],
  "ignored_abnormals": ["abnormal results not in differential reasoning"],
  "dropped_hypotheses_warning": ["hypotheses dropped without sufficient evidence"],
  "recommendation": "continue" | "reconsider_dropped" | "seek_new_evidence",
  "reasoning": "brief explanation of your assessment"
}}

Output ONLY valid JSON. Be concise. Only flag real issues, not speculative ones.
"""
