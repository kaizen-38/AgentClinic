"""Prompt templates for the IntakeParser module."""


INTAKE_SYSTEM = (
    "You are a clinical information extraction engine. "
    "Your task is to parse raw patient dialogue and extract structured clinical data "
    "following OSCE (Objective Structured Clinical Examination) format. "
    "Be precise and conservative — only extract information explicitly stated."
)


def intake_extract_prompt(patient_text: str, current_state_json: str) -> str:
    return f"""Given the patient's statement below and the current clinical state, extract any NEW clinical information.

Patient Statement:
\"\"\"{patient_text}\"\"\"

Current Clinical State (JSON):
{current_state_json}

Output a JSON object with ONLY the fields that have NEW or UPDATED information:
{{
  "age": "string or null",
  "gender": "string or null",
  "chief_complaint": "string or null",
  "hpi_additions": "new HPI details to append (string or null)",
  "new_positive_symptoms": ["list of newly confirmed positive symptoms"],
  "new_negative_symptoms": ["list of newly denied symptoms"],
  "medications": ["newly mentioned medications"],
  "allergies": ["newly mentioned allergies"],
  "past_medical_history": ["newly mentioned PMH items"],
  "family_history": ["newly mentioned FH items"],
  "social_history": ["newly mentioned SH items"],
  "physical_exam_findings": {{"exam_area": "finding"}},
  "reasoning": "brief explanation of what was extracted"
}}

Rules:
- Only include fields with new/updated data. Omit unchanged fields (use null).
- For symptoms, use lowercase standardized names (e.g., "chest pain" not "CHEST PAIN").
- If a patient denies a symptom, add it to new_negative_symptoms.
- If information is ambiguous, include it with a note in the value.
- Output ONLY valid JSON. No preamble or explanation outside the JSON block.
"""
