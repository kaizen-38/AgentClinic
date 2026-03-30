"""
medical_agent — SDRP (Structured Diagnostic Reasoning Pipeline)

A multi-agent medical diagnostic system that replaces AgentClinic's
monolithic DoctorAgent with a structured pipeline of specialized modules.

Quick start:
    from medical_agent.interface import SDRPDoctorAgent
    from medical_agent.config import PipelineConfig

    config = PipelineConfig(
        model_backend="voyager",
        api_key="...",
        api_base_url="https://openai.rc.asu.edu/v1",
        doctor_model="qwen3-235b",
        support_model="qwen3-30b",
        max_turns=20,
    )
    agent = SDRPDoctorAgent(config=config, scenario=scenario, dataset="MedQA")
    response = agent.inference(patient_text)
"""

__version__ = "0.1.0"
