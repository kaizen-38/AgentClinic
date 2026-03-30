"""
medical_agent — entry point CLI.

Provides two modes:
1. standalone  — run the pipeline against AgentClinic scenarios directly.
2. single-step — interactive single-turn mode for debugging.

Usage examples:
    python -m medical_agent.main --dataset MedQA --num_scenarios 10 \\
        --voyager_api_key KEY --voyager_api_base https://openai.rc.asu.edu/v1 \\
        --voyager_model qwen3-235b-a22b-instruct-2507 \\
        --voyager_lite_model qwen3-30b-a3b-instruct-2507

    python -m medical_agent.main --mode debug --query "Does the patient have chest pain?"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running as `python -m medical_agent.main` from the AgentClinic root
sys.path.insert(0, str(Path(__file__).parent.parent))

from medical_agent.config import PipelineConfig


def run_benchmark(args) -> None:
    """Run the full pipeline against an AgentClinic dataset."""
    import openai

    # Import AgentClinic components
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import agentclinic as ac

    from medical_agent.interface.agentclinic_adapter import SDRPDoctorAgent
    from medical_agent.monitoring.metrics import MetricsCollector

    # ── Config ────────────────────────────────────────────────────────────────
    config = PipelineConfig(
        doctor_model=args.voyager_model or "qwen3-235b",
        support_model=args.voyager_lite_model or "qwen3-30b",
        model_backend="voyager",
        api_base_url=args.voyager_api_base or "https://openai.rc.asu.edu/v1",
        api_key=args.voyager_api_key,
        max_turns=args.total_inferences,
        log_level=args.log_level,
        trajectory_output_dir=args.output_dir,
        # Ablation flags
        enable_kg_hypothesis=not args.disable_kg,
        enable_info_gain=not args.disable_info_gain,
        enable_osce_state=not args.disable_osce,
        enable_verifier=not args.disable_verifier,
        enable_confidence_termination=not args.disable_confidence_termination,
    )

    # ── Set up Voyager globals ────────────────────────────────────────────────
    ac._VOYAGER_API_BASE = config.api_base_url
    ac._VOYAGER_API_KEY = config.api_key
    ac._VOYAGER_MODEL_NAME = config.doctor_model
    ac._VOYAGER_LITE_MODEL_NAME = config.support_model

    # ── Load scenarios ────────────────────────────────────────────────────────
    dataset = args.dataset
    if dataset == "MedQA":
        loader = ac.ScenarioLoaderMedQA()
    elif dataset == "MedQA_Ext":
        loader = ac.ScenarioLoaderMedQAExtended()
    elif dataset == "NEJM_Ext":
        loader = ac.ScenarioLoaderNEJMExtended()
    elif dataset == "NEJM":
        loader = ac.ScenarioLoaderNEJM()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    num = args.num_scenarios or loader.num_scenarios
    collector = MetricsCollector(output_dir=args.metrics_dir)

    total_correct = 0
    total_seen = 0

    for scenario_id in range(min(num, loader.num_scenarios)):
        if args.start_scenario and scenario_id < args.start_scenario:
            continue

        scenario = loader.get_scenario(scenario_id)
        correct_diag = str(scenario.diagnosis_information())

        # Other agents
        patient_agent = ac.PatientAgent(scenario=scenario, backend_str=args.patient_llm)
        meas_agent = ac.MeasurementAgent(scenario=scenario, backend_str=args.meas_llm)

        # Our SDRP doctor agent
        doctor_agent = SDRPDoctorAgent(
            config=config,
            scenario=scenario,
            case_id=f"{dataset}_{scenario_id:04d}",
            dataset=dataset,
            correct_diagnosis=correct_diag,
        )

        print(f"\n{'='*60}")
        print(f"Case {scenario_id:04d} | {dataset} | Correct: {correct_diag}")
        print(f"{'='*60}")

        pi_dialogue = ""
        final_diagnosis = None
        case_start = time.time()

        for turn_id in range(args.total_inferences):
            if turn_id == args.total_inferences - 1:
                pi_dialogue += " This is the final question. Please provide a diagnosis."

            doctor_response = doctor_agent.inference(pi_dialogue)
            print(f"  Doctor [{turn_id+1}/{args.total_inferences}]: {doctor_response}")

            if "DIAGNOSIS READY" in doctor_response:
                final_diagnosis = doctor_response
                break

            if "REQUEST TEST" in doctor_response:
                pi_dialogue = meas_agent.inference_measurement(doctor_response)
                print(f"  Measurement: {pi_dialogue}")
                patient_agent.add_hist(pi_dialogue)
            else:
                pi_dialogue = patient_agent.inference_patient(doctor_response)
                print(f"  Patient: {pi_dialogue}")
                meas_agent.add_hist(pi_dialogue)

            time.sleep(0.5)

        # Evaluate
        if final_diagnosis:
            correctness = (
                ac.compare_results(final_diagnosis, correct_diag, args.mod_llm, None) == "yes"
            )
        else:
            correctness = False
            final_diagnosis = "(no diagnosis issued)"

        total_seen += 1
        if correctness:
            total_correct += 1

        acc = int(total_correct / total_seen * 100)
        elapsed = time.time() - case_start
        print(f"  → {'CORRECT' if correctness else 'INCORRECT'}  |  Running acc: {acc}%  |  {elapsed:.1f}s")

        # Flush trajectory
        traj_path = doctor_agent.finalize_case(correctness)
        print(f"  → Trajectory: {traj_path}")

        # Metrics
        doctor_agent.orchestrator.metrics.predicted_diagnosis = final_diagnosis
        doctor_agent.orchestrator.metrics.accuracy = correctness
        collector.add(doctor_agent.orchestrator.metrics)

    # Save summary
    summary_path = collector.save(f"metrics_{dataset}_{num}cases.json")
    print(f"\n[Summary saved → {summary_path}]")
    print(f"Final accuracy: {total_correct}/{total_seen} = {int(total_correct/max(total_seen,1)*100)}%")


def run_debug(args) -> None:
    """Interactive single-turn debug mode."""
    from medical_agent.config import PipelineConfig
    from medical_agent.orchestrator import Orchestrator
    from medical_agent.state.clinical_state import ClinicalState

    config = PipelineConfig(log_level="DEBUG", log_format="text")
    orch = Orchestrator(config, case_id="debug_case", dataset="MedQA")

    print("SDRP Debug Mode — type patient responses, 'quit' to exit")
    while True:
        user_input = input("\nPatient/measurement response: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        result = orch.step(user_input)
        print(f"\nDoctor: {result.utterance}")
        print(f"  Phase: {result.phase} | Top: {result.top_hypothesis} ({result.top_confidence:.2f})")
        if result.is_diagnosis:
            print("  *** DIAGNOSIS ISSUED ***")
            break


def main():
    parser = argparse.ArgumentParser(description="SDRP Medical Agent Pipeline")
    parser.add_argument("--mode", choices=["benchmark", "debug"], default="benchmark")
    parser.add_argument("--dataset", default="MedQA",
                        choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"])
    parser.add_argument("--num_scenarios", type=int, default=None)
    parser.add_argument("--start_scenario", type=int, default=None)
    parser.add_argument("--total_inferences", type=int, default=20)
    parser.add_argument("--output_dir", default="./trajectories/sdrp")
    parser.add_argument("--metrics_dir", default="./metrics/sdrp")
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    # API
    parser.add_argument("--voyager_api_key", default=os.environ.get("VOYAGER_API_KEY"))
    parser.add_argument("--voyager_api_base", default="https://openai.rc.asu.edu/v1")
    parser.add_argument("--voyager_model", default="qwen3-235b-a22b-instruct-2507")
    parser.add_argument("--voyager_lite_model", default="qwen3-30b-a3b-instruct-2507")
    # Other agents
    parser.add_argument("--patient_llm", default="voyager_lite")
    parser.add_argument("--meas_llm", default="voyager_lite")
    parser.add_argument("--mod_llm", default="voyager_lite")
    # Ablation flags
    parser.add_argument("--disable_kg", action="store_true")
    parser.add_argument("--disable_info_gain", action="store_true")
    parser.add_argument("--disable_osce", action="store_true")
    parser.add_argument("--disable_verifier", action="store_true")
    parser.add_argument("--disable_confidence_termination", action="store_true")

    args = parser.parse_args()

    if args.mode == "debug":
        run_debug(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
