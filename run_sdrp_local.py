"""
run_sdrp_local.py
─────────────────────────────────────────────────────────────────────────────
Local benchmark runner for the SDRP pipeline.
Self-contained: does NOT import agentclinic.py (which has a replicate
dependency that breaks on Python 3.14).

Doctor      → SDRPDoctorAgent  (our pipeline)
Patient     → lightweight API call
Measurement → lightweight API call

Default backend: ASU Voyager (qwen3-235b)
    python run_sdrp_local.py --api_key YOUR_VOYAGER_KEY
    python run_sdrp_local.py --api_key YOUR_VOYAGER_KEY --dataset MedQA_Ext --num_scenarios 10
    python run_sdrp_local.py --api_key YOUR_VOYAGER_KEY --dataset NEJM_Ext --num_scenarios 5 --verbose

OpenAI backend:
    python run_sdrp_local.py --api_key sk-... --api_base "" --doctor_model gpt-4.1-mini

Flags:
    --api_key             API key (or set OPENAI_API_KEY env var)
    --api_base            API base URL (default: https://openai.rc.asu.edu/v1 for Voyager;
                          pass empty string "" to use OpenAI directly)
    --dataset             MedQA | MedQA_Ext | NEJM | NEJM_Ext  (default: MedQA)
    --num_scenarios       Cases to run (default: all)
    --start_scenario      Start index (default: 0)
    --total_inferences    Max turns per case (default: 20)
    --output_dir          Trajectory dir (default: ./trajectories/sdrp_voyager)
    --doctor_model        Doctor LLM (default: qwen3-235b-a22b-instruct-2507)
    --support_model       Support LLM for parse/verify (default: qwen3-30b-a3b-instruct-2507)
    --disable_kg          Ablation: turn off KG grounding
    --disable_info_gain   Ablation: turn off info-gain questions
    --disable_osce        Ablation: turn off OSCE state parsing
    --disable_verifier    Ablation: turn off per-turn verification
    --verbose             Print full dialogue
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).parent))

from medical_agent.config import PipelineConfig
from medical_agent.interface.agentclinic_adapter import SDRPDoctorAgent
from medical_agent.monitoring.metrics import MetricsCollector


# ─────────────────────────────────────────────────────────────────────────────
# Minimal scenario wrappers (mirrors agentclinic.py without its imports)
# ─────────────────────────────────────────────────────────────────────────────

class _Scenario:
    """Wraps a MedQA-style OSCE scenario dict."""
    def __init__(self, d: dict):
        osce = d["OSCE_Examination"]
        self.diagnosis = osce["Correct_Diagnosis"]
        self.patient_info = osce["Patient_Actor"]
        self.examiner_info = osce["Objective_for_Doctor"]
        phys = osce.get("Physical_Examination_Findings", {})
        phys["tests"] = osce["Test_Results"]
        self.exam_info = phys
    def patient_information(self): return self.patient_info
    def examiner_information(self): return self.examiner_info
    def exam_information(self): return self.exam_info
    def diagnosis_information(self): return self.diagnosis


class _ScenarioNEJM:
    """Wraps a NEJM-style scenario dict."""
    def __init__(self, d: dict):
        self.diagnosis = next(a["text"] for a in d["answers"] if a["correct"])
        self.patient_info = d["patient_info"]
        self.physical_exams = d["physical_exams"]
        self.answers = d["answers"]
    def patient_information(self): return self.patient_info
    def examiner_information(self): return "What is the most likely diagnosis?"
    def exam_information(self): return self.physical_exams
    def diagnosis_information(self): return self.diagnosis


def _load_scenarios(dataset: str):
    files = {
        "MedQA":     "agentclinic_medqa.jsonl",
        "MedQA_Ext": "agentclinic_medqa_extended.jsonl",
        "NEJM":      "agentclinic_nejm.jsonl",
        "NEJM_Ext":  "agentclinic_nejm_extended.jsonl",
    }
    path = files[dataset]
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    if dataset in ("NEJM", "NEJM_Ext"):
        return [_ScenarioNEJM(r) for r in rows]
    return [_Scenario(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight OpenAI helper (no retry, that's fine for patient/measurement)
# ─────────────────────────────────────────────────────────────────────────────

def _chat(system: str, user: str, model: str, max_tokens: int = 200) -> str:
    for attempt in range(4):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.05,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [openai retry {attempt+1}/4]: {e}")
            time.sleep(wait)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Minimal patient / measurement / moderator agents
# ─────────────────────────────────────────────────────────────────────────────

class PatientAgent:
    def __init__(self, scenario, model: str):
        self._symptoms = scenario.patient_information()
        self._hist = ""
        self._model = model
        self._sys = (
            "You are a patient in a clinic. Respond only as dialogue (1-3 sentences). "
            "Do NOT reveal your diagnosis. Only reveal symptoms if asked. "
            f"Your information: {self._symptoms}"
        )

    def inference_patient(self, doctor_text: str) -> str:
        prompt = (
            f"Dialogue so far:\n{self._hist}\n\n"
            f"Doctor just said: {doctor_text}\n\nPatient:"
        )
        answer = _chat(self._sys, prompt, self._model)
        self._hist += f"Doctor: {doctor_text}\nPatient: {answer}\n\n"
        return answer

    def add_hist(self, text: str):
        self._hist += text + "\n\n"


class MeasurementAgent:
    def __init__(self, scenario, model: str):
        self._info = scenario.exam_information()
        self._hist = ""
        self._model = model
        self._sys = (
            "You are a medical test reader. When the doctor requests a test, return "
            "the result in format 'RESULTS: [result here]'. "
            f"Available data: {self._info}. "
            "If the test is not in your data, respond with: RESULTS: NORMAL READINGS."
        )

    def inference_measurement(self, doctor_text: str) -> str:
        prompt = (
            f"History:\n{self._hist}\n\n"
            f"Doctor request: {doctor_text}\n\nMeasurement reader:"
        )
        answer = _chat(self._sys, prompt, self._model)
        self._hist += doctor_text + "\n\n" + answer + "\n\n"
        return answer

    def add_hist(self, text: str):
        self._hist += text + "\n\n"


def compare_results(doctor_response: str, correct: str, model: str) -> bool:
    """Return True if the doctor's diagnosis matches the correct answer."""
    ans = _chat(
        "You determine if two diagnoses refer to the same disease. "
        "Respond ONLY with 'Yes' or 'No'.",
        f"Correct diagnosis: {correct}\nDoctor said: {doctor_response}\nSame disease?",
        model,
        max_tokens=5,
    )
    first_word = re.sub(r"[^a-z]", "", ans.lower().split()[0]) if ans.strip() else "no"
    return first_word == "yes"


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

VOYAGER_BASE    = "https://openai.rc.asu.edu/v1"
VOYAGER_DOCTOR  = "qwen3-235b-a22b-instruct-2507"
VOYAGER_SUPPORT = "qwen3-30b-a3b-instruct-2507"


def parse_args():
    p = argparse.ArgumentParser(
        description="SDRP pipeline — local run (Voyager or OpenAI)"
    )
    p.add_argument("--api_key",           default=os.environ.get("OPENAI_API_KEY"),
                   help="API key for Voyager or OpenAI")
    # keep old name as alias so existing scripts don't break
    p.add_argument("--openai_api_key",    default=None,
                   help="Alias for --api_key (deprecated)")
    p.add_argument("--api_base",          default=VOYAGER_BASE,
                   help="API base URL. Defaults to Voyager.")
    p.add_argument("--use_openai",        action="store_true",
                   help="Use OpenAI directly (api.openai.com). Overrides --api_base.")
    p.add_argument("--dataset",           default="MedQA",
                   choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"])
    p.add_argument("--num_scenarios",     type=int, default=None)
    p.add_argument("--start_scenario",    type=int, default=0)
    p.add_argument("--total_inferences",  type=int, default=20)
    p.add_argument("--output_dir",        default="./trajectories/sdrp_voyager")
    p.add_argument("--doctor_model",      default=VOYAGER_DOCTOR)
    p.add_argument("--support_model",     default=VOYAGER_SUPPORT)
    p.add_argument("--request_timeout",    type=float, default=600.0,
                   help="Per-request LLM timeout in seconds (default 600 for 235b)")
    p.add_argument("--disable_kg",          action="store_true")
    p.add_argument("--disable_info_gain",   action="store_true")
    p.add_argument("--disable_osce",        action="store_true")
    p.add_argument("--disable_verifier",    action="store_true")
    p.add_argument("--verbose",             action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # --openai_api_key is the old alias; --api_key takes precedence
    api_key = args.api_key or args.openai_api_key
    if not api_key:
        print("ERROR: provide --api_key (or set OPENAI_API_KEY env var)")
        sys.exit(1)

    # --use_openai overrides api_base to point at api.openai.com
    if args.use_openai:
        api_base = None
    else:
        api_base = args.api_base.strip() if args.api_base else None
        if not api_base:
            api_base = None

    # Set globals so _chat() (patient/measurement agents) also uses the right endpoint
    openai.api_key = api_key
    if api_base:
        openai.api_base = api_base

    # Detect backend name for PipelineConfig
    if api_base and "openai.rc.asu.edu" in api_base:
        backend = "voyager"
    elif api_base:
        backend = "vllm"
    else:
        backend = "openai"

    config = PipelineConfig(
        doctor_model=args.doctor_model,
        support_model=args.support_model,
        model_backend=backend,
        api_base_url=api_base,
        api_key=api_key,
        max_turns=args.total_inferences,
        enable_kg_hypothesis=not args.disable_kg,
        enable_info_gain=not args.disable_info_gain,
        enable_osce_state=not args.disable_osce,
        enable_verifier=not args.disable_verifier,
        enable_confidence_termination=True,
        request_timeout=args.request_timeout,
        log_level="INFO",
        log_format="json",
        trajectory_output_dir=args.output_dir,
        metrics_output_dir=args.output_dir + "/metrics",
    )

    scenarios = _load_scenarios(args.dataset)
    start = args.start_scenario
    end   = start + (args.num_scenarios or len(scenarios))
    end   = min(end, len(scenarios))
    run_scenarios = scenarios[start:end]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    collector = MetricsCollector(output_dir=args.output_dir + "/metrics")

    print("=" * 64)
    print(f"  SDRP Local Run — {args.dataset}  ({len(run_scenarios)} scenarios)")
    print(f"  Backend       : {backend}  ({api_base or 'api.openai.com'})")
    print(f"  Doctor model  : {args.doctor_model}")
    print(f"  Support model : {args.support_model}")
    print(f"  Max turns     : {args.total_inferences}")
    print(f"  KG            : {'ON' if not args.disable_kg else 'OFF (ablation)'}")
    print(f"  Info-gain     : {'ON' if not args.disable_info_gain else 'OFF (ablation)'}")
    print(f"  OSCE state    : {'ON' if not args.disable_osce else 'OFF (ablation)'}")
    print(f"  Verifier      : {'ON' if not args.disable_verifier else 'OFF (ablation)'}")
    print(f"  Output dir    : {args.output_dir}")
    print("=" * 64)

    total_correct = 0
    total_seen    = 0

    for idx, scenario in enumerate(run_scenarios):
        scenario_id  = start + idx
        correct_diag = str(scenario.diagnosis_information())
        case_id      = f"{args.dataset}_{scenario_id:04d}"

        # NEJM: extract answer choices for label normalization
        answer_choices = None
        if hasattr(scenario, "answers"):
            answer_choices = [a["text"] for a in scenario.answers]

        print(f"\n{'─'*64}")
        print(f"  Case {scenario_id:04d}/{len(scenarios)-1}  |  Correct: {correct_diag}")
        print(f"{'─'*64}")

        patient_agent = PatientAgent(scenario, model=args.doctor_model)
        meas_agent    = MeasurementAgent(scenario, model=args.doctor_model)

        doctor_agent  = SDRPDoctorAgent(
            config=config,
            scenario=scenario,
            case_id=case_id,
            dataset=args.dataset,
            correct_diagnosis=correct_diag,
            answer_choices=answer_choices,
        )

        pi_dialogue     = ""
        final_diagnosis = None
        case_start      = time.time()

        for turn_id in range(args.total_inferences):
            if turn_id == args.total_inferences - 1:
                pi_dialogue += " This is the final question. Please provide a diagnosis."

            doctor_response = doctor_agent.inference(pi_dialogue)

            if args.verbose:
                pct = int((turn_id + 1) / args.total_inferences * 100)
                print(f"  Doctor [{pct:3d}%]: {doctor_response}")

            if "DIAGNOSIS READY" in doctor_response:
                final_diagnosis = doctor_response
                if not args.verbose:
                    print(f"  → Diagnosis at turn {turn_id+1}: {doctor_response}")
                break

            if "REQUEST TEST" in doctor_response:
                pi_dialogue = meas_agent.inference_measurement(doctor_response)
                if args.verbose:
                    print(f"  Measurement : {pi_dialogue}")
                patient_agent.add_hist(pi_dialogue)
            else:
                pi_dialogue = patient_agent.inference_patient(doctor_response)
                if args.verbose:
                    print(f"  Patient     : {pi_dialogue}")
                meas_agent.add_hist(pi_dialogue)

            time.sleep(0.3)   # gentle rate-limit

        if final_diagnosis:
            correctness = compare_results(
                final_diagnosis, correct_diag, args.doctor_model
            )
        else:
            correctness     = False
            final_diagnosis = "(no diagnosis issued)"

        total_seen += 1
        if correctness:
            total_correct += 1

        elapsed = time.time() - case_start
        acc     = int(total_correct / total_seen * 100)
        status  = "CORRECT  ✓" if correctness else "INCORRECT ✗"
        print(f"  → {status}  |  Running acc: {acc}%  |  {elapsed:.1f}s")

        traj_path = doctor_agent.finalize_case(correctness)
        print(f"  → Trajectory: {traj_path}")

        doctor_agent.orchestrator.metrics.predicted_diagnosis = final_diagnosis
        doctor_agent.orchestrator.metrics.accuracy            = correctness
        collector.add(doctor_agent.orchestrator.metrics)

    summary_path = collector.save(
        f"metrics_{args.dataset}_{total_seen}cases.json"
    )
    print(f"\n{'='*64}")
    print(f"  RUN COMPLETE")
    print(f"  Accuracy    : {total_correct}/{total_seen} = "
          f"{int(total_correct/max(total_seen,1)*100)}%")
    print(f"  Metrics     : {summary_path}")
    print(f"  Trajectories: {args.output_dir}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
