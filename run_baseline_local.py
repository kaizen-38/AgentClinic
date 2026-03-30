"""
run_baseline_local.py
─────────────────────────────────────────────────────────────────────────────
Vanilla AgentClinic DoctorAgent baseline — self-contained, no replicate dep.

Mirrors the exact DoctorAgent logic from agentclinic.py:
  - Same system prompt (turn counter, examiner info, REQUEST TEST format)
  - Same inference loop (raw LLM call, full dialogue history)
  - No structured pipeline, no KG, no hypothesis register

Use this to compare against SDRP (run_sdrp_local.py) on the same datasets.

Usage:
    python run_baseline_local.py --api_key sk-... --use_openai --dataset MedQA
    python run_baseline_local.py --api_key VOYAGER_KEY --dataset MedQA
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

VOYAGER_BASE    = "https://openai.rc.asu.edu/v1"
VOYAGER_DOCTOR  = "qwen3-30b-a3b-instruct-2507"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario loaders (identical to run_sdrp_local.py)
# ─────────────────────────────────────────────────────────────────────────────

class _Scenario:
    def __init__(self, d: dict):
        osce = d["OSCE_Examination"]
        self.diagnosis   = osce["Correct_Diagnosis"]
        self.patient_info = osce["Patient_Actor"]
        self.examiner_info = osce["Objective_for_Doctor"]
        phys = osce.get("Physical_Examination_Findings", {})
        phys["tests"] = osce["Test_Results"]
        self.exam_info = phys
    def patient_information(self):  return self.patient_info
    def examiner_information(self): return self.examiner_info
    def exam_information(self):     return self.exam_info
    def diagnosis_information(self): return self.diagnosis


class _ScenarioNEJM:
    def __init__(self, d: dict):
        self.diagnosis     = next(a["text"] for a in d["answers"] if a["correct"])
        self.patient_info  = d["patient_info"]
        self.physical_exams = d["physical_exams"]
        self.answers       = d["answers"]
    def patient_information(self):  return self.patient_info
    def examiner_information(self): return "What is the most likely diagnosis?"
    def exam_information(self):     return self.physical_exams
    def diagnosis_information(self): return self.diagnosis


def _load_scenarios(dataset: str):
    files = {
        "MedQA":     "agentclinic_medqa.jsonl",
        "MedQA_Ext": "agentclinic_medqa_extended.jsonl",
        "NEJM":      "agentclinic_nejm.jsonl",
        "NEJM_Ext":  "agentclinic_nejm_extended.jsonl",
    }
    with open(files[dataset]) as f:
        rows = [json.loads(line) for line in f]
    if dataset in ("NEJM", "NEJM_Ext"):
        return [_ScenarioNEJM(r) for r in rows]
    return [_Scenario(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight LLM helper
# ─────────────────────────────────────────────────────────────────────────────

def _chat(system: str, user: str, model: str, max_tokens: int = 300) -> str:
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
            print(f"  [retry {attempt+1}/4]: {e}")
            time.sleep(wait)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla DoctorAgent — exact mirror of agentclinic.py DoctorAgent
# ─────────────────────────────────────────────────────────────────────────────

class VanillaDoctorAgent:
    """
    Reproduces the exact DoctorAgent from agentclinic.py lines 601-677.
    System prompt is identical. No structured pipeline.
    """
    def __init__(self, scenario, model: str, max_infs: int = 20):
        self.model     = model
        self.MAX_INFS  = max_infs
        self.infs      = 0
        self.agent_hist = ""
        self.presentation = scenario.examiner_information()

    def system_prompt(self) -> str:
        return (
            f"You are a doctor named Dr. Agent who only responds in the form of dialogue. "
            f"You are inspecting a patient who you will ask questions in order to understand "
            f"their disease. You are only allowed to ask {self.MAX_INFS} questions total before "
            f"you must make a decision. You have asked {self.infs} questions so far. "
            f"You can request test results using the format \"REQUEST TEST: [test]\". "
            f"For example, \"REQUEST TEST: Chest_X-Ray\". "
            f"Your dialogue will only be 1-3 sentences in length. "
            f"Once you have decided to make a diagnosis please type "
            f"\"DIAGNOSIS READY: [diagnosis here]\""
            f"\n\nBelow is all of the information you have. {self.presentation}. "
            f"\n\nRemember, you must discover their disease by asking them questions. "
            f"You are also able to provide exams."
        )

    def inference(self, patient_response: str) -> str:
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"
        prompt = (
            f"\nHere is a history of your dialogue: {self.agent_hist}"
            f"\n Here was the patient response: {patient_response}"
            f"Now please continue your dialogue\nDoctor: "
        )
        answer = _chat(self.system_prompt(), prompt, self.model)
        self.agent_hist += patient_response + "\n\n" + answer + "\n\n"
        self.infs += 1
        return answer


# ─────────────────────────────────────────────────────────────────────────────
# Patient / Measurement (identical to run_sdrp_local.py)
# ─────────────────────────────────────────────────────────────────────────────

class PatientAgent:
    def __init__(self, scenario, model: str):
        self._symptoms = scenario.patient_information()
        self._hist     = ""
        self._model    = model
        self._sys = (
            "You are a patient in a clinic. Respond only as dialogue (1-3 sentences). "
            "Do NOT reveal your diagnosis. Only reveal symptoms if asked. "
            f"Your information: {self._symptoms}"
        )

    def inference_patient(self, doctor_text: str) -> str:
        prompt = f"Dialogue so far:\n{self._hist}\n\nDoctor just said: {doctor_text}\n\nPatient:"
        answer = _chat(self._sys, prompt, self._model)
        self._hist += f"Doctor: {doctor_text}\nPatient: {answer}\n\n"
        return answer

    def add_hist(self, text: str):
        self._hist += text + "\n\n"


class MeasurementAgent:
    def __init__(self, scenario, model: str):
        self._info  = scenario.exam_information()
        self._hist  = ""
        self._model = model
        self._sys = (
            "You are a medical test reader. When the doctor requests a test, return "
            "the result in format 'RESULTS: [result here]'. "
            f"Available data: {self._info}. "
            "If the test is not in your data, respond with: RESULTS: NORMAL READINGS."
        )

    def inference_measurement(self, doctor_text: str) -> str:
        prompt = f"History:\n{self._hist}\n\nDoctor request: {doctor_text}\n\nMeasurement reader:"
        answer = _chat(self._sys, prompt, self._model)
        self._hist += doctor_text + "\n\n" + answer + "\n\n"
        return answer

    def add_hist(self, text: str):
        self._hist += text + "\n\n"


def compare_results(doctor_response: str, correct: str, model: str) -> bool:
    ans = _chat(
        "You determine if two diagnoses refer to the same disease. Respond ONLY with 'Yes' or 'No'.",
        f"Correct diagnosis: {correct}\nDoctor said: {doctor_response}\nSame disease?",
        model, max_tokens=5,
    )
    first_word = re.sub(r"[^a-z]", "", ans.lower().split()[0]) if ans.strip() else "no"
    return first_word == "yes"


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Vanilla AgentClinic baseline runner")
    p.add_argument("--api_key",          default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--use_openai",       action="store_true",
                   help="Use api.openai.com instead of Voyager")
    p.add_argument("--api_base",         default=VOYAGER_BASE)
    p.add_argument("--model",            default=VOYAGER_DOCTOR,
                   help="Doctor + support model")
    p.add_argument("--dataset",          default="MedQA",
                   choices=["MedQA", "MedQA_Ext", "NEJM", "NEJM_Ext"])
    p.add_argument("--num_scenarios",    type=int, default=None)
    p.add_argument("--start_scenario",   type=int, default=0)
    p.add_argument("--total_inferences", type=int, default=20)
    p.add_argument("--output_dir",       default="./trajectories/baseline")
    p.add_argument("--verbose",          action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not args.api_key:
        print("ERROR: provide --api_key or set OPENAI_API_KEY")
        sys.exit(1)

    api_base = None if args.use_openai else (args.api_base.strip() or None)

    openai.api_key = args.api_key
    if api_base:
        openai.api_base = api_base

    model = args.model

    scenarios = _load_scenarios(args.dataset)
    start = args.start_scenario
    end   = min(start + (args.num_scenarios or len(scenarios)), len(scenarios))
    run_scenarios = scenarios[start:end]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(f"  VANILLA BASELINE — {args.dataset}  ({len(run_scenarios)} scenarios)")
    print(f"  Backend : {'OpenAI (api.openai.com)' if api_base is None else api_base}")
    print(f"  Model   : {model}")
    print(f"  Turns   : {args.total_inferences}")
    print(f"  Output  : {args.output_dir}")
    print("=" * 64)

    total_correct = 0
    total_seen    = 0
    results       = []

    for idx, scenario in enumerate(run_scenarios):
        scenario_id  = start + idx
        correct_diag = str(scenario.diagnosis_information())
        case_id      = f"{args.dataset}_{scenario_id:04d}"

        print(f"\n{'─'*64}")
        print(f"  Case {scenario_id:04d}/{len(scenarios)-1}  |  Correct: {correct_diag}")
        print(f"{'─'*64}")

        patient_agent = PatientAgent(scenario, model=model)
        meas_agent    = MeasurementAgent(scenario, model=model)
        doctor_agent  = VanillaDoctorAgent(scenario, model=model, max_infs=args.total_inferences)

        pi_dialogue     = ""
        final_diagnosis = None
        case_start      = time.time()
        turns_used      = 0

        for turn_id in range(args.total_inferences):
            if turn_id == args.total_inferences - 1:
                pi_dialogue += " This is the final question. Please provide a diagnosis."

            doctor_response = doctor_agent.inference(pi_dialogue)
            turns_used += 1

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

            time.sleep(0.2)

        if final_diagnosis:
            correctness = compare_results(final_diagnosis, correct_diag, model)
        else:
            correctness     = False
            final_diagnosis = "(no diagnosis issued)"

        total_seen += 1
        if correctness:
            total_correct += 1

        elapsed = time.time() - case_start
        acc     = int(total_correct / total_seen * 100)
        status  = "CORRECT  ✓" if correctness else "INCORRECT ✗"
        print(f"  → {status}  |  Running acc: {acc}%  |  {elapsed:.1f}s  |  turns={turns_used}")

        results.append({
            "case_id":          case_id,
            "correct_diagnosis": correct_diag,
            "predicted":         final_diagnosis,
            "is_correct":        correctness,
            "turns_used":        turns_used,
            "elapsed_s":         round(elapsed, 1),
        })

        # Save trajectory
        traj_path = Path(args.output_dir) / f"trajectory_{case_id}.json"
        traj_path.write_text(json.dumps({
            "case_id": case_id,
            "correct_diagnosis": correct_diag,
            "predicted": final_diagnosis,
            "is_correct": correctness,
            "turns_used": turns_used,
            "dialogue_history": doctor_agent.agent_hist,
        }, indent=2))

    # Summary
    summary = {
        "model":         model,
        "dataset":       args.dataset,
        "total_cases":   total_seen,
        "correct":       total_correct,
        "accuracy_pct":  round(total_correct / max(total_seen, 1) * 100, 1),
        "avg_turns":     round(sum(r["turns_used"] for r in results) / max(len(results), 1), 1),
        "results":       results,
    }
    summary_path = Path(args.output_dir) / f"summary_{args.dataset}_{total_seen}cases.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*64}")
    print(f"  BASELINE COMPLETE")
    print(f"  Accuracy : {total_correct}/{total_seen} = {summary['accuracy_pct']}%")
    print(f"  Avg turns: {summary['avg_turns']}")
    print(f"  Summary  : {summary_path}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
