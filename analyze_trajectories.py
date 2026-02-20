#!/usr/bin/env python3
"""
analyze_trajectories.py
=======================
Post-hoc analysis of JSON trajectory files produced by agentclinic.py.

Usage:
    python analyze_trajectories.py --input_dir ./trajectories/
    python analyze_trajectories.py --input_dir ./trajectories/ --output summary_report.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_trajectories(input_dir):
    trajs = []
    paths = sorted(Path(input_dir).glob("trajectory_*.json"))
    if not paths:
        raise FileNotFoundError(f"No trajectory_*.json files found in {input_dir}")
    for p in paths:
        with open(p) as f:
            trajs.append(json.load(f))
    print(f"Loaded {len(trajs)} trajectories from {input_dir}")
    return trajs


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory features
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(traj):
    turns = traj.get("turns", [])
    doctor_turns    = [t for t in turns if t["role"] == "doctor"]
    patient_turns   = [t for t in turns if t["role"] == "patient"]
    meas_turns      = [t for t in turns if t["role"] == "measurement"]
    test_requests   = [t for t in turns if t["turn_type"] == "test_request"]

    # Did the doctor issue DIAGNOSIS READY?
    diag_ready = traj.get("diagnosis_ready_issued", False)

    # Turn number at which DIAGNOSIS READY was issued (None if never)
    diag_turn = None
    for t in turns:
        if t["turn_type"] == "diagnosis":
            diag_turn = t["turn_id"]
            break

    return {
        "scenario_id":           traj.get("scenario_id"),
        "dataset":               traj.get("dataset"),
        "correct_diagnosis":     traj.get("correct_diagnosis"),
        "final_diagnosis":       traj.get("final_diagnosis") or "",
        "is_correct":            traj.get("is_correct"),
        "diagnosis_ready":       diag_ready,
        "diagnosis_turn":        diag_turn,
        "total_turns":           traj.get("total_turns", len(turns)),
        "n_doctor_turns":        len(doctor_turns),
        "n_patient_turns":       len(patient_turns),
        "n_measurement_turns":   len(meas_turns),
        "n_tests_requested":     len(test_requests),
        "tests_requested":       traj.get("tests_requested", []),
        "doctor_llm":            traj.get("models", {}).get("doctor"),
        "patient_llm":           traj.get("models", {}).get("patient"),
        "doctor_bias":           traj.get("biases", {}).get("doctor"),
        "patient_bias":          traj.get("biases", {}).get("patient"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze(trajs):
    feats = [extract_features(t) for t in trajs]
    n = len(feats)

    correct           = [f for f in feats if f["is_correct"]]
    incorrect         = [f for f in feats if not f["is_correct"]]
    no_diag_ready     = [f for f in feats if not f["diagnosis_ready"]]
    gave_up           = [f for f in no_diag_ready]  # same: never typed DIAGNOSIS READY

    accuracy          = len(correct) / n if n else 0
    avg_turns         = sum(f["total_turns"] for f in feats) / n if n else 0
    avg_tests         = sum(f["n_tests_requested"] for f in feats) / n if n else 0

    # Turn at which correct vs incorrect scenarios were diagnosed
    correct_diag_turns  = [f["diagnosis_turn"] for f in correct  if f["diagnosis_turn"] is not None]
    incorrect_diag_turns= [f["diagnosis_turn"] for f in incorrect if f["diagnosis_turn"] is not None]
    avg_diag_turn_correct   = (sum(correct_diag_turns)  / len(correct_diag_turns))  if correct_diag_turns  else None
    avg_diag_turn_incorrect = (sum(incorrect_diag_turns)/ len(incorrect_diag_turns)) if incorrect_diag_turns else None

    # Most commonly requested tests
    all_tests = []
    for f in feats:
        all_tests.extend(f["tests_requested"])
    test_freq = Counter(all_tests).most_common(15)

    # Most common wrong diagnoses
    wrong_diags = [f["final_diagnosis"] for f in incorrect if f["final_diagnosis"] and "did not issue" not in f["final_diagnosis"]]
    wrong_diag_freq = Counter(wrong_diags).most_common(10)

    # Correct diagnoses in wrong cases (what should they have said?)
    correct_diags_when_wrong = Counter(f["correct_diagnosis"] for f in incorrect).most_common(10)

    # Bias breakdown
    bias_results = defaultdict(lambda: {"correct": 0, "total": 0})
    for f in feats:
        key = f"doctor={f['doctor_bias']}, patient={f['patient_bias']}"
        bias_results[key]["total"] += 1
        if f["is_correct"]:
            bias_results[key]["correct"] += 1

    return {
        "summary": {
            "total_scenarios":      n,
            "correct":              len(correct),
            "incorrect":            len(incorrect),
            "accuracy_pct":         round(accuracy * 100, 1),
            "never_issued_diag_ready": len(gave_up),
            "avg_total_turns":      round(avg_turns, 1),
            "avg_tests_requested":  round(avg_tests, 2),
            "avg_turn_when_correct_diag":   round(avg_diag_turn_correct, 1)   if avg_diag_turn_correct   else None,
            "avg_turn_when_incorrect_diag": round(avg_diag_turn_incorrect, 1) if avg_diag_turn_incorrect else None,
        },
        "failure_modes": {
            "gave_up_without_diagnosis": len(gave_up),
            "wrong_diagnosis_issued":    len(incorrect) - len(gave_up),
        },
        "most_common_tests_requested": test_freq,
        "most_common_wrong_diagnoses": wrong_diag_freq,
        "correct_answers_in_failed_cases": correct_diags_when_wrong,
        "results_by_bias": {k: {"accuracy_pct": round(v["correct"]/v["total"]*100, 1) if v["total"] else 0,
                                 "correct": v["correct"], "total": v["total"]}
                            for k, v in bias_results.items()},
        "per_scenario": feats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

def pretty_print(results):
    s = results["summary"]
    fm = results["failure_modes"]
    print("\n" + "="*60)
    print("  AgentClinic Trajectory Analysis")
    print("="*60)
    print(f"  Scenarios analysed : {s['total_scenarios']}")
    print(f"  Accuracy           : {s['correct']}/{s['total_scenarios']} = {s['accuracy_pct']}%")
    print(f"  Never issued DIAG  : {s['never_issued_diag_ready']}")
    print(f"  Avg total turns    : {s['avg_total_turns']}")
    print(f"  Avg tests ordered  : {s['avg_tests_requested']}")
    if s['avg_turn_when_correct_diag'] is not None:
        print(f"  Avg turn (correct) : {s['avg_turn_when_correct_diag']}")
    if s['avg_turn_when_incorrect_diag'] is not None:
        print(f"  Avg turn (wrong)   : {s['avg_turn_when_incorrect_diag']}")

    print("\n── Failure Modes ────────────────────────────────────────")
    print(f"  Gave up (no DIAGNOSIS READY) : {fm['gave_up_without_diagnosis']}")
    print(f"  Issued wrong diagnosis       : {fm['wrong_diagnosis_issued']}")

    print("\n── Most Requested Tests ─────────────────────────────────")
    for test, cnt in results["most_common_tests_requested"][:10]:
        print(f"  {cnt:3d}x  {test}")

    print("\n── Most Common Wrong Diagnoses ──────────────────────────")
    for diag, cnt in results["most_common_wrong_diagnoses"][:10]:
        print(f"  {cnt:3d}x  {diag[:70]}")

    print("\n── What Should Have Been Diagnosed (failed cases) ───────")
    for diag, cnt in results["correct_answers_in_failed_cases"][:10]:
        print(f"  {cnt:3d}x  {diag[:70]}")

    print("\n── Results by Bias Config ───────────────────────────────")
    for bias_key, v in results["results_by_bias"].items():
        print(f"  [{bias_key}]  {v['correct']}/{v['total']} = {v['accuracy_pct']}%")

    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze AgentClinic trajectory JSON files")
    parser.add_argument("--input_dir",  type=str, default="./trajectories", help="Directory containing trajectory_*.json files")
    parser.add_argument("--output",     type=str, default=None,             help="Optional: path to write summary_report.json")
    args = parser.parse_args()

    trajs   = load_trajectories(args.input_dir)
    results = analyze(trajs)
    pretty_print(results)

    if args.output:
        # Serialise Counter tuples to dicts for JSON
        out = json.loads(json.dumps(results, default=str))
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
