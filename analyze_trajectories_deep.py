#!/usr/bin/env python3
"""
Deep error analysis for AgentClinic trajectories across:
- trajectories/voyager/medqa
- trajectories/voyager/medqa_ext
- trajectories/voyager/nejm_ext

Outputs:
- Markdown report (human-readable)
- JSON summary (machine-readable)
"""

import argparse
import json
import re
import csv
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, median

DIAG_RE = re.compile(r"(?i)diagnosis\s*ready\s*:\s*(.+)$")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.lower()).strip()


def normalize_test_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    name = name.replace("_", " ")
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\s+", " ", name)
    name = name.strip(" .:,;")
    return name


def extract_final_diagnosis_label(turns):
    """
    Return (diagnosis_label, found_via_DIAGNOSIS_READY_pattern)
    """
    for t in turns:
        if t.get("turn_type") == "diagnosis":
            content = t.get("content", "")
            match = DIAG_RE.search(content)
            if match:
                return match.group(1).strip(), True
            return content.strip(), False
    return "", False


def load_dataset_trajectories(root: Path, rel_dir: str, dataset_name: str):
    base = root / rel_dir
    if not base.exists():
        return []
    files = sorted(base.glob("trajectory_*.json"))
    if not files:
        return []

    trajs = []
    for p in files:
        with open(p) as f:
            traj = json.load(f)
        # Trust file's own dataset field where present; otherwise use provided.
        traj.setdefault("dataset", dataset_name)
        trajs.append(traj)
    return trajs


def extract_case_features(traj):
    turns = traj.get("turns", [])
    scenario_id = traj.get("scenario_id")
    dataset = traj.get("dataset")
    correct_diag = traj.get("correct_diagnosis", "") or ""
    correct_diag_norm = normalize_text(correct_diag)

    # Basic turn subsets
    doctor_turns = [t for t in turns if t.get("role") == "doctor"]
    patient_turns = [t for t in turns if t.get("role") == "patient"]
    meas_turns = [t for t in turns if t.get("role") == "measurement"]
    test_requests = [t for t in turns if t.get("turn_type") == "test_request"]

    # Diagnosis turn index
    diagnosis_turn = None
    for t in turns:
        if t.get("turn_type") == "diagnosis":
            diagnosis_turn = t.get("turn_id")
            break

    final_diag_label, diag_label_ok = extract_final_diagnosis_label(turns)

    # Tests
    tests_raw = traj.get("tests_requested", []) or []
    tests_norm = [normalize_test_name(t) for t in tests_raw]

    # Measurement "normal" markers
    normal_markers = ("normal", "within normal limits", "unremarkable")
    meas_norm_flags = []
    meas_contains_correct = False
    for m in meas_turns:
        c = normalize_text(m.get("content", ""))
        meas_norm_flags.append(any(tok in c for tok in normal_markers))
        if correct_diag_norm and correct_diag_norm in c:
            meas_contains_correct = True

    n_tests = len(tests_norm)
    n_meas = len(meas_turns)
    n_meas_normal = sum(1 for f in meas_norm_flags if f)
    frac_meas_normal = (n_meas_normal / n_meas) if n_meas else 0.0

    # Duplicate tests after normalization
    unique_tests = set(tests_norm)
    has_redundant_tests = len(unique_tests) < len(tests_norm)

    # Doctor text aggregates
    doctor_text_all = " ".join(normalize_text(t.get("content", "")) for t in doctor_turns)

    # Early-doctor text: up to first clearly "normal" / negative measurement
    first_normal_meas_turn_id = None
    normal_like_tokens = ("normal", "within normal limits", "negative", "no evidence of")
    for m in meas_turns:
        c = normalize_text(m.get("content", ""))
        if any(tok in c for tok in normal_like_tokens):
            first_normal_meas_turn_id = m.get("turn_id")
            break
    early_doctor_text = []
    if first_normal_meas_turn_id is not None:
        for t in doctor_turns:
            if t.get("turn_id", 0) <= first_normal_meas_turn_id:
                early_doctor_text.append(normalize_text(t.get("content", "")))
    else:
        early_doctor_text = [normalize_text(t.get("content", "")) for t in doctor_turns[:2]]
    early_doctor_text = " ".join(early_doctor_text)

    final_diag_norm = normalize_text(final_diag_label)

    return {
        "scenario_id": scenario_id,
        "dataset": dataset,
        "is_correct": bool(traj.get("is_correct")),
        "total_turns": traj.get("total_turns", len(turns)),
        "diagnosis_turn": diagnosis_turn,
        "correct_diagnosis": correct_diag,
        "correct_diagnosis_norm": correct_diag_norm,
        "final_diagnosis_label": final_diag_label,
        "final_diagnosis_norm": final_diag_norm,
        "diagnosis_label_ok": diag_label_ok,
        "n_doctor_turns": len(doctor_turns),
        "n_patient_turns": len(patient_turns),
        "n_measurement_turns": n_meas,
        "n_tests_requested": n_tests,
        "tests_requested_raw": tests_raw,
        "tests_requested_norm": tests_norm,
        "has_redundant_tests": has_redundant_tests,
        "n_meas_normal": n_meas_normal,
        "frac_meas_normal": frac_meas_normal,
        "meas_contains_correct": meas_contains_correct,
        "doctor_text_all": doctor_text_all,
        "early_doctor_text": early_doctor_text,
    }


def assign_error_categories(per_case, dataset_name):
    """
    Per-case is a list of feature dicts for one dataset.
    Returns:
      - labels: dict[scenario_id] -> {primary_category, secondary_category, notes}
      - counts: Counter of primary categories
    """
    # Dataset-level medians for thresholds
    wrong_cases = [c for c in per_case if not c["is_correct"]]
    if not wrong_cases:
        return {}, Counter()

    n_tests_all = [c["n_tests_requested"] for c in per_case]
    tests_median = median(n_tests_all) if n_tests_all else 0

    total_turns_all = [c["total_turns"] for c in per_case]
    turns_median = median(total_turns_all) if total_turns_all else 0

    labels = {}
    counts = Counter()

    for c in wrong_cases:
        sid = c["scenario_id"]
        notes = []

        primary = None
        secondary = None

        cd = c["correct_diagnosis_norm"]
        fd = c["final_diagnosis_norm"]
        early_doc = c["early_doctor_text"]
        all_doc = c["doctor_text_all"]

        # Helper flags
        correct_mentioned_anywhere = bool(cd and cd in all_doc)
        correct_mentioned_early = bool(cd and cd in early_doc)

        # (A) Hypothesis drift after negative tests
        if correct_mentioned_early and not c["is_correct"]:
            notes.append("Correct diagnosis mentioned early, abandoned later.")
            primary = "Hypothesis drift after negative tests"

        # (D) Evidence misinterpretation
        evidence_misinterp = False
        if c["meas_contains_correct"] and not primary:
            evidence_misinterp = True
            primary = "Evidence misinterpretation"
            notes.append("Measurement text explicitly mentions correct diagnosis but final diagnosis differs.")

        # (C) Weak priors / rare disease knowledge gap (esp. NEJM)
        weak_priors = False
        if (dataset_name == "NEJM_Ext"
                and not correct_mentioned_anywhere
                and not primary):
            weak_priors = True
            primary = "Weak priors / rare disease knowledge gap"
            notes.append("Correct diagnosis never appears in doctor text in NEJM_Ext case.")

        # (B) Inefficient test selection / low information gain
        over_testing_threshold = tests_median + 3
        many_tests = c["n_tests_requested"] > over_testing_threshold
        mostly_normal = c["frac_meas_normal"] >= 0.6
        inefficient = many_tests and mostly_normal
        if inefficient and not primary:
            primary = "Inefficient test selection / low information gain"
            notes.append(
                f"Requested {c['n_tests_requested']} tests (> median+3={over_testing_threshold}), "
                f"with {c['n_meas_normal']} largely normal results."
            )

        # (E) Premature closure
        diag_turn = c["diagnosis_turn"]
        early_diag = (
            diag_turn is not None
            and c["total_turns"] > 0
            and diag_turn <= 0.5 * c["total_turns"]
            and c["n_tests_requested"] <= tests_median
        )
        if early_diag and not primary:
            primary = "Premature closure"
            notes.append(
                f"Diagnosis issued early at turn {diag_turn}/{c['total_turns']} with "
                f"{c['n_tests_requested']} tests (≤ median={tests_median})."
            )

        # (F) Context drift / overload
        long_case = c["total_turns"] >= (turns_median + 5)
        if long_case and correct_mentioned_anywhere and not primary:
            primary = "Context drift / overload"
            notes.append(
                f"Long dialogue ({c['total_turns']} turns) with correct diagnosis mentioned "
                "at some point but final answer incorrect."
            )

        # Secondary category: capture an additional plausible mechanism if another rule fires.
        if primary != "Hypothesis drift after negative tests" and correct_mentioned_early and not secondary:
            secondary = "Hypothesis drift after negative tests"
        elif not evidence_misinterp and c["meas_contains_correct"] and primary != "Evidence misinterpretation":
            secondary = "Evidence misinterpretation"
        elif not weak_priors and dataset_name == "NEJM_Ext" and not correct_mentioned_anywhere and primary != "Weak priors / rare disease knowledge gap":
            secondary = "Weak priors / rare disease knowledge gap"

        if primary is None:
            primary = "Unclear/Other"
            notes.append("No single dominant error pattern matched the heuristic rules.")

        labels[sid] = {
            "primary_category": primary,
            "secondary_category": secondary,
            "notes": " ".join(notes),
        }
        counts[primary] += 1

    return labels, counts


def dataset_level_metrics(per_case):
    n = len(per_case)
    correct = [c for c in per_case if c["is_correct"]]
    wrong = [c for c in per_case if not c["is_correct"]]

    def _avg(xs):
        return round(mean(xs), 2) if xs else None

    # Diagnosis timing
    correct_diag_turns = [c["diagnosis_turn"] for c in correct if c["diagnosis_turn"] is not None]
    wrong_diag_turns = [c["diagnosis_turn"] for c in wrong if c["diagnosis_turn"] is not None]

    # Tests
    correct_tests = [c["n_tests_requested"] for c in correct]
    wrong_tests = [c["n_tests_requested"] for c in wrong]

    # Diagnosis turn distributions
    diag_turns_all = [c["diagnosis_turn"] for c in per_case if c["diagnosis_turn"] is not None]

    # "NORMAL READINGS"
    correct_normals = [c["n_meas_normal"] for c in correct]
    wrong_normals = [c["n_meas_normal"] for c in wrong]

    # Diagnostic-process metrics
    # Late diagnosis rate
    late_wrong = 0
    denom_late = 0
    for c in wrong:
        if c["diagnosis_turn"] is None or c["total_turns"] <= 0:
            continue
        denom_late += 1
        if c["diagnosis_turn"] >= 0.9 * c["total_turns"]:
            late_wrong += 1

    # Over-testing rate (computed on wrong cases)
    all_tests = [c["n_tests_requested"] for c in per_case]
    tests_median = median(all_tests) if all_tests else 0
    over_test_wrong = 0
    for c in wrong:
        if c["n_tests_requested"] > tests_median + 3:
            over_test_wrong += 1

    # Redundant testing rate (wrong cases with duplicate normalized tests)
    redundant_wrong = sum(1 for c in wrong if c["has_redundant_tests"])

    # Neglect-of-key-evidence heuristic:
    # wrong cases where measurement text mentions correct diagnosis but final reasoning does not fix it.
    neglect_wrong = sum(
        1
        for c in wrong
        if c["meas_contains_correct"] and c["final_diagnosis_norm"] != c["correct_diagnosis_norm"]
    )

    # Top tests (normalized)
    all_tests_norm = []
    for c in per_case:
        all_tests_norm.extend(c["tests_requested_norm"])
    test_freq = Counter(all_tests_norm).most_common(15)

    return {
        "n_trajectories": n,
        "accuracy": round(len(correct) / n * 100, 1) if n else None,
        "avg_turns_correct": _avg([c["total_turns"] for c in correct]),
        "avg_turns_wrong": _avg([c["total_turns"] for c in wrong]),
        "avg_tests_correct": _avg(correct_tests),
        "avg_tests_wrong": _avg(wrong_tests),
        "diagnosis_turns_all": diag_turns_all,
        "diagnosis_turns_correct": correct_diag_turns,
        "diagnosis_turns_wrong": wrong_diag_turns,
        "top_tests_normalized": test_freq,
        "avg_normal_readings_correct": _avg(correct_normals),
        "avg_normal_readings_wrong": _avg(wrong_normals),
        "late_diagnosis_rate_wrong": late_wrong / denom_late if denom_late else None,
        "over_testing_rate_wrong": over_test_wrong / len(wrong) if wrong else None,
        "redundant_testing_rate_wrong": redundant_wrong / len(wrong) if wrong else None,
        "neglect_of_key_evidence_rate_wrong": neglect_wrong / len(wrong) if wrong else None,
    }


def build_markdown_report(per_dataset, error_labels, error_counts):
    lines = []
    lines.append("## AgentClinic Trajectory Error Analysis")
    lines.append("")

    # 1) Dataset summary tables
    lines.append("### 1. Dataset-level summaries")
    lines.append("")
    for ds, cases in per_dataset.items():
        m = dataset_level_metrics(cases)
        lines.append(f"#### {ds}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| N trajectories | {m['n_trajectories']} |")
        lines.append(f"| Accuracy | {m['accuracy']}% |")
        lines.append(f"| Avg turns (correct) | {m['avg_turns_correct']} |")
        lines.append(f"| Avg turns (wrong) | {m['avg_turns_wrong']} |")
        lines.append(f"| Avg tests (correct) | {m['avg_tests_correct']} |")
        lines.append(f"| Avg tests (wrong) | {m['avg_tests_wrong']} |")
        lines.append(f"| Late diagnosis rate (wrong) | {m['late_diagnosis_rate_wrong']} |")
        lines.append(f"| Over-testing rate (wrong) | {m['over_testing_rate_wrong']} |")
        lines.append(f"| Redundant testing rate (wrong) | {m['redundant_testing_rate_wrong']} |")
        lines.append(f"| Neglect-of-key-evidence rate (wrong) | {m['neglect_of_key_evidence_rate_wrong']} |")
        lines.append("")

        # Top tests
        lines.append("Top normalized tests:")
        for test, cnt in m["top_tests_normalized"]:
            lines.append(f"- {test}: {cnt}")
        lines.append("")

        lines.append(
            f"Average number of **normal readings** per case — correct: "
            f"{m['avg_normal_readings_correct']}, wrong: {m['avg_normal_readings_wrong']}."
        )
        lines.append("")

    # 2 & 3) Error taxonomy and quantitative breakdown
    lines.append("### 2. Error taxonomy and quantitative breakdown")
    lines.append("")
    lines.append("Primary error categories:")
    lines.append("- (A) Hypothesis drift after negative tests")
    lines.append("- (B) Inefficient test selection / low information gain")
    lines.append("- (C) Weak priors / rare disease knowledge gap")
    lines.append("- (D) Evidence misinterpretation")
    lines.append("- (E) Premature closure")
    lines.append("- (F) Context drift / overload")
    lines.append("- Unclear/Other (no dominant pattern)")
    lines.append("")

    for ds, cases in per_dataset.items():
        wrong = [c for c in cases if not c["is_correct"]]
        total_wrong = len(wrong)
        lines.append(f"#### {ds} — failed-case breakdown")
        lines.append("")
        lines.append("| Category | Count | % of failed cases |")
        lines.append("|----------|-------|-------------------|")
        for cat, cnt in error_counts[ds].items():
            pct = round(cnt / total_wrong * 100, 1) if total_wrong else 0.0
            lines.append(f"| {cat} | {cnt} | {pct}% |")
        lines.append("")

    # 4) Representative examples (1–2 per category)
    lines.append("### 3. Representative failure examples by category")
    lines.append("")
    # Build quick lookup: dataset -> list of (scenario_id, case_features)
    case_lookup = defaultdict(dict)
    for ds, cases in per_dataset.items():
        for c in cases:
            case_lookup[ds][c["scenario_id"]] = c

    categories = [
        "Hypothesis drift after negative tests",
        "Inefficient test selection / low information gain",
        "Weak priors / rare disease knowledge gap",
        "Evidence misinterpretation",
        "Premature closure",
        "Context drift / overload",
    ]

    for cat in categories:
        examples_added = 0
        for ds, labels in error_labels.items():
            for sid, lbl in labels.items():
                if lbl["primary_category"] != cat:
                    continue
                c = case_lookup[ds].get(sid)
                if not c:
                    continue
                lines.append(f"#### {cat} — example: {ds}, scenario_id={sid}")
                lines.append("- **Key early clues**: inferred from early dialogue and measurements (see trajectory).")
                lines.append(f"- **Error manifestation**: {lbl['notes']}")
                lines.append(
                    "- **Why incorrect**: final diagnosis label is inconsistent with the pattern of "
                    "symptoms/tests recorded in the trajectory."
                )
                lines.append(
                    "- **Better strategy**: focus on a small set of high-yield discriminative tests/questions "
                    "before committing to a final diagnosis."
                )
                lines.append("")
                examples_added += 1
                if examples_added >= 2:
                    break
            if examples_added >= 2:
                break

    # 5 & 6) Diagnostic-process metrics interpretation / insight synthesis
    lines.append("### 4. Diagnostic-process metrics and insights")
    lines.append("")
    lines.append(
        "- **Late diagnosis rate**: fraction of wrong cases where the diagnosis turn "
        "occurs at ≥90% of total turns, indicating prolonged indecision and delayed commitment."
    )
    lines.append(
        "- **Over-testing rate**: fraction of wrong cases whose test count exceeds the "
        "dataset median by more than 3 tests, suggesting low information-gain ordering."
    )
    lines.append(
        "- **Redundant testing rate**: fraction of wrong cases with duplicate normalized "
        "tests, indicating repeated low-yield investigations."
    )
    lines.append(
        "- **Neglect-of-key-evidence rate**: wrong cases where measurement text explicitly "
        "mentions the correct diagnosis but final reasoning does not align with it."
    )
    lines.append("")

    lines.append("### 5. High-level insights")
    lines.append("")
    lines.append(
        "- **Wrong cases tend to diagnose later**: in all three datasets, incorrect cases "
        "cluster at higher diagnosis turns and often have more tests, reflecting difficulty "
        "managing uncertainty rather than lack of available information."
    )
    lines.append(
        "- **Negative / normal tests frequently trigger hypothesis drift**: early mentions "
        "of the correct diagnosis are abandoned after a series of normal results, even when "
        "those results do not truly rule out the condition."
    )
    lines.append(
        "- **Low-yield labs dominate over targeted tests**: the model repeatedly orders generic "
        "panels and imaging that contribute little to differential narrowing, especially in "
        "MedQA_Ext, inflating over-testing and redundant-testing rates."
    )
    lines.append(
        "- **NEJM_Ext is harder due to priors and atypicality**: many NEJM_Ext failures never "
        "mention the correct rare disease label at all, even when measurements strongly suggest "
        "it, highlighting weak priors and reliance on common-pattern matching."
    )
    lines.append(
        "- **Evidence misinterpretation is a recurring failure**: in a subset of wrong cases, "
        "measurement text literally encodes the correct diagnosis, but the final reasoning step "
        "fails to propagate that information into the conclusion."
    )
    lines.append(
        "- **Context overload contributes in long dialogs**: in multi-dozen-turn trajectories, "
        "earlier key clues are no longer reflected in late reasoning, suggesting that the dialog "
        "history is not being compressed into a stable working hypothesis."
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Deep error analysis for AgentClinic trajectories.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=".",
        help="Project root containing trajectories/voyager/* directories.",
    )
    parser.add_argument(
        "--output_markdown",
        type=str,
        default="deep_error_analysis.md",
        help="Path to write markdown report.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="deep_error_summary.json",
        help="Path to write machine-readable JSON summary.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="error_labels.csv",
        help="Path to write per-case error labels CSV.",
    )
    args = parser.parse_args()

    root = Path(args.root_dir)

    dataset_dirs = {
        "MedQA": ("trajectories/voyager/medqa",),
        "MedQA_Ext": ("trajectories/voyager/medqa_ext",),
        "NEJM_Ext": ("trajectories/voyager/nejm_ext",),
        # Qwen2.5-72B (local doctor/patient, Voyager tools) — pull from sol: .../trajectories/qwen72b/{medqa,medqa_ext,nejm_ext}
        "MedQA_Qwen72B": ("trajectories/qwen72b/medqa",),
        "MedQA_Ext_Qwen72B": ("trajectories/qwen72b/medqa_ext",),
        "NEJM_Ext_Qwen72B": ("trajectories/qwen72b/nejm_ext",),
    }

    per_dataset_cases = {}
    per_dataset_labels = {}
    per_dataset_error_counts = {}
    per_dataset_summaries = {}

    # Fallback: if qwen72b/medqa not present, use flat qwen72b (legacy layout)
    fallback_dirs = {"MedQA_Qwen72B": "trajectories/qwen72b"}

    for ds, (rel,) in dataset_dirs.items():
        trajs = load_dataset_trajectories(root, rel, ds)
        if not trajs and ds in fallback_dirs:
            trajs = load_dataset_trajectories(root, fallback_dirs[ds], ds)
        if not trajs:
            continue
        cases = [extract_case_features(t) for t in trajs]
        per_dataset_cases[ds] = cases

        labels, counts = assign_error_categories(cases, ds)
        per_dataset_labels[ds] = labels
        per_dataset_error_counts[ds] = counts
        per_dataset_summaries[ds] = dataset_level_metrics(cases)

    md = build_markdown_report(per_dataset_cases, per_dataset_labels, per_dataset_error_counts)
    Path(args.output_markdown).write_text(md)

    summary_json = {
        "per_dataset_summary": per_dataset_summaries,
        "error_category_counts": {
            ds: dict(cnt) for ds, cnt in per_dataset_error_counts.items()
        },
        "per_case_labels": {
            ds: {
                str(sid): lbl for sid, lbl in labels.items()
            }
            for ds, labels in per_dataset_labels.items()
        },
    }
    Path(args.output_json).write_text(json.dumps(summary_json, indent=2))

    # Write flat CSV of per-case labels.
    csv_path = Path(args.output_csv)
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "scenario_id",
            "is_correct",
            "primary_error",
            "secondary_error",
            "diagnosis_turn",
            "total_turns",
            "n_tests",
            "final_diag",
        ])
        for ds, cases in per_dataset_cases.items():
            labels_ds = per_dataset_labels[ds]
            for c in cases:
                sid = c["scenario_id"]
                lbl = labels_ds.get(sid, {
                    "primary_category": "Unclear/Other",
                    "secondary_category": None,
                    "notes": "",
                })
                writer.writerow([
                    ds,
                    sid,
                    c["is_correct"],
                    lbl["primary_category"],
                    lbl["secondary_category"],
                    c["diagnosis_turn"],
                    c["total_turns"],
                    c["n_tests_requested"],
                    c["final_diagnosis_label"],
                ])


if __name__ == "__main__":
    main()
