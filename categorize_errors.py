#!/usr/bin/env python3
"""
categorize_errors.py
====================
Comprehensive error categorization across all AgentClinic trajectory runs.

Handles:
  Format A — Baseline    (has "is_correct" + "turns" list with role/content)
  Format B — SDRP        (has "correct" key, dialogue_turns, hypothesis_history)
  Format C — Baseline v2 (has "is_correct" + "dialogue_history" string + "predicted")

Outputs:
  error_categories_all.csv
  error_summary_all.json
  error_report.md
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TRAJECTORIES_ROOT = Path(__file__).parent / "trajectories"

LOW_YIELD_TESTS = [
    "cbc", "complete blood count", "bmp", "basic metabolic",
    "cmp", "comprehensive metabolic", "urinalysis", "cxr", "chest x-ray",
    "chest radiograph", "esr", "erythrocyte sedimentation",
    "crp", "c-reactive protein", "lft", "liver function",
    "kft", "kidney function", "renal function panel",
    "metabolic panel", "blood culture",
]

STOP_WORDS = {
    "a", "an", "the", "of", "in", "with", "and", "or", "is", "was",
    "for", "to", "due", "on", "by", "at", "as", "be", "are", "were",
    "has", "have", "had", "its", "it", "this", "that", "from", "not",
    "no", "do", "but", "if", "so", "can", "may", "we", "i", "my", "your",
}

# Mapping from directory path patterns to run names
# (We auto-detect by walking the tree, so this is just documentation.)

# ─────────────────────────────────────────────────────────────────────────────
# Directory discovery
# ─────────────────────────────────────────────────────────────────────────────

def detect_format(sample: dict) -> str:
    """
    Detect trajectory format from a sample document.
      A — has "is_correct" + "turns" (list with role/content)
      B — has "correct" (not "is_correct"), has "dialogue_turns"
      C — has "is_correct" + "dialogue_history" (raw string) + "predicted"
    """
    if "correct" in sample and "is_correct" not in sample:
        return "B"
    if "dialogue_history" in sample and "predicted" in sample:
        return "C"
    return "A"


def discover_runs(root: Path) -> list[tuple[str, Path, str]]:
    """
    Return list of (run_name, dir_path, format_hint) tuples.
    format_hint is 'A', 'B', or 'C' — detected from a sample file.
    """
    runs = []
    for p in sorted(root.rglob("trajectory_*.json")):
        # Skip metrics subdirectories
        if "metrics" in p.parts:
            continue
        parent = p.parent
        run_name = str(parent.relative_to(root))
        # Avoid duplicates — we collect directories, not individual files
        if any(r[1] == parent for r in runs):
            continue
        # Detect format from this sample file
        try:
            with open(p) as f:
                sample = json.load(f)
            fmt = detect_format(sample)
        except Exception:
            fmt = "A"
        runs.append((run_name, parent, fmt))
        print(f"  Found run: {run_name}  ({fmt})")
    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_run(dir_path: Path) -> list[dict]:
    trajs = []
    for p in sorted(dir_path.glob("trajectory_*.json")):
        try:
            with open(p) as f:
                trajs.append(json.load(f))
        except Exception as e:
            print(f"    WARNING: could not load {p}: {e}")
    return trajs


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_doctor_texts_A(traj: dict) -> list[str]:
    """All doctor turn content strings (Format A)."""
    return [
        t["content"] for t in traj.get("turns", [])
        if t.get("role") == "doctor"
    ]


def get_doctor_texts_B(traj: dict) -> list[str]:
    """All doctor utterances from dialogue_turns (Format B)."""
    return [
        t["utterance"] for t in traj.get("dialogue_turns", [])
        if t.get("action") in ("question", "diagnosis", "test")
    ]


def get_tests_A(traj: dict) -> list[str]:
    return traj.get("tests_requested", [])


def get_tests_B(traj: dict) -> list[str]:
    return traj.get("tests_ordered", [])


def is_low_yield(test_str: str) -> bool:
    s = test_str.lower()
    return any(kw in s for kw in LOW_YIELD_TESTS)


def count_low_yield(tests: list[str]) -> int:
    return sum(1 for t in tests if is_low_yield(t))


def has_test_repetition(tests: list[str]) -> bool:
    normalized = [t.lower().strip() for t in tests]
    return len(normalized) != len(set(normalized))


def keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def correct_dx_in_doctor_text(correct_dx: str, doctor_texts: list[str]) -> bool:
    """True if >2 non-stopword keywords from correct_dx appear in doctor dialogue."""
    dx_words = keywords(correct_dx)
    if len(dx_words) < 2:
        # Short diagnosis name — require at least 1 match
        if not dx_words:
            return False
        combined = " ".join(doctor_texts).lower()
        return any(w in combined for w in dx_words)
    combined = keywords(" ".join(doctor_texts))
    matches = dx_words & combined
    return len(matches) > 2


def find_anchor_bias(doctor_texts: list[str], correct_dx: str) -> bool:
    """
    True if a single WRONG disease name appears in >60% of doctor turns.
    We extract capitalized multi-word phrases as candidate disease mentions.
    """
    if not doctor_texts:
        return False
    # Extract candidate disease mentions: capitalized sequences of 1-4 words
    # Exclude generic phrases
    disease_pattern = re.compile(
        r'\b(?:[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,3})\b'
    )
    turn_mentions = []
    for text in doctor_texts:
        found = disease_pattern.findall(text)
        # Filter out noise
        filtered = [
            m for m in found
            if len(m) > 3
            and not m.lower().startswith(("thank", "hello", "i ", "the ", "you ", "we ", "can "))
        ]
        turn_mentions.append(filtered)

    # Count which disease appears in most turns
    disease_turn_count: Counter = Counter()
    all_diseases: set[str] = set()
    for mentions in turn_mentions:
        seen_in_turn = set(m.lower() for m in mentions)
        for d in seen_in_turn:
            disease_turn_count[d] += 1
        all_diseases.update(seen_in_turn)

    n_turns = len(doctor_texts)
    correct_lower = correct_dx.lower()

    for disease, count in disease_turn_count.items():
        if count / n_turns > 0.60:
            # Check it's not the correct diagnosis
            if correct_lower not in disease and disease not in correct_lower:
                return True
    return False


def find_diagnosis_turn_A(traj: dict) -> int | None:
    """Turn ID of the DIAGNOSIS READY turn in Format A."""
    for t in traj.get("turns", []):
        if t.get("turn_type") == "diagnosis":
            return t.get("turn_id")
    return None


def correct_dx_in_hypothesis_history(correct_dx: str, hyp_history: list[list[dict]]) -> bool:
    """True if correct diagnosis (fuzzy) appears anywhere in hypothesis_history."""
    dx_words = keywords(correct_dx)
    for turn_hyps in hyp_history:
        for h in turn_hyps:
            disease = h.get("disease", "").lower()
            # Exact or near-exact match
            if correct_dx.lower() in disease or disease in correct_dx.lower():
                return True
            # Keyword overlap: if >60% of dx keywords appear in disease name
            if dx_words:
                d_words = keywords(disease)
                overlap = dx_words & d_words
                if len(overlap) / len(dx_words) >= 0.6:
                    return True
    return False


def top_hypothesis_changes(hyp_history: list[list[dict]]) -> int:
    """Count how many times the top hypothesis (rank 0) changes disease name."""
    if not hyp_history:
        return 0
    prev = None
    changes = 0
    for turn_hyps in hyp_history:
        if not turn_hyps:
            continue
        top = turn_hyps[0].get("disease", "").lower()
        if prev is not None and top != prev:
            changes += 1
        prev = top
    return changes


def max_confidence_wrong(confidence_trajectory: list[float]) -> float:
    if not confidence_trajectory:
        return 0.0
    return max(confidence_trajectory)


def early_fixation(doctor_texts: list[str]) -> bool:
    """True if the first doctor utterance mentions a specific disease name."""
    if not doctor_texts:
        return False
    first = doctor_texts[0]
    # Look for capitalized multi-word medical terms (disease names)
    # Heuristic: contains a word like "gravis", "syndrome", "disease", "disorder",
    # "infection", "sclerosis", "carcinoma", "lymphoma", etc.
    medical_terms = re.compile(
        r'\b(?:gravis|syndrome|disease|disorder|infection|sclerosis|carcinoma|'
        r'lymphoma|leukemia|anemia|arthritis|hepatitis|pneumonia|tuberculosis|'
        r'diabetes|thyroiditis|nephritis|colitis|gastritis|myopathy|neuropathy|'
        r'vasculitis|sarcoidosis|amyloidosis|malignancy|cancer|tumor|tumour)\b',
        re.IGNORECASE
    )
    return bool(medical_terms.search(first))


# ─────────────────────────────────────────────────────────────────────────────
# Primary error classification
# ─────────────────────────────────────────────────────────────────────────────

def get_doctor_texts_C(traj: dict) -> list[str]:
    """
    Format C has dialogue_history as a raw alternating string.
    We treat odd-indexed lines (0-indexed) as doctor turns.
    """
    raw = traj.get("dialogue_history", "") or ""
    lines = [ln.strip() for ln in raw.strip().split("\n") if ln.strip()]
    # Odd indices (0-based) are doctor turns in an interleaved dialogue
    return [lines[i] for i in range(0, len(lines), 2)]


def classify_primary(traj: dict, fmt: str) -> tuple[str, dict]:
    """
    Returns (primary_error_code, metadata_dict).
    metadata_dict has keys used for secondary flag computation.
    """
    dataset = traj.get("dataset", "") or ""

    if fmt == "C":
        is_correct = traj.get("is_correct", False)
        correct_dx = traj.get("correct_diagnosis", "")
        predicted_dx = traj.get("predicted", "") or ""
        turns_used = traj.get("turns_used", 0)
        doctor_texts = get_doctor_texts_C(traj)
        # No structured tests in Format C
        tests: list[str] = []

        # Determine if diagnosis was issued
        no_diag_phrases = ["(no diagnosis issued)", "no diagnosis", "timed out", ""]
        diag_ready = bool(
            predicted_dx.strip()
            and predicted_dx.strip().lower() not in {"(no diagnosis issued)", "no diagnosis"}
        )

        meta = {
            "correct_dx": correct_dx,
            "final_dx": predicted_dx,
            "is_correct": is_correct,
            "total_turns": turns_used,
            "n_tests": 0,
            "tests": [],
            "doctor_texts": doctor_texts,
            "diagnosis_turn": turns_used if diag_ready else None,
            "diag_ready": diag_ready,
            "hyp_history": [],
            "verifier_flags": [],
            "confidence_trajectory": [],
            "fmt": "C",
        }

        # Step 1
        if is_correct:
            return "correct", meta

        # Step 2: no diagnosis
        if not diag_ready:
            return "no_diagnosis", meta

        # Step 3: premature (committed early — for C we use turns_used ≤ 3)
        if turns_used <= 3 and diag_ready:
            return "premature_diagnosis", meta

        # Step 5: anchor bias
        if find_anchor_bias(doctor_texts, correct_dx):
            return "anchor_bias", meta

        # Step 8: knowledge gap
        if "nejm" in dataset.lower():
            return "knowledge_gap", meta

        return "incorrect_other", meta

    if fmt == "A":
        is_correct = traj.get("is_correct", False)
        correct_dx = traj.get("correct_diagnosis", "")
        final_dx = traj.get("final_diagnosis", "") or ""
        diag_ready = traj.get("diagnosis_ready_issued", False)
        total_turns = traj.get("total_turns", 0)
        tests = get_tests_A(traj)
        doctor_texts = get_doctor_texts_A(traj)
        diagnosis_turn = find_diagnosis_turn_A(traj)

        meta = {
            "correct_dx": correct_dx,
            "final_dx": final_dx,
            "is_correct": is_correct,
            "total_turns": total_turns,
            "n_tests": len(tests),
            "tests": tests,
            "doctor_texts": doctor_texts,
            "diagnosis_turn": diagnosis_turn,
            "diag_ready": diag_ready,
            "hyp_history": [],
            "verifier_flags": [],
            "confidence_trajectory": [],
            "fmt": "A",
        }

        # Step 1
        if is_correct:
            return "correct", meta

        # Step 2: no diagnosis
        if not diag_ready:
            return "no_diagnosis", meta

        # Step 3: premature
        if diagnosis_turn is not None and diagnosis_turn <= 3:
            return "premature_diagnosis", meta

        # Steps 4 skipped (no hyp_history in Format A)

        # Step 5: anchor bias
        if find_anchor_bias(doctor_texts, correct_dx):
            return "anchor_bias", meta

        # Step 6: low yield testing
        if count_low_yield(tests) >= 3:
            return "low_yield_testing", meta

        # Step 7: over testing
        if len(tests) > 5:
            return "over_testing", meta

        # Step 8: knowledge gap (NEJM + wrong)
        if "nejm" in dataset.lower():
            return "knowledge_gap", meta

        return "incorrect_other", meta

    else:  # Format B
        is_correct = traj.get("correct", False)
        correct_dx = traj.get("correct_diagnosis", "")
        predicted_dx = traj.get("predicted_diagnosis", "") or ""
        turns_used = traj.get("turns_used", 0)
        tests = get_tests_B(traj)
        doctor_texts = get_doctor_texts_B(traj)
        hyp_history = traj.get("hypothesis_history", [])
        verifier_flags = traj.get("verifier_flags", [])
        confidence_traj = traj.get("confidence_trajectory", [])

        # Determine "diagnosis turn" for SDRP: find turn where action=diagnosis
        diagnosis_turn_sdrp = None
        for dt in traj.get("dialogue_turns", []):
            if dt.get("action") == "diagnosis":
                diagnosis_turn_sdrp = dt.get("turn")
                break

        meta = {
            "correct_dx": correct_dx,
            "final_dx": predicted_dx,
            "is_correct": is_correct,
            "total_turns": turns_used,
            "n_tests": len(tests),
            "tests": tests,
            "doctor_texts": doctor_texts,
            "diagnosis_turn": diagnosis_turn_sdrp,
            "diag_ready": bool(predicted_dx and predicted_dx.strip()
                               and "unknown" not in predicted_dx.lower()),
            "hyp_history": hyp_history,
            "verifier_flags": verifier_flags,
            "confidence_trajectory": confidence_traj,
            "fmt": "B",
        }

        # Step 1
        if is_correct:
            return "correct", meta

        # Step 2: no diagnosis
        if not predicted_dx or not predicted_dx.strip() or "unknown" in predicted_dx.lower():
            return "no_diagnosis", meta

        # Step 3: premature (committed at turn ≤3)
        if diagnosis_turn_sdrp is not None and diagnosis_turn_sdrp <= 3:
            return "premature_diagnosis", meta

        # Step 4: wrong differential (SDRP only)
        if hyp_history and not correct_dx_in_hypothesis_history(correct_dx, hyp_history):
            return "wrong_differential", meta

        # Step 5: anchor bias
        if find_anchor_bias(doctor_texts, correct_dx):
            return "anchor_bias", meta

        # Step 6: low yield testing
        if count_low_yield(tests) >= 3:
            return "low_yield_testing", meta

        # Step 7: over testing
        if len(tests) > 5:
            return "over_testing", meta

        # Step 8: knowledge gap
        if "nejm" in dataset.lower():
            return "knowledge_gap", meta

        return "incorrect_other", meta


# ─────────────────────────────────────────────────────────────────────────────
# Secondary flags
# ─────────────────────────────────────────────────────────────────────────────

def compute_secondary_flags(meta: dict) -> list[str]:
    flags = []
    fmt = meta["fmt"]
    is_correct = meta["is_correct"]

    # verifier_inconsistency (SDRP only)
    if fmt == "B":
        vf = meta.get("verifier_flags", [])
        if any(not f.get("is_consistent", True) for f in vf):
            flags.append("verifier_inconsistency")

    # hypothesis_drift (SDRP only)
    if fmt == "B":
        changes = top_hypothesis_changes(meta.get("hyp_history", []))
        if changes >= 3:
            flags.append("hypothesis_drift")

    # confidence_overload (SDRP only, wrong cases)
    if fmt == "B" and not is_correct:
        ct = meta.get("confidence_trajectory", [])
        if max_confidence_wrong(ct) > 0.85:
            flags.append("confidence_overload")

    # test_repetition
    tests = meta.get("tests", [])
    if has_test_repetition(tests):
        flags.append("test_repetition")

    # early_fixation
    if early_fixation(meta.get("doctor_texts", [])):
        flags.append("early_fixation")

    # late_correct_mention (wrong cases only)
    if not is_correct:
        if correct_dx_in_doctor_text(meta["correct_dx"], meta.get("doctor_texts", [])):
            flags.append("late_correct_mention")

    # never_requested_test
    if not meta.get("tests", []):
        flags.append("never_requested_test")

    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Per-trajectory row builder
# ─────────────────────────────────────────────────────────────────────────────

def infer_dataset(traj: dict, run_name: str) -> str:
    """Return dataset name, inferring from run_name if not in trajectory."""
    ds = traj.get("dataset") or ""
    if ds:
        return ds
    run_lower = run_name.lower()
    if "nejm_ext" in run_lower:
        return "NEJM_Ext"
    if "medqa_ext" in run_lower:
        return "MedQA_Ext"
    if "medqa" in run_lower or "nejm" not in run_lower:
        return "MedQA"
    return "unknown"


def process_trajectory(traj: dict, run_name: str, fmt: str) -> dict:
    # Scenario/case ID
    scenario_id = traj.get("scenario_id") or traj.get("case_id", "")
    dataset = infer_dataset(traj, run_name)

    primary, meta = classify_primary(traj, fmt)
    secondary = compute_secondary_flags(meta)

    return {
        "run": run_name,
        "dataset": dataset,
        "scenario_id": str(scenario_id),
        "correct_diagnosis": meta["correct_dx"],
        "final_diagnosis": meta["final_dx"],
        "is_correct": meta["is_correct"],
        "primary_error": primary,
        "secondary_flags": ",".join(secondary) if secondary else "",
        "total_turns": meta["total_turns"],
        "n_tests": meta["n_tests"],
        "turn_at_diagnosis": meta["diagnosis_turn"] if meta["diagnosis_turn"] is not None else "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_rows(rows: list[dict]) -> dict:
    def stats(subset: list[dict]) -> dict:
        total = len(subset)
        correct = sum(1 for r in subset if r["is_correct"])
        acc = round(correct / total * 100, 1) if total else 0.0
        by_error: Counter = Counter(r["primary_error"] for r in subset)
        return {
            "total": total,
            "correct": correct,
            "accuracy_pct": acc,
            "by_error_category": dict(by_error),
        }

    # Overall
    overall = stats(rows)

    # By run
    by_run: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_run[r["run"]].append(r)

    # By dataset
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(r)

    # Secondary flag counts
    sec_counts: Counter = Counter()
    for r in rows:
        if r["secondary_flags"]:
            for flag in r["secondary_flags"].split(","):
                sec_counts[flag] += 1

    return {
        "overall": overall,
        "by_run": {k: stats(v) for k, v in sorted(by_run.items())},
        "by_dataset": {k: stats(v) for k, v in sorted(by_dataset.items())},
        "secondary_flag_counts": dict(sec_counts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(rows: list[dict], summary: dict) -> str:
    lines = ["# AgentClinic Error Categorization Report\n"]

    # ── Summary table by run ──────────────────────────────────────────────────
    lines.append("## Accuracy by Run\n")
    lines.append("| Run | Dataset(s) | Total | Correct | Accuracy |")
    lines.append("|-----|------------|-------|---------|----------|")

    for run, s in summary["by_run"].items():
        # Collect datasets in this run
        run_rows = [r for r in rows if r["run"] == run]
        datasets = sorted(set(r["dataset"] for r in run_rows))
        ds_str = ", ".join(datasets)
        lines.append(
            f"| {run} | {ds_str} | {s['total']} | {s['correct']} | {s['accuracy_pct']}% |"
        )
    lines.append("")

    # ── Error category breakdown per dataset ─────────────────────────────────
    lines.append("## Error Categories by Dataset\n")
    all_error_cats = [
        "correct", "no_diagnosis", "premature_diagnosis", "wrong_differential",
        "anchor_bias", "low_yield_testing", "over_testing", "knowledge_gap",
        "context_drift", "hallucination", "incorrect_other",
    ]

    for ds, s in summary["by_dataset"].items():
        lines.append(f"### {ds}  (n={s['total']}, accuracy={s['accuracy_pct']}%)\n")
        lines.append("| Error Category | Count | % of total |")
        lines.append("|----------------|-------|------------|")
        total = s["total"]
        for cat in all_error_cats:
            count = s["by_error_category"].get(cat, 0)
            if count == 0:
                continue
            pct = round(count / total * 100, 1) if total else 0.0
            lines.append(f"| {cat} | {count} | {pct}% |")
        lines.append("")

    # ── Top 10 wrong diagnoses ────────────────────────────────────────────────
    lines.append("## Top 10 Wrong Diagnoses (Overall)\n")
    wrong_rows = [r for r in rows if not r["is_correct"] and r["final_diagnosis"].strip()]
    # Normalize final_diagnosis: extract text after "DIAGNOSIS READY:" if present
    def extract_dx(text: str) -> str:
        if "DIAGNOSIS READY:" in text.upper():
            idx = text.upper().index("DIAGNOSIS READY:")
            return text[idx + 16:].strip().split("\n")[0].strip()
        # For SDRP the field is already the predicted_diagnosis
        return text.strip()

    wrong_dx_counter: Counter = Counter(extract_dx(r["final_diagnosis"]) for r in wrong_rows)
    lines.append("| Rank | Diagnosis | Count |")
    lines.append("|------|-----------|-------|")
    for rank, (dx, cnt) in enumerate(wrong_dx_counter.most_common(10), 1):
        lines.append(f"| {rank} | {dx[:80]} | {cnt} |")
    lines.append("")

    # ── Top wrong diagnoses by error category ────────────────────────────────
    lines.append("## Top Wrong Diagnoses by Error Category\n")
    for cat in all_error_cats:
        if cat == "correct":
            continue
        cat_rows = [r for r in wrong_rows if r["primary_error"] == cat]
        if not cat_rows:
            continue
        counter: Counter = Counter(extract_dx(r["final_diagnosis"]) for r in cat_rows)
        top = counter.most_common(5)
        lines.append(f"### {cat}  (n={len(cat_rows)})\n")
        lines.append("| Diagnosis | Count |")
        lines.append("|-----------|-------|")
        for dx, cnt in top:
            lines.append(f"| {dx[:80]} | {cnt} |")
        lines.append("")

    # ── Secondary flags summary ───────────────────────────────────────────────
    lines.append("## Secondary Flag Counts (Overall)\n")
    lines.append("| Flag | Count |")
    lines.append("|------|-------|")
    for flag, cnt in sorted(summary["secondary_flag_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {flag} | {cnt} |")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"Scanning trajectory directories under: {TRAJECTORIES_ROOT}\n")
    runs = discover_runs(TRAJECTORIES_ROOT)
    print(f"\nFound {len(runs)} runs total.\n")

    all_rows: list[dict] = []

    for run_name, dir_path, fmt in runs:
        trajs = load_run(dir_path)
        print(f"  Processing {run_name}: {len(trajs)} trajectories (format {fmt})")
        for traj in trajs:
            row = process_trajectory(traj, run_name, fmt)
            all_rows.append(row)

    print(f"\nProcessed {len(all_rows)} trajectories total.\n")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = Path(__file__).parent / "error_categories_all.csv"
    fieldnames = [
        "run", "dataset", "scenario_id", "correct_diagnosis", "final_diagnosis",
        "is_correct", "primary_error", "secondary_flags", "total_turns",
        "n_tests", "turn_at_diagnosis",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Wrote: {csv_path}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = aggregate_rows(all_rows)

    # ── Write JSON summary ────────────────────────────────────────────────────
    json_path = Path(__file__).parent / "error_summary_all.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote: {json_path}")

    # ── Write Markdown report ─────────────────────────────────────────────────
    report = generate_report(all_rows, summary)
    md_path = Path(__file__).parent / "error_report.md"
    with open(md_path, "w") as f:
        f.write(report)
    print(f"Wrote: {md_path}")

    # ── Print quick summary ───────────────────────────────────────────────────
    ov = summary["overall"]
    print("\n" + "=" * 60)
    print("  Overall Summary")
    print("=" * 60)
    print(f"  Total trajectories : {ov['total']}")
    print(f"  Correct            : {ov['correct']}")
    print(f"  Accuracy           : {ov['accuracy_pct']}%")
    print()
    print("  Error category breakdown:")
    for cat, cnt in sorted(ov["by_error_category"].items(), key=lambda x: -x[1]):
        pct = round(cnt / ov["total"] * 100, 1) if ov["total"] else 0
        print(f"    {cat:<25} {cnt:5d}  ({pct}%)")
    print()
    print("  Accuracy by run:")
    for run, s in summary["by_run"].items():
        print(f"    {run:<45} {s['correct']:3d}/{s['total']:3d} = {s['accuracy_pct']}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
