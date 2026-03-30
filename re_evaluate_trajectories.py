#!/usr/bin/env python3
"""
Re-evaluate trajectory correctness using the same moderator as AgentClinic.

Uses the same prompt and normalization as agentclinic.compare_results(), but calls
the Voyager (OpenAI-compatible) API directly so we do not import agentclinic (which
pulls in replicate/pydantic and can fail on Python 3.14+).

Usage:
  python re_evaluate_trajectories.py --input_dir trajectories/voyager/medqa
  python re_evaluate_trajectories.py --input_dir trajectories/voyager/medqa --write_back
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

try:
    import openai
except ImportError:
    openai = None

MODERATOR_SYSTEM = "You are responsible for determining if the correct diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else."
MODERATOR_USER_TEMPLATE = "\nHere is the correct diagnosis: {correct_diagnosis}\n Here was the doctor dialogue: {diagnosis}\nAre these the same?"


def load_env(root_dir: Path) -> None:
    env_path = root_dir / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    # Support "export KEY=value" (shell-style) and "KEY=value"
                    if line.lower().startswith("export "):
                        line = line[7:].strip()
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    # Expand $VAR to existing env value (e.g. VOYAGER_API_KEY=$USER_MODEL_API_KEY)
                    if v.startswith("$") and v[1:] in os.environ:
                        v = os.environ[v[1:]]
                    os.environ.setdefault(k, v)


def compare_results_voyager(diagnosis: str, correct_diagnosis: str) -> str:
    """Same logic as agentclinic.compare_results for moderator_llm=voyager. Returns 'yes' or 'no'."""
    if not openai:
        raise RuntimeError("openai package is required; pip install openai")
    api_key = os.environ.get("VOYAGER_API_KEY") or os.environ.get("USER_MODEL_API_KEY")
    if not api_key:
        raise RuntimeError("Set VOYAGER_API_KEY or USER_MODEL_API_KEY (e.g. in .env)")
    base = os.environ.get("VOYAGER_API_BASE", "https://openai.rc.asu.edu/v1")
    model = os.environ.get("VOYAGER_MODEL_NAME", "qwen3-235b-a22b-instruct-2507")

    user_msg = MODERATOR_USER_TEMPLATE.format(
        correct_diagnosis=correct_diagnosis,
        diagnosis=diagnosis,
    )
    # openai 0.28.x uses module-level api_key/api_base and ChatCompletion.create
    openai.api_key = api_key
    openai.api_base = base
    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": MODERATOR_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.05,
        max_tokens=200,
    )
    answer = (resp["choices"][0]["message"]["content"] or "").strip()
    first_word = re.sub(r"[^a-z]", "", answer.lower().split()[0]) if answer else "no"
    return first_word


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate trajectories using same moderator as AgentClinic (Voyager).")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing trajectory_*.json")
    parser.add_argument("--output_dir", type=str, default=None, help="If --write_back, write here (default: input_dir)")
    parser.add_argument("--write_back", action="store_true", help="Write updated is_correct back to trajectory JSONs")
    parser.add_argument("--root_dir", type=str, default=".", help="Project root for .env")
    args = parser.parse_args()

    root = Path(args.root_dir)
    load_env(root)

    input_dir = Path(args.input_dir)
    paths = sorted(input_dir.glob("trajectory_*.json"))
    if not paths:
        raise SystemExit(f"No trajectory_*.json found in {input_dir}")

    correct = 0
    total = 0
    updated = []

    for p in paths:
        with open(p) as f:
            traj = json.load(f)
        correct_diag = traj.get("correct_diagnosis") or ""
        final_diag = traj.get("final_diagnosis") or ""

        if not final_diag or "(Doctor did not issue" in final_diag or "DIAGNOSIS READY" not in final_diag:
            is_correct = False
        else:
            result = compare_results_voyager(final_diag, correct_diag)
            is_correct = (result == "yes")
            time.sleep(0.5)

        total += 1
        if is_correct:
            correct += 1
        traj["is_correct"] = is_correct
        updated.append((p, traj))

    accuracy_pct = round(100 * correct / total, 1) if total else 0
    print(f"Re-evaluated {total} trajectories (Voyager moderator, same prompt as AgentClinic)")
    print(f"Accuracy: {correct}/{total} = {accuracy_pct}%")

    if args.write_back:
        out_dir = Path(args.output_dir) if args.output_dir else input_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        for p, traj in updated:
            out_path = out_dir / p.name
            with open(out_path, "w") as f:
                json.dump(traj, f, indent=2)
        print(f"Wrote {len(updated)} trajectories to {out_dir}")


if __name__ == "__main__":
    main()
