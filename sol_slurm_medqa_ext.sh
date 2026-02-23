#!/bin/bash
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=agentclinic-medqa-ext
#SBATCH --output=/scratch/%u/agentclinic/logs/%x-%j.out
#SBATCH --error=/scratch/%u/agentclinic/logs/%x-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# AgentClinic — MedQA_Extended dataset, all-Voyager, second API key
#
#   Doctor    → voyager      qwen3-235b-a22b-instruct-2507  (22B active, strongest)
#   Patient   → voyager_lite qwen3-30b-a3b-instruct-2507    ( 3B active, fast)
#   Measurement→voyager_lite qwen3-30b-a3b-instruct-2507    ( 3B active, fast)
#   Moderator → voyager      qwen3-235b-a22b-instruct-2507  (22B active, reliable)
#
# Submit:   sbatch sol_slurm_medqa_ext.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_BASE="/scratch/$USER"
AC_DIR="/scratch/$USER/agentclinic"
mkdir -p "$AC_DIR"/{logs,trajectories}

# ── Model configuration ───────────────────────────────────────────────────────
DOCTOR_LLM="voyager"
PATIENT_LLM="voyager_lite"
MEASUREMENT_LLM="voyager_lite"
MODERATOR_LLM="voyager"

VOYAGER_MODEL_NAME="qwen3-235b-a22b-instruct-2507"
VOYAGER_LITE_MODEL_NAME="qwen3-30b-a3b-instruct-2507"
# ─────────────────────────────────────────────────────────────────────────────

# ── AgentClinic run settings ──────────────────────────────────────────────────
DATASET="MedQA_Ext"
TOTAL_INFERENCES=20
DOCTOR_BIAS="None"
PATIENT_BIAS="None"
OUTPUT_DIR="$AC_DIR/trajectories/voyager/medqa_ext"
# ─────────────────────────────────────────────────────────────────────────────

# ── Use second API key to avoid rate-limit collisions with MedQA job ─────────
VOYAGER_API_BASE=https://openai.rc.asu.edu/v1
VOYAGER_API_KEY=sk-LIUEJYkOM8UuCkgBgKOsog
# ─────────────────────────────────────────────────────────────────────────────

# ── Python venv (pre-built manually — see README setup steps) ────────────────
VENV_PY="$AC_DIR/venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
  echo "ERROR: venv not found at $AC_DIR/venv" >&2
  echo "  Run setup first: python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
echo "Using venv: $($VENV_PY --version) at $VENV_PY"
# ─────────────────────────────────────────────────────────────────────────────

cd "$AC_DIR"

echo "============================================"
echo "  AGENTCLINIC — MEDQA_EXT VOYAGER RUN"
echo "============================================"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Node          : $(hostname)"
echo "  Dataset       : $DATASET  ($TOTAL_INFERENCES inferences/scenario)"
echo "  Doctor        : $VOYAGER_MODEL_NAME"
echo "  Patient       : $VOYAGER_LITE_MODEL_NAME"
echo "  Measurement   : $VOYAGER_LITE_MODEL_NAME"
echo "  Moderator     : $VOYAGER_MODEL_NAME"
echo "  Output dir    : $OUTPUT_DIR"
echo "  Start time    : $(date)"
echo "============================================"

"$VENV_PY" agentclinic.py \
  --openai_api_key          "EMPTY" \
  --inf_type                llm \
  --doctor_llm              "$DOCTOR_LLM" \
  --patient_llm             "$PATIENT_LLM" \
  --measurement_llm         "$MEASUREMENT_LLM" \
  --moderator_llm           "$MODERATOR_LLM" \
  --agent_dataset           "$DATASET" \
  --total_inferences        "$TOTAL_INFERENCES" \
  --doctor_bias             "$DOCTOR_BIAS" \
  --patient_bias            "$PATIENT_BIAS" \
  --output_dir              "$OUTPUT_DIR" \
  --voyager_api_key         "$VOYAGER_API_KEY" \
  --voyager_api_base        "$VOYAGER_API_BASE" \
  --voyager_model_name      "$VOYAGER_MODEL_NAME" \
  --voyager_lite_model_name "$VOYAGER_LITE_MODEL_NAME"

echo "============================================"
echo "  JOB COMPLETE: $(date)"
echo "  Trajectories: $OUTPUT_DIR"
echo "============================================"
