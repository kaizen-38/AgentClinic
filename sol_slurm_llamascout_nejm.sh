#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=agentclinic-llamascout-nejm
#SBATCH --output=/scratch/%u/agentclinic/logs/%x-%j.out
#SBATCH --error=/scratch/%u/agentclinic/logs/%x-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# AgentClinic — NEJM_Ext with Llama-4-Scout as doctor (Voyager API, vision)
#
# Llama-4-Scout is a multimodal MoE model served on ASU Voyager.
# No local GPU needed — all inference is via the Voyager OpenAI-compatible API.
#
#   Doctor    → voyager  llama4-scout-17b  (vision-enabled)
#   Patient   → voyager_lite  qwen3-30b-a3b-instruct-2507
#   Measurement→voyager_lite  qwen3-30b-a3b-instruct-2507
#   Moderator → voyager       qwen3-235b-a22b-instruct-2507
#
# Submit:  sbatch sol_slurm_llamascout_nejm.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1

SCRATCH_BASE="/scratch/$USER"
AC_DIR="/scratch/$USER/agentclinic"
mkdir -p "$AC_DIR"/{logs,trajectories}

# ── Voyager API config ────────────────────────────────────────────────────────
VOYAGER_API_BASE="https://openai.rc.asu.edu/v1"

# Doctor: Llama-4-Scout on Voyager (multimodal, vision-enabled)
# If the model name below doesn't match, run the curl command in the header
# to find the correct name and update VOYAGER_DOCTOR_MODEL.
VOYAGER_DOCTOR_MODEL="llama4-scout-17b"

# Supporting agents: Qwen3 on Voyager (text-only, fast)
VOYAGER_LITE_MODEL_NAME="qwen3-30b-a3b-instruct-2507"
VOYAGER_MODERATOR_MODEL="qwen3-235b-a22b-instruct-2507"

if [ -f "$AC_DIR/.env" ]; then
  set -a; source "$AC_DIR/.env"; set +a
elif [ -f "$SCRATCH_BASE/.agentclinic_secrets" ]; then
  set -a; source "$SCRATCH_BASE/.agentclinic_secrets"; set +a
fi
VOYAGER_API_KEY="${VOYAGER_API_KEY:-${USER_MODEL_API_KEY:-}}"
if [[ -z "${VOYAGER_API_KEY:-}" ]]; then
  echo "ERROR: No Voyager API key found." >&2
  echo "  Create $AC_DIR/.env with: export USER_MODEL_API_KEY=your-key-here" >&2
  exit 1
fi
# ─────────────────────────────────────────────────────────────────────────────

# ── Run config ────────────────────────────────────────────────────────────────
DATASET="NEJM_Ext"
TOTAL_INFERENCES=20
DOCTOR_BIAS="None"
PATIENT_BIAS="None"
DOCTOR_IMAGE_REQUEST="True"
OUTPUT_DIR="$AC_DIR/trajectories/llamascout/nejm_ext"
mkdir -p "$OUTPUT_DIR"
# ─────────────────────────────────────────────────────────────────────────────

VENV_PY="$AC_DIR/venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
  echo "ERROR: venv not found at $AC_DIR/venv" >&2
  exit 1
fi

echo "============================================================"
echo "  AGENTCLINIC — Llama-4-Scout NEJM_Ext run"
echo "============================================================"
echo "  Job ID      : $SLURM_JOB_ID"
echo "  Node        : $(hostname)"
echo "  Dataset     : $DATASET  (119 cases, images enabled)"
echo "  Doctor      : $VOYAGER_DOCTOR_MODEL (Voyager, vision)"
echo "  Patient     : $VOYAGER_LITE_MODEL_NAME (Voyager)"
echo "  Measurement : $VOYAGER_LITE_MODEL_NAME (Voyager)"
echo "  Moderator   : $VOYAGER_MODERATOR_MODEL (Voyager)"
echo "  Start time  : $(date)"
echo "============================================================"

cd "$AC_DIR"

"$VENV_PY" agentclinic.py \
  --openai_api_key          "EMPTY" \
  --inf_type                llm \
  --doctor_llm              "voyager" \
  --patient_llm             "voyager_lite" \
  --measurement_llm         "voyager_lite" \
  --moderator_llm           "voyager" \
  --agent_dataset           "$DATASET" \
  --total_inferences        "$TOTAL_INFERENCES" \
  --doctor_bias             "$DOCTOR_BIAS" \
  --patient_bias            "$PATIENT_BIAS" \
  --doctor_image_request    "$DOCTOR_IMAGE_REQUEST" \
  --output_dir              "$OUTPUT_DIR" \
  --voyager_api_key         "$VOYAGER_API_KEY" \
  --voyager_api_base        "$VOYAGER_API_BASE" \
  --voyager_model_name      "$VOYAGER_DOCTOR_MODEL" \
  --voyager_lite_model_name "$VOYAGER_LITE_MODEL_NAME"

echo "============================================================"
echo "  JOB COMPLETE: $(date)"
echo "  Trajectories: $OUTPUT_DIR"
echo "============================================================"
