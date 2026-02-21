#!/bin/bash
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=agentclinic-voyager
#SBATCH --output=/scratch/%u/agentclinic/logs/%x-%j.out
#SBATCH --error=/scratch/%u/agentclinic/logs/%x-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# AgentClinic — All-Voyager configuration (no local GPU needed)
#
# Agent routing:
#   Doctor    → voyager      (qwen3-235b-a22b-instruct-2507)  ← strongest, 22B active
#   Patient   → voyager_lite (qwen3-30b-a3b-instruct-2507)    ← fast,  3B active
#   Measurement→voyager_lite (qwen3-30b-a3b-instruct-2507)    ← simple template lookup
#   Moderator → voyager      (qwen3-235b-a22b-instruct-2507)  ← reliable yes/no scoring
#
# Submit:
#   sbatch sol_slurm_job.sh
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

# Strong model  — doctor + moderator (22B active params)
VOYAGER_MODEL_NAME="qwen3-235b-a22b-instruct-2507"

# Fast model    — patient + measurement (3B active params, cheaper/faster)
VOYAGER_LITE_MODEL_NAME="qwen3-30b-a3b-instruct-2507"
# ─────────────────────────────────────────────────────────────────────────────

# ── AgentClinic run settings ──────────────────────────────────────────────────
DATASET="MedQA"        # MedQA | MedQA_Ext | NEJM | NEJM_Ext
TOTAL_INFERENCES=20
DOCTOR_BIAS="None"
PATIENT_BIAS="None"
OUTPUT_DIR="$AC_DIR/trajectories/voyager/medqa"
# ─────────────────────────────────────────────────────────────────────────────

# ── Load Voyager API keys ─────────────────────────────────────────────────────
VOYAGER_API_BASE="https://openai.rc.asu.edu/v1"
if [ -f "$AC_DIR/.env" ]; then
  set -a; source "$AC_DIR/.env"; set +a
elif [ -f "$SCRATCH_BASE/.agentclinic_secrets" ]; then
  set -a; source "$SCRATCH_BASE/.agentclinic_secrets"; set +a
fi
VOYAGER_API_KEY="${VOYAGER_API_KEY:-${USER_MODEL_API_KEY:-}}"
VOYAGER_API_BASE="${VOYAGER_API_BASE:-${USER_MODEL_API_BASE:-https://openai.rc.asu.edu/v1}}"
if [[ -z "${VOYAGER_API_KEY:-}" ]]; then
  echo "ERROR: No Voyager API key found." >&2
  echo "  Create $AC_DIR/.env with: export USER_MODEL_API_KEY=your-key-here" >&2
  exit 1
fi
# ─────────────────────────────────────────────────────────────────────────────

# ── Python venv setup ─────────────────────────────────────────────────────────
VENV_PY="$AC_DIR/venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
  echo "Creating Python venv at $AC_DIR/venv ..."
  python3 -m venv "$AC_DIR/venv"
  "$AC_DIR/venv/bin/pip" install --quiet --upgrade pip
  "$AC_DIR/venv/bin/pip" install --quiet -r "$AC_DIR/requirements.txt"
  echo "Venv ready."
fi
# ─────────────────────────────────────────────────────────────────────────────

cd "$AC_DIR"

echo "============================================"
echo "  AGENTCLINIC RUN CONFIG"
echo "============================================"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Node          : $(hostname)"
echo "  Dataset       : $DATASET"
echo "  Doctor LLM    : $DOCTOR_LLM ($VOYAGER_MODEL_NAME)"
echo "  Patient LLM   : $PATIENT_LLM ($VOYAGER_LITE_MODEL_NAME)"
echo "  Measurement   : $MEASUREMENT_LLM ($VOYAGER_LITE_MODEL_NAME)"
echo "  Moderator     : $MODERATOR_LLM ($VOYAGER_MODEL_NAME)"
echo "  Inferences/sc : $TOTAL_INFERENCES"
echo "  Output dir    : $OUTPUT_DIR"
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
