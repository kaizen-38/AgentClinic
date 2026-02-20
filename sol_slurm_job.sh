#!/usr/bin/env bash
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=agentclinic-gaudi
#SBATCH --output=/scratch/%u/agentclinic/logs/%x-%j.out
#SBATCH --error=/scratch/%u/agentclinic/logs/%x-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# AgentClinic on ASU SOL — Gaudi 2 (Doctor/Patient vLLM) + Voyager (Moderator)
#
# Architecture:
#   Doctor  Agent → local Gaudi vLLM  (--doctor_llm  local)
#   Patient Agent → local Gaudi vLLM  (--patient_llm local)
#   Measurement   → Voyager API       (--measurement_llm  gpt4o)
#   Moderator     → Voyager API       (--moderator_llm    gpt4)
#
# Submit (single job, all N scenarios):
#   sbatch sol_slurm_job.sh
#
# Submit (job array, 1 scenario per task — parallel):
#   sbatch --array=0-214 sol_slurm_job.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1

# ── Container ─────────────────────────────────────────────────────────────────
CTR=/usr/bin/apptainer
CONTAINER="/data/sse/gaudi/containers/vllm-gaudi.sif"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_BASE="/scratch/$USER"
AC_DIR="/scratch/$USER/agentclinic"   # copy of this repo on SOL scratch
VENV_PY="$AC_DIR/venv/bin/python"

mkdir -p "$SCRATCH_BASE"/{logs,habana_logs,home,.cache/huggingface}
mkdir -p "$AC_DIR/logs"

export HOME="$SCRATCH_BASE/home"
export HF_HOME="$SCRATCH_BASE/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$SCRATCH_BASE/.cache"
export HABANA_LOGS="$SCRATCH_BASE/habana_logs"

# ── Model configuration ───────────────────────────────────────────────────────
#   LOCAL_MODEL  : served on Gaudi via vLLM (Doctor + Patient)
#   Voyager API  : used for Measurement + Moderator
LOCAL_MODEL="meta-llama/Meta-Llama-3-70B-Instruct"
DOCTOR_LLM="local"
PATIENT_LLM="local"
MEASUREMENT_LLM="gpt4o"       # routed via Voyager
MODERATOR_LLM="gpt4"          # routed via Voyager (hardcoded grader)

# ── AgentClinic run settings ──────────────────────────────────────────────────
DATASET="MedQA"                # MedQA | MedQA_Ext | NEJM | NEJM_Ext
TOTAL_INFERENCES=10
DOCTOR_BIAS="None"
PATIENT_BIAS="None"
OUTPUT_DIR="$AC_DIR/trajectories"
mkdir -p "$OUTPUT_DIR"

# ── API Keys ──────────────────────────────────────────────────────────────────
# Voyager keys — load from .env file (NEVER commit keys to git)
#   Create once on SOL:  echo 'export VOYAGER_API_KEY="sk-..."' > ~/.agentclinic_secrets
VOYAGER_API_BASE="https://openai.rc.asu.edu/v1"
if [ -f "$HOME/.agentclinic_secrets" ]; then
  set -a; source "$HOME/.agentclinic_secrets"; set +a
elif [ -f "$AC_DIR/.env" ]; then
  set -a; source "$AC_DIR/.env"; set +a
fi
VOYAGER_API_KEY="${VOYAGER_API_KEY:-${USER_MODEL_API_KEY:-}}"
if [[ -z "$VOYAGER_API_KEY" ]]; then
  echo "ERROR: VOYAGER_API_KEY not set. Create ~/.agentclinic_secrets or $AC_DIR/.env" >&2
  exit 1
fi

# ── Gaudi vLLM Server ─────────────────────────────────────────────────────────
PORT=$((8000 + SLURM_JOB_ID % 1000))

echo "Launching Gaudi vLLM server on 127.0.0.1:${PORT}..."
echo "  Model: $LOCAL_MODEL"

$CTR exec --writable-tmpfs \
  --bind /scratch:/scratch \
  --bind /data:/data \
  --bind "$HABANA_LOGS":"$HABANA_LOGS" \
  --env HABANA_VISIBLE_DEVICES=0 \
  --env HABANA_LOGS="$HABANA_LOGS" \
  --env HF_HOME="$HF_HOME" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env HOME="$HOME" \
  --env VLLM_SKIP_WARMUP=true \
  "$CONTAINER" \
  bash -c "pip install --no-cache-dir 'transformers>=4.51.0' -q && \
    vllm serve '$LOCAL_MODEL' \
      --device hpu \
      --dtype bfloat16 \
      --block-size 128 \
      --max-model-len 16384 \
      --tensor-parallel-size 1 \
      --port $PORT" \
  > "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM to be ready (up to 15 min)..."
for i in {1..450}; do
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM exited early. Log tail:"
    tail -n 80 "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" || true
    exit 1
  fi
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready on port $PORT."
    break
  fi
  sleep 2
done

if ! curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM never became ready. Log tail:"
  tail -n 80 "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" || true
  exit 1
fi

LOCAL_API_BASE="http://127.0.0.1:${PORT}/v1"

# ── Run AgentClinic ───────────────────────────────────────────────────────────
cd "$AC_DIR"

echo "============================================"
echo "  AGENTCLINIC RUN CONFIG"
echo "============================================"
echo "  Job ID         : $SLURM_JOB_ID"
echo "  Node           : $(hostname)"
echo "  Dataset        : $DATASET"
echo "  Doctor LLM     : $DOCTOR_LLM  →  $LOCAL_MODEL  (local Gaudi vLLM)"
echo "  Patient LLM    : $PATIENT_LLM →  $LOCAL_MODEL  (local Gaudi vLLM)"
echo "  Measurement    : $MEASUREMENT_LLM (Voyager)"
echo "  Moderator      : $MODERATOR_LLM   (Voyager)"
echo "  Inferences/sc  : $TOTAL_INFERENCES"
echo "  Output dir     : $OUTPUT_DIR"
echo "============================================"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  # ── Array mode: one task = scenarios 0..SLURM_ARRAY_TASK_ID (cumulative)
  # Because agentclinic.py has no --start_scenario, only num_scenarios,
  # we run up to (TASK_ID+1) scenarios and collect only the last file.
  TASK_ID="${SLURM_ARRAY_TASK_ID}"
  TASK_OUT="$OUTPUT_DIR/array_task_${TASK_ID}"
  mkdir -p "$TASK_OUT"

  "$VENV_PY" agentclinic.py \
    --openai_api_key     "EMPTY" \
    --inf_type           llm \
    --doctor_llm         "$DOCTOR_LLM" \
    --patient_llm        "$PATIENT_LLM" \
    --measurement_llm    "$MEASUREMENT_LLM" \
    --moderator_llm      "$MODERATOR_LLM" \
    --agent_dataset      "$DATASET" \
    --num_scenarios      "$((TASK_ID + 1))" \
    --total_inferences   "$TOTAL_INFERENCES" \
    --doctor_bias        "$DOCTOR_BIAS" \
    --patient_bias       "$PATIENT_BIAS" \
    --output_dir         "$TASK_OUT" \
    --openai_api_base    "$LOCAL_API_BASE" \
    --local_model_name   "$LOCAL_MODEL" \
    --voyager_api_key    "$VOYAGER_API_KEY" \
    --voyager_api_base   "$VOYAGER_API_BASE"

  # Copy the single trajectory file we care about to the shared output dir
  TARGET="$TASK_OUT/trajectory_$(printf '%04d' $TASK_ID).json"
  if [[ -f "$TARGET" ]]; then
    cp "$TARGET" "$OUTPUT_DIR/"
    echo "Trajectory for scenario $TASK_ID saved."
  else
    echo "WARNING: Expected $TARGET not found."
  fi

else
  # ── Single-job mode: run all scenarios sequentially
  "$VENV_PY" agentclinic.py \
    --openai_api_key     "EMPTY" \
    --inf_type           llm \
    --doctor_llm         "$DOCTOR_LLM" \
    --patient_llm        "$PATIENT_LLM" \
    --measurement_llm    "$MEASUREMENT_LLM" \
    --moderator_llm      "$MODERATOR_LLM" \
    --agent_dataset      "$DATASET" \
    --total_inferences   "$TOTAL_INFERENCES" \
    --doctor_bias        "$DOCTOR_BIAS" \
    --patient_bias       "$PATIENT_BIAS" \
    --output_dir         "$OUTPUT_DIR" \
    --openai_api_base    "$LOCAL_API_BASE" \
    --local_model_name   "$LOCAL_MODEL" \
    --voyager_api_key    "$VOYAGER_API_KEY" \
    --voyager_api_base   "$VOYAGER_API_BASE"
fi

echo "============================================"
echo "  JOB COMPLETE: $(date)"
echo "  Trajectories : $OUTPUT_DIR"
echo "============================================"

# Gracefully shut down the vLLM server
kill "$VLLM_PID" || true
