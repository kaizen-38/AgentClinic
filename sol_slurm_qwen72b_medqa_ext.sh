#!/bin/bash
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=agentclinic-qwen72b-medqa-ext
#SBATCH --output=/scratch/%u/agentclinic/logs/%x-%j.out
#SBATCH --error=/scratch/%u/agentclinic/logs/%x-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# AgentClinic — 4x Gaudi (Qwen2.5-72B doctor/patient) + Voyager (measurement/moderator)
# Dataset: MedQA_Ext (full benchmark)
#
#   Doctor    → local vLLM    Qwen2.5-72B-Instruct  (4x Gaudi, 22B active after TP)
#   Patient   → local vLLM    Qwen2.5-72B-Instruct  (same server)
#   Measurement→voyager_lite  qwen3-30b-a3b          (fast, 3B active)
#   Moderator → voyager       qwen3-235b-a22b        (22B active, reliable yes/no)
#
# Submit:  sbatch sol_slurm_qwen72b_medqa_ext.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export PYTHONUNBUFFERED=1

CTR=/usr/bin/apptainer
CONTAINER="/data/sse/gaudi/containers/vllm-gaudi.sif"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRATCH_BASE="/scratch/$USER"
AC_DIR="/scratch/$USER/agentclinic"
mkdir -p "$SCRATCH_BASE"/{logs,habana_logs,home,.cache/huggingface}
mkdir -p "$AC_DIR"/{logs,trajectories}

export HOME="$SCRATCH_BASE/home"
export HF_HOME="$SCRATCH_BASE/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export XDG_CACHE_HOME="$SCRATCH_BASE/.cache"
export HABANA_LOGS="$SCRATCH_BASE/habana_logs"

# ── Model configuration ───────────────────────────────────────────────────────
# Open-source 72B — no HF token required, strong medical reasoning
LOCAL_MODEL="Qwen/Qwen2.5-72B-Instruct"
# Alternative (gated — needs HF_TOKEN + accepted Meta license on huggingface.co):
# LOCAL_MODEL="meta-llama/Llama-3.3-70B-Instruct"

DOCTOR_LLM="local"
PATIENT_LLM="local"
MEASUREMENT_LLM="voyager_lite"
MODERATOR_LLM="voyager"

VOYAGER_MODEL_NAME="qwen3-235b-a22b-instruct-2507"       # reliable yes/no for moderator
VOYAGER_LITE_MODEL_NAME="qwen3-30b-a3b-instruct-2507"   # fast lookup for measurement
# ─────────────────────────────────────────────────────────────────────────────

# ── AgentClinic run settings ──────────────────────────────────────────────────
DATASET="MedQA_Ext"
TOTAL_INFERENCES=20
DOCTOR_BIAS="None"
PATIENT_BIAS="None"
OUTPUT_DIR="$AC_DIR/trajectories/qwen72b/medqa_ext"
# ─────────────────────────────────────────────────────────────────────────────

# ── Load Voyager API keys ─────────────────────────────────────────────────────
VOYAGER_API_BASE="https://openai.rc.asu.edu/v1"
if [ -f "$AC_DIR/.env" ]; then
  set -a; source "$AC_DIR/.env"; set +a
elif [ -f "$SCRATCH_BASE/.agentclinic_secrets" ]; then
  set -a; source "$SCRATCH_BASE/.agentclinic_secrets"; set +a
fi
VOYAGER_API_KEY="${VOYAGER_API_KEY:-${USER_MODEL_API_KEY:-}}"
if [[ -z "${VOYAGER_API_KEY:-}" ]]; then
  echo "ERROR: No Voyager API key found." >&2
  exit 1
fi
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

# ── Launch Gaudi vLLM (4-GPU tensor parallel for 72B model) ──────────────────
PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "Launching 4-GPU Gaudi vLLM on port $PORT for $LOCAL_MODEL ..."

$CTR exec --writable-tmpfs \
  --bind /scratch:/scratch \
  --bind /data:/data \
  --bind "$HABANA_LOGS":"$HABANA_LOGS" \
  --env HABANA_VISIBLE_DEVICES=0,1,2,3 \
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
      --tensor-parallel-size 4 \
      --port $PORT" \
  > "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM to be ready (up to 20 min for 72B model)..."
for i in {1..600}; do
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
# ─────────────────────────────────────────────────────────────────────────────

cd "$AC_DIR"

echo "============================================"
echo "  AGENTCLINIC — QWEN 2.5-72B RUN (4x Gaudi)"
echo "  DATASET: MedQA_Ext"
echo "============================================"
echo "  Job ID        : $SLURM_JOB_ID"
echo "  Node          : $(hostname)"
echo "  Dataset       : $DATASET  ($TOTAL_INFERENCES inferences/scenario)"
echo "  Doctor        : $LOCAL_MODEL (4x Gaudi)"
echo "  Patient       : $LOCAL_MODEL (4x Gaudi)"
echo "  Measurement   : $VOYAGER_LITE_MODEL_NAME (Voyager)"
echo "  Moderator     : $VOYAGER_MODEL_NAME (Voyager)"
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
  --openai_api_base         "$LOCAL_API_BASE" \
  --local_model_name        "$LOCAL_MODEL" \
  --voyager_api_key         "$VOYAGER_API_KEY" \
  --voyager_api_base        "$VOYAGER_API_BASE" \
  --voyager_model_name      "$VOYAGER_MODEL_NAME" \
  --voyager_lite_model_name "$VOYAGER_LITE_MODEL_NAME"

echo "============================================"
echo "  JOB COMPLETE: $(date)"
echo "  Trajectories: $OUTPUT_DIR"
echo "============================================"

kill "$VLLM_PID" || true

