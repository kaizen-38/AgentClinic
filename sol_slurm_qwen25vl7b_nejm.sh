#!/bin/bash
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=agentclinic-qwen25vl7b-nejm
#SBATCH --output=/scratch/%u/agentclinic/logs/%x-%j.out
#SBATCH --error=/scratch/%u/agentclinic/logs/%x-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# AgentClinic — NEJM dataset with Qwen2.5-VL-7B-Instruct (vision doctor)
#
# SUPERSEDED — prefer the larger VLM scripts for better accuracy:
#   sol_slurm_qwen25vl32b_nejm.sh   — Qwen2.5-VL-32B, Apache 2.0, explicit Gaudi docs
#   sol_slurm_internvl25_38b_nejm.sh — InternVL2.5-38B, MIT, best medical benchmarks
#
# Keep this script for quick/cheap ablation runs (7B is fast, uses 1 GPU).
#
#   Doctor    → local vLLM    Qwen2.5-VL-7B-Instruct  (2x Gaudi, vision-enabled)
#   Patient   → voyager_lite  qwen3-30b-a3b-instruct-2507  (Apache 2.0)
#   Measurement→voyager_lite  qwen3-30b-a3b-instruct-2507  (Apache 2.0)
#   Moderator → voyager       qwen3-235b-a22b-instruct-2507 (Apache 2.0)
#
# Submit:  sbatch sol_slurm_qwen25vl7b_nejm.sh
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
LOCAL_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

DOCTOR_LLM="local"
PATIENT_LLM="voyager_lite"
MEASUREMENT_LLM="voyager_lite"
MODERATOR_LLM="voyager"

VOYAGER_MODEL_NAME="qwen3-235b-a22b-instruct-2507"
VOYAGER_LITE_MODEL_NAME="qwen3-30b-a3b-instruct-2507"
# ─────────────────────────────────────────────────────────────────────────────

# ── AgentClinic run settings ──────────────────────────────────────────────────
# NEJM base dataset: 14 cases, each with a real clinical photo (image_url).
# --doctor_image_request True enables the doctor to call REQUEST IMAGES and
# receive the clinical photo via the VLM multimodal API.
DATASET="NEJM"
TOTAL_INFERENCES=20
DOCTOR_BIAS="None"
PATIENT_BIAS="None"
DOCTOR_IMAGE_REQUEST="True"
OUTPUT_DIR="$AC_DIR/trajectories/qwen25vl7b/nejm"
mkdir -p "$OUTPUT_DIR"
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
  echo "  Create $AC_DIR/.env with: export USER_MODEL_API_KEY=your-key-here" >&2
  exit 1
fi
# ─────────────────────────────────────────────────────────────────────────────

# ── Python venv ───────────────────────────────────────────────────────────────
VENV_PY="$AC_DIR/venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
  echo "ERROR: venv not found at $AC_DIR/venv" >&2
  echo "  Run: python3 -m venv venv && venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
echo "Using venv: $($VENV_PY --version) at $VENV_PY"
# ─────────────────────────────────────────────────────────────────────────────

# ── Launch Gaudi vLLM (2-GPU tensor parallel for Qwen2.5-VL-7B) ──────────────
PORT=$((8000 + SLURM_JOB_ID % 1000))
echo "Launching 2-GPU Gaudi vLLM on port $PORT for $LOCAL_MODEL ..."

$CTR exec --writable-tmpfs \
  --bind /scratch:/scratch \
  --bind /data:/data \
  --bind "$HABANA_LOGS":"$HABANA_LOGS" \
  --env HABANA_VISIBLE_DEVICES=0,1 \
  --env HABANA_LOGS="$HABANA_LOGS" \
  --env HF_HOME="$HF_HOME" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env HOME="$HOME" \
  --env VLLM_SKIP_WARMUP=true \
  "$CONTAINER" \
  bash -c "pip install --no-cache-dir 'transformers>=4.51.0' 'qwen_vl_utils>=0.0.8' -q && \
    vllm serve '$LOCAL_MODEL' \
      --device hpu \
      --dtype bfloat16 \
      --block-size 128 \
      --max-model-len 16384 \
      --tensor-parallel-size 2 \
      --limit-mm-per-prompt image=1 \
      --port $PORT" \
  > "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM to be ready (up to 15 min for 7B VLM)..."
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
# ─────────────────────────────────────────────────────────────────────────────

cd "$AC_DIR"

echo "============================================================"
echo "  AGENTCLINIC — Qwen2.5-VL-7B NEJM VLM RUN (2x Gaudi)"
echo "============================================================"
echo "  Job ID           : $SLURM_JOB_ID"
echo "  Node             : $(hostname)"
echo "  Dataset          : $DATASET  ($TOTAL_INFERENCES inferences/scenario)"
echo "  Doctor           : $LOCAL_MODEL (2x Gaudi, vision-enabled)"
echo "  Patient          : $VOYAGER_LITE_MODEL_NAME (Voyager)"
echo "  Measurement      : $VOYAGER_LITE_MODEL_NAME (Voyager)"
echo "  Moderator        : $VOYAGER_MODEL_NAME (Voyager)"
echo "  Image requests   : $DOCTOR_IMAGE_REQUEST"
echo "  Output dir       : $OUTPUT_DIR"
echo "  Start time       : $(date)"
echo "============================================================"

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
  --doctor_image_request    "$DOCTOR_IMAGE_REQUEST" \
  --output_dir              "$OUTPUT_DIR" \
  --openai_api_base         "$LOCAL_API_BASE" \
  --local_model_name        "$LOCAL_MODEL" \
  --voyager_api_key         "$VOYAGER_API_KEY" \
  --voyager_api_base        "$VOYAGER_API_BASE" \
  --voyager_model_name      "$VOYAGER_MODEL_NAME" \
  --voyager_lite_model_name "$VOYAGER_LITE_MODEL_NAME"

echo "============================================================"
echo "  JOB COMPLETE: $(date)"
echo "  Trajectories: $OUTPUT_DIR"
echo "============================================================"

kill "$VLLM_PID" || true
