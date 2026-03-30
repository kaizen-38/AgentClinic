#!/bin/bash
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH -A class_cse59827694spring2026
#SBATCH --job-name=agentclinic-internvl25-38b-nejm
#SBATCH --output=/scratch/%u/agentclinic/logs/%x-%j.out
#SBATCH --error=/scratch/%u/agentclinic/logs/%x-%j.err

# ─────────────────────────────────────────────────────────────────────────────
# AgentClinic — NEJM_Ext dataset with InternVL2.5-38B (vision doctor)
#
# Why InternVL2.5-38B:
#   - MIT license (code) + Apache 2.0 (Qwen2.5-32B backbone) — fully open
#   - Best medical benchmarks among open-source VLMs ≤40B:
#       OmniMedVQA : 79.9   (vs GPT-4o: ~85)
#       VQA-RAD    : 61.4
#       SLAKE      : 70.3
#       PathVQA    : 46.9
#   - InternViT-6B vision encoder handles NEJM clinical photos well
#     (tile-based 448px processing → good for high-res dermatology, radiology)
#   - 38B × 2 bytes BF16 = ~76 GB weights → 2× HL-225 (192 GB total) comfortable
#   - InternVLChatModel arch is supported in vLLM; requires --trust-remote-code
#
# NOTE: Gaudi-specific support for InternVL2.5 is via the generic vLLM-Gaudi
# backend (not explicitly documented like Qwen2.5-VL). If this script fails
# during vLLM serve, try sol_slurm_qwen25vl32b_nejm.sh instead — Qwen2.5-VL-32B
# has explicit Gaudi docs and the same GPU footprint.
#
#   Doctor    → local vLLM    InternVL2_5-38B  (2× Gaudi HL-225, vision-enabled)
#   Patient   → voyager_lite  qwen3-30b-a3b-instruct-2507   (Apache 2.0, Alibaba)
#   Measurement→voyager_lite  qwen3-30b-a3b-instruct-2507   (Apache 2.0, Alibaba)
#   Moderator → voyager       qwen3-235b-a22b-instruct-2507  (Apache 2.0, Alibaba)
#
# All models are open-source. No closed-source APIs used.
#
# Submit:  sbatch sol_slurm_internvl25_38b_nejm.sh
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
# InternVL2.5-38B: MIT + Apache 2.0, best medical benchmarks, uses custom code
LOCAL_MODEL="OpenGVLab/InternVL2_5-38B"

DOCTOR_LLM="local"
PATIENT_LLM="voyager_lite"
MEASUREMENT_LLM="voyager_lite"
MODERATOR_LLM="voyager"

VOYAGER_MODEL_NAME="qwen3-235b-a22b-instruct-2507"
VOYAGER_LITE_MODEL_NAME="qwen3-30b-a3b-instruct-2507"
# ─────────────────────────────────────────────────────────────────────────────

# ── AgentClinic run settings ──────────────────────────────────────────────────
DATASET="NEJM_Ext"
TOTAL_INFERENCES=20
DOCTOR_BIAS="None"
PATIENT_BIAS="None"
DOCTOR_IMAGE_REQUEST="True"
OUTPUT_DIR="$AC_DIR/trajectories/internvl25_38b/nejm_ext"
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

# ── Launch Gaudi vLLM (2× HL-225 tensor parallel for InternVL2.5-38B) ─────────
# InternVL2.5 uses custom model code → --trust-remote-code required.
# Also needs einops and timm for the InternViT vision encoder.
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
  bash -c "pip install --no-cache-dir 'transformers>=4.51.0' 'einops>=0.8.0' 'timm>=1.0.0' -q && \
    vllm serve '$LOCAL_MODEL' \
      --device hpu \
      --dtype bfloat16 \
      --block-size 128 \
      --max-model-len 16384 \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --limit-mm-per-prompt image=1 \
      --port $PORT" \
  > "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &

VLLM_PID=$!

echo "Waiting for vLLM to be ready (up to 25 min for 38B VLM with vision encoder)..."
for i in {1..750}; do
  if ! kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "vLLM exited early. Log tail:"
    tail -n 80 "$SCRATCH_BASE/logs/vllm-$SLURM_JOB_ID.log" || true
    echo ""
    echo "If the error mentions InternVL or trust-remote-code, the Gaudi vLLM"
    echo "container version may not support InternVLChatModel. Use:"
    echo "  sbatch sol_slurm_qwen25vl32b_nejm.sh  (explicit Gaudi support)"
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
echo "  AGENTCLINIC — InternVL2.5-38B NEJM_Ext VLM RUN (2× Gaudi)"
echo "============================================================"
echo "  Job ID           : $SLURM_JOB_ID"
echo "  Node             : $(hostname)"
echo "  Dataset          : $DATASET  (119 cases, clinical images enabled)"
echo "  Doctor           : $LOCAL_MODEL (2× Gaudi HL-225, vision-enabled)"
echo "  Patient          : $VOYAGER_LITE_MODEL_NAME (Voyager, text-only)"
echo "  Measurement      : $VOYAGER_LITE_MODEL_NAME (Voyager, text-only)"
echo "  Moderator        : $VOYAGER_MODEL_NAME (Voyager, text-only)"
echo "  Image requests   : $DOCTOR_IMAGE_REQUEST"
echo "  Max turns/case   : $TOTAL_INFERENCES"
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
