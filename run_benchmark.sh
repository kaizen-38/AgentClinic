#!/usr/bin/env bash
# run_benchmark.sh
# ─────────────────────────────────────────────────────────────────────────────
# Convenience script to run AgentClinic with trajectory saving.
#
# Usage:
#   chmod +x run_benchmark.sh
#   ./run_benchmark.sh --openai_api_key sk-...
#   ./run_benchmark.sh --openai_api_key sk-... --num_scenarios 10
#
# Environment variables (optional, override CLI):
#   OPENAI_API_KEY    - OpenAI API key
#   ANTHROPIC_API_KEY - Anthropic API key
#   REPLICATE_API_KEY - Replicate API key
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DOCTOR_LLM="gpt4o"
PATIENT_LLM="gpt4o"
MEASUREMENT_LLM="gpt4"
MODERATOR_LLM="gpt4"
DATASET="MedQA"            # MedQA | MedQA_Ext | NEJM | NEJM_Ext
NUM_SCENARIOS=10           # number of scenarios to run
TOTAL_INFERENCES=10        # max turns per scenario
DOCTOR_BIAS="None"
PATIENT_BIAS="None"
OUTPUT_DIR="./trajectories"
INF_TYPE="llm"
OPENAI_KEY="${OPENAI_API_KEY:-}"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --openai_api_key)   OPENAI_KEY="$2";          shift 2 ;;
    --doctor_llm)       DOCTOR_LLM="$2";          shift 2 ;;
    --patient_llm)      PATIENT_LLM="$2";         shift 2 ;;
    --dataset)          DATASET="$2";             shift 2 ;;
    --num_scenarios)    NUM_SCENARIOS="$2";        shift 2 ;;
    --total_inferences) TOTAL_INFERENCES="$2";    shift 2 ;;
    --doctor_bias)      DOCTOR_BIAS="$2";          shift 2 ;;
    --patient_bias)     PATIENT_BIAS="$2";         shift 2 ;;
    --output_dir)       OUTPUT_DIR="$2";           shift 2 ;;
    *)                  echo "Unknown option: $1"; exit 1  ;;
  esac
done

if [[ -z "$OPENAI_KEY" ]]; then
  echo "ERROR: --openai_api_key is required (or set OPENAI_API_KEY env var)"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "────────────────────────────────────────────────────────"
echo "  AgentClinic Benchmark Run"
echo "  Dataset       : $DATASET"
echo "  Scenarios     : $NUM_SCENARIOS"
echo "  Inferences    : $TOTAL_INFERENCES"
echo "  Doctor LLM    : $DOCTOR_LLM"
echo "  Patient LLM   : $PATIENT_LLM"
echo "  Doctor bias   : $DOCTOR_BIAS"
echo "  Patient bias  : $PATIENT_BIAS"
echo "  Output dir    : $OUTPUT_DIR"
echo "  Log file      : $LOG_FILE"
echo "────────────────────────────────────────────────────────"

python agentclinic.py \
  --openai_api_key     "$OPENAI_KEY" \
  --inf_type           "$INF_TYPE" \
  --doctor_llm         "$DOCTOR_LLM" \
  --patient_llm        "$PATIENT_LLM" \
  --measurement_llm    "$MEASUREMENT_LLM" \
  --moderator_llm      "$MODERATOR_LLM" \
  --agent_dataset      "$DATASET" \
  --num_scenarios      "$NUM_SCENARIOS" \
  --total_inferences   "$TOTAL_INFERENCES" \
  --doctor_bias        "$DOCTOR_BIAS" \
  --patient_bias       "$PATIENT_BIAS" \
  --output_dir         "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "Run complete. Trajectories saved to: $OUTPUT_DIR"
echo "Run analysis with:"
echo "  python analyze_trajectories.py --input_dir $OUTPUT_DIR --output summary_report.json"
