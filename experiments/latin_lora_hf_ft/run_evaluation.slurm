#!/usr/bin/env bash
#SBATCH --job-name=latin-qwen-hf-eval
#SBATCH --gpus=a100-40
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

RUN_ID="${RUN_ID:-hf-head-deprel-eval-001}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH-hf_outputs/qwen25-05b-head-deprel-a100-lora-full}"
SYSTEM_NAME="${SYSTEM_NAME:-hf_lora_head_deprel}"
TEST_FILE="${TEST_FILE:-latin_head_deprel_data/test.jsonl}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-4096}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
PRECISION="${PRECISION:-bf16}"
LIMIT="${LIMIT:-}"

MODEL="$MODEL" \
ADAPTER_PATH="$ADAPTER_PATH" \
SYSTEM_NAME="$SYSTEM_NAME" \
TEST_FILE="$TEST_FILE" \
MAX_INPUT_LENGTH="$MAX_INPUT_LENGTH" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
PRECISION="$PRECISION" \
LIMIT="$LIMIT" \
./run_evaluation.sh "$RUN_ID"
