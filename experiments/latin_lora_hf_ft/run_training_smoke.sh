#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv-hf ]; then
  python3 -m venv .venv-hf
fi
source .venv-hf/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export TOKENIZERS_PARALLELISM=false

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-latin_head_deprel_data_macsafe_2048}"
OUTPUT_DIR="${OUTPUT_DIR:-hf_outputs/smoke-qwen25-05b-head-deprel-lora}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MAX_STEPS="${MAX_STEPS:-10}"
PRECISION="${PRECISION:-fp16}"

PRECISION_ARGS=""
if [ "$PRECISION" = "bf16" ]; then
  PRECISION_ARGS="--bf16"
elif [ "$PRECISION" = "fp16" ]; then
  PRECISION_ARGS="--fp16"
elif [ "$PRECISION" != "fp32" ]; then
  echo "Unsupported PRECISION=$PRECISION; use fp16, bf16, or fp32" >&2
  exit 2
fi

python scripts/train_hf_lora.py \
  --model "$MODEL" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --max-steps "$MAX_STEPS" \
  --qlora \
  $PRECISION_ARGS \
  --gradient-checkpointing \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --learning-rate 2e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --logging-steps 1 \
  --eval-steps 5 \
  --save-steps 5 \
  --save-total-limit 1
