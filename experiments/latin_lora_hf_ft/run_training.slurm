#!/usr/bin/env bash
#SBATCH --job-name=latin-qwen-hf-lora
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

if [ "${RECREATE_VENV:-0}" = "1" ]; then
  rm -rf .venv-hf
fi

python3 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install --upgrade pip --no-cache-dir
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install --no-cache-dir -r requirements.txt

export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
DATA_DIR="${DATA_DIR:-latin_head_deprel_data}"
OUTPUT_DIR="${OUTPUT_DIR:-hf_outputs/qwen25-05b-head-deprel-lora}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_STEPS="${MAX_STEPS:--1}"
EPOCHS="${EPOCHS:-3}"
PRECISION="${PRECISION:-fp16}"
QLORA="${QLORA:-0}"

PRECISION_ARGS=""
if [ "$PRECISION" = "bf16" ]; then
  PRECISION_ARGS="--bf16"
elif [ "$PRECISION" = "fp16" ]; then
  PRECISION_ARGS="--fp16"
elif [ "$PRECISION" != "fp32" ]; then
  echo "Unsupported PRECISION=$PRECISION; use fp16, bf16, or fp32" >&2
  exit 2
fi

QLORA_ARGS=()
if [ "$QLORA" = "1" ]; then
  python -m pip install --no-cache-dir bitsandbytes==0.45.0
  QLORA_ARGS=(--qlora)
fi

python scripts/train_hf_lora.py \
  --model "$MODEL" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  "${QLORA_ARGS[@]}" \
  $PRECISION_ARGS \
  --gradient-checkpointing \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --logging-steps 10 \
  --eval-steps 50 \
  --save-steps 100 \
  --save-total-limit 2
