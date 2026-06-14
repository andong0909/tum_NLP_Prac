# Latin LoRA Hugging Face Fine-Tuning

CUDA/Hugging Face PEFT experiment for fine-tuning a small causal language model
to predict EvaLatin dependency columns.

This folder is for remote GPU clusters. The Apple MLX experiment and data
preparation live in:

```text
../latin_lora_mlx_ft/
```

## Protocol

The model sees the full input CoNLL-U sentence, including all original token
columns. The target is only the dependency-related output:

```text
HEAD<TAB>DEPREL
HEAD<TAB>DEPREL
...
```

There is one target line per syntactic token, in input token order. Multiword
token rows such as `7-8` are ignored in the target.

After generation, a renderer script can replace the blank CoNLL-U `HEAD` and
`DEPREL` columns with these model-generated lines and then call the official
scorer. This keeps formatting deterministic while still evaluating the model's
dependency predictions.

## Data

Build the HEAD/DEPREL-only training splits from the MLX/data-prep JSONL:

```sh
python3 scripts/convert_chat_conllu_to_head_deprel.py \
  --input-dir ../latin_lora_mlx_ft/latin_lora_data \
  --out-dir latin_head_deprel_data

python3 scripts/convert_chat_conllu_to_head_deprel.py \
  --input-dir ../latin_lora_mlx_ft/latin_lora_data_macsafe_2048 \
  --out-dir latin_head_deprel_data_macsafe_2048
```

The generated folders are:

```text
latin_head_deprel_data/
latin_head_deprel_data_macsafe_2048/
```

## Copy To Cluster

From the Mac:

```sh
rsync -av --exclude ".venv" --exclude ".venv-hf" --exclude "adapters" --exclude "hf_outputs" --exclude "runs" \
  /Users/antoniii/Desktop/tum_NLP_Prac/ \
  andong@xlogin1.comp.nus.edu.sg:~/tum_NLP_Prac/
```

## Interactive Smoke Run

On the cluster:

```sh
cd ~/tum_NLP_Prac/experiments/latin_lora_hf_ft
srun --gpus=a100-40 --mem=48G --time=00:15:00 --pty bash -l
```

The A100 40 GB target is recommended for this experiment because it supports
bf16 and has enough memory for normal LoRA without bitsandbytes/QLoRA.

Inside the GPU session, confirm CUDA and run a short smoke train:

```sh
cd ~/tum_NLP_Prac/experiments/latin_lora_hf_ft
nvidia-smi

python3 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install --upgrade pip --no-cache-dir
python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
python -m pip install --no-cache-dir transformers peft accelerate datasets

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

python scripts/train_hf_lora.py \
  --model "Qwen/Qwen2.5-0.5B-Instruct" \
  --data-dir "latin_head_deprel_data_macsafe_2048" \
  --output-dir "hf_outputs/interactive-smoke-a100-noqlora" \
  --max-seq-length 1024 \
  --max-steps 10 \
  --bf16 \
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
```

This smoke test has been verified on `xgph13` with an A100 MIG slice.

## Slurm Job

Submit full training with a requested A100 40 GB GPU:

```sh
cd ~/tum_NLP_Prac/experiments/latin_lora_hf_ft

MODEL="Qwen/Qwen2.5-0.5B-Instruct" \
DATA_DIR="latin_head_deprel_data" \
OUTPUT_DIR="hf_outputs/qwen25-05b-head-deprel-a100-lora-full" \
MAX_SEQ_LENGTH=4096 \
MAX_STEPS=-1 \
EPOCHS=3 \
PRECISION=bf16 \
sbatch --gpus=a100-40 run_training.slurm
```

The successful full run produced:

```text
eval_loss: 0.5038
train_loss: 0.7946
```

Useful commands:

```sh
squeue -u "$USER"
tail -f slurm-latin-qwen-hf-lora-<jobid>.out
scancel <jobid>
```

The saved adapter is intentionally ignored by Git:

```text
hf_outputs/qwen25-05b-head-deprel-a100-lora-full/
```

Defaults:

```text
MODEL=Qwen/Qwen2.5-0.5B-Instruct
DATA_DIR=latin_head_deprel_data
OUTPUT_DIR=hf_outputs/qwen25-05b-head-deprel-lora
MAX_SEQ_LENGTH=4096
PRECISION=fp16
```

The Slurm script defaults to `fp16` for older GPUs. Use `PRECISION=bf16` with
A100/H100 jobs.

Override example:

```sh
MODEL="Qwen/Qwen2.5-1.5B-Instruct" \
DATA_DIR="latin_head_deprel_data" \
OUTPUT_DIR="hf_outputs/qwen25-15b-head-deprel-lora" \
MAX_SEQ_LENGTH=4096 \
PRECISION=fp16 \
sbatch run_training.slurm
```

## Render Predictions

After inference, render model-generated `HEAD<TAB>DEPREL` lines back into
CoNLL-U for scoring:

```sh
python3 scripts/render_head_deprel_to_conllu.py \
  --input-jsonl latin_head_deprel_data/test.jsonl \
  --pred-jsonl runs/my_model_head_deprel_predictions.jsonl \
  --output-conllu runs/my_model.conllu \
  --report runs/my_model_render_report.json
```

For a smoke subset, add the same limit used during generation:

```sh
--limit 5
```

## Evaluate Adapter

Evaluation is strict: prediction rendering fails unless the model produces
exactly one `HEAD<TAB>DEPREL` line per syntactic token.

Smoke-test the base model:

```sh
LIMIT=5 \
RUN_ID="qwen25-base-head-deprel-smoke-005" \
SYSTEM_NAME="base_qwen25_head_deprel" \
ADAPTER_PATH="" \
PRECISION=bf16 \
MAX_NEW_TOKENS=1024 \
sbatch --gpus=a100-40 run_evaluation.slurm
```

Smoke-test the adapter:

```sh
LIMIT=5 \
RUN_ID="qwen25-lora-head-deprel-smoke-005" \
SYSTEM_NAME="lora_qwen25_head_deprel" \
ADAPTER_PATH="hf_outputs/qwen25-05b-head-deprel-a100-lora-full" \
PRECISION=bf16 \
MAX_NEW_TOKENS=1024 \
sbatch --gpus=a100-40 run_evaluation.slurm
```

Current strict smoke-test status:

- base model: unscoreable, generated too few dependency lines
- LoRA adapter: unscoreable, generated extra dependency lines

These are useful failures: training works, but inference needs stronger output
boundary control before UAS/LAS can be reported for the adapted model.

Once smoke rendering succeeds, run the full held-out test split:

```sh
RUN_ID="qwen25-head-deprel-full-001" \
SYSTEM_NAME="lora_qwen25_head_deprel" \
ADAPTER_PATH="hf_outputs/qwen25-05b-head-deprel-a100-lora-full" \
PRECISION=bf16 \
MAX_NEW_TOKENS=1024 \
sbatch --gpus=a100-40 run_evaluation.slurm
```

Outputs:

```text
runs/<run_id>/
  gold.conllu
  head_deprel_predictions.jsonl
  pred.conllu
  render_report.json
  format_report.txt
  hf_lora_head_deprel.score.txt
  summary.md
```
