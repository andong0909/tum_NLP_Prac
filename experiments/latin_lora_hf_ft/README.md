# Latin LoRA Hugging Face Fine-Tuning

CUDA/Hugging Face PEFT experiment for fine-tuning a small causal language model
to predict EvaLatin dependency columns.

This folder is for remote GPU clusters. The Apple MLX experiment and data
preparation live in:

```text
../latin_lora_mlx_ft/
```

For the chronological record of attempts, motivations, outcomes, and next
steps, see [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## Current Status

The most successful protocol so far is the sentence-level
`ID<TAB>HEAD<TAB>DEPREL` variant. It keeps the full blank CoNLL-U sentence as
input, uses a short system prompt, and asks the model to emit one compact row
per syntactic token.

Latest result from the Mac-safe 2048-token split:

```text
Model: Qwen/Qwen2.5-0.5B-Instruct
Adapter: hf_outputs/qwen25-05b-sentence-id-head-deprel-a100-lora-full/
Training data: latin_sentence_id_head_deprel_data_macsafe_2048/
Training: normal LoRA, bf16, 3 epochs, A100 40 GB Slurm job
Evaluation data: latin_sentence_id_head_deprel_data_macsafe_2048/test.jsonl
```

The official scorer could not run on all 58 Mac-safe test sentences because:

- 1 sentence failed rendering: `TacGerma-Q-01-112`
- 2 rendered sentences had invalid dependency trees: `SenHerFu-P-15-401`, `TacGerma-Q-01-93`

Partial diagnostic score over the remaining 55/58 renderable and tree-valid
sentences:

| System | Scope | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `lora_qwen25_sentence_id_head_deprel_partial` | 55/58 Mac-safe test sentences | 100.00 | 49.36 | 40.47 | 39.02 | 33.70 | 39.02 |

This is not an official full-split score. It is a useful diagnostic that shows
the sentence-ID protocol is scoreable for most short examples and materially
better than the original unanchored `HEAD<TAB>DEPREL` target.

Next protocol change to try: add `# token_count = N` to the user message and an
explicit `END` line to the assistant target, so the model learns when to stop.

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

### Sentence-Level ID/HEAD/DEPREL Variant

This is the recommended next candidate for the actual parsing task. It keeps
the full blank CoNLL-U sentence as input, but makes the target row-anchored:

```text
ID<TAB>HEAD<TAB>DEPREL
ID<TAB>HEAD<TAB>DEPREL
...
```

The system prompt is intentionally short:

```text
Predict Latin dependencies. Given blank CoNLL-U, output only ID<TAB>HEAD<TAB>DEPREL rows, one per syntactic token, in input order.
```

This keeps sentence context while avoiding the earlier ambiguity of returning
bare `HEAD<TAB>DEPREL` lines.

Build the splits:

```sh
python3 scripts/convert_chat_conllu_to_sentence_id_head_deprel.py \
  --input-dir ../latin_lora_mlx_ft/latin_lora_data \
  --out-dir latin_sentence_id_head_deprel_data

python3 scripts/convert_chat_conllu_to_sentence_id_head_deprel.py \
  --input-dir ../latin_lora_mlx_ft/latin_lora_data_macsafe_2048 \
  --out-dir latin_sentence_id_head_deprel_data_macsafe_2048
```

Smoke-train:

```sh
DATA_DIR="latin_sentence_id_head_deprel_data_macsafe_2048" \
OUTPUT_DIR="hf_outputs/qwen25-05b-sentence-id-head-deprel-smoke" \
MAX_SEQ_LENGTH=2048 \
MAX_STEPS=100 \
PRECISION=bf16 \
./run_training_smoke.sh
```

Full Slurm run:

```sh
DATA_DIR="latin_sentence_id_head_deprel_data" \
OUTPUT_DIR="hf_outputs/qwen25-05b-sentence-id-head-deprel-a100-lora-full" \
MAX_SEQ_LENGTH=4096 \
MAX_STEPS=-1 \
EPOCHS=3 \
PRECISION=bf16 \
sbatch run_training.slurm
```

Evaluate:

```sh
LIMIT=5 \
RUN_ID="qwen25-sentence-id-lora-smoke-005" \
SYSTEM_NAME="lora_qwen25_sentence_id_head_deprel" \
ADAPTER_PATH="hf_outputs/qwen25-05b-sentence-id-head-deprel-smoke" \
TEST_FILE="latin_sentence_id_head_deprel_data/test.jsonl" \
MAX_NEW_TOKENS=1024 \
PRECISION=bf16 \
./run_sentence_id_evaluation.sh qwen25-sentence-id-lora-smoke-005
```

### Token-Level Variant

The sentence-level `HEAD<TAB>DEPREL` protocol is efficient, but brittle: the
model must generate exactly one line per token. The token-level variant turns
each sentence into one example per syntactic token.

Input:

```text
full CoNLL-U sentence with blank HEAD/DEPREL
target token ID/FORM/LEMMA/UPOS/XPOS/FEATS
```

Output:

```text
ID<TAB>HEAD<TAB>DEPREL
```

This makes each generation tiny and easier to validate. The tradeoff is that
the model predicts arcs independently, so a later validation step still needs to
check for impossible trees, missing roots, cycles, or inconsistent global
structure.

Build the token-level splits:

```sh
python3 scripts/convert_chat_conllu_to_token_head_deprel.py \
  --input-dir ../latin_lora_mlx_ft/latin_lora_data \
  --out-dir latin_token_head_deprel_data

python3 scripts/convert_chat_conllu_to_token_head_deprel.py \
  --input-dir ../latin_lora_mlx_ft/latin_lora_data_macsafe_2048 \
  --out-dir latin_token_head_deprel_data_macsafe_2048
```

Smoke-train the token-level format:

```sh
DATA_DIR="latin_token_head_deprel_data_macsafe_2048" \
OUTPUT_DIR="hf_outputs/qwen25-05b-token-head-deprel-smoke" \
MAX_SEQ_LENGTH=2048 \
MAX_STEPS=20 \
PRECISION=bf16 \
./run_training_smoke.sh
```

Submit a full token-level Slurm run:

```sh
DATA_DIR="latin_token_head_deprel_data" \
OUTPUT_DIR="hf_outputs/qwen25-05b-token-head-deprel-a100-lora-full" \
MAX_SEQ_LENGTH=4096 \
MAX_STEPS=-1 \
EPOCHS=3 \
PRECISION=bf16 \
sbatch run_training.slurm
```

For token-level inference, reuse the HF generator with a small output budget:

```sh
python3 scripts/generate_hf_head_deprel_predictions.py \
  --input latin_token_head_deprel_data/test.jsonl \
  --output runs/qwen25-token/head_deprel_token_predictions.jsonl \
  --model "Qwen/Qwen2.5-0.5B-Instruct" \
  --adapter-path hf_outputs/qwen25-05b-token-head-deprel-a100-lora-full \
  --max-input-length 4096 \
  --max-new-tokens 24 \
  --bf16
```

Then render token predictions back into CoNLL-U:

```sh
python3 scripts/render_token_head_deprel_to_conllu.py \
  --input-jsonl latin_token_head_deprel_data/test.jsonl \
  --pred-jsonl runs/qwen25-token/head_deprel_token_predictions.jsonl \
  --output-conllu runs/qwen25-token/pred.conllu \
  --report runs/qwen25-token/render_report.json
```

Or use the wrapper, which limits smoke tests by complete sentences:

```sh
LIMIT_SENTENCES=5 \
RUN_ID="qwen25-token-lora-smoke-005" \
SYSTEM_NAME="lora_qwen25_token_head_deprel" \
ADAPTER_PATH="hf_outputs/qwen25-05b-token-head-deprel-a100-lora-full" \
TEST_FILE="latin_token_head_deprel_data/test.jsonl" \
MAX_NEW_TOKENS=24 \
PRECISION=bf16 \
./run_token_evaluation.sh qwen25-token-lora-smoke-005
```

### Compact Word-Lines Variant

This is a diagnostic format for testing whether the model can reliably output
the correct number of dependency rows. The prompt removes the full CoNLL-U
table and shows only word lines:

```text
# sent_id = ...
# text = ...
1<TAB>quae
2<TAB>fera
3<TAB>tyranni
...
```

The target is:

```text
1<TAB>4<TAB>det
2<TAB>4<TAB>amod
3<TAB>4<TAB>nmod
...
```

This is not expected to be the strongest linguistic setup because the model
loses lemma, UPOS, morphology, and XPOS. It is mainly a formatting/capability
test: can the model produce one structured dependency row per input word?

Build the compact splits:

```sh
python3 scripts/convert_chat_conllu_to_word_lines_head_deprel.py \
  --input-dir ../latin_lora_mlx_ft/latin_lora_data \
  --out-dir latin_word_lines_head_deprel_data

python3 scripts/convert_chat_conllu_to_word_lines_head_deprel.py \
  --input-dir ../latin_lora_mlx_ft/latin_lora_data_macsafe_2048 \
  --out-dir latin_word_lines_head_deprel_data_macsafe_2048
```

Smoke-train the compact format:

```sh
DATA_DIR="latin_word_lines_head_deprel_data_macsafe_2048" \
OUTPUT_DIR="hf_outputs/qwen25-05b-word-lines-head-deprel-smoke" \
MAX_SEQ_LENGTH=1024 \
MAX_STEPS=50 \
PRECISION=bf16 \
./run_training_smoke.sh
```

Submit a full compact-format Slurm run:

```sh
DATA_DIR="latin_word_lines_head_deprel_data" \
OUTPUT_DIR="hf_outputs/qwen25-05b-word-lines-head-deprel-a100-lora-full" \
MAX_SEQ_LENGTH=2048 \
MAX_STEPS=-1 \
EPOCHS=3 \
PRECISION=bf16 \
sbatch run_training.slurm
```

Evaluate a compact-format adapter:

```sh
LIMIT=5 \
RUN_ID="qwen25-word-lines-lora-smoke-005" \
SYSTEM_NAME="lora_qwen25_word_lines_head_deprel" \
ADAPTER_PATH="hf_outputs/qwen25-05b-word-lines-head-deprel-a100-lora-full" \
TEST_FILE="latin_word_lines_head_deprel_data/test.jsonl" \
MAX_NEW_TOKENS=512 \
PRECISION=bf16 \
./run_word_lines_evaluation.sh qwen25-word-lines-lora-smoke-005
```

### Single-Row Diagnostic Variant

This is the most artificial troubleshooting setup. Each JSONL record contains
only one syntactic CoNLL-U token row, regardless of sentence boundaries:

```text
1<TAB>quae<TAB>quis<TAB>PRON<TAB>K<TAB>Case=Nom|...<TAB>_<TAB>_<TAB>_<TAB>...
```

The target is exactly one row:

```text
1<TAB>4<TAB>det
```

This should not be treated as a real dependency parser because a dependency
decision needs sentence context. The purpose is narrower: compare prompt wording
and check whether the model can obey the required output shape when the task is
reduced to a single CoNLL-U row.

Build prompt variants:

```sh
for style in minimal strict verbose; do
  python3 scripts/convert_chat_conllu_to_row_head_deprel.py \
    --input-dir ../latin_lora_mlx_ft/latin_lora_data \
    --out-dir "latin_row_head_deprel_data_${style}" \
    --prompt-style "$style"
done
```

Recommended first smoke train:

```sh
DATA_DIR="latin_row_head_deprel_data_strict" \
OUTPUT_DIR="hf_outputs/qwen25-05b-row-head-deprel-strict-smoke" \
MAX_SEQ_LENGTH=512 \
MAX_STEPS=100 \
PRECISION=bf16 \
./run_training_smoke.sh
```

Run the same command with `minimal` and `verbose` folders to compare prompt
wording:

```sh
DATA_DIR="latin_row_head_deprel_data_minimal" \
OUTPUT_DIR="hf_outputs/qwen25-05b-row-head-deprel-minimal-smoke" \
MAX_SEQ_LENGTH=512 \
MAX_STEPS=100 \
PRECISION=bf16 \
./run_training_smoke.sh

DATA_DIR="latin_row_head_deprel_data_verbose" \
OUTPUT_DIR="hf_outputs/qwen25-05b-row-head-deprel-verbose-smoke" \
MAX_SEQ_LENGTH=512 \
MAX_STEPS=100 \
PRECISION=bf16 \
./run_training_smoke.sh
```

Evaluate a row-level adapter:

```sh
LIMIT_SENTENCES=5 \
RUN_ID="qwen25-row-strict-smoke-005" \
SYSTEM_NAME="lora_qwen25_row_strict" \
ADAPTER_PATH="hf_outputs/qwen25-05b-row-head-deprel-strict-smoke" \
TEST_FILE="latin_row_head_deprel_data_strict/test.jsonl" \
MAX_INPUT_LENGTH=512 \
MAX_NEW_TOKENS=16 \
PRECISION=bf16 \
./run_row_evaluation.sh qwen25-row-strict-smoke-005
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
sbatch run_training.slurm
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
sbatch run_evaluation.slurm
```

Smoke-test the adapter:

```sh
LIMIT=5 \
RUN_ID="qwen25-lora-head-deprel-smoke-005" \
SYSTEM_NAME="lora_qwen25_head_deprel" \
ADAPTER_PATH="hf_outputs/qwen25-05b-head-deprel-a100-lora-full" \
PRECISION=bf16 \
MAX_NEW_TOKENS=1024 \
sbatch run_evaluation.slurm
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
sbatch run_evaluation.slurm
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
