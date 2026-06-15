# Latin LoRA Experiment Log

This log records the major EvaLatin dependency-parsing experiments, why each was tried, what happened, and what to try next. It is intentionally a lab notebook rather than a polished paper result.

## Current Best Read

The best LLM fine-tuning direction so far is the **sentence-level `ID<TAB>HEAD<TAB>DEPREL` protocol**:

- input: full blank CoNLL-U sentence with tokenization, lemmas, POS, morphology, and blank `HEAD`/`DEPREL`
- target: one compact dependency row per syntactic token, in input order
- model: `Qwen/Qwen2.5-0.5B-Instruct`
- training: Hugging Face PEFT normal LoRA, bf16, A100 Slurm job
- strongest diagnostic result: 55/58 Mac-safe test sentences scored after excluding one render failure and two tree-invalid predictions

| Scope | UPOS | UAS | LAS | CLAS | MLAS | BLEX | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Mac-safe 55/58 renderable + tree-valid subset | 100.00 | 49.36 | 40.47 | 39.02 | 33.70 | 39.02 | Partial diagnostic, not official |
| `END` Mac-safe 52/58 tree-valid subset | 100.00 | 40.36 | 31.81 | 30.53 | 26.44 | 30.53 | Partial diagnostic, worse than H2 |
| `END` Mac-safe adapter on full 62/85 tree-valid subset | 100.00 | 32.95 | 25.75 | 23.20 | 19.38 | 23.20 | Partial diagnostic, generalization check |

The result is promising because it is the first LoRA setup that mostly renders and scores. It is not yet a valid full-test result because `TacGerma-Q-01-112` failed rendering and `SenHerFu-P-15-401` / `TacGerma-Q-01-93` formed cycles.

## Experiment Table

| ID | Approach | Motivation | Data / Target | Model / Runtime | Outcome | Lesson / Next Action |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | UDPipe EvaLatin baseline | Establish a non-LLM reference and verify the official scorer workflow. | EvaLatin test CoNLL-U with gold tokenization/morphology; parser fills `HEAD`/`DEPREL`. | UDPipe API, `latin-evalatin24-240520`. | Verified scores: prose UAS 80.49 / LAS 75.20; poetry UAS 78.31 / LAS 72.36. | This remains the practical baseline to beat. LLM scores should be compared against it only when full-split scoring is valid. |
| B1 | UDPipe Latin model sweep | Check whether general Latin UDPipe models are competitive and get a broader baseline picture. | EvaLatin prose and poetry test files. | Six Latin UD 2.17 UDPipe models. | Best general model was CIRCSE: avg UAS 73.75 / LAS 67.58. | EvaLatin-specific UDPipe is stronger. General Latin models are useful comparison points but not the main target. |
| M0 | Apple MLX LoRA setup | Learn the fine-tuning loop locally on Mac before using cluster GPUs. | Initial chat JSONL from EvaLatin gold; full CoNLL-U output target. | MLX / `mlx_lm.lora`, small Qwen MLX model. | Training ran, but long examples caused truncation warnings and later OOM when sequence length was increased. | MLX is useful for learning and smoke tests, but the Mac is memory-limited for this task. Keep Mac-safe splits and move serious training to Slurm. |
| M1 | Full CoNLL-U generation | Try the most direct task: given Latin text or blank CoNLL-U, output valid full CoNLL-U. | User prompt contained Latin sentence or blank CoNLL-U; assistant target full CoNLL-U. | MLX Qwen 0.6B / Qwen-like small models. | Predictions were often malformed, with invalid column counts and unscoreable files. | Too much formatting burden. Do not ask a small LLM to regenerate full CoNLL-U. Render into a trusted skeleton instead. |
| H0 | HF/PEFT on Slurm | Move training from Mac to school GPU cluster after MLX OOMs. | Same EvaLatin-derived train/valid/test JSONL. | Hugging Face Trainer, PEFT LoRA, Qwen2.5-0.5B, A100 40 GB. | Environment was initially unstable: bf16 support, CPU fallback, PyTorch/CUDA package drift, and login-node CUDA import errors. | Use Slurm for all model work. Pin PyTorch to cu124, make QLoRA opt-in, and rebuild venv with `RECREATE_VENV=1` when polluted. |
| H1 | Sentence-level `HEAD<TAB>DEPREL` | Reduce output burden by predicting only dependency fields, then render them back into CoNLL-U. | Input: full blank CoNLL-U. Target: one `HEAD<TAB>DEPREL` line per token. | Qwen2.5-0.5B, HF LoRA, A100. | Training completed (`eval_loss 0.5038`, `train_loss 0.7946`) but evaluation failed when outputs had too few/too many lines. | Removing full CoNLL-U helped, but unanchored rows are brittle. Add token IDs to every output row. |
| D0 | Row-level diagnostic | Test whether the model can obey a tiny one-row output contract independent of syntax. | One JSONL example per CoNLL-U token row. Target: `ID<TAB>HEAD<TAB>DEPREL`. Prompt variants: minimal, strict, verbose. | Qwen2.5-0.5B, short LoRA smoke. | Format mostly worked; renderer produced valid CoNLL-U for many cases, but independent rows created cycles and invalid trees. | Good formatting diagnostic, bad final parser. Dependency parsing needs sentence context or tree constraints. |
| D1 | Token-level with sentence context | Reduce generation length while keeping full sentence context. | One example per target token; user includes full sentence plus token to label. Target: one `ID<TAB>HEAD<TAB>DEPREL` row. | HF LoRA-compatible data and renderer added. | Mechanically validated with gold render scoring 100. Not selected as main path because dataset becomes much larger and training was slow. | Keep as fallback if sentence-level generation remains too brittle. It can score per token but may need tree repair or constraints. |
| D2 | Compact word-lines diagnostic | Test row-count behavior with very small prompts, without full morphology. | Input: `ID<TAB>FORM` lines per sentence. Target: `ID<TAB>HEAD<TAB>DEPREL` lines. | HF-compatible data and renderer. | Gold-path validation scored 100. Not pursued deeply because syntax loses morphology/POS context. | Useful for output-shape debugging, not expected to be the strongest parser. |
| H2 | Sentence-level `ID<TAB>HEAD<TAB>DEPREL` | Combine sentence context with row anchors learned from row-level diagnostics. | Input: full blank CoNLL-U. Target: one `ID<TAB>HEAD<TAB>DEPREL` line per token. Short system prompt. | Qwen2.5-0.5B, HF LoRA. Smoke and full Mac-safe runs. | 5-sentence smoke scored. Full Mac-safe run rendered 57/58, tree-valid subset 55/58 scored: UAS 49.36 / LAS 40.47. | Current best direction. Remaining problems are early stopping on one sentence and cycles on two sentences. Add `token_count` and `END`. |
| H3 | Sentence-level `ID<TAB>HEAD<TAB>DEPREL` + `END` | Fix the H2 row-count and stopping failures directly. | Input: full blank CoNLL-U with `# token_count = N`. Target: exactly N `ID<TAB>HEAD<TAB>DEPREL` rows followed by `END`. | Qwen2.5-0.5B, HF LoRA, Mac-safe split, A100 Slurm job. | Mac-safe rendered 58/58, but only 52/58 were tree-valid: UAS 40.36 / LAS 31.81. Full-set diagnostic with the Mac-safe adapter rendered 82/85 and scored 62/85 tree-valid: UAS 32.95 / LAS 25.75. | `END` improves row-count control but worsens tree validity and score versus H2. Do not treat H3 as the best model; next work should target tree constraints. |
| S0 | Partial renderable/tree-valid scoring | Get an honest diagnostic score when a nearly valid run has a few failures. | Filter render failures, then filter dependency-tree-invalid predictions before scoring. | `score_renderable_sentence_id_subset.py`. | Produced clear 55/58 Mac-safe partial score and recorded exclusions. | Useful for analysis only. Do not present as official benchmark result. Official result still requires all sentences to render and form valid trees. |

## Important Failure Modes Observed

| Failure mode | What it looked like | Likely cause | Mitigation |
| --- | --- | --- | --- |
| Invalid CoNLL-U shape | Rows with fewer than 10 columns, relation labels in wrong columns. | Asking model to regenerate too much structure. | Render model predictions into a trusted original CoNLL-U skeleton. |
| Line-count mismatch | Expected N dependency rows, got N-1 or extra rows. | Model stops early or continues beyond task boundary. | Include token IDs, add `# token_count = N`, add `END`, and keep strict render validation. |
| Truncation | Last output row cut off, e.g. `33<TAB>1`. | `MAX_NEW_TOKENS` too low for long sentences. | Increase output cap or use token-count/END stopping protocol. |
| Repetition / rambling | Model generates rows far beyond actual token count. | No learned stop condition. | Add explicit `END`; optionally stop decoding at END during inference. |
| Cycles | Official scorer rejects rendered file with `There is a cycle in a sentence`. | Independent or weakly constrained arc predictions do not guarantee a tree. | Add tree-validity diagnostics; later consider constrained decoding or post-hoc tree repair only if clearly labeled. |
| CUDA/PyTorch import errors | `libtorch_cuda.so` failed to map, missing `typing_extensions`, CUDA 13 package drift. | Polluted venv or pip installing incompatible newest packages. | Use Slurm GPU nodes, pinned cu124 PyTorch, `RECREATE_VENV=1`, QLoRA opt-in only. |

## Current Artifacts

| Artifact | Purpose | Tracked? |
| --- | --- | --- |
| `latin_sentence_id_head_deprel_data/` | Full sentence-ID train/valid/test split. | Yes |
| `latin_sentence_id_head_deprel_data_macsafe_2048/` | Shorter Mac-safe sentence-ID split used for latest partial result. | Yes |
| `latin_sentence_id_head_deprel_end_data/` | Full sentence-ID split with `# token_count = N` input and `END` target marker. | Yes |
| `latin_sentence_id_head_deprel_end_data_macsafe_2048/` | Mac-safe sentence-ID + `END` split for the next priority training run. | Yes |
| `latin_row_head_deprel_data_{minimal,strict,verbose}/` | Prompt wording diagnostics for one-row examples. | Yes |
| `latin_token_head_deprel_data*/` | Token-level sentence-context diagnostic data. | Yes |
| `latin_word_lines_head_deprel_data*/` | Compact word-line diagnostic data. | Yes |
| `runs/qwen25-sentence-id-full-trained-macsafe-eval-001/` | Synced Mac-safe evaluation artifacts and partial scores. | Yes |
| `runs/qwen25-sentence-id-end-macsafe-001/` | `END` protocol Mac-safe evaluation artifacts and partial tree-valid score. | Yes |
| `runs/qwen25-sentence-id-end-macsafe-adapter-on-full-001/` | Full-test diagnostic using the Mac-safe-trained `END` adapter. | Yes |
| `hf_outputs/qwen25-05b-sentence-id-head-deprel-a100-lora-full/` | Trained adapter weights on cluster. | No, ignored because large |
| `hf_outputs/qwen25-05b-sentence-id-head-deprel-end-macsafe/` | Trained `END` adapter on the cluster. | No, ignored because large |

## Near-Term Plan

1. Add tree-validity constraints to the dependency target and prompt.

   The most common H3 failures were cycles, including self-head predictions
   such as `18 -> 18`. The next format should explicitly state:

   ```text
   HEAD must be 0 or another token ID. HEAD must never equal ID. Exactly one row must have HEAD=0.
   ```

   This should be tested on the Mac-safe split first and judged by both render
   validity and tree-validity counts.

2. Keep H2 as the current best LLM diagnostic baseline.

   H3 should be recorded as a formatting experiment, not as a replacement for
   H2, because it lowered Mac-safe partial LAS from 40.47 to 31.81.

3. Try ByT5-small with the best dependency target once the tree-validity
   constraint is explicit.

   ByT5 may be a better text-to-text fit than chat-style Qwen, but the core
   metric should be whether it reduces cycles and self-heads, not just whether
   it follows row-count instructions.

## Completed H3 Setup Notes

The sentence-ID format with explicit token count and end marker used:

   System prompt:

   ```text
   Predict Latin dependencies. Given blank CoNLL-U and # token_count, output exactly that many ID<TAB>HEAD<TAB>DEPREL rows, then END.
   ```

   Proposed target:

   ```text
   1<TAB>4<TAB>det
   2<TAB>4<TAB>amod
   ...
   END
   ```

   Status: data conversion, rendering, and gold-path scoring are implemented and
   validated for both Mac-safe and full test splits. Model evaluation showed
   improved rendering but worse tree validity.

## Reporting Rules

- **Official score**: all examples in the chosen split render, form valid dependency trees, and are scored by `conll18_ud_eval.py` without exclusions.
- **Partial diagnostic score**: any score that excludes render failures, cycles, malformed predictions, or other invalid outputs.
- **Assisted score**: any score using truncation repair, missing-row filling, tree repair, or fallback predictions. These must be labeled separately from raw model output.
