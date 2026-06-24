# Latin LoRA Experiment Log

This log records the major EvaLatin dependency-parsing experiments, why each was tried, what happened, and what to try next. It is intentionally a lab notebook rather than a polished paper result.

For a requirements-style plan for the next major experiment, including
tree-validity constraints and multilingual data expansion, see
[`NEXT_EXPERIMENT_REQUIREMENTS.md`](NEXT_EXPERIMENT_REQUIREMENTS.md).

## Current Best Read

The best LLM fine-tuning direction so far is the **sentence-level
`ID<TAB>HEAD<TAB>DEPREL` protocol** scaled to `Qwen/Qwen2.5-3B-Instruct`:

- input: full blank CoNLL-U sentence with tokenization, lemmas, POS, morphology, and blank `HEAD`/`DEPREL`
- target: one compact dependency row per syntactic token, in input order
- training: Hugging Face PEFT normal LoRA, bf16, A100 Slurm job
- final reporting target: full 85-sentence EvaLatin test split
- fairer score type: penalized full score, where invalid predicted trees are replaced by adversarial dummy valid trees

| Model | Score Type | Scope | UPOS | UAS | LAS | CLAS | MLAS | BLEX | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-3B LoRA | Penalized full | 85/85, 9 dummy-replaced | 100.00 | 62.11 | 58.03 | 54.13 | 51.83 | 54.13 | Best current full-split score |
| Qwen2.5-1.5B LoRA | Penalized full | 85/85, 8 dummy-replaced | 100.00 | 51.16 | 47.07 | 42.71 | 40.44 | 42.71 | Full-split comparison |
| Qwen2.5-3B LoRA | Partial diagnostic | 76/85 tree-valid only | 100.00 | 70.34 | 65.72 | 63.44 | 60.74 | 63.44 | Optimistic diagnostic |
| Qwen2.5-1.5B LoRA | Partial diagnostic | 77/85 tree-valid only | 100.00 | 64.16 | 59.04 | 56.30 | 53.31 | 56.30 | Optimistic diagnostic |

The 3B model is clearly stronger than 1.5B under penalized full scoring. The
main remaining failure mode is invalid dependency trees, not CoNLL-U formatting
or row count.

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
| H2 | Sentence-level `ID<TAB>HEAD<TAB>DEPREL` | Combine sentence context with row anchors learned from row-level diagnostics. | Input: full blank CoNLL-U. Target: one `ID<TAB>HEAD<TAB>DEPREL` line per token. Short system prompt. | Qwen2.5-0.5B, HF LoRA. Smoke and full Mac-safe runs. | 5-sentence smoke scored. Full Mac-safe run rendered 57/58, tree-valid subset 55/58 scored: UAS 49.36 / LAS 40.47. | Established the winning output format. Later H4 kept this target and improved it with a larger model. |
| H3 | Sentence-level `ID<TAB>HEAD<TAB>DEPREL` + `END` | Fix the H2 row-count and stopping failures directly. | Input: full blank CoNLL-U with `# token_count = N`. Target: exactly N `ID<TAB>HEAD<TAB>DEPREL` rows followed by `END`. | Qwen2.5-0.5B, HF LoRA, Mac-safe split, A100 Slurm job. | Mac-safe rendered 58/58, but only 52/58 were tree-valid: UAS 40.36 / LAS 31.81. Full-set diagnostic with the Mac-safe adapter rendered 82/85 and scored 62/85 tree-valid: UAS 32.95 / LAS 25.75. | `END` improves row-count control but worsens tree validity and score versus H2. Do not treat H3 as the best model; next work should target tree constraints. |
| E1 | Sentence-level `ID<TAB>HEAD<TAB>DEPREL` + tree constraints | Keep the better H2 output shape while targeting H3's major failure mode: self-heads and cycles. | Input: full blank CoNLL-U. Target: one `ID<TAB>HEAD<TAB>DEPREL` line per token. System prompt states `HEAD != ID`, exactly one root, and acyclic tree. | Qwen2.5-0.5B, HF LoRA, Mac-safe split, A100 Slurm job. | Mac-safe rendered 58/58 and scored 55/58 tree-valid: UAS 43.17 / LAS 35.79. Tree exclusions: `SenHerFu-P-15-527`, `SenHerFu-P-15-131`, `TacGerma-Q-01-279`. | The constraint prompt reduced invalid trees versus H3 and restored the H2 tree-valid count, but LAS remained below H2. Next step should improve tree validity metrics/evaluator and then test Latin data expansion or ByT5 on the same E1/H2-style target. |
| H4 | Larger Qwen2.5-1.5B sentence-ID model | Test whether model scale solves more of the task without changing the H2 target contract. | Input: full blank CoNLL-U. Target: one `ID<TAB>HEAD<TAB>DEPREL` line per token. Trained on the full H2 data and evaluated on Mac-safe plus full test. | Qwen2.5-1.5B, HF LoRA, full H2 data, A100 Slurm job. | Mac-safe rendered and scored 58/58 with UAS 66.33 / LAS 60.44. Full test rendered 85/85; partial tree-valid score over 77/85 was UAS 64.16 / LAS 59.04. | Model scale helps substantially. The next end-goal step is not more prompt nudging; it is reducing the remaining full-test cycles, then testing 3B/ByT5 or Latin expansion on the same protocol. |
| H5 | Larger Qwen2.5-3B sentence-ID model | Test whether scaling beyond 1.5B improves accuracy on the same H2 target. | Same full blank CoNLL-U input and `ID<TAB>HEAD<TAB>DEPREL` target as H4. Trained on full H2 data and evaluated on the full 85-sentence test split. | Qwen2.5-3B, HF LoRA, full H2 data, A100 Slurm job. | Full test rendered 85/85. Partial tree-valid score over 76/85 was UAS 70.34 / LAS 65.72. Penalized full score over all 85, with 9 invalid trees dummy-replaced, was UAS 62.11 / LAS 58.03. | 3B is the strongest model by penalized full score. Next work should target invalid-tree reduction, tree-valid decoding/repair, or compare 7B/ByT5 with the same penalized scoring protocol. |
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
| `latin_sentence_id_tree_constraints_data/` | Full E1 split using the H2 output shape with explicit tree-validity constraints. | Yes |
| `latin_sentence_id_tree_constraints_data_macsafe_2048/` | Mac-safe E1 split for the next Slurm training run. | Yes |
| `latin_row_head_deprel_data_{minimal,strict,verbose}/` | Prompt wording diagnostics for one-row examples. | Yes |
| `latin_token_head_deprel_data*/` | Token-level sentence-context diagnostic data. | Yes |
| `latin_word_lines_head_deprel_data*/` | Compact word-line diagnostic data. | Yes |
| `runs/qwen25-sentence-id-full-trained-macsafe-eval-001/` | Synced Mac-safe evaluation artifacts and partial scores. | Yes |
| `runs/qwen25-sentence-id-end-macsafe-001/` | `END` protocol Mac-safe evaluation artifacts and partial tree-valid score. | Yes |
| `runs/qwen25-sentence-id-end-macsafe-adapter-on-full-001/` | Full-test diagnostic using the Mac-safe-trained `END` adapter. | Yes |
| `runs/qwen25-sentence-id-tree-constraints-macsafe-001/` | E1 tree-constrained Mac-safe evaluation artifacts and partial score. | Yes |
| `runs/qwen25-15b-sentence-id-macsafe-001/` | Qwen2.5-1.5B Mac-safe official evaluation artifacts. | Yes |
| `runs/qwen25-15b-sentence-id-full-001/` | Qwen2.5-1.5B full-test artifacts, partial score, and penalized full score. | Yes |
| `runs/qwen25-3b-sentence-id-macsafe-001/` | Qwen2.5-3B Mac-safe rendered artifacts and partial tree-valid score. | Yes |
| `runs/qwen25-3b-sentence-id-full-001/` | Qwen2.5-3B full-test artifacts, partial score, and penalized full score. | Yes |
| `hf_outputs/qwen25-05b-sentence-id-head-deprel-a100-lora-full/` | Trained adapter weights on cluster. | No, ignored because large |
| `hf_outputs/qwen25-05b-sentence-id-head-deprel-end-macsafe/` | Trained `END` adapter on the cluster. | No, ignored because large |
| `hf_outputs/qwen25-15b-sentence-id-full/` | Trained Qwen2.5-1.5B H2 adapter on the cluster. | No, ignored because large |
| `hf_outputs/qwen25-3b-sentence-id-full/` | Trained Qwen2.5-3B H2 adapter on the cluster. | No, ignored because large |

## Near-Term Plan

1. Treat Qwen2.5-3B H2 as the new full-test baseline.

   The 3B run reaches penalized full LAS 58.03 on the full 85-sentence split,
   compared with 47.07 for 1.5B. Future experiments should compare against
   penalized full scores first and partial diagnostics second.

2. Improve tree-validity diagnostics in the evaluator.

   H5 renders all full-test sentences, so the main remaining blocker is invalid
   tree structure. Before another major training run, the evaluator should
   summarize outside-head errors, self-heads, root-count errors, missing heads,
   and cycles automatically.

3. Add tree-validity constraints or decoding-time validation only if clearly
   separated from raw model scoring.

   The most common remaining full-test failures are cycles. Possible next
   approaches are constrained decoding, reranking, or a post-hoc MST repair, but
   repaired scores must be labeled assisted rather than raw.

4. Try Latin data expansion or ByT5 on the H2 target.

   ByT5 may be a better text-to-text fit than chat-style Qwen, while more Latin
   UD data may teach more dependency structure. The core metric should be
   whether either path reduces cycles on the full 85-sentence split while
   preserving Mac-safe LAS.

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
