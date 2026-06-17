# Next Experiment Requirements: Valid Latin Dependency Parsing

This document defines the next research step after the H2/H3/H4 LoRA
experiments.
The goal is to move toward a usable parser rather than keep nudging output
formatting.

## 1. End Goal

Build a Latin dependency parsing workflow that can take Latin literary text and
produce scoreable CoNLL-U with reliable `HEAD` and `DEPREL` predictions.

The final workflow does not need to be a single free-form LLM generation step.
The preferred architecture is a controlled pipeline:

```text
Latin text
  -> tokenization / morphology / blank CoNLL-U skeleton
  -> dependency model predicts HEAD/DEPREL
  -> validator enforces CoNLL-U shape and tree sanity
  -> official CoNLL-18 evaluation
```

The immediate next milestone is narrower:

```text
Given a blank CoNLL-U sentence with tokenization, lemma, UPOS, XPOS, and FEATS,
predict valid HEAD/DEPREL rows that render into a valid dependency tree.
```

## 2. Lessons From Current Experiments

### H2/H4: Sentence-ID Baseline

Current best LLM protocol:

```text
Input: full blank CoNLL-U sentence
Target: ID<TAB>HEAD<TAB>DEPREL rows
```

The 0.5B H2 run established the format:

| Scope | Rendered | Tree-valid scored | UAS | LAS | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H2 sentence-ID | 57/58 | 55/58 | 49.36 | 40.47 | Best LLM diagnostic so far |

The 1.5B H4 run confirmed that model scale helps:

| Scope | Rendered | Tree-valid scored | UAS | LAS | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H4 Qwen2.5-1.5B Mac-safe | 58/58 | 58/58 | 66.33 | 60.44 | Best official LLM result so far |
| H4 Qwen2.5-1.5B full 85 | 85/85 | 77/85 | 64.16 | 59.04 | Best full-test diagnostic so far |

Main failure modes:

- occasional dependency cycles on the full 85-sentence test set
- official scorer cannot produce a full score when any sentence is cyclic

### H3: Token Count + END

The `# token_count = N` and `END` marker improved rendering but hurt tree
validity and score.

| Scope | Rendered | Tree-valid scored | UAS | LAS | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| H3 Mac-safe | 58/58 | 52/58 | 40.36 | 31.81 | Better rendering, worse trees |
| H3 Mac-safe adapter on full 85 | 82/85 | 62/85 | 32.95 | 25.75 | Generalization diagnostic |

Conclusion:

```text
Row-count control is not enough. The next bottleneck is dependency-tree validity.
```

## 3. Required Next Experiment

The next experiment must target valid trees directly while preserving the H4
score gains.

### Requirement R1: Preserve the H2 Input Shape

Use the full blank CoNLL-U sentence as input.

Reason:

- H2 performed better than H3.
- Full morphology and POS context are useful.
- Word-only and row-only diagnostics lose too much syntax.

Required input:

```text
# sent_id = ...
# text = ...
1 FORM LEMMA UPOS XPOS FEATS _ _ DEPS MISC
2 FORM LEMMA UPOS XPOS FEATS _ _ DEPS MISC
...
```

Optional additions:

- `# token_count = N`
- `# valid_tree_required = true`

Do not make `END` mandatory unless an ablation shows it helps without worsening
tree validity.

### Requirement R2: Add Explicit Tree Constraints to the Training Target

The system prompt and target contract must include the following constraints:

```text
Output one ID<TAB>HEAD<TAB>DEPREL row per syntactic token.
HEAD must be 0 or the ID of another syntactic token.
HEAD must never equal ID.
Exactly one row must have HEAD=0.
The output must form one acyclic dependency tree.
```

This is not expected to guarantee correctness by itself, but it targets the
observed failure mode directly: self-heads and cycles.

### Requirement R3: Keep Raw and Assisted Scores Separate

There must be three reporting modes:

| Score type | Meaning | Allowed transformations |
| --- | --- | --- |
| Official raw score | Model output renders and scores directly for all sentences. | None beyond deterministic skeleton rendering. |
| Partial diagnostic score | Excludes render failures and tree-invalid predictions. | Exclusion only, clearly reported. |
| Assisted valid-tree score | Applies post-hoc tree repair or fallback. | Allowed only if labeled as assisted. |

The official raw score is the main benchmark. Partial and assisted scores are
diagnostics only.

### Requirement R4: Add Tree-Validity Metrics Before Official Scoring

Every evaluation run must report:

- total sentences
- rendered sentences
- render-excluded sentences
- tree-valid sentences
- tree-excluded sentences
- self-head count
- multi-root count
- zero-root count
- missing-head count
- cycle count
- official score if possible
- partial score if official score fails

This turns the current """ERROR""" situation into measurable progress.

### Requirement R5: Add a Self-Head/Cycle Error Analyzer

The evaluator must produce a compact JSON file:

```json
{
  "total_sentences": 58,
  "tree_invalid": [
    {
      "sent_id": "SenHerFu-P-15-65",
      "error_type": "cycle",
      "details": "token 8 revisits 12"
    }
  ]
}
```

This should be generated for every run, not only after manual debugging.

## 4. Dataset Expansion Requirements

Dataset expansion is worth trying, but it must be controlled. More data can
help only if it teaches transferable dependency behavior without washing out
EvaLatin conventions.

Universal Dependencies is the preferred source because it uses a consistent
cross-lingual annotation framework for POS, morphology, and dependency
relations.

### Requirement D1: Latin Must Remain the Target Domain

EvaLatin remains the evaluation target.

Non-EvaLatin data must be used only for training or pretraining. It must not
enter the EvaLatin test set.

### Requirement D2: Use a Tiered Expansion Plan

Use tiers so we can tell what helped.

| Tier | Data | Motivation | Risk |
| --- | --- | --- | --- |
| 0 | EvaLatin only | Cleanest target-domain result. | Small data. |
| 1 | Additional Latin UD treebanks | Same language, more syntax. | Domain mismatch: medieval, biblical, prose-heavy. |
| 2 | Ancient Greek / Gothic / Old Church Slavonic / Sanskrit UD | Ancient, morphologically rich Indo-European languages. | Cross-lingual syntax and label distribution mismatch. |
| 3 | Broader morphologically rich UD data | More general tree-shape learning. | Highest noise and lowest interpretability. |

### Requirement D3: Recommended Candidate Treebanks

Start with these candidates:

| Candidate | Why it is relevant | Initial use |
| --- | --- | --- |
| UD_Latin-CIRCSE | Latin prose and poetry from different periods. Closest supplemental target. | First Latin expansion candidate. |
| UD_Latin-Perseus | Classical Latin material from the Ancient Greek and Latin Dependency Treebank conversion. | Useful but watch conversion artifacts. |
| UD_Latin-PROIEL | Large Latin PROIEL corpus with Vulgate and classical prose selections. | Good for volume and PROIEL-style annotation. |
| UD_Ancient_Greek-PROIEL | Ancient Greek PROIEL, similar annotation lineage to Latin PROIEL. | Cross-lingual syntax transfer candidate. |
| UD_Ancient_Greek-Perseus | Ancient Greek literary material. | Cross-lingual literary syntax candidate. |
| UD_Gothic-PROIEL | Gothic Bible translation, PROIEL lineage. | Small but structurally related ancient IE data. |
| UD_Old_Church_Slavonic-PROIEL | Canonical OCS data from PROIEL/TOROT. | Ancient IE with rich morphology. |
| UD_Sanskrit-Vedic | Ancient Indo-European, morphologically rich. | Use cautiously; script and syntax differ. |

### Requirement D4: Normalize Training Records Across Languages

All multilingual data must be converted into the same training contract:

```text
Input: blank CoNLL-U sentence with HEAD/DEPREL blanked
Target: ID<TAB>HEAD<TAB>DEPREL rows
```

Add optional metadata comments:

```text
# source_treebank = UD_Latin-CIRCSE
# language = la
# genre = poetry|prose|unknown
```

The model should not be asked to output the metadata.

### Requirement D5: Preserve a Latin-Heavy Sampling Ratio

The first multilingual experiment should not drown Latin.

Recommended sampling:

```text
60% EvaLatin / additional Latin
25% Ancient Greek + Gothic + OCS
15% Sanskrit or broader ancient IE
```

Alternative conservative sampling:

```text
80% Latin
20% related ancient IE
```

Use the conservative ratio first if training budget is limited.

### Requirement D6: Track Source-Specific Evaluation

The evaluation report must include:

- EvaLatin Mac-safe result
- EvaLatin full result
- optional dev-set result by source treebank
- error counts by source treebank

If multilingual data improves general tree validity but hurts EvaLatin LAS, it
is not yet a win.

## 5. Model Requirements

### Requirement M1: Keep Qwen2.5-1.5B H2 as the Main Control

The next experiment should compare against the Qwen2.5-1.5B H2 result, because
it is now the strongest LLM baseline.

Current control:

```text
Mac-safe official: UAS 66.33 / LAS 60.44 on 58/58
Full diagnostic: UAS 64.16 / LAS 59.04 on 77/85 tree-valid
```

Keep Qwen2.5-0.5B only as a cheap ablation/control when debugging format or
evaluation scripts.

### Requirement M2: Add ByT5 as a Separate Branch

ByT5 should be tested as a seq2seq model, not as a chat model.

Recommended ByT5 input:

```text
Predict Latin dependencies.
Return one ID<TAB>HEAD<TAB>DEPREL row per token.
HEAD cannot equal ID. Exactly one HEAD must be 0. The tree must be acyclic.

<blank CoNLL-U sentence>
```

Recommended ByT5 target:

```text
1<TAB>4<TAB>det
2<TAB>4<TAB>amod
...
```

Do not mix ByT5 results with Qwen results unless the training data and
evaluation split are identical.

### Requirement M3: Larger Models Are Now a Confirmed Positive Branch

Qwen2.5-1.5B improved substantially over Qwen2.5-0.5B. Larger Qwen models are
worth trying, but only if the run uses the same H2 target and reports the same
tree-validity diagnostics.

Recommended next larger-model candidates:

```text
Qwen/Qwen2.5-3B-Instruct
Qwen/Qwen2.5-7B-Instruct, only if GPU time and memory allow
```

Success should mean fewer full-test cycles or higher full-test partial LAS, not
only lower training loss.

## 6. Evaluation Gates

The next experiment is successful only if it passes gates in order.

### Gate 1: Gold-Path Validation

Generated gold targets must render and self-score 100.

Required:

```text
Mac-safe gold path: 58/58 render, 100 score
Full gold path: 85/85 render, 100 score
```

### Gate 2: Mac-Safe Raw Render Validity

Required minimum:

```text
58/58 rendered
at least 55/58 tree-valid
```

Target:

```text
58/58 rendered
58/58 tree-valid
```

### Gate 3: Mac-Safe Score Improvement

The next experiment must beat H2 on the same diagnostic basis.

Current 1.5B H4 target:

```text
58/58 tree-valid
UAS 66.33
LAS 60.44
```

Minimum meaningful improvement:

```text
58/58 tree-valid
LAS > 60.44
```

### Gate 4: Full-Set Generalization

Only run the full 85-sentence evaluation after Gate 2 passes.

Target:

```text
85/85 rendered
no official scorer crash
```

Diagnostic minimum:

```text
85/85 rendered
more than 77/85 tree-valid
LAS > 59.04 on comparable partial score
```

## 7. Proposed Experiment Sequence

### E1: Constraint Prompt on H2 Format

Purpose:

```text
Test whether explicit tree constraints reduce self-heads/cycles without END.
```

Change:

```text
Use H2 target format, but add HEAD != ID, exactly one root, acyclic tree constraints to system prompt.
```

Success:

```text
Mac-safe tree-valid count improves beyond 55/58 or LAS improves beyond 40.47.
```

Result:

```text
Rendered 58/58.
Tree-valid scored 55/58.
Partial UAS 43.17 / LAS 35.79.
This improved over H3 but did not beat the H2 LAS baseline.
```

### H4: Larger Qwen2.5-1.5B on H2 Format

Purpose:

```text
Test whether model scale improves parsing while keeping the same H2 data and target.
```

Result:

```text
Mac-safe official: 58/58 rendered and tree-valid, UAS 66.33 / LAS 60.44.
Full test: 85/85 rendered, 77/85 tree-valid, partial UAS 64.16 / LAS 59.04.
```

Conclusion:

```text
Scale helps. The remaining end-goal blocker is full-test tree validity.
```

### E2: Constraint Prompt on H3 Format

Purpose:

```text
Test whether END can be retained if tree constraints are stronger.
```

Success:

```text
58/58 renderable and at least 55/58 tree-valid.
```

### E3: Latin Expansion on H4/H2 Format

Purpose:

```text
Add Latin UD data before adding other languages.
```

Recommended sources:

```text
UD_Latin-CIRCSE
UD_Latin-Perseus
UD_Latin-PROIEL
```

Success:

```text
Improves full-set tree-valid count beyond 77/85 and LAS beyond 59.04 without
reducing Mac-safe official LAS below 60.44.
```

### E4: Related Ancient IE Expansion

Purpose:

```text
Teach more general dependency-tree behavior from morphologically rich ancient languages.
```

Recommended sources:

```text
UD_Ancient_Greek-PROIEL
UD_Ancient_Greek-Perseus
UD_Gothic-PROIEL
UD_Old_Church_Slavonic-PROIEL
UD_Sanskrit-Vedic
```

Success:

```text
Fewer cycles/self-heads on EvaLatin without lowering LAS.
```

### E5: ByT5-small

Purpose:

```text
Test whether a byte-level seq2seq model handles structured CoNLL-U transduction better than chat Qwen.
```

Success:

```text
Lower cycle rate than Qwen2.5-1.5B on identical data, with comparable or better
LAS.
```

## 8. Non-Goals

Do not optimize for:

- making partial scores look better by excluding more examples
- silently repairing predictions and calling them raw model scores
- regenerating full CoNLL-U free-form
- using the EvaLatin test set for training
- comparing multilingual models without identical evaluation splits

## 9. Immediate Implementation Checklist

1. Add a converter variant for H2 + tree-constraint prompt. Done:
   `convert_chat_conllu_to_sentence_id_tree_constraints.py`.
2. Add evaluator fields for self-head, root count, missing heads, and cycle
   count.
3. Train Qwen2.5-0.5B on Mac-safe H2 + constraints. Done as E1.
4. Compare against H2 and H3 on the same Mac-safe split. Done: E1 improves on
   H3 but remains below H2 LAS.
5. Train Qwen2.5-1.5B on full H2 data. Done as H4.
6. Record H4 as the new control: Mac-safe official LAS 60.44; full diagnostic
   LAS 59.04 on 77/85 tree-valid.
7. Improve full-test tree-validity reporting and target the 8 remaining cycles.
8. Build the Latin expansion dataset on the H2/H4 target.
9. After Latin expansion, branch into related ancient IE data.
10. Test ByT5-small on the same target and splits.

## 10. Source Notes

Universal Dependencies is the data source to prioritize for multilingual
expansion because it uses a consistent annotation framework across languages.
The UD project page lists the current release and treebanks, including Latin,
Ancient Greek, Gothic, Old Church Slavonic, and Sanskrit candidates.

Useful references:

- Universal Dependencies: https://universaldependencies.org/
- Hugging Face ByT5 model docs: https://huggingface.co/docs/transformers/model_doc/byt5
