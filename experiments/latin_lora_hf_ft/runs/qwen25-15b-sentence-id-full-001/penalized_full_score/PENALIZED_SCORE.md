# Penalized Full Score

This score keeps every sentence in the split. Any rendered prediction with an invalid dependency tree is replaced by an adversarial dummy valid tree before running the official CoNLL-18 scorer.

- Total sentences: 85
- Raw tree-valid sentences: 77
- Dummy-replaced sentences: 8
- Dummy-replaced sent_ids: SenHerFu-P-15-75, TacGerma-Q-01-125, SenHerFu-P-15-449, TacGerma-Q-01-92, SenHerFu-P-15-521, TacGerma-Q-01-236, TacGerma-Q-01-221, TacGerma-Q-01-87
- Dummy strategy: choose a gold non-root leaf as dummy root; attach all other tokens to it; use `dep:dummy`.
- Guarantee: multi-token dummy replacements receive 0 UAS/LAS for that sentence.

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_15b_sentence_id_full_penalized | 100.00 | 51.16 | 47.07 | 42.71 | 40.44 | 42.71 |
