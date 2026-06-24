# Penalized Full Score

This score keeps every sentence in the split. Any rendered prediction with an invalid dependency tree is replaced by an adversarial dummy valid tree before running the official CoNLL-18 scorer.

- Total sentences: 85
- Raw tree-valid sentences: 76
- Dummy-replaced sentences: 9
- Dummy-replaced sent_ids: SenHerFu-P-15-72, SenHerFu-P-15-335, SenHerFu-P-15-89, SenHerFu-P-15-449, TacGerma-Q-01-97, SenHerFu-P-15-50, TacGerma-Q-01-236, SenHerFu-P-15-306, TacGerma-Q-01-102
- Dummy strategy: choose a gold non-root leaf as dummy root; attach all other tokens to it; use `dep:dummy`.
- Guarantee: multi-token dummy replacements receive 0 UAS/LAS for that sentence.

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_3b_sentence_id_full_penalized | 100.00 | 62.11 | 58.03 | 54.13 | 51.83 | 54.13 |
