# Partial Renderable-Subset Score

This is a diagnostic score over only the sentences whose predictions rendered successfully and passed a basic dependency-tree validity check. It is not an official full-split score.

- Total sentences: 58
- Renderable sentences: 57
- Tree-valid sentences scored: 55
- Render-excluded sentences: 1
- Tree-excluded sentences: 2
- Render-excluded indices: 57
- Tree-excluded sent_ids: SenHerFu-P-15-401, TacGerma-Q-01-93

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_sentence_id_head_deprel_partial | 100.00 | 49.36 | 40.47 | 39.02 | 33.70 | 39.02 |
