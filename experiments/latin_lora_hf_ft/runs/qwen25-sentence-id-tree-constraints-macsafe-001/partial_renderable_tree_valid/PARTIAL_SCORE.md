# Partial Renderable-Subset Score

This is a diagnostic score over only the sentences whose predictions rendered successfully and passed a basic dependency-tree validity check. It is not an official full-split score.

- Total sentences: 58
- Renderable sentences: 58
- Tree-valid sentences scored: 55
- Render-excluded sentences: 0
- Tree-excluded sentences: 3
- Render-excluded indices: 
- Tree-excluded sent_ids: SenHerFu-P-15-527, SenHerFu-P-15-131, TacGerma-Q-01-279

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_sentence_id_tree_constraints_macsafe_partial | 100.00 | 43.17 | 35.79 | 35.44 | 29.80 | 35.44 |
