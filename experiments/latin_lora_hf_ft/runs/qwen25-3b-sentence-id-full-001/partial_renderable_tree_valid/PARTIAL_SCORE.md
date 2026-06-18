# Partial Renderable-Subset Score

This is a diagnostic score over only the sentences whose predictions rendered successfully and passed a basic dependency-tree validity check. It is not an official full-split score.

- Total sentences: 85
- Renderable sentences: 85
- Tree-valid sentences scored: 76
- Render-excluded sentences: 0
- Tree-excluded sentences: 9
- Render-excluded indices: 
- Tree-excluded sent_ids: SenHerFu-P-15-72, SenHerFu-P-15-335, SenHerFu-P-15-89, SenHerFu-P-15-449, TacGerma-Q-01-97, SenHerFu-P-15-50, TacGerma-Q-01-236, SenHerFu-P-15-306, TacGerma-Q-01-102

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_3b_sentence_id_full_partial | 100.00 | 70.34 | 65.72 | 63.44 | 60.74 | 63.44 |
