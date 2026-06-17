# Partial Renderable-Subset Score

This is a diagnostic score over only the sentences whose predictions rendered successfully and passed a basic dependency-tree validity check. It is not an official full-split score.

- Total sentences: 85
- Renderable sentences: 85
- Tree-valid sentences scored: 77
- Render-excluded sentences: 0
- Tree-excluded sentences: 8
- Render-excluded indices: 
- Tree-excluded sent_ids: SenHerFu-P-15-75, TacGerma-Q-01-125, SenHerFu-P-15-449, TacGerma-Q-01-92, SenHerFu-P-15-521, TacGerma-Q-01-236, TacGerma-Q-01-221, TacGerma-Q-01-87

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_15b_sentence_id_full_partial | 100.00 | 64.16 | 59.04 | 56.30 | 53.31 | 56.30 |
