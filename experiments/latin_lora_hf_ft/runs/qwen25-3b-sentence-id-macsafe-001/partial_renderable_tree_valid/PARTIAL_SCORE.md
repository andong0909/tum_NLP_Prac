# Partial Renderable-Subset Score

This is a diagnostic score over only the sentences whose predictions rendered successfully and passed a basic dependency-tree validity check. It is not an official full-split score.

- Total sentences: 58
- Renderable sentences: 58
- Tree-valid sentences scored: 55
- Render-excluded sentences: 0
- Tree-excluded sentences: 3
- Render-excluded indices: 
- Tree-excluded sent_ids: SenHerFu-P-15-335, TacGerma-Q-01-97, SenHerFu-P-15-50

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_3b_sentence_id_macsafe_partial | 100.00 | 74.37 | 67.92 | 65.69 | 63.88 | 65.69 |
