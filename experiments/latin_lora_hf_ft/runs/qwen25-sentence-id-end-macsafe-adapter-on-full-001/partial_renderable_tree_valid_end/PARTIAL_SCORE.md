# Partial Renderable-Subset Score

This is a diagnostic score over only the sentences whose predictions rendered successfully and passed a basic dependency-tree validity check. It is not an official full-split score.

- Total sentences: 85
- Renderable sentences: 82
- Tree-valid sentences scored: 62
- Render-excluded sentences: 3
- Tree-excluded sentences: 20
- Render-excluded indices: 24, 31, 73
- Tree-excluded sent_ids: SenHerFu-P-15-65, SenHerFu-P-15-352, SenHerFu-P-15-166, TacGerma-Q-01-231, TacGerma-Q-01-45, SenHerFu-P-15-72, SenHerFu-P-15-2, TacGerma-Q-01-78-bis, SenHerFu-P-15-89, SenHerFu-P-15-54, TacGerma-Q-01-92, SenHerFu-P-15-546, SenHerFu-P-15-483, SenHerFu-P-15-521, TacGerma-Q-01-7, SenHerFu-P-15-131, SenHerFu-P-15-219, SenHerFu-P-15-306, TacGerma-Q-01-87, TacGerma-Q-01-226

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_sentence_id_head_deprel_end_macsafe_on_full_partial | 100.00 | 32.95 | 25.75 | 23.20 | 19.38 | 23.20 |
