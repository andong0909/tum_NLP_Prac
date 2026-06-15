# Partial Renderable-Subset Score

This is a diagnostic score over only the sentences whose predictions rendered successfully and passed a basic dependency-tree validity check. It is not an official full-split score.

- Total sentences: 58
- Renderable sentences: 58
- Tree-valid sentences scored: 52
- Render-excluded sentences: 0
- Tree-excluded sentences: 6
- Render-excluded indices: 
- Tree-excluded sent_ids: SenHerFu-P-15-365, SenHerFu-P-15-65, SenHerFu-P-15-352, TacGerma-Q-01-231, TacGerma-Q-01-45, SenHerFu-P-15-131

# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_sentence_id_head_deprel_end_macsafe_partial | 100.00 | 40.36 | 31.81 | 30.53 | 26.44 | 30.53 |
