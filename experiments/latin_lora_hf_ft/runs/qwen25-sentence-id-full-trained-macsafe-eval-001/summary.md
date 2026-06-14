# CoNLL-U Evaluation Summary

| System | UPOS | UAS | LAS | CLAS | MLAS | BLEX |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lora_qwen25_sentence_id_head_deprel | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |

## Render Error

```text
Traceback (most recent call last):
  File "/home/a/andong/tum_NLP_Prac/experiments/latin_lora_hf_ft/scripts/render_sentence_id_head_deprel_to_conllu.py", line 127, in <module>
    main()
  File "/home/a/andong/tum_NLP_Prac/experiments/latin_lora_hf_ft/scripts/render_sentence_id_head_deprel_to_conllu.py", line 106, in main
    raise ValueError(json.dumps({"errors": errors[:20], "error_count": len(errors)}, ensure_ascii=False, indent=2))
ValueError: {
  "errors": [
    {
      "index": 57,
      "sent_id": "TacGerma-Q-01-112",
      "error": "Predicted token IDs do not match input order: expected ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18'], got ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']",
      "prediction": "1\t17\tadvmod:tmod\n2\t7\tdiscourse\n3\t7\tadvmod:tmod\n4\t5\tnsubj\n5\t7\tnsubj:pass\n6\t7\tcase\n7\t0\troot\n8\t17\tadvmod\n9\t17\tnsubj\n10\t17\tconj\n11\t12\tcc\n12\t10\tconj\n13\t10\tobl\n14\t13\tconj\n15\t14\tcc\n16\t14\tnmod\n17\t10\tconj"
    }
  ],
  "error_count": 1
}
```
