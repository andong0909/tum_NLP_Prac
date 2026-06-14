#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_examples(path, limit):
    examples = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit is not None:
        examples = examples[:limit]
    return examples


def build_prompt(tokenizer, messages):
    prompt_messages = messages[:2]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in prompt_messages
    ) + "\n\nASSISTANT:\n"


def clean_generation(text):
    for marker in ["<|im_end|>", "</s>", "<end_of_turn>"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate HEAD/DEPREL predictions with a Hugging Face base model and optional LoRA adapter."
    )
    parser.add_argument("--input", type=Path, default=Path("latin_head_deprel_data/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter_path if args.adapter_path else args.model,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
    model.eval()

    examples = load_examples(args.input, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, 1):
            prompt = build_prompt(tokenizer, example["messages"])
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, encoded["input_ids"].shape[1] :]
            prediction = clean_generation(tokenizer.decode(new_tokens, skip_special_tokens=True))
            sent_id = next(
                (
                    line.split("=", 1)[1].strip()
                    for line in example["messages"][1]["content"].splitlines()
                    if line.startswith("# sent_id")
                ),
                str(index),
            )
            handle.write(
                json.dumps(
                    {"sent_id": sent_id, "assistant": prediction},
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(f"Generated {index}/{len(examples)}")

    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
