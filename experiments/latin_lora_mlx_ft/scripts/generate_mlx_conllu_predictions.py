#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load

from prompting import SYSTEM_PROMPT


def build_prompt(tokenizer, sentence, chat_template_config):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence},
    ]
    template_kwargs = json.loads(chat_template_config) if chat_template_config else {}
    if tokenizer.has_chat_template:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        return tokenizer.encode(prompt, add_special_tokens=False)
    return tokenizer.encode(f"{SYSTEM_PROMPT}\n\n{sentence}\n\n")


def clean_generation(text):
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[-1]
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    lines = [line.rstrip() for line in text.splitlines()]
    start = 0
    for index, line in enumerate(lines):
        if line.startswith("# sent_id = ") or line.startswith("# text = "):
            start = index
            break
    return "\n".join(lines[start:]).strip()


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoNLL-U predictions from base or LoRA-adapted MLX model."
    )
    parser.add_argument("--input", type=Path, default=Path("latin_lora_data/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--chat-template-config",
        default='{"enable_thinking": false}',
        help="JSON kwargs passed to tokenizer.apply_chat_template.",
    )
    args = parser.parse_args()

    examples = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        examples = examples[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load(
        args.model,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
        tokenizer_config={"trust_remote_code": None},
    )
    sampler = make_sampler(args.temp)

    predictions = []
    for index, example in enumerate(examples, 1):
        sentence = example["messages"][1]["content"]
        prompt = build_prompt(tokenizer, sentence, args.chat_template_config)
        generated = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        prediction = clean_generation(generated)
        predictions.append(prediction)
        print(f"Generated {index}/{len(examples)}")

    args.output.write_text("\n\n".join(predictions) + "\n", encoding="utf-8")
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
