#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from prompting import load_prompt, wrap_user_conllu


def format_feats(feats):
    if not feats:
        return "_"
    return "|".join(f"{key}={feats[key]}" for key in sorted(feats))


def value_or_underscore(value):
    if value is None or value == "":
        return "_"
    return str(value)


def expected_output_to_conllu(expected_output):
    lines = [
        f"# sent_id = {value_or_underscore(expected_output.get('sent_id'))}",
        f"# text = {expected_output['text']}",
    ]

    for token in expected_output["tokens"]:
        columns = [
            value_or_underscore(token.get("id")),
            value_or_underscore(token.get("form")),
            value_or_underscore(token.get("lemma")),
            value_or_underscore(token.get("upos")),
            value_or_underscore(token.get("xpos")),
            format_feats(token.get("feats")),
            value_or_underscore(token.get("head")),
            value_or_underscore(token.get("deprel")),
            "_",
            "_",
        ]
        lines.append("\t".join(columns))

    return "\n".join(lines)


def canonical_to_chat(example, system_prompt, wrap_input):
    if example["input"].get("format") == "conllu":
        user_content = example["input"]["conllu"]
        if wrap_input:
            user_content = wrap_user_conllu(user_content)
        assistant_content = example["expected_output"]["conllu"]
    else:
        user_content = example["input"]["text"]
        assistant_content = expected_output_to_conllu(example["expected_output"])

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def write_jsonl(path, examples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")


def render_chat_for_length(messages, tokenizer):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def sent_id_from_chat(example):
    user_content = example["messages"][1]["content"]
    for line in user_content.splitlines():
        if line.startswith("# sent_id"):
            return line.split("=", 1)[1].strip()
    return "_"


def filter_by_token_length(chat_examples, tokenizer_model, max_seq_tokens):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The --max-seq-tokens option requires transformers. "
            "Install the experiment requirements first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model,
        local_files_only=True,
        trust_remote_code=False,
    )

    kept = []
    dropped = []
    for index, example in enumerate(chat_examples, 1):
        rendered = render_chat_for_length(example["messages"], tokenizer)
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        item = {
            "index": index,
            "sent_id": sent_id_from_chat(example),
            "tokens": token_count,
            "lines": len(example["messages"][1]["content"].splitlines()),
        }
        if token_count <= max_seq_tokens:
            kept.append(example)
        else:
            dropped.append(item)

    return kept, dropped


def main():
    parser = argparse.ArgumentParser(
        description="Split canonical Latin dependency JSONL into MLX-LM chat JSONL."
    )
    parser.add_argument("input", type=Path, help="Canonical processed JSONL file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("latin_lora_data"),
        help="Output folder for train.jsonl, valid.jsonl, and test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap for a tiny first experiment.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Custom system prompt string. Defaults to scripts/prompting.py.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        default=None,
        help="File containing a custom system prompt.",
    )
    parser.add_argument(
        "--wrap-input",
        action="store_true",
        help="Wrap user CoNLL-U in <INPUT_CONLLU> tags.",
    )
    parser.add_argument(
        "--max-seq-tokens",
        type=int,
        default=None,
        help="Drop examples whose rendered chat sequence exceeds this tokenizer length.",
    )
    parser.add_argument(
        "--tokenizer-model",
        default="mlx-community/Qwen3.5-0.8B-MLX-4bit",
        help="Tokenizer used with --max-seq-tokens.",
    )
    args = parser.parse_args()

    examples = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    system_prompt = load_prompt(args.system_prompt, args.system_prompt_file)
    chat_examples = [
        canonical_to_chat(example, system_prompt, args.wrap_input)
        for example in examples
    ]

    dropped = []
    if args.max_seq_tokens is not None:
        chat_examples, dropped = filter_by_token_length(
            chat_examples,
            args.tokenizer_model,
            args.max_seq_tokens,
        )

    total = len(chat_examples)
    test_count = max(1, round(total * args.test_ratio)) if total >= 3 else 0
    valid_count = max(1, round(total * args.valid_ratio)) if total >= 3 else 0
    train_count = total - valid_count - test_count

    train = chat_examples[:train_count]
    valid = chat_examples[train_count : train_count + valid_count]
    test = chat_examples[train_count + valid_count :]

    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "valid.jsonl", valid)
    write_jsonl(args.out_dir / "test.jsonl", test)

    print(f"Read {len(examples)} examples from {args.input}")
    if args.max_seq_tokens is not None:
        print(
            f"Kept {total} examples at <= {args.max_seq_tokens} tokens; "
            f"dropped {len(dropped)} examples"
        )
        if dropped:
            dropped_path = args.out_dir / "dropped_too_long.jsonl"
            write_jsonl(
                dropped_path,
                [
                    {
                        "sent_id": item["sent_id"],
                        "tokens": item["tokens"],
                        "lines": item["lines"],
                    }
                    for item in dropped
                ],
            )
            print(f"Wrote dropped-example report to {dropped_path}")
    print(f"Wrote {len(train)} train examples to {args.out_dir / 'train.jsonl'}")
    print(f"Wrote {len(valid)} valid examples to {args.out_dir / 'valid.jsonl'}")
    print(f"Wrote {len(test)} test examples to {args.out_dir / 'test.jsonl'}")


if __name__ == "__main__":
    main()
