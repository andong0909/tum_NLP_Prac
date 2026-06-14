#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

from mlx_lm.generate import generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load


DEFAULT_SYSTEM_PROMPT = (
    "Predict Latin dependency arcs. Return only a JSON array of objects with "
    "integer id, integer head, and string deprel."
)


def read_prompt(prompt_file):
    if prompt_file:
        return prompt_file.read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT


def conllu_token_rows(conllu):
    rows = []
    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if not columns or "-" in columns[0] or "." in columns[0]:
            continue
        rows.append(columns)
    return rows


def compact_token_input(conllu):
    lines = ["TOKENS"]
    for columns in conllu_token_rows(conllu):
        while len(columns) < 10:
            columns.append("_")
        token_id, form, lemma, upos, _xpos, feats = columns[:6]
        lines.append("\t".join([token_id, form, lemma, upos, feats]))
    return "\n".join(lines)


def build_prompt(tokenizer, system_prompt, user_content, chat_template_config):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
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
    return tokenizer.encode(f"{system_prompt}\n\n{user_content}\n\n")


def clean_generation(text):
    for marker in ["<|im_start|>assistant", "<start_of_turn>model", "<|assistant|>"]:
        if marker in text:
            text = text.split(marker, 1)[-1]
    for marker in ["<|im_end|>", "<end_of_turn>", "</s>"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def extract_json_array(text):
    text = clean_generation(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed, None
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return [], "no_json_array"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return [], f"json_decode_error: {exc}"
    if not isinstance(parsed, list):
        return [], "json_value_is_not_array"
    return parsed, None


def root_token_id(rows):
    for columns in rows:
        if len(columns) > 3 and columns[3] in {"VERB", "AUX"}:
            return columns[0]
    return rows[0][0] if rows else "0"


def fallback_arc(columns, root_id):
    token_id = columns[0]
    upos = columns[3] if len(columns) > 3 else "_"
    if token_id == root_id:
        return 0, "root"
    if upos == "PUNCT":
        return int(root_id), "punct"
    if upos == "CCONJ":
        return int(root_id), "cc"
    if upos == "SCONJ":
        return int(root_id), "mark"
    return int(root_id), "dep"


def valid_deprel(value):
    return isinstance(value, str) and bool(value) and value != "_" and not value.isspace()


def normalize_arc(item):
    if not isinstance(item, dict):
        return None
    try:
        token_id = int(item["id"])
        head = int(item["head"])
    except (KeyError, TypeError, ValueError):
        return None
    deprel = item.get("deprel")
    if not valid_deprel(deprel):
        return None
    return token_id, head, deprel


def render_conllu(input_conllu, arcs, fallback):
    rows = conllu_token_rows(input_conllu)
    token_ids = {int(row[0]) for row in rows}
    root_id = root_token_id(rows)
    arc_by_id = {}
    duplicate_ids = 0
    invalid_items = 0

    for item in arcs:
        normalized = normalize_arc(item)
        if normalized is None:
            invalid_items += 1
            continue
        token_id, head, deprel = normalized
        if token_id not in token_ids or (head != 0 and head not in token_ids):
            invalid_items += 1
            continue
        if token_id in arc_by_id:
            duplicate_ids += 1
        arc_by_id[token_id] = (head, deprel)

    missing_ids = []
    rendered_lines = []
    for line in input_conllu.strip().splitlines():
        if line.startswith("#") or not line:
            rendered_lines.append(line)
            continue

        columns = line.split("\t")
        while len(columns) < 10:
            columns.append("_")
        columns = columns[:10]
        token_id_text = columns[0]
        if "-" in token_id_text or "." in token_id_text:
            rendered_lines.append("\t".join(columns))
            continue

        token_id = int(token_id_text)
        arc = arc_by_id.get(token_id)
        if arc is None:
            missing_ids.append(token_id)
            if fallback:
                arc = fallback_arc(columns, root_id)
            else:
                columns[6] = "_"
                columns[7] = "_"
                rendered_lines.append("\t".join(columns))
                continue

        columns[6] = str(arc[0])
        columns[7] = arc[1]
        rendered_lines.append("\t".join(columns))

    stats = {
        "token_count": len(token_ids),
        "valid_arc_count": len(arc_by_id),
        "missing_arc_count": len(missing_ids),
        "invalid_item_count": invalid_items,
        "duplicate_id_count": duplicate_ids,
        "fallback_used": bool(fallback and missing_ids),
    }
    return "\n".join(rendered_lines), stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate dependency arc JSON with MLX and render it to CoNLL-U."
    )
    parser.add_argument("--input", type=Path, default=Path("latin_lora_data/test.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-conllu", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="mlx-community/Qwen3.5-0.8B-MLX-4bit")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--system-prompt-file", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--chat-template-config",
        default='{"enable_thinking": false}',
        help="JSON kwargs passed to tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--fallback-root",
        action="store_true",
        help="Make every sentence scoreable by root-attaching missing/invalid arcs.",
    )
    args = parser.parse_args()

    examples = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError(f"No examples found in {args.input}")

    system_prompt = read_prompt(args.system_prompt_file)
    model, tokenizer = load(
        args.model,
        adapter_path=str(args.adapter_path) if args.adapter_path else None,
        tokenizer_config={"trust_remote_code": None},
    )
    sampler = make_sampler(args.temp)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_conllu.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    prediction_records = []
    conllu_blocks = []
    report_records = []

    for index, example in enumerate(examples, 1):
        input_conllu = example["messages"][1]["content"]
        user_content = compact_token_input(input_conllu)
        prompt = build_prompt(
            tokenizer,
            system_prompt,
            user_content,
            args.chat_template_config,
        )
        generated = generate(
            model,
            tokenizer,
            prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        arcs, parse_error = extract_json_array(generated)
        conllu, stats = render_conllu(input_conllu, arcs, args.fallback_root)
        sent_id = next(
            (
                line.split("=", 1)[1].strip()
                for line in input_conllu.splitlines()
                if line.startswith("# sent_id")
            ),
            str(index),
        )

        prediction_records.append(
            {
                "sent_id": sent_id,
                "raw_response": clean_generation(generated),
                "arcs": arcs,
                "parse_error": parse_error,
            }
        )
        report_records.append({"sent_id": sent_id, "parse_error": parse_error, **stats})
        conllu_blocks.append(conllu)
        print(f"Generated {index}/{len(examples)}")

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in prediction_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    args.output_conllu.write_text("\n\n".join(conllu_blocks) + "\n\n", encoding="utf-8")

    totals = {
        "sentences": len(report_records),
        "token_count": sum(item["token_count"] for item in report_records),
        "valid_arc_count": sum(item["valid_arc_count"] for item in report_records),
        "missing_arc_count": sum(item["missing_arc_count"] for item in report_records),
        "invalid_item_count": sum(item["invalid_item_count"] for item in report_records),
        "duplicate_id_count": sum(item["duplicate_id_count"] for item in report_records),
        "parse_error_count": sum(1 for item in report_records if item["parse_error"]),
        "fallback_sentence_count": sum(1 for item in report_records if item["fallback_used"]),
    }
    args.report.write_text(
        json.dumps({"totals": totals, "sentences": report_records}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote arc predictions to {args.output_jsonl}")
    print(f"Wrote rendered CoNLL-U to {args.output_conllu}")
    print(f"Wrote arc report to {args.report}")


if __name__ == "__main__":
    main()
