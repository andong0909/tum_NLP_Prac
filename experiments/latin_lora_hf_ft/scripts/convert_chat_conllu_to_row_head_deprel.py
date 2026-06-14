#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PROMPTS = {
    "minimal": (
        "Given one CoNLL-U token row, output exactly one line: "
        "ID<TAB>HEAD<TAB>DEPREL."
    ),
    "strict": (
        "You are a CoNLL-U dependency-column predictor. You will receive exactly "
        "one CoNLL-U token row whose HEAD and DEPREL columns are blank. Return "
        "exactly one tab-separated output row: ID<TAB>HEAD<TAB>DEPREL. Do not "
        "return comments, markdown, full CoNLL-U, explanations, or extra lines."
    ),
    "verbose": (
        "You are testing strict output formatting for Latin dependency parsing. "
        "Input is one syntactic token row from a CoNLL-U file. The row has 10 "
        "tab-separated columns: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, "
        "DEPS, MISC. The input HEAD and DEPREL are blank underscores. Predict "
        "only the missing dependency columns for this same token. Your entire "
        "answer must be exactly one line with exactly three tab-separated fields: "
        "ID<TAB>HEAD<TAB>DEPREL. HEAD must be an integer. DEPREL must be a UD "
        "dependency relation. No prose. No code block. No extra whitespace. No "
        "second line."
    ),
}


def conllu_comments(conllu):
    comments = {}
    for line in conllu.splitlines():
        if line.startswith("# sent_id = "):
            comments["sent_id"] = line.removeprefix("# sent_id = ").strip()
        elif line.startswith("# text = "):
            comments["text"] = line.removeprefix("# text = ").strip()
    return comments


def conllu_token_rows(conllu):
    rows = []
    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if not columns or "-" in columns[0] or "." in columns[0]:
            continue
        if len(columns) != 10:
            raise ValueError(f"Expected 10 CoNLL-U columns, got {len(columns)}: {line}")
        rows.append(columns)
    return rows


def convert_record(record, system_prompt, include_context):
    messages = record.get("messages", [])
    if len(messages) != 3:
        raise ValueError("Expected chat record with exactly three messages")

    user_conllu = messages[1]["content"].strip()
    gold_conllu = messages[2]["content"].strip()
    user_rows = conllu_token_rows(user_conllu)
    gold_rows = conllu_token_rows(gold_conllu)
    if len(user_rows) != len(gold_rows):
        raise ValueError(
            f"Input/gold token count mismatch: {len(user_rows)} input rows, {len(gold_rows)} gold rows"
        )

    comments = conllu_comments(gold_conllu)
    examples = []
    for row_index, (user_columns, gold_columns) in enumerate(zip(user_rows, gold_rows)):
        if user_columns[0] != gold_columns[0] or user_columns[1] != gold_columns[1]:
            raise ValueError(
                f"Input/gold token mismatch: {user_columns[0]} {user_columns[1]} vs "
                f"{gold_columns[0]} {gold_columns[1]}"
            )

        user_content = "\t".join(user_columns)
        if include_context:
            context = []
            if comments.get("sent_id"):
                context.append(f"# sent_id = {comments['sent_id']}")
            if comments.get("text"):
                context.append(f"# text = {comments['text']}")
            context.append(user_content)
            user_content = "\n".join(context)

        examples.append(
            {
                "sent_id": comments.get("sent_id", ""),
                "text": comments.get("text", ""),
                "row_index": row_index,
                "token_id": gold_columns[0],
                "form": gold_columns[1],
                "input_conllu": user_conllu,
                "gold_conllu": gold_conllu,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": f"{gold_columns[0]}\t{gold_columns[6]}\t{gold_columns[7]}"},
                ],
            }
        )
    return examples


def convert_file(input_path, output_path, system_prompt, include_context):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_count = 0
    row_count = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                examples = convert_record(
                    json.loads(line),
                    system_prompt=system_prompt,
                    include_context=include_context,
                )
            except Exception as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            for example in examples:
                dst.write(json.dumps(example, ensure_ascii=False) + "\n")
            sentence_count += 1
            row_count += len(examples)
    return sentence_count, row_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert full CoNLL-U chat JSONL into one-example-per-token-row ID/HEAD/DEPREL JSONL."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prompt-style", choices=sorted(PROMPTS), default="strict")
    parser.add_argument("--system-prompt")
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include # sent_id and # text above the single CoNLL-U token row.",
    )
    args = parser.parse_args()

    system_prompt = args.system_prompt if args.system_prompt is not None else PROMPTS[args.prompt_style]
    total_sentences = 0
    total_rows = 0
    for split in ["train", "valid", "test"]:
        sentence_count, row_count = convert_file(
            args.input_dir / f"{split}.jsonl",
            args.out_dir / f"{split}.jsonl",
            system_prompt=system_prompt,
            include_context=args.include_context,
        )
        total_sentences += sentence_count
        total_rows += row_count
        print(
            f"Wrote {row_count} row examples from {sentence_count} {split} sentences "
            f"to {args.out_dir / f'{split}.jsonl'}"
        )
    print(f"Total: {total_rows} row examples from {total_sentences} sentences")


if __name__ == "__main__":
    main()
