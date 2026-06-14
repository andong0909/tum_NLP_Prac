#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a Latin dependency parser. Given one full CoNLL-U sentence whose "
    "HEAD and DEPREL columns are blank, predict the dependency label for one "
    "specified token. Return exactly one tab-separated line: ID<TAB>HEAD<TAB>DEPREL. "
    "Do not output markdown, comments, extra tokens, or explanations."
)


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


def target_token_prompt(user_conllu, token_columns, include_sentence_text):
    token_id, form, lemma, upos, xpos, feats = token_columns[:6]
    parts = []
    if include_sentence_text:
        comments = conllu_comments(user_conllu)
        if comments.get("text"):
            parts.append(f"Sentence text: {comments['text']}")
    parts.extend(
        [
            "Input CoNLL-U with blank dependency columns:",
            user_conllu,
            "Predict dependency for this token only:",
            f"ID={token_id}\tFORM={form}\tLEMMA={lemma}\tUPOS={upos}\tXPOS={xpos}\tFEATS={feats}",
        ]
    )
    return "\n\n".join(parts)


def convert_record(record, system_prompt, include_sentence_text):
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
    for user_columns, gold_columns in zip(user_rows, gold_rows):
        if user_columns[0] != gold_columns[0]:
            raise ValueError(f"Token ID mismatch: {user_columns[0]} vs {gold_columns[0]}")
        target = f"{gold_columns[0]}\t{gold_columns[6]}\t{gold_columns[7]}"
        examples.append(
            {
                "sent_id": comments.get("sent_id", ""),
                "token_id": gold_columns[0],
                "form": gold_columns[1],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": target_token_prompt(
                            user_conllu,
                            user_columns,
                            include_sentence_text=include_sentence_text,
                        ),
                    },
                    {"role": "assistant", "content": target},
                ],
            }
        )
    return examples


def convert_file(input_path, output_path, system_prompt, include_sentence_text):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_count = 0
    token_count = 0
    with input_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line_number, line in enumerate(src, 1):
            if not line.strip():
                continue
            try:
                examples = convert_record(
                    json.loads(line),
                    system_prompt=system_prompt,
                    include_sentence_text=include_sentence_text,
                )
            except Exception as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            for example in examples:
                dst.write(json.dumps(example, ensure_ascii=False) + "\n")
            sentence_count += 1
            token_count += len(examples)
    return sentence_count, token_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert full CoNLL-U chat JSONL into token-level ID/HEAD/DEPREL chat JSONL."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument(
        "--no-sentence-text",
        action="store_true",
        help="Do not duplicate the plain sentence text before the CoNLL-U block.",
    )
    args = parser.parse_args()

    total_sentences = 0
    total_tokens = 0
    for split in ["train", "valid", "test"]:
        sentence_count, token_count = convert_file(
            args.input_dir / f"{split}.jsonl",
            args.out_dir / f"{split}.jsonl",
            args.system_prompt,
            include_sentence_text=not args.no_sentence_text,
        )
        total_sentences += sentence_count
        total_tokens += token_count
        print(
            f"Wrote {token_count} token examples from {sentence_count} {split} sentences "
            f"to {args.out_dir / f'{split}.jsonl'}"
        )
    print(f"Total: {total_tokens} token examples from {total_sentences} sentences")


if __name__ == "__main__":
    main()
