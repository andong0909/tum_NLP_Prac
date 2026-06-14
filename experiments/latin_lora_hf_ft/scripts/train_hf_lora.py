#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


IGNORE_INDEX = -100


class ChatJsonlDataset(Dataset):
    def __init__(self, path, tokenizer, max_seq_length):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages")
            if not messages or len(messages) != 3:
                raise ValueError(f"{path}:{line_number}: expected exactly 3 chat messages")
            self.examples.append(messages)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        messages = self.examples[index]
        prompt_messages = messages[:2]
        full_messages = messages

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_length,
        )["input_ids"]

        labels = full_ids.copy()
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [IGNORE_INDEX] * prompt_length

        if all(label == IGNORE_INDEX for label in labels):
            # If the assistant part was truncated away, leave the final token as
            # a target so the trainer does not see an all-masked example.
            labels[-1] = full_ids[-1]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }


class DataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of:
            remainder = max_length % self.pad_to_multiple_of
            if remainder:
                max_length += self.pad_to_multiple_of - remainder

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        pad_id = self.tokenizer.pad_token_id
        for item in features:
            length = len(item["input_ids"])
            pad_length = max_length - length
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad_length)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_length)
            batch["labels"].append(item["labels"] + [IGNORE_INDEX] * pad_length)

        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Hugging Face PEFT/LoRA training for Latin HEAD/DEPREL prediction.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-dir", type=Path, default=Path("latin_head_deprel_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("hf_outputs/qwen-head-deprel-lora"))
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--qlora", action="store_true", help="Load base model in 4-bit with bitsandbytes.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training when supported.")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 training.")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--report-to", default="none")
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    torch_dtype = torch.float32
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    if args.qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype if torch_dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype if not args.qlora else None,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.use_cache = False

    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[item.strip() for item in args.target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = ChatJsonlDataset(args.data_dir / "train.jsonl", tokenizer, args.max_seq_length)
    eval_dataset = ChatJsonlDataset(args.data_dir / "valid.jsonl", tokenizer, args.max_seq_length)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollator(tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
