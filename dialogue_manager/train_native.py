#!/usr/bin/env python3
"""Train on the repository's native complete, interleaved dialogue sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments


TOKENS = [
    "<|continue-listen|>", "<|switch-to-speak|>", "<|switch-to-listen|>",
    "<|continue_speak|>", "<|interrupted|>",
]


class NativeDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_assistant_tokens_mask=True, add_generation_prompt=False,
                enable_thinking=False,
            )
            full_ids = encoded["input_ids"]
            mask = encoded.get("assistant_masks") or encoded.get("assistant_tokens_mask")
            if mask is None or not any(mask):
                mask = self._assistant_mask(messages, tokenizer, full_ids)
            ids = full_ids[-max_length:]
            mask = mask[-max_length:]
            labels = [token if keep else -100 for token, keep in zip(ids, mask)]
            if all(label == -100 for label in labels):
                continue
            self.rows.append({"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels})

    @staticmethod
    def _assistant_mask(messages, tokenizer, full_ids):
        """Locate assistant spans when the chat template lacks generation tags."""
        mask = [0] * len(full_ids)
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            empty = [dict(item) for item in messages[:index]] + [{"role": "assistant", "content": ""}]
            actual = messages[: index + 1]
            empty_ids = tokenizer.apply_chat_template(
                empty, tokenize=True, return_dict=True,
                add_generation_prompt=False, enable_thinking=False,
            )["input_ids"]
            actual_ids = tokenizer.apply_chat_template(
                actual, tokenize=True, return_dict=True,
                add_generation_prompt=False, enable_thinking=False,
            )["input_ids"]
            start = 0
            for left, right in zip(empty_ids, actual_ids):
                if left != right:
                    break
                start += 1
            end = min(len(actual_ids), len(full_ids))
            for position in range(start, end):
                mask[position] = 1
        return mask

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        length = max(len(row["input_ids"]) for row in rows)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            pad = length - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            result["attention_mask"].append(row["attention_mask"] + [0] * pad)
            result["labels"].append(row["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": TOKENS})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train = NativeDataset(args.train_file, tokenizer, args.max_length)
    evaluation = NativeDataset(args.eval_file, tokenizer, args.max_length)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), bf16=True, tf32=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True, learning_rate=args.learning_rate,
        max_steps=args.max_steps, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=4,
        logging_steps=5, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, seed=args.seed, data_seed=args.seed,
        remove_unused_columns=False, report_to=["tensorboard"],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(json.dumps({**vars(args), "train_dialogues": len(train), "eval_dialogues": len(evaluation)}, default=str, indent=2) + "\n")
    trainer = Trainer(model=model, args=training_args, train_dataset=train, eval_dataset=evaluation, data_collator=Collator(tokenizer), callbacks=[EarlyStoppingCallback(early_stopping_patience=6)])
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    (args.output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
