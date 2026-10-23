#!/usr/bin/env python3
"""Fine-tune a lightweight four-action Vietnamese dialogue manager."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


CONTROL_TOKENS = ["<|C-L|>", "<|S-S|>", "<|S-L|>", "<|C-S|>"]


class DecisionDataset(Dataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        max_length: int,
        limit: int | None = None,
        per_label_limit: int | None = None,
        balance: bool = False,
    ) -> None:
        self.rows = []
        rows_by_label = {token: [] for token in CONTROL_TOKENS}
        label_counts = {token: 0 for token in CONTROL_TOKENS}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"]
            target = messages[-1]["content"]
            if target not in CONTROL_TOKENS:
                raise ValueError(f"Invalid target in {row.get('example_id')}: {target}")
            if (
                per_label_limit is not None
                and label_counts[target] >= per_label_limit
            ):
                continue
            prefix = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            target_id = tokenizer.convert_tokens_to_ids(target)
            input_ids = (prefix_ids + [target_id])[-max_length:]
            labels = [-100] * (len(input_ids) - 1) + [target_id]
            encoded = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }
            self.rows.append(encoded)
            rows_by_label[target].append(encoded)
            label_counts[target] += 1
            if limit is not None and len(self.rows) >= limit:
                break
            if per_label_limit is not None and all(
                count >= per_label_limit for count in label_counts.values()
            ):
                break
        if balance:
            max_count = max(len(rows) for rows in rows_by_label.values())
            if any(not rows for rows in rows_by_label.values()):
                raise ValueError(f"Cannot balance missing labels: {label_counts}")
            self.rows = [
                row
                for token in CONTROL_TOKENS
                for row in (
                    rows_by_label[token]
                    * math.ceil(max_count / len(rows_by_label[token]))
                )[:max_count]
            ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class DecisionCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in features)
        inputs, masks, labels = [], [], []
        for row in features:
            pad = max_len - len(row["input_ids"])
            inputs.append(row["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            masks.append(row["attention_mask"] + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def class_metrics(predictions: np.ndarray, labels: np.ndarray, token_ids: dict) -> dict:
    # AutoModelForCausalLM computes next-token loss by shifting internally.
    # Mirror that shift here so metrics score the logit immediately before the
    # control token, rather than the logit after it.
    predictions = predictions[..., :-1]
    labels = labels[..., 1:]
    mask = labels != -100
    truth = labels[mask].reshape(-1)
    pred = predictions[mask].reshape(-1)
    result = {"accuracy": float((truth == pred).mean())}
    f1_values = []
    for token, token_id in token_ids.items():
        tp = int(((truth == token_id) & (pred == token_id)).sum())
        fp = int(((truth != token_id) & (pred == token_id)).sum())
        fn = int(((truth == token_id) & (pred != token_id)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        name = token[2:-2].lower().replace("-", "_")
        result[f"{name}_precision"] = precision
        result[f"{name}_recall"] = recall
        result[f"{name}_f1"] = f1
        f1_values.append(f1)
    result["macro_f1"] = float(sum(f1_values) / len(f1_values))
    return result


class FloorLinearTrainer(Trainer):
    """Paper-like linear decay from LR to 10% of LR, rather than zero."""

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            optimizer = optimizer or self.optimizer

            def factor(step: int) -> float:
                return 1.0 - 0.9 * min(step / max(num_training_steps, 1), 1.0)

            self.lr_scheduler = LambdaLR(optimizer, factor)
        return self.lr_scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--per-label-limit", type=int)
    parser.add_argument("--balance-train", action="store_true")
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": CONTROL_TOKENS})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_ids = {
        token: tokenizer.convert_tokens_to_ids(token) for token in CONTROL_TOKENS
    }
    if len(set(token_ids.values())) != len(CONTROL_TOKENS):
        raise RuntimeError(f"Control tokens are not unique: {token_ids}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    train_dataset = DecisionDataset(
        args.train_file,
        tokenizer,
        args.max_length,
        args.train_limit,
        args.per_label_limit,
        args.balance_train,
    )
    eval_dataset = DecisionDataset(
        args.eval_file,
        tokenizer,
        args.max_length,
        args.eval_limit,
        args.per_label_limit,
        False,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        logging_dir=str(args.output_dir / "tensorboard"),
        report_to=["tensorboard"],
        run_name="vi-dm-qwen3-0.6b-paper-baseline",
        bf16=True,
        tf32=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=4,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    def preprocess_logits(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        # The dialogue manager has exactly four legal outputs. Score with the
        # same constrained decoding used at inference instead of allowing an
        # unrelated vocabulary token to win the argmax.
        allowed_ids = torch.tensor(
            list(token_ids.values()), device=logits.device, dtype=torch.long
        )
        class_index = logits.index_select(-1, allowed_ids).argmax(dim=-1)
        return allowed_ids[class_index]

    trainer = FloorLinearTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DecisionCollator(tokenizer),
        compute_metrics=lambda value: class_metrics(
            value.predictions, value.label_ids, token_ids
        ),
        preprocess_logits_for_metrics=preprocess_logits,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "train_examples": len(train_dataset),
                "eval_examples": len(eval_dataset),
                "control_token_ids": token_ids,
                "global_batch_size": (
                    args.batch_size
                    * args.gradient_accumulation
                    * max(torch.cuda.device_count(), 1)
                ),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    (args.output_dir / "final_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
