#!/usr/bin/env python3
"""Evaluate a native-sequence DM checkpoint on label-only state snapshots."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MAP = {
    "<|C-L|>": "<|continue-listen|>", "<|S-S|>": "<|switch-to-speak|>",
    "<|S-L|>": "<|switch-to-listen|>", "<|C-S|>": "<|continue_speak|>",
}


def prompt_messages(model_input):
    messages = [dict(message) for message in model_input["dialogue_history"]]
    ongoing = model_input.get("assistant_ongoing", "").strip()
    if model_input["assistant_state"] == "speaking" and ongoing:
        if not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content", "").strip() != ongoing:
            messages.append({"role": "assistant", "content": ongoing})
    messages.append({"role": "user", "content": model_input["streaming_asr"]})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    native_ids = {label: tokenizer.convert_tokens_to_ids(native) for label, native in MAP.items()}
    confusion = defaultdict(Counter)
    predictions = []
    with torch.inference_mode():
        for line in args.eval_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            text = tokenizer.apply_chat_template(prompt_messages(row.get("input") or {}), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            batch = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            logits = model(**batch).logits[0, -1]
            pred = max(native_ids, key=lambda label: float(logits[native_ids[label]]))
            truth = row.get("target") or row["messages"][-1]["content"]
            confusion[truth][pred] += 1
            predictions.append({"example_id": row.get("example_id"), "truth": truth, "prediction": pred})
    labels = list(MAP)
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in labels)
    metrics = {"total": total, "accuracy": correct / total, "confusion": {truth: dict(confusion[truth]) for truth in labels}}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[truth][label] for truth in labels if truth != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    metrics["macro_f1"] = sum(f1s) / len(f1s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": metrics, "predictions": predictions}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
