#!/usr/bin/env python3
"""Export constrained four-label predictions and an HTML failure review."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LABELS = ["<|C-L|>", "<|S-S|>", "<|S-L|>", "<|C-S|>"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    label_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(label) for label in LABELS],
        device="cuda",
    )
    rows = []
    confusion: dict[str, Counter] = defaultdict(Counter)
    with torch.inference_mode():
        for line in args.eval_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source = json.loads(line)
            messages = source["messages"]
            truth = messages[-1]["content"]
            prompt = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to("cuda")
            logits = model(**encoded).logits[0, -1].index_select(0, label_ids)
            probabilities = logits.float().softmax(-1).cpu().tolist()
            prediction = LABELS[max(range(len(LABELS)), key=probabilities.__getitem__)]
            confusion[truth][prediction] += 1
            rows.append(
                {
                    "example_id": source["example_id"],
                    "conversation_id": source["conversation_id"],
                    "truth": truth,
                    "prediction": prediction,
                    "correct": truth == prediction,
                    "confidence": max(probabilities),
                    "probabilities": dict(zip(LABELS, probabilities)),
                    "model_input": messages[-2]["content"],
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "total": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "confusion": {
            truth: {prediction: confusion[truth][prediction] for prediction in LABELS}
            for truth in LABELS
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    focus = [
        row
        for row in rows
        if row["truth"] in {"<|C-L|>", "<|S-L|>"}
        or row["prediction"] in {"<|C-L|>", "<|S-L|>"}
    ]
    cards = []
    for row in sorted(
        focus,
        key=lambda item: (
            item["correct"],
            item["truth"],
            -item["confidence"],
        ),
    ):
        status = "PASS" if row["correct"] else "FAIL"
        probs = " · ".join(
            f"{label} {probability:.1%}"
            for label, probability in row["probabilities"].items()
        )
        cards.append(
            f"""
<article class="card {status.lower()}">
  <header><b>{html.escape(row["example_id"])}</b>
    <span>{status} · truth {html.escape(row["truth"])}
    → predicted {html.escape(row["prediction"])}</span></header>
  <pre>{html.escape(row["model_input"])}</pre>
  <footer>{html.escape(probs)}</footer>
</article>"""
        )
    matrix_rows = "".join(
        "<tr><th>{}</th>{}</tr>".format(
            html.escape(truth),
            "".join(f"<td>{confusion[truth][prediction]}</td>" for prediction in LABELS),
        )
        for truth in LABELS
    )
    page = f"""<!doctype html>
<html lang="vi"><meta charset="utf-8">
<title>Vi-DM failure review</title>
<style>
body{{font:16px system-ui;max-width:1200px;margin:32px auto;color:#172033}}
table{{border-collapse:collapse}}th,td{{padding:9px 14px;border:1px solid #ccd5e3}}
.card{{border:1px solid #cad5e5;border-radius:12px;margin:16px 0;overflow:hidden}}
.card header,.card footer{{padding:12px 16px;background:#eef4fc;display:flex;
justify-content:space-between;gap:20px}}.card.fail{{border-color:#d66}}
.card.fail header{{background:#ffe8e8}}pre{{white-space:pre-wrap;padding:16px;margin:0}}
footer{{font-size:14px}}code{{background:#eef2f7;padding:2px 5px}}
</style>
<h1>Vi-DM: <code>&lt;|C-L|&gt;</code> and <code>&lt;|S-L|&gt;</code></h1>
<p>Best checkpoint, constrained four-label decoding. Failure cards appear first.</p>
<h2>Confusion matrix</h2>
<table><tr><th>Truth ↓ / prediction →</th>
{''.join(f'<th>{html.escape(label)}</th>' for label in LABELS)}</tr>
{matrix_rows}</table>
<h2>Focused examples</h2>
{''.join(cards)}
</html>"""
    (args.output_dir / "failure_review.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
