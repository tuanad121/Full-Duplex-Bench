#!/usr/bin/env python3
"""Create leakage-free train/eval splits for paper-format DM examples."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_model_jsonl(path: Path, rows: list[dict]) -> None:
    """Write only fields consumed by an ordinary chat-SFT trainer."""
    path.write_text(
        "".join(
            json.dumps(
                {
                    "messages": row["messages"],
                    "example_id": row["example_id"],
                    "conversation_id": row["conversation_id"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def stats(rows: list[dict]) -> dict:
    conversation_ids = {row["conversation_id"] for row in rows}
    first_by_conversation = {
        cid: next(row for row in rows if row["conversation_id"] == cid)
        for cid in conversation_ids
    }
    return {
        "conversations": len(conversation_ids),
        "decisions": len(rows),
        "action_counts": dict(sorted(Counter(row["target"] for row in rows).items())),
        "pattern_counts": dict(
            sorted(
                Counter(
                    first_by_conversation[cid]["audit"]["pattern"]
                    for cid in conversation_ids
                ).items()
            )
        ),
        "provider_model_counts": dict(
            sorted(
                Counter(
                    first_by_conversation[cid]["audit"].get(
                        "provider_model", "unknown"
                    )
                    for cid in conversation_ids
                ).items()
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args()
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval-ratio must be between 0 and 1")

    rows = load_jsonl(args.input)
    by_conversation: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_conversation[row["conversation_id"]].append(row)

    by_pattern: dict[str, list[str]] = defaultdict(list)
    for cid, conversation_rows in by_conversation.items():
        patterns = {row["audit"]["pattern"] for row in conversation_rows}
        if len(patterns) != 1:
            raise ValueError(f"{cid} has inconsistent patterns: {patterns}")
        by_pattern[patterns.pop()].append(cid)

    rng = random.Random(args.seed)
    eval_ids: set[str] = set()
    for pattern in sorted(by_pattern):
        ids = sorted(by_pattern[pattern])
        rng.shuffle(ids)
        eval_count = max(1, round(len(ids) * args.eval_ratio))
        eval_ids.update(ids[:eval_count])
    train_ids = set(by_conversation) - eval_ids
    if train_ids & eval_ids:
        raise AssertionError("conversation leakage between train and eval")

    train_rows = [row for row in rows if row["conversation_id"] in train_ids]
    eval_rows = [row for row in rows if row["conversation_id"] in eval_ids]
    args.output.mkdir(parents=True, exist_ok=True)
    write_model_jsonl(args.output / "train.jsonl", train_rows)
    write_model_jsonl(args.output / "eval.jsonl", eval_rows)

    manifest = {
        "schema_version": "vi_dm_train_eval_split_v1",
        "source": str(args.input),
        "split_unit": "conversation",
        "stratified_by": "generation_pattern",
        "seed": args.seed,
        "requested_eval_ratio": args.eval_ratio,
        "usage": {
            "train": "parameter optimization",
            "eval": "internal checkpoint selection only",
            "external_qc": "must be a separate unseen dataset",
        },
        "train": stats(train_rows),
        "eval": stats(eval_rows),
        "conversation_overlap": len(train_ids & eval_ids),
        "train_conversation_ids": sorted(train_ids),
        "eval_conversation_ids": sorted(eval_ids),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in manifest.items() if not k.endswith("_ids")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
