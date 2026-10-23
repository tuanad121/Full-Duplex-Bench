#!/usr/bin/env python3
"""Resumable independent Text QC for production conversations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from openai import OpenAI

from common import chat_json, choose_reviewer, read_jsonl, validate, write_jsonl
from review import SYSTEM


def review_one(client, cfg: dict, row: dict, provenance: dict) -> dict:
    errors = validate(row)
    if errors:
        return {"conversation_id": row.get("conversation_id"), "decision": "reject", "deterministic_errors": errors}
    authors = [x["requested_model"] for x in provenance["author_calls"]]
    prompt = f"""Đánh giá hội thoại sau. Nguồn tạo đã được ẩn:
{json.dumps(row, ensure_ascii=False)}
Schema output: {{"naturalness":1-5,"role_distinctiveness":1-5,
"context_consistency":1-5,"events_correct":true,"hidden_information_leak":false,
"renderable":true,"issues":["..."],"decision":"accept|reject"}}.
Chỉ accept nếu mọi điểm >=3, event đúng, không leak và renderable."""
    reviewer_models = cfg["reviewer_models"]
    # Try each reviewer model (cycling) across 3 attempts; the first model is chosen
    # by vendor-independence, subsequent ones are fallbacks on parse failure.
    first = choose_reviewer(authors, reviewer_models)
    others = [m for m in reviewer_models if m != first]
    candidates = [first, first] + others  # two attempts on preferred, then alternatives
    last = None
    for model in candidates:
        try:
            body, meta = chat_json(client, model, SYSTEM, prompt, float(cfg.get("review_temperature", 0)), 2000)
            valid = all(isinstance(body.get(k), (int, float)) and body[k] >= 3 for k in ("naturalness", "role_distinctiveness", "context_consistency"))
            valid = valid and body.get("events_correct") is True and body.get("hidden_information_leak") is False and body.get("renderable") is True
            return {"conversation_id": row["conversation_id"], "strategy": provenance["strategy"],
                    "decision": "accept" if valid else "reject", "deterministic_errors": [],
                    "llm_review": body, "reviewer": meta}
        except Exception as exc:
            last = exc
    return {"conversation_id": row["conversation_id"], "strategy": provenance["strategy"],
            "decision": "reject", "deterministic_errors": [], "review_error": f"{type(last).__name__}: {last}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--provenance", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.input)
    provenance = {x["conversation_id"]: x for x in read_jsonl(args.provenance)}
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    client = OpenAI(api_key=key, base_url=cfg["openrouter_base_url"], timeout=180, max_retries=1)
    item_dir = args.output / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    pending = [x for x in rows if not (item_dir / f"{x['conversation_id']}.json").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        fs = {pool.submit(review_one, client, cfg, row, provenance[row["conversation_id"]]): row for row in pending}
        for n, f in enumerate(concurrent.futures.as_completed(fs), 1):
            audit = f.result()
            (item_dir / f"{audit['conversation_id']}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
            print(f"[{n}/{len(pending)}] {audit['conversation_id']}: {audit['decision']}", flush=True)
    audits = [json.loads((item_dir / f"{r['conversation_id']}.json").read_text(encoding="utf-8")) for r in rows]
    by_id = {r["conversation_id"]: r for r in rows}
    accepted = [by_id[x["conversation_id"]] for x in audits if x["decision"] == "accept"]
    rejected = [by_id[x["conversation_id"]] for x in audits if x["decision"] != "accept"]
    write_jsonl(args.output / "reviews.jsonl", audits)
    write_jsonl(args.output / "accepted.jsonl", accepted)
    write_jsonl(args.output / "rejected.jsonl", rejected)
    summary = {"reviewed": len(audits), "accepted": len(accepted), "rejected": len(rejected)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

