#!/usr/bin/env python3
"""Apply deterministic and independent-LLM text quality gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from schema import read_jsonl, validate_corpus, write_jsonl


SYSTEM = """Bạn là người kiểm định độc lập cho dữ liệu huấn luyện hội thoại tiếng Việt
full-duplex. Chỉ trả về JSON hợp lệ. Đánh giá nghiêm khắc nhưng không sửa nội dung.
Kiểm tra tính tự nhiên, mạch hội thoại, bảo toàn thông tin, vai trò event và hành vi
agent. Backchannel không được đòi phản hồi. Interruption phải thực sự sửa, phản đối,
đổi hoặc bổ sung yêu cầu và agent_after phải theo yêu cầu mới."""


def review(client: OpenAI, model: str, row: dict) -> dict:
    prompt = f"""Đánh giá hội thoại sau:
{json.dumps(row, ensure_ascii=False)}

Schema output:
{{"naturalness":1-5,"context_consistency":1-5,"event_roles_correct":true,
"target_behaviors_correct":true,"issues":["..."],"decision":"accept|reject"}}
Chỉ accept nếu hai điểm đều >= 3 và cả hai boolean đều true."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1200,
        response_format={"type": "json_object"},
        extra_headers={"X-Title": "Vietnamese Full-Duplex Text QC"},
    )
    result = json.loads(response.choices[0].message.content or "{}")
    required = {"naturalness", "context_consistency", "event_roles_correct", "target_behaviors_correct", "issues", "decision"}
    if set(result) != required:
        raise ValueError("reviewer returned invalid fields")
    computed_accept = (
        result["naturalness"] >= 3 and result["context_consistency"] >= 3
        and result["event_roles_correct"] is True and result["target_behaviors_correct"] is True
    )
    result["decision"] = "accept" if computed_accept else "reject"
    result["provider_model"] = response.model
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = read_jsonl(args.input)
    deterministic = validate_corpus(rows)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    client = OpenAI(api_key=key, base_url=config["openrouter_base_url"], timeout=180, max_retries=1)
    accepted, rejected, audits = [], [], []
    for index, row in enumerate(rows, 1):
        cid = row["conversation_id"]
        errors = deterministic[cid]
        if errors:
            audit = {"conversation_id": cid, "deterministic_errors": errors, "decision": "reject"}
        else:
            try:
                llm = review(client, config["reviewer_model"], row)
                audit = {"conversation_id": cid, "deterministic_errors": [], "llm_review": llm, "decision": llm["decision"]}
            except Exception as exc:
                audit = {"conversation_id": cid, "deterministic_errors": [], "review_error": f"{type(exc).__name__}: {exc}", "decision": "reject"}
        audits.append(audit)
        (accepted if audit["decision"] == "accept" else rejected).append(row)
        print(f"[{index}/{len(rows)}] {cid}: {audit['decision']}", flush=True)
    write_jsonl(args.output / "accepted.jsonl", accepted)
    write_jsonl(args.output / "rejected.jsonl", rejected)
    write_jsonl(args.output / "text_qc.jsonl", audits)
    summary = {"input": len(rows), "accepted": len(accepted), "rejected": len(rejected)}
    (args.output / "text_qc_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
