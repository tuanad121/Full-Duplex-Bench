#!/usr/bin/env python3
"""Blind independent LLM review for matched dialogue-generation strategies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from common import chat_json, choose_reviewer, read_jsonl, validate, vendor, write_jsonl

SYSTEM = """Bạn kiểm định độc lập dữ liệu hội thoại tiếng Việt cho mô hình speech-to-speech
full-duplex. Không đoán cách dữ liệu được tạo và không sửa lời thoại. Chỉ trả JSON.

CÁC LOẠI EVENT HỢP LỆ (chỉ ba loại này):
  • backchannel — user phát tín hiệu đang nghe ngắn khi agent đang nói
  • real_interruption — user ngắt lời agent bằng yêu cầu/thông tin mới thực sự
  • ambiguous_interruption — onset giống backchannel nhưng trở thành real interruption
  Đây là ba loại duy nhất. Đừng báo lỗi khi gặp "ambiguous_interruption".

THỨ TỰ CỐ ĐỊNH trong mỗi round:
  1. user nói (trường "user")
  2. agent bắt đầu trả lời (trường "agent")
  3. USER chen vào giữa lời agent tại overlap_fraction (event.text) — LUÔN LÀ USER, không phải agent
  4. Nếu interruption/ambiguous: agent dừng rồi nói tiếp (event.agent_after, không rỗng)
     Nếu backchannel: agent KHÔNG dừng → event.agent_after PHẢI RỖNG (""). Đây là ĐÚNG.

VÍ DỤ HỢP LỆ:
  backchannel:  {"type":"backchannel","text":"Ừ.","agent_after":"","overlap_fraction":0.4}
  interruption: {"type":"real_interruption","text":"Dừng lại, đổi sang sáng mai nhé.","agent_after":"Được, sáng mai vậy.","overlap_fraction":0.5}
  ambiguous:    {"type":"ambiguous_interruption","text":"Ừm... thật ra...","agent_after":"Bạn muốn thay đổi gì?","overlap_fraction":0.35}

VÍ DỤ SAI — event.text viết như lời agent, không phải user:
  {"type":"real_interruption","text":"Tôi sẽ kiểm tra lại.","agent_after":"..."}  ← SAI

Đánh giá: lời nói tự nhiên, sự khác biệt hai vai, mạch thông tin, trợ lý không biết
trước thông tin ẩn, và tính đúng của event. Chỉ trả JSON."""


def author_models(provenance: dict, strategy: str) -> list[str]:
    if strategy == "single_author":
        return [provenance["single_author"]["requested_model"]]
    return [x["requested_model"] for x in provenance["role_play_calls"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="matched_pairs.jsonl")
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    pairs = read_jsonl(args.input)
    provenance = {x["pair_key"]: x for x in read_jsonl(args.provenance)}
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    client = OpenAI(api_key=key, base_url=cfg["openrouter_base_url"], timeout=180, max_retries=1)
    audits, accepted = [], []
    for pair in pairs:
        for strategy in ("single_author", "role_play"):
            row = pair[strategy]
            deterministic = validate(row)
            model = choose_reviewer(author_models(provenance[pair["pair_key"]], strategy), cfg["reviewer_models"])
            if deterministic:
                result = {"decision": "reject", "deterministic_errors": deterministic}
            else:
                prompt = f"""Đánh giá hội thoại (không có nhãn nguồn):
{json.dumps(row, ensure_ascii=False)}

Schema:
{{"naturalness":1-5,"role_distinctiveness":1-5,"context_consistency":1-5,
"events_correct":true,"hidden_information_leak":false,"renderable":true,
"issues":["..."],"decision":"accept|reject"}}
Chỉ accept nếu mọi điểm >=3, event đúng, không leak và renderable."""
                try:
                    last_error = None
                    for attempt in range(2):
                        try:
                            body, meta = chat_json(client, model, SYSTEM, prompt, float(cfg.get("review_temperature", 0)), 1800)
                            break
                        except (json.JSONDecodeError, ValueError) as exc:
                            last_error = exc
                    else:
                        raise last_error or ValueError("invalid reviewer JSON")
                    valid = all(isinstance(body.get(x), (int, float)) and body[x] >= 3 for x in ("naturalness", "role_distinctiveness", "context_consistency"))
                    valid = valid and body.get("events_correct") is True and body.get("hidden_information_leak") is False and body.get("renderable") is True
                    body["decision"] = "accept" if valid else "reject"
                    result = {"decision": body["decision"], "deterministic_errors": [], "llm_review": body, "reviewer": meta}
                except Exception as exc:
                    result = {"decision": "reject", "deterministic_errors": [], "review_error": f"{type(exc).__name__}: {exc}"}
            audit = {"pair_key": pair["pair_key"], "conversation_id": row["conversation_id"], "strategy": strategy, **result}
            audits.append(audit)
            if result["decision"] == "accept":
                accepted.append(row)
            print(f"{pair['pair_key']} {strategy}: {result['decision']}", flush=True)
    write_jsonl(args.output / "reviews.jsonl", audits)
    write_jsonl(args.output / "accepted.jsonl", accepted)
    summary = {}
    for strategy in ("single_author", "role_play"):
        subset = [x for x in audits if x["strategy"] == strategy]
        summary[strategy] = {"reviewed": len(subset), "accepted": sum(x["decision"] == "accept" for x in subset)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
