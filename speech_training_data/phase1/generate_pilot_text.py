#!/usr/bin/env python3
"""Generate four Vietnamese Phase-1 full-duplex pilot conversations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


SYSTEM = """Bạn thiết kế hội thoại tiếng Việt tự nhiên để tạo dữ liệu âm thanh
full-duplex. Chỉ trả về một JSON object hợp lệ, không markdown. Không dùng dấu ba
chấm để mô phỏng cắt lời. Nội dung phải giống hội thoại đời thường, không phải lời
giải thích về nhãn. Lời trợ lý trong trường hợp bị cắt lời phải đủ dài để người dùng
có thể chen vào tự nhiên. Backchannel phải ngắn và không đòi hỏi câu trả lời."""

PROMPT = """Tạo đúng bốn mẫu, mỗi loại một mẫu:

1. normal_turn: người dùng hỏi xong; trợ lý bắt đầu trả lời sau khi người dùng kết thúc.
2. real_interruption: người dùng hỏi; trợ lý đang trả lời dài; người dùng chen vào bằng
   một yêu cầu sửa đổi thực sự; trợ lý phải dừng và sau đó trả lời yêu cầu mới.
3. backchannel: người dùng hỏi; trong lúc trợ lý trả lời dài, người dùng chỉ nói một
   backchannel ngắn như “ừ”, “vâng”, “dạ đúng”; trợ lý phải nói tiếp, không phản hồi cue.
4. ambiguous_interruption: câu chen bắt đầu giống backchannel nhưng tiếp tục thành sửa
   đổi thực sự, ví dụ “ừ, nhưng mà...”; trợ lý phải dừng và xử lý yêu cầu mới.

Chủ đề của bốn mẫu phải khác nhau. Văn phong khẩu ngữ tự nhiên. Tránh thông tin y tế,
pháp lý hoặc tài chính rủi ro cao.

Schema bắt buộc:
{"schema_version":1,"language":"vi","samples":[
 {"id":"normal_turn_01","case":"normal_turn","topic":"...",
  "user_initial":"...","agent_ongoing":"...","user_event":"",
  "agent_after_event":""},
 {"id":"real_interruption_01","case":"real_interruption","topic":"...",
  "user_initial":"...","agent_ongoing":"...","user_event":"...",
  "agent_after_event":"..."},
 {"id":"backchannel_01","case":"backchannel","topic":"...",
  "user_initial":"...","agent_ongoing":"...","user_event":"...",
  "agent_after_event":""},
 {"id":"ambiguous_interruption_01","case":"ambiguous_interruption","topic":"...",
  "user_initial":"...","agent_ongoing":"...","user_event":"...",
  "agent_after_event":"..."}
]}"""

CASES = ["normal_turn", "real_interruption", "backchannel", "ambiguous_interruption"]


def validate(payload: dict) -> None:
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 4:
        raise ValueError("expected exactly four samples")
    if [sample.get("case") for sample in samples] != CASES:
        raise ValueError("cases are missing or out of order")
    required = {"id", "case", "topic", "user_initial", "agent_ongoing", "user_event", "agent_after_event"}
    for sample in samples:
        if set(sample) != required:
            raise ValueError(f"invalid fields for {sample.get('id')}")
        for key in required:
            if not isinstance(sample[key], str):
                raise ValueError(f"{sample['id']}.{key} must be a string")
        if not sample["user_initial"] or not sample["agent_ongoing"]:
            raise ValueError(f"{sample['id']} lacks the initial exchange")
    if samples[0]["user_event"] or samples[0]["agent_after_event"]:
        raise ValueError("normal turn must not have an overlap event")
    if not samples[1]["user_event"] or not samples[1]["agent_after_event"]:
        raise ValueError("real interruption must contain event and recovery reply")
    if not samples[2]["user_event"] or samples[2]["agent_after_event"]:
        raise ValueError("backchannel must have a cue and no reply to the cue")
    if not samples[3]["user_event"] or not samples[3]["agent_after_event"]:
        raise ValueError("ambiguous interruption must contain event and recovery reply")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=180)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}],
        temperature=0.8,
        max_tokens=3000,
        response_format={"type": "json_object"},
        extra_headers={"X-Title": "Vietnamese Full-Duplex Phase 1 Pilot"},
    )
    payload = json.loads(response.choices[0].message.content or "{}")
    validate(payload)
    payload["generator"] = {"requested_model": args.model, "provider_model": response.model}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
