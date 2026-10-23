#!/usr/bin/env python3
"""Generate long Vietnamese full-duplex conversations through OpenRouter."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

from schema import validate_conversation, write_jsonl


TOPICS = [
    "đặt bàn và chọn món", "lên kế hoạch du lịch", "sắp lịch họp",
    "hỗ trợ kỹ thuật", "nhà thông minh", "mua sắm và đổi trả",
    "giao hàng", "dẫn đường", "việc nhà", "lên kế hoạch với bạn bè",
]
STYLES = [
    "thân mật tự nhiên", "lịch sự", "ngắn gọn", "do dự có từ đệm",
    "tự sửa lời", "khẩu ngữ miền Bắc", "khẩu ngữ miền Nam",
]
SYSTEM = """Bạn tạo dữ liệu hội thoại tiếng Việt để huấn luyện mô hình speech-to-speech
full-duplex. Chỉ trả về JSON hợp lệ, không markdown. Hội thoại phải liên tục và giữ
đúng thông tin từ các lượt trước. Event xảy ra trong lúc agent đang nói. Backchannel
chỉ là tín hiệu nghe và agent_after phải rỗng. Real/ambiguous interruption thay đổi,
phản đối hoặc bổ sung yêu cầu; agent_after phải xử lý đúng nội dung mới. Không nhồi
event vào mọi lượt. Không dùng lại ví dụ benchmark hoặc câu mẫu trong prompt."""


def make_prompt(job: dict) -> str:
    return f"""Tạo một hội thoại {job['rounds']} vòng về {job['topic']}, phong cách
{job['style']}. conversation_id phải là {job['conversation_id']}.

Chèn đúng {job['event_count']} event vào các lượt không phải lượt đầu hoặc cuối.
Phải có các loại: {', '.join(job['required_events'])}. Các event còn lại chọn hợp lý.
Mỗi câu agent ở lượt có event phải đủ dài để overlap nghe tự nhiên.

Schema chính xác:
{{"conversation_id":"...","topic":"...","style":"...","rounds":[
{{"user":"...","agent":"...","event":null}},
{{"user":"...","agent":"câu agent đầy đủ trước khi bị xử lý timeline",
"event":{{"type":"backchannel|real_interruption|ambiguous_interruption",
"text":"lời user chen vào","agent_after":"rỗng cho backchannel, câu đáp mới cho interruption",
"overlap_fraction":0.2 đến 0.8}}}}
]}}

Seed nội dung: {job['seed']}"""


def build_jobs(config: dict, start: int, count: int) -> list[dict]:
    rng = random.Random(int(config["seed"]))
    models = config["generator_models"]
    jobs = []
    for offset in range(count):
        index = start + offset
        event_count = 2 if index % 3 else 3
        jobs.append({
            "conversation_id": f"vi_fd_{index:06d}",
            "rounds": rng.randint(int(config["rounds_min"]), int(config["rounds_max"])),
            "topic": TOPICS[index % len(TOPICS)],
            "style": STYLES[(index * 3) % len(STYLES)],
            "event_count": event_count,
            "required_events": ["backchannel", "real_interruption"],
            "seed": rng.randrange(1_000_000_000),
            "model": models[offset % len(models)],
        })
    return jobs


def generate_one(client: OpenAI, job: dict, retries: int) -> dict:
    last_error = "unknown"
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=job["model"],
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": make_prompt(job)}],
                temperature=0.85,
                max_tokens=5000,
                response_format={"type": "json_object"},
                extra_headers={"X-Title": "Vietnamese Full-Duplex Speech Data"},
            )
            row = json.loads(response.choices[0].message.content or "{}")
            errors = validate_conversation(row)
            actual_events = [turn["event"]["type"] for turn in row.get("rounds", []) if isinstance(turn, dict) and isinstance(turn.get("event"), dict)]
            if len(actual_events) != job["event_count"]:
                errors.append("event_count_mismatch")
            for required in job["required_events"]:
                if required not in actual_events:
                    errors.append(f"missing_{required}")
            if not errors:
                return {"ok": True, "conversation": row, "provenance": {
                    "requested_model": job["model"], "provider_model": response.model,
                    "job": job, "usage": response.usage.model_dump() if response.usage else None,
                }}
            last_error = ",".join(errors)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2 ** attempt, 8))
    return {"ok": False, "job": job, "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--count", type=int)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    count = args.count or int(config["conversation_count"])
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    client = OpenAI(api_key=key, base_url=config["openrouter_base_url"], timeout=180, max_retries=1)
    jobs = build_jobs(config, args.start, count)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(generate_one, client, job, args.retries) for job in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{index}/{len(jobs)}] {'accepted' if result['ok'] else 'rejected'}", flush=True)
    accepted = [result for result in results if result["ok"]]
    write_jsonl(args.output / "generated.jsonl", (result["conversation"] for result in accepted))
    write_jsonl(args.output / "generation_provenance.jsonl", (result["provenance"] for result in accepted))
    write_jsonl(args.output / "generation_rejected.jsonl", (result for result in results if not result["ok"]))
    summary = {"requested": len(jobs), "accepted": len(accepted), "rejected": len(jobs) - len(accepted)}
    (args.output / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
