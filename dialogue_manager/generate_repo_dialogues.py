#!/usr/bin/env python3
"""Generate complete Vietnamese dialogues for the authors' deterministic pipeline."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import time
from pathlib import Path

from openai import OpenAI

MODELS = [
    "google/gemini-2.5-flash",
    "qwen/qwen3-235b-a22b-2507",
    "openai/gpt-4.1-mini",
    "deepseek/deepseek-v4-flash",
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-4-maverick",
    "anthropic/claude-sonnet-4.6",
]
TOPICS = [
    "lễ tân khách sạn", "đặt lịch khám", "ngân hàng", "đổi trả hàng",
    "giao hàng", "nhà hàng", "du lịch", "chuyến bay và tàu",
    "trợ lý trong ô tô", "nhà thông minh", "lịch họp", "học tập",
    "hỗ trợ kỹ thuật", "gia đình và việc nhà", "dịch vụ công",
]
STYLES = [
    "tự nhiên trung tính", "thân mật khẩu ngữ", "lịch sự trang trọng",
    "do dự có từ đệm", "tự sửa lời", "ngắn gọn trực tiếp",
    "người lớn tuổi nói đơn giản", "chuyển mã Việt-Anh nhẹ",
]

SYSTEM = """Bạn tạo hội thoại tiếng Việt tự nhiên để huấn luyện quản lý lượt lời.
Chỉ xuất đúng một JSON object, không markdown. Tạo hội thoại HOÀN CHỈNH trước;
không cắt câu, không chèn nhãn điều khiển, không viết transcript ASR, không bỏ dấu
tiếng Việt. Mỗi lượt gồm câu người dùng hoàn chỉnh và câu trợ lý hoàn chỉnh.
`type=interruption` chỉ có nghĩa câu người dùng là lời sửa, phản đối, câu hỏi tiếp
theo hoặc chuyển yêu cầu so với câu trả lời trước; vẫn viết toàn bộ câu hoàn chỉnh.
Không dùng backchannel, lời nền hoặc lời nói với người khác ở bước này."""


def prompt(index: int, seed: int, rounds: int, topic: str, style: str) -> str:
    return f"""Tạo {rounds} vòng hội thoại liên tục về "{topic}", phong cách "{style}".
Ít nhất một vòng sau vòng đầu phải là `interruption`, còn lại là `normal`.
Schema bắt buộc:
{{"conversation_id":"repo_vi_{index:06d}","rounds":[
{{"type":"normal|interruption","user":"...","assistant":"..."}}
]}}
Mỗi user/assistant phải là câu hoàn chỉnh, có dấu tiếng Việt, hợp ngữ cảnh.
Seed nội dung: {seed}."""


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def validate(obj: dict, index: int, rounds: int) -> list[str]:
    errors = []
    if obj.get("conversation_id") != f"repo_vi_{index:06d}":
        errors.append("conversation_id")
    items = obj.get("rounds")
    if not isinstance(items, list) or len(items) != rounds:
        return errors + ["round_count"]
    if not any(item.get("type") == "interruption" for item in items[1:]):
        errors.append("missing_interruption_round")
    for pos, item in enumerate(items):
        if item.get("type") not in {"normal", "interruption"}:
            errors.append(f"{pos}:type")
        for field in ("user", "assistant"):
            value = item.get(field)
            if not isinstance(value, str) or len(value.strip()) < 8:
                errors.append(f"{pos}:{field}")
            if isinstance(value, str) and "<|" in value:
                errors.append(f"{pos}:{field}_token_leak")
    return errors


def to_repo_schema(obj: dict) -> dict:
    return {
        f"Round {index}": {
            "type": item["type"],
            "user": item["user"].strip(),
            "assistant": item["assistant"].strip(),
        }
        for index, item in enumerate(obj["rounds"], 1)
    }


def generate(client: OpenAI, job: dict, retries: int) -> dict:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=job["model"],
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt(**job["prompt"])},
                ],
                temperature=0.85,
                max_tokens=3500,
                response_format={"type": "json_object"},
                extra_headers={"X-Title": "Vietnamese Full Duplex Repo Pipeline"},
            )
            obj = parse_json(response.choices[0].message.content or "")
            errors = validate(obj, job["prompt"]["index"], job["prompt"]["rounds"])
            if not errors:
                return {
                    "ok": True,
                    "dialogue": to_repo_schema(obj),
                    "provenance": {
                        "requested_model": job["model"],
                        "provider_model": response.model,
                        "topic": job["prompt"]["topic"],
                        "style": job["prompt"]["style"],
                        "usage": response.usage.model_dump() if response.usage else None,
                    },
                }
            last_error = ",".join(errors)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2**attempt, 8))
    return {"ok": False, "error": last_error, "job": job}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=28)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    if any(model.endswith(":free") or model == "openrouter/free" for model in args.models):
        raise SystemExit("Free OpenRouter models are prohibited")

    rng = random.Random(args.seed)
    jobs = []
    for offset in range(args.count):
        index = args.start + offset
        jobs.append(
            {
                "model": args.models[offset % len(args.models)],
                "prompt": {
                    "index": index,
                    "seed": rng.randrange(1_000_000_000),
                    "rounds": rng.randint(4, 7),
                    "topic": TOPICS[offset % len(TOPICS)],
                    "style": STYLES[(offset * 3) % len(STYLES)],
                },
            }
        )

    client = OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        timeout=180,
        max_retries=1,
    )
    results = [None] * len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_map = {
            pool.submit(generate, client, job, args.retries): pos
            for pos, job in enumerate(jobs)
        }
        for done, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            pos = future_map[future]
            results[pos] = future.result()
            print(f"[{done}/{len(jobs)}] {'accepted' if results[pos]['ok'] else 'rejected'}")

    args.output.mkdir(parents=True, exist_ok=True)
    accepted = [result for result in results if result["ok"]]
    rejected = [result for result in results if not result["ok"]]
    (args.output / "answers_clean.jsonl").write_text(
        "".join(
            json.dumps(result["dialogue"], ensure_ascii=False) + "\n"
            for result in accepted
        ),
        encoding="utf-8",
    )
    (args.output / "provenance.jsonl").write_text(
        "".join(
            json.dumps(result["provenance"], ensure_ascii=False) + "\n"
            for result in accepted
        ),
        encoding="utf-8",
    )
    (args.output / "rejected.jsonl").write_text(
        "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in rejected),
        encoding="utf-8",
    )
    summary = {"requested": len(jobs), "accepted": len(accepted), "rejected": len(rejected)}
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
