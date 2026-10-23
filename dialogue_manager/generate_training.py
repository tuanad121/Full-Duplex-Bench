#!/usr/bin/env python3
"""Generate balanced Vietnamese semantic-DM conversations via OpenRouter."""

from __future__ import annotations

import argparse
import concurrent.futures
import difflib
import json
import os
import random
import re
import threading
import time
import unicodedata
from collections import Counter
from pathlib import Path

from openai import OpenAI

TOKENS = {"<|C-L|>", "<|S-S|>", "<|S-L|>", "<|C-S|>"}
# Explicit paid endpoints from different model families. Do not use a router
# alias here: exact model IDs make diversity quotas and provenance reproducible.
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
    "lễ tân và khách sạn", "đặt lịch khám", "nhà thuốc", "ngân hàng",
    "mua sắm và đổi trả", "giao hàng", "nhà hàng", "du lịch",
    "chuyến bay và tàu", "dẫn đường trong ô tô", "nhà thông minh",
    "lịch họp công việc", "học tập", "hỗ trợ kỹ thuật", "giải trí",
    "gia đình và việc nhà", "lên kế hoạch với bạn bè", "dịch vụ công",
    "an toàn thường ngày không chẩn đoán y khoa",
]
STYLES = [
    "trung tính tự nhiên", "ngắn gọn trực tiếp", "lịch sự trang trọng",
    "thân mật khẩu ngữ", "do dự có từ đệm", "tự sửa lời",
    "dài dòng nhưng tự nhiên", "sốt ruột", "không chắc và hay xác nhận",
    "cách nói đơn giản của người lớn tuổi", "nhanh và đứt đoạn",
    "chuyển mã Việt-Anh nhẹ", "từ vựng vùng miền dễ hiểu",
    "giống ASR không dấu câu", "giống ASR có một lỗi nhận dạng hợp lý",
]
CONTEXTS = [
    "receptionist_service", "smart_home", "in_car", "human_to_human",
]
PATTERNS = [
    "normal", "incomplete", "real_interruption", "fake_backchannel",
    "fake_talking_to_other", "fake_background_speech",
]

SYSTEM = """Bạn tạo dữ liệu huấn luyện tiếng Việt cho bộ quản lý lượt lời full-duplex.
Chỉ xuất một JSON object hợp lệ, không markdown. Viết hội thoại tự nhiên, nhất quán.
Nhãn đã được giao sẵn; tuyệt đối không đổi nhãn. Không sao chép ví dụ benchmark.
Mỗi decision là ảnh chụp trạng thái mà mô hình quản lý hội thoại sẽ nhìn thấy.
speaker_state phải là listening hoặc speaking. Nếu speaking, assistant_history phải
có nội dung đang nói. Với C-L/S-S hệ thống đang listening; với S-L/C-S hệ thống đang
speaking. C-S chỉ dành cho backchannel/lời nói với người khác/nền không cần phản hồi;
S-L dành cho sửa đổi, phản đối, câu hỏi mới hoặc đổi yêu cầu thực sự."""


def assigned_tokens(pattern: str, rounds: int) -> list[str]:
    base = ["<|S-S|>"] * rounds
    if pattern == "incomplete":
        base[0] = "<|C-L|>"
        if rounds > 1:
            base[1] = "<|S-S|>"
    elif pattern == "real_interruption":
        base[min(1, rounds - 1)] = "<|S-L|>"
    elif pattern.startswith("fake_"):
        base[min(1, rounds - 1)] = "<|C-S|>"
    return base


def assigned_roles(pattern: str, rounds: int) -> list[str]:
    roles = ["normal_query"] * rounds
    event_index = min(1, rounds - 1)
    if pattern == "incomplete":
        roles[0] = "incomplete_query"
    elif pattern == "real_interruption":
        roles[event_index] = "real_interruption"
    elif pattern == "fake_backchannel":
        roles[event_index] = "backchannel"
    elif pattern == "fake_talking_to_other":
        roles[event_index] = "talking_to_other"
    elif pattern == "fake_background_speech":
        roles[event_index] = "background_speech"
    return roles


def make_job(index: int, seed: int, models: list[str] = MODELS) -> dict:
    rng = random.Random(seed + index * 7919)
    rounds = rng.randint(2, 12)
    pattern = PATTERNS[index % len(PATTERNS)]
    return {
        "conversation_id": f"vi_dm_{index + 1:06d}",
        "topic": TOPICS[index % len(TOPICS)],
        "style": STYLES[(index * 7) % len(STYLES)],
        "context_family": CONTEXTS[(index * 5) % len(CONTEXTS)],
        "pattern": pattern,
        "round_count": rounds,
        "required_tokens": assigned_tokens(pattern, rounds),
        "required_roles": assigned_roles(pattern, rounds),
        "model": models[index % len(models)],
        "seed": rng.randrange(1_000_000_000),
    }


def prompt(job: dict) -> str:
    return f"""Tạo đúng một hội thoại theo đặc tả sau:
{json.dumps(job, ensure_ascii=False)}

JSON schema:
{{"conversation_id":"...","topic":"...","style":"...","context_family":"...",
"pattern":"...","turns":[{{"round":1,"user_text":"bản sạch tiếng Việt",
"assistant_history":"nội dung trợ lý đã/đang nói, có thể rỗng khi listening",
"speaker_state":"listening|speaking","streaming_asr":"văn bản ASR hiện có",
"event_role":"normal_query|incomplete_query|real_interruption|backchannel|talking_to_other|background_speech",
"action":"một token được giao","assistant_reply":"câu trả lời tự nhiên hoặc rỗng nếu chưa cần CDE"}}]}}

Yêu cầu: đúng {job['round_count']} turns; action theo đúng thứ tự required_tokens và
event_role theo đúng thứ tự required_roles.
Các lượt phải tạo thành một hội thoại liên tục. Với incomplete, phần ASR đầu phải dang dở
nhưng là chuỗi từ thực tế, không dùng "..." hay ký hiệu ba chấm; lượt sau hoàn tất ý.
streaming_asr phải viết thường và không dấu câu, giống đầu ra ChunkFormer.
Với interruption, assistant_history phải cho thấy trợ lý đang nói.
Với lời nền/nói với người khác, không được giả vờ đó là yêu cầu dành cho trợ lý."""


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


def normalize_streaming_asr(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char for char in text
    )
    return " ".join(text.split())


def fold_for_match(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = "".join(char if char.isalnum() else " " for char in text)
    return " ".join(text.split())


def asr_matches_user(user_text: str, streaming_asr: str) -> bool:
    user = fold_for_match(user_text)
    asr = fold_for_match(streaming_asr)
    if not user or not asr:
        return False
    if asr in user or user in asr:
        return True
    return difflib.SequenceMatcher(None, user, asr).ratio() >= 0.68


def validate_paid_models(models: list[str]) -> None:
    invalid = [
        model
        for model in models
        if model == "openrouter/free" or model.endswith(":free")
    ]
    if invalid:
        raise ValueError(f"Free/router models are prohibited: {invalid}")


def normalize_assigned_fields(obj: dict, job: dict) -> dict:
    """Stamp control fields supplied by the job; generators only author content."""
    turns = obj.get("turns")
    if not isinstance(turns, list) or len(turns) != job["round_count"]:
        return obj
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        action = job["required_tokens"][index]
        turn["action"] = action
        turn["event_role"] = job["required_roles"][index]
        turn["speaker_state"] = "speaking" if action in {"<|S-L|>", "<|C-S|>"} else "listening"
        if not str(turn.get("streaming_asr", "")).strip() and str(turn.get("user_text", "")).strip():
            turn["streaming_asr"] = turn["user_text"]
        turn["streaming_asr"] = normalize_streaming_asr(str(turn.get("streaming_asr", "")))
    return obj


def validate(obj: dict, job: dict) -> list[str]:
    errors = []
    if obj.get("conversation_id") != job["conversation_id"]:
        errors.append("conversation_id mismatch")
    turns = obj.get("turns")
    if not isinstance(turns, list) or len(turns) != job["round_count"]:
        return errors + ["wrong turn count"]
    actions = [turn.get("action") for turn in turns]
    if actions != job["required_tokens"]:
        errors.append("action sequence mismatch")
    roles = [turn.get("event_role") for turn in turns]
    if roles != job["required_roles"]:
        errors.append("event role sequence mismatch")
    for i, turn in enumerate(turns):
        action = turn.get("action")
        state = turn.get("speaker_state")
        if action not in TOKENS:
            errors.append(f"turn {i}: invalid action")
        if action in {"<|C-L|>", "<|S-S|>"} and state != "listening":
            errors.append(f"turn {i}: listening action/state mismatch")
        if action in {"<|S-L|>", "<|C-S|>"} and state != "speaking":
            errors.append(f"turn {i}: speaking action/state mismatch")
        if state == "speaking" and not str(turn.get("assistant_history", "")).strip():
            errors.append(f"turn {i}: speaking state missing assistant_history")
        for field in ("user_text", "streaming_asr", "event_role"):
            if not isinstance(turn.get(field), str) or not turn[field].strip():
                errors.append(f"turn {i}: missing {field}")
        if not asr_matches_user(
            str(turn.get("user_text", "")), str(turn.get("streaming_asr", ""))
        ):
            errors.append(f"turn {i}: streaming_asr unrelated to user_text")
    return errors


def atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def generate_one(client: OpenAI, job: dict, root: Path, retries: int) -> tuple[str, str]:
    accepted = root / "accepted" / f"{job['conversation_id']}.json"
    if accepted.exists():
        previous = json.loads(accepted.read_text(encoding="utf-8"))
        if not validate(previous.get("response", {}), job):
            return "skipped", job["conversation_id"]
        accepted.unlink()
    last_error = "unknown"
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=job["model"],
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(job)}],
                temperature=0.85,
                max_tokens=5000,
                response_format={"type": "json_object"},
                extra_headers={"X-Title": "Vietnamese Semantic DM Dataset"},
            )
            raw = response.choices[0].message.content or ""
            obj = normalize_assigned_fields(extract_json(raw), job)
            errors = validate(obj, job)
            audit = {"job": job, "response": obj, "validation_errors": errors,
                     "provider_model": response.model, "usage": response.usage.model_dump() if response.usage else None}
            atomic_write(root / "raw" / f"{job['conversation_id']}.json", audit)
            if not errors:
                atomic_write(accepted, audit)
                return "accepted", job["conversation_id"]
            last_error = "; ".join(errors)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(2 ** attempt, 8))
    atomic_write(root / "rejected" / f"{job['conversation_id']}.json", {"job": job, "error": last_error})
    return "rejected", f"{job['conversation_id']}: {last_error}"


def summarize(root: Path, jobs: list[dict]) -> dict:
    accepted = list((root / "accepted").glob("*.json")) if (root / "accepted").exists() else []
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in accepted]
    actions = Counter()
    models, provider_models, patterns = Counter(), Counter(), Counter()
    turns = 0
    for row in rows:
        job, response = row["job"], row["response"]
        models[job["model"]] += 1
        provider_models[row.get("provider_model", "unknown")] += 1
        patterns[job["pattern"]] += 1
        turns += len(response["turns"])
        actions.update(turn["action"] for turn in response["turns"])
    result = {"requested": len(jobs), "accepted": len(rows), "dialogue_rounds": turns,
              "actions": actions, "models": models, "provider_models": provider_models,
              "patterns": patterns}
    atomic_write(root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    validate_paid_models(args.models)
    jobs = [make_job(i, args.seed, args.models) for i in range(args.start, args.start + args.count)]
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output / "config.json", {"count": args.count, "start": args.start,
                 "seed": args.seed, "models": args.models, "topics": TOPICS, "styles": STYLES})
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=180, max_retries=1)
    counts = Counter()
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(generate_one, client, job, args.output, args.retries) for job in jobs]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            status, detail = future.result()
            with lock:
                counts[status] += 1
            print(f"[{done}/{len(jobs)}] {status}: {detail}", flush=True)
    print(json.dumps(summarize(args.output, jobs), ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
