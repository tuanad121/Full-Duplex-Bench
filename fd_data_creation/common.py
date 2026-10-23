"""Shared I/O, validation, job construction, and OpenRouter helpers."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

EVENT_TYPES = {"backchannel", "real_interruption", "ambiguous_interruption"}
TOPICS = [
    "đặt bàn và chọn món", "lên kế hoạch du lịch", "sắp lịch họp",
    "hỗ trợ kỹ thuật", "nhà thông minh", "mua sắm và đổi trả",
    "giao hàng", "dẫn đường", "việc nhà", "lên kế hoạch với bạn bè",
]
STYLES = [
    "thân mật tự nhiên", "lịch sự", "ngắn gọn", "do dự có từ đệm",
    "tự sửa lời", "khẩu ngữ miền Bắc", "khẩu ngữ miền Nam",
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    tmp.replace(path)


def vendor(model: str) -> str:
    return model.split("/", 1)[0]


def choose_reviewer(author_models: list[str], reviewers: list[str]) -> str:
    author_vendors = {vendor(x) for x in author_models}
    return next((x for x in reviewers if vendor(x) not in author_vendors), reviewers[0])


def build_pronoun_note(cfg: dict) -> str:
    """Return a Vietnamese pronoun-consistency constraint derived from voice genders in config."""
    g = cfg.get("voice_gender", {"user": "nữ", "agent": "nam"})
    ug, ag = g.get("user", "nữ"), g.get("agent", "nam")
    _PAIRS = {
        ("nữ", "nam"): ('USER là NỮ: tự xưng "em"/"chị"/"tôi"/"mình", gọi agent là "anh". '
                        'AGENT là NAM: tự xưng "anh"/"tôi", gọi user là "em"/"chị"/"bạn". '
                        'NGHIÊM CẤM: gọi agent là "bà"/"cô", gọi user là "anh"/"ông".'),
        ("nam", "nam"): ('USER là NAM: tự xưng "anh"/"tôi"/"mình", gọi agent là "bạn"/"anh". '
                         'AGENT là NAM: tự xưng "anh"/"tôi", gọi user là "bạn"/"anh". '
                         'NGHIÊM CẤM: gọi bên kia là "chị"/"bà"/"cô".'),
        ("nữ", "nữ"): ('USER là NỮ: tự xưng "em"/"chị"/"tôi"/"mình", gọi agent là "chị"/"bạn". '
                       'AGENT là NỮ: tự xưng "em"/"chị"/"tôi", gọi user là "em"/"chị"/"bạn". '
                       'NGHIÊM CẤM: gọi bên kia là "anh"/"ông".'),
        ("nam", "nữ"): ('USER là NAM: tự xưng "anh"/"tôi"/"mình", gọi agent là "chị"/"bạn". '
                        'AGENT là NỮ: tự xưng "em"/"chị"/"tôi", gọi user là "anh"/"bạn". '
                        'NGHIÊM CẤM: gọi agent là "anh"/"ông", gọi user là "chị"/"bà".'),
    }
    return _PAIRS.get((ug, ag), 'Dùng "bạn" (trung tính) cho cả hai vai để tránh nhầm giới tính.')


def build_jobs(cfg: dict, start: int, count: int) -> list[dict]:
    rng = random.Random(int(cfg["seed"]))
    weights = cfg["strategy_weights"]
    strategies = list(weights)
    jobs = []
    for offset in range(count):
        index = start + offset
        strategy = rng.choices(strategies, weights=[weights[x] for x in strategies], k=1)[0]
        jobs.append({
            "conversation_id": f"vi_fd_{index:06d}",
            "topic": TOPICS[index % len(TOPICS)],
            "style": STYLES[(index * 3) % len(STYLES)],
            "round_count": rng.randint(int(cfg["rounds_min"]), int(cfg["rounds_max"])),
            "event_types": ["backchannel", "real_interruption"] + (["ambiguous_interruption"] if index % 3 == 0 else []),
            "strategy": strategy,
            "seed": rng.randrange(1_000_000_000),
            "pair_key": f"{index:06d}",
        })
    return jobs


def validate(row: dict, expected: dict | None = None) -> list[str]:
    errors: list[str] = []
    if not re.fullmatch(r"vi_fd_\d{6}(?:_(?:sa|rp))?", str(row.get("conversation_id", ""))):
        errors.append("invalid_conversation_id")
    rounds = row.get("rounds")
    if not isinstance(rounds, list) or not 6 <= len(rounds) <= 15:
        return errors + ["invalid_round_count"]
    seen: set[str] = set()
    actual_events: list[str] = []
    for i, turn in enumerate(rounds):
        if not isinstance(turn, dict) or set(turn) != {"user", "agent", "event"}:
            errors.append(f"round_{i}:invalid_fields")
            continue
        for role in ("user", "agent"):
            text = turn.get(role)
            if not isinstance(text, str) or len(text.strip()) < 6:
                errors.append(f"round_{i}:{role}_too_short")
            else:
                folded = re.sub(r"\W+", " ", text.lower()).strip()
                if folded in seen:
                    errors.append(f"round_{i}:{role}_duplicate")
                seen.add(folded)
        event = turn.get("event")
        if event is None:
            continue
        if not isinstance(event, dict) or set(event) != {"type", "text", "agent_after", "overlap_fraction"}:
            errors.append(f"round_{i}:invalid_event")
            continue
        kind = event.get("type")
        actual_events.append(kind)
        if kind not in EVENT_TYPES:
            errors.append(f"round_{i}:invalid_event_type")
        if not isinstance(event.get("text"), str) or len(event["text"].strip()) < 2:
            errors.append(f"round_{i}:event_text")
        if not isinstance(event.get("overlap_fraction"), (int, float)) or not 0.2 <= event["overlap_fraction"] <= 0.8:
            errors.append(f"round_{i}:overlap_fraction")
        after = event.get("agent_after")
        if kind == "backchannel" and after != "":
            errors.append(f"round_{i}:backchannel_reply")
        if kind != "backchannel" and (not isinstance(after, str) or len(after.strip()) < 6):
            errors.append(f"round_{i}:missing_interruption_reply")
    if expected:
        if len(rounds) != expected["round_count"]:
            errors.append("round_count_mismatch")
        for kind in expected["event_types"]:
            if kind not in actual_events:
                errors.append(f"missing_{kind}")
    # Dialect consistency: collect all user speech (main turns + event.text)
    user_speech = []
    for turn in rounds:
        if isinstance(turn.get("user"), str):
            user_speech.append(turn["user"])
        ev = turn.get("event")
        if isinstance(ev, dict) and isinstance(ev.get("text"), str):
            user_speech.append(ev["text"])
    combined = " ".join(user_speech).lower()
    has_north = bool(re.search(r'\bnhỉ\b', combined))
    has_south = bool(re.search(r'\b(nha|hén|vậy nè|nghen)\b', combined))
    if has_north and has_south:
        errors.append("dialect_mix")
    return errors


def _repair_truncated_json(raw: str) -> dict:
    """Close a JSON object that was cut off mid-response by finding the last safe position."""
    in_str = escaped = False
    depth = 0
    last_comma_at_depth1 = -1
    for i, ch in enumerate(raw):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_str:
            escaped = True
            continue
        if ch == '"':
            in_str = not in_str
        elif not in_str:
            if ch in ("{", "["):
                depth += 1
            elif ch in ("}", "]"):
                depth -= 1
            elif ch == "," and depth == 1:
                last_comma_at_depth1 = i
    if last_comma_at_depth1 >= 0:
        return json.loads(raw[:last_comma_at_depth1].rstrip() + "}")
    raise json.JSONDecodeError("unrepairable truncated JSON", raw, len(raw))


def chat_json(client, model: str, system: str, prompt: str, temperature: float, max_tokens: int = 5000) -> tuple[dict, dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_headers={"X-Title": "Vietnamese Full-Duplex Data Creation"},
    )
    choice = response.choices[0]
    raw = choice.message.content or ""
    if not raw.strip():
        raise ValueError(f"{model} returned empty content; finish_reason={choice.finish_reason}")
    if raw.lstrip().startswith("```"):
        raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.I | re.S)
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        if choice.finish_reason == "length":
            raise ValueError(f"{model} response truncated (finish_reason=length); increase max_tokens")
        body = _repair_truncated_json(raw)
    usage = response.usage.model_dump() if response.usage else None
    return body, {"requested_model": model, "provider_model": response.model, "usage": usage}
