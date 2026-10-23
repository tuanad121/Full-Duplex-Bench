"""Shared schema and deterministic validation for full-duplex conversations."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable


EVENT_ACTIONS = {
    "backchannel": "continue_speaking",
    "real_interruption": "stop_and_listen_then_reply",
    "ambiguous_interruption": "stop_and_listen_then_reply",
}
EVENT_TYPES = set(EVENT_ACTIONS)
REQUIRED_EVENT_FIELDS = {"type", "text", "agent_after", "overlap_fraction"}
VIETNAMESE_MARKS = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồẩẫậớờởỡợúùủũụứừửữựýỳỷỹỵ")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temp.replace(path)


def fold_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE).split())


def validate_conversation(row: dict, min_rounds: int = 6, max_rounds: int = 15) -> list[str]:
    errors: list[str] = []
    cid = row.get("conversation_id")
    if not isinstance(cid, str) or not re.fullmatch(r"vi_fd_[0-9]{6}(?:_(?:sa|rp))?", cid):
        errors.append("invalid_conversation_id")
    for field in ("topic", "style"):
        if not isinstance(row.get(field), str) or len(row[field].strip()) < 3:
            errors.append(f"missing_{field}")
    rounds = row.get("rounds")
    if not isinstance(rounds, list) or not min_rounds <= len(rounds) <= max_rounds:
        return errors + ["invalid_round_count"]
    event_types: list[str] = []
    seen_text: set[str] = set()
    for index, turn in enumerate(rounds):
        if not isinstance(turn, dict) or set(turn) != {"user", "agent", "event"}:
            errors.append(f"round_{index}:invalid_fields")
            continue
        for speaker in ("user", "agent"):
            value = turn.get(speaker)
            if not isinstance(value, str) or len(value.strip()) < 8:
                errors.append(f"round_{index}:{speaker}_too_short")
                continue
            folded = fold_text(value)
            if folded in seen_text:
                errors.append(f"round_{index}:{speaker}_duplicate")
            seen_text.add(folded)
            if not any(char in VIETNAMESE_MARKS for char in value.lower()):
                errors.append(f"round_{index}:{speaker}_unlikely_vietnamese")
        event = turn.get("event")
        if event is None:
            continue
        if not isinstance(event, dict) or set(event) != REQUIRED_EVENT_FIELDS:
            errors.append(f"round_{index}:invalid_event_fields")
            continue
        kind = event.get("type")
        if kind not in EVENT_TYPES:
            errors.append(f"round_{index}:invalid_event_type")
            continue
        event_types.append(kind)
        text = event.get("text")
        if not isinstance(text, str) or len(text.strip()) < 2:
            errors.append(f"round_{index}:event_text_too_short")
        fraction = event.get("overlap_fraction")
        if not isinstance(fraction, (int, float)) or not 0.2 <= fraction <= 0.8:
            errors.append(f"round_{index}:invalid_overlap_fraction")
        after = event.get("agent_after")
        if kind == "backchannel" and after != "":
            errors.append(f"round_{index}:backchannel_has_reply")
        if kind != "backchannel" and (not isinstance(after, str) or len(after.strip()) < 8):
            errors.append(f"round_{index}:interruption_missing_reply")
    if not event_types:
        errors.append("missing_behavioral_event")
    if len(event_types) > 4:
        errors.append("too_many_behavioral_events")
    return errors


def validate_corpus(rows: list[dict]) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    ids: set[str] = set()
    for position, row in enumerate(rows):
        cid = str(row.get("conversation_id") or f"row_{position}")
        errors = validate_conversation(row)
        if cid in ids:
            errors.append("duplicate_conversation_id")
        ids.add(cid)
        results[cid] = errors
    return results
