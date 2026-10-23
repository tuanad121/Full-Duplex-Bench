#!/usr/bin/env python3
"""Export Vietnamese DM conversations in the paper's per-decision format."""

from __future__ import annotations

import argparse
import difflib
import json
import unicodedata
from collections import Counter
from pathlib import Path


ACTIONS = {"<|C-L|>", "<|S-S|>", "<|S-L|>", "<|C-S|>"}
SYSTEM_PROMPT = (
    "Bạn là bộ quản lý hội thoại song công. Dựa trên lịch sử hội thoại, "
    "trạng thái hiện tại của trợ lý và giả thuyết ASR đang phát trực tuyến, "
    "chỉ xuất đúng một nhãn: <|C-L|>, <|S-S|>, <|S-L|>, hoặc <|C-S|>."
)


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = "".join(c if c.isalnum() else " " for c in text)
    return " ".join(text.split())


def asr_matches_user(turn: dict) -> bool:
    user = fold(str(turn.get("user_text", "")))
    asr = fold(str(turn.get("streaming_asr", "")))
    if not user or not asr or user == "ban sach tieng viet":
        return False
    if asr in user or user in asr:
        return True
    return difflib.SequenceMatcher(None, user, asr).ratio() >= 0.68


def quality_issues(row: dict) -> list[str]:
    issues: list[str] = []
    turns = row.get("response", {}).get("turns", [])
    if not turns:
        return ["no_turns"]
    for i, turn in enumerate(turns, 1):
        if turn.get("action") not in ACTIONS:
            issues.append(f"round_{i}:non_paper_action")
        if not asr_matches_user(turn):
            issues.append(f"round_{i}:asr_user_mismatch")
        state = turn.get("speaker_state")
        if state not in {"listening", "speaking"}:
            issues.append(f"round_{i}:invalid_speaker_state")
        if state == "speaking" and not str(turn.get("assistant_history", "")).strip():
            issues.append(f"round_{i}:missing_ongoing_assistant")
    return issues


def history_before(turns: list[dict], index: int) -> list[dict[str, str]]:
    """Return completed dialogue before a decision, without target leakage."""
    history: list[dict[str, str]] = []
    for prior in turns[:index]:
        user = str(prior.get("user_text", "")).strip()
        reply = str(prior.get("assistant_reply", "")).strip()
        if user:
            history.append({"role": "user", "content": user})
        if reply:
            history.append({"role": "assistant", "content": reply})
    return history


def decision_input(turns: list[dict], index: int) -> dict:
    turn = turns[index]
    state = turn["speaker_state"]
    ongoing = ""
    if state == "speaking":
        # These are the CDE response tokens already emitted at this decision.
        ongoing = str(turn.get("assistant_history", "")).strip()
    return {
        "dialogue_history": history_before(turns, index),
        "assistant_state": state,
        "assistant_ongoing": ongoing,
        "streaming_asr": str(turn["streaming_asr"]).strip(),
    }


def serialize_input(model_input: dict) -> str:
    lines = ["Lịch sử hội thoại:"]
    history = model_input["dialogue_history"]
    if history:
        for message in history:
            who = "Người dùng" if message["role"] == "user" else "Trợ lý"
            lines.append(f"{who}: {message['content']}")
    else:
        lines.append("∅")
    lines.append(f"Trạng thái trợ lý: {model_input['assistant_state']}")
    if model_input["assistant_ongoing"]:
        lines.append(f"Lời trợ lý đang nói: {model_input['assistant_ongoing']}")
    lines.append(f"ASR hiện tại: {model_input['streaming_asr']}")
    return "\n".join(lines)


def export_row(row: dict) -> list[dict]:
    response, job = row["response"], row["job"]
    cid = response["conversation_id"]
    turns = response["turns"]
    examples = []
    for index, turn in enumerate(turns):
        model_input = decision_input(turns, index)
        target = turn["action"]
        # Array order is authoritative. Some generated records omit or repeat
        # the display round number; retain that value only for audit.
        round_number = index + 1
        examples.append(
            {
                "example_id": f"{cid}_r{round_number:04d}",
                "conversation_id": cid,
                "round": round_number,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": serialize_input(model_input)},
                    {"role": "assistant", "content": target},
                ],
                "input": model_input,
                "target": target,
                "audit": {
                    "pattern": job["pattern"],
                    "event_role": turn.get("event_role"),
                    "full_user_text": turn.get("user_text"),
                    "assistant_reply": turn.get("assistant_reply"),
                    "topic": job.get("topic"),
                    "style": job.get("style"),
                    "context_family": job.get("context_family"),
                    "provider_model": row.get("provider_model"),
                    "source_round": turn.get("round"),
                },
            }
        )
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_paths = sorted((args.input / "accepted").glob("*.json"))
    safe_examples: list[dict] = []
    quarantined: list[dict] = []
    action_counts: Counter[str] = Counter()
    safe_conversations = 0
    for path in source_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        issues = quality_issues(row)
        if issues:
            quarantined.append(
                {
                    "conversation_id": row.get("response", {}).get("conversation_id"),
                    "source": str(path),
                    "issues": issues,
                }
            )
            continue
        safe_conversations += 1
        examples = export_row(row)
        safe_examples.extend(examples)
        action_counts.update(example["target"] for example in examples)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "decisions.jsonl", safe_examples)
    write_jsonl(args.output / "quarantine.jsonl", quarantined)
    manifest = {
        "schema_version": "vi_dm_paper_format_v1",
        "source": str(args.input),
        "format": "progressive multi-turn context; one control-token target per decision",
        "paper_action_space": sorted(ACTIONS),
        "source_conversations": len(source_paths),
        "accepted_conversations": safe_conversations,
        "quarantined_conversations": len(quarantined),
        "decision_examples": len(safe_examples),
        "action_counts": dict(sorted(action_counts.items())),
        "files": {
            "training_examples": "decisions.jsonl",
            "quality_quarantine": "quarantine.jsonl",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
