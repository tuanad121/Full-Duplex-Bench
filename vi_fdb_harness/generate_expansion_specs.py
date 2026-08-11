#!/usr/bin/env python3
"""Generate and validate the 240-case Vietnamese FDB expansion text specs."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


TASKS = {
    "pause_handling": ("v1_0", "within_utterance_pause"),
    "smooth_turn_taking": ("v1_0", "turn_end"),
    "backchannel": ("v1_0", "addressed_backchannel"),
    "user_interruption_v1_0": ("v1_0", "addressed_request_change"),
    "user_interruption_v1_5": ("v1_5", "addressed_request_change"),
    "user_backchannel": ("v1_5", "addressed_backchannel"),
    "talking_to_other": ("v1_5", "non_addressed_side_conversation"),
    "background_speech": ("v1_5", "non_addressed_background"),
}
FAMILIES = ["service_assistant", "smart_home", "in_car", "human_human"]

SYSTEM = """You design a Vietnamese full-duplex spoken-dialogue benchmark.
Return natural, concise Vietnamese suitable for TTS. Do not copy examples from an existing dataset.
Use diverse receptionist/service, smart-home, in-car, and human-human settings.
Avoid unsafe, private, or time-sensitive factual content. Return JSON only."""


def prompt(task: str, count: int, start_id: int) -> str:
    version, role = TASKS[task]
    special = {
        "pause_handling": "primary_text must contain exactly one literal ' ... ' at a meaningful mid-utterance pause. Include a mixture of pauses between words and pauses splitting Vietnamese compounds such as 'tổng ... quát'. event_text must be '[PAUSE]'.",
        "smooth_turn_taking": "event_text must be '[TURN_END]'. The primary request must be complete and natural.",
        "backchannel": "primary_text asks for a multi-step explanation; event_text is a short addressed backchannel that means continue listening.",
        "user_interruption_v1_0": "event_text changes or replaces at least one concrete constraint in primary_text.",
        "user_interruption_v1_5": "event_text changes or replaces at least one concrete constraint in primary_text.",
        "user_backchannel": "event_text is a short acknowledgement, not a changed request.",
        "talking_to_other": "event_text is clearly addressed to another nearby human and unrelated to primary_text.",
        "background_speech": "event_text is a non-addressed announcement/background utterance. Include both unrelated topics and semantically related-but-non-addressed distractors.",
    }[task]
    quotas = {family: count // len(FAMILIES) for family in FAMILIES}
    for family in FAMILIES[: count % len(FAMILIES)]:
        quotas[family] += 1
    return f"""Create exactly {count} NEW cases for task={task}, version={version}, reference_role={role}.
IDs must be zero-padded strings {start_id:06d} through {start_id + count - 1:06d}.
{special}
Use these exact context_family counts: {json.dumps(quotas)}. Human-human cases are required even for talking_to_other: the primary conversation is between humans and the event addresses a different nearby person. Vary setting, intent, wording, names, numbers, dates, and constraints. Do not repeat a primary_text or event_text.
Each item must contain exactly: id, primary_text, event_text, context_family, interaction_type, setting, scenario_id, primary_speaker, event_speaker.
Speakers must be one of north_female, north_male, south_female, south_male. interaction_type is human_machine or human_human.
Return {{"samples": [...]}}."""


def validate(all_rows: list[dict], per_task: int, start_id: int) -> dict:
    errors = []
    seen_primary, seen_pair = set(), set()
    counts = {task: 0 for task in TASKS}
    required = {"id", "primary_text", "event_text", "context_family", "interaction_type", "setting", "scenario_id", "primary_speaker", "event_speaker", "task", "version", "event_role"}
    for row in all_rows:
        missing = required - row.keys()
        if missing:
            errors.append(f"{row.get('task')}/{row.get('id')}: missing {sorted(missing)}")
            continue
        counts[row["task"]] = counts.get(row["task"], 0) + 1
        primary = " ".join(row["primary_text"].casefold().split())
        pair = (primary, " ".join(row["event_text"].casefold().split()))
        if primary in seen_primary:
            errors.append(f"duplicate primary_text: {row['primary_text']}")
        if pair in seen_pair:
            errors.append(f"duplicate pair: {row['task']}/{row['id']}")
        seen_primary.add(primary); seen_pair.add(pair)
        if row["task"] == "pause_handling" and row["primary_text"].count(" ... ") != 1:
            errors.append(f"pause syntax: {row['id']}")
        if row["context_family"] not in FAMILIES:
            errors.append(f"bad family: {row['task']}/{row['id']}")
    for task, count in counts.items():
        if count != per_task:
            errors.append(f"{task}: expected {per_task}, got {count}")
        ids = sorted(int(row["id"]) for row in all_rows if row.get("task") == task)
        if ids != list(range(start_id, start_id + per_task)):
            errors.append(f"{task}: incorrect id range")
        family_counts = [sum(1 for row in all_rows if row.get("task") == task and row.get("context_family") == family) for family in FAMILIES]
        if family_counts and max(family_counts) - min(family_counts) > 1:
            errors.append(f"{task}: unbalanced context families {dict(zip(FAMILIES, family_counts))}")
    return {"ok": not errors, "total": len(all_rows), "counts": counts, "errors": errors}


def build_html(rows: list[dict], destination: Path) -> None:
    cards = []
    for row in rows:
        cards.append(f"""<article><b>{html.escape(row['task'])} · {row['id']}</b><span>{html.escape(row['context_family'])} · {html.escape(row['event_role'])}</span><p><strong>Primary:</strong> {html.escape(row['primary_text'])}</p><p><strong>Event:</strong> {html.escape(row['event_text'])}</p></article>""")
    destination.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Vi-FDB 240 expansion text review</title><style>body{{font:15px/1.45 system-ui;background:#eef2f7;color:#172033;max-width:1100px;margin:30px auto;padding:0 18px}}article{{background:white;border:1px solid #dce2ec;border-radius:12px;padding:16px;margin:12px 0}}span{{float:right;color:#667085}}strong{{color:#344054}}</style></head><body><h1>Vietnamese FDB expansion — 240 text candidates</h1>{''.join(cards)}</body></html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=30)
    parser.add_argument("--start-id", type=int, default=21)
    parser.add_argument("--model", default=os.getenv("OPENAI_SPEC_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.merge_only:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY"))
        selected = {args.task: TASKS[args.task]} if args.task else TASKS
        for task, (version, role) in selected.items():
            response = client.chat.completions.create(
                model=args.model,
                response_format={"type": "json_object"},
                temperature=0.8,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(task, args.per_task, args.start_id)}],
            )
            samples = json.loads(response.choices[0].message.content)["samples"]
            for sample in samples:
                sample.update(task=task, version=version, event_role=role)
            (args.output / f"specs_{task}.json").write_text(json.dumps({"samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{task}: {len(samples)}", flush=True)
        if args.task:
            return
    for task in TASKS:
        checkpoint = args.output / f"specs_{task}.json"
        if checkpoint.exists():
            rows.extend(json.loads(checkpoint.read_text(encoding="utf-8"))["samples"])
    result = validate(rows, args.per_task, args.start_id)
    (args.output / "specs.json").write_text(json.dumps({"schema_version": 1, "samples": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    build_html(rows, args.output / "text_review.html")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
