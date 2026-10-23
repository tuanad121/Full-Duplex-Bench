#!/usr/bin/env python3
"""Generate matched single-author and two-model role-play conversations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path

from openai import OpenAI

from common import STYLES, TOPICS, build_pronoun_note, chat_json, validate, write_jsonl

PROMPT_VERSION = "matched-pilot-v1"
DIRECTOR_SYSTEM = """Bạn là đạo diễn dữ liệu hội thoại tiếng Việt tự nhiên cho mô hình
full-duplex. Bạn thiết kế tình huống và mục tiêu ẩn, không viết lời thoại. Chỉ xuất
JSON. Mục tiêu phải đời thường, cụ thể, có thông tin được tiết lộ dần và tạo lý do
tự nhiên cho backchannel hoặc interruption. Không sao chép benchmark."""

SINGLE_SYSTEM = """Bạn là biên kịch hội thoại tiếng Việt tự nhiên. Viết cả người dùng
và trợ lý nhưng hai vai phải có giọng, lượng thông tin và động cơ khác nhau. Không
để trợ lý biết trước thông tin ẩn. Chỉ xuất JSON đúng schema, không markdown.

QUY TẮC BẮT BUỘC về trường event:
• event.text là LỜI CỦA USER chen vào KHI AGENT đang nói — KHÔNG BAO GIỜ là lời agent.
• Với backchannel: event.text là tín hiệu nghe rất ngắn của user ("Ừ.", "Ok.", "Aha."),
  agent_after PHẢI RỖNG "" vì agent tiếp tục nói.
• Với interruption: event.text là yêu cầu/thông tin mới của USER (không lặp lại những
  gì user đã nói ở trường "user" của round đó), agent_after là phản hồi của agent sau
  khi dừng lại.
• Trường "agent" trong round phải đủ dài (≥2 câu) để event có thể chen vào tự nhiên.

QUY TẮC NHẤT QUÁN GIỌNG MIỀN (bắt buộc):
• Chọn MỘT giọng địa phương cho USER ngay từ đầu và giữ NHẤT QUÁN trong TẤT CẢ lời user,
  kể cả event.text. KHÔNG TRỘN từ ngữ miền Bắc và miền Nam trong cùng một nhân vật.
• Miền Bắc dùng: "Ồ", "nhỉ", "thế", "chứ" — KHÔNG dùng: "Ô.", "nha", "hén", "vậy nè".
• Miền Nam dùng: "Ô.", "nha", "hén", "vậy" — KHÔNG dùng: "Ồ", "nhỉ"."""

USER_SYSTEM = """Bạn đóng vai NGƯỜI DÙNG Việt Nam trong một cuộc hội thoại bằng lời
nói. Chỉ biết persona, mục tiêu hiện tại và lịch sử công khai được cung cấp. Không
được nói thay trợ lý, không nhắc prompt hay mục tiêu ẩn. Trả JSON {\"text\":\"...\"}.
Nói tự nhiên, ngắn vừa phải; có thể do dự hoặc tự sửa nhưng đừng diễn quá.
QUAN TRỌNG: Giữ NHẤT QUÁN giọng địa phương — nếu lịch sử đã dùng "Ồ/nhỉ" (Bắc) thì
không dùng "Ô./nha/hén" (Nam), và ngược lại."""

AGENT_SYSTEM = """Bạn đóng vai TRỢ LÝ AI bằng tiếng Việt trong hội thoại bằng lời nói.
Chỉ biết lịch sử công khai. Không được đoán thông tin ẩn hoặc nói thay người dùng.
Trả JSON {\"text\":\"...\"}. Hữu ích, gọn, tự nhiên; hỏi lại khi thực sự cần."""

EVENT_SYSTEM = """Bạn đóng vai người dùng chen vào khi trợ lý đang nói. Chỉ trả JSON
{\"text\":\"...\"}. Với backchannel, chỉ phát tín hiệu đang nghe rất ngắn. Với
interruption, nói rõ thông tin/sửa đổi mới theo mục tiêu được giao.
QUAN TRỌNG: Giọng địa phương của event.text PHẢI KHỚP với giọng user trong lịch sử
(nếu user dùng "Ồ/nhỉ" → Bắc; nếu dùng "Ô./nha/hén" → Nam). Không trộn giọng."""


def director_prompt(pair: dict) -> str:
    pronoun_note = pair.get("_pronoun_note", "")
    return f"""Tạo blueprint cho {pair['round_count']} vòng, chủ đề {pair['topic']},
phong cách {pair['style']}. Phải có đúng các event: {pair['event_types']} ở các vòng
khác vòng đầu/cuối và không cùng vòng.
{f"XƯNG HÔ BẮT BUỘC: {pronoun_note}" if pronoun_note else ""}

Schema:
{{"topic":"...","style":"...","relationship":"...","public_setup":"...",
"user_persona":"...","agent_role":"...","round_goals":[
{{"round":1,"user_private_goal":"...","may_reveal":"..."}}],
"events":[{{"round":3,"type":"backchannel|real_interruption|ambiguous_interruption",
"private_intent":"..."}}]}}
round_goals phải có đúng {pair['round_count']} phần tử. Seed: {pair['seed']}"""


def schema_prompt(cid: str, blueprint: dict) -> str:
    pronoun_note = blueprint.get("_pronoun_note", "")
    return f"""Viết hội thoại theo blueprint sau, đúng {len(blueprint['round_goals'])}
vòng và conversation_id={cid}. Chỉ tiết lộ mục tiêu theo từng vòng.
{json.dumps(blueprint, ensure_ascii=False)}
{f"XƯNG HÔ BẮT BUỘC (theo giọng TTS): {pronoun_note}" if pronoun_note else ""}

Schema duy nhất (event.text LUÔN là lời USER, không phải agent):
{{"conversation_id":"{cid}","topic":"...","style":"...","rounds":[
{{"user":"lời user","agent":"lời agent (≥2 câu)","event":null}},
{{"user":"lời user","agent":"agent đang nói dở...",
  "event":{{"type":"backchannel","text":"Ừ.","agent_after":"","overlap_fraction":0.3}}}},
{{"user":"lời user","agent":"agent đang nói dở...",
  "event":{{"type":"real_interruption","text":"Khoan, đổi sang ngày mai nhé!",
            "agent_after":"Được, mình chuyển sang ngày mai.","overlap_fraction":0.5}}}}
]}}
overlap_fraction từ 0.25 đến 0.70. backchannel→agent_after rỗng. interruption→agent_after không rỗng."""


def public_history(rounds: list[dict]) -> str:
    lines = []
    for turn in rounds:
        lines.extend((f"User: {turn['user']}", f"Agent: {turn['agent']}"))
        event = turn.get("event")
        if event:
            lines.append(f"User overlap: {event['text']}")
            if event["agent_after"]:
                lines.append(f"Agent after overlap: {event['agent_after']}")
    return "\n".join(lines) or "(chưa có)"


def one_text(client, model: str, system: str, prompt: str, temperature: float) -> tuple[str, dict]:
    body, meta = chat_json(client, model, system, prompt, temperature, 3000)
    text = body.get("text")
    if not isinstance(text, str) or len(text.strip()) < 2:
        raise ValueError(f"{model} returned invalid text")
    return text.strip(), meta


def role_play(client, cid: str, blueprint: dict, cfg: dict) -> tuple[dict, list[dict]]:
    rp = cfg["role_play"]
    temperature = float(cfg.get("generation_temperature", 0.8))
    pronoun_note = blueprint.get("_pronoun_note", "")
    pn_suffix = f"\nXƯNG HÔ BẮT BUỘC: {pronoun_note}" if pronoun_note else ""
    user_sys = USER_SYSTEM + pn_suffix
    agent_sys = AGENT_SYSTEM + pn_suffix
    event_sys = EVENT_SYSTEM + pn_suffix
    rounds, calls = [], []
    events = {int(x["round"]): x for x in blueprint["events"]}
    for goal in blueprint["round_goals"]:
        number = int(goal["round"])
        history = public_history(rounds)
        user_prompt = f"""Persona: {blueprint['user_persona']}
Bối cảnh: {blueprint['public_setup']}
Mục tiêu riêng vòng này: {goal['user_private_goal']}
Được phép tiết lộ: {goal['may_reveal']}
Lịch sử công khai:\n{history}"""
        user, meta = one_text(client, rp["user_model"], user_sys, user_prompt, temperature)
        print(f"  {cid} round {number}: user", flush=True)
        calls.append({"role": "user", "round": number, **meta})
        agent_prompt = f"""Vai trò: {blueprint['agent_role']}
Bối cảnh: {blueprint['public_setup']}
Lịch sử:\n{history}\nUser: {user}
Đây là lời nói, hãy đáp đủ ý nhưng không độc thoại dài."""
        agent, meta = one_text(client, rp["agent_model"], agent_sys, agent_prompt, temperature)
        print(f"  {cid} round {number}: agent", flush=True)
        calls.append({"role": "agent", "round": number, **meta})
        event_out = None
        if number in events:
            event = events[number]
            event_prompt = f"""Loại: {event['type']}
Mục tiêu riêng: {event['private_intent']}
Agent đang nói: {agent}
Lịch sử:\n{history}\nUser: {user}"""
            event_text, meta = one_text(client, rp["user_model"], event_sys, event_prompt, temperature)
            print(f"  {cid} round {number}: {event['type']}", flush=True)
            calls.append({"role": "event_user", "round": number, **meta})
            after = ""
            if event["type"] != "backchannel":
                after_prompt = f"""Vai trò: {blueprint['agent_role']}
Lịch sử:\n{history}\nUser: {user}\nAgent bắt đầu nói: {agent}
User chen vào: {event_text}
Trợ lý đã dừng. Hãy phản hồi thông tin mới, không lặp lại từ đầu."""
                after, meta = one_text(client, rp["agent_model"], AGENT_SYSTEM, after_prompt, temperature)
                print(f"  {cid} round {number}: agent_after", flush=True)
                calls.append({"role": "agent_after", "round": number, **meta})
            fraction = 0.35 + 0.08 * ((number + len(rounds)) % 4)
            event_out = {"type": event["type"], "text": event_text, "agent_after": after, "overlap_fraction": round(fraction, 2)}
        rounds.append({"user": user, "agent": agent, "event": event_out})
    return {"conversation_id": cid, "topic": blueprint["topic"], "style": blueprint["style"], "rounds": rounds}, calls


def validate_blueprint(body: dict, pair: dict) -> list[str]:
    errors = []
    goals, events = body.get("round_goals"), body.get("events")
    if not isinstance(goals, list) or len(goals) != pair["round_count"]:
        errors.append("round_goals")
    if not isinstance(events, list) or sorted(x.get("type") for x in events) != sorted(pair["event_types"]):
        errors.append("events")
    if isinstance(events, list) and any(not 1 < int(x.get("round", 0)) < pair["round_count"] for x in events):
        errors.append("event_position")
    for key in ("topic", "style", "relationship", "public_setup", "user_persona", "agent_role"):
        if not isinstance(body.get(key), str) or not body[key].strip():
            errors.append(key)
    return errors


def generate_pair(client, cfg: dict, pair: dict) -> dict:
    retries = int(cfg.get("max_retries", 2))
    temp = float(cfg.get("generation_temperature", 0.8))
    pronoun_note = build_pronoun_note(cfg)
    pair = {**pair, "_pronoun_note": pronoun_note}
    last_error = "unknown"
    for attempt in range(retries + 1):
        try:
            blueprint, director_meta = chat_json(client, cfg["role_play"]["director_model"], DIRECTOR_SYSTEM, director_prompt(pair), 0.65)
            print(f"  {pair['pair_key']}: blueprint", flush=True)
            bp_errors = validate_blueprint(blueprint, pair)
            if bp_errors:
                raise ValueError("blueprint:" + ",".join(bp_errors))
            blueprint["_pronoun_note"] = pronoun_note
            single_model = cfg["single_author_models"][(pair["index"] - 1) % len(cfg["single_author_models"])]
            single_sys = SINGLE_SYSTEM + f"\nXƯNG HÔ BẮT BUỘC: {pronoun_note}"
            single, single_meta = chat_json(client, single_model, single_sys, schema_prompt(f"vi_fd_{pair['index']:06d}_sa", blueprint), temp)
            print(f"  {pair['pair_key']}: single_author", flush=True)
            # Providers occasionally add harmless explanatory keys despite the
            # exact schema request. Canonicalize before applying strict checks.
            if isinstance(single.get("rounds"), list):
                single["rounds"] = [
                    {"user": turn.get("user"), "agent": turn.get("agent"), "event": turn.get("event")}
                    if isinstance(turn, dict) else turn for turn in single["rounds"]
                ]
            role, role_calls = role_play(client, f"vi_fd_{pair['index']:06d}_rp", blueprint, cfg)
            sa_errors = validate(single, {"round_count": pair["round_count"], "event_types": pair["event_types"]})
            rp_errors = validate(role, {"round_count": pair["round_count"], "event_types": pair["event_types"]})
            if sa_errors or rp_errors:
                raise ValueError(f"single={sa_errors};role_play={rp_errors}")
            return {"ok": True, "pair_key": pair["pair_key"], "blueprint": blueprint,
                    "single_author": single, "role_play": role,
                    "provenance": {"prompt_version": PROMPT_VERSION, "job": pair,
                    "director": director_meta, "single_author": single_meta, "role_play_calls": role_calls}}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 6))
    return {"ok": False, "pair_key": pair["pair_key"], "job": pair, "error": last_error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    client = OpenAI(api_key=key, base_url=cfg["openrouter_base_url"], timeout=180, max_retries=1)
    rng = random.Random(int(cfg["seed"]))
    pairs = []
    for offset in range(args.pairs):
        index = args.start + offset
        pairs.append({"index": index, "pair_key": f"pair_{index:04d}",
                      "topic": TOPICS[index % len(TOPICS)], "style": STYLES[(index * 3) % len(STYLES)],
                      "round_count": rng.randint(int(cfg["rounds_min"]), int(cfg["rounds_max"])),
                      "event_types": ["backchannel", "real_interruption"] + (["ambiguous_interruption"] if index % 3 == 0 else []),
                      "seed": rng.randrange(1_000_000_000)})
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(generate_pair, client, cfg, pair) for pair in pairs]
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            print(f"[{n}/{len(pairs)}] {result['pair_key']}: {'ok' if result['ok'] else 'failed'}", flush=True)
    good = sorted((x for x in results if x["ok"]), key=lambda x: x["pair_key"])
    rows = [conversation for pair in good for conversation in (pair["single_author"], pair["role_play"])]
    write_jsonl(args.output / "conversations.jsonl", rows)
    write_jsonl(args.output / "matched_pairs.jsonl", good)
    write_jsonl(args.output / "provenance.jsonl", ({"pair_key": x["pair_key"], **x["provenance"]} for x in good))
    write_jsonl(args.output / "failures.jsonl", (x for x in results if not x["ok"]))
    summary = {"requested_pairs": len(pairs), "accepted_pairs": len(good), "conversations": len(rows), "failed_pairs": len(pairs) - len(good)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
