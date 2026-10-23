#!/usr/bin/env python3
"""Resumable production generation with a configured strategy mixture."""

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
from generate import (DIRECTOR_SYSTEM, PROMPT_VERSION, SINGLE_SYSTEM, director_prompt,
                      role_play, schema_prompt, validate_blueprint)


def make_jobs(cfg: dict, start: int, count: int) -> list[dict]:
    rng = random.Random(int(cfg["seed"]))
    weights = cfg["strategy_weights"]
    # Exact counts are more auditable than probabilistic proportions.
    role_count = round(count * float(weights.get("role_play", 0)))
    strategies = ["role_play"] * role_count + ["single_author"] * (count - role_count)
    rng.shuffle(strategies)
    jobs = []
    for offset, strategy in enumerate(strategies):
        index = start + offset
        jobs.append({"index": index, "pair_key": f"production_{index:06d}",
                     "topic": TOPICS[index % len(TOPICS)], "style": STYLES[(index * 3) % len(STYLES)],
                     "round_count": rng.randint(int(cfg["rounds_min"]), int(cfg["rounds_max"])),
                     "event_types": ["backchannel", "real_interruption"] + (["ambiguous_interruption"] if index % 3 == 0 else []),
                     "strategy": strategy, "seed": rng.randrange(1_000_000_000)})
    return jobs


def generate_one(client: OpenAI, cfg: dict, job: dict) -> dict:
    retries, temp = int(cfg.get("max_retries", 2)), float(cfg.get("generation_temperature", .8))
    cid = f"vi_fd_{job['index']:06d}_{'rp' if job['strategy'] == 'role_play' else 'sa'}"
    pronoun_note = build_pronoun_note(cfg)
    job = {**job, "_pronoun_note": pronoun_note}
    last_error = "unknown"
    for attempt in range(retries + 1):
        try:
            blueprint, director = chat_json(client, cfg["role_play"]["director_model"], DIRECTOR_SYSTEM, director_prompt(job), .65)
            errors = validate_blueprint(blueprint, job)
            if errors:
                raise ValueError("blueprint:" + ",".join(errors))
            blueprint["_pronoun_note"] = pronoun_note
            if job["strategy"] == "single_author":
                model = cfg["single_author_models"][(job["index"] - 1) % len(cfg["single_author_models"])]
                single_sys = SINGLE_SYSTEM + f"\nXƯNG HÔ BẮT BUỘC: {pronoun_note}"
                row, author = chat_json(client, model, single_sys, schema_prompt(cid, blueprint), temp, 8000)
                if isinstance(row.get("rounds"), list):
                    row["rounds"] = [{"user": x.get("user"), "agent": x.get("agent"), "event": x.get("event")} if isinstance(x, dict) else x for x in row["rounds"]]
                author_calls = [author]
            else:
                row, author_calls = role_play(client, cid, blueprint, cfg)
            errors = validate(row, {"round_count": job["round_count"], "event_types": job["event_types"]})
            if errors:
                raise ValueError("conversation:" + ",".join(errors))
            return {"ok": True, "conversation": row, "provenance": {
                "conversation_id": cid, "strategy": job["strategy"], "prompt_version": PROMPT_VERSION,
                "job": job, "director": director, "author_calls": author_calls}}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 6))
    return {"ok": False, "job": job, "error": last_error}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--start", type=int, default=1001)
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    client = OpenAI(api_key=key, base_url=cfg["openrouter_base_url"], timeout=180, max_retries=1)
    item_dir = args.output / "items"
    item_dir.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(cfg, args.start, args.count)
    pending = [j for j in jobs if not (item_dir / f"{j['index']:06d}.json").exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_jobs = {pool.submit(generate_one, client, cfg, job): job for job in pending}
        for n, future in enumerate(concurrent.futures.as_completed(future_jobs), 1):
            result = future.result()
            job = future_jobs[future]
            path = item_dir / f"{job['index']:06d}.json"
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"[{n}/{len(pending)}] {job['index']:06d} {job['strategy']}: {'ok' if result['ok'] else 'failed'}", flush=True)
    results = [json.loads((item_dir / f"{j['index']:06d}.json").read_text(encoding="utf-8")) for j in jobs]
    good = [x for x in results if x["ok"]]
    write_jsonl(args.output / "generated.jsonl", (x["conversation"] for x in good))
    write_jsonl(args.output / "provenance.jsonl", (x["provenance"] for x in good))
    write_jsonl(args.output / "failures.jsonl", (x for x in results if not x["ok"]))
    summary = {"requested": len(jobs), "accepted": len(good), "failed": len(jobs)-len(good),
               "by_strategy": {s: sum(x["ok"] and x["provenance"]["strategy"] == s for x in results) for s in ("single_author", "role_play")}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

