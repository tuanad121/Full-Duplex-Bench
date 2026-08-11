#!/usr/bin/env python3
"""Package a sanitized Vi-FDB reference run for public distribution."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def normalized_release_id(version: str, task: str, sample_id: str) -> str:
    if task != "user_interruption":
        return sample_id
    prefix = "standard" if version == "v1_0" else "counterfactual"
    return f"{prefix}_{sample_id}"


def sanitized_events(timing: dict) -> list[dict]:
    return [
        {
            key: event.get(key)
            for key in ("type", "time", "status")
            if event.get(key) is not None
        }
        for event in timing.get("events", [])
    ]


def load_review_transcripts(path: Path) -> dict[tuple[str, str, str], str]:
    """Recover the full ASR text captured before compatibility slicing."""
    page = path.read_text(encoding="utf-8")
    transcripts = {}
    for card in re.findall(r'<article class="card".*?</article>', page, re.DOTALL):
        source = re.search(r'src="(v1_[05])/([^/]+)/([^/]+)/input\.wav"', card)
        transcript = re.search(r'<p><b>chunkformer:</b>\s*(.*?)</p>', card, re.DOTALL)
        if not source or not transcript:
            continue
        text = re.sub(r"<[^>]+>", "", transcript.group(1))
        transcripts[source.groups()] = html.unescape(text).strip()
    return transcripts


def build_row(
    *,
    source: Path,
    destination: Path,
    benchmark_repo: str,
    benchmark_revision: str,
    version: str,
    task: str,
    sample_id: str,
    condition: str,
    judge: dict | None,
    human_review: dict | None,
    review_transcript: str | None,
) -> dict:
    prefix = "" if condition == "event" else "clean_"
    audio_source = source / f"{prefix}output.wav"
    asr = read_json(source / f"{prefix}output.chunkformer.json")
    timing = read_json(source / f"{prefix}output_timing.json")
    release_id = normalized_release_id(version, task, sample_id)
    release_path = f"{task}/{release_id}"
    audio_relative = f"audio/{version}/{task}/{sample_id}/{condition}.wav"
    audio_destination = destination / "data" / audio_relative
    audio_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(audio_source, audio_destination)

    effective = ({**judge, **human_review} if judge and human_review else judge)
    compatibility_slice = asr.get("event_slice")
    recovered_full_transcript = condition == "event" and compatibility_slice and review_transcript is not None
    transcript = review_transcript if recovered_full_transcript else asr.get("text", "")
    chunks = [] if recovered_full_transcript else asr.get("chunks", [])
    base_url = (
        f"https://huggingface.co/datasets/{benchmark_repo}/resolve/"
        f"{benchmark_revision}/data/pilot_160/{release_path}"
    )
    row = {
        "file_name": audio_relative,
        "benchmark_repo": benchmark_repo,
        "benchmark_revision": benchmark_revision,
        "benchmark_sample_id": release_id,
        "source_suite": version.replace("_", "."),
        "task": task,
        "condition": condition,
        "model": "gpt-realtime",
        "benchmark_input_url": f"{base_url}/{'input.wav' if condition == 'event' else 'clean_input.wav'}",
        "benchmark_metadata_url": f"{base_url}/metadata.json",
        "transcript": transcript,
        "transcript_source": (
            "final_review_snapshot_before_compatibility_slice"
            if recovered_full_transcript else "chunkformer_artifact"
        ),
        "asr_backend": asr.get("backend", "chunkformer"),
        "asr_chunks_json": json.dumps(chunks, ensure_ascii=False),
        "asr_chunks_scope": "unavailable_for_recovered_full_transcript" if recovered_full_transcript else "full_output",
        "clock": timing.get("clock"),
        "input_duration_seconds": timing.get("input_duration"),
        "output_duration_seconds": timing.get("output_timeline_duration"),
        "timing_events_json": json.dumps(sanitized_events(timing), ensure_ascii=False),
        "automated_pass": judge.get("pass") if judge and condition == "event" else None,
        "final_pass": effective.get("pass") if effective and condition == "event" else None,
        "adjudication": (
            "human" if human_review and condition == "event"
            else "automated" if judge and condition == "event"
            else None
        ),
        "observed_behavior": effective.get("observed_behavior") if effective and condition == "event" else None,
        "desired_behavior": judge.get("desired_behavior") if judge and condition == "event" else None,
        "judge_model": judge.get("judge_model") if judge and condition == "event" else None,
        "judge_confidence": judge.get("confidence") if judge and condition == "event" else None,
        "judge_evidence_vi": judge.get("evidence_vi") if judge and condition == "event" else None,
    }
    return row


def build_summary(rows: list[dict]) -> dict:
    event_rows = [row for row in rows if row["condition"] == "event"]
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    behaviors = Counter()
    for row in event_rows:
        groups[(row["source_suite"], row["task"])]["total"] += 1
        groups[(row["source_suite"], row["task"])]["pass"] += int(bool(row["final_pass"]))
        behaviors[row["observed_behavior"] or "UNKNOWN"] += 1
    per_task = []
    for (suite, task), counts in sorted(groups.items()):
        per_task.append({
            "source_suite": suite,
            "task": task,
            "pass": counts["pass"],
            "total": counts["total"],
            "pass_rate": counts["pass"] / counts["total"],
        })
    passed = sum(bool(row["final_pass"]) for row in event_rows)
    return {
        "model": "gpt-realtime",
        "event_samples": len(event_rows),
        "clean_control_samples": len(rows) - len(event_rows),
        "pass": passed,
        "total": len(event_rows),
        "pass_rate": passed / len(event_rows),
        "human_adjudications": sum(row["adjudication"] == "human" for row in event_rows),
        "per_task": per_task,
        "observed_behaviors": dict(sorted(behaviors.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-repo", default="tuanamz/vi-fdb-v1")
    parser.add_argument("--benchmark-revision", required=True)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = args.run_root / "human_review_overrides.json"
    reviews = read_json(reviews_path)
    human_decisions = reviews.get("decisions", {})
    review_transcripts = load_review_transcripts(
        args.run_root / "review_final_adjudicated.html"
    )
    if len(review_transcripts) != 160:
        raise SystemExit(
            f"Expected 160 review transcripts, found {len(review_transcripts)}"
        )
    rows = []

    for version in ("v1_0", "v1_5"):
        version_root = args.run_root / version
        for task_root in sorted(path for path in version_root.iterdir() if path.is_dir()):
            for sample_root in sorted(path for path in task_root.iterdir() if path.is_dir()):
                sample_id = sample_root.name
                key = f"{version}/{task_root.name}/{sample_id}"
                judge_path = sample_root / "judge.json"
                judge = read_json(judge_path) if judge_path.exists() else None
                human_review = human_decisions.get(key)
                rows.append(build_row(
                    source=sample_root,
                    destination=args.output_dir,
                    benchmark_repo=args.benchmark_repo,
                    benchmark_revision=args.benchmark_revision,
                    version=version,
                    task=task_root.name,
                    sample_id=sample_id,
                    condition="event",
                    judge=judge,
                    human_review=human_review,
                    review_transcript=review_transcripts.get(
                        (version, task_root.name, sample_id)
                    ),
                ))
                if (sample_root / "clean_output.wav").exists():
                    rows.append(build_row(
                        source=sample_root,
                        destination=args.output_dir,
                        benchmark_repo=args.benchmark_repo,
                        benchmark_revision=args.benchmark_revision,
                        version=version,
                        task=task_root.name,
                        sample_id=sample_id,
                        condition="clean",
                        judge=None,
                        human_review=None,
                        review_transcript=None,
                    ))

    metadata_path = args.output_dir / "data" / "metadata.jsonl"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(args.output_dir / "evaluation" / "summary.json", build_summary(rows))
    write_json(args.output_dir / "evaluation" / "human_review_overrides.json", reviews)
    for name in ("upstream_metrics.json", "interruption_relevance_summary.json"):
        source = args.run_root.parent / "english_fdb_compat" / name
        if source.exists():
            shutil.copy2(source, args.output_dir / "evaluation" / name)
    print(json.dumps(build_summary(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
