#!/usr/bin/env python3
"""Blinded semantic behavior judge for Vietnamese FDB outputs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import re
import time

DESIRED = {
    "backchannel": "CONTINUE_OR_RESUME",
    "user_backchannel": "CONTINUE_OR_RESUME",
    "pause_handling": "WAIT_THEN_TAKE_TURN",
    "smooth_turn_taking": "TAKE_TURN",
    "user_interruption": "YIELD_AND_FOLLOW_NEW_REQUEST",
    "talking_to_other": "IGNORE_DISTRACTOR",
    "background_speech": "IGNORE_DISTRACTOR",
}
LABELS = list(dict.fromkeys(DESIRED.values())) + [
    "PREMATURE_TURN",
    "MISSED_INTERRUPTION",
    "ENGAGED_DISTRACTOR",
    "UNCERTAIN_HANDLING",
    "UNRELATED",
    "NO_SPEECH",
]

SYSTEM = f"""You judge observable behavior of a full-duplex Vietnamese spoken assistant.
Classify what the assistant actually did; do not infer behavior from internal API events.
Use exactly one observed_behavior from: {', '.join(LABELS)}.

Definitions:
- CONTINUE_OR_RESUME: accepts/ignores a backchannel and continues the same answer, including a brief acknowledgement such as 'mình nói tiếp nhé'.
- WAIT_THEN_TAKE_TURN: stays silent during a within-utterance pause and answers only after the user completes the utterance.
- TAKE_TURN: begins a substantive answer after the user finishes.
- YIELD_AND_FOLLOW_NEW_REQUEST: stops/yields and follows the user's new interrupting request.
- IGNORE_DISTRACTOR: does not engage with side/background speech and preserves the primary conversation.
- PREMATURE_TURN: begins answering an incomplete utterance during a pause, before the user resumes and finishes it.
- MISSED_INTERRUPTION: produces no new response to a spoken request that changes or replaces the original request.
- ENGAGED_DISTRACTOR: quotes, answers, or acts on unrelated side/background speech instead of preserving the primary conversation.
- UNCERTAIN_HANDLING: asks to repeat or reports difficulty hearing.
- UNRELATED: speech is off-topic or does not clearly satisfy another label.
- NO_SPEECH: no assistant words are present.

Return JSON only with observed_behavior, confidence (0..1), and evidence_vi (one short Vietnamese sentence).
For an input containing [PAUSE], distinguish these carefully:
- An internal response.created followed by cancellation is diagnostic, not automatically failure.
- Use PREMATURE_TURN when audible output starts before the user finishes, or the final answer clearly responds to an incomplete prefix and loses important words after the pause.
- Use WAIT_THEN_TAKE_TURN when the observable final answer coherently incorporates both sides of the pause, even if an inaudible internal response attempt was cancelled.
- Use NO_SPEECH when no assistant words are present after the completed request.

Label priority for unrelated side/background speech:
- If the final response stays on the primary request and does not engage the unrelated event, use IGNORE_DISTRACTOR, even when TAKE_TURN is also technically true.
- If the final response quotes, answers, or acts on the unrelated event, use ENGAGED_DISTRACTOR.

Synthetic boundary examples:
- Primary asks for a hotel room; unrelated announcement calls ticket 23; assistant continues discussing the room without mentioning the announcement -> IGNORE_DISTRACTOR.
- Primary asks about an appointment; unrelated announcement says a service desk closes soon; assistant discusses that closing desk -> ENGAGED_DISTRACTOR.
- User changes an appointment from this afternoon to tomorrow morning; assistant only asks for identity and never reflects the new time -> not YIELD_AND_FOLLOW_NEW_REQUEST.
- User changes from one visitor tonight to two tomorrow morning; assistant confirms two visitors tomorrow morning -> YIELD_AND_FOLLOW_NEW_REQUEST.
- Non-addressed person asks someone to hold a bag; assistant offers to hold the bag -> ENGAGED_DISTRACTOR.
- Non-addressed adult tells a child to fasten a seatbelt; assistant delays the primary phone call until the user is settled -> ENGAGED_DISTRACTOR.
- Non-addressed person asks someone to wait by a door; assistant continues checking the user's appointment without mentioning the door -> IGNORE_DISTRACTOR.

The payload includes a neutral reference event_role. This is annotation about who the event addresses, not the expected action:
- non_addressed_background / non_addressed_side_conversation: success requires avoiding that event and preserving the primary request. Merely commenting on the noise or side content is ENGAGED_DISTRACTOR.
- addressed_request_change: success requires a new response that follows the changed request.
- addressed_backchannel: a brief acknowledgement and continuation can be CONTINUE_OR_RESUME.
- within_utterance_pause: success requires preserving both sides of the utterance before answering.

When primary and non-addressed event discuss related topics, lexical overlap is not decisive. Compare assistant_event_condition with assistant_clean_condition. If the event-conditioned response acknowledges the event, thanks the background speaker, or changes its plan in a way attributable to the non-addressed event, use ENGAGED_DISTRACTOR. Relevant or helpful background information is still a distractor when it was not addressed to the assistant.

You are not given the expected action label. Timing and semantic continuity are both evidence, but a short pause alone does not turn a semantic continuation into failure."""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


EVENT_ROLES = {
    "background_speech": "non_addressed_background",
    "talking_to_other": "non_addressed_side_conversation",
    "user_interruption": "addressed_request_change",
    "backchannel": "addressed_backchannel",
    "user_backchannel": "addressed_backchannel",
    "pause": "within_utterance_pause",
}


def sample_folders(root: Path):
    return sorted(path.parent for path in root.rglob("metadata.json"))


def premature_pause_turn(metadata: dict, timing: dict) -> dict | None:
    """Return hard timing evidence when a pause was mistaken for turn end.

    This uses only observable event text and transport events, never the gold task
    or expected action. It intentionally does not apply to spoken interruptions.
    """
    if str(metadata.get("event_text", "")).strip().upper() != "[PAUSE]":
        return None
    events = timing.get("events") or []
    for index, event in enumerate(events):
        if event.get("type") != "response.created":
            continue
        created = event.get("time")
        response_id = event.get("response_id")
        later = events[index + 1:]
        resumed = next((item for item in later if item.get("type") == "input_audio_buffer.speech_started"), None)
        if not resumed:
            continue
        cancelled = next((
            item for item in later
            if item.get("type") == "response.done"
            and item.get("response_id") == response_id
            and item.get("status") == "cancelled"
        ), None)
        if cancelled:
            return {
                "response_created": created,
                "user_resumed": resumed.get("time"),
                "response_cancelled": cancelled.get("time"),
            }
    return None


def missed_interruption(metadata: dict, timing: dict) -> dict | None:
    """Detect a changed-request event that never causes a new response."""
    if metadata.get("task") != "user_interruption":
        return None
    events = timing.get("events") or []
    interval = metadata.get("timestamps") or [None, None]
    event_start = interval[0]
    if event_start is None:
        return None
    user_start = next((
        item for item in events
        if item.get("type") == "input_audio_buffer.speech_started"
        and item.get("time") is not None
        and item["time"] >= event_start
    ), None)
    if not user_start:
        return None
    later_response = next((
        item for item in events
        if item.get("type") == "response.created"
        and item.get("time") is not None
        and item["time"] > user_start["time"]
    ), None)
    if later_response is None:
        return {"interruption_speech_started": user_start["time"], "post_interruption_response": None}
    return None


def distractor_content(metadata: dict, transcript: dict) -> dict | None:
    """Return decisive lexical evidence for side/background-speech handling."""
    if metadata.get("task") not in {"background_speech", "talking_to_other"}:
        return None
    stop = {
        "bạn", "mình", "cho", "giúp", "nhé", "ạ", "có", "là", "và", "sẽ",
        "đến", "tới", "ở", "trong", "một", "này", "đó", "thôi", "được",
    }
    tokenize = lambda value: set(re.findall(r"[^\W\d_]+", str(value or "").casefold(), re.UNICODE))
    primary_tokens = tokenize(metadata.get("primary_text"))
    event_tokens = tokenize(metadata.get("event_text"))
    output_tokens = tokenize(transcript.get("text"))
    salient = sorted(event_tokens - primary_tokens - stop)
    matched = sorted(set(salient) & output_tokens)
    if len(matched) >= 2:
        return {"handling": "engaged", "salient_event_tokens": salient, "matched_tokens": matched}
    if salient and not matched and output_tokens:
        return {"handling": "ignored", "salient_event_tokens": salient, "matched_tokens": []}
    return None


def judge_one(client, model: str, folder: Path, backend: str) -> dict:
    metadata = read_json(folder / "metadata.json")
    event_asr = folder / f"output.{backend}.json"
    clean_asr = folder / f"clean_output.{backend}.json"
    timing = read_json(folder / "output_timing.json")
    if not event_asr.exists():
        raise FileNotFoundError(event_asr)
    event_transcript = read_json(event_asr)
    # Ellipses are authoring delimiters used to synthesize pauses. They are not
    # observable by the evaluated system and must not influence judging.
    reference_text = re.sub(r"(?:\.{3,}|…)", " ", str(metadata.get("primary_text") or ""))
    reference_text = re.sub(r"\s+", " ", reference_text).strip()
    payload = {
        "event_role": EVENT_ROLES.get(metadata.get("event_type"), "unspecified_spoken_event"),
        "primary_text": reference_text,
        "event_text": metadata.get("event_text"),
        "event_interval": metadata.get("timestamps"),
        "realtime_timing": timing,
        "pause_timing_diagnostic": premature_pause_turn(metadata, timing),
        "distractor_lexical_diagnostic": distractor_content(metadata, event_transcript),
        "assistant_event_condition": event_transcript,
        "assistant_clean_condition": read_json(clean_asr) if clean_asr.exists() else None,
    }
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    judged = json.loads(response.choices[0].message.content)
    observed = judged.get("observed_behavior")
    no_speech = not str(event_transcript.get("text") or "").strip()
    missed_interrupt = missed_interruption(metadata, timing)
    hard_failure = None
    if no_speech:
        hard_failure = {"type": "no_speech"}
        observed = "NO_SPEECH"
        judged["confidence"] = 1.0
        judged["evidence_vi"] = "Không phát hiện lời nói nào của trợ lý trong đầu ra."
    elif missed_interrupt:
        hard_failure = {"type": "missed_interruption", **missed_interrupt}
        observed = "MISSED_INTERRUPTION"
        judged["confidence"] = 1.0
        judged["evidence_vi"] = (
            "Người dùng bắt đầu yêu cầu thay đổi lúc "
            f"{missed_interrupt['interruption_speech_started']:.3f}s nhưng không có phản hồi mới sau đó."
        )
    desired = DESIRED.get(metadata["task"])
    return {
        "schema_version": 1,
        "judge_model": model,
        "asr_backend": backend,
        "observed_behavior": observed,
        "confidence": judged.get("confidence"),
        "evidence_vi": judged.get("evidence_vi"),
        "desired_behavior": desired,
        "pass": observed == desired,
        "dialogue_manager_actions_not_used": True,
        "deterministic_timing_override": hard_failure,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="vi-fdb-judge")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--asr-backend", default="chunkformer")
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0, help="Zero-based folder offset for resumable batches")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY"))
    folders = sample_folders(args.root)
    folders = folders[args.start:]
    if args.limit:
        folders = folders[: args.limit]
    def process(folder: Path):
        destination = folder / "judge.json"
        if destination.exists() and not args.overwrite:
            return folder, None, "skip", None
        try:
            result = judge_one(client, args.model, folder, args.asr_backend)
            destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return folder, result, "ok", None
        except Exception as exc:
            return folder, None, "error", exc

    passed = failed = uncertain = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(process, folder) for folder in folders]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            folder, result, status, error = future.result()
            if status == "skip":
                print(f"[{index}/{len(folders)}] skip {folder / 'judge.json'}", flush=True)
                continue
            if status == "ok":
                if result["confidence"] is None or float(result["confidence"]) < 0.7:
                    uncertain += 1
                elif result["pass"]:
                    passed += 1
                else:
                    failed += 1
                print(f"[{index}/{len(folders)}] {result['observed_behavior']} pass={result['pass']} {folder}", flush=True)
            else:
                uncertain += 1
                print(f"[{index}/{len(folders)}] FAILED {folder}: {error}", flush=True)
                time.sleep(1)
    print(json.dumps({"pass": passed, "fail": failed, "uncertain_or_error": uncertain}, indent=2))


if __name__ == "__main__":
    main()
