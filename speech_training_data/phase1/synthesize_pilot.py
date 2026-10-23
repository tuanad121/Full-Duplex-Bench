#!/usr/bin/env python3
"""Synthesize and timeline-render the Phase-1 full-duplex pilot."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from vieneu import Vieneu


SPEAKERS = {
    "user": "src/vieneu/assets/samples/Ly (nữ miền Bắc).wav",
    "agent": "src/vieneu/assets/samples/Vĩnh (nam miền Nam).wav",
}
REACTION_SECONDS = 0.32
TURN_GAP_SECONDS = 0.42
POST_EVENT_GAP_SECONDS = 0.48
TAIL_SECONDS = 0.5


def put(track: np.ndarray, audio: np.ndarray, start: int) -> None:
    track[start:start + len(audio)] += audio


def synth(tts: Vieneu, text: str, ref: Path) -> tuple[np.ndarray, int]:
    audio = np.asarray(tts.infer(text, ref_audio=str(ref), temperature=0.8, show_progress=False), dtype=np.float32)
    return audio, int(tts.sample_rate)


def fade_cut(audio: np.ndarray, stop: int, fade_samples: int) -> np.ndarray:
    stop = min(max(stop, 1), len(audio))
    out = audio[:stop].copy()
    n = min(fade_samples, len(out))
    if n:
        out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def render_sample(tts: Vieneu, sample: dict, root: Path, refs: dict[str, Path]) -> dict:
    sample_dir = root / sample["id"]
    source_dir = sample_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    initial, sr = synth(tts, sample["user_initial"], refs["user"])
    ongoing, sr2 = synth(tts, sample["agent_ongoing"], refs["agent"])
    if sr != sr2:
        raise ValueError("sample-rate mismatch")
    sf.write(source_dir / "user_initial.wav", initial, sr)
    sf.write(source_dir / "agent_ongoing_full.wav", ongoing, sr)

    initial_start = 0
    agent_start = len(initial) + round(TURN_GAP_SECONDS * sr)
    event = np.zeros(0, dtype=np.float32)
    event_start = None
    event_end = None
    agent_stop = None
    agent_after = np.zeros(0, dtype=np.float32)

    if sample["user_event"]:
        event, event_sr = synth(tts, sample["user_event"], refs["user"])
        if event_sr != sr:
            raise ValueError("sample-rate mismatch")
        sf.write(source_dir / "user_event.wav", event, sr)
        overlap_fraction = 0.46 if sample["case"] == "backchannel" else 0.42
        event_start = agent_start + round(len(ongoing) * overlap_fraction)
        event_end = event_start + len(event)

    rendered_agent = ongoing
    if sample["case"] in {"real_interruption", "ambiguous_interruption"}:
        agent_stop = event_start + round(REACTION_SECONDS * sr)
        rendered_agent = fade_cut(ongoing, agent_stop - agent_start, round(0.05 * sr))
        agent_after, after_sr = synth(tts, sample["agent_after_event"], refs["agent"])
        if after_sr != sr:
            raise ValueError("sample-rate mismatch")
        sf.write(source_dir / "agent_after_event.wav", agent_after, sr)

    after_start = None
    if len(agent_after):
        after_start = event_end + round(POST_EVENT_GAP_SECONDS * sr)
    end_candidates = [agent_start + len(rendered_agent), len(initial)]
    if event_end is not None:
        end_candidates.append(event_end)
    if after_start is not None:
        end_candidates.append(after_start + len(agent_after))
    total = max(end_candidates) + round(TAIL_SECONDS * sr)
    user_track = np.zeros(total, dtype=np.float32)
    agent_track = np.zeros(total, dtype=np.float32)
    put(user_track, initial, initial_start)
    put(agent_track, rendered_agent, agent_start)
    if event_start is not None:
        put(user_track, event, event_start)
    if after_start is not None:
        put(agent_track, agent_after, after_start)
    peak = max(float(np.max(np.abs(user_track))), float(np.max(np.abs(agent_track))), 1e-6)
    scale = min(0.94 / peak, 1.0)
    user_track *= scale
    agent_track *= scale
    mixture = np.clip(user_track + agent_track, -1.0, 1.0)
    sf.write(sample_dir / "user.wav", user_track, sr)
    sf.write(sample_dir / "agent.wav", agent_track, sr)
    sf.write(sample_dir / "mixture.wav", mixture, sr)

    expected = {
        "normal_turn": "start_speaking",
        "real_interruption": "stop_and_listen_then_reply",
        "backchannel": "continue_speaking",
        "ambiguous_interruption": "stop_and_listen_then_reply",
    }[sample["case"]]
    metadata = {
        **sample,
        "sample_rate": sr,
        "duration_sec": round(total / sr, 3),
        "expected_action": expected,
        "timeline": {
            "user_initial": [0.0, round(len(initial) / sr, 3)],
            "agent_ongoing": [round(agent_start / sr, 3), round((agent_start + len(rendered_agent)) / sr, 3)],
            "user_event": None if event_start is None else [round(event_start / sr, 3), round(event_end / sr, 3)],
            "agent_stop_target_sec": None if agent_stop is None else round(agent_stop / sr, 3),
            "agent_after_event": None if after_start is None else [round(after_start / sr, 3), round((after_start + len(agent_after)) / sr, 3)],
        },
        "files": {"user": "user.wav", "agent": "agent.wav", "mixture": "mixture.wav"},
    }
    (sample_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def build_html(rows: list[dict], root: Path) -> None:
    labels = {
        "normal_turn": "Ordinary turn",
        "real_interruption": "Real interruption",
        "backchannel": "Backchannel",
        "ambiguous_interruption": "Ambiguous onset → interruption",
    }
    cards = []
    for row in rows:
        event = row["user_event"] or "—"
        after = row["agent_after_event"] or "—"
        base = html.escape(row["id"])
        cards.append(f'''<article data-id="{base}"><header><span class="case">{html.escape(labels[row["case"]])}</span><span class="action">{html.escape(row["expected_action"])}</span></header><h2>{html.escape(row["topic"])}</h2><dl><dt>User</dt><dd>{html.escape(row["user_initial"])}</dd><dt>Agent ongoing</dt><dd>{html.escape(row["agent_ongoing"])}</dd><dt>User event</dt><dd>{html.escape(event)}</dd><dt>Agent after event</dt><dd>{html.escape(after)}</dd></dl><div class="players"><label>Mixture<audio controls preload="metadata" src="{base}/mixture.wav"></audio></label><label>User stem<audio controls preload="metadata" src="{base}/user.wav"></audio></label><label>Agent target<audio controls preload="metadata" src="{base}/agent.wav"></audio></label></div><div class="review"><button data-v="pass">Good</button><button data-v="fail">Problem</button><input placeholder="Optional listening note"></div></article>''')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase 1 full-duplex pilot</title><style>:root{{font-family:Inter,system-ui,sans-serif;color:#172033;background:#f3f6fb}}*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1100px;margin:auto;padding:32px 20px 60px}}h1{{margin-bottom:4px}}.lead{{color:#64748b;max-width:820px}}article{{background:#fff;border:1px solid #dbe3ef;border-radius:16px;padding:20px;margin:18px 0;box-shadow:0 4px 16px #1e293b0b}}article[data-review=pass]{{border:2px solid #22c55e}}article[data-review=fail]{{border:2px solid #ef4444}}header{{display:flex;justify-content:space-between;gap:12px}}.case,.action{{padding:5px 10px;border-radius:999px;background:#e8eefc;font-weight:700}}.action{{background:#ecfdf5;color:#166534}}h2{{font-size:20px}}dl{{display:grid;grid-template-columns:140px 1fr;gap:7px 12px}}dt{{font-weight:700;color:#475569}}dd{{margin:0}}.players{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}}label{{font-size:13px;font-weight:700;color:#475569}}audio{{width:100%;display:block;margin-top:5px}}.review{{display:flex;gap:8px;margin-top:16px}}button,input{{font:inherit;padding:8px 12px;border:1px solid #bcc8d8;border-radius:8px;background:white}}button{{cursor:pointer}}input{{flex:1}}@media(max-width:760px){{.players{{grid-template-columns:1fr}}dl{{grid-template-columns:1fr}}}}</style></head><body><main><h1>Phase 1 full-duplex speech pilot</h1><p class="lead">Four clean-speech cases. Listen to the mixture first, then inspect isolated stems. Confirm naturalness, overlap timing, and whether the agent target demonstrates the expected behavior.</p>{''.join(cards)}<button id="export">Export review JSON</button><script>const key='phase1_fd_pilot_review_v1',saved=JSON.parse(localStorage.getItem(key)||'{{}}');for(const c of document.querySelectorAll('article')){{const r=saved[c.dataset.id];if(r){{c.dataset.review=r.verdict;c.querySelector('input').value=r.note||''}}}}document.addEventListener('click',e=>{{if(!e.target.dataset.v)return;const c=e.target.closest('article');c.dataset.review=e.target.dataset.v;saved[c.dataset.id]={{verdict:e.target.dataset.v,note:c.querySelector('input').value}};localStorage.setItem(key,JSON.stringify(saved))}});document.addEventListener('input',e=>{{if(!e.target.matches('input'))return;const c=e.target.closest('article');if(saved[c.dataset.id]){{saved[c.dataset.id].note=e.target.value;localStorage.setItem(key,JSON.stringify(saved))}}}});export.onclick=()=>{{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(saved,null,2)],{{type:'application/json'}}));a.download='phase1_pilot_review.json';a.click()}};</script></main></body></html>'''
    (root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vieneu-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    payload = json.loads(args.spec.read_text(encoding="utf-8"))
    refs = {name: args.vieneu_root / path for name, path in SPEAKERS.items()}
    missing = [str(path) for path in refs.values() if not path.exists()]
    if missing:
        raise SystemExit(f"missing voice references: {missing}; set --vieneu-root")
    args.output.mkdir(parents=True, exist_ok=True)
    tts = Vieneu(mode="v3turbo", backend="pytorch")
    rows = [render_sample(tts, sample, args.output, refs) for sample in payload["samples"]]
    (args.output / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    build_html(rows, args.output)
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
