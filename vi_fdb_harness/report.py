#!/usr/bin/env python3
"""Generate a synchronized, self-contained-index HTML review for a Vi-FDB run."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def rel(path: Path, parent: Path) -> str:
    return path.relative_to(parent).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--asr-backend", default="chunkformer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = (args.output or root / "review.html").resolve()
    human_reviews = read_json(root / "human_review_overrides.json", {}).get("decisions", {})
    cards = []
    summary = {"samples": 0, "judged": 0, "pass": 0, "fail": 0, "uncertain": 0}
    for metadata_path in sorted(root.rglob("metadata.json")):
        folder = metadata_path.parent
        event_output = folder / "output.wav"
        if not event_output.exists():
            continue
        metadata = read_json(metadata_path, {})
        timing = read_json(folder / "output_timing.json", {})
        asr = read_json(folder / f"output.{args.asr_backend}.json", {})
        judge = read_json(folder / "judge.json")
        sample_key = folder.relative_to(root).as_posix()
        human_review = human_reviews.get(sample_key)
        effective_judge = ({**(judge or {}), **human_review, "adjudication": "human"}
                           if human_review else judge)
        summary["samples"] += 1
        if effective_judge:
            summary["judged"] += 1
            if not human_review and effective_judge.get("confidence", 0) < 0.7:
                summary["uncertain"] += 1
            elif effective_judge.get("pass"):
                summary["pass"] += 1
            else:
                summary["fail"] += 1
        card_id = f"c{summary['samples']}"
        start, end = metadata.get("timestamps", [0, 0])
        if not (folder / f"output.{args.asr_backend}.json").exists():
            transcript = "ASR pending"
        elif not str(asr.get("text") or "").strip():
            transcript = "[NO ASSISTANT SPEECH DETECTED]"
        else:
            transcript = asr["text"]
        verdict = "Judge pending" if not effective_judge else f"{effective_judge.get('observed_behavior')} · {'PASS' if effective_judge.get('pass') else 'REVIEW'}{' · HUMAN' if human_review else ''}"
        cards.append(f"""
<article class="card" data-task="{html.escape(metadata.get('task',''))}">
  <div class="top"><div><b>{html.escape(metadata.get('task',''))}</b> · {html.escape(metadata.get('sample_id',''))}</div><span>{html.escape(verdict)}</span></div>
  <p><b>Primary:</b> {html.escape(metadata.get('primary_text',''))}</p>
  <p><b>Event:</b> {html.escape(metadata.get('event_text',''))} <small>({start:.2f}–{end:.2f}s)</small></p>
  <div class="players">
    <label>User/event input<audio id="{card_id}i" controls preload="none" src="{rel(folder/'input.wav', output.parent)}"></audio></label>
    <label>Aligned assistant output<audio id="{card_id}o" controls preload="none" src="{rel(event_output, output.parent)}"></audio></label>
  </div>
  <button onclick="playPair('{card_id}')">Play synchronized</button>
  <button onclick="seekPair('{card_id}',{start})">Jump to event</button>
  <p><b>{html.escape(args.asr_backend)}:</b> {html.escape(transcript)}</p>
  <details><summary>Timing and judge evidence</summary><pre>{html.escape(json.dumps({'timing':timing,'judge':judge,'human_review':human_review},ensure_ascii=False,indent=2))}</pre></details>
</article>""")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Vietnamese FDB run review</title>
<style>body{{font:15px/1.45 system-ui;margin:0;background:#eef2f7;color:#172033}}main{{max-width:1120px;margin:30px auto;padding:0 18px}}header,.card{{background:white;border:1px solid #dce2ec;border-radius:14px;padding:18px;margin:14px 0}}h1{{margin:0}}.top{{display:flex;justify-content:space-between;gap:12px}}.top span{{background:#edf2ff;padding:4px 8px;border-radius:20px}}.players{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}audio{{display:block;width:100%;margin:6px 0 10px}}button{{padding:7px 10px;margin-right:7px}}small{{color:#667085}}pre{{white-space:pre-wrap;background:#f6f7f9;padding:12px;max-height:280px;overflow:auto}}@media(max-width:720px){{.players{{grid-template-columns:1fr}}}}</style></head><body><main>
<header><h1>Vietnamese FDB v1.0 run</h1><p>{html.escape(json.dumps(summary))}</p></header>
{''.join(cards)}
</main><script>
function sync(id){{const i=document.getElementById(id+'i'),o=document.getElementById(id+'o');if(Math.abs(i.currentTime-o.currentTime)>.12)o.currentTime=i.currentTime;}}
function playPair(id){{const i=document.getElementById(id+'i'),o=document.getElementById(id+'o');o.currentTime=i.currentTime;i.ontimeupdate=()=>sync(id);i.onpause=()=>o.pause();i.onended=()=>o.pause();Promise.allSettled([i.play(),o.play()]);}}
function seekPair(id,t){{const i=document.getElementById(id+'i'),o=document.getElementById(id+'o');i.currentTime=t;o.currentTime=t;playPair(id);}}
</script></body></html>""", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
