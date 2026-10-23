#!/usr/bin/env python3
"""Render an optional, non-gating HTML vibe-check for accepted audio."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

from schema import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        conversation = row["conversation"]
        cid = conversation["conversation_id"]
        event_list = "".join(f'<li><b>{html.escape(event["type"])}</b> at {event["start_sec"]:.2f}s → {html.escape(event["expected_action"])}</li>' for event in row["events"])
        turns = "".join(f'<div><b>User:</b> {html.escape(turn["user"])}<br><b>Agent:</b> {html.escape(turn["agent"])}</div>' for turn in conversation["rounds"])
        relative = os.path.relpath(Path(row["files"]["mixture"]), args.output.parent)
        cards.append(f'''<article data-id="{cid}"><header><h2>{html.escape(cid)} · {html.escape(conversation["topic"])}</h2><span>{row["duration_sec"]:.1f}s</span></header><audio controls preload="metadata" src="{html.escape(relative)}"></audio><ul>{event_list}</ul><details><summary>Transcript</summary>{turns}</details><div class="review"><button data-v="pass">Good</button><button data-v="fail">Problem</button><input placeholder="Optional note"></div></article>''')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Full-duplex audio vibe-check</title><style>body{{max-width:1100px;margin:35px auto;padding:0 18px;background:#f3f6fb;color:#172033;font:16px/1.5 system-ui}}article{{background:#fff;border:1px solid #dbe3ef;border-radius:15px;padding:18px;margin:15px 0}}article[data-review=pass]{{border:2px solid #22c55e}}article[data-review=fail]{{border:2px solid #ef4444}}header,.review{{display:flex;align-items:center;gap:10px;justify-content:space-between}}h2{{font-size:19px}}audio{{width:100%}}details div{{padding:6px;border-top:1px solid #edf1f6}}button,input{{font:inherit;padding:8px 12px;border:1px solid #bdc9d8;border-radius:8px;background:white}}input{{flex:1}}</style></head><body><h1>Optional full-duplex audio vibe-check</h1><p>This listening review is unofficial and does not gate production acceptance. Use it for pilots, spot checks, and debugging.</p>{''.join(cards)}<button id="export">Export reviews</button><script>const key='vi_fd_audio_vibe_v1',saved=JSON.parse(localStorage.getItem(key)||'{{}}');for(const c of document.querySelectorAll('article')){{const r=saved[c.dataset.id];if(r){{c.dataset.review=r.verdict;c.querySelector('input').value=r.note||''}}}}document.addEventListener('click',e=>{{if(!e.target.dataset.v)return;const c=e.target.closest('article');c.dataset.review=e.target.dataset.v;saved[c.dataset.id]={{verdict:e.target.dataset.v,note:c.querySelector('input').value}};localStorage.setItem(key,JSON.stringify(saved))}});document.addEventListener('input',e=>{{const c=e.target.closest('article');if(c&&saved[c.dataset.id]){{saved[c.dataset.id].note=e.target.value;localStorage.setItem(key,JSON.stringify(saved))}}}});export.onclick=()=>{{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(saved,null,2)],{{type:'application/json'}}));a.download='audio_vibe_reviews.json';a.click()}};</script></body></html>'''
    args.output.write_text(page, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
