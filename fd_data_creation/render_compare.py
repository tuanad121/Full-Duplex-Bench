#!/usr/bin/env python3
"""Side-by-side SA vs RP comparison HTML with optional audio.

Reads matched_pairs.jsonl (from generate.py) and an optional rendered audio
directory (from render_timelines.py) to produce a single self-contained HTML
file showing both strategies for each blueprint.

Usage (text only):
    python fd_data_creation/render_compare.py \
        --pairs   outputs/full_duplex_training/compare_8pairs/matched_pairs.jsonl \
        --output  outputs/full_duplex_training/compare_8pairs/compare.html

With audio (after render_timelines.py has run):
    python fd_data_creation/render_compare.py \
        --pairs   outputs/full_duplex_training/compare_8pairs/matched_pairs.jsonl \
        --audio   outputs/full_duplex_training/compare_8pairs/04_rendered \
        --output  outputs/full_duplex_training/compare_8pairs/compare.html
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

EVENT_COLOR = {
    "backchannel":          ("#7c3aed", "#f5f3ff"),
    "real_interruption":    ("#dc2626", "#fff5f5"),
    "ambiguous_interruption": ("#d97706", "#fffbeb"),
}

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f1f5f9; color: #1e293b; }
h1 { font-size: 18px; font-weight: 700; padding: 20px 24px 0; color: #0f172a; }
.subtitle { font-size: 12px; color: #64748b; padding: 4px 24px 16px; }
nav { background: #1e293b; padding: 10px 16px; display: flex; gap: 6px;
      flex-wrap: wrap; position: sticky; top: 0; z-index: 10; }
nav button { background: transparent; border: 1px solid #334155; color: #94a3b8;
             padding: 5px 12px; border-radius: 6px; font: 12px/1 inherit; cursor: pointer; }
nav button.active { background: #6366f1; border-color: #6366f1; color: #fff; }
.pair-card { display: none; padding: 0 24px 32px; }
.pair-card.visible { display: block; }
.blueprint { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
             padding: 14px 18px; margin-bottom: 16px; }
.blueprint h2 { font-size: 15px; font-weight: 600; margin-bottom: 6px; }
.blueprint .meta { font-size: 12px; color: #64748b; }
.blueprint .events-planned { margin-top: 8px; font-size: 12px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.col { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; overflow: hidden; }
.col-header { padding: 12px 16px; font-size: 13px; font-weight: 700; border-bottom: 1px solid #e2e8f0; }
.col-header.sa { background: #eff6ff; color: #1d4ed8; }
.col-header.rp { background: #f0fdf4; color: #166534; }
.col-header .model { font-size: 10px; font-weight: 400; color: #64748b; margin-top: 2px; }
audio { width: 100%; display: block; border-radius: 0; border-bottom: 1px solid #e2e8f0; }
.rounds { padding: 12px 16px; }
.round { margin-bottom: 10px; }
.round.has-event { border-radius: 8px; padding: 10px; margin-bottom: 12px; }
.bubble { display: flex; gap: 8px; margin-bottom: 5px; }
.bubble.user { flex-direction: row; }
.bubble.agent { flex-direction: row-reverse; }
.avatar { width: 22px; height: 22px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; font-size: 9px; font-weight: 700; flex-shrink: 0; }
.avatar.u { background: #dbeafe; color: #1d4ed8; }
.avatar.a { background: #dcfce7; color: #166534; }
.text { max-width: 85%; padding: 6px 10px; border-radius: 12px; font-size: 12px; line-height: 1.5; }
.bubble.user .text { background: #eff6ff; color: #1e3a8a; border-radius: 12px 12px 12px 3px; }
.bubble.agent .text { background: #f0fdf4; color: #14532d; border-radius: 12px 12px 3px 12px; }
.event-label { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px;
               display: inline-flex; align-items: center; gap: 4px; margin: 4px 0; }
.event-user-text { font-size: 12px; padding: 5px 10px; border-radius: 12px 12px 12px 3px;
                   display: inline-block; max-width: 85%; line-height: 1.5; }
.event-agent-after { font-size: 12px; padding: 5px 10px; border-radius: 12px 12px 3px 12px;
                     background: #fef3c7; color: #78350f;
                     display: inline-block; max-width: 85%; margin-top: 3px;
                     float: right; clear: both; line-height: 1.5; }
.event-meta { font-size: 10px; color: #94a3b8; clear: both; margin-top: 4px; }
.diff-badge { font-size: 10px; padding: 1px 6px; border-radius: 8px;
              background: #fef9c3; color: #713f12; margin-left: 6px; }
"""

_JS = """
const cards = document.querySelectorAll('.pair-card');
const btns  = document.querySelectorAll('nav button');

function show(i) {
  cards.forEach((c, j) => c.classList.toggle('visible', i === j));
  btns.forEach((b, j) => b.classList.toggle('active', i === j));
  window.scrollTo(0, 0);
}

btns.forEach((b, i) => b.addEventListener('click', () => show(i)));
if (cards.length) show(0);

document.addEventListener('keydown', e => {
  const cur = [...btns].findIndex(b => b.classList.contains('active'));
  if (e.key === 'ArrowRight' && cur < btns.length - 1) show(cur + 1);
  if (e.key === 'ArrowLeft'  && cur > 0)               show(cur - 1);
});
"""


def _e(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def render_round(turn: dict, idx: int) -> str:
    event = turn.get("event")
    bg = border = ""
    if event:
        ec, ebg = EVENT_COLOR.get(event["type"], ("#6b7280", "#f9fafb"))
        bg = f'background:{ebg};border-left:3px solid {ec};'
        border = ec

    parts = [f'<div class="round{"  has-event" if event else ""}" style="{bg}">']
    parts.append(f'<div class="bubble user"><div class="avatar u">U</div>'
                 f'<div class="text">{_e(turn["user"])}</div></div>')
    parts.append(f'<div class="bubble agent"><div class="avatar a">A</div>'
                 f'<div class="text">{_e(turn["agent"])}</div></div>')

    if event:
        ec, _ = EVENT_COLOR.get(event["type"], ("#6b7280", ""))
        label = event["type"].replace("_", " ")
        frac = event.get("overlap_fraction", 0.5)
        parts.append(
            f'<div><span class="event-label" style="background:{ec}22;color:{ec}">'
            f'⚡ {_e(label)} @ {int(frac*100)}%</span></div>'
        )
        parts.append(f'<div><span class="event-user-text" style="background:{ec}18;color:{ec}">'
                     f'{_e(event["text"])}</span></div>')
        if event.get("agent_after"):
            parts.append(f'<div><span class="event-agent-after">{_e(event["agent_after"])}</span></div>')
        parts.append('<div class="event-meta" style="clear:both"></div>')
    parts.append("</div>")
    return "".join(parts)


def render_conv(conv: dict, strategy: str, audio_path: "Path | None", provenance: dict | None, html_dir: "Path | None" = None) -> str:
    cls = "sa" if strategy == "single_author" else "rp"
    label = "Single Author" if strategy == "single_author" else "Role Play (Director + 2 Actors)"

    model_info = ""
    if provenance:
        if strategy == "single_author":
            m = provenance.get("single_author", {}).get("provider_model", "")
            model_info = f'<div class="model">Model: {_e(m)}</div>'
        else:
            calls = provenance.get("role_play_calls", [])
            models = {c.get("role"): c.get("provider_model") for c in calls if c.get("role")}
            parts = [f'{r}: {_e(m)}' for r, m in models.items() if m]
            model_info = f'<div class="model">{" · ".join(parts[:3])}</div>'

    html = [f'<div class="col">']
    html.append(f'<div class="col-header {cls}">{_e(label)}{model_info}</div>')
    if audio_path:
        rel = os.path.relpath(audio_path, html_dir) if html_dir else str(audio_path)
        html.append(f'<audio controls preload="metadata" src="{_e(rel)}"></audio>')
    html.append('<div class="rounds">')
    for i, turn in enumerate(conv.get("rounds", [])):
        html.append(render_round(turn, i))
    html.append("</div></div>")
    return "".join(html)


def render_pair(pair: dict, idx: int, audio_dir: Path | None, prov: dict | None, html_dir: Path | None = None) -> tuple[str, str]:
    blueprint = pair.get("blueprint", {})
    sa = pair["single_author"]
    rp = pair["role_play"]
    cid_sa = sa.get("conversation_id", "")
    cid_rp = rp.get("conversation_id", "")
    topic = blueprint.get("topic") or sa.get("topic", "")
    style = blueprint.get("style") or sa.get("style", "")
    bp_events = blueprint.get("events", [])
    event_types = [e.get("type", "") for e in bp_events]

    audio_sa = audio_rp = None
    if audio_dir:
        sa_mix = audio_dir / cid_sa / "mixture.wav"
        rp_mix = audio_dir / cid_rp / "mixture.wav"
        if sa_mix.exists(): audio_sa = sa_mix
        if rp_mix.exists(): audio_rp = rp_mix

    prov_sa = prov_rp = None
    if prov:
        prov_sa = {"single_author": prov.get("single_author")}
        prov_rp = {"role_play_calls": prov.get("role_play_calls", [])}

    rounds_sa = len(sa.get("rounds", []))
    rounds_rp = len(rp.get("rounds", []))
    diff_badge = (f'<span class="diff-badge">⚠ round count differs '
                  f'({rounds_sa} vs {rounds_rp})</span>') if rounds_sa != rounds_rp else ""

    event_pills = "".join(
        f'<span style="background:{EVENT_COLOR.get(t,("#6b7280",""))[0]}22;'
        f'color:{EVENT_COLOR.get(t,("#6b7280",""))[0]};'
        f'font-size:11px;padding:2px 7px;border-radius:8px;margin-right:4px">{_e(t)}</span>'
        for t in event_types
    )

    card = (
        f'<div class="pair-card" id="pair-{idx}">'
        f'<div class="blueprint">'
        f'<h2>{_e(topic)}{diff_badge}</h2>'
        f'<div class="meta">Style: {_e(style)} · '
        f'{rounds_sa} rounds (SA) / {rounds_rp} rounds (RP)</div>'
        f'<div class="events-planned">Planned events: {event_pills or "(none)"}</div>'
        f'</div>'
        f'<div class="cols">'
        + render_conv(sa, "single_author", audio_sa, prov_sa, html_dir)
        + render_conv(rp, "role_play",    audio_rp, prov_rp, html_dir)
        + '</div></div>'
    )
    btn = f'<button id="btn-{idx}">#{idx+1} {_e(topic[:28])}</button>'
    return card, btn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs",  type=Path, required=True, help="matched_pairs.jsonl")
    p.add_argument("--audio",  type=Path, default=None,  help="04_rendered/ directory (optional)")
    p.add_argument("--provenance", type=Path, default=None, help="provenance.jsonl (optional)")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    pairs = [json.loads(l) for l in args.pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
    prov_map: dict = {}
    if args.provenance and args.provenance.exists():
        for l in args.provenance.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                prov_map[r["pair_key"]] = r

    html_dir = args.output.resolve().parent
    cards_html, btns_html = [], []
    for i, pair in enumerate(pairs):
        prov = prov_map.get(pair.get("pair_key", ""))
        card, btn = render_pair(pair, i, args.audio, prov, html_dir)
        cards_html.append(card)
        btns_html.append(btn)

    audio_note = "with audio" if args.audio else "text only — rerun with --audio after synthesis"
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SA vs RP — {len(pairs)} pairs</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Single Author vs Role Play — {len(pairs)} matched pairs</h1>
<p class="subtitle">Same blueprint, two generation strategies · {audio_note} · ← → keys to navigate</p>
<nav>{"".join(btns_html)}</nav>
{"".join(cards_html)}
<script>{_JS}</script>
</body>
</html>"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs → {args.output}")


if __name__ == "__main__":
    main()
