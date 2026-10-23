#!/usr/bin/env python3
"""Render a self-contained HTML vibe-check page from generated conversations (text-only).

Usage:
    python fd_data_creation/render_vibe_text.py \
        --input outputs/full_duplex_training/production_text_v1/qc/accepted.jsonl \
        --output outputs/full_duplex_training/production_text_v1/vibe_text.html \
        --sample 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

EVENT_LABELS = {
    "backchannel": ("Backchannel", "#7c3aed"),
    "real_interruption": ("Real interruption", "#dc2626"),
    "ambiguous_interruption": ("Ambiguous interruption", "#d97706"),
}

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       background: #f1f5f9; color: #1e293b; display: flex; height: 100vh; overflow: hidden; }
#sidebar { width: 280px; min-width: 200px; background: #1e293b; color: #cbd5e1;
           overflow-y: auto; padding: 12px 0; flex-shrink: 0; }
#sidebar h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
              color: #64748b; padding: 0 16px 8px; }
#sidebar .conv-item { padding: 8px 16px; cursor: pointer; border-left: 3px solid transparent;
                      font-size: 13px; line-height: 1.4; transition: background .1s; }
#sidebar .conv-item:hover { background: #334155; }
#sidebar .conv-item.active { border-left-color: #6366f1; background: #273549; color: #f8fafc; }
#sidebar .conv-item .cid { font-size: 10px; color: #64748b; font-family: monospace; }
#sidebar .conv-item .ctopic { font-weight: 500; }
#sidebar .conv-item .cstyle { font-size: 11px; color: #94a3b8; }
#main { flex: 1; overflow-y: auto; padding: 24px; }
#stats { background: #fff; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;
         display: flex; gap: 24px; flex-wrap: wrap; box-shadow: 0 1px 3px rgba(0,0,0,.07); }
#stats .stat { }
#stats .stat-val { font-size: 22px; font-weight: 700; color: #6366f1; }
#stats .stat-lbl { font-size: 11px; color: #64748b; }
.conv-card { background: #fff; border-radius: 10px; padding: 20px 24px;
             box-shadow: 0 1px 3px rgba(0,0,0,.07); }
.conv-header { margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #e2e8f0; }
.conv-header h3 { font-size: 14px; font-weight: 600; color: #334155; }
.conv-header .meta { font-size: 12px; color: #64748b; margin-top: 4px; }
.round { margin-bottom: 4px; }
.round.has-event { background: #faf5ff; border-radius: 8px; padding: 10px 12px;
                   margin-bottom: 8px; border-left: 3px solid #7c3aed; }
.round.event-interruption { background: #fff5f5; border-left-color: #dc2626; }
.round.event-ambiguous { background: #fffbeb; border-left-color: #d97706; }
.bubble { display: flex; gap: 8px; margin-bottom: 6px; align-items: flex-start; }
.bubble.user { flex-direction: row; }
.bubble.agent { flex-direction: row-reverse; }
.avatar { width: 26px; height: 26px; border-radius: 50%; display: flex;
          align-items: center; justify-content: center; font-size: 10px;
          font-weight: 700; flex-shrink: 0; }
.avatar.u { background: #dbeafe; color: #1d4ed8; }
.avatar.a { background: #dcfce7; color: #166534; }
.text { max-width: 75%; padding: 8px 12px; border-radius: 14px; font-size: 13px;
        line-height: 1.5; }
.bubble.user .text { background: #eff6ff; color: #1e3a8a; border-radius: 14px 14px 14px 4px; }
.bubble.agent .text { background: #f0fdf4; color: #14532d; border-radius: 14px 14px 4px 14px; }
.event-badge { display: inline-flex; align-items: center; gap: 6px; margin: 6px 0 4px;
               font-size: 11px; font-weight: 600; padding: 3px 9px; border-radius: 12px; }
.event-block { margin-top: 4px; }
.event-user { background: #fde8ff; color: #6b21a8; border-radius: 14px 14px 14px 4px;
              padding: 6px 12px; font-size: 13px; line-height: 1.5;
              display: inline-block; max-width: 75%; }
.event-agent-after { background: #fef3c7; color: #78350f; border-radius: 14px 14px 4px 14px;
                     padding: 6px 12px; font-size: 13px; line-height: 1.5;
                     display: inline-block; max-width: 75%; margin-top: 4px;
                     float: right; clear: both; }
.event-meta { font-size: 11px; color: #94a3b8; margin-top: 6px; clear: both; }
.overlap-bar { display: inline-block; width: 48px; height: 6px; background: #e2e8f0;
               border-radius: 3px; vertical-align: middle; position: relative; margin: 0 4px; }
.overlap-fill { position: absolute; left: 0; top: 0; height: 100%; border-radius: 3px; }
"""

_JS = """
const cards = document.querySelectorAll('.conv-card');
const items = document.querySelectorAll('.conv-item');

function show(idx) {
  cards.forEach((c, i) => c.style.display = i === idx ? 'block' : 'none');
  items.forEach((it, i) => it.classList.toggle('active', i === idx));
  items[idx].scrollIntoView({block: 'nearest'});
}

items.forEach((it, i) => it.addEventListener('click', () => show(i)));
if (cards.length) show(0);

document.addEventListener('keydown', e => {
  const cur = [...items].findIndex(it => it.classList.contains('active'));
  if (e.key === 'ArrowDown' && cur < items.length - 1) show(cur + 1);
  if (e.key === 'ArrowUp' && cur > 0) show(cur - 1);
});
"""


def overlap_bar_html(frac: float, color: str) -> str:
    pct = int(frac * 100)
    return (f'<span class="overlap-bar"><span class="overlap-fill" '
            f'style="width:{pct}%;background:{color}"></span></span>')


def render_round(i: int, turn: dict) -> str:
    event = turn.get("event")
    extra_cls = ""
    if event:
        t = event.get("type", "")
        if t == "real_interruption":
            extra_cls = " event-interruption"
        elif t == "ambiguous_interruption":
            extra_cls = " event-ambiguous"
        else:
            extra_cls = ""

    parts = [f'<div class="round{"  has-event" + extra_cls if event else ""}">']
    parts.append(f'<div class="bubble user"><div class="avatar u">U</div>'
                 f'<div class="text">{_esc(turn["user"])}</div></div>')
    parts.append(f'<div class="bubble agent"><div class="avatar a">A</div>'
                 f'<div class="text">{_esc(turn["agent"])}</div></div>')

    if event:
        label, color = EVENT_LABELS.get(event["type"], (event["type"], "#6b7280"))
        frac = event.get("overlap_fraction", 0.5)
        parts.append(f'<div class="event-block">')
        parts.append(f'<span class="event-badge" style="background:{color}22;color:{color}">'
                     f'⚡ {label} @ {int(frac*100)}%'
                     f'{overlap_bar_html(frac, color)}</span>')
        parts.append(f'<div><span class="event-user">{_esc(event["text"])}</span></div>')
        if event.get("agent_after"):
            parts.append(f'<div><span class="event-agent-after">{_esc(event["agent_after"])}</span></div>')
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def render_conversation(conv: dict, idx: int) -> tuple[str, str]:
    cid = conv.get("conversation_id", f"conv_{idx}")
    topic = conv.get("topic", "")
    style = conv.get("style", "")
    rounds = conv.get("rounds", [])
    events = [r["event"] for r in rounds if r.get("event")]
    event_types = [e["type"] for e in events]

    type_badges = "".join(
        f'<span class="event-badge" style="background:{EVENT_LABELS.get(t,("",""+"#6b7280"))[1]}22;'
        f'color:{EVENT_LABELS.get(t, (t,"#6b7280"))[1]};font-size:10px">'
        f'{EVENT_LABELS.get(t,(t,""))[0]}</span>'
        for t in event_types
    )

    card = (f'<div class="conv-card" id="card-{idx}" style="display:none">'
            f'<div class="conv-header"><h3>{_esc(topic)} '
            f'<code style="font-size:11px;color:#64748b">{_esc(cid)}</code></h3>'
            f'<div class="meta">Style: {_esc(style)} · {len(rounds)} rounds · {type_badges}</div></div>')
    for i, turn in enumerate(rounds):
        card += render_round(i, turn)
    card += "</div>"

    item = (f'<div class="conv-item" id="item-{idx}">'
            f'<div class="ctopic">{_esc(topic)}</div>'
            f'<div class="cstyle">{_esc(style)}</div>'
            f'<div class="cid">{_esc(cid)}</div></div>')
    return card, item


def build_stats(convs: list[dict]) -> str:
    from collections import Counter
    event_counts: Counter = Counter()
    total_rounds = 0
    for c in convs:
        for r in c.get("rounds", []):
            total_rounds += 1
            if r.get("event"):
                event_counts[r["event"]["type"]] += 1
    stats = [
        (str(len(convs)), "conversations"),
        (str(total_rounds), "total rounds"),
        (str(event_counts.get("backchannel", 0)), "backchannels"),
        (str(event_counts.get("real_interruption", 0)), "real interruptions"),
        (str(event_counts.get("ambiguous_interruption", 0)), "ambiguous interruptions"),
    ]
    html = '<div id="stats">'
    for val, lbl in stats:
        html += f'<div class="stat"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>'
    html += "</div>"
    return html


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="generated.jsonl or accepted.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample", type=int, default=0, help="random sample size (0 = all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    convs = [json.loads(l) for l in args.input.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.sample and args.sample < len(convs):
        rng = random.Random(args.seed)
        convs = rng.sample(convs, args.sample)

    cards_html, items_html = [], []
    for i, conv in enumerate(convs):
        card, item = render_conversation(conv, i)
        cards_html.append(card)
        items_html.append(item)

    stats_html = build_stats(convs)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Vibe check — {len(convs)} conversations</title>
<style>{_CSS}</style>
</head>
<body>
<div id="sidebar">
  <h2>{len(convs)} conversations</h2>
  {"".join(items_html)}
</div>
<div id="main">
  {stats_html}
  {"".join(cards_html)}
</div>
<script>{_JS}</script>
</body>
</html>"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {len(convs)} conversations → {args.output}")


if __name__ == "__main__":
    main()
