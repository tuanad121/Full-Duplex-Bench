# Full-duplex training data pipeline — proven commands

All commands run from repo root. Load the API key first:

```bash
export OPENROUTER_API_KEY="$(tr -d '\r\n' < 'docs/open router key.txt')"
```

Use `uv run --project fd_data_creation` for every Python step (the venv lives in
`fd_data_creation/.venv`). The pipeline is resumable: every script skips items
that already have output files.

---

## Step 1 — Generate conversations

```bash
/Users/tuandinh/.local/bin/uv run --project fd_data_creation \
  python fd_data_creation/generate_batch.py \
  --config fd_data_creation/config.production.json \
  --output outputs/full_duplex_training/production_text_v1 \
  --count 1000 \
  --start 1001 \
  --jobs 8
```

Outputs in `outputs/full_duplex_training/production_text_v1/`:
- `items/NNNNNN.json` — one file per conversation (resumable unit)
- `generated.jsonl` — all accepted conversations
- `provenance.jsonl` — model / token / cost per conversation
- `failures.jsonl` — failed conversations with error messages
- `summary.json` — counts

To continue an interrupted run, rerun the same command; completed items are skipped.
To start a second batch (different index range), change `--start` and `--output`.

---

## Step 2 — Review conversations

```bash
/Users/tuandinh/.local/bin/uv run --project fd_data_creation \
  python fd_data_creation/review_batch.py \
  --config fd_data_creation/config.production.json \
  --input  outputs/full_duplex_training/production_text_v1/generated.jsonl \
  --provenance outputs/full_duplex_training/production_text_v1/provenance.jsonl \
  --output outputs/full_duplex_training/production_text_v1/qc \
  --jobs 4
```

Outputs in `.../qc/`:
- `items/<conversation_id>.json` — per-item review (resumable unit)
- `accepted.jsonl` — conversations that passed LLM review
- `rejected.jsonl` — rejected conversations
- `reviews.jsonl` — full review records with scores and issues
- `summary.json` — counts

---

## Step 3 — Vibe check (text)

Generate an HTML page showing all accepted conversations; use `--sample N`
for a random subset:

```bash
/Users/tuandinh/.local/bin/uv run --project fd_data_creation \
  python fd_data_creation/render_vibe_text.py \
  --input  outputs/full_duplex_training/production_text_v1/qc/accepted.jsonl \
  --output outputs/full_duplex_training/production_text_v1/vibe_text.html \
  --sample 50
open outputs/full_duplex_training/production_text_v1/vibe_text.html
```

Arrow keys navigate between conversations in the sidebar.

---

## Step 3b — Merge batches into synthesis queue

After reviewing multiple batches, merge all accepted conversations into one file:

```bash
cat outputs/full_duplex_training/production_text_v*/qc/accepted.jsonl \
  > outputs/full_duplex_training/accepted_for_synthesis.jsonl
wc -l outputs/full_duplex_training/accepted_for_synthesis.jsonl
```

---

## Step 4 — Transfer to cluster and synthesize audio (A100)

The Python interpreter for synthesis is the VieNeu-TTS venv on the cluster, not
fd_data_creation. The fd_data_creation code and pipeline scripts must be rsynced first.

### 4a — Sync code to cluster (run on Mac)

```bash
# Pipeline scripts (exclude venv — it's platform-specific)
rsync -av --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
  fd_data_creation/ \
  myserver:/home/tuanda67/projects/Full-Duplex-Bench/fd_data_creation/

rsync -av --exclude='__pycache__' --exclude='*.pyc' \
  speech_training_data/pipeline/ \
  myserver:/home/tuanda67/projects/Full-Duplex-Bench/speech_training_data/pipeline/

# Synthesis queue (the only text file the synthesis step needs)
rsync --partial --append \
  outputs/full_duplex_training/accepted_for_synthesis.jsonl \
  myserver:/home/tuanda67/projects/Full-Duplex-Bench/outputs/full_duplex_training/accepted_for_synthesis.jsonl
```

### 4b — Start synthesis in a persistent tmux session (run from Mac)

`srun` kills background processes when the step exits — use direct SSH to the
compute node instead so tmux persists.

```bash
VIENEU_ROOT="/home/tuanda67/projects/sommelier/external/VieNeu-TTS"
PYTHON="${VIENEU_ROOT}/.venv/bin/python"
PROJ="/home/tuanda67/projects/Full-Duplex-Bench"

ssh myserver "ssh dgx-a100-5 \"
  mkdir -p ${PROJ}/outputs/full_duplex_training/production_audio_v1 && \
  tmux new-session -d -s synthesis \
    \\\"cd ${PROJ} && ${PYTHON} speech_training_data/pipeline/synthesize_utterances.py \
    --config fd_data_creation/config.production.json \
    --input  outputs/full_duplex_training/accepted_for_synthesis.jsonl \
    --output outputs/full_duplex_training/production_audio_v1/03_utterances \
    --vieneu-root ${VIENEU_ROOT} \
    > outputs/full_duplex_training/production_audio_v1/synthesize.log 2>&1\\\" && \
  tmux list-sessions
\""
```

### 4c — Monitor synthesis

```bash
# Progress: count completed conversation directories
ssh myserver 'ssh dgx-a100-5 "ls /home/tuanda67/projects/Full-Duplex-Bench/outputs/full_duplex_training/production_audio_v1/03_utterances/ | wc -l"'

# Tail the log
ssh myserver 'ssh dgx-a100-5 "tail -5 /home/tuanda67/projects/Full-Duplex-Bench/outputs/full_duplex_training/production_audio_v1/synthesize.log"'

# GPU utilization
ssh myserver 'ssh dgx-a100-5 "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits"'
```

### 4d — Download utterance manifest after synthesis completes

```bash
rsync --partial \
  myserver:/home/tuanda67/projects/Full-Duplex-Bench/outputs/full_duplex_training/production_audio_v1/03_utterances/utterance_manifest.jsonl \
  outputs/full_duplex_training/production_audio_v1/03_utterances/utterance_manifest.jsonl
```

### 4e — Start production_audio_v2 (v3–v7 batches, 3932 conversations)

After production_audio_v1 completes, start a second synthesis run for the remaining
batches. The v2 manifest was already synced to the cluster.

```bash
VIENEU_ROOT="/home/tuanda67/projects/sommelier/external/VieNeu-TTS"
PYTHON="${VIENEU_ROOT}/.venv/bin/python"
PROJ="/home/tuanda67/projects/Full-Duplex-Bench"

ssh myserver "ssh dgx-a100-5 \"
  mkdir -p ${PROJ}/outputs/full_duplex_training/production_audio_v2 && \
  tmux new-session -d -s synthesis2 \
    \\\"cd ${PROJ} && ${PYTHON} speech_training_data/pipeline/synthesize_utterances.py \
    --config fd_data_creation/config.production.json \
    --input  outputs/full_duplex_training/accepted_for_synthesis_v2.jsonl \
    --output outputs/full_duplex_training/production_audio_v2/03_utterances \
    --vieneu-root ${VIENEU_ROOT} \
    > outputs/full_duplex_training/production_audio_v2/synthesize.log 2>&1\\\" && \
  tmux list-sessions
\""
```

Monitor:
```bash
ssh myserver 'ssh dgx-a100-5 "ls /home/tuanda67/projects/Full-Duplex-Bench/outputs/full_duplex_training/production_audio_v2/03_utterances/ | wc -l"'
ssh myserver 'ssh dgx-a100-5 "tail -5 /home/tuanda67/projects/Full-Duplex-Bench/outputs/full_duplex_training/production_audio_v2/synthesize.log"'
```

Download manifest:
```bash
rsync --partial \
  myserver:/home/tuanda67/projects/Full-Duplex-Bench/outputs/full_duplex_training/production_audio_v2/03_utterances/utterance_manifest.jsonl \
  outputs/full_duplex_training/production_audio_v2/03_utterances/utterance_manifest.jsonl
```

---

## Step 5 — Render timelines, QC, extract windows

Use the local VieNeu-TTS venv (has soundfile/numpy). Run from repo root.

```bash
VIENEU_PY="external/VieNeu-TTS/.venv/bin/python3"

$VIENEU_PY speech_training_data/pipeline/render_timelines.py \
  --config fd_data_creation/config.production.json \
  --input  outputs/full_duplex_training/production_audio_v1/03_utterances/utterance_manifest.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/04_rendered

$VIENEU_PY speech_training_data/pipeline/audio_qc.py \
  --config fd_data_creation/config.production.json \
  --input  outputs/full_duplex_training/production_audio_v1/04_rendered/rendered_manifest.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/05_audio_qc

$VIENEU_PY speech_training_data/pipeline/extract_windows.py \
  --config fd_data_creation/config.production.json \
  --input  outputs/full_duplex_training/production_audio_v1/05_audio_qc/accepted.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/06_windows
```

---

## Step 6 — Audio vibe check

Audio HTML must be served via HTTP — browsers block local audio on `file://`.

```bash
VIENEU_PY="external/VieNeu-TTS/.venv/bin/python3"

# Production audio vibe check (after audio_qc step):
$VIENEU_PY speech_training_data/pipeline/render_vibe.py \
  --input  outputs/full_duplex_training/production_audio_v1/05_audio_qc/accepted.jsonl \
  --output outputs/full_duplex_training/production_audio_v1/vibe_audio.html

# SA vs RP side-by-side comparison (after render_timelines.py):
/Users/tuandinh/.local/bin/uv run --project fd_data_creation \
  python fd_data_creation/render_compare.py \
  --pairs      outputs/full_duplex_training/compare_8pairs/matched_pairs.jsonl \
  --provenance outputs/full_duplex_training/compare_8pairs/provenance.jsonl \
  --audio      outputs/full_duplex_training/compare_8pairs/04_rendered \
  --output     outputs/full_duplex_training/compare_8pairs/compare_audio.html

# Serve whichever directory the HTML lives in, then open in browser:
pkill -f "http.server 8899"; python3 -m http.server 8899 --directory <html-parent-dir> &
open "http://localhost:8899/<filename>.html"
# Kill server when done:
pkill -f "http.server 8899"
```

### Quick vibe check from partial synthesis (while synthesis is still running)

Download a sample, build a manifest, render, and view — all without waiting for
full synthesis to complete:

```bash
VIENEU_PY="external/VieNeu-TTS/.venv/bin/python3"
SAMPLE_DIR="outputs/full_duplex_training/vibe_audio_sample"
CLUSTER_BASE="/home/tuanda67/projects/Full-Duplex-Bench/outputs/full_duplex_training/production_audio_v1/03_utterances"

# 1. Pick ~20 conversation IDs from cluster (10 SA, 10 RP)
ssh myserver "ssh dgx-a100-5 \"ls ${CLUSTER_BASE} | grep _sa | sed -n '1~8p' | head -10 && ls ${CLUSTER_BASE} | grep _rp | sed -n '1~3p' | head -10\""

# 2. Download them (replace IDs with the ones from step 1)
mkdir -p ${SAMPLE_DIR}/03_utterances
rsync --partial -r \
  myserver:${CLUSTER_BASE}/vi_fd_001001_sa \
  myserver:${CLUSTER_BASE}/vi_fd_001012_rp \
  ${SAMPLE_DIR}/03_utterances/

# 3. Build utterance manifest
$VIENEU_PY fd_data_creation/build_sample_manifest.py \
  --audio  ${SAMPLE_DIR}/03_utterances \
  --text   outputs/full_duplex_training/accepted_for_synthesis.jsonl \
  --output ${SAMPLE_DIR}/utterance_manifest.jsonl

# 4. Render and view
$VIENEU_PY speech_training_data/pipeline/render_timelines.py \
  --config fd_data_creation/config.production.json \
  --input  ${SAMPLE_DIR}/utterance_manifest.jsonl \
  --output ${SAMPLE_DIR}/04_rendered

$VIENEU_PY speech_training_data/pipeline/render_vibe.py \
  --input  ${SAMPLE_DIR}/04_rendered/rendered_manifest.jsonl \
  --output ${SAMPLE_DIR}/vibe_audio.html

pkill -f "http.server 8899"; python3 -m http.server 8899 --directory ${SAMPLE_DIR} &
open "http://localhost:8899/vibe_audio.html"
```

---

## Monitoring a running generation

```bash
# Count completed items
ls outputs/full_duplex_training/production_text_v1/items/ | wc -l

# Count accepted so far
wc -l outputs/full_duplex_training/production_text_v1/generated.jsonl 2>/dev/null || echo "not written yet"

# Tail live log
tail -f outputs/full_duplex_training/production_text_v1/generate.log

# Failure breakdown
python3 -c "
import json
fails = [json.loads(l) for l in open('outputs/full_duplex_training/production_text_v1/failures.jsonl') if l.strip()]
from collections import Counter
print(Counter(f['error'].split(':')[0] for f in fails))
"
```

---

## Configuration

| File | Purpose |
|---|---|
| `fd_data_creation/config.production.json` | Production runs (1k+ conversations) |
| `fd_data_creation/config.smoke.json` | Quick 2-conversation sanity check |
| `fd_data_creation/config.example.json` | Reference / documentation |

Key fields:
- `seed` — change per batch to get different topic/style assignments
- `strategy_weights` — `{"single_author": 0.75, "role_play": 0.25}` is the default mix
- `single_author_models` — cycled round-robin across conversations
- `rounds_min` / `rounds_max` — 6–10 rounds per conversation

---

## Scale reference

| Target audio | Training windows | Conversations needed (≈50% acceptance) |
|---|---|---|
| 10 hours | ~1 500 | ~1 500 |
| 100 hours | ~15 000 | ~15 000 |
| 1 000 hours | ~150 000 | ~150 000 |

Each conversation yields at most 2 training windows (24 s each). Text generation
costs roughly $0.02–0.05 per conversation at current OpenRouter prices. Audio
synthesis is the GPU bottleneck and runs on the A100 cluster.
