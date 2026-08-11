# Vietnamese FDB v1.0 Harness

This directory is the canonical harness for Vietnamese FDB v1.0. It keeps user audio and assistant audio on separate tracks with one monotonic clock, runs both event and clean conditions, normalizes Vietnamese ASR timestamps, applies a blinded behavior judge, and creates synchronized review HTML.

It is reserved for benchmark creation and evaluation. Dialogue-manager data
generation, training, and evaluation live in the sibling `dialogue_manager/`
project.

## Data and result terminology

- **Event condition** (`input.wav` → `output.wav`): the user question plus a later backchannel, interruption, side utterance, or background event.
- **Clean condition** (`clean_input.wav` → `clean_output.wav`): the same primary question and duration, with silence replacing the event. This exists for the 80 v1.5-derived cases.
- “Event condition” replaces the ambiguous earlier name “mixed.” User and assistant audio are never mono-mixed for scoring.

## Shared-clock contract

1. Connect to the live model and finish session configuration.
2. Set `t=0` immediately before transmitting the first user PCM frame.
3. Pace input against absolute source timestamps.
4. Position each assistant audio delta on the same monotonic clock.
5. Queue deltas for each response at audible playout speed.
6. Truncate queued audio at a response cancellation time.
7. Make the assistant timeline at least as long as the input timeline.
8. Save VAD and response lifecycle events in `*_timing.json`.

ASR runs independently on user and assistant tracks. Stereo/dual playback exists only for human review.

## Setup

Core validation and orchestration have no Python dependencies:

```bash
cd vi_fdb_harness
uv sync
```

GPU ASR:

```bash
uv sync --group asr
```

LLM judging:

```bash
uv sync --group judge
```

The GPT-Realtime JavaScript transport is installed separately:

```bash
cd ../v1_v1.5/model_inference/gpt-realtime
npm install
```

## 1. Validate the canonical 160-sample dataset

From `vi_fdb_harness/`:

```bash
uv run python harness.py validate-dataset \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710
```

Expected balance: 20 samples in each of eight tasks; 160 event inputs and 80 paired clean inputs.

## 2. Run GPT-Realtime

Never store a key in the repository. Export it in the shell:

```bash
export OPENAI_API_KEY=...
```

One-sample check:

```bash
uv run python harness.py run-openai \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710 \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --node-cli ../v1_v1.5/model_inference/gpt-realtime/cli.js \
  --condition both --jobs 1 --limit 1
```

Official 240-condition run:

```bash
uv run python harness.py run-openai \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710 \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --node-cli ../v1_v1.5/model_inference/gpt-realtime/cli.js \
  --condition both --jobs 4
```

The harness is resumable by default. Add `--overwrite` only to intentionally regenerate existing outputs.

Validate results:

```bash
uv run python harness.py validate-run \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710 \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime
```

## 3. Vietnamese ASR

Primary Vietnamese transcription and timestamps:

```bash
uv run python transcribe.py \
  --root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --backend chunkformer
```

Whisper timestamp verification on a 24-file bake-off:

```bash
uv run python transcribe.py \
  --root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --backend whisper --limit 24
```

Compare ASRs and flag text similarity below 0.75 or onset disagreement above 300 ms:

```bash
uv run python compare_asr.py \
  --root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --primary chunkformer --verifier whisper
```

Run PhoWhisper only on flagged cases or for an additional audit. Every backend writes the same normalized schema:

```json
{
  "schema_version": 1,
  "backend": "chunkformer",
  "audio": "output.wav",
  "text": "...",
  "chunks": [{"text": "...", "timestamp": [4.92, 5.24]}]
}
```

## 4. Blinded behavior judge

The judge sees the task, source text/event interval, and timestamped clean/event assistant transcripts. Dialogue-manager action tokens are not part of benchmark scoring. Internal API cancellation events are diagnostic only; the judge scores observable behavior.

```bash
export OPENAI_API_KEY=...
uv run python judge.py \
  --root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --asr-backend chunkformer
```

Cases below 0.7 judge confidence and ASR-disagreement cases require human review.

## 5. Synchronized review report

```bash
uv run python report.py \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --asr-backend chunkformer
```

The generated `review.html` plays the event input and assistant output from the same timestamp. It does not alter scoring audio.

## Release gate

A run is eligible for scoring only when:

- Dataset validation reports zero errors.
- There are 160 event outputs and 80 clean outputs.
- Every output has a readable WAV and a corrected monotonic timing sidecar.
- ASR produces normalized timestamp chunks or explicitly records no speech.
- All ASR/judge disagreements are resolved or listed for human review.
- Model, voice, VAD settings, concurrency, and run date are recorded in `run_config.json`.
