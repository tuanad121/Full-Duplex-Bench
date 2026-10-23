# Full-duplex speech training data

This directory builds synchronized conversational audio for training a
full-duplex speech-to-speech model. Speech is synthesized into clean user and
agent stems first; overlap, interruption timing, and target behavior are applied
after synthesis. A mono mixture is retained for listening and optional training.

## Phased plan

### Phase 1 — core behavior

1. Ordinary turn taking: start speaking after a completed user turn.
2. Real interruption: stop the agent after a short reaction delay and listen.
3. Backchannel: continue speaking through a short user acknowledgment.
4. Ambiguous onset: distinguish a backchannel-like prefix followed by a
   substantive correction from a true backchannel.

Phase 1 begins with clean speech so semantic and timing failures are not
confounded by acoustic noise.

### Phase 2 — speech not addressed to the agent

Add named third-party speech, nearby conversations, phone calls, and television
dialogue. Both semantic addressing cues and acoustic distance must vary.

### Phase 3 — acoustic robustness

Apply licensed noise, room impulse responses, codec artifacts, microphone
distance, and residual agent echo independently to the clean stems.

## Quality-control policy

Text acceptance is automated through deterministic validation followed by an
independent LLM review of naturalness, continuity, event role, and target
behavior. Audio acceptance uses automated structural, ASR, VAD, speaker,
clipping, and timing checks. Human listening is an unofficial vibe-check used
for pilots, debugging, and small spot checks; it is not a required production
gate.

## Phase 1 pilot

Generate text with OpenRouter:

```bash
OPENROUTER_API_KEY=... python speech_training_data/phase1/generate_pilot_text.py \
  --output outputs/full_duplex_training/phase1_pilot/spec.json
```

Synthesize and assemble audio from the repository root:

```bash
python speech_training_data/phase1/synthesize_pilot.py \
  --spec outputs/full_duplex_training/phase1_pilot/spec.json \
  --output outputs/full_duplex_training/phase1_pilot
```

Then open `outputs/full_duplex_training/phase1_pilot/index.html`.

Each sample contains equal-length `user.wav`, `agent.wav`, and `mixture.wav`
files plus exact event and target-action timestamps in `metadata.json`.

## Production pipeline

The reusable implementation lives in `speech_training_data/pipeline/`:

1. `generate_dialogues.py` creates 6–15-round Vietnamese conversations through
   explicit paid OpenRouter models. Each conversation contains a small number of
   backchannels and real or ambiguous interruptions.
2. `review_text.py` applies deterministic schema checks and an independent LLM
   semantic review. Only accepted scripts proceed to synthesis.
3. `synthesize_utterances.py` generates resumable atomic user, agent, event, and
   post-interruption clips with VieNeu-TTS.
4. `render_timelines.py` assembles synchronized full conversations. It preserves
   agent speech through backchannels and cuts it after configurable interruption
   reaction latency.
5. `audio_qc.py` checks track structure, synchronization, finite samples,
   clipping, event intervals, overlap behavior, and reaction timing.
6. `extract_windows.py` creates event-centered 24-second causal windows. A window
   is skipped if it would contain another behavioral event.
7. `render_vibe.py` creates an optional, non-gating HTML listening page.

Copy the example configuration before a run:

```bash
cp speech_training_data/pipeline/config.example.json \
  outputs/full_duplex_training/config.json
```

### Recommended Mac/A100 split

Run generation and Text QC on the Mac:

```bash
export OPENROUTER_API_KEY="$(tr -d '\r\n' < 'docs/open router key.txt')"
python speech_training_data/pipeline/run_pipeline.py \
  --config outputs/full_duplex_training/config.json \
  --root outputs/full_duplex_training/run_001 \
  --vieneu-root external/VieNeu-TTS \
  --from-stage generate --to-stage text-qc
```

Copy the configuration, pipeline code, and `02_text_qc/accepted.jsonl` to the
A100 workspace. Inside the existing VieNeu virtual environment, run synthesis:

```bash
python speech_training_data/pipeline/run_pipeline.py \
  --config outputs/full_duplex_training/config.json \
  --root outputs/full_duplex_training/run_001 \
  --vieneu-root /path/to/VieNeu-TTS \
  --from-stage synthesize --to-stage synthesize
```

Copy `03_utterances/` back to the Mac and finish the inexpensive stages:

```bash
python speech_training_data/pipeline/run_pipeline.py \
  --config outputs/full_duplex_training/config.json \
  --root outputs/full_duplex_training/run_001 \
  --vieneu-root external/VieNeu-TTS \
  --from-stage render --to-stage vibe
```

The canonical outputs are full synchronized conversations under
`04_rendered/`; model-facing examples are under `06_windows/`. The generator,
reviewer, and synthesis stages are independently resumable.

### Dependencies

- Generation/Text QC: `openai`
- Rendering/QC/windowing: `numpy`, `soundfile`
- Synthesis: the VieNeu-TTS environment plus its model dependencies

ASR transcript verification and neural speaker-similarity scoring are planned
extensions to the Audio QC stage; the implemented gate currently covers
structural audio and behavioral timing.
