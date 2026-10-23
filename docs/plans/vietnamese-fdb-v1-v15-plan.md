# Vietnamese FDB v1.0/v1.5 Plan

## Goal

Create a Vietnamese version of Full-Duplex-Bench v1.0/v1.5 to evaluate turn-taking and overlap handling in Vietnamese spoken dialogue systems.

## Scope

- Reproduce the v1.0 turn-taking tasks:
  - Pause handling
  - Backchanneling
  - Smooth turn-taking
  - User interruption
- Reproduce the v1.5 overlap tasks:
  - User interruption
  - User backchannel
  - Talking to others
  - Background speech

Recommended first milestone: build a Vietnamese v1.5 mini-benchmark with 20 samples per overlap scenario before scaling.

## Scenario Design

- Write natural Vietnamese conversational contexts instead of literal English translations.
- Include Vietnamese-specific speech phenomena:
  - Fillers: "ờ", "ừm", "à", "thì", "kiểu như"
  - Hesitation and self-repair
  - Politeness particles: "ạ", "nhé", "nha", "được không"
  - Backchannels: "ừ", "ừm", "dạ", "vâng", "đúng rồi", "à ha"
- Keep utterances short, spoken, and contextually natural.

## Dataset Format

Mirror the current repo format where possible:

```text
vietnamese_fdb/
  v1_0/
    pause_handling/
    backchannel/
    smooth_turn_taking/
    user_interruption/
  v1_5/
    user_interruption/
    user_backchannel/
    talking_to_other/
    background_speech/
  evaluation/
  get_transcript/
  README.md
```

Each v1.5 sample should include:

- `input.wav`: Vietnamese user audio with overlap event
- `clean_input.wav`: same user audio without overlap
- `metadata.json`: text and overlap timing

Example metadata:

```json
{
  "context_text": "Mình đang tìm quán ăn tối nay.",
  "current_turn_text": "À, bạn xem giúp mình quán nào còn mở sau chín giờ được không?",
  "timestamps": [7.275, 11.467],
  "scenario": "user_interruption"
}
```

## Audio Creation

- Best quality: record native Vietnamese speakers.
- Faster prototype: use Vietnamese TTS and manually verify naturalness.
- For v1.5 overlap:
  - Create or record main user speech.
  - Create or record overlap speech separately.
  - Mix overlap into the selected timestamp.
  - Keep a clean version without overlap.

## Timing Annotation

Annotate event timings in seconds:

- Pause start/end
- Turn end
- Interruption start/end
- Overlap start/end

Keep timing fields compatible with the existing evaluation scripts where possible.

## ASR Adaptation

The current repo uses English-oriented ASR. For Vietnamese, replace or add a Vietnamese-capable ASR while preserving the output format:

```json
{
  "text": "...",
  "chunks": [
    {"text": "xin", "timestamp": [0.12, 0.32]}
  ]
}
```

Candidate ASR options:

- Whisper large-v3 / large-v3-turbo
- Vietnamese wav2vec2 models
- Commercial ASR if timestamp quality is better

## Evaluation Adaptation

- Keep timing metrics mostly unchanged.
- Localize text-based heuristics:
  - Vietnamese backchannel word list
  - Vietnamese tokenization
  - Short utterance thresholds
- Translate or make bilingual LLM judge prompts for behavior and relevance scoring.

## Vietnamese Backchannel Detector

Add a Vietnamese-specific backchannel list:

- "ừ"
- "ừm"
- "ờ"
- "dạ"
- "vâng"
- "rồi"
- "đúng rồi"
- "ok"
- "à"

Short acknowledgements should count as backchannels, not full turn-taking.

## Expected Behavior For v1.5

- User interruption: model should stop and respond to the new request.
- User backchannel: model should ignore the cue and continue.
- Talking to others: model should ignore or briefly clarify.
- Background speech: model should ignore and continue.

## Pilot

Build 20-30 samples first and test:

- ASR timestamp quality
- Overlap timestamp accuracy
- Vietnamese backchannel detection
- Script compatibility
- LLM judge reliability in Vietnamese

## Human Validation

Ask native Vietnamese speakers to review:

- Scenario naturalness
- Audio quality
- Overlap realism
- Expected behavior labels
- Whether translated prompts sound natural

## Metrics

Reuse original headline metrics where possible:

- TOR / takeover rate
- Backchannel frequency
- Response latency
- Stop latency
- Behavior category distribution
- Prosodic shifts if needed

