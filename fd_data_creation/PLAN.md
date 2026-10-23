# Vietnamese full-duplex training-data plan

## Goal

Create long, natural Vietnamese user/assistant conversations, preserve the full
source conversation, and render event-centred 20–30 second training windows.
Phase 1 covers clean turn taking, backchannels, real interruptions, and ambiguous
interruptions. Background speakers and speech addressed to other people remain
deferred until the core behaviours are reliable.

## Model policy

The pilot compares two authoring methods instead of assuming one is superior:

1. **Single author** — one frontier model writes both roles and event metadata in
   one pass. This is the production default because it gives strong global
   coherence and exact structural control.
2. **Role play** — a director creates a private scenario blueprint, then separate
   user and assistant models alternate turns. A separate annotator converts the
   exchange into the training schema. This may improve spontaneity and reduce the
   characteristic rhythm of a single model, but costs more and can drift.

Use a frontier model for part of creation. Cheap models are appropriate for
schema repair and bulk expansion only after the pilot establishes that their
acceptance rate and diversity are competitive. Do not spend frontier-model
tokens on deterministic validation.

The reviewer must not review its own output when an alternative model family is
configured. Every record stores the requested and returned model IDs, strategy,
prompt version, usage, and seed.

## Pilot experiment

Generate at least 100 accepted conversations in matched pairs:

- 50 `single_author`
- 50 `role_play`
- identical topic, style, turn-count, and event distributions
- 6–10 user/assistant rounds, normally 2 behavioural events

Compare blind human preference on a 20-conversation sample and automated QC on
all records:

- acceptance and regeneration rate
- continuity and factual consistency
- naturalness and role distinctiveness
- event correctness and event-position diversity
- lexical/template duplication
- token cost and wall-clock latency per accepted conversation

Adopt role play only if it produces a meaningful quality/diversity gain. A likely
production mix is 70–80% single-author and 20–30% role-play conversations.

## Pipeline

1. Build deterministic, balanced jobs.
2. Generate scripts through OpenRouter (`single_author` or `role_play`).
3. Apply schema, event, duplication, and leakage checks.
4. Apply an independent LLM text review with a strict JSON verdict.
5. Synthesize atomic clips with VieNeu-TTS using fixed user/assistant identities.
6. Render synchronized user, assistant, and mixture stems.
7. Apply structural/timing audio QC; later add ASR, VAD, and speaker similarity.
8. Extract event-centred 24-second windows while retaining full conversations.
9. Produce a non-gating HTML listening page for human vibe checks.

## Acceptance policy

Text QC rejects unnatural Vietnamese, contradictions, role leakage, implausible
assistant behaviour, incorrect backchannels/interruptions, templated repetition,
and events that cannot be rendered naturally. Audio QC rejects missing or corrupt
tracks, sample-rate/duration mismatch, clipping, missing overlap, incorrect stop
latency, or speaker-identity drift once speaker scoring is enabled.

LLM review is advisory evidence wrapped in a deterministic acceptance rule. The
raw review is retained for audit. Audio listening is useful for pilots but is not
a production gate.

