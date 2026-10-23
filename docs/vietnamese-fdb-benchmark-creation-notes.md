# Experience Building the Vietnamese Full-Duplex Benchmark

We adapted the English Full-Duplex Benchmark methodology into a Vietnamese benchmark rather than translating its samples directly.

## 1. Start from behaviors, not sentences

We defined eight turn-taking situations:

- Pause handling
- Smooth turn-taking
- Backchannel
- User interruption
- User backchannel
- Talking to another person
- Background speech
- Overlapping interruption

Each sample has an expected dialogue-manager action, such as continuing, listening, stopping, or yielding.

## 2. Diversify conversational contexts

Early samples were overly concentrated on receptionist robots. We expanded them across:

- Reception and service assistants
- Smart homes
- In-car assistants
- Human-to-human conversations

Context changes the vocabulary, conversational relationship, expected politeness, and what constitutes a natural interruption or backchannel.

## 3. Generate atomic audio streams

We synthesize the main question and conversational event separately using VieNeu-TTS. This gives precise control over:

- Speaker identity and dialect
- Event placement
- Silence before and after events
- Foreground versus distant speech
- Clean and event-containing versions

The final timeline is assembled from these atomic streams with exact timestamps.

## 4. Human review is indispensable

Automated validation catches schema, duration, missing files, and timestamp errors, but native listeners are needed for:

- Mispronunciations
- Hallucinated or missing syllables
- Unnatural wording
- Whether a question naturally elicits a sufficiently long answer
- Whether an event sounds like a backchannel or interruption

For the 160-sample pilot, repeated problems allowed the reviewer to comment on representative failures rather than label every duplicate issue.

## 5. Trailing silence matters

A full-duplex system needs enough time to begin and finish responding. Most Vietnamese cases use approximately five seconds of trailing silence, matching English FDB practice. Task-specific exceptions, such as pause handling, use shorter intentional tails.

## 6. Real-time alignment is more difficult than ordinary TTS evaluation

The user and assistant cannot simply be concatenated. Both require one shared monotonic clock:

```text
question -> assistant starts -> user event -> assistant reaction
```

Important lessons:

- Start `t=0` at the first transmitted audio frame, not during API setup.
- Pace audio against absolute timestamps to avoid accumulated drift.
- Preserve assistant arrival/playout timing.
- Truncate buffered assistant audio when a response is cancelled.
- Store VAD, response creation, cancellation, and completion events.
- Review user and assistant as synchronized stereo channels.

An early implementation accidentally included WebSocket setup latency in the assistant timeline. The synchronized vibe check exposed this error.

## 7. Judge observable behavior, not internal model events

GPT-Realtime internally cancelled a response when it heard a backchannel, then said “rồi, mình nói tiếp nhé…” and resumed.

The cancellation event alone does not imply failure. Benchmark scoring must evaluate what the user hears:

- Did the assistant resume the original answer?
- Did it respond to the event as a new request?
- Did it become confused?
- Did it stop completely?
- How long was the recovery pause?

Semantic behavior and timing quality should be scored separately.

## 8. Use ASR conservatively

Our planned transcription pipeline is:

- ChunkFormer for Vietnamese transcription
- Whisper large-v3 for word alignment
- PhoWhisper for disagreements
- Human review for ambiguous cases

A large ASR committee is unnecessary. Two complementary systems plus selective human arbitration provide better cost and interpretability.

## 9. Evaluate clean and event-containing pairs

For v1.5, every event-containing input has a clean counterpart with identical timing. Comparing the system’s outputs reveals whether the backchannel, interruption, or background speech caused a meaningful behavioral change.

## 10. Current scope and next target

We currently have a balanced **160-sample pilot: 20 samples across each of eight tasks**. This is enough to validate methodology and run systems, but not a strong final leaderboard.

Recommended milestones:

- **160 samples:** pilot and methodology validation
- **400 samples:** credible Vietnamese FDB v1.0 release
- **800 samples:** stronger leaderboard and statistical analysis

The biggest lesson is that benchmark quality comes less from generating many clips and more from preserving **natural context, controlled timing, observable behavior, and reviewable evidence**.
