# Paper repository versus Vietnamese DM pipeline

Compared against `HaoZhang6720/fullduplex-dialogue-data` (main branch, cloned
2026-07-29).

## Equivalent components

| Function | Authors' repository | Vietnamese pipeline |
|---|---|---|
| Multi-round generation | 1–8 or configurable rounds | 2–12 rounds |
| Topic/style sampling | EN/ZH topic and persona pools | Vietnamese topic, style, and context-family pools |
| LLM API | OpenAI-compatible caller, one configured model | OpenRouter-compatible caller, seven explicit paid models |
| Four actions | switch/continue speak/listen | paper abbreviations: `<|S-S|>`, `<|S-L|>`, `<|C-S|>`, `<|C-L|>` |
| Filtering | parse/schema/artifact filters | schema, state/action, ongoing-speech, ASR/user, and duplicate filters |
| Provenance | intermediate JSONL stages | raw/accepted/rejected files plus provider model and usage |
| SFT export | OpenAI messages or HY format | OpenAI messages with one control-token target per decision |

## Important differences

1. **Training unit**
   - Authors: one complete dialogue sequence. Control tokens are interleaved
     with user and assistant messages, and assistant answers remain in the SFT
     output.
   - Ours: one example per decision point. Input is accumulated dialogue,
     current assistant state/ongoing response, and streaming ASR; output is
     exactly one control token.
   - Assessment: ours isolates the semantic-DM objective more directly, but is
     not byte-for-byte the authors' canonical SFT representation.

2. **Interaction construction**
   - Authors: generate ordinary/interruption dialogues, then deterministically
     cut user or assistant text and probabilistically insert real/fake
     interruptions and one or two continue-listen events.
   - Ours: assign a required event/action sequence first and ask the LLM to
     author a natural example satisfying it.
   - Assessment: the authors' post-processing gives tighter class ratios and
     multiple event positions; ours gives more natural cut points but relies
     more heavily on semantic filtering.

3. **Assistant state**
   - Authors: state is implicit in the preceding control token and message
     sequence; `<|interrupted|>` marks truncated assistant speech internally.
   - Ours: `listening`/`speaking` and already emitted assistant text are explicit
     input fields. Audit-only event roles are withheld from the model.

4. **ASR augmentation**
   - Authors: emit base, trailing-punctuation-stripped, and punctuation-added
     variants (3× canonical augmentation).
   - Ours: use normalized punctuation-free streaming-ASR hypotheses and retain
     clean text only for audit.
   - Recommended change: add controlled history depth and limited
     punctuation/ASR variants without triplicating the eval set.

5. **Event density**
   - Authors: may insert several full-duplex decisions into a conversation.
   - Ours: typically has one designated special event per conversation plus
     normal decisions.
   - Recommended change: add deterministic multi-event augmentation to improve
     minority-label density before the production-scale run.

6. **Splitting and metrics**
   - Authors' released repository prepares data but does not include model
     training, checkpoint selection, TensorBoard, or evaluation code.
   - Ours splits by conversation, records model provenance, computes macro-F1
     and per-label metrics, early-stops, and writes TensorBoard events.

## Repository caveat

In the inspected commit, `scripts/06_to_openai_jsonl.py` references
`messages_to_hy_yzs` for the canonical HY export but does not define or import
that function. The OpenAI `messages` export is implemented. Therefore the
repository is highly useful as the reference data recipe, but it is not a
complete executable reproduction of model training.

## Baseline decision

Retain our label-only DM trainer and explicit runtime state because they match
the intended deployed module. Before full training:

1. run a tiny end-to-end smoke test;
2. establish an unmodified current-data baseline;
3. add an authors-recipe ablation with deterministic multi-event and
   punctuation/ASR augmentation;
4. compare both using the same conversation-held-out eval set.
