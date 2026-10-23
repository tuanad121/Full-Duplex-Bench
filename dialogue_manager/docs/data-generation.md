# Vietnamese Dialogue-Manager Training Data Generation

## Objective

Create Vietnamese instruction-tuning data for the lightweight semantic dialogue
manager described by Zhang et al., *LLM-Enhanced Dialogue Management for
Full-Duplex Spoken Dialogue Systems* (arXiv:2502.14145). The model consumes
streaming ASR text plus dialogue/system state and predicts one action:

- `<|C-L|>`: continue listening; the user's query is incomplete.
- `<|S-S|>`: start speaking; the user's query is complete.
- `<|S-L|>`: stop speaking and listen; the user made a real interruption.
- `<|C-S|>`: continue speaking; the overlap is a fake interruption.

The Vietnamese Full-Duplex Benchmark's 400 v1.0 cases are held out and never
copied into dialogue-manager training.

## Paper-aligned generation procedure

1. Sample a topic, setting, interaction type, and Vietnamese speaking style.
2. Sample 2–12 dialogue rounds.
3. Assign each round one of: normal query, incomplete query, real interruption,
   fake interruption, or an ordinary uninterrupted exchange.
4. Give the generator the required control-token sequence. The LLM writes the
   dialogue but does not decide the ground-truth labels.
5. Produce streaming decision records containing dialogue history, assistant
   state, partial/current ASR text, and the next control token.
6. Create punctuation-reduced and plausible-ASR variants only as additional
   views; retain the clean transcript as provenance.
7. Validate schema, token/state compatibility, Vietnamese content, round count,
   duplication, and class/context/style balance.
8. Keep train, validation, and test splits separated by conversation and semantic
   scenario. Retain normal uninterrupted dialogues to reduce overfitting.

## Topic pool

Reception/hotel, healthcare scheduling, pharmacy, banking, shopping/returns,
delivery, restaurants, travel, flights/trains, in-car navigation, smart home,
work meetings, education, technical support, entertainment, family/household,
friend-to-friend planning, public services, and non-diagnostic safety scenarios.

Contexts are balanced across receptionist/service assistants, smart home,
in-car assistants, and human-to-human conversations.

## Speaking-style pool

Neutral, concise, polite/formal, casual/colloquial, hesitant, self-correcting,
rambling, impatient, uncertain, elderly/simple phrasing, fragmented speech,
mild Vietnamese-English code-switching, regional vocabulary, punctuation-free
ASR style, and ASR text with plausible recognition errors.

## Multi-model policy

The initial pilot used OpenRouter's rotating free pool. Scale-up prohibits both
`openrouter/free` and every `:free` endpoint. Jobs are rotated deterministically
and evenly across explicit paid endpoints from seven independent families:

- Google Gemini 2.5 Flash
- Qwen3 235B A22B Instruct
- OpenAI GPT-4.1 mini
- DeepSeek V4 Flash
- Mistral Small 3.2 24B Instruct
- Meta Llama 4 Maverick
- Anthropic Claude Sonnet 4.6

The requested and actual provider model IDs are retained separately. This makes
model quotas, acceptance rates, cost, and linguistic diversity reproducible.
A generator cannot approve its own output: local rules validate the schema,
token/state agreement, required ongoing speech, and user/ASR consistency.
Later semantic auditing uses a different model family plus sampled human review.

## Scale and outputs

- Pilot: 1,000 conversations, approximately 7,000 dialogue rounds.
- Training v1 target: 10,000–12,000 conversations and 70,000–80,000 decisions.
- Raw API responses, accepted JSONL, rejected examples, job specifications,
  configuration, model provenance, and summary statistics are retained.

The generator is resumable: completed IDs are skipped, failed IDs can be retried,
and partial output remains auditable.
