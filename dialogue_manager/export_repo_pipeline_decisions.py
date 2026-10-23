#!/usr/bin/env python3
"""Convert the authors-recipe tokenized dialogues to label-only DM examples.

The upstream recipe represents a dialogue as interleaved user text, control
tokens, and assistant text.  Our runtime DM predicts only the control action,
so this exporter reconstructs the state visible at every control-token
decision while retaining progressive conversation history.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from export_paper_format import SYSTEM_PROMPT, serialize_input


TOKEN_TO_ACTION = {
    "<|continue-listen|>": "<|C-L|>",
    "<|switch-to-speak|>": "<|S-S|>",
    "<|switch-to-listen|>": "<|S-L|>",
    "<|continue_speak|>": "<|C-S|>",
}
TOKEN_RE = re.compile(
    "(" + "|".join(re.escape(token) for token in TOKEN_TO_ACTION) + "|<\\|interrupted\\|>)"
)


def clean_assistant(text: str) -> str:
    return " ".join(TOKEN_RE.sub("", text).split())


def first_action(text: str) -> tuple[str, str]:
    found = sorted(
        ((text.find(token), token) for token in TOKEN_TO_ACTION if token in text),
        key=lambda item: item[0],
    )
    if not found:
        raise ValueError(f"expected a decision token: {text!r}")
    # A normal response and a fake-interruption continuation may end with
    # switch-to-listen.  That closing marker describes the state after the
    # response; only the first marker is the decision being supervised.
    token = found[0][1]
    if len(found) > 1 and not (
        found[1][1] == "<|switch-to-listen|>"
        and token in {"<|switch-to-speak|>", "<|continue_speak|>"}
    ):
        raise ValueError(f"ambiguous decision tokens {found}: {text!r}")
    return token, TOKEN_TO_ACTION[token]


def make_example(
    conversation_id: str,
    decision_number: int,
    history: list[dict[str, str]],
    state: str,
    ongoing: str,
    streaming_asr: str,
    target: str,
) -> dict:
    model_input = {
        "dialogue_history": [dict(message) for message in history],
        "assistant_state": state,
        "assistant_ongoing": ongoing if state == "speaking" else "",
        "streaming_asr": " ".join(streaming_asr.split()),
    }
    example_id = f"{conversation_id}_d{decision_number:04d}"
    return {
        "example_id": example_id,
        "conversation_id": conversation_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": serialize_input(model_input)},
            {"role": "assistant", "content": target},
        ],
        "input": model_input,
        "target": target,
    }


def convert_dialogue(row: dict, conversation_id: str) -> list[dict]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing messages")

    history: list[dict[str, str]] = []
    examples: list[dict] = []
    current_user = ""
    pending_primary_user = ""
    ongoing = ""
    state = "listening"

    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role == "user":
            # Recipe cuts may occur in the middle of a Vietnamese word, so the
            # streaming fragments must be concatenated byte-for-byte.
            current_user = (current_user + content).strip()
            continue
        if role != "assistant":
            raise ValueError(f"unsupported role {role!r}")

        token, target = first_action(content)
        if not current_user:
            raise ValueError(f"assistant decision {target} has no current user text")
        required_state = "speaking" if target in {"<|S-L|>", "<|C-S|>"} else "listening"
        if state != required_state:
            raise ValueError(
                f"state/action mismatch: state={state}, target={target}, content={content!r}"
            )

        examples.append(
            make_example(
                conversation_id,
                len(examples) + 1,
                history,
                state,
                ongoing,
                current_user,
                target,
            )
        )
        spoken = clean_assistant(content)

        if target == "<|C-L|>":
            # More user audio follows; retain the accumulated ASR hypothesis.
            continue

        if target == "<|S-S|>":
            pending_primary_user = current_user
            current_user = ""
            ongoing = spoken
            state = "speaking"
            if "<|switch-to-listen|>" in content:
                history.append({"role": "user", "content": pending_primary_user})
                if ongoing:
                    history.append({"role": "assistant", "content": ongoing})
                pending_primary_user = ""
                ongoing = ""
                state = "listening"
            continue

        if target == "<|S-L|>":
            # The interrupted assistant prefix becomes completed history.  The
            # interrupting user's remaining fragment continues the same ASR.
            if pending_primary_user:
                history.append({"role": "user", "content": pending_primary_user})
            if ongoing:
                history.append({"role": "assistant", "content": ongoing})
            pending_primary_user = ""
            ongoing = ""
            state = "listening"
            continue

        # C-S: ignore the non-addressed barge-in as dialogue history, finish
        # the pending response, and return to listening when the recipe says so.
        ongoing = " ".join(part for part in (ongoing, spoken) if part).strip()
        current_user = ""
        if "<|switch-to-listen|>" in content:
            if pending_primary_user:
                history.append({"role": "user", "content": pending_primary_user})
            if ongoing:
                history.append({"role": "assistant", "content": ongoing})
            pending_primary_user = ""
            ongoing = ""
            state = "listening"

    if current_user:
        raise ValueError("dialogue ended with unconsumed user text")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-prefix", default="repo_vi")
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output: list[dict] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows, 1):
        examples = convert_dialogue(row, f"{args.id_prefix}_{index:06d}")
        output.extend(examples)
        counts.update(example["target"] for example in examples)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "vi_dm_repo_recipe_decisions_v1",
        "source": str(args.input),
        "conversations": len(rows),
        "decisions": len(output),
        "action_counts": dict(sorted(counts.items())),
    }
    manifest_path = args.output.with_name("manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
