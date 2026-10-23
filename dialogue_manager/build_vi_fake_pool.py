#!/usr/bin/env python3
"""Build an 80/20 Vietnamese affirmation/interruption pool for recipe step 05."""

from __future__ import annotations

import argparse
from pathlib import Path


def lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = lines(args.base)
    affirmations = base[len(base) // 2 :]
    extracted = list(dict.fromkeys(lines(args.extracted)))
    # Uniform sampling from this line list approximates the repository's 0.8
    # affirmation / 0.2 unrelated-or-query mixture.
    repeats = max(1, (4 * len(extracted) + len(affirmations) - 1) // len(affirmations))
    pool = (affirmations * repeats)[: 4 * len(extracted)] + extracted
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(pool) + "\n", encoding="utf-8")
    print(f"affirmation_entries={4 * len(extracted)} extracted_entries={len(extracted)} total={len(pool)}")


if __name__ == "__main__":
    main()
