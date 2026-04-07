"""Install recommended Ollama model cohorts for local or H200 experimentation."""

from __future__ import annotations

import argparse
import subprocess
from typing import Sequence


LOCAL_COHORT: tuple[str, ...] = (
    "qwen3:8b",
    "deepseek-r1:8b",
    "gemma3:12b-it-qat",
    "mistral-nemo:12b",
    "ministral-3:8b",
    "nemotron-mini:4b",
)

H200_COHORT: tuple[str, ...] = (
    "qwen3:30b",
    "qwen3:32b",
    "deepseek-r1:32b",
    "gemma3:27b-it-qat",
    "gpt-oss:120b",
    "nemotron:70b",
)

COHORTS: dict[str, tuple[str, ...]] = {
    "local": LOCAL_COHORT,
    "h200": H200_COHORT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        choices=tuple(COHORTS),
        default="local",
        help="Which predefined Ollama model cohort to install.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Additional Ollama model tags to pull.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the models that would be installed without pulling them.",
    )
    return parser.parse_args()


def iter_models(cohort: str, extras: Sequence[str]) -> list[str]:
    models: list[str] = []
    seen: set[str] = set()
    for model in [*COHORTS[cohort], *extras]:
        if model in seen:
            continue
        seen.add(model)
        models.append(model)
    return models


def pull_model(model: str) -> None:
    subprocess.run(["ollama", "pull", model], check=True)


def main() -> int:
    args = parse_args()
    models = iter_models(args.cohort, args.model)

    print(f"Selected Ollama cohort: {args.cohort}")
    for model in models:
        print(f" - {model}")

    if args.dry_run:
        return 0

    for model in models:
        print(f"\nPulling {model}...")
        pull_model(model)

    print("\nDone. These models will now appear in the experiment CLI automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
