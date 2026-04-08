#!/usr/bin/env python3
"""Launch the canonical long-horizon ecology suite with consistent settings."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_SELECTORS = [
    "cerebras:llama3.1-8b",
    "nvidia:deepseek-ai/deepseek-v3.2",
    "nvidia:moonshotai/kimi-k2-instruct-0905",
]


@dataclass(frozen=True)
class SuiteRun:
    name: str
    config_path: str
    results_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the canonical ecology evidence suite: baseline scarcity, "
            "public reputation, and event-stress."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help="Repeatable model selector override. Defaults to the stable triplet cohort.",
    )
    parser.add_argument(
        "--results-root",
        default="results/canonical_ecology_suite",
        help="Root directory where per-track outputs will be written.",
    )
    parser.add_argument(
        "--from-run",
        choices=["baseline", "reputation", "event-stress"],
        default="baseline",
        help="Start from this suite stage instead of always launching from baseline.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass through to run_experiment.py for deterministic mock responses.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the planned commands without launching them.",
    )
    parser.add_argument(
        "--resume-log",
        help=(
            "Optional prior JSONL log to reuse completed trial summaries from "
            "when launching the first selected suite stage."
        ),
    )
    return parser.parse_args()


def suite_runs(results_root: str) -> list[SuiteRun]:
    root = Path(results_root)
    return [
        SuiteRun(
            name="baseline",
            config_path="configs/part2/society_baseline.yaml",
            results_dir=str(root / "baseline"),
        ),
        SuiteRun(
            name="reputation",
            config_path="configs/part3/society_reputation.yaml",
            results_dir=str(root / "reputation"),
        ),
        SuiteRun(
            name="event-stress",
            config_path="configs/part2/society_event_stress.yaml",
            results_dir=str(root / "event_stress"),
        ),
    ]


def selected_runs(runs: list[SuiteRun], start_name: str) -> list[SuiteRun]:
    start_index = next(index for index, run in enumerate(runs) if run.name == start_name)
    return runs[start_index:]


def build_command(
    run: SuiteRun,
    *,
    models: list[str],
    dry_run: bool,
    resume_log: str | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        run.config_path,
        "--results-dir",
        run.results_dir,
    ]
    for selector in models:
        command.extend(["--model", selector])
    if dry_run:
        command.append("--dry-run")
    if resume_log:
        command.extend(["--resume-log", resume_log])
    return command


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def main() -> int:
    args = parse_args()
    models = args.model or list(DEFAULT_MODEL_SELECTORS)
    runs = selected_runs(suite_runs(args.results_root), args.from_run)

    resume_log = args.resume_log
    for run in runs:
        command = build_command(run, models=models, dry_run=args.dry_run, resume_log=resume_log)
        print(f"[{run.name}] {format_command(command)}")
        if args.print_only:
            continue
        Path(run.results_dir).mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
        resume_log = None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
