#!/usr/bin/env python3
"""CLI entrypoint for running llm-altruism experiments."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.report import render_text_report
from src.experiments.runner import run_experiment_from_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run an llm-altruism experiment.")
    parser.add_argument("--config", required=True, help="Path to an experiment YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without provider API calls using deterministic mock responses.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where JSON/JSONL outputs should be written.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    result = await run_experiment_from_path(
        args.config,
        dry_run=args.dry_run,
        results_dir=args.results_dir,
    )
    print(render_text_report(result))
    return 0


def main() -> int:
    """Run the CLI."""
    args = parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
