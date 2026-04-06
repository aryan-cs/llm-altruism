#!/usr/bin/env python3
"""Summarize paper-batch experiment outputs into CSV and Markdown."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.report import load_result_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paper-batch result files.")
    parser.add_argument(
        "results",
        nargs="+",
        help="Paths or globs pointing to result JSON files or directories.",
    )
    parser.add_argument(
        "--csv",
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--markdown",
        help="Optional Markdown output path.",
    )
    return parser.parse_args()


def expand_paths(inputs: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for item in inputs:
        matches = [Path(match) for match in glob.glob(item)]
        if matches:
            expanded.extend(matches)
        else:
            expanded.append(Path(item))
    return expanded


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    experiment = summary.get("config", {}).get("experiment", summary.get("config", {}))
    run_metadata = summary.get("run_metadata", {})
    row = {
        "experiment_id": summary.get("experiment_id"),
        "name": experiment.get("name"),
        "part": experiment.get("part"),
        "game": experiment.get("game"),
        "track": run_metadata.get("track"),
        "presentation": run_metadata.get("presentation"),
        "trial_count": len(summary.get("trials", [])),
        "total_cost_usd": summary.get("total_cost_usd", 0.0),
        "total_duration_seconds": summary.get("total_duration_seconds", 0.0),
        "skipped_model_count": len(summary.get("skipped_models", [])),
        "skipped_trial_count": len(summary.get("skipped_trials", [])),
    }

    aggregate = summary.get("aggregate_summary", {})
    for key, value in aggregate.items():
        if isinstance(value, (int, float)):
            row[key] = value
    return row


def is_experiment_summary(summary: dict[str, Any]) -> bool:
    """Return True when a JSON artifact looks like a real experiment summary."""
    return bool(summary.get("experiment_id")) and (
        "trials" in summary or "aggregate_summary" in summary
    )


def render_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Paper Summary\n\nNo result rows found.\n"

    columns = [
        column
        for column in [
            "experiment_id",
            "track",
            "presentation",
            "part",
            "game",
            "trial_count",
            "cooperation_rate_a",
            "cooperation_rate_b",
            "average_payoff_a",
            "average_payoff_b",
            "survival_rate",
            "final_survival_rate",
            "extinction_event",
            "total_duration_seconds",
        ]
        if column in frame.columns
    ]

    lines = ["# Paper Summary", ""]
    subset = frame[columns].fillna("")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines.append(header)
    lines.append(divider)
    for _, row in subset.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rows = []
    for path in expand_paths(args.results):
        if path.is_dir():
            for json_path in sorted(path.glob("*.json")):
                summary = load_result_artifact(json_path)
                if is_experiment_summary(summary):
                    rows.append(flatten_summary(summary))
        elif path.suffix == ".json":
            summary = load_result_artifact(path)
            if is_experiment_summary(summary):
                rows.append(flatten_summary(summary))

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("No result rows found.")
        return 1

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)

    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(frame), encoding="utf-8")
    else:
        print(render_markdown(frame))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
