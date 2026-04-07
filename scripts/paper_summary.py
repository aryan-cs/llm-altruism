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


def flatten_prompt_variant_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate trial summaries by prompt variant for a single experiment."""
    experiment = summary.get("config", {}).get("experiment", summary.get("config", {}))
    run_metadata = summary.get("run_metadata", {})
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trial in summary.get("trials", []):
        prompt_variant = trial.get("prompt_variant")
        if not prompt_variant or "summary" not in trial:
            continue
        grouped.setdefault(prompt_variant, []).append(trial["summary"])

    rows: list[dict[str, Any]] = []
    for prompt_variant, trial_summaries in grouped.items():
        row: dict[str, Any] = {
            "experiment_id": summary.get("experiment_id"),
            "name": experiment.get("name"),
            "part": experiment.get("part"),
            "game": experiment.get("game"),
            "track": run_metadata.get("track"),
            "presentation": run_metadata.get("presentation"),
            "prompt_variant": prompt_variant,
            "trial_count": len(trial_summaries),
        }
        numeric_keys = {
            key
            for item in trial_summaries
            for key, value in item.items()
            if isinstance(value, (int, float))
        }
        for key in sorted(numeric_keys):
            values = [item[key] for item in trial_summaries if isinstance(item.get(key), (int, float))]
            if values:
                row[key] = sum(values) / len(values)
        rows.append(row)
    return rows


def flatten_trial_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten raw per-trial metrics for cross-experiment pooling."""
    experiment = summary.get("config", {}).get("experiment", summary.get("config", {}))
    run_metadata = summary.get("run_metadata", {})
    rows: list[dict[str, Any]] = []
    for trial in summary.get("trials", []):
        trial_summary = trial.get("summary")
        prompt_variant = trial.get("prompt_variant")
        if not prompt_variant or not isinstance(trial_summary, dict):
            continue
        row: dict[str, Any] = {
            "experiment_id": summary.get("experiment_id"),
            "name": experiment.get("name"),
            "part": experiment.get("part"),
            "game": experiment.get("game"),
            "track": run_metadata.get("track"),
            "presentation": run_metadata.get("presentation"),
            "prompt_variant": prompt_variant,
            "trial_id": trial.get("trial_id"),
            "repetition": trial.get("repetition"),
        }
        for key, value in trial_summary.items():
            if isinstance(value, (int, float)):
                row[key] = value
        rows.append(row)
    return rows


def pooled_prompt_variant_frame(trial_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw trial rows across experiments by prompt condition."""
    if trial_frame.empty:
        return pd.DataFrame()

    group_columns = ["track", "presentation", "game", "prompt_variant"]
    numeric_columns = [
        column
        for column in trial_frame.columns
        if column not in {"part", "trial_id", "repetition"}
        and pd.api.types.is_numeric_dtype(trial_frame[column])
    ]

    rows: list[dict[str, Any]] = []
    for keys, group in trial_frame.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys, strict=False))
        row["experiment_count"] = int(group["experiment_id"].nunique())
        row["trial_count"] = int(len(group))
        for column in numeric_columns:
            values = group[column].dropna()
            if not values.empty:
                row[column] = float(values.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def is_experiment_summary(summary: dict[str, Any]) -> bool:
    """Return True when a JSON artifact looks like a real experiment summary."""
    return bool(summary.get("experiment_id")) and (
        "trials" in summary or "aggregate_summary" in summary
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Render a simple markdown table from selected columns."""
    if frame.empty:
        return ["No rows found.", ""]

    subset = frame[[column for column in columns if column in frame.columns]].fillna("")
    header = "| " + " | ".join(subset.columns) + " |"
    divider = "| " + " | ".join("---" for _ in subset.columns) + " |"
    lines = [header, divider]
    for _, row in subset.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sort by the subset of requested columns that are present."""
    sort_columns = [column for column in columns if column in frame.columns]
    if not sort_columns:
        return frame
    return frame.sort_values(sort_columns)


def benchmark_delta_frame(experiment_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute benchmark-presentation deltas relative to canonical rows."""
    benchmark = experiment_frame[experiment_frame.get("track") == "benchmark"].copy()
    if benchmark.empty or "presentation" not in benchmark.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for game, group in benchmark.groupby("game"):
        canonical_rows = group[group["presentation"] == "canonical"]
        if canonical_rows.empty:
            continue
        canonical = canonical_rows.iloc[0]
        for _, row in group.iterrows():
            presentation = row.get("presentation")
            if presentation == "canonical":
                continue
            rows.append(
                {
                    "game": game,
                    "presentation": presentation,
                    "cooperation_rate_a": row.get("cooperation_rate_a"),
                    "cooperation_rate_b": row.get("cooperation_rate_b"),
                    "average_payoff_a": row.get("average_payoff_a"),
                    "average_payoff_b": row.get("average_payoff_b"),
                    "delta_cooperation_rate_a": row.get("cooperation_rate_a", 0.0)
                    - canonical.get("cooperation_rate_a", 0.0),
                    "delta_cooperation_rate_b": row.get("cooperation_rate_b", 0.0)
                    - canonical.get("cooperation_rate_b", 0.0),
                    "delta_average_payoff_a": row.get("average_payoff_a", 0.0)
                    - canonical.get("average_payoff_a", 0.0),
                    "delta_average_payoff_b": row.get("average_payoff_b", 0.0)
                    - canonical.get("average_payoff_b", 0.0),
                }
            )
    return pd.DataFrame(rows)


def render_markdown(
    experiment_frame: pd.DataFrame,
    prompt_variant_frame: pd.DataFrame,
    pooled_prompt_frame: pd.DataFrame,
) -> str:
    if experiment_frame.empty:
        return "# Paper Summary\n\nNo result rows found.\n"

    lines = ["# Paper Summary", ""]

    lines.append("## Experiment Summary")
    lines.append("")
    lines.extend(
        markdown_table(
            sort_frame(experiment_frame, ["track", "game", "presentation", "experiment_id"]),
            [
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
            ],
        )
    )

    if not prompt_variant_frame.empty:
        lines.append("## Prompt Variant Breakdown")
        lines.append("")
        lines.extend(
            markdown_table(
                sort_frame(
                    prompt_variant_frame,
                    ["track", "game", "presentation", "prompt_variant", "experiment_id"]
                ),
                [
                    "experiment_id",
                    "track",
                    "presentation",
                    "game",
                    "prompt_variant",
                    "trial_count",
                    "cooperation_rate_a",
                    "cooperation_rate_b",
                    "average_payoff_a",
                    "average_payoff_b",
                    "survival_rate",
                    "final_survival_rate",
                    "extinction_event",
                ],
            )
        )

    if not pooled_prompt_frame.empty:
        lines.append("## Pooled Prompt Variant Summary")
        lines.append("")
        lines.extend(
            markdown_table(
                sort_frame(
                    pooled_prompt_frame,
                    ["track", "game", "presentation", "prompt_variant"]
                ),
                [
                    "track",
                    "presentation",
                    "game",
                    "prompt_variant",
                    "experiment_count",
                    "trial_count",
                    "cooperation_rate_a",
                    "cooperation_rate_b",
                    "average_payoff_a",
                    "average_payoff_b",
                    "survival_rate",
                    "final_survival_rate",
                    "average_trade_volume",
                    "average_gini",
                    "commons_health",
                    "alliance_count",
                    "extinction_event",
                ],
            )
        )

    benchmark_deltas = benchmark_delta_frame(experiment_frame)
    if not benchmark_deltas.empty:
        lines.append("## Benchmark Presentation Deltas")
        lines.append("")
        lines.extend(
            markdown_table(
                sort_frame(benchmark_deltas, ["game", "presentation"]),
                [
                    "game",
                    "presentation",
                    "cooperation_rate_a",
                    "cooperation_rate_b",
                    "average_payoff_a",
                    "average_payoff_b",
                    "delta_cooperation_rate_a",
                    "delta_cooperation_rate_b",
                    "delta_average_payoff_a",
                    "delta_average_payoff_b",
                ],
            )
        )

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rows = []
    prompt_variant_rows = []
    trial_rows = []
    for path in expand_paths(args.results):
        if path.is_dir():
            for json_path in sorted(path.glob("*.json")):
                summary = load_result_artifact(json_path)
                if is_experiment_summary(summary):
                    rows.append(flatten_summary(summary))
                    prompt_variant_rows.extend(flatten_prompt_variant_rows(summary))
                    trial_rows.extend(flatten_trial_rows(summary))
        elif path.suffix == ".json":
            summary = load_result_artifact(path)
            if is_experiment_summary(summary):
                rows.append(flatten_summary(summary))
                prompt_variant_rows.extend(flatten_prompt_variant_rows(summary))
                trial_rows.extend(flatten_trial_rows(summary))

    experiment_frame = pd.DataFrame(rows)
    prompt_variant_frame = pd.DataFrame(prompt_variant_rows)
    trial_frame = pd.DataFrame(trial_rows)
    pooled_prompt_frame = pooled_prompt_variant_frame(trial_frame)
    if experiment_frame.empty:
        print("No result rows found.")
        return 1

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_frame.to_csv(csv_path, index=False)

    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            render_markdown(experiment_frame, prompt_variant_frame, pooled_prompt_frame),
            encoding="utf-8",
        )
    else:
        print(render_markdown(experiment_frame, prompt_variant_frame, pooled_prompt_frame))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
