"""Utilities for reading experiment artifacts and generating summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import ExperimentLogger


def load_result_artifact(path: str | Path) -> dict[str, Any]:
    """Load either a nested JSON summary or derive one from a JSONL log."""
    artifact_path = Path(path)

    if artifact_path.is_dir():
        json_candidates = sorted(artifact_path.glob("*.json"))
        jsonl_candidates = sorted(artifact_path.glob("*.jsonl"))
        if json_candidates:
            artifact_path = json_candidates[0]
        elif jsonl_candidates:
            artifact_path = jsonl_candidates[0]
        else:
            raise FileNotFoundError(f"No experiment artifacts found in {artifact_path}")

    if artifact_path.suffix == ".json":
        return json.loads(artifact_path.read_text(encoding="utf-8"))

    if artifact_path.suffix == ".jsonl":
        return summarize_jsonl_log(ExperimentLogger.read_log(artifact_path))

    raise ValueError(f"Unsupported artifact type: {artifact_path}")


def summarize_jsonl_log(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a JSONL event log into a compact experiment summary."""
    config: dict[str, Any] = {}
    experiment_id = "unknown"
    total_cost = 0.0
    total_duration = 0.0
    rounds = 0
    trial_summaries: list[dict[str, Any]] = []

    for entry in entries:
        entry_type = entry.get("type")
        if entry_type == "experiment_start":
            experiment_id = entry.get("experiment_id", experiment_id)
            config = entry.get("config", {})
        elif entry_type == "round":
            rounds += 1
        elif entry_type == "trial_summary":
            trial_summaries.append(entry.get("summary", {}))
        elif entry_type == "experiment_summary":
            total_cost = float(entry.get("total_cost_usd", 0.0))
            total_duration = float(entry.get("total_duration_seconds", 0.0))

    return {
        "experiment_id": experiment_id,
        "config": config,
        "trial_count": len(trial_summaries),
        "round_events": rounds,
        "total_cost_usd": total_cost,
        "total_duration_seconds": total_duration,
        "trial_summaries": trial_summaries,
    }


def collect_experiment_summaries(paths: list[str | Path]) -> list[dict[str, Any]]:
    """Load and summarize multiple result artifacts."""
    return [load_result_artifact(path) for path in paths]


def comparison_table(paths: list[str | Path]) -> pd.DataFrame:
    """Create a flat comparison dataframe from experiment outputs."""
    rows: list[dict[str, Any]] = []

    for summary in collect_experiment_summaries(paths):
        config = summary.get("config", {})
        experiment = config.get("experiment", config)
        row = {
            "experiment_id": summary.get("experiment_id"),
            "name": experiment.get("name"),
            "part": experiment.get("part"),
            "game": experiment.get("game"),
            "trial_count": summary.get("trial_count", len(summary.get("trials", []))),
            "total_cost_usd": summary.get("total_cost_usd", 0.0),
            "total_duration_seconds": summary.get("total_duration_seconds", 0.0),
        }

        aggregate = summary.get("aggregate_summary", {})
        for key, value in aggregate.items():
            if isinstance(value, (int, float)):
                row[key] = value

        rows.append(row)

    return pd.DataFrame(rows)


def render_text_report(summary: dict[str, Any]) -> str:
    """Render a human-readable text summary from experiment output."""
    config = summary.get("config", {})
    experiment = config.get("experiment", config)
    lines = [
        f"Experiment: {summary.get('experiment_id', 'unknown')}",
        f"Name: {experiment.get('name', 'n/a')}",
        f"Part: {experiment.get('part', 'n/a')}",
    ]

    if "game" in experiment and experiment["game"]:
        lines.append(f"Game: {experiment['game']}")

    lines.append(f"Trials: {summary.get('trial_count', len(summary.get('trials', [])))}")
    lines.append(f"Total cost (USD): {summary.get('total_cost_usd', 0.0):.4f}")
    lines.append(f"Duration (s): {summary.get('total_duration_seconds', 0.0):.2f}")

    aggregate = summary.get("aggregate_summary", {})
    if aggregate:
        lines.append("")
        lines.append("Aggregate metrics:")
        for key, value in sorted(aggregate.items()):
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")

    return "\n".join(lines)
