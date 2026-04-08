#!/usr/bin/env python3
"""Summarize paper-batch experiment outputs into CSV and Markdown."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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


def aggregate_trial_summaries(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric trial-summary fields across completed trials."""
    summaries = [trial.get("summary", {}) for trial in trials if isinstance(trial.get("summary"), dict)]
    numeric_keys = {
        key
        for summary in summaries
        for key, value in summary.items()
        if isinstance(value, (int, float))
    }
    aggregate: dict[str, Any] = {}
    for key in sorted(numeric_keys):
        values = [summary[key] for summary in summaries if isinstance(summary.get(key), (int, float))]
        if values:
            aggregate[key] = float(sum(values) / len(values))
    return aggregate


def decode_trial_metadata(experiment: dict[str, Any], trial_id: int) -> dict[str, Any]:
    """Reconstruct prompt/repetition metadata from deterministic trial ordering."""
    prompt_variants = experiment.get("prompt_variants", []) or []
    temperatures = experiment.get("parameters", {}).get("temperature", [0.0]) or [0.0]
    repetitions = int(experiment.get("repetitions", 1) or 1)
    prompt_count = max(1, len(prompt_variants))
    temp_count = max(1, len(temperatures))
    conditions_per_group = prompt_count * temp_count * repetitions
    inner = trial_id % conditions_per_group
    prompt_variant_index = inner // (temp_count * repetitions)
    prompt_variant = None
    if prompt_variants:
        prompt_variant = prompt_variants[prompt_variant_index].get("name")
    repetition = inner % repetitions
    return {
        "prompt_variant": prompt_variant,
        "repetition": repetition,
    }


def load_partial_jsonl_summary(path: Path) -> dict[str, Any] | None:
    """Convert an active JSONL experiment log into a partial summary payload."""
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        lines.append(json.loads(raw_line))

    start = next((item for item in lines if item.get("type") == "experiment_start"), None)
    if start is None:
        return None

    config = start.get("config", {})
    experiment = config.get("experiment", config)
    trial_summary_lines = sorted(
        (item for item in lines if item.get("type") == "trial_summary"),
        key=lambda item: item.get("trial_id", 0),
    )
    if not trial_summary_lines:
        return None

    trials: list[dict[str, Any]] = []
    for item in trial_summary_lines:
        trial_id = int(item.get("trial_id", 0))
        metadata = decode_trial_metadata(experiment, trial_id)
        trials.append(
            {
                "trial_id": trial_id,
                "prompt_variant": metadata["prompt_variant"],
                "repetition": metadata["repetition"],
                "summary": item.get("summary", {}),
            }
        )

    return {
        "experiment_id": start.get("experiment_id"),
        "config": config,
        "run_metadata": config.get("run_metadata", {}),
        "trials": trials,
        "aggregate_summary": aggregate_trial_summaries(trials),
    }


def summary_source_priority(source_kind: str) -> int:
    """Prefer finalized JSON summaries over active JSONL partials."""
    if source_kind == "json":
        return 2
    if source_kind == "jsonl":
        return 1
    return 0


def summary_identity_key(summary: dict[str, Any]) -> str:
    """Return the logical experiment name used to collapse retries and reruns."""
    config = summary.get("config", {})
    experiment = config.get("experiment", config)
    name = experiment.get("name")
    if isinstance(name, str) and name:
        return name
    return str(summary.get("experiment_id"))


def summary_recency_key(summary: dict[str, Any]) -> str:
    """Use lexicographic experiment ids as a stable proxy for run recency."""
    return str(summary.get("experiment_id") or "")


def collect_unique_summaries(paths: list[Path]) -> list[dict[str, Any]]:
    """Load summaries and de-duplicate by logical experiment name across retries."""
    collected: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def register(summary: dict[str, Any], *, source_kind: str) -> None:
        if not is_experiment_summary(summary):
            return
        identity_key = summary_identity_key(summary)
        candidate = {
            "summary": summary,
            "priority": summary_source_priority(source_kind),
            "trial_count": len(summary.get("trials", [])),
            "recency": summary_recency_key(summary),
        }
        existing = collected.get(identity_key)
        if existing is None:
            collected[identity_key] = candidate
            order.append(identity_key)
            return
        if candidate["priority"] > existing["priority"] or (
            candidate["priority"] == existing["priority"]
            and (
                candidate["recency"] > existing["recency"]
                or (
                    candidate["recency"] == existing["recency"]
                    and candidate["trial_count"] > existing["trial_count"]
                )
            )
        ):
            collected[identity_key] = candidate

    for path in paths:
        if path.is_dir():
            for json_path in sorted(path.glob("*.json")):
                register(load_result_artifact(json_path), source_kind="json")
            for jsonl_path in sorted(path.glob("*.jsonl")):
                summary = load_partial_jsonl_summary(jsonl_path)
                if summary is not None:
                    register(summary, source_kind="jsonl")
        elif path.suffix == ".json":
            register(load_result_artifact(path), source_kind="json")
        elif path.suffix == ".jsonl":
            summary = load_partial_jsonl_summary(path)
            if summary is not None:
                register(summary, source_kind="jsonl")

    return [collected[identity_key]["summary"] for identity_key in order]


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
                mean, ci_low, ci_high = bootstrap_mean_ci(values)
                row[key] = mean
                row[f"{key}_ci95_low"] = ci_low
                row[f"{key}_ci95_high"] = ci_high
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
                mean, ci_low, ci_high = bootstrap_mean_ci(values.tolist())
                row[column] = mean
                row[f"{column}_ci95_low"] = ci_low
                row[f"{column}_ci95_high"] = ci_high
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: list[float] | tuple[float, ...],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
) -> tuple[float, float, float]:
    """Estimate a mean and bootstrap confidence interval from trial-level values."""
    numeric = np.asarray([float(value) for value in values], dtype=float)
    if numeric.size == 0:
        raise ValueError("bootstrap_mean_ci requires at least one value")

    mean = float(numeric.mean())
    if numeric.size == 1:
        return mean, mean, mean

    rng = np.random.default_rng(0)
    samples = rng.choice(numeric, size=(resamples, numeric.size), replace=True)
    sample_means = samples.mean(axis=1)
    alpha = 1.0 - confidence
    ci_low = float(np.quantile(sample_means, alpha / 2.0))
    ci_high = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return mean, ci_low, ci_high


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
        prompt_variant_display = format_ci_display_columns(
            prompt_variant_frame,
            [
                "cooperation_rate_a",
                "cooperation_rate_b",
                "average_payoff_a",
                "average_payoff_b",
                "survival_rate",
                "final_survival_rate",
            ],
        )
        lines.append("## Prompt Variant Breakdown")
        lines.append("")
        lines.extend(
            markdown_table(
                sort_frame(
                    prompt_variant_display,
                    ["track", "game", "presentation", "prompt_variant", "experiment_id"]
                ),
                [
                    "experiment_id",
                    "track",
                    "presentation",
                    "game",
                    "prompt_variant",
                    "trial_count",
                    "cooperation_rate_a (95% CI)",
                    "cooperation_rate_b (95% CI)",
                    "average_payoff_a (95% CI)",
                    "average_payoff_b (95% CI)",
                    "survival_rate (95% CI)",
                    "final_survival_rate (95% CI)",
                    "extinction_event",
                ],
            )
        )

    if not pooled_prompt_frame.empty:
        pooled_prompt_display = format_ci_display_columns(
            pooled_prompt_frame,
            [
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
            ],
        )
        lines.append("## Pooled Prompt Variant Summary")
        lines.append("")
        lines.extend(
            markdown_table(
                sort_frame(
                    pooled_prompt_display,
                    ["track", "game", "presentation", "prompt_variant"]
                ),
                [
                    "track",
                    "presentation",
                    "game",
                    "prompt_variant",
                    "experiment_count",
                    "trial_count",
                    "cooperation_rate_a (95% CI)",
                    "cooperation_rate_b (95% CI)",
                    "average_payoff_a (95% CI)",
                    "average_payoff_b (95% CI)",
                    "survival_rate (95% CI)",
                    "final_survival_rate (95% CI)",
                    "average_trade_volume (95% CI)",
                    "average_gini (95% CI)",
                    "commons_health (95% CI)",
                    "alliance_count (95% CI)",
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


def format_ci_display_columns(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Attach compact `mean [low, high]` display columns when CI bounds exist."""
    display = frame.copy()
    for metric in metrics:
        low = f"{metric}_ci95_low"
        high = f"{metric}_ci95_high"
        if metric not in display.columns or low not in display.columns or high not in display.columns:
            continue
        display[f"{metric} (95% CI)"] = [
            f"{mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]"
            for mean, ci_low, ci_high in zip(display[metric], display[low], display[high], strict=False)
        ]
    return display


def main() -> int:
    args = parse_args()
    rows = []
    prompt_variant_rows = []
    trial_rows = []
    for summary in collect_unique_summaries(expand_paths(args.results)):
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
