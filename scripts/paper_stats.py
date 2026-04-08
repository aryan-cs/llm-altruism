#!/usr/bin/env python3
"""Compute exact paired tests for key paper contrasts."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_PD_PATHS = [
    ROOT
    / "results"
    / "paper_ready_baseline_triplet"
    / "paper-baseline-prisoners_dilemma-20260407T173331Z.json",
    ROOT
    / "results"
    / "paper_ready_replications"
    / "paper-baseline-prisoners_dilemma-20260407T172819Z.json",
]
DEFAULT_BENCHMARK_DIR = ROOT / "results" / "paper_ready_benchmark_triplet"
DEFAULT_SUSCEPTIBILITY_DIR = ROOT / "results" / "paper_ready_susceptibility_triplet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exact paired statistical tests.")
    parser.add_argument(
        "--csv",
        default=str(ROOT / "paper" / "tables" / "paired_statistical_tests.csv"),
        help="CSV output path.",
    )
    parser.add_argument(
        "--markdown",
        default=str(ROOT / "paper" / "tables" / "paired_statistical_tests.md"),
        help="Markdown output path.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trial_mean_metric(trial: dict[str, Any], metric_root: str) -> float:
    summary = trial["summary"]
    return float((summary[f"{metric_root}_a"] + summary[f"{metric_root}_b"]) / 2.0)


def trial_action_trace(trial: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((item["action_a"], item["action_b"]) for item in trial.get("rounds", []))


def exact_sign_flip_p_value(differences: list[float]) -> float:
    """Return an exact two-sided sign-flip p-value for a paired mean difference."""
    nonzero = [abs(float(value)) for value in differences if abs(float(value)) > 1e-12]
    if not differences or not nonzero:
        return 1.0

    denominator = len(differences)
    observed = abs(sum(differences) / denominator)
    extreme = 0
    total = 0
    for signs in itertools.product([-1.0, 1.0], repeat=len(nonzero)):
        statistic = abs(sum(sign * value for sign, value in zip(signs, nonzero, strict=False)) / denominator)
        total += 1
        if statistic >= observed - 1e-12:
            extreme += 1
    return float(extreme / total)


def summarize_paired_values(
    *,
    analysis_id: str,
    track: str,
    game: str,
    condition_a: str,
    condition_b: str,
    values_a: list[float],
    values_b: list[float],
    identical_action_traces: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    if len(values_a) != len(values_b):
        raise ValueError("Paired summaries require equal-length value lists")
    if not values_a:
        raise ValueError("Paired summaries require at least one matched trial")

    differences = [b - a for a, b in zip(values_a, values_b, strict=False)]
    row = {
        "analysis_id": analysis_id,
        "track": track,
        "game": game,
        "condition_a": condition_a,
        "condition_b": condition_b,
        "matched_trials": len(values_a),
        "mean_condition_a": float(sum(values_a) / len(values_a)),
        "mean_condition_b": float(sum(values_b) / len(values_b)),
        "mean_diff_b_minus_a": float(sum(differences) / len(differences)),
        "exact_p_two_sided": exact_sign_flip_p_value(differences),
        "identical_action_traces": identical_action_traces,
        "notes": notes,
    }
    return row


def pd_neutral_rows() -> list[dict[str, Any]]:
    minimal: list[float] = []
    abstract_family: list[float] = []
    compact: list[float] = []
    institutional: list[float] = []
    identical_traces = 0
    matched = 0

    for path in DEFAULT_BASELINE_PD_PATHS:
        summary = load_summary(path)
        trial_map: dict[tuple[tuple[str, str], int], dict[str, dict[str, Any]]] = {}
        for trial in summary["trials"]:
            key = (tuple(trial["pairing"]), int(trial.get("repetition", 0)))
            trial_map.setdefault(key, {})[trial["prompt_variant"]] = trial

        for condition_trials in trial_map.values():
            required = {
                "minimal-neutral",
                "minimal-neutral-compact",
                "minimal-neutral-institutional",
            }
            if not required.issubset(condition_trials):
                continue
            minimal_trial = condition_trials["minimal-neutral"]
            compact_trial = condition_trials["minimal-neutral-compact"]
            institutional_trial = condition_trials["minimal-neutral-institutional"]

            minimal_value = trial_mean_metric(minimal_trial, "cooperation_rate")
            compact_value = trial_mean_metric(compact_trial, "cooperation_rate")
            institutional_value = trial_mean_metric(institutional_trial, "cooperation_rate")

            minimal.append(minimal_value)
            compact.append(compact_value)
            institutional.append(institutional_value)
            abstract_family.append((compact_value + institutional_value) / 2.0)

            matched += 1
            if trial_action_trace(compact_trial) == trial_action_trace(institutional_trial):
                identical_traces += 1

    return [
        summarize_paired_values(
            analysis_id="pd-neutral-minimal-vs-abstract-family",
            track="baseline",
            game="prisoners_dilemma",
            condition_a="minimal-neutral",
            condition_b="abstract-neutral-family",
            values_a=minimal,
            values_b=abstract_family,
            identical_action_traces=None,
            notes=(
                "Abstract-neutral family averages the compact and institutional prompts. "
                "Across the pooled PD cohorts these two prompts produced identical action "
                "traces on every matched trial."
            ),
        ),
        summarize_paired_values(
            analysis_id="pd-neutral-compact-vs-institutional",
            track="baseline",
            game="prisoners_dilemma",
            condition_a="minimal-neutral-compact",
            condition_b="minimal-neutral-institutional",
            values_a=compact,
            values_b=institutional,
            identical_action_traces=identical_traces,
            notes=f"Identical action traces on {identical_traces}/{matched} matched trials.",
        ),
    ]


def benchmark_rows() -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, dict[tuple[str, str], dict[str, Any]]]] = {}
    for path in sorted(DEFAULT_BENCHMARK_DIR.glob("*.json")):
        summary = load_summary(path)
        if summary.get("run_metadata", {}).get("track") != "benchmark":
            continue
        game = str(summary["config"]["experiment"]["game"])
        selected_models = tuple(summary.get("run_metadata", {}).get("selected_models", []))
        presentation = str(summary["run_metadata"]["presentation"])
        grouped.setdefault((game, "|".join(selected_models)), {})[presentation] = {
            tuple(trial["pairing"]): trial for trial in summary["trials"]
        }

    rows: list[dict[str, Any]] = []
    for (game, _cohort), presentations in sorted(grouped.items()):
        for condition_a, condition_b in [("canonical", "unnamed"), ("canonical", "resource")]:
            if condition_a not in presentations or condition_b not in presentations:
                continue
            common_keys = sorted(set(presentations[condition_a]) & set(presentations[condition_b]))
            values_a = [
                trial_mean_metric(presentations[condition_a][key], "cooperation_rate") for key in common_keys
            ]
            values_b = [
                trial_mean_metric(presentations[condition_b][key], "cooperation_rate") for key in common_keys
            ]
            rows.append(
                summarize_paired_values(
                    analysis_id=f"benchmark-{game}-{condition_a}-vs-{condition_b}",
                    track="benchmark",
                    game=game,
                    condition_a=condition_a,
                    condition_b=condition_b,
                    values_a=values_a,
                    values_b=values_b,
                    notes="Exact paired test on matched model pairings within the stable triplet cohort.",
                )
            )
    return rows


def susceptibility_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(DEFAULT_SUSCEPTIBILITY_DIR.glob("*.json")):
        summary = load_summary(path)
        if summary.get("run_metadata", {}).get("track") != "susceptibility":
            continue
        game = str(summary["config"]["experiment"]["game"])
        variants = {
            prompt_variant: {
                tuple(trial["pairing"]): trial
                for trial in summary["trials"]
                if trial["prompt_variant"] == prompt_variant
            }
            for prompt_variant in sorted({trial["prompt_variant"] for trial in summary["trials"]})
        }
        if "competitive" not in variants or "cooperative" not in variants:
            continue
        common_keys = sorted(set(variants["competitive"]) & set(variants["cooperative"]))
        values_a = [trial_mean_metric(variants["competitive"][key], "cooperation_rate") for key in common_keys]
        values_b = [trial_mean_metric(variants["cooperative"][key], "cooperation_rate") for key in common_keys]
        rows.append(
            summarize_paired_values(
                analysis_id=f"susceptibility-{game}-competitive-vs-cooperative",
                track="susceptibility",
                game=game,
                condition_a="competitive",
                condition_b="cooperative",
                values_a=values_a,
                values_b=values_b,
                notes="Exact paired test on matched model pairings within the stable triplet cohort.",
            )
        )
    return rows


def build_stats_frame() -> pd.DataFrame:
    rows = []
    rows.extend(pd_neutral_rows())
    rows.extend(benchmark_rows())
    rows.extend(susceptibility_rows())
    return pd.DataFrame(rows)


def render_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Paired Statistical Tests\n\nNo rows found.\n"

    display = frame.copy()
    for column in [
        "mean_condition_a",
        "mean_condition_b",
        "mean_diff_b_minus_a",
        "exact_p_two_sided",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.5f}")

    lines = [
        "# Paired Statistical Tests",
        "",
        "Exact two-sided sign-flip randomization tests on matched trial-level mean cooperation.",
        "",
        "| analysis_id | track | game | condition_a | condition_b | matched_trials | mean_condition_a | mean_condition_b | mean_diff_b_minus_a | exact_p_two_sided | identical_action_traces | notes |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in display.iterrows():
        values = [
            str(row["analysis_id"]),
            str(row["track"]),
            str(row["game"]),
            str(row["condition_a"]),
            str(row["condition_b"]),
            str(int(row["matched_trials"])),
            str(row["mean_condition_a"]),
            str(row["mean_condition_b"]),
            str(row["mean_diff_b_minus_a"]),
            str(row["exact_p_two_sided"]),
            "" if pd.isna(row["identical_action_traces"]) else str(int(row["identical_action_traces"])),
            str(row["notes"]),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    frame = build_stats_frame()

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)

    markdown_path = Path(args.markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(frame), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
