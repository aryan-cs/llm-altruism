#!/usr/bin/env python3
"""Generate paper-ready figures from summarized experiment artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]

VARIANT_ORDER = [
    "minimal-neutral",
    "minimal-neutral-compact",
    "minimal-neutral-institutional",
    "competitive",
    "cooperative",
    "task-only",
]
VARIANT_COLORS = {
    "minimal-neutral": "#264653",
    "minimal-neutral-compact": "#2a9d8f",
    "minimal-neutral-institutional": "#8ab17d",
    "competitive": "#d62828",
    "cooperative": "#f4a261",
    "task-only": "#577590",
}
PRESENTATION_COLORS = {
    "canonical": "#264653",
    "resource_disguise": "#2a9d8f",
    "unnamed_isomorphic": "#e9c46a",
}
TRACK_COLORS = {
    "society": "#577590",
    "reputation": "#bc4749",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from result artifacts.")
    parser.add_argument(
        "results",
        nargs="+",
        help="Paths or globs pointing to result JSON files or directories.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where PNG figures and a manifest should be written.",
    )
    return parser.parse_args()


def load_paper_summary_module():
    module_path = Path(__file__).with_name("paper_summary.py")
    spec = importlib.util.spec_from_file_location("paper_summary_helpers", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load summary helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_frames(result_inputs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_module = load_paper_summary_module()
    summaries = summary_module.collect_unique_summaries(summary_module.expand_paths(result_inputs))
    rows = [summary_module.flatten_summary(summary) for summary in summaries]
    prompt_variant_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for summary in summaries:
        prompt_variant_rows.extend(summary_module.flatten_prompt_variant_rows(summary))
        trial_rows.extend(summary_module.flatten_trial_rows(summary))
    experiment_frame = pd.DataFrame(rows)
    prompt_variant_frame = pd.DataFrame(prompt_variant_rows)
    trial_frame = pd.DataFrame(trial_rows)
    pooled_prompt_frame = summary_module.pooled_prompt_variant_frame(trial_frame)
    return experiment_frame, prompt_variant_frame, pooled_prompt_frame


def prettify_slug(value: str) -> str:
    return value.replace("_", " ").replace("-", "\n")


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def side_average_metric_frame(frame: pd.DataFrame, metric_root: str) -> pd.DataFrame:
    """Collapse agent A/B metrics into a single row mean for plotting."""
    metric_columns = [
        column for column in [f"{metric_root}_a", f"{metric_root}_b"] if column in frame.columns
    ]
    if not metric_columns:
        return pd.DataFrame()

    averaged = frame.copy()
    averaged[metric_root] = averaged[metric_columns].mean(axis=1)

    low_columns = [
        column
        for column in [f"{metric_root}_a_ci95_low", f"{metric_root}_b_ci95_low"]
        if column in frame.columns
    ]
    high_columns = [
        column
        for column in [f"{metric_root}_a_ci95_high", f"{metric_root}_b_ci95_high"]
        if column in frame.columns
    ]
    if low_columns and high_columns:
        averaged[f"{metric_root}_ci95_low"] = averaged[low_columns].mean(axis=1)
        averaged[f"{metric_root}_ci95_high"] = averaged[high_columns].mean(axis=1)
    return averaged


def add_error_bars(
    ax: plt.Axes,
    x_positions: list[int],
    means: list[float],
    lowers: list[float] | None,
    uppers: list[float] | None,
) -> None:
    if lowers is None or uppers is None:
        return
    errors = [
        [max(0.0, mean - low) for mean, low in zip(means, lowers, strict=False)],
        [max(0.0, high - mean) for mean, high in zip(means, uppers, strict=False)],
    ]
    ax.errorbar(
        x_positions,
        means,
        yerr=errors,
        fmt="none",
        ecolor="#222222",
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        zorder=3,
    )


def save_prompt_variant_track_figure(
    frame: pd.DataFrame,
    *,
    track: str,
    metric_root: str,
    output_path: Path,
    title: str,
    y_label: str,
    limit: tuple[float, float] | None = None,
) -> Path | None:
    subset = frame[frame.get("track") == track].copy()
    subset = side_average_metric_frame(subset, metric_root)
    if subset.empty or "game" not in subset.columns or "prompt_variant" not in subset.columns:
        return None

    games = [game for game in subset["game"].dropna().drop_duplicates().tolist()]
    if not games:
        return None

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, len(games), figsize=(5.2 * len(games), 4.8), squeeze=False)

    for index, game in enumerate(games):
        ax = axes[0][index]
        game_frame = subset[subset["game"] == game].copy()
        order_lookup = {name: idx for idx, name in enumerate(VARIANT_ORDER)}
        game_frame["_sort"] = game_frame["prompt_variant"].map(lambda item: order_lookup.get(item, 999))
        game_frame = game_frame.sort_values(["_sort", "prompt_variant"])

        x_positions = list(range(len(game_frame)))
        heights = game_frame[metric_root].tolist()
        colors = [
            VARIANT_COLORS.get(prompt_variant, "#6c757d")
            for prompt_variant in game_frame["prompt_variant"].tolist()
        ]
        ax.bar(x_positions, heights, color=colors, width=0.7, zorder=2)

        lower = (
            game_frame[f"{metric_root}_ci95_low"].tolist()
            if f"{metric_root}_ci95_low" in game_frame.columns
            else None
        )
        upper = (
            game_frame[f"{metric_root}_ci95_high"].tolist()
            if f"{metric_root}_ci95_high" in game_frame.columns
            else None
        )
        add_error_bars(ax, x_positions, heights, lower, upper)

        ax.set_title(prettify_slug(str(game)), fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [prettify_slug(str(item)) for item in game_frame["prompt_variant"].tolist()],
            fontsize=10,
        )
        ax.set_ylabel(y_label if index == 0 else "")
        if limit is not None:
            ax.set_ylim(*limit)

    fig.suptitle(title, fontsize=16, y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_benchmark_figure(experiment_frame: pd.DataFrame, output_path: Path) -> Path | None:
    subset = experiment_frame[experiment_frame.get("track") == "benchmark"].copy()
    subset = side_average_metric_frame(subset, "cooperation_rate")
    if subset.empty or "game" not in subset.columns or "presentation" not in subset.columns:
        return None

    games = [game for game in subset["game"].dropna().drop_duplicates().tolist()]
    if not games:
        return None

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(1, len(games), figsize=(5.2 * len(games), 4.8), squeeze=False)

    for index, game in enumerate(games):
        ax = axes[0][index]
        game_frame = subset[subset["game"] == game].copy()
        game_frame = game_frame.sort_values("presentation")
        x_positions = list(range(len(game_frame)))
        heights = game_frame["cooperation_rate"].tolist()
        colors = [
            PRESENTATION_COLORS.get(presentation, "#6c757d")
            for presentation in game_frame["presentation"].tolist()
        ]
        ax.bar(x_positions, heights, color=colors, width=0.7, zorder=2)
        ax.set_title(prettify_slug(str(game)), fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [prettify_slug(str(item)) for item in game_frame["presentation"].tolist()],
            fontsize=10,
        )
        ax.set_ylabel("Mean cooperation" if index == 0 else "")
        ax.set_ylim(0, 1)

    fig.suptitle("Benchmark presentation shifts measured cooperation", fontsize=16, y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_society_survival_figure(pooled_prompt_frame: pd.DataFrame, output_path: Path) -> Path | None:
    subset = pooled_prompt_frame[
        pooled_prompt_frame.get("track").isin(["society", "reputation"])
    ].copy()
    if subset.empty or "prompt_variant" not in subset.columns or "track" not in subset.columns:
        return None

    metric_root = "final_survival_rate"
    subset = subset[subset[metric_root].notna()].copy()
    if subset.empty:
        return None

    sns.set_theme(style="whitegrid", context="talk")
    tracks = [track for track in ["society", "reputation"] if track in subset["track"].tolist()]
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.2 * len(tracks), 4.8), squeeze=False)

    for index, track in enumerate(tracks):
        ax = axes[0][index]
        track_frame = subset[subset["track"] == track].copy()
        order_lookup = {name: idx for idx, name in enumerate(VARIANT_ORDER)}
        track_frame["_sort"] = track_frame["prompt_variant"].map(
            lambda item: order_lookup.get(item, 999)
        )
        track_frame = track_frame.sort_values(["_sort", "prompt_variant"])
        x_positions = list(range(len(track_frame)))
        heights = track_frame[metric_root].tolist()
        colors = [TRACK_COLORS.get(track, "#6c757d")] * len(track_frame)
        ax.bar(x_positions, heights, color=colors, width=0.7, zorder=2)

        lower = (
            track_frame[f"{metric_root}_ci95_low"].tolist()
            if f"{metric_root}_ci95_low" in track_frame.columns
            else None
        )
        upper = (
            track_frame[f"{metric_root}_ci95_high"].tolist()
            if f"{metric_root}_ci95_high" in track_frame.columns
            else None
        )
        add_error_bars(ax, x_positions, heights, lower, upper)

        ax.set_title(track.title(), fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [prettify_slug(str(item)) for item in track_frame["prompt_variant"].tolist()],
            fontsize=10,
        )
        ax.set_ylabel("Final survival rate" if index == 0 else "")
        ax.set_ylim(0, 1)

    fig.suptitle("Scarcity and reputation outcomes by prompt condition", fontsize=16, y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    experiment_frame, prompt_variant_frame, pooled_prompt_frame = build_frames(args.results)
    if experiment_frame.empty:
        print("No experiment rows found.")
        return 1

    created: list[Path] = []
    for maybe_path in [
        save_prompt_variant_track_figure(
            pooled_prompt_frame if not pooled_prompt_frame.empty else prompt_variant_frame,
            track="baseline",
            metric_root="cooperation_rate",
            output_path=output_dir / "baseline_prompt_variants_cooperation.png",
            title="Neutral baseline prompt variants",
            y_label="Mean cooperation",
            limit=(0, 1),
        ),
        save_prompt_variant_track_figure(
            pooled_prompt_frame if not pooled_prompt_frame.empty else prompt_variant_frame,
            track="susceptibility",
            metric_root="cooperation_rate",
            output_path=output_dir / "susceptibility_prompt_variants_cooperation.png",
            title="Prompt susceptibility across games",
            y_label="Mean cooperation",
            limit=(0, 1),
        ),
        save_benchmark_figure(
            experiment_frame,
            output_dir / "benchmark_presentations_cooperation.png",
        ),
        save_society_survival_figure(
            pooled_prompt_frame,
            output_dir / "society_reputation_final_survival.png",
        ),
    ]:
        if maybe_path is not None:
            created.append(maybe_path)

    manifest = {
        "source_results": args.results,
        "experiment_count": int(len(experiment_frame)),
        "figures": [path.name for path in created],
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    if not created:
        print("No figures generated.")
        return 1

    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
