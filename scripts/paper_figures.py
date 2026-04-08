#!/usr/bin/env python3
"""Generate paper-ready figures from summarized experiment artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
SOCIETY_VARIANT_ORDER = ["task-only", "cooperative", "competitive"]
EVENT_CATEGORY_ORDER = ["gather", "share", "steal", "communicate", "rest", "reproduce", "system"]
EVENT_CATEGORY_COLORS = {
    "gather": "#577590",
    "share": "#2a9d8f",
    "steal": "#d62828",
    "communicate": "#e9c46a",
    "rest": "#6d597a",
    "reproduce": "#f28482",
    "system": "#adb5bd",
}
FIGURE_DPI = 300


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.05)


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


def load_summaries(result_inputs: list[str]) -> tuple[Any, list[dict[str, Any]]]:
    summary_module = load_paper_summary_module()
    summaries = summary_module.collect_unique_summaries(summary_module.expand_paths(result_inputs))
    return summary_module, summaries


def build_frames_from_summaries(
    summary_module: Any,
    summaries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def build_frames(result_inputs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_module, summaries = load_summaries(result_inputs)
    return build_frames_from_summaries(summary_module, summaries)


def summary_run_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("run_metadata") or summary.get("config", {}).get("run_metadata", {})


def summary_track(summary: dict[str, Any]) -> str | None:
    experiment = summary.get("config", {}).get("experiment", summary.get("config", {}))
    run_metadata = summary_run_metadata(summary)
    track = run_metadata.get("track")
    if isinstance(track, str) and track:
        return track
    part = experiment.get("part")
    if part == 2:
        return "society"
    if part == 3:
        return "reputation"
    return None


def flatten_society_round_rows(summaries: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for summary in summaries:
        experiment = summary.get("config", {}).get("experiment", summary.get("config", {}))
        track = summary_track(summary)
        if track not in {"society", "reputation"}:
            continue

        for trial in summary.get("trials", []):
            prompt_variant = trial.get("prompt_variant")
            if not prompt_variant:
                continue
            for round_payload in trial.get("rounds", []):
                events = round_payload.get("events", []) or []
                round_rows.append(
                    {
                        "experiment_id": summary.get("experiment_id"),
                        "name": experiment.get("name"),
                        "track": track,
                        "prompt_variant": prompt_variant,
                        "trial_id": trial.get("trial_id"),
                        "repetition": trial.get("repetition"),
                        "timestep": round_payload.get("timestep"),
                        "alive_count": round_payload.get("alive_count"),
                        "total_agents": round_payload.get("total_agents"),
                        "public_food": round_payload.get("public_food"),
                        "public_water": round_payload.get("public_water"),
                        "public_resources": round_payload.get("public_resources"),
                        "trade_volume": round_payload.get("trade_volume"),
                        "average_health": round_payload.get("average_health"),
                        "average_energy": round_payload.get("average_energy"),
                        "birth_count": len(round_payload.get("spawned_agents", []) or []),
                        "death_count": len(round_payload.get("newly_dead", []) or []),
                        "event_count": len(events),
                    }
                )
                for event in events:
                    event_rows.append(
                        {
                            "experiment_id": summary.get("experiment_id"),
                            "name": experiment.get("name"),
                            "track": track,
                            "prompt_variant": prompt_variant,
                            "trial_id": trial.get("trial_id"),
                            "repetition": trial.get("repetition"),
                            "timestep": round_payload.get("timestep"),
                            "event_kind": event.get("kind", "unknown"),
                            "actor": event.get("actor"),
                            "target": event.get("target"),
                            "amount": event.get("amount"),
                            "message": event.get("message"),
                        }
                    )

    return pd.DataFrame(round_rows), pd.DataFrame(event_rows)


def flatten_society_model_rows(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract per-model alive-population counts from society rounds."""
    model_rows: list[dict[str, Any]] = []

    for summary in summaries:
        experiment = summary.get("config", {}).get("experiment", summary.get("config", {}))
        track = summary_track(summary)
        if track not in {"society", "reputation"}:
            continue

        for trial in summary.get("trials", []):
            prompt_variant = trial.get("prompt_variant")
            if not prompt_variant:
                continue
            for round_payload in trial.get("rounds", []):
                counts: dict[str, int] = {}
                for vital in (round_payload.get("agent_vitals") or {}).values():
                    if not isinstance(vital, dict) or not vital.get("alive", False):
                        continue
                    model = str(vital.get("model") or "unknown")
                    counts[model] = counts.get(model, 0) + 1
                for model, alive_count_model in counts.items():
                    model_rows.append(
                        {
                            "experiment_id": summary.get("experiment_id"),
                            "name": experiment.get("name"),
                            "track": track,
                            "prompt_variant": prompt_variant,
                            "trial_id": trial.get("trial_id"),
                            "repetition": trial.get("repetition"),
                            "timestep": round_payload.get("timestep"),
                            "model": model,
                            "alive_count_model": alive_count_model,
                        }
                    )

    return pd.DataFrame(model_rows)


def flatten_society_agent_vital_rows(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract per-agent vital signs from society rounds."""
    vital_rows: list[dict[str, Any]] = []

    for summary in summaries:
        experiment = summary.get("config", {}).get("experiment", summary.get("config", {}))
        track = summary_track(summary)
        if track not in {"society", "reputation"}:
            continue

        for trial in summary.get("trials", []):
            prompt_variant = trial.get("prompt_variant")
            if not prompt_variant:
                continue
            for round_payload in trial.get("rounds", []):
                for agent_id, vital in (round_payload.get("agent_vitals") or {}).items():
                    if not isinstance(vital, dict) or not vital.get("alive", False):
                        continue
                    vital_rows.append(
                        {
                            "experiment_id": summary.get("experiment_id"),
                            "name": experiment.get("name"),
                            "track": track,
                            "prompt_variant": prompt_variant,
                            "trial_id": trial.get("trial_id"),
                            "repetition": trial.get("repetition"),
                            "timestep": round_payload.get("timestep"),
                            "agent_id": str(vital.get("agent_id") or agent_id),
                            "model": str(vital.get("model") or "unknown"),
                            "food": vital.get("food"),
                            "water": vital.get("water"),
                            "energy": vital.get("energy"),
                            "health": vital.get("health"),
                            "resources_total": vital.get("resources_total"),
                        }
                    )

    return pd.DataFrame(vital_rows)


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


def ordered_prompt_variants(values: list[str], *, society_only: bool = False) -> list[str]:
    if society_only:
        order = SOCIETY_VARIANT_ORDER
    else:
        order = VARIANT_ORDER
    order_lookup = {name: idx for idx, name in enumerate(order)}
    return sorted(values, key=lambda item: (order_lookup.get(item, 999), item))


def event_category(kind: str | None) -> str:
    normalized = str(kind or "unknown")
    if normalized in {"broadcast", "whisper", "message"}:
        return "communicate"
    if normalized in {"gather", "forage_food", "draw_water"}:
        return "gather"
    if normalized in {"share", "offer_trade", "accept_trade", "trade_completed"}:
        return "share"
    if normalized in {"steal"}:
        return "steal"
    if normalized in {"sleep"}:
        return "rest"
    if normalized in {"reproduce"}:
        return "reproduce"
    return "system"


def summarize_round_metric_frame(
    round_frame: pd.DataFrame,
    *,
    metric_root: str,
    summary_module: Any | None = None,
) -> pd.DataFrame:
    if round_frame.empty or metric_root not in round_frame.columns or "track" not in round_frame.columns:
        return pd.DataFrame()

    if summary_module is None:
        summary_module = load_paper_summary_module()

    subset = round_frame[
        round_frame.get("track").isin(["society", "reputation"]) & round_frame[metric_root].notna()
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in subset.groupby(["track", "prompt_variant", "timestep"], dropna=False):
        values = group[metric_root].dropna().tolist()
        if not values:
            continue
        mean, ci_low, ci_high = summary_module.bootstrap_mean_ci(values)
        rows.append(
            {
                "track": keys[0],
                "prompt_variant": keys[1],
                "timestep": keys[2],
                metric_root: mean,
                f"{metric_root}_ci95_low": ci_low,
                f"{metric_root}_ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def summarize_society_phase_frame(
    round_frame: pd.DataFrame,
    *,
    summary_module: Any | None = None,
) -> pd.DataFrame:
    """Aggregate collapse-window and plateau-window means across society trials."""
    required = {
        "track",
        "prompt_variant",
        "trial_id",
        "timestep",
        "alive_count",
        "total_agents",
        "public_food",
        "public_water",
        "average_health",
        "average_energy",
        "death_count",
    }
    if round_frame.empty or not required.issubset(set(round_frame.columns)):
        return pd.DataFrame()

    if summary_module is None:
        summary_module = load_paper_summary_module()

    subset = round_frame[round_frame.get("track").isin(["society", "reputation"])].copy()
    if subset.empty:
        return pd.DataFrame()

    metric_names = [
        "alive_count",
        "public_food",
        "public_water",
        "average_health",
        "average_energy",
    ]
    trial_phase_rows: list[dict[str, Any]] = []

    for (track, prompt_variant, trial_id), group in subset.groupby(
        ["track", "prompt_variant", "trial_id"],
        dropna=False,
    ):
        trial_frame = group.sort_values("timestep").copy()
        latest_alive = trial_frame["alive_count"].iloc[-1]
        first_loss_timestep = next(
            (
                int(row.timestep)
                for row in trial_frame.itertuples()
                if pd.notna(row.alive_count)
                and pd.notna(row.total_agents)
                and float(row.alive_count) < float(row.total_agents)
            ),
            None,
        )
        last_death_timestep = next(
            (
                int(row.timestep)
                for row in reversed(list(trial_frame.itertuples()))
                if pd.notna(row.death_count) and float(row.death_count) > 0
            ),
            None,
        )
        stability_start_timestep = next(
            (
                int(row.timestep) + 1
                for row in reversed(list(trial_frame.itertuples()))
                if pd.notna(row.alive_count) and float(row.alive_count) != float(latest_alive)
            ),
            int(trial_frame["timestep"].iloc[0]),
        )
        stabilized_post_collapse = bool(
            last_death_timestep is not None
            and int(trial_frame["timestep"].iloc[-1]) - int(last_death_timestep) >= 5
            and stability_start_timestep == last_death_timestep
        )

        phase_specs: list[tuple[str, pd.DataFrame]] = []
        if first_loss_timestep is not None and last_death_timestep is not None:
            collapse_frame = trial_frame[
                (trial_frame["timestep"] >= first_loss_timestep)
                & (trial_frame["timestep"] <= last_death_timestep)
            ].copy()
            if not collapse_frame.empty:
                phase_specs.append(("collapse", collapse_frame))
        if stabilized_post_collapse:
            plateau_frame = trial_frame[trial_frame["timestep"] >= stability_start_timestep].copy()
            if not plateau_frame.empty:
                phase_specs.append(("plateau", plateau_frame))

        for phase, phase_frame in phase_specs:
            row: dict[str, Any] = {
                "track": track,
                "prompt_variant": prompt_variant,
                "trial_id": trial_id,
                "phase": phase,
                "phase_rounds": int(len(phase_frame)),
            }
            for metric_name in metric_names:
                row[metric_name] = float(phase_frame[metric_name].mean())
            trial_phase_rows.append(row)

    if not trial_phase_rows:
        return pd.DataFrame()

    trial_phase_frame = pd.DataFrame(trial_phase_rows)
    rows: list[dict[str, Any]] = []
    for (track, prompt_variant, phase), group in trial_phase_frame.groupby(
        ["track", "prompt_variant", "phase"],
        dropna=False,
    ):
        row: dict[str, Any] = {
            "track": track,
            "prompt_variant": prompt_variant,
            "phase": phase,
        }
        for metric_name in ["phase_rounds", *metric_names]:
            values = group[metric_name].dropna().tolist()
            if not values:
                continue
            mean, ci_low, ci_high = summary_module.bootstrap_mean_ci(values)
            row[metric_name] = mean
            row[f"{metric_name}_ci95_low"] = ci_low
            row[f"{metric_name}_ci95_high"] = ci_high
        rows.append(row)

    return pd.DataFrame(rows)


def build_event_mix_frame(event_frame: pd.DataFrame) -> pd.DataFrame:
    if event_frame.empty or "event_kind" not in event_frame.columns or "track" not in event_frame.columns:
        return pd.DataFrame()

    subset = event_frame[event_frame.get("track").isin(["society", "reputation"])].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["event_category"] = subset["event_kind"].map(event_category)
    grouped = (
        subset.groupby(["track", "prompt_variant", "event_category"], dropna=False)
        .size()
        .reset_index(name="event_count")
    )
    totals = grouped.groupby(["track", "prompt_variant"], dropna=False)["event_count"].transform("sum")
    grouped["event_share"] = grouped["event_count"] / totals
    return grouped


def summarize_model_population_frame(
    model_frame: pd.DataFrame,
    *,
    summary_module: Any | None = None,
) -> pd.DataFrame:
    """Aggregate per-model alive counts across repeated trials."""
    if model_frame.empty or "alive_count_model" not in model_frame.columns:
        return pd.DataFrame()

    if summary_module is None:
        summary_module = load_paper_summary_module()

    rows: list[dict[str, Any]] = []
    for keys, group in model_frame.groupby(
        ["track", "prompt_variant", "model", "timestep"],
        dropna=False,
    ):
        values = group["alive_count_model"].dropna().tolist()
        if not values:
            continue
        mean, ci_low, ci_high = summary_module.bootstrap_mean_ci(values)
        rows.append(
            {
                "track": keys[0],
                "prompt_variant": keys[1],
                "model": keys[2],
                "timestep": keys[3],
                "alive_count_model": mean,
                "alive_count_model_ci95_low": ci_low,
                "alive_count_model_ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def summarize_model_vital_frame(
    agent_frame: pd.DataFrame,
    *,
    metric_root: str,
    summary_module: Any | None = None,
) -> pd.DataFrame:
    """Aggregate per-model vital metrics across repeated trials."""
    if agent_frame.empty or metric_root not in agent_frame.columns:
        return pd.DataFrame()

    if summary_module is None:
        summary_module = load_paper_summary_module()

    subset = agent_frame[
        agent_frame.get("track").isin(["society", "reputation"]) & agent_frame[metric_root].notna()
    ].copy()
    if subset.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in subset.groupby(
        ["track", "prompt_variant", "model", "timestep"],
        dropna=False,
    ):
        values = group[metric_root].dropna().tolist()
        if not values:
            continue
        mean, ci_low, ci_high = summary_module.bootstrap_mean_ci(values)
        rows.append(
            {
                "track": keys[0],
                "prompt_variant": keys[1],
                "model": keys[2],
                "timestep": keys[3],
                metric_root: mean,
                f"{metric_root}_ci95_low": ci_low,
                f"{metric_root}_ci95_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


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


def annotate_bar_values(ax: plt.Axes, heights: list[float]) -> None:
    for index, height in enumerate(heights):
        ax.text(
            index,
            height + 0.02,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#222222",
            zorder=4,
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

    configure_plot_style()
    fig, axes = plt.subplots(1, len(games), figsize=(6.0 * len(games), 5.8), squeeze=False)

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
        annotate_bar_values(ax, heights)

        ax.set_title(prettify_slug(str(game)), fontsize=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [prettify_slug(str(item)) for item in game_frame["prompt_variant"].tolist()],
            fontsize=11,
        )
        ax.set_ylabel(y_label if index == 0 else "")
        if limit is not None:
            ax.set_ylim(*limit)
        else:
            ax.set_ylim(0, max(1.0, max(heights) + 0.12))

    fig.suptitle(title, fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
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

    configure_plot_style()
    fig, axes = plt.subplots(1, len(games), figsize=(6.0 * len(games), 5.8), squeeze=False)

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
        annotate_bar_values(ax, heights)
        ax.set_title(prettify_slug(str(game)), fontsize=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [prettify_slug(str(item)) for item in game_frame["presentation"].tolist()],
            fontsize=11,
        )
        ax.set_ylabel("Mean cooperation" if index == 0 else "")
        ax.set_ylim(0, 1)

    fig.suptitle("Benchmark presentation shifts measured cooperation", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_society_metric_figure(
    pooled_prompt_frame: pd.DataFrame,
    *,
    metric_root: str,
    output_path: Path,
    title: str,
    y_label: str,
    limit: tuple[float, float] | None = None,
) -> Path | None:
    subset = pooled_prompt_frame[
        pooled_prompt_frame.get("track").isin(["society", "reputation"])
    ].copy()
    if subset.empty or "prompt_variant" not in subset.columns or "track" not in subset.columns:
        return None

    subset = subset[subset[metric_root].notna()].copy()
    if subset.empty:
        return None

    configure_plot_style()
    tracks = [track for track in ["society", "reputation"] if track in subset["track"].tolist()]
    fig, axes = plt.subplots(1, len(tracks), figsize=(6.0 * len(tracks), 5.8), squeeze=False)

    for index, track in enumerate(tracks):
        ax = axes[0][index]
        track_frame = subset[subset["track"] == track].copy()
        order_lookup = {name: idx for idx, name in enumerate(SOCIETY_VARIANT_ORDER)}
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
        annotate_bar_values(ax, heights)

        ax.set_title(track.title(), fontsize=15)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [prettify_slug(str(item)) for item in track_frame["prompt_variant"].tolist()],
            fontsize=11,
        )
        ax.set_ylabel(y_label if index == 0 else "")
        if limit is not None:
            ax.set_ylim(*limit)
        else:
            ax.set_ylim(0, max(max(heights) + 0.25, 1.0))

    fig.suptitle(title, fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_society_survival_figure(pooled_prompt_frame: pd.DataFrame, output_path: Path) -> Path | None:
    return save_society_metric_figure(
        pooled_prompt_frame,
        metric_root="final_survival_rate",
        output_path=output_path,
        title="Scarcity and reputation outcomes by prompt condition",
        y_label="Final survival rate",
        limit=(0, 1),
    )


def save_society_timeline_figure(
    round_frame: pd.DataFrame,
    *,
    metric_root: str,
    output_path: Path,
    title: str,
    y_label: str,
    limit: tuple[float, float] | None = None,
) -> Path | None:
    summary_frame = summarize_round_metric_frame(round_frame, metric_root=metric_root)
    if summary_frame.empty:
        return None

    configure_plot_style()
    tracks = [track for track in ["society", "reputation"] if track in summary_frame["track"].tolist()]
    fig, axes = plt.subplots(1, len(tracks), figsize=(6.8 * len(tracks), 5.6), squeeze=False)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for index, track in enumerate(tracks):
        ax = axes[0][index]
        track_frame = summary_frame[summary_frame["track"] == track].copy()
        prompt_variants = ordered_prompt_variants(
            track_frame["prompt_variant"].dropna().drop_duplicates().tolist(),
            society_only=True,
        )
        for prompt_variant in prompt_variants:
            variant_frame = track_frame[track_frame["prompt_variant"] == prompt_variant].copy()
            variant_frame = variant_frame.sort_values("timestep")
            x_values = variant_frame["timestep"].tolist()
            y_values = variant_frame[metric_root].tolist()
            line = ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2.5,
                color=VARIANT_COLORS.get(prompt_variant, "#6c757d"),
                label=prettify_slug(prompt_variant).replace("\n", " "),
                zorder=3,
            )[0]
            ax.fill_between(
                x_values,
                variant_frame[f"{metric_root}_ci95_low"].tolist(),
                variant_frame[f"{metric_root}_ci95_high"].tolist(),
                color=VARIANT_COLORS.get(prompt_variant, "#6c757d"),
                alpha=0.15,
                zorder=2,
            )
            if index == 0:
                legend_handles.append(line)
                legend_labels.append(prettify_slug(prompt_variant).replace("\n", " "))

        ax.set_title(track.title(), fontsize=15)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(y_label if index == 0 else "")
        ax.set_xticks(sorted(track_frame["timestep"].dropna().unique().tolist()))
        if limit is not None:
            ax.set_ylim(*limit)
        else:
            high_column = f"{metric_root}_ci95_high"
            upper = track_frame[high_column].max() if high_column in track_frame.columns else track_frame[metric_root].max()
            ax.set_ylim(0, max(1.0, float(upper) * 1.08))

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(3, len(legend_labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.06),
        )
    fig.suptitle(title, fontsize=18, y=1.10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_society_event_mix_figure(
    event_frame: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> Path | None:
    mix_frame = build_event_mix_frame(event_frame)
    if mix_frame.empty:
        return None

    configure_plot_style()
    tracks = [track for track in ["society", "reputation"] if track in mix_frame["track"].tolist()]
    fig, axes = plt.subplots(1, len(tracks), figsize=(6.8 * len(tracks), 5.6), squeeze=False)

    for index, track in enumerate(tracks):
        ax = axes[0][index]
        track_frame = mix_frame[mix_frame["track"] == track].copy()
        prompt_variants = ordered_prompt_variants(
            track_frame["prompt_variant"].dropna().drop_duplicates().tolist(),
            society_only=True,
        )
        y_positions = list(range(len(prompt_variants)))
        cumulative = [0.0] * len(prompt_variants)

        for category in EVENT_CATEGORY_ORDER:
            category_values: list[float] = []
            for prompt_variant in prompt_variants:
                rows = track_frame[
                    (track_frame["prompt_variant"] == prompt_variant)
                    & (track_frame["event_category"] == category)
                ]
                category_values.append(
                    float(rows.iloc[0]["event_share"]) if not rows.empty else 0.0
                )

            bars = ax.barh(
                y_positions,
                category_values,
                left=cumulative,
                color=EVENT_CATEGORY_COLORS.get(category, "#6c757d"),
                label=prettify_slug(category).replace("\n", " "),
                zorder=2,
            )
            for bar, share in zip(bars, category_values, strict=False):
                if share >= 0.10:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_y() + bar.get_height() / 2.0,
                        f"{share:.0%}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#111111",
                        zorder=3,
                    )
            cumulative = [left + value for left, value in zip(cumulative, category_values, strict=False)]

        ax.set_title(track.title(), fontsize=15)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([prettify_slug(item).replace("\n", " ") for item in prompt_variants])
        ax.invert_yaxis()
        ax.set_xlabel("Share of logged events")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_xlim(0, 1)
        if index == 0:
            ax.set_ylabel("Prompt variant")
        else:
            ax.set_ylabel("")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 5),
            frameon=False,
            bbox_to_anchor=(0.5, 1.07),
        )
    fig.suptitle(title, fontsize=18, y=1.11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_society_model_population_figure(
    model_frame: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> Path | None:
    summary_frame = summarize_model_population_frame(model_frame)
    if summary_frame.empty:
        return None

    configure_plot_style()
    tracks = [track for track in ["society", "reputation"] if track in summary_frame["track"].tolist()]
    prompt_variants = ordered_prompt_variants(
        summary_frame["prompt_variant"].dropna().drop_duplicates().tolist(),
        society_only=True,
    )
    if not tracks or not prompt_variants:
        return None

    models = sorted(summary_frame["model"].dropna().drop_duplicates().tolist())
    palette = sns.color_palette("colorblind", n_colors=max(1, len(models)))
    model_colors = {model: palette[index] for index, model in enumerate(models)}

    fig, axes = plt.subplots(
        len(tracks),
        len(prompt_variants),
        figsize=(5.8 * len(prompt_variants), 4.8 * len(tracks)),
        squeeze=False,
    )
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for row_index, track in enumerate(tracks):
        for col_index, prompt_variant in enumerate(prompt_variants):
            ax = axes[row_index][col_index]
            panel = summary_frame[
                (summary_frame["track"] == track)
                & (summary_frame["prompt_variant"] == prompt_variant)
            ].copy()
            if panel.empty:
                ax.axis("off")
                continue

            for model in models:
                model_panel = panel[panel["model"] == model].copy().sort_values("timestep")
                if model_panel.empty:
                    continue
                x_values = model_panel["timestep"].tolist()
                y_values = model_panel["alive_count_model"].tolist()
                line = ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2.2,
                    color=model_colors[model],
                    label=model,
                    zorder=3,
                )[0]
                ax.fill_between(
                    x_values,
                    model_panel["alive_count_model_ci95_low"].tolist(),
                    model_panel["alive_count_model_ci95_high"].tolist(),
                    color=model_colors[model],
                    alpha=0.12,
                    zorder=2,
                )
                if row_index == 0 and col_index == 0:
                    legend_handles.append(line)
                    legend_labels.append(model)

            ax.set_title(
                f"{track.title()}: {prettify_slug(prompt_variant).replace(chr(10), ' ')}",
                fontsize=14,
            )
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Alive agents" if col_index == 0 else "")
            ax.set_xticks(sorted(panel["timestep"].dropna().unique().tolist()))
            upper = float(panel["alive_count_model_ci95_high"].max())
            ax.set_ylim(0, max(1.0, upper * 1.10))

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(3, len(legend_labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(title, fontsize=18, y=1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_society_vitals_heatmap(
    agent_frame: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> Path | None:
    if agent_frame.empty:
        return None

    required = {"track", "prompt_variant", "trial_id", "timestep", "agent_id", "model"}
    if not required.issubset(set(agent_frame.columns)):
        return None

    panels: list[tuple[tuple[str, str, int], pd.DataFrame]] = []
    for keys, group in agent_frame.groupby(["track", "prompt_variant", "trial_id"], dropna=False):
        latest_timestep = group["timestep"].max()
        latest_group = group[group["timestep"] == latest_timestep].copy()
        if latest_group.empty:
            continue
        latest_group = latest_group.sort_values(["model", "agent_id"])
        latest_group["agent_label"] = [
            f"{agent_id} | {str(model).split('/')[-1]}"
            for agent_id, model in zip(
                latest_group["agent_id"],
                latest_group["model"],
                strict=False,
            )
        ]
        latest_group.attrs["latest_timestep"] = latest_timestep
        panels.append((keys, latest_group))

    if not panels:
        return None

    ncols = 1 if len(panels) == 1 else min(2, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    max_agents = max(len(panel) for _, panel in panels)
    figure_height = max(4.5 * nrows, 0.33 * max_agents * nrows + 1.8)

    configure_plot_style()
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, figure_height),
        squeeze=False,
    )
    flat_axes = axes.flatten()
    metrics = ["food", "water", "energy", "health"]

    for index, ((track, prompt_variant, trial_id), panel) in enumerate(panels):
        ax = flat_axes[index]
        heatmap_frame = (
            panel.set_index("agent_label")[metrics]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )
        sns.heatmap(
            heatmap_frame,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="#ffffff",
            cbar=index == 0,
            ax=ax,
        )
        latest_timestep = panel.attrs.get("latest_timestep")
        ax.set_title(
            f"{track.title()}: {prettify_slug(prompt_variant).replace(chr(10), ' ')} "
            f"(trial {trial_id}, t={latest_timestep})",
            fontsize=13,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Agent")

    for ax in flat_axes[len(panels) :]:
        ax.axis("off")

    fig.suptitle(title, fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_society_model_vitals_timeline_figure(
    agent_frame: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> Path | None:
    if agent_frame.empty:
        return None

    metrics = ["food", "water", "energy", "health"]
    metric_labels = {
        "food": "Average food",
        "water": "Average water",
        "energy": "Average energy",
        "health": "Average health",
    }
    summary_frames = {
        metric: summarize_model_vital_frame(agent_frame, metric_root=metric)
        for metric in metrics
    }
    if all(frame.empty for frame in summary_frames.values()):
        return None

    group_rows: list[tuple[str, str]] = []
    for frame in summary_frames.values():
        if frame.empty:
            continue
        for keys, _group in frame.groupby(["track", "prompt_variant"], dropna=False):
            pair = (str(keys[0]), str(keys[1]))
            if pair not in group_rows:
                group_rows.append(pair)
    if not group_rows:
        return None

    track_order = {"society": 0, "reputation": 1}
    prompt_order = {name: idx for idx, name in enumerate(SOCIETY_VARIANT_ORDER)}
    group_rows.sort(key=lambda item: (track_order.get(item[0], 99), prompt_order.get(item[1], 99), item))

    models = sorted(
        {
            str(model)
            for frame in summary_frames.values()
            if not frame.empty
            for model in frame["model"].dropna().tolist()
        }
    )
    palette = sns.color_palette("colorblind", n_colors=max(1, len(models)))
    model_colors = {model: palette[index] for index, model in enumerate(models)}

    configure_plot_style()
    fig, axes = plt.subplots(
        len(group_rows),
        len(metrics),
        figsize=(5.0 * len(metrics), 4.5 * len(group_rows)),
        squeeze=False,
    )
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for row_index, (track, prompt_variant) in enumerate(group_rows):
        for col_index, metric in enumerate(metrics):
            ax = axes[row_index][col_index]
            frame = summary_frames[metric]
            panel = frame[
                (frame["track"] == track)
                & (frame["prompt_variant"] == prompt_variant)
            ].copy()
            if panel.empty:
                ax.axis("off")
                continue
            for model in models:
                model_panel = panel[panel["model"] == model].copy().sort_values("timestep")
                if model_panel.empty:
                    continue
                x_values = model_panel["timestep"].tolist()
                y_values = model_panel[metric].tolist()
                line = ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2.1,
                    color=model_colors[model],
                    label=model,
                    zorder=3,
                )[0]
                ax.fill_between(
                    x_values,
                    model_panel[f"{metric}_ci95_low"].tolist(),
                    model_panel[f"{metric}_ci95_high"].tolist(),
                    color=model_colors[model],
                    alpha=0.12,
                    zorder=2,
                )
                if row_index == 0 and col_index == 0:
                    legend_handles.append(line)
                    legend_labels.append(model)
            title_prefix = (
                f"{track.title()}: {prettify_slug(prompt_variant).replace(chr(10), ' ')}"
                if col_index == 0
                else metric_labels[metric]
            )
            ax.set_title(title_prefix, fontsize=13)
            ax.set_xlabel("Timestep")
            ax.set_ylabel(metric_labels[metric] if col_index == 0 else "")
            ax.set_xticks(sorted(panel["timestep"].dropna().unique().tolist()))
            upper = float(panel[f"{metric}_ci95_high"].max())
            ax.set_ylim(0, max(1.0, upper * 1.10))

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(3, len(legend_labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(title, fontsize=18, y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_society_phase_window_figure(
    round_frame: pd.DataFrame,
    *,
    output_path: Path,
    title: str,
) -> Path | None:
    summary_frame = summarize_society_phase_frame(round_frame)
    if summary_frame.empty:
        return None

    metrics = [
        ("phase_rounds", "Phase duration"),
        ("alive_count", "Alive agents"),
        ("public_food", "Public food"),
        ("public_water", "Public water"),
        ("average_health", "Average health"),
        ("average_energy", "Average energy"),
    ]
    phase_order = ["collapse", "plateau"]
    phase_colors = {"collapse": "#d62828", "plateau": "#2a9d8f"}
    group_rows = sorted(
        {
            (str(track), str(prompt_variant))
            for track, prompt_variant in summary_frame[["track", "prompt_variant"]].itertuples(index=False, name=None)
        },
        key=lambda item: (
            {"society": 0, "reputation": 1}.get(item[0], 99),
            {name: idx for idx, name in enumerate(SOCIETY_VARIANT_ORDER)}.get(item[1], 99),
            item,
        ),
    )

    configure_plot_style()
    fig, axes = plt.subplots(
        len(group_rows),
        len(metrics),
        figsize=(4.1 * len(metrics), 4.8 * len(group_rows)),
        squeeze=False,
    )

    for row_index, (track, prompt_variant) in enumerate(group_rows):
        panel = summary_frame[
            (summary_frame["track"] == track)
            & (summary_frame["prompt_variant"] == prompt_variant)
        ].copy()
        for col_index, (metric_name, metric_label) in enumerate(metrics):
            ax = axes[row_index][col_index]
            metric_panel = panel[panel[metric_name].notna()].copy()
            if metric_panel.empty:
                ax.axis("off")
                continue
            metric_panel["phase"] = pd.Categorical(metric_panel["phase"], categories=phase_order, ordered=True)
            metric_panel = metric_panel.sort_values("phase")
            x_positions = list(range(len(metric_panel)))
            values = metric_panel[metric_name].tolist()
            ci_lows = metric_panel[f"{metric_name}_ci95_low"].tolist()
            ci_highs = metric_panel[f"{metric_name}_ci95_high"].tolist()
            errors = [
                [value - low for value, low in zip(values, ci_lows, strict=False)],
                [high - value for value, high in zip(values, ci_highs, strict=False)],
            ]
            ax.bar(
                x_positions,
                values,
                color=[phase_colors.get(str(phase), "#6c757d") for phase in metric_panel["phase"].tolist()],
                yerr=errors,
                capsize=4,
                zorder=3,
            )
            ax.set_xticks(x_positions)
            ax.set_xticklabels([prettify_slug(str(phase)).replace("\n", " ") for phase in metric_panel["phase"].tolist()])
            ax.set_title(
                f"{track.title()}: {prettify_slug(prompt_variant).replace(chr(10), ' ')}"
                if col_index == 0
                else metric_label,
                fontsize=13,
            )
            ax.set_ylabel(metric_label if col_index == 0 else "")
            ax.set_xlabel("")
            upper = max(ci_highs) if ci_highs else max(values)
            ax.set_ylim(0, max(1.0, float(upper) * 1.15))

    fig.suptitle(title, fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    summary_module, summaries = load_summaries(args.results)
    experiment_frame, prompt_variant_frame, pooled_prompt_frame = build_frames_from_summaries(
        summary_module,
        summaries,
    )
    society_round_frame, society_event_frame = flatten_society_round_rows(summaries)
    society_model_frame = flatten_society_model_rows(summaries)
    society_agent_frame = flatten_society_agent_vital_rows(summaries)
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
        save_society_metric_figure(
            pooled_prompt_frame,
            metric_root="average_trade_volume",
            output_path=output_dir / "society_reputation_trade_volume.png",
            title="Scarcity and reputation trade volume by prompt condition",
            y_label="Average trade volume",
        ),
        save_society_metric_figure(
            pooled_prompt_frame,
            metric_root="alliance_count",
            output_path=output_dir / "society_reputation_alliance_count.png",
            title="Scarcity and reputation alliances by prompt condition",
            y_label="Average alliance count",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="alive_count",
            output_path=output_dir / "society_reputation_population_over_time.png",
            title="Population over time in the LLM societies",
            y_label="Alive agents",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="public_resources",
            output_path=output_dir / "society_reputation_public_resources_over_time.png",
            title="Public resources over time in the LLM societies",
            y_label="Public resources",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="public_food",
            output_path=output_dir / "society_reputation_public_food_over_time.png",
            title="Public food over time in the LLM societies",
            y_label="Public food",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="public_water",
            output_path=output_dir / "society_reputation_public_water_over_time.png",
            title="Public water over time in the LLM societies",
            y_label="Public water",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="trade_volume",
            output_path=output_dir / "society_reputation_trade_volume_over_time.png",
            title="Trade intensity over time in the LLM societies",
            y_label="Trade volume per round",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="average_health",
            output_path=output_dir / "society_reputation_health_over_time.png",
            title="Average health over time in the LLM societies",
            y_label="Average health",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="average_energy",
            output_path=output_dir / "society_reputation_energy_over_time.png",
            title="Average energy over time in the LLM societies",
            y_label="Average energy",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="birth_count",
            output_path=output_dir / "society_reputation_births_over_time.png",
            title="Birth events over time in the LLM societies",
            y_label="Births per round",
        ),
        save_society_timeline_figure(
            society_round_frame,
            metric_root="death_count",
            output_path=output_dir / "society_reputation_deaths_over_time.png",
            title="Death events over time in the LLM societies",
            y_label="Deaths per round",
        ),
        save_society_event_mix_figure(
            society_event_frame,
            output_path=output_dir / "society_reputation_behavior_mix.png",
            title="Behavior mix in the LLM societies",
        ),
        save_society_model_population_figure(
            society_model_frame,
            output_path=output_dir / "society_reputation_population_by_model_over_time.png",
            title="Population trajectories by model in the LLM societies",
        ),
        save_society_vitals_heatmap(
            society_agent_frame,
            output_path=output_dir / "society_reputation_survivor_vitals_heatmap.png",
            title="Latest observed survivor vitals in the LLM societies",
        ),
        save_society_model_vitals_timeline_figure(
            society_agent_frame,
            output_path=output_dir / "society_reputation_model_vitals_over_time.png",
            title="Model-level vital trajectories in the LLM societies",
        ),
        save_society_phase_window_figure(
            society_round_frame,
            output_path=output_dir / "society_reputation_phase_window_summary.png",
            title="Collapse and plateau phase summary in the LLM societies",
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
