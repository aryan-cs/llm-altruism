"""Render visual diagnostics used by the conference submission.

These figures combine the graph-independent summary tables with raw Part 2
trajectories. They are intentionally separated from the older per-part graph
scripts because several outputs span benchmark parts.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from data.graphs.part_2_graphs import (
    _model_bar_color,
    _model_family_name,
    _model_line_style,
    _model_sort_key,
)

TABLES_DIR = Path("data") / "analysis" / "tables"
OUTPUT_DIR = Path("data") / "graphs" / "paper_visuals"
MODEL_DECISIONS_PER_FULL_COMMONS_RUN = 5000
COMMONS_HORIZON_DAYS = 100
COMMONS_SOCIETY_SIZE = 50
LOW_SCORE_COLOR = "#d55e00"
HIGH_SCORE_COLOR = "#0072b2"
MISSING_SCORE_COLOR = "#e5e7eb"
NEUTRAL_SCORE_COLOR = "#f6f7fb"
DISPLAY_BLEND_AMOUNT = 0.38
FRAME_ORDER = ("self_direct", "advice", "observer_evaluation", "prediction")
FRAME_LABELS = {
    "self_direct": "Self-direct",
    "advice": "Advice",
    "observer_evaluation": "Observer eval.",
    "prediction": "Prediction",
}
GAME_ORDER = ("prisoners_dilemma", "temptation_or_commons")
GAME_LABELS = {
    "prisoners_dilemma": "Prisoner's dilemma",
    "temptation_or_commons": "Temptation / commons",
}


def _blend_hex(foreground: str, background: str = NEUTRAL_SCORE_COLOR, amount: float = DISPLAY_BLEND_AMOUNT) -> str:
    fg = np.array([int(foreground[index : index + 2], 16) for index in (1, 3, 5)], dtype=float)
    bg = np.array([int(background[index : index + 2], 16) for index in (1, 3, 5)], dtype=float)
    rgb = np.rint(fg * (1.0 - amount) + bg * amount).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


LOW_SCORE_DISPLAY_COLOR = _blend_hex(LOW_SCORE_COLOR)
HIGH_SCORE_DISPLAY_COLOR = _blend_hex(HIGH_SCORE_COLOR)
LOW_ACTION_COLOR = LOW_SCORE_DISPLAY_COLOR
HIGH_ACTION_COLOR = HIGH_SCORE_DISPLAY_COLOR


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def _model_label_for_style(model: str) -> str:
    return f"ollama/{model}"


def _short_model_label(model: str) -> str:
    replacements = {
        "gpt-oss-safeguard:20b": "gpt-oss-sg",
        "gurubot/gpt-oss-derestricted:20b": "gpt-oss-der",
        "huihui_ai/qwen2.5-abliterate:7b-instruct": "qwen2.5-abl-inst",
        "huihui_ai/qwen2.5-abliterate:7b": "qwen2.5-abl",
        "qwen2.5:7b-instruct": "qwen2.5-inst",
        "sorc/qwen3.5-instruct": "qwen3.5-inst",
        "sorc/qwen3.5-instruct-uncensored": "qwen3.5-inst-unc",
        "aratan/qwen3.5-uncensored:9b": "qwen3.5-unc",
    }
    return replacements.get(model, model)


def _model_order(models: Iterable[str]) -> list[str]:
    return sorted(models, key=lambda model: (*_model_sort_key(_model_label_for_style(model)), model))


def _family_legend_handles(models: Iterable[str]) -> list[object]:
    import matplotlib.patches as mpatches

    present: dict[str, str] = {}
    for model in models:
        style_label = _model_label_for_style(model)
        family = _model_family_name(style_label)
        present.setdefault(family, _model_bar_color(style_label))

    return [
        mpatches.Patch(facecolor=color, edgecolor="none", label=family)
        for family, color in present.items()
    ]


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    cmap = plt.colormaps["turbo"]
    return plt, cmap


def _heatmap_text_color(value: float) -> str:
    return "white" if value <= 25.0 or value >= 93.0 else "#111827"


def _frame_rates() -> dict[tuple[str, str], float]:
    rows = _read_rows(TABLES_DIR / "part1_dimension_summary.csv")
    rates: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["dimension"] != "frame":
            continue
        rates[(row["model"], row["value"])] = float(row["cooperation_rate"]) * 100.0
    return rates


def _part1_dimension_rates(dimension: str) -> dict[tuple[str, str], float]:
    rows = _read_rows(TABLES_DIR / "part1_dimension_summary.csv")
    rates: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["dimension"] != dimension:
            continue
        rates[(row["model"], row["value"])] = float(row["cooperation_rate"]) * 100.0
    return rates


def _part2_rows() -> dict[str, dict[str, str]]:
    return {
        row["model"]: row
        for row in _read_rows(TABLES_DIR / "part2_model_summary.csv")
    }


def render_frame_sensitivity_heatmap() -> Path:
    plt, cmap = _setup_matplotlib()
    frame_rates = _frame_rates()
    models = _model_order({model for model, _frame in frame_rates})
    matrix = np.asarray(
        [
            [frame_rates.get((model, frame), 0.0) for frame in FRAME_ORDER]
            for model in models
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(FRAME_ORDER)))
    ax.set_xticklabels([FRAME_LABELS[frame] for frame in FRAME_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_short_model_label(model) for model in models])
    ax.set_title("Prompt-semantics calibration across role instructions")
    ax.set_xlabel("Prompt frame")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(-0.5, len(FRAME_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                color=_heatmap_text_color(value),
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Cooperative action-label rate (%)")
    fig.tight_layout()
    output = OUTPUT_DIR / "frame_sensitivity_heatmap.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_part1_game_heatmap() -> Path:
    plt, cmap = _setup_matplotlib()
    rates = _part1_dimension_rates("game")
    models = _model_order({model for model, _game in rates})
    matrix = np.asarray(
        [
            [rates.get((model, game), 0.0) for game in GAME_ORDER]
            for model in models
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(GAME_ORDER)))
    ax.set_xticklabels([GAME_LABELS[game] for game in GAME_ORDER], rotation=25, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_short_model_label(model) for model in models])
    ax.set_title("Direct focal choices by game family")
    ax.set_xlabel("Game family")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(-0.5, len(GAME_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(
                col_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=9,
                color=_heatmap_text_color(value),
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Cooperation rate (%)")
    fig.tight_layout()
    output = OUTPUT_DIR / "part1_cooperation_by_game_heatmap.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def _part2_source_paths() -> dict[str, Path]:
    rows = _read_rows(TABLES_DIR / "part2_model_summary.csv")
    return {row["model"]: Path(row["csv_path"]) for row in rows}


def _agent_index(agent: str) -> int | None:
    match = re.search(r"(\d+)$", agent)
    if not match:
        return None
    index = int(match.group(1)) - 1
    if index < 0 or index >= COMMONS_SOCIETY_SIZE:
        return None
    return index


def _action_code(action: str) -> int | None:
    normalized = action.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in {"A", "OPTION_A", "ACTION_A", "RESTRAIN"}:
        return 1
    if normalized in {"B", "OPTION_B", "ACTION_B", "OVERUSE"}:
        return 0
    return None


def _part2_trajectories() -> tuple[list[str], dict[str, list[dict[str, float]]], np.ndarray]:
    paths = _part2_source_paths()
    models = _model_order(paths)
    trajectories: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    raster = np.full((len(models) * COMMONS_SOCIETY_SIZE, COMMONS_HORIZON_DAYS), 2, dtype=int)

    for model_index, model in enumerate(models):
        path = paths[model]
        rows = _read_rows(path)
        for row in rows:
            day = int(row["day"])
            if day < 1 or day > COMMONS_HORIZON_DAYS:
                continue
            action = _action_code(row["action"])
            agent_index = _agent_index(row["agent"])
            if action is not None and agent_index is not None:
                raster[model_index * COMMONS_SOCIETY_SIZE + agent_index, day - 1] = action

            if day not in trajectories[model]:
                total = int(row["restrain_count"]) + int(row["overuse_count"])
                capacity = int(row.get("resource_capacity") or 2500)
                trajectories[model][day] = {
                    "day": float(day),
                    "restraint_rate": (int(row["restrain_count"]) / total * 100.0) if total else 0.0,
                    "resource_units_remaining": float(row["resource_units_remaining"]),
                    "resource_capacity": float(capacity),
                    "resource_percent": float(row["resource_units_remaining"]) / capacity * 100.0,
                    "population": float(row["population_end"]),
                }

    ordered = {
        model: [trajectories[model][day] for day in sorted(trajectories[model])]
        for model in models
    }
    return models, ordered, raster


def render_agent_day_raster() -> Path:
    import matplotlib.patches as mpatches
    from matplotlib.colors import BoundaryNorm, ListedColormap

    plt, _cmap = _setup_matplotlib()
    models, _trajectories, raster = _part2_trajectories()
    cmap = ListedColormap([LOW_ACTION_COLOR, HIGH_ACTION_COLOR, MISSING_SCORE_COLOR])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.imshow(raster, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    centers = [index * COMMONS_SOCIETY_SIZE + (COMMONS_SOCIETY_SIZE - 1) / 2 for index in range(len(models))]
    ax.set_yticks(centers)
    ax.set_yticklabels([_short_model_label(model) for model in models])
    ax.set_xticks([0, 24, 49, 74, 99])
    ax.set_xticklabels(["1", "25", "50", "75", "100"])
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Model society")
    ax.set_title("Stored Part 2 tokens under the mismatched contract")
    for boundary in range(1, len(models)):
        ax.axhline(boundary * COMMONS_SOCIETY_SIZE - 0.5, color="white", linewidth=1.1)
    handles = [
        mpatches.Patch(color=HIGH_ACTION_COLOR, label="OPTION_A token"),
        mpatches.Patch(color=LOW_ACTION_COLOR, label="OPTION_B token"),
        mpatches.Patch(color=MISSING_SCORE_COLOR, label="No active decision"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.065), ncol=3, frameon=False)
    fig.tight_layout()
    output = OUTPUT_DIR / "part2_agent_day_raster.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_part2_line_chart(metric: str, ylabel: str, title: str, filename: str) -> Path:
    plt, _cmap = _setup_matplotlib()
    models, trajectories, _raster = _part2_trajectories()
    family_models: dict[str, list[str]] = defaultdict(list)
    for model in models:
        family_models[_model_family_name(_model_label_for_style(model))].append(model)
    families = list(family_models)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    for ax, family in zip(axes_flat, families):
        for model in family_models[family]:
            rows = trajectories[model]
            if not rows:
                continue
            style_label = _model_label_for_style(model)
            ax.plot(
                [row["day"] for row in rows],
                [row[metric] for row in rows],
                label=_short_model_label(model),
                color=_model_bar_color(style_label),
                linestyle=_model_line_style(style_label),
                linewidth=1.9,
                alpha=0.95,
            )
        ax.set_title(family, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(1, COMMONS_HORIZON_DAYS)
        ax.set_xticks([1, 25, 50, 75, 100])
        if metric in {"restraint_rate", "resource_percent"}:
            ax.set_ylim(0, 105)
            ax.set_yticks(range(0, 101, 20))
        elif metric == "population":
            ax.set_ylim(0, COMMONS_SOCIETY_SIZE + 2)
            ax.set_yticks(range(0, COMMONS_SOCIETY_SIZE + 1, 10))
        elif metric == "resource_units_remaining":
            ax.set_ylim(0, 2600)
        ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    for ax in axes_flat[len(families) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=14)
    fig.supxlabel("Simulation day", fontsize=12)
    fig.supylabel(ylabel, fontsize=12)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
    output = OUTPUT_DIR / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_part2_restraint_choice_heatmap() -> Path:
    plt, cmap = _setup_matplotlib()
    models, trajectories, _raster = _part2_trajectories()
    matrix = np.full((len(models), COMMONS_HORIZON_DAYS), np.nan, dtype=float)

    for model_index, model in enumerate(models):
        for row in trajectories[model]:
            day_index = int(row["day"]) - 1
            if 0 <= day_index < COMMONS_HORIZON_DAYS:
                matrix[model_index, day_index] = row["restraint_rate"]

    heatmap_cmap = cmap.copy()
    heatmap_cmap.set_bad(MISSING_SCORE_COLOR)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    image = ax.imshow(
        matrix,
        cmap=heatmap_cmap,
        vmin=0,
        vmax=100,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title("Stored OPTION_A token rate over time")
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Model society")
    ax.set_xticks([0, 24, 49, 74, 99])
    ax.set_xticklabels(["1", "25", "50", "75", "100"])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_short_model_label(model) for model in models])
    ax.set_xticks(np.arange(-0.5, COMMONS_HORIZON_DAYS, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Daily stored OPTION_A token rate (%)")
    fig.tight_layout()
    output = OUTPUT_DIR / "part2_restraint_choice_over_time.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_part2_restraint_bar() -> Path:
    plt, _cmap = _setup_matplotlib()
    rows = _part2_rows()
    models = _model_order(rows)
    values = [float(rows[model]["restraint_rate"]) * 100.0 for model in models]
    x_positions = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(
        x_positions,
        values,
        color=[_model_bar_color(_model_label_for_style(model)) for model in models],
        edgecolor="none",
        width=0.74,
    )
    for x_pos, value in zip(x_positions, values):
        label_y = min(104.5, value + 1.4)
        ax.text(x_pos, label_y, f"{value:.1f}", ha="center", va="bottom", fontsize=10, rotation=90)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_short_model_label(model) for model in models], rotation=45, ha="right")
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel("Stored OPTION_A token rate (%)")
    ax.set_title("Contract-mismatched Part 2 token records")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        handles=_family_legend_handles(models),
        title="Model family",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    output = OUTPUT_DIR / "part2_restraint_rate_by_model.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    outputs = [
        render_frame_sensitivity_heatmap(),
        render_part1_game_heatmap(),
        render_agent_day_raster(),
        render_part2_line_chart(
            "resource_units_remaining",
            "Shared reserve units",
            "Shared reserve over time by model",
            "part2_shared_reserve_over_time.png",
        ),
        render_part2_line_chart(
            "population",
            "Living population",
            "Living population over time by model",
            "part2_population_over_time.png",
        ),
        render_part2_restraint_bar(),
        render_part2_restraint_choice_heatmap(),
    ]
    for output in outputs:
        print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
