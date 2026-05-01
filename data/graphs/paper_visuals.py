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
    EDGE_COLOR,
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
LOW_SCORE_COLOR = "#f04357"
HIGH_SCORE_COLOR = "#0c9430"
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


def _blend_hex(foreground: str, background: str = NEUTRAL_SCORE_COLOR, amount: float = DISPLAY_BLEND_AMOUNT) -> str:
    fg = np.array([int(foreground[index : index + 2], 16) for index in (1, 3, 5)], dtype=float)
    bg = np.array([int(background[index : index + 2], 16) for index in (1, 3, 5)], dtype=float)
    rgb = np.rint(fg * (1.0 - amount) + bg * amount).astype(int)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


LOW_SCORE_DISPLAY_COLOR = _blend_hex(LOW_SCORE_COLOR)
HIGH_SCORE_DISPLAY_COLOR = _blend_hex(HIGH_SCORE_COLOR)


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
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "prosocial_score",
        [LOW_SCORE_DISPLAY_COLOR, NEUTRAL_SCORE_COLOR, HIGH_SCORE_DISPLAY_COLOR],
        N=256,
    )
    return plt, cmap


def _frame_rates() -> dict[tuple[str, str], float]:
    rows = _read_rows(TABLES_DIR / "part1_dimension_summary.csv")
    rates: dict[tuple[str, str], float] = {}
    for row in rows:
        if row["dimension"] != "frame":
            continue
        rates[(row["model"], row["value"])] = float(row["cooperation_rate"]) * 100.0
    return rates


def _cross_part_rows() -> dict[str, dict[str, str]]:
    return {
        row["model"]: row
        for row in _read_rows(TABLES_DIR / "cross_part_model_summary.csv")
    }


def _part2_rows() -> dict[str, dict[str, str]]:
    return {
        row["model"]: row
        for row in _read_rows(TABLES_DIR / "part2_model_summary.csv")
    }


def _fingerprint_matrix() -> tuple[list[str], list[str], np.ndarray]:
    cross = _cross_part_rows()
    part2 = _part2_rows()
    frame_rates = _frame_rates()
    models = _model_order(cross)
    columns = [
        "Safety refusal",
        "Cooperation",
        "Self-direct",
        "Advice",
        "Observer eval.",
        "Prediction",
        "Commons restraint",
        "No-depletion horizon",
        "Final population",
        "Final reserve",
    ]
    matrix: list[list[float]] = []
    for model in models:
        first_depletion = part2[model]["first_depletion_day"]
        depletion_score = (
            100.0
            if not first_depletion
            else max(0.0, min(100.0, float(first_depletion) / COMMONS_HORIZON_DAYS * 100.0))
        )
        resource_capacity = float(part2[model]["resource_capacity"])
        final_reserve = float(part2[model]["final_resource_units"]) / resource_capacity * 100.0
        row = [
            float(cross[model]["safety_refusal_rate"]) * 100.0,
            float(cross[model]["cooperation_rate"]) * 100.0,
            frame_rates.get((model, "self_direct"), 0.0),
            frame_rates.get((model, "advice"), 0.0),
            frame_rates.get((model, "observer_evaluation"), 0.0),
            frame_rates.get((model, "prediction"), 0.0),
            float(cross[model]["restraint_rate"]) * 100.0,
            depletion_score,
            float(cross[model]["final_population"]) / COMMONS_SOCIETY_SIZE * 100.0,
            final_reserve,
        ]
        matrix.append(row)
    return models, columns, np.asarray(matrix, dtype=float)


def render_behavioral_fingerprint_heatmap() -> Path:
    plt, cmap = _setup_matplotlib()
    models, columns, matrix = _fingerprint_matrix()

    fig, ax = plt.subplots(figsize=(12.8, 6.2))
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=35, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([_short_model_label(model) for model in models])
    ax.set_title("Behavioral fingerprint across benchmark axes")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(col_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=7, color="#111827")

    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Score (%)")
    fig.tight_layout()
    output = OUTPUT_DIR / "behavioral_fingerprint_heatmap.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_frame_sensitivity_heatmap() -> Path:
    plt, cmap = _setup_matplotlib()
    cross = _cross_part_rows()
    models = _model_order(cross)
    frame_rates = _frame_rates()
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
    ax.set_title("Frame sensitivity in dyadic cooperation")
    ax.set_xlabel("Prompt frame")
    ax.set_ylabel("Model")
    ax.set_xticks(np.arange(-0.5, len(FRAME_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            ax.text(col_index, row_index, f"{value:.0f}", ha="center", va="center", fontsize=7, color="#111827")

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Cooperation rate (%)")
    fig.tight_layout()
    output = OUTPUT_DIR / "frame_sensitivity_heatmap.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_model_behavior_pca() -> Path:
    plt, _cmap = _setup_matplotlib()
    models, _columns, matrix = _fingerprint_matrix()
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, ddof=1, keepdims=True)
    scale[scale == 0] = 1.0
    standardized = centered / scale
    _u, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ vt[:2].T
    variances = singular_values**2 / max(1, standardized.shape[0] - 1)
    explained = variances / variances.sum()

    label_offsets = {
        "gpt-oss:20b": (7, -12),
        "gpt-oss-safeguard:20b": (8, -4),
        "gurubot/gpt-oss-derestricted:20b": (-8, -18),
        "llama2": (7, -12),
        "llama2-uncensored": (8, 8),
        "qwen2.5:7b": (8, -16),
        "huihui_ai/qwen2.5-abliterate:7b": (-8, 12),
        "qwen2.5:7b-instruct": (8, 10),
        "huihui_ai/qwen2.5-abliterate:7b-instruct": (8, -15),
        "qwen3.5": (8, -6),
        "aratan/qwen3.5-uncensored:9b": (8, 11),
        "sorc/qwen3.5-instruct": (8, -11),
        "sorc/qwen3.5-instruct-uncensored": (7, -10),
    }

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for model, (x_pos, y_pos) in zip(models, scores[:, :2]):
        style_label = _model_label_for_style(model)
        x_offset, y_offset = label_offsets.get(model, (5, 4))
        ax.scatter(
            x_pos,
            y_pos,
            s=70,
            color=_model_bar_color(style_label),
            edgecolor=EDGE_COLOR,
            linewidth=0.7,
            zorder=3,
        )
        ax.annotate(
            _short_model_label(model),
            (x_pos, y_pos),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=7.5,
            ha="right" if x_offset < 0 else "left",
            va="top" if y_offset < 0 else "bottom",
        )

    ax.axhline(0, color="#9ca3af", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#9ca3af", linewidth=0.8, zorder=1)
    ax.margins(x=0.08, y=0.12)
    ax.grid(True, alpha=0.22, linewidth=0.7)
    ax.set_title("Model behavior map from benchmark fingerprints")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    handles = _family_legend_handles(models)
    ax.legend(handles=handles, title="Model family", loc="best", frameon=False)
    fig.tight_layout()
    output = OUTPUT_DIR / "model_behavior_pca.png"
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
    cmap = ListedColormap([LOW_SCORE_DISPLAY_COLOR, HIGH_SCORE_DISPLAY_COLOR, MISSING_SCORE_COLOR])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(12.5, 10.8))
    ax.imshow(raster, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    centers = [index * COMMONS_SOCIETY_SIZE + (COMMONS_SOCIETY_SIZE - 1) / 2 for index in range(len(models))]
    ax.set_yticks(centers)
    ax.set_yticklabels([_short_model_label(model) for model in models])
    ax.set_xticks([0, 24, 49, 74, 99])
    ax.set_xticklabels(["1", "25", "50", "75", "100"])
    ax.set_xlabel("Simulation day")
    ax.set_ylabel("Model society")
    ax.set_title("Agent-day action raster in the repeated commons")
    for boundary in range(1, len(models)):
        ax.axhline(boundary * COMMONS_SOCIETY_SIZE - 0.5, color="white", linewidth=1.1)
    handles = [
        mpatches.Patch(color=HIGH_SCORE_DISPLAY_COLOR, label="Restrain"),
        mpatches.Patch(color=LOW_SCORE_DISPLAY_COLOR, label="Overuse"),
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
    fig, ax = plt.subplots(figsize=(12.2, 6.8))
    for model in models:
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
    ax.set_title(title)
    ax.set_xlabel("Simulation day")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(1, COMMONS_HORIZON_DAYS)
    ax.set_xticks([1, 25, 50, 75, 100])
    if metric in {"restraint_rate", "resource_percent"}:
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 10))
    elif metric == "resource_units_remaining":
        ax.set_ylim(0, 2600)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)
    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    image = ax.imshow(
        matrix,
        cmap=heatmap_cmap,
        vmin=0,
        vmax=100,
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_title("Restraint choice over time by model")
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
    cbar.set_label("Daily restraint choice rate (%)")
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
    lows = [float(rows[model]["wilson_low"]) * 100.0 for model in models]
    highs = [float(rows[model]["wilson_high"]) * 100.0 for model in models]
    yerr = np.asarray(
        [
            [value - low for value, low in zip(values, lows)],
            [high - value for value, high in zip(values, highs)],
        ]
    )
    x_positions = np.arange(len(models))

    fig, ax = plt.subplots(figsize=(12.4, 6.6))
    ax.bar(
        x_positions,
        values,
        yerr=yerr,
        capsize=3,
        color=[_model_bar_color(_model_label_for_style(model)) for model in models],
        edgecolor="none",
        width=0.74,
    )
    for x_pos, value in zip(x_positions, values):
        ax.text(x_pos, min(103, value + 2.0), f"{value:.1f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_short_model_label(model) for model in models], rotation=45, ha="right")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 10))
    ax.set_ylabel("Restraint choice rate (%)")
    ax.set_title("Commons restraint rate by model")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=_family_legend_handles(models), title="Model family", loc="upper left", frameon=False)
    fig.tight_layout()
    output = OUTPUT_DIR / "part2_restraint_rate_by_model.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    outputs = [
        render_behavioral_fingerprint_heatmap(),
        render_model_behavior_pca(),
        render_frame_sensitivity_heatmap(),
        render_agent_day_raster(),
        render_part2_line_chart(
            "resource_units_remaining",
            "Shared reserve units",
            "Shared reserve over time by model",
            "part2_shared_reserve_over_time.png",
        ),
        render_part2_restraint_bar(),
        render_part2_restraint_choice_heatmap(),
    ]
    for output in outputs:
        print(f"wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
