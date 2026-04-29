"""Render cooperative-choice charts for part_1 benchmark outputs.

The default mode selects the latest non-empty part_1 CSV for each model from
``data/raw/part_1`` and combines them into a single comparison set.

For each model, the script computes:

- cooperative choice rate = cooperative / (cooperative + selfish)
- confidence interval around that rate

The cooperative action is derived from ``experiments/part1/part_1_prompt.json``
for each game, so mixed-game outputs can be compared on one common rate.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from experiments.misc.prompt_loader import load_prompt_config

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_PART_1_DIR = Path("data") / "raw" / "part_1"
DEFAULT_GRAPH_DIR = Path("data") / "graphs" / "part_1"
DEFAULT_CI_METHOD: Literal["wilson", "wald"] = "wilson"
MASTER_PLOTS_DIRNAME = "master-plots"
OVERALL_TOP_PADDING = 6.0
BREAKDOWN_TOP_PADDING = 10.0
HATCH_PALETTE: tuple[str, ...] = ("", "//", "xx", "..", "++", "||", "\\\\", "--")
GPT_OSS_MODEL_COLORS = {
    "standard": "#4fc9b0",
    "standard_instruct": "#3bbfa5",
    "safeguard": "#1f9e83",
    "unrestricted": "#7edcc8",
    "unrestricted_instruct": "#66d4bd",
}
QWEN25_MODEL_COLORS = {
    "standard": "#a47bd6",
    "standard_instruct": "#9468cc",
    "safeguard": "#7544b8",
    "unrestricted": "#bea0e2",
    "unrestricted_instruct": "#b18edc",
}
QWEN35_MODEL_COLORS = {
    "standard": "#cc7bd6",
    "standard_instruct": "#be68cc",
    "safeguard": "#a344b5",
    "unrestricted": "#dca0e5",
    "unrestricted_instruct": "#d48edf",
}
LLAMA_MODEL_COLORS = {
    "standard": "#7aade8",
    "standard_instruct": "#6a9fe2",
    "safeguard": "#4a84d2",
    "unrestricted": "#9ec3ef",
    "unrestricted_instruct": "#8db8ec",
}
FAMILY_COLOR_PALETTES = {
    "GPT-OSS": GPT_OSS_MODEL_COLORS,
    "Qwen 2.5": QWEN25_MODEL_COLORS,
    "Qwen 3.5": QWEN35_MODEL_COLORS,
    "Llama": LLAMA_MODEL_COLORS,
}
EDGE_COLOR = "#6b7280"
HATCH_COLOR = "#000000"
DEFAULT_FALLBACK_COLOR = "#b0b8c4"
MODEL_BAR_WIDTH = 0.62
OVERALL_FIG_HEIGHT = 8.0
BREAKDOWN_DIMENSIONS = ("game", "frame", "domain", "presentation")
PART_1_PROMPTS = load_prompt_config("part_1")

CiMethod = Literal["wilson", "wald"]
ScopeFilter = Literal["full", "subset", "smoke", "any"]


@dataclass
class Aggregate:
    """Per-model/per-breakdown count summary for one part_1 split."""

    cooperative: int = 0
    selfish: int = 0
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate part_1 cooperative-choice graphs with confidence intervals "
            "for model-level and breakdown-level rates."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--csv",
        nargs="+",
        help=(
            "One or more CSV filenames or paths inside data/raw/part_1 to combine. "
            "Example: --csv part1__ollama__gpt-oss-20b__full__20260420_005501.csv"
        ),
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest non-empty CSV for each model in data/raw/part_1.",
    )

    parser.add_argument(
        "--part1-dir",
        default=str(DEFAULT_PART_1_DIR),
        help="Directory containing part_1 CSV files (default: data/raw/part_1).",
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(DEFAULT_GRAPH_DIR),
        help="Directory for output graph files (default: data/graphs/part_1).",
    )
    parser.add_argument(
        "--scope",
        default="full",
        choices=["full", "subset", "smoke", "any"],
        help=(
            "When using latest discovery, limit source files by result scope. "
            "'full' is default; 'any' uses the latest non-empty file regardless of scope."
        ),
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix (defaults to the selected source set).",
    )
    parser.add_argument(
        "--confidence",
        default=0.95,
        type=float,
        help="Confidence level for intervals (default: 0.95).",
    )
    parser.add_argument(
        "--ci-method",
        default=DEFAULT_CI_METHOD,
        choices=["wilson", "wald"],
        help=(
            "Confidence interval method. 'wilson' is the default and preferred "
            "for proportions; 'wald' uses the historical normal approximation."
        ),
    )
    parser.add_argument(
        "--dimensions",
        nargs="*",
        choices=list(BREAKDOWN_DIMENSIONS),
        default=list(BREAKDOWN_DIMENSIONS),
        help=(
            "Breakdown charts to render in addition to the overall model chart. "
            "Defaults to all part_1 dimensions."
        ),
    )
    return parser.parse_args()


def z_from_confidence(confidence: float) -> float:
    """Convert a confidence level to a two-sided standard-normal z-score."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return statistics.NormalDist().inv_cdf((1.0 + confidence) / 2.0)


def cooperative_action_for_game(game_id: str) -> str | None:
    """Return the configured cooperative action label for a part_1 game."""
    game = PART_1_PROMPTS.get("games", {}).get(game_id)
    if not isinstance(game, dict):
        return None
    action_labels = game.get("action_labels", {})
    if not isinstance(action_labels, dict):
        return None
    cooperative = action_labels.get("cooperative")
    if isinstance(cooperative, str) and cooperative.strip():
        return cooperative.strip()
    return None


def parse_action_state(row: dict[str, str]) -> bool | None:
    """Parse a row action into cooperative/selfish/unknown.

    Returns:
        True when the chosen action is the configured cooperative action,
        False when the row contains a different non-empty action,
        None when the game/action is missing or unrecognized.
    """
    game = (row.get("game", "") or "").strip()
    action = (row.get("action", "") or "").strip()
    if not game or not action:
        return None

    cooperative_action = cooperative_action_for_game(game)
    if cooperative_action is None:
        return None

    if action == cooperative_action:
        return True

    allowed_actions = PART_1_PROMPTS.get("games", {}).get(game, {}).get("action_descriptions", {})
    if action in allowed_actions:
        return False
    return None


def model_label(row: dict[str, str]) -> str:
    """Build a stable model label from provider + model columns."""
    provider = (row.get("provider", "") or "unknown").strip()
    model = (row.get("model", "") or "unknown").strip()
    return f"{provider}/{model}"


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load rows from a CSV path produced by part_1."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(
    rows: Sequence[dict[str, str]],
    dimension: str,
) -> tuple[
    dict[str, Aggregate],
    dict[tuple[str, str], Aggregate],
]:
    """Aggregate counts by model and model+dimension."""
    overall: defaultdict[str, Aggregate] = defaultdict(Aggregate)
    by_dimension: defaultdict[tuple[str, str], Aggregate] = defaultdict(Aggregate)

    for row in rows:
        key = model_label(row)
        bucket_value = (row.get(dimension, "") or "unknown").strip()
        state = parse_action_state(row)

        agg = overall[key]
        dim_agg = by_dimension[(key, bucket_value)]

        if state is True:
            agg.cooperative += 1
            dim_agg.cooperative += 1
        elif state is False:
            agg.selfish += 1
            dim_agg.selfish += 1
        else:
            agg.skipped += 1
            dim_agg.skipped += 1

    return dict(overall), dict(by_dimension)


def _center_and_half_width_wilson(cooperative: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score center and half-width for binomial proportion."""
    if total == 0:
        raise ValueError("total must be > 0")

    p = cooperative / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = (z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / denominator

    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    center = (lower + upper) / 2
    half = (upper - lower) / 2
    return center, half


def _center_and_half_width_wald(cooperative: int, total: int, z: float) -> tuple[float, float]:
    """Wald/normal confidence interval center and half-width (legacy)."""
    if total == 0:
        raise ValueError("total must be > 0")

    p = cooperative / total
    half = z * math.sqrt(p * (1 - p) / total)
    lower = max(0.0, p - half)
    upper = min(1.0, p + half)
    center = (lower + upper) / 2
    half = (upper - lower) / 2
    return center, half


def cooperation_rate_and_error(
    agg: Aggregate,
    z: float,
    ci_method: CiMethod,
) -> tuple[float | None, float, int]:
    """Compute cooperative-choice percentage and CI half-width."""
    total = agg.cooperative + agg.selfish
    if total == 0:
        return None, 0.0, 0

    if ci_method == "wilson":
        center, half = _center_and_half_width_wilson(agg.cooperative, total, z)
    else:
        center, half = _center_and_half_width_wald(agg.cooperative, total, z)

    return center * 100, half * 100, total


def _y_axis_upper_bound(
    values: Sequence[float],
    errors: Sequence[float],
    top_padding: float,
) -> float:
    """Compute chart-safe y-axis max with dynamic headroom."""
    finite = [
        value + error
        for value, error in zip(values, errors)
        if not math.isnan(value) and not math.isnan(error)
    ]
    if not finite:
        return 100.0
    return min(110.0, max(finite) + top_padding)


def _apply_percent_axis_labels(ax: object, values: Sequence[float], errors: Sequence[float]) -> None:
    """Set percent ticks through 100 while leaving top headroom."""
    upper = _y_axis_upper_bound(values, errors, OVERALL_TOP_PADDING)
    ax.set_ylim(0.0, upper)
    ax.set_yticks(list(range(0, 101, 10)))


def _annotate_percent_bars(
    ax: object,
    xs: Sequence[float],
    ys: Sequence[float],
    yerrs: Sequence[float],
    *,
    rotate: float = 0.0,
) -> None:
    """Add percentage labels above bars and their error bars."""
    _, ymax = ax.get_ylim()
    for x_pos, rate, err in zip(xs, ys, yerrs):
        label = f"{rate:.1f}%"
        y_pos = min(rate + err + 1.0, ymax - 0.2)
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
            rotation=rotate,
        )


def _ci_label(ci_method: CiMethod, confidence: float) -> str:
    """Human-readable interval label for titles."""
    method = "Wilson" if ci_method == "wilson" else "Wald"
    return f"{method} {confidence:.0%} CI"


def _model_leaf(model_label: str) -> str:
    """Return the terminal model-name component without the provider prefix."""
    _, _, model_path = model_label.partition("/")
    model = model_path or model_label
    _, _, leaf = model.rpartition("/")
    return leaf or model


def _model_family_name(model_label: str) -> str:
    """Choose the display family used for color mapping."""
    lowered = _model_leaf(model_label).lower()
    if "qwen2.5" in lowered or "qwen2-5" in lowered:
        return "Qwen 2.5"
    if "qwen3.5" in lowered or "qwen3-5" in lowered:
        return "Qwen 3.5"
    if "llama" in lowered:
        return "Llama"
    return "GPT-OSS"


def _model_color_variant(model_label: str) -> str:
    """Choose the color shade within a model family."""
    lowered = model_label.lower()
    is_instruct = "instruct" in lowered
    is_unrestricted = (
        "uncensored" in lowered
        or "abliterate" in lowered
        or "derestricted" in lowered
    )
    if "safeguard" in lowered:
        return "safeguard"
    if is_unrestricted and is_instruct:
        return "unrestricted_instruct"
    if is_unrestricted:
        return "unrestricted"
    if is_instruct:
        return "standard_instruct"
    return "standard"


def _model_bar_color(model_label: str) -> str:
    """Choose a bar color based on model family and variant."""
    family = _model_family_name(model_label)
    variant = _model_color_variant(model_label)
    palette = FAMILY_COLOR_PALETTES.get(family)
    if palette is None:
        return DEFAULT_FALLBACK_COLOR
    return palette.get(variant, palette.get("standard", DEFAULT_FALLBACK_COLOR))


def _model_bar_edge_color(model_label: str) -> str:
    """Uniform thin gray border for all model bars."""
    return EDGE_COLOR


def _model_bar_line_width(model_label: str) -> float:
    """Uniform thin border width for all model bars."""
    return 0.4


def _model_bar_hatch(model_label: str) -> str:
    """No model-variant hatches — variants are distinguished by color shade and grouping."""
    return ""


def _apply_model_bar_styles(patches: Sequence[object], labels: Sequence[str]) -> None:
    """Apply hatches and instruct borders to model bars."""
    for patch, model in zip(patches, labels):
        patch.set_hatch(_model_bar_hatch(model))
        patch.set_edgecolor(_model_bar_edge_color(model))
        patch.set_linewidth(_model_bar_line_width(model))


def _model_legend_handles(model_labels: Sequence[str]) -> list[object]:
    """Build family/color and variant/style legend handles for visible models."""
    try:
        from matplotlib.patches import Patch
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    present_families = {_model_family_name(label) for label in model_labels}

    handles: list[object] = []
    for family, palette in FAMILY_COLOR_PALETTES.items():
        if family in present_families:
            handles.append(
                Patch(
                    facecolor=palette["standard"],
                    edgecolor=EDGE_COLOR,
                    label=family,
                )
            )

    return handles


def _legend_column_count(handles: Sequence[object], maximum: int = 4) -> int:
    """Choose a compact legend column count for the handles that are visible."""
    return max(1, min(maximum, len(handles)))


def _dimension_hatches(count: int) -> list[str]:
    """Return hatch patterns for grouped breakdown bars."""
    if count <= 0:
        return []
    return [HATCH_PALETTE[index % len(HATCH_PALETTE)] for index in range(count)]


def _dimension_legend_handles(dim_labels: Sequence[str], hatches: Sequence[str]) -> list[object]:
    """Build hatch-only legend entries for dimension categories."""
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=DEFAULT_FALLBACK_COLOR, edgecolor=HATCH_COLOR, hatch=hatches[i], label=label)
        for i, label in enumerate(dim_labels)
    ]


def build_overall_title() -> str:
    """Build a descriptive title for the overall chart."""
    return (
        "Cooperation Rate by Model\n"
        "Higher bars indicate more cooperative choices in social dilemmas."
    )


def build_dimension_title(*, dimension: str) -> str:
    """Build a descriptive title for one part_1 breakdown chart."""
    return (
        f"Cooperation Rate by Model and {dimension.capitalize()}\n"
        f"Bars compare cooperation rates across {dimension} categories."
    )


def _normalize_model_family(model_label: str) -> str:
    """Collapse variant suffixes to group related model families together."""
    _, _, model = model_label.rpartition("/")
    model = (model or model_label).strip().lower()

    if ":" in model:
        model_core, _ = model.split(":", 1)
    else:
        model_core = model

    core_parts = model_core.split("-")
    while core_parts and core_parts[-1] in {
        "instruct",
        "uncensored",
        "safeguard",
        "derestricted",
        "abliterate",
    }:
        core_parts.pop()

    collapsed_core = "-".join(core_parts) if core_parts else model_core
    return collapsed_core


def _model_family_category(model_label: str) -> int:
    """Return a coarse family variant rank for stable grouping order."""
    lowered = model_label.lower()
    if "safeguard" in lowered:
        return 1
    if "uncensored" in lowered or "abliterate" in lowered or "derestricted" in lowered:
        return 2
    return 0


def _model_instruct_group(model_label: str) -> int:
    """Separate instruct models from non-instruct for grouping."""
    return 1 if "instruct" in model_label.lower() else 0


def _model_sort_key(model_label: str) -> tuple[str, int, int]:
    """Build an ordering key used for deterministic model grouping."""
    category = _model_family_category(model_label)
    instruct_group = _model_instruct_group(model_label)
    return _normalize_model_family(model_label), instruct_group, category


def _model_plot_folder(model_label: str) -> str:
    """Return the subfolder name for a model-family plot bundle."""
    leaf = _model_leaf(model_label)
    lowered = leaf.lower()
    is_instruct = "instruct" in lowered

    if "gpt-oss" in lowered:
        size = "20b"
        if ":" in leaf:
            size = leaf.rsplit(":", 1)[1].lower()
        return f"gpt-oss:{size}-plots"
    if "llama2" in lowered:
        return "llama2-plots"
    if "qwen2.5" in lowered or "qwen2-5" in lowered:
        return "qwen2.5-instruct-plots" if is_instruct else "qwen2.5-plots"
    if "qwen3.5" in lowered or "qwen3-5" in lowered:
        return "qwen3.5-instruct-plots" if is_instruct else "qwen3.5-plots"

    safe = "".join(
        character if character.isalnum() or character in ".:-_" else "-"
        for character in lowered
    ).strip("-")
    return f"{safe or 'unknown'}-plots"


def build_model_plot_groups(model_order: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group sorted model labels into output plot folders."""
    grouped: dict[str, list[str]] = {}
    for model in model_order:
        folder = _model_plot_folder(model)
        grouped.setdefault(folder, []).append(model)
    return list(grouped.items())


def filter_model_rows(
    labels: Sequence[str],
    rates: Sequence[float],
    errors: Sequence[float],
    totals: Sequence[int],
    keep_labels: Sequence[str],
) -> tuple[list[str], list[float], list[float], list[int]]:
    """Filter precomputed model rows while preserving a requested order."""
    rows_by_model = {
        model: (rate, error, total)
        for model, rate, error, total in zip(labels, rates, errors, totals)
    }
    filtered_labels: list[str] = []
    filtered_rates: list[float] = []
    filtered_errors: list[float] = []
    filtered_totals: list[int] = []

    for model in keep_labels:
        row = rows_by_model.get(model)
        if row is None:
            continue
        rate, error, total = row
        filtered_labels.append(model)
        filtered_rates.append(rate)
        filtered_errors.append(error)
        filtered_totals.append(total)

    return filtered_labels, filtered_rates, filtered_errors, filtered_totals


def _grouped_bar_positions(labels: Sequence[str]) -> list[float]:
    """Compute evenly spaced x positions for any number of bars."""
    return [float(i) for i in range(len(labels))]


def _apply_bar_xlim(
    ax: object,
    positions: Sequence[float],
    *,
    bar_width: float = MODEL_BAR_WIDTH,
) -> None:
    """Set x-axis limits so the margin on each side equals the gap between bars."""
    if not positions:
        return
    step = (positions[1] - positions[0]) if len(positions) >= 2 else 1.0
    padding = step - bar_width / 2
    ax.set_xlim(min(positions) - padding, max(positions) + padding)


def _place_legend_and_adjust(
    fig: object,
    *,
    handles: Sequence[object] | None = None,
    title: str,
    ncol: int,
    left: float = 0.14,
    right: float = 0.99,
    bottom: float = 0.26,
    gap: float = 0.01,
) -> None:
    """Place a legend between the suptitle and the axes, adjusting top margin dynamically."""
    fig.canvas.draw()
    suptitle_obj = fig._suptitle
    if suptitle_obj is not None:
        title_bbox = suptitle_obj.get_window_extent().transformed(
            fig.transFigure.inverted()
        )
        legend_top = title_bbox.y0 - gap
    else:
        legend_top = 0.97

    kwargs: dict[str, object] = dict(
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_top),
        ncol=ncol,
        frameon=False,
    )
    if handles is not None:
        kwargs["handles"] = handles
    legend = fig.legend(**kwargs)
    fig.canvas.draw()
    legend_bbox = legend.get_window_extent().transformed(
        fig.transFigure.inverted()
    )
    top = max(0.5, legend_bbox.y0 - gap)
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)


def build_model_rows(
    overall: dict[str, Aggregate],
    z: float,
    ci_method: CiMethod,
) -> tuple[list[str], list[float], list[float], list[int]]:
    """Build sorted model arrays for plotting from aggregated totals."""
    data: list[tuple[str, float, float, int]] = []
    for model, agg in overall.items():
        rate, err, total = cooperation_rate_and_error(agg, z, ci_method)
        if rate is None:
            continue
        data.append((model, rate, err, total))

    data.sort(key=lambda item: (*_model_sort_key(item[0]), -item[1], item[0]))
    labels = [item[0] for item in data]
    rates = [item[1] for item in data]
    errors = [item[2] for item in data]
    totals = [item[3] for item in data]
    return labels, rates, errors, totals


MASTER_PLOTS_DIRNAME = "master-plots"


def _model_plot_folder(model_label: str) -> str:
    """Return the subfolder name for a model-family plot bundle."""
    leaf = _model_leaf(model_label)
    lowered = leaf.lower()
    is_instruct = "instruct" in lowered

    if "gpt-oss" in lowered:
        size = "20b"
        if ":" in leaf:
            size = leaf.rsplit(":", 1)[1].lower()
        return f"gpt-oss:{size}-plots"
    if "llama2" in lowered:
        return "llama2-plots"
    if "qwen2.5" in lowered or "qwen2-5" in lowered:
        return "qwen2.5-instruct-plots" if is_instruct else "qwen2.5-plots"
    if "qwen3.5" in lowered or "qwen3-5" in lowered:
        return "qwen3.5-instruct-plots" if is_instruct else "qwen3.5-plots"

    safe = "".join(
        character if character.isalnum() or character in ".:-_" else "-"
        for character in lowered
    ).strip("-")
    return f"{safe or 'unknown'}-plots"


def build_model_plot_groups(model_order: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group sorted model labels into output plot folders."""
    grouped: dict[str, list[str]] = {}
    for model in model_order:
        folder = _model_plot_folder(model)
        grouped.setdefault(folder, []).append(model)
    return list(grouped.items())


def filter_model_rows(
    labels: Sequence[str],
    rates: Sequence[float],
    errors: Sequence[float],
    totals: Sequence[int],
    keep_labels: Sequence[str],
) -> tuple[list[str], list[float], list[float], list[int]]:
    """Filter precomputed model rows while preserving a requested order."""
    rows_by_model = {
        model: (rate, error, total)
        for model, rate, error, total in zip(labels, rates, errors, totals)
    }
    filtered_labels: list[str] = []
    filtered_rates: list[float] = []
    filtered_errors: list[float] = []
    filtered_totals: list[int] = []

    for model in keep_labels:
        row = rows_by_model.get(model)
        if row is None:
            continue
        rate, error, total = row
        filtered_labels.append(model)
        filtered_rates.append(rate)
        filtered_errors.append(error)
        filtered_totals.append(total)

    return filtered_labels, filtered_rates, filtered_errors, filtered_totals


def render_overall_chart(
    labels: Sequence[str],
    rates: Sequence[float],
    errors: Sequence[float],
    totals: Sequence[int],
    *,
    title: str,
    output: Path,
) -> None:
    """Render and save the overall model cooperation bar chart."""
    del totals
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    x = _grouped_bar_positions(labels)
    bar_colors = [_model_bar_color(model) for model in labels]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), OVERALL_FIG_HEIGHT))

    ax.bar(
        x,
        rates,
        yerr=errors,
        capsize=4,
        width=MODEL_BAR_WIDTH,
        alpha=1.0,
        color=bar_colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
    )
    _apply_model_bar_styles(ax.patches, labels)
    _annotate_percent_bars(ax, x, rates, errors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _apply_bar_xlim(ax, x)
    _apply_percent_axis_labels(ax, rates, errors)
    ax.set_ylabel("Cooperative choice rate (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, y=0.985)
    legend_handles = _model_legend_handles(labels)
    _place_legend_and_adjust(
        fig,
        handles=legend_handles,
        title="Model family / style",
        ncol=_legend_column_count(legend_handles),
        left=0.14,
        right=0.99,
        bottom=0.26,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _dimension_display_label(dimension: str, value: str) -> str:
    """Map config ids to human-readable labels for legends."""
    section_name = f"{dimension}s"
    section = PART_1_PROMPTS.get(section_name, {})
    entry = section.get(value, {})
    if isinstance(entry, dict):
        label = entry.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return value


def ordered_dimension_values(
    dimension: str,
    by_dimension: dict[tuple[str, str], Aggregate],
) -> list[str]:
    """Return a stable display order for chart breakdown categories."""
    available = {value for _, value in by_dimension.keys()}
    defaults = PART_1_PROMPTS.get("defaults", {}).get(f"{dimension}s", [])
    ordered = [value for value in defaults if value in available]
    remainder = sorted(available - set(ordered))
    return ordered + remainder


def render_dimension_chart(
    by_dimension: dict[tuple[str, str], Aggregate],
    z: float,
    ci_method: CiMethod,
    model_order: Sequence[str],
    *,
    dimension: str,
    title: str,
    output: Path,
) -> None:
    """Render and save a grouped chart for one part_1 dimension."""
    available_models = {model for model, _ in by_dimension.keys()}
    models = [model for model in model_order if model in available_models]
    dimension_values = ordered_dimension_values(dimension, by_dimension)
    if not models or not dimension_values:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    import matplotlib as mpl
    mpl.rcParams["hatch.color"] = HATCH_COLOR
    mpl.rcParams["hatch.linewidth"] = 0.5

    width = 0.75 / len(dimension_values)
    model_positions = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.6), 8))
    dimension_hatches = _dimension_hatches(len(dimension_values))
    dim_labels = [_dimension_display_label(dimension, v) for v in dimension_values]

    all_rates: list[float] = []
    all_errors: list[float] = []

    for dim_idx, value in enumerate(dimension_values):
        xs: list[float] = []
        ys: list[float] = []
        yerrs: list[float] = []
        bar_colors: list[str] = []

        for model_idx, model in enumerate(models):
            agg = by_dimension.get((model, value), Aggregate())
            rate, err, _ = cooperation_rate_and_error(agg, z, ci_method)
            if rate is None:
                continue

            xs.append(model_idx + (dim_idx - len(dimension_values) / 2 + 0.5) * width)
            ys.append(rate)
            yerrs.append(err)
            bar_colors.append(_model_bar_color(model))
            all_rates.append(rate)
            all_errors.append(err)

        bar_models = [
            model
            for model in models
            if cooperation_rate_and_error(
                by_dimension.get((model, value), Aggregate()),
                z,
                ci_method,
            )[0]
            is not None
        ]
        bars = ax.bar(
            xs,
            ys,
            width=width,
            yerr=yerrs,
            capsize=3,
            label=dim_labels[dim_idx],
            color=bar_colors,
            alpha=1.0,
            edgecolor=EDGE_COLOR,
            linewidth=0.25,
            hatch=dimension_hatches[dim_idx % len(dimension_hatches)],
        )
        for patch, model in zip(bars.patches, bar_models):
            patch.set_edgecolor(_model_bar_edge_color(model))
            patch.set_linewidth(_model_bar_line_width(model))
        _annotate_percent_bars(ax, xs, ys, yerrs, rotate=90)

    ax.set_xticks(model_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    if model_positions:
        group_width = len(dimension_values) * width
        edge_pad = 1.0 - group_width / 2
        ax.set_xlim(model_positions[0] - edge_pad, model_positions[-1] + edge_pad)
    upper = _y_axis_upper_bound(all_rates, all_errors, BREAKDOWN_TOP_PADDING)
    ax.set_ylim(0.0, upper)
    ax.set_yticks(list(range(0, 101, 10)))
    ax.set_ylabel("Cooperative choice rate (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, y=0.985)
    legend_handles = _model_legend_handles(models) + _dimension_legend_handles(dim_labels, dimension_hatches)
    _place_legend_and_adjust(
        fig,
        handles=legend_handles,
        title=f"Model / {dimension.capitalize()}",
        ncol=_legend_column_count(legend_handles),
        left=0.08,
        right=0.99,
        bottom=0.18,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _scope_matches(scope_label: str, scope_filter: ScopeFilter) -> bool:
    """Return whether a filename scope matches the requested filter."""
    if scope_filter == "any":
        return True
    if scope_filter == "full":
        return scope_label == "full"
    if scope_filter == "subset":
        return scope_label.startswith("subset-")
    if scope_filter == "smoke":
        return scope_label.startswith("smoke-")
    return False


def _parse_csv_metadata_from_name(path: Path) -> tuple[str, datetime] | None:
    """Extract scope + timestamp from a part_1 result filename."""
    stem_parts = path.stem.split("__")
    if len(stem_parts) != 5 or stem_parts[0] != "part1":
        return None
    scope_label = stem_parts[3]
    try:
        timestamp = datetime.strptime(stem_parts[4], TIMESTAMP_FORMAT)
    except ValueError:
        return None
    return scope_label, timestamp


def _is_interrupted_run(path: Path) -> bool:
    return path.with_name(f"{path.stem}_meta.json").exists()


def find_latest_nonempty_csvs_by_model(
    directory: Path,
    *,
    scope: ScopeFilter = "full",
) -> list[Path]:
    """Select the latest non-empty part_1 CSV for each model."""
    candidates: dict[str, tuple[datetime, Path]] = {}

    for path in sorted(directory.glob("*.csv")):
        parsed = _parse_csv_metadata_from_name(path)
        if parsed is None:
            continue
        scope_label, timestamp = parsed
        if not _scope_matches(scope_label, scope):
            continue
        if _is_interrupted_run(path):
            continue

        rows = read_rows(path)
        if not rows:
            continue

        model = model_label(rows[0])
        previous = candidates.get(model)
        if previous is None or timestamp > previous[0]:
            candidates[model] = (timestamp, path)

    if not candidates:
        raise ValueError(f"No non-empty part_1 CSV files found in {directory} for scope={scope}.")

    return [path for _, path in sorted(candidates.values(), key=lambda item: item[0])]


def resolve_csv_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve the input CSV set from either explicit files or latest discovery."""
    part1_dir = Path(args.part1_dir)
    if args.csv:
        selected: list[Path] = []
        for item in args.csv:
            path = Path(item)
            if not path.is_absolute():
                path = part1_dir / item
            selected.append(path)
        return selected

    return find_latest_nonempty_csvs_by_model(part1_dir, scope=args.scope)


def default_out_prefix(paths: Sequence[Path], *, scope: ScopeFilter) -> str:
    """Choose a stable default output prefix for the selected input set."""
    if len(paths) == 1:
        return paths[0].stem
    return f"part1_latest_{scope}_per_model"


def render_chart_bundle(
    *,
    rows: Sequence[dict[str, str]],
    labels: Sequence[str],
    rates: Sequence[float],
    errors: Sequence[float],
    totals: Sequence[int],
    z: float,
    ci_method: CiMethod,
    dimensions: Sequence[str],
    output_dir: Path,
    prefix: str,
    source_note: str,
    ci_label: str,
) -> list[Path]:
    """Render the selected part_1 chart set into one output directory."""
    written: list[Path] = []
    if not labels:
        return written

    overall_output = output_dir / f"{prefix}_cooperation_by_model.png"
    render_overall_chart(
        labels,
        rates,
        errors,
        totals,
        title=build_overall_title(source_note=source_note, ci_label=ci_label),
        output=overall_output,
    )
    written.append(overall_output)

    for dimension in dimensions:
        _, by_dimension = aggregate_rows(rows, dimension)
        dimension_output = output_dir / f"{prefix}_cooperation_by_model_and_{dimension}.png"
        render_dimension_chart(
            by_dimension,
            z,
            ci_method,
            labels,
            dimension=dimension,
            title=build_dimension_title(
                dimension=dimension,
                source_note=source_note,
                ci_label=ci_label,
            ),
            output=dimension_output,
        )
        written.append(dimension_output)

    return written


def main() -> int:
    """Entry point for `uv run python data/graphs/part_1_graphs.py ...`."""
    args = parse_args()
    csv_paths = resolve_csv_paths(args)

    rows: list[dict[str, str]] = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        rows.extend(read_rows(path))

    if not rows:
        raise ValueError("No rows found in the selected part_1 CSV files.")

    overall, _ = aggregate_rows(rows, "game")
    z = z_from_confidence(args.confidence)
    ci_method: CiMethod = args.ci_method

    labels, rates, errors, totals = build_model_rows(overall, z, ci_method)
    if not labels:
        raise ValueError("No model rows with judged cooperative/selfish actions were found.")

    graphs_dir = Path(args.graphs_dir)
    prefix = args.out_prefix or default_out_prefix(csv_paths, scope=args.scope)

    # --- master-plots (all models) ---
    master_dir = graphs_dir / MASTER_PLOTS_DIRNAME

    overall_output = master_dir / f"{prefix}_cooperation_by_model.png"
    render_overall_chart(
        labels,
        rates,
        errors,
        totals,
        title=build_overall_title(),
        output=overall_output,
    )
    for output in master_outputs:
        print(f"wrote: {output}")

    for dimension in args.dimensions:
        _, by_dimension = aggregate_rows(rows, dimension)
        dimension_output = master_dir / f"{prefix}_cooperation_by_model_and_{dimension}.png"
        render_dimension_chart(
            by_dimension,
            z,
            ci_method,
            labels,
            dimension=dimension,
            title=build_dimension_title(dimension=dimension),
            output=dimension_output,
        )
        for output in group_outputs:
            print(f"wrote: {output}")

    # --- per-family model-specific plots ---
    for folder_name, group_models in build_model_plot_groups(labels):
        group_labels, group_rates, group_errors, group_totals = filter_model_rows(
            labels, rates, errors, totals, group_models,
        )
        group_dir = graphs_dir / folder_name

        group_overall = group_dir / f"{prefix}_cooperation_by_model.png"
        render_overall_chart(
            group_labels,
            group_rates,
            group_errors,
            group_totals,
            title=build_overall_title(),
            output=group_overall,
        )
        print(f"wrote: {group_overall}")

        for dimension in args.dimensions:
            _, by_dimension = aggregate_rows(rows, dimension)
            group_dim = group_dir / f"{prefix}_cooperation_by_model_and_{dimension}.png"
            render_dimension_chart(
                by_dimension,
                z,
                ci_method,
                group_labels,
                dimension=dimension,
                title=build_dimension_title(dimension=dimension),
                output=group_dim,
            )
            print(f"wrote: {group_dim}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
