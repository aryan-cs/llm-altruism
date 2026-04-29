"""Render society-dynamics charts for part_2 benchmark outputs.

The default mode selects the latest non-empty part_2 CSV for each model from
``data/raw/part_2`` and combines them into a single comparison set.

For each model, the script computes:

- restraint choice rate = RESTRAIN / (RESTRAIN + OVERUSE)
- confidence interval around that rate
- per-day restraint, resource, population, and death trajectories
- final resource, final population, and first reserve-depletion day summaries

The chart styling intentionally matches the part_0 and part_1 graph modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_PART_2_DIR = Path("data") / "raw" / "part_2"
DEFAULT_GRAPH_DIR = Path("data") / "graphs" / "part_2"
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
LINE_STYLES: tuple[str, ...] = ("-", "--", ":", "-.")
PART_2_CHARTS = (
    "restraint",
    "restraint-by-day",
    "resource-by-day",
    "population-by-day",
    "deaths-by-day",
    "final-resource",
    "final-population",
    "collapse-day",
)
TIME_SERIES_SPECS: dict[str, tuple[str, str, str]] = {
    "restraint-by-day": (
        "Restraint-choice rate",
        "Restraint choice rate (%)",
        "restraint_by_day",
    ),
    "resource-by-day": (
        "Shared reserve remaining",
        "Resource remaining (%)",
        "resource_by_day",
    ),
    "population-by-day": (
        "Population remaining",
        "Population remaining (%)",
        "population_by_day",
    ),
    "deaths-by-day": (
        "Daily deaths",
        "Deaths",
        "deaths_by_day",
    ),
}
RESULT_HEADERS = [
    "provider",
    "model",
    "day",
    "agent",
    "action",
    "reasoning",
    "population_start",
    "population_end",
    "restrain_count",
    "overuse_count",
    "resource_units_remaining",
    "resource_capacity",
    "deaths",
    "resource",
    "selfish_gain",
    "depletion_units",
    "community_benefit",
]
LEGACY_RESULT_HEADERS = [
    "day",
    "agent",
    "action",
    "reasoning",
    "population_start",
    "population_end",
    "restrain_count",
    "overuse_count",
    "resource_units_remaining",
    "deaths",
]
ACTION_ALIASES = {
    "A": "RESTRAIN",
    "OPTION_A": "RESTRAIN",
    "ACTION_A": "RESTRAIN",
    "RESTRAIN": "RESTRAIN",
    "B": "OVERUSE",
    "OPTION_B": "OVERUSE",
    "ACTION_B": "OVERUSE",
    "OVERUSE": "OVERUSE",
}

CiMethod = Literal["wilson", "wald"]
Part2Chart = Literal[
    "restraint",
    "restraint-by-day",
    "resource-by-day",
    "population-by-day",
    "deaths-by-day",
    "final-resource",
    "final-population",
    "collapse-day",
]


@dataclass
class Aggregate:
    """Per-model count summary for part_2 actions."""

    restrain: int = 0
    overuse: int = 0
    skipped: int = 0


@dataclass(frozen=True)
class CsvMetadata:
    """Metadata parsed from a part_2 result filename."""

    provider_slug: str
    model_slug: str
    society_size: int
    days_label: str
    resource: str
    timestamp: datetime


@dataclass(frozen=True)
class DaySummary:
    """One model's repeated day-level state summary."""

    model: str
    day: int
    population_start: int
    population_end: int
    restrain_count: int
    overuse_count: int
    resource_units_remaining: int | None
    resource_capacity: int | None
    deaths: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate part_2 society-dynamics graphs with confidence intervals "
            "and model-level time series."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--csv",
        nargs="+",
        help=(
            "One or more CSV filenames or paths inside data/raw/part_2 to combine. "
            "Example: --csv part2__ollama__gpt-oss-20b__n50__d100__water__20260426_161639.csv"
        ),
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest non-empty CSV for each model in data/raw/part_2.",
    )

    parser.add_argument(
        "--part2-dir",
        default=str(DEFAULT_PART_2_DIR),
        help="Directory containing part_2 CSV files (default: data/raw/part_2).",
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(DEFAULT_GRAPH_DIR),
        help="Directory for output graph subfolders (default: data/graphs/part_2).",
    )
    parser.add_argument(
        "--society-size",
        type=int,
        default=None,
        help="When using latest discovery, only include filenames with this society size.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "When using latest discovery, only include filenames with this day count. "
            "Use 0 for open-ended runs."
        ),
    )
    parser.add_argument(
        "--resource",
        default=None,
        help="When using latest discovery, only include filenames with this resource slug.",
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
        "--charts",
        nargs="*",
        choices=list(PART_2_CHARTS),
        default=list(PART_2_CHARTS),
        help="Charts to render. Defaults to all part_2 chart types.",
    )
    return parser.parse_args()


def z_from_confidence(confidence: float) -> float:
    """Convert a confidence level to a two-sided standard-normal z-score."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return statistics.NormalDist().inv_cdf((1.0 + confidence) / 2.0)


def parse_action_state(row: dict[str, str]) -> bool | None:
    """Parse a row action into restrain/overuse/unknown.

    Returns:
        True when the chosen action is RESTRAIN,
        False when the chosen action is OVERUSE,
        None when the action is missing or unrecognized.
    """
    raw_action = (row.get("action", "") or "").strip().upper().replace("-", "_").replace(" ", "_")
    action = ACTION_ALIASES.get(raw_action, raw_action)
    if action == "RESTRAIN":
        return True
    if action == "OVERUSE":
        return False
    return None


def model_label(row: dict[str, str]) -> str:
    """Build a stable model label from provider + model columns."""
    provider = (row.get("provider", "") or "unknown").strip()
    model = (row.get("model", "") or "unknown").strip()
    return f"{provider}/{model}"


def _parse_int(raw: str | None) -> int | None:
    """Parse an integer field, returning None for missing/invalid values."""
    try:
        value = (raw or "").strip()
        if not value:
            return None
        return int(value)
    except ValueError:
        return None


def _resource_capacity_from_row(row: dict[str, str]) -> int | None:
    capacity = _parse_int(row.get("resource_capacity"))
    if capacity is not None:
        return capacity

    population_start = _parse_int(row.get("population_start"))
    resource_units = _parse_int(row.get("resource_units_remaining"))
    if population_start is None or resource_units is None:
        return None
    return max(resource_units, population_start)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load rows from a CSV path produced by part_2."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        source_header = list(reader.fieldnames)
        valid_headers = {tuple(RESULT_HEADERS), tuple(LEGACY_RESULT_HEADERS)}
        if tuple(source_header) not in valid_headers:
            raise ValueError(
                f"Unexpected CSV header for {csv_path}: expected {RESULT_HEADERS}, found {source_header}."
            )
        return [
            {column: row.get(column, "") or "" for column in RESULT_HEADERS}
            for row in reader
        ]


def aggregate_rows(
    rows: Sequence[dict[str, str]],
) -> tuple[dict[str, Aggregate], dict[tuple[str, int], DaySummary]]:
    """Aggregate counts by model and one day summary per model/day."""
    overall: defaultdict[str, Aggregate] = defaultdict(Aggregate)
    day_summaries: dict[tuple[str, int], DaySummary] = {}

    for row in rows:
        key = model_label(row)
        state = parse_action_state(row)
        agg = overall[key]

        if state is True:
            agg.restrain += 1
        elif state is False:
            agg.overuse += 1
        else:
            agg.skipped += 1

        day = _parse_int(row.get("day"))
        if day is None:
            continue
        summary_key = (key, day)
        if summary_key in day_summaries:
            continue

        population_start = _parse_int(row.get("population_start"))
        population_end = _parse_int(row.get("population_end"))
        restrain_count = _parse_int(row.get("restrain_count"))
        overuse_count = _parse_int(row.get("overuse_count"))
        deaths = _parse_int(row.get("deaths"))
        if (
            population_start is None
            or population_end is None
            or restrain_count is None
            or overuse_count is None
            or deaths is None
        ):
            continue

        day_summaries[summary_key] = DaySummary(
            model=key,
            day=day,
            population_start=population_start,
            population_end=population_end,
            restrain_count=restrain_count,
            overuse_count=overuse_count,
            resource_units_remaining=_parse_int(row.get("resource_units_remaining")),
            resource_capacity=_resource_capacity_from_row(row),
            deaths=deaths,
        )

    return dict(overall), day_summaries


def _center_and_half_width_wilson(restrain: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score center and half-width for binomial proportion."""
    if total == 0:
        raise ValueError("total must be > 0")

    p = restrain / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = (z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / denominator

    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    center = (lower + upper) / 2
    half = (upper - lower) / 2
    return center, half


def _center_and_half_width_wald(restrain: int, total: int, z: float) -> tuple[float, float]:
    """Wald/normal confidence interval center and half-width (legacy)."""
    if total == 0:
        raise ValueError("total must be > 0")

    p = restrain / total
    half = z * math.sqrt(p * (1 - p) / total)
    lower = max(0.0, p - half)
    upper = min(1.0, p + half)
    center = (lower + upper) / 2
    half = (upper - lower) / 2
    return center, half


def restraint_rate_and_error(
    agg: Aggregate,
    z: float,
    ci_method: CiMethod,
) -> tuple[float | None, float, int]:
    """Compute restraint-choice percentage and CI half-width."""
    total = agg.restrain + agg.overuse
    if total == 0:
        return None, 0.0, 0

    if ci_method == "wilson":
        center, half = _center_and_half_width_wilson(agg.restrain, total, z)
    else:
        center, half = _center_and_half_width_wald(agg.restrain, total, z)

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


def _annotate_value_bars(
    ax: object,
    xs: Sequence[float],
    ys: Sequence[float],
    labels: Sequence[str],
    *,
    rotate: float = 0.0,
) -> None:
    """Add arbitrary value labels above bars."""
    _, ymax = ax.get_ylim()
    for x_pos, value, label in zip(xs, ys, labels):
        y_pos = min(value + (ymax * 0.015), ymax - (ymax * 0.005))
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


def _palette_for_count(count: int) -> list[str]:
    """Return visibly separated teal shades instead of adjacent near-matches."""
    if count <= 0:
        return []
    if count <= len(TEAL_PALETTE):
        if count == 1:
            return [TEAL_PALETTE[2]]
        step = (len(TEAL_PALETTE) - 1) / (count - 1)
        indices = [round(index * step) for index in range(count)]
        return [TEAL_PALETTE[index] for index in indices]
    return [TEAL_PALETTE[index % len(TEAL_PALETTE)] for index in range(count)]


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


def _model_bar_edge_color(model_label: str) -> str:
    """No outline for model bars."""
    return "none"


def _model_bar_line_width(model_label: str) -> float:
    """No outline for model bars."""
    return 0.0


def _model_bar_hatch(model_label: str) -> str:
    """No model-variant hatches — variants are distinguished by color shade and grouping."""
    return ""


def _model_line_style(model_label: str) -> str:
    """Choose a line style based on model family variant."""
    lowered = model_label.lower()
    if "safeguard" in lowered:
        return "--"
    if "uncensored" in lowered or "abliterate" in lowered or "derestricted" in lowered:
        return ":"
    return "-"


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
                    edgecolor="none",
                    linewidth=0.0,
                    label=family,
                )
            )

    return handles


def _apply_model_bar_styles(patches: Sequence[object], labels: Sequence[str]) -> None:
    """Apply model bar styling without outlines."""
    for patch, model in zip(patches, labels):
        patch.set_hatch(_model_bar_hatch(model))
        patch.set_edgecolor(_model_bar_edge_color(model))
        patch.set_linewidth(_model_bar_line_width(model))


def build_overall_title() -> str:
    """Build a descriptive title for the overall restraint chart."""
    return (
        "Restraint Rate by Model\n"
        "Higher bars indicate more sustainable resource use."
    )


def build_time_series_title(*, metric_label: str) -> str:
    """Build a descriptive title for one part_2 time-series chart."""
    return (
        f"{metric_label} Over Time by Model\n"
        "Each line follows one model's society through the resource-allocation run."
    )


def build_final_title(*, metric_label: str) -> str:
    """Build a descriptive title for final-state summary charts."""
    return (
        f"{metric_label} by Model\n"
        "Each bar uses the final completed day for that model's society."
    )


def build_collapse_title() -> str:
    """Build a descriptive title for the first-depletion chart."""
    return (
        "First Resource Depletion Day by Model\n"
        "Higher bars indicate the shared resource lasted longer."
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


INTER_GROUP_GAP = 0.45


def _bar_group_key(model_label: str) -> tuple[str, int]:
    """Return the grouping key for bar clustering: (family, instruct_group)."""
    return _normalize_model_family(model_label), _model_instruct_group(model_label)


def _grouped_bar_positions(labels: Sequence[str], *, cluster: bool = True) -> list[float]:
    """Position bars so same-family bars touch and different families have a gap.

    When *cluster* is False, bars are evenly spaced for per-family sub-plots.
    """
    if not labels:
        return []
    if not cluster:
        return [float(i) for i in range(len(labels))]
    positions: list[float] = [0.0]
    for index in range(1, len(labels)):
        if _bar_group_key(labels[index]) == _bar_group_key(labels[index - 1]):
            positions.append(positions[-1] + MODEL_BAR_WIDTH)
        else:
            positions.append(positions[-1] + MODEL_BAR_WIDTH + INTER_GROUP_GAP)
    return positions


def _apply_bar_xlim(
    ax: object,
    positions: Sequence[float],
    *,
    bar_width: float = MODEL_BAR_WIDTH,
) -> None:
    """Set x-axis limits with consistent padding on each side."""
    if not positions:
        return
    if len(positions) >= 2:
        min_step = min(positions[index] - positions[index - 1] for index in range(1, len(positions)))
    else:
        min_step = 1.0
    padding = max(INTER_GROUP_GAP, min_step - bar_width / 2)
    ax.set_xlim(min(positions) - bar_width / 2 - padding, max(positions) + bar_width / 2 + padding)


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


def _legend_column_count(handles: Sequence[object], maximum: int = 4) -> int:
    """Choose a compact legend column count for the handles that are visible."""
    return max(1, min(maximum, len(handles)))


def build_model_rows(
    overall: dict[str, Aggregate],
    z: float,
    ci_method: CiMethod,
) -> tuple[list[str], list[float], list[float], list[int]]:
    """Build sorted model arrays for plotting from aggregated totals."""
    data: list[tuple[str, float, float, int]] = []
    for model, agg in overall.items():
        rate, err, total = restraint_rate_and_error(agg, z, ci_method)
        if rate is None:
            continue
        data.append((model, rate, err, total))

    data.sort(key=lambda item: (*_model_sort_key(item[0]), -item[1], item[0]))
    labels = [item[0] for item in data]
    rates = [item[1] for item in data]
    errors = [item[2] for item in data]
    totals = [item[3] for item in data]
    return labels, rates, errors, totals


def _summaries_by_model(
    day_summaries: dict[tuple[str, int], DaySummary],
) -> dict[str, list[DaySummary]]:
    """Group day summaries by model in day order."""
    grouped: defaultdict[str, list[DaySummary]] = defaultdict(list)
    for summary in day_summaries.values():
        grouped[summary.model].append(summary)
    return {
        model: sorted(summaries, key=lambda summary: summary.day)
        for model, summaries in grouped.items()
    }


def _final_summary(summaries: Sequence[DaySummary]) -> DaySummary | None:
    """Return the final day summary for a model."""
    if not summaries:
        return None
    return max(summaries, key=lambda summary: summary.day)


def _initial_population(summaries: Sequence[DaySummary]) -> int | None:
    """Return initial population from the first completed day."""
    if not summaries:
        return None
    first = min(summaries, key=lambda summary: summary.day)
    return first.population_start if first.population_start > 0 else None


def build_final_population_rows(
    day_summaries: dict[tuple[str, int], DaySummary],
    model_order: Sequence[str],
) -> tuple[list[str], list[float]]:
    """Build final population retention percentages by model."""
    grouped = _summaries_by_model(day_summaries)
    labels: list[str] = []
    values: list[float] = []

    for model in model_order:
        summaries = grouped.get(model, [])
        final = _final_summary(summaries)
        initial_population = _initial_population(summaries)
        if final is None or initial_population is None:
            continue
        labels.append(model)
        values.append((final.population_end / initial_population) * 100)

    return labels, values


def build_final_resource_rows(
    day_summaries: dict[tuple[str, int], DaySummary],
    model_order: Sequence[str],
) -> tuple[list[str], list[float]]:
    """Build final resource percentages by model."""
    grouped = _summaries_by_model(day_summaries)
    labels: list[str] = []
    values: list[float] = []

    for model in model_order:
        final = _final_summary(grouped.get(model, []))
        if (
            final is None
            or final.resource_units_remaining is None
            or final.resource_capacity is None
            or final.resource_capacity <= 0
        ):
            continue
        labels.append(model)
        values.append((final.resource_units_remaining / final.resource_capacity) * 100)

    return labels, values


def build_collapse_day_rows(
    day_summaries: dict[tuple[str, int], DaySummary],
    model_order: Sequence[str],
) -> tuple[list[str], list[float], list[str]]:
    """Build first reserve-depletion day rows by model."""
    grouped = _summaries_by_model(day_summaries)
    labels: list[str] = []
    values: list[float] = []
    annotations: list[str] = []

    for model in model_order:
        summaries = grouped.get(model, [])
        if not summaries:
            continue
        first_depletion = next(
            (
                summary
                for summary in summaries
                if summary.resource_units_remaining is not None
                and summary.resource_units_remaining <= 0
            ),
            None,
        )
        final = _final_summary(summaries)
        if first_depletion is not None:
            labels.append(model)
            values.append(float(first_depletion.day))
            annotations.append(str(first_depletion.day))
        elif final is not None:
            labels.append(model)
            values.append(float(final.day))
            annotations.append(f">{final.day}")

    return labels, values, annotations


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
    cluster: bool = True,
) -> None:
    """Render and save the overall model restraint bar chart."""
    del totals
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    x = _grouped_bar_positions(labels, cluster=cluster)
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
        edgecolor="none",
        linewidth=0.0,
    )
    _apply_model_bar_styles(ax.patches, labels)
    _annotate_percent_bars(ax, x, rates, errors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _apply_bar_xlim(ax, x)
    _apply_percent_axis_labels(ax, rates, errors)
    ax.set_ylabel("Restraint choice rate (%)")
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


def render_percent_bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str,
    output: Path,
    cluster: bool = True,
) -> None:
    """Render and save a percent-valued model bar chart."""
    if not labels:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    x = _grouped_bar_positions(labels, cluster=cluster)
    errors = [0.0 for _ in values]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), OVERALL_FIG_HEIGHT))
    ax.bar(
        x,
        values,
        yerr=errors,
        capsize=4,
        width=MODEL_BAR_WIDTH,
        alpha=1.0,
        color=[_model_bar_color(model) for model in labels],
        edgecolor="none",
        linewidth=0.0,
    )
    _apply_model_bar_styles(ax.patches, labels)
    _annotate_percent_bars(ax, x, values, errors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _apply_bar_xlim(ax, x)
    _apply_percent_axis_labels(ax, values, errors)
    ax.set_ylabel(ylabel)
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


def render_collapse_day_chart(
    labels: Sequence[str],
    values: Sequence[float],
    annotations: Sequence[str],
    *,
    title: str,
    output: Path,
    cluster: bool = True,
) -> None:
    """Render and save the first reserve-depletion day chart."""
    if not labels:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    x = _grouped_bar_positions(labels, cluster=cluster)
    ymax = max(values) if values else 1.0
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), OVERALL_FIG_HEIGHT))
    ax.bar(
        x,
        values,
        width=MODEL_BAR_WIDTH,
        alpha=1.0,
        color=[_model_bar_color(model) for model in labels],
        edgecolor="none",
        linewidth=0.0,
    )
    _apply_model_bar_styles(ax.patches, labels)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _apply_bar_xlim(ax, x)
    ax.set_ylim(0.0, ymax + max(2.0, ymax * 0.12))
    _annotate_value_bars(ax, x, values, annotations)
    ax.set_ylabel("Day")
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


def _series_values(
    summaries: Sequence[DaySummary],
    metric: Part2Chart,
) -> tuple[list[int], list[float]]:
    """Build x/y arrays for a part_2 time-series metric."""
    days: list[int] = []
    values: list[float] = []
    for summary in summaries:
        value: float | None
        if metric == "restraint-by-day":
            total = summary.restrain_count + summary.overuse_count
            value = (summary.restrain_count / total) * 100 if total > 0 else None
        elif metric == "resource-by-day":
            if summary.resource_units_remaining is None or not summary.resource_capacity:
                value = None
            else:
                value = (summary.resource_units_remaining / summary.resource_capacity) * 100
        elif metric == "population-by-day":
            initial = summaries[0].population_start if summaries else 0
            value = (summary.population_end / initial) * 100 if initial > 0 else None
        elif metric == "deaths-by-day":
            value = float(summary.deaths)
        else:
            value = None

        if value is None:
            continue
        days.append(summary.day)
        values.append(value)
    return days, values


def render_time_series_chart(
    day_summaries: dict[tuple[str, int], DaySummary],
    model_order: Sequence[str],
    *,
    metric: Part2Chart,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    """Render and save a model-level part_2 time-series chart."""
    grouped = _summaries_by_model(day_summaries)
    models = [model for model in model_order if model in grouped]
    if not models:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.6), 8))
    all_values: list[float] = []

    for model in models:
        days, values = _series_values(grouped[model], metric)
        if not days:
            continue
        all_values.extend(values)
        ax.plot(
            days,
            values,
            label=model,
            color=_model_bar_color(model),
            linestyle=_model_line_style(model),
            linewidth=1.7,
            alpha=0.95,
        )

    if not all_values:
        plt.close(fig)
        return

    if metric in {"restraint-by-day", "resource-by-day", "population-by-day"}:
        upper = min(110.0, max(all_values) + BREAKDOWN_TOP_PADDING)
        ax.set_ylim(0.0, upper)
        ax.set_yticks(list(range(0, 101, 10)))
    else:
        upper = max(all_values) + max(1.0, max(all_values) * 0.12)
        ax.set_ylim(0.0, upper)

    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, y=0.985)
    _place_legend_and_adjust(
        fig,
        title="Model",
        ncol=3,
        left=0.08,
        right=0.99,
        bottom=0.10,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_csv_metadata_from_name(path: Path) -> CsvMetadata | None:
    """Extract config + timestamp from a part_2 result filename."""
    stem_parts = path.stem.split("__")
    if len(stem_parts) != 7 or stem_parts[0] != "part2":
        return None

    size_label = stem_parts[3]
    days_label = stem_parts[4]
    if not size_label.startswith("n"):
        return None
    try:
        society_size = int(size_label[1:])
        datetime_stamp = datetime.strptime(stem_parts[6], TIMESTAMP_FORMAT)
    except ValueError:
        return None

    if days_label == "open":
        parsed_days_label = "open"
    elif days_label.startswith("d"):
        try:
            int(days_label[1:])
        except ValueError:
            return None
        parsed_days_label = days_label
    else:
        return None

    return CsvMetadata(
        provider_slug=stem_parts[1],
        model_slug=stem_parts[2],
        society_size=society_size,
        days_label=parsed_days_label,
        resource=stem_parts[5],
        timestamp=datetime_stamp,
    )


def _days_label(days: int | None) -> str | None:
    """Convert a requested day count to the filename label."""
    if days is None:
        return None
    if days == 0:
        return "open"
    return f"d{days}"


def _is_interrupted_run(path: Path) -> bool:
    metadata_path = path.with_name(f"{path.stem}_meta.json")
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    return metadata.get("status") != "complete"


def _metadata_matches_filters(
    metadata: CsvMetadata,
    *,
    society_size: int | None,
    days: int | None,
    resource: str | None,
) -> bool:
    """Return whether filename metadata matches latest-discovery filters."""
    if society_size is not None and metadata.society_size != society_size:
        return False
    requested_days_label = _days_label(days)
    if requested_days_label is not None and metadata.days_label != requested_days_label:
        return False
    if resource is not None and metadata.resource != resource:
        return False
    return True


def find_latest_nonempty_csvs_by_model(
    directory: Path,
    *,
    society_size: int | None = None,
    days: int | None = None,
    resource: str | None = None,
) -> list[Path]:
    """Select the latest non-empty part_2 CSV for each model."""
    candidates: dict[str, tuple[datetime, Path]] = {}

    for path in sorted(directory.glob("*.csv")):
        metadata = _parse_csv_metadata_from_name(path)
        if metadata is None:
            continue
        if not _metadata_matches_filters(
            metadata,
            society_size=society_size,
            days=days,
            resource=resource,
        ):
            continue
        if _is_interrupted_run(path):
            continue

        rows = read_rows(path)
        if not rows:
            continue

        model = model_label(rows[0])
        previous = candidates.get(model)
        if previous is None or metadata.timestamp > previous[0]:
            candidates[model] = (metadata.timestamp, path)

    if not candidates:
        raise ValueError(f"No non-empty part_2 CSV files found in {directory}.")

    return [path for _, path in sorted(candidates.values(), key=lambda item: item[0])]


def resolve_csv_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve the input CSV set from either explicit files or latest discovery."""
    part2_dir = Path(args.part2_dir)
    if args.csv:
        selected: list[Path] = []
        for item in args.csv:
            path = Path(item)
            if not path.is_absolute():
                path = part2_dir / item
            selected.append(path)
        return selected

    return find_latest_nonempty_csvs_by_model(
        part2_dir,
        society_size=args.society_size,
        days=args.days,
        resource=args.resource,
    )


def default_out_prefix(paths: Sequence[Path]) -> str:
    """Choose a stable default output prefix for the selected input set."""
    if len(paths) == 1:
        return paths[0].stem

    parsed = [_parse_csv_metadata_from_name(path) for path in paths]
    if all(item is not None for item in parsed):
        config_keys = {
            (item.society_size, item.days_label, item.resource)
            for item in parsed
            if item is not None
        }
        if len(config_keys) == 1:
            society_size, days_label, resource = next(iter(config_keys))
            return f"part2_latest_n{society_size}_{days_label}_{resource}_per_model"

    return "part2_latest_per_model"


def render_chart_bundle(
    *,
    labels: Sequence[str],
    rates: Sequence[float],
    errors: Sequence[float],
    totals: Sequence[int],
    day_summaries: dict[tuple[str, int], DaySummary],
    charts: set[str],
    output_dir: Path,
    prefix: str,
    cluster: bool = True,
) -> list[Path]:
    """Render the selected part_2 chart set into one output directory."""
    written: list[Path] = []
    if not labels:
        return written

    if "restraint" in charts:
        output = output_dir / f"{prefix}_restraint_by_model.png"
        render_overall_chart(
            labels,
            rates,
            errors,
            totals,
            title=build_overall_title(),
            output=output,
            cluster=cluster,
        )
        written.append(output)

    for chart, (metric_label, ylabel, filename_suffix) in TIME_SERIES_SPECS.items():
        if chart not in charts:
            continue
        output = output_dir / f"{prefix}_{filename_suffix}.png"
        render_time_series_chart(
            day_summaries,
            labels,
            metric=chart,
            title=build_time_series_title(metric_label=metric_label),
            ylabel=ylabel,
            output=output,
        )
        written.append(output)

    if "final-resource" in charts:
        resource_labels, resource_values = build_final_resource_rows(day_summaries, labels)
        if resource_labels:
            output = output_dir / f"{prefix}_final_resource_by_model.png"
            render_percent_bar_chart(
                resource_labels,
                resource_values,
                title=build_final_title(
                    metric_label="Final shared reserve remaining",
                ),
                ylabel="Final resource remaining (%)",
                output=output,
                cluster=cluster,
            )
            written.append(output)

    if "final-population" in charts:
        population_labels, population_values = build_final_population_rows(day_summaries, labels)
        if population_labels:
            output = output_dir / f"{prefix}_final_population_by_model.png"
            render_percent_bar_chart(
                population_labels,
                population_values,
                title=build_final_title(
                    metric_label="Final population remaining",
                ),
                ylabel="Final population remaining (%)",
                output=output,
                cluster=cluster,
            )
            written.append(output)

    if "collapse-day" in charts:
        collapse_labels, collapse_values, collapse_annotations = build_collapse_day_rows(
            day_summaries,
            labels,
        )
        if collapse_labels:
            output = output_dir / f"{prefix}_collapse_day_by_model.png"
            render_collapse_day_chart(
                collapse_labels,
                collapse_values,
                collapse_annotations,
                title=build_collapse_title(),
                output=output,
                cluster=cluster,
            )
            written.append(output)

    return written


def main() -> int:
    """Entry point for `uv run python data/graphs/part_2_graphs.py ...`."""
    args = parse_args()
    csv_paths = resolve_csv_paths(args)

    rows: list[dict[str, str]] = []
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        rows.extend(read_rows(path))

    if not rows:
        raise ValueError("No rows found in the selected part_2 CSV files.")

    overall, day_summaries = aggregate_rows(rows)
    z = z_from_confidence(args.confidence)
    ci_method: CiMethod = args.ci_method

    labels, rates, errors, totals = build_model_rows(overall, z, ci_method)
    if not labels:
        raise ValueError("No model rows with recognized RESTRAIN/OVERUSE actions were found.")

    graphs_dir = Path(args.graphs_dir)
    prefix = args.out_prefix or default_out_prefix(csv_paths)

    charts: set[str] = set(args.charts)

    master_outputs = render_chart_bundle(
        labels=labels,
        rates=rates,
        errors=errors,
        totals=totals,
        day_summaries=day_summaries,
        charts=charts,
        output_dir=graphs_dir / MASTER_PLOTS_DIRNAME,
        prefix=prefix,
    )
    for output in master_outputs:
        print(f"wrote: {output}")

    for folder_name, group_models in build_model_plot_groups(labels):
        (
            group_labels,
            group_rates,
            group_errors,
            group_totals,
        ) = filter_model_rows(labels, rates, errors, totals, group_models)
        group_outputs = render_chart_bundle(
            labels=group_labels,
            rates=group_rates,
            errors=group_errors,
            totals=group_totals,
            day_summaries=day_summaries,
            charts=charts,
            output_dir=graphs_dir / folder_name,
            prefix=prefix,
            cluster=False,
        )
        for output in group_outputs:
            print(f"wrote: {output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
