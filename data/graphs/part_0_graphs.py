"""Render alignment-rate charts for part_0 benchmark outputs.

The input is expected to be a part_0 CSV in ``data/raw/part_0`` with a
``complied?`` column containing true/false values.

For each model, the script computes:

- alignment rate = denied / (denied + complied)
- confidence interval around that rate

By default, it uses the Wilson score interval, which is the standard practical
choice for proportions in model-eval publications.
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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import matplotlib.axes

TIMESTAMP_FORMAT = "%m-%d-%Y_%H_%M_%S"
DEFAULT_ALIGNMENT_DIR = Path("data") / "raw" / "part_0"
DEFAULT_GRAPH_DIR = Path("data") / "graphs" / "part_0"
DEFAULT_CI_METHOD: Literal["wilson", "wald"] = "wilson"
MASTER_PLOTS_DIRNAME = "master-plots"
OVERALL_TOP_PADDING = 6.0
LANGUAGE_TOP_PADDING = 10.0
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

CiMethod = Literal["wilson", "wald"]


@dataclass
class Aggregate:
    """Per-model/per-language count summary for one benchmark split."""

    denied: int = 0
    complied: int = 0
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate part_0 alignment graphs with confidence intervals "
            "for model-level alignment rates."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--csv",
        nargs="+",
        help=(
            "One or more CSV filenames or paths inside data/raw/part_0 to use. "
            "Example: --csv 04-11-2026_13_04_37.csv"
        ),
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest timestamped CSV in data/raw/part_0.",
    )

    parser.add_argument(
        "--alignment-dir",
        default=str(DEFAULT_ALIGNMENT_DIR),
        help="Directory containing part_0 alignment CSV files (default: data/raw/part_0).",
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(DEFAULT_GRAPH_DIR),
        help="Directory for output graph files (default: data/graphs/part_0).",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix (defaults to the CSV stem).",
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
        "--no-language-breakdown",
        action="store_true",
        help="Skip the model-by-language chart.",
    )
    return parser.parse_args()


def z_from_confidence(confidence: float) -> float:
    """Convert a confidence level to a two-sided standard-normal z-score."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return statistics.NormalDist().inv_cdf((1.0 + confidence) / 2.0)


def parse_complied_value(raw: str) -> bool | None:
    """Parse ``complied?`` values.

    Returns:
        True when model complied,
        False when denied,
        None for unknown / skipped rows.
    """
    normalized = (raw or "").strip().lower()
    if normalized in {"true", "1", "yes", "y", "complied"}:
        return True
    if normalized in {"false", "0", "no", "n", "denied"}:
        return False
    return None


def model_label(row: dict[str, str]) -> str:
    """Build a stable model label from provider + model columns."""
    provider = (row.get("provider", "") or "unknown").strip()
    model = (row.get("model", "") or "unknown").strip()
    return f"{provider}/{model}"


def find_latest_csv(directory: Path) -> Path:
    """Find the newest timestamped part_0 results CSV in `alignment-dir`."""
    candidates: list[tuple[datetime, Path]] = []
    for path in directory.glob("*.csv"):
        stem = path.stem
        if stem.endswith("_pending") or stem.endswith("_meta"):
            continue
        try:
            candidates.append((datetime.strptime(stem, TIMESTAMP_FORMAT), path))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"No timestamped CSV found in {directory}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load rows from a CSV path produced by part_0."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(
    rows: Sequence[dict[str, str]],
) -> tuple[
    dict[str, Aggregate],
    dict[tuple[str, str], Aggregate],
]:
    """Aggregate counts by model and model+language.

    Returns:
        (overall_by_model, by_model_language)
    """
    overall: defaultdict[str, Aggregate] = defaultdict(Aggregate)
    by_language: defaultdict[tuple[str, str], Aggregate] = defaultdict(Aggregate)

    for row in rows:
        key = model_label(row)
        language = (row.get("language", "") or "unknown").strip()
        state = parse_complied_value(row.get("complied?", ""))

        agg = overall[key]
        lang_agg = by_language[(key, language)]

        if state is True:
            agg.complied += 1
            lang_agg.complied += 1
        elif state is False:
            agg.denied += 1
            lang_agg.denied += 1
        else:
            agg.skipped += 1
            lang_agg.skipped += 1

    return dict(overall), dict(by_language)


def _center_and_half_width_wilson(denied: int, total: int, z: float) -> tuple[float, float]:
    """Wilson score center and half-width for binomial proportion.

    Returns:
        (center, half_width) where both are in raw probability scale [0, 1].
    """
    if total == 0:
        raise ValueError("total must be > 0")

    p = denied / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = (z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))) / denominator

    lower = max(0.0, center - half)
    upper = min(1.0, center + half)
    center = (lower + upper) / 2
    half = (upper - lower) / 2
    return center, half


def _center_and_half_width_wald(denied: int, total: int, z: float) -> tuple[float, float]:
    """Wald/normal confidence interval center and half-width (legacy)."""
    if total == 0:
        raise ValueError("total must be > 0")

    p = denied / total
    half = z * math.sqrt(p * (1 - p) / total)
    lower = max(0.0, p - half)
    upper = min(1.0, p + half)
    center = (lower + upper) / 2
    half = (upper - lower) / 2
    return center, half


def alignment_rate_and_error(
    agg: Aggregate,
    z: float,
    ci_method: CiMethod,
) -> tuple[float | None, float, int]:
    """Compute alignment percentage and CI half-width.

    Returns:
        (rate_percent, half_error_percent, judged_total).
        If no judged samples exist, returns (None, 0, 0).
    """
    total = agg.denied + agg.complied
    if total == 0:
        return None, 0.0, 0

    if ci_method == "wilson":
        center, half = _center_and_half_width_wilson(agg.denied, total, z)
    else:
        center, half = _center_and_half_width_wald(agg.denied, total, z)

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


def _apply_percent_axis_labels(ax: matplotlib.axes.Axes, values: Sequence[float], errors: Sequence[float]) -> None:
    """Set percent ticks through 100 while leaving top headroom."""
    # Use a modest default headroom consistent with the single-model chart.
    upper = _y_axis_upper_bound(values, errors, OVERALL_TOP_PADDING)
    ax.set_ylim(0.0, upper)
    ax.set_yticks(list(range(0, 101, 10)))


def _annotate_percent_bars(
    ax: matplotlib.axes.Axes,
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
        va = "bottom"

        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va=va,
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


def _normalize_language_name(language: str) -> str:
    """Normalize language labels for color mapping."""
    return (language or "").strip().lower()


def _language_hatches_by_name(languages: Sequence[str]) -> list[str]:
    """Map language names to distinct hatch patterns for clearer legends."""
    return [HATCH_PALETTE[index % len(HATCH_PALETTE)] for index, _ in enumerate(languages)]


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


def _language_legend_handles(languages: Sequence[str], hatches: Sequence[str]) -> list[object]:
    """Build hatch-only legend entries for languages."""
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=DEFAULT_FALLBACK_COLOR, edgecolor=HATCH_COLOR, hatch=hatches[i], label=lang)
        for i, lang in enumerate(languages)
    ]


def build_overall_title() -> str:
    """Build a descriptive title for the overall chart."""
    return (
        "Alignment Rate by Model\n"
        "Higher bars indicate more refusals of harmful prompts."
    )


def build_language_title() -> str:
    """Build a descriptive title for the language breakdown chart."""
    return (
        "Alignment Rate by Model and Language\n"
        "Bars compare refusal rates across response languages."
    )


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
    """Place a legend between the suptitle and the axes, adjusting top margin dynamically.

    Measures the suptitle's bottom edge so the legend is anchored just
    below it, then measures the legend's own height and sets
    ``subplots_adjust(top=...)`` so the axes never overlap the legend.
    """
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


def build_model_rows(
    overall: dict[str, Aggregate],
    z: float,
    ci_method: CiMethod,
) -> tuple[list[str], list[float], list[float], list[int]]:
    """Build sorted model arrays for plotting from aggregated totals."""
    data: list[tuple[str, float, float, int]] = []
    for model, agg in overall.items():
        rate, err, total = alignment_rate_and_error(agg, z, ci_method)
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
    """Render and save the overall model alignment bar chart."""
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
    ax.set_ylabel("Alignment rate (%)")
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


def render_language_chart(
    by_language: dict[tuple[str, str], Aggregate],
    z: float,
    ci_method: CiMethod,
    model_order: Sequence[str],
    *,
    title: str,
    output: Path,
) -> None:
    """Render and save the model-by-language grouped chart."""
    available_models = {model for model, _ in by_language.keys()}
    models = [model for model in model_order if model in available_models]
    if not models:
        return
    languages = sorted({language for _, language in by_language.keys()})
    if not models or not languages:
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

    width = 0.75 / len(languages)
    model_positions = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.6), 8))
    language_hatches = _language_hatches_by_name(languages)

    all_rates: list[float] = []
    all_errors: list[float] = []

    for lang_idx, language in enumerate(languages):
        xs: list[float] = []
        ys: list[float] = []
        yerrs: list[float] = []
        bar_colors: list[str] = []

        for m_idx, model in enumerate(models):
            agg = by_language.get((model, language), Aggregate())
            rate, err, _ = alignment_rate_and_error(agg, z, ci_method)
            if rate is None:
                continue

            xs.append(m_idx + (lang_idx - len(languages) / 2 + 0.5) * width)
            ys.append(rate)
            yerrs.append(err)
            bar_colors.append(_model_bar_color(model))
            all_rates.append(rate)
            all_errors.append(err)

        bar_models = [
            model
            for model in models
            if alignment_rate_and_error(
                by_language.get((model, language), Aggregate()),
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
            label=language,
            color=bar_colors,
            alpha=1.0,
            edgecolor=EDGE_COLOR,
            linewidth=0.25,
            hatch=language_hatches[lang_idx % len(language_hatches)],
        )
        for patch, model in zip(bars.patches, bar_models):
            patch.set_edgecolor(_model_bar_edge_color(model))
            patch.set_linewidth(_model_bar_line_width(model))
        _annotate_percent_bars(ax, xs, ys, yerrs, rotate=90)

    ax.set_xticks(model_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    if model_positions:
        group_width = len(languages) * width
        edge_pad = 1.0 - group_width / 2
        ax.set_xlim(model_positions[0] - edge_pad, model_positions[-1] + edge_pad)
    upper = _y_axis_upper_bound(all_rates, all_errors, LANGUAGE_TOP_PADDING)
    ax.set_ylim(0.0, upper)
    ax.set_yticks(list(range(0, 101, 10)))
    ax.set_ylabel("Alignment rate (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, y=0.985)
    legend_handles = _model_legend_handles(models) + _language_legend_handles(languages, language_hatches)
    _place_legend_and_adjust(
        fig,
        handles=legend_handles,
        title="Model / Language",
        ncol=_legend_column_count(legend_handles),
        left=0.08,
        right=0.99,
        bottom=0.18,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_csv_paths(args: argparse.Namespace) -> list[Path]:
    """Resolve CSV paths from either explicit filenames or --latest."""
    alignment_dir = Path(args.alignment_dir)
    if args.latest:
        return [find_latest_csv(alignment_dir)]

    if not args.csv:
        raise ValueError("Either --csv or --latest is required.")

    selected_paths: list[Path] = []
    for item in args.csv:
        selected = Path(item)
        if not selected.is_absolute():
            selected = alignment_dir / item
        selected_paths.append(selected)
    return selected_paths


def default_out_prefix(csv_paths: Sequence[Path]) -> str:
    """Choose a stable default output prefix for the selected input set."""
    if len(csv_paths) == 1:
        return csv_paths[0].stem
    return f"alignment_combined_{len(csv_paths)}files"


def render_chart_bundle(
    *,
    labels: Sequence[str],
    rates: Sequence[float],
    errors: Sequence[float],
    totals: Sequence[int],
    by_language: dict[tuple[str, str], Aggregate],
    z: float,
    ci_method: CiMethod,
    output_dir: Path,
    prefix: str,
    source_note: str,
    ci_label: str,
    include_language_breakdown: bool,
) -> list[Path]:
    """Render the selected part_0 chart set into one output directory."""
    written: list[Path] = []
    if not labels:
        return written

    overall_output = output_dir / f"{prefix}_alignment_by_model.png"
    render_overall_chart(
        labels,
        rates,
        errors,
        totals,
        title=build_overall_title(source_note=source_note, ci_label=ci_label),
        output=overall_output,
    )
    written.append(overall_output)

    if include_language_breakdown:
        language_output = output_dir / f"{prefix}_alignment_by_model_and_language.png"
        render_language_chart(
            by_language,
            z,
            ci_method,
            labels,
            title=build_language_title(source_note=source_note, ci_label=ci_label),
            output=language_output,
        )
        written.append(language_output)

    return written


def main() -> int:
    """Entry point for `uv run python data/graphs/part_0_graphs.py ...`."""
    args = parse_args()
    csv_paths = resolve_csv_paths(args)
    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        rows.extend(read_rows(csv_path))
    if not rows:
        raise ValueError("No rows found in the selected alignment CSV files.")

    overall, by_language = aggregate_rows(rows)
    z = z_from_confidence(args.confidence)
    ci_method: CiMethod = args.ci_method

    labels, rates, errors, totals = build_model_rows(overall, z, ci_method)
    if not labels:
        raise ValueError("No model rows with judged results were found.")

    graphs_dir = Path(args.graphs_dir)
    prefix = args.out_prefix or default_out_prefix(csv_paths)

    # --- master-plots (all models) ---
    master_dir = graphs_dir / MASTER_PLOTS_DIRNAME

    overall_output = master_dir / f"{prefix}_alignment_by_model.png"
    render_overall_chart(
        labels,
        rates,
        errors,
        totals,
        title=build_overall_title(),
        output=overall_output,
    )
    print(f"wrote: {overall_output}")

    if not args.no_language_breakdown:
        language_output = master_dir / f"{prefix}_alignment_by_model_and_language.png"
        render_language_chart(
            by_language,
            z,
            ci_method,
            labels,
            title=build_language_title(),
            output=language_output,
        )
        print(f"wrote: {language_output}")

    # --- per-family model-specific plots ---
    for folder_name, group_models in build_model_plot_groups(labels):
        group_labels, group_rates, group_errors, group_totals = filter_model_rows(
            labels, rates, errors, totals, group_models,
        )
        group_dir = graphs_dir / folder_name

        group_overall = group_dir / f"{prefix}_alignment_by_model.png"
        render_overall_chart(
            group_labels,
            group_rates,
            group_errors,
            group_totals,
            title=build_overall_title(),
            output=group_overall,
        )
        print(f"wrote: {group_overall}")

        if not args.no_language_breakdown:
            group_lang = group_dir / f"{prefix}_alignment_by_model_and_language.png"
            render_language_chart(
                by_language,
                z,
                ci_method,
                group_labels,
                title=build_language_title(),
                output=group_lang,
            )
            print(f"wrote: {group_lang}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
