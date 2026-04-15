"""Render alignment-rate charts for part_0 benchmark outputs.

The input is expected to be a part_0 CSV in ``results/alignment`` with a
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
from typing import Literal

TIMESTAMP_FORMAT = "%m-%d-%Y_%H_%M_%S"
DEFAULT_ALIGNMENT_DIR = Path("results") / "alignment"
DEFAULT_GRAPH_DIR = Path("data") / "graphs"
DEFAULT_CI_METHOD: Literal["wilson", "wald"] = "wilson"
OVERALL_TOP_PADDING = 6.0
LANGUAGE_TOP_PADDING = 10.0
TEAL_PALETTE: tuple[str, ...] = (
    "#036f5a",
    "#03826a",
    "#048a73",
    "#059482",
    "#06a991",
    "#0abfa0",
    "#10c6a8",
    "#2cd4b8",
)
STANDARD_MODEL_COLOR = "#048a73"
SAFEGUARD_MODEL_COLOR = "#025e4f"
UNRESTRICTED_MODEL_COLOR = "#2cd4b8"
EDGE_COLOR = "#025e4f"
ENGLISH_LANGUAGE_COLOR = "#036f5a"
RUSSIAN_LANGUAGE_COLOR = "#2cd4b8"
MODEL_BAR_WIDTH = 0.90
MODEL_FAMILY_GAP = 0.35
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
        help=(
            "CSV filename or path inside results/alignment to use. "
            "Example: --csv 04-11-2026_13_04_37.csv"
        ),
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest timestamped CSV in results/alignment.",
    )

    parser.add_argument(
        "--alignment-dir",
        default=str(DEFAULT_ALIGNMENT_DIR),
        help="Directory containing part_0 alignment CSV files (default: results/alignment).",
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(DEFAULT_GRAPH_DIR),
        help="Directory for output graph files (default: data/graphs).",
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


def _palette_for_count(count: int) -> list[str]:
    """Return a list of shades based on #048a73."""
    if count <= 0:
        return []
    if count <= len(TEAL_PALETTE):
        return list(TEAL_PALETTE[:count])
    repeats = []
    for index in range(count):
        repeats.append(TEAL_PALETTE[index % len(TEAL_PALETTE)])
    return repeats


def _normalize_language_name(language: str) -> str:
    """Normalize language labels for color mapping."""
    return (language or "").strip().lower()


def _language_colors_by_name(languages: Sequence[str]) -> list[str]:
    """Map language names to a readable color set with higher English-vs-Russian contrast."""
    palette = _palette_for_count(len(languages))
    color_by_lang: dict[str, str] = {}

    fallback_colors = [c for c in palette if c not in {ENGLISH_LANGUAGE_COLOR, RUSSIAN_LANGUAGE_COLOR}]
    fallback_idx = 0

    for language in languages:
        normalized = _normalize_language_name(language)
        if normalized == "english" and ENGLISH_LANGUAGE_COLOR not in color_by_lang.values():
            color_by_lang[language] = ENGLISH_LANGUAGE_COLOR
        elif normalized == "russian":
            color_by_lang[language] = RUSSIAN_LANGUAGE_COLOR
        elif fallback_colors:
            color_by_lang[language] = fallback_colors[fallback_idx % len(fallback_colors)]
            fallback_idx += 1
        elif palette:
            color_by_lang[language] = palette[fallback_idx % len(palette)]
            fallback_idx += 1
        else:
            color_by_lang[language] = STANDARD_MODEL_COLOR

    return [color_by_lang[language] for language in languages]


def _model_bar_color(model_label: str) -> str:
    """Choose a bar color based on model family variant."""
    lowered = model_label.lower()
    if "safeguard" in lowered:
        return SAFEGUARD_MODEL_COLOR
    if "uncensored" in lowered or "abliterate" in lowered or "derestricted" in lowered:
        return UNRESTRICTED_MODEL_COLOR
    return STANDARD_MODEL_COLOR


def _grouped_bar_positions(labels: Sequence[str]) -> list[float]:
    """Compute x positions so model variants in the same family touch."""
    positions: list[float] = []
    previous_group = None
    index_in_family = 0
    cursor_x = 0.0

    for model in labels:
        family = _normalize_model_family(model)
        group = _model_instruct_group(model)
        family_group = (family, group)
        if previous_group is None:
            previous_group = family_group
        elif family_group != previous_group:
            cursor_x += (index_in_family * MODEL_BAR_WIDTH) + MODEL_FAMILY_GAP
            previous_group = family_group
            index_in_family = 0

        positions.append(cursor_x + (index_in_family * MODEL_BAR_WIDTH))
        index_in_family += 1

    return positions


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
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
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
    _annotate_percent_bars(ax, x, rates, errors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _apply_percent_axis_labels(ax, rates, errors)
    ax.set_ylabel("Alignment rate (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, y=0.97)
    fig.legend(
        handles=[
            Patch(facecolor=STANDARD_MODEL_COLOR, edgecolor=EDGE_COLOR, label="Standard"),
            Patch(facecolor=SAFEGUARD_MODEL_COLOR, edgecolor=EDGE_COLOR, label="Safeguard"),
            Patch(
                facecolor=UNRESTRICTED_MODEL_COLOR,
                edgecolor=EDGE_COLOR,
                label="Uncensored / Abliterate",
            ),
        ],
        title="Model type",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=0.85, bottom=0.26)

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

    width = 0.75 / len(languages)
    model_positions = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.6), 8))
    language_colors = _language_colors_by_name(languages)

    all_rates: list[float] = []
    all_errors: list[float] = []

    for lang_idx, language in enumerate(languages):
        xs: list[float] = []
        ys: list[float] = []
        yerrs: list[float] = []

        for m_idx, model in enumerate(models):
            agg = by_language.get((model, language), Aggregate())
            rate, err, _ = alignment_rate_and_error(agg, z, ci_method)
            if rate is None:
                continue

            xs.append(m_idx + (lang_idx - len(languages) / 2 + 0.5) * width)
            ys.append(rate)
            yerrs.append(err)
            all_rates.append(rate)
            all_errors.append(err)

        ax.bar(
            xs,
            ys,
            width=width,
            yerr=yerrs,
            capsize=3,
            label=language,
            color=language_colors[lang_idx % len(language_colors)],
            alpha=1.0,
            edgecolor=EDGE_COLOR,
            linewidth=0.25,
        )
        _annotate_percent_bars(ax, xs, ys, yerrs, rotate=90)

    ax.set_xticks(model_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    if model_positions:
        left_pad = 0.5 + (len(languages) * width) / 2
        right_pad = 0.5 + (len(languages) * width) / 2
        ax.set_xlim(model_positions[0] - left_pad, model_positions[-1] + right_pad)
    upper = _y_axis_upper_bound(all_rates, all_errors, LANGUAGE_TOP_PADDING)
    ax.set_ylim(0.0, upper)
    ax.set_yticks(list(range(0, 101, 10)))
    ax.set_ylabel("Alignment rate (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, y=0.97)
    fig.legend(
        title="Language",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=min(len(languages), len(language_colors)),
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.18)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def resolve_csv_path(args: argparse.Namespace) -> Path:
    """Resolve the CSV path from either explicit filename or --latest."""
    alignment_dir = Path(args.alignment_dir)
    if args.latest:
        return find_latest_csv(alignment_dir)

    if not args.csv:
        raise ValueError("Either --csv or --latest is required.")

    selected = Path(args.csv)
    if not selected.is_absolute():
        selected = alignment_dir / selected
    return selected


def main() -> int:
    """Entry point for `uv run python data/graphs/part_0_graphs.py ...`."""
    args = parse_args()
    csv_path = resolve_csv_path(args)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    overall, by_language = aggregate_rows(rows)
    z = z_from_confidence(args.confidence)
    ci_method: CiMethod = args.ci_method

    labels, rates, errors, totals = build_model_rows(overall, z, ci_method)
    if not labels:
        raise ValueError("No model rows with judged results were found.")

    graphs_dir = Path(args.graphs_dir)
    prefix = args.out_prefix or csv_path.stem
    ci_label = _ci_label(ci_method, args.confidence)

    overall_output = graphs_dir / f"{prefix}_alignment_by_model.png"
    render_overall_chart(
        labels,
        rates,
        errors,
        totals,
        title=f"Alignment by model ({csv_path.name})\n{ci_label}",
        output=overall_output,
    )

    if not args.no_language_breakdown:
        language_output = graphs_dir / f"{prefix}_alignment_by_model_and_language.png"
        render_language_chart(
            by_language,
            z,
            ci_method,
            labels,
            title=f"Alignment by model and language ({csv_path.name})\n{ci_label}",
            output=language_output,
        )
        print(f"wrote: {language_output}")

    print(f"wrote: {overall_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
