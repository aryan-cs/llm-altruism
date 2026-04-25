"""Render cooperative-choice charts for part_1 benchmark outputs.

The default mode selects the latest non-empty part_1 CSV for each model from
``results/part_1`` and combines them into a single comparison set.

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
DEFAULT_PART_1_DIR = Path("results") / "part_1"
DEFAULT_GRAPH_DIR = Path("data") / "graphs" / "part_1"
DEFAULT_CI_METHOD: Literal["wilson", "wald"] = "wilson"
OVERALL_TOP_PADDING = 6.0
BREAKDOWN_TOP_PADDING = 10.0
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
HATCH_PALETTE: tuple[str, ...] = ("", "////", "xxxx", "....", "++++", "||||", "\\\\\\\\", "----")
STANDARD_MODEL_COLOR = "#048a73"
SAFEGUARD_MODEL_COLOR = "#01483d"
UNRESTRICTED_MODEL_COLOR = "#34d6bc"
EDGE_COLOR = "#025e4f"
MODEL_BAR_WIDTH = 0.90
MODEL_FAMILY_GAP = 0.35
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
            "One or more CSV filenames or paths inside results/part_1 to combine. "
            "Example: --csv part1__ollama__gpt-oss-20b__full__20260420_005501.csv"
        ),
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest non-empty CSV for each model in results/part_1.",
    )

    parser.add_argument(
        "--part1-dir",
        default=str(DEFAULT_PART_1_DIR),
        help="Directory containing part_1 CSV files (default: results/part_1).",
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


def _model_bar_color(model_label: str) -> str:
    """Choose a bar color based on model family variant."""
    lowered = model_label.lower()
    if "safeguard" in lowered:
        return SAFEGUARD_MODEL_COLOR
    if "uncensored" in lowered or "abliterate" in lowered or "derestricted" in lowered:
        return UNRESTRICTED_MODEL_COLOR
    return STANDARD_MODEL_COLOR


def _model_bar_hatch(model_label: str) -> str:
    """Choose a hatch pattern based on model family variant."""
    lowered = model_label.lower()
    if "safeguard" in lowered:
        return "////"
    if "uncensored" in lowered or "abliterate" in lowered or "derestricted" in lowered:
        return "xxxx"
    return ""


def _dimension_hatches(count: int) -> list[str]:
    """Return hatch patterns for grouped breakdown bars."""
    if count <= 0:
        return []
    return [HATCH_PALETTE[index % len(HATCH_PALETTE)] for index in range(count)]


def build_overall_title(*, source_note: str, ci_label: str) -> str:
    """Build a descriptive title for the overall chart."""
    return (
        "Overall cooperative-choice rate by model\n"
        "Each bar aggregates all part 1 prompts for one model; higher means more COOPERATE / RESTRAIN choices.\n"
        f"Source: {source_note} | {ci_label}"
    )


def build_dimension_title(*, dimension: str, source_note: str, ci_label: str) -> str:
    """Build a descriptive title for one part_1 breakdown chart."""
    descriptions = {
        "game": "split by game type (Prisoner's Dilemma vs. Temptation / Commons)",
        "frame": "split by prompt framing (self choice, advice, observer judgment, prediction)",
        "domain": "split by scenario domain",
        "presentation": "split by prompt presentation style (narrative vs. structured)",
    }
    detail = descriptions.get(dimension, f"split by {dimension}")
    return (
        f"Cooperative-choice rate by model, {detail}\n"
        "Within each model, separate bars show the rate for each category in that breakdown.\n"
        f"Source: {source_note} | {ci_label}"
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
    for patch, model in zip(ax.patches, labels):
        patch.set_hatch(_model_bar_hatch(model))
    _annotate_percent_bars(ax, x, rates, errors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _apply_percent_axis_labels(ax, rates, errors)
    ax.set_ylabel("Cooperative choice rate (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, y=0.985)
    fig.legend(
        handles=[
            Patch(facecolor=STANDARD_MODEL_COLOR, edgecolor=EDGE_COLOR, label="Standard"),
            Patch(
                facecolor=SAFEGUARD_MODEL_COLOR,
                edgecolor=EDGE_COLOR,
                hatch="////",
                label="Safeguard",
            ),
            Patch(
                facecolor=UNRESTRICTED_MODEL_COLOR,
                edgecolor=EDGE_COLOR,
                hatch="xxxx",
                label="Uncensored / Abliterate",
            ),
        ],
        title="Model type",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.14, right=0.99, top=0.78, bottom=0.26)

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

    width = 0.75 / len(dimension_values)
    model_positions = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.6), 8))
    dimension_colors = _palette_for_count(len(dimension_values))
    dimension_hatches = _dimension_hatches(len(dimension_values))

    all_rates: list[float] = []
    all_errors: list[float] = []

    for dim_idx, value in enumerate(dimension_values):
        xs: list[float] = []
        ys: list[float] = []
        yerrs: list[float] = []

        for model_idx, model in enumerate(models):
            agg = by_dimension.get((model, value), Aggregate())
            rate, err, _ = cooperation_rate_and_error(agg, z, ci_method)
            if rate is None:
                continue

            xs.append(model_idx + (dim_idx - len(dimension_values) / 2 + 0.5) * width)
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
            label=_dimension_display_label(dimension, value),
            color=dimension_colors[dim_idx % len(dimension_colors)],
            alpha=1.0,
            edgecolor=EDGE_COLOR,
            linewidth=0.25,
            hatch=dimension_hatches[dim_idx % len(dimension_hatches)],
        )
        _annotate_percent_bars(ax, xs, ys, yerrs, rotate=90)

    ax.set_xticks(model_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    if model_positions:
        left_pad = 0.5 + (len(dimension_values) * width) / 2
        right_pad = 0.5 + (len(dimension_values) * width) / 2
        ax.set_xlim(model_positions[0] - left_pad, model_positions[-1] + right_pad)
    upper = _y_axis_upper_bound(all_rates, all_errors, BREAKDOWN_TOP_PADDING)
    ax.set_ylim(0.0, upper)
    ax.set_yticks(list(range(0, 101, 10)))
    ax.set_ylabel("Cooperative choice rate (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, y=0.985)
    fig.legend(
        title=dimension.capitalize(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=min(len(dimension_values), len(dimension_colors)),
        frameon=False,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.18)

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
    ci_label = _ci_label(ci_method, args.confidence)
    source_note = (
        f"{len(csv_paths)} latest part_1 CSVs"
        if len(csv_paths) > 1
        else csv_paths[0].name
    )

    overall_output = graphs_dir / f"{prefix}_cooperation_by_model.png"
    render_overall_chart(
        labels,
        rates,
        errors,
        totals,
        title=build_overall_title(source_note=source_note, ci_label=ci_label),
        output=overall_output,
    )
    print(f"wrote: {overall_output}")

    for dimension in args.dimensions:
        _, by_dimension = aggregate_rows(rows, dimension)
        dimension_output = graphs_dir / f"{prefix}_cooperation_by_model_and_{dimension}.png"
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
        print(f"wrote: {dimension_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
