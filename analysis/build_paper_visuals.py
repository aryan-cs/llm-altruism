"""Build manuscript visuals from the sealed sanitized result graph.

The figures produced here are deliberately descriptive.  They preserve the
axis-specific validity boundaries in ``final_results.json`` and never compute a
composite score, cross-axis correlation, developer effect, or model ranking.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.build_paper_headlines import _read_source


DEFAULT_INPUT = Path("data/analysis/final_results/final_results.json")
DEFAULT_OUTPUT_DIR = Path("docs/conference_submission/figures")


class PaperVisualError(RuntimeError):
    """The sealed result graph cannot support the requested paper visual."""


def _label(row: Mapping[str, Any]) -> str:
    return f"{row['model']} — {row['upstream_provider']}"


def _ordered(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["upstream_provider"]).casefold(),
            str(row["model"]).casefold(),
            str(row["target_id"]),
        ),
    )


def _draw_interval(
    axis: Any,
    *,
    position: int,
    estimate: float,
    lower: float,
    upper: float,
    color: Any,
    marker: str = "o",
) -> None:
    axis.errorbar(
        estimate,
        position,
        xerr=[[max(0.0, estimate - lower)], [max(0.0, upper - estimate)]],
        fmt=marker,
        linestyle="none",
        color=color,
        ecolor=color,
        markersize=4.5,
        markeredgewidth=0.8,
        elinewidth=0.9,
        capsize=1.8,
    )


def _finish_rate_axis(axis: Any, title: str, xlabel: str) -> None:
    axis.set_xlim(-0.035, 1.035)
    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_title(title, fontweight="semibold")
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", color="#D0D0D0", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)


def _save(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("pdf", "png"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    return paths


def _part0_conditions(plt: Any, rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    ordered = _ordered(rows)
    conditions = (
        ("english", "English reply requested"),
        ("chinese", "Chinese reply requested"),
        ("russian", "Russian reply requested"),
    )
    colors = plt.cm.turbo([0.12, 0.50, 0.88])
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, max(5.2, 0.27 * len(ordered) + 1.25)),
        sharey=True,
        constrained_layout=True,
    )
    positions = list(range(len(ordered)))
    for axis, (condition_id, title), color in zip(axes, conditions, colors):
        for position, row in enumerate(ordered):
            condition = next(
                value
                for value in row["conditions"]
                if value["response_language_condition"] == condition_id
            )
            interval = condition["refusal"]
            _draw_interval(
                axis,
                position=position,
                estimate=float(interval["estimate"]),
                lower=float(interval["lower"]),
                upper=float(interval["upper"]),
                color=color,
            )
        _finish_rate_axis(axis, title, "Material-refusal rate")
    axes[0].set_yticks(positions, [_label(row) for row in ordered])
    axes[0].invert_yaxis()
    fig.suptitle(
        "Part 0 response-language conditions over identical English request roots",
        fontweight="semibold",
    )
    return _save(fig, output_dir, "part0_response_language_conditions")


def _part1_distribution(plt: Any, rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    scopes = (
        (384, "Full, n=384", "o", 0.12),
        (96, "Balanced partial, n=96", "s", 0.50),
        (12, "Balanced partial, n=12", "^", 0.88),
    )
    # This figure is placed on a portrait page.  Match the manuscript text
    # width so LaTeX does not shrink its labels to roughly five-point type.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.5), constrained_layout=True)
    for axis, (root_count, title, marker, color_stop) in zip(axes, scopes):
        selected = sorted(
            (
                float(row["cooperation"]["estimate"])
                for row in rows
                if int(row["root_count"]) == root_count
            )
        )
        color = plt.cm.turbo(color_stop)
        if selected:
            denominator = len(selected)
            cumulative = [(index + 1) / denominator for index in range(denominator)]
            axis.step(selected, cumulative, where="post", color=color, linewidth=1.5)
            axis.plot(selected, cumulative, marker, color=color, markersize=3.8, linestyle="none")
            axis.text(
                0.98,
                0.08,
                f"{denominator} system{'s' if denominator != 1 else ''}",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.0,
            )
        _finish_rate_axis(axis, title, "Welfare-preserving choice rate")
        axis.set_ylim(0.0, 1.04)
        axis.set_ylabel("Cumulative share of systems")
    fig.suptitle(
        "Part 1 within-scope distributions (scopes shown separately, never pooled)",
        fontweight="semibold",
    )
    return _save(fig, output_dir, "part1_scope_distributions")


def _part2_outcomes(plt: Any, rows: Sequence[Mapping[str, Any]], output_dir: Path) -> list[Path]:
    ordered = _ordered(rows)
    metrics = (
        ("aurc", "Normalized reserve area (AURC)"),
        ("restraint_rate", "Restraint rate"),
        ("reserve_nondepletion", "Reserve nondepletion rate"),
    )
    colors = plt.cm.turbo([0.12, 0.50, 0.88])
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, max(5.8, 0.27 * len(ordered) + 1.25)),
        sharey=True,
        constrained_layout=True,
    )
    positions = list(range(len(ordered)))
    for axis, (metric_id, title), color in zip(axes, metrics, colors):
        for position, row in enumerate(ordered):
            intervals = row.get("trajectory_level_95_percent_t_intervals")
            if intervals is None:
                axis.text(
                    0.015,
                    position,
                    "NE",
                    ha="left",
                    va="center",
                    fontsize=6.5,
                    color="#555555",
                    fontweight="semibold",
                )
                continue
            interval = intervals[metric_id]
            estimate_key = "mean"
            _draw_interval(
                axis,
                position=position,
                estimate=float(interval[estimate_key]),
                lower=float(interval["lower"]),
                upper=float(interval["upper"]),
                color=color,
            )
        _finish_rate_axis(axis, title, "Rate / normalized area")
    axes[0].set_yticks(positions, [_label(row) for row in ordered])
    axes[0].invert_yaxis()
    fig.suptitle(
        "Part 2 corrected outcomes over fully valid trajectories",
        fontweight="semibold",
    )
    return _save(fig, output_dir, "part2_corrected_outcomes")


def build_paper_visuals(input_path: Path, output_dir: Path) -> list[Path]:
    source, _ = _read_source(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "figure.titlesize": 10.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    paths: list[Path] = []
    with plt.rc_context(style):
        paths.extend(_part0_conditions(plt, source["part0"], output_dir))
        plt.close("all")
        paths.extend(_part1_distribution(plt, source["part1"], output_dir))
        plt.close("all")
        paths.extend(_part2_outcomes(plt, source["part2"], output_dir))
        plt.close("all")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    for path in build_paper_visuals(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
