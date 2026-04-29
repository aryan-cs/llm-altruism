"""Render cross-part scatter plots from canonical summary tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

from data.graphs.part_2_graphs import (
    EDGE_COLOR,
    _model_bar_color,
    _model_leaf,
    _model_legend_handles,
)

DEFAULT_TABLES_DIR = Path("data") / "analysis" / "tables"
DEFAULT_GRAPHS_DIR = Path("data") / "graphs" / "cross_part" / "master-plots"
DEFAULT_INDIVIDUAL_GRAPHS_DIR = Path("data") / "graphs" / "cross_part" / "individual-plots"
MATRIX_POINT_SIZE = 20
INDIVIDUAL_POINT_SIZE = 30

PLOT_SPECS = [
    (
        "safety_refusal_rate",
        "cooperation_rate",
        "Safety refusal (%)",
        "Cooperation (%)",
        "safety_refusal_vs_cooperation.png",
    ),
    (
        "safety_refusal_rate",
        "restraint_rate",
        "Safety refusal (%)",
        "Commons restraint (%)",
        "safety_refusal_vs_restraint.png",
    ),
    (
        "safety_refusal_rate",
        "final_population",
        "Safety refusal (%)",
        "Final population",
        "safety_refusal_vs_final_population.png",
    ),
    (
        "cooperation_rate",
        "restraint_rate",
        "Cooperation (%)",
        "Commons restraint (%)",
        "cooperation_vs_restraint.png",
    ),
    (
        "cooperation_rate",
        "final_population",
        "Cooperation (%)",
        "Final population",
        "cooperation_vs_final_population.png",
    ),
    (
        "restraint_rate",
        "final_population",
        "Commons restraint (%)",
        "Final population",
        "restraint_vs_final_population.png",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cross-part scatter plots from summary tables."
    )
    parser.add_argument(
        "--tables-dir",
        default=str(DEFAULT_TABLES_DIR),
        help="Directory containing cross_part_model_summary.csv and cross_part_correlations.csv.",
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(DEFAULT_GRAPHS_DIR),
        help="Directory for cross-part graph outputs.",
    )
    parser.add_argument(
        "--individual-graphs-dir",
        default=str(DEFAULT_INDIVIDUAL_GRAPHS_DIR),
        help="Directory for individual cross-part scatter outputs.",
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def _percentage(row: dict[str, str], key: str) -> float:
    return float(row[key]) * 100.0


def _correlation_lookup(rows: Sequence[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["metric_x"], row["metric_y"]): row for row in rows}


def _short_label(model: str) -> str:
    leaf = _model_leaf(f"ollama/{model}")
    replacements = {
        "gpt-oss-safeguard:20b": "gpt-oss-sg",
        "gpt-oss-derestricted:20b": "gpt-oss-der",
        "qwen2.5:7b-instruct": "qwen2.5-inst",
        "qwen2.5-abliterate:7b-instruct": "qwen2.5-abl-inst",
        "qwen2.5-abliterate:7b": "qwen2.5-abl",
        "qwen3.5-instruct-uncensored": "qwen3.5-inst-unc",
        "qwen3.5-uncensored:9b": "qwen3.5-unc",
    }
    return replacements.get(leaf, leaf)


def _metric_value(row: dict[str, str], key: str) -> float:
    return _percentage(row, key) if key.endswith("_rate") else float(row[key])


def _correlation_subtitle(correlation_rows: Sequence[dict[str, str]], x_key: str, y_key: str) -> str:
    corr = _correlation_lookup(correlation_rows).get((x_key, y_key), {})
    if not corr:
        return ""
    return (
        f"Pearson r={float(corr['pearson_r']):.2f}; "
        f"Spearman rho={float(corr['spearman_r']):.2f}"
    )


def _style_scatter_axis(axis, x_key: str, y_key: str, x_label: str, y_label: str) -> None:
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, axis="both", alpha=0.28, linewidth=0.6)
    if x_key.endswith("_rate"):
        axis.set_xlim(-5, 105)
    if y_key.endswith("_rate"):
        axis.set_ylim(-5, 105)
    elif y_key == "final_population":
        axis.set_ylim(-4, 56)


def _bbox_overlap_area(left, right) -> float:
    width = max(0.0, min(left.x1, right.x1) - max(left.x0, right.x0))
    height = max(0.0, min(left.y1, right.y1) - max(left.y0, right.y0))
    return width * height


def _outside_axis_penalty(bbox, axis_bbox) -> float:
    padding = 3.0
    left = max(0.0, axis_bbox.x0 + padding - bbox.x0)
    right = max(0.0, bbox.x1 - (axis_bbox.x1 - padding))
    bottom = max(0.0, axis_bbox.y0 + padding - bbox.y0)
    top = max(0.0, bbox.y1 - (axis_bbox.y1 - padding))
    return left + right + bottom + top


def _candidate_offsets() -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    directions = [
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
    ]
    for radius in (10, 16, 23, 31, 41, 54, 70, 88, 110):
        for dx, dy in directions:
            offsets.append((dx * radius, dy * radius))
    return offsets


def _point_collision_boxes(fig, axis, points, marker_size: int):
    from matplotlib.transforms import Bbox

    radius_px = max(7.0, math.sqrt(marker_size) * fig.dpi / 72.0 + 4.0)
    boxes = []
    for x, y, _label, _color in points:
        x_px, y_px = axis.transData.transform((x, y))
        boxes.append(
            Bbox.from_extents(
                x_px - radius_px,
                y_px - radius_px,
                x_px + radius_px,
                y_px + radius_px,
            )
        )
    return boxes


def _point_density_order(axis, points: Sequence[tuple[float, float, str, str]]):
    x_min, x_max = axis.get_xlim()
    y_min, y_max = axis.get_ylim()
    x_span = max(1e-9, x_max - x_min)
    y_span = max(1e-9, y_max - y_min)

    def density(point: tuple[float, float, str, str]) -> float:
        score = 0.0
        x, y = point[0], point[1]
        for other in points:
            if other is point:
                continue
            dx = (x - other[0]) / x_span
            dy = (y - other[1]) / y_span
            distance = math.hypot(dx, dy)
            score += 1.0 / max(0.035, distance)
        return score

    return sorted(points, key=lambda item: (-density(item), -item[1], item[0], item[2]))


def _annotate_label(
    axis,
    x: float,
    y: float,
    label: str,
    offset: tuple[int, int],
    fontsize: int,
    color: str,
):
    horizontal = "left" if offset[0] >= 0 else "right"
    vertical = "bottom" if offset[1] >= 0 else "top"
    if offset[0] == 0:
        horizontal = "center"
    if offset[1] == 0:
        vertical = "center"
    return axis.annotate(
        label,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        color="#111827",
        ha=horizontal,
        va=vertical,
        annotation_clip=False,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": color,
            "edgecolor": EDGE_COLOR,
            "linewidth": 0.4,
            "alpha": 0.20,
        },
        zorder=5,
    )


def _clamp_annotation_inside_axis(fig, axis, annotation) -> None:
    """Move a placed offset annotation back inside its final axes bbox."""
    for _ in range(3):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = annotation.get_window_extent(renderer)
        axis_bbox = axis.get_window_extent(renderer)
        padding = 3.0
        shift_x = 0.0
        shift_y = 0.0

        if bbox.x0 < axis_bbox.x0 + padding:
            shift_x += axis_bbox.x0 + padding - bbox.x0
        if bbox.x1 > axis_bbox.x1 - padding:
            shift_x -= bbox.x1 - (axis_bbox.x1 - padding)
        if bbox.y0 < axis_bbox.y0 + padding:
            shift_y += axis_bbox.y0 + padding - bbox.y0
        if bbox.y1 > axis_bbox.y1 - padding:
            shift_y -= bbox.y1 - (axis_bbox.y1 - padding)

        if abs(shift_x) < 0.5 and abs(shift_y) < 0.5:
            return

        current_x, current_y = annotation.get_position()
        annotation.set_position(
            (
                current_x + shift_x * 72.0 / fig.dpi,
                current_y + shift_y * 72.0 / fig.dpi,
            )
        )


def _place_labels(
    fig,
    axis,
    points: Sequence[tuple[float, float, str, str]],
    *,
    fontsize: int,
    marker_size: int,
) -> None:
    candidate_offsets = _candidate_offsets()
    placed = []
    renderer = fig.canvas.get_renderer()
    axis_bbox = axis.get_window_extent(renderer)
    point_boxes = _point_collision_boxes(fig, axis, points, marker_size)

    ordered_points = _point_density_order(axis, points)
    for x, y, label, color in ordered_points:
        best_offset = candidate_offsets[0]
        best_score = float("inf")

        for offset in candidate_offsets:
            annotation = _annotate_label(axis, x, y, label, offset, fontsize, color)
            bbox = annotation.get_window_extent(renderer).expanded(1.08, 1.16)
            label_overlap = sum(_bbox_overlap_area(bbox, existing) for existing in placed)
            point_overlap = sum(_bbox_overlap_area(bbox, point_box) for point_box in point_boxes)
            outside = _outside_axis_penalty(bbox, axis_bbox)
            distance = math.hypot(offset[0], offset[1])
            score = (
                label_overlap * 40.0
                + point_overlap * 200.0
                + outside * 100000.0
                + distance * 0.25
            )
            annotation.remove()
            if score < best_score:
                best_score = score
                best_offset = offset
            if score == 0:
                break

        final_annotation = _annotate_label(axis, x, y, label, best_offset, fontsize, color)
        _clamp_annotation_inside_axis(fig, axis, final_annotation)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        placed.append(
            final_annotation.get_window_extent(renderer).expanded(1.08, 1.16)
        )


def _draw_scatter_points(
    axis,
    model_rows: Sequence[dict[str, str]],
    *,
    x_key: str,
    y_key: str,
    marker_size: int,
) -> list[tuple[float, float, str, str]]:
    points: list[tuple[float, float, str, str]] = []
    for row in model_rows:
        model = row["model"]
        model_label = f"ollama/{model}"
        x = _metric_value(row, x_key)
        y = _metric_value(row, y_key)
        axis.scatter(
            x,
            y,
            s=marker_size,
            color=_model_bar_color(model_label),
            edgecolors=EDGE_COLOR,
            linewidths=0.7,
            zorder=3,
        )
        points.append((x, y, _short_label(model), _model_bar_color(model_label)))
    return points


def render_scatter_matrix(
    model_rows: Sequence[dict[str, str]],
    correlation_rows: Sequence[dict[str, str]],
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 11.2), dpi=150)
    axes_flat = [axis for row in axes for axis in row]
    points_by_axis = []

    for axis, (x_key, y_key, x_label, y_label, _filename) in zip(axes_flat, PLOT_SPECS):
        points = _draw_scatter_points(
            axis,
            model_rows,
            x_key=x_key,
            y_key=y_key,
            marker_size=MATRIX_POINT_SIZE,
        )
        axis.set_title(_correlation_subtitle(correlation_rows, x_key, y_key), fontsize=10)
        _style_scatter_axis(axis, x_key, y_key, x_label, y_label)
        points_by_axis.append((axis, points))

    handles = _model_legend_handles([f"ollama/{row['model']}" for row in model_rows])
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=max(1, len(handles)),
        frameon=False,
        title="Model family",
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        "Cross-Part Relations Across Model-Level Outcomes",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.91))
    fig.canvas.draw()
    for axis, points in points_by_axis:
        _place_labels(fig, axis, points, fontsize=5.5, marker_size=MATRIX_POINT_SIZE)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_individual_scatters(
    model_rows: Sequence[dict[str, str]],
    correlation_rows: Sequence[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with uv: uv add matplotlib (or uv sync)."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    handles = _model_legend_handles([f"ollama/{row['model']}" for row in model_rows])
    outputs: list[Path] = []

    for x_key, y_key, x_label, y_label, filename in PLOT_SPECS:
        fig, axis = plt.subplots(figsize=(9.2, 6.8), dpi=150)
        points = _draw_scatter_points(
            axis,
            model_rows,
            x_key=x_key,
            y_key=y_key,
            marker_size=INDIVIDUAL_POINT_SIZE,
        )
        _style_scatter_axis(axis, x_key, y_key, x_label, y_label)
        axis.set_title(
            f"{x_label} vs. {y_label}\n{_correlation_subtitle(correlation_rows, x_key, y_key)}",
            fontsize=12,
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=max(1, len(handles)),
            frameon=False,
            title="Model family",
            bbox_to_anchor=(0.5, 0.985),
        )
        fig.tight_layout(rect=(0.03, 0.03, 0.98, 0.88))
        fig.canvas.draw()
        _place_labels(fig, axis, points, fontsize=8, marker_size=INDIVIDUAL_POINT_SIZE)
        output = output_dir / filename
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        outputs.append(output)

    return outputs


def main() -> int:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    graphs_dir = Path(args.graphs_dir)
    individual_graphs_dir = Path(args.individual_graphs_dir)
    model_rows = _read_rows(tables_dir / "cross_part_model_summary.csv")
    correlation_rows = _read_rows(tables_dir / "cross_part_correlations.csv")
    output = graphs_dir / "cross_part_scatter_matrix.png"
    render_scatter_matrix(model_rows, correlation_rows, output)
    print(f"wrote: {output}")
    for individual_output in render_individual_scatters(
        model_rows,
        correlation_rows,
        individual_graphs_dir,
    ):
        print(f"wrote: {individual_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
