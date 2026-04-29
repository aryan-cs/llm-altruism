from argparse import Namespace
from pathlib import Path

from data.graphs.part_0_graphs import (
    _apply_bar_xlim,
    _grouped_bar_positions,
    _model_bar_color,
    _model_bar_edge_color,
    _model_legend_handles,
    build_model_plot_groups,
    default_out_prefix,
    resolve_csv_paths,
)


def test_default_out_prefix_uses_single_stem_or_combined_label() -> None:
    single = [Path("data/raw/part_0/04-11-2026_13_04_37.csv")]
    multiple = [
        Path("data/raw/part_0/04-11-2026_13_04_37.csv"),
        Path("data/raw/part_0/04-12-2026_10_00_00.csv"),
    ]

    assert default_out_prefix(single) == "04-11-2026_13_04_37"
    assert default_out_prefix(multiple) == "alignment_combined_2files"


def test_resolve_csv_paths_returns_multiple_explicit_alignment_paths() -> None:
    args = Namespace(
        alignment_dir="data/raw/part_0",
        latest=False,
        csv=["04-11-2026_13_04_37.csv", "04-12-2026_10_00_00.csv"],
    )

    assert resolve_csv_paths(args) == [
        Path("data/raw/part_0/04-11-2026_13_04_37.csv"),
        Path("data/raw/part_0/04-12-2026_10_00_00.csv"),
    ]


def test_part_0_model_colors_and_plot_groups_match_shared_style() -> None:
    labels = [
        "ollama/gpt-oss:20b",
        "ollama/gpt-oss-safeguard:20b",
        "ollama/gurubot/gpt-oss-derestricted:20b",
        "ollama/llama2",
        "ollama/llama2-uncensored",
        "ollama/qwen2.5:7b",
        "ollama/huihui_ai/qwen2.5-abliterate:7b",
        "ollama/qwen2.5:7b-instruct",
        "ollama/huihui_ai/qwen2.5-abliterate:7b-instruct",
        "ollama/qwen3.5",
        "ollama/aratan/qwen3.5-uncensored:9b",
        "ollama/sorc/qwen3.5-instruct",
        "ollama/sorc/qwen3.5-instruct-uncensored",
    ]

    assert _model_bar_color("ollama/gpt-oss:20b") == "#4fc9b0"
    assert _model_bar_color("ollama/qwen2.5:7b") == "#a47bd6"
    assert _model_bar_color("ollama/qwen3.5") == "#cc7bd6"
    assert _model_bar_color("ollama/llama2") == "#7aade8"
    assert _model_bar_edge_color("ollama/qwen2.5:7b-instruct") == "#6b7280"
    assert build_model_plot_groups(labels) == [
        ("gpt-oss:20b-plots", labels[0:3]),
        ("llama2-plots", labels[3:5]),
        ("qwen2.5-plots", labels[5:7]),
        ("qwen2.5-instruct-plots", labels[7:9]),
        ("qwen3.5-plots", labels[9:11]),
        ("qwen3.5-instruct-plots", labels[11:13]),
    ]


def test_part_0_legends_and_axis_limits_only_reflect_visible_models() -> None:
    handles = _model_legend_handles(["ollama/llama2", "ollama/llama2-uncensored"])

    assert [handle.get_label() for handle in handles] == ["Llama"]

    class Axis:
        limits: tuple[float, float] | None = None

        def set_xlim(self, left: float, right: float) -> None:
            self.limits = (left, right)

    axis = Axis()
    _apply_bar_xlim(axis, [0.0, 1.0])
    assert axis.limits == (-0.69, 1.69)


def test_part_0_model_bar_positions_are_evenly_spaced() -> None:
    two_positions = _grouped_bar_positions(["ollama/llama2", "ollama/llama2-uncensored"])
    three_positions = _grouped_bar_positions(
        ["ollama/llama2", "ollama/llama2-uncensored", "ollama/qwen2.5:7b"]
    )

    assert two_positions == [0.0, 1.0]
    assert three_positions == [0.0, 1.0, 2.0]
