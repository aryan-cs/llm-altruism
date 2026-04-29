from pathlib import Path

import pytest

from data.graphs.part_1_graphs import (
    MODEL_BAR_WIDTH,
    _apply_bar_xlim,
    _breakdown_bar_width,
    _grouped_bar_positions,
    _model_bar_color,
    _model_bar_edge_color,
    _model_legend_handles,
    aggregate_rows,
    build_model_plot_groups,
    default_out_prefix,
    find_latest_nonempty_csvs_by_model,
)


def test_aggregate_rows_counts_cooperative_choices_by_game_definition() -> None:
    rows = [
        {
            "provider": "ollama",
            "model": "model-a",
            "game": "prisoners_dilemma",
            "frame": "self_direct",
            "domain": "crime",
            "presentation": "narrative",
            "action": "COOPERATE",
        },
        {
            "provider": "ollama",
            "model": "model-a",
            "game": "prisoners_dilemma",
            "frame": "self_direct",
            "domain": "sports",
            "presentation": "structured",
            "action": "DEFECT",
        },
        {
            "provider": "ollama",
            "model": "model-a",
            "game": "temptation_or_commons",
            "frame": "advice",
            "domain": "crime",
            "presentation": "narrative",
            "action": "RESTRAIN",
        },
        {
            "provider": "ollama",
            "model": "model-a",
            "game": "temptation_or_commons",
            "frame": "advice",
            "domain": "sports",
            "presentation": "structured",
            "action": "OVERUSE",
        },
    ]

    overall, by_frame = aggregate_rows(rows, "frame")

    assert overall["ollama/model-a"].cooperative == 2
    assert overall["ollama/model-a"].selfish == 2
    assert overall["ollama/model-a"].skipped == 0

    assert by_frame[("ollama/model-a", "self_direct")].cooperative == 1
    assert by_frame[("ollama/model-a", "self_direct")].selfish == 1
    assert by_frame[("ollama/model-a", "advice")].cooperative == 1
    assert by_frame[("ollama/model-a", "advice")].selfish == 1


def test_find_latest_nonempty_csvs_by_model_skips_header_only_files(tmp_path: Path) -> None:
    older = tmp_path / "part1__ollama__model-a__full__20260419_210000.csv"
    newer_empty = tmp_path / "part1__ollama__model-a__full__20260420_210000.csv"
    newer_other = tmp_path / "part1__ollama__model-b__full__20260420_220000.csv"

    older.write_text(
        "provider,model,game,frame,domain,presentation,prompt_id,action,justification,prompt_text\n"
        "ollama,model-a,prisoners_dilemma,self_direct,crime,narrative,p1,COOPERATE,ok,prompt\n",
        encoding="utf-8",
    )
    newer_empty.write_text(
        "provider,model,game,frame,domain,presentation,prompt_id,action,justification,prompt_text\n",
        encoding="utf-8",
    )
    newer_other.write_text(
        "provider,model,game,frame,domain,presentation,prompt_id,action,justification,prompt_text\n"
        "ollama,model-b,temptation_or_commons,self_direct,crime,narrative,p2,RESTRAIN,ok,prompt\n",
        encoding="utf-8",
    )

    selected = find_latest_nonempty_csvs_by_model(tmp_path, scope="full")

    assert selected == [older, newer_other]


def test_find_latest_nonempty_csvs_by_model_skips_interrupted_partial_runs(tmp_path: Path) -> None:
    partial = tmp_path / "part1__ollama__model-a__full__20260420_210000.csv"
    partial_meta = tmp_path / "part1__ollama__model-a__full__20260420_210000_meta.json"
    complete = tmp_path / "part1__ollama__model-a__full__20260420_220000.csv"

    partial.write_text(
        "provider,model,game,frame,domain,presentation,prompt_id,action,justification,prompt_text\n"
        "ollama,model-a,prisoners_dilemma,self_direct,crime,narrative,p1,COOPERATE,ok,prompt\n",
        encoding="utf-8",
    )
    partial_meta.write_text("{}", encoding="utf-8")
    complete.write_text(
        "provider,model,game,frame,domain,presentation,prompt_id,action,justification,prompt_text\n"
        "ollama,model-a,prisoners_dilemma,self_direct,crime,narrative,p2,COOPERATE,ok,prompt\n",
        encoding="utf-8",
    )

    selected = find_latest_nonempty_csvs_by_model(tmp_path, scope="full")

    assert selected == [complete]


def test_default_out_prefix_uses_single_stem_or_latest_bundle() -> None:
    single = [Path("data/raw/part_1/part1__ollama__llama2__full__20260419_210124.csv")]
    multiple = [
        Path("data/raw/part_1/part1__ollama__llama2__full__20260419_210124.csv"),
        Path("data/raw/part_1/part1__ollama__gpt-oss-20b__full__20260420_005501.csv"),
    ]

    assert default_out_prefix(single, scope="full") == "part1__ollama__llama2__full__20260419_210124"
    assert default_out_prefix(multiple, scope="full") == "part1_latest_full_per_model"


def test_part_1_model_colors_and_plot_groups_match_shared_style() -> None:
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
    assert _model_bar_edge_color("ollama/qwen2.5:7b-instruct") == "none"
    assert build_model_plot_groups(labels) == [
        ("gpt-oss:20b-plots", labels[0:3]),
        ("llama2-plots", labels[3:5]),
        ("qwen2.5-plots", labels[5:7]),
        ("qwen2.5-instruct-plots", labels[7:9]),
        ("qwen3.5-plots", labels[9:11]),
        ("qwen3.5-instruct-plots", labels[11:13]),
    ]


def test_part_1_legends_and_axis_limits_only_reflect_visible_models() -> None:
    handles = _model_legend_handles(["ollama/llama2", "ollama/llama2-uncensored"])

    assert [handle.get_label() for handle in handles] == ["Llama"]

    class Axis:
        limits: tuple[float, float] | None = None

        def set_xlim(self, left: float, right: float) -> None:
            self.limits = (left, right)

    axis = Axis()
    _apply_bar_xlim(axis, [0.0, 1.0])
    assert axis.limits == (-1.0, 2.0)


def test_part_1_model_bar_positions_separate_instruct_groups() -> None:
    positions = _grouped_bar_positions(
        [
            "ollama/qwen3.5",
            "ollama/aratan/qwen3.5-uncensored:9b",
            "ollama/sorc/qwen3.5-instruct",
            "ollama/sorc/qwen3.5-instruct-uncensored",
        ]
    )

    assert positions[1] - positions[0] == pytest.approx(MODEL_BAR_WIDTH)
    assert positions[2] - positions[1] > MODEL_BAR_WIDTH
    assert positions[3] - positions[2] == pytest.approx(MODEL_BAR_WIDTH)


def test_part_1_individual_breakdown_bar_width_leaves_model_gap() -> None:
    width = _breakdown_bar_width([0.0, 1.0], 4, cluster=False)

    assert width == MODEL_BAR_WIDTH / 4
