from pathlib import Path

from data.graphs.part_2_graphs import (
    RESULT_HEADERS,
    _grouped_bar_positions,
    _minimum_x_limits,
    _model_bar_color,
    _model_bar_edge_color,
    _model_legend_handles,
    aggregate_rows,
    build_model_plot_groups,
    build_collapse_day_rows,
    build_final_population_rows,
    build_final_resource_rows,
    build_model_rows,
    build_overall_title,
    build_time_series_title,
    default_out_prefix,
    find_latest_nonempty_csvs_by_model,
    render_collapse_day_chart,
    render_overall_chart,
    render_percent_bar_chart,
    render_time_series_chart,
    z_from_confidence,
)


def _csv_text(*rows: str) -> str:
    return ",".join(RESULT_HEADERS) + "\n" + "\n".join(rows) + ("\n" if rows else "")


def test_aggregate_rows_counts_actions_and_deduplicates_day_summaries() -> None:
    rows = [
        {
            "provider": "ollama",
            "model": "model-a",
            "day": "1",
            "agent": "society_1",
            "action": "RESTRAIN",
            "population_start": "2",
            "population_end": "2",
            "restrain_count": "1",
            "overuse_count": "1",
            "resource_units_remaining": "8",
            "resource_capacity": "10",
            "deaths": "0",
        },
        {
            "provider": "ollama",
            "model": "model-a",
            "day": "1",
            "agent": "society_2",
            "action": "OVERUSE",
            "population_start": "2",
            "population_end": "2",
            "restrain_count": "1",
            "overuse_count": "1",
            "resource_units_remaining": "8",
            "resource_capacity": "10",
            "deaths": "0",
        },
        {
            "provider": "ollama",
            "model": "model-a",
            "day": "2",
            "agent": "society_1",
            "action": "OPTION_A",
            "population_start": "2",
            "population_end": "1",
            "restrain_count": "2",
            "overuse_count": "0",
            "resource_units_remaining": "0",
            "resource_capacity": "10",
            "deaths": "1",
        },
        {
            "provider": "ollama",
            "model": "model-a",
            "day": "2",
            "agent": "society_2",
            "action": "A",
            "population_start": "2",
            "population_end": "1",
            "restrain_count": "2",
            "overuse_count": "0",
            "resource_units_remaining": "0",
            "resource_capacity": "10",
            "deaths": "1",
        },
    ]

    overall, day_summaries = aggregate_rows(rows)

    assert overall["ollama/model-a"].restrain == 3
    assert overall["ollama/model-a"].overuse == 1
    assert overall["ollama/model-a"].skipped == 0
    assert len(day_summaries) == 2
    assert day_summaries[("ollama/model-a", 1)].restrain_count == 1
    assert day_summaries[("ollama/model-a", 2)].population_end == 1
    assert day_summaries[("ollama/model-a", 2)].deaths == 1


def test_final_and_collapse_rows_use_completed_day_summaries() -> None:
    _, day_summaries = aggregate_rows(
        [
            {
                "provider": "ollama",
                "model": "model-a",
                "day": "1",
                "agent": "society_1",
                "action": "RESTRAIN",
                "population_start": "2",
                "population_end": "2",
                "restrain_count": "2",
                "overuse_count": "0",
                "resource_units_remaining": "8",
                "resource_capacity": "10",
                "deaths": "0",
            },
            {
                "provider": "ollama",
                "model": "model-a",
                "day": "2",
                "agent": "society_1",
                "action": "RESTRAIN",
                "population_start": "2",
                "population_end": "1",
                "restrain_count": "2",
                "overuse_count": "0",
                "resource_units_remaining": "0",
                "resource_capacity": "10",
                "deaths": "1",
            },
            {
                "provider": "ollama",
                "model": "model-b",
                "day": "1",
                "agent": "society_1",
                "action": "RESTRAIN",
                "population_start": "4",
                "population_end": "4",
                "restrain_count": "4",
                "overuse_count": "0",
                "resource_units_remaining": "10",
                "resource_capacity": "10",
                "deaths": "0",
            },
        ]
    )

    model_order = ["ollama/model-a", "ollama/model-b"]

    assert build_final_population_rows(day_summaries, model_order) == (
        model_order,
        [50.0, 100.0],
    )
    assert build_final_resource_rows(day_summaries, model_order) == (
        model_order,
        [0.0, 100.0],
    )
    assert build_collapse_day_rows(day_summaries, model_order) == (
        model_order,
        [2.0, 1.0],
        ["2", ">1"],
    )


def test_find_latest_nonempty_csvs_by_model_skips_header_only_and_interrupted_files(
    tmp_path: Path,
) -> None:
    older = tmp_path / "part2__ollama__model-a__n2__d2__water__20260419_210000.csv"
    newer_empty = tmp_path / "part2__ollama__model-a__n2__d2__water__20260420_210000.csv"
    partial = tmp_path / "part2__ollama__model-b__n2__d2__water__20260420_220000.csv"
    partial_meta = tmp_path / "part2__ollama__model-b__n2__d2__water__20260420_220000_meta.json"
    complete_other = tmp_path / "part2__ollama__model-b__n2__d2__water__20260420_230000.csv"
    different_config = tmp_path / "part2__ollama__model-c__n3__d2__water__20260420_230000.csv"

    older.write_text(
        _csv_text("ollama,model-a,1,society_1,RESTRAIN,ok,2,2,1,1,8,10,0,water,2,2,5"),
        encoding="utf-8",
    )
    newer_empty.write_text(_csv_text(), encoding="utf-8")
    partial.write_text(
        _csv_text("ollama,model-b,1,society_1,RESTRAIN,ok,2,2,2,0,10,10,0,water,2,2,5"),
        encoding="utf-8",
    )
    partial_meta.write_text("{}", encoding="utf-8")
    complete_other.write_text(
        _csv_text("ollama,model-b,1,society_1,OVERUSE,ok,2,2,1,1,8,10,0,water,2,2,5"),
        encoding="utf-8",
    )
    different_config.write_text(
        _csv_text("ollama,model-c,1,society_1,RESTRAIN,ok,3,3,3,0,10,10,0,water,2,2,5"),
        encoding="utf-8",
    )

    selected = find_latest_nonempty_csvs_by_model(
        tmp_path,
        society_size=2,
        days=2,
        resource="water",
    )

    assert selected == [older, complete_other]


def test_default_out_prefix_uses_single_stem_or_latest_config_bundle() -> None:
    single = [Path("data/raw/part_2/part2__ollama__llama2__n50__d100__water__20260426_214029.csv")]
    multiple = [
        Path("data/raw/part_2/part2__ollama__llama2__n50__d100__water__20260426_214029.csv"),
        Path("data/raw/part_2/part2__ollama__gpt-oss-20b__n50__d100__water__20260426_161639.csv"),
    ]

    assert default_out_prefix(single) == "part2__ollama__llama2__n50__d100__water__20260426_214029"
    assert default_out_prefix(multiple) == "part2_latest_n50_d100_water_per_model"


def test_model_colors_use_requested_family_palettes_and_instruct_border() -> None:
    assert _model_bar_color("ollama/gpt-oss:20b") == "#048a73"
    assert _model_bar_color("ollama/gpt-oss-safeguard:20b") == "#01483d"
    assert _model_bar_color("ollama/gurubot/gpt-oss-derestricted:20b") == "#34d6bc"
    assert _model_bar_color("ollama/qwen2.5:7b") == "#7416c7"
    assert _model_bar_color("ollama/huihui_ai/qwen2.5-abliterate:7b-instruct") == "#8e38d8"
    assert _model_bar_color("ollama/qwen3.5") == "#a707b5"
    assert _model_bar_color("ollama/sorc/qwen3.5-instruct-uncensored") == "#bf2ece"
    assert _model_bar_color("ollama/llama2") == "#105bcc"
    assert _model_bar_color("ollama/llama2-uncensored") == "#5a95e8"
    assert _model_bar_edge_color("ollama/qwen2.5:7b-instruct") == "#000000"
    assert _model_bar_edge_color("ollama/qwen2.5:7b") != "#000000"


def test_build_model_plot_groups_uses_requested_subfolders() -> None:
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

    assert build_model_plot_groups(labels) == [
        ("gpt-oss:20b-plots", labels[0:3]),
        ("llama2-plots", labels[3:5]),
        ("qwen2.5-plots", labels[5:7]),
        ("qwen2.5-instruct-plots", labels[7:9]),
        ("qwen3.5-plots", labels[9:11]),
        ("qwen3.5-instruct-plots", labels[11:13]),
    ]


def test_part_2_legends_and_axis_limits_only_reflect_visible_models() -> None:
    handles = _model_legend_handles(["ollama/llama2", "ollama/llama2-uncensored"])

    assert [handle.get_label() for handle in handles] == [
        "Llama",
        "Standard",
        "Unrestricted",
    ]

    left, right = _minimum_x_limits(-0.45, 1.35)
    assert round(right - left, 6) == 6.0
    assert _minimum_x_limits(0.0, 8.0) == (0.0, 8.0)


def test_part_2_model_bar_positions_are_evenly_spaced() -> None:
    two_positions = _grouped_bar_positions(["ollama/llama2", "ollama/llama2-uncensored"])
    three_positions = _grouped_bar_positions(
        ["ollama/llama2", "ollama/llama2-uncensored", "ollama/qwen2.5:7b"]
    )

    assert two_positions == [1.5, 4.5]
    assert [round(position, 6) for position in three_positions] == [1.2, 3.0, 4.8]


def test_render_functions_write_part_2_graph_files(tmp_path: Path) -> None:
    rows = [
        {
            "provider": "ollama",
            "model": "model-a",
            "day": "1",
            "agent": "society_1",
            "action": "RESTRAIN",
            "population_start": "2",
            "population_end": "2",
            "restrain_count": "1",
            "overuse_count": "1",
            "resource_units_remaining": "8",
            "resource_capacity": "10",
            "deaths": "0",
        },
        {
            "provider": "ollama",
            "model": "model-a",
            "day": "1",
            "agent": "society_2",
            "action": "OVERUSE",
            "population_start": "2",
            "population_end": "2",
            "restrain_count": "1",
            "overuse_count": "1",
            "resource_units_remaining": "8",
            "resource_capacity": "10",
            "deaths": "0",
        },
        {
            "provider": "ollama",
            "model": "model-a",
            "day": "2",
            "agent": "society_1",
            "action": "RESTRAIN",
            "population_start": "2",
            "population_end": "1",
            "restrain_count": "1",
            "overuse_count": "0",
            "resource_units_remaining": "0",
            "resource_capacity": "10",
            "deaths": "1",
        },
    ]
    overall, day_summaries = aggregate_rows(rows)
    labels, rates, errors, totals = build_model_rows(overall, z_from_confidence(0.95), "wilson")

    overall_output = tmp_path / "restraint.png"
    render_overall_chart(
        labels,
        rates,
        errors,
        totals,
        title=build_overall_title(source_note="synthetic", ci_label="Wilson 95% CI"),
        output=overall_output,
    )

    series_output = tmp_path / "resource_by_day.png"
    render_time_series_chart(
        day_summaries,
        labels,
        metric="resource-by-day",
        title=build_time_series_title(metric_label="Shared reserve remaining", source_note="synthetic"),
        ylabel="Resource remaining (%)",
        output=series_output,
    )

    percent_output = tmp_path / "final_population.png"
    population_labels, population_values = build_final_population_rows(day_summaries, labels)
    render_percent_bar_chart(
        population_labels,
        population_values,
        title="Final population by model",
        ylabel="Final population remaining (%)",
        output=percent_output,
    )

    collapse_output = tmp_path / "collapse.png"
    collapse_labels, collapse_values, collapse_annotations = build_collapse_day_rows(
        day_summaries,
        labels,
    )
    render_collapse_day_chart(
        collapse_labels,
        collapse_values,
        collapse_annotations,
        title="First depletion day by model",
        output=collapse_output,
    )

    assert overall_output.exists()
    assert series_output.exists()
    assert percent_output.exists()
    assert collapse_output.exists()
