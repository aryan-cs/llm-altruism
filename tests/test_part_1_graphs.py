from pathlib import Path

from data.graphs.part_1_graphs import (
    aggregate_rows,
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
    single = [Path("results/part_1/part1__ollama__llama2__full__20260419_210124.csv")]
    multiple = [
        Path("results/part_1/part1__ollama__llama2__full__20260419_210124.csv"),
        Path("results/part_1/part1__ollama__gpt-oss-20b__full__20260420_005501.csv"),
    ]

    assert default_out_prefix(single, scope="full") == "part1__ollama__llama2__full__20260419_210124"
    assert default_out_prefix(multiple, scope="full") == "part1_latest_full_per_model"
