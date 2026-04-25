import csv
import io
import json
from pathlib import Path

from rich.console import Console

from experiments.part2.part_2 import run_part_2, run_part_2_until_complete


def test_run_part_2_writes_multi_day_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del query, json_mode
        return json.dumps(
            {
                "reasoning": "Protecting the commons is better for long-term stability.",
                "action": "RESTRAIN",
            }
        )

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", fake_query)

    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=3,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
    )

    result_path = Path(csv_path)
    assert result_path.exists()

    with result_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 6
    assert result_path.name.startswith("part2__openai__gpt-4-1-mini__n3__d2__water__")
    assert rows[0]["provider"] == "openai"
    assert rows[0]["model"] == "gpt-4.1-mini"
    assert rows[0]["day"] == "1"
    assert rows[-1]["day"] == "2"
    assert {row["action"] for row in rows} == {"RESTRAIN"}
    assert not list((tmp_path / "results" / "part_2").glob("*_meta.json"))


def test_run_part_2_resumes_from_completed_days(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    calls = {"count": 0}

    def interrupted_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        if calls["count"] <= 2:
            return json.dumps(
                {"reasoning": "First day completes.", "action": "RESTRAIN"}
            )
        raise KeyboardInterrupt()

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", interrupted_query)

    csv_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
    )

    metadata_files = list((tmp_path / "results" / "part_2").glob("*_meta.json"))
    assert len(metadata_files) == 1

    def resumed_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        return json.dumps(
            {"reasoning": "Second day completes after resume.", "action": "RESTRAIN"}
        )

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", resumed_query)

    resumed_csv_path = run_part_2(resume=True)

    assert resumed_csv_path == csv_path
    with Path(resumed_csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert [row["day"] for row in rows] == ["1", "1", "2", "2"]
    assert rows[-1]["reasoning"] == "Second day completes after resume."
    assert not list((tmp_path / "results" / "part_2").glob("*_meta.json"))


def test_run_part_2_headless_prints_compact_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.part2.part_2.Agent2.query",
        lambda self, query, json_mode=False: json.dumps(
            {"reasoning": "Compact progress.", "action": "RESTRAIN"}
        ),
    )

    buffer = io.StringIO()
    test_console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=200,
    )
    monkeypatch.setattr("experiments.part2.part_2.console", test_console)

    run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=1,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        headless=True,
    )

    output = buffer.getvalue()
    assert "Model openai/gpt-4.1-mini [day 1/1] [--------------------] RUNNING" in output
    assert "Model openai/gpt-4.1-mini [day 1/1] [####################] done" in output


def test_run_part_2_until_complete_resumes_matching_partial_run(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part2.part_2.run_experiment_preflight",
        lambda *args, **kwargs: None,
    )

    calls = {"count": 0}

    def flaky_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        if calls["count"] == 3:
            raise KeyboardInterrupt()
        return json.dumps({"reasoning": "eventual success", "action": "RESTRAIN"})

    monkeypatch.setattr("experiments.part2.part_2.Agent2.query", flaky_query)

    partial_path = run_part_2(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
    )
    assert Path(partial_path).with_name(f"{Path(partial_path).stem}_meta.json").exists()

    final_path = run_part_2_until_complete(
        provider="openai",
        model="gpt-4.1-mini",
        society_size=2,
        days=2,
        resource="water",
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        headless=True,
    )

    assert final_path == partial_path
    with Path(final_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert not Path(final_path).with_name(f"{Path(final_path).stem}_meta.json").exists()
