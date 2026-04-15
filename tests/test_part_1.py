import csv
import json
from pathlib import Path

from experiments.part1.part_1 import run_part_1


def test_run_part_1_writes_results_incrementally(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight", lambda *args, **kwargs: None
    )

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del query, json_mode
        if self.id == "A":
            return json.dumps({"reasoning": "A reasoning", "action": "SNITCH"})
        return json.dumps({"reasoning": "B reasoning", "action": "STAY_SILENT"})

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", fake_query)

    csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        prompt_style="direct",
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"agent": "A", "action": "SNITCH", "reasoning": "A reasoning"},
        {"agent": "B", "action": "STAY_SILENT", "reasoning": "B reasoning"},
    ]


def test_run_part_1_preserves_first_agent_row_when_interrupted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "experiments.part1.part_1.run_experiment_preflight", lambda *args, **kwargs: None
    )

    calls = {"count": 0}

    def fake_query(self, query: str, json_mode: bool = False) -> str:
        del self, query, json_mode
        calls["count"] += 1
        if calls["count"] == 1:
            return json.dumps({"reasoning": "A reasoning", "action": "SNITCH"})
        raise KeyboardInterrupt()

    monkeypatch.setattr("experiments.part1.part_1.Agent1.query", fake_query)

    csv_path = run_part_1(
        provider="openai",
        model="gpt-4.1-mini",
        prompt_style="direct",
    )

    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"agent": "A", "action": "SNITCH", "reasoning": "A reasoning"},
    ]
