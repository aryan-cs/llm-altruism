import csv
import json
from pathlib import Path

from experiments.part2.part_2 import run_part_2


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
    assert rows[0]["day"] == "1"
    assert rows[-1]["day"] == "2"
    assert {row["action"] for row in rows} == {"RESTRAIN"}
