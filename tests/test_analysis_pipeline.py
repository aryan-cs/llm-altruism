import csv
import json
from pathlib import Path

from analysis.build_manifest import build_manifest
from analysis.summarize_results import _wilson_interval
from analysis.validation import validate_part2_file
from experiments.part2.part_2 import RESULT_HEADERS


def _write_part2_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_HEADERS})


def test_part2_validation_flags_reasoning_mismatch_without_failing(tmp_path: Path) -> None:
    csv_path = tmp_path / "part2__openai__model__n1__d1__water__20260429_000000.csv"
    _write_part2_rows(
        csv_path,
        [
            {
                "provider": "openai",
                "model": "model",
                "day": 1,
                "agent": "society_1",
                "action": "OVERUSE",
                "reasoning": "This will keep the reserve stable and benefit the group.",
                "population_start": 1,
                "population_end": 1,
                "restrain_count": 0,
                "overuse_count": 1,
                "resource_units_remaining": 8,
                "resource_capacity": 10,
                "deaths": 0,
                "resource": "water",
                "selfish_gain": 2,
                "depletion_units": 2,
                "community_benefit": 5,
            }
        ],
    )

    report = validate_part2_file(csv_path)

    assert report.status == "warn"
    assert report.metrics["reasoning_mismatch_flags"] == 1
    assert report.errors == []


def test_manifest_links_metadata_without_embedding_raw_metadata(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    part_dir = raw_dir / "part_1"
    csv_path = part_dir / "part1__openai__model__smoke-1prompts__20260429_000000.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("provider,model\nopenai,model\n", encoding="utf-8")
    csv_path.with_name(f"{csv_path.stem}_meta.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "provider": "openai",
                "model": "model",
                "timestamp": "20260429_000000",
                "prompt_config_hash": "abc",
            }
        ),
        encoding="utf-8",
    )

    entries = build_manifest(raw_dir)

    assert len(entries) == 1
    assert entries[0]["metadata_status"] == "complete"
    assert "metadata" not in entries[0]


def test_wilson_interval_handles_empty_and_nonempty_rates() -> None:
    assert _wilson_interval(0, 0) == (0.0, 0.0)
    low, high = _wilson_interval(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0
