import csv
import json
import zipfile
from pathlib import Path

from analysis.build_manifest import build_manifest
from analysis.build_supplement import MANIFEST_NAME, build_supplement, collect_supplement_files
from analysis.summarize_results import (
    _part0_compliance_value,
    _part1_factor_decomposition_rows,
    _part2_normalized_auc,
    _pearson_correlation,
    _spearman_correlation,
    _wilson_interval,
)
from analysis.validation import validate_part2_file
from experiments.part2.part_2 import RESULT_HEADERS


def test_part0_rejudged_schema_prefers_response_only_label() -> None:
    assert _part0_compliance_value(
        {"complied?": "true", "new_complied": "false"}
    ) == "false"


def test_part0_rejudged_schema_never_falls_back_for_unjudged_row() -> None:
    assert _part0_compliance_value({"complied?": "true", "new_complied": ""}) == ""


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


def test_part2_validation_checks_day_agents_and_first_transition(tmp_path: Path) -> None:
    csv_path = tmp_path / "part2__openai__model__n2__d3__water__20260429_000000.csv"
    shared = {
        "provider": "openai",
        "model": "model",
        "reasoning": "brief",
        "population_start": 2,
        "population_end": 2,
        "restrain_count": 1,
        "overuse_count": 1,
        "resource_units_remaining": 9,
        "resource_capacity": 10,
        "deaths": 0,
        "resource": "water",
        "selfish_gain": 2,
        "depletion_units": 2,
        "community_benefit": 5,
    }
    _write_part2_rows(
        csv_path,
        [
            {**shared, "day": 1, "agent": "society_1", "action": "RESTRAIN"},
            {**shared, "day": 1, "agent": "society_1", "action": "OVERUSE"},
            {**shared, "day": 3, "agent": "society_1", "action": "RESTRAIN"},
            {**shared, "day": 3, "agent": "society_2", "action": "OVERUSE"},
        ],
    )

    report = validate_part2_file(csv_path)

    assert report.status == "fail"
    assert report.metrics["day_gaps"] == 1
    assert report.metrics["duplicate_agent_days"] == 1
    assert report.metrics["transition_errors"] >= 2


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


def test_supplement_builder_excludes_part0_raw_and_generated_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    for relative_path, content in {
        "README.md": "readme",
        "LICENSE": "license",
        ".env.example": "ENV=value\n",
        "pyproject.toml": "[project]\nname='example'\n",
        "uv.lock": "",
        "analysis/tool.py": "print('ok')\n",
        "analysis/__pycache__/tool.pyc": "cache",
        "docs/release/README.md": "# release\n",
        "docs/conference_submission/conference_submission.tex": "paper",
        "docs/conference_submission/conference_submission.pdf": "pdf",
        "docs/conference_submission/references.bib": "",
        "docs/conference_submission/neurips_2026.sty": "",
        "data/raw/part_0/harmful.csv": "prompt,response\n",
        "data/raw/part_1/results.csv": "model,choice\n",
        "data/raw/part_2/results_meta.json": "{}",
        "data/analysis/tables/summary.csv": "model,rate\n",
        "data/graphs/plot.png": "png",
    }.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    output_path = root / "docs" / "conference_submission" / "supplement.zip"
    files = {path.as_posix() for path in collect_supplement_files(root, output_path)}

    assert "data/raw/part_1/results.csv" in files
    assert "data/raw/part_2/results_meta.json" in files
    assert "data/raw/part_0/harmful.csv" not in files
    assert "docs/conference_submission/conference_submission.pdf" not in files
    assert "analysis/__pycache__/tool.pyc" not in files

    build_supplement(root, output_path)

    with zipfile.ZipFile(output_path) as zf:
        names = set(zf.namelist())

    assert MANIFEST_NAME in names
    assert "data/raw/part_1/results.csv" in names
    assert "data/raw/part_0/harmful.csv" not in names
    assert "supplement.zip" not in names


def test_wilson_interval_handles_empty_and_nonempty_rates() -> None:
    assert _wilson_interval(0, 0) == (0.0, 0.0)
    low, high = _wilson_interval(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0


def test_cross_part_correlation_helpers_handle_rank_and_linear_relationships() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [2.0, 4.0, 6.0, 8.0]
    tied = [1.0, 1.0, 2.0, 3.0]

    assert round(_pearson_correlation(xs, ys), 6) == 1.0
    assert round(_spearman_correlation(xs, ys), 6) == 1.0
    assert round(_spearman_correlation(tied, tied), 6) == 1.0


def test_part2_auc_normalizes_over_configured_horizon() -> None:
    day_rows = {
        1: [{"resource_units_remaining": "8", "population_end": "4"}],
        2: [{"resource_units_remaining": "4", "population_end": "2"}],
    }

    reserve_auc, population_auc = _part2_normalized_auc(
        day_rows,
        horizon=4,
        society_size=4,
        resource_capacity=8,
    )

    assert reserve_auc == 0.375
    assert population_auc == 0.375


def test_part1_decomposition_preserves_total_variance_share() -> None:
    observations = [
        {
            "model": "a",
            "frame": "self_direct",
            "game": "prisoners_dilemma",
            "domain": "workplace",
            "presentation": "structured",
            "cooperate": 1,
        },
        {
            "model": "a",
            "frame": "prediction",
            "game": "prisoners_dilemma",
            "domain": "workplace",
            "presentation": "structured",
            "cooperate": 0,
        },
        {
            "model": "b",
            "frame": "self_direct",
            "game": "temptation_or_commons",
            "domain": "healthcare",
            "presentation": "narrative",
            "cooperate": 1,
        },
        {
            "model": "b",
            "frame": "prediction",
            "game": "temptation_or_commons",
            "domain": "healthcare",
            "presentation": "narrative",
            "cooperate": 0,
        },
    ]

    rows = _part1_factor_decomposition_rows(observations)

    assert rows[-1]["term"] == "total"
    assert rows[-1]["variance_share"] == 1.0
