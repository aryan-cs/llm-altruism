import csv
import hashlib
import json
import math
import zipfile
from pathlib import Path

import pytest

import analysis.summarize_results as summary_module

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
from analysis.validation import _csv_paths, validate_part1_file, validate_part2_file
from experiments.part1.part_1 import (
    PRE_ORDERING_RESULT_HEADERS as PART1_PRE_ORDERING_RESULT_HEADERS,
)
from experiments.part2.part_2 import PILOT_RESULT_HEADERS


def test_default_summary_build_does_not_emit_withdrawn_part0_or_cross_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[str] = []
    monkeypatch.setattr(
        summary_module,
        "summarize_part0",
        lambda *_args: called.append("part0") or (),
    )
    monkeypatch.setattr(
        summary_module,
        "summarize_part1",
        lambda *_args: called.append("part1") or (),
    )
    monkeypatch.setattr(
        summary_module,
        "summarize_part2",
        lambda *_args: called.append("part2") or (tmp_path / "part2.csv"),
    )
    monkeypatch.setattr(
        summary_module,
        "summarize_cross_part",
        lambda *_args: called.append("cross_part") or (),
    )

    summary_module.summarize_all(tmp_path / "raw", tmp_path / "tables")
    assert called == ["part1", "part2"]

    called.clear()
    summary_module.summarize_all(
        tmp_path / "raw",
        tmp_path / "tables-forensic",
        include_withdrawn_part0=True,
    )
    assert called == ["part0", "part1", "part2", "cross_part"]


def test_part0_rejudged_schema_prefers_response_only_label() -> None:
    assert _part0_compliance_value(
        {"complied?": "true", "new_complied": "false"}
    ) == "false"


def test_part0_rejudged_schema_never_falls_back_for_unjudged_row() -> None:
    assert _part0_compliance_value({"complied?": "true", "new_complied": ""}) == ""


def test_hash_bound_smoke_marker_excludes_sacrificial_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "smoke.csv"
    csv_path.write_text("provider,model\nprovider,model\n", encoding="utf-8")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    csv_path.with_name("smoke.analysis_exclude.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "canonical_analysis_and_validation",
                "reason": "sacrificial_campaign_smoke",
                "csv_sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    assert _csv_paths(tmp_path) == []

    csv_path.write_text("provider,model\nprovider,tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not hash-bound"):
        _csv_paths(tmp_path)


def test_part1_validation_accepts_pilot_schema_before_order_tracking(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "part1__ollama__model__full__20260424_000000.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PART1_PRE_ORDERING_RESULT_HEADERS)
        writer.writeheader()
        writer.writerow(
            {
                "provider": "ollama",
                "model": "model",
                "game": "prisoners_dilemma",
                "frame": "self_direct",
                "domain": "workplace",
                "scenario_variant": "workplace_pd_1",
                "presentation": "narrative",
                "prompt_id": "pilot-row",
                "action": "COOPERATE",
                "justification": "brief",
                "prompt_text": "stored prompt",
            }
        )

    report = validate_part1_file(csv_path)

    assert report.status == "warn"
    assert report.errors == []


def _write_part2_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_RESULT_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in PILOT_RESULT_HEADERS})
    first = rows[0]
    path.with_name(f"{path.stem}_meta.json").write_text(
        json.dumps(
            {
                "provider": first["provider"],
                "model": first["model"],
                "parameters": {
                    "society_config": {
                        "society_size": int(first["population_start"]),
                        "days": max(int(row["day"]) for row in rows),
                        "resource": first["resource"],
                        "selfish_gain": int(first["selfish_gain"]),
                        "depletion_units": int(first["depletion_units"]),
                        "community_benefit": int(first["community_benefit"]),
                    },
                    "resource_capacity": int(first["resource_capacity"]),
                    "collapse_death_rate": 0.2,
                },
            }
        ),
        encoding="utf-8",
    )


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
    assert entries[0]["csv_integrity"]["row_count"] == 1
    assert entries[0]["csv_integrity"]["columns"] == ["provider", "model"]
    assert len(entries[0]["csv_integrity"]["sha256"]) == 64
    assert entries[0]["csv_integrity"]["identifier_values"] == {
        "provider": ["openai"],
        "model": ["model"],
    }
    assert len(entries[0]["metadata_integrity"]["sha256"]) == 64
    assert entries[0]["run_contract"] == {"prompt_config_hash": "abc"}
    assert len(entries[0]["run_contract_sha256"]) == 64

    original_csv_hash = entries[0]["csv_integrity"]["sha256"]
    csv_path.write_text(
        "provider,model\nopenai,model\nopenai,model\n",
        encoding="utf-8",
    )
    changed = build_manifest(raw_dir)
    assert changed[0]["csv_integrity"]["row_count"] == 2
    assert changed[0]["csv_integrity"]["sha256"] != original_csv_hash


def test_supplement_builder_excludes_all_raw_and_legacy_generated_artifacts(tmp_path: Path) -> None:
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
        "docs/conference_submission/conference_submission.tex": (
            "\\author{Fixture Author}\npaper\n"
        ),
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

    assert "data/raw/part_0/harmful.csv" not in files
    assert "data/raw/part_1/results.csv" not in files
    assert "data/raw/part_2/results_meta.json" not in files
    assert "data/analysis/tables/summary.csv" not in files
    assert "data/graphs/plot.png" not in files
    assert "docs/conference_submission/conference_submission.pdf" not in files
    assert "analysis/__pycache__/tool.pyc" not in files

    build_supplement(root, output_path)

    with zipfile.ZipFile(output_path) as zf:
        names = set(zf.namelist())

    assert MANIFEST_NAME in names
    assert "data/raw/part_0/harmful.csv" not in names
    assert "data/raw/part_1/results.csv" not in names
    assert "data/raw/part_2/results_meta.json" not in names
    assert "data/analysis/tables/summary.csv" not in names
    assert "data/graphs/plot.png" not in names
    assert "supplement.zip" not in names


def test_supplement_builder_uses_reproducible_manifest_epoch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("readme\n", encoding="utf-8")

    first, _ = build_supplement(
        project_root=root, output_path=tmp_path / "first.zip"
    )
    second, _ = build_supplement(
        project_root=root, output_path=tmp_path / "second.zip"
    )

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read(MANIFEST_NAME))
    assert manifest["created_utc"] == "2026-01-01T00:00:00+00:00"


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


def test_part2_auc_does_not_impute_censored_tail_as_zero() -> None:
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

    assert math.isnan(reserve_auc)
    assert math.isnan(population_auc)


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
