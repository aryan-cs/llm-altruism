from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pytest

from analysis.build_croissant_metadata import (
    CORE_SPEC,
    CROISSANT_CONTEXT,
    RAI_SPEC,
    CroissantBuildError,
    _canonical_hash,
    build_metadata,
)


SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _final_fixture(tmp_path: Path) -> Path:
    directory = tmp_path / "data/analysis/final_results"
    p0_ids = [f"matched-{index:02d}" for index in range(24)]
    part0 = [
        {
            "target_id": target_id,
            "root_count_per_condition": 24,
            "human_validation_complete": False,
            "paper_eligible": False,
        }
        for target_id in p0_ids
    ]
    part1 = [
        {
            "target_id": target_id,
            "scope": "balanced_partial",
            "root_count": 96,
            "paper_eligible": False,
        }
        for target_id in [*p0_ids, *(f"expanded-{index:02d}" for index in range(51))]
    ]
    part1.extend(
        {
            "target_id": f"slow-{index:02d}",
            "scope": "balanced_partial",
            "root_count": 12,
            "paper_eligible": False,
        }
        for index in range(2)
    )
    part1.append(
        {
            "target_id": "glm-5.1-full",
            "scope": "full_384",
            "root_count": 384,
            "paper_eligible": True,
        }
    )
    part2 = [
        {
            "target_id": target_id,
            "trajectory_count": 8,
            "paper_eligible": False,
        }
        for target_id in p0_ids
    ]
    tables = {
        "part0_model_rates.csv": [
            {"target_id": target_id, "refusal_rate": 0.5} for target_id in p0_ids
        ],
        "part1_model_rates.csv": [
            {"target_id": row["target_id"], "scope": row["scope"], "cooperation_rate": 0.5}
            for row in part1
        ],
        "part2_model_metrics.csv": [
            {"target_id": target_id, "trajectory_count": 8, "restraint_rate_mean": 0.5}
            for target_id in p0_ids
        ],
    }
    for filename, rows in tables.items():
        _write_csv(directory / filename, rows)
    outputs = {
        "part0_csv": {
            "path": "part0_model_rates.csv",
            "file_sha256": hashlib.sha256((directory / "part0_model_rates.csv").read_bytes()).hexdigest(),
        },
        "part1_csv": {
            "path": "part1_model_rates.csv",
            "file_sha256": hashlib.sha256((directory / "part1_model_rates.csv").read_bytes()).hexdigest(),
        },
        "part2_csv": {
            "path": "part2_model_metrics.csv",
            "file_sha256": hashlib.sha256((directory / "part2_model_metrics.csv").read_bytes()).hexdigest(),
        },
        "cross_axis_csv": None,
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "prosocial_readiness_final_sanitized_results",
        "privacy_contract": {
            "contains_prompt_text": False,
            "contains_response_text": False,
            "contains_reasoning": False,
            "contains_raw_responses": False,
            "contains_routes": False,
        },
        "part0": part0,
        "part1": part1,
        "part2": part2,
        "cross_axis": {"status": "withheld_evidence_gates"},
        "outputs": outputs,
    }
    artifact["evidence_sha256"] = _canonical_hash(artifact)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "final_results.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return directory


def test_missing_final_results_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(CroissantBuildError, match="unavailable"):
        build_metadata(
            repository_root=tmp_path,
            final_results_dir=tmp_path / "missing",
            output_path=tmp_path / "croissant.json",
        )


def test_metadata_uses_restored_title_design_and_evidence_status(tmp_path: Path) -> None:
    final_dir = _final_fixture(tmp_path)
    metadata = build_metadata(
        repository_root=tmp_path,
        final_results_dir=final_dir,
        output_path=tmp_path / "croissant.json",
    )

    assert metadata["@context"] == CROISSANT_CONTEXT
    assert metadata["@type"] == "sc:Dataset"
    assert metadata["conformsTo"] == [CORE_SPEC, RAI_SPEC]
    assert metadata["name"] == "Prosocial Readiness Bench"
    assert "Safety Beyond Refusal" in metadata["citeAs"]
    assert "negative benchmark" not in json.dumps(metadata).lower()
    assert "75 Part 1 routes at 96 roots, two at 12 roots" in metadata["prov:wasDerivedFrom"][0]["description"]
    assert "78 observed of 81" in metadata["prov:wasDerivedFrom"][0]["description"]
    assert "eight independent" in metadata["rai:dataCollection"]
    assert "exploratory" in metadata["description"]
    assert "private manifests" in metadata["conditionsOfAccess"]
    assert SEMVER.fullmatch(metadata["version"])
    assert SEMVER.fullmatch(metadata["cr:sdVersion"])
    assert date.fromisoformat(metadata["dateCreated"]) <= date.fromisoformat(metadata["datePublished"]) <= date.fromisoformat(metadata["dateModified"])


def test_distributions_are_only_hash_bound_final_artifacts(tmp_path: Path) -> None:
    final_dir = _final_fixture(tmp_path)
    output = tmp_path / "metadata/croissant.json"
    metadata = build_metadata(
        repository_root=tmp_path, final_results_dir=final_dir, output_path=output
    )
    distributions = metadata["distribution"]

    assert {item["name"] for item in distributions} == {
        "final_results.json",
        "part0_model_rates.csv",
        "part1_model_rates.csv",
        "part2_model_metrics.csv",
    }
    assert not any("raw" in item["name"] or "private" in item["contentUrl"] for item in distributions)
    for item in distributions:
        resolved = (output.parent / item["contentUrl"]).resolve()
        assert resolved.is_file()
        assert item["sha256"] == hashlib.sha256(resolved.read_bytes()).hexdigest()


def test_every_final_csv_has_a_complete_record_set(tmp_path: Path) -> None:
    final_dir = _final_fixture(tmp_path)
    output = tmp_path / "croissant.json"
    metadata = build_metadata(
        repository_root=tmp_path, final_results_dir=final_dir, output_path=output
    )
    distributions = {item["@id"]: item for item in metadata["distribution"]}
    record_sets = metadata["recordSet"]

    assert len(record_sets) == 3
    for record_set in record_sets:
        source_id = record_set["field"][0]["source"]["fileObject"]["@id"]
        source = next(item for item in metadata["distribution"] if item["@id"] == source_id)
        with (output.parent / source["contentUrl"]).resolve().open(newline="", encoding="utf-8") as handle:
            columns = next(csv.reader(handle))
        assert [field["name"] for field in record_set["field"]] == columns
        assert distributions[source_id]["encodingFormat"] == "text/csv"


def test_hash_privacy_and_executed_coverage_changes_fail_closed(tmp_path: Path) -> None:
    final_dir = _final_fixture(tmp_path)
    artifact_path = final_dir / "final_results.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["part0"][0]["prompt_text"] = "must never be released"
    artifact["evidence_sha256"] = _canonical_hash(artifact)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(CroissantBuildError, match="forbidden private field"):
        build_metadata(final_results_dir=final_dir, output_path=tmp_path / "out.json")

    final_dir = _final_fixture(tmp_path / "coverage")
    artifact_path = final_dir / "final_results.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["part2"][0]["trajectory_count"] = 12
    artifact["evidence_sha256"] = _canonical_hash(artifact)
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(CroissantBuildError, match=r"24-matched/75x96\+2x12\+1x384"):
        build_metadata(final_results_dir=final_dir, output_path=tmp_path / "out.json")

    final_dir = _final_fixture(tmp_path / "hash")
    (final_dir / "part1_model_rates.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CroissantBuildError, match="hash changed"):
        build_metadata(final_results_dir=final_dir, output_path=tmp_path / "out.json")


def test_dataset_url_rejects_placeholders_and_accepts_review_host(tmp_path: Path) -> None:
    final_dir = _final_fixture(tmp_path)
    with pytest.raises(ValueError, match="public HTTPS"):
        build_metadata(final_results_dir=final_dir, dataset_url="./")
    with pytest.raises(ValueError, match="public HTTPS"):
        build_metadata(final_results_dir=final_dir, dataset_url="https://example.com/dataset")
    metadata = build_metadata(
        final_results_dir=final_dir,
        output_path=tmp_path / "metadata.json",
        dataset_url="https://openreview.net/",
    )
    assert metadata["url"] == "https://openreview.net/"


def test_mlcroissant_loads_final_record_sets_and_rows(tmp_path: Path) -> None:
    mlcroissant = pytest.importorskip("mlcroissant")
    final_dir = _final_fixture(tmp_path)
    output = tmp_path / "croissant.json"
    metadata = build_metadata(final_results_dir=final_dir, output_path=output)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    dataset = mlcroissant.Dataset(str(output))
    assert [record_set.uuid for record_set in dataset.metadata.record_sets] == [
        "part0-model-rates-records",
        "part1-model-rates-records",
        "part2-model-metrics-records",
    ]
    for record_set_id in [record_set.uuid for record_set in dataset.metadata.record_sets]:
        assert next(iter(dataset.records(record_set=record_set_id)))
