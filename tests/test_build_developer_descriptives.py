import json
from pathlib import Path
from typing import Any

import pytest

from analysis.build_developer_descriptives import (
    OUTPUT_ARTIFACT_TYPE,
    DeveloperDescriptiveError,
    _self_hash,
    build_developer_descriptives,
    main,
)
from analysis.build_final_results import build_final_results
from analysis.build_paper_headlines import _seal
from test_build_final_results import (
    SUBJECT,
    _panel,
    _part0_fixture,
    _part1_fixture,
    _part2_fixture,
)
from test_build_paper_headlines import _source


def _write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build(tmp_path: Path, source: dict[str, Any] | None = None):
    input_path = tmp_path / "final_results.json"
    output_path = tmp_path / "developer_descriptives.json"
    _write(input_path, source or _source())
    return build_developer_descriptives(input_path, output_path), output_path


def test_emits_only_alphabetical_multi_system_groups_and_separate_scopes(
    tmp_path: Path,
) -> None:
    source = _source()
    # Exercise alphabetical ordering without ever sorting on outcomes.
    source.pop("evidence_sha256")
    for row in source["part0"]:
        row["upstream_provider"] = "Zulu Developer"
    for row in source["part2"]:
        row["upstream_provider"] = "Alpha Developer"
    source["part1"][0]["upstream_provider"] = "Beta Developer"
    source["part1"][1]["upstream_provider"] = "Beta Developer"
    source["part1"][2]["upstream_provider"] = "Singleton"
    source["part1"][3]["upstream_provider"] = "Singleton Two"
    source["part1"][4]["upstream_provider"] = "Singleton Full"
    _seal(source)

    artifact, output = _build(tmp_path, source)
    assert artifact["artifact_type"] == OUTPUT_ARTIFACT_TYPE
    assert artifact["evidence_sha256"] == _self_hash(artifact)
    assert artifact["source"]["path"] == "final_results.json"
    assert len(artifact["source"]["file_sha256"]) == 64
    assert artifact["source"]["evidence_sha256"] == source["evidence_sha256"]
    assert artifact["reporting_contract"]["within_axis_only"] is True
    assert artifact["reporting_contract"]["part1_scopes_pooled"] is False
    assert artifact["reporting_contract"]["rankings_computed"] is False

    assert artifact["part0"]["groups_alphabetical"] == [{
        "developer_route_group": "Zulu Developer",
        "system_count": 2,
        "overall_refusal": {"median": 0.666667, "minimum": 0.5, "maximum": 0.833333},
    }]
    scopes = artifact["part1"]["scopes_alphabetical"]
    assert [scope["root_count"] for scope in scopes] == [12, 96, 384]
    assert scopes[0]["groups_alphabetical"] == []
    assert scopes[1]["groups_alphabetical"] == [{
        "developer_route_group": "Beta Developer",
        "system_count": 2,
        "self_choice": {"median": 0.5, "minimum": 0.25, "maximum": 0.75},
    }]
    assert scopes[2]["groups_alphabetical"] == []
    assert artifact["part2"]["groups_alphabetical"] == [{
        "developer_route_group": "Alpha Developer",
        "system_count": 2,
        "aurc": {"median": 0.6, "minimum": 0.5, "maximum": 0.7},
        "restraint_rate": {"median": 0.375, "minimum": 0.25, "maximum": 0.5},
        "aupc": {"median": 0.825, "minimum": 0.75, "maximum": 0.9},
    }]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.__setitem__("schema_version", 2), "schema/type/self-hash"),
        (
            lambda value: value["privacy_contract"].__setitem__("contains_routes", True),
            "privacy contract",
        ),
        (lambda value: value["part0"][0].__setitem__("route", "secret/route"), "Forbidden"),
        (
            lambda value: value["part1"][0].__setitem__("scope", "full_384"),
            "scope/root_count",
        ),
        (
            lambda value: value["part2"][0]["trajectory_level_95_percent_t_intervals"][
                "aurc"
            ].__setitem__("mean", "nan"),
            "finite",
        ),
    ],
)
def test_rejects_unsafe_or_inconsistent_source(
    tmp_path: Path, mutation: Any, match: str,
) -> None:
    value = _source()
    value.pop("evidence_sha256")
    mutation(value)
    _seal(value)
    input_path = tmp_path / "final_results.json"
    _write(input_path, value)
    with pytest.raises(DeveloperDescriptiveError, match=match):
        build_developer_descriptives(input_path, tmp_path / "out.json")


def test_rejects_tampered_hash_and_overwrite(tmp_path: Path) -> None:
    value = _source()
    value["part0"][0]["model"] = "tampered"
    input_path = tmp_path / "final_results.json"
    _write(input_path, value)
    with pytest.raises(DeveloperDescriptiveError, match="self-hash"):
        build_developer_descriptives(input_path, tmp_path / "out.json")

    _write(input_path, _source())
    output_path = tmp_path / "out.json"
    output_path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(DeveloperDescriptiveError, match="overwrite"):
        build_developer_descriptives(input_path, output_path)


def test_end_to_end_from_final_results_builder_and_cli(tmp_path: Path) -> None:
    second = {
        "target_id": "subject.same-developer",
        "upstream_provider": SUBJECT["upstream_provider"],
        "model": "second-model",
        "route": "region/second-model",
    }
    subjects = (SUBJECT, second)
    final_directory = tmp_path / "final"
    build_final_results(
        part0_manifest=_part0_fixture(tmp_path, subjects=subjects),
        part1_full_manifests=[
            _part1_fixture(tmp_path, count=384, name="part1-full", subjects=subjects)
        ],
        part1_n96_manifests=[
            _part1_fixture(tmp_path, count=96, name="part1-n96", subjects=subjects),
            _part1_fixture(tmp_path, count=12, name="part1-n12", subjects=subjects),
        ],
        part2_manifest=_part2_fixture(tmp_path, subjects=subjects),
        panel_path=_panel(tmp_path),
        output_dir=final_directory,
    )

    output = tmp_path / "selected/developer.json"
    assert main([
        "--input", str(final_directory / "final_results.json"),
        "--output", str(output),
    ]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["part0"]["groups_alphabetical"][0]["system_count"] == 2
    assert all(
        scope["groups_alphabetical"][0]["system_count"] == 2
        for scope in artifact["part1"]["scopes_alphabetical"]
    )
    assert artifact["part2"]["groups_alphabetical"][0]["system_count"] == 2
    serialized = output.read_text(encoding="utf-8")
    for forbidden in ("region/", '"route"', '"prompt_text"', '"raw_response"'):
        assert forbidden not in serialized
