import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from analysis import confirmatory_estimators
from analysis import confirmatory_data_lock
from experiments.misc.run_metadata import sha256_file, stable_json_hash
from experiments.part0 import confirmatory_runner as part0_runner


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_SHA = "c" * 64


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    argument_strings = [str(argument) for argument in arguments]
    input_path = argument_strings[argument_strings.index("--input") + 1]
    wrapper = (
        "import sys; from pathlib import Path; "
        "from analysis import confirmatory_estimators as module; "
        "module.materialize_native_units=lambda *args,**kwargs:"
        "module._load_json_object(Path(sys.argv[1]))[0]; "
        "raise SystemExit(module.main(sys.argv[2:]))"
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            wrapper,
            input_path,
            *argument_strings,
            "--data-lock",
            "/private/confirmatory-data-lock.json",
            "--data-lock-sha256",
            "d" * 64,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_private(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    path.chmod(0o600)
    return raw


def _verify_artifact(path: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    recorded = artifact.pop("artifact_sha256")
    canonical = (
        json.dumps(
            artifact,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    assert recorded == hashlib.sha256(canonical).hexdigest()
    artifact["artifact_sha256"] = recorded
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return artifact


def _seal_input(value: dict[str, object]) -> dict[str, object]:
    value.pop("artifact_sha256", None)
    canonical = (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    value["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    return value


def _part0_input() -> dict[str, object]:
    systems = ["system-a", "system-b"]
    roots = [
        {
            "prompt_root_id": "harmful-1",
            "semantic_cluster_id": "harmful-cluster-1",
            "arm": "harmful",
        },
        {
            "prompt_root_id": "harmful-2",
            "semantic_cluster_id": "harmful-cluster-2",
            "arm": "harmful",
        },
        {
            "prompt_root_id": "control-1",
            "semantic_cluster_id": "control-cluster-1",
            "arm": "control",
        },
        {
            "prompt_root_id": "control-2",
            "semantic_cluster_id": "control-cluster-2",
            "arm": "control",
        },
    ]
    rows: list[dict[str, object]] = []
    for system in systems:
        for root in roots:
            for language in ("english", "chinese", "russian"):
                for block in (1,):
                    if root["prompt_root_id"] == "harmful-1":
                        outcome = "REFUSAL"
                    elif root["prompt_root_id"] == "harmful-2":
                        outcome = "INVALID" if system == "system-a" else "COMPLIANCE"
                    elif root["prompt_root_id"] == "control-1":
                        outcome = "COMPLIANCE"
                    else:
                        outcome = "REFUSAL"
                    rows.append(
                        {
                            "system_id": system,
                            **root,
                            "language": language,
                            "generation_block": block,
                            "outcome": outcome,
                        }
                    )
    return _seal_input({
        "schema_version": 1,
        "artifact_type": "part0_confirmatory_units",
        "campaign_manifest_sha256": CAMPAIGN_SHA,
        "source_data_lock_path": "/private/confirmatory-data-lock.json",
        "source_data_lock_sha256": "d" * 64,
        "frozen_system_ids": systems,
        "system_metadata": [
            {
                "system_id": system,
                "cohort_id": "current_sota",
                "developer_id": f"developer-{index}",
            }
            for index, system in enumerate(systems)
        ],
        "languages": ["english", "chinese", "russian"],
        "generation_blocks": [1],
        "prompt_roots": roots,
        "rows": rows,
    })


def _result(
    artifact: dict[str, object], estimand: str, arm: str, language: str
) -> dict[str, object]:
    return next(
        row
        for row in artifact["results"]
        if row["estimand"] == estimand
        and row["system_id"] == "finite_panel_equal_system_weight"
        and row["arm"] == arm
        and row["language"] == language
    )


def test_part0_cli_two_way_clusters_and_retains_invalids_exactly(tmp_path: Path) -> None:
    input_path = tmp_path / "part0-private.json"
    raw = _write_private(input_path, _part0_input())
    output = tmp_path / "part0-estimates.json"
    completed = _run("part0", "--input", input_path, "--output", output)
    assert completed.returncode == 0, completed.stderr
    artifact = _verify_artifact(output)

    assert artifact["private_input_sha256"] == hashlib.sha256(raw).hexdigest()
    assert artifact["campaign_manifest_sha256"] == CAMPAIGN_SHA
    assert artifact["source_artifact_sha256"] == _part0_input()["artifact_sha256"]
    assert artifact["bootstrap_replicates"] == 2000
    assert artifact["bootstrap_method"] == (
        "semantic_prompt_cluster_bootstrap_stratified_by_arm"
    )
    assert artifact["system_resampling"] == "none_primary_finite_panel"
    assert artifact["retained_invalid_count"] == 3
    harmful = _result(artifact, "primary_rate", "harmful", "english")
    invalid = _result(artifact, "invalid_rate", "harmful", "english")
    contrast = _result(
        artifact,
        "harmful_minus_control_refusal_rate",
        "harmful_minus_control",
        "english",
    )
    assert harmful["estimate"] == 0.5
    assert invalid["estimate"] == 0.25
    assert contrast["estimate"] == 0.0
    assert _result(
        artifact,
        "language_minus_english_primary_rate",
        "harmful",
        "chinese",
    )["estimate"] == 0.0

    second_output = tmp_path / "part0-estimates-2.json"
    assert _run("part0", "--input", input_path, "--output", second_output).returncode == 0
    second = _verify_artifact(second_output)
    assert second["artifact_sha256"] == artifact["artifact_sha256"]

    refused = _run("part0", "--input", input_path, "--output", output)
    assert refused.returncode == 2
    assert "refusing to overwrite" in refused.stderr
    assert "Traceback" not in refused.stderr


def test_part0_cli_fails_on_incomplete_duplicate_or_row_iid_request(tmp_path: Path) -> None:
    incomplete = _part0_input()
    incomplete["rows"] = incomplete["rows"][:-1]
    _seal_input(incomplete)
    path = tmp_path / "incomplete.json"
    _write_private(path, incomplete)
    rejected = _run("part0", "--input", path, "--output", tmp_path / "out.json")
    assert rejected.returncode == 2
    assert "exactly 24 rows" in rejected.stderr
    assert "Traceback" not in rejected.stderr

    duplicate = _part0_input()
    duplicate["rows"][-1] = dict(duplicate["rows"][0])
    _seal_input(duplicate)
    path = tmp_path / "duplicate.json"
    _write_private(path, duplicate)
    rejected = _run("part0", "--input", path, "--output", tmp_path / "dup-out.json")
    assert rejected.returncode == 2
    assert "duplicate system/root/language/block" in rejected.stderr

    valid_path = tmp_path / "valid.json"
    _write_private(valid_path, _part0_input())
    rejected = _run(
        "part0",
        "--input",
        valid_path,
        "--replicates",
        1999,
        "--output",
        tmp_path / "row-iid.json",
    )
    assert rejected.returncode == 2
    assert "at least 2000" in rejected.stderr
    assert not (tmp_path / "row-iid.json").exists()


def test_part0_estimator_uses_exact_native_runner_block_contract() -> None:
    native_fixture = _part0_input()
    assert confirmatory_estimators.PART0_BLOCKS == part0_runner.EXPECTED_BLOCKS
    assert native_fixture["generation_blocks"] == list(part0_runner.EXPECTED_BLOCKS)
    assert {row["generation_block"] for row in native_fixture["rows"]} == set(
        part0_runner.EXPECTED_BLOCKS
    )


def test_part0_bootstrap_resamples_roots_but_not_frozen_systems(tmp_path: Path) -> None:
    root_varying = _part0_input()
    varying_path = tmp_path / "root-varying.json"
    _write_private(varying_path, root_varying)
    varying_output = tmp_path / "root-varying-output.json"
    assert _run("part0", "--input", varying_path, "--output", varying_output).returncode == 0
    varying = _verify_artifact(varying_output)
    interval = _result(varying, "primary_rate", "harmful", "english")
    assert interval["ci_low"] < interval["estimate"] < interval["ci_high"]

    original = _part0_input()
    original_path = tmp_path / "original.json"
    _write_private(original_path, original)
    original_output = tmp_path / "original-output.json"
    assert _run("part0", "--input", original_path, "--output", original_output).returncode == 0
    original_artifact = _verify_artifact(original_output)

    duplicated = json.loads(json.dumps(original))
    duplicated.pop("artifact_sha256")
    duplicated["frozen_system_ids"].extend(("system-a-copy", "system-b-copy"))
    duplicated["system_metadata"].extend(
        (
            {
                "system_id": "system-a-copy",
                "cohort_id": "current_sota",
                "developer_id": "developer-0",
            },
            {
                "system_id": "system-b-copy",
                "cohort_id": "current_sota",
                "developer_id": "developer-1",
            },
        )
    )
    copied_rows: list[dict[str, object]] = []
    for row in duplicated["rows"]:
        if row["system_id"] in {"system-a", "system-b"}:
            copy = dict(row)
            copy["system_id"] = f"{row['system_id']}-copy"
            copied_rows.append(copy)
    duplicated["rows"].extend(copied_rows)
    _seal_input(duplicated)
    duplicated_path = tmp_path / "duplicated-systems.json"
    _write_private(duplicated_path, duplicated)
    duplicated_output = tmp_path / "duplicated-systems-output.json"
    assert _run(
        "part0", "--input", duplicated_path, "--output", duplicated_output
    ).returncode == 0
    duplicated_artifact = _verify_artifact(duplicated_output)
    for language in ("english", "chinese", "russian", "all_languages"):
        before = _result(original_artifact, "primary_rate", "harmful", language)
        after = _result(duplicated_artifact, "primary_rate", "harmful", language)
        assert (before["estimate"], before["ci_low"], before["ci_high"]) == (
            after["estimate"],
            after["ci_low"],
            after["ci_high"],
        )


def _part1_input() -> dict[str, object]:
    systems = ["system-a", "system-b"]
    games = ("prisoners_dilemma", "temptation_or_commons")
    domains = (
        "shared_workspaces",
        "scientific_facilities",
        "civic_infrastructure",
        "education_resources",
        "healthcare_operations",
        "digital_services",
    )
    roots: list[dict[str, str]] = []
    for game in games:
        for domain in domains:
            for index in range(32):
                root_id = f"{game}-{domain}-{index:02d}"
                roots.append({"root_id": root_id, "game": game, "domain": domain})
    rows: list[dict[str, object]] = []
    for system in systems:
        for root in roots:
            root_index = int(root["root_id"].rsplit("-", 1)[1])
            primary_outcome = (
                "COOPERATE" if system == "system-b" or root_index < 16 else "INVALID"
            )
            rows.append(
                {
                    "system_id": system,
                    **root,
                    "phase": "primary",
                    "frame": "self_direct",
                    "execution_block": 0,
                    "outcome": primary_outcome,
                }
            )
    return _seal_input({
        "schema_version": 1,
        "artifact_type": "part1_confirmatory_units",
        "campaign_manifest_sha256": CAMPAIGN_SHA,
        "source_data_lock_path": "/private/confirmatory-data-lock.json",
        "source_data_lock_sha256": "d" * 64,
        "frozen_system_ids": systems,
        "system_metadata": [
            {
                "system_id": system,
                "cohort_id": "current_sota",
                "developer_id": f"developer-{index}",
            }
            for index, system in enumerate(systems)
        ],
        "primary_root_design": roots,
        "secondary_root_ids": [],
        "rows": rows,
    })


def _part1_result(
    artifact: dict[str, object], estimand: str, system: str, domain: str, frame: str
) -> dict[str, object]:
    return next(
        row
        for row in artifact["results"]
        if row["estimand"] == estimand
        and row["system_id"] == system
        and row["domain"] == domain
        and row["frame"] == frame
    )


def test_part1_cli_root_by_block_within_system_domain_frame_effects(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "part1-private.json"
    _write_private(input_path, _part1_input())
    output = tmp_path / "part1-estimates.json"
    completed = _run("part1", "--input", input_path, "--output", output)
    assert completed.returncode == 0, completed.stderr
    artifact = _verify_artifact(output)
    assert artifact["primary_bootstrap_method"] == (
        "root_cluster_bootstrap_stratified_by_domain"
    )
    assert artifact["secondary_bootstrap_method"] is None
    assert artifact["primary_calls_per_system"] == 384
    assert artifact["secondary_calls_per_system"] == 0
    assert artifact["total_calls_per_system"] == 384
    assert artifact["retained_row_count"] == 768
    assert artifact["retained_invalid_count"] == 192
    assert _part1_result(
        artifact,
        "primary_domain_rate",
        "system-a",
        "shared_workspaces",
        "self_direct",
    )["estimate"] == 0.5
    assert _part1_result(
        artifact,
        "primary_domain_invalid_rate",
        "system-a",
        "shared_workspaces",
        "self_direct",
    )["estimate"] == 0.5


def test_part1_cli_rejects_incomplete_and_duplicate_root_block_units(tmp_path: Path) -> None:
    incomplete = _part1_input()
    incomplete["rows"] = incomplete["rows"][:-1]
    _seal_input(incomplete)
    path = tmp_path / "incomplete.json"
    _write_private(path, incomplete)
    rejected = _run("part1", "--input", path, "--output", tmp_path / "out.json")
    assert rejected.returncode == 2
    assert "exactly 768 rows" in rejected.stderr

    duplicate = _part1_input()
    duplicate["rows"][-1] = dict(duplicate["rows"][0])
    _seal_input(duplicate)
    path = tmp_path / "duplicate.json"
    _write_private(path, duplicate)
    rejected = _run("part1", "--input", path, "--output", tmp_path / "dup.json")
    assert rejected.returncode == 2
    assert "duplicate system/root/phase/frame/block" in rejected.stderr
    assert "Traceback" not in rejected.stderr


def _part2_input() -> dict[str, object]:
    systems = ["system-a", "system-b"]
    materialized: list[dict[str, object]] = []
    for system_index, system_id in enumerate(systems):
        units = []
        for run_index in range(20):
            aurc = (
                run_index / 19
                if system_index == 0
                else 0.25 + 0.5 * run_index / 19
            )
            survived = float(run_index >= 10)
            units.append(
                {
                    "unit_id": f"trajectory-{system_index}-{run_index:02d}",
                    "analysis_source": "final_baseline",
                    "structural_cell_id": "baseline-cell",
                    "horizon_days": 30,
                    "restraint_rate": 0.2 + 0.01 * run_index,
                    "normalized_aurc": aurc,
                    "normalized_aupc": min(1.0, aurc + 0.1),
                    "restricted_time_to_depletion": (
                        30.0 if survived else 10.0 + run_index
                    ),
                    "survived_through_horizon": survived,
                }
            )
        materialized.append(
            {
                "system_id": system_id,
                "cohort_id": "current_sota",
                "developer_id": f"developer-{system_index}",
                "part2_units": units,
            }
        )
    return _seal_input(
        {
            "schema_version": 1,
            "artifact_type": "part2_confirmatory_units",
            "campaign_manifest_sha256": CAMPAIGN_SHA,
            "source_data_lock_path": "/private/confirmatory-data-lock.json",
            "source_data_lock_sha256": "d" * 64,
            "frozen_system_ids": systems,
            "system_metadata": [
                {
                    "system_id": system_id,
                    "cohort_id": "current_sota",
                    "developer_id": f"developer-{index}",
                }
                for index, system_id in enumerate(systems)
            ],
            "selected_run_count": 20,
            "structural_cell_id": "baseline-cell",
            "horizon_days": 30,
            "systems": materialized,
        }
    )


def test_part2_cli_reports_primary_and_secondary_run_level_intervals(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "part2-private.json"
    _write_private(input_path, _part2_input())
    output = tmp_path / "part2-estimates.json"
    completed = _run("part2", "--input", input_path, "--output", output)
    assert completed.returncode == 0, completed.stderr
    artifact = _verify_artifact(output)
    assert artifact["analysis_source"] == "locked_part2_baseline_production_only"
    assert artifact["primary_metric"] == "normalized_aurc"
    assert artifact["selected_run_count_per_system"] == 20
    assert artifact["structural_cell_id"] == "baseline-cell"
    assert len(artifact["results"]) == 10
    primary = next(
        row
        for row in artifact["results"]
        if row["system_id"] == "system-a" and row["metric"] == "normalized_aurc"
    )
    assert primary["estimand_role"] == "primary"
    assert primary["estimate"] == pytest.approx(0.5)
    assert primary["t_ci_low"] < primary["estimate"] < primary["t_ci_high"]
    assert primary["bca_ci_low"] < primary["estimate"] < primary["bca_ci_high"]
    assert primary["interval_unit"] == "independent_final_baseline_trajectory"
    assert {
        row["metric"] for row in artifact["results"] if row["system_id"] == "system-a"
    } == {
        "normalized_aurc",
        "restraint_rate",
        "normalized_aupc",
        "restricted_mean_time_to_depletion",
        "survival_through_horizon",
    }
    assert {
        row["weighting"] for row in artifact["panel_results"]
    } == {"equal_system", "equal_developer"}


def test_part2_fixed_production_uses_fixed_trajectory_interval_label(
    tmp_path: Path,
) -> None:
    payload = _part2_input()
    payload["selected_run_count"] = 24
    payload["structural_cell_id"] = "fixed-cell"
    for system in payload["systems"]:
        original = system["part2_units"]
        for index in range(20, 24):
            original.append(
                {
                    **original[index % 20],
                    "unit_id": f"fixed-{system['system_id']}-{index}",
                }
            )
        for unit in original:
            unit["analysis_source"] = "fixed_production"
            unit["structural_cell_id"] = "fixed-cell"
    _seal_input(payload)
    input_path = tmp_path / "part2-fixed-private.json"
    _write_private(input_path, payload)
    output = tmp_path / "part2-fixed-estimates.json"

    completed = _run("part2", "--input", input_path, "--output", output)

    assert completed.returncode == 0, completed.stderr
    artifact = _verify_artifact(output)
    assert artifact["analysis_source"] == "locked_part2_fixed_production_only"
    assert {row["interval_unit"] for row in artifact["results"]} == {
        "independent_fixed_production_trajectory"
    }


def test_part2_cli_rejects_cross_system_duplicate_trajectory_and_wrong_n(
    tmp_path: Path,
) -> None:
    duplicate = _part2_input()
    duplicate["systems"][1]["part2_units"][0]["unit_id"] = duplicate["systems"][0][
        "part2_units"
    ][0]["unit_id"]
    _seal_input(duplicate)
    duplicate_path = tmp_path / "part2-duplicate.json"
    _write_private(duplicate_path, duplicate)
    rejected = _run(
        "part2", "--input", duplicate_path, "--output", tmp_path / "duplicate-out.json"
    )
    assert rejected.returncode == 2
    assert "duplicate trajectory_id across systems" in rejected.stderr

    wrong_n = _part2_input()
    wrong_n["systems"][0]["part2_units"] = wrong_n["systems"][0]["part2_units"][:-1]
    _seal_input(wrong_n)
    wrong_n_path = tmp_path / "part2-wrong-n.json"
    _write_private(wrong_n_path, wrong_n)
    rejected = _run(
        "part2", "--input", wrong_n_path, "--output", tmp_path / "wrong-n-out.json"
    )
    assert rejected.returncode == 2
    assert "exactly 20 selected trajectories" in rejected.stderr


def test_native_part2_materializer_selects_only_baseline_stage_and_rejects_duplicates(
    tmp_path: Path, monkeypatch
) -> None:
    systems = ["system-a", "system-b"]
    baseline_jobs = [
        {
            "id": f"baseline-{system_id}-{index:02d}",
            "target_id": system_id,
            "experiment": "part2",
            "stage": "part2_baseline_production",
        }
        for system_id in systems
        for index in range(20)
    ]
    contaminating_jobs = [
        {
            "id": f"pilot-{system_id}",
            "target_id": system_id,
            "experiment": "part2",
            "stage": "part2_variance_pilot",
        }
        for system_id in systems
    ] + [
        {
            "id": f"raw-{system_id}",
            "target_id": system_id,
            "experiment": "part2",
            "stage": "production",
        }
        for system_id in systems
    ]
    baseline = {
        "targets": [{"id": system_id} for system_id in systems],
        "part2_design": {"variance_selected_n": 20},
        "jobs": baseline_jobs + contaminating_jobs,
    }
    lock = {"campaigns": {"baseline_stage": {"file_sha256": "b" * 64}}}
    metadata = [
        {
            "system_id": system_id,
            "cohort_id": "current_sota",
            "developer_id": f"developer-{index}",
        }
        for index, system_id in enumerate(systems)
    ]
    called: list[str] = []

    def native(job):
        called.append(job["id"])
        return {
            "unit_id": f"trajectory-{job['id']}",
            "analysis_source": "final_baseline",
            "structural_cell_id": "baseline-cell",
            "horizon_days": 30,
            "restraint_rate": 0.5,
            "normalized_aurc": 0.5,
            "normalized_aupc": 0.5,
            "restricted_time_to_depletion": 30.0,
            "survived_through_horizon": 1.0,
        }

    monkeypatch.setattr(confirmatory_estimators, "_native_part2_unit", native)
    artifact = confirmatory_estimators._materialize_part2_from_context(
        lock,
        baseline,
        metadata,
        data_lock_path=tmp_path / "lock.json",
        data_lock_sha256="d" * 64,
    )
    assert set(called) == {job["id"] for job in baseline_jobs}
    assert not set(called) & {job["id"] for job in contaminating_jobs}
    assert artifact["selected_run_count"] == 20
    assert sum(len(system["part2_units"]) for system in artifact["systems"]) == 40

    monkeypatch.setattr(
        confirmatory_estimators,
        "_native_part2_unit",
        lambda job: {**native(job), "unit_id": "duplicate-trajectory"},
    )
    with pytest.raises(ValueError, match="duplicate trajectory_id"):
        confirmatory_estimators._materialize_part2_from_context(
            lock,
            baseline,
            metadata,
            data_lock_path=tmp_path / "lock.json",
            data_lock_sha256="d" * 64,
        )


def _cross_input(
    *, current_count: int = 12, historical_count: int = 0
) -> dict[str, object]:
    total = current_count + historical_count
    frozen = [f"system-{index:02d}" for index in range(total)]
    systems: list[dict[str, object]] = []
    part1_order = list(range(total))
    developer_modulus = 8 if current_count >= 24 else 4
    for index, system_id in enumerate(frozen):
        p0 = 5 + index * 3
        p1 = 5 + part1_order[index] * 3
        if index < current_count:
            p2 = (current_count - index) / (current_count + 1)
            cohort_id = "current_sota"
            developer_id = f"developer-{index % developer_modulus}"
        else:
            historical_index = index - current_count
            p2 = (historical_index + 1) / (historical_count + 1)
            cohort_id = "historical"
            developer_id = f"historical-developer-{historical_index % 3}"
        systems.append(
            {
                "system_id": system_id,
                "cohort_id": cohort_id,
                "developer_id": developer_id,
                "part0_units": [
                    {
                        "unit_id": f"p0-{index}-a",
                        "arm": "harmful",
                        "language_scope": "all_languages",
                        "refusal_count": p0,
                        "invalid_count": 5,
                        "total_count": 100,
                    },
                    {
                        "unit_id": f"p0-{index}-b",
                        "arm": "harmful",
                        "language_scope": "all_languages",
                        "refusal_count": p0 + 2,
                        "invalid_count": 4,
                        "total_count": 100,
                    },
                ],
                "part1_units": [
                    {
                        "unit_id": f"p1-{index}-a",
                        "phase": "primary",
                        "frame": "self_direct",
                        "cooperation_count": p1,
                        "invalid_count": 3,
                        "total_count": 100,
                    },
                    {
                        "unit_id": f"p1-{index}-b",
                        "phase": "primary",
                        "frame": "self_direct",
                        "cooperation_count": p1 + 1,
                        "invalid_count": 2,
                        "total_count": 100,
                    },
                ],
                "part2_units": [
                    {
                        "unit_id": f"p2-{index}-a",
                        "analysis_source": "final_baseline",
                        "structural_cell_id": "baseline-cell",
                        "horizon_days": 30,
                        "restraint_rate": p2,
                        "normalized_aurc": p2,
                        "normalized_aupc": p2,
                        "restricted_time_to_depletion": 15.0,
                        "survived_through_horizon": 0.0,
                    },
                    {
                        "unit_id": f"p2-{index}-b",
                        "analysis_source": "final_baseline",
                        "structural_cell_id": "baseline-cell",
                        "horizon_days": 30,
                        "restraint_rate": min(1.0, p2 + 0.01),
                        "normalized_aurc": min(1.0, p2 + 0.01),
                        "normalized_aupc": min(1.0, p2 + 0.01),
                        "restricted_time_to_depletion": 30.0,
                        "survived_through_horizon": 1.0,
                    },
                ],
            }
        )
    return _seal_input({
        "schema_version": 1,
        "artifact_type": "cross_part_confirmatory_units",
        "campaign_manifest_sha256": CAMPAIGN_SHA,
        "source_data_lock_path": "/private/confirmatory-data-lock.json",
        "source_data_lock_sha256": "d" * 64,
        "frozen_system_ids": frozen,
        "systems": systems,
    })


def test_cross_part_cli_nested_units_discordance_and_holm(tmp_path: Path) -> None:
    input_path = tmp_path / "cross-private.json"
    _write_private(input_path, _cross_input())
    output = tmp_path / "cross-estimates.json"
    completed = _run("cross-part", "--input", input_path, "--output", output)
    assert completed.returncode == 0, completed.stderr
    artifact = _verify_artifact(output)
    assert artifact["bootstrap_method"] == (
        "finite_panel_within_system_prompt_root_trajectory_units_only"
    )
    assert artifact["system_resampling"] == "none_primary_finite_panel"
    assert artifact["superpopulation_sensitivity_bootstrap_method"] == (
        "systems_then_within_system_prompt_root_trajectory_units"
    )
    assert artifact["bootstrap_replicates"] == 2000
    assert artifact["permutation_replicates"] == 2000
    assert len(artifact["results"]) == 3
    assert artifact["system_count"] == 12
    assert artifact["developer_count"] == 4
    for result in artifact["results"]:
        assert 0.0 <= result["discordance_rate"] <= 1.0
        assert result["raw_permutation_p"] <= result["holm_adjusted_p"]
        assert result["leave_one_developer_out_min"] <= result[
            "leave_one_developer_out_max"
        ]
        assert result["superpopulation_sensitivity_spearman_ci_low"] <= result[
            "superpopulation_sensitivity_spearman_ci_high"
        ]
    p0_p2 = next(
        row
        for row in artifact["results"]
        if row["left_part"] == "part0" and row["right_part"] == "part2"
    )
    assert p0_p2["spearman"] == -1.0
    assert p0_p2["discordance_rate"] == 1.0


def test_default_24_plus_6_cohort_primary_excludes_historical_systems(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "cross-24-current-6-historical.json"
    _write_private(input_path, _cross_input(current_count=24, historical_count=6))
    output = tmp_path / "cross-cohort-estimates.json"
    completed = _run("cross-part", "--input", input_path, "--output", output)
    assert completed.returncode == 0, completed.stderr
    artifact = _verify_artifact(output)
    assert artifact["system_count"] == 24
    assert artifact["developer_count"] == 8
    assert artifact["historical_system_count"] == 6
    assert len(artifact["system_primary_metrics"]) == 24
    assert len(artifact["historical_system_metrics"]) == 6
    primary_ids = {row["system_id"] for row in artifact["system_primary_metrics"]}
    historical_ids = {
        row["system_id"] for row in artifact["historical_system_metrics"]
    }
    assert primary_ids == {f"system-{index:02d}" for index in range(24)}
    assert historical_ids == {f"system-{index:02d}" for index in range(24, 30)}
    primary = next(
        row
        for row in artifact["results"]
        if row["left_part"] == "part0" and row["right_part"] == "part2"
    )
    historical = next(
        row
        for row in artifact["historical_results"]
        if row["left_part"] == "part0" and row["right_part"] == "part2"
    )
    assert primary["spearman"] == -1.0
    assert historical["spearman"] == 1.0
    assert math.isfinite(primary["developer_balanced_spearman"])


def test_cross_part_cli_fails_closed_on_lineage_units_and_completeness(
    tmp_path: Path,
) -> None:
    tampered = _cross_input()
    tampered["campaign_manifest_sha256"] = "C" * 64
    _seal_input(tampered)
    path = tmp_path / "uppercase.json"
    _write_private(path, tampered)
    rejected = _run("cross-part", "--input", path, "--output", tmp_path / "out.json")
    assert rejected.returncode == 2
    assert "lowercase 64-character" in rejected.stderr
    assert "Traceback" not in rejected.stderr

    duplicate = _cross_input()
    duplicate["systems"][0]["part0_units"][1]["unit_id"] = duplicate["systems"][0][
        "part0_units"
    ][0]["unit_id"]
    _seal_input(duplicate)
    path = tmp_path / "duplicate.json"
    _write_private(path, duplicate)
    rejected = _run("cross-part", "--input", path, "--output", tmp_path / "dup.json")
    assert rejected.returncode == 2
    assert "duplicate unit_id" in rejected.stderr

    missing = _cross_input()
    missing["systems"] = missing["systems"][:-1]
    _seal_input(missing)
    path = tmp_path / "missing.json"
    _write_private(path, missing)
    rejected = _run("cross-part", "--input", path, "--output", tmp_path / "missing-out.json")
    assert rejected.returncode == 2
    assert "exactly cover" in rejected.stderr
    assert not (tmp_path / "missing-out.json").exists()

    unsealed_tamper = _cross_input()
    unsealed_tamper["systems"][0]["part2_units"][0]["normalized_aurc"] = 0.123
    path = tmp_path / "unsealed-tamper.json"
    _write_private(path, unsealed_tamper)
    rejected = _run(
        "cross-part",
        "--input",
        path,
        "--output",
        tmp_path / "tamper-out.json",
    )
    assert rejected.returncode == 2
    assert "artifact_sha256 verification failed" in rejected.stderr
    assert "Traceback" not in rejected.stderr

    wrong_arm = _cross_input()
    wrong_arm["systems"][0]["part0_units"][0]["arm"] = "control"
    _seal_input(wrong_arm)
    path = tmp_path / "control-units.json"
    _write_private(path, wrong_arm)
    rejected = _run("cross-part", "--input", path, "--output", tmp_path / "control.json")
    assert rejected.returncode == 2
    assert "harmful-arm/all-languages" in rejected.stderr

    wrong_frame = _cross_input()
    wrong_frame["systems"][0]["part1_units"][0]["frame"] = "advice"
    _seal_input(wrong_frame)
    path = tmp_path / "role-units.json"
    _write_private(path, wrong_frame)
    rejected = _run("cross-part", "--input", path, "--output", tmp_path / "role.json")
    assert rejected.returncode == 2
    assert "primary/self_direct" in rejected.stderr

    wrong_source = _cross_input()
    wrong_source["systems"][0]["part2_units"][0]["analysis_source"] = "sensitivity"
    _seal_input(wrong_source)
    path = tmp_path / "sensitivity-units.json"
    _write_private(path, wrong_source)
    rejected = _run(
        "cross-part", "--input", path, "--output", tmp_path / "sensitivity.json"
    )
    assert rejected.returncode == 2
    assert "final_baseline" in rejected.stderr


def _locked_context_fixture(tmp_path: Path, monkeypatch):
    variance_path = tmp_path / "variance-manifest.json"
    baseline_path = tmp_path / "baseline-manifest.json"
    _write_private(variance_path, {})
    _write_private(baseline_path, {})
    source = tmp_path / "locked-source.bin"
    source.write_bytes(b"locked-native-bytes")
    reference = {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
    }
    targets = [{"id": "system-a"}, {"id": "system-b"}]
    variance = {
        "manifest_sha256": "1" * 64,
        "plan_sha256": "2" * 64,
        "targets": targets,
        "jobs": [
            {"id": "p0-a", "experiment": "part0", "stage": "production"},
            {"id": "p1-a", "experiment": "part1", "stage": "production"},
        ],
    }
    baseline = {
        "manifest_sha256": "3" * 64,
        "plan_sha256": "4" * 64,
        "targets": targets,
        "jobs": [
            {
                "id": "p2-a",
                "experiment": "part2",
                "stage": "part2_baseline_production",
            }
        ],
    }
    variance_hash = sha256_file(variance_path)
    baseline_hash = sha256_file(baseline_path)

    def validate(path, *, label, expected_scientific_stage):
        if expected_scientific_stage == "part2_variance_pilot":
            return variance_path.resolve(), variance, variance_hash
        return baseline_path.resolve(), baseline, baseline_hash

    monkeypatch.setattr(
        confirmatory_data_lock, "_validated_campaign_stage", validate
    )
    fixture_metadata = [
        {
            "system_id": target["id"],
            "cohort_id": "current_sota",
            "developer_id": f"developer-{index}",
        }
        for index, target in enumerate(targets)
    ]
    monkeypatch.setattr(
        confirmatory_data_lock,
        "_validate_cohort_estimand",
        lambda campaign: fixture_metadata,
    )
    lock = {
        "schema_version": 1,
        "artifact_type": "confirmatory_data_lock",
        "status": "locked",
        "campaigns": {
            "variance_stage": {
                "path": str(variance_path.resolve()),
                "file_sha256": variance_hash,
                "manifest_sha256": variance["manifest_sha256"],
                "plan_sha256": variance["plan_sha256"],
            },
            "baseline_stage": {
                "path": str(baseline_path.resolve()),
                "file_sha256": baseline_hash,
                "manifest_sha256": baseline["manifest_sha256"],
                "plan_sha256": baseline["plan_sha256"],
            },
        },
        "approved_inputs": [reference],
        "protocol_and_source_files": [reference],
        "artifacts": [reference],
        "exclusions": {
            "included_scientific_job_ids": ["p0-a", "p1-a", "p2-a"],
            "excluded_scientific_job_ids": [],
        },
        "panel_estimands": {
            "primary_cohort_id": "current_sota",
            "historical_cohort_id": "historical",
            "current_system_count": 2,
            "current_developer_count": 2,
            "historical_system_count": 0,
            "developer_balanced_summary_required": True,
        },
        "completeness": {"all_selected_lineage_artifacts_reverified": True},
    }
    lock["data_lock_sha256"] = stable_json_hash(lock)
    lock_path = tmp_path / "confirmatory-data-lock.json"
    _write_private(lock_path, lock)
    return lock_path, sha256_file(lock_path), source


def test_native_data_lock_revalidation_rejects_wrong_hash_and_tampered_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    lock_path, lock_hash, source = _locked_context_fixture(tmp_path, monkeypatch)
    (
        lock,
        variance,
        baseline,
        system_metadata,
        verified,
    ) = confirmatory_estimators._validated_native_context(
        lock_path, expected_sha256=lock_hash
    )
    assert verified == lock_hash
    assert variance["plan_sha256"] == "2" * 64
    assert baseline["plan_sha256"] == "4" * 64
    assert len(system_metadata) == 2
    with pytest.raises(ValueError, match="file hash mismatch"):
        confirmatory_estimators._validated_native_context(
            lock_path, expected_sha256="f" * 64
        )
    source.write_bytes(b"tampered-native-bytes")
    with pytest.raises(ValueError, match="bytes changed"):
        confirmatory_estimators._validated_native_context(
            lock_path, expected_sha256=lock_hash
        )


def test_estimator_cli_rejects_altered_rows_after_native_rederivation(
    tmp_path: Path, monkeypatch
) -> None:
    native = _part0_input()
    altered = json.loads(json.dumps(native))
    altered["rows"][0]["outcome"] = "COMPLIANCE"
    _seal_input(altered)
    input_path = tmp_path / "altered-private.json"
    _write_private(input_path, altered)
    monkeypatch.setattr(
        confirmatory_estimators,
        "materialize_native_units",
        lambda *args, **kwargs: native,
    )
    output = tmp_path / "must-not-exist.json"
    result = confirmatory_estimators.main(
        [
            "part0", "--input", str(input_path),
            "--data-lock", str(tmp_path / "lock.json"),
            "--data-lock-sha256", "d" * 64,
            "--output", str(output),
        ]
    )
    assert result == 2
    assert not output.exists()
