from __future__ import annotations

import hashlib
import json
import csv
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments import confirmatory_campaign as campaign
from experiments.confirmatory_campaign import (
    ConfirmatoryCampaignError,
    ProcessResult,
    build_plan,
    create_manifest,
    execute_manifest,
    load_manifest,
    validate_manifest,
)
from analysis import part2_confirmatory
from analysis.part2_dynamics import (
    structural_cell_id_for_contract,
    trajectory_id_for_contract,
)
from experiments.part2 import part_2


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verification(route: str, index: int) -> dict[str, Any]:
    return {
        "verified_at_utc": "2026-08-02T00:00:00+00:00",
        "discovery_sha256": _sha(f"discovery-{index}"),
        "smoke_test": {
            "completed_at_utc": "2026-08-02T00:00:00+00:00",
            "request_id": f"request-{index}",
            "response_model": route,
            "response_sha256": _sha(f"response-{index}"),
            "finish_reason": "stop",
            "usage_sha256": _sha(f"usage-{index}"),
            "generation_controls": {
                "temperature": 0,
                "top_p": 1,
                "seed": 20_260_801,
                "max_tokens": 16,
                "stream": False,
                "structured_output": True,
                "json_schema_sha256": _sha("schema"),
            },
        },
    }


def _targets(count: int = 30) -> list[dict[str, Any]]:
    result = []
    for index in range(count):
        route = f"vendor/model-{index:02d}"
        result.append(
            {
                "id": f"target-{index:02d}",
                "provider": "inference_hub",
                "upstream_provider": f"vendor-{index % 5}",
                "model": route,
                "route": route,
                "endpoint_profile": "inference_hub",
                "verification_status": "verified",
                "route_source": "inference_hub_models_api",
                "verification_evidence": _verification(route, index),
            }
        )
    return result


def _install_cohorts(
    monkeypatch: pytest.MonkeyPatch, targets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    split = min(24, max(1, len(targets) - 1))
    cohorts = {
        "current_sota": targets[:split],
        "historical": targets[split:],
    }
    monkeypatch.setattr(campaign, "require_fresh_route_verification", lambda entry: None)
    if not cohorts["historical"]:
        cohorts["current_sota"] = targets
        cohorts["historical"] = []

    def load(name: str):
        if name not in cohorts or not cohorts[name]:
            raise KeyError(name)
        return {
            "id": name,
            "version": "fixture-v1",
            "registry_version": "fixture-registry-v1",
            "registry_hash": _sha("fixture-registry"),
            "routing_roster_hash": _sha("fixture-routing-roster"),
            "route_verification_policy": {
                "schema_version": 1,
                "max_age_hours": 168,
                "require_complete_registry_bundle": True,
            },
            "verification_bundle": {
                "schema_version": 1,
                "bundle_sha256": _sha("fixture-bundle"),
                "verified_at_utc": "2026-08-02T00:00:00+00:00",
            },
            "targets": deepcopy(cohorts[name]),
        }

    monkeypatch.setattr(campaign, "load_model_cohort", load)
    return [target for name in ("current_sota", "historical") for target in cohorts[name]]


def _evidence(targets: list[dict[str, Any]]) -> dict[str, Any]:
    split = min(24, max(1, len(targets) - 1))
    cohort_rows = [
        {
            "id": "current_sota",
            "version": "fixture-v1",
            "target_ids": [target["id"] for target in targets[:split]],
        },
        {
            "id": "historical",
            "version": "fixture-v1",
            "target_ids": [target["id"] for target in targets[split:]],
        },
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "verified_at_utc": "2026-08-02T00:00:00+00:00",
        "endpoint": "https://inference-api.nvidia.com/v1",
        "registry_version": "fixture-registry-v1",
        "registry_hash": _sha("fixture-registry"),
        "routing_roster_sha256": _sha("fixture-routing-roster"),
        "catalog_source_payload_sha256": _sha("catalog"),
        "cohorts": cohort_rows,
        "target_count": len(targets),
        "targets": [
            {
                "target_id": target["id"],
                "upstream_provider": target["upstream_provider"],
                "route": target["route"],
                "evidence": {
                    "schema_version": 2,
                    "verification_status": "verified",
                    "endpoint": "https://inference-api.nvidia.com/v1",
                    "requested_route": target["route"],
                    "provider_response_model": target["route"],
                    "verification_evidence": deepcopy(
                        target["verification_evidence"]
                    ),
                },
            }
            for target in targets
        ],
    }
    payload["bundle_sha256"] = campaign.stable_json_hash(payload)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(
    tmp_path: Path,
    evidence_path: Path,
    evidence_hash: str,
    *,
    part2_stage: str = "variance-pilot",
    gate_path: Path | None = None,
    gate_hash: str | None = None,
) -> Any:
    p0 = tmp_path / "part0.json"
    p1 = tmp_path / "part1.json"
    if not p0.exists():
        p0.write_text("{}\n", encoding="utf-8")
    if not p1.exists():
        p1.write_text("{}\n", encoding="utf-8")
    return SimpleNamespace(
        campaign_id="fixture-campaign",
        cohort=None,
        target_id=None,
        extractor_target_id="target-00",
        judge_target_id="target-01",
        part0_registry=str(p0),
        part0_registry_sha256=hashlib.sha256(p0.read_bytes()).hexdigest(),
        part1_bank=str(p1),
        part1_bank_sha256=hashlib.sha256(p1.read_bytes()).hexdigest(),
        endpoint_evidence=str(evidence_path),
        endpoint_evidence_sha256=evidence_hash,
        part2_stage=part2_stage,
        variance_selection=str(gate_path) if gate_path else None,
        variance_selection_sha256=gate_hash,
        variance_pilot_manifest=None,
        variance_pilot_manifest_sha256=None,
        part2_society_size=50,
        part2_days=100,
        part2_resource="water",
        part2_selfish_gain=2,
        part2_depletion_units=2,
        part2_community_benefit=5,
        timeout_seconds=100,
    )


@pytest.fixture()
def planned_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    targets = _install_cohorts(monkeypatch, _targets())
    evidence_path = tmp_path / "endpoint-evidence.json"
    evidence_hash = _write_json(evidence_path, _evidence(targets))
    monkeypatch.setattr(campaign.part0_runner, "load_production_registry", lambda *a, **k: object())
    monkeypatch.setattr(campaign.part1_runner, "load_production_bank", lambda *a, **k: object())
    freeze = {
        "schema_version": 1,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "python_executable": "/fixture/python",
        "source_bundle": {"files": [], "bundle_sha256": _sha("sources")},
        "dependency_lock": {"files": [], "bundle_sha256": _sha("lock")},
    }
    monkeypatch.setattr(campaign, "_execution_freeze", lambda: deepcopy(freeze))
    route_by_name = {target["route"]: target for target in targets}
    from experiments.misc import final_answer

    monkeypatch.setattr(
        final_answer,
        "resolve_model_registry_entry",
        lambda provider, model: {
            **deepcopy(route_by_name[model]),
            "registry_version": "fixture-registry-v1",
            "registry_hash": _sha("fixture-registry"),
        },
    )
    args = _args(tmp_path, evidence_path, evidence_hash)
    return args, targets, build_plan(args), freeze


def test_default_exact_30_target_union_and_variance_pilot_job_counts(
    planned_fixture,
) -> None:
    _, targets, manifest, _ = planned_fixture
    assert len(targets) == len(manifest["targets"]) == 30
    assert [row["id"] for row in manifest["cohorts"]] == [
        "current_sota",
        "historical",
    ]
    assert len({row["route"] for row in manifest["targets"]}) == 30
    assert manifest["part2_design"]["replicates_per_target"] == 8
    assert len(manifest["part2_design"]["common_generation_seeds"]) == 8
    assert len(manifest["jobs"]) == 390
    assert manifest["scheduling"] == {
        "protocol": campaign.CAMPAIGN_SCHEDULING_PROTOCOL,
        "seed": campaign.CAMPAIGN_SCHEDULING_SEED,
        "unit": "target_then_part_dependency_block",
        "smoke_dependency": "same_target_same_part_smoke_precedes_science",
    }
    positions = {job["id"]: index for index, job in enumerate(manifest["jobs"])}
    assert all(
        job["smoke_job_id"] is None
        or positions[job["smoke_job_id"]] < positions[job["id"]]
        for job in manifest["jobs"]
    )
    assert [job["id"] for job in build_plan(planned_fixture[0])["jobs"]] == [
        job["id"] for job in manifest["jobs"]
    ]
    assert [job["stage"] for job in manifest["jobs"]] != sorted(
        job["stage"] for job in manifest["jobs"]
    )
    counts: dict[tuple[str, str], int] = {}
    for job in manifest["jobs"]:
        counts[(job["stage"], job["experiment"])] = (
            counts.get((job["stage"], job["experiment"]), 0) + 1
        )
        assert isinstance(job["argv_fresh"], list)
        assert isinstance(job["argv_resume"], list)
        assert job["provider"] == "inference_hub"
    assert counts == {
        ("smoke", "part0"): 30,
        ("smoke", "part1"): 30,
        ("smoke", "part2"): 30,
        ("production", "part0"): 30,
        ("production", "part1"): 30,
        ("part2_variance_pilot", "part2"): 240,
    }
    for job in (
        row
        for row in manifest["jobs"]
        if row["stage"] == "production" and row["experiment"] == "part0"
    ):
        option_index = job["argv_fresh"].index("--completed-smoke-dir")
        assert job["argv_fresh"][option_index + 1].endswith(
            f"/{job['target_id']}/smoke"
        )
        assert job["argv_resume"][option_index + 1] == job["argv_fresh"][
            option_index + 1
        ]
    for job in (
        row
        for row in manifest["jobs"]
        if row["stage"] == "production" and row["experiment"] == "part1"
    ):
        option_index = job["argv_fresh"].index("--completed-smoke-directory")
        assert job["argv_fresh"][option_index + 1].endswith(
            f"/{job['target_id']}/smoke"
        )
        assert job["argv_resume"][option_index + 1] == job["argv_fresh"][
            option_index + 1
        ]
    scientific = [
        job for job in manifest["jobs"] if job["stage"] == "part2_variance_pilot"
    ]
    by_target: dict[str, set[tuple[int, int]]] = {}
    for job in scientific:
        expected = job["expected"]
        by_target.setdefault(job["target_id"], set()).add(
            (expected["generation_seed"], expected["environment_seed"])
        )
    assert len({frozenset(value) for value in by_target.values()}) == 1
    assert all(len(value) == 8 for value in by_target.values())
    part2_smokes = [
        job
        for job in manifest["jobs"]
        if job["stage"] == "smoke" and job["experiment"] == "part2"
    ]
    assert all(
        job["expected"]["society_config"]
        == {
            "society_size": 4,
            "days": 3,
            "resource": "water",
            "selfish_gain": 2,
            "depletion_units": 2,
            "community_benefit": 5,
        }
        and job["expected"]["resource_capacity"] == 8
        and job["expected"]["collapse_death_rate"] == 0.2
        and job["expected"]["row_count_upper_bound"] == 12
        for job in part2_smokes
    )


def test_hash_bound_target_shard_keeps_full_union_attestation(
    planned_fixture,
) -> None:
    args, _targets_all, _manifest, _freeze = planned_fixture
    args.target_id = ["target-00"]

    manifest = build_plan(args)

    assert manifest["target_selection"] == {
        "mode": "shard",
        "selected_target_ids": ["target-00"],
        "complete_union_target_count": 30,
    }
    assert [target["id"] for target in manifest["targets"]] == ["target-00"]
    assert len(manifest["jobs"]) == 13
    assert {job["target_id"] for job in manifest["jobs"]} == {"target-00"}
    assert len(manifest["cohorts"][0]["target_ids"]) == 24
    assert len(manifest["cohorts"][1]["target_ids"]) == 6


def test_duplicate_target_route_and_unverified_routes_fail_before_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets()
    targets[25]["route"] = targets[0]["route"]
    _install_cohorts(monkeypatch, targets)
    evidence_path = tmp_path / "evidence.json"
    evidence_hash = _write_json(evidence_path, _evidence(targets))
    args = _args(tmp_path, evidence_path, evidence_hash)
    with pytest.raises(ConfirmatoryCampaignError, match="duplicated"):
        build_plan(args)

    targets = _targets()
    targets[0]["verification_status"] = "unverified"
    _install_cohorts(monkeypatch, targets)
    evidence_hash = _write_json(evidence_path, _evidence(targets))
    args = _args(tmp_path, evidence_path, evidence_hash)
    with pytest.raises(ConfirmatoryCampaignError, match="not verified"):
        build_plan(args)


def test_stale_route_attestation_fails_before_input_loaders_or_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = _targets()
    _install_cohorts(monkeypatch, targets)
    monkeypatch.setattr(
        campaign,
        "require_fresh_route_verification",
        lambda entry: (_ for _ in ()).throw(ValueError("stale route evidence")),
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_hash = _write_json(evidence_path, _evidence(targets))
    args = _args(tmp_path, evidence_path, evidence_hash)
    called = False

    def forbidden_loader(*args, **kwargs):
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(campaign.part0_runner, "load_production_registry", forbidden_loader)
    with pytest.raises(ConfirmatoryCampaignError, match="not fresh"):
        build_plan(args)
    assert called is False


def test_endpoint_evidence_route_coverage_and_file_hash_tampering_fail(
    planned_fixture,
    tmp_path: Path,
) -> None:
    args, targets, _, _ = planned_fixture
    evidence_path = Path(args.endpoint_evidence)
    payload = _evidence(targets)
    payload["targets"][0]["route"] = "wrong/route"
    payload["bundle_sha256"] = campaign.stable_json_hash(
        {key: value for key, value in payload.items() if key != "bundle_sha256"}
    )
    args.endpoint_evidence_sha256 = _write_json(evidence_path, payload)
    with pytest.raises(ConfirmatoryCampaignError, match="exact verified route"):
        build_plan(args)

    args.endpoint_evidence_sha256 = "0" * 64
    with pytest.raises(ConfirmatoryCampaignError, match="hash mismatch"):
        build_plan(args)


def test_campaign_json_loader_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}\n', encoding="utf-8")
    with pytest.raises(ConfirmatoryCampaignError, match="not valid strict UTF-8 JSON"):
        campaign._load_json(path, label="duplicate fixture")


def _gate(selected_n: int, *, pilot_manifest_sha256: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "part2_identity_masked_variance_selection",
        "private_input_sha256": _sha("private-variance-input"),
        "pilot_campaign_manifest_sha256": (
            pilot_manifest_sha256 or _sha("pilot-manifest")
        ),
        "selection": {
            "schema_version": 1,
            "selection_rule": "smallest_n_with_t95_half_width_at_most_0.05_capped_20_40",
            "identity_masked": True,
            "group_count": 30,
            "pilot_runs_per_group": 8,
            "common_environment_seed_count": 8,
            "common_environment_seeds_sha256": _sha("environment-seeds"),
            "target_half_width": 0.05,
            "s_max": 0.1,
            "selected_common_run_count": selected_n,
            "t_critical": 2.093024,
            "achieved_half_width": 0.0468,
            "capped_at_maximum": False,
            "location_removed_masked_input_sha256": _sha("centered-input"),
        },
    }
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    payload["artifact_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def test_baseline_count_comes_only_from_hash_pinned_blinded_gate(
    planned_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from analysis import part2_confirmatory

    args, _, pilot_plan, _ = planned_fixture
    pilot_path = tmp_path / "pilot-manifest.json"
    pilot_path.write_text("{}\n", encoding="utf-8")
    pilot_hash = hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    args.variance_pilot_manifest = str(pilot_path)
    args.variance_pilot_manifest_sha256 = pilot_hash
    monkeypatch.setattr(
        campaign,
        "_validate_completed_variance_pilot_manifest",
        lambda path, *, expected_sha256: {
            **deepcopy(pilot_plan),
            "campaign_id": "pilot-fixture",
            "plan_sha256": _sha("pilot-plan"),
            "manifest_sha256": _sha("pilot-payload"),
        },
    )
    masked_ids = [f"masked-{index:02d}" for index in range(30)]
    seed_keys = [
        str(campaign.DEFAULT_PART2_ENVIRONMENT_SEED_BASE + index)
        for index in range(campaign.VARIANCE_PILOT_REPLICATES)
    ]
    pilot_values = {
        masked_id: {seed: 0.5 for seed in seed_keys}
        for masked_id in masked_ids
    }
    native_input = part2_confirmatory._sealed_artifact(
        {
            "schema_version": 1,
            "artifact_type": "part2_native_identity_masked_variance_input",
            "pilot_campaign_path": str(pilot_path.resolve()),
            "pilot_campaign_manifest_sha256": pilot_hash,
            "pilot_campaign_payload_sha256": _sha("pilot-payload"),
            "native_artifact_set_sha256": _sha("native-artifacts"),
            "identity_masking_scheme": "automated_plan_bound_location_insensitive_v1",
            "frozen_blinded_group_ids": masked_ids,
            "pilot_by_blinded_group": pilot_values,
        }
    )
    monkeypatch.setattr(
        part2_confirmatory,
        "_derive_native_blinded_variance_input",
        lambda path, *, expected_sha256: deepcopy(native_input),
    )
    gate_path = tmp_path / "variance-gate.json"
    valid_gate = _gate(20, pilot_manifest_sha256=pilot_hash)
    valid_gate["private_input_sha256"] = hashlib.sha256(
        part2_confirmatory._canonical_json_bytes(native_input)
    ).hexdigest()
    valid_gate["selection"] = part2_confirmatory.select_blinded_variance_run_count(
        {
            masked_id: {int(seed): value for seed, value in values.items()}
            for masked_id, values in pilot_values.items()
        },
        expected_blinded_groups=masked_ids,
    )
    valid_gate["artifact_sha256"] = hashlib.sha256(
        part2_confirmatory._canonical_json_bytes(
            {
                key: value
                for key, value in valid_gate.items()
                if key != "artifact_sha256"
            }
        )
    ).hexdigest()
    gate_hash = _write_json(gate_path, valid_gate)
    args.part2_stage = "baseline-production"
    args.variance_selection = str(gate_path)
    args.variance_selection_sha256 = gate_hash
    manifest = build_plan(args)
    assert manifest["part2_design"]["variance_selected_n"] == 20
    assert manifest["part2_design"]["replicates_per_target"] == 20
    assert len(manifest["jobs"]) == 630
    assert sum(
        job["stage"] == "part2_baseline_production" for job in manifest["jobs"]
    ) == 600
    assert all(job["experiment"] == "part2" for job in manifest["jobs"])
    monkeypatch.setattr(
        campaign,
        "require_fresh_route_verification",
        lambda entry: (_ for _ in ()).throw(ValueError("elapsed freshness window")),
    )
    continued = build_plan(args)
    assert continued["plan_sha256"] == manifest["plan_sha256"]
    args.judge_target_id = "target-02"
    with pytest.raises(ConfirmatoryCampaignError, match="continue the pilot's exact"):
        build_plan(args)
    args.judge_target_id = "target-01"

    forged = deepcopy(valid_gate)
    forged["selection"]["selected_common_run_count"] = 40
    forged["selection"]["t_critical"] = 2.022691
    forged["artifact_sha256"] = hashlib.sha256(
        part2_confirmatory._canonical_json_bytes(
            {key: value for key, value in forged.items() if key != "artifact_sha256"}
        )
    ).hexdigest()
    args.variance_selection_sha256 = _write_json(gate_path, forged)
    with pytest.raises(ConfirmatoryCampaignError, match="native rederived pilot rule"):
        build_plan(args)

    args.variance_selection_sha256 = _write_json(
        gate_path, _gate(19, pilot_manifest_sha256=pilot_hash)
    )
    with pytest.raises(ConfirmatoryCampaignError, match="20..40"):
        build_plan(args)
    bad = _gate(20, pilot_manifest_sha256=pilot_hash)
    bad["selection"]["identity_masked"] = False
    without_hash = {key: value for key, value in bad.items() if key != "artifact_sha256"}
    bad["artifact_sha256"] = hashlib.sha256(
        (
            json.dumps(without_hash, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    args.variance_selection_sha256 = _write_json(gate_path, bad)
    with pytest.raises(ConfirmatoryCampaignError, match="prespecified identity-masked"):
        build_plan(args)


def test_clean_commit_and_source_dependency_freeze_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "git_commit", lambda: "a" * 40)
    monkeypatch.setattr(campaign, "git_dirty", lambda: True)
    with pytest.raises(ConfirmatoryCampaignError, match="clean Git"):
        campaign._execution_freeze()

    monkeypatch.setattr(campaign, "git_dirty", lambda: False)
    monkeypatch.setattr(campaign, "git_commit", lambda: "short")
    with pytest.raises(ConfirmatoryCampaignError, match="exact Git"):
        campaign._execution_freeze()


def _small_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Any, dict[str, Any]]:
    targets = _install_cohorts(monkeypatch, _targets(2))
    evidence_path = tmp_path / "small-evidence.json"
    evidence_hash = _write_json(evidence_path, _evidence(targets))
    monkeypatch.setattr(campaign.part0_runner, "load_production_registry", lambda *a, **k: object())
    monkeypatch.setattr(campaign.part1_runner, "load_production_bank", lambda *a, **k: object())
    freeze = {
        "schema_version": 1,
        "git_commit": "b" * 40,
        "git_dirty": False,
        "python_executable": "/fixture/python",
        "source_bundle": {"files": [], "bundle_sha256": _sha("small-source")},
        "dependency_lock": {"files": [], "bundle_sha256": _sha("small-lock")},
    }
    monkeypatch.setattr(campaign, "_execution_freeze", lambda: deepcopy(freeze))
    from experiments.misc import final_answer

    by_route = {target["route"]: target for target in targets}
    monkeypatch.setattr(
        final_answer,
        "resolve_model_registry_entry",
        lambda provider, model: {
            **deepcopy(by_route[model]),
            "registry_version": "fixture-registry-v1",
            "registry_hash": _sha("fixture-registry"),
        },
    )
    args = _args(tmp_path, evidence_path, evidence_hash)
    monkeypatch.setattr(campaign, "CAMPAIGN_ROOT", tmp_path / "campaigns")
    return args, build_plan(args)


def test_native_variance_adapter_revalidates_completed_campaign_and_derives_aurcs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _args_value, planned = _small_fixture(tmp_path, monkeypatch)
    manifest_path = create_manifest(planned)
    completed = load_manifest(manifest_path)
    artifacts: dict[str, dict[str, Any]] = {}
    dynamics = {"policy": "native-variance-fixture-v1"}
    society = {
        "society_size": 1,
        "days": 2,
        "resource": "water",
        "selfish_gain": 2,
        "depletion_units": 1,
        "community_benefit": 5,
    }
    structural_cell_id = structural_cell_id_for_contract(
        society_config=society,
        resource_capacity=2,
        collapse_death_rate=0.2,
        dynamics=dynamics,
    )
    for job in completed["jobs"]:
        job["status"] = "complete"
        if job["experiment"] != "part2" or job["stage"] != "part2_variance_pilot":
            job["artifact"] = {"fixture": job["id"]}
            continue
        expected = job["expected"]
        environment_seed = expected["environment_seed"]
        generation_seed = expected["generation_seed"]
        trajectory_id = trajectory_id_for_contract(
            structural_cell_id=structural_cell_id,
            environment_seed=environment_seed,
        )
        run_id = f"run-{job['id']}"
        csv_path = tmp_path / "native-pilot" / f"{job['id']}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for day, reserve in ((1, 2), (2, 1)):
            rows.append(
                {
                    "run_id": run_id,
                    "trajectory_id": trajectory_id,
                    "structural_cell_id": structural_cell_id,
                    "environment_seed": environment_seed,
                    "generation_seed": generation_seed,
                    "anonymous_agent_slot": 1,
                    "died_today": 0,
                    "attrition_rank": "",
                    "attrition_seed": "",
                    "death_selected_slots_json": "[]",
                    "provider": "inference_hub",
                    "model": job["route"],
                    "day": day,
                    "population_start": 1,
                    "population_end": 1,
                    "resource_units_remaining": reserve,
                    "resource_capacity": 2,
                    "resource": "water",
                    "selfish_gain": 2,
                    "depletion_units": 1,
                    "community_benefit": 5,
                }
            )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        metadata_path = csv_path.with_name(f"{csv_path.stem}_meta.json")
        metadata_path.write_text(
            json.dumps(
                {
                    "provider": "inference_hub",
                    "model": job["route"],
                    "parameters": {
                        "part_2_schema_version": part_2.PART_2_SCHEMA_VERSION,
                        "run_id": run_id,
                        "trajectory_id": trajectory_id,
                        "structural_cell_id": structural_cell_id,
                        "society_config": society,
                        "resource_capacity": 2,
                        "collapse_death_rate": 0.2,
                        "environment_seed": environment_seed,
                        "generation_seed": generation_seed,
                        "dynamics": dynamics,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact = {
            "csv_path": str(csv_path),
            "metadata_path": str(metadata_path),
            "rows": len(rows),
        }
        job["artifact"] = deepcopy(artifact)
        artifacts[job["id"]] = artifact
    completed["status"] = "complete"
    campaign._write_manifest(manifest_path, completed)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def resolve(job, before):
        return deepcopy(artifacts.get(job["id"], job.get("artifact", {})))

    monkeypatch.setattr(campaign, "resolve_job_artifact", resolve)
    derived = part2_confirmatory._derive_native_blinded_variance_input(
        manifest_path,
        expected_sha256=manifest_hash,
    )
    assert derived["artifact_type"] == "part2_native_identity_masked_variance_input"
    assert derived["identity_masking_scheme"] == (
        "automated_plan_bound_location_insensitive_v1"
    )
    assert len(derived["frozen_blinded_group_ids"]) == 2
    expected_seeds = {
        str(campaign.DEFAULT_PART2_ENVIRONMENT_SEED_BASE + offset)
        for offset in range(8)
    }
    assert all(
        set(values) == expected_seeds and set(values.values()) == {0.75}
        for values in derived["pilot_by_blinded_group"].values()
    )
    part2_confirmatory._verify_sealed_artifact(derived, hash_field="artifact_sha256")

    input_path = tmp_path / "native-variance-input.json"
    part2_confirmatory._atomic_write_fresh_json(input_path, derived)
    output_path = tmp_path / "native-variance-selection.json"
    result = part2_confirmatory._run_select_variance(
        SimpleNamespace(
            input=input_path,
            pilot_campaign=manifest_path,
            pilot_campaign_sha256=manifest_hash,
            output=output_path,
        )
    )
    assert result["selection"]["group_count"] == 2
    assert result["selection"]["identity_masked"] is True


def test_smoke_failure_blocks_only_matching_scientific_job_and_uses_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest = _small_fixture(tmp_path, monkeypatch)
    path = create_manifest(manifest)
    observed: list[list[str]] = []

    def process(argv, cwd, env, log_path, timeout):
        observed.append(list(argv))
        if "smoke-part0-target-00" in log_path.name:
            return ProcessResult(7, error="synthetic smoke failure")
        return ProcessResult(0)

    def artifact(job, before):
        return {"job_id": job["id"], "sha256": _sha(job["id"])}

    result = execute_manifest(
        manifest,
        path,
        process_runner=process,
        artifact_resolver=artifact,
    )
    failed = next(job for job in result["jobs"] if job["id"] == "smoke-part0-target-00")
    blocked = next(
        job for job in result["jobs"] if job["id"] == "production-part0-target-00"
    )
    other = next(
        job for job in result["jobs"] if job["id"] == "production-part0-target-01"
    )
    assert failed["status"] == "failed"
    assert blocked["status"] == "blocked_smoke"
    assert other["status"] == "complete"
    assert all(isinstance(argv, list) and argv[0] == "/fixture/python" for argv in observed)


def test_manifest_resume_payload_tamper_and_source_drift_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest = _small_fixture(tmp_path, monkeypatch)
    path = create_manifest(manifest)
    execute_manifest(
        manifest,
        path,
        process_runner=lambda *args: ProcessResult(0),
        artifact_resolver=lambda job, before: {"job_id": job["id"]},
    )
    resumed = load_manifest(path)
    assert resumed["status"] == "complete"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["jobs"][0]["route"] = "tampered/route"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ConfirmatoryCampaignError, match="plan hash"):
        load_manifest(path)

    campaign._write_manifest(path, resumed)
    old_freeze = campaign._execution_freeze()
    monkeypatch.setattr(
        campaign,
        "_execution_freeze",
        lambda: {**deepcopy(old_freeze), "source_bundle": {"bundle_sha256": "changed"}},
    )
    with pytest.raises(ConfirmatoryCampaignError, match="drifted"):
        load_manifest(path)


def test_rehashed_manifest_cannot_delete_a_planned_scientific_job(
    planned_fixture,
) -> None:
    _, _, manifest, _ = planned_fixture
    tampered = deepcopy(manifest)
    tampered["jobs"] = [
        job
        for job in tampered["jobs"]
        if job["id"] != "part2_variance_pilot-target-00-s01"
    ]
    tampered["plan_sha256"] = campaign._plan_hash(tampered)
    tampered["manifest_sha256"] = campaign._manifest_hash(tampered)
    with pytest.raises(ConfirmatoryCampaignError, match="job matrix"):
        validate_manifest(tampered)


def test_dry_run_validates_everything_without_creating_campaign_files(
    planned_fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args, _, _, _ = planned_fixture
    monkeypatch.setattr(campaign, "CAMPAIGN_ROOT", tmp_path / "dry-campaigns")
    argv = [
        "--campaign-id",
        args.campaign_id,
        "--extractor-target-id",
        args.extractor_target_id,
        "--judge-target-id",
        args.judge_target_id,
        "--part0-registry",
        args.part0_registry,
        "--part0-registry-sha256",
        args.part0_registry_sha256,
        "--part1-bank",
        args.part1_bank,
        "--part1-bank-sha256",
        args.part1_bank_sha256,
        "--endpoint-evidence",
        args.endpoint_evidence,
        "--endpoint-evidence-sha256",
        args.endpoint_evidence_sha256,
        "--dry-run",
    ]
    assert campaign.main(argv) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dry_run"] is True
    assert summary["target_count"] == 30
    assert not (tmp_path / "dry-campaigns").exists()


def test_run_process_always_disables_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeProcess:
        def wait(self, timeout):
            return 0

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return FakeProcess()

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    result = campaign.run_process(
        ["/fixture/python", "-m", "module"],
        tmp_path,
        {},
        tmp_path / "run.log",
        10,
    )
    assert result.returncode == 0
    assert captured["argv"] == ["/fixture/python", "-m", "module"]
    assert captured["shell"] is False


def test_part2_resolver_refuses_multiple_changed_seed_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = {
        "id": "p2",
        "stage": "part2_variance_pilot",
        "experiment": "part2",
        "expected": {},
    }
    monkeypatch.setattr(
        campaign,
        "_part2_snapshot",
        lambda: {"/tmp/a_meta.json": (1, 1, "a"), "/tmp/b_meta.json": (1, 1, "b")},
    )
    with pytest.raises(ConfirmatoryCampaignError, match="exactly one"):
        campaign._verify_part2_artifact(job, {})
