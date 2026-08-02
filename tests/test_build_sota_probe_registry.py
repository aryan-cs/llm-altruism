from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from analysis.build_sota_inference_hub_roster import build_roster
from analysis.build_sota_probe_registry import (
    ProbeRegistryBuildError,
    build_probe_registry,
    main,
    validate_probe_registry,
)
from analysis.reconcile_inference_hub_routes import reconcile_routes
from experiments.misc.inference_hub_compatibility import _validate_inputs


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/private/inference_hub/catalog-live-20260802-v3.json"
PRODUCTION_REGISTRY_PATH = ROOT / "agents/agent_config.registry.json"


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture(scope="module")
def live_inputs() -> tuple[dict, dict, dict]:
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    production_registry = json.loads(
        PRODUCTION_REGISTRY_PATH.read_text(encoding="utf-8")
    )
    roster = build_roster(catalog=catalog, registry=production_registry)
    return catalog, production_registry, roster


def test_live_roster_builds_84_exploratory_only_exact_identity_targets(
    live_inputs: tuple[dict, dict, dict],
) -> None:
    catalog, _, roster = live_inputs
    production_before = PRODUCTION_REGISTRY_PATH.read_bytes()
    registry = build_probe_registry(roster=roster)

    assert PRODUCTION_REGISTRY_PATH.read_bytes() == production_before
    assert len(registry["targets"]) == 84
    assert registry["default_cohort"] == "exploratory_sota"
    assert registry["cohorts"]["exploratory_sota"]["targets"] == [
        target["id"] for target in registry["targets"]
    ]
    assert registry["source_roster_artifact_sha256"] == roster["artifact_sha256"]
    assert registry["source_catalog_input_sha256"] == roster["catalog"]["input_sha256"]
    assert registry["source_catalog_input_sha256"] == _canonical_hash(catalog)
    assert registry["source_catalog_payload_sha256"] == catalog[
        "source_payload_sha256"
    ]
    assert registry["confirmatory_promotion_permitted"] is False
    assert registry["paper_result_promotion_permitted"] is False
    assert registry["production_registry_mutation_permitted"] is False

    subjects = {row["subject_id"]: row for row in roster["subject_roster"]}
    assert set(subjects) == {target["id"] for target in registry["targets"]}
    for target in registry["targets"]:
        subject = subjects[target["id"]]
        assert target == {
            "id": subject["subject_id"],
            "provider": "inference_hub",
            "upstream_provider": subject["upstream_provider"],
            "model": subject["model_identifier"],
            "route": subject["preferred_candidate"],
            "endpoint_profile": "inference_hub",
            "verification_status": "unverified",
            "route_source": "sealed_sota_roster_preferred_candidate",
        }

    validate_probe_registry(registry)
    assert registry["artifact_sha256"] == _canonical_hash(
        {key: value for key, value in registry.items() if key != "artifact_sha256"}
    )


def test_live_bridge_contains_no_judge_evals_safety_or_non_text_target(
    live_inputs: tuple[dict, dict, dict],
) -> None:
    _, _, roster = live_inputs
    registry = build_probe_registry(roster=roster)
    subject_by_id = {row["subject_id"]: row for row in roster["subject_roster"]}
    inventory = {row["route"]: row for row in roster["route_inventory"]}

    assert all(subject_by_id[target["id"]]["modality"] == "text" for target in registry["targets"])
    assert all(
        subject_by_id[target["id"]]["role"]
        in {"general_chat", "coding_chat", "search_chat"}
        for target in registry["targets"]
    )
    assert all(
        inventory[target["route"]]["eligible_text_chat"] is True
        for target in registry["targets"]
    )
    assert not any(
        marker in "/".join(
            (target["id"], target["model"], target["route"])
        ).casefold()
        for target in registry["targets"]
        for marker in ("evals", "judge", "guard", "safety")
    )


def test_all_84_targets_fully_reconcile_and_replay_in_compatibility_validator(
    live_inputs: tuple[dict, dict, dict],
) -> None:
    catalog, _, roster = live_inputs
    registry = build_probe_registry(roster=roster)
    report = reconcile_routes(catalog=catalog, registry=registry)

    assert report["target_count"] == 84
    assert report["candidate_selected_count"] == 84
    assert report["unresolved_count"] == 0
    assert _validate_inputs(catalog, registry, report) == report

    subjects = {row["subject_id"]: row for row in roster["subject_roster"]}
    for resolution in report["resolutions"]:
        subject = subjects[resolution["target_id"]]
        assert resolution["cohort"] == "exploratory_sota"
        assert resolution["planned_route"] == subject["preferred_candidate"]
        assert resolution["model"] == subject["model_identifier"]
        assert set(resolution["all_exact_suffix_candidates"]) == set(
            subject["all_exact_candidate_backends"]
        )
        assert resolution["selected_candidate"] in subject[
            "all_exact_candidate_backends"
        ]


def test_tampered_or_resealed_unsafe_rosters_and_registries_fail_closed(
    live_inputs: tuple[dict, dict, dict],
) -> None:
    _, _, roster = live_inputs
    tampered = copy.deepcopy(roster)
    tampered["subject_roster"][0]["model_identifier"] = "forged-model"
    with pytest.raises(ProbeRegistryBuildError, match="artifact_sha256 is invalid"):
        build_probe_registry(roster=tampered)

    resealed = copy.deepcopy(roster)
    resealed["subject_roster"][0]["role"] = "judge"
    resealed.pop("artifact_sha256")
    resealed["artifact_sha256"] = _canonical_hash(resealed)
    with pytest.raises(ProbeRegistryBuildError, match="allowed text-chat"):
        build_probe_registry(roster=resealed)

    registry = build_probe_registry(roster=roster)
    registry["targets"][0]["verification_status"] = "verified"
    with pytest.raises(ProbeRegistryBuildError, match="artifact_sha256 is invalid"):
        validate_probe_registry(registry)


def test_cli_writes_byte_deterministic_atomic_registry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    live_inputs: tuple[dict, dict, dict],
) -> None:
    _, _, roster = live_inputs
    roster_path = tmp_path / "roster.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    roster_path.write_text(json.dumps(roster), encoding="utf-8")

    assert main(["--roster", str(roster_path), "--output", str(first)]) == 0
    assert main(["--roster", str(roster_path), "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))
    validate_probe_registry(json.loads(first.read_text(encoding="utf-8")))
    assert "Built 84 exploratory SOTA probe targets" in capsys.readouterr().out
