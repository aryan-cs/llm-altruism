from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import analysis.validate_inference_hub_part2_operational_overlays as validator
from experiments.misc import inference_hub_part2_panel as runner
from experiments.misc.inference_hub_part1_panel import _seal, _sha256_file, _sha256_json


ENDPOINT = "https://inference-api.nvidia.com/v1"
CURSOR_EPOCH = "cursor-epoch-11111111111111111111111111111111"
JUDGE = "judge.nvidia-evals-nemotron-3-30b-a3b"
PANEL_PATH = Path("experiments/sota_cross_axis_part2_100day_panel.json").resolve()
ROOT = Path(__file__).resolve().parents[1]
IS_ANONYMOUS_SUPPLEMENT = (ROOT / "SUPPLEMENT_MANIFEST.json").is_file()
PRODUCTION_PAIRS = (
    (
        ROOT / "data/private/inference_hub/full-part2-n12-n50-d100-main21-v5/private/manifest.json",
        ROOT / "data/private/inference_hub/full-part2-n12-n50-d100-main21-v5-operational-completion-capability-v4/private/manifest.json",
    ),
    (
        ROOT / "data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-recovered-v1/private/manifest.json",
        ROOT / "data/private/inference_hub/full-part2-n12-n50-d100-nemotron-3-ultra-operational-repair-v1/private/manifest.json",
    ),
    (
        ROOT / "data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-recovered-v1/private/manifest.json",
        ROOT / "data/private/inference_hub/full-part2-n12-n50-d100-deepseek-v4-flash-operational-repair-v1/private/manifest.json",
    ),
)


def _subject(target_id: str, index: int) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "upstream_provider": f"provider-{index}",
        "model": f"model-{index}",
        "route": f"region/model-{index}",
        "candidate_index": 0,
        "supported_controls": ["seed", "temperature", "top_p", "structured_response"],
        "selected_profile_id": f"profile-{index}",
        "selected_profile_request_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
    }


def _effective_row(
    subject: dict[str, Any], trajectory_index: int, environment_seed: int,
) -> dict[str, Any]:
    invalid = int(trajectory_index == 0)
    scheduled = 5000
    restraint = 4000
    overuse = scheduled - restraint - invalid
    return {
        "schema_version": 1,
        "target_id": subject["target_id"],
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "trajectory_index": trajectory_index,
        "environment_seed_index": trajectory_index,
        "environment_seed": environment_seed,
        "operationally_eligible": True,
        "scheduled_agent_days": scheduled,
        "responses_received": scheduled,
        "invalid_count": invalid,
        "identity_mismatch_count": 0,
        "transport_failure_count": 0,
        "restraint_count": restraint,
        "overuse_count": overuse,
        "restraint_rate": restraint / scheduled,
        "aurc": 0.8,
        "aupc": 1.0,
        "reserve_nondepletion": True,
        "final_reserve": 500,
        "final_population": 50,
        "population_retention": 1.0,
        "cumulative_private_payoff": restraint + 2 * overuse,
        "cumulative_group_payoff": 0,
        "operational_repair_round": None,
        "source_replaced_for_operational_failure": False,
    }


def _production_shaped_pairs() -> list[validator._ValidatedPair]:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    target_ids = [
        target_id for target_id in panel["subject_target_ids"]
        if target_id not in validator.EXPECTED_EXCLUDED_TARGET_IDS
    ]
    subject_by_id = {
        target_id: _subject(target_id, index)
        for index, target_id in enumerate(target_ids)
    }
    singleton_sets = [
        {"deepseek-ai/deepseek-v4-flash"},
        {"nvidia/nemotron-3-ultra"},
    ]
    main_set = set(target_ids) - set().union(*singleton_sets)
    shard_sets = [main_set, *singleton_sets]
    contract = runner.Part2Contract(50, 100, 12, 2500, 2, 2, 5, 0.2)
    seeds = validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS
    panel_ref = {
        "path": str(PANEL_PATH),
        "file_sha256": _sha256_file(PANEL_PATH),
        "canonical_sha256": _sha256_json(panel),
    }
    judge = {
        "target_id": JUDGE,
        "upstream_provider": "judge",
        "model": "judge-model",
        "route": "region/judge-model",
        "dispatch_permitted_in_this_runner": False,
        "role": "fixed_disjoint_judge_reserved_for_cross_axis_analysis",
    }
    pairs = []
    for shard_index, shard in enumerate(shard_sets):
        subjects = tuple(subject_by_id[target_id] for target_id in target_ids if target_id in shard)
        rows = tuple(
            _effective_row(subject, trajectory_index, seeds[trajectory_index])
            for subject in subjects
            for trajectory_index in range(12)
        )
        models = tuple(runner._aggregate_models(
            rows, subjects, expected_trajectories=12, capacity=2500,
        ))
        source_manifest = {
            "input_artifacts": {"panel": panel_ref},
            "judge_reservation": judge,
            "base_seed": validator.EXPECTED_BASE_SEED,
            "execution_contract": deepcopy(
                validator.EXPECTED_SOURCE_EXECUTION_CONTRACT
            ),
        }
        pairs.append(validator._ValidatedPair(
            source_manifest_path=Path(f"/source-{shard_index}/private/manifest.json"),
            overlay_manifest_path=Path(f"/overlay-{shard_index}/private/manifest.json"),
            source_manifest=source_manifest,
            overlay_manifest={},
            subjects=subjects,
            contract=contract,
            environment_seeds=seeds,
            effective_trajectories=rows,
            effective_models=models,
            source_operational_failure_count=shard_index + 1,
            repair_round_count=shard_index + 2,
        ))
    return pairs


def test_exact_23_route_276_trajectory_union_reconciles_denominators() -> None:
    result = validator._validate_union(_production_shaped_pairs())

    assert result["status"] == "passed"
    assert result["route_count"] == 23
    assert result["trajectory_count"] == 276
    assert result["common_environment_seed_count"] == 12
    assert result["base_seed"] == 20_260_802
    assert result["common_environment_seeds"] == list(
        validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS
    )
    assert result["excluded_target_ids"] == ["anthropic/claude-opus-4-5"]
    assert result["source_operational_failure_trajectories"] == 6
    assert result["executed_operational_repair_rounds"] == 9
    provenance = result["overlay_execution_provenance"]
    assert provenance["manifest_bound_maximum_rounds"] == 8
    assert provenance["journal_observable_dispatch_attempt_ceiling"] == 8
    assert provenance["configured_attempt_ceiling_manifest_bound_for_overlay"] is False
    assert provenance["nonsemantic_local_scheduling_settings_not_manifest_bound"] == [
        "trajectory_workers",
        "participant_workers",
        "initial_exponential_backoff_seconds",
    ]
    denominators = result["denominators"]
    assert denominators["behavioral_all_scheduled_living_agent_days"] == 1_380_000
    assert denominators["behavioral_valid_actions"] == 1_379_977
    assert denominators["semantic_invalid_actions"] == 23
    assert denominators["environmental_eligible_zero_invalid_trajectories"] == 253


def test_frozen_original_scale_source_design_is_exact_and_fail_closed() -> None:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    contract = runner.Part2Contract(50, 100, 12, 2500, 2, 2, 5, 0.2)
    manifest = {
        "base_seed": validator.EXPECTED_BASE_SEED,
        "execution_contract": deepcopy(
            validator.EXPECTED_SOURCE_EXECUTION_CONTRACT
        ),
    }

    assert _sha256_file(PANEL_PATH) == validator.EXPECTED_PANEL_FILE_SHA256
    assert _sha256_json(panel) == validator.EXPECTED_PANEL_CANONICAL_SHA256
    assert validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS == (
        945_353_965,
        674_434_863,
        373_620_026,
        161_949_049,
        305_447_851,
        1_694_051_603,
        827_330_313,
        526_445_251,
        1_853_673_051,
        1_941_677_986,
        1_517_370_141,
        597_333_924,
    )
    assert validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS == tuple(
        runner._environment_seeds(
            panel["panel_id"], validator.EXPECTED_BASE_SEED, 12,
        )
    )
    assert validator._validate_frozen_source_design(
        manifest,
        contract=contract,
        environment_seeds=validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS,
    ) == manifest["execution_contract"]

    wrong_seed = deepcopy(manifest)
    wrong_seed["base_seed"] += 1
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="base seed",
    ):
        validator._validate_frozen_source_design(
            wrong_seed,
            contract=contract,
            environment_seeds=validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS,
        )

    changed_vector = list(validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS)
    changed_vector[-1] += 1
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="seed vector",
    ):
        validator._validate_frozen_source_design(
            manifest, contract=contract, environment_seeds=changed_vector,
        )


def _hydrated_subject(subject: dict[str, Any]) -> dict[str, Any]:
    return {
        **subject,
        "id": subject["target_id"],
        "provider": "inference_hub",
        "target_model": subject["model"],
        "endpoint_profile": "inference_hub",
        "verification_status": "unverified",
        "route_source": "sealed_sota_roster_preferred_candidate",
        "compatibility_max_tokens": 256,
    }


def test_cascading_subject_routes_accept_authentic_registry_superset() -> None:
    source = [_subject("subject.alpha", 0)]
    hydrated = [_hydrated_subject(source[0])]

    validator._validate_cascading_subject_route_lineage(
        deepcopy(hydrated),
        source_manifest_routes=source,
        hydrated_source_routes=hydrated,
    )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda row: row.__setitem__("route", "region/changed-route"),
            "frozen route/control/hash binding",
        ),
        (
            lambda row: row.__setitem__("candidate_index", False),
            "frozen route/control/hash binding",
        ),
        (
            lambda row: row.pop("supported_controls"),
            "lineage schema changed",
        ),
        (
            lambda row: row.pop("compatibility_max_tokens"),
            "lineage schema changed",
        ),
        (
            lambda row: row.__setitem__("request_timeout_seconds", 1),
            "lineage schema changed",
        ),
        (
            lambda row: row.__setitem__("compatibility_max_tokens", 1),
            "bound registry route metadata",
        ),
        (
            lambda row: row.__setitem__("compatibility_max_tokens", 256.0),
            "bound registry route metadata",
        ),
    ],
)
def test_cascading_subject_routes_reject_changed_missing_or_unknown_values(
    mutator: Any, message: str,
) -> None:
    source = [_subject("subject.alpha", 0)]
    hydrated = [_hydrated_subject(source[0])]
    overlay = deepcopy(hydrated)
    mutator(overlay[0])

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match=message,
    ):
        validator._validate_cascading_subject_route_lineage(
            overlay,
            source_manifest_routes=source,
            hydrated_source_routes=hydrated,
        )


def test_cascading_subject_routes_reject_reordered_hydrated_pairing() -> None:
    source = [_subject("subject.alpha", 0), _subject("subject.beta", 1)]
    hydrated = [_hydrated_subject(row) for row in source]
    reordered = list(reversed(hydrated))
    overlay = [
        {
            **source_row,
            **{
                key: hydrated_row[key]
                for key in validator.CASCADING_SUBJECT_ROUTE_REGISTRY_METADATA_KEYS
            },
        }
        for source_row, hydrated_row in zip(source, reordered, strict=True)
    ]

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="frozen route/control/hash binding",
    ):
        validator._validate_cascading_subject_route_lineage(
            overlay,
            source_manifest_routes=source,
            hydrated_source_routes=reordered,
        )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (
            lambda execution: execution.__setitem__("trajectory_workers", 47),
            "original-scale execution design",
        ),
        (
            lambda execution: execution.__setitem__("participant_workers", 15),
            "original-scale execution design",
        ),
        (
            lambda execution: execution.__setitem__("max_transport_attempts", 7),
            "original-scale execution design",
        ),
        (
            lambda execution: execution.__setitem__(
                "initial_exponential_backoff_seconds", 2.0
            ),
            "original-scale execution design",
        ),
        (
            lambda execution: execution.__setitem__("unbound_extra", True),
            "schema changed",
        ),
    ],
)
def test_source_execution_contract_rejects_every_changed_bound_field(
    mutator: Any, message: str,
) -> None:
    execution = deepcopy(validator.EXPECTED_SOURCE_EXECUTION_CONTRACT)
    mutator(execution)
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match=message,
    ):
        validator._validate_execution_contract(execution)


def test_source_execution_contract_rejects_resealed_rate_policy_tamper() -> None:
    execution = deepcopy(validator.EXPECTED_SOURCE_EXECUTION_CONTRACT)
    rate = execution["shared_rate_limit"]
    rate["provider_requests_per_second"] = 2.6
    rate["policy_sha256"] = _sha256_json(
        {key: value for key, value in rate.items() if key != "policy_sha256"}
    )

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="shared rate-limit policy changed",
    ):
        validator._validate_execution_contract(execution)


def _credential_pool_binding() -> dict[str, Any]:
    return {
        "account_count": 3,
        "selection_policy": "thread_safe_round_robin",
        "rate_limit_scope": "independent_per_account",
        "rate_limit_contract": deepcopy(
            validator.EXPECTED_SHARED_RATE_LIMIT_CONTRACT
        ),
        "implementation_files": {
            str(Path(validator.repair.__file__).resolve()): (
                validator.EXPECTED_REPAIR_IMPLEMENTATION_SHA256
            ),
            str(Path(runner.__file__).resolve()): (
                validator.EXPECTED_RUNNER_IMPLEMENTATION_SHA256
            ),
        },
    }


def _credential_source_manifest() -> dict[str, Any]:
    return {
        "source_artifacts": {
            str(Path(runner.__file__).resolve()): (
                validator.EXPECTED_RUNNER_IMPLEMENTATION_SHA256
            ),
        },
    }


def _bind_anonymous_archive_implementation_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise structural checks against intentionally anonymized source bytes.

    Production constants remain untouched. The supplement rewrites private
    deployment literals, so only its tests bind the distributed bytes locally.
    """

    if not IS_ANONYMOUS_SUPPLEMENT:
        return
    cascading_path = Path(validator.cascading.__file__).resolve()
    paths = {
        "EXPECTED_CASCADING_REPAIR_IMPLEMENTATION_SHA256": cascading_path,
        "EXPECTED_RUNNER_IMPLEMENTATION_SHA256": Path(runner.__file__).resolve(),
        "EXPECTED_REPAIR_IMPLEMENTATION_SHA256": Path(
            validator.repair.__file__
        ).resolve(),
        "EXPECTED_DISCOVERY_IMPLEMENTATION_SHA256": cascading_path.with_name(
            "inference_hub_discovery.py"
        ),
        "EXPECTED_RATE_LIMIT_IMPLEMENTATION_SHA256": cascading_path.with_name(
            "inference_hub_rate_limit.py"
        ),
    }
    for attribute, path in paths.items():
        monkeypatch.setattr(validator, attribute, _sha256_file(path))


def test_multikey_credential_pool_exact_provenance_and_implementation_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_anonymous_archive_implementation_hashes(monkeypatch)
    binding = _credential_pool_binding()
    assert _sha256_file(Path(runner.__file__).resolve()) == (
        validator.EXPECTED_RUNNER_IMPLEMENTATION_SHA256
    )
    assert _sha256_file(Path(validator.repair.__file__).resolve()) == (
        validator.EXPECTED_REPAIR_IMPLEMENTATION_SHA256
    )

    tracker = validator._FileTracker()
    validator._validate_credential_pool(
        binding, tracker, source_manifest=_credential_source_manifest(),
    )
    tracker.verify()


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("account_count", 2),
        ("selection_policy", "random"),
        ("rate_limit_scope", "shared_across_accounts"),
    ],
)
def test_multikey_credential_pool_rejects_identity_contract_tamper(
    field: str, replacement: object,
) -> None:
    binding = _credential_pool_binding()
    binding[field] = replacement
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="Credential-pool provenance changed",
    ):
        validator._validate_credential_pool(
            binding,
            validator._FileTracker(),
            source_manifest=_credential_source_manifest(),
        )


def test_multikey_credential_pool_rejects_rate_or_runner_hash_tamper() -> None:
    changed_rate = _credential_pool_binding()
    rate = changed_rate["rate_limit_contract"]
    rate["global_concurrency"] = 59
    rate["policy_sha256"] = _sha256_json(
        {key: value for key, value in rate.items() if key != "policy_sha256"}
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="rate-limit contract changed",
    ):
        validator._validate_credential_pool(
            changed_rate,
            validator._FileTracker(),
            source_manifest=_credential_source_manifest(),
        )

    changed_runner = _credential_pool_binding()
    changed_runner["implementation_files"][
        str(Path(runner.__file__).resolve())
    ] = "0" * 64
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="implementation bindings changed",
    ):
        validator._validate_credential_pool(
            changed_runner,
            validator._FileTracker(),
            source_manifest=_credential_source_manifest(),
        )

    source_mismatch = _credential_source_manifest()
    source_mismatch["source_artifacts"][
        str(Path(runner.__file__).resolve())
    ] = "0" * 64
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="does not match the source campaign runner",
    ):
        validator._validate_credential_pool(
            _credential_pool_binding(),
            validator._FileTracker(),
            source_manifest=source_mismatch,
        )


def test_union_rejects_nonfrozen_base_seed_or_common_seed_vector() -> None:
    wrong_base = _production_shaped_pairs()
    manifest = dict(wrong_base[0].source_manifest)
    manifest["base_seed"] = validator.EXPECTED_BASE_SEED + 1
    wrong_base[0] = validator._ValidatedPair(
        **{**wrong_base[0].__dict__, "source_manifest": manifest}
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="exact frozen base seed",
    ):
        validator._validate_union(wrong_base)

    wrong_vector = _production_shaped_pairs()
    changed = list(validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS)
    changed[0] += 1
    wrong_vector[2] = validator._ValidatedPair(
        **{**wrong_vector[2].__dict__, "environment_seeds": tuple(changed)}
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="exact frozen base seed",
    ):
        validator._validate_union(wrong_vector)


def test_union_rejects_missing_trajectory_and_model_aggregate_tamper() -> None:
    missing = _production_shaped_pairs()
    pair = missing[0]
    missing[0] = validator._ValidatedPair(
        **{
            **pair.__dict__,
            "effective_trajectories": pair.effective_trajectories[:-1],
        }
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="276 unique",
    ):
        validator._validate_union(missing)

    aggregate = _production_shaped_pairs()
    pair = aggregate[0]
    models = [dict(row) for row in pair.effective_models]
    models[0]["total_invalid_count"] += 1
    aggregate[0] = validator._ValidatedPair(
        **{**pair.__dict__, "effective_models": tuple(models)}
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="aggregate union",
    ):
        validator._validate_union(aggregate)


def test_union_rejects_cross_shard_route_identity_overlap() -> None:
    pairs = _production_shaped_pairs()
    first = pairs[0]
    subjects = [dict(row) for row in first.subjects]
    subjects[0]["route"] = pairs[1].subjects[0]["route"]
    pairs[0] = validator._ValidatedPair(
        **{**first.__dict__, "subjects": tuple(subjects)}
    )

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="route identities overlap",
    ):
        validator._validate_union(pairs)


def test_strict_effective_schema_rejects_private_response_field() -> None:
    row = dict(_production_shaped_pairs()[0].effective_trajectories[0])
    row["raw_response"] = {"private": "must never be sanitized"}
    payload = {
        "schema_version": 1,
        "artifact_type": "inference_hub_part2_operational_repair_effective_trajectory_metrics_v1",
        "panel_id": validator.EXPECTED_PANEL_ID,
        "generated_at_utc": "2026-09-11T00:00:00Z",
        "source_manifest_evidence_sha256": "0" * 64,
        "rows": [row],
        "evidence_sha256": "0" * 64,
    }

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="schema changed",
    ):
        validator._validate_trajectory_payload_schema(payload, overlay=True)


def _compat_target(target_id: str, model: str, route: str) -> dict[str, Any]:
    controls = ["seed", "temperature", "top_p", "structured_response"]
    profile = {
        "attempt_id": f"probe-{target_id}",
        "controls": controls,
        "profile_id": f"profile-{target_id}",
        "request_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
        "resumed_from_ledger": False,
        "status": "passed",
        "validation_source": "execution_profile_probe",
    }
    return {
        "target_id": target_id,
        "model": model,
        "candidate_count": 1,
        "frozen_candidate_order": [route],
        "selected_execution_candidate": route,
        "selected_execution_profile": {**profile, "route": route},
        "selection_basis": "first_execution_compatible_in_reconciliation_frozen_order",
        "status": "execution_candidate_selected",
        "candidates": [{
            "route": route,
            "candidate_index": 0,
            "max_tokens": 64,
            "execution_compatible": True,
            "selected_execution_profile": profile,
        }],
    }


class _Client:
    base_url = ENDPOINT

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        return {
            "id": "fixture-response",
            "model": body["model"],
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "action": "OPTION_A",
                        "reasoning": "Preserve the shared reserve.",
                    }),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }


def _small_replay_fixture(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], runner.Part2Contract, dict[str, Any]]:
    subject_id = "subject.alpha"
    route = "region/alpha-model"
    panel = {
        "schema_version": 1,
        "panel_id": "validator-replay-test",
        "judge_target_id": JUDGE,
        "subject_target_ids": [subject_id],
        "part2": {
            "society_size": 5,
            "days": 12,
            "independent_trajectories": 12,
            "resource_capacity": 50,
            "private_gain_for_option_b": 2,
            "reserve_cost_for_option_b": 2,
            "common_environment_seeds": True,
        },
    }
    registry = {
        "schema_version": 1,
        "targets": [
            {"id": subject_id, "provider": "inference_hub", "upstream_provider": "developer", "model": "alpha-model"},
            {"id": JUDGE, "provider": "inference_hub", "upstream_provider": "judge", "model": "judge-model"},
        ],
    }
    compatibility = {
        "schema_version": 2,
        "artifact_type": "inference_hub_provider_compatibility",
        "endpoint": ENDPOINT,
        "registry_sha256": _sha256_json(registry),
        "targets": [
            _compat_target(subject_id, "alpha-model", route),
            _compat_target(JUDGE, "judge-model", "region/judge-model"),
        ],
        "target_count": 2,
        "selected_count": 2,
        "unresolved_count": 0,
    }
    compatibility["evidence_sha256"] = _sha256_json(compatibility)
    paths = [tmp_path / name for name in ("panel.json", "compatibility.json", "registry.json")]
    for path, value in zip(paths, (panel, compatibility, registry)):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    output = tmp_path / "run"
    manifest = runner.run_panel(
        panel_path=paths[0], compatibility_path=paths[1], registry_path=paths[2],
        output_dir=output, client=_Client(), selected_ids=[subject_id],
        trajectory_limit=1, trajectory_workers=1, participant_workers=5,
        max_attempts=1, initial_backoff_seconds=0,
    )
    reference = manifest["journals"][f"{subject_id}::0"]
    journal_path = Path(reference["path"])
    records = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    subject = dict(manifest["subject_routes"][0])
    subject["compatibility_max_tokens"] = 64
    contract = runner.Part2Contract(5, 12, 1, 50, 2, 2)
    return records, subject, contract, manifest


def test_real_journal_replay_validates_attempts_prompts_and_transitions(tmp_path: Path) -> None:
    records, subject, contract, manifest = _small_replay_fixture(tmp_path)
    observed_ids: set[str] = set()

    replayed = validator._replay_trajectory(
        records,
        subject=subject,
        trajectory_index=0,
        environment_seed=manifest["common_environment_seeds"][0],
        contract=contract,
        execution_contract=manifest["execution_contract"],
        global_attempt_ids=observed_ids,
    )

    trajectory_payload = json.loads(
        Path(manifest["sanitized_artifacts"]["trajectory_metrics"]["path"]).read_text(encoding="utf-8")
    )
    assert replayed == trajectory_payload["rows"][0]
    assert len(observed_ids) == replayed["scheduled_agent_days"]


def test_attempt_binding_tamper_and_global_duplicate_are_rejected(tmp_path: Path) -> None:
    records, subject, contract, manifest = _small_replay_fixture(tmp_path)
    seed = manifest["common_environment_seeds"][0]
    tampered = [dict(row) for row in records]
    terminal = next(row for row in tampered if row["event"] == "semantic_result")
    terminal["request_sha256"] = "0" * 64
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="changed its unit or request",
    ):
        validator._replay_trajectory(
            tampered, subject=subject, trajectory_index=0, environment_seed=seed,
            contract=contract, execution_contract=manifest["execution_contract"],
            global_attempt_ids=set(),
        )

    observed: set[str] = set()
    validator._replay_trajectory(
        records, subject=subject, trajectory_index=0, environment_seed=seed,
        contract=contract, execution_contract=manifest["execution_contract"],
        global_attempt_ids=observed,
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="duplicated",
    ):
        validator._replay_trajectory(
            records, subject=subject, trajectory_index=0, environment_seed=seed,
            contract=contract, execution_contract=manifest["execution_contract"],
            global_attempt_ids=observed,
        )

    reference = manifest["journals"][f"{subject['target_id']}::0"]
    journal_path = Path(reference["path"])
    broken_chain = [dict(row) for row in records]
    next(row for row in broken_chain if row["event"] == "semantic_result")["action"] = "OPTION_B"
    journal_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in broken_chain),
        encoding="utf-8",
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="hash chain failed",
    ):
        validator._read_journal_reference(
            reference, expected_path=journal_path, label="tampered journal",
            tracker=validator._FileTracker(),
        )


def _first_successful_unit(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    semantic = next(row for row in records if row["event"] == "semantic_result")
    reservation = next(
        row
        for row in records
        if row["event"] == "reserved_before_dispatch"
        and row["attempt_id"] == semantic["attempt_id"]
    )
    return deepcopy(reservation), deepcopy(semantic)


def _transient_failure_for(reservation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": reservation["schema_version"],
        "artifact_type": reservation["artifact_type"],
        "event": "attempt_failed",
        "attempt_id": reservation["attempt_id"],
        "target_id": reservation["target_id"],
        "trajectory_index": reservation["trajectory_index"],
        "day": reservation["day"],
        "slot": reservation["slot"],
        "request_sha256": reservation["request_sha256"],
        "failure": {
            "failure_code": "http_503",
            "transient": True,
            "http_status": 503,
            "error_type": "FixtureTransientError",
        },
        "completed_at_utc": reservation["reserved_at_utc"],
    }


def test_journal_rejects_overlapping_and_over_ceiling_attempts(tmp_path: Path) -> None:
    records, subject, _contract, manifest = _small_replay_fixture(tmp_path)
    seed = manifest["common_environment_seeds"][0]
    first, semantic = _first_successful_unit(records)
    second = deepcopy(first)
    second["attempt_id"] = "part2_fixture_second_attempt"
    second["attempt_number"] = 2
    second_semantic = deepcopy(semantic)
    second_semantic["attempt_id"] = second["attempt_id"]
    second_semantic["attempt_number"] = 2

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="overlaps an active attempt",
    ):
        validator._validate_attempt_bindings(
            [first, second, _transient_failure_for(first), second_semantic],
            subject=subject,
            trajectory_index=0,
            environment_seed=seed,
            global_attempt_ids=set(),
            observable_dispatch_attempt_ceiling=1,
        )

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="exceeds its observable dispatch-attempt ceiling",
    ):
        validator._validate_attempt_bindings(
            [first, _transient_failure_for(first), second, second_semantic],
            subject=subject,
            trajectory_index=0,
            environment_seed=seed,
            global_attempt_ids=set(),
            observable_dispatch_attempt_ceiling=1,
        )


def test_journal_rejects_invalid_synthetic_ceiling_terminal_and_day_order(
    tmp_path: Path,
) -> None:
    records, subject, _contract, manifest = _small_replay_fixture(tmp_path)
    seed = manifest["common_environment_seeds"][0]
    first, semantic = _first_successful_unit(records)
    synthetic = deepcopy(first)
    synthetic["attempt_id"] = "part2_terminal_fixture"
    synthetic["attempt_number"] = 2
    synthetic["dispatch_skipped"] = True
    invalid_terminal = deepcopy(semantic)
    invalid_terminal["attempt_id"] = synthetic["attempt_id"]
    invalid_terminal["attempt_number"] = 2
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="synthetic exhausted-attempt terminal is inconsistent",
    ):
        validator._validate_attempt_bindings(
            [
                first,
                _transient_failure_for(first),
                synthetic,
                invalid_terminal,
            ],
            subject=subject,
            trajectory_index=0,
            environment_seed=seed,
            global_attempt_ids=set(),
            observable_dispatch_attempt_ceiling=1,
        )

    day_one = [deepcopy(row) for row in records if row.get("day") == 1]
    day_two = [deepcopy(row) for row in records if row.get("day") == 2]
    assert day_one and day_two
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="days are not in execution order",
    ):
        validator._validate_attempt_bindings(
            day_two + day_one,
            subject=subject,
            trajectory_index=0,
            environment_seed=seed,
            global_attempt_ids=set(),
            observable_dispatch_attempt_ceiling=1,
        )


def test_read_only_lock_check_rejects_active_run_without_touching_files(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    manifest_path = private / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    lock_path = private / ".run.lock"
    lock_path.touch(mode=0o600)
    before = (manifest_path.stat().st_mtime_ns, lock_path.stat().st_mtime_ns)

    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            validator.Part2OperationalOverlayValidationError, match="active",
        ):
            with validator._hold_run_locks([manifest_path]):
                raise AssertionError("active run must never yield a snapshot")

    assert before == (manifest_path.stat().st_mtime_ns, lock_path.stat().st_mtime_ns)


def test_incomplete_overlay_fails_before_journal_validation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "inference_hub_part2_operational_trajectory_repair_v1",
        "source_manifest": {},
        "panel_id": validator.EXPECTED_PANEL_ID,
        "part2_contract": {},
        "common_environment_seeds": [],
        "subject_routes": [],
        "repair_policy": "whole_trajectory_day_one_exact_route_separate_overlay",
        "maximum_rounds": 8,
        "created_at_utc": "2026-09-11T00:00:00Z",
        "last_updated_at_utc": "2026-09-11T00:00:00Z",
        "complete": False,
        "summary": {},
        "journals": {"unreadable": {"path": "/private/must-not-be-read"}},
        "sanitized_artifacts": {},
    }
    _seal(manifest)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="incomplete",
    ):
        validator._preflight_overlay(manifest_path, validator._FileTracker())


def test_complete_overlay_requires_ordered_canonical_utc_timestamps(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "artifact_type": "inference_hub_part2_operational_trajectory_repair_v1",
        "source_manifest": {},
        "panel_id": validator.EXPECTED_PANEL_ID,
        "part2_contract": {},
        "common_environment_seeds": list(
            validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS
        ),
        "subject_routes": [],
        "repair_policy": "whole_trajectory_day_one_exact_route_separate_overlay",
        "maximum_rounds": 8,
        "created_at_utc": "2026-09-11T00:00:02Z",
        "last_updated_at_utc": "2026-09-11T00:00:01Z",
        "completed_at_utc": "2026-09-11T00:00:03Z",
        "complete": True,
        "summary": {},
        "journals": {},
        "sanitized_artifacts": {},
    }
    _seal(manifest)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="timestamps are out of order",
    ):
        validator._preflight_overlay(manifest_path, validator._FileTracker())

    manifest["created_at_utc"] = "2026-09-11T00:00:00+00:00"
    _seal(manifest)
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="canonical UTC timestamp",
    ):
        validator._preflight_overlay(manifest_path, validator._FileTracker())


def _overlay_contract_fixture(
    *, main_shard: bool, maximum_rounds: int = 8,
    credential_pool: object = ...,
) -> tuple[validator._SourceEvidence, dict[str, Any]]:
    pair = _production_shaped_pairs()[0 if main_shard else 1]
    source_manifest_path = Path("/source/private/manifest.json")
    source_manifest_sha = "1" * 64
    source_evidence_sha = "2" * 64
    part2_contract = {
        "society_size": 50,
        "days": 100,
        "independent_trajectories": 12,
        "resource_capacity": 2500,
        "option_a_private_gain": 1,
        "option_b_private_gain": 2,
        "option_b_reserve_cost": 2,
        "unanimous_a_group_payoff": 5,
        "unanimous_b_group_payoff": -5,
        "invalid_policy": "retain_as_INVALID_zero_effect_no_semantic_retry",
        "collapse_death_rate": 0.2,
        "attrition_policy": "matched_seed_day_random_sample_v1",
    }
    source_manifest = {
        "evidence_sha256": source_evidence_sha,
        "panel_id": validator.EXPECTED_PANEL_ID,
        "part2_contract": part2_contract,
        "common_environment_seeds": list(
            validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS
        ),
        "subject_routes": list(pair.subjects),
        "execution_contract": deepcopy(
            validator.EXPECTED_SOURCE_EXECUTION_CONTRACT
        ),
        **_credential_source_manifest(),
    }
    source = validator._SourceEvidence(
        manifest_path=source_manifest_path,
        manifest_file_sha256=source_manifest_sha,
        manifest=source_manifest,
        subjects=pair.subjects,
        contract=pair.contract,
        environment_seeds=pair.environment_seeds,
        trajectories=(),
        models=(),
    )
    overlay = {
        "source_manifest": {
            "path": str(source_manifest_path),
            "file_sha256": source_manifest_sha,
            "evidence_sha256": source_evidence_sha,
        },
        "panel_id": validator.EXPECTED_PANEL_ID,
        "part2_contract": part2_contract,
        "common_environment_seeds": list(
            validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS
        ),
        "subject_routes": list(pair.subjects),
        "repair_policy": "whole_trajectory_day_one_exact_route_separate_overlay",
        "maximum_rounds": maximum_rounds,
    }
    if credential_pool is not ...:
        overlay["credential_pool"] = credential_pool
    return source, overlay


def test_overlay_requires_exact_max_rounds_and_main_pool_provenance() -> None:
    source, wrong_rounds = _overlay_contract_fixture(
        main_shard=True, maximum_rounds=7, credential_pool=_credential_pool_binding(),
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="maximum-round binding changed",
    ):
        validator._validate_overlay(
            Path("/overlay/private/manifest.json"),
            wrong_rounds,
            source=source,
            tracker=validator._FileTracker(),
            global_attempt_ids=set(),
        )

    source, missing_pool = _overlay_contract_fixture(main_shard=True)
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="lacks its bound credential-pool provenance",
    ):
        validator._validate_overlay(
            Path("/overlay/private/manifest.json"),
            missing_pool,
            source=source,
            tracker=validator._FileTracker(),
            global_attempt_ids=set(),
        )

    singleton_source, explicit_null_pool = _overlay_contract_fixture(
        main_shard=False, credential_pool=None,
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="Credential-pool provenance schema changed",
    ):
        validator._validate_overlay(
            Path("/overlay/private/manifest.json"),
            explicit_null_pool,
            source=singleton_source,
            tracker=validator._FileTracker(),
            global_attempt_ids=set(),
        )


def _cascading_manifest_fixture() -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "artifact_type": validator.cascading.ARTIFACT_TYPE,
        "source_manifest": {},
        "parent_overlay_manifest": {},
        "panel_id": validator.EXPECTED_PANEL_ID,
        "base_seed": validator.EXPECTED_BASE_SEED,
        "part2_contract": {},
        "common_environment_seeds": list(
            validator.EXPECTED_COMMON_ENVIRONMENT_SEEDS
        ),
        "subject_routes": [],
        "repair_policy": validator.cascading.REPAIR_POLICY,
        "selected_trajectory": {},
        "composition_contract": validator.cascading._composition_contract(),
        "maximum_rounds": 8,
        "execution_contract": validator.cascading._execution_contract(
            maximum_rounds=validator.cascading.DEFAULT_MAXIMUM_ROUNDS,
            trajectory_workers=validator.cascading.DEFAULT_TRAJECTORY_WORKERS,
            participant_workers=validator.cascading.DEFAULT_PARTICIPANT_WORKERS,
            max_attempts=validator.cascading.DEFAULT_MAX_ATTEMPTS,
            initial_backoff_seconds=validator.cascading.DEFAULT_BACKOFF_SECONDS,
            timeout_seconds=validator.cascading.DEFAULT_TIMEOUT_SECONDS,
            rate_profile=validator.cascading.DEFAULT_RATE_PROFILE,
        ),
        "credential_pool": {},
        "preflight_ledger": {},
        "created_at_utc": "2026-09-12T00:00:01Z",
        "last_updated_at_utc": "2026-09-12T00:00:02Z",
        "completed_at_utc": "2026-09-12T00:00:03Z",
        "complete": True,
        "summary": {},
        "journals": {},
        "sanitized_artifacts": {},
    }
    _seal(manifest)
    return manifest


def test_cascading_preflight_rejects_incomplete_or_secret_bearing_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    incomplete = _cascading_manifest_fixture()
    incomplete["complete"] = False
    incomplete.pop("completed_at_utc")
    _seal(incomplete)
    path.write_text(json.dumps(incomplete) + "\n", encoding="utf-8")
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="incomplete",
    ):
        validator._preflight_cascading_overlay(path, validator._FileTracker())

    exposed = _cascading_manifest_fixture()
    exposed["credential_pool"] = {"api_key": "must-not-persist"}
    _seal(exposed)
    path.write_text(json.dumps(exposed) + "\n", encoding="utf-8")
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError, match="secret-like field",
    ):
        validator._preflight_cascading_overlay(path, validator._FileTracker())


def test_cascading_preflight_ledger_rejects_symlink(tmp_path: Path) -> None:
    private_dir = tmp_path / "private"
    preflight_dir = private_dir / "preflight"
    preflight_dir.mkdir(parents=True)
    manifest_path = private_dir / "manifest.json"
    target_path = tmp_path / "preflight-target.json"
    target_path.write_text("{}\n", encoding="utf-8")
    target_path.chmod(0o600)
    ledger_path = preflight_dir / "preflight-000.json"
    ledger_path.symlink_to(target_path)
    reference = {
        "path": str(ledger_path),
        "file_sha256": validator._sha256_file(target_path),
        "evidence_sha256": "0" * 64,
    }

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="regular nonsymlink 0600 file",
    ):
        validator._validate_cascading_preflight_ledger(
            manifest_path, reference,
            exact_route="gcp/google/gemini-3.5-flash",
            tracker=validator._FileTracker(), resume_count=0,
        )


def _preflight_ledger_fixture(route: str) -> dict[str, Any]:
    slots = list(validator.cascading._account_slots(3))
    commitments = {
        slot: hashlib.sha256(f"preflight::{slot}".encode()).hexdigest()
        for slot in slots
    }
    results = [
        {
            "account_slot": slots[0],
            "account_commitment": commitments[slots[0]],
            "started_at_utc": "2026-09-12T00:00:01Z",
            "completed_at_utc": "2026-09-12T00:00:02Z",
            "authentication_succeeded": True,
            "catalog_http_status": 200,
            "catalog_failure_code": None,
            "exact_route_catalog_listed": True,
            "chat_attempted": True,
            "chat_http_status": 200,
            "chat_failure_code": None,
            "chat_response_identity_matched": True,
            "qualified": True,
        },
        {
            "account_slot": slots[1],
            "account_commitment": commitments[slots[1]],
            "started_at_utc": "2026-09-12T00:00:03Z",
            "completed_at_utc": "2026-09-12T00:00:04Z",
            "authentication_succeeded": False,
            "catalog_http_status": 401,
            "catalog_failure_code": "http_error",
            "exact_route_catalog_listed": False,
            "chat_attempted": False,
            "chat_http_status": None,
            "chat_failure_code": None,
            "chat_response_identity_matched": None,
            "qualified": False,
        },
        {
            "account_slot": slots[2],
            "account_commitment": commitments[slots[2]],
            "started_at_utc": "2026-09-12T00:00:05Z",
            "completed_at_utc": "2026-09-12T00:00:06Z",
            "authentication_succeeded": True,
            "catalog_http_status": 200,
            "catalog_failure_code": None,
            "exact_route_catalog_listed": True,
            "chat_attempted": True,
            "chat_http_status": 503,
            "chat_failure_code": "http_error",
            "chat_response_identity_matched": False,
            "qualified": False,
        },
    ]
    ledger = {
        "schema_version": 1,
        "artifact_type": validator.cascading.PREFLIGHT_ARTIFACT_TYPE,
        "created_at_utc": "2026-09-12T00:00:00Z",
        "completed_at_utc": "2026-09-12T00:00:07Z",
        "initial_completed_at_utc": "2026-09-12T00:00:07Z",
        "configured_account_slots": slots,
        "exact_route": route,
        "base_url": validator.runner.DEFAULT_BASE_URL,
        "request_contract": validator.cascading.PREFLIGHT_REQUEST_CONTRACT,
        "request_contract_sha256": validator._sha256_json(
            validator.cascading._preflight_request(route)
        ),
        "results": results,
        "qualified_account_slots": [slots[0]],
        "qualified_set_sha256": validator.cascading._qualified_set_sha256(
            [slots[0]], commitments,
        ),
        "cursor_epoch": CURSOR_EPOCH,
        "resume_preflight_sessions": [],
    }
    _reseal_preflight_ledger(ledger)
    return ledger


def _reseal_preflight_ledger(ledger: dict[str, Any]) -> None:
    initial_session = {
        "session_index": 0,
        "created_at_utc": ledger["created_at_utc"],
        "completed_at_utc": ledger["initial_completed_at_utc"],
        "results": ledger["results"],
        "qualified_account_slots": ledger["qualified_account_slots"],
        "qualified_set_sha256": ledger["qualified_set_sha256"],
        "starting_global_dispatch_ordinal": 0,
        "previous_session_sha256": None,
    }
    ledger["initial_session_sha256"] = (
        validator.cascading._preflight_session_sha256(initial_session)
    )
    _seal(ledger)


@pytest.mark.parametrize(
    "result_index,status_field,failure_field,status,failure_code",
    [
        (1, "catalog_http_status", "catalog_failure_code", 401, "connection_timeout"),
        (2, "chat_http_status", "chat_failure_code", None, "http_error"),
    ],
)
def test_cascading_preflight_rejects_impossible_failure_code_status_pairs(
    tmp_path: Path, result_index: int, status_field: str, failure_field: str,
    status: object, failure_code: str,
) -> None:
    route = "gcp/google/gemini-3.5-flash"
    ledger = _preflight_ledger_fixture(route)
    ledger["results"][result_index][status_field] = status
    ledger["results"][result_index][failure_field] = failure_code
    _reseal_preflight_ledger(ledger)
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="failure code/status pairing changed",
    ):
        validator._validate_cascading_preflight_failure_pair(
            status, failure_code, label="fixture preflight"
        )
    private_dir = tmp_path / "private"
    ledger_path = private_dir / "preflight" / "preflight-000.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    ledger_path.chmod(0o600)
    reference = {
        "path": str(ledger_path),
        "file_sha256": validator._sha256_file(ledger_path),
        "evidence_sha256": ledger["evidence_sha256"],
    }

    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="failure code/status pairing changed|Preflight .* failure is inconsistent",
    ):
        validator._validate_cascading_preflight_ledger(
            private_dir / "manifest.json", reference, exact_route=route,
            tracker=validator._FileTracker(), resume_count=0,
        )


def test_cascading_preflight_requires_chat_for_authenticated_listed_route(
    tmp_path: Path,
) -> None:
    route = "gcp/google/gemini-3.5-flash"
    ledger = _preflight_ledger_fixture(route)
    retained = ledger["results"][2]
    retained.update({
        "chat_http_status": 200,
        "chat_failure_code": None,
        "chat_response_identity_matched": True,
        "qualified": True,
    })
    impossible = ledger["results"][0]
    impossible.update({
        "chat_attempted": False,
        "chat_http_status": None,
        "chat_failure_code": None,
        "chat_response_identity_matched": None,
        "qualified": False,
    })
    ledger["qualified_account_slots"] = [retained["account_slot"]]
    commitments = {
        result["account_slot"]: result["account_commitment"]
        for result in ledger["results"]
    }
    ledger["qualified_set_sha256"] = (
        validator.cascading._qualified_set_sha256(
            ledger["qualified_account_slots"], commitments,
        )
    )
    _reseal_preflight_ledger(ledger)
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="chat-attempt control flow changed",
    ):
        validator._validate_cascading_preflight_control_flow(
            True, True, False, label="fixture preflight"
        )

    private_dir = tmp_path / "private"
    ledger_path = private_dir / "preflight" / "preflight-000.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps(ledger) + "\n", encoding="utf-8")
    ledger_path.chmod(0o600)
    reference = {
        "path": str(ledger_path),
        "file_sha256": validator._sha256_file(ledger_path),
        "evidence_sha256": ledger["evidence_sha256"],
    }
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="catalog success is inconsistent|chat-attempt control flow changed",
    ):
        validator._validate_cascading_preflight_ledger(
            private_dir / "manifest.json", reference, exact_route=route,
            tracker=validator._FileTracker(), resume_count=0,
        )


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("trajectory_workers",), 2),
        (("participant_workers",), 15),
        (("max_transport_attempts",), 7),
        (("initial_exponential_backoff_seconds",), 2.0),
        (("request_timeout_seconds",), 60.0),
        (("maximum_rounds",), 7),
        (("rate_profile",), "standard"),
        (("retry_policy", "http_statuses"), [429, *range(500, 600)]),
    ],
)
def test_cascading_execution_contract_binds_every_argument_and_http400_retry(
    path: tuple[str, ...], replacement: object,
) -> None:
    execution = validator.cascading._execution_contract(
        maximum_rounds=validator.cascading.DEFAULT_MAXIMUM_ROUNDS,
        trajectory_workers=validator.cascading.DEFAULT_TRAJECTORY_WORKERS,
        participant_workers=validator.cascading.DEFAULT_PARTICIPANT_WORKERS,
        max_attempts=validator.cascading.DEFAULT_MAX_ATTEMPTS,
        initial_backoff_seconds=validator.cascading.DEFAULT_BACKOFF_SECONDS,
        timeout_seconds=validator.cascading.DEFAULT_TIMEOUT_SECONDS,
        rate_profile=validator.cascading.DEFAULT_RATE_PROFILE,
    )
    target = execution
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="frozen exact design",
    ):
        validator._validate_cascading_execution_contract(execution)


def test_cascading_frozen_execution_constants_are_explicit() -> None:
    assert validator.EXPECTED_CASCADING_PARTICIPANT_WORKERS == 30
    assert validator.EXPECTED_CASCADING_CONFIGURED_ACCOUNT_COUNT == 3
    assert validator.cascading._execution_contract(
        maximum_rounds=validator.cascading.DEFAULT_MAXIMUM_ROUNDS,
        trajectory_workers=validator.cascading.DEFAULT_TRAJECTORY_WORKERS,
        participant_workers=validator.cascading.DEFAULT_PARTICIPANT_WORKERS,
        max_attempts=validator.cascading.DEFAULT_MAX_ATTEMPTS,
        initial_backoff_seconds=validator.cascading.DEFAULT_BACKOFF_SECONDS,
        timeout_seconds=validator.cascading.DEFAULT_TIMEOUT_SECONDS,
        rate_profile=validator.cascading.DEFAULT_RATE_PROFILE,
    ) == validator.EXPECTED_CASCADING_EXECUTION_CONTRACT
    assert (
        validator.cascading._composition_contract()
        == validator.EXPECTED_CASCADING_COMPOSITION_CONTRACT
    )


def _cascading_credential_pool_fixture() -> tuple[
    dict[str, Any], tuple[str, ...], dict[str, str], dict[str, Any]
]:
    slots = tuple(validator.cascading._account_slots(3))
    qualified = (slots[0], slots[2])
    commitments = {
        slot: hashlib.sha256(f"commitment::{slot}".encode()).hexdigest()
        for slot in slots
    }
    epoch = CURSOR_EPOCH
    cascading_path = Path(validator.cascading.__file__).resolve()
    runner_path = Path(validator.runner.__file__).resolve()
    repair_path = Path(validator.repair.__file__).resolve()
    pool = {
        "configured_account_count": 3,
        "configured_account_slots": list(slots),
        "account_slot_commitments": commitments,
        "qualified_account_count": 2,
        "qualified_account_slots": list(qualified),
        "rejected_account_slots": [slots[1]],
        "qualified_set_sha256": validator.cascading._qualified_set_sha256(
            qualified, commitments,
        ),
        "selection_policy": validator.cascading.SELECTION_POLICY,
        "rate_limit_scope": validator.cascading.RATE_LIMIT_SCOPE,
        "rate_limit_contract": deepcopy(
            validator.EXPECTED_SHARED_RATE_LIMIT_CONTRACT
        ),
        "qualified_account_rate_limiters": {
            slot: {
                "scope_label": f"{epoch}::{slot}",
                "state_path_sha256": hashlib.sha256(
                    f"limiter::{slot}".encode()
                ).hexdigest(),
                "rate_limit_contract": deepcopy(
                    validator.EXPECTED_SHARED_RATE_LIMIT_CONTRACT
                ),
            }
            for slot in qualified
        },
        "exact_route": "gcp/google/gemini-3.5-flash",
        "base_url": validator.runner.DEFAULT_BASE_URL,
        "cursor_epoch": epoch,
        "cursor_initial_global_dispatch_ordinal": 0,
        "implementation_files": {
            str(cascading_path): validator._sha256_file(cascading_path),
            str(runner_path): validator.EXPECTED_RUNNER_IMPLEMENTATION_SHA256,
            str(repair_path): validator.EXPECTED_REPAIR_IMPLEMENTATION_SHA256,
            str(cascading_path.with_name("inference_hub_discovery.py")):
                validator.EXPECTED_DISCOVERY_IMPLEMENTATION_SHA256,
            str(cascading_path.with_name("inference_hub_rate_limit.py")):
                validator.EXPECTED_RATE_LIMIT_IMPLEMENTATION_SHA256,
        },
    }
    source_manifest = {
        "source_artifacts": {
            str(runner_path): validator.EXPECTED_RUNNER_IMPLEMENTATION_SHA256,
        },
    }
    return pool, qualified, commitments, source_manifest


def test_cascading_credential_pool_binds_three_slots_and_independent_limiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_anonymous_archive_implementation_hashes(monkeypatch)
    pool, qualified, commitments, source_manifest = (
        _cascading_credential_pool_fixture()
    )
    current_hash = pool["implementation_files"][
        str(Path(validator.cascading.__file__).resolve())
    ]
    monkeypatch.setattr(
        validator, "EXPECTED_CASCADING_REPAIR_IMPLEMENTATION_SHA256",
        current_hash,
    )
    validator._validate_cascading_credential_pool(
        pool, qualified_slots=qualified,
        cursor_epoch=CURSOR_EPOCH,
        exact_route="gcp/google/gemini-3.5-flash",
        account_commitments=commitments,
        source_manifest=source_manifest, tracker=validator._FileTracker(),
    )

    duplicate_scope = deepcopy(pool)
    limiters = duplicate_scope["qualified_account_rate_limiters"]
    limiters[qualified[1]]["state_path_sha256"] = (
        limiters[qualified[0]]["state_path_sha256"]
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="independent limiter scopes",
    ):
        validator._validate_cascading_credential_pool(
            duplicate_scope, qualified_slots=qualified,
            cursor_epoch=CURSOR_EPOCH,
            exact_route="gcp/google/gemini-3.5-flash",
            account_commitments=commitments,
            source_manifest=source_manifest, tracker=validator._FileTracker(),
        )

    duplicate_credential = deepcopy(pool)
    duplicate_credential["account_slot_commitments"][
        "account-slot-02"
    ] = duplicate_credential["account_slot_commitments"]["account-slot-01"]
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="credential-pool provenance changed",
    ):
        validator._validate_cascading_credential_pool(
            duplicate_credential, qualified_slots=qualified,
            cursor_epoch=CURSOR_EPOCH,
            exact_route="gcp/google/gemini-3.5-flash",
            account_commitments=duplicate_credential[
                "account_slot_commitments"
            ],
            source_manifest=source_manifest, tracker=validator._FileTracker(),
        )


def _cascading_records(
    tmp_path: Path,
) -> tuple[list[dict[str, Any]], tuple[str, ...], str, dict[str, Any]]:
    records, _subject_row, _contract, _manifest = _small_replay_fixture(tmp_path)
    qualified = ("account-slot-01", "account-slot-03")
    epoch = CURSOR_EPOCH
    session_sha256 = "a" * 64
    ledger = {
        "initial_session_sha256": session_sha256,
        "created_at_utc": "2026-09-12T00:00:00Z",
        "initial_completed_at_utc": "2026-09-12T00:00:00Z",
        "resume_preflight_sessions": [],
    }
    ordinal = 0
    for row in records:
        if row["event"] == "reserved_before_dispatch":
            ordinal += 1
            row["global_dispatch_ordinal"] = ordinal
            row["account_slot"] = qualified[(ordinal - 1) % len(qualified)]
            row["cursor_epoch"] = epoch
            row["preflight_session_index"] = 0
            row["preflight_session_sha256"] = session_sha256
    return records, qualified, epoch, ledger


def test_cascading_dispatch_provenance_rejects_slot_ordinal_and_epoch_tamper(
    tmp_path: Path,
) -> None:
    records, qualified, epoch, ledger = _cascading_records(tmp_path)
    assert validator._validate_cascading_dispatch_provenance(
        records, qualified_slots=qualified, cursor_epoch=epoch,
        next_global_ordinal=0, max_attempts=8, preflight_ledger=ledger,
    ) == sum(row["event"] == "reserved_before_dispatch" for row in records)
    physical_count = sum(
        row["event"] == "reserved_before_dispatch" for row in records
    )
    validator._validate_cascading_dispatch_boundaries(
        ledger, final_global_ordinal=physical_count,
    )
    future_boundary = deepcopy(ledger)
    future_boundary["resume_preflight_sessions"] = [{
        "session_index": 1,
        "session_sha256": "b" * 64,
        "created_at_utc": "2026-09-12T01:00:00Z",
        "completed_at_utc": "2026-09-12T01:01:00Z",
        "starting_global_dispatch_ordinal": physical_count + 1,
    }]
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="future dispatch boundary",
    ):
        validator._validate_cascading_dispatch_boundaries(
            future_boundary, final_global_ordinal=physical_count,
        )

    mutators = (
        lambda row: row.__setitem__("account_slot", "account-slot-03"),
        lambda row: row.__setitem__("global_dispatch_ordinal", 2),
        lambda row: row.__setitem__(
            "cursor_epoch", "cursor-epoch-22222222222222222222222222222222"
        ),
    )
    for mutator in mutators:
        changed = deepcopy(records)
        reservation = next(
            row for row in changed if row["event"] == "reserved_before_dispatch"
        )
        mutator(reservation)
        with pytest.raises(
            validator.Part2OperationalOverlayValidationError,
            match="round-robin account dispatch provenance",
        ):
            validator._validate_cascading_dispatch_provenance(
                changed, qualified_slots=qualified, cursor_epoch=epoch,
                next_global_ordinal=0, max_attempts=8,
                preflight_ledger=ledger,
            )


def test_cascading_retry_policy_accepts_http400_and_rejects_hidden_401_retry(
    tmp_path: Path,
) -> None:
    records, qualified, epoch, ledger = _cascading_records(tmp_path)
    reservation = next(
        deepcopy(row) for row in records if row["event"] == "reserved_before_dispatch"
    )
    failed = {
        key: deepcopy(value)
        for key, value in next(
            row for row in records if row["event"] == "semantic_result"
        ).items()
        if key in validator._CASCADING_ATTEMPT_FAILED_KEYS
    }
    failed.update({
        "event": "attempt_failed",
        "attempt_id": reservation["attempt_id"],
        "day": reservation["day"],
        "slot": reservation["slot"],
        "request_sha256": reservation["request_sha256"],
        "failure": {
            "failure_code": "http_error",
            "transient": True,
            "http_status": 400,
            "error_type": "InferenceHubDiscoveryError",
        },
    })
    assert set(failed) == validator._CASCADING_ATTEMPT_FAILED_KEYS
    assert validator._validate_cascading_dispatch_provenance(
        [reservation, failed], qualified_slots=qualified, cursor_epoch=epoch,
        next_global_ordinal=0, max_attempts=8, preflight_ledger=ledger,
    ) == 1

    missing_error_type = deepcopy(failed)
    missing_error_type["failure"]["error_type"] = ""
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="exact bounded retry policy",
    ):
        validator._validate_cascading_dispatch_provenance(
            [reservation, missing_error_type], qualified_slots=qualified,
            cursor_epoch=epoch, next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
        )

    failed_401 = deepcopy(failed)
    failed_401["failure"]["http_status"] = 401
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="exact bounded retry policy",
    ):
        validator._validate_cascading_dispatch_provenance(
            [reservation, failed_401], qualified_slots=qualified,
            cursor_epoch=epoch, next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
        )


@pytest.mark.parametrize(
    "failure,expected",
    [
        ({"failure_code": "http_error", "http_status": 400}, True),
        ({"failure_code": "wrong_code", "http_status": 400}, False),
        ({"failure_code": "connection_error", "http_status": None}, True),
        ({"failure_code": "connection_error", "http_status": 400}, False),
    ],
)
def test_cascading_retry_code_status_pairs_are_exact(
    failure: dict[str, Any], expected: bool,
) -> None:
    assert validator._cascading_retryable_failure(failure) is expected


def test_cascading_session_authorization_rejects_hash_boundary_and_time_tamper(
    tmp_path: Path,
) -> None:
    records, qualified, epoch, ledger = _cascading_records(tmp_path)
    ledger["resume_preflight_sessions"] = [{
        "session_index": 1,
        "session_sha256": "b" * 64,
        "created_at_utc": "2026-09-12T01:00:00Z",
        "completed_at_utc": "2026-09-12T01:01:00Z",
        "starting_global_dispatch_ordinal": 1,
    }]
    assert validator._validate_cascading_cursor_epoch(CURSOR_EPOCH) == CURSOR_EPOCH
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="exact generated format",
    ):
        validator._validate_cascading_cursor_epoch("cursor-epoch-fixture")
    validator._validate_cascading_resume_session_times(
        ledger,
        child_created=validator._utc_datetime(
            "2026-09-11T23:59:59Z", label="fixture child creation"
        ),
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="predates child campaign creation",
    ):
        validator._validate_cascading_resume_session_times(
            ledger,
            child_created=validator._utc_datetime(
                "2026-09-12T01:00:01Z", label="fixture child creation"
            ),
        )
    for row in records:
        if row["event"] != "reserved_before_dispatch":
            continue
        if row["global_dispatch_ordinal"] == 1:
            row["reserved_at_utc"] = "2026-09-12T00:30:00Z"
        else:
            row["preflight_session_index"] = 1
            row["preflight_session_sha256"] = "b" * 64
            row["reserved_at_utc"] = "2026-09-12T02:00:00Z"
    reservation_ordinals = {
        row["attempt_id"]: row["global_dispatch_ordinal"]
        for row in records if row["event"] == "reserved_before_dispatch"
    }
    for row in records:
        if row["event"] in {"attempt_failed", "semantic_result"}:
            row["completed_at_utc"] = (
                "2026-09-12T00:40:00Z"
                if reservation_ordinals[row["attempt_id"]] == 1
                else "2026-09-12T02:10:00Z"
            )
    validator._validate_cascading_manifest_dispatch_times(
        records,
        child_created=validator._utc_datetime(
            "2026-09-11T23:59:59Z", label="fixture child creation"
        ),
        latest_session_index=1,
        latest_resume_at=validator._utc_datetime(
            "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
        ),
    )
    before_manifest = deepcopy(records)
    next(
        row for row in before_manifest
        if row["event"] == "reserved_before_dispatch"
    )["reserved_at_utc"] = "2026-09-11T23:59:58Z"
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="predates child manifest creation",
    ):
        validator._validate_cascading_manifest_dispatch_times(
            before_manifest,
            child_created=validator._utc_datetime(
                "2026-09-11T23:59:59Z", label="fixture child creation"
            ),
            latest_session_index=1,
            latest_resume_at=validator._utc_datetime(
                "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
            ),
        )
    before_resume_checkpoint = deepcopy(records)
    next(
        row for row in before_resume_checkpoint
        if row.get("preflight_session_index") == 1
    )["reserved_at_utc"] = "2026-09-12T01:15:00Z"
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="predates its resume manifest checkpoint",
    ):
        validator._validate_cascading_manifest_dispatch_times(
            before_resume_checkpoint,
            child_created=validator._utc_datetime(
                "2026-09-11T23:59:59Z", label="fixture child creation"
            ),
            latest_session_index=1,
            latest_resume_at=validator._utc_datetime(
                "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
            ),
        )
    assert validator._validate_cascading_dispatch_provenance(
        records, qualified_slots=qualified, cursor_epoch=epoch,
        next_global_ordinal=0, max_attempts=8, preflight_ledger=ledger,
    ) == sum(row["event"] == "reserved_before_dispatch" for row in records)

    wrong_hash = deepcopy(records)
    wrong_hash_reservation = next(
        row for row in wrong_hash
        if row.get("global_dispatch_ordinal") == 2
    )
    wrong_hash_reservation["preflight_session_sha256"] = "c" * 64
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="exact preflight session",
    ):
        validator._validate_cascading_dispatch_provenance(
            wrong_hash, qualified_slots=qualified, cursor_epoch=epoch,
            next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
        )

    wrong_boundary = deepcopy(records)
    wrong_boundary_reservation = next(
        row for row in wrong_boundary
        if row.get("global_dispatch_ordinal") == 2
    )
    wrong_boundary_reservation["preflight_session_index"] = 0
    wrong_boundary_reservation["preflight_session_sha256"] = "a" * 64
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="session boundary|wrong preflight session",
    ):
        validator._validate_cascading_dispatch_provenance(
            wrong_boundary, qualified_slots=qualified, cursor_epoch=epoch,
            next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
        )

    before_probe = deepcopy(records)
    before_probe_reservation = next(
        row for row in before_probe
        if row.get("global_dispatch_ordinal") == 2
    )
    before_probe_reservation["reserved_at_utc"] = "2026-09-12T01:00:30Z"
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="predates its authorizing preflight",
    ):
        validator._validate_cascading_dispatch_provenance(
            before_probe, qualified_slots=qualified, cursor_epoch=epoch,
            next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
        )


def _cascading_transport_semantic(
    semantic: dict[str, Any], *, attempt_id: str, attempt_number: int,
    failure: dict[str, Any],
) -> dict[str, Any]:
    row = deepcopy(semantic)
    row.update({
        "attempt_id": attempt_id,
        "attempt_number": attempt_number,
        "response_model": None,
        "model_identity_valid": False,
        "action": "INVALID",
        "reasoning": "",
        "invalid_reason": "transport_failure_exhausted",
        "format_valid": False,
        "visible_content": None,
        "visible_content_sha256": None,
        "request_id": None,
        "finish_reason": None,
        "usage": None,
        "raw_response": None,
        "raw_response_sha256": None,
        "failure": failure,
    })
    return row


def test_cascading_post_preflight_401_semantic_is_quarantined(
    tmp_path: Path,
) -> None:
    records, qualified, epoch, ledger = _cascading_records(tmp_path)
    reservation, semantic = _first_successful_unit(records)
    reservation["global_dispatch_ordinal"] = 1
    reservation["account_slot"] = qualified[0]
    reservation["cursor_epoch"] = epoch
    reservation["preflight_session_index"] = 0
    reservation["preflight_session_sha256"] = ledger["initial_session_sha256"]
    terminal = _cascading_transport_semantic(
        semantic, attempt_id=reservation["attempt_id"], attempt_number=1,
        failure={
            "failure_code": "http_error", "transient": False,
            "http_status": 401, "error_type": "InferenceHubDiscoveryError",
        },
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="invalidated a preflight-qualified account",
    ):
        validator._validate_cascading_dispatch_provenance(
            [reservation, terminal], qualified_slots=qualified,
            cursor_epoch=epoch, next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
        )

    mismatched_pair = deepcopy(terminal)
    mismatched_pair["failure"] = {
        "failure_code": "connection_error", "transient": False,
        "http_status": 400, "error_type": "InferenceHubDiscoveryError",
    }
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="terminal failure classification changed",
    ):
        validator._validate_cascading_dispatch_provenance(
            [reservation, mismatched_pair], qualified_slots=qualified,
            cursor_epoch=epoch, next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
        )

    empty_error_type = deepcopy(terminal)
    empty_error_type["failure"] = {
        "failure_code": "unexpected_client_error", "transient": False,
        "http_status": None, "error_type": "",
    }
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="terminal failure classification changed",
    ):
        validator._validate_cascading_dispatch_provenance(
            [reservation, empty_error_type], qualified_slots=qualified,
            cursor_epoch=epoch, next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
        )


def test_cascading_stale_attempt_eight_closes_only_with_synthetic_nine(
    tmp_path: Path,
) -> None:
    records, subject, _contract, manifest = _small_replay_fixture(tmp_path)
    base_reservation, base_semantic = _first_successful_unit(records)
    seed = manifest["common_environment_seeds"][0]
    qualified = ("account-slot-01", "account-slot-03")
    epoch = CURSOR_EPOCH
    session_sha256 = "a" * 64
    ledger = {
        "initial_session_sha256": session_sha256,
        "created_at_utc": "2026-09-12T00:00:00Z",
        "initial_completed_at_utc": "2026-09-12T00:00:00Z",
        "resume_preflight_sessions": [{
            "session_index": 1,
            "session_sha256": "b" * 64,
            "created_at_utc": "2026-09-12T01:00:00Z",
            "completed_at_utc": "2026-09-12T01:01:00Z",
            "starting_global_dispatch_ordinal": 8,
        }],
    }
    crash_records: list[dict[str, Any]] = []
    for attempt_number in range(1, 9):
        reservation = deepcopy(base_reservation)
        reservation_minute = 8 + 2 * attempt_number
        reservation.update({
            "attempt_id": f"v4-crash-{attempt_number}",
            "attempt_number": attempt_number,
            "global_dispatch_ordinal": attempt_number,
            "account_slot": qualified[(attempt_number - 1) % len(qualified)],
            "cursor_epoch": epoch,
            "preflight_session_index": 0,
            "preflight_session_sha256": session_sha256,
            "reserved_at_utc": (
                f"2026-09-12T00:{reservation_minute:02d}:00Z"
            ),
        })
        crash_records.append(reservation)
        failed = {
            key: deepcopy(value)
            for key, value in base_semantic.items()
            if key in validator._CASCADING_ATTEMPT_FAILED_KEYS
        }
        failed.update({
            "event": "attempt_failed",
            "attempt_id": reservation["attempt_id"],
            "request_sha256": reservation["request_sha256"],
            "failure": (
                {
                    "failure_code": "stale_reserved_attempt",
                    "transient": True,
                    "http_status": None,
                }
                if attempt_number == 8 else {
                    "failure_code": "http_error",
                    "transient": True,
                    "http_status": 400,
                    "error_type": "InferenceHubDiscoveryError",
                }
            ),
            "completed_at_utc": (
                "2026-09-12T02:00:00Z"
                if attempt_number == 8
                else f"2026-09-12T00:{reservation_minute + 1:02d}:00Z"
            ),
        })
        crash_records.append(failed)

    synthetic_reservation = deepcopy(base_reservation)
    synthetic_reservation.update({
        "attempt_id": "v4-crash-synthetic-9",
        "attempt_number": 9,
        "global_dispatch_ordinal": None,
        "account_slot": None,
        "cursor_epoch": epoch,
        "preflight_session_index": 1,
        "preflight_session_sha256": "b" * 64,
        "reserved_at_utc": "2026-09-12T02:00:01Z",
        "dispatch_skipped": True,
    })
    crash_records.append(synthetic_reservation)
    synthetic = _cascading_transport_semantic(
        base_semantic, attempt_id=synthetic_reservation["attempt_id"],
        attempt_number=9,
        failure={
            "failure_code": "resume_attempt_budget_exhausted",
            "transient": False, "http_status": None, "error_type": None,
        },
    )
    synthetic["completed_at_utc"] = "2026-09-12T02:00:02Z"
    crash_records.append(synthetic)

    assert validator._validate_cascading_dispatch_provenance(
        crash_records, qualified_slots=qualified, cursor_epoch=epoch,
        next_global_ordinal=0, max_attempts=8, preflight_ledger=ledger,
        latest_resume_at=validator._utc_datetime(
            "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
        ),
    ) == 8
    semantics = validator._validate_attempt_bindings(
        crash_records, subject=subject, trajectory_index=0,
        environment_seed=seed, global_attempt_ids=set(),
        observable_dispatch_attempt_ceiling=8,
    )
    assert semantics[(synthetic["day"], synthetic["slot"])] == synthetic

    backdated_retry = deepcopy(crash_records)
    backdated_retry[2]["reserved_at_utc"] = "2026-09-12T00:10:30Z"
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="retry reservation predates its failed predecessor",
    ):
        validator._validate_cascading_dispatch_provenance(
            backdated_retry, qualified_slots=qualified, cursor_epoch=epoch,
            next_global_ordinal=0, max_attempts=8, preflight_ledger=ledger,
            latest_resume_at=validator._utc_datetime(
                "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
            ),
        )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="retry reservation predates its failed predecessor",
    ):
        validator._validate_attempt_bindings(
            backdated_retry, subject=subject, trajectory_index=0,
            environment_seed=seed, global_attempt_ids=set(),
            observable_dispatch_attempt_ceiling=8,
        )

    second_resume_ledger = deepcopy(ledger)
    second_resume_ledger["resume_preflight_sessions"].append({
        "session_index": 2,
        "session_sha256": "c" * 64,
        "created_at_utc": "2026-09-12T03:00:00Z",
        "completed_at_utc": "2026-09-12T03:01:00Z",
        "starting_global_dispatch_ordinal": 8,
    })
    resumed_synthetic = deepcopy(crash_records)
    resumed_synthetic[-1]["completed_at_utc"] = "2026-09-12T04:00:00Z"
    assert validator._validate_cascading_dispatch_provenance(
        resumed_synthetic, qualified_slots=qualified, cursor_epoch=epoch,
        next_global_ordinal=0, max_attempts=8,
        preflight_ledger=second_resume_ledger,
        latest_resume_at=validator._utc_datetime(
            "2026-09-12T03:30:00Z", label="fixture resume checkpoint"
        ),
    ) == 8

    stale_before_checkpoint = deepcopy(crash_records)
    stale_before_checkpoint[15]["completed_at_utc"] = "2026-09-12T01:15:00Z"
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="stale recovery predates its manifest checkpoint",
    ):
        validator._validate_cascading_dispatch_provenance(
            stale_before_checkpoint, qualified_slots=qualified,
            cursor_epoch=epoch, next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
            latest_resume_at=validator._utc_datetime(
                "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
            ),
        )

    synthetic_before_checkpoint = deepcopy(resumed_synthetic)
    synthetic_before_checkpoint[-1]["completed_at_utc"] = (
        "2026-09-12T03:15:00Z"
    )
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="synthetic recovery predates its manifest checkpoint",
    ):
        validator._validate_cascading_dispatch_provenance(
            synthetic_before_checkpoint, qualified_slots=qualified,
            cursor_epoch=epoch, next_global_ordinal=0, max_attempts=8,
            preflight_ledger=second_resume_ledger,
            latest_resume_at=validator._utc_datetime(
                "2026-09-12T03:30:00Z", label="fixture resume checkpoint"
            ),
        )

    invalid = deepcopy(crash_records)
    invalid[-2]["dispatch_skipped"] = False
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="reservation schema|dispatch-attempt ceiling",
    ):
        validator._validate_cascading_dispatch_provenance(
            invalid, qualified_slots=qualified, cursor_epoch=epoch,
            next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
            latest_resume_at=validator._utc_datetime(
                "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
            ),
        )

    attempt_ten = deepcopy(crash_records)
    attempt_ten[-2]["attempt_number"] = 10
    attempt_ten[-1]["attempt_number"] = 10
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="exact dispatch-attempt ceiling",
    ):
        validator._validate_cascading_dispatch_provenance(
            attempt_ten, qualified_slots=qualified, cursor_epoch=epoch,
            next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
            latest_resume_at=validator._utc_datetime(
                "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
            ),
        )

    stale_skipped = deepcopy(crash_records)
    stale_skipped[-1] = {
        key: deepcopy(value)
        for key, value in crash_records[-1].items()
        if key in validator._CASCADING_ATTEMPT_FAILED_KEYS
    }
    stale_skipped[-1].update({
        "event": "attempt_failed",
        "attempt_id": stale_skipped[-2]["attempt_id"],
        "request_sha256": stale_skipped[-2]["request_sha256"],
        "failure": {
            "failure_code": "stale_reserved_attempt",
            "transient": True,
            "http_status": None,
        },
    })
    with pytest.raises(
        validator.Part2OperationalOverlayValidationError,
        match="stale reservation classification changed",
    ):
        validator._validate_cascading_dispatch_provenance(
            stale_skipped, qualified_slots=qualified, cursor_epoch=epoch,
            next_global_ordinal=0, max_attempts=8,
            preflight_ledger=ledger,
            latest_resume_at=validator._utc_datetime(
                "2026-09-12T01:30:00Z", label="fixture resume checkpoint"
            ),
        )


@pytest.mark.skipif(
    not all(path.is_file() for pair in PRODUCTION_PAIRS for path in pair),
    reason="Private production source/overlay evidence is intentionally not distributed.",
)
def test_production_three_pair_gate_replays_complete_union() -> None:
    result = validator.validate_operational_overlay_pairs(PRODUCTION_PAIRS)

    assert result["status"] == "passed"
    assert result["route_count"] == 23
    assert result["trajectory_count"] == 276
    assert result["common_environment_seed_count"] == 12


@pytest.mark.skipif(
    not all(path.is_file() for path in PRODUCTION_PAIRS[2]),
    reason="Private completed DeepSeek source/overlay evidence is intentionally not distributed.",
)
def test_completed_production_pair_replays_and_reconciles_end_to_end() -> None:
    source_path, overlay_path = (path.resolve() for path in PRODUCTION_PAIRS[2])
    tracker = validator._FileTracker()
    attempt_ids: set[str] = set()
    with validator._hold_run_locks([overlay_path, source_path]):
        overlay_manifest = validator._preflight_overlay(overlay_path, tracker)
        source_manifest = validator._safe_json(
            source_path, label="source manifest", tracker=tracker,
        )
        source = validator._validate_source(
            source_path, source_manifest, tracker=tracker,
            global_attempt_ids=attempt_ids, source_verification_root=None,
        )
        pair = validator._validate_overlay(
            overlay_path, overlay_manifest, source=source, tracker=tracker,
            global_attempt_ids=attempt_ids,
        )
        tracker.verify()

    assert len(pair.subjects) == 1
    assert len(pair.effective_trajectories) == 12
    assert pair.source_operational_failure_count == 9
    assert pair.repair_round_count == 10
    assert all(row["operationally_eligible"] is True for row in pair.effective_trajectories)
