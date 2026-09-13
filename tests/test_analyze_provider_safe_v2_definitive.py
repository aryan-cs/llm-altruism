from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

import analysis.analyze_provider_safe_v2_definitive as definitive
from analysis.analyze_provider_safe_v2_definitive import (
    DefinitiveAnalysisError,
    PART2_100DAY_CONTRACT,
    PART2_100DAY_DECLARED_EXCLUSION,
    PART2_100DAY_ORDERED_TARGET_IDS,
    PART2_EFFECTIVE_MODEL_TYPE,
    PART2_EFFECTIVE_TRAJECTORY_ROW_KEYS,
    PART2_EFFECTIVE_TRAJECTORY_TYPE,
    PART2_OPERATIONAL_REPAIR_TYPE,
    PART2_TRAJECTORY_ROW_KEYS,
    SENSITIVITY_DIAGNOSTIC_FAMILY,
    SENSITIVITY_HOLM_FAMILY,
    SENSITIVITY_HOLM_FAMILY_SIZE,
    _analyze_deadline_sensitivity,
    _provider_safe_contract,
    _mean_t_95,
    _parser,
    _part2_composition,
    _replay_part2_source_journal,
    _root_cluster_bootstrap_95,
    _self_hash,
    _stratified_root_bootstrap_95,
    _validate_role_journals,
    _wilson_95,
    analyze,
)
from experiments.misc import inference_hub_part2_panel as part2_panel
from experiments.misc.inference_hub_part2_sensitivity_v1 import load_sensitivity_design
from experiments.part1.confirmatory_design import COUNTERBALANCES, DOMAINS, GAMES


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "experiments/part2/part2_sensitivity_deadline_exploratory_v2.json"
JUDGE = {
    "target_id": "judge.nemotron", "route": "judge/nemotron",
    "upstream_provider": "judge-provider", "model": "nemotron-judge",
    "dispatch_permitted_in_this_runner": False,
}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = _self_hash(value)
    return value


def _route(index: int, prefix: str) -> dict[str, Any]:
    return {
        "target_id": f"{prefix}/model-{index:03d}",
        "route": f"route/{prefix}/model-{index:03d}",
        "upstream_provider": f"provider-{index:03d}",
        "model": f"model-{index:03d}",
    }


def _write_json(path: Path, value: Mapping[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    if private:
        os.chmod(path, 0o600)


def _journal(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    encoded = []
    for payload in rows:
        row = {**payload, "previous_record_sha256": previous}
        row["record_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        previous = row["record_sha256"]
        encoded.append(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    path.write_text("".join(encoded), encoding="utf-8")
    os.chmod(path, 0o600)
    return {
        "path": str(path.resolve()), "record_count": len(rows),
        "tail_record_sha256": previous, "file_sha256": _sha_file(path),
    }


class _MemoryChainedJournal:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        previous = self.records[-1]["record_sha256"] if self.records else None
        row = {**payload, "previous_record_sha256": previous}
        row["record_sha256"] = _canonical_sha(row)
        self.records.append(row)
        return row


class _CompositionTrajectoryClient:
    def __init__(self, *, semantic_invalid_first: bool) -> None:
        self.semantic_invalid_first = semantic_invalid_first
        self.calls = 0

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        self.calls += 1
        content = (
            "not-json"
            if self.semantic_invalid_first and self.calls == 1
            else json.dumps({"action": "OPTION_B", "reasoning": "Use reserve."})
        )
        return {
            "id": f"composition-{self.calls}",
            "model": body["model"],
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
            },
        }


def _persist_memory_journal(
    path: Path, journal: _MemoryChainedJournal
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in journal.records
        ),
        encoding="utf-8",
    )
    os.chmod(path, 0o600)
    return {
        "path": str(path.resolve()),
        "record_count": len(journal.records),
        "tail_record_sha256": (
            journal.records[-1]["record_sha256"] if journal.records else None
        ),
        "file_sha256": _sha_file(path),
    }


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _role_overlay_fixture(
    run: Path, *, replace_visible_semantic_invalid: bool = False
) -> tuple[dict[str, Any], Path, Path]:
    target = "role/model"
    route = "route/role/model"
    subject = {
        "target_id": target,
        "route": route,
        "upstream_provider": "provider",
        "model": "model",
    }
    visible_raw = {"model": route, "choices": [{"message": {"content": "OPTION_X"}}]}
    original_rows = [
        {
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_role_calibration_raw_response_v1",
            "target_id": target,
            "requested_route": route,
            "trial_id": "visible-invalid",
            "root_id": "root-visible",
            "frame_id": "advice",
            "generation_block": "block-0",
            "counterbalance_id": "cb-0",
            "prompt_sha256": "a" * 64,
            "request_sha256": "b" * 64,
            "raw_response": visible_raw,
            "raw_response_sha256": _canonical_sha(visible_raw),
            "model_identity_valid": True,
            "format_valid": False,
            "welfare_preserving": None,
        },
        {
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_role_calibration_raw_response_v1",
            "target_id": target,
            "requested_route": route,
            "trial_id": "transport-null",
            "root_id": "root-null",
            "frame_id": "prediction",
            "generation_block": "block-0",
            "counterbalance_id": "cb-1",
            "prompt_sha256": "c" * 64,
            "request_sha256": "d" * 64,
            "raw_response": None,
            "raw_response_sha256": None,
            "model_identity_valid": False,
            "format_valid": False,
            "welfare_preserving": None,
        },
    ]
    original_path = run / "private/raw/role.jsonl"
    original_ref = _journal(original_path, original_rows)
    retained = [json.loads(line) for line in original_path.read_text().splitlines()]
    selected = retained[0 if replace_visible_semantic_invalid else 1]

    repaired_raw = {
        "model": route,
        "choices": [{"message": {"content": "OPTION_A"}}],
    }
    attempt_id = "repair-attempt-1"
    repair_binding = {
        "repair_reason": "original_retained_raw_response_null",
        "replaces_original_record_sha256": selected["record_sha256"],
    }
    repair_row = {
        "schema_version": 1,
        "artifact_type": (
            "inference_hub_part1_role_calibration_operational_repair_response_v1"
        ),
        "target_id": target,
        "requested_route": route,
        "response_model": route,
        "model_identity_valid": True,
        "trial_id": selected["trial_id"],
        "root_id": selected["root_id"],
        "frame_id": selected["frame_id"],
        "generation_block": selected["generation_block"],
        "counterbalance_id": selected["counterbalance_id"],
        "prompt_sha256": selected["prompt_sha256"],
        "request_sha256": selected["request_sha256"],
        "attempt_id": attempt_id,
        **repair_binding,
        "raw_response": repaired_raw,
        "raw_response_sha256": _canonical_sha(repaired_raw),
        "response_text_sha256": "e" * 64,
        "format_valid": True,
        "welfare_preserving": True,
    }
    repair_path = run / "private/operational_repairs/raw/role.jsonl"
    repair_ref = _journal(repair_path, [repair_row])
    reservation = {
        "schema_version": 1,
        "artifact_type": (
            "inference_hub_part1_role_calibration_operational_repair_attempt_v1"
        ),
        "event": "reserved_before_dispatch",
        "attempt_id": attempt_id,
        "target_id": target,
        "trial_id": selected["trial_id"],
        "request_sha256": selected["request_sha256"],
        **repair_binding,
    }
    completion = {
        "schema_version": 1,
        "artifact_type": (
            "inference_hub_part1_role_calibration_operational_repair_attempt_v1"
        ),
        "event": "attempt_completed",
        "attempt_id": attempt_id,
        "outcome": "response_retained",
        "response_payload_sha256": repair_row["raw_response_sha256"],
        "response_text_sha256": repair_row["response_text_sha256"],
        "response_model": route,
        **repair_binding,
    }
    repair_ledger = _journal(
        run / "private/operational_repairs/attempt_ledger.jsonl",
        [reservation, completion],
    )
    original_ledger = _journal(run / "private/attempts.jsonl", [])
    repair_source = ROOT / "experiments/misc/inference_hub_part1_role_calibration_v1.py"
    manifest = {
        "subject_routes": [subject],
        "summary": {
            "operational_repair_eligible_originals": 1,
            "operational_repairs_succeeded": 1,
            "operational_repairs_unresolved": 0,
            "failed_without_response": 0,
        },
        "journals": {
            "attempt_ledger": original_ledger,
            "raw_responses": {target: original_ref},
            "operational_repair_overlay": {
                "policy": (
                    "overlay_only_original_retained_raw_response_null_"
                    "no_semantic_retry_v1"
                ),
                "implementation_source_artifacts": {
                    str(repair_source.resolve()): _sha_file(repair_source)
                },
                "attempt_ledger": repair_ledger,
                "raw_responses": {target: repair_ref},
            },
        },
    }
    return manifest, original_path, repair_path


def _base_manifest(kind: str, subjects: list[dict[str, Any]]) -> dict[str, Any]:
    shared = {
        "schema_version": 2,
        "algorithm": "cross_process_provider_aware_leaky_bucket_with_leases_all_http_5xx_full_throttle_cooldown",
        "global_concurrency": 16, "provider_concurrency": 1,
        "global_requests_per_second": 8.0, "provider_requests_per_second": 1.0,
        "lease_seconds": 900.0, "poll_seconds": 0.05,
        "throttle_cooldown_seconds": 30.0, "transient_cooldown_seconds": 5.0,
    }
    shared["policy_sha256"] = hashlib.sha256(
        json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1, "artifact_type": kind, "complete": True,
        "completed_at_utc": "2026-08-03T12:00:00Z", "subject_routes": subjects,
        "judge_reservation": dict(JUDGE),
        "execution_contract": {"shared_rate_limit": shared},
        "source_artifacts": {str(ROOT / "experiments/misc/inference_hub_provider_safe_v2.py"): "a" * 64},
    }


def _artifact(run: Path, name: str, artifact_type: str, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    payload = _seal({"schema_version": 1, "artifact_type": artifact_type, "rows": rows})
    path = run / "sanitized" / f"{name}.json"
    _write_json(path, payload)
    manifest.setdefault("sanitized_artifacts", {})[name] = {
        "path": str(path.resolve()), "file_sha256": _sha_file(path),
        "evidence_sha256": payload["evidence_sha256"],
    }


def _composition_subject(target_id: str, index: int) -> dict[str, Any]:
    provider, model = target_id.split("/", 1)
    return {
        "target_id": target_id,
        "upstream_provider": provider,
        "model": model,
        "route": f"fixture-route/{target_id}",
        "candidate_index": 0,
        "supported_controls": ["seed", "temperature", "top_p"],
        "selected_profile_id": "profile_00_seed_temperature_top_p",
        "selected_profile_request_sha256": hashlib.sha256(
            target_id.encode("utf-8")
        ).hexdigest(),
    }


def _composition_trajectory(
    subject: Mapping[str, Any],
    trajectory_index: int,
    environment_seed: int,
    *,
    transport_failure: bool = False,
    semantic_invalid: bool = False,
) -> dict[str, Any]:
    invalid = int(transport_failure or semantic_invalid)
    restraint = 3_999 if invalid else 4_000
    return {
        "schema_version": 1,
        "target_id": subject["target_id"],
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "trajectory_index": trajectory_index,
        "environment_seed_index": trajectory_index,
        "environment_seed": environment_seed,
        "operationally_eligible": not transport_failure,
        "scheduled_agent_days": 5_000,
        "responses_received": 4_999 if transport_failure else 5_000,
        "invalid_count": invalid,
        "identity_mismatch_count": 0,
        "transport_failure_count": int(transport_failure),
        "restraint_count": restraint,
        "overuse_count": 1_000,
        "restraint_rate": restraint / 5_000,
        "aurc": 0.8,
        "aupc": 0.9,
        "reserve_nondepletion": True,
        "final_reserve": 500,
        "final_population": 40,
        "population_retention": 0.8,
        "cumulative_private_payoff": 6_000,
        "cumulative_group_payoff": 0,
    }


def _composition_payload(
    run: Path,
    name: str,
    artifact_type: str,
    rows: list[dict[str, Any]],
    *,
    source_evidence: str | None = None,
) -> tuple[dict[str, Any], Path]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "panel_id": "sota_cross_axis_part2_corrected_original_scale_100d_v1",
        "generated_at_utc": "2026-08-22T12:00:00Z",
        "rows": rows,
    }
    if artifact_type == "inference_hub_part2_sanitized_trajectory_metrics":
        payload["independence_unit"] = "target_by_environment_seed_trajectory"
    elif artifact_type == "inference_hub_part2_sanitized_model_metrics":
        payload["uncertainty_unit"] = "independent_trajectory"
    if source_evidence is not None:
        payload["source_manifest_evidence_sha256"] = source_evidence
    _seal(payload)
    path = run / "sanitized" / f"{name}.json"
    _write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha_file(path),
        "evidence_sha256": payload["evidence_sha256"],
    }, path


def _composition_fixture(
    root: Path,
) -> list[tuple[Path, Path]]:
    panel_path = ROOT / "experiments/sota_cross_axis_part2_100day_panel.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_ref = {
        "path": str(panel_path.resolve()),
        "file_sha256": _sha_file(panel_path),
        "canonical_sha256": _canonical_sha(panel),
    }
    source_artifacts = {
        str(path.resolve()): _sha_file(path.resolve())
        for path in part2_panel._SOURCE_PATHS
    }
    common_seeds = part2_panel._environment_seeds(
        "sota_cross_axis_part2_corrected_original_scale_100d_v1",
        20_260_802,
        12,
    )
    panel_targets = list(panel["subject_target_ids"])
    composition_judge = {
        "target_id": panel["judge_target_id"],
        "upstream_provider": "nvidia",
        "model": "nvidia/evals-nemotron-3-30b-a3b",
        "route": "nvidia/nvidia/evals-nemotron-3-30b-a3b",
        "dispatch_permitted_in_this_runner": False,
        "role": "fixed_disjoint_judge_reserved_for_cross_axis_analysis",
    }
    all_routes = [_composition_subject(target, 0) for target in panel_targets]
    all_routes.append(
        {
            "target_id": composition_judge["target_id"],
            "upstream_provider": composition_judge["upstream_provider"],
            "model": composition_judge["model"],
            "route": composition_judge["route"],
            "candidate_index": 0,
            "supported_controls": ["seed", "temperature", "top_p"],
            "selected_profile_id": "profile_00_seed_temperature_top_p",
            "selected_profile_request_sha256": hashlib.sha256(
                composition_judge["target_id"].encode("utf-8")
            ).hexdigest(),
        }
    )
    registry = {
        "schema_version": 1,
        "artifact_type": "composition_test_registry",
        "targets": [
            {
                "id": route["target_id"],
                "provider": "inference_hub",
                "upstream_provider": route["upstream_provider"],
                "model": route["model"],
            }
            for route in all_routes
        ],
    }
    registry_path = root / "inputs/registry.json"
    _write_json(registry_path, registry)
    registry_ref = {
        "path": str(registry_path.resolve()),
        "file_sha256": _sha_file(registry_path),
        "canonical_sha256": _canonical_sha(registry),
    }
    compatibility_targets = []
    for route in all_routes:
        profile = {
            "route": route["route"],
            "status": "passed",
            "profile_id": route["selected_profile_id"],
            "controls": route["supported_controls"],
            "request_sha256": route["selected_profile_request_sha256"],
            "validation_source": "execution_profile_probe",
        }
        compatibility_targets.append(
            {
                "target_id": route["target_id"],
                "model": route["model"],
                "status": "execution_candidate_selected",
                "selected_execution_candidate": route["route"],
                "selected_execution_profile": profile,
                "candidates": [
                    {
                        "route": route["route"],
                        "execution_compatible": True,
                        "candidate_index": route["candidate_index"],
                        "max_tokens": 2_048,
                        "selected_execution_profile": profile,
                    }
                ],
            }
        )
    compatibility = _seal(
        {
            "schema_version": 2,
            "artifact_type": "inference_hub_provider_compatibility",
            "registry_sha256": _canonical_sha(registry),
            "target_count": len(compatibility_targets),
            "selected_count": len(compatibility_targets),
            "unresolved_count": 0,
            "targets": compatibility_targets,
        }
    )
    compatibility_path = root / "inputs/compatibility.json"
    _write_json(compatibility_path, compatibility)
    compatibility_ref = {
        "path": str(compatibility_path.resolve()),
        "file_sha256": _sha_file(compatibility_path),
        "evidence_sha256": compatibility["evidence_sha256"],
    }
    main_targets = [
        target
        for target in panel_targets
        if target
        not in {
            PART2_100DAY_DECLARED_EXCLUSION,
            "nvidia/nemotron-3-ultra",
            "deepseek-ai/deepseek-v4-flash",
        }
    ]
    pair_targets = [
        main_targets,
        ["nvidia/nemotron-3-ultra"],
        ["deepseek-ai/deepseek-v4-flash"],
    ]
    shared_rate = {
        "schema_version": 2,
        "algorithm": (
            "cross_process_provider_aware_leaky_bucket_with_leases_all_http_"
            "5xx_full_throttle_cooldown"
        ),
        "global_concurrency": 60,
        "provider_concurrency": 10,
        "global_requests_per_second": 12.0,
        "provider_requests_per_second": 2.5,
        "lease_seconds": 900.0,
        "poll_seconds": 0.05,
        "throttle_cooldown_seconds": 30.0,
        "transient_cooldown_seconds": 5.0,
    }
    shared_rate["policy_sha256"] = _canonical_sha(shared_rate)

    result: list[tuple[Path, Path]] = []
    for pair_index, targets in enumerate(pair_targets):
        source_run = root / f"source-{pair_index}"
        overlay_run = root / f"overlay-{pair_index}"
        subjects = [
            _composition_subject(target, index)
            for index, target in enumerate(targets)
        ]
        source_rows: list[dict[str, Any]] = []
        source_journals: dict[str, Any] = {}
        for subject_index, subject in enumerate(subjects):
            for trajectory_index, environment_seed in enumerate(common_seeds):
                failed = subject_index == 0 and trajectory_index == 0
                semantic_invalid = (
                    (pair_index == 2 and not failed)
                    or (
                        pair_index == 1
                        and trajectory_index in {1, 2, 3}
                    )
                )
                source_rows.append(
                    _composition_trajectory(
                        subject,
                        trajectory_index,
                        environment_seed,
                        transport_failure=failed,
                        semantic_invalid=semantic_invalid,
                    )
                )
                key = f"{subject['target_id']}::{trajectory_index}"
                source_journals[key] = _journal(
                    source_run
                    / "private/trajectories"
                    / f"{subject_index}-{trajectory_index}.jsonl",
                    [],
                )
        source_models = part2_panel._aggregate_models(
            source_rows,
            subjects,
            expected_trajectories=12,
            capacity=2_500,
        )
        source_manifest: dict[str, Any] = {
            "schema_version": 1,
            "artifact_type": "inference_hub_part2_corrected_matched_panel",
            "created_at_utc": "2026-08-20T12:00:00Z",
            "panel_id": "sota_cross_axis_part2_corrected_original_scale_100d_v1",
            "input_artifacts": {
                "panel": panel_ref,
                "compatibility": compatibility_ref,
                "registry": registry_ref,
            },
            "source_artifacts": source_artifacts,
            "subject_routes": subjects,
            "judge_reservation": composition_judge,
            "part2_contract": dict(PART2_100DAY_CONTRACT),
            "base_seed": 20_260_802,
            "common_environment_seeds": common_seeds,
            "execution_contract": {
                "strategy": (
                    "parallel_target_trajectory_and_parallel_participants_"
                    "with_sequential_days"
                ),
                "trajectory_workers": 4,
                "participant_workers": 4,
                "max_transport_attempts": 3,
                "initial_exponential_backoff_seconds": 1.0,
                "shared_rate_limit": shared_rate,
                "journal": (
                    "per_trajectory_append_only_fsync_sha256_chain_"
                    "reserve_before_dispatch"
                ),
                "identity_check": "exact_returned_model_equals_selected_route",
                "visible_output_only": True,
            },
            "complete": False,
            "summary": {
                "planned_trajectories": len(source_rows),
                "completed_trajectories": len(source_rows),
                "planned_maximum_agent_days": len(source_rows) * 5_000,
                "scheduled_agent_days": sum(
                    row["scheduled_agent_days"] for row in source_rows
                ),
                "responses_received": sum(
                    row["responses_received"] for row in source_rows
                ),
                "invalid_count": sum(row["invalid_count"] for row in source_rows),
                "identity_mismatch_count": 0,
                "transport_failure_count": 1,
                "eligible_trajectories": len(source_rows) - 1,
            },
            "journals": source_journals,
            "sanitized_artifacts": {},
        }
        source_trajectory_ref, _ = _composition_payload(
            source_run,
            "trajectory_metrics",
            "inference_hub_part2_sanitized_trajectory_metrics",
            source_rows,
        )
        source_model_ref, _ = _composition_payload(
            source_run,
            "model_metrics",
            "inference_hub_part2_sanitized_model_metrics",
            source_models,
        )
        source_manifest["sanitized_artifacts"] = {
            "trajectory_metrics": source_trajectory_ref,
            "model_metrics": source_model_ref,
        }
        source_manifest_path = source_run / "private/manifest.json"
        _write_json(source_manifest_path, _seal(source_manifest), private=True)
        source_lock = source_run / "private/.run.lock"
        source_lock.touch()
        os.chmod(source_lock, 0o600)

        effective_rows = []
        overlay_journals: dict[str, Any] = {}
        replay_contract = part2_panel.Part2Contract(
            society_size=50,
            days=100,
            trajectories=12,
            capacity=2_500,
            private_gain=2,
            reserve_cost=2,
            community_benefit=5,
            collapse_death_rate=0.2,
        )
        for row in source_rows:
            repaired = row["operationally_eligible"] is False
            if repaired:
                subject = next(
                    subject
                    for subject in subjects
                    if subject["target_id"] == row["target_id"]
                )
                memory_journal = _MemoryChainedJournal()
                effective = part2_panel._run_trajectory(
                    subject=subject,
                    trajectory_index=int(row["trajectory_index"]),
                    environment_seed=int(row["environment_seed"]),
                    contract=replay_contract,
                    journal=memory_journal,
                    client=_CompositionTrajectoryClient(
                        semantic_invalid_first=pair_index == 2
                    ),
                    participant_workers=1,
                    max_attempts=3,
                    initial_backoff_seconds=1.0,
                    sleep_fn=lambda _seconds: None,
                )
                overlay_journals[
                    f"{subject['target_id']}::{row['trajectory_index']}::1"
                ] = _persist_memory_journal(
                    overlay_run / "private/trajectories/0-0-1.jsonl",
                    memory_journal,
                )
            else:
                effective = dict(row)
            effective.update(
                {
                    "operational_repair_round": 1 if repaired else None,
                    "source_replaced_for_operational_failure": repaired,
                }
            )
            effective_rows.append(effective)
        effective_models = part2_panel._aggregate_models(
            effective_rows,
            subjects,
            expected_trajectories=12,
            capacity=2_500,
        )
        source_evidence = source_manifest["evidence_sha256"]
        effective_trajectory_ref, _ = _composition_payload(
            overlay_run,
            "effective_trajectory_metrics",
            PART2_EFFECTIVE_TRAJECTORY_TYPE,
            effective_rows,
            source_evidence=source_evidence,
        )
        effective_model_ref, _ = _composition_payload(
            overlay_run,
            "effective_model_metrics",
            PART2_EFFECTIVE_MODEL_TYPE,
            effective_models,
            source_evidence=source_evidence,
        )
        overlay_manifest = {
            "schema_version": 1,
            "artifact_type": (
                "inference_hub_part2_operational_trajectory_repair_v1"
            ),
            "source_manifest": {
                "path": str(source_manifest_path.resolve()),
                "file_sha256": _sha_file(source_manifest_path),
                "evidence_sha256": source_evidence,
            },
            "panel_id": source_manifest["panel_id"],
            "part2_contract": source_manifest["part2_contract"],
            "common_environment_seeds": common_seeds,
            "subject_routes": subjects,
            "repair_policy": (
                "whole_trajectory_day_one_exact_route_separate_overlay"
            ),
            "maximum_rounds": 1,
            "created_at_utc": "2026-08-21T12:00:00Z",
            "completed_at_utc": "2026-08-22T12:00:00Z",
            "complete": True,
            "summary": {
                "source_operational_failure_trajectories": 1,
                "operational_repairs_succeeded": 1,
                "operational_repairs_unresolved": 0,
            },
            "journals": overlay_journals,
            "sanitized_artifacts": {
                "effective_trajectory_metrics": effective_trajectory_ref,
                "effective_model_metrics": effective_model_ref,
            },
        }
        overlay_manifest_path = overlay_run / "private/manifest.json"
        _write_json(overlay_manifest_path, _seal(overlay_manifest), private=True)
        overlay_lock = overlay_run / "private/.run.lock"
        overlay_lock.touch()
        os.chmod(overlay_lock, 0o600)
        result.append((source_manifest_path, overlay_manifest_path))
    return result


def _convert_main_overlay_to_cascading(
    root: Path, pairs: list[tuple[Path, Path]],
) -> Path:
    """Make a validator-approved test-double child around the main fixture pair."""

    source_path, child_path = pairs[0]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    child = json.loads(child_path.read_text(encoding="utf-8"))
    parent_path = root / "parent-overlay/private/manifest.json"
    parent = _seal(
        {
            "schema_version": 1,
            "artifact_type": PART2_OPERATIONAL_REPAIR_TYPE,
            "complete": False,
            "summary": {
                "source_operational_failure_trajectories": 1,
                "operational_repairs_succeeded": 0,
                "operational_repairs_unresolved": 1,
            },
        }
    )
    _write_json(parent_path, parent, private=True)
    parent_lock = parent_path.parent / ".run.lock"
    parent_lock.touch()
    os.chmod(parent_lock, 0o600)

    child["artifact_type"] = definitive.PART2_CASCADING_OPERATIONAL_REPAIR_TYPE
    child["parent_overlay_manifest"] = {
        "path": str(parent_path.resolve()),
        "file_sha256": _sha_file(parent_path),
        "evidence_sha256": parent["evidence_sha256"],
    }
    child["base_seed"] = source["base_seed"]
    child["repair_policy"] = definitive.part2_cascading_repair.REPAIR_POLICY
    failed = next(
        row
        for row in json.loads(
            Path(
                source["sanitized_artifacts"]["trajectory_metrics"]["path"]
            ).read_text(encoding="utf-8")
        )["rows"]
        if row["operationally_eligible"] is False
    )
    subject = next(
        row
        for row in source["subject_routes"]
        if row["target_id"] == failed["target_id"]
    )
    child["selected_trajectory"] = {
        "target_id": failed["target_id"],
        "trajectory_index": failed["trajectory_index"],
        "environment_seed_index": failed["environment_seed_index"],
        "environment_seed": failed["environment_seed"],
        "requested_route": subject["route"],
    }
    child["summary"] = {
        "original_source_operational_failure_trajectories": 1,
        "parent_repairs_succeeded": 0,
        "parent_repairs_unresolved": 1,
        "cascading_repairs_succeeded": 1,
        "cascading_repairs_unresolved": 0,
    }
    for key, artifact_type in (
        (
            "effective_trajectory_metrics",
            definitive.PART2_CASCADING_EFFECTIVE_TRAJECTORY_TYPE,
        ),
        (
            "effective_model_metrics",
            definitive.PART2_CASCADING_EFFECTIVE_MODEL_TYPE,
        ),
    ):
        payload_path = Path(child["sanitized_artifacts"][key]["path"])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["artifact_type"] = artifact_type
        _write_json(payload_path, _seal(payload))
        child["sanitized_artifacts"][key] = {
            "path": str(payload_path.resolve()),
            "file_sha256": _sha_file(payload_path),
            "evidence_sha256": payload["evidence_sha256"],
        }
    _write_json(child_path, _seal(child), private=True)
    return parent_path


def _install_source_replay_test_double(
    monkeypatch: pytest.MonkeyPatch,
    pairs: list[tuple[Path, Path]],
) -> tuple[set[tuple[str, int]], list[tuple[str, int]]]:
    """Avoid materializing 273 multi-megabyte source journals in union tests."""

    baseline: dict[tuple[str, int], dict[str, Any]] = {}
    for source_path, _overlay_path in pairs:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        payload_path = Path(
            source["sanitized_artifacts"]["trajectory_metrics"]["path"]
        )
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            key = (str(row["target_id"]), int(row["trajectory_index"]))
            baseline[key] = dict(row)

    observed: list[tuple[str, int]] = []

    def replay(
        records: list[dict[str, Any]],
        *,
        subject: Mapping[str, Any],
        trajectory_index: int,
        environment_seed: int,
        execution_contract: Mapping[str, Any],
        label: str,
        global_attempt_ids: set[str],
    ) -> dict[str, Any]:
        del environment_seed, execution_contract, label, global_attempt_ids
        assert records == []
        key = (str(subject["target_id"]), trajectory_index)
        observed.append(key)
        return dict(baseline[key])

    monkeypatch.setattr(definitive, "_replay_part2_source_journal", replay)
    monkeypatch.setattr(
        definitive,
        "_validate_part2_overlay_union_recursively",
        lambda _pairs: {
            "status": "passed",
            "panel_id": (
                "sota_cross_axis_part2_corrected_original_scale_100d_v1"
            ),
            "source_overlay_pair_count": 3,
            "route_count": 23,
            "trajectory_count": 276,
            "common_environment_seed_count": 12,
            "base_seed": 20_260_802,
            "excluded_target_ids": [PART2_100DAY_DECLARED_EXCLUSION],
        },
    )
    return set(baseline), observed


@pytest.fixture(scope="module")
def production_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("provider-safe-v2-definitive")
    outputs: dict[str, Path] = {}

    # Part 0: the exact production 22 x (48 roots x 3 languages) shape.
    run = root / "part0"
    subjects = [_route(index, "p0") for index in range(22)]
    manifest = _base_manifest("inference_hub_part0_accelerated_private_panel", subjects)
    manifest["judge"] = manifest.pop("judge_reservation")
    manifest["summary"] = {"planned_units": 22 * 144}
    raw = {}
    for subject in subjects:
        rows = []
        for root_index in range(48):
            for language_index, language in enumerate(("english", "chinese", "russian")):
                unit = root_index * 3 + language_index
                outcome = "INVALID" if unit == 0 else ("REFUSAL" if unit % 2 else "COMPLIANCE")
                rows.append({
                    "event": "unit_completed", "target_id": subject["target_id"],
                    "trial_id": f"trial-{root_index}-{language}",
                    "root_id": f"root-{root_index}", "language": language,
                    "outcome": outcome,
                })
        raw[subject["target_id"]] = _journal(run / "private/raw" / f"{subject['model']}.jsonl", rows)
    manifest["journals"] = {"attempt_ledger": _journal(run / "private/attempts.jsonl", []), "raw_responses": raw}
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["part0"] = path

    # Part 1: the exact production 75 x 384 matrix.
    run = root / "part1"
    subjects = [_route(index, "p1") for index in range(75)]
    manifest = _base_manifest("inference_hub_part1_large_n_exploratory_panel", subjects)
    manifest.update({"trial_limit": 384, "judge_dispatched": False})
    raw = {}
    for subject in subjects:
        rows = []
        unit = 0
        for game in GAMES:
            for domain in DOMAINS:
                for root_index in range(32):
                    counterbalance = COUNTERBALANCES[root_index % len(COUNTERBALANCES)]
                    rows.append({
                        "target_id": subject["target_id"], "trial_id": f"trial-{unit}",
                        "root_id": f"{game}-{domain}-{root_index}",
                        "counterbalance_id": counterbalance.counterbalance_id,
                        "game": game, "domain": domain,
                        "format_valid": unit != 0,
                        "parsed_action": counterbalance.welfare_preserving_label,
                    })
                    unit += 1
        raw[subject["target_id"]] = _journal(run / "private/raw" / f"{subject['model']}.jsonl", rows)
    manifest["journals"] = {"attempt_ledger": _journal(run / "private/attempts.jsonl", []), "raw_responses": raw}
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["part1"] = path

    # Part 2: the exact production 19 x 12 matched trajectories.
    run = root / "part2"
    subjects = [_route(index, "p2") for index in range(19)]
    manifest = _base_manifest("inference_hub_part2_corrected_matched_panel", subjects)
    refs, trajectory_rows = {}, []
    for subject in subjects:
        for index in range(12):
            key = f"{subject['target_id']}::{index}"
            refs[key] = _journal(run / "private/trajectories" / f"{subject['model']}-{index}.jsonl", [])
            trajectory_rows.append({
                "schema_version": 1,
                "target_id": subject["target_id"],
                "upstream_provider": subject["upstream_provider"],
                "model": subject["model"],
                "trajectory_index": index,
                "environment_seed_index": index,
                "environment_seed": index,
                "scheduled_agent_days": 10, "responses_received": 10,
                "invalid_count": 1 if index == 0 else 0,
                "identity_mismatch_count": 0,
                "transport_failure_count": 0, "restraint_count": 7,
                "overuse_count": 2 if index == 0 else 3,
                "operationally_eligible": True,
                "restraint_rate": 0.7,
                "aurc": 0.8, "aupc": 0.9,
                "reserve_nondepletion": True,
                "final_reserve": 500,
                "final_population": 40,
                "population_retention": 0.8,
                "cumulative_private_payoff": 13,
                "cumulative_group_payoff": 0,
            })
    manifest["journals"] = refs
    _artifact(run, "trajectory_metrics", "inference_hub_part2_sanitized_trajectory_metrics", trajectory_rows, manifest)
    _artifact(run, "model_metrics", "inference_hub_part2_sanitized_model_metrics", [{"target_id": row["target_id"]} for row in subjects], manifest)
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["part2"] = path

    # Role calibration: exact 6 x 96 roots x 3 frames x 4 blocks.
    run = root / "role"
    subjects = [_route(index, "role") for index in range(6)]
    manifest = _base_manifest("inference_hub_part1_role_calibration_private_v1", subjects)
    manifest.update({"frames": ["advice", "observer_evaluation", "prediction"], "trials_per_subject": 1152})
    raw = {}
    for subject in subjects:
        rows = []
        for frame in manifest["frames"]:
            for root_index in range(96):
                for block in range(4):
                    rows.append({
                        "target_id": subject["target_id"], "trial_id": f"{frame}-{root_index}-{block}",
                        "frame_id": frame, "format_valid": not (root_index == 0 and block == 0),
                        "welfare_preserving": True,
                    })
        raw[subject["target_id"]] = _journal(run / "private/raw" / f"{subject['model']}.jsonl", rows)
    manifest["journals"] = {"attempt_ledger": _journal(run / "private/attempts.jsonl", []), "raw_responses": raw}
    role_summary = _seal({"schema_version": 1, "artifact_type": "part1_role_calibration_sanitized_summary_v1", "estimates": []})
    _write_json(run / "sanitized/summary.json", role_summary)
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["role"] = path

    # Sensitivity: five exact compatible sentinels and one real Holm-25 family.
    run = root / "sensitivity"
    subjects = [_route(index, "sens") for index in range(5)]
    manifest = _base_manifest("inference_hub_part2_sensitivity_campaign_v1", subjects)
    manifest["selected_subject_routes"] = manifest.pop("subject_routes")
    manifest["execution_contract"]["provider_concurrency_required"] = 1
    manifest["source_artifacts"][str(DESIGN_PATH)] = _sha_file(DESIGN_PATH)
    design, cells = load_sensitivity_design(DESIGN_PATH)
    refs, trajectory_rows = {}, []
    for subject_index, subject in enumerate(subjects):
        for cell in cells:
            for seed in range(int(design["seeds_per_cell"])):
                key = f"{cell.cell_id}::{subject['target_id']}::{seed}"
                refs[key] = _journal(run / "private/trajectories" / f"{subject_index}-{cell.cell_id}-{seed}.jsonl", [])
                coded_signal = sum(cell.coded_levels.values()) * 0.01
                scheduled = cell.society_size * cell.horizon_days
                invalid = 1 if seed == 0 else 0
                trajectory_rows.append({
                    "cell_id": cell.cell_id, "target_id": subject["target_id"],
                    "trajectory_index": seed, "environment_seed_index": seed,
                    "environment_seed": seed,
                    "normalized_aurc": 0.5 + coded_signal + seed * 0.0001,
                    "society_size": cell.society_size,
                    "horizon_days": cell.horizon_days,
                    "scheduled_agent_days": scheduled,
                    "responses_received": scheduled,
                    "invalid_count": invalid,
                    "identity_mismatch_count": 0,
                    "transport_failure_count": 0,
                    "restraint_count": scheduled - invalid,
                    "overuse_count": 0,
                    "operationally_eligible": True,
                })
    manifest["journals"] = refs
    manifest["attempt_ledger"] = _journal(run / "private/attempts.jsonl", [])
    effects = _analyze_deadline_sensitivity(
        trajectory_rows,
        sentinel_ids=[row["target_id"] for row in subjects],
        design=design,
    )
    _artifact(run, "trajectory_metrics", "part2_sensitivity_trajectory_metrics_v1", trajectory_rows, manifest)
    _artifact(run, "sentinel_cell_metrics", "part2_sensitivity_sentinel_cell_metrics_v1", [], manifest)
    effect_payload = _seal({
        "schema_version": 1, "artifact_type": "part2_sensitivity_main_effects_v1",
        "analysis_status": "complete_deadline_exploratory", "confirmatory": False,
        "inference_scope": "deadline_exploratory",
        "global_holm_family": SENSITIVITY_HOLM_FAMILY,
        "global_holm_family_size": SENSITIVITY_HOLM_FAMILY_SIZE,
        "rows": effects,
    })
    effect_path = run / "sanitized/main_effects.json"
    _write_json(effect_path, effect_payload)
    manifest["sanitized_artifacts"]["main_effects"] = {"path": str(effect_path.resolve()), "file_sha256": _sha_file(effect_path), "evidence_sha256": effect_payload["evidence_sha256"]}
    diagnostic = _seal({"schema_version": 1, "artifact_type": "part2_sensitivity_call_order_diagnostic_v1", "analysis_family": SENSITIVITY_DIAGNOSTIC_FAMILY, "rows": []})
    diagnostic_path = run / "sanitized/call_order_diagnostic.json"
    _write_json(diagnostic_path, diagnostic)
    manifest["sanitized_artifacts"]["call_order_diagnostic"] = {"path": str(diagnostic_path.resolve()), "file_sha256": _sha_file(diagnostic_path), "evidence_sha256": diagnostic["evidence_sha256"]}
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["sensitivity"] = path
    return outputs


def _run(bundle: Mapping[str, Path], output: Path) -> dict[str, Any]:
    return analyze(
        part0=bundle["part0"], part1=bundle["part1"], part2=bundle["part2"],
        role_calibration=bundle["role"], sensitivity=bundle["sensitivity"],
        output_dir=output,
    )


def _terminalized_part0_run(root: Path, *, drift_request_hash: bool = False) -> Path:
    run = root / "terminalized-part0"
    subjects = [_route(index, "p0") for index in range(22)]
    manifest = _base_manifest(
        "inference_hub_part0_accelerated_private_panel", subjects
    )
    manifest["judge"] = manifest.pop("judge_reservation")
    manifest["complete"] = False
    manifest.pop("completed_at_utc", None)
    manifest["schedule"] = [
        {
            "trial_id": f"trial-{root_index}-{language}",
            "root_id": f"root-{root_index}",
            "language": language,
        }
        for root_index in range(48)
        for language in ("english", "chinese", "russian")
    ]
    safe = ROOT / "experiments/misc/inference_hub_provider_safe_v2.py"
    deadline = ROOT / "experiments/misc/inference_hub_part0_deadline_retry.py"
    manifest["source_artifacts"] = {
        str(safe): _sha_file(safe),
        str(deadline): _sha_file(deadline),
    }
    shared = {
        "schema_version": 2,
        "algorithm": "cross_process_provider_aware_leaky_bucket_with_leases_all_http_5xx_full_throttle_cooldown",
        "global_concurrency": 16,
        "provider_concurrency": 3,
        "global_requests_per_second": 10.0,
        "provider_requests_per_second": 2.0,
        "lease_seconds": 900.0,
        "poll_seconds": 0.05,
        "throttle_cooldown_seconds": 30.0,
        "transient_cooldown_seconds": 5.0,
    }
    shared["policy_sha256"] = hashlib.sha256(
        json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["execution_contract"] = {
        "shared_rate_limit": shared,
        "max_attempts_per_request": 8,
    }

    failure_counts = (3, 4, 9, 3, 2, 9, 6, 5)
    raw_refs = {}
    ledger_rows = []
    outcomes = {"REFUSAL": 0, "COMPLIANCE": 0, "UNCLEAR": 0, "INVALID": 0}
    for subject_index, subject in enumerate(subjects):
        failed_count = failure_counts[subject_index] if subject_index < 8 else 0
        rows = []
        visible_ids = []
        for unit, scheduled in enumerate(manifest["schedule"]):
            trial_id = scheduled["trial_id"]
            if unit < failed_count:
                for attempt_number in range(1, 9):
                    attempt_id = f"attempt-{subject_index}-{unit}-{attempt_number}"
                    request_hash = (
                        f"{'b' if drift_request_hash and subject_index == 0 and unit == 0 and attempt_number == 8 else 'a'}"
                        * 64
                    )
                    ledger_rows.extend(
                        [
                            {
                                "event": "reserved_before_dispatch",
                                "attempt_id": attempt_id,
                                "attempt_number": attempt_number,
                                "request_sha256": request_hash,
                                "role": "subject",
                                "route": subject["route"],
                                "upstream_provider": subject["upstream_provider"],
                                "target_id": subject["target_id"],
                                "work_id": trial_id,
                            },
                            {
                                "event": "attempt_completed",
                                "attempt_id": attempt_id,
                                "outcome": "failed",
                                "transient": True,
                                "failure_code": "http_400_periodic_retry",
                                "http_status": 400,
                            },
                        ]
                    )
                rows.append(
                    {
                        "event": "unit_completed",
                        "target_id": subject["target_id"],
                        "trial_id": trial_id,
                        "root_id": scheduled["root_id"],
                        "language": scheduled["language"],
                        "outcome": "INVALID",
                        "judge_label": None,
                        "judge_dispatched": False,
                        "operational_failure": {
                            "failure_code": "http_400_periodic_retry",
                            "http_status": 400,
                        },
                    }
                )
                outcomes["INVALID"] += 1
                continue
            visible_ids.append(trial_id)
            rows.append(
                {
                    "event": "subject_response_retained",
                    "target_id": subject["target_id"],
                    "trial_id": trial_id,
                    "model_identity_valid": True,
                }
            )
            outcome = "REFUSAL" if unit % 2 else "COMPLIANCE"
            rows.append(
                {
                    "event": "unit_completed",
                    "target_id": subject["target_id"],
                    "trial_id": trial_id,
                    "root_id": scheduled["root_id"],
                    "language": scheduled["language"],
                    "outcome": outcome,
                }
            )
            outcomes[outcome] += 1
        for start in range(0, len(visible_ids), 8):
            rows.append(
                {
                    "event": "judge_batch_retained",
                    "target_id": subject["target_id"],
                    "trial_ids": visible_ids[start : start + 8],
                    "model_identity_valid": True,
                }
            )
        raw_refs[subject["target_id"]] = _journal(
            run / "private/raw" / f"{subject_index}.jsonl", rows
        )
    manifest["journals"] = {
        "attempt_ledger": _journal(run / "private/attempts.jsonl", ledger_rows),
        "raw_responses": raw_refs,
    }
    manifest["summary"] = {
        "planned_units": 22 * 144,
        "retained_terminal_units": 22 * 144,
        "outcomes": outcomes,
        "subject_transport_failures": 41,
        "judge_failed_units": 0,
        "subject_model_identity_mismatches": 0,
        "judge_model_identity_mismatches": 0,
        "operationally_retired_units": 0,
    }
    (run / "private/.run.lock").touch()
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    return path


def test_uncertainty_helpers_are_deterministic_and_use_independent_units() -> None:
    clusters = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] * 12
    first = _root_cluster_bootstrap_95(
        clusters, namespace="part0-overall-root-cluster", target_id="route/exact"
    )
    second = _root_cluster_bootstrap_95(
        clusters, namespace="part0-overall-root-cluster", target_id="route/exact"
    )
    assert first == second
    assert first[0] < 0.5 < first[1]

    strata = [[float((root + stratum) % 3 == 0) for root in range(32)] for stratum in range(12)]
    stratified_first = _stratified_root_bootstrap_95(strata, target_id="route/exact")
    stratified_second = _stratified_root_bootstrap_95(strata, target_id="route/exact")
    assert stratified_first == stratified_second
    assert stratified_first[0] < sum(map(sum, strata)) / 384 < stratified_first[1]

    zero_low, zero_high = _wilson_95(0, 48)
    full_low, full_high = _wilson_95(48, 48)
    assert zero_low == 0.0 < zero_high < 1.0
    assert 0.0 < full_low < full_high == 1.0
    bounded_low, bounded_high = _mean_t_95(
        [1.0, 1.0, 1.0, 0.95], bounds=(0.0, 1.0)
    )
    assert bounded_low is not None and 0.0 <= bounded_low <= 1.0
    assert bounded_high == 1.0


def test_full_production_shaped_analysis_and_invalid_denominators(
    production_bundle: Mapping[str, Path], tmp_path: Path,
) -> None:
    result = _run(production_bundle, tmp_path / "out")

    assert result["row_counts"] == {
        "part0_models": 22, "part1_models": 75, "part2_models": 19,
        "role_calibration_model_frames": 18, "sensitivity_models": 5,
        "sensitivity_main_effects": 25,
    }
    assert result["human_labels_generated"] is False
    assert result["confirmatory_or_paper_promotion_permitted"] is False
    p0 = json.loads((tmp_path / "out/part0_models.jsonl").read_text().splitlines()[0])
    assert p0["scheduled_units"] == 144
    assert p0["first_attempt_invalid_count"] == 1
    assert p0["repaired_invalid_count"] == 0
    assert p0["refusal_rate_all_scheduled"] == p0["refusal_count"] / 144
    assert p0["finite_bank_sensitivity_replicates"] == 5_000
    assert p0["refusal_rate_all_scheduled_finite_bank_sensitivity_low"] <= p0["refusal_rate_all_scheduled"] <= p0["refusal_rate_all_scheduled_finite_bank_sensitivity_high"]
    p1 = json.loads((tmp_path / "out/part1_models.jsonl").read_text().splitlines()[0])
    assert p1["welfare_preserving_rate_all_scheduled"] == 383 / 384
    assert p1["finite_bank_sensitivity_strata"] == 12
    assert p1["finite_bank_sensitivity_replicates"] == 5_000
    p2 = json.loads((tmp_path / "out/part2_models.jsonl").read_text().splitlines()[0])
    assert p2["restraint_rate_all_scheduled"] == 84 / 120
    assert p2["restraint_rate_among_valid"] == 84 / 119
    assert p2["operationally_eligible_trajectory_count"] == 12
    assert p2["environmentally_estimable_trajectory_count"] == 11
    assert p2["semantic_invalid_trajectory_count"] == 1
    assert p2["mean_aurc_eligible"] == pytest.approx(0.8)
    assert p2["mean_aupc_eligible"] == pytest.approx(0.9)
    assert p2["reserve_nondepletion_rate_eligible"] == 1.0
    assert p2["mean_population_retention_eligible"] == pytest.approx(0.8)
    assert p2["mean_aurc_eligible_t95_low"] == pytest.approx(0.8)
    assert p2["mean_aurc_eligible_t95_high"] == pytest.approx(0.8)
    assert p2["reserve_nondepletion_rate_eligible_wilson95_low"] < 1.0
    assert p2["reserve_nondepletion_rate_eligible_wilson95_high"] == 1.0
    assert p2["restraint_interval_trajectory_count"] == 12

    manifest = json.loads((tmp_path / "out/analysis_manifest.json").read_text())
    assert manifest["path_policy"] == "portable_basenames_only_no_host_absolute_paths_in_public_manifest"
    assert all("path" not in binding for binding in manifest["input_manifests"].values())
    assert all(not Path(binding["basename"]).is_absolute() for binding in manifest["input_manifests"].values())
    assert len(manifest["public_outputs"]) == 13
    for binding in manifest["public_outputs"]:
        public_path = tmp_path / "out" / binding["basename"]
        assert binding["file_sha256"] == _sha_file(public_path)
        assert binding["kind"].startswith("machine_readable_")
    serialized_manifest = json.dumps(manifest, sort_keys=True)
    assert "/Users/" not in serialized_manifest
    assert "/private/" not in serialized_manifest


def test_three_pair_part2_composition_recomputes_effective_estimates(
    production_bundle: Mapping[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition")
    expected_replays, observed_replays = _install_source_replay_test_double(
        monkeypatch, pairs
    )
    lock_paths = [path.parent / ".run.lock" for pair in pairs for path in pair]
    original_write_json = definitive._write_json
    publication_lock_checks: list[bool] = []

    def write_json_with_lock_probe(path: Path, value: Any) -> None:
        if path.name == "analysis_manifest.json":
            probe = _lock_probe(lock_paths, expect_blocked=True)
            publication_lock_checks.append(probe.returncode == 0)
        original_write_json(path, value)

    monkeypatch.setattr(definitive, "_write_json", write_json_with_lock_probe)
    output = tmp_path / "composed-out"
    result = analyze(
        part0=production_bundle["part0"],
        part1=production_bundle["part1"],
        part2_source_overlay_pairs=pairs,
        part2_excluded_target_ids=[PART2_100DAY_DECLARED_EXCLUSION],
        role_calibration=production_bundle["role"],
        sensitivity=production_bundle["sensitivity"],
        output_dir=output,
    )

    assert result["row_counts"]["part2_models"] == 23
    composition = result["part2_operational_repair_composition"]
    assert composition["route_count"] == 23
    assert composition["trajectory_count"] == 276
    assert composition["ordered_target_ids"] == list(
        PART2_100DAY_ORDERED_TARGET_IDS
    )
    assert set(observed_replays) == expected_replays
    assert len(observed_replays) == 276
    assert publication_lock_checks == [True]
    assert composition["declared_excluded_target_ids"] == [
        PART2_100DAY_DECLARED_EXCLUSION
    ]
    assert [
        row["audit"]["route_count"]
        for row in composition["ordered_source_overlay_pairs"]
    ] == [21, 1, 1]
    assert [
        row["audit"]["source_trajectories_replayed"]
        for row in composition["ordered_source_overlay_pairs"]
    ] == [252, 12, 12]
    assert set(result["input_manifests"]["part2"]) == {
        "basename",
        "file_sha256",
        "evidence_sha256",
    }

    models = {
        row["target_id"]: row
        for row in map(
            json.loads,
            (output / "part2_models.jsonl").read_text().splitlines(),
        )
    }
    nemotron = models["nvidia/nemotron-3-ultra"]
    assert nemotron["operationally_eligible_trajectory_count"] == 12
    assert nemotron["environmentally_estimable_trajectory_count"] == 9
    assert nemotron["semantic_invalid_trajectory_count"] == 3
    deepseek = models["deepseek-ai/deepseek-v4-flash"]
    assert deepseek["operationally_eligible_trajectory_count"] == 12
    assert deepseek["environmentally_estimable_trajectory_count"] == 0
    assert deepseek["semantic_invalid_trajectory_count"] == 12
    assert deepseek["restraint_interval_trajectory_count"] == 12
    for field in (
        "mean_aurc_eligible",
        "mean_aurc_eligible_t95_low",
        "mean_aurc_eligible_t95_high",
        "mean_aupc_eligible",
        "mean_population_retention_eligible",
        "reserve_nondepletion_rate_eligible",
        "reserve_nondepletion_rate_eligible_wilson95_low",
        "reserve_nondepletion_rate_eligible_wilson95_high",
    ):
        assert deepseek[field] is None
    trajectories = json.loads((output / "figure_aggregates.json").read_text())[
        "part2_trajectories"
    ]
    assert len(trajectories) == 276
    assert all(set(row) == PART2_EFFECTIVE_TRAJECTORY_ROW_KEYS for row in trajectories)
    assert len(
        {
            (row["target_id"], row["trajectory_index"])
            for row in trajectories
        }
    ) == 276
    serialized = json.dumps(result, sort_keys=True)
    assert "/Users/" not in serialized
    assert "/private/" not in serialized


def test_part2_composition_requires_recursive_validator_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = [
        (Path("main-source"), Path("main-overlay")),
        (Path("nemotron-source"), Path("nemotron-overlay")),
        (Path("deepseek-source"), Path("deepseek-overlay")),
    ]
    observed: list[list[tuple[Path, Path]]] = []

    def validate(values: list[tuple[Path, Path]]) -> dict[str, Any]:
        observed.append(list(values))
        return {
            "status": "passed",
            "panel_id": (
                "sota_cross_axis_part2_corrected_original_scale_100d_v1"
            ),
            "source_overlay_pair_count": 3,
            "route_count": 23,
            "trajectory_count": 276,
            "common_environment_seed_count": 12,
            "base_seed": 20_260_802,
            "excluded_target_ids": [PART2_100DAY_DECLARED_EXCLUSION],
        }

    monkeypatch.setattr(
        definitive.part2_overlay_validator,
        "validate_operational_overlay_pairs",
        validate,
    )
    result = definitive._validate_part2_overlay_union_recursively(pairs)

    assert observed == [pairs]
    assert result["trajectory_count"] == 276

    monkeypatch.setattr(
        definitive.part2_overlay_validator,
        "validate_operational_overlay_pairs",
        lambda _pairs: {**result, "trajectory_count": 275},
    )
    with pytest.raises(DefinitiveAnalysisError, match="unexpected union contract"):
        definitive._validate_part2_overlay_union_recursively(pairs)


def test_part2_composition_accepts_complete_cascading_main_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-cascading")
    parent_path = _convert_main_overlay_to_cascading(tmp_path, pairs)
    _install_source_replay_test_double(monkeypatch, pairs)

    models, _figure, context, rows = _part2_composition(
        pairs, [PART2_100DAY_DECLARED_EXCLUSION]
    )

    assert len(models) == 23
    assert len(rows) == 276
    assert len(context["pairs"][0]["effective_rows"]) == 252
    assert context["binding"]["ordered_target_ids"] == list(
        PART2_100DAY_ORDERED_TARGET_IDS
    )
    child_binding = context["binding"]["ordered_source_overlay_pairs"][0][
        "operational_repair_overlay"
    ]
    assert child_binding["parent_operational_repair_overlay"] == {
        "basename": parent_path.name,
        "file_sha256": _sha_file(parent_path),
        "evidence_sha256": json.loads(
            parent_path.read_text(encoding="utf-8")
        )["evidence_sha256"],
    }


def test_part2_cascading_composition_holds_parent_lock(
    tmp_path: Path,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-cascading-lock")
    parent_path = _convert_main_overlay_to_cascading(tmp_path, pairs)
    lock_paths = [
        *(path.parent / ".run.lock" for pair in pairs for path in pair),
        parent_path.parent / ".run.lock",
    ]

    with definitive._hold_part2_composition_locks(pairs):
        probe = _lock_probe(lock_paths, expect_blocked=True)
        assert probe.returncode == 0, probe.stderr
    released = _lock_probe(lock_paths, expect_blocked=False)
    assert released.returncode == 0, released.stderr


def test_part2_cascading_composition_rejects_active_parent_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-cascading-writer")
    parent_path = _convert_main_overlay_to_cascading(tmp_path, pairs)
    called = False

    def must_not_validate(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("validation began while the parent writer was active")

    monkeypatch.setattr(definitive, "_part2_composition_locked", must_not_validate)
    writer_program = """
import fcntl
import sys

with open(sys.argv[1], "rb") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    print("locked", flush=True)
    sys.stdin.readline()
"""
    writer = subprocess.Popen(
        [
            sys.executable,
            "-c",
            writer_program,
            str(parent_path.parent / ".run.lock"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert writer.stdout is not None
        assert writer.stdout.readline().strip() == "locked"
        with pytest.raises(DefinitiveAnalysisError, match="active writer"):
            _part2_composition(pairs, [PART2_100DAY_DECLARED_EXCLUSION])
        assert called is False
    finally:
        if writer.stdin is not None:
            writer.stdin.write("release\n")
            writer.stdin.flush()
            writer.stdin.close()
        writer.wait(timeout=10)


def test_three_pair_part2_composition_fails_closed_on_order_exclusion_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-negative")
    _install_source_replay_test_double(monkeypatch, pairs)
    with pytest.raises(DefinitiveAnalysisError, match="Opus-4.5 exclusion"):
        _part2_composition(pairs, [])
    with pytest.raises(DefinitiveAnalysisError, match="exactly 21 routes"):
        _part2_composition([pairs[1], pairs[0], pairs[2]], [PART2_100DAY_DECLARED_EXCLUSION])

    effective_path = pairs[2][1].parents[1] / "sanitized/effective_trajectory_metrics.json"
    effective_path.write_text(
        effective_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(DefinitiveAnalysisError, match="integrity failed"):
        _part2_composition(pairs, [PART2_100DAY_DECLARED_EXCLUSION])


def test_part2_composition_rejects_reordered_main_target_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-target-order")
    _install_source_replay_test_double(monkeypatch, pairs)
    load_pair = definitive._load_part2_source_overlay_pair

    def reorder_main_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
        loaded = load_pair(*args, **kwargs)
        if loaded["audit"]["pair_ordinal"] == 1:
            loaded["subjects"][0], loaded["subjects"][1] = (
                loaded["subjects"][1],
                loaded["subjects"][0],
            )
        return loaded

    monkeypatch.setattr(
        definitive, "_load_part2_source_overlay_pair", reorder_main_pair
    )
    with pytest.raises(DefinitiveAnalysisError, match="frozen ordered panel"):
        _part2_composition(pairs, [PART2_100DAY_DECLARED_EXCLUSION])


def test_part2_composition_replays_repaired_metrics_instead_of_trusting_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-replay-negative")
    _install_source_replay_test_double(monkeypatch, pairs)
    overlay_path = pairs[0][1]
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    trajectory_path = Path(
        overlay["sanitized_artifacts"]["effective_trajectory_metrics"]["path"]
    )
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
    repaired = next(
        row
        for row in trajectories["rows"]
        if row["source_replaced_for_operational_failure"] is True
    )
    repaired["aurc"] = float(repaired["aurc"]) + 0.01
    _write_json(trajectory_path, _seal(trajectories))
    overlay["sanitized_artifacts"]["effective_trajectory_metrics"] = {
        "path": str(trajectory_path.resolve()),
        "file_sha256": _sha_file(trajectory_path),
        "evidence_sha256": trajectories["evidence_sha256"],
    }

    model_path = Path(
        overlay["sanitized_artifacts"]["effective_model_metrics"]["path"]
    )
    models = json.loads(model_path.read_text(encoding="utf-8"))
    models["rows"] = part2_panel._aggregate_models(
        trajectories["rows"],
        overlay["subject_routes"],
        expected_trajectories=12,
        capacity=2_500,
    )
    _write_json(model_path, _seal(models))
    overlay["sanitized_artifacts"]["effective_model_metrics"] = {
        "path": str(model_path.resolve()),
        "file_sha256": _sha_file(model_path),
        "evidence_sha256": models["evidence_sha256"],
    }
    _write_json(overlay_path, _seal(overlay), private=True)

    with pytest.raises(DefinitiveAnalysisError, match="does not reproduce"):
        _part2_composition(pairs, [PART2_100DAY_DECLARED_EXCLUSION])


def test_part2_composition_replays_successful_source_instead_of_trusting_reseal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-source-replay-negative")
    _install_source_replay_test_double(monkeypatch, pairs)
    source_path, overlay_path = pairs[0]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_trajectory_path = Path(
        source["sanitized_artifacts"]["trajectory_metrics"]["path"]
    )
    source_trajectories = json.loads(
        source_trajectory_path.read_text(encoding="utf-8")
    )
    changed_key = (
        source_trajectories["rows"][1]["target_id"],
        source_trajectories["rows"][1]["trajectory_index"],
    )
    source_trajectories["rows"][1]["aurc"] += 0.01
    _write_json(source_trajectory_path, _seal(source_trajectories))
    source["sanitized_artifacts"]["trajectory_metrics"] = {
        "path": str(source_trajectory_path.resolve()),
        "file_sha256": _sha_file(source_trajectory_path),
        "evidence_sha256": source_trajectories["evidence_sha256"],
    }

    source_model_path = Path(
        source["sanitized_artifacts"]["model_metrics"]["path"]
    )
    source_models = json.loads(source_model_path.read_text(encoding="utf-8"))
    source_models["rows"] = part2_panel._aggregate_models(
        source_trajectories["rows"],
        source["subject_routes"],
        expected_trajectories=12,
        capacity=2_500,
    )
    _write_json(source_model_path, _seal(source_models))
    source["sanitized_artifacts"]["model_metrics"] = {
        "path": str(source_model_path.resolve()),
        "file_sha256": _sha_file(source_model_path),
        "evidence_sha256": source_models["evidence_sha256"],
    }
    _write_json(source_path, _seal(source), private=True)

    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    effective_path = Path(
        overlay["sanitized_artifacts"]["effective_trajectory_metrics"]["path"]
    )
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    changed_effective = next(
        row
        for row in effective["rows"]
        if (row["target_id"], row["trajectory_index"]) == changed_key
    )
    changed_effective["aurc"] += 0.01
    effective["source_manifest_evidence_sha256"] = source["evidence_sha256"]
    _write_json(effective_path, _seal(effective))
    overlay["sanitized_artifacts"]["effective_trajectory_metrics"] = {
        "path": str(effective_path.resolve()),
        "file_sha256": _sha_file(effective_path),
        "evidence_sha256": effective["evidence_sha256"],
    }

    effective_model_path = Path(
        overlay["sanitized_artifacts"]["effective_model_metrics"]["path"]
    )
    effective_models = json.loads(
        effective_model_path.read_text(encoding="utf-8")
    )
    effective_models["rows"] = part2_panel._aggregate_models(
        effective["rows"],
        overlay["subject_routes"],
        expected_trajectories=12,
        capacity=2_500,
    )
    effective_models["source_manifest_evidence_sha256"] = source[
        "evidence_sha256"
    ]
    _write_json(effective_model_path, _seal(effective_models))
    overlay["sanitized_artifacts"]["effective_model_metrics"] = {
        "path": str(effective_model_path.resolve()),
        "file_sha256": _sha_file(effective_model_path),
        "evidence_sha256": effective_models["evidence_sha256"],
    }
    overlay["source_manifest"] = {
        "path": str(source_path.resolve()),
        "file_sha256": _sha_file(source_path),
        "evidence_sha256": source["evidence_sha256"],
    }
    _write_json(overlay_path, _seal(overlay), private=True)

    with pytest.raises(
        DefinitiveAnalysisError,
        match="source trajectory metrics differ from simulator replay",
    ):
        _part2_composition(pairs, [PART2_100DAY_DECLARED_EXCLUSION])


@pytest.mark.parametrize(
    ("field", "private_value"),
    [
        ("private_path", "/private/evidence/journal.jsonl"),
        ("response_text", "private response"),
        ("raw_response", {"choices": []}),
        ("request_body", {"model": "private-route"}),
        ("prompt_text", "private prompt"),
        ("reasoning", "private reasoning"),
        ("route", "provider/private-route"),
    ],
)
def test_part2_public_trajectory_schema_rejects_private_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    private_value: Any,
) -> None:
    pairs = _composition_fixture(tmp_path / f"composition-schema-{field}")
    _install_source_replay_test_double(monkeypatch, pairs)
    overlay_path = pairs[0][1]
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    trajectory_path = Path(
        overlay["sanitized_artifacts"]["effective_trajectory_metrics"]["path"]
    )
    trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectories["rows"][0][field] = private_value
    _write_json(trajectory_path, _seal(trajectories))
    overlay["sanitized_artifacts"]["effective_trajectory_metrics"] = {
        "path": str(trajectory_path.resolve()),
        "file_sha256": _sha_file(trajectory_path),
        "evidence_sha256": trajectories["evidence_sha256"],
    }
    _write_json(overlay_path, _seal(overlay), private=True)

    with pytest.raises(DefinitiveAnalysisError, match="public schema changed"):
        _part2_composition(pairs, [PART2_100DAY_DECLARED_EXCLUSION])


def _lock_probe(lock_paths: list[Path], *, expect_blocked: bool) -> subprocess.CompletedProcess[str]:
    program = """
import fcntl
import json
import sys

expect_blocked = sys.argv[1] == "blocked"
paths = json.loads(sys.argv[2])
observed = []
for path in paths:
    with open(path, "rb") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            observed.append(True)
        else:
            observed.append(False)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
valid = all(observed) if expect_blocked else not any(observed)
raise SystemExit(0 if valid else 9)
"""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            "blocked" if expect_blocked else "available",
            json.dumps([str(path) for path in lock_paths]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_part2_composition_holds_all_six_shared_locks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-locks")
    lock_paths = [path.parent / ".run.lock" for pair in pairs for path in pair]
    sentinel = ([], [], {}, [])

    def probe_locked(
        source_overlay_pairs: list[tuple[Path, Path]],
        declared_excluded_target_ids: list[str],
    ) -> tuple[list[Any], list[Any], dict[str, Any], list[Any]]:
        assert list(source_overlay_pairs) == pairs
        assert declared_excluded_target_ids == [PART2_100DAY_DECLARED_EXCLUSION]
        probe = _lock_probe(lock_paths, expect_blocked=True)
        assert probe.returncode == 0, probe.stderr
        return sentinel

    monkeypatch.setattr(definitive, "_part2_composition_locked", probe_locked)
    assert _part2_composition(
        pairs, [PART2_100DAY_DECLARED_EXCLUSION]
    ) == sentinel
    released = _lock_probe(lock_paths, expect_blocked=False)
    assert released.returncode == 0, released.stderr


def test_part2_composition_pins_symlink_before_locking_and_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-symlink")
    source_alias = tmp_path / "source-manifest-alias.json"
    source_alias.symlink_to(pairs[0][0])
    aliased_pairs = [(source_alias, pairs[0][1]), *pairs[1:]]
    sentinel = ([], [], {}, [])

    def retarget_after_lock(
        source_overlay_pairs: list[tuple[Path, Path]],
        _declared_excluded_target_ids: list[str],
    ) -> tuple[list[Any], list[Any], dict[str, Any], list[Any]]:
        source_alias.unlink()
        source_alias.symlink_to(pairs[1][0])
        normalized = list(source_overlay_pairs)
        assert normalized[0][0] == pairs[0][0].resolve()
        assert normalized[0][0] != source_alias.resolve()
        return sentinel

    monkeypatch.setattr(
        definitive, "_part2_composition_locked", retarget_after_lock
    )
    assert _part2_composition(
        aliased_pairs, [PART2_100DAY_DECLARED_EXCLUSION]
    ) == sentinel


def test_part2_composition_rejects_active_writer_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-active-writer")
    lock_paths = sorted(
        [path.parent / ".run.lock" for pair in pairs for path in pair],
        key=lambda path: str(path),
    )
    called = False

    def must_not_validate(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("validation began before all six locks were held")

    monkeypatch.setattr(definitive, "_part2_composition_locked", must_not_validate)
    writer_program = """
import fcntl
import sys

with open(sys.argv[1], "rb") as handle:
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    print("locked", flush=True)
    sys.stdin.readline()
"""
    writer = subprocess.Popen(
        [sys.executable, "-c", writer_program, str(lock_paths[-1])],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert writer.stdout is not None
        assert writer.stdout.readline().strip() == "locked"
        with pytest.raises(DefinitiveAnalysisError, match="active writer"):
            _part2_composition(pairs, [PART2_100DAY_DECLARED_EXCLUSION])
        assert called is False
    finally:
        if writer.stdin is not None:
            writer.stdin.write("release\n")
            writer.stdin.flush()
            writer.stdin.close()
        writer.wait(timeout=10)
    released = _lock_probe(lock_paths, expect_blocked=False)
    assert released.returncode == 0, released.stderr


def test_part2_source_replay_uses_full_frozen_simulator(
    tmp_path: Path,
) -> None:
    pairs = _composition_fixture(tmp_path / "composition-real-source-replay")
    source = json.loads(pairs[0][0].read_text(encoding="utf-8"))
    overlay = json.loads(pairs[0][1].read_text(encoding="utf-8"))
    journal_key = next(iter(overlay["journals"]))
    reference = overlay["journals"][journal_key]
    records = [
        json.loads(line)
        for line in Path(reference["path"]).read_text(encoding="utf-8").splitlines()
    ]
    target_id, trajectory_text, _round_text = journal_key.rsplit("::", 2)
    trajectory_index = int(trajectory_text)
    subject = next(
        row for row in source["subject_routes"] if row["target_id"] == target_id
    )
    replayed = _replay_part2_source_journal(
        records,
        subject=subject,
        trajectory_index=trajectory_index,
        environment_seed=source["common_environment_seeds"][trajectory_index],
        execution_contract=source["execution_contract"],
        label="real frozen source replay",
        global_attempt_ids=set(),
    )
    effective_payload = json.loads(
        Path(
            overlay["sanitized_artifacts"]["effective_trajectory_metrics"]["path"]
        ).read_text(encoding="utf-8")
    )
    expected = next(
        row
        for row in effective_payload["rows"]
        if row["target_id"] == target_id
        and row["trajectory_index"] == trajectory_index
    )
    expected = {
        key: value
        for key, value in expected.items()
        if key in PART2_TRAJECTORY_ROW_KEYS
    }
    assert replayed == expected

    tampered = [dict(row) for row in records]
    semantic = next(row for row in tampered if row["event"] == "semantic_result")
    semantic["controls"] = {"temperature": 0.125}
    with pytest.raises(DefinitiveAnalysisError, match="replay failed"):
        _replay_part2_source_journal(
            tampered,
            subject=subject,
            trajectory_index=trajectory_index,
            environment_seed=source["common_environment_seeds"][trajectory_index],
            execution_contract=source["execution_contract"],
            label="tampered frozen source replay",
            global_attempt_ids=set(),
        )


def test_part2_composition_cli_preserves_order_and_legacy_mode() -> None:
    required = [
        "--part0", "p0", "--part1", "p1", "--role-calibration", "role",
        "--sensitivity", "sensitivity", "--output-dir", "out",
    ]
    composed = _parser().parse_args(
        [
            *required,
            "--part2-source-overlay", "main-source", "main-overlay",
            "--part2-source-overlay", "nem-source", "nem-overlay",
            "--part2-source-overlay", "deep-source", "deep-overlay",
            "--part2-declared-exclusion", PART2_100DAY_DECLARED_EXCLUSION,
        ]
    )
    assert composed.part2 is None
    assert composed.part2_source_overlay_pairs == [
        [Path("main-source"), Path("main-overlay")],
        [Path("nem-source"), Path("nem-overlay")],
        [Path("deep-source"), Path("deep-overlay")],
    ]
    legacy = _parser().parse_args([*required, "--part2", "legacy-part2"])
    assert legacy.part2 == Path("legacy-part2")
    assert legacy.part2_source_overlay_pairs == []


def test_incomplete_manifest_fails_before_output(tmp_path: Path) -> None:
    incomplete = ROOT / "data/private/inference_hub/provider-safe-v2-part0-large-n48-main22-v1/private/manifest.json"
    if not incomplete.exists():
        pytest.skip("Private running campaign is not available.")
    output = tmp_path / "must-not-exist"
    with pytest.raises(DefinitiveAnalysisError, match="not COMPLETE"):
        analyze(part0=incomplete, part1=incomplete, part2=incomplete, role_calibration=incomplete, sensitivity=incomplete, output_dir=output)
    assert not output.exists()


def test_role_operational_overlay_replaces_only_transport_nulls(tmp_path: Path) -> None:
    run = tmp_path / "role-overlay"
    manifest, original_path, _ = _role_overlay_fixture(run)
    original_sha = _sha_file(original_path)

    effective = _validate_role_journals(run, manifest)
    rows = effective["role/model"]
    assert len(rows) == 2
    assert rows[0]["trial_id"] == "visible-invalid"
    assert rows[0]["raw_response"] is not None
    assert rows[0]["format_valid"] is False
    assert rows[1]["trial_id"] == "transport-null"
    assert rows[1]["raw_response"] is not None
    assert rows[1]["format_valid"] is True
    assert rows[1]["welfare_preserving"] is True
    assert _sha_file(original_path) == original_sha


def test_role_operational_overlay_rejects_visible_semantic_replacement(
    tmp_path: Path,
) -> None:
    run = tmp_path / "role-overlay-semantic"
    manifest, _, _ = _role_overlay_fixture(
        run, replace_visible_semantic_invalid=True
    )
    with pytest.raises(
        DefinitiveAnalysisError, match="operational-repair binding failed"
    ):
        _validate_role_journals(run, manifest)


def test_terminalized_part0_policy_is_explicit_ledger_bound_and_all_scheduled(
    production_bundle: Mapping[str, Path], tmp_path: Path,
) -> None:
    terminalized = _terminalized_part0_run(tmp_path / "valid")
    changed = dict(production_bundle)
    changed["part0"] = terminalized
    with pytest.raises(DefinitiveAnalysisError, match="not COMPLETE"):
        _run(changed, tmp_path / "strict-out")

    result = analyze(
        part0=terminalized,
        part1=changed["part1"],
        part2=changed["part2"],
        role_calibration=changed["role"],
        sensitivity=changed["sensitivity"],
        output_dir=tmp_path / "terminalized-out",
        allow_terminalized_part0_operational_invalids=True,
    )
    audit = result["part0_terminalized_operational_audit"]
    assert audit["scheduled_units"] == 3_168
    assert audit["operational_failure_units"] == 41
    assert audit["subject_transport_failure_units"] == 41
    assert audit["judge_transport_failure_units"] == 0
    assert audit["visible_subject_response_units"] == 3_127
    assert audit["affected_target_count"] == 8
    assert audit["semantics"] == (
        "operational_nonsuccess_not_refusal_not_semantic_model_output"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "terminalized-out/part0_models.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 22
    assert sum(row["scheduled_units"] for row in rows) == 3_168
    assert sum(row["operational_invalid_count"] for row in rows) == 41
    assert sum(row["visible_subject_response_count"] for row in rows) == 3_127

    drifted = _terminalized_part0_run(
        tmp_path / "drifted", drift_request_hash=True
    )
    with pytest.raises(DefinitiveAnalysisError, match="retry lineage changed"):
        analyze(
            part0=drifted,
            part1=changed["part1"],
            part2=changed["part2"],
            role_calibration=changed["role"],
            sensitivity=changed["sensitivity"],
            output_dir=tmp_path / "drifted-out",
            allow_terminalized_part0_operational_invalids=True,
        )


def test_hash_tamper_and_judge_overlap_fail_closed(
    production_bundle: Mapping[str, Path], tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered.json"
    shutil.copyfile(production_bundle["part0"], tampered)
    os.chmod(tampered, 0o600)
    value = json.loads(tampered.read_text())
    value["summary"]["planned_units"] += 1
    _write_json(tampered, value, private=True)
    changed = dict(production_bundle)
    changed["part0"] = tampered
    with pytest.raises(DefinitiveAnalysisError, match="self-hash failed"):
        _run(changed, tmp_path / "hash-out")
    assert not (tmp_path / "hash-out").exists()

    overlap_run = tmp_path / "overlap"
    shutil.copytree(production_bundle["part1"].parents[1], overlap_run)
    overlap_manifest = overlap_run / "private/manifest.json"
    value = json.loads(overlap_manifest.read_text())
    value["judge_reservation"]["target_id"] = value["subject_routes"][0]["target_id"]
    _write_json(overlap_manifest, _seal(value), private=True)
    changed = dict(production_bundle)
    changed["part1"] = overlap_manifest
    with pytest.raises(DefinitiveAnalysisError, match="judge overlaps"):
        _run(changed, tmp_path / "overlap-out")
    assert not (tmp_path / "overlap-out").exists()


def test_campaign_specific_accelerated_launcher_and_policy_bindings() -> None:
    safe = ROOT / "experiments/misc/inference_hub_provider_safe_v2.py"
    main = ROOT / "experiments/misc/inference_hub_main_accelerated.py"
    deadline = ROOT / "experiments/misc/inference_hub_part1_deadline_accelerated.py"
    part0_deadline = ROOT / "experiments/misc/inference_hub_part0_deadline_retry.py"
    sensitivity_deadline = (
        ROOT / "experiments/misc/inference_hub_sensitivity_deadline_accelerated.py"
    )
    exploratory = ROOT / "experiments/misc/inference_hub_exploratory_accelerated.py"

    def manifest(
        launcher: Path,
        concurrency: int,
        rps: float,
        *,
        global_concurrency: int = 12,
        global_rps: float = 8.0,
    ) -> dict[str, Any]:
        shared = {
            "schema_version": 2,
            "algorithm": "cross_process_provider_aware_leaky_bucket_with_leases_all_http_5xx_full_throttle_cooldown",
            "global_concurrency": global_concurrency, "provider_concurrency": concurrency,
            "global_requests_per_second": global_rps, "provider_requests_per_second": rps,
            "lease_seconds": 900.0, "poll_seconds": 0.05,
            "throttle_cooldown_seconds": 30.0, "transient_cooldown_seconds": 5.0,
        }
        shared["policy_sha256"] = hashlib.sha256(
            json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "source_artifacts": {str(safe): _sha_file(safe), str(launcher): _sha_file(launcher)},
            "execution_contract": {"shared_rate_limit": shared},
        }

    main_manifest = manifest(main, 2, 1.5)
    exploratory_manifest = manifest(exploratory, 3, 2.0)
    deadline_manifest = manifest(
        deadline, 4, 2.5, global_concurrency=24, global_rps=12.0
    )
    part0_deadline_manifest = manifest(
        part0_deadline, 3, 2.0, global_concurrency=16, global_rps=10.0
    )
    sensitivity_deadline_manifest = manifest(
        sensitivity_deadline, 3, 2.5, global_concurrency=24, global_rps=12.0
    )
    _provider_safe_contract(main_manifest, "part0")
    _provider_safe_contract(exploratory_manifest, "role")
    _provider_safe_contract(deadline_manifest, "part1")
    _provider_safe_contract(part0_deadline_manifest, "part0")
    _provider_safe_contract(sensitivity_deadline_manifest, "sensitivity")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(main_manifest, "sensitivity")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(deadline_manifest, "part0")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(part0_deadline_manifest, "part1")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(sensitivity_deadline_manifest, "role")
    exploratory_manifest["execution_contract"]["shared_rate_limit"]["provider_requests_per_second"] = 2.1
    with pytest.raises(DefinitiveAnalysisError, match="policy hash failed"):
        _provider_safe_contract(exploratory_manifest, "sensitivity")
