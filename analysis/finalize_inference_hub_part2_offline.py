"""Fail-closed offline finalization for an interrupted hosted Part 2 panel.

The finalizer never dispatches a model request.  It acquires the panel run lock,
validates the frozen inputs and every journal hash chain, replays complete
retained trajectories through the original simulator, and emits behavioral
metrics only for those retained complete trajectories.  Explicitly unavailable
targets contribute operational failure evidence only.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.build_final_results import FinalResultsError, _reject_text_keys
from experiments.misc.inference_hub_part1_panel import (
    _ChainedJournal,
    _atomic_json,
    _read_json,
    _require_mode,
    _safe_file_stem,
    _seal,
    _self_hash,
    _sha256_file,
    _sha256_json,
    _validate_checkpoint_reference,
    InferenceHubPart1PanelError,
    select_routes,
)
from experiments.misc.inference_hub_part2_panel import (
    SCHEMA_VERSION,
    _SOURCE_PATHS,
    InferenceHubPart2PanelError,
    Part2Contract,
    _aggregate_models,
    _environment_seeds,
    _load_panel,
    _run_trajectory,
    _utc_now,
)


ARTIFACT_TYPE = "inference_hub_part2_corrected_matched_panel"
TRAJECTORY_ARTIFACT_TYPE = "inference_hub_part2_sanitized_trajectory_metrics"
MODEL_ARTIFACT_TYPE = "inference_hub_part2_sanitized_model_metrics"


class OfflinePart2FinalizationError(RuntimeError):
    """An interrupted panel cannot be finalized without inventing evidence."""


class _OfflineReplayRequired(RuntimeError):
    """A replay reached a participant unit absent from the retained journal."""


class _ReadOnlyJournal:
    """Journal-shaped replay input that makes every attempted append fatal."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self.records = [dict(row) for row in records]

    def append(self, _payload: Mapping[str, Any]) -> dict[str, Any]:
        raise _OfflineReplayRequired("Offline replay reached an unretained unit.")


def _strings(values: Sequence[str], *, label: str) -> list[str]:
    output: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise OfflinePart2FinalizationError(f"{label} is empty or invalid.")
        if value in output:
            raise OfflinePart2FinalizationError(f"{label} is duplicated: {value}.")
        output.append(value)
    if not output:
        raise OfflinePart2FinalizationError("At least one unavailable target is required.")
    return output


def _acquire_offline_lock(private_dir: Path):
    path = private_dir / ".run.lock"
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise OfflinePart2FinalizationError(
            "The Part 2 run lock is held; the live runner must exit first."
        ) from error
    return handle


def _validate_file_binding(
    reference: Any, *, label: str, canonical: Mapping[str, Any] | None = None,
    require_evidence_hash: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(reference, Mapping):
        raise OfflinePart2FinalizationError(f"{label} binding is absent.")
    path_value = reference.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise OfflinePart2FinalizationError(f"{label} path binding is absent.")
    path = Path(path_value).resolve()
    value = _read_json(path, label)
    if reference.get("file_sha256") != _sha256_file(path):
        raise OfflinePart2FinalizationError(f"{label} file hash changed.")
    if canonical is not None and reference.get("canonical_sha256") != _sha256_json(canonical):
        raise OfflinePart2FinalizationError(f"{label} canonical binding changed.")
    if require_evidence_hash and (
        value.get("evidence_sha256") != _self_hash(value)
        or reference.get("evidence_sha256") != value.get("evidence_sha256")
    ):
        raise OfflinePart2FinalizationError(f"{label} evidence binding changed.")
    return path, value


def _validate_frozen_inputs(
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Part2Contract, list[int]]:
    inputs = manifest.get("input_artifacts")
    if not isinstance(inputs, Mapping):
        raise OfflinePart2FinalizationError("Manifest input bindings are absent.")
    panel_ref = inputs.get("panel")
    if not isinstance(panel_ref, Mapping) or not isinstance(panel_ref.get("path"), str):
        raise OfflinePart2FinalizationError("Panel input binding is absent.")
    panel_path = Path(str(panel_ref["path"])).resolve()
    panel, frozen_contract = _load_panel(panel_path)
    _validate_file_binding(panel_ref, label="panel", canonical=panel)
    _, compatibility = _validate_file_binding(
        inputs.get("compatibility"), label="compatibility", require_evidence_hash=True,
    )
    _, registry = _validate_file_binding(
        inputs.get("registry"), label="registry", canonical=_read_json(
            Path(str(inputs.get("registry", {}).get("path", ""))).resolve(), "registry"
        ),
    )
    selected_rows = manifest.get("subject_routes")
    if not isinstance(selected_rows, list) or not selected_rows:
        raise OfflinePart2FinalizationError("Manifest subject routes are absent.")
    selected_ids = [
        row.get("target_id") for row in selected_rows if isinstance(row, Mapping)
    ]
    if len(selected_ids) != len(selected_rows) or any(
        not isinstance(target_id, str) or not target_id for target_id in selected_ids
    ) or len(set(selected_ids)) != len(selected_ids):
        raise OfflinePart2FinalizationError("Manifest subject identities are invalid.")
    subjects, judge = select_routes(
        registry=registry, compatibility=compatibility,
        selected_ids=selected_ids, judge_target_id=str(panel["judge_target_id"]),
    )
    reconstructed_subjects = [{
        "target_id": row["target_id"], "upstream_provider": row["upstream_provider"],
        "model": row["model"], "route": row["route"],
        "candidate_index": row["candidate_index"],
        "supported_controls": row["supported_controls"],
        "selected_profile_id": row["selected_profile_id"],
        "selected_profile_request_sha256": row["selected_profile_request_sha256"],
    } for row in subjects]
    if reconstructed_subjects != selected_rows:
        raise OfflinePart2FinalizationError("Subject route selection differs from frozen inputs.")
    expected_judge = {
        "target_id": judge["target_id"], "upstream_provider": judge["upstream_provider"],
        "model": judge["model"], "route": judge["route"],
        "dispatch_permitted_in_this_runner": False,
        "role": "fixed_disjoint_judge_reserved_for_cross_axis_analysis",
    }
    if manifest.get("judge_reservation") != expected_judge:
        raise OfflinePart2FinalizationError("Judge reservation differs from frozen inputs.")

    contract_row = manifest.get("part2_contract")
    if not isinstance(contract_row, Mapping):
        raise OfflinePart2FinalizationError("Part 2 contract is absent.")
    trajectory_count = contract_row.get("independent_trajectories")
    if (
        not isinstance(trajectory_count, int) or isinstance(trajectory_count, bool)
        or not 1 <= trajectory_count <= frozen_contract.trajectories
    ):
        raise OfflinePart2FinalizationError("Trajectory count is outside the frozen contract.")
    expected_contract = {
        "society_size": frozen_contract.society_size,
        "days": frozen_contract.days,
        "independent_trajectories": trajectory_count,
        "resource_capacity": frozen_contract.capacity,
        "option_a_private_gain": 1,
        "option_b_private_gain": frozen_contract.private_gain,
        "option_b_reserve_cost": frozen_contract.reserve_cost,
        "unanimous_a_group_payoff": frozen_contract.community_benefit,
        "unanimous_b_group_payoff": -frozen_contract.community_benefit,
        "invalid_policy": "retain_as_INVALID_zero_effect_no_semantic_retry",
        "collapse_death_rate": frozen_contract.collapse_death_rate,
        "attrition_policy": "matched_seed_day_random_sample_v1",
    }
    if dict(contract_row) != expected_contract:
        raise OfflinePart2FinalizationError("Part 2 execution contract changed.")
    base_seed = manifest.get("base_seed")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise OfflinePart2FinalizationError("Base seed is invalid.")
    environment_seeds = _environment_seeds(
        str(panel["panel_id"]), base_seed, trajectory_count,
    )
    if manifest.get("common_environment_seeds") != environment_seeds:
        raise OfflinePart2FinalizationError("Common environment seeds changed.")
    contract = Part2Contract(
        society_size=frozen_contract.society_size,
        days=frozen_contract.days,
        trajectories=trajectory_count,
        capacity=frozen_contract.capacity,
        private_gain=frozen_contract.private_gain,
        reserve_cost=frozen_contract.reserve_cost,
        community_benefit=frozen_contract.community_benefit,
        collapse_death_rate=frozen_contract.collapse_death_rate,
    )
    return subjects, contract, environment_seeds


def _validate_source_bindings(
    manifest: Mapping[str, Any], *, source_verification_root: Path | None,
) -> dict[str, Any]:
    recorded = manifest.get("source_artifacts")
    expected_paths = {str(path.resolve()) for path in _SOURCE_PATHS}
    if not isinstance(recorded, Mapping) or set(recorded) != expected_paths:
        raise OfflinePart2FinalizationError("Runner source binding set changed.")
    repository_root = Path(__file__).resolve().parents[1]
    verification_root = (
        repository_root
        if source_verification_root is None
        else source_verification_root.resolve()
    )
    if not verification_root.is_dir():
        raise OfflinePart2FinalizationError("Source verification root is absent.")
    direct_replay_paths = {
        Path(__file__).resolve().parents[1]
        / "experiments/misc/inference_hub_part2_panel.py",
        Path(__file__).resolve().parents[1] / "experiments/part2/part_2.py",
        Path(__file__).resolve().parents[1] / "agents/agent_2.py",
        Path(__file__).resolve().parents[1] / "experiments/part2/part_2_prompt.json",
    }
    verified: list[dict[str, str]] = []
    for path_value, digest in recorded.items():
        bound_path = Path(str(path_value)).resolve()
        try:
            relative = bound_path.relative_to(repository_root)
        except ValueError as error:
            raise OfflinePart2FinalizationError(
                "Manifest-bound source lies outside the repository root."
            ) from error
        candidate = (verification_root / relative).resolve()
        try:
            candidate.relative_to(verification_root)
        except ValueError as error:
            raise OfflinePart2FinalizationError(
                "Source verification path escapes its explicit root."
            ) from error
        if not candidate.is_file():
            raise OfflinePart2FinalizationError(
                f"Source verification file is missing: {relative.as_posix()}."
            )
        if digest != _sha256_file(candidate):
            raise OfflinePart2FinalizationError(
                f"Source verification bytes changed: {relative.as_posix()}."
            )
        # These files directly define replayed prompts, state transitions, and
        # metrics.  An alternate tree can supply historical support-module
        # provenance, but it cannot authorize execution of changed P2 logic.
        if bound_path in direct_replay_paths and digest != _sha256_file(bound_path):
            raise OfflinePart2FinalizationError(
                f"Loaded P2 replay code differs from recorded bytes: {relative.as_posix()}."
            )
        verified.append({
            "relative_path": relative.as_posix(), "file_sha256": str(digest),
        })
    return {
        "mode": (
            "current_repository_root"
            if verification_root == repository_root
            else "explicit_clean_source_root"
        ),
        # This remains in the private manifest. Sanitized outputs contain no
        # source-root paths.
        "verification_root": str(verification_root),
        "verified_files": sorted(verified, key=lambda row: row["relative_path"]),
    }


def _journal_failure_counts(records: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    identity = transport = 0
    for row in records:
        if row.get("event") != "semantic_result":
            continue
        reason = row.get("invalid_reason")
        if reason == "response_model_identity_mismatch":
            identity += 1
        elif reason == "transport_failure_exhausted":
            transport += 1
    return identity, transport


def _validate_journal_identity(
    records: Sequence[Mapping[str, Any]], *, subject: Mapping[str, Any],
    trajectory_index: int, environment_seed: int,
) -> None:
    for row in records:
        event = row.get("event")
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("artifact_type") != "inference_hub_part2_trajectory_event"
            or row.get("target_id") != subject["target_id"]
            or row.get("trajectory_index") != trajectory_index
        ):
            raise OfflinePart2FinalizationError("Trajectory journal identity changed.")
        if event in {"reserved_before_dispatch", "semantic_result"} and (
            row.get("upstream_provider") != subject["upstream_provider"]
            or row.get("model") != subject["model"]
            or row.get("requested_route") != subject["route"]
            or row.get("environment_seed") != environment_seed
        ):
            raise OfflinePart2FinalizationError("Trajectory route/seed binding changed.")
        if event != "semantic_result":
            continue
        reason = row.get("invalid_reason")
        if reason == "response_model_identity_mismatch":
            if (
                row.get("model_identity_valid") is not False
                or not isinstance(row.get("raw_response"), Mapping)
                or row.get("response_model") == subject["route"]
                or row.get("failure") is not None
                or row.get("action") != "INVALID"
            ):
                raise OfflinePart2FinalizationError("Identity-failure evidence is inconsistent.")
        elif reason == "transport_failure_exhausted":
            failure = row.get("failure")
            if (
                row.get("model_identity_valid") is not False
                or row.get("raw_response") is not None
                or row.get("response_model") is not None
                or not isinstance(failure, Mapping)
                or not isinstance(failure.get("failure_code"), str)
                or row.get("action") != "INVALID"
            ):
                raise OfflinePart2FinalizationError("Transport-failure evidence is inconsistent.")
        elif row.get("failure") is not None:
            raise OfflinePart2FinalizationError("Non-operational result carries failure evidence.")


def _operational_evidence_row(
    *, subject: Mapping[str, Any], trajectory_index: int,
    identity_count: int, transport_count: int, complete: bool,
) -> dict[str, Any] | None:
    if not complete and identity_count == transport_count == 0:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "target_id": subject["target_id"],
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "trajectory_index": trajectory_index,
        "operationally_eligible": complete and identity_count == transport_count == 0,
        "identity_mismatch_count": identity_count,
        "transport_failure_count": transport_count,
        "evidence_scope": "operational_only_no_behavioral_metrics",
    }


def _offline_replay(
    *, subject: Mapping[str, Any], trajectory_index: int, environment_seed: int,
    contract: Part2Contract, records: Sequence[Mapping[str, Any]],
    execution_contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    try:
        return _run_trajectory(
            subject=subject, trajectory_index=trajectory_index,
            environment_seed=environment_seed, contract=contract,
            journal=_ReadOnlyJournal(records), client=object(),
            participant_workers=int(execution_contract["participant_workers"]),
            max_attempts=int(execution_contract["max_transport_attempts"]),
            initial_backoff_seconds=float(
                execution_contract["initial_exponential_backoff_seconds"]
            ),
            sleep_fn=lambda _seconds: None,
        )
    except _OfflineReplayRequired:
        return None


def finalize_part2_offline(
    manifest_path: Path, *, unavailable_targets: Sequence[str],
    source_verification_root: Path | None = None,
) -> dict[str, Any]:
    """Finalize an interrupted Part 2 panel without dispatching any requests."""
    unavailable = set(_strings(unavailable_targets, label="Unavailable target"))
    manifest_path = manifest_path.resolve()
    private_dir = manifest_path.parent
    if manifest_path.name != "manifest.json" or private_dir.name != "private":
        raise OfflinePart2FinalizationError("Expected an existing private/manifest.json path.")
    _require_mode(private_dir, 0o700)
    lock = _acquire_offline_lock(private_dir)
    try:
        manifest_file_sha256 = _sha256_file(manifest_path)
        manifest = _read_json(manifest_path, "Part 2 private manifest")
        _require_mode(manifest_path, 0o600)
        if (
            manifest.get("schema_version") != SCHEMA_VERSION
            or manifest.get("artifact_type") != ARTIFACT_TYPE
            or manifest.get("evidence_sha256") != _self_hash(manifest)
        ):
            raise OfflinePart2FinalizationError("Part 2 manifest schema/type/self-hash failed.")
        if manifest.get("complete") is not False:
            raise OfflinePart2FinalizationError("Offline finalization requires an incomplete manifest.")
        if manifest.get("sanitized_artifacts") not in ({}, None):
            raise OfflinePart2FinalizationError("Sanitized artifacts already exist; refusing overwrite.")
        source_verification = _validate_source_bindings(
            manifest, source_verification_root=source_verification_root,
        )
        subjects, contract, environment_seeds = _validate_frozen_inputs(manifest)
        subject_by_id = {str(row["target_id"]): row for row in subjects}
        unknown = unavailable - set(subject_by_id)
        if unknown:
            raise OfflinePart2FinalizationError(
                "Unavailable target is absent from the manifest: " + ",".join(sorted(unknown))
            )
        retained = set(subject_by_id) - unavailable
        if not retained:
            raise OfflinePart2FinalizationError("At least one retained target is required.")
        execution = manifest.get("execution_contract")
        if not isinstance(execution, Mapping) or any(
            not isinstance(execution.get(key), int)
            or isinstance(execution.get(key), bool)
            or int(execution[key]) < 1
            for key in ("participant_workers", "max_transport_attempts")
        ) or not isinstance(execution.get("initial_exponential_backoff_seconds"), (int, float)):
            raise OfflinePart2FinalizationError("Execution replay controls are invalid.")

        refs = manifest.get("journals")
        expected_keys = {
            f"{target_id}::{index}"
            for target_id in subject_by_id for index in range(contract.trajectories)
        }
        if not isinstance(refs, Mapping) or set(refs) != expected_keys:
            raise OfflinePart2FinalizationError("Manifest trajectory checkpoint set changed.")
        journals: dict[tuple[str, int], _ChainedJournal] = {}
        initial_references: dict[str, dict[str, Any]] = {}
        retained_trajectories: list[dict[str, Any]] = []
        unavailable_evidence: dict[str, list[dict[str, Any]]] = {
            target_id: [] for target_id in unavailable
        }
        completed_count = 0
        unavailable_scheduled = unavailable_received = unavailable_invalid = 0

        for target_id, subject in subject_by_id.items():
            expected_directory = (
                private_dir / "trajectories" / _safe_file_stem(target_id)
            ).resolve()
            for index in range(contract.trajectories):
                key = f"{target_id}::{index}"
                reference = refs[key]
                if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
                    raise OfflinePart2FinalizationError(f"Journal reference is absent for {key}.")
                journal_path = Path(str(reference["path"])).resolve()
                if journal_path != expected_directory / f"seed-{index:03d}.jsonl":
                    raise OfflinePart2FinalizationError(f"Journal path changed for {key}.")
                journal = _ChainedJournal(journal_path)
                _validate_checkpoint_reference(journal, reference, label=f"trajectory {key}")
                _validate_journal_identity(
                    journal.records, subject=subject, trajectory_index=index,
                    environment_seed=environment_seeds[index],
                )
                journals[(target_id, index)] = journal
                initial_references[key] = journal.reference()
                replayed = _offline_replay(
                    subject=subject, trajectory_index=index,
                    environment_seed=environment_seeds[index], contract=contract,
                    records=journal.records, execution_contract=execution,
                )
                if target_id in retained:
                    if replayed is None:
                        raise OfflinePart2FinalizationError(
                            f"Retained target has an incomplete trajectory: {key}."
                        )
                    if replayed.get("operationally_eligible") is not True:
                        raise OfflinePart2FinalizationError(
                            f"Retained target has operational failure evidence: {key}."
                        )
                    retained_trajectories.append(replayed)
                    completed_count += 1
                    continue
                identity, transport = _journal_failure_counts(journal.records)
                complete = replayed is not None
                if complete:
                    completed_count += 1
                semantic_rows = [
                    row for row in journal.records
                    if row.get("event") == "semantic_result"
                ]
                unavailable_scheduled += len(semantic_rows)
                unavailable_received += sum(
                    isinstance(row.get("raw_response"), Mapping)
                    for row in semantic_rows
                )
                unavailable_invalid += sum(
                    row.get("action") == "INVALID" for row in semantic_rows
                )
                evidence = _operational_evidence_row(
                    subject=subject, trajectory_index=index,
                    identity_count=identity, transport_count=transport,
                    complete=complete,
                )
                if evidence is not None:
                    unavailable_evidence[target_id].append(evidence)

        for target_id, rows in unavailable_evidence.items():
            if not rows or sum(
                int(row["identity_mismatch_count"]) + int(row["transport_failure_count"])
                for row in rows
            ) == 0:
                raise OfflinePart2FinalizationError(
                    f"Unavailable target lacks identity/transport failure evidence: {target_id}."
                )

        trajectory_rows = list(retained_trajectories)
        for target_id in sorted(unavailable_evidence):
            trajectory_rows.extend(unavailable_evidence[target_id])
        trajectory_rows.sort(key=lambda row: (str(row["target_id"]), int(row["trajectory_index"])))
        retained_subjects = [row for row in subjects if row["target_id"] in retained]
        model_rows = _aggregate_models(
            retained_trajectories, retained_subjects,
            expected_trajectories=contract.trajectories, capacity=contract.capacity,
        )
        for target_id in sorted(unavailable_evidence):
            subject = subject_by_id[target_id]
            rows = unavailable_evidence[target_id]
            model_rows.append({
                "schema_version": SCHEMA_VERSION,
                "target_id": target_id,
                "upstream_provider": subject["upstream_provider"],
                "model": subject["model"],
                "trajectory_count": len(rows),
                "eligible_trajectory_count": sum(
                    row["operationally_eligible"] is True for row in rows
                ),
                "expected_trajectory_count": contract.trajectories,
                "complete_matched_panel": False,
                "total_identity_mismatch_count": sum(
                    int(row["identity_mismatch_count"]) for row in rows
                ),
                "total_transport_failure_count": sum(
                    int(row["transport_failure_count"]) for row in rows
                ),
                "evidence_scope": "operational_only_no_behavioral_metrics",
            })
        model_rows.sort(key=lambda row: str(row["target_id"]))
        trajectory_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": TRAJECTORY_ARTIFACT_TYPE,
            "panel_id": manifest.get("panel_id"),
            "generated_at_utc": _utc_now(),
            "independence_unit": "target_by_environment_seed_trajectory",
            "offline_finalization": True,
            "rows": trajectory_rows,
        }
        model_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": MODEL_ARTIFACT_TYPE,
            "panel_id": manifest.get("panel_id"),
            "generated_at_utc": _utc_now(),
            "uncertainty_unit": "independent_trajectory",
            "offline_finalization": True,
            "rows": model_rows,
        }
        _reject_text_keys(trajectory_payload)
        _reject_text_keys(model_payload)
        _seal(trajectory_payload)
        _seal(model_payload)

        # Recheck every byte binding immediately before the first write.
        if _sha256_file(manifest_path) != manifest_file_sha256:
            raise OfflinePart2FinalizationError("Manifest changed during offline finalization.")
        final_references = {
            f"{target_id}::{index}": journal.reference()
            for (target_id, index), journal in journals.items()
        }
        if final_references != initial_references:
            raise OfflinePart2FinalizationError("A journal changed during offline finalization.")

        output_dir = private_dir.parent
        sanitized_dir = output_dir / "sanitized"
        trajectory_path = sanitized_dir / "trajectory_metrics.json"
        model_path = sanitized_dir / "model_metrics.json"
        if trajectory_path.exists() or model_path.exists():
            raise OfflinePart2FinalizationError("Sanitized output exists; refusing overwrite.")
        sanitized_dir.mkdir(mode=0o700, exist_ok=True)
        _require_mode(sanitized_dir, 0o700)
        _atomic_json(trajectory_path, trajectory_payload)
        _atomic_json(model_path, model_payload)

        identity_total = sum(
            int(row["identity_mismatch_count"]) for row in trajectory_rows
        )
        transport_total = sum(
            int(row["transport_failure_count"]) for row in trajectory_rows
        )
        manifest["summary"] = {
            "planned_trajectories": len(subjects) * contract.trajectories,
            "completed_trajectories": completed_count,
            "planned_maximum_agent_days": (
                len(subjects) * contract.trajectories
                * contract.days * contract.society_size
            ),
            "scheduled_agent_days": sum(
                int(row["scheduled_agent_days"]) for row in retained_trajectories
            ) + unavailable_scheduled,
            "responses_received": sum(
                int(row["responses_received"]) for row in retained_trajectories
            ) + unavailable_received,
            "invalid_count": sum(
                int(row["invalid_count"]) for row in retained_trajectories
            ) + unavailable_invalid,
            "identity_mismatch_count": identity_total,
            "transport_failure_count": transport_total,
            "eligible_trajectories": sum(
                row.get("operationally_eligible") is True for row in trajectory_rows
            ),
            "offline_finalized_retained_targets": len(retained),
            "offline_declared_unavailable_targets": len(unavailable),
        }
        manifest["complete"] = False
        manifest["journals"] = final_references
        manifest["sanitized_artifacts"] = {
            "trajectory_metrics": {
                "path": str(trajectory_path.resolve()),
                "file_sha256": _sha256_file(trajectory_path),
                "evidence_sha256": trajectory_payload["evidence_sha256"],
            },
            "model_metrics": {
                "path": str(model_path.resolve()),
                "file_sha256": _sha256_file(model_path),
                "evidence_sha256": model_payload["evidence_sha256"],
            },
        }
        manifest["offline_finalization"] = {
            "status": "retained_complete_unavailable_operational_evidence_only",
            "unavailable_target_ids": sorted(unavailable),
            "script": Path(__file__).name,
            "script_file_sha256": _sha256_file(Path(__file__).resolve()),
            "source_verification": source_verification,
        }
        manifest["last_updated_at_utc"] = _utc_now()
        _seal(manifest)
        _atomic_json(manifest_path, manifest)
        return manifest
    except (
        FinalResultsError, InferenceHubPart1PanelError, InferenceHubPart2PanelError,
        OSError, ValueError, KeyError, TypeError,
    ) as error:
        if isinstance(error, OfflinePart2FinalizationError):
            raise
        raise OfflinePart2FinalizationError(str(error)) from error
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finalize an interrupted Part 2 panel without model dispatch."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--unavailable-target", action="append", default=[],
        help="Exact target id with retained identity/transport failure evidence.",
    )
    parser.add_argument(
        "--source-verification-root", type=Path,
        help=(
            "Explicit clean repository tree used to verify manifest-bound source bytes; "
            "production inputs and journals remain at their original paths."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = finalize_part2_offline(
            args.manifest, unavailable_targets=args.unavailable_target,
            source_verification_root=args.source_verification_root,
        )
    except OfflinePart2FinalizationError as error:
        print(f"Offline Part 2 finalization failed: {error}", file=os.sys.stderr)
        return 2
    print(f"Offline Part 2 manifest: {args.manifest}")
    print(f"Operational failures: {manifest['summary']['identity_mismatch_count'] + manifest['summary']['transport_failure_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
