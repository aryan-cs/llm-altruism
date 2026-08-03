"""Retire one failed Part 0/Part 1 target before an offline resume.

This tool never dispatches a request and never assigns a behavioral outcome.
It may run only after the panel process has stopped and its run lock is free.
The selected target must already have target-bound, durably reserved and
completed transport-failure evidence or a retained response-model identity
mismatch.  Remaining scheduled units receive explicit non-dispatched terminal
retirement records so a resumed runner can finish healthy targets without
redispatching the retired target.  With an explicit recovery flag, every
unmatched in-flight reservation across all targets is first closed using the
same transient stale-completion record as ordinary runner resume recovery;
healthy targets receive no terminal record and remain scheduled for retry.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from experiments.misc.inference_hub_part0_panel import (
    _SOURCE_PATHS as PART0_SOURCE_PATHS,
)
from experiments.misc.inference_hub_part1_panel import (
    _ChainedJournal,
    _SOURCE_PATHS as PART1_SOURCE_PATHS,
    _atomic_json,
    _safe_file_stem,
    _seal,
    _self_hash,
    _sha256_file,
    _sha256_json,
    build_draft_trials,
)
from experiments.misc.inference_hub_part1_stratified_panel import (
    build_stratified_trials,
)


SCHEMA_VERSION = 1
PART0_TYPE = "inference_hub_part0_accelerated_private_panel"
PART1_TYPE = "inference_hub_part1_large_n_exploratory_panel"
RETIREMENT_TYPE = "inference_hub_offline_target_retirement"
RETIREMENT_PROVENANCE = "offline_target_bound_operational_retirement"
RETIREMENT_FAILURE_CODE = "operational_target_retired"
_DISPATCH_FAILURE_CODES = {"http_error", "connection_error", "connection_timeout"}
PART1_STRATIFIED_SOURCE = Path(__file__).with_name(
    "inference_hub_part1_stratified_panel.py"
).resolve()


class TargetRetirementError(RuntimeError):
    """The artifact is unsafe, ineligible, active, or internally inconsistent."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TargetRetirementError(f"Manifest is unreadable: {path}") from error
    if not isinstance(value, dict):
        raise TargetRetirementError("Manifest is not an object.")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise TargetRetirementError("Manifest schema version is unsupported.")
    if value.get("artifact_type") not in {PART0_TYPE, PART1_TYPE}:
        raise TargetRetirementError("Manifest is not a supported Part 0/Part 1 artifact.")
    if value.get("evidence_sha256") != _self_hash(value):
        raise TargetRetirementError("Manifest self-hash failed.")
    if value.get("complete") is True:
        raise TargetRetirementError("A complete panel cannot be operationally retired.")
    return value


@contextmanager
def _exclusive_run_lock(private_dir: Path) -> Iterator[None]:
    path = private_dir / ".run.lock"
    handle = path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TargetRetirementError(
                "Panel run lock is live; stop the runner before retirement."
            ) from error
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _subjects(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = manifest.get("subject_routes")
    if not isinstance(rows, list) or not rows:
        raise TargetRetirementError("Manifest subject routes are absent.")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise TargetRetirementError("Manifest subject route is not an object.")
        target_id = row.get("target_id")
        if (
            not isinstance(target_id, str)
            or not target_id
            or target_id in result
            or not all(isinstance(row.get(key), str) and row.get(key) for key in (
                "upstream_provider", "model", "route",
            ))
        ):
            raise TargetRetirementError("Manifest subject identities are invalid.")
        result[target_id] = row
    return result


def _journal_from_reference(
    *, private_dir: Path, reference: object, expected_path: Path, label: str,
) -> _ChainedJournal:
    if not isinstance(reference, Mapping):
        raise TargetRetirementError(f"Manifest lacks {label} checkpoint.")
    recorded_path = reference.get("path")
    if (
        not isinstance(recorded_path, str)
        or Path(recorded_path).resolve() != expected_path.resolve()
        or expected_path.parent.resolve() not in {
            private_dir.resolve(), (private_dir / "raw_responses").resolve()
        }
    ):
        raise TargetRetirementError(f"Manifest {label} path binding is invalid.")
    try:
        journal = _ChainedJournal(expected_path)
    except Exception as error:
        raise TargetRetirementError(f"{label} hash chain is invalid.") from error
    count = reference.get("record_count")
    tail = reference.get("tail_record_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TargetRetirementError(f"Manifest {label} record count is invalid.")
    if len(journal.records) < count:
        raise TargetRetirementError(f"{label} was truncated after its checkpoint.")
    checkpoint_tail = journal.records[count - 1]["record_sha256"] if count else None
    if checkpoint_tail != tail:
        raise TargetRetirementError(f"{label} checkpoint tail changed.")
    if len(journal.records) == count:
        actual_file_hash = _sha256_file(expected_path) if expected_path.exists() else None
        if reference.get("file_sha256") != actual_file_hash:
            raise TargetRetirementError(f"{label} checkpoint bytes changed.")
    return journal


def _journals(
    manifest: Mapping[str, Any], private_dir: Path, subjects: Mapping[str, Any],
) -> tuple[_ChainedJournal, dict[str, _ChainedJournal]]:
    checkpoints = manifest.get("journals")
    if not isinstance(checkpoints, Mapping):
        raise TargetRetirementError("Manifest journal checkpoints are absent.")
    raw_refs = checkpoints.get("raw_responses")
    if not isinstance(raw_refs, Mapping) or set(raw_refs) != set(subjects):
        raise TargetRetirementError("Manifest raw journal target set changed.")
    ledger = _journal_from_reference(
        private_dir=private_dir,
        reference=checkpoints.get("attempt_ledger"),
        expected_path=private_dir / "attempt_ledger.jsonl",
        label="attempt ledger",
    )
    raw: dict[str, _ChainedJournal] = {}
    for target_id, reference in raw_refs.items():
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise TargetRetirementError(f"Raw journal checkpoint is absent for {target_id}.")
        expected_path = (
            private_dir / "raw_responses" / f"{_safe_file_stem(target_id)}.jsonl"
        )
        raw[target_id] = _journal_from_reference(
            private_dir=private_dir,
            reference=reference,
            expected_path=expected_path,
            label=f"raw responses for {target_id}",
        )
    return ledger, raw


def _source_hash_migrations(
    manifest: Mapping[str, Any], *, part: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Validate every frozen source path and narrowly migrate runner hashes.

    The scheduler must understand the new administrative terminal records, so
    its own source hash necessarily changes.  Part 0 also binds the Part 1
    runner because it imports the shared journal implementation from it.  No
    other frozen source hash is permitted to move.
    """
    recorded = manifest.get("source_artifacts")
    if not isinstance(recorded, Mapping) or not all(
        isinstance(path, str) and isinstance(digest, str)
        for path, digest in recorded.items()
    ):
        raise TargetRetirementError("Manifest source-artifact bindings are invalid.")
    base_paths = tuple(
        path.resolve()
        for path in (PART0_SOURCE_PATHS if part == "part0" else PART1_SOURCE_PATHS)
    )
    allowed_path_sets = [base_paths]
    if part == "part1":
        allowed_path_sets.append((*base_paths, PART1_STRATIFIED_SOURCE))
    expected_paths = next(
        (paths for paths in allowed_path_sets if set(recorded) == {str(path) for path in paths}),
        None,
    )
    if expected_paths is None:
        raise TargetRetirementError("Manifest source-artifact path set changed.")
    allowed_migrations = {
        str(Path(__file__).with_name("inference_hub_part1_panel.py").resolve())
    }
    if part == "part0":
        allowed_migrations.add(
            str(Path(__file__).with_name("inference_hub_part0_panel.py").resolve())
        )
    refreshed: dict[str, str] = {}
    migrations: list[dict[str, str]] = []
    for path in expected_paths:
        path_key = str(path)
        observed = _sha256_file(path)
        prior = str(recorded[path_key])
        if observed != prior:
            if path_key not in allowed_migrations:
                raise TargetRetirementError(
                    f"Frozen non-runner source changed: {path_key}"
                )
            migrations.append({
                "path": path_key,
                "prior_sha256": prior,
                "resumed_runner_sha256": observed,
            })
        refreshed[path_key] = observed
    return refreshed, migrations


def _ledger_evidence(
    ledger: _ChainedJournal, *, part: str, target_id: str,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], list[str]]:
    artifact_type = f"inference_hub_{part}_attempt_ledger"
    reservations: dict[str, Mapping[str, Any]] = {}
    completions: dict[str, Mapping[str, Any]] = {}
    for row in ledger.records:
        if row.get("schema_version") != SCHEMA_VERSION or row.get("artifact_type") != artifact_type:
            raise TargetRetirementError("Attempt ledger schema/type binding failed.")
        attempt_id = row.get("attempt_id")
        event = row.get("event")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise TargetRetirementError("Attempt ledger has an invalid attempt ID.")
        if event == "reserved_before_dispatch":
            if attempt_id in reservations:
                raise TargetRetirementError("Attempt ledger reserves an ID twice.")
            if (
                not isinstance(row.get("target_id"), str)
                or not isinstance(row.get("request_sha256"), str)
                or len(str(row["request_sha256"])) != 64
            ):
                raise TargetRetirementError("Attempt reservation binding is invalid.")
            reservations[attempt_id] = row
        elif event == "attempt_completed":
            if attempt_id in completions:
                raise TargetRetirementError("Attempt ledger completes an ID twice.")
            completions[attempt_id] = row
        else:
            raise TargetRetirementError("Attempt ledger event is unknown.")
    if set(completions) - set(reservations):
        raise TargetRetirementError("Attempt completion lacks its reservation.")

    genuine: list[str] = []
    for attempt_id, completion in completions.items():
        reservation = reservations[attempt_id]
        if reservation.get("target_id") != target_id or completion.get("outcome") != "failed":
            continue
        code = completion.get("failure_code")
        status = completion.get("http_status")
        if code not in _DISPATCH_FAILURE_CODES:
            continue
        if code == "http_error" and (
            isinstance(status, bool) or not isinstance(status, int) or not 400 <= status <= 599
        ):
            continue
        record_hash = completion.get("record_sha256")
        if isinstance(record_hash, str):
            genuine.append(record_hash)
    return reservations, completions, genuine


def _retained_attempt_ids(
    raw_journals: Mapping[str, _ChainedJournal], *, part: str,
) -> set[str]:
    retained: set[str] = set()
    for journal in raw_journals.values():
        for row in journal.records:
            is_retained = (
                row.get("event") in {"subject_response_retained", "judge_batch_retained"}
                if part == "part0"
                else row.get("raw_response") is not None
            )
            if not is_retained:
                continue
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                raise TargetRetirementError(
                    "A retained raw response lacks its dispatch attempt binding."
                )
            if attempt_id in retained:
                raise TargetRetirementError(
                    "One dispatch attempt retained multiple raw responses."
                )
            retained.add(attempt_id)
    return retained


def _close_all_stale_reservations(
    *, ledger: _ChainedJournal, raw_journals: Mapping[str, _ChainedJournal],
    reservations: Mapping[str, Mapping[str, Any]],
    completions: Mapping[str, Mapping[str, Any]], part: str,
    permitted: bool, checkpoint_record_count: int,
) -> tuple[list[dict[str, Any]], int]:
    """Mirror runner recovery for every unmatched reservation across targets."""
    unmatched = {
        attempt_id: reservation
        for attempt_id, reservation in reservations.items()
        if attempt_id not in completions
    }
    if unmatched and not permitted:
        raise TargetRetirementError(
            "Unmatched dispatch reservations remain; use the explicit all-target "
            "stale-reservation recovery option after stopping the runner."
        )
    retained_attempts = _retained_attempt_ids(raw_journals, part=part)
    retained_unmatched = sorted(set(unmatched) & retained_attempts)
    if retained_unmatched:
        raise TargetRetirementError(
            "An unmatched reservation has a retained raw response."
        )

    appended_ids: set[str] = set()
    for attempt_id in sorted(unmatched):
        ledger.append({
            "schema_version": SCHEMA_VERSION,
            "artifact_type": f"inference_hub_{part}_attempt_ledger",
            "event": "attempt_completed",
            "attempt_id": attempt_id,
            "outcome": "failed",
            "failure_code": "stale_reservation_retried",
            "transient": True,
            "http_status": None,
            "completed_at_utc": _now().replace("+00:00", "Z"),
        })
        appended_ids.add(attempt_id)

    stale_records = [
        {
            "target_id": str(reservations[str(row["attempt_id"])]["target_id"]),
            "attempt_id": str(row["attempt_id"]),
            "record_sha256": str(row["record_sha256"]),
            "appended_in_finalization_invocation": (
                row.get("attempt_id") in appended_ids
            ),
        }
        for index, row in enumerate(ledger.records)
        if index >= checkpoint_record_count
        and row.get("event") == "attempt_completed"
        and row.get("attempt_id") in reservations
        and row.get("outcome") == "failed"
        and row.get("failure_code") == "stale_reservation_retried"
        and row.get("transient") is True
        and row.get("http_status") is None
    ]
    stale_attempts = {row["attempt_id"] for row in stale_records}
    if stale_attempts & retained_attempts:
        raise TargetRetirementError(
            "A recovered stale reservation has a retained raw response."
        )
    return stale_records, len(appended_ids)


def _identity_evidence(
    *, rows: Sequence[Mapping[str, Any]], ledger: _ChainedJournal,
    target_id: str,
) -> list[str]:
    reservations = {
        str(row["attempt_id"]): row
        for row in ledger.records if row.get("event") == "reserved_before_dispatch"
    }
    completions = {
        str(row["attempt_id"]): row
        for row in ledger.records if row.get("event") == "attempt_completed"
    }
    evidence: list[str] = []
    for row in rows:
        if row.get("target_id") != target_id or row.get("model_identity_valid") is not False:
            continue
        raw = row.get("raw_response")
        attempt_id = row.get("attempt_id")
        reservation = reservations.get(str(attempt_id))
        completion = completions.get(str(attempt_id))
        if (
            not isinstance(raw, Mapping)
            or row.get("raw_response_sha256") != _sha256_json(raw)
            or reservation is None
            or reservation.get("target_id") != target_id
            or completion is None
            or completion.get("outcome") != "response_retained"
        ):
            continue
        record_hash = row.get("record_sha256")
        if isinstance(record_hash, str):
            evidence.append(record_hash)
    return evidence


def _part0_schedule(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    schedule = manifest.get("schedule")
    if not isinstance(schedule, list) or not schedule or not all(
        isinstance(row, Mapping) for row in schedule
    ):
        raise TargetRetirementError("Part 0 schedule is absent or malformed.")
    if manifest.get("schedule_sha256") != _sha256_json(schedule):
        raise TargetRetirementError("Part 0 schedule hash changed.")
    ids: set[str] = set()
    for row in schedule:
        trial_id = row.get("trial_id")
        if (
            not isinstance(trial_id, str)
            or not trial_id
            or trial_id in ids
            or not isinstance(row.get("root_id"), str)
            or not isinstance(row.get("language"), str)
            or not isinstance(row.get("prompt_sha256"), str)
        ):
            raise TargetRetirementError("Part 0 schedule unit is invalid or duplicated.")
        ids.add(trial_id)
    if manifest.get("executed_trial_count_per_subject") != len(schedule):
        raise TargetRetirementError("Part 0 scheduled unit count changed.")
    return schedule


def _part1_schedule(manifest: Mapping[str, Any]) -> tuple[Any, ...]:
    base_seed = manifest.get("base_seed")
    count = manifest.get("executed_trial_count_per_subject")
    if (
        isinstance(base_seed, bool) or not isinstance(base_seed, int)
        or isinstance(count, bool) or not isinstance(count, int)
        or not 1 <= count <= 384
    ):
        raise TargetRetirementError("Part 1 schedule parameters are invalid.")
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping):
        raise TargetRetirementError("Part 1 source-artifact bindings are invalid.")
    builder = (
        build_stratified_trials
        if str(PART1_STRATIFIED_SOURCE) in source_artifacts
        else build_draft_trials
    )
    trials = builder(base_seed=base_seed, limit=None if count == 384 else count)
    binding = [
        {
            "trial_id": trial.trial_id,
            "root_id": trial.root_id,
            "game": trial.game,
            "domain": trial.domain,
            "counterbalance_id": trial.counterbalance_id,
            "prompt_sha256": trial.prompt_hash,
            "generation_settings": {
                "temperature": trial.generation_settings.temperature,
                "top_p": trial.generation_settings.top_p,
                "max_output_tokens": trial.generation_settings.max_output_tokens,
                "generation_seed": trial.generation_settings.generation_seed,
                "seed_base": trial.generation_settings.seed_base,
                "seed_derivation": trial.generation_settings.seed_derivation,
            },
        }
        for trial in trials
    ]
    if manifest.get("executed_schedule_sha256") != _sha256_json(binding):
        raise TargetRetirementError("Part 1 reconstructed schedule hash changed.")
    return trials


def _part0_existing(
    *, rows: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]],
    subject: Mapping[str, Any], retirement_id: str, reason: str,
    genuine_evidence: Sequence[str],
) -> tuple[set[str], dict[str, Mapping[str, Any]], list[Mapping[str, Any]]]:
    scheduled = {str(row["trial_id"]): row for row in schedule}
    retained: dict[str, Mapping[str, Any]] = {}
    terminal: set[str] = set()
    orphaned_retirements: list[Mapping[str, Any]] = []
    for row in rows:
        if row.get("target_id") != subject["target_id"]:
            raise TargetRetirementError("Part 0 raw journal target binding failed.")
        event = row.get("event")
        if event == "subject_response_retained":
            trial = scheduled.get(str(row.get("trial_id")))
            if (
                trial is None or row.get("trial_id") in retained
                or row.get("root_id") != trial.get("root_id")
                or row.get("language") != trial.get("language")
                or row.get("prompt_sha256") != trial.get("prompt_sha256")
                or row.get("requested_route") != subject["route"]
                or not isinstance(row.get("raw_response"), Mapping)
                or row.get("raw_response_sha256") != _sha256_json(row["raw_response"])
            ):
                raise TargetRetirementError("Part 0 retained response binding failed.")
            retained[str(row["trial_id"])] = row
        elif event == "judge_batch_retained":
            raw = row.get("raw_response")
            if raw is not None and (
                not isinstance(raw, Mapping)
                or row.get("raw_response_sha256") != _sha256_json(raw)
            ):
                raise TargetRetirementError("Part 0 judge response hash changed.")
        elif event in {"unit_completed", "unit_operationally_retired"}:
            trial = scheduled.get(str(row.get("trial_id")))
            if (
                trial is None or row.get("trial_id") in terminal
                or row.get("root_id") != trial.get("root_id")
                or row.get("language") != trial.get("language")
            ):
                raise TargetRetirementError("Part 0 terminal unit binding failed.")
            if event == "unit_completed" and row.get("outcome") not in {
                "REFUSAL", "COMPLIANCE", "UNCLEAR", "INVALID",
            }:
                raise TargetRetirementError("Part 0 terminal behavioral outcome is invalid.")
            if event == "unit_operationally_retired" and (
                row.get("dispatched") is not False
                or "outcome" in row
                or not isinstance(row.get("retirement"), Mapping)
                or row["retirement"].get("retirement_id") != retirement_id
                or row["retirement"].get("provenance") != RETIREMENT_PROVENANCE
                or row["retirement"].get("reason") != reason
                or row["retirement"].get(
                    "genuine_failure_evidence_record_sha256"
                ) != list(genuine_evidence)
            ):
                raise TargetRetirementError("Part 0 prior retirement record is invalid.")
            if event == "unit_operationally_retired":
                orphaned_retirements.append(row)
            terminal.add(str(row["trial_id"]))
        else:
            raise TargetRetirementError("Part 0 raw journal event is unknown.")
    retained_without_terminal = {
        trial_id: retained[trial_id] for trial_id in sorted(set(retained) - terminal)
    }
    return terminal, retained_without_terminal, orphaned_retirements


def _part1_existing(
    *, rows: Sequence[Mapping[str, Any]], trials: Sequence[Any],
    subject: Mapping[str, Any], retirement_id: str, reason: str,
    genuine_evidence: Sequence[str],
) -> tuple[set[str], list[Mapping[str, Any]]]:
    by_id = {trial.trial_id: trial for trial in trials}
    completed: set[str] = set()
    orphaned_retirements: list[Mapping[str, Any]] = []
    for row in rows:
        trial_id = row.get("trial_id")
        trial = by_id.get(str(trial_id))
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("artifact_type") != "inference_hub_part1_raw_response"
            or row.get("target_id") != subject["target_id"]
            or trial is None
            or trial_id in completed
            or row.get("root_id") != trial.root_id
            or row.get("prompt_sha256") != trial.prompt_hash
            or row.get("requested_route") != subject["route"]
        ):
            raise TargetRetirementError("Part 1 retained/terminal binding failed.")
        raw = row.get("raw_response")
        if raw is not None and (
            not isinstance(raw, Mapping)
            or row.get("raw_response_sha256") != _sha256_json(raw)
        ):
            raise TargetRetirementError("Part 1 retained response hash changed.")
        failure = row.get("failure")
        if (
            isinstance(failure, Mapping)
            and failure.get("failure_code") == RETIREMENT_FAILURE_CODE
        ):
            retirement = row.get("retirement")
            if (
                raw is not None
                or row.get("raw_response_sha256") is not None
                or row.get("response_model") is not None
                or failure.get("dispatched") is not False
                or "parsed_action" in row
                or not isinstance(retirement, Mapping)
                or retirement.get("retirement_id") != retirement_id
                or retirement.get("provenance") != RETIREMENT_PROVENANCE
                or retirement.get("reason") != reason
                or retirement.get(
                    "genuine_failure_evidence_record_sha256"
                ) != list(genuine_evidence)
            ):
                raise TargetRetirementError("Part 1 prior retirement record is invalid.")
            orphaned_retirements.append(row)
        completed.add(str(trial_id))
    return completed, orphaned_retirements


def _audit_record(
    *, part: str, target_id: str, reason: str, prior_manifest_hash: str,
    genuine_evidence_hashes: Sequence[str], appended: Sequence[Mapping[str, Any]],
    scheduled_unit_ids: Sequence[str], retirement_id: str,
    source_hash_migrations: Sequence[Mapping[str, str]],
    stale_completion_records: Sequence[Mapping[str, Any]],
    stale_completions_appended: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": RETIREMENT_TYPE,
        "part": part,
        "target_id": target_id,
        "retirement_id": retirement_id,
        "reason": reason,
        "provenance": RETIREMENT_PROVENANCE,
        "network_dispatch_performed_by_tool": False,
        "behavioral_outcomes_assigned_by_tool": False,
        "prior_manifest_evidence_sha256": prior_manifest_hash,
        "genuine_failure_evidence_record_sha256": sorted(genuine_evidence_hashes),
        "scheduled_unit_ids_sha256": hashlib.sha256(
            json.dumps(sorted(scheduled_unit_ids), separators=(",", ":")).encode()
        ).hexdigest(),
        "non_dispatched_terminal_record_count": len(appended),
        "non_dispatched_terminal_record_sha256": [
            str(row["record_sha256"]) for row in appended
        ],
        "source_artifact_hash_migrations": [dict(row) for row in source_hash_migrations],
        "all_target_stale_reservation_recovery": {
            "scope": "all_unmatched_reservations_across_all_targets",
            "records": [dict(row) for row in stale_completion_records],
            "appended_in_finalization_invocation": stale_completions_appended,
            "behavioral_terminal_records_created": 0,
        },
        "retired_at_utc": _now(),
    }
    record["record_sha256"] = _sha256_json(record)
    return record


def retire_target(
    *, manifest_path: Path, target_id: str, reason: str,
    close_all_stale_reservations: bool = False,
) -> dict[str, Any]:
    """Retire exactly one eligible target and return its sealed audit record."""
    manifest_path = manifest_path.resolve()
    if manifest_path.name != "manifest.json" or manifest_path.parent.name != "private":
        raise TargetRetirementError("Expected a private/manifest.json artifact path.")
    if not isinstance(target_id, str) or not target_id.strip():
        raise TargetRetirementError("Target ID is empty.")
    if not isinstance(reason, str) or not reason.strip():
        raise TargetRetirementError("Retirement reason is empty.")
    private_dir = manifest_path.parent
    with _exclusive_run_lock(private_dir):
        manifest = _read_manifest(manifest_path)
        prior_hash = str(manifest["evidence_sha256"])
        subjects = _subjects(manifest)
        if target_id not in subjects:
            raise TargetRetirementError("Selected target is absent from the manifest.")
        prior_audits = manifest.get("target_retirements", [])
        if not isinstance(prior_audits, list):
            raise TargetRetirementError("Manifest target retirement audit list is malformed.")
        for row in prior_audits:
            if (
                not isinstance(row, Mapping)
                or row.get("schema_version") != SCHEMA_VERSION
                or row.get("artifact_type") != RETIREMENT_TYPE
                or row.get("record_sha256") != _sha256_json({
                    key: value for key, value in row.items() if key != "record_sha256"
                })
            ):
                raise TargetRetirementError(
                    "Manifest target retirement audit record is invalid."
                )
        if any(
            isinstance(row, Mapping) and row.get("target_id") == target_id
            for row in prior_audits
        ):
            raise TargetRetirementError("Selected target is already operationally retired.")
        ledger, raw_journals = _journals(manifest, private_dir, subjects)
        part = "part0" if manifest["artifact_type"] == PART0_TYPE else "part1"
        refreshed_sources, source_hash_migrations = _source_hash_migrations(
            manifest, part=part,
        )
        reservations, completions, transport_evidence = _ledger_evidence(
            ledger, part=part, target_id=target_id,
        )
        ledger_reference = manifest["journals"]["attempt_ledger"]
        checkpoint_record_count = ledger_reference.get("record_count")
        if (
            isinstance(checkpoint_record_count, bool)
            or not isinstance(checkpoint_record_count, int)
        ):
            raise TargetRetirementError("Attempt-ledger checkpoint count is invalid.")
        target_rows = list(raw_journals[target_id].records)
        identity_evidence = _identity_evidence(
            rows=target_rows, ledger=ledger, target_id=target_id,
        )
        genuine_evidence = sorted(set(transport_evidence + identity_evidence))
        if not genuine_evidence:
            raise TargetRetirementError(
                "Selected target lacks genuine dispatched transport/identity failure evidence."
            )
        stale_completion_records, stale_completions_appended = (
            _close_all_stale_reservations(
                ledger=ledger, raw_journals=raw_journals,
                reservations=reservations, completions=completions, part=part,
                permitted=close_all_stale_reservations,
                checkpoint_record_count=checkpoint_record_count,
            )
        )

        retirement_id = "retire_" + hashlib.sha256(
            json.dumps(
                [part, target_id, reason.strip(), prior_hash],
                separators=(",", ":"), ensure_ascii=False,
            ).encode()
        ).hexdigest()[:24]
        subject = subjects[target_id]
        appended: list[Mapping[str, Any]] = []
        scheduled_ids: list[str]
        if part == "part0":
            schedule = _part0_schedule(manifest)
            terminal, retained_without_terminal, orphaned = _part0_existing(
                rows=target_rows, schedule=schedule, subject=subject,
                retirement_id=retirement_id, reason=reason.strip(),
                genuine_evidence=genuine_evidence,
            )
            scheduled_ids = [str(row["trial_id"]) for row in schedule]
            remaining = [row for row in schedule if row["trial_id"] not in terminal]
            if not remaining and not orphaned:
                raise TargetRetirementError("Selected target has no remaining scheduled units.")
            appended.extend(orphaned)
            for trial in remaining:
                retained = retained_without_terminal.get(str(trial["trial_id"]))
                retained_provenance = (
                    {
                        "record_sha256": retained["record_sha256"],
                        "raw_response_sha256": retained["raw_response_sha256"],
                        "attempt_id": retained["attempt_id"],
                    }
                    if retained is not None else None
                )
                appended.append(raw_journals[target_id].append({
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "inference_hub_part0_private_record",
                    "event": "unit_operationally_retired",
                    "target_id": target_id,
                    "trial_id": trial["trial_id"],
                    "root_id": trial["root_id"],
                    "language": trial["language"],
                    "dispatched": False,
                    "retirement": {
                        "retirement_id": retirement_id,
                        "provenance": RETIREMENT_PROVENANCE,
                        "reason": reason.strip(),
                        "genuine_failure_evidence_record_sha256": genuine_evidence,
                        **(
                            {
                                "retained_subject_response": retained_provenance,
                                "judge_dispatch_performed": False,
                            }
                            if retained_provenance is not None else {}
                        ),
                    },
                    "completed_at_utc": _now(),
                }))
        else:
            trials = _part1_schedule(manifest)
            completed, orphaned = _part1_existing(
                rows=target_rows, trials=trials, subject=subject,
                retirement_id=retirement_id, reason=reason.strip(),
                genuine_evidence=genuine_evidence,
            )
            scheduled_ids = [trial.trial_id for trial in trials]
            remaining = [trial for trial in trials if trial.trial_id not in completed]
            if not remaining and not orphaned:
                raise TargetRetirementError("Selected target has no remaining scheduled units.")
            appended.extend(orphaned)
            for trial in remaining:
                appended.append(raw_journals[target_id].append({
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "inference_hub_part1_raw_response",
                    "target_id": target_id,
                    "upstream_provider": subject["upstream_provider"],
                    "model": subject["model"],
                    "requested_route": subject["route"],
                    "response_model": None,
                    "model_identity_valid": False,
                    "trial_id": trial.trial_id,
                    "root_id": trial.root_id,
                    "game": trial.game,
                    "domain": trial.domain,
                    "counterbalance_id": trial.counterbalance_id,
                    "prompt_sha256": trial.prompt_hash,
                    "raw_response": None,
                    "raw_response_sha256": None,
                    "failure": {
                        "failure_code": RETIREMENT_FAILURE_CODE,
                        "http_status": None,
                        "error_type": None,
                        "dispatched": False,
                    },
                    "retirement": {
                        "retirement_id": retirement_id,
                        "provenance": RETIREMENT_PROVENANCE,
                        "reason": reason.strip(),
                        "genuine_failure_evidence_record_sha256": genuine_evidence,
                    },
                    "finished_at_utc": _now(),
                }))

        audit = _audit_record(
            part=part, target_id=target_id, reason=reason.strip(),
            prior_manifest_hash=prior_hash,
            genuine_evidence_hashes=genuine_evidence, appended=appended,
            scheduled_unit_ids=scheduled_ids, retirement_id=retirement_id,
            source_hash_migrations=source_hash_migrations,
            stale_completion_records=stale_completion_records,
            stale_completions_appended=stale_completions_appended,
        )
        manifest["target_retirements"] = [*prior_audits, audit]
        manifest["source_artifacts"] = refreshed_sources
        manifest["journals"] = {
            "attempt_ledger": ledger.reference(),
            "raw_responses": {
                item_target: journal.reference()
                for item_target, journal in raw_journals.items()
            },
        }
        manifest["last_updated_at_utc"] = _now()
        _seal(manifest)
        _atomic_json(manifest_path, manifest)
        return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--close-all-stale-reservations",
        action="store_true",
        help=(
            "After the runner is stopped, close every unmatched reservation "
            "across all targets using the runner's transient stale record."
        ),
    )
    args = parser.parse_args()
    try:
        audit = retire_target(
            manifest_path=args.manifest, target_id=args.target_id, reason=args.reason,
            close_all_stale_reservations=(
                args.close_all_stale_reservations
            ),
        )
    except (TargetRetirementError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
