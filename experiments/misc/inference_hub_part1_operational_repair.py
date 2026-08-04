"""Repair only terminal Part 1 transport-null rows in a separate overlay.

The source campaign and every original response journal remain immutable. This
utility is eligible only after the source writer releases its run lock and the
source manifest accounts for every scheduled unit. It redispatches the exact
original request only for rows whose retained ``raw_response`` is null. Any
identity-valid provider response, including a genuine format-invalid response,
is retained once and is never semantically regenerated.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import stat
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.misc import inference_hub_part1_panel as base
from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
)
from experiments.misc.inference_hub_part1_deadline_accelerated import (
    _deadline_client,
)


SCHEMA_VERSION = 1
SOURCE_ARTIFACT_TYPE = "inference_hub_part1_large_n_exploratory_panel"
MANIFEST_ARTIFACT_TYPE = "inference_hub_part1_operational_repair_overlay_v1"
ATTEMPT_ARTIFACT_TYPE = "inference_hub_part1_operational_repair_attempt_v1"
RESPONSE_ARTIFACT_TYPE = "inference_hub_part1_operational_repair_response_v1"
SANITIZED_ARTIFACT_TYPE = (
    "inference_hub_part1_operational_repair_sanitized_summary_v1"
)
DEFAULT_MAX_ROUNDS = 8
DEFAULT_MAX_WORKERS = 24
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 900.0
DEADLINE_LAUNCHER = Path(__file__).with_name(
    "inference_hub_part1_deadline_accelerated.py"
)
_SOURCE_PATHS = (
    Path(__file__),
    Path(base.__file__),
    DEADLINE_LAUNCHER,
    Path(__file__).with_name("inference_hub_discovery.py"),
    Path(__file__).with_name("inference_hub_rate_limit.py"),
)


class Part1OperationalRepairError(RuntimeError):
    """The operational repair evidence or execution contract is unsafe."""


def _require_private_file(path: Path) -> None:
    if not path.is_file():
        raise Part1OperationalRepairError(f"Private evidence is missing: {path}.")
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise Part1OperationalRepairError(
            f"Private evidence must have mode 0600: {path}."
        )


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _load_source(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    tuple[Any, ...],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    """Validate an inactive, fully accounted source without changing it."""

    manifest_path = manifest_path.resolve()
    _require_private_file(manifest_path)
    manifest = base._read_json(manifest_path, "terminal Part 1 manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != SOURCE_ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != base._self_hash(manifest)
    ):
        raise Part1OperationalRepairError(
            "Source Part 1 manifest type or integrity failed."
        )
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, Mapping) or not sources or any(
        not isinstance(path, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        for path, digest in sources.items()
    ):
        raise Part1OperationalRepairError("Source provenance bindings are invalid.")
    subject_rows = manifest.get("subject_routes")
    if not isinstance(subject_rows, list) or not subject_rows:
        raise Part1OperationalRepairError("Source subject routes are missing.")
    subjects = {
        str(row.get("target_id")): row
        for row in subject_rows
        if isinstance(row, Mapping) and row.get("target_id")
    }
    if len(subjects) != len(subject_rows):
        raise Part1OperationalRepairError(
            "Source subject identities are invalid or duplicated."
        )
    judge = manifest.get("judge_reservation")
    if (
        not isinstance(judge, Mapping)
        or judge.get("dispatch_permitted_in_this_runner") is not False
        or str(judge.get("target_id")) in subjects
    ):
        raise Part1OperationalRepairError("Source judge reservation is unsafe.")

    trials = base.build_draft_trials(
        base_seed=int(manifest["base_seed"]),
        limit=manifest.get("trial_limit"),
    )
    trials_by_id = {trial.trial_id: trial for trial in trials}
    refs = manifest.get("journals")
    raw_refs = refs.get("raw_responses") if isinstance(refs, Mapping) else None
    if not isinstance(raw_refs, Mapping) or set(raw_refs) != set(subjects):
        raise Part1OperationalRepairError("Source raw journal set changed.")
    private_dir = manifest_path.parent
    attempt_ref = refs.get("attempt_ledger") if isinstance(refs, Mapping) else None
    if not isinstance(attempt_ref, Mapping) or not isinstance(
        attempt_ref.get("path"), str
    ):
        raise Part1OperationalRepairError("Source attempt checkpoint is malformed.")
    attempt_path = Path(attempt_ref["path"]).resolve()
    if not _within(attempt_path, private_dir):
        raise Part1OperationalRepairError("Source attempt journal escaped private data.")
    source_attempts = base._ChainedJournal(attempt_path)
    base._validate_checkpoint_reference(
        source_attempts, attempt_ref, label="source attempt ledger"
    )
    if len(source_attempts.records) != attempt_ref.get("record_count"):
        raise Part1OperationalRepairError(
            "Source attempt journal grew beyond its terminal checkpoint."
        )

    raw_journals: dict[str, base._ChainedJournal] = {}
    for target, reference in raw_refs.items():
        if not isinstance(reference, Mapping) or not isinstance(
            reference.get("path"), str
        ):
            raise Part1OperationalRepairError("Source raw checkpoint is malformed.")
        path = Path(reference["path"]).resolve()
        if not _within(path, private_dir):
            raise Part1OperationalRepairError("Source raw journal escaped private data.")
        journal = base._ChainedJournal(path)
        base._validate_checkpoint_reference(
            journal, reference, label=f"source raw/{target}"
        )
        if len(journal.records) != reference.get("record_count"):
            raise Part1OperationalRepairError(
                "Source raw journal grew beyond its terminal checkpoint."
            )
        raw_journals[str(target)] = journal

    retained = base._completed_index(
        raw_journals,
        trials_by_id=trials_by_id,
        subjects_by_id=subjects,
    )
    planned = len(subjects) * len(trials)
    summary = manifest.get("summary")
    if (
        not isinstance(summary, Mapping)
        or len(retained) != planned
        or summary.get("planned_generations") != planned
        or summary.get("retained_trial_records") != planned
    ):
        raise Part1OperationalRepairError(
            "Source has not terminalized with one retained row per scheduled unit."
        )
    rows = list(retained.values())
    null_count = sum(row.get("raw_response") is None for row in rows)
    if (
        summary.get("failed_without_response") != null_count
        or summary.get("responses_received") != planned - null_count
        or manifest.get("complete") is not (null_count == 0)
    ):
        raise Part1OperationalRepairError(
            "Source terminal operational accounting is inconsistent."
        )
    if any(
        row.get("raw_response") is not None
        and row.get("model_identity_valid") is not True
        for row in rows
    ):
        raise Part1OperationalRepairError(
            "Source contains a non-null identity mismatch; operational overlay refused."
        )

    for (target, trial_id), row in retained.items():
        subject = subjects[target]
        trial = trials_by_id[trial_id]
        body, _ = base._request_contract(subject, trial)
        if (
            row.get("target_id") != target
            or row.get("trial_id") != trial_id
            or row.get("root_id") != trial.root_id
            or row.get("prompt_sha256") != trial.prompt_hash
            or row.get("prompt_text") != trial.prompt_text
            or row.get("requested_route") != subject.get("route")
            or row.get("model") != subject.get("model")
            or row.get("upstream_provider") != subject.get("upstream_provider")
            or row.get("request_sha256") != base._sha256_json(body)
        ):
            raise Part1OperationalRepairError(
                f"Original request/prompt/identity binding changed for {target}/{trial_id}."
            )
    return manifest, trials, subjects, retained


def _attempt_id(original_sha256: str, round_index: int) -> str:
    digest = hashlib.sha256(
        f"part1-operational-repair-v1\0{original_sha256}\0{round_index}".encode()
    ).hexdigest()
    return f"p1oprepair_{digest[:32]}"


def _bindings(
    source_path: Path, source: Mapping[str, Any], max_rounds: int
) -> dict[str, Any]:
    source_artifacts = {
        str(path.resolve()): base._sha256_file(path.resolve()) for path in _SOURCE_PATHS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "source_manifest": {
            "path": str(source_path.resolve()),
            "file_sha256": base._sha256_file(source_path),
            "evidence_sha256": source["evidence_sha256"],
        },
        "source_artifact_type": SOURCE_ARTIFACT_TYPE,
        "source_provenance_sha256": base._sha256_json(source["source_artifacts"]),
        "repair_source_artifacts": source_artifacts,
        "max_operational_rounds": max_rounds,
        "request_policy": "exact_original_request_no_seed_or_prompt_change_v1",
        "eligibility_policy": "original_retained_provider_payload_null_only_v1",
        "original_manifest_mutated": False,
        "original_journals_mutated": False,
        "visible_format_invalid_rows_retried": False,
        "judge_dispatched": False,
    }


def _reconcile(
    ledger: base._ChainedJournal,
    raw_journals: Mapping[str, base._ChainedJournal],
    *,
    originals: Mapping[tuple[str, str], Mapping[str, Any]],
    subjects: Mapping[str, Mapping[str, Any]],
    trials_by_id: Mapping[str, Any],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, str, int], Mapping[str, Any]],
]:
    reservations: dict[str, Mapping[str, Any]] = {}
    completions: dict[str, Mapping[str, Any]] = {}
    for row in ledger.records:
        attempt_id = row.get("attempt_id")
        if row.get("artifact_type") != ATTEMPT_ARTIFACT_TYPE:
            raise Part1OperationalRepairError("Repair attempt artifact type changed.")
        if row.get("event") == "reserved_before_dispatch" and isinstance(
            attempt_id, str
        ):
            if attempt_id in reservations:
                raise Part1OperationalRepairError("Repair attempt was reserved twice.")
            reservations[attempt_id] = row
        elif row.get("event") == "attempt_completed" and isinstance(attempt_id, str):
            if attempt_id in completions or attempt_id not in reservations:
                raise Part1OperationalRepairError("Repair completion binding failed.")
            completions[attempt_id] = row
        else:
            raise Part1OperationalRepairError("Unknown repair attempt event.")

    raw_by_unit: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    raw_by_attempt: dict[str, Mapping[str, Any]] = {}
    for target, journal in raw_journals.items():
        for row in journal.records:
            attempt_id = row.get("attempt_id")
            trial_id = str(row.get("trial_id"))
            round_index = row.get("round_index")
            key = (target, trial_id, round_index)
            original = originals.get((target, trial_id))
            reservation = reservations.get(str(attempt_id))
            subject = subjects[target]
            trial = trials_by_id.get(trial_id)
            raw_response = row.get("raw_response")
            identity_valid = row.get("response_model") == subject["route"]
            if (
                row.get("artifact_type") != RESPONSE_ARTIFACT_TYPE
                or not isinstance(attempt_id, str)
                or isinstance(round_index, bool)
                or not isinstance(round_index, int)
                or round_index < 1
                or key in raw_by_unit
                or attempt_id in raw_by_attempt
                or original is None
                or original.get("raw_response") is not None
                or trial is None
                or reservation is None
                or reservation.get("target_id") != target
                or reservation.get("trial_id") != trial_id
                or reservation.get("round_index") != round_index
                or reservation.get("root_id") != trial.root_id
                or reservation.get("original_record_sha256")
                != original.get("record_sha256")
                or reservation.get("original_request_sha256")
                != original.get("request_sha256")
                or reservation.get("original_prompt_sha256")
                != original.get("prompt_sha256")
                or reservation.get("requested_route") != subject["route"]
                or reservation.get("request_sha256")
                != original.get("request_sha256")
                or row.get("original_record_sha256") != original.get("record_sha256")
                or row.get("original_request_sha256") != original.get("request_sha256")
                or row.get("original_prompt_sha256") != original.get("prompt_sha256")
                or row.get("requested_route") != subject["route"]
                or row.get("request_sha256") != original.get("request_sha256")
                or row.get("root_id") != trial.root_id
                or row.get("upstream_provider") != subject["upstream_provider"]
                or row.get("model") != subject["model"]
                or raw_response is None
                or row.get("raw_response_sha256") != base._sha256_json(raw_response)
                or row.get("model_identity_valid") is not identity_valid
            ):
                raise Part1OperationalRepairError(
                    f"Repair response binding failed for {target}/{trial_id}."
                )
            raw_by_unit[key] = row
            raw_by_attempt[attempt_id] = row
            if attempt_id not in completions:
                ledger.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": ATTEMPT_ARTIFACT_TYPE,
                        "event": "attempt_completed",
                        "attempt_id": attempt_id,
                        "outcome": (
                            "response_retained_for_overlay"
                            if identity_valid
                            else "identity_mismatch_retained_not_overlay"
                        ),
                        "recovered_after_raw_fsync": True,
                        "completed_at_utc": base._utc_now(),
                    }
                )
                completions[attempt_id] = ledger.records[-1]
    for attempt_id, completion in completions.items():
        if completion.get("outcome") in {
            "response_retained_for_overlay",
            "identity_mismatch_retained_not_overlay",
        } and attempt_id not in raw_by_attempt:
            raise Part1OperationalRepairError(
                "Retained repair completion lacks its raw response."
            )
    for attempt_id in sorted(set(reservations) - set(completions)):
        ledger.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ATTEMPT_ARTIFACT_TYPE,
                "event": "attempt_completed",
                "attempt_id": attempt_id,
                "outcome": "indeterminate_after_crash",
                "redispatch_same_round": False,
                "completed_at_utc": base._utc_now(),
            }
        )
        completions[attempt_id] = ledger.records[-1]
    return reservations, completions, raw_by_unit


def _effective_outputs(
    *,
    source: Mapping[str, Any],
    originals: Mapping[tuple[str, str], Mapping[str, Any]],
    raw_by_unit: Mapping[tuple[str, str, int], Mapping[str, Any]],
    subjects: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay: dict[tuple[str, str], Mapping[str, Any]] = {}
    for key in sorted(raw_by_unit):
        row = raw_by_unit[key]
        unit = (key[0], key[1])
        if row.get("model_identity_valid") is True and unit not in overlay:
            overlay[unit] = row
    effective = {**originals, **overlay}
    rows = list(effective.values())
    eligible = {
        key for key, row in originals.items() if row.get("raw_response") is None
    }
    effective_summary = {
        "planned_generations": len(originals),
        "retained_trial_records": len(rows),
        "responses_received": sum(row.get("raw_response") is not None for row in rows),
        "failed_without_response": sum(row.get("raw_response") is None for row in rows),
        "format_valid": sum(row.get("format_valid") is True for row in rows),
        "format_invalid_retained": sum(
            row.get("raw_response") is not None and row.get("format_valid") is not True
            for row in rows
        ),
        "response_model_identity_mismatches": sum(
            row.get("raw_response") is not None
            and row.get("model_identity_valid") is not True
            for row in rows
        ),
        "operational_repair_eligible_originals": len(eligible),
        "operational_repairs_succeeded": len(overlay),
        "operational_repairs_unresolved": len(eligible - set(overlay)),
        "non_null_format_invalid_originals_not_retried": sum(
            row.get("raw_response") is not None and row.get("format_valid") is not True
            for row in originals.values()
        ),
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for (target, _), row in effective.items():
        grouped[target].append(row)
    model_rows = []
    for target, group in sorted(grouped.items()):
        original_group = [
            row for (candidate, _), row in originals.items() if candidate == target
        ]
        repaired_count = sum(key[0] == target for key in overlay)
        valid = sum(row.get("format_valid") is True for row in group)
        model_rows.append(
            {
                "target_id": target,
                "upstream_provider": subjects[target]["upstream_provider"],
                "model": subjects[target]["model"],
                "scheduled_draws": len(group),
                "original_transport_null_draws": sum(
                    row.get("raw_response") is None for row in original_group
                ),
                "operationally_repaired_draws": repaired_count,
                "effective_response_draws": sum(
                    row.get("raw_response") is not None for row in group
                ),
                "effective_format_valid_draws": valid,
                "effective_format_invalid_draws": sum(
                    row.get("raw_response") is not None
                    and row.get("format_valid") is not True
                    for row in group
                ),
                "effective_x_draws": sum(row.get("parsed_action") == "X" for row in group),
                "effective_y_draws": sum(row.get("parsed_action") == "Y" for row in group),
                "effective_format_valid_rate_all_scheduled": valid / len(group),
            }
        )
    sanitized = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SANITIZED_ARTIFACT_TYPE,
        "source_manifest_evidence_sha256": source["evidence_sha256"],
        "raw_text_included": False,
        "original_manifest_mutated": False,
        "original_journals_mutated": False,
        "visible_format_invalid_rows_retried": False,
        "effective_summary": effective_summary,
        "rows": model_rows,
    }
    base._seal(sanitized)
    return effective_summary, sanitized


def run_repair(
    *,
    source_manifest_path: Path,
    output_dir: Path,
    client: Any | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    initial_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    resume: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run an immutable post-terminal operational overlay."""

    for name, value in (("max_rounds", max_rounds), ("max_workers", max_workers)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise Part1OperationalRepairError(f"{name} must be a positive integer.")
    if initial_backoff_seconds < 0:
        raise Part1OperationalRepairError(
            "initial_backoff_seconds must be nonnegative."
        )
    source_path = source_manifest_path.resolve()
    source_private = source_path.parent
    try:
        source_lock = base._acquire_run_lock(source_private)
    except base.InferenceHubPart1PanelError as error:
        raise Part1OperationalRepairError(
            "Source Part 1 writer is still active; repair must wait for terminalization."
        ) from error
    try:
        source, trials, subjects, originals = _load_source(source_path)
        trials_by_id = {trial.trial_id: trial for trial in trials}
        eligible = {
            key: row for key, row in originals.items() if row.get("raw_response") is None
        }
        bindings = _bindings(source_path, source, max_rounds)
        if output_dir.exists() and not resume:
            raise Part1OperationalRepairError(
                "Operational repair output exists; use --resume explicitly."
            )
        private_dir = output_dir / "private"
        raw_dir = private_dir / "raw_responses"
        sanitized_dir = output_dir / "sanitized"
        if resume:
            for directory in (output_dir, private_dir, raw_dir, sanitized_dir):
                if not directory.is_dir():
                    raise Part1OperationalRepairError(
                        f"Repair resume directory is missing: {directory}."
                    )
        else:
            output_dir.mkdir(parents=True, mode=0o700)
            for directory in (private_dir, raw_dir, sanitized_dir):
                directory.mkdir(mode=0o700)
        for directory in (output_dir, private_dir, raw_dir, sanitized_dir):
            base._secure_mode(directory, 0o700)
            base._require_mode(directory, 0o700)
        repair_lock = base._acquire_run_lock(private_dir)
        try:
            manifest_path = private_dir / "manifest.json"
            ledger = base._ChainedJournal(private_dir / "attempt_ledger.jsonl")
            raw_journals = {
                target: base._ChainedJournal(
                    raw_dir / f"{base._safe_file_stem(target)}.jsonl"
                )
                for target in subjects
            }
            if resume:
                manifest = base._read_json(manifest_path, "operational repair manifest")
                _require_private_file(manifest_path)
                if manifest.get("evidence_sha256") != base._self_hash(manifest):
                    raise Part1OperationalRepairError(
                        "Operational repair manifest self-hash failed."
                    )
                if {key: manifest.get(key) for key in bindings} != bindings:
                    raise Part1OperationalRepairError(
                        "Operational repair/source bindings changed on resume."
                    )
                refs = manifest.get("journals")
                raw_refs = refs.get("raw_responses") if isinstance(refs, Mapping) else None
                if not isinstance(raw_refs, Mapping) or set(raw_refs) != set(raw_journals):
                    raise Part1OperationalRepairError(
                        "Operational repair journal set changed."
                    )
                base._validate_checkpoint_reference(
                    ledger, refs.get("attempt_ledger"), label="repair attempt ledger"
                )
                for target, journal in raw_journals.items():
                    base._validate_checkpoint_reference(
                        journal, raw_refs[target], label=f"repair raw/{target}"
                    )
            else:
                manifest = {
                    **bindings,
                    "created_at_utc": base._utc_now(),
                    "last_updated_at_utc": base._utc_now(),
                    "complete": False,
                    "summary": {},
                    "journals": {
                        "attempt_ledger": ledger.reference(),
                        "raw_responses": {
                            target: journal.reference()
                            for target, journal in raw_journals.items()
                        },
                    },
                }
                base._seal(manifest)
                base._atomic_json(manifest_path, manifest)

            reservations, _, raw_by_unit = _reconcile(
                ledger,
                raw_journals,
                originals=originals,
                subjects=subjects,
                trials_by_id=trials_by_id,
            )
            model_client = client
            if model_client is None and eligible:
                model_client = _deadline_client(DEFAULT_TIMEOUT_SECONDS)

            def execute(original: Mapping[str, Any], round_index: int) -> None:
                target = str(original["target_id"])
                trial_id = str(original["trial_id"])
                subject = subjects[target]
                trial = trials_by_id[trial_id]
                body, controls = base._request_contract(subject, trial)
                request_sha256 = base._sha256_json(body)
                if request_sha256 != original.get("request_sha256"):
                    raise Part1OperationalRepairError(
                        "Exact operational repair request changed before dispatch."
                    )
                attempt_id = _attempt_id(str(original["record_sha256"]), round_index)
                binding = {
                    "target_id": target,
                    "trial_id": trial_id,
                    "root_id": trial.root_id,
                    "round_index": round_index,
                    "original_record_sha256": original["record_sha256"],
                    "original_request_sha256": original["request_sha256"],
                    "original_prompt_sha256": original["prompt_sha256"],
                    "requested_route": subject["route"],
                    "request_sha256": request_sha256,
                }
                ledger.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": ATTEMPT_ARTIFACT_TYPE,
                        "event": "reserved_before_dispatch",
                        "attempt_id": attempt_id,
                        "upstream_provider": subject["upstream_provider"],
                        "model": subject["model"],
                        "controls": controls,
                        **binding,
                        "reserved_at_utc": base._utc_now(),
                    }
                )
                try:
                    if model_client is None:  # pragma: no cover
                        raise Part1OperationalRepairError("Repair client is unavailable.")
                    if isinstance(model_client, InferenceHubClient):
                        response = model_client.post(
                            "/chat/completions",
                            body,
                            upstream_provider=str(subject["upstream_provider"]),
                        )
                    else:
                        response = model_client.post("/chat/completions", body)
                    if not isinstance(response, Mapping):
                        raise TypeError("client response is not an object")
                except Exception as error:
                    transient, code, status = base._transient(error)
                    ledger.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "artifact_type": ATTEMPT_ARTIFACT_TYPE,
                            "event": "attempt_completed",
                            "attempt_id": attempt_id,
                            "outcome": "transport_failure",
                            "failure_code": code,
                            "transient": transient,
                            "http_status": status,
                            "completed_at_utc": base._utc_now(),
                        }
                    )
                    return
                metadata = base._response_metadata(response)
                identity_valid = metadata["response_model"] == subject["route"]
                raw_row = raw_journals[target].append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": RESPONSE_ARTIFACT_TYPE,
                        "attempt_id": attempt_id,
                        "upstream_provider": subject["upstream_provider"],
                        "model": subject["model"],
                        "model_identity_valid": identity_valid,
                        **binding,
                        **metadata,
                        "raw_response": dict(response),
                        "raw_response_sha256": base._sha256_json(response),
                        "retained_at_utc": base._utc_now(),
                    }
                )
                ledger.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": ATTEMPT_ARTIFACT_TYPE,
                        "event": "attempt_completed",
                        "attempt_id": attempt_id,
                        "outcome": (
                            "response_retained_for_overlay"
                            if identity_valid
                            else "identity_mismatch_retained_not_overlay"
                        ),
                        "response_payload_sha256": raw_row["raw_response_sha256"],
                        "response_model": metadata["response_model"],
                        "format_valid": metadata["format_valid"],
                        "completed_at_utc": base._utc_now(),
                    }
                )

            for round_index in range(1, max_rounds + 1):
                reservations, _, raw_by_unit = _reconcile(
                    ledger,
                    raw_journals,
                    originals=originals,
                    subjects=subjects,
                    trials_by_id=trials_by_id,
                )
                successful = {
                    (target, trial_id)
                    for (target, trial_id, _), row in raw_by_unit.items()
                    if row.get("model_identity_valid") is True
                }
                work = [
                    original
                    for key, original in eligible.items()
                    if key not in successful
                    and _attempt_id(str(original["record_sha256"]), round_index)
                    not in reservations
                ]
                if work:
                    with ThreadPoolExecutor(
                        max_workers=min(max_workers, len(work)),
                        thread_name_prefix=f"part1-operational-repair-r{round_index}",
                    ) as executor:
                        futures = [
                            executor.submit(execute, original, round_index)
                            for original in work
                        ]
                        for future in as_completed(futures):
                            future.result()
                reservations, _, raw_by_unit = _reconcile(
                    ledger,
                    raw_journals,
                    originals=originals,
                    subjects=subjects,
                    trials_by_id=trials_by_id,
                )
                effective_summary, _ = _effective_outputs(
                    source=source,
                    originals=originals,
                    raw_by_unit=raw_by_unit,
                    subjects=subjects,
                )
                manifest["last_updated_at_utc"] = base._utc_now()
                manifest["summary"] = effective_summary
                manifest["journals"] = {
                    "attempt_ledger": ledger.reference(),
                    "raw_responses": {
                        target: journal.reference()
                        for target, journal in raw_journals.items()
                    },
                }
                base._seal(manifest)
                base._atomic_json(manifest_path, manifest)
                if effective_summary["operational_repairs_unresolved"] == 0:
                    break
                if round_index < max_rounds:
                    sleep_fn(initial_backoff_seconds * (2 ** (round_index - 1)))

            _, _, raw_by_unit = _reconcile(
                ledger,
                raw_journals,
                originals=originals,
                subjects=subjects,
                trials_by_id=trials_by_id,
            )
            effective_summary, sanitized = _effective_outputs(
                source=source,
                originals=originals,
                raw_by_unit=raw_by_unit,
                subjects=subjects,
            )
            sanitized_path = sanitized_dir / "summary.json"
            base._atomic_json(sanitized_path, sanitized)
            manifest["summary"] = effective_summary
            manifest["complete"] = (
                effective_summary["operational_repairs_unresolved"] == 0
                and effective_summary["response_model_identity_mismatches"] == 0
            )
            manifest["last_updated_at_utc"] = base._utc_now()
            if manifest["complete"]:
                manifest["completed_at_utc"] = base._utc_now()
            manifest["journals"] = {
                "attempt_ledger": ledger.reference(),
                "raw_responses": {
                    target: journal.reference()
                    for target, journal in raw_journals.items()
                },
            }
            manifest["sanitized_artifact"] = {
                "path": str(sanitized_path.resolve()),
                "file_sha256": base._sha256_file(sanitized_path),
                "evidence_sha256": sanitized["evidence_sha256"],
            }
            base._seal(manifest)
            base._atomic_json(manifest_path, manifest)
            return manifest
        finally:
            fcntl.flock(repair_lock.fileno(), fcntl.LOCK_UN)
            repair_lock.close()
    finally:
        fcntl.flock(source_lock.fileno(), fcntl.LOCK_UN)
        source_lock.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=_positive_int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--max-workers", type=_positive_int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--initial-backoff-seconds",
        type=_nonnegative_float,
        default=DEFAULT_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--timeout-seconds", type=_nonnegative_float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        client = _deadline_client(args.timeout_seconds)
        manifest = run_repair(
            source_manifest_path=args.source_manifest,
            output_dir=args.output_dir,
            client=client,
            max_rounds=args.max_rounds,
            max_workers=args.max_workers,
            initial_backoff_seconds=args.initial_backoff_seconds,
            resume=args.resume,
        )
    except (
        Part1OperationalRepairError,
        base.InferenceHubPart1PanelError,
        InferenceHubDiscoveryError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Part 1 operational repair failed: {error}")
        return 2
    print(f"Operational repairs unresolved: {manifest['summary']['operational_repairs_unresolved']}")
    print(f"Sanitized summary: {args.output_dir / 'sanitized' / 'summary.json'}")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
