"""Separate semantic-invalid repair for a completed definitive Part 1 panel.

The source panel stays immutable. Only retained first-response rows with
format_valid=false are eligible. Repair responses never replace, relabel, or
change a primary record or denominator.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.misc import inference_hub_part1_panel as base
from experiments.misc.inference_hub_main_accelerated import _accelerated_client


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "inference_hub_part1_semantic_invalid_repair_v1"
SOURCE_ARTIFACT_TYPE = "inference_hub_part1_large_n_exploratory_panel"
MAIN_LAUNCHER = Path(__file__).with_name("inference_hub_main_accelerated.py")
PROVIDER_SAFE_V2 = Path(__file__).with_name("inference_hub_provider_safe_v2.py")
EXPECTED_POLICY = {
    "global_concurrency": 12,
    "provider_concurrency": 2,
    "global_requests_per_second": 8.0,
    "provider_requests_per_second": 1.5,
}
DEFAULT_MAX_ROUNDS = 3
DEFAULT_MAX_WORKERS = 8
DEFAULT_ROUND_INTERVAL_SECONDS = 30.0


class Part1SemanticInvalidRepairError(RuntimeError):
    """The repair evidence or execution contract is unsafe."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_digest(sources: Mapping[str, Any], basename: str) -> str | None:
    values = [
        value for path, value in sources.items()
        if Path(str(path)).name == basename
    ]
    return values[0] if len(values) == 1 and isinstance(values[0], str) else None


def _policy_hash(shared: Mapping[str, Any]) -> str:
    return base._sha256_json({
        key: value for key, value in shared.items() if key != "policy_sha256"
    })


def _require_private(path: Path) -> None:
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise Part1SemanticInvalidRepairError(
            f"Private evidence must have mode 0600: {path}"
        )


def _validate_source_manifest(
    path: Path,
) -> tuple[
    dict[str, Any],
    tuple[Any, ...],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
]:
    path = path.resolve()
    _require_private(path)
    manifest = base._read_json(path, "completed definitive Part 1 manifest")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != SOURCE_ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != base._self_hash(manifest)
        or manifest.get("complete") is not True
        or not manifest.get("completed_at_utc")
    ):
        raise Part1SemanticInvalidRepairError(
            "Source Part 1 manifest is not COMPLETE and hash-valid."
        )
    sources = manifest.get("source_artifacts")
    if not isinstance(sources, Mapping):
        raise Part1SemanticInvalidRepairError(
            "Source manifest lacks source bindings."
        )
    if (
        _source_digest(sources, PROVIDER_SAFE_V2.name)
        != base._sha256_file(PROVIDER_SAFE_V2)
    ):
        raise Part1SemanticInvalidRepairError(
            "Source provider-safe-v2 binding failed."
        )
    if (
        _source_digest(sources, MAIN_LAUNCHER.name)
        != base._sha256_file(MAIN_LAUNCHER)
    ):
        raise Part1SemanticInvalidRepairError(
            "Source main-accelerated launcher binding failed."
        )
    contract = manifest.get("execution_contract")
    shared = (
        contract.get("shared_rate_limit")
        if isinstance(contract, Mapping) else None
    )
    if (
        not isinstance(shared, Mapping)
        or shared.get("policy_sha256") != _policy_hash(shared)
        or any(
            shared.get(key) != value
            for key, value in EXPECTED_POLICY.items()
        )
    ):
        raise Part1SemanticInvalidRepairError(
            "Source main-accelerated rate-limit policy is wrong."
        )
    subject_rows = manifest.get("subject_routes")
    if not isinstance(subject_rows, list) or not subject_rows:
        raise Part1SemanticInvalidRepairError(
            "Source subject routes are missing."
        )
    subjects = {
        str(row.get("target_id")): row
        for row in subject_rows
        if isinstance(row, Mapping) and row.get("target_id")
    }
    if len(subjects) != len(subject_rows):
        raise Part1SemanticInvalidRepairError(
            "Source subject identities are invalid or duplicated."
        )
    judge = manifest.get("judge_reservation")
    if (
        not isinstance(judge, Mapping)
        or judge.get("dispatch_permitted_in_this_runner") is not False
    ):
        raise Part1SemanticInvalidRepairError(
            "Source judge reservation is unsafe."
        )
    if str(judge.get("target_id")) in subjects:
        raise Part1SemanticInvalidRepairError(
            "Source judge overlaps a subject."
        )

    trials = base.build_draft_trials(
        base_seed=int(manifest["base_seed"]),
        limit=manifest.get("trial_limit"),
    )
    trials_by_id = {trial.trial_id: trial for trial in trials}
    refs = manifest.get("journals")
    raw_refs = (
        refs.get("raw_responses") if isinstance(refs, Mapping) else None
    )
    if not isinstance(raw_refs, Mapping) or set(raw_refs) != set(subjects):
        raise Part1SemanticInvalidRepairError(
            "Source raw journal set changed."
        )
    attempt_ref = (
        refs.get("attempt_ledger")
        if isinstance(refs, Mapping) else None
    )
    if (
        not isinstance(attempt_ref, Mapping)
        or not isinstance(attempt_ref.get("path"), str)
    ):
        raise Part1SemanticInvalidRepairError(
            "Source attempt-ledger reference is malformed."
        )
    attempt_path = Path(attempt_ref["path"]).resolve()
    try:
        attempt_path.relative_to(path.parent)
    except ValueError as error:
        raise Part1SemanticInvalidRepairError(
            "Source attempt ledger escaped its private directory."
        ) from error
    source_attempts = base._ChainedJournal(attempt_path)
    base._validate_checkpoint_reference(
        source_attempts,
        attempt_ref,
        label="source attempt ledger",
    )
    if len(source_attempts.records) != attempt_ref.get(
        "record_count"
    ):
        raise Part1SemanticInvalidRepairError(
            "Source attempt ledger grew beyond its COMPLETE checkpoint."
        )
    raw_journals: dict[str, base._ChainedJournal] = {}
    for target, reference in raw_refs.items():
        if (
            not isinstance(reference, Mapping)
            or not isinstance(reference.get("path"), str)
        ):
            raise Part1SemanticInvalidRepairError(
                "Source raw journal reference is malformed."
            )
        journal_path = Path(reference["path"]).resolve()
        try:
            journal_path.relative_to(path.parent)
        except ValueError as error:
            raise Part1SemanticInvalidRepairError(
                "Source journal escaped its private directory."
            ) from error
        journal = base._ChainedJournal(journal_path)
        base._validate_checkpoint_reference(
            journal, reference, label=f"source raw/{target}"
        )
        if len(journal.records) != reference.get("record_count"):
            raise Part1SemanticInvalidRepairError(
                "Source journal grew beyond its COMPLETE checkpoint."
            )
        raw_journals[str(target)] = journal
    retained = base._completed_index(
        raw_journals,
        trials_by_id=trials_by_id,
        subjects_by_id=subjects,
    )
    planned = len(subjects) * len(trials)
    if (
        len(retained) != planned
        or manifest.get("summary", {}).get("retained_trial_records") != planned
    ):
        raise Part1SemanticInvalidRepairError(
            "Source COMPLETE unit accounting failed."
        )
    if any(
        row.get("raw_response") is None
        or row.get("model_identity_valid") is not True
        for row in retained.values()
    ):
        raise Part1SemanticInvalidRepairError(
            "Source COMPLETE panel contains operational failures."
        )

    for (target, trial_id), row in retained.items():
        if row.get("format_valid") is not False:
            continue
        subject, trial = subjects[target], trials_by_id[trial_id]
        original_body, _ = base._request_contract(subject, trial)
        if (
            row.get("prompt_sha256") != trial.prompt_hash
            or row.get("prompt_text") != trial.prompt_text
            or row.get("requested_route") != subject.get("route")
            or row.get("request_sha256") != base._sha256_json(original_body)
            or row.get("target_id") != target
            or row.get("model") != subject.get("model")
            or row.get("upstream_provider")
            != subject.get("upstream_provider")
        ):
            raise Part1SemanticInvalidRepairError(
                "Original prompt/request/identity drifted for "
                f"{target}/{trial_id}."
            )
    return manifest, trials, subjects, retained


def _attempt_id(
    target_id: str,
    trial_id: str,
    original_record_sha256: str,
    round_index: int,
) -> str:
    digest = hashlib.sha256(
        (
            f"{target_id}\0{trial_id}\0"
            f"{original_record_sha256}\0{round_index}"
        ).encode()
    ).hexdigest()
    return f"p1repair_{digest[:32]}"


def _repair_seed(
    original_seed: int,
    target_id: str,
    trial_id: str,
    original_record_sha256: str,
    round_index: int,
) -> int:
    digest = hashlib.sha256(
        (
            "part1-semantic-repair-v1\0"
            f"{target_id}\0{trial_id}\0"
            f"{original_record_sha256}\0{round_index}"
        ).encode()
    ).digest()
    seed = int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF
    return (seed + 1) & 0x7FFF_FFFF if seed == original_seed else seed


def _manifest_bindings(
    source_path: Path,
    source: Mapping[str, Any],
    max_rounds: int,
) -> dict[str, Any]:
    this_file = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "source_manifest": {
            "path": str(source_path.resolve()),
            "file_sha256": base._sha256_file(source_path),
            "evidence_sha256": source["evidence_sha256"],
        },
        "source_artifact_type": SOURCE_ARTIFACT_TYPE,
        "source_completed_at_utc": source["completed_at_utc"],
        "max_semantic_rounds": max_rounds,
        "round_seed_policy":
            "sha256_v1_new_seed_only_when_route_supports_seed",
        "primary_records_mutated": False,
        "primary_denominators_changed": False,
        "judge_dispatched": False,
        "promotion_permitted": False,
        "source_artifacts": {
            str(this_file): base._sha256_file(this_file),
            str(MAIN_LAUNCHER.resolve()):
                base._sha256_file(MAIN_LAUNCHER),
        },
    }


def _repair_request(
    subject: Mapping[str, Any],
    trial: Any,
    original: Mapping[str, Any],
    round_index: int,
) -> tuple[dict[str, Any], int | None]:
    body, _ = base._request_contract(subject, trial)
    new_seed = None
    if "seed" in subject.get("supported_controls", []):
        original_seed = body.get("seed")
        if (
            not isinstance(original_seed, int)
            or isinstance(original_seed, bool)
        ):
            raise Part1SemanticInvalidRepairError(
                "Seed-supported original request lacks an integer seed."
            )
        new_seed = _repair_seed(
            original_seed,
            str(subject["target_id"]),
            str(trial.trial_id),
            str(original["record_sha256"]),
            round_index,
        )
        body["seed"] = new_seed
    elif "seed" in body:
        raise Part1SemanticInvalidRepairError(
            "Unsupported seed unexpectedly appears in request."
        )
    return body, new_seed


def _reconcile(
    ledger: base._ChainedJournal,
    raw: base._ChainedJournal,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, str, int], Mapping[str, Any]],
]:
    reservations: dict[str, Mapping[str, Any]] = {}
    completions: dict[str, Mapping[str, Any]] = {}
    for row in ledger.records:
        attempt_id = row.get("attempt_id")
        if (
            row.get("event") == "reserved_before_dispatch"
            and isinstance(attempt_id, str)
        ):
            if attempt_id in reservations:
                raise Part1SemanticInvalidRepairError(
                    "Repair attempt was reserved twice."
                )
            reservations[attempt_id] = row
        elif (
            row.get("event") == "attempt_completed"
            and isinstance(attempt_id, str)
        ):
            if (
                attempt_id in completions
                or attempt_id not in reservations
            ):
                raise Part1SemanticInvalidRepairError(
                    "Repair completion binding failed."
                )
            completions[attempt_id] = row
        else:
            raise Part1SemanticInvalidRepairError(
                "Unknown repair ledger event."
            )

    raw_by_unit: dict[
        tuple[str, str, int], Mapping[str, Any]
    ] = {}
    raw_by_attempt: dict[str, Mapping[str, Any]] = {}
    for row in raw.records:
        attempt_id = row.get("attempt_id")
        round_index = row.get("round_index")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
        ):
            raise Part1SemanticInvalidRepairError(
                "Repair response round is invalid."
            )
        key = (
            str(row.get("target_id")),
            str(row.get("trial_id")),
            round_index,
        )
        reservation = reservations.get(str(attempt_id))
        if (
            not isinstance(attempt_id, str)
            or attempt_id in raw_by_attempt
            or key in raw_by_unit
            or reservation is None
            or reservation.get("target_id") != row.get("target_id")
            or reservation.get("trial_id") != row.get("trial_id")
            or reservation.get("round_index") != round_index
            or reservation.get("original_record_sha256")
            != row.get("original_record_sha256")
            or reservation.get("request_sha256")
            != row.get("request_sha256")
            or row.get("raw_response_sha256")
            != base._sha256_json(row.get("raw_response"))
        ):
            raise Part1SemanticInvalidRepairError(
                "Private repair response binding failed."
            )
        if row.get("model_identity_valid") is not True:
            raise Part1SemanticInvalidRepairError(
                "Retained repair response has an identity mismatch."
            )
        raw_by_attempt[attempt_id] = row
        raw_by_unit[key] = row
        if attempt_id not in completions:
            ledger.append({
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "event": "attempt_completed",
                "attempt_id": attempt_id,
                "outcome": "response_retained",
                "recovered_after_raw_fsync": True,
                "completed_at_utc": _utc_now(),
            })
            completions[attempt_id] = ledger.records[-1]
    for attempt_id, completion in completions.items():
        if (
            completion.get("outcome") == "response_retained"
            and attempt_id not in raw_by_attempt
        ):
            raise Part1SemanticInvalidRepairError(
                "Retained repair completion lacks raw evidence."
            )
    for attempt_id in sorted(set(reservations) - set(completions)):
        ledger.append({
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "event": "attempt_completed",
            "attempt_id": attempt_id,
            "outcome": "indeterminate_after_crash",
            "redispatch_same_round": False,
            "completed_at_utc": _utc_now(),
        })
        completions[attempt_id] = ledger.records[-1]
    return reservations, completions, raw_by_unit


def _outcomes(
    eligible: Sequence[Mapping[str, Any]],
    raw_by_unit: Mapping[
        tuple[str, str, int], Mapping[str, Any]
    ],
    reservations: Mapping[str, Mapping[str, Any]],
    max_rounds: int,
) -> list[dict[str, Any]]:
    rows = []
    for original in sorted(
        eligible,
        key=lambda row: (
            str(row["target_id"]),
            str(row["trial_id"]),
        ),
    ):
        attempts = [
            raw_by_unit[key]
            for key in sorted(raw_by_unit)
            if (
                key[0] == original["target_id"]
                and key[1] == original["trial_id"]
            )
        ]
        success = next((
            row
            for row in attempts
            if (
                row.get("format_valid") is True
                and row.get("model_identity_valid") is True
            )
        ), None)
        reserved = sum(
            _attempt_id(
                original["target_id"],
                original["trial_id"],
                original["record_sha256"],
                index,
            ) in reservations
            for index in range(1, max_rounds + 1)
        )
        rows.append({
            "target_id": original["target_id"],
            "upstream_provider": original["upstream_provider"],
            "model": original["model"],
            "trial_id": original["trial_id"],
            "root_id": original["root_id"],
            "original_record_sha256":
                original["record_sha256"],
            "original_request_sha256":
                original["request_sha256"],
            "original_prompt_sha256":
                original["prompt_sha256"],
            "original_format_valid": False,
            "original_finish_reason":
                original.get("finish_reason"),
            "rounds_reserved": reserved,
            "rounds_with_retained_response": len(attempts),
            "repair_status": (
                "repaired_valid_separate"
                if success
                else "unrepaired_after_bounded_rounds"
            ),
            "repaired_format_valid": success is not None,
            "successful_round": (
                success.get("round_index") if success else None
            ),
            "repaired_action": (
                success.get("parsed_action") if success else None
            ),
            "repaired_response_text_sha256": (
                success.get("response_text_sha256")
                if success else None
            ),
            "primary_record_mutated": False,
            "primary_denominator_changed": False,
        })
    return rows


def run_repair(
    *,
    source_manifest_path: Path,
    output_dir: Path,
    client: Any | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    round_interval_seconds: float
        = DEFAULT_ROUND_INTERVAL_SECONDS,
    resume: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
    after_raw_hook: Callable[
        [Mapping[str, Any]], None
    ] | None = None,
) -> dict[str, Any]:
    """Run bounded repair rounds without changing source evidence."""

    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds < 1
    ):
        raise Part1SemanticInvalidRepairError(
            "max_rounds must be a positive integer."
        )
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers < 1
    ):
        raise Part1SemanticInvalidRepairError(
            "max_workers must be a positive integer."
        )
    if round_interval_seconds < 0:
        raise Part1SemanticInvalidRepairError(
            "round_interval_seconds cannot be negative."
        )
    source_path = source_manifest_path.resolve()
    source, trials, subjects, retained = (
        _validate_source_manifest(source_path)
    )
    trials_by_id = {
        trial.trial_id: trial for trial in trials
    }
    eligible = [
        row
        for row in retained.values()
        if row.get("format_valid") is False
    ]
    bindings = _manifest_bindings(
        source_path, source, max_rounds
    )

    if output_dir.exists() and not resume:
        raise Part1SemanticInvalidRepairError(
            "Repair output exists; use resume explicitly."
        )
    private = output_dir / "private"
    sanitized = output_dir / "sanitized"
    if not output_dir.exists():
        private.mkdir(parents=True, mode=0o700)
        sanitized.mkdir(parents=True)
        if os.name == "posix":
            private.chmod(0o700)
    manifest_path = private / "manifest.json"
    ledger = base._ChainedJournal(
        private / "attempt_ledger.jsonl"
    )
    raw = base._ChainedJournal(
        private / "raw_repair_responses.jsonl"
    )
    run_lock = base._acquire_run_lock(private)
    try:
        if resume:
            manifest = base._read_json(
                manifest_path, "repair manifest"
            )
            if (
                manifest.get("evidence_sha256")
                != base._self_hash(manifest)
            ):
                raise Part1SemanticInvalidRepairError(
                    "Repair manifest self-hash failed."
                )
            observed = {
                key: manifest.get(key) for key in bindings
            }
            if observed != bindings:
                raise Part1SemanticInvalidRepairError(
                    "Repair/source bindings changed on resume."
                )
            refs = manifest.get("journals")
            if not isinstance(refs, Mapping):
                raise Part1SemanticInvalidRepairError(
                    "Repair manifest lacks journal checkpoints."
                )
            base._validate_checkpoint_reference(
                ledger,
                refs.get("attempt_ledger"),
                label="repair attempt ledger",
            )
            base._validate_checkpoint_reference(
                raw,
                refs.get("raw_responses"),
                label="repair raw responses",
            )
        else:
            manifest = {
                **bindings,
                "created_at_utc": _utc_now(),
                "last_updated_at_utc": _utc_now(),
                "complete": False,
                "eligible_invalid_unit_count": len(eligible),
                "summary": {},
                "journals": {
                    "attempt_ledger": ledger.reference(),
                    "raw_responses": raw.reference(),
                },
            }
            base._seal(manifest)
            base._atomic_json(manifest_path, manifest)

        reservations, _, raw_by_unit = _reconcile(
            ledger, raw
        )
        model_client = client
        if model_client is None and eligible:
            model_client = _accelerated_client(180.0)

        def execute(
            original: Mapping[str, Any],
            round_index: int,
        ) -> None:
            target = str(original["target_id"])
            trial_id = str(original["trial_id"])
            subject = subjects[target]
            trial = trials_by_id[trial_id]
            body, new_seed = _repair_request(
                subject, trial, original, round_index
            )
            request_sha = base._sha256_json(body)
            attempt_id = _attempt_id(
                target,
                trial_id,
                str(original["record_sha256"]),
                round_index,
            )
            ledger.append({
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "event": "reserved_before_dispatch",
                "attempt_id": attempt_id,
                "target_id": target,
                "trial_id": trial_id,
                "original_record_sha256":
                    original["record_sha256"],
                "original_request_sha256":
                    original["request_sha256"],
                "round_index": round_index,
                "requested_route": subject["route"],
                "repair_seed": new_seed,
                "request_sha256": request_sha,
                "reserved_at_utc": _utc_now(),
            })
            try:
                if model_client is None:  # pragma: no cover - no-op guard
                    raise Part1SemanticInvalidRepairError(
                        "Repair client is unavailable."
                    )
                response = model_client.post(
                    "/chat/completions", body
                )
            except Exception as error:
                transient, code, status = base._transient(
                    error
                )
                ledger.append({
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": ARTIFACT_TYPE,
                    "event": "attempt_completed",
                    "attempt_id": attempt_id,
                    "outcome": "transport_failure",
                    "failure_code": code,
                    "transient": transient,
                    "http_status": status,
                    "completed_at_utc": _utc_now(),
                })
                if not transient:
                    raise Part1SemanticInvalidRepairError(
                        "Non-transient repair dispatch failure."
                    ) from error
                return
            metadata = base._response_metadata(response)
            identity_valid = (
                response.get("model") == subject["route"]
            )
            retained_row = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "attempt_id": attempt_id,
                "round_index": round_index,
                "target_id": target,
                "upstream_provider":
                    subject["upstream_provider"],
                "model": subject["model"],
                "requested_route": subject["route"],
                "trial_id": trial_id,
                "root_id": trial.root_id,
                "original_record_sha256":
                    original["record_sha256"],
                "original_request_sha256":
                    original["request_sha256"],
                "original_prompt_sha256":
                    original["prompt_sha256"],
                "request_sha256": request_sha,
                "repair_seed": new_seed,
                "model_identity_valid": identity_valid,
                **metadata,
                "raw_response": dict(response),
                "raw_response_sha256":
                    base._sha256_json(response),
                "retained_at_utc": _utc_now(),
            }
            raw.append(retained_row)
            if after_raw_hook is not None:
                after_raw_hook(retained_row)
            ledger.append({
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "event": "attempt_completed",
                "attempt_id": attempt_id,
                "outcome": "response_retained",
                "identity_valid": identity_valid,
                "format_valid": metadata["format_valid"],
                "completed_at_utc": _utc_now(),
            })
            if not identity_valid:
                raise Part1SemanticInvalidRepairError(
                    "Repair response model identity mismatch."
                )

        for round_index in range(1, max_rounds + 1):
            reservations, _, raw_by_unit = _reconcile(
                ledger, raw
            )
            successful = {
                (target, trial_id)
                for (
                    target, trial_id, _
                ), row in raw_by_unit.items()
                if (
                    row.get("format_valid") is True
                    and row.get("model_identity_valid") is True
                )
            }
            work = []
            for original in eligible:
                key = (
                    str(original["target_id"]),
                    str(original["trial_id"]),
                )
                attempt_id = _attempt_id(
                    key[0],
                    key[1],
                    str(original["record_sha256"]),
                    round_index,
                )
                if (
                    key not in successful
                    and attempt_id not in reservations
                ):
                    work.append(original)
            errors: list[BaseException] = []
            if work:
                with ThreadPoolExecutor(
                    max_workers=min(max_workers, len(work)),
                    thread_name_prefix=(
                        f"part1-repair-r{round_index}"
                    ),
                ) as executor:
                    futures = [
                        executor.submit(
                            execute, original, round_index
                        )
                        for original in work
                    ]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except BaseException as error:
                            errors.append(error)
                if errors:
                    raise errors[0]
            reservations, _, raw_by_unit = _reconcile(
                ledger, raw
            )
            unresolved = [
                original
                for original in eligible
                if not any(
                    key[0] == original["target_id"]
                    and key[1] == original["trial_id"]
                    and row.get("format_valid") is True
                    and row.get("model_identity_valid") is True
                    for key, row in raw_by_unit.items()
                )
            ]
            manifest["last_updated_at_utc"] = _utc_now()
            manifest["journals"] = {
                "attempt_ledger": ledger.reference(),
                "raw_responses": raw.reference(),
            }
            manifest["summary"] = {
                "rounds_completed": round_index,
                "unresolved_invalid_units": len(unresolved),
            }
            base._seal(manifest)
            base._atomic_json(manifest_path, manifest)
            if not unresolved:
                break
            if round_index < max_rounds:
                sleep_fn(round_interval_seconds)

        reservations, _, raw_by_unit = _reconcile(
            ledger, raw
        )
        outcome_rows = _outcomes(
            eligible,
            raw_by_unit,
            reservations,
            max_rounds,
        )
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type":
                "inference_hub_part1_semantic_invalid_"
                "repair_outcomes_v1",
            "source_manifest_evidence_sha256":
                source["evidence_sha256"],
            "raw_text_included": False,
            "original_records_mutated": False,
            "primary_denominators_changed": False,
            "repaired_estimates_separate_only": True,
            "judge_dispatched": False,
            "promotion_permitted": False,
            "rows": outcome_rows,
        }
        base._seal(payload)
        outcome_path = sanitized / "repair_outcomes.json"
        base._atomic_json(outcome_path, payload)
        manifest["complete"] = True
        manifest["completed_at_utc"] = _utc_now()
        manifest["last_updated_at_utc"] = _utc_now()
        manifest["journals"] = {
            "attempt_ledger": ledger.reference(),
            "raw_responses": raw.reference(),
        }
        manifest["summary"] = {
            "source_scheduled_unit_count": len(retained),
            "source_first_attempt_invalid_count":
                len(eligible),
            "repaired_valid_separate_count": sum(
                row["repaired_format_valid"]
                for row in outcome_rows
            ),
            "unrepaired_after_bounded_rounds_count": sum(
                not row["repaired_format_valid"]
                for row in outcome_rows
            ),
            "primary_denominator": len(retained),
            "primary_denominator_changed": False,
        }
        manifest["sanitized_artifact"] = {
            "path": str(outcome_path.resolve()),
            "file_sha256":
                base._sha256_file(outcome_path),
            "evidence_sha256":
                payload["evidence_sha256"],
        }
        base._seal(manifest)
        base._atomic_json(manifest_path, manifest)
        return manifest
    finally:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
        run_lock.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
    )
    parser.add_argument(
        "--round-interval-seconds",
        type=float,
        default=DEFAULT_ROUND_INTERVAL_SECONDS,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        run_repair(
            source_manifest_path=args.source_manifest,
            output_dir=args.output_dir,
            max_rounds=args.max_rounds,
            max_workers=args.max_workers,
            round_interval_seconds=
                args.round_interval_seconds,
            resume=args.resume,
        )
    except Part1SemanticInvalidRepairError as error:
        print(
            "Part 1 semantic-invalid repair failed: "
            f"{error}"
        )
        return 1
    print(
        "Wrote separate Part 1 repair evidence: "
        f"{args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
