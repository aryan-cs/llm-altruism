"""Build the final, fail-closed confirmatory data-lock artifact."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis import judge_audit
from experiments import confirmatory_campaign
from experiments.confirmatory_budget import validate_frozen_budget, validate_ledger
from experiments.misc.attempt_log import attempt_log_path_for_csv, load_attempt_records
from experiments.misc.run_metadata import sha256_file, stable_json_hash


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "confirmatory_data_lock"
EXCLUSIONS_TYPE = "confirmatory_exclusion_decisions"
OBJECTIVE_EXCLUSION_REASON_CODES = (
    "prespecified_route_identity_failure",
    "prespecified_artifact_integrity_failure",
    "prespecified_protocol_deviation",
    "prespecified_incomplete_execution_unit",
)
PRIMARY_COHORT_ID = "current_sota"
HISTORICAL_COHORT_ID = "historical"
EXPECTED_PRIMARY_SYSTEMS = 24
EXPECTED_PRIMARY_DEVELOPERS = 12
EXPECTED_HISTORICAL_SYSTEMS = 6
_EXCLUSION_KEYS = {
    "schema_version", "artifact_type", "status", "campaign_plan_sha256s",
    "policy_frozen_at_utc", "policy_frozen_by", "allowed_reason_codes",
    "decision_reviewed_at_utc", "reviewed_by", "excluded_job_ids", "decisions",
}
_DECISION_KEYS = {
    "job_id", "reason_code", "evidence_path", "evidence_sha256",
    "decided_by", "decided_at_utc", "outcome_blind",
}
_JUDGE_RESULT_KEYS = {
    "schema_version", "workflow", "annotator_ids", "primary_rows",
    "duplicate_rows_per_annotator", "adjudication", "weighting",
    "unclear_handling", "overall", "per_language",
    "bootstrap_confidence_intervals", "inter_rater_reliability",
    "intra_rater_reliability", "input_integrity", "criterion_promotion_gate",
    "scoring_parameters", "result_sha256",
}


class ConfirmatoryDataLockError(ValueError):
    """A required completed artifact, approval, or hash is missing."""


def _validate_cohort_estimand(
    campaign: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return exact cohort/developer membership and enforce primary-panel minima."""

    targets = campaign.get("targets")
    cohorts = campaign.get("cohorts")
    if not isinstance(targets, list) or not targets or not isinstance(cohorts, list):
        raise ConfirmatoryDataLockError("campaign lacks frozen targets/cohorts")
    target_by_id = {
        str(target.get("id")): target for target in targets if isinstance(target, Mapping)
    }
    if len(target_by_id) != len(targets):
        raise ConfirmatoryDataLockError("campaign target IDs are empty or duplicated")
    cohort_by_target: dict[str, str] = {}
    for cohort in cohorts:
        if not isinstance(cohort, Mapping):
            raise ConfirmatoryDataLockError("campaign cohort entry is invalid")
        cohort_id = cohort.get("id")
        members = cohort.get("target_ids")
        if cohort_id not in {PRIMARY_COHORT_ID, HISTORICAL_COHORT_ID}:
            raise ConfirmatoryDataLockError("campaign contains an unsupported cohort")
        if not isinstance(members, list) or not all(
            isinstance(member, str) and member for member in members
        ):
            raise ConfirmatoryDataLockError("campaign cohort target_ids are invalid")
        for member in members:
            if member not in target_by_id or member in cohort_by_target:
                raise ConfirmatoryDataLockError(
                    "campaign cohort membership is unknown, overlapping, or duplicated"
                )
            cohort_by_target[member] = str(cohort_id)
    if set(cohort_by_target) != set(target_by_id):
        raise ConfirmatoryDataLockError(
            "campaign cohorts do not exactly partition the frozen target panel"
        )
    metadata: list[dict[str, str]] = []
    for target in targets:
        system_id = str(target["id"])
        developer_id = target.get("upstream_provider")
        if not isinstance(developer_id, str) or not developer_id.strip():
            raise ConfirmatoryDataLockError(
                f"campaign target {system_id} lacks upstream_provider"
            )
        metadata.append(
            {
                "system_id": system_id,
                "cohort_id": cohort_by_target[system_id],
                "developer_id": developer_id.strip(),
            }
        )
    current = [row for row in metadata if row["cohort_id"] == PRIMARY_COHORT_ID]
    developers = {row["developer_id"] for row in current}
    historical = [
        row for row in metadata if row["cohort_id"] == HISTORICAL_COHORT_ID
    ]
    if (
        len(current) != EXPECTED_PRIMARY_SYSTEMS
        or len(developers) != EXPECTED_PRIMARY_DEVELOPERS
        or len(historical) != EXPECTED_HISTORICAL_SYSTEMS
    ):
        raise ConfirmatoryDataLockError(
            "fixed panel requires exactly 24 current systems from 12 developers "
            "and 6 historical systems"
        )
    return metadata


def _load(path: str | Path, label: str) -> tuple[Path, dict[str, Any], str]:
    resolved = Path(path).resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryDataLockError(f"{label} is not valid JSON: {resolved}") from error
    if not isinstance(payload, dict):
        raise ConfirmatoryDataLockError(f"{label} root must be an object")
    return resolved, payload, sha256_file(resolved)


def _utc(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfirmatoryDataLockError(f"{label} must be a nonempty UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ConfirmatoryDataLockError(f"{label} must be ISO-8601 UTC") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ConfirmatoryDataLockError(f"{label} must be ISO-8601 UTC")
    return value


def _utc_datetime(value: object, label: str) -> datetime:
    normalized = _utc(value, label)
    return datetime.fromisoformat(normalized.replace("Z", "+00:00"))


def _scientific_attempt_timing(
    job: Mapping[str, Any], *, campaign_stage: str
) -> dict[str, Any]:
    attempts = job.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ConfirmatoryDataLockError(
            f"scientific job {job.get('id')} has no timestamped execution attempts"
        )
    expected_keys = {
        "attempt", "started_at_utc", "argv_sha256", "resume",
        "finished_at_utc", "returncode", "timed_out", "error",
    }
    history: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, Mapping) or set(attempt) != expected_keys:
            raise ConfirmatoryDataLockError(
                f"scientific job {job.get('id')} attempt schema is not exact"
            )
        if attempt.get("attempt") != index:
            raise ConfirmatoryDataLockError(
                f"scientific job {job.get('id')} attempt sequence is invalid"
            )
        started = _utc_datetime(
            attempt.get("started_at_utc"),
            f"scientific job {job.get('id')} attempt {index} started_at_utc",
        )
        finished = _utc_datetime(
            attempt.get("finished_at_utc"),
            f"scientific job {job.get('id')} attempt {index} finished_at_utc",
        )
        if finished < started:
            raise ConfirmatoryDataLockError(
                f"scientific job {job.get('id')} attempt finished before it started"
            )
        if (
            not isinstance(attempt.get("argv_sha256"), str)
            or len(str(attempt["argv_sha256"])) != 64
            or not isinstance(attempt.get("resume"), bool)
            or not isinstance(attempt.get("returncode"), int)
            or not isinstance(attempt.get("timed_out"), bool)
            or not (
                attempt.get("error") is None
                or isinstance(attempt.get("error"), str)
            )
        ):
            raise ConfirmatoryDataLockError(
                f"scientific job {job.get('id')} attempt state is invalid"
            )
        history.append(
            {
                "attempt": index,
                "started_at_utc": attempt["started_at_utc"],
                "finished_at_utc": attempt["finished_at_utc"],
                "duration_seconds": (finished - started).total_seconds(),
                "returncode": attempt["returncode"],
                "timed_out": attempt["timed_out"],
                "resume": attempt["resume"],
            }
        )
    completed = attempts[-1]
    if (
        completed.get("returncode") != 0
        or completed.get("timed_out") is not False
        or completed.get("error") not in {None, ""}
    ):
        raise ConfirmatoryDataLockError(
            f"scientific job {job.get('id')} lacks a successful terminal attempt"
        )
    return {
        "campaign_stage": campaign_stage,
        "job_id": str(job["id"]),
        "experiment": str(job["experiment"]),
        "target_id": str(job.get("target_id", "")),
        "completed_attempt": history[-1],
        "attempt_history": history,
    }


def _collection_timing(
    campaigns: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    job_rows: list[dict[str, Any]] = []
    stage_ranges: dict[str, dict[str, Any]] = {}
    for campaign_stage, jobs in campaigns.items():
        stage_rows = [
            _scientific_attempt_timing(job, campaign_stage=campaign_stage)
            for job in jobs
            if job.get("stage") != "smoke"
        ]
        if not stage_rows:
            raise ConfirmatoryDataLockError(
                f"{campaign_stage} has no timestamped scientific jobs"
            )
        job_rows.extend(stage_rows)

        def range_for(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            starts = [
                (
                    _utc_datetime(
                        row["completed_attempt"]["started_at_utc"],
                        "completed attempt start",
                    ),
                    row["completed_attempt"]["started_at_utc"],
                )
                for row in rows
            ]
            finishes = [
                (
                    _utc_datetime(
                        row["completed_attempt"]["finished_at_utc"],
                        "completed attempt finish",
                    ),
                    row["completed_attempt"]["finished_at_utc"],
                )
                for row in rows
            ]
            first = min(starts, key=lambda value: value[0])
            last = max(finishes, key=lambda value: value[0])
            return {
                "job_count": len(rows),
                "started_at_utc": first[1],
                "finished_at_utc": last[1],
                "elapsed_seconds": (last[0] - first[0]).total_seconds(),
            }

        parts = sorted({str(row["experiment"]) for row in stage_rows})
        stage_ranges[campaign_stage] = {
            **range_for(stage_rows),
            "parts": {
                part: range_for(
                    [row for row in stage_rows if row["experiment"] == part]
                )
                for part in parts
            },
        }
    variance_finished = _utc_datetime(
        stage_ranges["variance_stage"]["finished_at_utc"],
        "variance-stage finish",
    )
    baseline_started = _utc_datetime(
        stage_ranges["baseline_stage"]["started_at_utc"],
        "baseline-stage start",
    )
    return {
        "policy": "stage_separated_timing_without_posthoc_cross_stage_cutoff",
        "exact_attempt_timestamps_preserved": True,
        "stage_part_temporal_confounding": {
            "flagged": True,
            "reason": (
                "Part 0/Part 1 final outcomes and Part 2 final outcomes were "
                "collected in separate campaign stages; calendar-time drift is "
                "therefore structurally confounded with part/stage."
            ),
            "baseline_start_minus_variance_finish_seconds": (
                baseline_started - variance_finished
            ).total_seconds(),
        },
        "per_stage": stage_ranges,
        "scientific_job_attempts": job_rows,
    }


def _single_stage_collection_timing(
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        _scientific_attempt_timing(job, campaign_stage="fixed_stage")
        for job in jobs
        if job.get("stage") != "smoke"
    ]
    if not rows:
        raise ConfirmatoryDataLockError("fixed_stage has no timestamped scientific jobs")
    starts = [
        _utc_datetime(row["completed_attempt"]["started_at_utc"], "attempt start")
        for row in rows
    ]
    finishes = [
        _utc_datetime(row["completed_attempt"]["finished_at_utc"], "attempt finish")
        for row in rows
    ]
    return {
        "policy": "single_block_randomized_campaign_timing_v1",
        "exact_attempt_timestamps_preserved": True,
        "stage_part_temporal_confounding": {
            "flagged": False,
            "reason": "All three parts were interleaved in one preregistered block-randomized campaign.",
        },
        "per_stage": {
            "fixed_stage": {
                "job_count": len(rows),
                "started_at_utc": min(starts).isoformat().replace("+00:00", "Z"),
                "finished_at_utc": max(finishes).isoformat().replace("+00:00", "Z"),
                "elapsed_seconds": (max(finishes) - min(starts)).total_seconds(),
            }
        },
        "scientific_job_attempts": rows,
    }


def _reference(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise ConfirmatoryDataLockError(f"required {kind} file is missing: {path}")
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_judge_gate(path: str | Path) -> dict[str, Any]:
    resolved, payload, digest = _load(path, "judge criterion gate")
    if set(payload) != _JUDGE_RESULT_KEYS or payload.get("schema_version") != 2 or payload.get("workflow") != "part0_multi_annotator_human_judge_validation":
        raise ConfirmatoryDataLockError("judge criterion artifact is not the multi-annotator confirmatory workflow")
    gate = payload.get("criterion_promotion_gate")
    if not isinstance(gate, dict) or set(gate) != {
        "passed", "promotion_authorized", "thresholds", "design_passed",
        "multifactor_allocation_passed", "failures"
    }:
        raise ConfirmatoryDataLockError("judge criterion gate schema is not exact")
    expected_thresholds = {
        "minimum_lower_95_bound": judge_audit.PROMOTION_MIN_LOWER_BOUND,
        "maximum_weighted_human_unclear_rate": (
            judge_audit.PROMOTION_MAX_UNCLEAR_RATE
        ),
        "maximum_weighted_automated_unclear_rate": (
            judge_audit.PROMOTION_MAX_AUTOMATED_UNCLEAR_RATE
        ),
        "minimum_inter_and_intra_rater_kappa": judge_audit.PROMOTION_MIN_KAPPA,
        "primary_items_per_stratum": judge_audit.DEFAULT_PER_STRATUM,
        "duplicates_per_annotator": judge_audit.DEFAULT_DUPLICATES,
        "annotators": judge_audit.DEFAULT_ANNOTATORS,
    }
    if gate.get("thresholds") != expected_thresholds:
        raise ConfirmatoryDataLockError(
            "judge criterion thresholds do not include the current automated-unclear gate"
        )
    recorded_result_hash = payload.get("result_sha256")
    expected_result_hash = stable_json_hash(
        {key: value for key, value in payload.items() if key != "result_sha256"}
    )
    if recorded_result_hash != expected_result_hash:
        raise ConfirmatoryDataLockError("judge criterion result self-hash is invalid")
    integrity = payload.get("input_integrity")
    annotators = payload.get("annotator_ids")
    if not isinstance(integrity, dict) or set(integrity) != {
        "audit_key", "annotations", "duplicate_annotations", "adjudications"
    } or not isinstance(annotators, list) or len(annotators) != judge_audit.DEFAULT_ANNOTATORS:
        raise ConfirmatoryDataLockError("judge criterion input-integrity schema is not exact")

    def validate_ref(value: object, label: str) -> None:
        if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
            raise ConfirmatoryDataLockError(f"judge criterion {label} reference is invalid")
        source = Path(str(value["path"]))
        if not source.is_file() or sha256_file(source) != value["sha256"] or source.stat().st_size != value["size_bytes"]:
            raise ConfirmatoryDataLockError(f"judge criterion {label} bytes changed")

    validate_ref(integrity["audit_key"], "audit key")
    validate_ref(integrity["adjudications"], "adjudications")
    for collection_name in ("annotations", "duplicate_annotations"):
        collection = integrity[collection_name]
        if not isinstance(collection, dict) or set(collection) != set(annotators):
            raise ConfirmatoryDataLockError(f"judge criterion {collection_name} coverage changed")
        for annotator_id, reference in collection.items():
            validate_ref(reference, f"{collection_name}:{annotator_id}")
    parameters = payload.get("scoring_parameters")
    if not isinstance(parameters, dict) or set(parameters) != {
        "bootstrap_replicates", "seed"
    }:
        raise ConfirmatoryDataLockError("judge criterion scoring parameters are missing")
    try:
        recomputed = judge_audit.score_multi_audit(
            integrity["audit_key"]["path"],
            {
                annotator_id: reference["path"]
                for annotator_id, reference in integrity["annotations"].items()
            },
            {
                annotator_id: reference["path"]
                for annotator_id, reference in integrity["duplicate_annotations"].items()
            },
            integrity["adjudications"]["path"],
            bootstrap_replicates=int(parameters["bootstrap_replicates"]),
            seed=int(parameters["seed"]),
        )
    except (judge_audit.AuditError, TypeError, ValueError) as error:
        raise ConfirmatoryDataLockError(
            f"judge criterion deterministic recomputation failed: {error}"
        ) from error
    if recomputed != payload:
        raise ConfirmatoryDataLockError(
            "judge criterion metrics/gate differ from deterministic recomputation"
        )
    try:
        judge_audit.require_criterion_promotion(recomputed)
    except judge_audit.AuditError as error:
        raise ConfirmatoryDataLockError(str(error)) from error
    return {
        "path": str(resolved),
        "sha256": digest,
        "workflow": payload["workflow"],
        "promotion_authorized": True,
        "input_integrity_sha256": stable_json_hash(integrity),
        "audit_key": dict(integrity["audit_key"]),
    }


def _validate_judge_campaign_lineage(
    judge_gate: Mapping[str, Any], campaign: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind the human-audit population to every frozen Part 0 production route."""

    key_ref = judge_gate.get("audit_key")
    if not isinstance(key_ref, Mapping):
        raise ConfirmatoryDataLockError("judge gate lacks its private audit-key reference")
    key_path = Path(str(key_ref.get("path", ""))).resolve()
    try:
        with key_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ConfirmatoryDataLockError("judge audit key is not readable CSV") from error
    primary = [row for row in rows if row.get("item_kind") == "primary"]
    if not primary:
        raise ConfirmatoryDataLockError("judge audit key has no primary rows")
    source_paths = {row.get("source_file", "") for row in primary}
    if not source_paths or any(
        Path(value).name != "confirmatory_audit_input.manifest.json"
        for value in source_paths
    ):
        raise ConfirmatoryDataLockError(
            "judge audit must use only native confirmatory adapter manifests"
        )
    from analysis.confirmatory_judge_adapter import (
        ConfirmatoryJudgeAdapterError,
        load_confirmatory_audit_input,
    )

    records: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for value in sorted(source_paths):
        try:
            manifest_records, manifest = load_confirmatory_audit_input(value)
        except ConfirmatoryJudgeAdapterError as error:
            raise ConfirmatoryDataLockError(
                f"judge confirmatory source failed replay: {error}"
            ) from error
        records.extend(manifest_records)
        source_runs.extend(manifest["source_runs"])
        manifests.append(
            {
                "path": str(Path(value).resolve()),
                "manifest_sha256": manifest["manifest_sha256"],
                "records_sha256": manifest["records_sha256"],
            }
        )

    jobs = [
        job for job in campaign.get("jobs", [])
        if job.get("experiment") == "part0" and job.get("stage") != "smoke"
    ]
    expected_by_directory = {
        str(Path(str(job.get("output_dir", ""))).resolve()): job for job in jobs
    }
    actual_by_directory = {
        str(Path(str(source.get("run_directory", ""))).resolve()): source
        for source in source_runs
    }
    if (
        not jobs
        or len(expected_by_directory) != len(jobs)
        or len(actual_by_directory) != len(source_runs)
        or set(actual_by_directory) != set(expected_by_directory)
    ):
        raise ConfirmatoryDataLockError(
            "judge audit sources do not exactly cover frozen Part 0 production jobs"
        )
    registry = campaign.get("inputs", {}).get("part0_registry", {})
    expected_targets: set[str] = set()
    for directory, job in expected_by_directory.items():
        source = actual_by_directory[directory]
        if (
            source.get("provider") != job.get("provider")
            or source.get("model") != job.get("route")
            or Path(str(source.get("registry_path", ""))).resolve()
            != Path(str(registry.get("path", ""))).resolve()
            or source.get("registry_sha256") != registry.get("sha256")
        ):
            raise ConfirmatoryDataLockError(
                f"judge audit route/registry lineage changed for {job.get('id')}"
            )
        expected_targets.add(str(job.get("target_id")))
    record_targets = {str(record.get("target_id")) for record in records}
    key_targets = {str(row.get("target_id")) for row in primary}
    if record_targets != expected_targets or not key_targets.issubset(expected_targets):
        raise ConfirmatoryDataLockError(
            "judge audit target population differs from the frozen Part 0 panel"
        )
    return {
        "status": "exact_campaign_part0_population_verified",
        "part0_production_jobs": len(jobs),
        "target_count": len(expected_targets),
        "adapter_manifests": manifests,
        "source_run_set_sha256": stable_json_hash(sorted(actual_by_directory)),
    }


def _validate_attempt_reconciliation(
    artifacts: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]
) -> dict[str, Any]:
    """Require an exact hash-level join between reservations and native attempts."""

    attempt_paths = sorted(
        {
            Path(str(reference["path"])).resolve()
            for reference in artifacts
            if str(reference.get("path", "")).endswith("_attempts.jsonl")
        }
    )
    if not attempt_paths:
        raise ConfirmatoryDataLockError("no native campaign attempt logs were retained")
    native_hashes: list[str] = []
    for path in attempt_paths:
        try:
            records = load_attempt_records(path)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            raise ConfirmatoryDataLockError(
                f"native attempt log could not be replayed: {path}"
            ) from error
        for record in records:
            unit = record.get("unit")
            value = unit.get("dispatch_request_sha256") if isinstance(unit, Mapping) else None
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ConfirmatoryDataLockError(
                    "native attempt lacks its pre-dispatch ledger request hash"
                )
            native_hashes.append(value)
    ledger_hashes = [
        str(record.get("request_sha256"))
        for record in ledger.get("records", [])
        if isinstance(record, Mapping) and record.get("role") != "discovery"
    ]
    if Counter(native_hashes) != Counter(ledger_hashes):
        raise ConfirmatoryDataLockError(
            "request ledger and native attempt logs do not have an exact hash multiset match"
        )
    return {
        "status": "exact_hash_multiset_verified",
        "scientific_and_smoke_physical_attempts": len(native_hashes),
        "native_attempt_log_count": len(attempt_paths),
        "request_hash_multiset_sha256": stable_json_hash(sorted(native_hashes)),
    }


def _validate_exclusions(
    path: str | Path,
    campaigns: Mapping[str, Mapping[str, Any]],
    scientific_jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    resolved, payload, digest = _load(path, "exclusion decisions")
    if set(payload) != _EXCLUSION_KEYS or payload.get("schema_version") != SCHEMA_VERSION or payload.get("artifact_type") != EXCLUSIONS_TYPE or payload.get("status") != "approved_outcome_blind":
        raise ConfirmatoryDataLockError("exclusion decision schema/status is not exact")
    expected_plans = {
        name: campaign["plan_sha256"] for name, campaign in campaigns.items()
    }
    if payload.get("campaign_plan_sha256s") != expected_plans:
        raise ConfirmatoryDataLockError("exclusion policy targets another immutable campaign plan pair")
    if payload.get("allowed_reason_codes") != list(OBJECTIVE_EXCLUSION_REASON_CODES):
        raise ConfirmatoryDataLockError("exclusion reason-code policy is not the prespecified objective policy")
    for field in ("policy_frozen_by", "reviewed_by"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ConfirmatoryDataLockError(f"exclusion decisions require {field}")
    policy_time = _utc(payload.get("policy_frozen_at_utc"), "exclusion policy_frozen_at_utc")
    _utc(payload.get("decision_reviewed_at_utc"), "exclusion decision_reviewed_at_utc")
    campaign_created = min(
        datetime.fromisoformat(
            _utc(campaign.get("created_at_utc"), f"{name} campaign created_at_utc").replace("Z", "+00:00")
        )
        for name, campaign in campaigns.items()
    )
    if datetime.fromisoformat(policy_time.replace("Z", "+00:00")) > campaign_created:
        raise ConfirmatoryDataLockError("exclusion policy was not frozen before outcome collection")
    ids = payload.get("excluded_job_ids")
    decisions = payload.get("decisions")
    if not isinstance(ids, list) or not all(isinstance(value, str) and value for value in ids) or len(set(ids)) != len(ids):
        raise ConfirmatoryDataLockError("excluded_job_ids must be unique strings")
    if not isinstance(decisions, list) or len(decisions) != len(ids):
        raise ConfirmatoryDataLockError("exclusion decisions must exactly cover excluded_job_ids")
    scientific_ids = {str(job["id"]) for job in scientific_jobs}
    decision_ids: list[str] = []
    for decision in decisions:
        if not isinstance(decision, dict) or set(decision) != _DECISION_KEYS:
            raise ConfirmatoryDataLockError("exclusion decision row schema is not exact")
        for field in _DECISION_KEYS - {"outcome_blind"}:
            if not isinstance(decision[field], str) or not decision[field].strip():
                raise ConfirmatoryDataLockError("exclusion decision fields must be nonempty")
        if decision["reason_code"] not in OBJECTIVE_EXCLUSION_REASON_CODES:
            raise ConfirmatoryDataLockError("exclusion decision uses an unprespecified reason code")
        if decision["outcome_blind"] is not True:
            raise ConfirmatoryDataLockError("exclusion decision lacks outcome-blind attestation")
        _utc(decision["decided_at_utc"], "exclusion decided_at_utc")
        evidence = Path(decision["evidence_path"]).resolve()
        if not evidence.is_file() or sha256_file(evidence) != decision["evidence_sha256"]:
            raise ConfirmatoryDataLockError("exclusion decision objective evidence changed")
        decision_ids.append(decision["job_id"])
    if decision_ids != ids or any(job_id not in scientific_ids for job_id in ids):
        raise ConfirmatoryDataLockError("exclusions contain duplicates, unknown jobs, or order drift")
    return {"path": str(resolved), "sha256": digest, **payload}


def _artifact_references(
    jobs: Sequence[Mapping[str, Any]], *, lineage_role: str
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for job in jobs:
        if job.get("status") != "complete" or not isinstance(job.get("artifact"), Mapping):
            raise ConfirmatoryDataLockError(f"job is not complete with an artifact: {job.get('id')}")
        try:
            verified = confirmatory_campaign.resolve_job_artifact(job, None)
        except Exception as error:
            raise ConfirmatoryDataLockError(f"native artifact validation failed for {job['id']}: {error}") from error
        stored = job["artifact"]
        paths: list[Path] = []
        if job["experiment"] in {"part0", "part1"}:
            if verified != stored or verified.get("kind") != "private_confirmatory_run":
                raise ConfirmatoryDataLockError(f"stored native artifact changed for {job['id']}")
            files = stored.get("files")
            if not isinstance(files, list) or not files:
                raise ConfirmatoryDataLockError(f"artifact file list is empty for {job['id']}")
            for item in files:
                if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size_bytes"}:
                    raise ConfirmatoryDataLockError("private artifact file schema is not exact")
                file_path = Path(str(item["path"])).resolve()
                if not file_path.is_file() or sha256_file(file_path) != item["sha256"] or file_path.stat().st_size != item["size_bytes"]:
                    raise ConfirmatoryDataLockError(f"private artifact bytes changed: {file_path}")
                paths.append(file_path)
        else:
            for key in ("metadata_path", "csv_path"):
                if verified.get(key) != stored.get(key):
                    raise ConfirmatoryDataLockError(f"Part 2 {key} changed for {job['id']}")
                paths.append((confirmatory_campaign.REPO_ROOT / str(stored[key])).resolve())
            if verified.get("rows") != stored.get("rows"):
                raise ConfirmatoryDataLockError(f"Part 2 row count changed for {job['id']}")
            attempts = attempt_log_path_for_csv(paths[-1])
            if attempts.is_file():
                paths.append(attempts)
            exclusion = stored.get("analysis_exclusion")
            if exclusion is not None:
                if not isinstance(exclusion, Mapping) or set(exclusion) != {"path", "sha256"}:
                    raise ConfirmatoryDataLockError("Part 2 exclusion schema is not exact")
                marker = (confirmatory_campaign.REPO_ROOT / str(exclusion["path"])).resolve()
                if not marker.is_file() or sha256_file(marker) != exclusion["sha256"]:
                    raise ConfirmatoryDataLockError("Part 2 exclusion marker changed")
                paths.append(marker)
        for artifact_path in paths:
            if artifact_path in seen_paths:
                raise ConfirmatoryDataLockError(f"artifact file is reused by multiple jobs: {artifact_path}")
            seen_paths.add(artifact_path)
            ref = _reference(artifact_path, kind="completed_job_artifact")
            ref.update(
                job_id=job["id"], experiment=job["experiment"], stage=job["stage"],
                analysis_eligible=job["stage"] != "smoke",
                lineage_role=lineage_role,
            )
            if job["stage"] == "smoke":
                ref.update(
                    publication_outcome_eligible=False,
                    publication_use="sacrificial_smoke_excluded",
                )
            elif (
                lineage_role == "part0_part1_final_and_part2_variance_only"
                and job["experiment"] == "part2"
            ):
                ref.update(
                    publication_outcome_eligible=False,
                    publication_use="variance_sample_size_selection_only",
                )
            else:
                ref.update(
                    publication_outcome_eligible=True,
                    publication_use="final_scientific_outcome",
                )
            references.append(ref)
    return references


def _validated_campaign_stage(
    path: str | Path, *, label: str, expected_scientific_stage: str
) -> tuple[Path, dict[str, Any], str]:
    manifest_path, manifest, file_hash = _load(path, label)
    try:
        confirmatory_campaign.validate_manifest(manifest)
    except Exception as error:
        raise ConfirmatoryDataLockError(f"{label} failed validation: {error}") from error
    if (
        manifest.get("status") != "complete"
        or not manifest.get("jobs")
        or any(job.get("status") != "complete" for job in manifest["jobs"])
    ):
        raise ConfirmatoryDataLockError(f"{label} and every planned job must be complete")
    selection = manifest.get("target_selection")
    if (
        not isinstance(selection, Mapping)
        or selection.get("mode") != "complete_union"
        or selection.get("selected_target_ids")
        != [target["id"] for target in manifest.get("targets", [])]
        or selection.get("complete_union_target_count") != len(manifest.get("targets", []))
    ):
        raise ConfirmatoryDataLockError(
            f"{label} must be an exact complete_union; execution shards cannot be published"
        )
    design = manifest.get("part2_design")
    if not isinstance(design, Mapping) or design.get("scientific_stage") != expected_scientific_stage:
        raise ConfirmatoryDataLockError(f"{label} has the wrong Part 2 scientific stage")
    part2_science = [
        job for job in manifest["jobs"]
        if job["experiment"] == "part2" and job["stage"] != "smoke"
    ]
    if not part2_science or any(job["stage"] != expected_scientific_stage for job in part2_science):
        raise ConfirmatoryDataLockError(f"{label} Part 2 job stages are not exact")
    return manifest_path, manifest, file_hash


def build_data_lock(
    *,
    variance_campaign_manifest_path: str | Path,
    baseline_campaign_manifest_path: str | Path,
    judge_criterion_path: str | Path,
    variance_selection_path: str | Path,
    exclusions_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Validate every final gate and atomically create a self-hashed lock."""

    variance_path, variance_campaign, variance_file_hash = _validated_campaign_stage(
        variance_campaign_manifest_path,
        label="variance-stage campaign manifest",
        expected_scientific_stage="part2_variance_pilot",
    )
    baseline_path, baseline_campaign, baseline_file_hash = _validated_campaign_stage(
        baseline_campaign_manifest_path,
        label="baseline-stage campaign manifest",
        expected_scientific_stage="part2_baseline_production",
    )
    for key in ("cohorts", "target_selection", "targets", "roles", "registry"):
        if variance_campaign.get(key) != baseline_campaign.get(key):
            raise ConfirmatoryDataLockError(f"campaign stage manifests disagree on {key}")
    system_metadata = _validate_cohort_estimand(variance_campaign)
    for key in ("part0_registry", "part1_bank", "endpoint_evidence"):
        if variance_campaign["inputs"].get(key) != baseline_campaign["inputs"].get(key):
            raise ConfirmatoryDataLockError(f"campaign stage manifests disagree on pinned {key}")
    if variance_campaign["inputs"].get("variance_selection") is not None:
        raise ConfirmatoryDataLockError("variance-stage campaign must not consume a variance selection")
    if variance_campaign.get("execution_freeze") != baseline_campaign.get("execution_freeze"):
        raise ConfirmatoryDataLockError("campaign stages do not share one frozen source/dependency state")

    variance_jobs = list(variance_campaign["jobs"])
    baseline_part2_jobs = [
        job for job in baseline_campaign["jobs"] if job["experiment"] == "part2"
    ]
    final_scientific_jobs = [
        job
        for job in variance_jobs
        if job["experiment"] in {"part0", "part1"} and job["stage"] != "smoke"
    ] + [job for job in baseline_part2_jobs if job["stage"] != "smoke"]
    collection_timing = _collection_timing(
        {
            "variance_stage": variance_jobs,
            "baseline_stage": baseline_part2_jobs,
        }
    )
    artifacts = _artifact_references(
        variance_jobs, lineage_role="part0_part1_final_and_part2_variance_only"
    ) + _artifact_references(
        baseline_part2_jobs, lineage_role="part2_final_baseline"
    )
    artifact_paths = [reference["path"] for reference in artifacts]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise ConfirmatoryDataLockError("campaign stages reuse an artifact file")
    judge_gate = _validate_judge_gate(judge_criterion_path)
    variance_file, variance, variance_digest = _load(variance_selection_path, "variance selection")
    try:
        selected_n = confirmatory_campaign._validate_variance_gate(
            variance,
            pilot_manifest_path=variance_path,
            expected_pilot_manifest_sha256=variance_file_hash,
        )
    except Exception as error:
        raise ConfirmatoryDataLockError(f"variance-selection gate failed: {error}") from error
    campaign_variance = baseline_campaign.get("inputs", {}).get("variance_selection")
    if not isinstance(campaign_variance, Mapping) or campaign_variance.get("sha256") != variance_digest or Path(str(campaign_variance.get("path", ""))).resolve() != variance_file or campaign_variance.get("selected_n") != selected_n:
        raise ConfirmatoryDataLockError("variance-selection gate is not the baseline campaign's pinned gate")
    pilot_record = campaign_variance.get("pilot_campaign")
    if (
        variance.get("pilot_campaign_manifest_sha256") != variance_file_hash
        or not isinstance(pilot_record, Mapping)
        or Path(str(pilot_record.get("path", ""))).resolve() != variance_path
        or pilot_record.get("sha256") != variance_file_hash
        or pilot_record.get("plan_sha256") != variance_campaign.get("plan_sha256")
        or pilot_record.get("manifest_payload_sha256")
        != variance_campaign.get("manifest_sha256")
    ):
        raise ConfirmatoryDataLockError("variance selection does not bind the completed variance-stage manifest")
    campaigns = {
        "variance_stage": variance_campaign,
        "baseline_stage": baseline_campaign,
    }
    exclusions = _validate_exclusions(
        exclusions_path, campaigns, final_scientific_jobs
    )
    inputs: list[dict[str, Any]] = []
    for key in ("part0_registry", "part1_bank", "endpoint_evidence"):
        value = variance_campaign["inputs"][key]
        if not isinstance(value, Mapping) or not value.get("path") or not value.get("sha256"):
            raise ConfirmatoryDataLockError(f"campaign input pin is invalid: {key}")
        ref = _reference(Path(str(value["path"])), kind=f"campaign_input:{key}")
        if ref["sha256"] != value["sha256"]:
            raise ConfirmatoryDataLockError(f"campaign input changed: {key}")
        inputs.append(ref)
    inputs.append(_reference(variance_file, kind="campaign_input:variance_selection"))
    registry_path = confirmatory_campaign.REPO_ROOT / "agents" / "agent_config.registry.json"
    inputs.append(_reference(registry_path, kind="model_registry"))
    protocol_paths = [
        confirmatory_campaign.REPO_ROOT / "docs" / "CONFIRMATORY_PROTOCOL.md",
        confirmatory_campaign.REPO_ROOT / "analysis" / "judge_audit.py",
        confirmatory_campaign.REPO_ROOT / "analysis" / "confirmatory_estimators.py",
        Path(__file__).resolve(),
        confirmatory_campaign.REPO_ROOT / "analysis" / "confirmatory_judge_adapter.py",
    ]
    protocols = [_reference(path, kind="protocol_or_lock_source") for path in protocol_paths]
    smoke_jobs = [
        {"campaign_stage": "variance_stage", "job_id": job["id"]}
        for job in variance_jobs if job["stage"] == "smoke"
    ] + [
        {"campaign_stage": "baseline_stage", "job_id": job["id"]}
        for job in baseline_part2_jobs if job["stage"] == "smoke"
    ]
    excluded_ids = set(exclusions["excluded_job_ids"])
    included_scientific = [str(job["id"]) for job in final_scientific_jobs if job["id"] not in excluded_ids]
    decision_by_id = {row["job_id"]: row for row in exclusions["decisions"]}
    excluded_audit_table = [
        {
            "job_id": job["id"],
            "experiment": job["experiment"],
            "stage": job["stage"],
            "target_id": job.get("target_id"),
            "route": job.get("route"),
            "campaign_stage": (
                "variance_stage"
                if job["experiment"] in {"part0", "part1"}
                else "baseline_stage"
            ),
            **decision_by_id[str(job["id"])],
        }
        for job in final_scientific_jobs
        if job["id"] in excluded_ids
    ]
    locked = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "locked",
        "campaigns": {
            "variance_stage": {
                "path": str(variance_path), "file_sha256": variance_file_hash,
                "manifest_sha256": variance_campaign["manifest_sha256"],
                "plan_sha256": variance_campaign["plan_sha256"],
                "part2_role": "sample_size_selection_only_not_final_outcomes",
            },
            "baseline_stage": {
                "path": str(baseline_path), "file_sha256": baseline_file_hash,
                "manifest_sha256": baseline_campaign["manifest_sha256"],
                "plan_sha256": baseline_campaign["plan_sha256"],
                "part2_role": "final_scientific_outcomes",
            },
        },
        "completed_jobs": {
            "variance_stage_total": len(variance_jobs),
            "baseline_stage_total": len(baseline_campaign["jobs"]),
            "final_analysis_source_total": len(final_scientific_jobs),
            "final_by_experiment": dict(sorted(Counter(str(job["experiment"]) for job in final_scientific_jobs).items())),
        },
        "analysis_sources": {
            "part0": "variance_stage",
            "part1": "variance_stage",
            "part2_variance_pilot": "variance_stage_sample_size_only",
            "part2_final": "baseline_stage",
        },
        "collection_timing": collection_timing,
        "panel_estimands": {
            "primary_cohort_id": PRIMARY_COHORT_ID,
            "historical_cohort_id": HISTORICAL_COHORT_ID,
            "current_system_count": sum(
                row["cohort_id"] == PRIMARY_COHORT_ID for row in system_metadata
            ),
            "current_developer_count": len(
                {
                    row["developer_id"]
                    for row in system_metadata
                    if row["cohort_id"] == PRIMARY_COHORT_ID
                }
            ),
            "historical_system_count": sum(
                row["cohort_id"] == HISTORICAL_COHORT_ID for row in system_metadata
            ),
            "developer_balanced_summary_required": True,
        },
        "deferred_non_lockable": {
            "part2_sensitivity": {
                "status": "deferred_pending_native_execution_lineage",
                "publication_outcome_eligible": False,
                "reason": (
                    "No confirmatory sensitivity result is lockable until its native "
                    "execution artifacts and campaign lineage exist."
                ),
            },
        },
        "artifacts": artifacts,
        "approved_inputs": inputs,
        "protocol_and_source_files": protocols,
        "gates": {
            "judge_criterion": judge_gate,
            "variance_selection": {"path": str(variance_file), "sha256": variance_digest, "selected_n": selected_n},
            "exclusion_decisions": {"path": exclusions["path"], "sha256": exclusions["sha256"], "status": "approved_outcome_blind"},
        },
        "exclusions": {
            "sacrificial_smoke_job_ids": smoke_jobs,
            "included_scientific_job_ids": included_scientific,
            "excluded_scientific_job_ids": exclusions["excluded_job_ids"],
            "excluded_scientific_job_audit_table": excluded_audit_table,
            "policy_frozen_at_utc": exclusions["policy_frozen_at_utc"],
            "allowed_reason_codes": list(OBJECTIVE_EXCLUSION_REASON_CODES),
        },
        "completeness": {
            "all_planned_jobs_complete": True,
            "all_selected_lineage_artifacts_reverified": True,
            "parts_present": ["part0", "part1", "part2"],
            "required_gate_count": 2,
            "required_gates_present": True,
            "scientific_attempt_timestamps_validated": True,
            "artifact_file_count": len(artifacts),
        },
    }
    locked["data_lock_sha256"] = stable_json_hash(locked)
    destination = Path(output_path).resolve()
    if destination.exists():
        raise ConfirmatoryDataLockError(f"data-lock output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(locked, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return locked


def build_fixed_data_lock(
    *,
    campaign_manifest_path: str | Path,
    judge_criterion_path: str | Path,
    exclusions_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create the final lock for the preregistered one-stage fixed design."""

    campaign_path, campaign, campaign_file_hash = _validated_campaign_stage(
        campaign_manifest_path,
        label="fixed-stage campaign manifest",
        expected_scientific_stage="part2_fixed_production",
    )
    design = campaign.get("part2_design", {})
    production_config = design.get("production_config", {})
    if (
        design.get("fixed_replicates") != confirmatory_campaign.FIXED_PART2_REPLICATES
        or production_config.get("society_size") != 10
        or production_config.get("days") != 30
        or production_config.get("resource_capacity") != 150
    ):
        raise ConfirmatoryDataLockError("fixed-stage Part 2 design changed")
    system_metadata = _validate_cohort_estimand(campaign)
    jobs = list(campaign["jobs"])
    scientific_jobs = [job for job in jobs if job["stage"] != "smoke"]
    artifacts = _artifact_references(jobs, lineage_role="single_fixed_stage_final")
    exclusions = _validate_exclusions(
        exclusions_path, {"fixed_stage": campaign}, scientific_jobs
    )
    if exclusions["excluded_job_ids"] or exclusions["decisions"]:
        raise ConfirmatoryDataLockError(
            "fixed-stage inference requires the exact complete scientific panel; "
            "the outcome-blind exclusion artifact must be empty"
        )
    judge_gate = _validate_judge_gate(judge_criterion_path)
    judge_lineage = _validate_judge_campaign_lineage(judge_gate, campaign)
    inputs: list[dict[str, Any]] = []
    for key in ("part0_registry", "part1_bank", "endpoint_evidence"):
        value = campaign["inputs"].get(key)
        if not isinstance(value, Mapping) or not value.get("path") or not value.get("sha256"):
            raise ConfirmatoryDataLockError(f"campaign input pin is invalid: {key}")
        reference = _reference(Path(str(value["path"])), kind=f"campaign_input:{key}")
        if reference["sha256"] != value["sha256"]:
            raise ConfirmatoryDataLockError(f"campaign input changed: {key}")
        inputs.append(reference)
    inputs.append(
        _reference(
            confirmatory_campaign.REPO_ROOT / "agents" / "agent_config.registry.json",
            kind="model_registry",
        )
    )
    budget_path = campaign_path.parent / "request_budget.json"
    ledger_path = campaign_path.parent / "request_ledger.json"
    _, budget_payload, _ = _load(budget_path, "fixed request budget")
    _, ledger_payload, _ = _load(ledger_path, "fixed request ledger")
    try:
        validate_frozen_budget(budget_payload)
        validate_ledger(ledger_payload, budget_payload)
    except Exception as error:
        raise ConfirmatoryDataLockError(f"request budget/ledger failed replay: {error}") from error
    if budget_payload != campaign.get("request_budget"):
        raise ConfirmatoryDataLockError("request budget differs from campaign freeze")
    attempt_reconciliation = _validate_attempt_reconciliation(
        artifacts, ledger_payload
    )
    inputs.extend(
        [
            _reference(budget_path, kind="confirmatory_request_budget"),
            _reference(ledger_path, kind="confirmatory_request_ledger"),
        ]
    )
    protocols = [
        _reference(path, kind="protocol_or_lock_source")
        for path in (
            confirmatory_campaign.REPO_ROOT / "docs" / "CONFIRMATORY_PROTOCOL.md",
            confirmatory_campaign.REPO_ROOT / "analysis" / "judge_audit.py",
            confirmatory_campaign.REPO_ROOT / "analysis" / "confirmatory_estimators.py",
            Path(__file__).resolve(),
            confirmatory_campaign.REPO_ROOT / "analysis" / "confirmatory_judge_adapter.py",
        )
    ]
    excluded_ids = set(exclusions["excluded_job_ids"])
    decisions = {row["job_id"]: row for row in exclusions["decisions"]}
    included = [str(job["id"]) for job in scientific_jobs if job["id"] not in excluded_ids]
    locked: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "locked",
        "campaigns": {
            "fixed_stage": {
                "path": str(campaign_path),
                "file_sha256": campaign_file_hash,
                "manifest_sha256": campaign["manifest_sha256"],
                "plan_sha256": campaign["plan_sha256"],
                "part2_role": "final_scientific_outcomes_exactly_24_runs",
            }
        },
        "completed_jobs": {
            "fixed_stage_total": len(jobs),
            "final_analysis_source_total": len(scientific_jobs),
            "final_by_experiment": dict(
                sorted(Counter(str(job["experiment"]) for job in scientific_jobs).items())
            ),
        },
        "analysis_sources": {
            "part0": "fixed_stage",
            "part1": "fixed_stage",
            "part2_final": "fixed_stage",
        },
        "collection_timing": _single_stage_collection_timing(jobs),
        "panel_estimands": {
            "primary_cohort_id": PRIMARY_COHORT_ID,
            "historical_cohort_id": HISTORICAL_COHORT_ID,
            "current_system_count": sum(row["cohort_id"] == PRIMARY_COHORT_ID for row in system_metadata),
            "current_developer_count": len({row["developer_id"] for row in system_metadata if row["cohort_id"] == PRIMARY_COHORT_ID}),
            "historical_system_count": sum(row["cohort_id"] == HISTORICAL_COHORT_ID for row in system_metadata),
            "developer_balanced_summary_required": True,
        },
        "deferred_non_lockable": {},
        "artifacts": artifacts,
        "approved_inputs": inputs,
        "protocol_and_source_files": protocols,
        "gates": {
            "judge_criterion": {
                **judge_gate,
                "campaign_lineage": judge_lineage,
            },
            "request_attempt_reconciliation": attempt_reconciliation,
            "exclusion_decisions": {
                "path": exclusions["path"],
                "sha256": exclusions["sha256"],
                "status": "approved_outcome_blind",
            },
        },
        "exclusions": {
            "sacrificial_smoke_job_ids": [
                {"campaign_stage": "fixed_stage", "job_id": job["id"]}
                for job in jobs if job["stage"] == "smoke"
            ],
            "included_scientific_job_ids": included,
            "excluded_scientific_job_ids": exclusions["excluded_job_ids"],
            "excluded_scientific_job_audit_table": [
                {
                    "job_id": job["id"],
                    "experiment": job["experiment"],
                    "stage": job["stage"],
                    "target_id": job.get("target_id"),
                    "route": job.get("route"),
                    "campaign_stage": "fixed_stage",
                    **decisions[str(job["id"])],
                }
                for job in scientific_jobs if job["id"] in excluded_ids
            ],
            "policy_frozen_at_utc": exclusions["policy_frozen_at_utc"],
            "allowed_reason_codes": list(OBJECTIVE_EXCLUSION_REASON_CODES),
        },
        "completeness": {
            "all_planned_jobs_complete": True,
            "all_selected_lineage_artifacts_reverified": True,
            "parts_present": ["part0", "part1", "part2"],
            "required_gate_count": 3,
            "required_gates_present": True,
            "scientific_attempt_timestamps_validated": True,
            "artifact_file_count": len(artifacts),
        },
    }
    locked["data_lock_sha256"] = stable_json_hash(locked)
    destination = Path(output_path).resolve()
    if destination.exists():
        raise ConfirmatoryDataLockError(f"data-lock output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(locked, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return locked


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the final self-hashed confirmatory data lock.")
    parser.add_argument("--fixed-campaign-manifest")
    parser.add_argument("--variance-campaign-manifest")
    parser.add_argument("--baseline-campaign-manifest")
    parser.add_argument("--judge-criterion", required=True)
    parser.add_argument("--variance-selection")
    parser.add_argument("--exclusions", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.fixed_campaign_manifest:
            if any(
                value is not None
                for value in (
                    args.variance_campaign_manifest,
                    args.baseline_campaign_manifest,
                    args.variance_selection,
                )
            ):
                raise ConfirmatoryDataLockError(
                    "fixed-stage lock cannot be combined with legacy two-stage inputs"
                )
            result = build_fixed_data_lock(
                campaign_manifest_path=args.fixed_campaign_manifest,
                judge_criterion_path=args.judge_criterion,
                exclusions_path=args.exclusions,
                output_path=args.output,
            )
        else:
            if not all(
                (args.variance_campaign_manifest, args.baseline_campaign_manifest, args.variance_selection)
            ):
                raise ConfirmatoryDataLockError(
                    "provide --fixed-campaign-manifest or all legacy two-stage inputs"
                )
            result = build_data_lock(
                variance_campaign_manifest_path=args.variance_campaign_manifest,
                baseline_campaign_manifest_path=args.baseline_campaign_manifest,
                judge_criterion_path=args.judge_criterion,
                variance_selection_path=args.variance_selection,
                exclusions_path=args.exclusions,
                output_path=args.output,
            )
    except ConfirmatoryDataLockError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(Path(args.output).resolve()), "data_lock_sha256": result["data_lock_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
