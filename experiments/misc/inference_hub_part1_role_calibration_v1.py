"""Provider-safe v1 Part 1 role-conditioned calibration runner.

This is a new, exploratory calibration path.  It does not alter or promote the
historical Part 1 artifacts.  The immutable six-model panel is executed over
the frozen 96-root role subset; advice, observer-evaluation, and prediction
remain separate estimands.  Raw prompts and responses stay in private,
append-only hash-chained journals.  Public output contains text-free root-aware
summaries only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dotenv import load_dotenv

from experiments.misc import inference_hub_part1_panel as base
from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
)
from experiments.misc.inference_hub_provider_safe_v2 import (
    _provider_safe_client,
    provider_round_robin,
)
from experiments.part1.confirmatory_design import (
    COUNTERBALANCE_BY_ID,
    DOMAINS,
    GAMES,
    ROLE_BLOCKS,
    ROLE_FRAME_IDS,
    WELFARE_PRESERVING,
    build_draft_bank,
    build_role_schedule,
    validate_role_schedule,
)


SCHEMA_VERSION = 1
CONFIG_PATH = Path(__file__).parents[1] / "part1" / "role_calibration_panel_v1.json"
CONFIG_FILE_SHA256 = "14012c3db18890dd970e826dcf59498eeba6b411fe0b5f04d2df52cc6c6e92b3"
EXPECTED_PANEL_ID = "part1-role-conditioned-sentinels-v1"
EXPECTED_ROOT_COUNT = 96
EXPECTED_TRIALS_PER_SUBJECT = 96 * 3 * 4
EXPECTED_SUBJECT_COUNT = 6
DEFAULT_MAX_WORKERS = 12
DEFAULT_MAX_WORKERS_PER_PROVIDER = 1
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.0

_SOURCE_PATHS = (
    Path(__file__),
    CONFIG_PATH,
    Path(base.__file__),
    Path(__file__).with_name("inference_hub_provider_safe.py"),
    Path(__file__).with_name("inference_hub_provider_safe_v2.py"),
    Path(__file__).with_name("inference_hub_rate_limit.py"),
    Path(__file__).parents[1] / "part1" / "confirmatory_design.py",
)


class RoleCalibrationError(RuntimeError):
    """The versioned role-calibration contract or evidence was invalid."""


class _RunLockedJournal(base._ChainedJournal):
    """Incremental journal writer used only while the directory run lock is held."""

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            row = {**payload, "previous_record_sha256": self.tail}
            row["record_sha256"] = base._sha256_json(row)
            encoded = json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            descriptor = os.open(
                self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
            )
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            base._secure_mode(self.path, 0o600)
            self.records.append(row)
            self.tail = row["record_sha256"]
            return row


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        base._secure_mode(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_frozen_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the one immutable v1 panel configuration, refusing substitutions."""

    resolved = path.resolve()
    if resolved != CONFIG_PATH.resolve():
        raise RoleCalibrationError("The v1 runner accepts only its repository config path.")
    if base._sha256_file(resolved) != CONFIG_FILE_SHA256:
        raise RoleCalibrationError("The immutable v1 panel configuration hash changed.")
    config = base._read_json(resolved, "role calibration configuration")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "panel_id",
        "analysis_role",
        "base_seed",
        "root_design",
        "frames",
        "generation_blocks",
        "estimand_contract",
        "execution_contract",
        "judge_target_id",
        "judge_dispatched",
        "subjects",
    }
    if set(config) != expected_keys:
        raise RoleCalibrationError("The immutable v1 config fields are not exact.")
    subjects = config.get("subjects")
    if (
        config.get("schema_version") != SCHEMA_VERSION
        or config.get("artifact_type") != "part1_role_conditioned_calibration_panel"
        or config.get("panel_id") != EXPECTED_PANEL_ID
        or config.get("analysis_role")
        != "exploratory_role_conditioned_calibration_only"
        or config.get("judge_dispatched") is not False
        or config.get("frames") != list(ROLE_FRAME_IDS)
        or config.get("generation_blocks") != list(ROLE_BLOCKS)
        or not isinstance(subjects, list)
        or len(subjects) != EXPECTED_SUBJECT_COUNT
    ):
        raise RoleCalibrationError("The immutable v1 panel contract is invalid.")
    target_ids = []
    developers = []
    for row in subjects:
        if not isinstance(row, Mapping) or set(row) != {
            "target_id",
            "developer_family",
            "capability_stratum",
            "selection_role",
        }:
            raise RoleCalibrationError("A sentinel definition is malformed.")
        if any(not isinstance(row[key], str) or not row[key] for key in row):
            raise RoleCalibrationError("A sentinel definition contains an empty value.")
        target_ids.append(row["target_id"])
        developers.append(row["developer_family"])
    if len(set(target_ids)) != EXPECTED_SUBJECT_COUNT:
        raise RoleCalibrationError("The sentinel target IDs are not unique.")
    if len(set(developers)) != EXPECTED_SUBJECT_COUNT:
        raise RoleCalibrationError("The sentinel panel is not developer-stratified.")
    root_design = config.get("root_design")
    estimand = config.get("estimand_contract")
    execution = config.get("execution_contract")
    if (
        not isinstance(root_design, Mapping)
        or root_design.get("root_count") != EXPECTED_ROOT_COUNT
        or root_design.get("roots_per_game_domain_cell") != 8
        or not isinstance(estimand, Mapping)
        or estimand.get("unit") != "scenario_root"
        or estimand.get("frames_pooled") is not False
        or estimand.get("models_pooled") is not False
        or estimand.get("counterbalance_draws_per_model_root_frame") != 4
        or estimand.get("cross_model_ranking_permitted") is not False
        or not isinstance(execution, Mapping)
        or execution.get("max_workers") != DEFAULT_MAX_WORKERS
        or execution.get("max_in_flight_per_upstream_provider") != 1
        or execution.get("max_transport_attempts_per_trial") != DEFAULT_MAX_ATTEMPTS
        or execution.get("initial_exponential_backoff_seconds")
        != DEFAULT_BACKOFF_SECONDS
        or execution.get("shared_cross_process_rate_limit_required") is not True
    ):
        raise RoleCalibrationError("The root or estimand contract is invalid.")
    return config


def build_frozen_trials(config: Mapping[str, Any]) -> tuple[Any, ...]:
    trials = build_role_schedule(
        build_draft_bank(),
        base_seed=int(config["base_seed"]),
        requested_provider="inference_hub",
        requested_model="compatibility-selected-six-sentinel-v1",
        production=False,
    )
    report = validate_role_schedule(trials, build_draft_bank(), production=False)
    if not report.valid:
        raise RoleCalibrationError("The frozen role schedule failed validation.")
    if (
        len(trials) != EXPECTED_TRIALS_PER_SUBJECT
        or len({trial.root_id for trial in trials}) != EXPECTED_ROOT_COUNT
        or {trial.frame_id for trial in trials} != set(ROLE_FRAME_IDS)
    ):
        raise RoleCalibrationError("The frozen role schedule has unexpected coverage.")
    return trials


@dataclass(frozen=True)
class _WorkItem:
    subject: Mapping[str, Any]
    trial: Any


def _completed_index(
    journals: Mapping[str, base._ChainedJournal],
    *,
    trials_by_id: Mapping[str, Any],
    subjects_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    complete: dict[tuple[str, str], dict[str, Any]] = {}
    for target_id, journal in journals.items():
        subject = subjects_by_id[target_id]
        for row in journal.records:
            trial = trials_by_id.get(str(row.get("trial_id")))
            key = (target_id, str(row.get("trial_id")))
            if (
                row.get("schema_version") != SCHEMA_VERSION
                or row.get("artifact_type")
                != "inference_hub_part1_role_calibration_raw_response_v1"
                or row.get("target_id") != target_id
                or trial is None
                or row.get("root_id") != trial.root_id
                or row.get("frame_id") != trial.frame_id
                or row.get("generation_block") != trial.generation_block
                or row.get("prompt_sha256") != trial.prompt_hash
                or row.get("requested_route") != subject["route"]
                or key in complete
            ):
                raise RoleCalibrationError(
                    f"Retained role-calibration binding is invalid for {target_id}."
                )
            raw = row.get("raw_response")
            if raw is not None and row.get("raw_response_sha256") != base._sha256_json(raw):
                raise RoleCalibrationError("A retained raw response hash changed.")
            complete[key] = row
    return complete


def _recover_ledger(
    ledger: base._ChainedJournal,
    raw_journals: Mapping[str, base._ChainedJournal],
) -> None:
    reservations: dict[str, Mapping[str, Any]] = {}
    completions: dict[str, Mapping[str, Any]] = {}
    for row in ledger.records:
        attempt_id = row.get("attempt_id")
        if row.get("event") == "reserved_before_dispatch" and isinstance(attempt_id, str):
            if attempt_id in reservations:
                raise RoleCalibrationError("An attempt was reserved twice.")
            reservations[attempt_id] = row
        elif row.get("event") == "attempt_completed" and isinstance(attempt_id, str):
            if attempt_id in completions:
                raise RoleCalibrationError("An attempt was completed twice.")
            completions[attempt_id] = row
    if set(completions) - set(reservations):
        raise RoleCalibrationError("A completion lacks a durable reservation.")
    raw_successes: dict[str, Mapping[str, Any]] = {}
    for target_id, journal in raw_journals.items():
        for row in journal.records:
            if row.get("raw_response") is None:
                continue
            attempt_id = row.get("attempt_id")
            reservation = reservations.get(str(attempt_id))
            if (
                not isinstance(attempt_id, str)
                or attempt_id in raw_successes
                or reservation is None
                or reservation.get("target_id") != target_id
                or reservation.get("trial_id") != row.get("trial_id")
                or reservation.get("request_sha256") != row.get("request_sha256")
            ):
                raise RoleCalibrationError("A raw success lacks its exact reservation.")
            raw_successes[attempt_id] = row
            if attempt_id not in completions:
                ledger.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "inference_hub_part1_role_calibration_attempt_v1",
                        "event": "attempt_completed",
                        "attempt_id": attempt_id,
                        "outcome": "response_retained",
                        "failure_code": None,
                        "transient": False,
                        "http_status": 200,
                        "request_id": row.get("request_id"),
                        "response_model": row.get("response_model"),
                        "response_payload_sha256": row.get("raw_response_sha256"),
                        "response_text_sha256": row.get("response_text_sha256"),
                        "recovered_after_raw_fsync": True,
                        "completed_at_utc": base._utc_now(),
                    }
                )
                completions[attempt_id] = ledger.records[-1]
    for attempt_id, completion in completions.items():
        if completion.get("outcome") == "response_retained" and attempt_id not in raw_successes:
            raise RoleCalibrationError(
                "A retained-success completion lacks its raw response; refusing redispatch."
            )
    for attempt_id in sorted(set(reservations) - set(completions)):
        ledger.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "inference_hub_part1_role_calibration_attempt_v1",
                "event": "attempt_completed",
                "attempt_id": attempt_id,
                "outcome": "failed",
                "failure_code": "stale_reservation_retried",
                "transient": True,
                "http_status": None,
                "completed_at_utc": base._utc_now(),
            }
        )


def _root_aware_summaries(
    *,
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    schedule_sha256: str,
    journal_tails: Mapping[str, str | None],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subject_config = {row["target_id"]: row for row in config["subjects"]}
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["target_id"]), str(row["frame_id"]), str(row["root_id"]))].append(row)

    roots: list[dict[str, Any]] = []
    for (target_id, frame_id, root_id), group in sorted(grouped.items()):
        valid = [row for row in group if row.get("format_valid") is True]
        estimable = [
            row
            for row in valid
            if row.get("model_identity_valid") is True and row.get("raw_response") is not None
        ]
        welfare = sum(row.get("welfare_preserving") is True for row in estimable)
        roots.append(
            {
                "target_id": target_id,
                "developer_family": subject_config[target_id]["developer_family"],
                "capability_stratum": subject_config[target_id]["capability_stratum"],
                "frame_id": frame_id,
                "root_id": root_id,
                "game": group[0]["game"],
                "domain": group[0]["domain"],
                "planned_draws": len(ROLE_BLOCKS),
                "retained_draws": len(group),
                "format_valid_draws": len(valid),
                "identity_valid_draws": sum(
                    row.get("model_identity_valid") is True for row in group
                ),
                "estimable_draws": len(estimable),
                "welfare_preserving_draws": welfare,
                "root_welfare_preserving_rate_among_valid": (
                    welfare / len(estimable) if estimable else None
                ),
            }
        )

    grouped_roots: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in roots:
        grouped_roots[(str(row["target_id"]), str(row["frame_id"]))].append(row)
    estimates = []
    for (target_id, frame_id), group in sorted(grouped_roots.items()):
        usable = [
            row
            for row in group
            if row["root_welfare_preserving_rate_among_valid"] is not None
        ]
        estimates.append(
            {
                "target_id": target_id,
                "developer_family": subject_config[target_id]["developer_family"],
                "capability_stratum": subject_config[target_id]["capability_stratum"],
                "frame_id": frame_id,
                "estimand_unit": "scenario_root",
                "planned_roots": EXPECTED_ROOT_COUNT,
                "retained_roots": len(group),
                "roots_with_format_valid_draw": len(usable),
                "planned_draws": EXPECTED_ROOT_COUNT * len(ROLE_BLOCKS),
                "retained_draws": sum(int(row["retained_draws"]) for row in group),
                "format_valid_draws": sum(int(row["format_valid_draws"]) for row in group),
                "identity_valid_draws": sum(
                    int(row["identity_valid_draws"]) for row in group
                ),
                "estimable_draws": sum(int(row["estimable_draws"]) for row in group),
                "root_weighted_welfare_preserving_rate_among_valid": (
                    sum(float(row["root_welfare_preserving_rate_among_valid"]) for row in usable)
                    / len(usable)
                    if usable
                    else None
                ),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "part1_role_calibration_sanitized_summary_v1",
        "panel_id": config["panel_id"],
        "analysis_role": config["analysis_role"],
        "config_file_sha256": CONFIG_FILE_SHA256,
        "schedule_sha256": schedule_sha256,
        "raw_journal_tail_sha256_by_target": dict(sorted(journal_tails.items())),
        "raw_text_included": False,
        "frames_pooled": False,
        "models_pooled": False,
        "cross_model_ranking_permitted": False,
        "root_summary_record_count": len(roots),
        "root_summaries_canonical_sha256": base._sha256_json(roots),
        "estimates": estimates,
    }
    summary["evidence_sha256"] = base._self_hash(summary)
    return summary, roots


def run_calibration(
    *,
    compatibility_path: Path,
    registry_path: Path,
    output_dir: Path,
    client: InferenceHubClient,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_workers_per_provider: int = DEFAULT_MAX_WORKERS_PER_PROVIDER,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    resume: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run or resume the complete immutable v1 six-sentinel calibration."""

    for name, value in (
        ("max_workers", max_workers),
        ("max_workers_per_provider", max_workers_per_provider),
        ("max_attempts", max_attempts),
    ):
        base._validate_positive_int(name, value)
    if max_workers_per_provider != 1:
        raise RoleCalibrationError(
            "The immutable v1 provider-safety contract permits one in-flight call per provider."
        )
    if initial_backoff_seconds < 0:
        raise RoleCalibrationError("initial_backoff_seconds cannot be negative.")
    if output_dir.exists() and not resume:
        raise RoleCalibrationError("Output directory already exists; use --resume.")
    if not output_dir.exists() and resume:
        raise RoleCalibrationError("Resume output directory does not exist.")

    config = load_frozen_config()
    trials = build_frozen_trials(config)
    private_dir = output_dir / "private"
    raw_dir = private_dir / "raw_responses"
    sanitized_dir = output_dir / "sanitized"
    if resume:
        for directory in (output_dir, private_dir, raw_dir, sanitized_dir):
            if not directory.is_dir():
                raise RoleCalibrationError(f"Resume directory is missing: {directory}.")
            base._require_mode(directory, 0o700)
    else:
        output_dir.mkdir(parents=True, mode=0o700)
        for directory in (private_dir, raw_dir, sanitized_dir):
            directory.mkdir(mode=0o700)
    for directory in (output_dir, private_dir, raw_dir, sanitized_dir):
        base._secure_mode(directory, 0o700)
        base._require_mode(directory, 0o700)
    run_lock = base._acquire_run_lock(private_dir)
    try:
        registry = base._read_json(registry_path, "registry")
        compatibility = base._read_json(compatibility_path, "compatibility evidence")
        if compatibility.get("endpoint") != client.base_url:
            raise RoleCalibrationError("Compatibility endpoint differs from the client endpoint.")
        selected_ids = [row["target_id"] for row in config["subjects"]]
        selected_subjects, judge = base.select_routes(
            registry=registry,
            compatibility=compatibility,
            selected_ids=selected_ids,
            judge_target_id=str(config["judge_target_id"]),
        )
        subjects = provider_round_robin(selected_subjects)
        if [row["target_id"] for row in subjects] != selected_ids:
            raise RoleCalibrationError("Compatibility selection changed the sentinel order.")
        stems = [base._safe_file_stem(str(subject["target_id"])) for subject in subjects]
        if len(stems) != len(set(stems)):
            raise RoleCalibrationError("Sentinel IDs collide as evidence filenames.")

        schedule_binding = [
            {
                "trial_id": trial.trial_id,
                "root_id": trial.root_id,
                "game": trial.game,
                "domain": trial.domain,
                "frame_id": trial.frame_id,
                "generation_block": trial.generation_block,
                "counterbalance_id": trial.counterbalance_id,
                "prompt_sha256": trial.prompt_hash,
                "generation_seed": trial.generation_settings.generation_seed,
            }
            for trial in trials
        ]
        schedule_sha256 = base._sha256_json(schedule_binding)
        source_artifacts = {
            str(path.resolve()): base._sha256_file(path.resolve()) for path in _SOURCE_PATHS
        }
        client_rate_limit_contract = getattr(client, "rate_limit_contract", None)
        if isinstance(client, InferenceHubClient):
            if not isinstance(client_rate_limit_contract, Mapping):
                raise RoleCalibrationError("Network client lacks a shared rate-limit contract.")
            rate_limit_contract = dict(client_rate_limit_contract)
        else:
            rate_limit_contract = {
                "enforcement": "external_non_network_test_double",
                "network_dispatch_permitted": False,
            }
        fresh_manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part1_role_calibration_private_v1",
            "created_at_utc": base._utc_now(),
            "panel_id": config["panel_id"],
            "analysis_role": config["analysis_role"],
            "draft_bank_human_approved": False,
            "confirmatory_or_paper_promotion_permitted": False,
            "config_file_sha256": CONFIG_FILE_SHA256,
            "source_artifacts": source_artifacts,
            "input_artifacts": {
                "registry_file_sha256": base._sha256_file(registry_path),
                "registry_canonical_sha256": base._sha256_json(registry),
                "compatibility_file_sha256": base._sha256_file(compatibility_path),
                "compatibility_evidence_sha256": compatibility["evidence_sha256"],
            },
            "root_count": EXPECTED_ROOT_COUNT,
            "frames": list(ROLE_FRAME_IDS),
            "frames_pooled": False,
            "generation_blocks": list(ROLE_BLOCKS),
            "trials_per_subject": len(trials),
            "schedule_sha256": schedule_sha256,
            "subject_routes": [
                {
                    "target_id": subject["target_id"],
                    "upstream_provider": subject["upstream_provider"],
                    "model": subject["model"],
                    "route": subject["route"],
                    "supported_controls": subject["supported_controls"],
                    "selected_profile_id": subject["selected_profile_id"],
                    "selected_profile_request_sha256": subject[
                        "selected_profile_request_sha256"
                    ],
                }
                for subject in subjects
            ],
            "judge_reservation": {
                "target_id": judge["target_id"],
                "upstream_provider": judge["upstream_provider"],
                "model": judge["model"],
                "route": judge["route"],
                "dispatch_permitted_in_this_runner": False,
                "subject_target_ids": selected_ids,
            },
            "execution_contract": {
                "configured_max_workers": max_workers,
                "max_workers_per_provider": max_workers_per_provider,
                "max_attempts_per_trial": max_attempts,
                "initial_exponential_backoff_seconds": initial_backoff_seconds,
                "shared_rate_limit": rate_limit_contract,
                "reserve_before_dispatch": True,
                "raw_response_commit_before_success_ledger": True,
            },
            "complete": False,
            "summary": {},
            "journals": {},
        }
        manifest_path = private_dir / "manifest.json"
        if resume:
            manifest = base._read_json(manifest_path, "role calibration manifest")
            base._require_mode(manifest_path, 0o600)
            if manifest.get("evidence_sha256") != base._self_hash(manifest):
                raise RoleCalibrationError("The private manifest self-hash failed.")
            if base._manifest_bindings(manifest) != base._manifest_bindings(fresh_manifest):
                raise RoleCalibrationError("Resume contract differs from immutable v1.")
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            manifest["last_resumed_at_utc"] = base._utc_now()
        else:
            manifest = fresh_manifest

        ledger = _RunLockedJournal(private_dir / "attempt_ledger.jsonl")
        raw_journals = {
            subject["target_id"]: _RunLockedJournal(
                raw_dir / f"{base._safe_file_stem(subject['target_id'])}.jsonl"
            )
            for subject in subjects
        }
        if resume:
            checkpoints = manifest.get("journals")
            if not isinstance(checkpoints, Mapping):
                raise RoleCalibrationError("Manifest journal checkpoints are invalid.")
            base._validate_checkpoint_reference(
                ledger, checkpoints.get("attempt_ledger"), label="attempt ledger"
            )
            raw_checkpoints = checkpoints.get("raw_responses")
            if not isinstance(raw_checkpoints, Mapping) or set(raw_checkpoints) != set(
                raw_journals
            ):
                raise RoleCalibrationError("Raw-response checkpoints differ from sentinels.")
            for target_id, journal in raw_journals.items():
                base._validate_checkpoint_reference(
                    journal, raw_checkpoints[target_id], label=f"raw responses for {target_id}"
                )
        else:
            manifest["journals"] = {
                "attempt_ledger": ledger.reference(),
                "raw_responses": {
                    target_id: journal.reference() for target_id, journal in raw_journals.items()
                },
            }
            base._seal(manifest)
            base._atomic_json(manifest_path, manifest)

        _recover_ledger(ledger, raw_journals)
        prior_attempts = base._prior_attempt_numbers(ledger)
        subjects_by_id = {subject["target_id"]: subject for subject in subjects}
        trials_by_id = {trial.trial_id: trial for trial in trials}
        completed = _completed_index(
            raw_journals, trials_by_id=trials_by_id, subjects_by_id=subjects_by_id
        )
        work = [
            _WorkItem(subject, trial)
            for trial in trials
            for subject in subjects
            if (subject["target_id"], trial.trial_id) not in completed
        ]
        provider_locks = {
            str(subject["upstream_provider"]): threading.BoundedSemaphore(
                max_workers_per_provider
            )
            for subject in subjects
        }

        def execute_serial(item: _WorkItem) -> dict[str, Any]:
            subject, trial = item.subject, item.trial
            body, controls = base._request_contract(subject, trial)
            request_bytes = base._canonical_bytes(body)
            request_sha256 = hashlib.sha256(request_bytes).hexdigest()
            key = (str(subject["target_id"]), trial.trial_id)
            first_attempt = prior_attempts.get(key, 0) + 1
            last_failure: dict[str, Any] | None = None
            for attempt_number in range(first_attempt, max_attempts + 1):
                attempt_id = f"part1_role_v1_{uuid.uuid4().hex}"
                ledger.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "inference_hub_part1_role_calibration_attempt_v1",
                        "event": "reserved_before_dispatch",
                        "attempt_id": attempt_id,
                        "target_id": subject["target_id"],
                        "upstream_provider": subject["upstream_provider"],
                        "model": subject["model"],
                        "route": subject["route"],
                        "trial_id": trial.trial_id,
                        "root_id": trial.root_id,
                        "frame_id": trial.frame_id,
                        "generation_block": trial.generation_block,
                        "attempt_number": attempt_number,
                        "request_sha256": request_sha256,
                        "request_body_bytes": len(request_bytes),
                        "prompt_sha256": trial.prompt_hash,
                        "controls": controls,
                        "reserved_at_utc": base._utc_now(),
                    }
                )
                try:
                    if isinstance(client, InferenceHubClient):
                        response = client.post(
                            "/chat/completions",
                            body,
                            upstream_provider=str(subject["upstream_provider"]),
                        )
                    else:
                        response = client.post("/chat/completions", body)
                    if not isinstance(response, Mapping):
                        raise TypeError("client response is not an object")
                except Exception as error:
                    retryable, failure_code, http_status = base._transient(error)
                    last_failure = {
                        "failure_code": failure_code,
                        "http_status": http_status,
                        "error_type": type(error).__name__,
                    }
                    ledger.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "artifact_type": "inference_hub_part1_role_calibration_attempt_v1",
                            "event": "attempt_completed",
                            "attempt_id": attempt_id,
                            "outcome": "failed",
                            "failure_code": failure_code,
                            "transient": retryable,
                            "http_status": http_status,
                            "completed_at_utc": base._utc_now(),
                        }
                    )
                    if retryable and attempt_number < max_attempts:
                        sleep_fn(initial_backoff_seconds * (2 ** (attempt_number - 1)))
                        continue
                    response = None
                if response is not None:
                    metadata = base._response_metadata(response)
                    response_sha256 = base._sha256_json(response)
                    expected_welfare_label = COUNTERBALANCE_BY_ID[
                        trial.counterbalance_id
                    ].label_for(WELFARE_PRESERVING)
                    row = raw_journals[subject["target_id"]].append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "artifact_type": "inference_hub_part1_role_calibration_raw_response_v1",
                            "target_id": subject["target_id"],
                            "upstream_provider": subject["upstream_provider"],
                            "model": subject["model"],
                            "requested_route": subject["route"],
                            "response_model": metadata["response_model"],
                            "model_identity_valid": metadata["response_model"] == subject["route"],
                            "trial_id": trial.trial_id,
                            "root_id": trial.root_id,
                            "game": trial.game,
                            "domain": trial.domain,
                            "frame_id": trial.frame_id,
                            "generation_block": trial.generation_block,
                            "counterbalance_id": trial.counterbalance_id,
                            "prompt_text": trial.prompt_text,
                            "prompt_sha256": trial.prompt_hash,
                            "request_sha256": request_sha256,
                            "controls": controls,
                            "attempt_id": attempt_id,
                            "attempt_number": attempt_number,
                            **metadata,
                            "welfare_preserving": (
                                metadata["parsed_action"] == expected_welfare_label
                                if metadata["format_valid"]
                                else None
                            ),
                            "raw_response": dict(response),
                            "raw_response_sha256": response_sha256,
                            "finished_at_utc": base._utc_now(),
                        }
                    )
                    ledger.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "artifact_type": "inference_hub_part1_role_calibration_attempt_v1",
                            "event": "attempt_completed",
                            "attempt_id": attempt_id,
                            "outcome": "response_retained",
                            "failure_code": None,
                            "transient": False,
                            "http_status": 200,
                            "request_id": metadata["request_id"],
                            "response_model": metadata["response_model"],
                            "response_payload_sha256": response_sha256,
                            "response_text_sha256": metadata["response_text_sha256"],
                            "completed_at_utc": base._utc_now(),
                        }
                    )
                    return row
                break
            return raw_journals[subject["target_id"]].append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "inference_hub_part1_role_calibration_raw_response_v1",
                    "target_id": subject["target_id"],
                    "upstream_provider": subject["upstream_provider"],
                    "model": subject["model"],
                    "requested_route": subject["route"],
                    "response_model": None,
                    "model_identity_valid": False,
                    "trial_id": trial.trial_id,
                    "root_id": trial.root_id,
                    "game": trial.game,
                    "domain": trial.domain,
                    "frame_id": trial.frame_id,
                    "generation_block": trial.generation_block,
                    "counterbalance_id": trial.counterbalance_id,
                    "prompt_text": trial.prompt_text,
                    "prompt_sha256": trial.prompt_hash,
                    "request_sha256": request_sha256,
                    "controls": controls,
                    "raw_response": None,
                    "raw_response_sha256": None,
                    "request_id": None,
                    "finish_reason": None,
                    "usage": None,
                    "reasoning_fields": {},
                    "output_field": None,
                    "response_text": None,
                    "response_text_sha256": None,
                    "parsed_action": None,
                    "format_valid": False,
                    "welfare_preserving": None,
                    "failure": last_failure,
                    "finished_at_utc": base._utc_now(),
                }
            )

        def execute(item: _WorkItem) -> dict[str, Any]:
            with provider_locks[str(item.subject["upstream_provider"])]:
                return execute_serial(item)

        if work:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(work)),
                thread_name_prefix="part1-role-calibration-v1",
            ) as executor:
                futures = [executor.submit(execute, item) for item in work]
                for index, future in enumerate(as_completed(futures), start=1):
                    future.result()
                    if index % 96 == 0:
                        manifest["last_updated_at_utc"] = base._utc_now()
                        manifest["journals"] = {
                            "attempt_ledger": ledger.reference(),
                            "raw_responses": {
                                target_id: journal.reference()
                                for target_id, journal in raw_journals.items()
                            },
                        }
                        base._seal(manifest)
                        base._atomic_json(manifest_path, manifest)

        retained = _completed_index(
            raw_journals, trials_by_id=trials_by_id, subjects_by_id=subjects_by_id
        )
        rows = list(retained.values())
        planned = len(subjects) * len(trials)
        manifest["summary"] = {
            "planned_generations": planned,
            "retained_trial_records": len(rows),
            "responses_received": sum(row.get("raw_response") is not None for row in rows),
            "failed_without_response": sum(row.get("raw_response") is None for row in rows),
            "format_valid": sum(row.get("format_valid") is True for row in rows),
            "format_invalid_retained": sum(
                row.get("raw_response") is not None and row.get("format_valid") is not True
                for row in rows
            ),
            "response_model_identity_mismatches": sum(
                row.get("raw_response") is not None and row.get("model_identity_valid") is not True
                for row in rows
            ),
        }
        manifest["journals"] = {
            "attempt_ledger": ledger.reference(),
            "raw_responses": {
                target_id: journal.reference() for target_id, journal in raw_journals.items()
            },
        }
        manifest["complete"] = (
            len(rows) == planned
            and manifest["summary"]["failed_without_response"] == 0
            and manifest["summary"]["response_model_identity_mismatches"] == 0
        )
        manifest["last_updated_at_utc"] = base._utc_now()
        if manifest["complete"]:
            manifest["completed_at_utc"] = base._utc_now()
        base._seal(manifest)
        base._atomic_json(manifest_path, manifest)

        summary, root_rows = _root_aware_summaries(
            rows=rows,
            config=config,
            schedule_sha256=schedule_sha256,
            journal_tails={target: journal.tail for target, journal in raw_journals.items()},
        )
        base._atomic_json(sanitized_dir / "summary.json", summary)
        _atomic_jsonl(sanitized_dir / "root_summaries.jsonl", root_rows)
        return manifest
    finally:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
        run_lock.close()


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
    parser = argparse.ArgumentParser(
        description="Run the immutable v1 Part 1 role-conditioned sentinel calibration."
    )
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=Path("agents/agent_config.registry.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-workers", type=_positive_int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--max-workers-per-provider",
        type=_positive_int,
        default=DEFAULT_MAX_WORKERS_PER_PROVIDER,
    )
    parser.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--initial-backoff-seconds",
        type=_nonnegative_float,
        default=DEFAULT_BACKOFF_SECONDS,
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()
    client = _provider_safe_client(args.timeout_seconds)
    manifest = run_calibration(
        compatibility_path=args.compatibility,
        registry_path=args.registry,
        output_dir=args.output_dir,
        client=client,
        max_workers=args.max_workers,
        max_workers_per_provider=args.max_workers_per_provider,
        max_attempts=args.max_attempts,
        initial_backoff_seconds=args.initial_backoff_seconds,
        resume=args.resume,
    )
    print(
        f"Retained {manifest['summary']['retained_trial_records']}/"
        f"{manifest['summary']['planned_generations']} role-calibration records."
    )
    print(f"Sanitized summary: {args.output_dir / 'sanitized' / 'summary.json'}")
    return 0 if manifest["complete"] else 1


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (
        RoleCalibrationError,
        base.InferenceHubPart1PanelError,
        InferenceHubDiscoveryError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Part 1 role calibration failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
