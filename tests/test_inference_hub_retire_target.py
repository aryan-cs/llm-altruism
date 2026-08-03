from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import experiments.misc.inference_hub_retire_target as retirement_module
from analysis.build_final_results import (
    _part0_unavailable_failure,
    _part1_unavailable_failure,
)
from experiments.misc.inference_hub_part0_panel import (
    _SOURCE_PATHS as PART0_SOURCE_PATHS,
    _manifest_bindings as _part0_manifest_bindings,
    _record_indexes,
)
from experiments.misc.inference_hub_part1_panel import (
    _ChainedJournal,
    _SOURCE_PATHS as PART1_SOURCE_PATHS,
    _atomic_json,
    _completed_index,
    _seal,
    _sha256_file,
    _sha256_json,
    _manifest_bindings as _part1_manifest_bindings,
    build_draft_trials,
)
from experiments.misc.inference_hub_retire_target import (
    PART1_STRATIFIED_SOURCE,
    RETIREMENT_FAILURE_CODE,
    TargetRetirementError,
    retire_target,
)
from experiments.misc.inference_hub_part1_stratified_panel import (
    build_stratified_trials,
)


TARGET = "subject.alpha"
HEALTHY = "subject.beta"


def _subject(target_id: str) -> dict[str, Any]:
    suffix = target_id.rsplit(".", 1)[-1]
    return {
        "target_id": target_id,
        "upstream_provider": suffix,
        "model": f"{suffix}-model",
        "route": f"region/{suffix}-model",
    }


def _private_tree(tmp_path: Path, name: str) -> tuple[Path, Path]:
    output = tmp_path / name
    private = output / "private"
    raw = private / "raw_responses"
    raw.mkdir(parents=True)
    for directory in (output, private, raw):
        os.chmod(directory, 0o700)
    return private, private / "manifest.json"


def _ledger_failure(
    ledger: _ChainedJournal, *, target_id: str, work_id: str, part: str,
) -> None:
    attempt_id = f"{part}-attempt-{target_id}-{work_id}"
    reservation: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": f"inference_hub_{part}_attempt_ledger",
        "event": "reserved_before_dispatch",
        "attempt_id": attempt_id,
        "target_id": target_id,
        "request_sha256": hashlib.sha256(attempt_id.encode()).hexdigest(),
    }
    if part == "part0":
        reservation.update({"role": "subject", "work_id": work_id})
    else:
        reservation["trial_id"] = work_id
    ledger.append(reservation)
    ledger.append({
        "schema_version": 1,
        "artifact_type": f"inference_hub_{part}_attempt_ledger",
        "event": "attempt_completed",
        "attempt_id": attempt_id,
        "outcome": "failed",
        "failure_code": "http_error",
        "http_status": 400,
    })


def _append_unmatched_reservation(
    manifest_path: Path, *, target_id: str, attempt_id: str, trial_id: str,
    part: str,
) -> None:
    manifest = _read_json(manifest_path)
    ledger = _ChainedJournal(Path(manifest["journals"]["attempt_ledger"]["path"]))
    reservation: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": f"inference_hub_{part}_attempt_ledger",
        "event": "reserved_before_dispatch",
        "attempt_id": attempt_id,
        "target_id": target_id,
        "request_sha256": hashlib.sha256(attempt_id.encode()).hexdigest(),
    }
    if part == "part0":
        reservation.update({"role": "subject", "work_id": trial_id})
    else:
        reservation["trial_id"] = trial_id
    ledger.append(reservation)


def _append_p0_retained_response(
    manifest_path: Path, *, target_id: str, trial_id: str,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    trial = next(row for row in manifest["schedule"] if row["trial_id"] == trial_id)
    attempt_id = f"part0-retained-{target_id}-{trial_id}"
    ledger = _ChainedJournal(Path(manifest["journals"]["attempt_ledger"]["path"]))
    ledger.append({
        "schema_version": 1,
        "artifact_type": "inference_hub_part0_attempt_ledger",
        "event": "reserved_before_dispatch",
        "attempt_id": attempt_id,
        "target_id": target_id,
        "role": "subject",
        "work_id": trial_id,
        "request_sha256": hashlib.sha256(attempt_id.encode()).hexdigest(),
    })
    route = _subject(target_id)["route"]
    raw = {"model": route, "choices": []}
    journal = _ChainedJournal(
        Path(manifest["journals"]["raw_responses"][target_id]["path"])
    )
    retained = journal.append({
        "schema_version": 1,
        "artifact_type": "inference_hub_part0_private_record",
        "event": "subject_response_retained",
        "target_id": target_id,
        "trial_id": trial_id,
        "root_id": trial["root_id"],
        "language": trial["language"],
        "prompt_sha256": trial["prompt_sha256"],
        "requested_route": route,
        "response_model": route,
        "model_identity_valid": True,
        "visible_response": "I cannot assist with that request.",
        "attempt_id": attempt_id,
        "raw_response": raw,
        "raw_response_sha256": _sha256_json(raw),
    })
    ledger.append({
        "schema_version": 1,
        "artifact_type": "inference_hub_part0_attempt_ledger",
        "event": "attempt_completed",
        "attempt_id": attempt_id,
        "outcome": "response_retained",
        "failure_code": None,
        "raw_record_sha256": retained["record_sha256"],
    })
    return retained


def _base_manifest(artifact_type: str) -> dict[str, Any]:
    source_paths = (
        PART0_SOURCE_PATHS
        if artifact_type == "inference_hub_part0_accelerated_private_panel"
        else PART1_SOURCE_PATHS
    )
    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "complete": False,
        "subject_routes": [_subject(TARGET), _subject(HEALTHY)],
        "source_artifacts": {
            str(path.resolve()): _sha256_file(path.resolve()) for path in source_paths
        },
        "summary": {},
    }


def _write_p0_fixture(
    tmp_path: Path, *, failure_target: str = TARGET,
    retained_conflict: bool = False,
) -> tuple[Path, dict[str, Any]]:
    private, manifest_path = _private_tree(tmp_path, "p0")
    schedule = [
        {
            "trial_id": f"trial-{index}",
            "root_id": f"root-{index}",
            "language": "english",
            "prompt_sha256": hashlib.sha256(f"prompt-{index}".encode()).hexdigest(),
            "source_stratum": "test",
        }
        for index in range(2)
    ]
    ledger = _ChainedJournal(private / "attempt_ledger.jsonl")
    _ledger_failure(
        ledger, target_id=failure_target, work_id="trial-0", part="part0"
    )
    journals = {
        target_id: _ChainedJournal(private / "raw_responses" / f"{target_id}.jsonl")
        for target_id in (TARGET, HEALTHY)
    }
    for journal in journals.values():
        journal.path.touch(mode=0o600)
    if failure_target == TARGET:
        journals[TARGET].append({
            "schema_version": 1,
            "artifact_type": "inference_hub_part0_private_record",
            "event": "unit_completed",
            "target_id": TARGET,
            "trial_id": "trial-0",
            "root_id": "root-0",
            "language": "english",
            "outcome": "INVALID",
            "operational_failure": {
                "failure_code": "http_error", "http_status": 400,
            },
        })
    if retained_conflict:
        attempt_id = "part0-retained-awaiting-judge"
        ledger.append({
            "schema_version": 1,
            "artifact_type": "inference_hub_part0_attempt_ledger",
            "event": "reserved_before_dispatch",
            "attempt_id": attempt_id,
            "target_id": TARGET,
            "role": "subject",
            "work_id": "trial-1",
            "request_sha256": hashlib.sha256(attempt_id.encode()).hexdigest(),
        })
        raw = {"model": _subject(TARGET)["route"], "choices": []}
        retained = journals[TARGET].append({
            "schema_version": 1,
            "artifact_type": "inference_hub_part0_private_record",
            "event": "subject_response_retained",
            "target_id": TARGET,
            "trial_id": "trial-1",
            "root_id": "root-1",
            "language": "english",
            "prompt_sha256": schedule[1]["prompt_sha256"],
            "requested_route": _subject(TARGET)["route"],
            "response_model": _subject(TARGET)["route"],
            "model_identity_valid": True,
            "visible_response": "I cannot help with that request.",
            "attempt_id": attempt_id,
            "raw_response": raw,
            "raw_response_sha256": _sha256_json(raw),
        })
        ledger.append({
            "schema_version": 1,
            "artifact_type": "inference_hub_part0_attempt_ledger",
            "event": "attempt_completed",
            "attempt_id": attempt_id,
            "outcome": "response_retained",
            "failure_code": None,
            "raw_record_sha256": retained["record_sha256"],
        })
    manifest = _base_manifest("inference_hub_part0_accelerated_private_panel")
    manifest.update({
        "executed_trial_count_per_subject": 2,
        "schedule": schedule,
        "schedule_sha256": _sha256_json(schedule),
        "summary": {
            "subject_transport_failures": int(failure_target == TARGET),
            "judge_failed_units": 0,
            "subject_model_identity_mismatches": 0,
            "judge_model_identity_mismatches": 0,
        },
        "judge": {"route": "judge/test-model"},
        "journals": {
            "attempt_ledger": ledger.reference(),
            "raw_responses": {
                target_id: journal.reference()
                for target_id, journal in journals.items()
            },
        },
    })
    _seal(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest_path, manifest


def _p1_schedule_binding(trials: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
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


def _write_p1_fixture(
    tmp_path: Path, *, failure_target: str = TARGET, semantic_only: bool = False,
    stratified: bool = False,
) -> tuple[Path, tuple[Any, ...]]:
    private, manifest_path = _private_tree(tmp_path, "p1")
    trials = (
        build_stratified_trials(base_seed=20260802, limit=12)
        if stratified else build_draft_trials(base_seed=20260802, limit=12)
    )
    first = trials[0]
    ledger = _ChainedJournal(private / "attempt_ledger.jsonl")
    journals = {
        target_id: _ChainedJournal(private / "raw_responses" / f"{target_id}.jsonl")
        for target_id in (TARGET, HEALTHY)
    }
    for journal in journals.values():
        journal.path.touch(mode=0o600)
    if semantic_only:
        attempt_id = "part1-semantic"
        ledger.append({
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_attempt_ledger",
            "event": "reserved_before_dispatch",
            "attempt_id": attempt_id,
            "target_id": TARGET,
            "trial_id": first.trial_id,
            "request_sha256": hashlib.sha256(attempt_id.encode()).hexdigest(),
        })
        raw = {"model": _subject(TARGET)["route"], "choices": []}
        retained = journals[TARGET].append({
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_raw_response",
            "target_id": TARGET,
            "upstream_provider": _subject(TARGET)["upstream_provider"],
            "model": _subject(TARGET)["model"],
            "requested_route": _subject(TARGET)["route"],
            "response_model": _subject(TARGET)["route"],
            "model_identity_valid": True,
            "trial_id": first.trial_id,
            "root_id": first.root_id,
            "prompt_sha256": first.prompt_hash,
            "attempt_id": attempt_id,
            "raw_response": raw,
            "raw_response_sha256": _sha256_json(raw),
            "format_valid": False,
        })
        ledger.append({
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_attempt_ledger",
            "event": "attempt_completed",
            "attempt_id": attempt_id,
            "outcome": "response_retained",
            "failure_code": None,
            "raw_record_sha256": retained["record_sha256"],
        })
    else:
        _ledger_failure(
            ledger, target_id=failure_target, work_id=first.trial_id, part="part1"
        )
        if failure_target == TARGET:
            journals[TARGET].append({
                "schema_version": 1,
                "artifact_type": "inference_hub_part1_raw_response",
                "target_id": TARGET,
                "upstream_provider": _subject(TARGET)["upstream_provider"],
                "model": _subject(TARGET)["model"],
                "requested_route": _subject(TARGET)["route"],
                "response_model": None,
                "model_identity_valid": False,
                "trial_id": first.trial_id,
                "root_id": first.root_id,
                "prompt_sha256": first.prompt_hash,
                "raw_response": None,
                "raw_response_sha256": None,
                "failure": {"failure_code": "http_error", "http_status": 400},
            })
    schedule = _p1_schedule_binding(trials)
    manifest = _base_manifest("inference_hub_part1_large_n_exploratory_panel")
    if stratified:
        manifest["source_artifacts"][str(PART1_STRATIFIED_SOURCE)] = _sha256_file(
            PART1_STRATIFIED_SOURCE
        )
    manifest.update({
        "base_seed": 20260802,
        "executed_trial_count_per_subject": 12,
        "trial_limit": 12,
        "executed_schedule_sha256": _sha256_json(schedule),
        "summary": {
            "failed_without_response": int(failure_target == TARGET and not semantic_only),
            "response_model_identity_mismatches": 0,
        },
        "journals": {
            "attempt_ledger": ledger.reference(),
            "raw_responses": {
                target_id: journal.reference()
                for target_id, journal in journals.items()
            },
        },
    })
    _seal(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest_path, trials


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_part0_retirement_is_nonbehavioral_isolated_and_final_results_compatible(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_p0_fixture(tmp_path)
    before = _read_json(manifest_path)
    healthy_path = Path(before["journals"]["raw_responses"][HEALTHY]["path"])
    healthy_hash = _sha256_file(healthy_path)

    audit = retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="reproducible HTTP 400 on the frozen route",
    )

    after = _read_json(manifest_path)
    target_path = Path(after["journals"]["raw_responses"][TARGET]["path"])
    rows = _ChainedJournal(target_path).records
    retired = [row for row in rows if row.get("event") == "unit_operationally_retired"]
    assert len(retired) == 1
    assert retired[0]["dispatched"] is False
    assert "outcome" not in retired[0]
    assert _sha256_file(healthy_path) == healthy_hash
    assert after["target_retirements"] == [audit]
    assert after["evidence_sha256"] != before["evidence_sha256"]
    assert _part0_unavailable_failure(after, TARGET)["total_failure_count"] == 1

    trials = {
        f"trial-{index}": SimpleNamespace(
            trial_id=f"trial-{index}", root_id=f"root-{index}",
            language="english", prompt_sha256=hashlib.sha256(
                f"prompt-{index}".encode()
            ).hexdigest(),
        )
        for index in range(2)
    }
    subject = _subject(TARGET)
    _, _, terminals = _record_indexes(
        _ChainedJournal(target_path), subject=subject, trials=trials
    )
    assert set(terminals) == {"trial-0", "trial-1"}


def test_part1_retirement_skips_remaining_without_actions_and_is_compatible(
    tmp_path: Path,
) -> None:
    manifest_path, trials = _write_p1_fixture(tmp_path)
    before = _read_json(manifest_path)
    healthy_path = Path(before["journals"]["raw_responses"][HEALTHY]["path"])
    healthy_hash = _sha256_file(healthy_path)

    audit = retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="transport attempt budget exhausted",
    )

    after = _read_json(manifest_path)
    target_path = Path(after["journals"]["raw_responses"][TARGET]["path"])
    rows = _ChainedJournal(target_path).records
    retired = [
        row for row in rows
        if isinstance(row.get("failure"), dict)
        and row["failure"].get("failure_code") == RETIREMENT_FAILURE_CODE
    ]
    assert len(retired) == 11
    assert all(row["failure"]["dispatched"] is False for row in retired)
    assert all("parsed_action" not in row for row in retired)
    assert _sha256_file(healthy_path) == healthy_hash
    assert audit["non_dispatched_terminal_record_count"] == 11
    assert _part1_unavailable_failure(after, TARGET)["total_failure_count"] == 1

    journals = {TARGET: _ChainedJournal(target_path)}
    completed = _completed_index(
        journals,
        trials_by_id={trial.trial_id: trial for trial in trials},
        subjects_by_id={TARGET: _subject(TARGET)},
    )
    assert len(completed) == 12


def test_rejects_semantic_invalid_without_genuine_operational_evidence(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_p1_fixture(tmp_path, semantic_only=True)
    before = manifest_path.read_bytes()
    with pytest.raises(TargetRetirementError, match="lacks genuine dispatched"):
        retire_target(
            manifest_path=manifest_path, target_id=TARGET,
            reason="format invalid is not operational evidence",
        )
    assert manifest_path.read_bytes() == before


def test_rejects_failure_bound_to_another_target_and_unknown_target(tmp_path: Path) -> None:
    manifest_path, _ = _write_p0_fixture(tmp_path, failure_target=HEALTHY)
    with pytest.raises(TargetRetirementError, match="lacks genuine dispatched"):
        retire_target(
            manifest_path=manifest_path, target_id=TARGET, reason="wrong target",
        )
    with pytest.raises(TargetRetirementError, match="absent from the manifest"):
        retire_target(
            manifest_path=manifest_path, target_id="subject.unknown", reason="unknown",
        )


def test_part0_retires_retained_unjudged_response_with_exact_provenance(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_p0_fixture(tmp_path, retained_conflict=True)
    retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="provider unavailable before retained response could be judged",
    )

    manifest = _read_json(manifest_path)
    rows = _ChainedJournal(
        Path(manifest["journals"]["raw_responses"][TARGET]["path"])
    ).records
    retained = next(
        row for row in rows
        if row.get("event") == "subject_response_retained"
        and row.get("trial_id") == "trial-1"
    )
    retired = next(
        row for row in rows
        if row.get("event") == "unit_operationally_retired"
        and row.get("trial_id") == "trial-1"
    )
    assert "outcome" not in retired
    assert retired["retirement"]["judge_dispatch_performed"] is False
    assert retired["retirement"]["retained_subject_response"] == {
        "record_sha256": retained["record_sha256"],
        "raw_response_sha256": retained["raw_response_sha256"],
        "attempt_id": retained["attempt_id"],
    }
    assert _part0_unavailable_failure(manifest, TARGET)["total_failure_count"] == 1


def test_part0_retirement_leaves_healthy_retained_response_judge_eligible(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_p0_fixture(tmp_path)
    retained = _append_p0_retained_response(
        manifest_path, target_id=HEALTHY, trial_id="trial-1",
    )
    before = _read_json(manifest_path)
    healthy_path = Path(before["journals"]["raw_responses"][HEALTHY]["path"])
    healthy_before = healthy_path.read_bytes()

    retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="selected provider unavailable; healthy response continues",
    )

    assert healthy_path.read_bytes() == healthy_before
    trials = {
        f"trial-{index}": SimpleNamespace(
            trial_id=f"trial-{index}", root_id=f"root-{index}",
            language="english", prompt_sha256=hashlib.sha256(
                f"prompt-{index}".encode()
            ).hexdigest(),
        )
        for index in range(2)
    }
    retained_subjects, _batches, terminals = _record_indexes(
        _ChainedJournal(healthy_path), subject=_subject(HEALTHY), trials=trials,
    )
    assert retained_subjects["trial-1"]["record_sha256"] == retained["record_sha256"]
    assert "trial-1" not in terminals
    assert retained_subjects["trial-1"]["model_identity_valid"] is True
    assert isinstance(retained_subjects["trial-1"]["visible_response"], str)


def test_rejects_live_lock_tampered_hash_and_repeated_retirement(tmp_path: Path) -> None:
    manifest_path, _ = _write_p1_fixture(tmp_path / "lock")
    lock_path = manifest_path.parent / ".run.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TargetRetirementError, match="run lock is live"):
            retire_target(
                manifest_path=manifest_path, target_id=TARGET, reason="locked",
            )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    retire_target(
        manifest_path=manifest_path, target_id=TARGET, reason="eligible failure",
    )
    with pytest.raises(TargetRetirementError, match="already operationally retired"):
        retire_target(
            manifest_path=manifest_path, target_id=TARGET, reason="repeat",
        )

    tampered_path, _ = _write_p0_fixture(tmp_path / "tamper")
    tampered = _read_json(tampered_path)
    tampered["summary"]["subject_transport_failures"] = 999
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(TargetRetirementError, match="self-hash failed"):
        retire_target(
            manifest_path=tampered_path, target_id=TARGET, reason="tampered",
        )


def test_audits_narrow_runner_source_hash_migration(tmp_path: Path) -> None:
    manifest_path, _ = _write_p1_fixture(tmp_path)
    manifest = _read_json(manifest_path)
    runner_path = str(Path(PART1_SOURCE_PATHS[0]).resolve())
    manifest["source_artifacts"][runner_path] = "0" * 64
    _seal(manifest)
    _atomic_json(manifest_path, manifest)

    audit = retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="runner gains administrative retirement support",
    )

    after = _read_json(manifest_path)
    assert after["source_artifacts"][runner_path] == _sha256_file(Path(runner_path))
    assert audit["source_artifact_hash_migrations"] == [{
        "path": runner_path,
        "prior_sha256": "0" * 64,
        "resumed_runner_sha256": _sha256_file(Path(runner_path)),
    }]
    resume_fresh = dict(manifest)
    resume_fresh["source_artifacts"] = dict(after["source_artifacts"])
    assert _part1_manifest_bindings(after) == _part1_manifest_bindings(resume_fresh)


def test_schema_faithful_stratified_part1_source_and_schedule_are_supported(
    tmp_path: Path,
) -> None:
    manifest_path, trials = _write_p1_fixture(tmp_path, stratified=True)
    before = _read_json(manifest_path)
    assert set(before["source_artifacts"]) == {
        *(str(path.resolve()) for path in PART1_SOURCE_PATHS),
        str(PART1_STRATIFIED_SOURCE),
    }

    retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="stratified selected route is unavailable",
    )

    after = _read_json(manifest_path)
    assert after["source_artifacts"][str(PART1_STRATIFIED_SOURCE)] == _sha256_file(
        PART1_STRATIFIED_SOURCE
    )
    rows = _ChainedJournal(
        Path(after["journals"]["raw_responses"][TARGET]["path"])
    ).records
    assert {row["trial_id"] for row in rows} == {
        trial.trial_id for trial in trials
    }


def test_stratified_wrapper_hash_is_frozen_and_cannot_migrate(tmp_path: Path) -> None:
    manifest_path, _ = _write_p1_fixture(tmp_path, stratified=True)
    manifest = _read_json(manifest_path)
    manifest["source_artifacts"][str(PART1_STRATIFIED_SOURCE)] = "0" * 64
    _seal(manifest)
    _atomic_json(manifest_path, manifest)

    with pytest.raises(TargetRetirementError, match="Frozen non-runner source changed"):
        retire_target(
            manifest_path=manifest_path, target_id=TARGET,
            reason="wrapper mutations are outside the migration contract",
        )


def test_part0_migrates_both_bound_runner_hashes_without_relaxing_resume(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_p0_fixture(tmp_path)
    manifest = _read_json(manifest_path)
    runner_paths = {
        str(Path(PART0_SOURCE_PATHS[0]).resolve()),
        str(Path(PART1_SOURCE_PATHS[0]).resolve()),
    }
    for path in runner_paths:
        manifest["source_artifacts"][path] = "0" * 64
    _seal(manifest)
    _atomic_json(manifest_path, manifest)

    retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="both Part 0 bound runner files gain retirement support",
    )

    after = _read_json(manifest_path)
    migrations = after["target_retirements"][0]["source_artifact_hash_migrations"]
    assert {row["path"] for row in migrations} == runner_paths
    resume_fresh = dict(manifest)
    resume_fresh["source_artifacts"] = dict(after["source_artifacts"])
    assert _part0_manifest_bindings(after) == _part0_manifest_bindings(resume_fresh)


def test_rejects_nonrunner_source_change_and_journal_checkpoint_tamper(
    tmp_path: Path,
) -> None:
    source_path, _ = _write_p1_fixture(tmp_path / "source")
    manifest = _read_json(source_path)
    nonrunner = next(
        path for path in manifest["source_artifacts"]
        if path != str(Path(PART1_SOURCE_PATHS[0]).resolve())
    )
    manifest["source_artifacts"][nonrunner] = "0" * 64
    _seal(manifest)
    _atomic_json(source_path, manifest)
    with pytest.raises(TargetRetirementError, match="Frozen non-runner source changed"):
        retire_target(
            manifest_path=source_path, target_id=TARGET,
            reason="must not relax the frozen source contract",
        )

    checkpoint_path, _ = _write_p0_fixture(tmp_path / "checkpoint")
    checkpoint = _read_json(checkpoint_path)
    checkpoint["journals"]["attempt_ledger"]["tail_record_sha256"] = "0" * 64
    _seal(checkpoint)
    _atomic_json(checkpoint_path, checkpoint)
    with pytest.raises(TargetRetirementError, match="checkpoint tail changed"):
        retire_target(
            manifest_path=checkpoint_path, target_id=TARGET,
            reason="must not accept a changed chain checkpoint",
        )


def test_recovers_append_complete_manifest_write_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_p1_fixture(tmp_path)
    real_atomic_json = retirement_module._atomic_json

    def fail_manifest_write(_path: Path, _value: object) -> None:
        raise OSError("simulated manifest write interruption")

    monkeypatch.setattr(retirement_module, "_atomic_json", fail_manifest_write)
    with pytest.raises(OSError, match="simulated manifest write interruption"):
        retire_target(
            manifest_path=manifest_path, target_id=TARGET,
            reason="recover this exact interrupted retirement",
        )
    monkeypatch.setattr(retirement_module, "_atomic_json", real_atomic_json)

    audit = retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="recover this exact interrupted retirement",
    )
    rows = _ChainedJournal(
        Path(_read_json(manifest_path)["journals"]["raw_responses"][TARGET]["path"])
    ).records
    retired = [
        row for row in rows
        if isinstance(row.get("failure"), dict)
        and row["failure"].get("failure_code") == RETIREMENT_FAILURE_CODE
    ]
    assert len(retired) == 11
    assert audit["non_dispatched_terminal_record_count"] == 11


def test_explicitly_closes_stale_reservation_without_using_it_as_evidence(
    tmp_path: Path,
) -> None:
    manifest_path, trials = _write_p1_fixture(tmp_path)
    attempt_id = "part1-graceful-term-selected"
    _append_unmatched_reservation(
        manifest_path, target_id=TARGET, attempt_id=attempt_id,
        trial_id=trials[1].trial_id, part="part1",
    )
    manifest_before = manifest_path.read_bytes()
    ledger_path = Path(
        _read_json(manifest_path)["journals"]["attempt_ledger"]["path"]
    )
    ledger_before = ledger_path.read_bytes()
    with pytest.raises(TargetRetirementError, match="explicit all-target"):
        retire_target(
            manifest_path=manifest_path, target_id=TARGET,
            reason="runner stopped during one selected dispatch",
        )
    assert manifest_path.read_bytes() == manifest_before
    assert ledger_path.read_bytes() == ledger_before

    audit = retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="runner stopped during one selected dispatch",
        close_all_stale_reservations=True,
    )

    ledger_rows = _ChainedJournal(ledger_path).records
    stale = [
        row for row in ledger_rows
        if row.get("attempt_id") == attempt_id
        and row.get("event") == "attempt_completed"
    ]
    assert len(stale) == 1
    assert {
        key: stale[0].get(key)
        for key in ("outcome", "failure_code", "transient", "http_status")
    } == {
        "outcome": "failed",
        "failure_code": "stale_reservation_retried",
        "transient": True,
        "http_status": None,
    }
    recovery = audit["all_target_stale_reservation_recovery"]
    assert recovery["records"] == [{
        "target_id": TARGET,
        "attempt_id": attempt_id,
        "record_sha256": stale[0]["record_sha256"],
        "appended_in_finalization_invocation": True,
    }]
    assert stale[0]["record_sha256"] not in audit[
        "genuine_failure_evidence_record_sha256"
    ]
    assert len(audit["genuine_failure_evidence_record_sha256"]) == 1


def test_part0_closes_selected_stale_with_runner_exact_record(tmp_path: Path) -> None:
    manifest_path, _ = _write_p0_fixture(tmp_path)
    attempt_id = "part0-graceful-term-selected"
    healthy_attempt_id = "part0-graceful-term-healthy"
    _append_unmatched_reservation(
        manifest_path, target_id=TARGET, attempt_id=attempt_id,
        trial_id="trial-1", part="part0",
    )
    _append_unmatched_reservation(
        manifest_path, target_id=HEALTHY, attempt_id=healthy_attempt_id,
        trial_id="trial-1", part="part0",
    )
    before = _read_json(manifest_path)
    healthy_raw_path = Path(
        before["journals"]["raw_responses"][HEALTHY]["path"]
    )
    healthy_raw_before = healthy_raw_path.read_bytes()

    audit = retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="Part 0 runner stopped during selected dispatch",
        close_all_stale_reservations=True,
    )

    ledger = _ChainedJournal(
        Path(_read_json(manifest_path)["journals"]["attempt_ledger"]["path"])
    )
    stale = [
        row for row in ledger.records
        if row.get("attempt_id") in {attempt_id, healthy_attempt_id}
        and row.get("event") == "attempt_completed"
    ]
    assert len(stale) == 2
    assert all(
        row["artifact_type"] == "inference_hub_part0_attempt_ledger"
        and row["failure_code"] == "stale_reservation_retried"
        and row["transient"] is True
        and row["http_status"] is None
        and row["completed_at_utc"].endswith("Z")
        for row in stale
    )
    assert all(
        row["record_sha256"] not in audit[
            "genuine_failure_evidence_record_sha256"
        ]
        for row in stale
    )
    assert healthy_raw_path.read_bytes() == healthy_raw_before
    trials = {
        f"trial-{index}": SimpleNamespace(
            trial_id=f"trial-{index}", root_id=f"root-{index}",
            language="english", prompt_sha256=hashlib.sha256(
                f"prompt-{index}".encode()
            ).hexdigest(),
        )
        for index in range(2)
    }
    retained, _batches, terminals = _record_indexes(
        _ChainedJournal(healthy_raw_path), subject=_subject(HEALTHY), trials=trials,
    )
    assert retained == {}
    assert terminals == {}


def test_all_target_stale_recovery_preserves_healthy_retry_scheduling(
    tmp_path: Path,
) -> None:
    manifest_path, trials = _write_p1_fixture(tmp_path)
    open_attempts = [
        (TARGET, "selected-open-1", trials[1].trial_id),
        (TARGET, "selected-open-2", trials[2].trial_id),
        (HEALTHY, "healthy-open-1", trials[1].trial_id),
        (HEALTHY, "healthy-open-2", trials[2].trial_id),
        (HEALTHY, "healthy-open-3", trials[3].trial_id),
    ]
    for target_id, attempt_id, trial_id in open_attempts:
        _append_unmatched_reservation(
            manifest_path, target_id=target_id, attempt_id=attempt_id,
            trial_id=trial_id, part="part1",
        )
    before = _read_json(manifest_path)
    healthy_raw_path = Path(
        before["journals"]["raw_responses"][HEALTHY]["path"]
    )
    healthy_raw_before = healthy_raw_path.read_bytes()

    audit = retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="graceful TERM left concurrent provider calls",
        close_all_stale_reservations=True,
    )

    recovery = audit["all_target_stale_reservation_recovery"]
    assert recovery["scope"] == "all_unmatched_reservations_across_all_targets"
    assert recovery["appended_in_finalization_invocation"] == len(open_attempts)
    assert recovery["behavioral_terminal_records_created"] == 0
    assert {
        (row["target_id"], row["attempt_id"]) for row in recovery["records"]
    } == {(target_id, attempt_id) for target_id, attempt_id, _ in open_attempts}
    assert healthy_raw_path.read_bytes() == healthy_raw_before

    after = _read_json(manifest_path)
    assert [row["target_id"] for row in after["target_retirements"]] == [TARGET]
    journals = {
        target_id: _ChainedJournal(
            Path(after["journals"]["raw_responses"][target_id]["path"])
        )
        for target_id in (TARGET, HEALTHY)
    }
    completed = _completed_index(
        journals,
        trials_by_id={trial.trial_id: trial for trial in trials},
        subjects_by_id={
            TARGET: _subject(TARGET), HEALTHY: _subject(HEALTHY),
        },
    )
    assert all((HEALTHY, trial.trial_id) not in completed for trial in trials)
    assert all(
        row.get("target_id") == TARGET
        for row in journals[TARGET].records
        if isinstance(row.get("failure"), dict)
        and row["failure"].get("failure_code") == RETIREMENT_FAILURE_CODE
    )


def test_all_target_stale_recovery_rejects_any_retained_raw_response(
    tmp_path: Path,
) -> None:
    retained_path, trials = _write_p1_fixture(tmp_path)

    retained_path, trials = _write_p1_fixture(tmp_path / "retained")
    attempt_id = "healthy-open-with-raw"
    _append_unmatched_reservation(
        retained_path, target_id=HEALTHY, attempt_id=attempt_id,
        trial_id=trials[1].trial_id, part="part1",
    )
    retained_manifest = _read_json(retained_path)
    raw_path = Path(
        retained_manifest["journals"]["raw_responses"][HEALTHY]["path"]
    )
    raw = {"model": _subject(HEALTHY)["route"], "choices": []}
    _ChainedJournal(raw_path).append({
        "schema_version": 1,
        "artifact_type": "inference_hub_part1_raw_response",
        "target_id": HEALTHY,
        "upstream_provider": _subject(HEALTHY)["upstream_provider"],
        "model": _subject(HEALTHY)["model"],
        "requested_route": _subject(HEALTHY)["route"],
        "response_model": _subject(HEALTHY)["route"],
        "model_identity_valid": True,
        "trial_id": trials[1].trial_id,
        "root_id": trials[1].root_id,
        "prompt_sha256": trials[1].prompt_hash,
        "attempt_id": attempt_id,
        "raw_response": raw,
        "raw_response_sha256": _sha256_json(raw),
    })
    with pytest.raises(TargetRetirementError, match="has a retained raw response"):
        retire_target(
            manifest_path=retained_path, target_id=TARGET,
            reason="raw must be recovered, not made stale",
            close_all_stale_reservations=True,
        )


def test_stale_completion_cannot_create_its_own_genuine_failure_evidence(
    tmp_path: Path,
) -> None:
    manifest_path, trials = _write_p1_fixture(tmp_path, semantic_only=True)
    attempt_id = "selected-open-without-prior-genuine-failure"
    _append_unmatched_reservation(
        manifest_path, target_id=TARGET, attempt_id=attempt_id,
        trial_id=trials[1].trial_id, part="part1",
    )
    ledger_path = Path(
        _read_json(manifest_path)["journals"]["attempt_ledger"]["path"]
    )
    before = ledger_path.read_bytes()
    with pytest.raises(TargetRetirementError, match="lacks genuine dispatched"):
        retire_target(
            manifest_path=manifest_path, target_id=TARGET,
            reason="synthetic stale completion is not independent evidence",
            close_all_stale_reservations=True,
        )
    assert ledger_path.read_bytes() == before


def test_stale_closure_is_idempotent_after_crash_before_retirement_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, trials = _write_p1_fixture(tmp_path)
    attempt_id = "selected-open-before-crash"
    healthy_attempt_id = "healthy-open-before-crash"
    _append_unmatched_reservation(
        manifest_path, target_id=TARGET, attempt_id=attempt_id,
        trial_id=trials[1].trial_id, part="part1",
    )
    _append_unmatched_reservation(
        manifest_path, target_id=HEALTHY, attempt_id=healthy_attempt_id,
        trial_id=trials[1].trial_id, part="part1",
    )
    real_schedule = retirement_module._part1_schedule

    def fail_after_stale_closure(_manifest: object) -> tuple[Any, ...]:
        raise RuntimeError("simulated crash after stale closure")

    monkeypatch.setattr(retirement_module, "_part1_schedule", fail_after_stale_closure)
    with pytest.raises(RuntimeError, match="simulated crash"):
        retire_target(
            manifest_path=manifest_path, target_id=TARGET,
            reason="idempotent stale close",
            close_all_stale_reservations=True,
        )
    monkeypatch.setattr(retirement_module, "_part1_schedule", real_schedule)

    audit = retire_target(
        manifest_path=manifest_path, target_id=TARGET,
        reason="idempotent stale close",
        close_all_stale_reservations=True,
    )
    ledger = _ChainedJournal(
        Path(_read_json(manifest_path)["journals"]["attempt_ledger"]["path"])
    )
    stale = [
        row for row in ledger.records
        if row.get("attempt_id") in {attempt_id, healthy_attempt_id}
        and row.get("event") == "attempt_completed"
        and row.get("failure_code") == "stale_reservation_retried"
    ]
    assert len(stale) == 2
    stale_by_id = {row["attempt_id"]: row for row in stale}
    recovery = audit["all_target_stale_reservation_recovery"]
    assert recovery["records"] == [
        {
            "target_id": HEALTHY,
            "attempt_id": healthy_attempt_id,
            "record_sha256": stale_by_id[healthy_attempt_id]["record_sha256"],
            "appended_in_finalization_invocation": False,
        },
        {
            "target_id": TARGET,
            "attempt_id": attempt_id,
            "record_sha256": stale_by_id[attempt_id]["record_sha256"],
            "appended_in_finalization_invocation": False,
        },
    ]
    assert recovery["appended_in_finalization_invocation"] == 0
