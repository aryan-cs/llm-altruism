from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from analysis.analyze_provider_safe_v2_definitive import (
    DefinitiveAnalysisError,
    _part1,
    _validate_part1_operational_repair,
)
from experiments.misc import inference_hub_part1_panel as base
from experiments.misc.inference_hub_part1_operational_repair import (
    ATTEMPT_ARTIFACT_TYPE,
    RESPONSE_ARTIFACT_TYPE,
    SANITIZED_ARTIFACT_TYPE,
    Part1OperationalRepairError,
    _rehydrate_compatibility_controls,
    run_repair,
)


BASE_SEED = 20_260_802


def test_rehydrates_bound_compatibility_max_tokens_omitted_from_manifest_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path = tmp_path / "registry.json"
    compatibility_path = tmp_path / "compatibility.json"
    registry_path.write_text("{}\n", encoding="utf-8")
    compatibility_path.write_text("{}\n", encoding="utf-8")
    target = "provider/model"
    subject = {
        "target_id": target,
        "route": "served/model",
        "supported_controls": ["seed", "temperature"],
        "selected_profile_id": "profile_01",
        "selected_profile_request_sha256": "a" * 64,
        "candidate_index": 0,
        "model": "model",
        "upstream_provider": "provider",
    }
    manifest = {
        "input_artifacts": {
            "registry": {
                "path": str(registry_path),
                "file_sha256": base._sha256_file(registry_path),
            },
            "compatibility": {
                "path": str(compatibility_path),
                "file_sha256": base._sha256_file(compatibility_path),
            },
        }
    }
    monkeypatch.setattr(
        base,
        "_registry_targets",
        lambda registry: {
            target: {"model": "model", "upstream_provider": "provider"}
        },
    )
    monkeypatch.setattr(
        base,
        "_validated_compatibility",
        lambda compatibility, registry: {
            target: {
                "route": "served/model",
                "supported_controls": ["seed", "temperature"],
                "selected_profile_id": "profile_01",
                "selected_profile_request_sha256": "a" * 64,
                "candidate_index": 0,
                "compatibility_max_tokens": 8192,
            }
        },
    )

    hydrated = _rehydrate_compatibility_controls(manifest, {target: subject})

    assert hydrated[target]["compatibility_max_tokens"] == 8192
    assert "compatibility_max_tokens" not in subject


def _journal(path: Path, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    encoded = []
    for payload in payloads:
        row = {**payload, "previous_record_sha256": previous}
        row["record_sha256"] = base._sha256_json(row)
        previous = row["record_sha256"]
        encoded.append(json.dumps(row, sort_keys=True) + "\n")
    path.write_text("".join(encoded), encoding="utf-8")
    os.chmod(path, 0o600)
    return {
        "path": str(path.resolve()),
        "record_count": len(payloads),
        "tail_record_sha256": previous,
        "file_sha256": base._sha256_file(path),
    }


def _source(
    root: Path,
    *,
    subject_count: int = 3,
    trial_limit: int = 4,
    nulls: set[tuple[int, int]] | None = None,
    semantic_invalids: set[tuple[int, int]] | None = None,
) -> tuple[Path, dict[tuple[int, int], dict[str, Any]]]:
    nulls = nulls or set()
    semantic_invalids = semantic_invalids or set()
    private = root / "private"
    raw_dir = private / "raw_responses"
    raw_dir.mkdir(parents=True)
    os.chmod(root, 0o700)
    os.chmod(private, 0o700)
    os.chmod(raw_dir, 0o700)
    trials = base.build_draft_trials(base_seed=BASE_SEED, limit=trial_limit)
    subjects = []
    raw_refs = {}
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for subject_index in range(subject_count):
        target = f"provider/model-{subject_index}"
        subject = {
            "target_id": target,
            "upstream_provider": f"provider-{subject_index}",
            "model": f"model-{subject_index}",
            "route": f"served/model-{subject_index}",
            "supported_controls": ["seed", "temperature", "top_p"],
        }
        subjects.append(subject)
        payloads = []
        for trial_index, trial in enumerate(trials):
            is_null = (subject_index, trial_index) in nulls
            is_invalid = (subject_index, trial_index) in semantic_invalids
            body, controls = base._request_contract(subject, trial)
            response = None
            metadata: dict[str, Any]
            if is_null:
                metadata = {
                    "request_id": None,
                    "response_model": None,
                    "finish_reason": None,
                    "usage": None,
                    "reasoning_fields": {},
                    "output_field": None,
                    "response_text": None,
                    "response_text_sha256": None,
                    "parsed_action": None,
                    "format_valid": False,
                }
            else:
                response = {
                    "id": f"source-{subject_index}-{trial_index}",
                    "model": subject["route"],
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "unfinished" if is_invalid else "X",
                            },
                            "finish_reason": "length" if is_invalid else "stop",
                        }
                    ],
                }
                metadata = base._response_metadata(response)
            payloads.append(
                {
                    "schema_version": 1,
                    "artifact_type": "inference_hub_part1_raw_response",
                    "target_id": target,
                    "upstream_provider": subject["upstream_provider"],
                    "model": subject["model"],
                    "requested_route": subject["route"],
                    "model_identity_valid": False if is_null else True,
                    "trial_id": trial.trial_id,
                    "root_id": trial.root_id,
                    "game": trial.game,
                    "domain": trial.domain,
                    "counterbalance_id": trial.counterbalance_id,
                    "prompt_text": trial.prompt_text,
                    "prompt_sha256": trial.prompt_hash,
                    "request_sha256": base._sha256_json(body),
                    "controls": controls,
                    **metadata,
                    "raw_response": response,
                    "raw_response_sha256": (
                        base._sha256_json(response) if response is not None else None
                    ),
                    "failure": (
                        {"failure_code": "connection_timeout"} if is_null else None
                    ),
                }
            )
        path = raw_dir / f"model-{subject_index}.jsonl"
        raw_refs[target] = _journal(path, payloads)
        journal = base._ChainedJournal(path)
        for trial_index, row in enumerate(journal.records):
            indexed[(subject_index, trial_index)] = row
    attempt_ref = _journal(private / "attempt_ledger.jsonl", [])
    planned = subject_count * trial_limit
    manifest = {
        "schema_version": 1,
        "artifact_type": "inference_hub_part1_large_n_exploratory_panel",
        "analysis_role": "exploratory_hosted_scale_panel_only",
        "complete": len(nulls) == 0,
        "last_updated_at_utc": "2026-08-03T12:00:00Z",
        "base_seed": BASE_SEED,
        "trial_limit": trial_limit,
        "executed_trial_count_per_subject": trial_limit,
        "subject_routes": subjects,
        "judge_reservation": {
            "target_id": "judge.nemotron",
            "upstream_provider": "judge",
            "model": "nemotron",
            "route": "served/judge",
            "dispatch_permitted_in_this_runner": False,
        },
        "source_artifacts": {
            str(Path(base.__file__).resolve()): base._sha256_file(Path(base.__file__))
        },
        "summary": {
            "planned_generations": planned,
            "retained_trial_records": planned,
            "responses_received": planned - len(nulls),
            "failed_without_response": len(nulls),
            "format_valid": planned - len(nulls) - len(semantic_invalids),
            "format_invalid_retained": len(semantic_invalids),
            "response_model_identity_mismatches": 0,
        },
        "journals": {
            "attempt_ledger": attempt_ref,
            "raw_responses": raw_refs,
        },
    }
    if manifest["complete"]:
        manifest["completed_at_utc"] = "2026-08-03T12:00:00Z"
    base._seal(manifest)
    base._atomic_json(private / "manifest.json", manifest)
    return private / "manifest.json", indexed


class _RepairClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        key = body["model"]
        with self.lock:
            self.calls.append(json.loads(json.dumps(body)))
            count = self.counts.get(key, 0) + 1
            self.counts[key] = count
        if key == "served/model-0" and count == 1:
            raise RuntimeError("simulated transport failure")
        if key == "served/model-1" and count == 1:
            route = "served/wrong-identity"
            content = "X"
            finish_reason = "stop"
        elif key == "served/model-1":
            route = key
            content = "unfinished"
            finish_reason = "length"
        else:
            route = key
            content = "X"
            finish_reason = "stop"
        return {
            "id": f"repair-{len(self.calls)}",
            "model": route,
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
        }


def test_full_operational_overlay_preserves_source_and_semantic_invalids(
    tmp_path: Path,
) -> None:
    source, indexed = _source(
        tmp_path / "source",
        subject_count=75,
        trial_limit=384,
        nulls={(0, 0), (1, 1)},
        semantic_invalids={(2, 2)},
    )
    source_files = {
        path: path.read_bytes()
        for path in source.parent.rglob("*")
        if path.is_file() and path.name != ".run.lock"
    }
    client = _RepairClient()
    output = tmp_path / "repair"
    result = run_repair(
        source_manifest_path=source,
        output_dir=output,
        client=client,
        max_rounds=3,
        max_workers=2,
        initial_backoff_seconds=0,
    )

    assert result["complete"] is True
    assert result["original_manifest_mutated"] is False
    assert result["original_journals_mutated"] is False
    assert result["visible_format_invalid_rows_retried"] is False
    assert result["summary"] == {
        "planned_generations": 75 * 384,
        "retained_trial_records": 75 * 384,
        "responses_received": 75 * 384,
        "failed_without_response": 0,
        "format_valid": 75 * 384 - 2,
        "format_invalid_retained": 2,
        "response_model_identity_mismatches": 0,
        "operational_repair_eligible_originals": 2,
        "operational_repairs_succeeded": 2,
        "operational_repairs_unresolved": 0,
        "non_null_format_invalid_originals_not_retried": 1,
    }
    assert len(client.calls) == 4
    original_bodies = {}
    trials = base.build_draft_trials(base_seed=BASE_SEED, limit=384)
    for subject_index, trial_index in ((0, 0), (1, 1)):
        subject = {
            "target_id": f"provider/model-{subject_index}",
            "upstream_provider": f"provider-{subject_index}",
            "model": f"model-{subject_index}",
            "route": f"served/model-{subject_index}",
            "supported_controls": ["seed", "temperature", "top_p"],
        }
        body, _ = base._request_contract(subject, trials[trial_index])
        original_bodies[subject["route"]] = body
    assert all(call == original_bodies[call["model"]] for call in client.calls)

    for path, before in source_files.items():
        assert path.read_bytes() == before
    response_rows = [
        json.loads(line)
        for path in (output / "private/raw_responses").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(response_rows) == 3
    assert {row["artifact_type"] for row in response_rows} == {
        RESPONSE_ARTIFACT_TYPE
    }
    assert {row["original_record_sha256"] for row in response_rows} <= {
        indexed[(0, 0)]["record_sha256"],
        indexed[(1, 1)]["record_sha256"],
    }
    assert indexed[(2, 2)]["record_sha256"] not in {
        row["original_record_sha256"] for row in response_rows
    }
    attempt_rows = [
        json.loads(line)
        for line in (output / "private/attempt_ledger.jsonl").read_text().splitlines()
    ]
    assert {row["artifact_type"] for row in attempt_rows} == {
        ATTEMPT_ARTIFACT_TYPE
    }
    sanitized_text = (output / "sanitized/summary.json").read_text()
    assert "prompt_text" not in sanitized_text
    assert "raw_response" not in sanitized_text
    sanitized = json.loads(sanitized_text)
    assert sanitized["artifact_type"] == SANITIZED_ARTIFACT_TYPE
    assert sanitized["raw_text_included"] is False
    assert sanitized["effective_summary"] == result["summary"]

    no_op_client = _RepairClient()
    resumed = run_repair(
        source_manifest_path=source,
        output_dir=output,
        client=no_op_client,
        max_rounds=3,
        max_workers=2,
        initial_backoff_seconds=0,
        resume=True,
    )
    assert resumed["complete"] is True
    assert no_op_client.calls == []

    source_manifest = base._read_json(source, "source")
    effective, audit, repair_manifest_path, repair_manifest = (
        _validate_part1_operational_repair(
            source_run=source.parent.parent,
            source_manifest_path=source,
            source_manifest=source_manifest,
            repair_value=output,
        )
    )
    assert repair_manifest_path == output / "private/manifest.json"
    assert repair_manifest["complete"] is True
    assert audit == {
        "planned_generations": 75 * 384,
        "visible_original_responses": 75 * 384 - 2,
        "original_transport_nulls": 2,
        "visible_original_format_invalids": 1,
        "operational_repairs_succeeded": 2,
        "operational_repairs_unresolved": 0,
        "effective_responses": 75 * 384,
        "effective_format_invalids": 2,
        "visible_format_invalid_rows_retried": False,
        "status": "complete_exact_source_bound_transport_null_overlay_applied",
    }
    assert sum(len(rows) for rows in effective.values()) == 75 * 384
    semantic_original = indexed[(2, 2)]
    semantic_effective = next(
        row
        for row in effective["provider/model-2"]
        if row["trial_id"] == semantic_original["trial_id"]
    )
    assert semantic_effective == semantic_original
    assert semantic_effective["raw_response"] is not None
    assert semantic_effective["format_valid"] is False
    repaired_original = indexed[(0, 0)]
    repaired_effective = next(
        row
        for row in effective["provider/model-0"]
        if row["trial_id"] == repaired_original["trial_id"]
    )
    assert repaired_effective["operational_repair_overlay_applied"] is True
    assert repaired_effective["raw_response"] is not None
    assert repaired_effective["game"] == repaired_original["game"]
    assert repaired_effective["domain"] == repaired_original["domain"]
    assert repaired_effective["counterbalance_id"] == repaired_original[
        "counterbalance_id"
    ]

    models, figure = _part1(
        source.parent.parent,
        source_manifest,
        effective_journals=effective,
    )
    assert len(models) == 75
    assert len(figure) == 75 * 12
    assert sum(row["scheduled_units"] for row in models) == 75 * 384
    assert sum(row["first_attempt_invalid_count"] for row in models) == 2

    # The public analyzer binding is exact: even a self-hash-valid manifest
    # pointing at a different source evidence hash must fail closed.
    repair_manifest_bytes = repair_manifest_path.read_bytes()
    changed_manifest = base._read_json(repair_manifest_path, "repair")
    changed_manifest["source_manifest"]["evidence_sha256"] = "0" * 64
    base._seal(changed_manifest)
    base._atomic_json(repair_manifest_path, changed_manifest)
    with pytest.raises(DefinitiveAnalysisError, match="exact-source binding"):
        _validate_part1_operational_repair(
            source_run=source.parent.parent,
            source_manifest_path=source,
            source_manifest=source_manifest,
            repair_value=output,
        )
    repair_manifest_path.write_bytes(repair_manifest_bytes)
    os.chmod(repair_manifest_path, 0o600)

    # A rehashed raw journal still cannot drift from the original route/trial
    # binding. Restore both files after the mutation so the fixture remains
    # usable and source immutability can be checked again below.
    repair_raw_path = next((output / "private/raw_responses").glob("*.jsonl"))
    repair_raw_bytes = repair_raw_path.read_bytes()
    repair_raw_rows = [
        json.loads(line) for line in repair_raw_path.read_text().splitlines()
    ]
    repair_raw_payloads = [
        {
            key: value
            for key, value in row.items()
            if key not in {"previous_record_sha256", "record_sha256"}
        }
        for row in repair_raw_rows
    ]
    repair_raw_payloads[0]["original_record_sha256"] = "f" * 64
    changed_raw_ref = _journal(repair_raw_path, repair_raw_payloads)
    changed_manifest = base._read_json(repair_manifest_path, "repair")
    changed_manifest["journals"]["raw_responses"][
        repair_raw_payloads[0]["target_id"]
    ] = changed_raw_ref
    base._seal(changed_manifest)
    base._atomic_json(repair_manifest_path, changed_manifest)
    with pytest.raises(DefinitiveAnalysisError, match="payload binding failed"):
        _validate_part1_operational_repair(
            source_run=source.parent.parent,
            source_manifest_path=source,
            source_manifest=source_manifest,
            repair_value=output,
        )
    repair_raw_path.write_bytes(repair_raw_bytes)
    os.chmod(repair_raw_path, 0o600)
    repair_manifest_path.write_bytes(repair_manifest_bytes)
    os.chmod(repair_manifest_path, 0o600)

    for path, before in source_files.items():
        assert path.read_bytes() == before


def test_refuses_nonterminal_source_without_dispatch(tmp_path: Path) -> None:
    source, _ = _source(
        tmp_path / "source",
        nulls={(0, 0)},
    )
    manifest = base._read_json(source, "source")
    manifest["summary"] = {}
    base._seal(manifest)
    base._atomic_json(source, manifest)
    client = _RepairClient()
    with pytest.raises(Part1OperationalRepairError, match="terminalized"):
        run_repair(
            source_manifest_path=source,
            output_dir=tmp_path / "repair",
            client=client,
            max_rounds=2,
            initial_backoff_seconds=0,
        )
    assert client.calls == []


def test_refuses_active_source_writer_without_dispatch(tmp_path: Path) -> None:
    source, _ = _source(tmp_path / "source", nulls={(0, 0)})
    lock = base._acquire_run_lock(source.parent)
    try:
        client = _RepairClient()
        with pytest.raises(Part1OperationalRepairError, match="still active"):
            run_repair(
                source_manifest_path=source,
                output_dir=tmp_path / "repair",
                client=client,
                max_rounds=2,
                initial_backoff_seconds=0,
            )
        assert client.calls == []
    finally:
        import fcntl

        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
