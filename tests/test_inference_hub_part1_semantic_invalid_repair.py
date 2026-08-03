from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest

from experiments.misc import inference_hub_part1_panel as base
from experiments.misc.inference_hub_part1_semantic_invalid_repair import (
    DEADLINE_LAUNCHER,
    MAIN_LAUNCHER,
    PROVIDER_SAFE_V2,
    Part1SemanticInvalidRepairError,
    run_repair,
)


BASE_SEED = 20_260_802


def _journal(
    path: Path,
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    rows = []
    encoded = []
    for payload in payloads:
        row = {**payload, "previous_record_sha256": previous}
        row["record_sha256"] = base._sha256_json(row)
        previous = row["record_sha256"]
        rows.append(row)
        encoded.append(
            json.dumps(
                row, sort_keys=True, ensure_ascii=False
            ) + "\n"
        )
    path.write_text("".join(encoded), encoding="utf-8")
    os.chmod(path, 0o600)
    return ({
        "path": str(path.resolve()),
        "record_count": len(rows),
        "tail_record_sha256": previous,
        "file_sha256": base._sha256_file(path),
    }, rows)


def _write_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    base._seal(manifest)
    base._atomic_json(path, manifest)


def _shared_policy(*, deadline: bool = False) -> dict[str, Any]:
    shared = {
        "schema_version": 2,
        "algorithm":
            "cross_process_provider_aware_leaky_bucket_with_"
            "leases_all_http_5xx_full_throttle_cooldown",
        "global_concurrency": 24 if deadline else 12,
        "provider_concurrency": 4 if deadline else 2,
        "global_requests_per_second": 12.0 if deadline else 8.0,
        "provider_requests_per_second": 2.5 if deadline else 1.5,
        "lease_seconds": 900.0,
        "poll_seconds": 0.05,
        "throttle_cooldown_seconds": 30.0,
        "transient_cooldown_seconds": 5.0,
    }
    shared["policy_sha256"] = base._sha256_json(shared)
    return shared


def _source(
    root: Path,
    *,
    subject_count: int,
    trial_limit: int,
    invalids: Mapping[tuple[int, int], str],
    deadline: bool = False,
) -> tuple[Path, dict[tuple[int, int], dict[str, Any]]]:
    private = root / "private"
    private.mkdir(parents=True)
    os.chmod(private, 0o700)
    trials = base.build_draft_trials(
        base_seed=BASE_SEED, limit=trial_limit
    )
    subjects = []
    references = {}
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for subject_index in range(subject_count):
        target = f"provider/model-{subject_index:03d}"
        controls = (
            ["temperature", "top_p", "structured_response"]
            if subject_index == 1
            else [
                "seed",
                "temperature",
                "top_p",
                "structured_response",
            ]
        )
        subject = {
            "target_id": target,
            "upstream_provider": f"provider-{subject_index:03d}",
            "model": f"model-{subject_index:03d}",
            "route": f"route/provider/model-{subject_index:03d}",
            "supported_controls": controls,
        }
        subjects.append(subject)
        payloads = []
        for trial_index, trial in enumerate(trials):
            finish_reason = invalids.get(
                (subject_index, trial_index)
            )
            invalid = finish_reason is not None
            original_body, _ = base._request_contract(
                subject, trial
            )
            raw_response = {
                "id": "source-response",
                "model": subject["route"],
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": (
                            None
                            if finish_reason == "content_filter"
                            else (
                                "unfinished"
                                if finish_reason == "length"
                                else "X"
                            )
                        ),
                    },
                    "finish_reason": (
                        finish_reason
                        if invalid else "stop"
                    ),
                }],
            }
            payloads.append({
                "schema_version": 1,
                "artifact_type":
                    "inference_hub_part1_raw_response",
                "target_id": target,
                "upstream_provider":
                    subject["upstream_provider"],
                "model": subject["model"],
                "trial_id": trial.trial_id,
                "root_id": trial.root_id,
                "prompt_sha256": trial.prompt_hash,
                "prompt_text": trial.prompt_text,
                "requested_route": subject["route"],
                "request_sha256":
                    base._sha256_json(original_body),
                "raw_response": raw_response,
                "raw_response_sha256":
                    base._sha256_json(raw_response),
                "model_identity_valid": True,
                "format_valid": not invalid,
                "parsed_action": None if invalid else "X",
                "finish_reason":
                    finish_reason if invalid else "stop",
            })
        reference, rows = _journal(
            private / "raw" / f"{subject_index:03d}.jsonl",
            payloads,
        )
        references[target] = reference
        for trial_index, row in enumerate(rows):
            indexed[(subject_index, trial_index)] = row
    attempt_ref, _ = _journal(
        private / "attempt_ledger.jsonl", []
    )
    manifest = {
        "schema_version": 1,
        "artifact_type":
            "inference_hub_part1_large_n_exploratory_panel",
        "complete": True,
        "completed_at_utc": "2026-08-03T12:00:00Z",
        "base_seed": BASE_SEED,
        "trial_limit": trial_limit,
        "subject_routes": subjects,
        "judge_reservation": {
            "target_id": "judge.nemotron",
            "upstream_provider": "judge",
            "model": "nemotron",
            "route": "judge/nemotron",
            "dispatch_permitted_in_this_runner": False,
        },
        "execution_contract": {
            "shared_rate_limit": _shared_policy(deadline=deadline)
        },
        "source_artifacts": {
            str(PROVIDER_SAFE_V2.resolve()):
                base._sha256_file(PROVIDER_SAFE_V2),
            str(
                DEADLINE_LAUNCHER.resolve()
                if deadline else MAIN_LAUNCHER.resolve()
            ): base._sha256_file(
                DEADLINE_LAUNCHER if deadline else MAIN_LAUNCHER
            ),
        },
        "summary": {
            "retained_trial_records":
                subject_count * trial_limit,
        },
        "journals": {
            "attempt_ledger": attempt_ref,
            "raw_responses": references,
        },
    }
    path = private / "manifest.json"
    _write_manifest(path, manifest)
    return path, indexed


def test_accepts_definitive_deadline_source_contract(
    tmp_path: Path,
) -> None:
    source, _ = _source(
        tmp_path / "deadline-source",
        subject_count=1,
        trial_limit=2,
        invalids={(0, 0): "length"},
        deadline=True,
    )
    manifest = run_repair(
        source_manifest_path=source,
        output_dir=tmp_path / "deadline-output",
        client=_Client(invalid_first_round=False),
        max_rounds=2,
        round_interval_seconds=0,
    )
    assert manifest["complete"] is True
    assert manifest["summary"][
        "source_first_attempt_invalid_count"
    ] == 1
    assert manifest["summary"][
        "repaired_valid_separate_count"
    ] == 1
    assert manifest["summary"][
        "unrepaired_after_bounded_rounds_count"
    ] == 0


class _Client:
    def __init__(
        self,
        *,
        invalid_first_round: bool = True,
        wrong_identity: bool = False,
    ) -> None:
        self.invalid_first_round = invalid_first_round
        self.wrong_identity = wrong_identity
        self.calls: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def post(
        self, path: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        assert path == "/chat/completions"
        prompt = body["messages"][1]["content"]
        with self.lock:
            self.calls.append(body)
            count = self.counts.get(prompt, 0) + 1
            self.counts[prompt] = count
        invalid = self.invalid_first_round and count == 1
        return {
            "id": f"repair-{len(self.calls)}",
            "model": (
                "wrong/model"
                if self.wrong_identity else body["model"]
            ),
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": (
                        "unfinished" if invalid else "X"
                    ),
                },
                "finish_reason":
                    "length" if invalid else "stop",
            }],
        }


@pytest.fixture(scope="module")
def production_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[tuple[int, int], dict[str, Any]]]:
    return _source(
        tmp_path_factory.mktemp("repair-production"),
        subject_count=75,
        trial_limit=384,
        invalids={(0, 0): "content_filter", (1, 1): "length"},
    )


def test_production_shaped_content_filter_and_length_repairs_are_separate(
    production_source: tuple[
        Path, dict[tuple[int, int], dict[str, Any]]
    ],
    tmp_path: Path,
) -> None:
    source_path, source_rows = production_source
    source_before = base._sha256_file(source_path)
    client = _Client()
    result = run_repair(
        source_manifest_path=source_path,
        output_dir=tmp_path / "repair",
        client=client,
        max_rounds=3,
        max_workers=2,
        round_interval_seconds=0,
        sleep_fn=lambda _: None,
    )

    assert result["complete"] is True
    assert result["summary"] == {
        "source_scheduled_unit_count": 75 * 384,
        "source_first_attempt_invalid_count": 2,
        "repaired_valid_separate_count": 2,
        "unrepaired_after_bounded_rounds_count": 0,
        "primary_denominator": 75 * 384,
        "primary_denominator_changed": False,
    }
    assert result["judge_dispatched"] is False
    assert result["primary_records_mutated"] is False
    assert base._sha256_file(source_path) == source_before
    assert len(client.calls) == 4

    payload = json.loads(
        (
            tmp_path / "repair/sanitized/repair_outcomes.json"
        ).read_text()
    )
    assert payload["raw_text_included"] is False
    assert payload["repaired_estimates_separate_only"] is True
    assert {
        row["original_finish_reason"]
        for row in payload["rows"]
    } == {"content_filter", "length"}
    assert all(
        row["successful_round"] == 2
        and row["primary_denominator_changed"] is False
        for row in payload["rows"]
    )
    original_hashes = {
        source_rows[(0, 0)]["record_sha256"],
        source_rows[(1, 1)]["record_sha256"],
    }
    assert {
        row["original_record_sha256"]
        for row in payload["rows"]
    } == original_hashes

    calls_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for call in client.calls:
        calls_by_prompt.setdefault(
            call["messages"][1]["content"], []
        ).append(call)
    seeded = calls_by_prompt[
        source_rows[(0, 0)]["prompt_text"]
    ]
    unseeded = calls_by_prompt[
        source_rows[(1, 1)]["prompt_text"]
    ]
    assert len({call["seed"] for call in seeded}) == 2
    assert all("seed" not in call for call in unseeded)


def test_raw_fsync_crash_is_recovered_without_redispatch(
    production_source: tuple[
        Path, dict[tuple[int, int], dict[str, Any]]
    ],
    tmp_path: Path,
) -> None:
    source_path, _ = production_source
    output = tmp_path / "crash-repair"
    client = _Client(invalid_first_round=False)
    crashed = False

    def crash_once(_: Mapping[str, Any]) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise KeyboardInterrupt("simulated crash after raw fsync")

    with pytest.raises(KeyboardInterrupt):
        run_repair(
            source_manifest_path=source_path,
            output_dir=output,
            client=client,
            max_rounds=2,
            max_workers=1,
            round_interval_seconds=0,
            sleep_fn=lambda _: None,
            after_raw_hook=crash_once,
        )
    call_count = len(client.calls)
    resumed = run_repair(
        source_manifest_path=source_path,
        output_dir=output,
        client=client,
        max_rounds=2,
        max_workers=1,
        round_interval_seconds=0,
        sleep_fn=lambda _: None,
        resume=True,
    )
    assert resumed["complete"] is True
    assert len(client.calls) == call_count
    ledger = base._ChainedJournal(
        output / "private/attempt_ledger.jsonl"
    )
    assert any(
        row.get("recovered_after_raw_fsync") is True
        for row in ledger.records
    )


def test_zero_invalid_is_keyless_noop(
    tmp_path: Path,
) -> None:
    source, _ = _source(
        tmp_path / "zero-source",
        subject_count=75,
        trial_limit=384,
        invalids={},
    )
    result = run_repair(
        source_manifest_path=source,
        output_dir=tmp_path / "zero-output",
        client=None,
        max_rounds=3,
        round_interval_seconds=0,
    )
    assert result["summary"][
        "source_first_attempt_invalid_count"
    ] == 0
    assert result["summary"][
        "repaired_valid_separate_count"
    ] == 0
    payload = json.loads(
        (
            tmp_path
            / "zero-output/sanitized/repair_outcomes.json"
        ).read_text()
    )
    assert payload["rows"] == []
    assert base._ChainedJournal(
        tmp_path
        / "zero-output/private/attempt_ledger.jsonl"
    ).records == []


def test_fail_closed_on_source_state_policy_prompt_and_identity(
    tmp_path: Path,
) -> None:
    source, _ = _source(
        tmp_path / "small-source",
        subject_count=1,
        trial_limit=2,
        invalids={(0, 0): "length"},
    )

    incomplete = json.loads(source.read_text())
    incomplete["complete"] = False
    incomplete_path = (
        tmp_path / "incomplete/private/manifest.json"
    )
    incomplete_path.parent.mkdir(parents=True)
    _write_manifest(incomplete_path, incomplete)
    with pytest.raises(
        Part1SemanticInvalidRepairError,
        match="not COMPLETE",
    ):
        run_repair(
            source_manifest_path=incomplete_path,
            output_dir=tmp_path / "incomplete-output",
            client=_Client(),
        )
    assert not (tmp_path / "incomplete-output").exists()

    wrong_policy = json.loads(source.read_text())
    wrong_policy["execution_contract"][
        "shared_rate_limit"
    ]["provider_concurrency"] = 3
    wrong_policy["execution_contract"][
        "shared_rate_limit"
    ]["policy_sha256"] = base._sha256_json({
        key: value
        for key, value in wrong_policy[
            "execution_contract"
        ]["shared_rate_limit"].items()
        if key != "policy_sha256"
    })
    wrong_policy_path = (
        tmp_path / "wrong-policy/private/manifest.json"
    )
    wrong_policy_path.parent.mkdir(parents=True)
    _write_manifest(wrong_policy_path, wrong_policy)
    with pytest.raises(
        Part1SemanticInvalidRepairError,
        match="policy is wrong",
    ):
        run_repair(
            source_manifest_path=wrong_policy_path,
            output_dir=tmp_path / "policy-output",
            client=_Client(),
        )

    manifest = json.loads(source.read_text())
    raw_ref = next(iter(
        manifest["journals"]["raw_responses"].values()
    ))
    raw_path = Path(raw_ref["path"])
    payloads = [
        {
            key: value
            for key, value in json.loads(line).items()
            if key not in {
                "record_sha256",
                "previous_record_sha256",
            }
        }
        for line in raw_path.read_text().splitlines()
    ]
    payloads[0]["prompt_text"] += " drift"
    new_ref, _ = _journal(raw_path, payloads)
    target = next(iter(
        manifest["journals"]["raw_responses"]
    ))
    manifest["journals"]["raw_responses"][
        target
    ] = new_ref
    _write_manifest(source, manifest)
    with pytest.raises(
        Part1SemanticInvalidRepairError,
        match="prompt/request/identity drifted",
    ):
        run_repair(
            source_manifest_path=source,
            output_dir=tmp_path / "drift-output",
            client=_Client(),
        )
    assert not (tmp_path / "drift-output").exists()

    identity_source, _ = _source(
        tmp_path / "identity-source",
        subject_count=1,
        trial_limit=1,
        invalids={(0, 0): "content_filter"},
    )
    with pytest.raises(
        Part1SemanticInvalidRepairError,
        match="identity mismatch",
    ):
        run_repair(
            source_manifest_path=identity_source,
            output_dir=tmp_path / "identity-output",
            client=_Client(
                invalid_first_round=False,
                wrong_identity=True,
            ),
            max_rounds=1,
            round_interval_seconds=0,
        )
    assert not (
        tmp_path
        / "identity-output/sanitized/repair_outcomes.json"
    ).exists()
