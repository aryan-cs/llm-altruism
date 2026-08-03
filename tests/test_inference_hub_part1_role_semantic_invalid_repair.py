from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest

from experiments.misc import inference_hub_part1_panel as base
from experiments.misc import inference_hub_part1_role_calibration_v1 as role
from experiments.misc.inference_hub_part1_role_semantic_invalid_repair import (
    EXPLORATORY_LAUNCHER,
    PROVIDER_SAFE_V2,
    ROLE_RUNNER,
    RoleSemanticInvalidRepairError,
    _schedule_hash,
    run_repair,
)


def _journal(
    path: Path,
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    rows = []
    encoded = []
    for payload in payloads:
        row = {
            **payload,
            "previous_record_sha256": previous,
        }
        row["record_sha256"] = (
            base._sha256_json(row)
        )
        previous = row["record_sha256"]
        rows.append(row)
        encoded.append(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
            ) + "\n"
        )
    path.write_text(
        "".join(encoded), encoding="utf-8"
    )
    os.chmod(path, 0o600)
    return ({
        "path": str(path.resolve()),
        "record_count": len(rows),
        "tail_record_sha256": previous,
        "file_sha256": base._sha256_file(path),
    }, rows)


def _write_manifest(
    path: Path, manifest: dict[str, Any]
) -> None:
    base._seal(manifest)
    base._atomic_json(path, manifest)


def _policy() -> dict[str, Any]:
    policy = {
        "schema_version": 2,
        "algorithm":
            "cross_process_provider_aware_leaky_"
            "bucket_with_leases_all_http_5xx_full_"
            "throttle_cooldown",
        "global_concurrency": 12,
        "provider_concurrency": 3,
        "global_requests_per_second": 8.0,
        "provider_requests_per_second": 2.0,
        "lease_seconds": 900.0,
        "poll_seconds": 0.05,
        "throttle_cooldown_seconds": 30.0,
        "transient_cooldown_seconds": 5.0,
    }
    policy["policy_sha256"] = (
        base._sha256_json(policy)
    )
    return policy


def _source(
    root: Path,
    *,
    invalids: Mapping[tuple[int, int], str],
) -> tuple[
    Path, dict[tuple[int, int], dict[str, Any]]
]:
    private = root / "private"
    private.mkdir(parents=True)
    os.chmod(private, 0o700)
    config = role.load_frozen_config()
    trials = role.build_frozen_trials(config)
    subjects = []
    raw_refs = {}
    indexed = {}
    for subject_index, config_subject in enumerate(
        config["subjects"]
    ):
        controls = (
            [
                "temperature",
                "top_p",
                "structured_response",
            ]
            if subject_index == 1
            else [
                "seed",
                "temperature",
                "top_p",
                "structured_response",
            ]
        )
        subject = {
            "target_id":
                config_subject["target_id"],
            "upstream_provider":
                config_subject["developer_family"],
            "model":
                f"model-{subject_index}",
            "route":
                f"route/role/model-{subject_index}",
            "supported_controls": controls,
        }
        subjects.append(subject)
        payloads = []
        for trial_index, trial in enumerate(trials):
            finish = invalids.get(
                (subject_index, trial_index)
            )
            invalid = finish is not None
            body, controls_used = (
                base._request_contract(subject, trial)
            )
            content = (
                None
                if finish == "content_filter"
                else (
                    "unfinished"
                    if finish == "length"
                    else "X"
                )
            )
            response = {
                "id": "source-role-response",
                "model": subject["route"],
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason":
                        finish if invalid else "stop",
                }],
            }
            payloads.append({
                "schema_version": 1,
                "artifact_type":
                    "inference_hub_part1_role_"
                    "calibration_raw_response_v1",
                "target_id": subject["target_id"],
                "upstream_provider":
                    subject["upstream_provider"],
                "model": subject["model"],
                "requested_route":
                    subject["route"],
                "response_model":
                    subject["route"],
                "model_identity_valid": True,
                "trial_id": trial.trial_id,
                "root_id": trial.root_id,
                "game": trial.game,
                "domain": trial.domain,
                "frame_id": trial.frame_id,
                "generation_block":
                    trial.generation_block,
                "counterbalance_id":
                    trial.counterbalance_id,
                "prompt_text": trial.prompt_text,
                "prompt_sha256":
                    trial.prompt_hash,
                "request_sha256":
                    base._sha256_json(body),
                "controls": controls_used,
                "finish_reason":
                    finish if invalid else "stop",
                "parsed_action":
                    None if invalid else "X",
                "format_valid": not invalid,
                "welfare_preserving":
                    None if invalid else True,
                "raw_response": response,
                "raw_response_sha256":
                    base._sha256_json(response),
            })
        ref, rows = _journal(
            private / "raw" / f"{subject_index}.jsonl",
            payloads,
        )
        raw_refs[subject["target_id"]] = ref
        for trial_index, row in enumerate(rows):
            indexed[(subject_index, trial_index)] = row
    ledger_ref, _ = _journal(
        private / "attempt_ledger.jsonl", []
    )
    planned = len(subjects) * len(trials)
    manifest = {
        "schema_version": 1,
        "artifact_type":
            "inference_hub_part1_role_"
            "calibration_private_v1",
        "complete": True,
        "completed_at_utc":
            "2026-08-03T12:00:00Z",
        "config_file_sha256":
            role.CONFIG_FILE_SHA256,
        "root_count": role.EXPECTED_ROOT_COUNT,
        "frames": list(role.ROLE_FRAME_IDS),
        "frames_pooled": False,
        "generation_blocks":
            list(role.ROLE_BLOCKS),
        "trials_per_subject":
            role.EXPECTED_TRIALS_PER_SUBJECT,
        "schedule_sha256":
            _schedule_hash(trials),
        "subject_routes": subjects,
        "judge_reservation": {
            "target_id": "judge.nemotron",
            "upstream_provider": "judge",
            "model": "nemotron",
            "route": "judge/nemotron",
            "dispatch_permitted_in_this_runner":
                False,
        },
        "execution_contract": {
            "shared_rate_limit": _policy()
        },
        "source_artifacts": {
            str(PROVIDER_SAFE_V2.resolve()):
                base._sha256_file(
                    PROVIDER_SAFE_V2
                ),
            str(EXPLORATORY_LAUNCHER.resolve()):
                base._sha256_file(
                    EXPLORATORY_LAUNCHER
                ),
            str(ROLE_RUNNER):
                base._sha256_file(ROLE_RUNNER),
            str(role.CONFIG_PATH.resolve()):
                base._sha256_file(
                    role.CONFIG_PATH
                ),
        },
        "summary": {
            "retained_trial_records": planned,
            "failed_without_response": 0,
            "response_model_identity_mismatches": 0,
        },
        "journals": {
            "attempt_ledger": ledger_ref,
            "raw_responses": raw_refs,
        },
    }
    path = private / "manifest.json"
    _write_manifest(path, manifest)
    return path, indexed


class _Client:
    def __init__(
        self,
        *,
        persistent_prompt: str | None = None,
        immediately_valid: bool = False,
        wrong_identity: bool = False,
    ) -> None:
        self.persistent_prompt = persistent_prompt
        self.immediately_valid = immediately_valid
        self.wrong_identity = wrong_identity
        self.calls: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()

    def post(
        self,
        path: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        assert path == "/chat/completions"
        prompt = body["messages"][1]["content"]
        with self.lock:
            self.calls.append(body)
            count = self.counts.get(prompt, 0) + 1
            self.counts[prompt] = count
        invalid = (
            prompt == self.persistent_prompt
            or (
                not self.immediately_valid
                and count == 1
            )
        )
        return {
            "id": f"role-repair-{len(self.calls)}",
            "model": (
                "wrong/model"
                if self.wrong_identity
                else body["model"]
            ),
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content":
                        "unfinished" if invalid else "X",
                },
                "finish_reason":
                    "length" if invalid else "stop",
            }],
        }


@pytest.fixture(scope="module")
def production_source(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    Path, dict[tuple[int, int], dict[str, Any]]
]:
    return _source(
        tmp_path_factory.mktemp(
            "role-repair-production"
        ),
        invalids={
            (0, 0): "content_filter",
            (1, 1): "length",
        },
    )


def test_production_shaped_repairs_and_final_invalid_persistence(
    production_source: tuple[
        Path, dict[tuple[int, int], dict[str, Any]]
    ],
    tmp_path: Path,
) -> None:
    source, source_rows = production_source
    source_hash = base._sha256_file(source)
    persistent_prompt = source_rows[
        (1, 1)
    ]["prompt_text"]
    client = _Client(
        persistent_prompt=persistent_prompt
    )
    result = run_repair(
        source_manifest_path=source,
        output_dir=tmp_path / "repair",
        client=client,
        max_rounds=2,
        max_workers=2,
        round_interval_seconds=0,
        sleep_fn=lambda _: None,
    )

    assert result["complete"] is True
    assert result["summary"] == {
        "source_scheduled_draw_count":
            6 * 96 * 3 * 4,
        "source_first_response_invalid_count": 2,
        "repaired_valid_separate_count": 1,
        "unrepaired_after_bounded_rounds_count":
            1,
        "primary_denominator":
            6 * 96 * 3 * 4,
        "primary_denominator_changed": False,
        "frames_pooled": False,
        "models_pooled": False,
    }
    assert base._sha256_file(source) == source_hash
    assert len(client.calls) == 4
    payload = json.loads(
        (
            tmp_path
            / "repair/sanitized/repair_outcomes.json"
        ).read_text()
    )
    assert payload["raw_text_included"] is False
    assert payload["frames_pooled"] is False
    assert payload["models_pooled"] is False
    assert {
        row["original_finish_reason"]
        for row in payload["rows"]
    } == {"content_filter", "length"}
    repaired = next(
        row for row in payload["rows"]
        if row["repaired_format_valid"]
    )
    persistent = next(
        row for row in payload["rows"]
        if not row["repaired_format_valid"]
    )
    assert repaired["successful_round"] == 2
    assert (
        persistent["repair_status"]
        == "unrepaired_after_bounded_rounds"
    )
    assert persistent["rounds_reserved"] == 2
    assert all(
        row["primary_denominator_changed"]
        is False
        and row["frame_pooled"] is False
        for row in payload["rows"]
    )
    calls_by_prompt: dict[
        str, list[dict[str, Any]]
    ] = {}
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
    assert len({
        call["seed"] for call in seeded
    }) == 2
    assert all(
        "seed" not in call for call in unseeded
    )


def test_role_raw_fsync_crash_recovers_without_redispatch(
    production_source: tuple[
        Path, dict[tuple[int, int], dict[str, Any]]
    ],
    tmp_path: Path,
) -> None:
    source, _ = production_source
    output = tmp_path / "crash"
    client = _Client(immediately_valid=True)
    crashed = False

    def crash_once(_: Mapping[str, Any]) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise KeyboardInterrupt(
                "crash after role raw fsync"
            )

    with pytest.raises(KeyboardInterrupt):
        run_repair(
            source_manifest_path=source,
            output_dir=output,
            client=client,
            max_rounds=2,
            max_workers=1,
            round_interval_seconds=0,
            sleep_fn=lambda _: None,
            after_raw_hook=crash_once,
        )
    calls = len(client.calls)
    result = run_repair(
        source_manifest_path=source,
        output_dir=output,
        client=client,
        max_rounds=2,
        max_workers=1,
        round_interval_seconds=0,
        sleep_fn=lambda _: None,
        resume=True,
    )
    assert result["complete"] is True
    assert len(client.calls) == calls
    ledger = base._ChainedJournal(
        output / "private/attempt_ledger.jsonl"
    )
    assert any(
        row.get("recovered_after_raw_fsync")
        is True
        for row in ledger.records
    )


def test_role_zero_invalid_is_keyless_noop(
    tmp_path: Path,
) -> None:
    source, _ = _source(
        tmp_path / "zero-source",
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
        "source_scheduled_draw_count"
    ] == 6 * 96 * 3 * 4
    assert result["summary"][
        "source_first_response_invalid_count"
    ] == 0
    payload = json.loads(
        (
            tmp_path
            / "zero-output/sanitized/"
            "repair_outcomes.json"
        ).read_text()
    )
    assert payload["rows"] == []


def test_role_fail_closed_on_prompt_policy_source_and_identity(
    tmp_path: Path,
) -> None:
    source, _ = _source(
        tmp_path / "tamper-source",
        invalids={(0, 0): "length"},
    )
    manifest = json.loads(source.read_text())

    incomplete = json.loads(
        json.dumps(manifest)
    )
    incomplete["complete"] = False
    incomplete_path = (
        tmp_path / "incomplete/private/manifest.json"
    )
    incomplete_path.parent.mkdir(parents=True)
    _write_manifest(
        incomplete_path, incomplete
    )
    with pytest.raises(
        RoleSemanticInvalidRepairError,
        match="not COMPLETE",
    ):
        run_repair(
            source_manifest_path=incomplete_path,
            output_dir=
                tmp_path / "incomplete-output",
            client=_Client(),
        )
    assert not (
        tmp_path / "incomplete-output"
    ).exists()

    wrong_policy = json.loads(
        json.dumps(manifest)
    )
    shared = wrong_policy[
        "execution_contract"
    ]["shared_rate_limit"]
    shared["provider_concurrency"] = 2
    shared["policy_sha256"] = (
        base._sha256_json({
            key: value
            for key, value in shared.items()
            if key != "policy_sha256"
        })
    )
    wrong_policy_path = (
        tmp_path / "wrong-policy/private/manifest.json"
    )
    wrong_policy_path.parent.mkdir(parents=True)
    _write_manifest(
        wrong_policy_path, wrong_policy
    )
    with pytest.raises(
        RoleSemanticInvalidRepairError,
        match="policy is wrong",
    ):
        run_repair(
            source_manifest_path=wrong_policy_path,
            output_dir=tmp_path / "policy-output",
            client=_Client(),
        )

    wrong_source = json.loads(
        json.dumps(manifest)
    )
    launcher_key = next(
        key
        for key in wrong_source[
            "source_artifacts"
        ]
        if Path(key).name
        == EXPLORATORY_LAUNCHER.name
    )
    wrong_source["source_artifacts"][
        launcher_key
    ] = "0" * 64
    wrong_source_path = (
        tmp_path / "wrong-source/private/manifest.json"
    )
    wrong_source_path.parent.mkdir(parents=True)
    _write_manifest(
        wrong_source_path, wrong_source
    )
    with pytest.raises(
        RoleSemanticInvalidRepairError,
        match="Source binding failed",
    ):
        run_repair(
            source_manifest_path=wrong_source_path,
            output_dir=tmp_path / "source-output",
            client=_Client(),
        )

    raw_refs = manifest["journals"][
        "raw_responses"
    ]
    target, raw_ref = next(iter(raw_refs.items()))
    raw_path = Path(raw_ref["path"])
    payloads = [
        {
            key: value
            for key, value
            in json.loads(line).items()
            if key not in {
                "record_sha256",
                "previous_record_sha256",
            }
        }
        for line in raw_path.read_text().splitlines()
    ]
    payloads[0]["prompt_text"] += " drift"
    new_ref, _ = _journal(raw_path, payloads)
    manifest["journals"]["raw_responses"][
        target
    ] = new_ref
    _write_manifest(source, manifest)
    with pytest.raises(
        (
            RoleSemanticInvalidRepairError,
            role.RoleCalibrationError,
        ),
        match="prompt/frame/root/block drifted",
    ):
        run_repair(
            source_manifest_path=source,
            output_dir=tmp_path / "prompt-output",
            client=_Client(),
        )
    assert not (
        tmp_path / "prompt-output"
    ).exists()

    identity_source, _ = _source(
        tmp_path / "identity-source",
        invalids={(0, 0): "content_filter"},
    )
    with pytest.raises(
        RoleSemanticInvalidRepairError,
        match="identity mismatch",
    ):
        run_repair(
            source_manifest_path=identity_source,
            output_dir=tmp_path / "identity-output",
            client=_Client(
                immediately_valid=True,
                wrong_identity=True,
            ),
            max_rounds=1,
            round_interval_seconds=0,
        )
    assert not (
        tmp_path
        / "identity-output/sanitized/"
        "repair_outcomes.json"
    ).exists()
