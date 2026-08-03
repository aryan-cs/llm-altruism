import hashlib
import json
import stat
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from experiments.misc import inference_hub_part1_role_calibration_v1 as role_runner
from experiments.misc.inference_hub_part1_panel import (
    InferenceHubPart1PanelError,
    _read_json,
    _sha256_json,
    select_routes,
)
from experiments.misc.inference_hub_part1_role_calibration_v1 import (
    CONFIG_FILE_SHA256,
    CONFIG_PATH,
    EXPECTED_ROOT_COUNT,
    EXPECTED_SUBJECT_COUNT,
    EXPECTED_TRIALS_PER_SUBJECT,
    RoleCalibrationError,
    _recover_ledger,
    _RunLockedJournal,
    _validate_operational_resume_contract,
    build_frozen_trials,
    load_frozen_config,
    run_calibration,
)
from experiments.misc.inference_hub_provider_safe_v2 import provider_round_robin
from experiments.part1.confirmatory_design import DOMAINS, GAMES


ENDPOINT = "https://inference-api.nvidia.com/v1"
JUDGE_ID = "judge.nvidia-evals-nemotron-3-30b-a3b"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _registry_and_compatibility(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    config = load_frozen_config()
    target_ids = [row["target_id"] for row in config["subjects"]]
    targets = []
    compatibility_targets = []
    for index, target_id in enumerate([*target_ids, JUDGE_ID]):
        upstream = "judge-provider" if target_id == JUDGE_ID else f"developer-{index}"
        model = "judge-model" if target_id == JUDGE_ID else f"model-{index}"
        route = f"served/{model}"
        targets.append(
            {
                "id": target_id,
                "provider": "inference_hub",
                "upstream_provider": upstream,
                "model": model,
                "route": f"display/{model}",
            }
        )
        profile = {
            "attempt_id": f"probe-{index}",
            "controls": ["seed", "temperature", "top_p"],
            "profile_id": f"profile-{index}",
            "request_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
            "resumed_from_ledger": False,
            "status": "passed",
            "validation_source": "execution_profile_probe",
        }
        compatibility_targets.append(
            {
                "target_id": target_id,
                "model": model,
                "candidate_count": 1,
                "frozen_candidate_order": [route],
                "selected_execution_candidate": route,
                "selected_execution_profile": {**profile, "route": route},
                "selection_basis": "first_execution_compatible_in_reconciliation_frozen_order",
                "status": "execution_candidate_selected",
                "candidates": [
                    {
                        "route": route,
                        "candidate_index": 0,
                        "max_tokens": 64,
                        "execution_compatible": True,
                        "selected_execution_profile": profile,
                    }
                ],
            }
        )
    registry = {"schema_version": 1, "registry_version": "test-v1", "targets": targets}
    compatibility: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "inference_hub_provider_compatibility",
        "endpoint": ENDPOINT,
        "registry_sha256": _sha256_json(registry),
        "targets": compatibility_targets,
        "target_count": len(compatibility_targets),
        "selected_count": len(compatibility_targets),
        "unresolved_count": 0,
    }
    compatibility["evidence_sha256"] = _sha256_json(compatibility)
    registry_path = tmp_path / "registry.json"
    compatibility_path = tmp_path / "compatibility.json"
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    compatibility_path.write_text(json.dumps(compatibility) + "\n", encoding="utf-8")
    return registry_path, compatibility_path, target_ids


class _FullFakeClient:
    base_url = ENDPOINT

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.threads: set[str] = set()
        self.lock = threading.Lock()

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        with self.lock:
            self.calls.append(dict(body))
            self.threads.add(threading.current_thread().name)
        return {
            "id": f"request-{len(self.calls)}",
            "model": body["model"],
            "choices": [
                {"message": {"role": "assistant", "content": "X"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101},
        }


class _OneIdentityDriftClient(_FullFakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.drift_emitted = False

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = super().post(path, body)
        with self.lock:
            if not self.drift_emitted:
                response["model"] = "served/transient-wrong-route"
                self.drift_emitted = True
        return response


class _TwoTransportNullsOneSemanticInvalidClient(_FullFakeClient):
    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = super().post(path, body)
        call_number = len(self.calls)
        if call_number <= 2:
            raise RuntimeError("simulated transport failure")
        if call_number == 3:
            response["choices"][0]["message"]["content"] = (
                "This is a retained semantic response without a final X or Y."
            )
        return response


def test_immutable_panel_and_full_role_schedule_are_exact() -> None:
    config = load_frozen_config()
    trials = build_frozen_trials(config)

    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == CONFIG_FILE_SHA256
    assert len(config["subjects"]) == EXPECTED_SUBJECT_COUNT
    assert len({row["developer_family"] for row in config["subjects"]}) == 6
    assert config["estimand_contract"]["frames_pooled"] is False
    assert config["estimand_contract"]["models_pooled"] is False
    assert len(trials) == EXPECTED_TRIALS_PER_SUBJECT
    assert len({trial.root_id for trial in trials}) == EXPECTED_ROOT_COUNT
    assert Counter((trial.game, trial.domain) for trial in trials) == Counter(
        {(game, domain): 96 for game in GAMES for domain in DOMAINS}
    )
    assert Counter(trial.frame_id for trial in trials) == {
        "advice": 384,
        "observer_evaluation": 384,
        "prediction": 384,
    }


def test_exact_sentinels_select_from_real_combined_registry_and_compatibility() -> None:
    registry_path = (
        REPO_ROOT / "data/private/inference_hub/sota-combined-registry-v1.json"
    )
    compatibility_path = (
        REPO_ROOT / "data/private/inference_hub/sota-combined-compatibility-v1.json"
    )
    if not registry_path.is_file() or not compatibility_path.is_file():
        pytest.skip("Private combined route evidence is intentionally not distributed.")
    config = load_frozen_config()
    expected = [row["target_id"] for row in config["subjects"]]
    subjects, judge = select_routes(
        registry=_read_json(registry_path, "combined registry"),
        compatibility=_read_json(compatibility_path, "combined compatibility"),
        selected_ids=expected,
        judge_target_id=config["judge_target_id"],
    )
    ordered = provider_round_robin(subjects)

    assert [row["target_id"] for row in ordered] == expected
    assert len({row["upstream_provider"] for row in ordered}) == 6
    assert judge["target_id"] == JUDGE_ID


def test_full_six_model_run_resume_sanitization_and_tamper_refusal(tmp_path: Path) -> None:
    registry_path, compatibility_path, target_ids = _registry_and_compatibility(tmp_path)
    output_dir = tmp_path / "role-calibration-v1"
    client = _FullFakeClient()

    manifest = run_calibration(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=client,
        max_workers=12,
        max_workers_per_provider=1,
        initial_backoff_seconds=0,
    )

    assert manifest["complete"] is True
    assert manifest["summary"]["planned_generations"] == 6 * 1152 == len(client.calls)
    assert manifest["summary"]["format_valid"] == 6 * 1152
    assert manifest["judge_reservation"]["dispatch_permitted_in_this_runner"] is False
    judge_route = manifest["judge_reservation"]["route"]
    assert all(call["model"] != judge_route for call in client.calls)
    assert len(client.threads) > 1

    private = output_dir / "private"
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE((private / "manifest.json").stat().st_mode) == 0o600
    ledger_rows = [
        json.loads(line)
        for line in (private / "attempt_ledger.jsonl").read_text().splitlines()
    ]
    assert len(ledger_rows) == 2 * 6 * 1152
    previous = None
    for row in ledger_rows:
        assert row["previous_record_sha256"] == previous
        assert row["record_sha256"] == _sha256_json(
            {key: value for key, value in row.items() if key != "record_sha256"}
        )
        previous = row["record_sha256"]

    sanitized_text = (output_dir / "sanitized" / "summary.json").read_text()
    root_text = (output_dir / "sanitized" / "root_summaries.jsonl").read_text()
    assert "prompt_text" not in sanitized_text + root_text
    assert "raw_response" not in sanitized_text + root_text
    assert "served/model" not in sanitized_text + root_text
    sanitized = json.loads(sanitized_text)
    assert sanitized["evidence_sha256"] == _sha256_json(
        {key: value for key, value in sanitized.items() if key != "evidence_sha256"}
    )
    assert sanitized["raw_text_included"] is False
    assert sanitized["frames_pooled"] is False
    assert sanitized["models_pooled"] is False
    assert len(sanitized["estimates"]) == 6 * 3
    assert sanitized["root_summary_record_count"] == 6 * 3 * 96
    root_rows = [json.loads(line) for line in root_text.splitlines()]
    assert sanitized["root_summaries_canonical_sha256"] == _sha256_json(root_rows)
    assert {row["target_id"] for row in sanitized["estimates"]} == set(target_ids)
    assert all(row["planned_roots"] == 96 for row in sanitized["estimates"])
    assert all(
        row["root_weighted_welfare_preserving_rate_among_valid"] == pytest.approx(0.5)
        for row in sanitized["estimates"]
    )

    resumed_client = _FullFakeClient()
    resumed = run_calibration(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=resumed_client,
        max_workers=12,
        max_workers_per_provider=1,
        initial_backoff_seconds=0,
        resume=True,
    )
    assert resumed["complete"] is True
    assert resumed["resume_count"] == 1
    assert resumed_client.calls == []

    raw_path = next((private / "raw_responses").glob("*.jsonl"))
    rows = raw_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["prompt_sha256"] = "0" * 64
    rows[0] = json.dumps(first, sort_keys=True)
    raw_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(InferenceHubPart1PanelError, match="hash chain"):
        run_calibration(
            registry_path=registry_path,
            compatibility_path=compatibility_path,
            output_dir=output_dir,
            client=_FullFakeClient(),
            max_workers=12,
            max_workers_per_provider=1,
            initial_backoff_seconds=0,
            resume=True,
        )


def test_response_identity_drift_is_retried_before_retention(
    tmp_path: Path,
) -> None:
    registry_path, compatibility_path, _ = _registry_and_compatibility(
        tmp_path
    )
    output_dir = tmp_path / "role-identity-retry"
    client = _OneIdentityDriftClient()
    manifest = run_calibration(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=client,
        max_workers=12,
        max_workers_per_provider=1,
        max_attempts=3,
        initial_backoff_seconds=0,
    )
    assert manifest["complete"] is True
    assert manifest["summary"][
        "response_model_identity_mismatches"
    ] == 0
    assert len(client.calls) == EXPECTED_SUBJECT_COUNT * (
        EXPECTED_TRIALS_PER_SUBJECT
    ) + 1
    ledger = [
        json.loads(line)
        for line in (
            output_dir / "private/attempt_ledger.jsonl"
        ).read_text().splitlines()
    ]
    failures = [
        row
        for row in ledger
        if row.get("failure_code")
        == "response_model_identity_mismatch"
    ]
    assert len(failures) == 1
    assert failures[0]["outcome"] == "failed"
    raw_rows = []
    for path in (
        output_dir / "private/raw_responses"
    ).glob("*.jsonl"):
        raw_rows.extend(
            json.loads(line) for line in path.read_text().splitlines()
        )
    assert len(raw_rows) == (
        EXPECTED_SUBJECT_COUNT * EXPECTED_TRIALS_PER_SUBJECT
    )
    assert all(row["model_identity_valid"] is True for row in raw_rows)


def test_resume_repairs_only_transport_nulls_in_immutable_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path, compatibility_path, _ = _registry_and_compatibility(tmp_path)
    output_dir = tmp_path / "role-operational-repair"
    frozen_subset = build_frozen_trials(load_frozen_config())[:2]
    monkeypatch.setattr(
        role_runner,
        "build_frozen_trials",
        lambda config: frozen_subset,
    )

    initial_client = _TwoTransportNullsOneSemanticInvalidClient()
    initial = run_calibration(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=initial_client,
        max_workers=1,
        max_workers_per_provider=1,
        max_attempts=1,
        initial_backoff_seconds=0,
    )
    assert initial["complete"] is False
    assert initial["summary"]["planned_generations"] == 12
    assert initial["summary"]["failed_without_response"] == 2
    assert initial["summary"]["format_invalid_retained"] == 1

    raw_dir = output_dir / "private/raw_responses"
    original_bytes = {path.name: path.read_bytes() for path in raw_dir.glob("*.jsonl")}
    original_rows = [
        json.loads(line)
        for path in raw_dir.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    transport_nulls = [row for row in original_rows if row["raw_response"] is None]
    semantic_invalids = [
        row
        for row in original_rows
        if row["raw_response"] is not None and row["format_valid"] is False
    ]
    assert len(transport_nulls) == 2
    assert len(semantic_invalids) == 1

    repair_client = _FullFakeClient()
    repaired = run_calibration(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=repair_client,
        max_workers=1,
        max_workers_per_provider=1,
        max_attempts=1,
        initial_backoff_seconds=0,
        resume=True,
    )
    assert repaired["complete"] is True
    assert len(repair_client.calls) == 2
    assert {path.name: path.read_bytes() for path in raw_dir.glob("*.jsonl")} == (
        original_bytes
    )
    assert repaired["summary"]["failed_without_response"] == 0
    assert repaired["summary"]["format_invalid_retained"] == 1
    assert repaired["summary"]["operational_repair_eligible_originals"] == 2
    assert repaired["summary"]["operational_repairs_succeeded"] == 2
    assert repaired["summary"]["operational_repairs_unresolved"] == 0
    assert repaired["summary"]["non_null_format_invalid_originals_not_retried"] == 1

    repair_rows = [
        json.loads(line)
        for path in (
            output_dir / "private/operational_repairs/raw_responses"
        ).glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(repair_rows) == 2
    assert {row["artifact_type"] for row in repair_rows} == {
        "inference_hub_part1_role_calibration_operational_repair_response_v1"
    }
    assert {row["replaces_original_record_sha256"] for row in repair_rows} == {
        row["record_sha256"] for row in transport_nulls
    }
    assert semantic_invalids[0]["record_sha256"] not in {
        row["replaces_original_record_sha256"] for row in repair_rows
    }
    sanitized = json.loads(
        (output_dir / "sanitized/summary.json").read_text(encoding="utf-8")
    )
    assert sum(row["estimable_draws"] for row in sanitized["estimates"]) == 11
    assert sanitized["operational_repair_policy"] == (
        "transport_null_overlay_only_no_semantic_retry_v1"
    )

    no_op_client = _FullFakeClient()
    no_op = run_calibration(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=no_op_client,
        max_workers=1,
        max_workers_per_provider=1,
        max_attempts=1,
        initial_backoff_seconds=0,
        resume=True,
    )
    assert no_op["complete"] is True
    assert no_op_client.calls == []


def test_config_path_substitution_is_refused(tmp_path: Path) -> None:
    copied = tmp_path / CONFIG_PATH.name
    copied.write_bytes(CONFIG_PATH.read_bytes())
    with pytest.raises(RoleCalibrationError, match="repository config path"):
        load_frozen_config(copied)


def test_operational_resume_allows_only_the_overlay_runner_source_change() -> None:
    runner_path = str(Path(role_runner.__file__).resolve())
    sibling_path = str((Path(role_runner.__file__).parent / "sibling.py").resolve())
    retained = {
        "panel_id": "frozen-panel",
        "source_artifacts": {
            runner_path: "1" * 64,
            sibling_path: "2" * 64,
        },
    }
    fresh = {
        "panel_id": "frozen-panel",
        "source_artifacts": {
            runner_path: "3" * 64,
            sibling_path: "2" * 64,
        },
    }
    _validate_operational_resume_contract(retained, fresh)

    changed_sibling = json.loads(json.dumps(fresh))
    changed_sibling["source_artifacts"][sibling_path] = "4" * 64
    with pytest.raises(RoleCalibrationError, match="outside the repair runner"):
        _validate_operational_resume_contract(retained, changed_sibling)


def test_provider_safety_refuses_more_than_three_in_flight_calls_per_provider(
    tmp_path: Path,
) -> None:
    with pytest.raises(RoleCalibrationError, match="at most three"):
        run_calibration(
            registry_path=tmp_path / "unused-registry.json",
            compatibility_path=tmp_path / "unused-compatibility.json",
            output_dir=tmp_path / "unused-output",
            client=_FullFakeClient(),
            max_workers_per_provider=4,
        )


def test_raw_first_crash_window_is_recovered_without_provider_dispatch(
    tmp_path: Path,
) -> None:
    ledger = _RunLockedJournal(tmp_path / "attempts.jsonl")
    raw = _RunLockedJournal(tmp_path / "raw.jsonl")
    request_sha256 = "1" * 64
    ledger.append(
        {
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_role_calibration_attempt_v1",
            "event": "reserved_before_dispatch",
            "attempt_id": "attempt-1",
            "target_id": "target-1",
            "trial_id": "trial-1",
            "request_sha256": request_sha256,
            "attempt_number": 1,
        }
    )
    raw.append(
        {
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_role_calibration_raw_response_v1",
            "attempt_id": "attempt-1",
            "target_id": "target-1",
            "trial_id": "trial-1",
            "request_sha256": request_sha256,
            "raw_response": {"id": "response-1"},
            "raw_response_sha256": _sha256_json({"id": "response-1"}),
            "request_id": "response-1",
            "response_model": "served/model-1",
            "response_text_sha256": "2" * 64,
        }
    )

    _recover_ledger(ledger, {"target-1": raw})
    completions = [row for row in ledger.records if row["event"] == "attempt_completed"]
    assert len(completions) == 1
    assert completions[0]["outcome"] == "response_retained"
    assert completions[0]["recovered_after_raw_fsync"] is True

    _recover_ledger(ledger, {"target-1": raw})
    assert len([row for row in ledger.records if row["event"] == "attempt_completed"]) == 1
