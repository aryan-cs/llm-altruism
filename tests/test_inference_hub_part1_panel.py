import hashlib
import json
import os
import shutil
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from analysis.analyze_inference_hub_part1_panel import (
    HostedPart1AnalysisError,
    _self_hash as _analysis_self_hash,
    analyze_panel,
    cli as analysis_cli,
)
from experiments.misc.inference_hub_discovery import InferenceHubDiscoveryError
from experiments.misc.inference_hub_part1_panel import (
    InferenceHubPart1PanelError,
    _canonical_bytes,
    _sha256_json,
    build_draft_trials,
    parse_final_action,
    run_panel,
    select_routes,
)


ENDPOINT = "https://inference-api.nvidia.com/v1"
JUDGE_ID = "judge.route"


def _registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry_version": "unit-v1",
        "targets": [
            {
                "id": "subject.alpha",
                "provider": "inference_hub",
                "upstream_provider": "alpha",
                "model": "alpha-model",
                "route": "display/alpha-model",
            },
            {
                "id": "subject.beta",
                "provider": "inference_hub",
                "upstream_provider": "beta",
                "model": "beta-model",
                "route": "display/beta-model",
            },
            {
                "id": JUDGE_ID,
                "provider": "inference_hub",
                "upstream_provider": "judge-provider",
                "model": "judge-model",
                "route": "display/judge-model",
            },
        ],
    }


def _target(
    target_id: str, model: str, route: str, controls: list[str]
) -> dict[str, Any]:
    profile = {
        "attempt_id": f"probe-{target_id}",
        "controls": controls,
        "profile_id": f"profile-{target_id}",
        "request_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
        "resumed_from_ledger": False,
        "status": "passed",
        "validation_source": "execution_profile_probe",
    }
    return {
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
                "max_tokens": 48,
                "execution_compatible": True,
                "selected_execution_profile": profile,
            }
        ],
    }


def _compatibility(registry: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 2,
        "artifact_type": "inference_hub_provider_compatibility",
        "endpoint": ENDPOINT,
        "registry_sha256": _sha256_json(registry),
        "targets": [
            _target(
                "subject.alpha",
                "alpha-model",
                "region/alpha-model",
                ["seed", "temperature", "top_p", "structured_response"],
            ),
            _target("subject.beta", "beta-model", "region/beta-model", ["seed"]),
            _target(
                JUDGE_ID,
                "judge-model",
                "region/judge-model",
                ["seed", "structured_response"],
            ),
        ],
        "target_count": 3,
        "selected_count": 3,
        "unresolved_count": 0,
    }
    payload["evidence_sha256"] = _sha256_json(payload)
    return payload


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    registry = _registry()
    compatibility = _compatibility(registry)
    registry_path = tmp_path / "registry.json"
    compatibility_path = tmp_path / "compatibility.json"
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    compatibility_path.write_text(json.dumps(compatibility) + "\n", encoding="utf-8")
    return registry_path, compatibility_path


def _response(body: dict[str, Any]) -> dict[str, Any]:
    text = "analysis\nX" if body["model"].endswith("alpha-model") else "not X or Y"
    return {
        "id": f"request-{body['seed']}",
        "model": body["model"],
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": text,
                    "reasoning_content": "private reasoning",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


class _FakeClient:
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
        return _response(body)


class _WrongIdentityClient(_FakeClient):
    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = super().post(path, body)
        response["model"] = "wrong/provider/model"
        return response


def test_schedule_and_parser_are_complete_and_deterministic() -> None:
    first = build_draft_trials(base_seed=20_260_802)
    second = build_draft_trials(base_seed=20_260_802)
    assert len(first) == len({trial.root_id for trial in first}) == 384
    assert [trial.prompt_hash for trial in first] == [trial.prompt_hash for trial in second]
    assert parse_final_action("reasoning\nX") == "X"
    assert parse_final_action("Y\nextra") is None
    assert parse_final_action("answer: X") is None


def test_response_identity_mismatch_prevents_completion(tmp_path: Path) -> None:
    registry_path, compatibility_path = _write_inputs(tmp_path)
    manifest = run_panel(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=tmp_path / "identity-mismatch",
        client=_WrongIdentityClient(),
        selected_ids=["subject.alpha"],
        judge_target_id=JUDGE_ID,
        limit=1,
        max_workers=1,
        max_attempts=1,
        initial_backoff_seconds=0,
    )

    assert manifest["summary"]["response_model_identity_mismatches"] == 1
    assert manifest["complete"] is False


def test_panel_uses_profiles_excludes_judge_and_retains_invalids_privately(
    tmp_path: Path,
) -> None:
    registry_path, compatibility_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "panel"
    client = _FakeClient()

    manifest = run_panel(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=client,
        selected_ids=["subject.alpha", "subject.beta"],
        judge_target_id=JUDGE_ID,
        limit=2,
        max_workers=4,
        initial_backoff_seconds=0,
    )

    assert manifest["draft_bank_human_approved"] is False
    assert manifest["confirmatory_or_paper_promotion_permitted"] is False
    assert manifest["judge_dispatched"] is False
    assert manifest["judge_reservation"]["target_id"] == JUDGE_ID
    assert manifest["judge_reservation"]["subject_target_ids"] == [
        "subject.alpha",
        "subject.beta",
    ]
    assert manifest["full_primary_root_count"] == 384
    assert manifest["executed_trial_count_per_subject"] == 2
    assert manifest["summary"]["retained_trial_records"] == 4
    assert manifest["summary"]["format_valid"] == 2
    assert manifest["summary"]["format_invalid_retained"] == 2
    assert manifest["complete"] is True
    assert len(client.calls) == 4
    assert all(call["model"] != "region/judge-model" for call in client.calls)
    alpha = [call for call in client.calls if call["model"] == "region/alpha-model"]
    beta = [call for call in client.calls if call["model"] == "region/beta-model"]
    assert all(call["temperature"] == 0 and call["top_p"] == 1 for call in alpha)
    assert all("temperature" not in call and "top_p" not in call for call in beta)
    assert all("response_format" not in call for call in client.calls)
    assert all(call["max_tokens"] >= 2048 for call in client.calls)
    assert all(
        call["messages"][0]["content"].startswith("Return exactly one character")
        for call in client.calls
    )

    private = output_dir / "private"
    raw = private / "raw_responses" / "subject.alpha.jsonl"
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE(raw.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(raw.stat().st_mode) == 0o600
    assert stat.S_IMODE((private / "attempt_ledger.jsonl").stat().st_mode) == 0o600
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["reasoning_fields"] == {"reasoning_content": "private reasoning"}
    assert rows[0]["usage"]["total_tokens"] == 13
    assert rows[0]["controls"]["omitted"]["structured_response"] == (
        "bare_terminal_XY_parser_contract"
    )

    ledger = [
        json.loads(line)
        for line in (private / "attempt_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for reservation in [row for row in ledger if row["event"] == "reserved_before_dispatch"]:
        call = next(
            body
            for body in client.calls
            if body["model"] == reservation["route"]
            and hashlib.sha256(_canonical_bytes(body)).hexdigest()
            == reservation["request_sha256"]
        )
        assert "api_key" not in call and "authorization" not in call
    assert all(
        next(i for i, row in enumerate(ledger) if row.get("attempt_id") == reservation["attempt_id"] and row["event"] == "attempt_completed")
        > index
        for index, reservation in enumerate(ledger)
        if reservation["event"] == "reserved_before_dispatch"
    )


def test_transient_retry_then_resume_does_not_redispatch(tmp_path: Path) -> None:
    registry_path, compatibility_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "panel"

    class ThrottledOnce(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.throttled = False

        def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
            with self.lock:
                self.calls.append(dict(body))
            if not self.throttled:
                self.throttled = True
                raise InferenceHubDiscoveryError(
                    "throttled",
                    failure_code="http_error",
                    http_status=429,
                )
            return _response(body)

    client = ThrottledOnce()
    sleeps: list[float] = []
    first = run_panel(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=client,
        selected_ids=["subject.alpha"],
        judge_target_id=JUDGE_ID,
        limit=1,
        max_workers=1,
        max_attempts=3,
        initial_backoff_seconds=0.25,
        sleep_fn=sleeps.append,
    )
    assert first["complete"] is True
    assert len(client.calls) == 2
    assert sleeps == [0.25]

    resumed_client = _FakeClient()
    second = run_panel(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=resumed_client,
        selected_ids=["subject.alpha"],
        judge_target_id=JUDGE_ID,
        limit=1,
        max_workers=1,
        max_attempts=3,
        initial_backoff_seconds=0.25,
        resume=True,
    )
    assert second["complete"] is True
    assert second["resume_count"] == 1
    assert resumed_client.calls == []


def test_tampered_raw_response_blocks_resume(tmp_path: Path) -> None:
    registry_path, compatibility_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "panel"
    run_panel(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=_FakeClient(),
        selected_ids=["subject.alpha"],
        judge_target_id=JUDGE_ID,
        limit=1,
        max_workers=1,
    )
    raw = output_dir / "private" / "raw_responses" / "subject.alpha.jsonl"
    raw.write_text(raw.read_text(encoding="utf-8").replace("private reasoning", "changed"), encoding="utf-8")

    with pytest.raises(InferenceHubPart1PanelError, match="hash chain"):
        run_panel(
            registry_path=registry_path,
            compatibility_path=compatibility_path,
            output_dir=output_dir,
            client=_FakeClient(),
            selected_ids=["subject.alpha"],
            judge_target_id=JUDGE_ID,
            limit=1,
            max_workers=1,
            resume=True,
        )


def test_subject_filter_cannot_include_judge() -> None:
    registry = _registry()
    with pytest.raises(InferenceHubPart1PanelError, match="cannot be a subject"):
        select_routes(
            registry=registry,
            compatibility=_compatibility(registry),
            selected_ids=[JUDGE_ID],
            judge_target_id=JUDGE_ID,
        )


@pytest.fixture(scope="module")
def complete_analysis_panel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Production-sized private evidence with two full 384-root subjects."""

    root = tmp_path_factory.mktemp("complete-hosted-part1-analysis")
    registry_path, compatibility_path = _write_inputs(root)
    output_dir = root / "panel"
    manifest = run_panel(
        registry_path=registry_path,
        compatibility_path=compatibility_path,
        output_dir=output_dir,
        client=_FakeClient(),
        selected_ids=["subject.alpha", "subject.beta"],
        judge_target_id=JUDGE_ID,
        max_workers=16,
        max_workers_per_subject=2,
        initial_backoff_seconds=0,
    )
    assert manifest["complete"] is True
    assert manifest["summary"]["retained_trial_records"] == 768
    return output_dir


def _panel_copy(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "panel"
    shutil.copytree(source, destination)
    # Journal references are intentionally absolute.  A copied fixture must be
    # explicitly rebound and resealed before it represents a coherent artifact.
    manifest_path = destination / "private" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    private = destination / "private"
    copied_inputs = destination / "bound-inputs"
    copied_inputs.mkdir()
    for name, reference in manifest["input_artifacts"].items():
        copied = copied_inputs / f"{name}.json"
        shutil.copy2(reference["path"], copied)
        reference["path"] = str(copied.resolve())
    manifest["journals"]["attempt_ledger"]["path"] = str(
        (private / "attempt_ledger.jsonl").resolve()
    )
    for target_id, reference in manifest["journals"]["raw_responses"].items():
        reference["path"] = str((private / "raw_responses" / f"{target_id}.jsonl").resolve())
    manifest["evidence_sha256"] = _analysis_self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    return destination


def _rewrite_chain(panel: Path, relative: str, mutate: Any) -> None:
    private = panel / "private"
    path = private / relative
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    previous = None
    encoded: list[str] = []
    for row in rows:
        row["previous_record_sha256"] = previous
        row.pop("record_sha256", None)
        row["record_sha256"] = _sha256_json(row)
        previous = row["record_sha256"]
        encoded.append(json.dumps(row, sort_keys=True, ensure_ascii=False))
    path.write_text("\n".join(encoded) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)

    manifest_path = private / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if relative == "attempt_ledger.jsonl":
        reference = manifest["journals"]["attempt_ledger"]
    else:
        target_id = Path(relative).stem
        reference = manifest["journals"]["raw_responses"][target_id]
    reference["record_count"] = len(rows)
    reference["tail_record_sha256"] = previous
    reference["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["evidence_sha256"] = _analysis_self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)


def test_analysis_full_384_schedule_is_safe_deterministic_and_atomic(
    complete_analysis_panel: Path, tmp_path: Path
) -> None:
    panel = _panel_copy(complete_analysis_panel, tmp_path)
    manifest_path = panel / "private" / "manifest.json"
    first = analyze_panel(manifest_path)

    assert first["coverage"] == {
        "planned_subject_count": 2,
        "aggregate_eligible_subject_count": 2,
        "quarantined_subject_count": 0,
        "roots_per_subject": 384,
        "planned_subject_trial_count": 768,
        "retained_subject_trial_count": 768,
        "game_domain_strata": 12,
        "roots_per_stratum_per_subject": 32,
        "panel_wide_complete_for_all_subjects": True,
        "runner_complete_flag": True,
        "judge_rows": 0,
    }
    assert first["parameters"]["bootstrap_replicates"] == 5_000
    assert first["confirmatory_or_paper_promotion_permitted"] is False
    alpha, beta = first["subjects"]
    assert alpha["target_id"] == "subject.alpha"
    assert alpha["counts"]["format_valid_rate"] == 1.0
    assert alpha["counts"]["action_x_count"] == 384
    assert alpha["counts"]["primary_action_x_rate_format_invalid_retained_as_non_x"] == 1.0
    assert len(alpha["per_game_domain"]) == 12
    assert beta["counts"]["format_valid_rate"] == 0.0
    assert beta["counts"]["valid_only_action_x_rate_descriptive"] is None
    assert first["overall_equal_subject"]["primary_action_x_rate_95_ci"]["estimate"] == 0.5
    assert first["evidence_sha256"] == _analysis_self_hash(first)
    serialized = json.dumps(first)
    assert "private reasoning" not in serialized
    assert all("prompt_text" not in row for row in first["subjects"])
    assert all("raw_response" not in row for row in first["subjects"])
    assert all("response_text" not in row for row in first["subjects"])
    assert all("reasoning_fields" not in row for row in first["subjects"])

    output = tmp_path / "derived" / "analysis.json"
    assert analysis_cli(["--manifest", str(manifest_path), "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == first
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not list(output.parent.glob(".analysis.json.*.tmp"))


@pytest.mark.parametrize(
    ("relative", "mutation", "message"),
    [
        (
            "raw_responses/subject.alpha.jsonl",
            lambda rows: rows.__setitem__(0, {**rows[0], "response_text": "changed\nX"}),
            "parser/text-hash",
        ),
        (
            "raw_responses/subject.alpha.jsonl",
            lambda rows: rows.__setitem__(0, {**rows[0], "response_model": "wrong/model"}),
            "route/model identity",
        ),
        (
            "raw_responses/subject.alpha.jsonl",
            lambda rows: rows.__setitem__(0, {**rows[0], "target_id": JUDGE_ID}),
            "judge record",
        ),
        (
            "attempt_ledger.jsonl",
            lambda rows: rows.__setitem__(0, {**rows[0], "route": "wrong/route"}),
            "reservation binding",
        ),
    ],
)
def test_analysis_catches_semantic_tamper_after_attacker_rechains_outer_hashes(
    complete_analysis_panel: Path,
    tmp_path: Path,
    relative: str,
    mutation: Any,
    message: str,
) -> None:
    panel = _panel_copy(complete_analysis_panel, tmp_path)
    _rewrite_chain(panel, relative, mutation)
    with pytest.raises(HostedPart1AnalysisError, match=message):
        analyze_panel(panel / "private" / "manifest.json")


def test_analysis_catches_dropped_raw_row_even_when_rechained(
    complete_analysis_panel: Path, tmp_path: Path
) -> None:
    panel = _panel_copy(complete_analysis_panel, tmp_path)
    _rewrite_chain(panel, "raw_responses/subject.alpha.jsonl", lambda rows: rows.pop())
    with pytest.raises(HostedPart1AnalysisError, match="exact 384-root coverage"):
        analyze_panel(panel / "private" / "manifest.json")


def test_analysis_quarantines_entire_identity_mismatched_target_from_every_aggregate(
    complete_analysis_panel: Path, tmp_path: Path
) -> None:
    panel = _panel_copy(complete_analysis_panel, tmp_path)
    private = panel / "private"
    captured: dict[str, str] = {}

    def mismatch_one_raw(rows: list[dict[str, Any]]) -> None:
        row = rows[0]
        row["raw_response"]["model"] = "unexpected/backend/model"
        row["response_model"] = "unexpected/backend/model"
        row["model_identity_valid"] = False
        row["raw_response_sha256"] = _sha256_json(row["raw_response"])
        captured["attempt_id"] = row["attempt_id"]
        captured["payload_sha256"] = row["raw_response_sha256"]

    _rewrite_chain(panel, "raw_responses/subject.beta.jsonl", mismatch_one_raw)

    def match_ledger_completion(rows: list[dict[str, Any]]) -> None:
        completion = next(
            row
            for row in rows
            if row.get("event") == "attempt_completed"
            and row.get("attempt_id") == captured["attempt_id"]
        )
        completion["response_model"] = "unexpected/backend/model"
        completion["response_payload_sha256"] = captured["payload_sha256"]

    _rewrite_chain(panel, "attempt_ledger.jsonl", match_ledger_completion)
    manifest_path = private / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"]["response_model_identity_mismatches"] = 1
    manifest["complete"] = False
    manifest["evidence_sha256"] = _analysis_self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)

    result = analyze_panel(manifest_path)
    assert result["coverage"]["panel_wide_complete_for_all_subjects"] is False
    assert result["coverage"]["aggregate_eligible_subject_count"] == 1
    assert result["coverage"]["quarantined_subject_count"] == 1
    assert result["quarantined_targets"] == [
        {
            "target_id": "subject.beta",
            "operational_reasons": ["response_model_identity_mismatch"],
            "response_model_identity_mismatch_count": 1,
            "failed_without_response_count": 0,
            "excluded_row_count": 384,
            "exclusion_scope": "entire_target_excluded_from_all_aggregates_and_bootstrap",
            "exclusion_basis": "protocol_and_model_identity_only_not_observed_action",
        }
    ]
    assert [row["target_id"] for row in result["subjects"]] == ["subject.alpha"]
    assert result["overall_equal_subject"]["subject_count"] == 1
    assert result["overall_equal_subject"]["primary_action_x_rate_95_ci"]["estimate"] == 1.0
    assert {row["upstream_provider"] for row in result["provider_descriptive"]} == {"alpha"}
    assert all("subject.beta" not in row["subject_target_ids"] for row in result["family_descriptive"])


def test_analysis_quarantines_entire_target_after_one_terminal_provider_failure(
    complete_analysis_panel: Path, tmp_path: Path
) -> None:
    panel = _panel_copy(complete_analysis_panel, tmp_path)
    private = panel / "private"
    captured: dict[str, str] = {}

    def fail_one_raw(rows: list[dict[str, Any]]) -> None:
        row = rows[0]
        captured["attempt_id"] = row["attempt_id"]
        row.update(
            {
                "raw_response": None,
                "raw_response_sha256": None,
                "response_model": None,
                "model_identity_valid": False,
                "request_id": None,
                "finish_reason": None,
                "usage": None,
                "reasoning_fields": {},
                "output_field": None,
                "response_text": None,
                "response_text_sha256": None,
                "parsed_action": None,
                "format_valid": False,
                "failure": {
                    "failure_code": "unexpected_client_error",
                    "http_status": None,
                    "error_type": "RuntimeError",
                },
            }
        )

    _rewrite_chain(panel, "raw_responses/subject.beta.jsonl", fail_one_raw)

    def fail_ledger_completion(rows: list[dict[str, Any]]) -> None:
        completion = next(
            row
            for row in rows
            if row.get("event") == "attempt_completed"
            and row.get("attempt_id") == captured["attempt_id"]
        )
        completion.update(
            {
                "outcome": "failed",
                "failure_code": "unexpected_client_error",
                "transient": False,
                "http_status": None,
            }
        )
        for key in (
            "request_id", "response_model", "response_payload_sha256",
            "response_text_sha256", "finish_reason", "usage",
            "reasoning_fields_sha256",
        ):
            completion.pop(key, None)

    _rewrite_chain(panel, "attempt_ledger.jsonl", fail_ledger_completion)
    manifest_path = private / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["summary"].update(
        {
            "responses_received": 767,
            "failed_without_response": 1,
            "format_invalid_retained": 383,
        }
    )
    manifest["complete"] = False
    manifest["evidence_sha256"] = _analysis_self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)

    result = analyze_panel(manifest_path)
    assert [row["target_id"] for row in result["subjects"]] == ["subject.alpha"]
    assert result["quarantined_targets"][0]["operational_reasons"] == [
        "failed_without_response"
    ]
    assert result["quarantined_targets"][0]["failed_without_response_count"] == 1
    assert result["overall_equal_subject"]["subject_count"] == 1
    assert result["overall_equal_subject"]["primary_action_x_rate_95_ci"]["estimate"] == 1.0


def test_analysis_catches_manifest_input_source_and_raw_file_hash_tamper(
    complete_analysis_panel: Path, tmp_path: Path
) -> None:
    panel = _panel_copy(complete_analysis_panel, tmp_path)
    manifest_path = panel / "private" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_artifacts"][next(iter(manifest["source_artifacts"]))] = "0" * 64
    manifest["evidence_sha256"] = _analysis_self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    with pytest.raises(HostedPart1AnalysisError, match="Source artifact hash changed"):
        analyze_panel(manifest_path)

    panel = _panel_copy(complete_analysis_panel, tmp_path / "input")
    manifest_path = panel / "private" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    registry_path = Path(manifest["input_artifacts"]["registry"]["path"])
    registry_path.write_text(registry_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(HostedPart1AnalysisError, match="Registry input file hash changed"):
        analyze_panel(manifest_path)

    panel = _panel_copy(complete_analysis_panel, tmp_path / "raw")
    raw = panel / "private" / "raw_responses" / "subject.alpha.jsonl"
    raw.write_text(raw.read_text(encoding="utf-8").replace("private reasoning", "tampered", 1), encoding="utf-8")
    with pytest.raises(HostedPart1AnalysisError, match="hash chain"):
        analyze_panel(panel / "private" / "manifest.json")


def test_analysis_catches_nonpromotion_and_private_permission_changes(
    complete_analysis_panel: Path, tmp_path: Path
) -> None:
    panel = _panel_copy(complete_analysis_panel, tmp_path)
    manifest_path = panel / "private" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["confirmatory_or_paper_promotion_permitted"] = True
    manifest["evidence_sha256"] = _analysis_self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    with pytest.raises(HostedPart1AnalysisError, match="nonpromotion flags"):
        analyze_panel(manifest_path)

    panel = _panel_copy(complete_analysis_panel, tmp_path / "permissions")
    manifest_path = panel / "private" / "manifest.json"
    os.chmod(manifest_path, 0o644)
    with pytest.raises(HostedPart1AnalysisError, match="Unsafe private permissions"):
        analyze_panel(manifest_path)
