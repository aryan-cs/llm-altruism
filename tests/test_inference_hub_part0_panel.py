import hashlib
import json
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from experiments.misc.inference_hub_discovery import InferenceHubDiscoveryError
from experiments.misc.inference_hub_part0_panel import (
    DEFAULT_JUDGE_TARGET_ID,
    InferenceHubPart0PanelError,
    build_legacy_trials,
    parse_judge_batch,
    run_panel,
)
from experiments.misc.inference_hub_part1_panel import _sha256_json


ENDPOINT = "https://inference-api.nvidia.com/v1"


def _registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "registry_version": "part0-test-v1",
        "targets": [
            {
                "id": "subject.alpha",
                "provider": "inference_hub",
                "upstream_provider": "alpha-provider",
                "model": "alpha-model",
            },
            {
                "id": DEFAULT_JUDGE_TARGET_ID,
                "provider": "inference_hub",
                "upstream_provider": "nvidia-evals",
                "model": "nemotron-judge-model",
            },
        ],
    }


def _compatibility_target(
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
                "max_tokens": 64,
                "execution_compatible": True,
                "selected_execution_profile": profile,
            }
        ],
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    registry = _registry()
    compatibility = {
        "schema_version": 2,
        "artifact_type": "inference_hub_provider_compatibility",
        "endpoint": ENDPOINT,
        "registry_sha256": _sha256_json(registry),
        "targets": [
            _compatibility_target(
                "subject.alpha",
                "alpha-model",
                "region/alpha-model",
                ["seed", "temperature", "top_p"],
            ),
            _compatibility_target(
                DEFAULT_JUDGE_TARGET_ID,
                "nemotron-judge-model",
                "nvidia/nemotron-judge-model",
                ["seed", "structured_response"],
            ),
        ],
        "target_count": 2,
        "selected_count": 2,
        "unresolved_count": 0,
    }
    compatibility["evidence_sha256"] = _sha256_json(compatibility)
    registry_path = tmp_path / "registry.json"
    compatibility_path = tmp_path / "compatibility.json"
    registry_path.write_text(json.dumps(registry) + "\n", encoding="utf-8")
    compatibility_path.write_text(json.dumps(compatibility) + "\n", encoding="utf-8")
    return registry_path, compatibility_path


class _FakeClient:
    base_url = ENDPOINT

    def __init__(self, *, malformed_judge: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.malformed_judge = malformed_judge

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        with self.lock:
            self.calls.append(body)
        if body["model"] == "nvidia/nemotron-judge-model":
            payload = json.loads(body["messages"][1]["content"])
            ids = [item["trial_id"] for item in payload["items"]]
            verdicts = [
                {"trial_id": trial_id, "label": "REFUSAL"} for trial_id in ids
            ]
            if self.malformed_judge:
                verdicts = list(reversed(verdicts))
            content = json.dumps({"verdicts": verdicts})
        else:
            content = "I cannot assist with that request."
        return {
            "id": f"request-{len(self.calls)}",
            "model": body["model"],
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "reasoning_content": "PRIVATE HIDDEN REASONING SENTINEL",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }


class _WrongSubjectIdentityClient(_FakeClient):
    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        response = super().post(path, body)
        if body["model"] == "region/alpha-model":
            response["model"] = "wrong/subject-model"
        return response


class _RetryOnceClient(_FakeClient):
    def __init__(self, status: int) -> None:
        super().__init__()
        self.status = status
        self.failed = False

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if body["model"] == "region/alpha-model" and not self.failed:
            self.failed = True
            raise InferenceHubDiscoveryError(
                "temporary transport failure",
                failure_code="http_error",
                http_status=self.status,
            )
        return super().post(path, body)


def _run(tmp_path: Path, client: _FakeClient, **overrides: Any) -> dict[str, Any]:
    registry_path, compatibility_path = _write_inputs(tmp_path)
    arguments = {
        "registry_path": registry_path,
        "compatibility_path": compatibility_path,
        "output_dir": tmp_path / "panel",
        "client": client,
        "selected_ids": ["subject.alpha"],
        "limit_per_language": 1,
        "max_workers": 8,
        "max_attempts": 2,
        "initial_backoff_seconds": 0,
        "sleep_fn": lambda _seconds: None,
    }
    arguments.update(overrides)
    return run_panel(**arguments)


def test_archived_bank_is_exact_stratified_and_deterministic() -> None:
    first = build_legacy_trials(limit_per_language=48)
    second = build_legacy_trials(limit_per_language=48)
    full = build_legacy_trials(limit_per_language=99)

    assert len(first) == 144
    assert [trial.trial_id for trial in first] == [trial.trial_id for trial in second]
    assert len({trial.root_id for trial in first}) == 48
    assert {trial.language for trial in first} == {"english", "chinese", "russian"}
    assert all(
        len({trial.prompt_text for trial in first if trial.root_id == root_id}) == 1
        for root_id in {trial.root_id for trial in first}
    )
    assert len(full) == 297
    roots_by_stratum = {
        stratum: len({trial.root_id for trial in first if trial.source_stratum == stratum})
        for stratum in {trial.source_stratum for trial in first}
    }
    assert roots_by_stratum == {"harmbench": 39, "jbb": 8, "harmbench_and_jbb": 1}


def test_judge_batch_parser_is_fail_closed() -> None:
    ids = ["one", "two"]
    valid = json.dumps(
        {"verdicts": [{"trial_id": "one", "label": "REFUSAL"}, {"trial_id": "two", "label": "UNCLEAR"}]}
    )
    assert parse_judge_batch(valid, ids) == {"one": "REFUSAL", "two": "UNCLEAR"}
    assert parse_judge_batch(valid, list(reversed(ids))) is None
    assert parse_judge_batch('{"verdicts":[{"trial_id":"one","label":"OTHER"}]}', ["one"]) is None
    assert parse_judge_batch('{"verdicts":[{"trial_id":"one","label":"REFUSAL"}]}', ids) is None


def test_full_collection_is_private_batched_visible_only_and_resumable(tmp_path: Path) -> None:
    client = _FakeClient()
    manifest = _run(tmp_path, client)

    assert manifest["analysis_role"].startswith("exploratory_deadline_panel")
    assert manifest["confirmatory_or_paper_promotion_permitted"] is False
    assert manifest["summary"]["planned_units"] == 3
    assert manifest["summary"]["outcomes"]["REFUSAL"] == 3
    assert manifest["complete"] is True
    subject_calls = [call for call in client.calls if call["model"] == "region/alpha-model"]
    judge_calls = [call for call in client.calls if call["model"] == "nvidia/nemotron-judge-model"]
    assert len(subject_calls) == 3
    assert len(judge_calls) == 1
    judge_payload = json.loads(judge_calls[0]["messages"][1]["content"])
    assert len(judge_payload["items"]) == 3
    assert "PRIVATE HIDDEN REASONING SENTINEL" not in json.dumps(judge_calls)
    assert all(call["model"] != judge_calls[0]["model"] for call in subject_calls)

    private = tmp_path / "panel" / "private"
    raw = private / "raw_responses" / "subject.alpha.jsonl"
    manifest_path = private / "manifest.json"
    assert stat.S_IMODE((tmp_path / "panel").stat().st_mode) == 0o700
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE(raw.stat().st_mode) == 0o600
    manifest_text = manifest_path.read_text(encoding="utf-8")
    first_prompt = build_legacy_trials(limit_per_language=1)[0].prompt_text
    assert first_prompt not in manifest_text
    assert "I cannot assist with that request." not in manifest_text
    assert first_prompt in raw.read_text(encoding="utf-8")

    call_count = len(client.calls)
    resumed = _run(tmp_path, client, resume=True)
    assert len(client.calls) == call_count
    assert resumed["resume_count"] == 1
    assert resumed["complete"] is True


def test_malformed_judge_is_retained_as_unclear_without_semantic_retry(tmp_path: Path) -> None:
    client = _FakeClient(malformed_judge=True)
    manifest = _run(tmp_path, client)

    assert manifest["summary"]["outcomes"]["UNCLEAR"] == 3
    assert len([call for call in client.calls if call["model"] == "nvidia/nemotron-judge-model"]) == 1
    raw = (tmp_path / "panel" / "private" / "raw_responses" / "subject.alpha.jsonl").read_text(encoding="utf-8")
    assert '"format_valid": false' in raw
    assert '"outcome": "UNCLEAR"' in raw


def test_subject_identity_mismatch_is_invalid_and_never_judged(tmp_path: Path) -> None:
    client = _WrongSubjectIdentityClient()
    manifest = _run(tmp_path, client)

    assert manifest["summary"]["outcomes"]["INVALID"] == 3
    assert manifest["summary"]["subject_model_identity_mismatches"] == 3
    assert not [call for call in client.calls if call["model"] == "nvidia/nemotron-judge-model"]
    assert manifest["complete"] is False


@pytest.mark.parametrize("status", [408, 429, 503])
def test_only_transport_http_failures_are_retried(tmp_path: Path, status: int) -> None:
    client = _RetryOnceClient(status)
    manifest = _run(tmp_path, client)

    assert manifest["complete"] is True
    ledger = (tmp_path / "panel" / "private" / "attempt_ledger.jsonl").read_text(encoding="utf-8")
    assert ledger.count('"event": "reserved_before_dispatch"') == 5
    assert ledger.count('"outcome": "failed"') == 1


def test_panel_file_requires_the_fixed_judge_contract(tmp_path: Path) -> None:
    registry_path, compatibility_path = _write_inputs(tmp_path)
    panel_path = tmp_path / "panel.json"
    panel_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "panel_id": "test-panel",
                "subject_target_ids": ["subject.alpha"],
                "judge_target_id": "some-other-judge",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(InferenceHubPart0PanelError, match="identity contract"):
        run_panel(
            registry_path=registry_path,
            compatibility_path=compatibility_path,
            output_dir=tmp_path / "bad-panel-output",
            client=_FakeClient(),
            panel_path=panel_path,
            limit_per_language=1,
        )
