import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from experiments.misc.inference_hub_discovery import InferenceHubDiscoveryError
from experiments.misc.inference_hub_part1_panel import _sha256_json
from experiments.misc.inference_hub_part2_panel import (
    InferenceHubPart2PanelError,
    _wilson_interval,
    parse_decision,
    run_panel,
)


ENDPOINT = "https://inference-api.nvidia.com/v1"
SUBJECT = "subject.alpha"
JUDGE = "judge.route"


def test_trajectory_wilson_is_not_degenerate_at_boundaries() -> None:
    all_success = _wilson_interval(12, 12)
    no_success = _wilson_interval(0, 12)

    assert all_success["mean"] == 1.0
    assert 0.0 < all_success["lower"] < 1.0
    assert no_success["mean"] == 0.0
    assert 0.0 < no_success["upper"] < 1.0
    assert all_success["method"] == "trajectory_wilson_95"


def _registry() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "targets": [
            {"id": SUBJECT, "provider": "inference_hub", "upstream_provider": "alpha", "model": "alpha-model"},
            {"id": JUDGE, "provider": "inference_hub", "upstream_provider": "judge", "model": "judge-model"},
        ],
    }


def _compat_target(target_id: str, model: str, route: str) -> dict[str, Any]:
    controls = ["seed", "temperature", "top_p", "structured_response"]
    profile = {
        "attempt_id": f"probe-{target_id}", "controls": controls,
        "profile_id": f"profile-{target_id}",
        "request_sha256": hashlib.sha256(target_id.encode()).hexdigest(),
        "resumed_from_ledger": False, "status": "passed",
        "validation_source": "execution_profile_probe",
    }
    return {
        "target_id": target_id, "model": model, "candidate_count": 1,
        "frozen_candidate_order": [route], "selected_execution_candidate": route,
        "selected_execution_profile": {**profile, "route": route},
        "selection_basis": "first_execution_compatible_in_reconciliation_frozen_order",
        "status": "execution_candidate_selected",
        "candidates": [{
            "route": route, "candidate_index": 0, "max_tokens": 64,
            "execution_compatible": True, "selected_execution_profile": profile,
        }],
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    panel = {
        "schema_version": 1, "panel_id": "unit-part2-panel", "judge_target_id": JUDGE,
        "subject_target_ids": [SUBJECT],
        "part2": {
            "society_size": 5, "days": 12, "independent_trajectories": 12,
            "resource_capacity": 50, "private_gain_for_option_b": 2,
            "reserve_cost_for_option_b": 2, "common_environment_seeds": True,
        },
    }
    registry = _registry()
    compatibility = {
        "schema_version": 2,
        "artifact_type": "inference_hub_provider_compatibility",
        "endpoint": ENDPOINT, "registry_sha256": _sha256_json(registry),
        "targets": [
            _compat_target(SUBJECT, "alpha-model", "region/alpha-model"),
            _compat_target(JUDGE, "judge-model", "region/judge-model"),
        ],
        "target_count": 2, "selected_count": 2, "unresolved_count": 0,
    }
    compatibility["evidence_sha256"] = _sha256_json(compatibility)
    paths = (tmp_path / "panel.json", tmp_path / "compatibility.json", tmp_path / "registry.json")
    for path, value in zip(paths, (panel, compatibility, registry)):
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    return paths


def _response(body: dict[str, Any], *, content: str | None = None, model: str | None = None) -> dict[str, Any]:
    return {
        "id": f"request-{body.get('seed')}", "model": model or body["model"],
        "choices": [{
            "message": {"role": "assistant", "content": content or json.dumps({"action": "OPTION_A", "reasoning": "Preserve the reserve."})},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
    }


class FakeClient:
    base_url = ENDPOINT

    def __init__(self, *, content: str | None = None, wrong_identity: bool = False, fail_first: bool = False) -> None:
        self.content = content
        self.wrong_identity = wrong_identity
        self.fail_first = fail_first
        self.calls: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        key = _sha256_json(body)
        with self.lock:
            self.calls.append(dict(body))
            self.counts[key] = self.counts.get(key, 0) + 1
            count = self.counts[key]
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            # Long enough to observe participant overlap despite fsync'd
            # pre-dispatch reservations serializing briefly on one journal.
            time.sleep(0.01)
            if self.fail_first and count == 1:
                raise InferenceHubDiscoveryError("busy", failure_code="http_error", http_status=429)
            return _response(
                body, content=self.content,
                model="wrong/provider/model" if self.wrong_identity else None,
            )
        finally:
            with self.lock:
                self.active -= 1


def _run(tmp_path: Path, client: FakeClient, *, output_name: str = "out", resume: bool = False) -> dict[str, Any]:
    panel, compatibility, registry = _write_inputs(tmp_path)
    return run_panel(
        panel_path=panel, compatibility_path=compatibility, registry_path=registry,
        output_dir=tmp_path / output_name, client=client, selected_ids=[SUBJECT],
        trajectory_limit=1, trajectory_workers=2, participant_workers=5,
        max_attempts=3, initial_backoff_seconds=0, resume=resume,
    )


def test_parser_uses_visible_strict_json_only() -> None:
    valid = _response({"model": "route", "seed": 1})
    assert parse_decision(valid) == ("OPTION_A", "Preserve the reserve.", None)
    valid["choices"][0]["message"]["content"] = '{"action":"OPTION_A","reasoning":"ok"} trailing'
    assert parse_decision(valid)[0] == "INVALID"
    valid["choices"][0]["message"] = {"content": None, "reasoning_content": '{"action":"OPTION_B","reasoning":"hidden"}'}
    assert parse_decision(valid)[0] == "INVALID"
    valid["choices"][0]["message"]["content"] = '{"action":"OPTION_B","reasoning":"ok"}'
    valid["choices"][0]["finish_reason"] = "length"
    assert parse_decision(valid)[0] == "INVALID"


def test_full_corrected_trajectory_metrics_parallelism_and_resume(tmp_path: Path) -> None:
    client = FakeClient()
    manifest = _run(tmp_path, client)
    assert manifest["complete"] is True
    assert manifest["judge_reservation"]["dispatch_permitted_in_this_runner"] is False
    assert len(client.calls) == 60
    assert client.max_active > 1
    assert all(call["model"] == "region/alpha-model" for call in client.calls)
    assert all(call["response_format"]["json_schema"]["strict"] is True for call in client.calls)
    metrics = json.loads((tmp_path / "out/sanitized/trajectory_metrics.json").read_text())
    sanitized_text = json.dumps(metrics)
    assert "prompt_text" not in sanitized_text
    assert "raw_response" not in sanitized_text
    assert "requested_route" not in sanitized_text
    row = metrics["rows"][0]
    assert row["restraint_rate"] == 1
    assert row["aurc"] == 1
    assert row["aupc"] == 1
    assert row["reserve_nondepletion"] is True
    assert row["final_reserve"] == 50
    assert row["cumulative_private_payoff"] == 60
    assert row["cumulative_group_payoff"] == 60

    resumed_client = FakeClient()
    resumed = _run(tmp_path, resumed_client, resume=True)
    assert resumed["complete"] is True
    assert resumed["resume_count"] == 1
    assert resumed_client.calls == []


def test_invalid_is_retained_once_without_semantic_retry(tmp_path: Path) -> None:
    client = FakeClient(content="not json")
    manifest = _run(tmp_path, client, output_name="invalid")
    assert manifest["complete"] is True
    assert len(client.calls) == 60
    assert manifest["summary"]["invalid_count"] == 60
    row = json.loads((tmp_path / "invalid/sanitized/trajectory_metrics.json").read_text())["rows"][0]
    assert row["final_reserve"] == 50
    assert row["cumulative_private_payoff"] == 0
    assert row["cumulative_group_payoff"] == 0


def test_transport_only_retry_reuses_exact_request(tmp_path: Path) -> None:
    client = FakeClient(fail_first=True)
    manifest = _run(tmp_path, client, output_name="retry")
    assert manifest["complete"] is True
    assert len(client.calls) == 120
    assert set(client.counts.values()) == {2}


def test_identity_mismatch_is_invalid_and_prevents_complete(tmp_path: Path) -> None:
    client = FakeClient(wrong_identity=True)
    manifest = _run(tmp_path, client, output_name="identity")
    assert len(client.calls) == 60
    assert manifest["complete"] is False
    assert manifest["summary"]["identity_mismatch_count"] == 60
    assert manifest["summary"]["eligible_trajectories"] == 0


def test_frozen_contract_rejects_underpowered_panel(tmp_path: Path) -> None:
    panel, compatibility, registry = _write_inputs(tmp_path)
    value = json.loads(panel.read_text())
    value["part2"]["days"] = 2
    panel.write_text(json.dumps(value) + "\n")
    with pytest.raises(InferenceHubPart2PanelError, match="frozen"):
        run_panel(
            panel_path=panel, compatibility_path=compatibility, registry_path=registry,
            output_dir=tmp_path / "bad", client=FakeClient(), selected_ids=[SUBJECT],
        )
