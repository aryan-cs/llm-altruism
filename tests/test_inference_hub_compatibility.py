import json
import threading
from pathlib import Path
from typing import Any

import pytest

from analysis.reconcile_inference_hub_routes import reconcile_routes
from experiments.misc.inference_hub_compatibility import (
    COMPATIBILITY_SEED,
    EXECUTION_PROFILE_ORDER,
    OPTIONAL_CONTROLS,
    REASONING_MAX_TOKENS_FLOOR,
    CompatibilityError,
    cli,
    main,
    output_token_budget,
    probe_compatibility,
)
from experiments.misc.inference_hub_discovery import (
    InferenceHubDiscoveryError,
    _load_discovery_ledger,
)


def _catalog(routes: list[str]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "captured_at_utc": "2026-08-01T00:00:00Z",
        "endpoint": "https://inference-api.nvidia.com/v1",
        "route_source": "inference_hub_models_api",
        "source_endpoints": ["/models"],
        "source_payload_sha256": {"models": "a" * 64},
        "route_count": len(routes),
        "routes": [
            {
                "route": route,
                "listed_by_models": True,
                "chat_capability": "unverified_until_structured_smoke",
            }
            for route in routes
        ],
    }


def _registry(model: str = "model-x") -> dict[str, Any]:
    return {
        "registry_version": "test-v1",
        "targets": [
            {
                "id": "target-x",
                "model": model,
                "route": f"planned/{model}",
                "upstream_provider": "test",
            }
        ],
        "cohorts": {"primary": {"targets": ["target-x"]}},
    }


def _response(
    route: str, *, structured: bool, reasoning: bool = False
) -> dict[str, Any]:
    text = '{"ok":"OK"}' if structured else "OK"
    message = (
        {"role": "assistant", "content": None, "reasoning_content": text}
        if reasoning
        else {"role": "assistant", "content": text}
    )
    return {
        "id": f"request-{route}",
        "model": route,
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }


class _FakeClient:
    base_url = "https://inference-api.nvidia.com/v1"

    def __init__(self, *, reject_temperature_for: str | None = None) -> None:
        self.reject_temperature_for = reject_temperature_for
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        assert path == "/chat/completions"
        with self._lock:
            self.calls.append(body)
        if body["model"] == self.reject_temperature_for and "temperature" in body:
            raise InferenceHubDiscoveryError(
                "provider rejected control",
                failure_code="http_error",
                http_status=400,
                evidence={
                    "error_body_sha256": "b" * 64,
                    "error_body_bytes": 37,
                    "provider_error_message_sha256": "c" * 64,
                },
            )
        return _response(
            body["model"],
            structured="response_format" in body,
            reasoning="response_format" not in body,
        )


def _inputs(routes: list[str], *, model: str = "model-x") -> tuple[dict[str, Any], ...]:
    catalog = _catalog(routes)
    registry = _registry(model)
    reconciliation = reconcile_routes(catalog=catalog, registry=registry)
    return catalog, registry, reconciliation


def test_selects_first_candidate_with_a_working_supported_control_profile(
    tmp_path: Path,
) -> None:
    preferred = "openai/openai/model-x"
    fallback = "nvidia/model-x"
    catalog, registry, reconciliation = _inputs([fallback, preferred])
    client = _FakeClient(reject_temperature_for=preferred)
    ledger_path = tmp_path / "attempts.json"

    evidence = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=ledger_path,
        requested_floor=48,
        max_workers=2,
    )

    target = evidence["targets"][0]
    assert target["frozen_candidate_order"] == [preferred, fallback]
    assert [row["route"] for row in target["candidates"]] == [preferred, fallback]
    assert target["selected_execution_candidate"] == preferred
    assert target["selected_execution_profile"]["route"] == preferred
    assert target["selected_execution_profile"]["controls"] == [
        "seed",
        "top_p",
        "structured_response",
    ]
    assert target["selection_basis"] == (
        "first_execution_compatible_in_reconciliation_frozen_order"
    )
    assert target["candidates"][0]["execution_compatible"] is True
    assert target["candidates"][1]["execution_compatible"] is True
    assert target["candidates"][0]["selected_execution_profile"]["controls"] == [
        "seed",
        "top_p",
        "structured_response",
    ]
    assert target["candidates"][1]["selected_execution_profile"]["controls"] == list(
        OPTIONAL_CONTROLS
    )
    assert [
        profile["controls"]
        for profile in target["candidates"][0]["tested_profiles"]
    ] == [
        list(EXECUTION_PROFILE_ORDER[0]),
        list(EXECUTION_PROFILE_ORDER[1]),
        list(EXECUTION_PROFILE_ORDER[2]),
        list(EXECUTION_PROFILE_ORDER[3]),
    ]
    assert len(target["candidates"][1]["tested_profiles"]) == 1
    assert len(client.calls) == 7
    assert all(call["max_tokens"] == 48 for call in client.calls)
    assert any(call.get("seed") == COMPATIBILITY_SEED for call in client.calls)
    assert any("response_format" in call for call in client.calls)
    failed = target["candidates"][0]["tested_profiles"][0]
    assert failed["evidence"]["http_status"] == 400
    assert failed["evidence"]["error_body_sha256"] == "b" * 64
    serialized = json.dumps(evidence)
    assert "provider rejected control" not in serialized
    assert "reasoning_content" in serialized
    assert "OK" not in serialized
    assert evidence["evidence_sha256"]
    ledger = _load_discovery_ledger(ledger_path)
    assert len(ledger["records"]) == 7
    assert all(
        record["outcome"] != "reserved_before_dispatch"
        for record in ledger["records"]
    )


def test_execution_profile_order_is_complete_and_descending_by_cardinality() -> None:
    assert len(EXECUTION_PROFILE_ORDER) == (2 ** len(OPTIONAL_CONTROLS)) - 1
    assert len(set(EXECUTION_PROFILE_ORDER)) == len(EXECUTION_PROFILE_ORDER)
    assert {control for profile in EXECUTION_PROFILE_ORDER for control in profile} == set(
        OPTIONAL_CONTROLS
    )
    cardinalities = [len(profile) for profile in EXECUTION_PROFILE_ORDER]
    assert cardinalities == sorted(cardinalities, reverse=True)
    assert all(
        tuple(control for control in OPTIONAL_CONTROLS if control in profile) == profile
        for profile in EXECUTION_PROFILE_ORDER
    )


def test_structured_response_is_optional_for_plain_part1_profile(
    tmp_path: Path,
) -> None:
    route = "anthropic/model-x"
    catalog, registry, reconciliation = _inputs([route])

    class NoStructuredResponseClient(_FakeClient):
        def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
            with self._lock:
                self.calls.append(body)
            if "response_format" in body:
                raise InferenceHubDiscoveryError(
                    "structured responses are unavailable",
                    failure_code="http_error",
                    http_status=400,
                    evidence={"error_body_sha256": "e" * 64},
                )
            return _response(body["model"], structured=False)

    client = NoStructuredResponseClient()
    evidence = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=tmp_path / "attempts.json",
        max_workers=1,
    )

    candidate = evidence["targets"][0]["candidates"][0]
    selected = candidate["selected_execution_profile"]
    assert candidate["execution_compatible"] is True
    assert selected["controls"] == ["seed", "temperature", "top_p"]
    assert selected["request_sha256"] == candidate["tested_profiles"][1][
        "request_sha256"
    ]
    assert [row["controls"] for row in candidate["tested_profiles"]] == [
        list(EXECUTION_PROFILE_ORDER[0]),
        list(EXECUTION_PROFILE_ORDER[1]),
    ]
    assert len(client.calls) == 3


def test_minimal_only_route_records_explicit_empty_selected_profile(
    tmp_path: Path,
) -> None:
    route = "anthropic/model-x"
    catalog, registry, reconciliation = _inputs([route])

    class MinimalOnlyClient(_FakeClient):
        def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
            with self._lock:
                self.calls.append(body)
            has_optional_control = any(
                key in body for key in ("seed", "temperature", "top_p", "response_format")
            )
            if has_optional_control:
                raise InferenceHubDiscoveryError(
                    "control combination rejected",
                    failure_code="http_error",
                    http_status=400,
                    evidence={"error_body_sha256": "f" * 64},
                )
            return _response(body["model"], structured=False)

    client = MinimalOnlyClient()
    evidence = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=tmp_path / "attempts.json",
        max_workers=1,
    )

    candidate = evidence["targets"][0]["candidates"][0]
    selected = candidate["selected_execution_profile"]
    assert candidate["execution_compatible"] is True
    assert selected["profile_id"] == "minimal"
    assert selected["controls"] == []
    assert selected["validation_source"] == "minimal_stage"
    assert selected["request_sha256"] == candidate["stages"]["minimal"][
        "request_sha256"
    ]
    assert candidate["profile_search_status"] == (
        "minimal_profile_selected_no_optional_controls"
    )
    assert [row["controls"] for row in candidate["tested_profiles"]] == [
        list(profile) for profile in EXECUTION_PROFILE_ORDER
    ]
    assert len(client.calls) == 1 + len(EXECUTION_PROFILE_ORDER)
    assert evidence["selected_count"] == 1


@pytest.mark.parametrize(
    "identity",
    [
        "openai/gpt-5.6",
        "moonshot/kimi-k2",
        "zai/glm-5",
        "deepseek/deepseek-v3",
        "qwen/qwen3",
        "nvidia/nemotron-3",
        "openai/gpt-oss-120b",
    ],
)
def test_reasoning_families_receive_2048_output_budget(identity: str) -> None:
    assert (
        output_token_budget(target_model=identity, route=identity, requested_floor=32)
        == REASONING_MAX_TOKENS_FLOOR
    )


def test_request_hash_resume_avoids_redispatch(tmp_path: Path) -> None:
    route = "nvidia/model-x"
    catalog, registry, reconciliation = _inputs([route])
    client = _FakeClient()
    ledger_path = tmp_path / "attempts.json"
    first = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=ledger_path,
        max_workers=1,
    )
    assert first["selected_count"] == 1
    assert len(client.calls) == 2

    second = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=ledger_path,
        max_workers=1,
    )

    assert len(client.calls) == 2
    stages = second["targets"][0]["candidates"][0]["stages"]
    assert all(stage["resumed_from_ledger"] is True for stage in stages.values())
    assert len(_load_discovery_ledger(ledger_path)["records"]) == 2


def test_transient_terminal_failure_is_retried_on_resume(tmp_path: Path) -> None:
    route = "nvidia/model-x"
    catalog, registry, reconciliation = _inputs([route])

    class ThrottledOnceClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.throttled = False

        def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
            if not self.throttled:
                self.throttled = True
                with self._lock:
                    self.calls.append(body)
                raise InferenceHubDiscoveryError(
                    "throttled",
                    failure_code="http_error",
                    http_status=429,
                    evidence={"error_body_sha256": "d" * 64},
                )
            return super().post(path, body)

    client = ThrottledOnceClient()
    ledger_path = tmp_path / "attempts.json"
    first = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=ledger_path,
        max_workers=1,
    )
    assert first["selected_count"] == 0
    assert len(client.calls) == 1

    second = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=ledger_path,
        max_workers=1,
    )
    assert second["selected_count"] == 1
    assert len(client.calls) == 3
    assert len(_load_discovery_ledger(ledger_path)["records"]) == 3


def test_exact_response_model_identity_is_required_and_later_stages_skip(
    tmp_path: Path,
) -> None:
    route = "nvidia/model-x"
    catalog, registry, reconciliation = _inputs([route])

    class WrongIdentityClient(_FakeClient):
        def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
            with self._lock:
                self.calls.append(body)
            return _response("model-x", structured=False)

    client = WrongIdentityClient()
    evidence = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=tmp_path / "attempts.json",
        max_workers=1,
    )
    candidate = evidence["targets"][0]["candidates"][0]
    assert len(client.calls) == 1
    assert candidate["stages"]["minimal"]["failure_code"] == (
        "response_model_identity_mismatch"
    )
    assert candidate["tested_profiles"] == []
    assert candidate["selected_execution_profile"] is None
    assert candidate["profile_search_status"] == "skipped_minimal_failed"


def test_tampered_reconciliation_is_rejected_before_network(tmp_path: Path) -> None:
    route = "nvidia/model-x"
    catalog, registry, reconciliation = _inputs([route])
    reconciliation["resolutions"][0]["all_exact_suffix_candidates"] = []
    client = _FakeClient()
    with pytest.raises(CompatibilityError, match="report_sha256"):
        probe_compatibility(
            client,
            catalog=catalog,
            registry=registry,
            reconciliation=reconciliation,
            ledger_path=tmp_path / "attempts.json",
        )
    assert client.calls == []


def test_catalog_endpoint_must_match_probe_endpoint(tmp_path: Path) -> None:
    route = "nvidia/model-x"
    catalog, registry, reconciliation = _inputs([route])
    client = _FakeClient()
    client.base_url = "https://integrate.api.nvidia.com/v1"

    with pytest.raises(CompatibilityError, match="endpoint"):
        probe_compatibility(
            client,
            catalog=catalog,
            registry=registry,
            reconciliation=reconciliation,
            ledger_path=tmp_path / "attempts.json",
        )
    assert client.calls == []


def test_cli_writes_bound_evidence_and_returns_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    route = "nvidia/model-x"
    catalog, registry, reconciliation = _inputs([route])
    catalog_path = tmp_path / "catalog.json"
    registry_path = tmp_path / "registry.json"
    reconciliation_path = tmp_path / "reconciliation.json"
    output_path = tmp_path / "compatibility.json"
    ledger_path = tmp_path / "attempts.json"
    for path, payload in (
        (catalog_path, catalog),
        (registry_path, registry),
        (reconciliation_path, reconciliation),
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
    client = _FakeClient()
    monkeypatch.setattr(
        "experiments.misc.inference_hub_compatibility._client_from_environment",
        lambda _timeout: client,
    )

    exit_code = main(
        [
            "--catalog",
            str(catalog_path),
            "--registry",
            str(registry_path),
            "--reconciliation",
            str(reconciliation_path),
            "--attempt-ledger",
            str(ledger_path),
            "--output",
            str(output_path),
            "--max-workers",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["selected_count"] == 1
    assert payload["inputs"]["catalog_file_sha256"]
    without_hash = {
        key: value for key, value in payload.items() if key != "evidence_sha256"
    }
    from experiments.misc.inference_hub_discovery import _sha256_json

    assert payload["evidence_sha256"] == _sha256_json(without_hash)


def test_cli_reports_invalid_input_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli(
        [
            "--catalog",
            str(tmp_path / "missing-catalog.json"),
            "--registry",
            str(tmp_path / "missing-registry.json"),
            "--reconciliation",
            str(tmp_path / "missing-reconciliation.json"),
            "--attempt-ledger",
            str(tmp_path / "attempts.json"),
            "--output",
            str(tmp_path / "compatibility.json"),
        ]
    )
    assert exit_code == 2
    assert "compatibility probe failed" in capsys.readouterr().err
