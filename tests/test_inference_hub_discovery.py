import hashlib
import json
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from analysis.reconcile_inference_hub_routes import reconcile_routes
from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
    _NoRedirectHandler,
    _build_parser,
    _load_discovery_ledger,
    _reserve_discovery_attempt,
    capture_catalog,
    chat_probe_route,
    cli,
    probe_catalog_routes,
    smoke_verify_cohorts,
    smoke_verify_reconciled_candidates,
    smoke_verify_route,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _install_responses(
    monkeypatch: pytest.MonkeyPatch,
    responses: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_urlopen(
        request: Any,
        *,
        timeout: float,
        context: Any,
    ) -> _FakeHTTPResponse:
        body = json.loads(request.data) if request.data is not None else None
        calls.append(
            {
                "method": request.method,
                "url": request.full_url,
                "authorization": request.get_header("Authorization"),
                "body": body,
                "timeout": timeout,
            }
        )
        key = (request.method, request.full_url)
        assert key in responses
        return _FakeHTTPResponse(responses[key])

    monkeypatch.setattr(
        "experiments.misc.inference_hub_discovery._urlopen_no_redirect",
        fake_urlopen,
    )
    return calls


def _catalog_payloads() -> dict[tuple[str, str], dict[str, Any]]:
    base = "https://inference-api.nvidia.com/v1"
    return {
        ("GET", f"{base}/models"): {
            "object": "list",
            "data": [
                {"id": "aws/anthropic/claude-sonnet", "object": "model"},
                {"id": "us/azure/openai/gpt-5", "object": "model"},
            ],
        }
    }


def test_catalog_uses_virtual_key_models_route_and_sanitizes_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "nvapi-secret-test-value"
    calls = _install_responses(monkeypatch, _catalog_payloads())
    client = InferenceHubClient(api_key=secret)

    catalog = capture_catalog(client)

    assert catalog["route_count"] == 2
    assert {row["route"] for row in catalog["routes"]} == {
        "aws/anthropic/claude-sonnet",
        "us/azure/openai/gpt-5",
    }
    assert all(row["listed_by_models"] is True for row in catalog["routes"])
    assert all(
        row["chat_capability"] == "unverified_until_structured_smoke"
        for row in catalog["routes"]
    )
    serialized = json.dumps(catalog)
    assert secret not in serialized
    assert all(call["authorization"] == f"Bearer {secret}" for call in calls)
    assert [call["url"] for call in calls] == [
        "https://inference-api.nvidia.com/v1/models"
    ]


def test_smoke_verification_records_hashes_not_generated_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _catalog_payloads()
    payloads[("POST", "https://inference-api.nvidia.com/v1/chat/completions")] = {
        "id": "completion-id",
        "model": "aws/anthropic/claude-sonnet",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"ok":"OK"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
    }
    calls = _install_responses(monkeypatch, payloads)
    client = InferenceHubClient(api_key="test-key")
    catalog = capture_catalog(client)

    evidence = smoke_verify_route(
        client,
        catalog=catalog,
        route="aws/anthropic/claude-sonnet",
        max_tokens=8,
    )

    assert evidence["verification_status"] == "verified"
    assert evidence["requested_route"] == "aws/anthropic/claude-sonnet"
    assert evidence["provider_response_model"] == "aws/anthropic/claude-sonnet"
    assert evidence["route_source"] == "inference_hub_models_api"
    smoke = evidence["verification_evidence"]["smoke_test"]
    assert smoke["completed_at_utc"] == evidence["verified_at_utc"]
    assert smoke["request_id"] == "completion-id"
    assert smoke["response_model"] == "aws/anthropic/claude-sonnet"
    assert smoke["response_sha256"] == evidence["response"]["payload_sha256"]
    assert smoke["finish_reason"] == "stop"
    assert smoke["usage_sha256"] == evidence["response"]["usage_sha256"]
    assert smoke["generation_controls"] == {
        key: value
        for key, value in evidence["request"].items()
        if key not in {"prompt_sha256", "request_sha256"}
    }
    assert smoke["generation_controls"]["temperature"] == 0
    assert smoke["generation_controls"]["top_p"] == 1
    assert smoke["generation_controls"]["seed"] == 20260801
    assert smoke["generation_controls"]["structured_output"] is True
    assert evidence["response"]["usage"]["total_tokens"] == 12
    assert len(evidence["response"]["payload_sha256"]) == 64
    assert '{"ok":"OK"}' not in json.dumps(evidence)
    assert calls[-1]["body"] == {
        "max_tokens": 8,
        "messages": [
            {
                "content": 'Return exactly the JSON object {"ok":"OK"}.',
                "role": "user",
            }
        ],
        "model": "aws/anthropic/claude-sonnet",
        "temperature": 0,
        "top_p": 1,
        "seed": 20260801,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "inference_hub_route_smoke",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "string", "enum": ["OK"]}
                    },
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            },
        },
    }


def test_smoke_rejects_provider_model_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _catalog_payloads()
    payloads[("POST", "https://inference-api.nvidia.com/v1/chat/completions")] = {
        "id": "completion-id",
        "model": "silent-alias/other-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"ok":"OK"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 1},
    }
    _install_responses(monkeypatch, payloads)
    client = InferenceHubClient(api_key="test-key")
    catalog = capture_catalog(client)

    with pytest.raises(InferenceHubDiscoveryError, match="identity does not match"):
        smoke_verify_route(
            client,
            catalog=catalog,
            route="aws/anthropic/claude-sonnet",
        )


def test_minimal_chat_probe_asserts_no_optional_generation_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _catalog_payloads()
    payloads[("POST", "https://inference-api.nvidia.com/v1/chat/completions")] = {
        "id": "completion-minimal",
        "model": "aws/anthropic/claude-sonnet",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "OK"},
                "finish_reason": "length",
            }
        ],
    }
    calls = _install_responses(monkeypatch, payloads)
    client = InferenceHubClient(api_key="test-key")
    catalog = capture_catalog(client)

    evidence = chat_probe_route(
        client,
        catalog=catalog,
        route="aws/anthropic/claude-sonnet",
        max_tokens=8,
    )

    assert evidence["verification_status"] == "chat_callable"
    assert evidence["request"]["optional_generation_controls_asserted"] == []
    assert evidence["provider_response_model"] == "aws/anthropic/claude-sonnet"
    assert "content" not in evidence["response"]
    assert evidence["response"]["truncated"] is True
    assert evidence["response"]["usage"] is None
    assert calls[-1]["body"] == {
        "model": "aws/anthropic/claude-sonnet",
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 8,
        "stream": False,
    }


def test_full_catalog_probe_attempts_every_route_and_retains_rejections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _candidate_catalog()
    rendezvous = threading.Barrier(2)

    def probe(_client, *, catalog, route, max_tokens):
        del catalog, max_tokens
        rendezvous.wait(timeout=2)
        if route.endswith("model-b"):
            raise InferenceHubDiscoveryError("route is not a chat model")
        return {
            "requested_route": route,
            "verification_status": "chat_callable",
            "response": {"request_id": "completion-a"},
        }

    monkeypatch.setattr(
        "experiments.misc.inference_hub_discovery.chat_probe_route", probe
    )
    bundle = probe_catalog_routes(
        InferenceHubClient(api_key="test-key"),
        catalog=catalog,
        attempt_ledger_path=tmp_path / "catalog-attempts.json",
        max_workers=2,
    )

    assert bundle["status"] == "complete"
    assert bundle["attempted_route_count"] == 2
    assert bundle["chat_callable_route_count"] == 1
    assert bundle["rejected_route_count"] == 1
    assert bundle["chat_callable_routes"][0]["route"] == "openai/openai/model-a"
    assert bundle["rejected_routes"] == [
        {
            "route": "gcp/google/model-b",
            "failure_code": "minimal_chat_probe_failed",
        }
    ]
    assert len(bundle["run_attempt_ids"]) == 2
    assert bundle["discovery_attempt_ledger"]["record_count"] == 2
    assert bundle["execution"] == {
        "strategy": "bounded_thread_pool",
        "configured_max_workers": 2,
        "effective_worker_count": 2,
        "result_ordering": "route_lexicographic",
        "ledger_strategy": "locked_reservation_before_each_dispatch",
    }


def test_cohort_verification_covers_every_exact_target_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cohorts = {
        "current_sota": {
            "id": "current_sota",
            "version": "v1",
            "registry_version": "registry-v1",
            "registry_hash": "a" * 64,
            "routing_roster_hash": "c" * 64,
            "targets": [
                {
                    "id": "openai.current",
                    "provider": "inference_hub",
                    "upstream_provider": "openai",
                    "route": "us/openai/current",
                },
                {
                    "id": "anthropic.current",
                    "provider": "inference_hub",
                    "upstream_provider": "anthropic",
                    "route": "aws/anthropic/current",
                },
            ],
        },
        "historical": {
            "id": "historical",
            "version": "v1",
            "registry_version": "registry-v1",
            "registry_hash": "a" * 64,
            "routing_roster_hash": "c" * 64,
            "targets": [
                {
                    "id": "openai.historical",
                    "provider": "inference_hub",
                    "upstream_provider": "openai",
                    "route": "us/openai/historical",
                }
            ],
        },
    }
    monkeypatch.setattr(
        "experiments.misc.inference_hub_discovery.load_model_cohort",
        lambda cohort_id: cohorts[cohort_id],
    )
    calls: list[str] = []
    rendezvous = threading.Barrier(2)

    def verify(_client, *, catalog, route, max_tokens):
        del catalog
        calls.append(route)
        if route != "us/openai/historical":
            rendezvous.wait(timeout=2)
        return {
            "requested_route": route,
            "verification_status": "verified",
            "max_tokens": max_tokens,
            "verification_evidence": {
                "smoke_test": {"request_id": f"request-{len(calls)}"}
            },
        }

    monkeypatch.setattr(
        "experiments.misc.inference_hub_discovery.smoke_verify_route", verify
    )
    client = InferenceHubClient(api_key="test-key")

    bundle = smoke_verify_cohorts(
        client,
        catalog={
            "source_payload_sha256": "b" * 64,
            "routes": [
                {
                    "route": route,
                    "listed_by_models": True,
                    "chat_capability": "unverified_until_structured_smoke",
                }
                for route in (
                    "us/openai/current",
                    "aws/anthropic/current",
                    "us/openai/historical",
                    "other/chat-model",
                )
            ],
        },
        cohort_ids=["current_sota", "historical"],
        max_tokens=9,
        attempt_ledger_path=tmp_path / "discovery-attempts.json",
        max_workers=2,
    )

    assert set(calls) == {
        "us/openai/current",
        "aws/anthropic/current",
        "us/openai/historical",
    }
    assert bundle["target_count"] == 3
    assert [target["target_id"] for target in bundle["targets"]] == [
        "openai.current",
        "anthropic.current",
        "openai.historical",
    ]
    assert len(bundle["bundle_sha256"]) == 64
    assert bundle["routing_roster_sha256"] == "c" * 64
    assert bundle["status"] == "verified"
    assert bundle["verified_target_count"] == 3
    assert bundle["discovery_attempt_ledger"]["record_count"] == 3
    assert bundle["execution"]["effective_worker_count"] == 2
    assert bundle["execution"]["result_ordering"] == (
        "requested_cohort_then_registry_target"
    )
    assert bundle["catalog_census"][-1] == {
        "route": "other/chat-model",
        "decision": "excluded_outside_frozen_panel",
        "target_id": None,
    }


def test_cohort_verification_rejects_non_inference_hub_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "experiments.misc.inference_hub_discovery.load_model_cohort",
        lambda _cohort_id: {
            "id": "mixed",
            "version": "v1",
            "registry_version": "registry-v1",
            "registry_hash": "a" * 64,
            "routing_roster_hash": "c" * 64,
            "targets": [
                {
                    "id": "direct.openai",
                    "provider": "openai",
                    "upstream_provider": "openai",
                    "route": "gpt-direct",
                }
            ],
        },
    )

    with pytest.raises(InferenceHubDiscoveryError, match="not routed through"):
        smoke_verify_cohorts(
            InferenceHubClient(api_key="test-key"),
            catalog={},
            cohort_ids=["mixed"],
        )


def _candidate_catalog() -> dict[str, Any]:
    routes = ["openai/openai/model-a", "gcp/google/model-b"]
    return {
        "schema_version": 2,
        "captured_at_utc": "2026-08-02T00:00:00Z",
        "source_endpoints": ["/models"],
        "source_payload_sha256": {"models": "b" * 64},
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


def _candidate_registry() -> dict[str, Any]:
    return {
        "registry_version": "registry-v1",
        "targets": [
            {
                "id": "provider.model-a",
                "upstream_provider": "openai",
                "route": "model-a",
            },
            {
                "id": "provider.model-b",
                "upstream_provider": "google",
                "route": "model-b",
            },
            {
                "id": "provider.missing",
                "upstream_provider": "other",
                "route": "missing",
            },
        ],
        "cohorts": {
            "panel": {
                "targets": [
                    "provider.model-a",
                    "provider.model-b",
                    "provider.missing",
                ]
            }
        },
    }


def _candidate_reconciliation(catalog: dict[str, Any]) -> dict[str, Any]:
    return reconcile_routes(catalog=catalog, registry=_candidate_registry())


def test_reconciled_candidate_verification_retains_passes_and_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _candidate_catalog()
    reconciliation = _candidate_reconciliation(catalog)
    rendezvous = threading.Barrier(2)

    def verify(_client, *, catalog, route, max_tokens):
        del catalog, max_tokens
        rendezvous.wait(timeout=2)
        if route.endswith("model-b"):
            raise InferenceHubDiscoveryError("structured controls rejected")
        return {
            "requested_route": route,
            "verification_status": "verified",
            "verification_evidence": {
                "smoke_test": {"request_id": "completion-a"}
            },
        }

    monkeypatch.setattr(
        "experiments.misc.inference_hub_discovery.smoke_verify_route", verify
    )
    bundle = smoke_verify_reconciled_candidates(
        InferenceHubClient(api_key="test-key"),
        catalog=catalog,
        registry=_candidate_registry(),
        reconciliation=reconciliation,
        attempt_ledger_path=tmp_path / "candidate-attempts.json",
        max_workers=2,
    )

    assert bundle["status"] == "incomplete"
    assert bundle["candidate_target_count"] == 2
    assert bundle["verified_target_count"] == 1
    assert bundle["targets"][0]["route"] == "openai/openai/model-a"
    assert bundle["rejected_targets"] == [
        {
            "target_id": "provider.model-b",
            "route": "gcp/google/model-b",
            "failure_code": "candidate_route_verification_failed",
        }
    ]
    assert bundle["unresolved_targets"] == [
        {
            "target_id": "provider.missing",
            "planned_route": "missing",
            "status": "unresolved_no_exact_suffix",
        }
    ]
    assert bundle["automatic_registry_promotion"] is False
    assert bundle["discovery_attempt_ledger"]["record_count"] == 2
    assert bundle["execution"]["effective_worker_count"] == 2
    assert bundle["execution"]["result_ordering"] == (
        "reconciliation_resolution_order"
    )
    assert len(bundle["bundle_sha256"]) == 64


def test_reconciled_candidate_verification_rejects_catalog_drift(
    tmp_path: Path,
) -> None:
    catalog = _candidate_catalog()
    reconciliation = _candidate_reconciliation(catalog)
    catalog["source_payload_sha256"] = {"models": "c" * 64}

    with pytest.raises(InferenceHubDiscoveryError, match="current catalog"):
        smoke_verify_reconciled_candidates(
            InferenceHubClient(api_key="test-key"),
            catalog=catalog,
            registry=_candidate_registry(),
            reconciliation=reconciliation,
            attempt_ledger_path=tmp_path / "candidate-attempts.json",
        )


def test_reconciliation_is_recomputed_from_fixed_catalog_and_registry(
    tmp_path: Path,
) -> None:
    catalog = _candidate_catalog()
    catalog["routes"].append(
        {
            "route": "gcp/google/unrelated",
            "listed_by_models": True,
            "chat_capability": "unverified_until_structured_smoke",
        }
    )
    catalog["route_count"] = len(catalog["routes"])
    reconciliation = _candidate_reconciliation(catalog)
    reconciliation["resolutions"][0]["selected_candidate"] = "gcp/google/unrelated"
    reconciliation["resolutions"][0]["all_exact_suffix_candidates"] = [
        "gcp/google/unrelated"
    ]
    reconciliation.pop("report_sha256")
    reconciliation["report_sha256"] = hashlib.sha256(
        json.dumps(
            reconciliation,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(InferenceHubDiscoveryError, match="exactly reproduce"):
        smoke_verify_reconciled_candidates(
            InferenceHubClient(api_key="test-key"),
            catalog=catalog,
            registry=_candidate_registry(),
            reconciliation=reconciliation,
            attempt_ledger_path=tmp_path / "candidate-attempts.json",
        )


def test_full_catalog_rejection_retains_sanitized_response_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _candidate_catalog()
    catalog["routes"] = catalog["routes"][:1]
    catalog["route_count"] = 1
    client = InferenceHubClient(api_key="test-key")
    raw_content = "sensitive generated response"
    monkeypatch.setattr(
        client,
        "post",
        lambda *_args, **_kwargs: {
            "id": "completion-wrong-model",
            "model": "silent-alias/other-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": raw_content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 7},
        },
    )
    ledger_path = tmp_path / "catalog-attempts.json"

    bundle = probe_catalog_routes(
        client,
        catalog=catalog,
        attempt_ledger_path=ledger_path,
    )

    rejection = bundle["rejected_routes"][0]
    assert rejection["failure_code"] == "response_model_identity_mismatch"
    assert rejection["failure_evidence"]["request_id"] == "completion-wrong-model"
    assert rejection["failure_evidence"]["response_model"] == "silent-alias/other-model"
    assert rejection["failure_evidence"]["finish_reason"] == "stop"
    assert len(rejection["failure_evidence"]["response_sha256"]) == 64
    assert len(rejection["failure_evidence"]["content_sha256"]) == 64
    assert raw_content not in json.dumps(bundle)
    ledger = _load_discovery_ledger(ledger_path)
    assert ledger["records"][0]["request_id"] == "completion-wrong-model"
    assert ledger["records"][0]["response_model"] == "silent-alias/other-model"
    assert ledger["records"][0]["response_sha256"] == rejection["failure_evidence"][
        "response_sha256"
    ]


def test_discovery_ledger_concurrent_reservations_are_never_lost(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "concurrent-attempts.json"

    def reserve(index: int) -> str:
        return _reserve_discovery_attempt(
            ledger_path,
            target_id=f"target-{index}",
            route=f"route-{index}",
            request_body={"model": f"route-{index}"},
            max_tokens=1,
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        attempt_ids = list(pool.map(reserve, range(48)))

    ledger = _load_discovery_ledger(ledger_path)
    assert len(ledger["records"]) == 48
    assert {row["attempt_id"] for row in ledger["records"]} == set(attempt_ids)
    assert all(row["outcome"] == "reserved_before_dispatch" for row in ledger["records"])


def test_redirect_handler_refuses_to_create_redirect_request() -> None:
    request = urllib.request.Request(
        "https://inference-api.nvidia.com/v1/models",
        headers={"Authorization": "Bearer must-not-forward"},
    )
    redirected = _NoRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://attacker.invalid/collect",
    )
    assert redirected is None


@pytest.mark.parametrize(
    "argv",
    [
        [
            "verify-cohorts",
            "--cohort",
            "panel",
            "--output",
            "evidence.json",
            "--attempt-ledger",
            "ledger.json",
        ],
        [
            "verify-candidates",
            "--catalog-input",
            "catalog.json",
            "--registry-input",
            "registry.json",
            "--reconciliation",
            "routes.json",
            "--output",
            "evidence.json",
            "--attempt-ledger",
            "ledger.json",
        ],
        [
            "probe-catalog",
            "--catalog-input",
            "catalog.json",
            "--output",
            "evidence.json",
            "--attempt-ledger",
            "ledger.json",
        ],
    ],
)
def test_parallel_cli_has_positive_default_and_rejects_zero(
    argv: list[str],
) -> None:
    parser = _build_parser()
    assert parser.parse_args(argv).max_workers == 16
    assert parser.parse_args([*argv, "--max-workers", "7"]).max_workers == 7
    with pytest.raises(SystemExit):
        parser.parse_args([*argv, "--max-workers", "0"])


def test_smoke_refuses_route_absent_from_models_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _catalog_payloads()
    payloads[("GET", "https://inference-api.nvidia.com/v1/models")]["data"] = []
    _install_responses(monkeypatch, payloads)
    client = InferenceHubClient(api_key="test-key")
    catalog = capture_catalog(client)

    with pytest.raises(InferenceHubDiscoveryError, match="does not occur once"):
        smoke_verify_route(
            client,
            catalog=catalog,
            route="aws/anthropic/claude-sonnet",
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://inference-api.nvidia.com/v1",
        "https://integrate.api.nvidia.com/v1",
        "https://inference-api.nvidia.com/v1/chat/completions",
        "https://user:password@inference-api.nvidia.com/v1",
    ],
)
def test_client_rejects_wrong_trust_boundary(base_url: str) -> None:
    with pytest.raises(ValueError):
        InferenceHubClient(api_key="test-key", base_url=base_url)


def test_http_errors_do_not_echo_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "nvapi-never-echo-this"

    def fail_urlopen(*_args: object, **_kwargs: object) -> Any:
        raise urllib.error.HTTPError(
            url="https://inference-api.nvidia.com/v1/models",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(
        "experiments.misc.inference_hub_discovery._urlopen_no_redirect",
        fail_urlopen,
    )
    client = InferenceHubClient(api_key=secret)

    with pytest.raises(InferenceHubDiscoveryError) as captured:
        client.get("/models")
    assert secret not in str(captured.value)
    assert captured.value.failure_code == "http_error"
    assert captured.value.http_status == 401


def test_cli_missing_key_fails_cleanly_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Any,
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(
        "experiments.misc.inference_hub_discovery.load_dotenv",
        lambda: False,
    )

    exit_code = cli(["catalog", "--output", str(tmp_path / "catalog.json")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "NVIDIA_API_KEY is required" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
