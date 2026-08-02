import json
import urllib.error
from typing import Any

import pytest

from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
    capture_catalog,
    cli,
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

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
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
        },
        ("GET", f"{base}/model/info"): {
            "data": [
                {
                    "model_name": "aws/anthropic/claude-sonnet",
                    "litellm_params": {"api_key": "must-not-be-persisted"},
                    "model_info": {
                        "mode": "chat",
                        "max_input_tokens": 200000,
                        "supported_openai_params": ["max_tokens", "temperature"],
                        "private_backend": "must-not-be-persisted",
                    },
                },
                {
                    "model_name": "us/azure/openai/gpt-5",
                    "model_info": {
                        "mode": "chat",
                        "max_input_tokens": 128000,
                    },
                },
            ]
        },
    }


def test_catalog_requires_exact_api_agreement_and_sanitizes_payload(
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
    assert all(row["listed_by_model_info"] is True for row in catalog["routes"])
    serialized = json.dumps(catalog)
    assert secret not in serialized
    assert "must-not-be-persisted" not in serialized
    assert all(call["authorization"] == f"Bearer {secret}" for call in calls)
    assert [call["url"] for call in calls] == [
        "https://inference-api.nvidia.com/v1/models",
        "https://inference-api.nvidia.com/v1/model/info",
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
        if key != "prompt_sha256"
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


def test_smoke_refuses_route_not_confirmed_by_both_catalog_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _catalog_payloads()
    payloads[("GET", "https://inference-api.nvidia.com/v1/model/info")]["data"] = []
    _install_responses(monkeypatch, payloads)
    client = InferenceHubClient(api_key="test-key")
    catalog = capture_catalog(client)

    with pytest.raises(InferenceHubDiscoveryError, match="absent from /model/info"):
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

    monkeypatch.setattr("urllib.request.urlopen", fail_urlopen)
    client = InferenceHubClient(api_key=secret)

    with pytest.raises(InferenceHubDiscoveryError) as captured:
        client.get("/models")
    assert secret not in str(captured.value)


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
