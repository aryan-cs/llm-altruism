from copy import deepcopy

import pytest

from agents.agent_config import (
    _validate_model_registry,
    load_agent_config,
    load_all_model_options,
    load_experiment_model_options,
    load_endpoint_profile,
    load_model_cohort,
    load_model_registry,
    model_registry_hash,
    resolve_model_registry_entry,
)


def _registry_with_verified_first_target() -> dict:
    registry = deepcopy(load_model_registry())
    target = registry["targets"][0]
    target["verification_status"] = "verified"
    target["route_source"] = "inference_hub_models_api"
    target["verification_evidence"] = {
        "verified_at_utc": "2026-08-02T01:02:03+00:00",
        "discovery_sha256": "a" * 64,
        "smoke_test": {
            "completed_at_utc": "2026-08-02T01:03:04Z",
            "request_id": "request-123",
            "response_model": target["route"],
            "response_sha256": "b" * 64,
            "finish_reason": "stop",
            "usage_sha256": "c" * 64,
            "generation_controls": {
                "temperature": 0,
                "top_p": 1,
                "seed": 20260801,
                "max_tokens": 16,
                "stream": False,
                "structured_output": True,
                "json_schema_sha256": "d" * 64,
            },
        },
    }
    return registry


def test_load_agent_config_supports_comment_lines() -> None:
    load_agent_config.cache_clear()
    config = load_agent_config()

    assert "part_0" in config
    assert "part_1" in config
    assert "part_2" in config


def test_load_experiment_model_options_returns_enabled_models() -> None:
    load_agent_config.cache_clear()
    models = load_experiment_model_options("part_0")

    assert "ollama" in models
    assert "gpt-oss-safeguard:20b" in models["ollama"]
    assert "gpt-oss-safeguard:120b" in models["ollama"]


def test_load_all_model_options_combines_experiment_catalogs() -> None:
    load_agent_config.cache_clear()
    models = load_all_model_options()

    assert "openai" in models
    assert "gpt-4.1-mini" in models["openai"]


def test_part_1_model_catalog_matches_the_recorded_part_0_run() -> None:
    load_agent_config.cache_clear()
    part_1_models = load_experiment_model_options("part_1")

    assert part_1_models["ollama"] == [
        "gpt-oss:20b",
        "gpt-oss-safeguard:20b",
        "gurubot/gpt-oss-derestricted:20b",
        "llama2",
        "llama2-uncensored",
        "qwen2.5:7b",
        "qwen2.5:7b-instruct",
        "huihui_ai/qwen2.5-abliterate:7b",
        "huihui_ai/qwen2.5-abliterate:7b-instruct",
        "qwen3.5",
        "sorc/qwen3.5-instruct",
        "aratan/qwen3.5-uncensored:9b",
        "sorc/qwen3.5-instruct-uncensored",
    ]
    assert part_1_models["inference_hub"] == [
        target["model"] for target in load_model_cohort("current_sota")["targets"]
    ]


def test_part_2_still_matches_part_0_catalog() -> None:
    load_agent_config.cache_clear()
    part_0_models = load_experiment_model_options("part_0")
    part_2_models = load_experiment_model_options("part_2")

    assert part_2_models == part_0_models


def test_versioned_current_sota_cohort_contains_exact_inference_hub_roster() -> None:
    load_model_registry.cache_clear()
    cohort = load_model_cohort("current_sota")

    assert cohort["registry_version"] == "2026-08-01.2"
    assert cohort["version"] == "2026-08-01.2"
    assert [target["model"] for target in cohort["targets"]] == [
        "gpt-5.6-sol",
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "google/gemma-4-31b-it",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "deepseek-ai/deepseek-v4-pro",
        "qwen/qwen3-next-80b-a3b-thinking",
        "moonshotai/kimi-k2-thinking",
        "z-ai/glm-5.2",
        "mistralai/mistral-nemotron",
    ]
    assert {target["provider"] for target in cohort["targets"]} == {
        "inference_hub"
    }
    assert {target["verification_status"] for target in cohort["targets"]} == {
        "unverified"
    }
    assert {target["route_source"] for target in cohort["targets"]} == {
        "catalog_display_only"
    }


def test_historical_cohort_and_route_metadata_are_pinned() -> None:
    load_model_registry.cache_clear()
    cohort = load_model_cohort("historical")

    assert [target["model"] for target in cohort["targets"]] == [
        "gpt-3.5-turbo-0125",
        "gpt-4.1-2025-04-14",
        "gpt-5-2025-08-07",
        "gemini-2.5-pro",
        "google/gemma-3-27b-it",
        "openai/gpt-oss-120b",
    ]
    entry = resolve_model_registry_entry(
        "inference-hub",
        "openai/gpt-oss-120b",
    )
    assert entry is not None
    assert entry["upstream_provider"] == "openai"
    assert entry["route"] == "openai/gpt-oss-120b"
    assert entry["cohorts"] == ["historical"]
    assert entry["endpoint"]["credential_env"] == "NVIDIA_API_KEY"
    assert len(model_registry_hash()) == 64


def test_internal_and_public_nvidia_endpoint_profiles_are_separate() -> None:
    load_model_registry.cache_clear()

    internal = load_endpoint_profile("inference_hub")
    public = load_endpoint_profile("nvidia")

    assert internal["base_url_env"] == "INFERENCE_HUB_BASE_URL"
    assert "default_base_url" not in internal
    assert internal["credential_env"] == "NVIDIA_API_KEY"
    assert public == {
        "provider": "nvidia",
        "protocol": "openai_chat_completions",
        "base_url_env": "NVIDIA_NIM_BASE_URL",
        "default_base_url": "https://integrate.api.nvidia.com/v1",
        "credential_env": "NVIDIA_NIM_API_KEY",
    }


def test_verified_route_requires_authoritative_source_and_complete_evidence() -> None:
    registry = _registry_with_verified_first_target()

    validated = _validate_model_registry(registry)

    assert validated["targets"][0]["verification_status"] == "verified"


def test_verified_route_rejects_non_authoritative_source() -> None:
    registry = _registry_with_verified_first_target()
    registry["targets"][0]["route_source"] = "catalog_display_only"

    with pytest.raises(ValueError, match="verified route_source"):
        _validate_model_registry(registry)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("verified_at_utc", "2026-08-02T01:02:03", "UTC timestamp"),
        ("discovery_sha256", "not-a-digest", "SHA-256"),
    ],
)
def test_verified_route_rejects_invalid_discovery_evidence(
    field: str,
    value: str,
    message: str,
) -> None:
    registry = _registry_with_verified_first_target()
    registry["targets"][0]["verification_evidence"][field] = value

    with pytest.raises(ValueError, match=message):
        _validate_model_registry(registry)


def test_verified_route_rejects_smoke_response_model_mismatch() -> None:
    registry = _registry_with_verified_first_target()
    registry["targets"][0]["verification_evidence"]["smoke_test"][
        "response_model"
    ] = "different/backend"

    with pytest.raises(ValueError, match="must match the exact route"):
        _validate_model_registry(registry)


def test_verified_route_requires_nonempty_smoke_request_id() -> None:
    registry = _registry_with_verified_first_target()
    registry["targets"][0]["verification_evidence"]["smoke_test"][
        "request_id"
    ] = "  "

    with pytest.raises(ValueError, match="request_id must be a non-empty string"):
        _validate_model_registry(registry)


def test_unverified_route_cannot_carry_verification_evidence() -> None:
    registry = deepcopy(load_model_registry())
    registry["targets"][0]["verification_evidence"] = {
        "verified_at_utc": "2026-08-02T01:02:03Z"
    }

    with pytest.raises(ValueError, match="must not carry verification evidence"):
        _validate_model_registry(registry)
