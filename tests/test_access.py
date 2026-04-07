"""Tests for live model-access probing helpers."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

from src.experiments.access import (
    ModelAccessResult,
    ModelExperimentReadinessResult,
    probe_model_experiment_readiness,
    probe_accessible_model_catalog,
    probe_model_access,
    spec_selector,
)
from src.experiments.config import ModelSpec
from src.experiments.runner import apply_model_max_token_floor
from src.providers import LLMResponse, RateLimitProviderError, UsageInfo
from src.providers.nvidia_provider import NVIDIAProvider

ROOT = Path(__file__).resolve().parents[1]


def _load_run_experiment_module():
    """Import the CLI module for direct helper testing."""
    module_path = ROOT / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_model_access_reports_missing_credentials_without_creating_client(monkeypatch):
    """Missing provider credentials should short-circuit before any client is built."""
    monkeypatch.setattr("src.experiments.access.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        "src.experiments.access.get_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("client should not be created")),
    )

    result = asyncio.run(probe_model_access(ModelSpec(model="gpt-4o", provider="openai")))

    assert not result.accessible
    assert "OPENAI_API_KEY" in result.status


def test_probe_model_access_marks_successful_response_as_verified(monkeypatch):
    """A successful provider response should mark the model as accessible."""
    monkeypatch.setattr("src.experiments.access.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            return LLMResponse(
                content="SAFE",
                model="gpt-4o",
                provider="openai",
                usage=UsageInfo(),
                latency_ms=1.0,
                cost_usd=0.0,
            )

        async def close(self):
            return None

    monkeypatch.setattr("src.experiments.access.get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(probe_model_access(ModelSpec(model="gpt-4o", provider="openai")))

    assert result.accessible
    assert result.status == "verified"


def test_probe_model_access_treats_rate_limit_as_accessible(monkeypatch):
    """Rate limits still prove that the credentials and model access are valid."""
    monkeypatch.setattr("src.experiments.access.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            raise RateLimitProviderError("too many requests", retry_after_seconds=1.0)

        async def close(self):
            return None

    monkeypatch.setattr("src.experiments.access.get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(probe_model_access(ModelSpec(model="gpt-4o", provider="openai")))

    assert result.accessible
    assert result.status == "rate limited"


def test_probe_model_experiment_readiness_accepts_explicit_json_action(monkeypatch):
    """Experiment readiness should require an explicit action, not fuzzy inference."""
    monkeypatch.setattr("src.experiments.access.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            return LLMResponse(
                content='{"action":"cooperate","reasoning":"Testing explicit action output."}',
                model="gpt-4o",
                provider="openai",
                usage=UsageInfo(),
                latency_ms=1.0,
                cost_usd=0.0,
            )

        async def close(self):
            return None

    monkeypatch.setattr("src.experiments.access.get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(
        probe_model_experiment_readiness(ModelSpec(model="gpt-4o", provider="openai"))
    )

    assert result.ready
    assert result.parsed_action == "cooperate"
    assert "verified" in result.status


def test_probe_model_experiment_readiness_rejects_ambiguous_action_output(monkeypatch):
    """Long reasoning that only mentions actions should not count as experiment-ready."""
    monkeypatch.setattr("src.experiments.access.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            return LLMResponse(
                content=(
                    "In a Prisoner's Dilemma, cooperation can build trust, but defection is "
                    "often individually dominant in the final round."
                ),
                model="gpt-4o",
                provider="openai",
                usage=UsageInfo(),
                latency_ms=1.0,
                cost_usd=0.0,
            )

        async def close(self):
            return None

    monkeypatch.setattr("src.experiments.access.get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(
        probe_model_experiment_readiness(ModelSpec(model="gpt-4o", provider="openai"))
    )

    assert not result.ready
    assert result.parsed_action is None
    assert result.status == "ambiguous action output"


def test_probe_model_experiment_readiness_accepts_decision_key(monkeypatch):
    """Some providers emit `decision` instead of `action`; accept that when explicit."""
    monkeypatch.setattr("src.experiments.access.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            return LLMResponse(
                content='{"decision":"defect"}',
                model="gpt-4o",
                provider="openai",
                usage=UsageInfo(),
                latency_ms=1.0,
                cost_usd=0.0,
            )

        async def close(self):
            return None

    monkeypatch.setattr("src.experiments.access.get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(
        probe_model_experiment_readiness(ModelSpec(model="gpt-4o", provider="openai"))
    )

    assert result.ready
    assert result.parsed_action == "defect"


def test_nvidia_role_restricted_models_normalize_messages():
    provider = NVIDIAProvider(model="google/gemma-3-1b-it", api_key="test-key")

    normalized = provider._normalize_role_restricted_messages(
        [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "History block"},
            {"role": "user", "content": "Current prompt"},
        ]
    )

    assert normalized == [
        {"role": "user", "content": "System instructions\n\nHistory block\n\nCurrent prompt"},
    ]


def test_nvidia_chatqa_models_preserve_context_role():
    provider = NVIDIAProvider(model="nvidia/llama3-chatqa-1.5-8b", api_key="test-key")

    normalized = provider._normalize_role_restricted_messages(
        [
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "History block"},
            {"role": "user", "content": "Current prompt"},
        ]
    )

    assert normalized == [
        {"role": "context", "content": "System instructions"},
        {"role": "user", "content": "History block\n\nCurrent prompt"},
    ]


def test_model_max_token_floor_applies_to_verbose_reasoning_models():
    assert apply_model_max_token_floor("z-ai/glm4.7", 300) == 768
    assert apply_model_max_token_floor("moonshotai/kimi-k2-thinking", 300) == 1024
    assert apply_model_max_token_floor("deepseek-ai/deepseek-v3.2", 300) == 300


def test_probe_accessible_model_catalog_filters_out_inaccessible_models(monkeypatch):
    """The catalog helper should only return specs that passed the live probe."""
    good = ModelSpec(model="gpt-4o", provider="openai")
    bad = ModelSpec(model="gpt-4o-mini", provider="openai")

    async def fake_probe(spec: ModelSpec) -> ModelAccessResult:
        return ModelAccessResult(
            spec=spec,
            accessible=spec.model == "gpt-4o",
            status="verified" if spec.model == "gpt-4o" else "missing OPENAI_API_KEY",
        )

    monkeypatch.setattr("src.experiments.access.probe_model_access", fake_probe)

    accessible, results = asyncio.run(probe_accessible_model_catalog([good, bad, good]))

    assert accessible == [good]
    assert results[spec_selector(good)].accessible
    assert not results[spec_selector(bad)].accessible


def test_verify_selected_models_access_fails_fast_for_inaccessible_model(monkeypatch):
    """CLI preflight should reject models that failed the startup access tests."""
    module = _load_run_experiment_module()
    bad_selector = "cerebras:gpt-oss-120b"
    access_results = {
        bad_selector: ModelAccessResult(
            spec=ModelSpec(model="gpt-oss-120b", provider="cerebras"),
            accessible=False,
            status="API returned status 404: model_not_found",
        )
    }

    try:
        module.verify_selected_models_access(
            [bad_selector],
            dry_run=False,
            access_results=access_results,
        )
    except ValueError as exc:
        assert bad_selector in str(exc)
        assert "startup access tests" in str(exc)
    else:
        raise AssertionError("Expected live preflight to fail for an inaccessible model")


def test_verify_selected_models_access_skips_preflight_for_dry_runs():
    """Dry runs should not perform live access validation."""
    module = _load_run_experiment_module()
    assert (
        module.verify_selected_models_access(
            ["cerebras:gpt-oss-120b"],
            dry_run=True,
            access_results={},
        )
        is None
    )
