"""Tests for live model-access probing helpers."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

from src.experiments.access import (
    ModelAccessResult,
    probe_accessible_model_catalog,
    probe_model_access,
    spec_selector,
)
from src.experiments.config import ModelSpec
from src.providers import LLMResponse, RateLimitProviderError, UsageInfo

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
    """CLI preflight should reject bad live selections before the run starts."""
    module = _load_run_experiment_module()
    bad_selector = "cerebras:gpt-oss-120b"

    async def fake_probe(_specs):
        spec = ModelSpec(model="gpt-oss-120b", provider="cerebras")
        return {
            bad_selector: ModelAccessResult(
                spec=spec,
                accessible=False,
                status="API returned status 404: model_not_found",
            )
        }

    monkeypatch.setattr(module, "probe_model_access_results", fake_probe)

    try:
        module.verify_selected_models_access([bad_selector], dry_run=False)
    except ValueError as exc:
        assert bad_selector in str(exc)
        assert "before the run started" in str(exc)
    else:
        raise AssertionError("Expected live preflight to fail for an inaccessible model")


def test_verify_selected_models_access_skips_preflight_for_dry_runs():
    """Dry runs should not perform live access validation."""
    module = _load_run_experiment_module()
    assert module.verify_selected_models_access(["cerebras:gpt-oss-120b"], dry_run=True) is None
