"""Tests for live model-access probing helpers."""

from __future__ import annotations

import asyncio

from src.experiments.access import (
    ModelAccessResult,
    probe_accessible_model_catalog,
    probe_model_access,
    spec_selector,
)
from src.experiments.config import ModelSpec
from src.providers import LLMResponse, RateLimitProviderError, UsageInfo


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
