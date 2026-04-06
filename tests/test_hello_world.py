"""Tests for the Hello, World! smoke test script."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import sys

from src.providers import LLMResponse, RateLimitProviderError, UsageInfo

ROOT = Path(__file__).resolve().parents[1]


def _load_hello_world_module():
    """Import the root smoke-test script for direct helper testing."""
    module_path = ROOT / "hello_world.py"
    spec = importlib.util.spec_from_file_location("hello_world_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_check_model_hello_world_reports_missing_credentials_without_building_client(monkeypatch):
    """Missing env should short-circuit before any provider client is created."""
    module = _load_hello_world_module()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        module,
        "get_provider",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("client should not be created")),
    )

    result = asyncio.run(module.check_model_hello_world("openai:gpt-4o"))

    assert not result.reachable
    assert "OPENAI_API_KEY" in result.status


def test_check_model_hello_world_marks_exact_match(monkeypatch):
    """Exact Hello, World! responses should be classified separately."""
    module = _load_hello_world_module()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            return LLMResponse(
                content="Hello, World!",
                model="gpt-4o",
                provider="openai",
                usage=UsageInfo(),
                latency_ms=1.0,
                cost_usd=0.0,
            )

        async def close(self):
            return None

    monkeypatch.setattr(module, "get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(module.check_model_hello_world("openai:gpt-4o"))

    assert result.reachable
    assert result.exact_match
    assert result.status == "exact hello world"


def test_check_model_hello_world_marks_non_empty_response_as_reachable(monkeypatch):
    """A non-empty response still proves that the endpoint is working."""
    module = _load_hello_world_module()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            return LLMResponse(
                content="Hi there.",
                model="gpt-4o",
                provider="openai",
                usage=UsageInfo(),
                latency_ms=1.0,
                cost_usd=0.0,
            )

        async def close(self):
            return None

    monkeypatch.setattr(module, "get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(module.check_model_hello_world("openai:gpt-4o"))

    assert result.reachable
    assert not result.exact_match
    assert result.status == "responded"
    assert result.response_excerpt == "Hi there."


def test_check_model_hello_world_treats_rate_limit_as_reachable(monkeypatch):
    """Rate limits still prove that the configured endpoint is reachable."""
    module = _load_hello_world_module()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            raise RateLimitProviderError("too many requests", retry_after_seconds=1.0)

        async def close(self):
            return None

    monkeypatch.setattr(module, "get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(module.check_model_hello_world("openai:gpt-4o"))

    assert result.reachable
    assert not result.exact_match
    assert "rate limited" in result.status


def test_check_model_hello_world_treats_429_provider_error_as_reachable(monkeypatch):
    """Some providers stringify 429s instead of raising a typed rate-limit error."""
    module = _load_hello_world_module()
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_BASE_URL", "https://example.com/v1")

    class DummyProvider:
        async def complete(self, **_kwargs):
            raise module.ProviderError(
                "Unexpected error calling Google API: 429 RESOURCE_EXHAUSTED"
            )

        async def close(self):
            return None

    monkeypatch.setattr(module, "get_provider", lambda *args, **kwargs: DummyProvider())

    result = asyncio.run(module.check_model_hello_world("google:gemini-2.0-flash"))

    assert result.reachable
    assert not result.exact_match
    assert "reachable but rate limited" in result.status
