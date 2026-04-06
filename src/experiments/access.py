"""Live model-access probing helpers for the interactive CLI."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.providers import ProviderError, RateLimitProviderError, TemporaryProviderError, get_provider

from .config import ModelSpec
from .runner import API_KEY_ENV, ENDPOINT_ENV, infer_provider_name

ROOT = Path(__file__).resolve().parents[2]
PROBE_MESSAGES = [
    {
        "role": "system",
        "content": "You are running a connectivity smoke test. Reply with only SAFE or UNSAFE.",
    },
    {
        "role": "user",
        "content": "Classify this harmless message: hello there",
    },
]
PROBE_MAX_TOKENS = 64
PROBE_MAX_ATTEMPTS = 3
PROBE_MAX_WAIT_SECONDS = 20.0


@dataclass(frozen=True)
class ModelAccessResult:
    """Outcome of probing whether a model is currently reachable."""

    spec: ModelSpec
    accessible: bool
    status: str


def spec_selector(spec: ModelSpec) -> str:
    """Return a stable `provider:model` selector for a model spec."""
    provider_name = spec.provider or infer_provider_name(spec.model)
    return f"{provider_name}:{spec.model}"


def _unique_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    """Deduplicate model specs while preserving order."""
    deduped: list[ModelSpec] = []
    seen: set[str] = set()
    for spec in specs:
        selector = spec_selector(spec)
        if selector in seen:
            continue
        seen.add(selector)
        deduped.append(spec)
    return deduped


def _missing_provider_requirements(provider_name: str) -> str | None:
    """Return a missing-config status string for a provider, if any."""
    missing = []
    key_env = API_KEY_ENV.get(provider_name, "")
    endpoint_env = ENDPOINT_ENV.get(provider_name, "")

    if key_env and not os.getenv(key_env):
        missing.append(key_env)
    if endpoint_env and not os.getenv(endpoint_env):
        missing.append(endpoint_env)

    if not missing:
        return None
    return "missing " + ", ".join(missing)


async def probe_model_access(spec: ModelSpec) -> ModelAccessResult:
    """Probe whether a model is reachable with the current environment."""
    load_dotenv(ROOT / ".env", override=False)

    provider_name = spec.provider or infer_provider_name(spec.model)
    missing_requirements = _missing_provider_requirements(provider_name)
    if missing_requirements:
        return ModelAccessResult(spec=spec, accessible=False, status=missing_requirements)

    api_key = os.getenv(API_KEY_ENV.get(provider_name, "")) or None
    base_url = os.getenv(ENDPOINT_ENV.get(provider_name, "")) or None

    try:
        client = get_provider(
            provider_name,
            model=spec.model,
            api_key=api_key,
            base_url=base_url,
        )
    except ProviderError as exc:
        return ModelAccessResult(spec=spec, accessible=False, status=str(exc))

    last_retryable_error: TemporaryProviderError | None = None

    try:
        for attempt in range(PROBE_MAX_ATTEMPTS):
            try:
                response = await client.complete(
                    messages=PROBE_MESSAGES,
                    temperature=0.0,
                    max_tokens=PROBE_MAX_TOKENS,
                )
                if response.content.strip():
                    return ModelAccessResult(spec=spec, accessible=True, status="verified")
                return ModelAccessResult(spec=spec, accessible=False, status="empty response")
            except RateLimitProviderError:
                return ModelAccessResult(spec=spec, accessible=True, status="rate limited")
            except TemporaryProviderError as exc:
                last_retryable_error = exc
                if attempt == PROBE_MAX_ATTEMPTS - 1:
                    break
                suggested_wait = getattr(exc, "retry_after_seconds", None)
                wait_seconds = max(1.0, suggested_wait or (2**attempt))
                await asyncio.sleep(min(PROBE_MAX_WAIT_SECONDS, wait_seconds))
            except ProviderError as exc:
                return ModelAccessResult(spec=spec, accessible=False, status=str(exc))
    finally:
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    if last_retryable_error is not None:
        return ModelAccessResult(spec=spec, accessible=False, status=str(last_retryable_error))
    return ModelAccessResult(spec=spec, accessible=False, status="probe failed")


async def probe_accessible_model_catalog(
    specs: list[ModelSpec],
) -> tuple[list[ModelSpec], dict[str, ModelAccessResult]]:
    """Return the subset of models that are currently reachable."""
    results: dict[str, ModelAccessResult] = {}
    accessible_specs: list[ModelSpec] = []

    for spec in _unique_specs(specs):
        result = await probe_model_access(spec)
        results[spec_selector(spec)] = result
        if result.accessible:
            accessible_specs.append(spec)

    return accessible_specs, results
