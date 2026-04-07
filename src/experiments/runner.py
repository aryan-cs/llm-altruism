"""Shared experiment runner infrastructure and dispatcher."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agents.base import Agent
from src.agents.memory import Memory
from src.agents.prompts import load_prompt
from src.providers import (
    LLMResponse,
    ProviderError,
    RateLimitProviderError,
    TemporaryProviderError,
    UsageInfo,
    get_provider,
)
from src.utils.cost_tracker import CostTracker
from src.utils.logging import ExperimentLogger

from .config import ExperimentSettings, ModelSpec, PromptVariantConfig

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "ollama": "",
}

ENDPOINT_ENV = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
    "google": "GOOGLE_BASE_URL",
    "xai": "XAI_BASE_URL",
    "cerebras": "CEREBRAS_BASE_URL",
    "openrouter": "OPENROUTER_BASE_URL",
    "nvidia": "NVIDIA_BASE_URL",
    "ollama": "OLLAMA_BASE_URL",
}

JSON_OUTPUT_CONTRACT = (
    "Output contract: return only one compact JSON object and nothing else. "
    "Do not include markdown, XML tags such as <think>, analysis before the JSON, "
    "or commentary after the JSON. If a reasoning field is requested, keep it to a "
    "single short sentence inside the JSON object."
)


class BudgetExceededError(RuntimeError):
    """Raised when an experiment exceeds its configured budget."""


class ModelUnavailableError(RuntimeError):
    """Raised when a provider/model is skipped because config or endpoint is unavailable."""


def infer_provider_name(model: str) -> str:
    """Infer provider name from a model identifier without instantiating a client."""
    mapping = {
        "gpt-": "openai",
        "o1": "openai",
        "o3": "openai",
        "claude-": "anthropic",
        "gemini-": "google",
        "grok-": "xai",
        "llama-": "cerebras",
        "nemotron-": "nvidia",
    }
    for prefix, provider_name in mapping.items():
        if model.startswith(prefix):
            return provider_name
    return "openrouter"


def apply_response_format_contract(
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Append a strict output contract when machine-readable JSON is required."""
    if not response_format or response_format.get("type") != "json_object":
        return messages

    updated = [dict(message) for message in messages]
    if updated and updated[0].get("role") == "system":
        existing = updated[0].get("content", "").rstrip()
        updated[0]["content"] = f"{existing}\n\n{JSON_OUTPUT_CONTRACT}" if existing else JSON_OUTPUT_CONTRACT
        return updated

    updated.insert(0, {"role": "system", "content": JSON_OUTPUT_CONTRACT})
    return updated


class ResponseCache:
    """Simple file-backed cache keyed by model, prompt, and parameters."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self,
        *,
        model: str,
        provider: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> Path:
        payload = {
            "model": model,
            "provider": provider,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def load(self, **kwargs: Any) -> LLMResponse | None:
        """Load a cached response if present."""
        path = self._cache_path(**kwargs)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse.model_validate(data)

    def save(self, response: LLMResponse, **kwargs: Any) -> None:
        """Persist a response to the cache."""
        path = self._cache_path(**kwargs)
        path.write_text(response.model_dump_json(indent=2), encoding="utf-8")


class BaseExperimentRunner(ABC):
    """Shared functionality for all experiment runners."""

    def __init__(
        self,
        config: ExperimentSettings,
        *,
        results_dir: str = "results",
        dry_run: bool = False,
        run_metadata: dict[str, Any] | None = None,
    ):
        load_dotenv()
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.run_metadata = run_metadata or {}
        self.cost_tracker = CostTracker()
        self.logger = ExperimentLogger(results_dir=str(self.results_dir))
        self.cache = ResponseCache(self.results_dir.parent / ".cache" / "responses")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.experiment_id = f"{self.config.name}-{timestamp}"
        self.logger.start_experiment(
            self.experiment_id,
            {
                "experiment": self.config.model_dump(mode="json"),
                "run_metadata": self.run_metadata,
            },
        )
        self._providers: dict[tuple[str, str], Any] = {}
        self._provider_backoff_until: dict[str, float] = {}
        self.skipped_models: dict[str, str] = {}
        self.skipped_trials: list[dict[str, Any]] = []

    def spec_key(self, spec: ModelSpec) -> str:
        """Return a stable key for a model specification."""
        provider_name = spec.provider or infer_provider_name(spec.model)
        return f"{provider_name}:{spec.model}"

    def mark_model_unavailable(self, spec: ModelSpec, reason: str) -> None:
        """Record that a model should be skipped for the rest of the run."""
        self.skipped_models[self.spec_key(spec)] = reason

    def provider_backoff_key(self, spec: ModelSpec) -> str:
        """Return the shared retry bucket for a provider."""
        return spec.provider or infer_provider_name(spec.model)

    def get_skip_reason(self, spec: ModelSpec) -> str | None:
        """Return the reason a model should be skipped, if any."""
        if self.dry_run:
            return None

        key = self.spec_key(spec)
        if key in self.skipped_models:
            return self.skipped_models[key]

        provider_name = spec.provider or infer_provider_name(spec.model)
        key_env = API_KEY_ENV.get(provider_name, "")
        endpoint_env = ENDPOINT_ENV.get(provider_name, "")

        if key_env and not os.getenv(key_env):
            reason = f"missing {key_env}"
            self.mark_model_unavailable(spec, reason)
            return reason

        if endpoint_env and not os.getenv(endpoint_env):
            reason = f"missing {endpoint_env}"
            self.mark_model_unavailable(spec, reason)
            return reason

        return None

    def get_provider_credentials(self, provider_name: str) -> tuple[str | None, str | None]:
        """Resolve API key and endpoint for a provider from environment variables."""
        key_env = API_KEY_ENV.get(provider_name, "")
        endpoint_env = ENDPOINT_ENV.get(provider_name, "")
        api_key = os.getenv(key_env) if key_env else None
        base_url = os.getenv(endpoint_env) if endpoint_env else None
        return api_key or None, base_url or None

    def record_skipped_trial(self, data: dict[str, Any]) -> None:
        """Append a skipped-trial record."""
        if not isinstance(self.skipped_trials, list):
            self.skipped_trials = []
        self.skipped_trials.append(data)

    async def wait_for_provider_backoff(self, provider_key: str) -> None:
        """Sleep until a provider's retry window has elapsed."""
        while True:
            resume_at = self._provider_backoff_until.get(provider_key)
            if resume_at is None:
                return
            delay = resume_at - time.monotonic()
            if delay <= 0:
                self._provider_backoff_until.pop(provider_key, None)
                return
            await asyncio.sleep(delay)

    def set_provider_backoff(self, provider_key: str, delay_seconds: float) -> None:
        """Extend a provider-wide retry window."""
        if delay_seconds <= 0:
            return
        resume_at = time.monotonic() + delay_seconds
        self._provider_backoff_until[provider_key] = max(
            self._provider_backoff_until.get(provider_key, 0.0),
            resume_at,
        )

    def compute_retry_delay(self, *, attempt: int, suggested_delay: float | None) -> float:
        """Compute the next retry delay, respecting provider hints when available."""
        if suggested_delay is not None and suggested_delay > 0:
            return max(1.0, suggested_delay)

        base_delay = min(
            self.config.parameters.initial_retry_delay_seconds * (2**attempt),
            self.config.parameters.max_retry_delay_seconds,
        )
        jitter = min(5.0, max(0.25, base_delay * 0.1))
        return min(
            self.config.parameters.max_retry_delay_seconds,
            base_delay + random.uniform(0.0, jitter),
        )

    def build_agent(
        self,
        *,
        agent_id: str,
        spec: ModelSpec,
        prompt_variant: PromptVariantConfig,
        temperature: float,
    ) -> Agent:
        """Construct an Agent instance from a model spec and prompt variant."""
        provider_name = spec.provider or infer_provider_name(spec.model)
        framing = load_prompt(prompt_variant.framing) if prompt_variant.framing else None
        persona = load_prompt(prompt_variant.persona) if prompt_variant.persona else None
        return Agent(
            agent_id=agent_id,
            model=spec.model,
            provider_name=provider_name,
            system_prompt=load_prompt(prompt_variant.system_prompt),
            temperature=temperature,
            framing=framing,
            persona=persona,
            memory=Memory(
                mode=self.config.history.to_memory_mode(),
                window_size=self.config.history.window_size or 10,
            ),
        )

    async def request_completion(
        self,
        *,
        spec: ModelSpec,
        messages: list[dict[str, str]],
        temperature: float,
        response_format: dict[str, Any] | None = None,
        mock_content: str | None = None,
    ) -> LLMResponse:
        """Call a real provider or return a deterministic dry-run response."""
        provider_name = spec.provider or infer_provider_name(spec.model)
        max_tokens = self.config.parameters.max_tokens
        use_cache = (not self.dry_run) and abs(float(temperature)) < 1e-9
        effective_messages = apply_response_format_contract(messages, response_format)

        if self.dry_run:
            content = mock_content or '{"action": "idle"}'
            return LLMResponse(
                content=content,
                model=spec.model,
                provider=provider_name,
                usage=UsageInfo(
                    input_tokens=max(1, sum(len(message["content"]) for message in effective_messages) // 4),
                    output_tokens=max(1, len(content) // 4),
                ),
                latency_ms=0.0,
                cost_usd=0.0,
            )

        cache_kwargs = {
            "model": spec.model,
            "provider": provider_name,
            "messages": effective_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
        if use_cache:
            cached = self.cache.load(**cache_kwargs)
            if cached is not None:
                return cached

        skip_reason = self.get_skip_reason(spec)
        if skip_reason is not None:
            raise ModelUnavailableError(f"{self.spec_key(spec)} skipped: {skip_reason}")

        provider = self._providers.get((provider_name, spec.model))
        if provider is None:
            api_key, base_url = self.get_provider_credentials(provider_name)
            try:
                provider = get_provider(
                    provider_name,
                    model=spec.model,
                    api_key=api_key,
                    base_url=base_url,
                )
            except ProviderError as exc:
                self.mark_model_unavailable(spec, str(exc))
                raise ModelUnavailableError(f"{self.spec_key(spec)} skipped: {exc}") from exc
            self._providers[(provider_name, spec.model)] = provider

        provider_key = self.provider_backoff_key(spec)
        rate_limit_attempt = 0
        transient_attempt = 0

        while True:
            await self.wait_for_provider_backoff(provider_key)

            try:
                response = await provider.complete(
                    messages=effective_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                break
            except RateLimitProviderError as exc:
                if not self.config.parameters.retry_on_rate_limit:
                    self.mark_model_unavailable(spec, str(exc))
                    raise ModelUnavailableError(f"{self.spec_key(spec)} skipped: {exc}") from exc

                max_rate_limit_retries = self.config.parameters.max_rate_limit_retries
                if (
                    max_rate_limit_retries is not None
                    and rate_limit_attempt >= max_rate_limit_retries
                ):
                    self.mark_model_unavailable(spec, str(exc))
                    raise ModelUnavailableError(f"{self.spec_key(spec)} skipped: {exc}") from exc

                delay_seconds = self.compute_retry_delay(
                    attempt=rate_limit_attempt,
                    suggested_delay=exc.retry_after_seconds,
                )
                self.set_provider_backoff(provider_key, delay_seconds)
                self.logger.log_event(
                    "provider_retry",
                    {
                        "provider": provider_name,
                        "model": spec.model,
                        "retry_kind": "rate_limit",
                        "attempt": rate_limit_attempt + 1,
                        "wait_seconds": delay_seconds,
                        "reason": str(exc),
                    },
                )
                rate_limit_attempt += 1
                continue
            except TemporaryProviderError as exc:
                if transient_attempt >= self.config.parameters.max_transient_retries:
                    self.mark_model_unavailable(spec, str(exc))
                    raise ModelUnavailableError(f"{self.spec_key(spec)} skipped: {exc}") from exc

                delay_seconds = self.compute_retry_delay(
                    attempt=transient_attempt,
                    suggested_delay=exc.retry_after_seconds,
                )
                self.set_provider_backoff(provider_key, delay_seconds)
                self.logger.log_event(
                    "provider_retry",
                    {
                        "provider": provider_name,
                        "model": spec.model,
                        "retry_kind": "temporary_error",
                        "attempt": transient_attempt + 1,
                        "wait_seconds": delay_seconds,
                        "reason": str(exc),
                    },
                )
                transient_attempt += 1
                continue
            except ProviderError as exc:
                self.mark_model_unavailable(spec, str(exc))
                raise ModelUnavailableError(f"{self.spec_key(spec)} skipped: {exc}") from exc

        if use_cache:
            self.cache.save(response, **cache_kwargs)
        try:
            self.cost_tracker.add(
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except ValueError:
            pass

        budget = self.config.parameters.budget_usd
        if budget is not None and self.cost_tracker.total() > budget:
            raise BudgetExceededError(
                f"Experiment budget exceeded: {self.cost_tracker.total():.4f} > {budget:.4f}"
            )

        return response

    async def close(self) -> None:
        """Close any provider clients that expose async cleanup."""
        for provider in self._providers.values():
            close_method = getattr(provider, "close", None)
            if close_method is not None:
                await close_method()

    def finalize(self, payload: dict[str, Any], duration_seconds: float) -> dict[str, Any]:
        """Write final outputs for an experiment run."""
        payload["experiment_id"] = self.experiment_id
        payload["config"] = {"experiment": self.config.model_dump(mode="json")}
        payload["run_metadata"] = self.run_metadata
        payload["total_cost_usd"] = self.cost_tracker.total()
        payload["total_duration_seconds"] = duration_seconds
        payload["skipped_models"] = [
            {"model": model_key, "reason": reason}
            for model_key, reason in sorted(self.skipped_models.items())
        ]
        payload["skipped_trials"] = self.skipped_trials if isinstance(self.skipped_trials, list) else []
        self.logger.finalize(total_cost=self.cost_tracker.total(), total_duration=duration_seconds)

        summary_path = self.results_dir / f"{self.experiment_id}.json"
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    @abstractmethod
    async def run(self) -> dict[str, Any]:
        """Execute the experiment and return a nested summary payload."""


async def run_experiment_from_path(
    config_path: str | Path,
    *,
    dry_run: bool = False,
    results_dir: str = "results",
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load an experiment config from disk and dispatch to the right runner."""
    from .config import load_experiment_config

    config = load_experiment_config(config_path)
    return await run_experiment_config(
        config,
        dry_run=dry_run,
        results_dir=results_dir,
        run_metadata=run_metadata,
    )


async def run_experiment_config(
    config: ExperimentSettings,
    *,
    dry_run: bool = False,
    results_dir: str = "results",
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch an already-loaded experiment config to the correct runner."""
    from .part1_runner import Part1Runner
    from .part2_runner import Part2Runner
    from .part3_runner import Part3Runner

    runner_map = {1: Part1Runner, 2: Part2Runner, 3: Part3Runner}
    runner_cls = runner_map[config.part]
    runner = runner_cls(
        config,
        results_dir=results_dir,
        dry_run=dry_run,
        run_metadata=run_metadata,
    )

    try:
        return await runner.run()
    except ProviderError as exc:
        raise RuntimeError(f"Provider call failed during experiment run: {exc}") from exc
    finally:
        await runner.close()
