"""Shared experiment runner infrastructure and dispatcher."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agents.base import Agent
from src.agents.memory import Memory
from src.agents.prompts import load_prompt
from src.providers import LLMResponse, ProviderError, UsageInfo, get_provider
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
    ):
        load_dotenv()
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.cost_tracker = CostTracker()
        self.logger = ExperimentLogger(results_dir=str(self.results_dir))
        self.cache = ResponseCache(self.results_dir.parent / ".cache" / "responses")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.experiment_id = f"{self.config.name}-{timestamp}"
        self.logger.start_experiment(self.experiment_id, {"experiment": self.config.model_dump(mode="json")})
        self._providers: dict[tuple[str, str], Any] = {}
        self.skipped_models: dict[str, str] = {}
        self.skipped_trials: list[dict[str, Any]] = []

    def spec_key(self, spec: ModelSpec) -> str:
        """Return a stable key for a model specification."""
        provider_name = spec.provider or infer_provider_name(spec.model)
        return f"{provider_name}:{spec.model}"

    def mark_model_unavailable(self, spec: ModelSpec, reason: str) -> None:
        """Record that a model should be skipped for the rest of the run."""
        self.skipped_models[self.spec_key(spec)] = reason

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

        if self.dry_run:
            content = mock_content or '{"action": "idle"}'
            return LLMResponse(
                content=content,
                model=spec.model,
                provider=provider_name,
                usage=UsageInfo(
                    input_tokens=max(1, sum(len(message["content"]) for message in messages) // 4),
                    output_tokens=max(1, len(content) // 4),
                ),
                latency_ms=0.0,
                cost_usd=0.0,
            )

        cache_kwargs = {
            "model": spec.model,
            "provider": provider_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
        }
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

        try:
            response = await provider.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except ProviderError as exc:
            self.mark_model_unavailable(spec, str(exc))
            raise ModelUnavailableError(f"{self.spec_key(spec)} skipped: {exc}") from exc

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
) -> dict[str, Any]:
    """Load an experiment config and dispatch to the right runner."""
    from .config import load_experiment_config
    from .part1_runner import Part1Runner
    from .part2_runner import Part2Runner
    from .part3_runner import Part3Runner

    config = load_experiment_config(config_path)
    runner_map = {1: Part1Runner, 2: Part2Runner, 3: Part3Runner}
    runner_cls = runner_map[config.part]
    runner = runner_cls(config, results_dir=results_dir, dry_run=dry_run)

    try:
        return await runner.run()
    except ProviderError as exc:
        raise RuntimeError(f"Provider call failed during experiment run: {exc}") from exc
    finally:
        await runner.close()
