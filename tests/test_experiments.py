"""End-to-end dry-run tests for config loading and runners."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.experiments import load_experiment_config
from src.experiments.part1_runner import Part1Runner
from src.experiments.runner import ModelUnavailableError, run_experiment_from_path
from src.providers import LLMResponse, RateLimitProviderError, TemporaryProviderError, UsageInfo


def test_can_load_sample_part1_config():
    config = load_experiment_config("configs/part1/prisoners_dilemma_baseline.yaml")
    assert config.part == 1
    assert config.game == "prisoners_dilemma"
    assert config.pairings


def test_part1_dry_run_produces_trials(tmp_path: Path):
    result = asyncio.run(
        run_experiment_from_path(
            "configs/part1/prisoners_dilemma_baseline.yaml",
            dry_run=True,
            results_dir=str(tmp_path),
        )
    )
    assert result["trials"]
    assert result["aggregate_summary"]["total_payoff_a"] >= 0


def test_run_metadata_is_saved_in_results(tmp_path: Path):
    result = asyncio.run(
        run_experiment_from_path(
            "configs/part1/prisoners_dilemma_baseline.yaml",
            dry_run=True,
            results_dir=str(tmp_path),
            run_metadata={
                "template_path": "configs/part1/prisoners_dilemma_baseline.yaml",
                "selected_models": [
                    "cerebras:llama3.1-8b",
                    "nvidia:deepseek-ai/deepseek-v3.2",
                ],
            },
        )
    )
    assert result["run_metadata"]["template_path"] == "configs/part1/prisoners_dilemma_baseline.yaml"
    assert result["run_metadata"]["selected_models"] == [
        "cerebras:llama3.1-8b",
        "nvidia:deepseek-ai/deepseek-v3.2",
    ]


def test_part2_dry_run_produces_society_metrics(tmp_path: Path):
    result = asyncio.run(
        run_experiment_from_path(
            "configs/part2/society_baseline.yaml",
            dry_run=True,
            results_dir=str(tmp_path),
        )
    )
    assert result["trials"][0]["summary"]["average_gini"] >= 0


def test_part3_dry_run_logs_reputation(tmp_path: Path):
    result = asyncio.run(
        run_experiment_from_path(
            "configs/part3/society_reputation.yaml",
            dry_run=True,
            results_dir=str(tmp_path),
        )
    )
    first_round = result["trials"][0]["rounds"][0]
    assert "ratings" in first_round


def test_real_run_skips_unconfigured_models_instead_of_crashing(monkeypatch, tmp_path: Path):
    for env_var in [
        "CEREBRAS_API_KEY",
        "CEREBRAS_BASE_URL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "OLLAMA_BASE_URL",
    ]:
        monkeypatch.setenv(env_var, "")

    result = asyncio.run(
        run_experiment_from_path(
            "configs/part1/prisoners_dilemma_baseline.yaml",
            dry_run=False,
            results_dir=str(tmp_path),
        )
    )
    assert result["trials"] == []
    assert result["skipped_trials"]
    assert result["skipped_models"]


class FlakyProvider:
    """Simple fake provider that returns queued results for retry tests."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    async def complete(self, **_kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _success_response(model: str, provider: str) -> LLMResponse:
    return LLMResponse(
        content='{"action": "cooperate"}',
        model=model,
        provider=provider,
        usage=UsageInfo(input_tokens=5, output_tokens=5),
        latency_ms=12.0,
        cost_usd=0.0,
    )


def test_request_completion_retries_rate_limits_until_success(monkeypatch, tmp_path: Path):
    config = load_experiment_config("configs/part1/prisoners_dilemma_baseline.yaml")
    runner = Part1Runner(config, results_dir=str(tmp_path), dry_run=False)
    spec = config.pairings[2][1]
    provider = FlakyProvider(
        [
            RateLimitProviderError("too many requests", retry_after_seconds=12.0),
            _success_response(spec.model, spec.provider or "ollama"),
        ]
    )
    runner._providers[(spec.provider or "ollama", spec.model)] = provider
    backoff_delays: list[float] = []

    async def no_wait(_provider_key: str) -> None:
        return None

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(runner, "wait_for_provider_backoff", no_wait)
    monkeypatch.setattr(runner, "set_provider_backoff", lambda _provider_key, delay: backoff_delays.append(delay))

    response = asyncio.run(
        runner.request_completion(
            spec=spec,
            messages=[{"role": "user", "content": "Choose an action."}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    )

    assert response.content == '{"action": "cooperate"}'
    assert provider.calls == 2
    assert backoff_delays == [12.0]
    assert not runner.skipped_models


def test_request_completion_skips_cache_for_nonzero_temperature(monkeypatch, tmp_path: Path):
    config = load_experiment_config("configs/part1/prisoners_dilemma_baseline.yaml")
    runner = Part1Runner(config, results_dir=str(tmp_path), dry_run=False)
    spec = config.pairings[2][1]
    provider = FlakyProvider([_success_response(spec.model, spec.provider or "ollama")])
    runner._providers[(spec.provider or "ollama", spec.model)] = provider
    cache_calls: list[str] = []

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(runner.cache, "load", lambda **_kwargs: cache_calls.append("load"))
    monkeypatch.setattr(runner.cache, "save", lambda *_args, **_kwargs: cache_calls.append("save"))

    response = asyncio.run(
        runner.request_completion(
            spec=spec,
            messages=[{"role": "user", "content": "Choose an action."}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
    )

    assert response.content == '{"action": "cooperate"}'
    assert provider.calls == 1
    assert cache_calls == []


def test_request_completion_skips_after_exhausting_transient_retries(monkeypatch, tmp_path: Path):
    config = load_experiment_config("configs/part1/prisoners_dilemma_baseline.yaml")
    config.parameters.max_transient_retries = 2
    runner = Part1Runner(config, results_dir=str(tmp_path), dry_run=False)
    spec = config.pairings[2][1]
    provider = FlakyProvider(
        [
            TemporaryProviderError("temporary outage"),
            TemporaryProviderError("temporary outage"),
            TemporaryProviderError("temporary outage"),
        ]
    )
    runner._providers[(spec.provider or "ollama", spec.model)] = provider
    backoff_delays: list[float] = []

    async def no_wait(_provider_key: str) -> None:
        return None

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setattr(runner, "wait_for_provider_backoff", no_wait)
    monkeypatch.setattr(runner, "set_provider_backoff", lambda _provider_key, delay: backoff_delays.append(delay))

    with pytest.raises(ModelUnavailableError):
        asyncio.run(
            runner.request_completion(
                spec=spec,
                messages=[{"role": "user", "content": "Choose an action after a transient outage."}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        )

    assert provider.calls == 3
    assert len(backoff_delays) == 2
    assert f"{spec.provider}:{spec.model}" in runner.skipped_models
