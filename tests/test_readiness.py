"""Readiness checks for local experiment execution.

These tests are stricter than the unit tests: they validate that the local
workspace, `.env`, sample configs, and currently selected providers are ready
for experiment runs. The live smoke test at the bottom intentionally makes
real provider calls for each locally configured model in the sample configs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from dotenv import dotenv_values, load_dotenv

from src.experiments import ModelSpec, load_experiment_config, probe_model_access
from src.experiments.runner import API_KEY_ENV, ENDPOINT_ENV, infer_provider_name, run_experiment_from_path
from src.providers import get_provider


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
CONFIG_PATHS = sorted((ROOT / "configs").glob("*/*.yaml"))
FREE_TIER_CATALOG_PATH = ROOT / "configs" / "part1" / "free_tier_model_catalog.yaml"
def _load_env_file() -> dict[str, str | None]:
    """Load the repository `.env` file and current process environment."""
    load_dotenv(ENV_PATH, override=True)
    return dict(dotenv_values(ENV_PATH))


def _iter_model_specs():
    """Yield unique `(provider, model)` pairs from the sample configs."""
    seen: set[tuple[str, str]] = set()

    for config_path in CONFIG_PATHS:
        config = load_experiment_config(config_path)

        if config.part == 1:
            for left, right in config.pairings:
                for spec in (left, right):
                    provider = spec.provider or infer_provider_name(spec.model)
                    item = (provider, spec.model)
                    if item not in seen:
                        seen.add(item)
                        yield item
        else:
            for spec in config.agents:
                provider = spec.provider or infer_provider_name(spec.model)
                item = (provider, spec.model)
                if item not in seen:
                    seen.add(item)
                    yield item


def _is_locally_configured(
    provider: str,
    env_values: dict[str, str | None],
) -> bool:
    """Return True when the local environment has what a provider needs."""
    key_env = API_KEY_ENV.get(provider, "")
    endpoint_env = ENDPOINT_ENV.get(provider, "")
    key_ready = not key_env or bool(env_values.get(key_env))
    endpoint_ready = not endpoint_env or bool(env_values.get(endpoint_env))
    return key_ready and endpoint_ready


def _live_smoke_cases() -> list[object]:
    """Build parametrized provider/model cases for the live readiness test."""
    cases = []
    for provider, model in _iter_model_specs():
        cases.append(pytest.param(provider, model, id=f"{provider}:{model}"))
    return cases


def test_env_file_exists_and_has_all_supported_provider_entries():
    """The real `.env` should define the full provider key/endpoint surface."""
    env_values = _load_env_file()

    expected_vars = {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "GOOGLE_API_KEY",
        "GOOGLE_BASE_URL",
        "XAI_API_KEY",
        "XAI_BASE_URL",
        "CEREBRAS_API_KEY",
        "CEREBRAS_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_BASE_URL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "OLLAMA_API_KEY",
        "OLLAMA_BASE_URL",
    }

    assert ENV_PATH.exists(), "Expected a real .env file at repo root"
    missing_entries = sorted(var for var in expected_vars if var not in env_values)
    assert not missing_entries, f".env is missing entries: {missing_entries}"


def test_all_current_sample_config_files_load():
    """Every sample config should parse successfully."""
    assert CONFIG_PATHS, "No sample experiment configs were found"
    for config_path in CONFIG_PATHS:
        config = load_experiment_config(config_path)
        assert config.name
        assert config.part in {1, 2, 3}


def test_configured_sample_model_providers_can_be_constructed():
    """Construct provider clients for all current sample-config models."""
    env_values = _load_env_file()
    configured_models = []

    for provider, model in _iter_model_specs():
        if not _is_locally_configured(provider, env_values):
            continue
        api_key = env_values.get(API_KEY_ENV.get(provider, "")) or None
        base_url = env_values.get(ENDPOINT_ENV.get(provider, "")) or None
        client = get_provider(provider, model=model, api_key=api_key, base_url=base_url)
        assert client.get_provider_name() == provider
        configured_models.append((provider, model))

    assert configured_models, "Expected at least one sample-config model to be locally runnable"


def test_free_tier_catalog_config_matches_requested_models():
    """The dedicated free-tier catalog config should include the requested model sets."""
    config = load_experiment_config(FREE_TIER_CATALOG_PATH)
    pairings = {(left.provider, left.model) for left, _right in config.pairings}

    expected_models = {
        ("nvidia", "z-ai/glm4.7"),
        ("nvidia", "z-ai/glm5"),
        ("nvidia", "google/gemma-3-1b-it"),
        ("nvidia", "google/gemma-3-27b-it"),
        ("nvidia", "bytedance/seed-oss-36b-instruct"),
        ("nvidia", "nvidia/nemotron-content-safety-reasoning-4b"),
        ("nvidia", "nvidia/nemotron-3-nano-30b-a3b"),
        ("nvidia", "nvidia/nemotron-3-super-120b-a12b"),
        ("nvidia", "deepseek-ai/deepseek-v3.2"),
        ("nvidia", "moonshotai/kimi-k2-instruct"),
        ("nvidia", "moonshotai/kimi-k2-instruct-0905"),
        ("nvidia", "moonshotai/kimi-k2-thinking"),
        ("nvidia", "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"),
        ("nvidia", "stepfun-ai/step-3-5-flash"),
        ("cerebras", "gpt-oss-120b"),
        ("cerebras", "llama3.1-8b"),
        ("cerebras", "qwen-3-235b-a22b-instruct-2507"),
        ("cerebras", "zai-glm-4.7"),
        ("openrouter", "nvidia/nemotron-3-nano-30b-a3b:free"),
        ("openrouter", "openai/gpt-oss-120b:free"),
        ("openrouter", "openai/gpt-oss-20b:free"),
        ("openrouter", "z-ai/glm-4.5-air:free"),
        ("openrouter", "google/gemma-3-4b-it:free"),
        ("openrouter", "google/gemma-3-12b-it:free"),
        ("openrouter", "google/gemma-3-27b-it:free"),
        ("openrouter", "meta-llama/llama-3.3-70b-instruct:free"),
        ("openrouter", "meta-llama/llama-3.2-3b-instruct:free"),
        ("openrouter", "qwen/qwen3.6-plus:free"),
        ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
    }

    assert len(config.pairings) == len(expected_models)
    assert pairings == expected_models


def test_all_sample_configs_can_dry_run(tmp_path: Path):
    """Every sample config should complete a dry-run end to end."""
    for config_path in CONFIG_PATHS:
        result = asyncio.run(
            run_experiment_from_path(
                config_path,
                dry_run=True,
                results_dir=str(tmp_path / config_path.stem),
            )
        )
        assert "experiment_id" in result
        assert "config" in result
        assert "trials" in result


def test_ollama_endpoint_is_reachable_for_current_sample_configs():
    """If Ollama is used by the current configs, its local endpoint should respond."""
    uses_ollama = any(provider == "ollama" for provider, _model in _iter_model_specs())
    assert uses_ollama, "Expected current sample configs to include Ollama"

    env_values = _load_env_file()
    base_url = (env_values.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
    response = httpx.get(f"{base_url}/api/tags", timeout=5.0)
    assert response.status_code == 200


@pytest.mark.live_api
@pytest.mark.asyncio
@pytest.mark.parametrize(("provider_name", "model"), _live_smoke_cases())
async def test_locally_configured_models_return_live_smoke_responses(
    provider_name: str,
    model: str,
):
    """Each locally configured sample-config model should return one real response."""
    env_values = _load_env_file()
    if not _is_locally_configured(provider_name, env_values):
        pytest.skip(f"{provider_name}:{model} is not fully configured in .env")

    result = await probe_model_access(ModelSpec(model=model, provider=provider_name))

    assert result.spec.provider == provider_name
    assert result.spec.model == model
    assert result.accessible, f"{provider_name}:{model} failed startup access test: {result.status}"
