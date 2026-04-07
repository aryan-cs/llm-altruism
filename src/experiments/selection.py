"""Interactive and non-interactive experiment model selection helpers."""

from __future__ import annotations

import os
import textwrap
from itertools import combinations
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv

from .config import ExperimentSettings, ModelSpec, PopulationSpec, load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = ROOT / "configs"

TEMPLATE_DESCRIPTIONS: dict[str, str] = {
    "free-tier-model-catalog": (
        "Broad Part 1 screening run across the requested free NVIDIA, Cerebras, and "
        "OpenRouter models using Prisoner's Dilemma. Good for overnight catalog sweeps."
    ),
    "pd-baseline-cross-family": (
        "Baseline cross-family Prisoner's Dilemma benchmark with neutral, cooperative, "
        "and competitive prompt framings. Good first live experiment."
    ),
    "pd-cross-family-sweep": (
        "Focused Prisoner's Dilemma comparison with analytical/business framing and "
        "shorter memory, aimed at cross-family behavior differences."
    ),
    "stag-hunt-persona-sweep": (
        "Coordination experiment in Stag Hunt that compares persona effects, especially "
        "cooperative versus grudge-holding behavior."
    ),
    "society-baseline": (
        "Multi-agent scarce-resource society without public reputation. Tests gathering, "
        "sharing, stealing, trade, reproduction, and messaging under pressure."
    ),
    "society-reputation": (
        "Society baseline plus public ratings and reputation decay, to test whether "
        "reputation changes partner choice and pro-social behavior."
    ),
}

KNOWN_MODELS_BY_PROVIDER: dict[str, tuple[str, ...]] = {
    "openai": (
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o3-mini",
    ),
    "anthropic": (
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
    ),
    "google": (
        "gemini-2.0-flash",
        "gemini-2.5-pro",
    ),
    "xai": ("grok-3",),
    "cerebras": (
        "gpt-oss-120b",
        "llama3.1-8b",
        "qwen-3-235b-a22b-instruct-2507",
        "zai-glm-4.7",
    ),
    "nvidia": (
        "mistralai/mistral-small-24b-instruct",
        "mistralai/magistral-small-2506",
        "tiiuae/falcon3-7b-instruct",
        "rakuten/rakutenai-7b-instruct",
        "z-ai/glm4.7",
        "z-ai/glm5",
        "google/gemma-3-1b-it",
        "google/gemma-3-27b-it",
        "bytedance/seed-oss-36b-instruct",
        "nvidia/nemotron-content-safety-reasoning-4b",
        "nvidia/nemotron-3-nano-30b-a3b",
        "nvidia/nemotron-3-super-120b-a12b",
        "deepseek-ai/deepseek-v3.2",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
        "moonshotai/kimi-k2-thinking",
        "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        "stepfun-ai/step-3-5-flash",
    ),
    "openrouter": (
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "openai/gpt-oss-120b:free",
        "openai/gpt-oss-20b:free",
        "z-ai/glm-4.5-air:free",
        "google/gemma-3-4b-it:free",
        "google/gemma-3-12b-it:free",
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "qwen/qwen3.6-plus:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
    ),
    "ollama": ("llama3.2:3b",),
}


def discover_ollama_models() -> tuple[str, ...]:
    """Return locally installed Ollama models when the local endpoint is available."""
    load_dotenv(ROOT / ".env", override=False)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    static_models = KNOWN_MODELS_BY_PROVIDER["ollama"]

    try:
        with urlopen(f"{base_url}/api/tags", timeout=3.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return static_models

    dynamic_models = [
        item.get("name", "").strip()
        for item in payload.get("models", [])
        if isinstance(item, dict) and item.get("name")
    ]
    merged = []
    seen: set[str] = set()
    for model in [*dynamic_models, *static_models]:
        if not model or model in seen:
            continue
        seen.add(model)
        merged.append(model)
    return tuple(merged) if merged else static_models


def list_experiment_templates() -> list[Path]:
    """Return all YAML experiment templates in the repo config directory."""
    return sorted(CONFIGS_ROOT.glob("*/*.yaml"))


def template_label(path: str | Path) -> str:
    """Build a short human-readable label for a config template path."""
    config = load_experiment_config(path)
    part_label = f"Part {config.part}"
    game_label = f" | {config.game}" if config.game else ""
    return f"{part_label} | {config.name}{game_label}"


def template_description(path: str | Path) -> str:
    """Return a human-readable description of the experiment template and setup."""
    config = load_experiment_config(path)
    setup = template_setup_summary(config)
    base = TEMPLATE_DESCRIPTIONS.get(config.name)
    if base:
        return f"{base} Setup: {setup}."
    return setup[0].upper() + setup[1:] + "."


def wrap_picker_description(
    description: str,
    *,
    columns: int,
    prefix: str = "  Description: ",
) -> str:
    """Wrap picker preview text for terminal display.

    `questionary` renders descriptions as a single token line, so we insert
    explicit newlines and indentation ourselves.
    """
    available_width = max(20, columns - len(prefix))
    wrapped_lines = textwrap.wrap(
        description,
        width=available_width,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped_lines:
        return ""
    if len(wrapped_lines) == 1:
        return wrapped_lines[0]
    indent = " " * len(prefix)
    return ("\n" + indent).join(wrapped_lines)


def template_setup_summary(config: ExperimentSettings) -> str:
    """Summarize the concrete setup encoded by an experiment template."""
    prompt_count = len(config.prompt_variants)
    if config.part == 1:
        model_count = len(models_from_config(config))
        return (
            f"{config.rounds} rounds x {config.repetitions} repetition(s), "
            f"{prompt_count} prompt variant(s), {model_count} default model(s), "
            f"memory={config.history.mode}"
        )

    total_agents = sum(population.count for population in config.agents)
    model_count = len(models_from_config(config))
    extra_features = [
        "trade" if config.society and config.society.trade_offer_ttl else None,
        "private messages" if config.society and config.society.allow_private_messages else None,
        "stealing" if config.society and config.society.allow_steal else None,
        "reputation" if config.part == 3 else None,
    ]
    enabled = ", ".join(feature for feature in extra_features if feature)
    summary = (
        f"{config.rounds} rounds x {config.repetitions} repetition(s), "
        f"{total_agents} total agent slots across {model_count} default model(s), "
        f"{prompt_count} prompt variant(s), memory={config.history.mode}"
    )
    if enabled:
        summary += f", features={enabled}"
    return summary


def known_model_specs() -> list[ModelSpec]:
    """Return the flattened interactive catalog of known models."""
    specs: list[ModelSpec] = []
    for provider, models in KNOWN_MODELS_BY_PROVIDER.items():
        if provider == "ollama":
            models = discover_ollama_models()
        for model in models:
            specs.append(ModelSpec(model=model, provider=provider))
    return specs


def unique_specs(specs: list[ModelSpec]) -> list[ModelSpec]:
    """Deduplicate model specs while preserving order."""
    deduped: list[ModelSpec] = []
    seen: set[tuple[str | None, str]] = set()
    for spec in specs:
        key = (spec.provider, spec.model)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def parse_model_selector(value: str) -> ModelSpec:
    """Parse a CLI selector in `provider:model-id` format."""
    if ":" not in value:
        raise ValueError(
            f"Invalid model selector {value!r}. Use the form provider:model-id."
        )
    provider, model = value.split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if not provider or not model:
        raise ValueError(
            f"Invalid model selector {value!r}. Use the form provider:model-id."
        )
    return ModelSpec(model=model, provider=provider)


def parse_model_selectors(values: list[str]) -> list[ModelSpec]:
    """Parse one or more CLI model selectors."""
    return unique_specs([parse_model_selector(value) for value in values])


def models_from_config(config: ExperimentSettings) -> list[ModelSpec]:
    """Extract the current model set from a config template."""
    if config.part == 1:
        specs = [spec for pairing in config.pairings for spec in pairing]
    else:
        specs = [ModelSpec(model=agent.model, provider=agent.provider) for agent in config.agents]
    return unique_specs(specs)


def apply_model_selection(
    config: ExperimentSettings,
    selected_models: list[ModelSpec],
) -> ExperimentSettings:
    """Return a config copy with pairings or populations rebuilt from selected models."""
    selected = unique_specs(selected_models)
    if not selected:
        raise ValueError("At least one model must be selected.")

    if config.part == 1:
        if len(selected) == 1:
            pairings = [(selected[0], selected[0])]
        else:
            pairings = list(combinations(selected, 2))
        return config.model_copy(update={"pairings": pairings}, deep=True)

    original_total = sum(population.count for population in config.agents) or len(selected)
    total_population = max(original_total, len(selected))
    base_count, remainder = divmod(total_population, len(selected))
    rebuilt_agents = [
        PopulationSpec(
            model=spec.model,
            provider=spec.provider,
            count=base_count + (1 if index < remainder else 0),
        )
        for index, spec in enumerate(selected)
    ]
    return config.model_copy(update={"agents": rebuilt_agents}, deep=True)


def apply_runtime_overrides(
    config: ExperimentSettings,
    *,
    rounds: int | None = None,
    repetitions: int | None = None,
    temperatures: list[float] | None = None,
    concurrency: int | None = None,
) -> ExperimentSettings:
    """Return a config copy with interactive/CLI runtime overrides applied."""
    parameter_updates: dict[str, object] = {}
    if temperatures is not None:
        parameter_updates["temperature"] = temperatures
    if concurrency is not None:
        parameter_updates["concurrency"] = concurrency

    updates: dict[str, object] = {}
    if rounds is not None:
        updates["rounds"] = rounds
    if repetitions is not None:
        updates["repetitions"] = repetitions
    if parameter_updates:
        updates["parameters"] = config.parameters.model_copy(update=parameter_updates, deep=True)

    if not updates:
        return config
    return config.model_copy(update=updates, deep=True)


def estimate_trial_conditions(config: ExperimentSettings) -> int:
    """Estimate the number of top-level trial conditions a config will run."""
    prompt_count = max(1, len(config.prompt_variants))
    temperature_count = max(1, len(config.parameters.temperature))
    repetitions = max(1, config.repetitions)

    if config.part == 1:
        return len(config.pairings) * prompt_count * temperature_count * repetitions
    return prompt_count * temperature_count * repetitions
