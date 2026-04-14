# print("[AGENT CONFIG] Hello, World!")

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

AGENT_CONFIG_PATH = Path(__file__).resolve().with_name("agent_config.json")


def _strip_json_comments(raw_text: str) -> str:
    kept_lines: list[str] = []
    for line in raw_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//") or stripped.startswith("#"):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _strip_trailing_commas(raw_text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", raw_text)


@lru_cache(maxsize=None)
def load_agent_config() -> dict[str, Any]:
    if not AGENT_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Missing agent config: {AGENT_CONFIG_PATH}")

    raw_text = AGENT_CONFIG_PATH.read_text(encoding="utf-8")
    config = json.loads(_strip_trailing_commas(_strip_json_comments(raw_text)))
    if not isinstance(config, dict):
        raise ValueError(f"Agent config must be a JSON object: {AGENT_CONFIG_PATH}")
    return config


def _normalize_provider_models(value: Any, *, section_name: str) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        raise ValueError(f"{section_name} must map providers to model lists.")

    normalized: dict[str, list[str]] = {}
    for provider, models in value.items():
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError(f"{section_name} contains an invalid provider name.")
        if not isinstance(models, list):
            raise ValueError(f"{section_name}.{provider} must be a list of model ids.")

        cleaned_models = [
            model.strip()
            for model in models
            if isinstance(model, str) and model.strip()
        ]
        if cleaned_models:
            normalized[provider.strip().lower()] = cleaned_models

    if not normalized:
        raise ValueError(f"{section_name} must define at least one enabled model.")
    return normalized


def load_experiment_model_options(experiment_key: str) -> dict[str, list[str]]:
    normalized_key = experiment_key.strip().lower()
    if not normalized_key:
        raise ValueError("experiment_key must be a non-empty string.")

    config = load_agent_config()
    if normalized_key not in config:
        available = ", ".join(sorted(config))
        raise KeyError(
            f"Unknown experiment key '{experiment_key}'. Available sections: {available}."
        )

    return _normalize_provider_models(
        config[normalized_key],
        section_name=normalized_key,
    )


def load_all_model_options() -> dict[str, list[str]]:
    config = load_agent_config()
    combined: dict[str, list[str]] = {}

    for section_name, section in config.items():
        section_models = _normalize_provider_models(section, section_name=section_name)
        for provider, models in section_models.items():
            combined.setdefault(provider, [])
            for model in models:
                if model not in combined[provider]:
                    combined[provider].append(model)

    if not combined:
        raise ValueError("agent_config.json does not define any enabled models.")
    return combined
