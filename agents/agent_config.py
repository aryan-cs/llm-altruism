# print("[AGENT CONFIG] Hello, World!")

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

AGENT_CONFIG_PATH = Path(__file__).resolve().with_name("agent_config.json")
MODEL_REGISTRY_PATH = Path(__file__).resolve().with_name(
    "agent_config.registry.json"
)


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


def _required_non_empty_string(
    value: Any,
    *,
    label: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _normalize_provider_name(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    return {
        "cerebris": "cerebras",
        "inferencehub": "inference_hub",
        "olama": "ollama",
        "openaicompatible": "openai_compatible",
        "x.ai": "xai",
    }.get(normalized, normalized)


def _validate_model_registry(registry: Any) -> dict[str, Any]:
    if not isinstance(registry, dict):
        raise ValueError(f"Model registry must be a JSON object: {MODEL_REGISTRY_PATH}")
    if registry.get("schema_version") != 1:
        raise ValueError("Model registry schema_version must be 1.")
    _required_non_empty_string(
        registry.get("registry_version"),
        label="registry_version",
    )

    profiles = registry.get("endpoint_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Model registry must define endpoint_profiles.")
    for profile_id, profile in profiles.items():
        _required_non_empty_string(profile_id, label="endpoint profile id")
        if not isinstance(profile, dict):
            raise ValueError(f"endpoint_profiles.{profile_id} must be an object.")
        for field in ("provider", "protocol", "base_url_env", "credential_env"):
            _required_non_empty_string(
                profile.get(field),
                label=f"endpoint_profiles.{profile_id}.{field}",
            )
        if profile["protocol"] != "openai_chat_completions":
            raise ValueError(
                f"endpoint_profiles.{profile_id}.protocol is not supported: "
                f"{profile['protocol']}"
            )
        default_base_url = profile.get("default_base_url")
        if default_base_url is not None:
            normalized_url = _required_non_empty_string(
                default_base_url,
                label=f"endpoint_profiles.{profile_id}.default_base_url",
            )
            parsed_url = urlparse(normalized_url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise ValueError(
                    f"endpoint_profiles.{profile_id}.default_base_url must be an "
                    "absolute HTTP(S) URL."
                )

    raw_targets = registry.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("Model registry must define at least one target.")
    targets_by_id: dict[str, dict[str, str]] = {}
    route_keys: set[tuple[str, str]] = set()
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise ValueError(f"targets[{index}] must be an object.")
        target = {
            field: _required_non_empty_string(
                raw_target.get(field),
                label=f"targets[{index}].{field}",
            )
            for field in (
                "id",
                "provider",
                "upstream_provider",
                "model",
                "route",
                "endpoint_profile",
            )
        }
        target_id = target["id"]
        if target_id in targets_by_id:
            raise ValueError(f"Duplicate model registry target id: {target_id}")
        if target["endpoint_profile"] not in profiles:
            raise ValueError(
                f"Target {target_id} references unknown endpoint profile "
                f"'{target['endpoint_profile']}'."
            )
        profile_provider = profiles[target["endpoint_profile"]]["provider"]
        if target["provider"].lower() != profile_provider.lower():
            raise ValueError(
                f"Target {target_id} provider '{target['provider']}' does not match "
                f"endpoint profile provider '{profile_provider}'."
            )
        route_key = (target["provider"].lower(), target["route"])
        if route_key in route_keys:
            raise ValueError(
                f"Duplicate provider/route in model registry: {route_key[0]}/{route_key[1]}"
            )
        route_keys.add(route_key)
        targets_by_id[target_id] = target

    cohorts = registry.get("cohorts")
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("Model registry must define cohorts.")
    for cohort_id, cohort in cohorts.items():
        _required_non_empty_string(cohort_id, label="cohort id")
        if not isinstance(cohort, dict):
            raise ValueError(f"cohorts.{cohort_id} must be an object.")
        _required_non_empty_string(
            cohort.get("version"),
            label=f"cohorts.{cohort_id}.version",
        )
        members = cohort.get("targets")
        if not isinstance(members, list) or not members:
            raise ValueError(f"cohorts.{cohort_id}.targets must be a non-empty list.")
        if len(members) != len(set(members)):
            raise ValueError(f"cohorts.{cohort_id} contains duplicate targets.")
        unknown = [member for member in members if member not in targets_by_id]
        if unknown:
            raise ValueError(
                f"cohorts.{cohort_id} references unknown targets: {', '.join(unknown)}"
            )

    default_cohort = _required_non_empty_string(
        registry.get("default_cohort"),
        label="default_cohort",
    )
    if default_cohort not in cohorts:
        raise ValueError(f"Unknown default cohort: {default_cohort}")
    return registry


@lru_cache(maxsize=None)
def load_model_registry() -> dict[str, Any]:
    if not MODEL_REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Missing model registry: {MODEL_REGISTRY_PATH}")
    raw_text = MODEL_REGISTRY_PATH.read_text(encoding="utf-8")
    return _validate_model_registry(json.loads(raw_text))


def model_registry_hash() -> str:
    canonical = json.dumps(
        load_model_registry(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_endpoint_profile(profile_id: str) -> dict[str, Any]:
    registry = load_model_registry()
    normalized_id = _normalize_provider_name(profile_id)
    profiles = registry["endpoint_profiles"]
    if normalized_id not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(
            f"Unknown endpoint profile '{profile_id}'. Available profiles: {available}."
        )
    return dict(profiles[normalized_id])


def load_model_cohort(cohort_id: str | None = None) -> dict[str, Any]:
    registry = load_model_registry()
    resolved_id = (cohort_id or registry["default_cohort"]).strip()
    cohorts = registry["cohorts"]
    if resolved_id not in cohorts:
        available = ", ".join(sorted(cohorts))
        raise KeyError(
            f"Unknown model cohort '{resolved_id}'. Available cohorts: {available}."
        )
    targets_by_id = {target["id"]: target for target in registry["targets"]}
    cohort = cohorts[resolved_id]
    return {
        "id": resolved_id,
        "version": cohort["version"],
        "registry_version": registry["registry_version"],
        "registry_hash": model_registry_hash(),
        "targets": [dict(targets_by_id[target_id]) for target_id in cohort["targets"]],
    }


def resolve_model_registry_entry(
    provider: str,
    model: str,
) -> dict[str, Any] | None:
    normalized_provider = _normalize_provider_name(provider)
    normalized_model = model.strip()
    if not normalized_provider or not normalized_model:
        return None
    registry = load_model_registry()
    memberships: dict[str, list[str]] = {}
    for cohort_id, cohort in registry["cohorts"].items():
        for target_id in cohort["targets"]:
            memberships.setdefault(target_id, []).append(cohort_id)
    for target in registry["targets"]:
        target_provider = target["provider"].lower().replace("-", "_")
        if target_provider != normalized_provider:
            continue
        if normalized_model not in {target["model"], target["route"]}:
            continue
        profile = registry["endpoint_profiles"][target["endpoint_profile"]]
        return {
            **target,
            "cohorts": memberships.get(target["id"], []),
            "endpoint": dict(profile),
            "registry_version": registry["registry_version"],
            "registry_hash": model_registry_hash(),
        }
    return None


def registry_metadata_for_targets(
    targets: list[tuple[str, str]],
) -> dict[str, Any]:
    registry = load_model_registry()
    resolved_targets: list[dict[str, Any]] = []
    for provider, model in targets:
        entry = resolve_model_registry_entry(provider, model)
        normalized_provider = _normalize_provider_name(provider)
        endpoint = registry["endpoint_profiles"].get(normalized_provider)
        resolved_targets.append(
            entry
            or {
                "provider": normalized_provider,
                "model": model.strip(),
                "route": model.strip(),
                **(
                    {
                        "endpoint_profile": normalized_provider,
                        "endpoint": dict(endpoint),
                    }
                    if isinstance(endpoint, dict)
                    else {}
                ),
                "registered": False,
            }
        )
    cohort_ids = sorted(
        {
            cohort_id
            for target in resolved_targets
            for cohort_id in target.get("cohorts", [])
        }
    )
    return {
        "registry_version": registry["registry_version"],
        "registry_hash": model_registry_hash(),
        "cohorts": [
            {
                "id": cohort_id,
                "version": registry["cohorts"][cohort_id]["version"],
            }
            for cohort_id in cohort_ids
        ],
        "targets": resolved_targets,
    }
