# print("[AGENT CONFIG] Hello, World!")

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

AGENT_CONFIG_PATH = Path(__file__).resolve().with_name("agent_config.json")
MODEL_REGISTRY_PATH = Path(__file__).resolve().with_name(
    "agent_config.registry.json"
)
STRICT_ENDPOINT_HOSTS = {
    "inference_hub": "inference-api.nvidia.com",
    "nvidia": "integrate.api.nvidia.com",
}
STRICT_ENDPOINT_PATHS = {
    "inference_hub": "/v1",
    "nvidia": "/v1",
}
AUTHORITATIVE_ROUTE_SOURCES = {
    "inference_hub_models_api",
}
UNVERIFIED_ROUTE_SOURCES = {"catalog_display_only"}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ROUTE_VERIFICATION_POLICY_SCHEMA_VERSION = 1


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


def _required_sha256(value: Any, *, label: str) -> str:
    normalized = _required_non_empty_string(value, label=label)
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return normalized


def _required_utc_timestamp(value: Any, *, label: str) -> str:
    normalized = _required_non_empty_string(value, label=label)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp.") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp.")
    return normalized


def _canonical_json_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _routing_roster_payload(registry: dict[str, Any]) -> dict[str, Any]:
    """Return the route identity surface that live evidence must bind.

    Verification fields are intentionally excluded so a bundle captured against
    an unverified candidate registry remains valid when those same exact routes
    are promoted to ``verified``.
    """

    return {
        "registry_version": registry.get("registry_version"),
        "endpoint_profiles": registry.get("endpoint_profiles"),
        "cohorts": registry.get("cohorts"),
        "targets": [
            {
                field: target.get(field)
                for field in (
                    "id",
                    "provider",
                    "upstream_provider",
                    "model",
                    "route",
                    "endpoint_profile",
                )
            }
            for target in registry.get("targets", [])
            if isinstance(target, dict)
        ],
    }


def _routing_roster_sha256(registry: dict[str, Any]) -> str:
    return _canonical_json_sha256(_routing_roster_payload(registry))


def _validate_verification_evidence(
    raw_target: dict[str, Any],
    *,
    index: int,
    route: str,
) -> None:
    evidence = raw_target.get("verification_evidence")
    label = f"targets[{index}].verification_evidence"
    if not isinstance(evidence, dict):
        raise ValueError(f"{label} must be an object for a verified route.")
    _required_utc_timestamp(
        evidence.get("verified_at_utc"),
        label=f"{label}.verified_at_utc",
    )
    _required_sha256(
        evidence.get("discovery_sha256"),
        label=f"{label}.discovery_sha256",
    )
    smoke_test = evidence.get("smoke_test")
    if not isinstance(smoke_test, dict):
        raise ValueError(f"{label}.smoke_test must be an object.")
    _required_utc_timestamp(
        smoke_test.get("completed_at_utc"),
        label=f"{label}.smoke_test.completed_at_utc",
    )
    _required_non_empty_string(
        smoke_test.get("request_id"),
        label=f"{label}.smoke_test.request_id",
    )
    response_model = _required_non_empty_string(
        smoke_test.get("response_model"),
        label=f"{label}.smoke_test.response_model",
    )
    if response_model != route:
        raise ValueError(
            f"{label}.smoke_test.response_model must match the exact route."
        )
    _required_sha256(
        smoke_test.get("response_sha256"),
        label=f"{label}.smoke_test.response_sha256",
    )
    finish_reason = _required_non_empty_string(
        smoke_test.get("finish_reason"),
        label=f"{label}.smoke_test.finish_reason",
    )
    if finish_reason.lower() in {"length", "max_tokens", "max_output_tokens"}:
        raise ValueError(f"{label}.smoke_test.finish_reason indicates truncation.")
    _required_sha256(
        smoke_test.get("usage_sha256"),
        label=f"{label}.smoke_test.usage_sha256",
    )
    controls = smoke_test.get("generation_controls")
    if not isinstance(controls, dict):
        raise ValueError(f"{label}.smoke_test.generation_controls must be an object.")
    if (
        controls.get("temperature") != 0
        or controls.get("top_p") != 1
        or not isinstance(controls.get("seed"), int)
        or isinstance(controls.get("seed"), bool)
        or not isinstance(controls.get("max_tokens"), int)
        or isinstance(controls.get("max_tokens"), bool)
        or controls.get("max_tokens", 0) <= 0
        or controls.get("stream") is not False
        or controls.get("structured_output") is not True
    ):
        raise ValueError(
            f"{label}.smoke_test.generation_controls do not prove the frozen controls."
        )
    _required_sha256(
        controls.get("json_schema_sha256"),
        label=f"{label}.smoke_test.generation_controls.json_schema_sha256",
    )


def _validate_route_verification_policy(registry: dict[str, Any]) -> dict[str, Any]:
    policy = registry.get("route_verification_policy")
    if not isinstance(policy, dict):
        raise ValueError("Model registry must define route_verification_policy.")
    if policy.get("schema_version") != ROUTE_VERIFICATION_POLICY_SCHEMA_VERSION:
        raise ValueError(
            "route_verification_policy.schema_version must be "
            f"{ROUTE_VERIFICATION_POLICY_SCHEMA_VERSION}."
        )
    max_age_hours = policy.get("max_age_hours")
    if (
        not isinstance(max_age_hours, int)
        or isinstance(max_age_hours, bool)
        or max_age_hours <= 0
    ):
        raise ValueError(
            "route_verification_policy.max_age_hours must be a positive integer."
        )
    if policy.get("require_complete_registry_bundle") is not True:
        raise ValueError(
            "route_verification_policy must require a complete registry bundle."
        )
    return policy


def _validate_verification_bundle(
    registry: dict[str, Any],
    *,
    targets_by_id: dict[str, dict[str, Any]],
) -> None:
    """Recompute and cross-check the complete retained live smoke bundle."""

    verified_targets = {
        target_id: target
        for target_id, target in targets_by_id.items()
        if target.get("verification_status") == "verified"
    }
    bundle = registry.get("verification_bundle")
    if not verified_targets:
        if bundle is not None:
            raise ValueError(
                "An all-unverified registry must not carry a verification_bundle."
            )
        return
    if not isinstance(bundle, dict):
        raise ValueError(
            "A registry with verified routes must embed verification_bundle."
        )
    if bundle.get("schema_version") != 2:
        raise ValueError("verification_bundle.schema_version must be 2.")
    recorded_bundle_sha256 = _required_sha256(
        bundle.get("bundle_sha256"),
        label="verification_bundle.bundle_sha256",
    )
    unhashed_bundle = dict(bundle)
    unhashed_bundle.pop("bundle_sha256", None)
    if _canonical_json_sha256(unhashed_bundle) != recorded_bundle_sha256:
        raise ValueError("verification_bundle.bundle_sha256 does not match its content.")
    bundle_timestamp = _required_utc_timestamp(
        bundle.get("verified_at_utc"),
        label="verification_bundle.verified_at_utc",
    )
    bundle_completed_at = datetime.fromisoformat(
        bundle_timestamp.replace("Z", "+00:00")
    )
    endpoint = _required_non_empty_string(
        bundle.get("endpoint"),
        label="verification_bundle.endpoint",
    )
    if validate_endpoint_base_url("inference_hub", endpoint) != endpoint:
        raise ValueError("verification_bundle.endpoint is not canonical.")
    if bundle.get("registry_version") != registry.get("registry_version"):
        raise ValueError("verification_bundle.registry_version does not match.")
    roster_sha256 = _required_sha256(
        bundle.get("routing_roster_sha256"),
        label="verification_bundle.routing_roster_sha256",
    )
    if roster_sha256 != _routing_roster_sha256(registry):
        raise ValueError("verification_bundle does not bind the current routing roster.")
    catalog_hashes = bundle.get("catalog_source_payload_sha256")
    if not isinstance(catalog_hashes, dict) or set(catalog_hashes) != {"models"}:
        raise ValueError("verification_bundle must bind the authorized /models payload.")
    for source_name, digest in catalog_hashes.items():
        _required_sha256(
            digest,
            label=f"verification_bundle.catalog_source_payload_sha256.{source_name}",
        )
    expected_cohorts = [
        {
            "id": cohort_id,
            "version": cohort["version"],
            "target_ids": list(cohort["targets"]),
        }
        for cohort_id, cohort in registry.get("cohorts", {}).items()
    ]
    if bundle.get("cohorts") != expected_cohorts:
        raise ValueError(
            "verification_bundle cohorts do not match the complete registry."
        )

    raw_bundle_targets = bundle.get("targets")
    if not isinstance(raw_bundle_targets, list):
        raise ValueError("verification_bundle.targets must be a list.")
    if bundle.get("target_count") != len(raw_bundle_targets):
        raise ValueError("verification_bundle.target_count does not match targets.")
    expected_ids = {
        target_id
        for target_id, target in targets_by_id.items()
        if target.get("provider", "").lower() == "inference_hub"
    }
    bundle_targets: dict[str, dict[str, Any]] = {}
    bundle_routes: set[str] = set()
    for index, raw_bundle_target in enumerate(raw_bundle_targets):
        label = f"verification_bundle.targets[{index}]"
        if not isinstance(raw_bundle_target, dict):
            raise ValueError(f"{label} must be an object.")
        target_id = _required_non_empty_string(
            raw_bundle_target.get("target_id"),
            label=f"{label}.target_id",
        )
        if target_id in bundle_targets:
            raise ValueError(f"Duplicate target in verification_bundle: {target_id}")
        registry_target = targets_by_id.get(target_id)
        if registry_target is None:
            raise ValueError(
                f"verification_bundle references unknown target: {target_id}"
            )
        route = _required_non_empty_string(
            raw_bundle_target.get("route"),
            label=f"{label}.route",
        )
        if route in bundle_routes:
            raise ValueError(f"Duplicate route in verification_bundle: {route}")
        if route != registry_target.get("route"):
            raise ValueError(f"{label}.route does not match the registry.")
        if raw_bundle_target.get("upstream_provider") != registry_target.get(
            "upstream_provider"
        ):
            raise ValueError(f"{label}.upstream_provider does not match the registry.")
        evidence = raw_bundle_target.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError(f"{label}.evidence must be an object.")
        if (
            evidence.get("verification_status") != "verified"
            or evidence.get("route_source") not in AUTHORITATIVE_ROUTE_SOURCES
            or evidence.get("endpoint") != endpoint
            or evidence.get("requested_route") != route
            or evidence.get("provider_response_model") != route
        ):
            raise ValueError(f"{label}.evidence route identity is invalid.")
        evidence_timestamp = _required_utc_timestamp(
            evidence.get("verified_at_utc"),
            label=f"{label}.evidence.verified_at_utc",
        )
        if datetime.fromisoformat(
            evidence_timestamp.replace("Z", "+00:00")
        ) > bundle_completed_at:
            raise ValueError(f"{label}.evidence completed after its bundle.")
        if evidence.get("catalog_source_payload_sha256") != catalog_hashes:
            raise ValueError(f"{label}.evidence catalog binding does not match.")
        nested = evidence.get("verification_evidence")
        if nested != registry_target.get("verification_evidence"):
            raise ValueError(f"{label}.evidence does not match registry evidence.")
        response = evidence.get("response")
        if not isinstance(response, dict) or response.get(
            "structured_output_validated"
        ) is not True:
            raise ValueError(f"{label}.evidence lacks validated structured output.")
        usage = response.get("usage")
        if not isinstance(usage, dict) or not any(
            isinstance(usage.get(key), int)
            and not isinstance(usage.get(key), bool)
            and usage.get(key, 0) > 0
            for key in ("completion_tokens", "output_tokens", "total_tokens")
        ):
            raise ValueError(f"{label}.evidence lacks positive token usage.")
        if response.get("usage_sha256") != _canonical_json_sha256(usage):
            raise ValueError(f"{label}.evidence usage hash does not match.")
        bundle_targets[target_id] = raw_bundle_target
        bundle_routes.add(route)
    if set(bundle_targets) != expected_ids:
        raise ValueError(
            "verification_bundle must cover every InferenceHub registry target exactly."
        )
    if not set(verified_targets).issubset(bundle_targets):
        raise ValueError("Verified registry targets are missing from verification_bundle.")


def _normalize_provider_name(provider: str) -> str:
    normalized = provider.strip().lower().replace("-", "_")
    return {
        "cerebris": "cerebras",
        "inferencehub": "inference_hub",
        "olama": "ollama",
        "openaicompatible": "openai_compatible",
        "x.ai": "xai",
    }.get(normalized, normalized)


def validate_endpoint_base_url(profile_id: str, value: str) -> str:
    """Validate and normalize a configured provider API base URL.

    NVIDIA's internal InferenceHub and public API Catalog are deliberately
    separate trust domains. Their endpoint profiles therefore accept only the
    exact HTTPS host assigned to that profile and never accept URL-embedded
    credentials, ports, queries, or fragments.
    """

    normalized_profile = _normalize_provider_name(profile_id)
    normalized_url = _required_non_empty_string(
        value,
        label=f"endpoint_profiles.{normalized_profile}.base_url",
    ).rstrip("/")
    if any(character.isspace() for character in normalized_url):
        raise ValueError(f"{normalized_profile} base URL must not contain whitespace.")
    try:
        parsed_url = urlsplit(normalized_url)
        parsed_port = parsed_url.port
    except ValueError as error:
        raise ValueError(
            f"{normalized_profile} base URL is not a valid absolute URL."
        ) from error
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(
            f"{normalized_profile} base URL must be an absolute HTTP(S) URL."
        )

    exact_host = STRICT_ENDPOINT_HOSTS.get(normalized_profile)
    if exact_host is None:
        strict_profile_by_host = {
            host: profile for profile, host in STRICT_ENDPOINT_HOSTS.items()
        }
        required_profile = strict_profile_by_host.get(parsed_url.hostname or "")
        if required_profile is not None:
            raise ValueError(
                f"{normalized_profile} base URL cannot target the strict "
                f"{required_profile} host; use its dedicated endpoint profile."
            )
        return normalized_url
    if parsed_url.scheme != "https":
        raise ValueError(f"{normalized_profile} base URL must use HTTPS.")
    if parsed_url.hostname != exact_host:
        raise ValueError(
            f"{normalized_profile} base URL must use exact host {exact_host}."
        )
    if parsed_url.username is not None or parsed_url.password is not None:
        raise ValueError(
            f"{normalized_profile} base URL must not contain URL credentials."
        )
    if parsed_port is not None:
        raise ValueError(f"{normalized_profile} base URL must not specify a port.")
    exact_path = STRICT_ENDPOINT_PATHS[normalized_profile]
    if parsed_url.path != exact_path:
        raise ValueError(
            f"{normalized_profile} base URL must use exact path {exact_path}."
        )
    if "?" in normalized_url or parsed_url.query:
        raise ValueError(f"{normalized_profile} base URL must not contain a query.")
    if "#" in normalized_url or parsed_url.fragment:
        raise ValueError(
            f"{normalized_profile} base URL must not contain a fragment."
        )
    return normalized_url


def _validate_model_registry(registry: Any) -> dict[str, Any]:
    if not isinstance(registry, dict):
        raise ValueError(f"Model registry must be a JSON object: {MODEL_REGISTRY_PATH}")
    if registry.get("schema_version") != 1:
        raise ValueError("Model registry schema_version must be 1.")
    _required_non_empty_string(
        registry.get("registry_version"),
        label="registry_version",
    )
    _validate_route_verification_policy(registry)

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
            validate_endpoint_base_url(profile_id, normalized_url)

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
        verification_status = raw_target.get("verification_status")
        route_source = raw_target.get("route_source")
        if target["provider"].lower() == "inference_hub":
            verification_status = _required_non_empty_string(
                verification_status,
                label=f"targets[{index}].verification_status",
            )
            route_source = _required_non_empty_string(
                route_source,
                label=f"targets[{index}].route_source",
            )
            if verification_status not in {"verified", "unverified"}:
                raise ValueError(
                    f"targets[{index}].verification_status must be verified or "
                    "unverified."
                )
            if verification_status == "verified":
                if route_source not in AUTHORITATIVE_ROUTE_SOURCES:
                    allowed = ", ".join(sorted(AUTHORITATIVE_ROUTE_SOURCES))
                    raise ValueError(
                        f"Target {target['id']} verified route_source must be one "
                        f"of: {allowed}."
                    )
                _validate_verification_evidence(
                    raw_target,
                    index=index,
                    route=target["route"],
                )
            else:
                if route_source not in UNVERIFIED_ROUTE_SOURCES:
                    allowed = ", ".join(sorted(UNVERIFIED_ROUTE_SOURCES))
                    raise ValueError(
                        f"Target {target['id']} unverified route_source must be one "
                        f"of: {allowed}."
                    )
                if raw_target.get("verification_evidence") is not None:
                    raise ValueError(
                        f"Target {target['id']} must not carry verification evidence "
                        "while unverified."
                    )
            target["verification_status"] = verification_status
            target["route_source"] = route_source
            if verification_status == "verified":
                target["verification_evidence"] = raw_target["verification_evidence"]
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

    _validate_verification_bundle(registry, targets_by_id=targets_by_id)

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


def model_routing_roster_hash() -> str:
    """Hash exact routes/cohorts without circular verification fields."""

    return _routing_roster_sha256(load_model_registry())


def require_fresh_route_verification(
    entry: dict[str, Any],
    *,
    now_utc: datetime | None = None,
) -> None:
    """Fail unless a resolved route has complete, fresh live smoke evidence."""

    if entry.get("verification_status") != "verified":
        raise ValueError("Model route is not verified.")
    evidence = entry.get("verification_evidence")
    bundle = entry.get("verification_bundle")
    policy = entry.get("route_verification_policy")
    if not isinstance(evidence, dict) or not isinstance(bundle, dict):
        raise ValueError("Verified model route lacks its retained evidence bundle.")
    if not isinstance(policy, dict):
        raise ValueError("Verified model route lacks its freshness policy.")
    bundle_timestamp = _required_utc_timestamp(
        bundle.get("verified_at_utc"),
        label="verification_bundle.verified_at_utc",
    )
    evidence_timestamp = _required_utc_timestamp(
        evidence.get("verified_at_utc"),
        label="verification_evidence.verified_at_utc",
    )
    bundle_completed_at = datetime.fromisoformat(
        bundle_timestamp.replace("Z", "+00:00")
    )
    completed_at = datetime.fromisoformat(
        evidence_timestamp.replace("Z", "+00:00")
    )
    now = now_utc or datetime.now(timezone.utc)
    if now.utcoffset() != timedelta(0):
        raise ValueError("now_utc must be timezone-aware UTC.")
    if completed_at > now + timedelta(minutes=5):
        raise ValueError("Route verification timestamp is implausibly in the future.")
    if completed_at > bundle_completed_at:
        raise ValueError("Route verification completed after its evidence bundle.")
    max_age_hours = policy.get("max_age_hours")
    if not isinstance(max_age_hours, int) or isinstance(max_age_hours, bool):
        raise ValueError("Route verification freshness policy is invalid.")
    if now - completed_at > timedelta(hours=max_age_hours):
        raise ValueError(
            "Route verification evidence is stale; rerun the exact cohort smoke gate."
        )


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
        "routing_roster_hash": model_routing_roster_hash(),
        "route_verification_policy": dict(registry["route_verification_policy"]),
        "verification_bundle": registry.get("verification_bundle"),
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
            "routing_roster_hash": model_routing_roster_hash(),
            "route_verification_policy": dict(
                registry["route_verification_policy"]
            ),
            "verification_bundle": registry.get("verification_bundle"),
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
