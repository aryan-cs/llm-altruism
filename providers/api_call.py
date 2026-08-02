# print("[API CALL] Hello, World!")

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_SITE_URL = "https://openrouter.ai"
OPENROUTER_APP_NAME = "llm-altruism"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
XAI_API_HOST = "api.x.ai"
OLLAMA_MODEL_ALIASES = {
    "aratan/qwen3.5-uncensored": "aratan/qwen3.5-uncensored:9b",
}


class OllamaConnectionError(RuntimeError):
    pass


FAILURE_GATEWAY = "gateway"
FAILURE_PROVIDER = "provider"
FAILURE_PARSER = "parser"
FAILURE_TRANSPORT = "transport"
FAILURE_PROVENANCE_ATTRIBUTE = "llm_altruism_failure_provenance"


class ResponseParseError(ValueError):
    """The provider returned a response that the benchmark could not parse."""


class UnsupportedControlError(ValueError):
    """A requested generation control is not supported by the provider adapter."""


class ResponseModelIdentityError(RuntimeError):
    """The provider omitted or changed the model identity for a strict route."""

    def __init__(
        self,
        *,
        provider: str,
        requested_model: str,
        response_model: str | None,
    ) -> None:
        self.provider = provider
        self.requested_model = requested_model
        self.response_model = response_model
        if response_model is None:
            message = (
                f"{provider}/{requested_model} did not report a response model."
            )
        else:
            message = (
                f"{provider} response model did not match the requested route "
                f"({requested_model!r} != {response_model!r})."
            )
        super().__init__(message)


TRUNCATION_FINISH_REASONS = {
    "length",
    "max_tokens",
    "max_output_tokens",
    "model_length",
    "token_limit",
}


@dataclass(frozen=True)
class ProviderResponse:
    """Lossless-enough provider response used by auditable benchmark runs.

    ``api_call`` intentionally keeps returning a plain string for compatibility.
    New experiment code uses ``api_call_detailed`` to retain the provider body,
    visible assistant content, separately exposed reasoning, finish reason, and
    usage before any answer extraction or grading takes place.
    """

    provider: str
    model: str
    content: str
    reasoning: str
    raw_response: Any
    finish_reason: str | None
    truncated: bool | None
    usage: dict[str, Any] | None
    request_id: str | None
    requested_model: str | None = None
    response_model: str | None = None
    model_identity_match: bool | None = None

    def __post_init__(self) -> None:
        # ``model`` remains the backwards-compatible requested-model alias.
        if self.requested_model is None:
            object.__setattr__(self, "requested_model", self.model)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProviderText(str):
    """String-compatible response carrying detailed provider provenance."""

    details: ProviderResponse

    def __new__(cls, value: str, details: ProviderResponse) -> "ProviderText":
        instance = str.__new__(cls, value)
        instance.details = details
        return instance


def require_response_model_identity(response: ProviderResponse | ProviderText) -> None:
    """Fail unless a provider reports the exact model route that was requested."""

    details = response.details if isinstance(response, ProviderText) else response
    if details.response_model is None:
        raise ResponseModelIdentityError(
            provider=details.provider,
            requested_model=str(details.requested_model),
            response_model=None,
        )
    if details.model_identity_match is not True:
        raise ResponseModelIdentityError(
            provider=details.provider,
            requested_model=str(details.requested_model),
            response_model=details.response_model,
        )


@dataclass(frozen=True)
class FailureProvenance:
    category: str
    provider: str
    model: str
    upstream_provider: str | None = None
    route: str | None = None
    status_code: int | None = None
    requested_model: str | None = None
    response_model: str | None = None
    model_identity_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


OllamaProgressCallback = Callable[[str], None]


def _float_env(var_name: str, default: float) -> float:
    raw_value = os.getenv(var_name, "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _int_env(var_name: str, default: int) -> int:
    raw_value = os.getenv(var_name, "").strip()
    if not raw_value:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


OLLAMA_ADMIN_TIMEOUT_SECONDS = _float_env("OLLAMA_ADMIN_TIMEOUT_SECONDS", 10.0)
# Maximum wall-clock seconds allowed for a single generation request.
# Prevents models from hanging indefinitely on problematic prompts (e.g. safety
# refusal loops that never emit an EOS token).  Override via env var.
OLLAMA_GENERATION_TIMEOUT_SECONDS = _float_env("OLLAMA_GENERATION_TIMEOUT_SECONDS", 120.0)
# Maximum tokens the model may generate per request.  Caps runaway generation
# loops before the HTTP timeout would kick in.  Override via env var.
OLLAMA_NUM_PREDICT = _int_env("OLLAMA_NUM_PREDICT", 2048)


def api_call(
    provider: str,
    model: str,
    system_prompt: str,
    query: str,
    *,
    json_mode: bool = False,
    json_schema: dict[str, Any] | type[BaseModel] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    keep_alive: float | str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> str:
    provider_key = _normalize_provider(provider)

    dispatch = {
        "anthropic": _query_anthropic,
        "openai": _query_openai,
        "openai_compatible": _query_openai_compatible,
        "inference_hub": _query_inference_hub,
        "nvidia": _query_nvidia,
        "cerebras": _query_cerebras,
        "ollama": _query_ollama,
        "openrouter": _query_openrouter,
        "groq": _query_groq,
        "xai": _query_xai,
    }

    if provider_key not in dispatch:
        supported = ", ".join(sorted(dispatch))
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers: {supported}."
        )

    if not model or not model.strip():
        raise ValueError("Model must be a non-empty string.")
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")
    _validate_generation_controls(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )
    _validate_provider_controls(
        provider=provider_key,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )

    request_kwargs = {
        "model": model,
        "system_prompt": system_prompt,
        "query": query,
        "json_mode": json_mode,
        "json_schema": json_schema,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "reasoning_effort": reasoning_effort,
        "timeout": timeout,
    }
    if provider_key == "ollama":
        request_kwargs["keep_alive"] = keep_alive
    if provider_key in {
        "openai",
        "openai_compatible",
        "inference_hub",
        "nvidia",
    }:
        request_kwargs["base_url"] = base_url
        request_kwargs["api_key"] = api_key

    try:
        return dispatch[provider_key](**request_kwargs)
    except Exception as error:
        _annotate_failure(error, provider=provider_key, model=model)
        raise


def api_call_detailed(
    provider: str,
    model: str,
    system_prompt: str,
    query: str,
    **kwargs: Any,
) -> ProviderResponse:
    """Call a provider while retaining the response fields needed for audit.

    Adapters that predate detailed capture still return a safe compatibility
    record with an unknown finish reason. Current hosted-NVIDIA/OpenAI-compatible,
    Ollama, Anthropic, Groq, Cerebras, OpenRouter, and xAI adapters emit full
    records. Callers must not infer ``truncated=False`` when it is ``None``.
    """

    response = api_call(provider, model, system_prompt, query, **kwargs)
    if isinstance(response, ProviderText):
        return response.details
    return ProviderResponse(
        provider=_normalize_provider(provider),
        model=model,
        content=str(response),
        reasoning="",
        raw_response=str(response),
        finish_reason=None,
        truncated=None,
        usage=None,
        request_id=None,
        requested_model=model,
        response_model=None,
        model_identity_match=None,
    )


def _normalize_provider(provider: str) -> str:
    provider_key = provider.strip().lower()
    aliases = {
        "cerebris": "cerebras",
        "inference-hub": "inference_hub",
        "inferencehub": "inference_hub",
        "olama": "ollama",
        "openai-compatible": "openai_compatible",
        "openaicompatible": "openai_compatible",
        "x.ai": "xai",
    }
    return aliases.get(provider_key, provider_key)


def _validate_generation_controls(
    *,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
) -> None:
    if temperature is not None:
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
            raise TypeError("temperature must be numeric.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
    if top_p is not None:
        if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
            raise TypeError("top_p must be numeric.")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1 inclusive.")
    if max_tokens is not None:
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            raise TypeError("max_tokens must be an integer.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer.")
    if reasoning_effort is not None:
        if not isinstance(reasoning_effort, str):
            raise TypeError("reasoning_effort must be a string.")
        if not reasoning_effort.strip():
            raise ValueError("reasoning_effort must be a non-empty string.")


def _validate_provider_controls(
    *,
    provider: str,
    seed: int | None,
    reasoning_effort: str | None,
) -> None:
    unsupported: list[str] = []
    if provider in {"anthropic", "xai"}:
        if seed is not None:
            unsupported.append("seed")
        if reasoning_effort is not None:
            unsupported.append("reasoning_effort")
    if unsupported:
        raise UnsupportedControlError(
            f"{provider} does not support these controls in this adapter: "
            + ", ".join(unsupported)
        )


def _validated_base_url(value: str, *, label: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{label} must be an absolute HTTP(S) URL.")
    return normalized


def _status_code(error: Exception) -> int | None:
    direct = getattr(error, "status_code", None)
    if isinstance(direct, int):
        return direct
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _is_transport_error(error: Exception) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError, OllamaConnectionError)):
        return True
    error_type = type(error)
    label = f"{error_type.__module__}.{error_type.__name__}".lower()
    markers = ("timeout", "connect", "connection", "network", "transport")
    return any(marker in label for marker in markers)


def _registry_route(provider: str, model: str) -> dict[str, Any]:
    try:
        from agents.agent_config import resolve_model_registry_entry

        entry = resolve_model_registry_entry(provider, model)
    except Exception:
        entry = None
    return entry or {}


def _require_verified_inference_hub_route(model: str) -> dict[str, Any]:
    """Resolve only an exact, registered, verified InferenceHub callable route."""

    from agents.agent_config import resolve_model_registry_entry

    normalized_model = model.strip()
    entry = resolve_model_registry_entry("inference_hub", normalized_model)
    if entry is None:
        raise ValueError(
            f"InferenceHub route is not registered and cannot execute: {normalized_model}"
        )
    if entry.get("route") != normalized_model:
        raise ValueError(
            "InferenceHub calls must use the registry's exact callable route, not "
            f"its display label: {normalized_model}"
        )
    if entry.get("verification_status") != "verified":
        raise ValueError(
            "InferenceHub route is not executable until verified with authoritative "
            f"discovery and smoke evidence: {entry.get('id', normalized_model)}"
        )
    return entry


def classify_api_failure(
    error: Exception,
    *,
    provider: str,
    model: str,
) -> FailureProvenance:
    """Classify a failure without changing the exception raised to callers."""

    provider_key = _normalize_provider(provider)
    status_code = _status_code(error)
    route_metadata = _registry_route(provider_key, model)
    upstream_provider = route_metadata.get("upstream_provider")
    route = route_metadata.get("route")

    if isinstance(error, ResponseParseError):
        category = FAILURE_PARSER
    elif _is_transport_error(error):
        category = FAILURE_TRANSPORT
    elif status_code in {502, 503, 504} and provider_key in {
        "inference_hub",
        "nvidia",
        "openai_compatible",
        "openrouter",
    }:
        category = FAILURE_GATEWAY
    else:
        category = FAILURE_PROVIDER

    return FailureProvenance(
        category=category,
        provider=provider_key,
        model=model,
        upstream_provider=(
            str(upstream_provider) if upstream_provider is not None else None
        ),
        route=str(route) if route is not None else model,
        status_code=status_code,
        requested_model=(
            error.requested_model
            if isinstance(error, ResponseModelIdentityError)
            else None
        ),
        response_model=(
            error.response_model
            if isinstance(error, ResponseModelIdentityError)
            else None
        ),
        model_identity_match=(
            error.response_model == error.requested_model
            if isinstance(error, ResponseModelIdentityError)
            and error.response_model is not None
            else None
        ),
    )


def _annotate_failure(error: Exception, *, provider: str, model: str) -> None:
    provenance = classify_api_failure(
        error,
        provider=provider,
        model=model,
    ).to_dict()
    try:
        setattr(error, FAILURE_PROVENANCE_ATTRIBUTE, provenance)
    except Exception:
        return


def failure_provenance(
    error: Exception,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any] | None:
    """Return serializable failure origin data for logs and run metadata."""

    existing = getattr(error, FAILURE_PROVENANCE_ATTRIBUTE, None)
    if isinstance(existing, dict):
        return dict(existing)
    if provider is None or model is None:
        return None
    return classify_api_failure(error, provider=provider, model=model).to_dict()


def is_retryable_api_failure(error: Exception) -> bool:
    """Return whether repeating the same request can plausibly succeed."""

    if isinstance(error, UnsupportedControlError):
        return False
    if isinstance(error, ResponseParseError):
        return True
    provenance = failure_provenance(error)
    if provenance is None:
        return False
    if provenance.get("category") in {FAILURE_GATEWAY, FAILURE_TRANSPORT}:
        return True
    status_code = provenance.get("status_code")
    return isinstance(status_code, int) and (
        status_code in {408, 409, 425, 429} or status_code >= 500
    )


def _resolve_ollama_model_name(model: str) -> str:
    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("Ollama model must be a non-empty string.")
    return OLLAMA_MODEL_ALIASES.get(normalized_model, normalized_model)


def _ollama_model_name_matches(candidate: str, target: str) -> bool:
    normalized_candidate = _resolve_ollama_model_name(candidate)
    normalized_target = _resolve_ollama_model_name(target)
    candidate_variants = {normalized_candidate}
    target_variants = {normalized_target}
    if normalized_candidate.endswith(":latest"):
        candidate_variants.add(normalized_candidate.removesuffix(":latest"))
    else:
        candidate_variants.add(f"{normalized_candidate}:latest")
    if normalized_target.endswith(":latest"):
        target_variants.add(normalized_target.removesuffix(":latest"))
    else:
        target_variants.add(f"{normalized_target}:latest")
    return not candidate_variants.isdisjoint(target_variants)


def _build_messages(system_prompt: str, query: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    return messages


def _resolve_schema(
    json_schema: dict[str, Any] | type[BaseModel] | None,
) -> dict[str, Any] | None:
    if json_schema is None:
        return None
    if isinstance(json_schema, dict):
        return json_schema
    if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
        return json_schema.model_json_schema()
    raise TypeError(
        "json_schema must be a dict, a Pydantic BaseModel subclass, or None."
    )


def _resolve_schema_name(
    json_schema: dict[str, Any] | type[BaseModel] | None,
    default: str = "structured_response",
) -> str:
    if json_schema is None:
        return default
    if isinstance(json_schema, dict):
        title = json_schema.get("title")
        return title if isinstance(title, str) and title.strip() else default
    if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
        return json_schema.__name__
    return default


def _build_openai_response_format(
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
) -> dict[str, Any] | None:
    if not json_mode:
        return None

    schema = _resolve_schema(json_schema)
    if schema is None:
        return {"type": "json_object"}

    return {
        "type": "json_schema",
        "json_schema": {
            "name": _resolve_schema_name(json_schema),
            "strict": True,
            "schema": schema,
        },
    }


def _ensure_env(var_name: str) -> str:
    value = os.getenv(var_name, "").strip()
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value


def _resolve_api_key(api_key: str | None, *, env_name: str) -> str:
    if api_key is None:
        return _ensure_env(env_name)
    normalized = api_key.strip()
    if not normalized:
        raise ValueError("Explicit api_key must be a non-empty string.")
    return normalized


def _resolve_endpoint_connection(
    profile_id: str,
    *,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str | None, str]:
    from agents.agent_config import (
        load_endpoint_profile,
        validate_endpoint_base_url,
    )

    profile = load_endpoint_profile(profile_id)
    base_url_env = str(profile["base_url_env"])
    credential_env = str(profile["credential_env"])
    if str(profile["provider"]) in {"inference_hub", "nvidia"} and api_key is not None:
        raise ValueError(
            f"{profile['provider']} does not accept an explicit api_key override; "
            f"set {credential_env} in the environment."
        )
    resolved_base_url = (
        base_url
        if base_url is not None
        else (
            os.getenv(base_url_env, "").strip()
            or str(profile.get("default_base_url", "")).strip()
            or None
        )
    )
    if resolved_base_url is None or not resolved_base_url.strip():
        raise EnvironmentError(
            f"Missing required environment variable: {base_url_env}"
        )
    validated_base_url = validate_endpoint_base_url(
        profile_id,
        resolved_base_url,
    )
    return validated_base_url, _resolve_api_key(api_key, env_name=credential_env)


def _build_ollama_client(*, timeout: float | None = None) -> Any:
    from ollama import Client

    client_kwargs: dict[str, Any] = {
        "host": (
            os.getenv("OLLAMA_BASE_URL", "").strip()
            or os.getenv("OLLAMA_HOST", "").strip()
            or OLLAMA_BASE_URL
        ),
    }
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    api_key = os.getenv("OLLAMA_API_KEY", "").strip()
    if api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
    return Client(**client_kwargs)


def _raise_ollama_connection_error(error: Exception) -> None:
    host = (
        os.getenv("OLLAMA_BASE_URL", "").strip()
        or os.getenv("OLLAMA_HOST", "").strip()
        or OLLAMA_BASE_URL
    )
    raise OllamaConnectionError(
        "Could not connect to Ollama at "
        f"{host}. Start the Ollama server, then rerun the experiment. "
        "Example: `ollama serve`."
    ) from error


def _is_ollama_transport_error(error: Exception) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True

    error_type = type(error)
    error_name = error_type.__name__.lower()
    error_module = error_type.__module__.lower()
    return "httpx" in error_module and (
        "timeout" in error_name
        or "connect" in error_name
        or "network" in error_name
        or "transport" in error_name
    )


def _is_ollama_not_found_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code == 404:
        return True

    error_text = str(error).lower()
    return "404" in error_text or "not found" in error_text


def _is_ollama_runner_terminated_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    error_text = str(error).lower()
    if status_code != 500:
        return False
    return (
        "runner process has terminated" in error_text
        or "llama runner process has terminated" in error_text
    )


def _format_ollama_progress(update: Any) -> str:
    status = getattr(update, "status", None) or "pulling"
    completed = getattr(update, "completed", None)
    total = getattr(update, "total", None)
    if isinstance(completed, int) and isinstance(total, int) and total > 0:
        percent = completed / total * 100
        return f"{status} ({percent:.1f}% | {completed / (1024 ** 3):.1f} / {total / (1024 ** 3):.1f} GiB)"
    return status


def _pull_ollama_model(
    client: Any,
    model: str,
    progress_callback: OllamaProgressCallback | None = None,
) -> None:
    try:
        progress_stream = client.pull(model, stream=True)
        for update in progress_stream:
            if progress_callback is not None:
                progress_callback(_format_ollama_progress(update))
    except TypeError:
        client.pull(model)


def _ollama_model_available_locally_with_client(
    client: Any,
    model: str,
) -> bool:
    try:
        client.show(model)
        return True
    except Exception as error:
        if _is_ollama_transport_error(error):
            _raise_ollama_connection_error(error)
        if _is_ollama_not_found_error(error):
            return False
        raise


def _list_loaded_ollama_models_with_client(client: Any) -> list[str]:
    try:
        response = client.ps()
    except Exception as error:
        if _is_ollama_transport_error(error):
            _raise_ollama_connection_error(error)
        raise

    loaded_models: list[str] = []
    seen: set[str] = set()
    for model_info in getattr(response, "models", []) or []:
        model_name = getattr(model_info, "model", None) or getattr(model_info, "name", None)
        if not isinstance(model_name, str):
            continue
        normalized_model = model_name.strip()
        if not normalized_model or normalized_model in seen:
            continue
        seen.add(normalized_model)
        loaded_models.append(normalized_model)
    return loaded_models


def _list_local_ollama_models_with_client(client: Any) -> list[str]:
    try:
        response = client.list()
    except Exception as error:
        if _is_ollama_transport_error(error):
            _raise_ollama_connection_error(error)
        raise

    local_models: list[str] = []
    seen: set[str] = set()
    for model_info in getattr(response, "models", []) or []:
        model_name = getattr(model_info, "model", None) or getattr(model_info, "name", None)
        if not isinstance(model_name, str):
            continue
        normalized_model = model_name.strip()
        if not normalized_model or normalized_model in seen:
            continue
        seen.add(normalized_model)
        local_models.append(normalized_model)
    return local_models


def _unload_ollama_model_with_client(
    client: Any,
    model: str,
) -> None:
    try:
        client.generate(model=model, keep_alive=0)
    except Exception as error:
        if _is_ollama_transport_error(error):
            _raise_ollama_connection_error(error)
        if _is_ollama_not_found_error(error):
            return
        raise


def _delete_ollama_model_with_client(
    client: Any,
    model: str,
) -> None:
    try:
        client.delete(model)
    except Exception as error:
        if _is_ollama_transport_error(error):
            _raise_ollama_connection_error(error)
        if _is_ollama_not_found_error(error):
            return
        raise


def unload_ollama_model(model: str) -> None:
    normalized_model = _resolve_ollama_model_name(model)
    client = _build_ollama_client(timeout=OLLAMA_ADMIN_TIMEOUT_SECONDS)
    _unload_ollama_model_with_client(client, normalized_model)


def unload_all_ollama_models() -> None:
    client = _build_ollama_client(timeout=OLLAMA_ADMIN_TIMEOUT_SECONDS)
    for loaded_model in _list_loaded_ollama_models_with_client(client):
        _unload_ollama_model_with_client(client, loaded_model)


def delete_other_ollama_models(keep_model: str) -> None:
    normalized_keep_model = _resolve_ollama_model_name(keep_model)
    client = _build_ollama_client(timeout=OLLAMA_ADMIN_TIMEOUT_SECONDS)
    for local_model in _list_local_ollama_models_with_client(client):
        if _ollama_model_name_matches(local_model, normalized_keep_model):
            continue
        _delete_ollama_model_with_client(client, local_model)


def _unload_other_ollama_models_with_client(
    client: Any,
    keep_model: str,
) -> None:
    for loaded_model in _list_loaded_ollama_models_with_client(client):
        if _ollama_model_name_matches(loaded_model, keep_model):
            continue
        _unload_ollama_model_with_client(client, loaded_model)


def _ensure_ollama_model_with_client(
    client: Any,
    model: str,
    progress_callback: OllamaProgressCallback | None = None,
) -> None:
    if _ollama_model_available_locally_with_client(client, model):
        return
    try:
        _pull_ollama_model(
            client,
            model,
            progress_callback=progress_callback,
        )
    except Exception as pull_error:
        if _is_ollama_transport_error(pull_error):
            _raise_ollama_connection_error(pull_error)
        raise


def ollama_model_available_locally(model: str) -> bool:
    normalized_model = _resolve_ollama_model_name(model)
    client = _build_ollama_client(timeout=OLLAMA_ADMIN_TIMEOUT_SECONDS)
    return _ollama_model_available_locally_with_client(client, normalized_model)


def ensure_ollama_model_available(
    model: str,
    progress_callback: OllamaProgressCallback | None = None,
) -> None:
    normalized_model = _resolve_ollama_model_name(model)
    client = _build_ollama_client()
    _ensure_ollama_model_with_client(
        client,
        normalized_model,
        progress_callback=progress_callback,
    )


def _extract_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.append(_extract_content(item))
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value", "output_text"):
            extracted = _extract_content(value.get(key))
            if extracted:
                return extracted
        return json.dumps(value, ensure_ascii=False)
    text = getattr(value, "text", None)
    if text:
        return str(text).strip()
    content = getattr(value, "content", None)
    if content is not None:
        return _extract_content(content)
    output_text = getattr(value, "output_text", None)
    if output_text:
        return str(output_text).strip()
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return str(value).strip()


def _finalize_text(provider: str, model: str, text: str) -> str:
    text = text.strip()
    if not text:
        raise ResponseParseError(
            f"{provider}/{model} returned empty assistant content."
        )
    return text


def _jsonable_provider_value(value: Any) -> Any:
    """Convert an SDK response body to JSON-safe data without headers/secrets."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _jsonable_provider_value(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_provider_value(child) for child in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable_provider_value(model_dump(mode="json"))
        except TypeError:
            return _jsonable_provider_value(model_dump())
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable_provider_value(child)
            for key, child in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _string_attribute(value: Any, *names: str) -> str:
    for name in names:
        candidate = (
            value.get(name)
            if isinstance(value, dict)
            else getattr(value, name, None)
        )
        text = _extract_content(candidate)
        if text:
            return text
    return ""


def _finish_reason_is_truncated(reason: str | None) -> bool | None:
    if reason is None or not reason.strip():
        return None
    return reason.strip().lower() in TRUNCATION_FINISH_REASONS


def _provider_text(
    *,
    provider: str,
    model: str,
    response: Any,
    content: str,
    reasoning: str = "",
    finish_reason: str | None = None,
    usage: Any = None,
    request_id: Any = None,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
) -> ProviderText:
    finalized = _finalize_response_text(
        provider=provider,
        model=model,
        text=content,
        json_mode=json_mode,
        json_schema=json_schema,
    )
    normalized_reason = finish_reason.strip() if isinstance(finish_reason, str) else None
    response_model = _string_attribute(response, "model") or None
    details = ProviderResponse(
        provider=provider,
        model=model,
        content=content.strip(),
        reasoning=reasoning.strip(),
        raw_response=_jsonable_provider_value(response),
        finish_reason=normalized_reason,
        truncated=_finish_reason_is_truncated(normalized_reason),
        usage=(
            _jsonable_provider_value(usage)
            if usage is not None
            else None
        ),
        request_id=(
            str(request_id).strip()
            if request_id is not None and str(request_id).strip()
            else None
        ),
        requested_model=model,
        response_model=response_model,
        model_identity_match=(
            response_model == model
            if response_model is not None
            else None
        ),
    )
    return ProviderText(finalized, details)


def _openai_style_provider_text(
    *,
    provider: str,
    model: str,
    response: Any,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
) -> ProviderText:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    choice = choices[0]
    message = choice.get("message", {}) if isinstance(choice, dict) else choice.message
    content = _extract_content(
        message.get("content") if isinstance(message, dict) else message.content
    )
    reasoning = _string_attribute(
        message,
        "reasoning_content",
        "reasoning",
        "thinking",
        "analysis",
    )
    finish_reason = (
        choice.get("finish_reason")
        if isinstance(choice, dict)
        else getattr(choice, "finish_reason", None)
    )
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    request_id = response.get("id") if isinstance(response, dict) else getattr(response, "id", None)
    return _provider_text(
        provider=provider,
        model=model,
        response=response,
        content=content,
        reasoning=reasoning,
        finish_reason=finish_reason,
        usage=usage,
        request_id=request_id,
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _ollama_is_gpt_oss_model(model: str) -> bool:
    normalized_model = _resolve_ollama_model_name(model).lower()
    return "gpt-oss" in normalized_model


def _ollama_is_thinking_model(model: str) -> bool:
    normalized_model = _resolve_ollama_model_name(model).lower()
    thinking_markers = (
        "gpt-oss",
        "qwen3",
        "deepseek-r1",
        "deepseek-v3.1",
    )
    return any(marker in normalized_model for marker in thinking_markers)


def _ollama_think_value(
    *,
    model: str,
    json_mode: bool,
) -> bool | str | None:
    if _ollama_is_gpt_oss_model(model):
        # GPT-OSS models do not support disabling thinking via booleans.
        return "low" if json_mode else "low"
    if json_mode and _ollama_is_thinking_model(model):
        # Keep structured-output calls out of the hidden thinking path. Leaving
        # thinking enabled makes Qwen/DeepSeek-style models slow and much more
        # likely to return prose instead of the requested JSON object.
        return False
    return False


def _ollama_reasoning_effort(value: str) -> bool | str:
    normalized = value.strip().lower()
    if normalized in {"none", "off", "disabled"}:
        return False
    if normalized in {"low", "medium", "high"}:
        return normalized
    raise ValueError(
        "Ollama reasoning_effort must be one of: none, off, disabled, low, medium, high."
    )


def _strip_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _extract_last_balanced_json(text: str) -> str | None:
    candidate_starts = [index for index, char in enumerate(text) if char in "{["]
    for start in reversed(candidate_starts):
        opening = text[start]
        closing = "}" if opening == "{" else "]"
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            char = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
                continue
            if char == opening:
                depth += 1
                continue
            if char == closing:
                depth -= 1
                if depth == 0:
                    snippet = text[start : end + 1].strip()
                    try:
                        json.loads(snippet)
                    except json.JSONDecodeError:
                        break
                    return snippet
    return None


def _coerce_single_enum_action_json(
    text: str,
    schema: dict[str, Any] | None,
) -> str | None:
    if not schema:
        return None

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    action_property = properties.get("action")
    reasoning_property = properties.get("reasoning")
    enum_values = (
        action_property.get("enum")
        if isinstance(action_property, dict)
        else None
    )
    if not isinstance(reasoning_property, dict) or not isinstance(enum_values, list):
        return None

    action_values = [value for value in enum_values if isinstance(value, str)]
    if not action_values:
        return None

    matches = [
        action
        for action in action_values
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(action)}(?![A-Za-z0-9_])", text)
    ]
    if len(matches) != 1:
        return None

    reasoning = text.strip()
    if not reasoning:
        return None

    return json.dumps(
        {
            "action": matches[0],
            "reasoning": reasoning,
        },
        ensure_ascii=False,
    )


def _normalize_json_text(
    text: str,
    *,
    schema: dict[str, Any] | None = None,
) -> str:
    candidates = [
        text.strip(),
        _strip_markdown_code_fence(text),
    ]
    balanced = _extract_last_balanced_json(text)
    if balanced is not None:
        candidates.append(balanced)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            json.loads(normalized)
        except json.JSONDecodeError:
            continue
        return normalized

    coerced = _coerce_single_enum_action_json(text, schema)
    if coerced is not None:
        return coerced

    raise ResponseParseError(
        "Expected valid JSON output but no parseable JSON object was found."
    )


def _finalize_response_text(
    *,
    provider: str,
    model: str,
    text: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
) -> str:
    if json_mode:
        return _normalize_json_text(text, schema=_resolve_schema(json_schema))
    return _finalize_text(provider, model, text)


def _build_openrouter_response_format(
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
) -> dict[str, Any] | None:
    if not json_mode:
        return None

    schema = _resolve_schema(json_schema)
    if schema is None:
        return {"type": "json_object"}

    return {
        "type": "json_schema",
        "json_schema": {
            "name": _resolve_schema_name(json_schema),
            "strict": True,
            "schema": schema,
        },
    }


def _query_openai(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
    base_url: str | None,
    api_key: str | None,
) -> str:
    resolved_base_url, resolved_api_key = _resolve_endpoint_connection(
        "openai",
        base_url=base_url,
        api_key=api_key,
    )
    return _query_openai_compatible_endpoint(
        provider_label="openai",
        model=model,
        system_prompt=system_prompt,
        query=query,
        json_mode=json_mode,
        json_schema=json_schema,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )


def _apply_openai_generation_controls(
    payload: dict[str, Any],
    *,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
) -> None:
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort


def _query_openai_compatible_endpoint(
    *,
    provider_label: str,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
    base_url: str | None,
    api_key: str,
) -> str:
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url is not None:
        client_kwargs["base_url"] = _validated_base_url(
            base_url,
            label=f"{provider_label} base URL",
        )
    client = OpenAI(**client_kwargs)
    if timeout is not None:
        client = client.with_options(timeout=timeout)

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, query),
    }
    _apply_openai_generation_controls(
        payload,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )
    response_format = _build_openai_response_format(json_mode, json_schema)
    if response_format is not None:
        payload["response_format"] = response_format

    response = client.chat.completions.create(**payload)
    return _openai_style_provider_text(
        provider=provider_label,
        model=model,
        response=response,
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _query_openai_compatible(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
    base_url: str | None,
    api_key: str | None,
) -> str:
    resolved_base_url, resolved_api_key = _resolve_endpoint_connection(
        "openai_compatible",
        base_url=base_url,
        api_key=api_key,
    )
    if not resolved_base_url:
        raise EnvironmentError(
            "Missing required environment variable: OPENAI_COMPATIBLE_BASE_URL"
        )
    return _query_openai_compatible_endpoint(
        provider_label="openai_compatible",
        model=model,
        system_prompt=system_prompt,
        query=query,
        json_mode=json_mode,
        json_schema=json_schema,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )


def _query_inference_hub(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
    base_url: str | None,
    api_key: str | None,
) -> str:
    _require_verified_inference_hub_route(model)
    resolved_base_url, resolved_api_key = _resolve_endpoint_connection(
        "inference_hub",
        base_url=base_url,
        api_key=api_key,
    )
    response = _query_openai_compatible_endpoint(
        provider_label="inference_hub",
        model=model,
        system_prompt=system_prompt,
        query=query,
        json_mode=json_mode,
        json_schema=json_schema,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )
    require_response_model_identity(response)
    return response


def _query_nvidia(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
    base_url: str | None,
    api_key: str | None,
) -> str:
    resolved_base_url, resolved_api_key = _resolve_endpoint_connection(
        "nvidia",
        base_url=base_url,
        api_key=api_key,
    )
    return _query_openai_compatible_endpoint(
        provider_label="nvidia",
        model=model,
        system_prompt=system_prompt,
        query=query,
        json_mode=json_mode,
        json_schema=json_schema,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
        timeout=timeout,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )


def _query_groq(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
) -> str:
    from groq import Groq

    client = Groq(api_key=_ensure_env("GROQ_API_KEY"))
    if timeout is not None and hasattr(client, "with_options"):
        client = client.with_options(timeout=timeout)

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, query),
    }
    _apply_openai_generation_controls(
        payload,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )

    response_format = _build_openai_response_format(json_mode, json_schema)
    if response_format is not None:
        payload["response_format"] = response_format

    response = client.chat.completions.create(**payload)
    return _openai_style_provider_text(
        provider="groq",
        model=model,
        response=response,
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _query_cerebras(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
) -> str:
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=_ensure_env("CEREBRAS_API_KEY"))

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, query),
    }
    _apply_openai_generation_controls(
        payload,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )
    if timeout is not None:
        payload["timeout"] = timeout

    response_format = _build_openai_response_format(json_mode, json_schema)
    if response_format is not None:
        payload["response_format"] = response_format

    response = client.chat.completions.create(**payload)
    return _openai_style_provider_text(
        provider="cerebras",
        model=model,
        response=response,
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _query_openrouter(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
) -> str:
    from openrouter import OpenRouter

    client = OpenRouter(api_key=_ensure_env("OPENROUTER_API_KEY"))

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, query),
        "stream": False,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if seed is not None:
        payload["seed"] = seed
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    if timeout is not None:
        payload["timeout_ms"] = int(timeout * 1000)

    response_format = _build_openrouter_response_format(json_mode, json_schema)
    if response_format is not None:
        payload["response_format"] = response_format

    app_name = os.getenv("OPENROUTER_APP_NAME", OPENROUTER_APP_NAME)
    http_referer = os.getenv("OPENROUTER_HTTP_REFERER", OPENROUTER_SITE_URL)
    if app_name:
        payload["x_open_router_title"] = app_name
    if http_referer:
        payload["http_referer"] = http_referer

    response = client.chat.send(**payload)
    return _openai_style_provider_text(
        provider="openrouter",
        model=model,
        response=response,
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _query_ollama(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
    keep_alive: float | str | None,
) -> str:
    resolved_model = _resolve_ollama_model_name(model)
    admin_timeout = min(timeout, OLLAMA_ADMIN_TIMEOUT_SECONDS) if timeout is not None else OLLAMA_ADMIN_TIMEOUT_SECONDS
    prepare_per_request = keep_alive is None
    model_available = True
    if prepare_per_request:
        admin_client = _build_ollama_client(timeout=admin_timeout)
        _unload_other_ollama_models_with_client(admin_client, keep_model=resolved_model)
        model_available = _ollama_model_available_locally_with_client(admin_client, resolved_model)

    generation_timeout = timeout if timeout is not None else OLLAMA_GENERATION_TIMEOUT_SECONDS
    client = _build_ollama_client(timeout=generation_timeout)
    if not model_available:
        try:
            _pull_ollama_model(client, resolved_model)
        except Exception as pull_error:
            if _is_ollama_transport_error(pull_error):
                _raise_ollama_connection_error(pull_error)
            raise

    payload: dict[str, Any] = {
        "model": resolved_model,
        "messages": _build_messages(system_prompt, query),
        # Ollama supports keep_alive=0 to unload immediately after the response.
        "keep_alive": 0 if keep_alive is None else keep_alive,
    }
    think_value = (
        _ollama_reasoning_effort(reasoning_effort)
        if reasoning_effort is not None
        else _ollama_think_value(model=resolved_model, json_mode=json_mode)
    )
    if think_value is not None:
        payload["think"] = think_value

    schema = _resolve_schema(json_schema)
    if json_mode and schema is not None:
        payload["format"] = schema
    elif json_mode:
        payload["format"] = "json"

    options: dict[str, Any] = {
        "num_predict": max_tokens if max_tokens is not None else OLLAMA_NUM_PREDICT
    }
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p
    if seed is not None:
        options["seed"] = seed
    payload["options"] = options

    try:
        response = client.chat(**payload)
    except Exception as error:
        if _is_ollama_transport_error(error):
            _raise_ollama_connection_error(error)
        if _is_ollama_runner_terminated_error(error):
            recovery_client = _build_ollama_client(timeout=admin_timeout)
            try:
                _unload_ollama_model_with_client(recovery_client, resolved_model)
            except Exception as unload_error:
                if _is_ollama_transport_error(unload_error):
                    _raise_ollama_connection_error(unload_error)
                if (
                    not _is_ollama_not_found_error(unload_error)
                    and not _is_ollama_runner_terminated_error(unload_error)
                ):
                    raise
            retry_client = _build_ollama_client(timeout=generation_timeout)
            response = retry_client.chat(**payload)
            return _ollama_provider_text(
                response=response,
                model=resolved_model,
                json_mode=json_mode,
                json_schema=json_schema,
            )
        if not _is_ollama_not_found_error(error):
            raise
        try:
            _pull_ollama_model(client, resolved_model)
        except Exception as pull_error:
            if _is_ollama_transport_error(pull_error):
                _raise_ollama_connection_error(pull_error)
            raise
        response = client.chat(**payload)
    return _ollama_provider_text(
        response=response,
        model=resolved_model,
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _ollama_provider_text(
    *,
    response: Any,
    model: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
) -> ProviderText:
    message = response.get("message", {})
    content = _extract_content(message.get("content"))
    reasoning = _extract_content(message.get("thinking"))
    # A few Ollama thinking models place structured output in `thinking` while
    # leaving content empty. Preserve both fields and only use that fallback for
    # the legacy structured-call path.
    text_for_call = content or (reasoning if json_mode else "")
    finish_reason = response.get("done_reason")
    usage = {
        key: response[key]
        for key in (
            "prompt_eval_count",
            "eval_count",
            "prompt_eval_duration",
            "eval_duration",
        )
        if response.get(key) is not None
    }
    return _provider_text(
        provider="ollama",
        model=model,
        response=response,
        content=text_for_call,
        reasoning=reasoning,
        finish_reason=finish_reason,
        usage=usage or None,
        request_id=None,
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _query_anthropic(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_ensure_env("ANTHROPIC_API_KEY"))

    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens if max_tokens is not None else 2048,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    # Rejected centrally by _validate_provider_controls before dispatch.
    del seed, reasoning_effort
    if timeout is not None:
        payload["timeout"] = timeout

    if json_mode:
        schema = _resolve_schema(json_schema)
        if schema is None:
            payload["output_config"] = {"format": {"type": "json"}}
        else:
            payload["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "name": _resolve_schema_name(json_schema),
                    "schema": schema,
                }
            }

    response = client.messages.create(**payload)

    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type in {"thinking", "reasoning"}:
            reasoning_parts.append(
                _string_attribute(block, "thinking", "reasoning", "text")
            )
        else:
            visible_parts.append(_extract_content(block))
    text = "\n".join(part for part in visible_parts if part).strip()
    return _provider_text(
        provider="anthropic",
        model=model,
        response=response,
        content=text,
        reasoning="\n".join(part for part in reasoning_parts if part),
        finish_reason=getattr(response, "stop_reason", None),
        usage=getattr(response, "usage", None),
        request_id=getattr(response, "id", None),
        json_mode=json_mode,
        json_schema=json_schema,
    )


def _query_xai(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    seed: int | None,
    reasoning_effort: str | None,
    timeout: float | None,
) -> str:
    from xai_sdk import Client
    from xai_sdk.chat import system, user

    client_kwargs: dict[str, Any] = {"api_key": _ensure_env("XAI_API_KEY")}
    api_host = os.getenv("XAI_API_HOST", XAI_API_HOST).strip()
    if api_host:
        client_kwargs["api_host"] = api_host

    client = Client(**client_kwargs)
    response_format: str | type[BaseModel] | None = None
    if json_mode and isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
        response_format = json_schema
    elif json_mode:
        response_format = "json_object"

    chat = client.chat.create(model=model, response_format=response_format)
    if system_prompt.strip():
        chat.append(system(system_prompt))
    chat.append(user(query))

    if temperature is not None:
        chat.temperature = temperature
    if top_p is not None:
        chat.top_p = top_p
    if max_tokens is not None:
        chat.max_tokens = max_tokens
    # Rejected centrally by _validate_provider_controls before dispatch.
    del seed, reasoning_effort
    if timeout is not None:
        chat.timeout = timeout

    completion = chat.sample()
    text = _extract_content(completion)

    if json_mode:
        # xAI's native SDK parse flow is strongest with Pydantic models. When a raw
        # schema dict is supplied, we still require valid JSON text from the model.
        try:
            json.loads(text)
        except json.JSONDecodeError as exc:
            raise ResponseParseError(
                "xAI json_mode expected valid JSON output. "
                "Pass a Pydantic schema class for native parsing."
            ) from exc

    return _provider_text(
        provider="xai",
        model=model,
        response=completion,
        content=text,
        reasoning=_string_attribute(completion, "reasoning_content", "reasoning"),
        finish_reason=getattr(completion, "finish_reason", None),
        usage=getattr(completion, "usage", None),
        request_id=getattr(completion, "id", None),
        json_mode=json_mode,
        json_schema=json_schema,
    )
