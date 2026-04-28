# print("[API CALL] Hello, World!")

import json
import os
import re
from typing import Any, Callable

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_SITE_URL = "https://openrouter.ai"
OPENROUTER_APP_NAME = "llm-altruism"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
XAI_API_HOST = "api.x.ai"
OLLAMA_MODEL_ALIASES = {
    "aratan/qwen3.5-uncensored": "aratan/qwen3.5-uncensored:9b",
}


class OllamaConnectionError(RuntimeError):
    pass


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
    timeout: float | None = None,
    keep_alive: float | str | None = None,
) -> str:
    provider_key = _normalize_provider(provider)

    dispatch = {
        "anthropic": _query_anthropic,
        "openai": _query_openai,
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

    request_kwargs = {
        "model": model,
        "system_prompt": system_prompt,
        "query": query,
        "json_mode": json_mode,
        "json_schema": json_schema,
        "temperature": temperature,
        "timeout": timeout,
    }
    if provider_key == "ollama":
        request_kwargs["keep_alive"] = keep_alive

    return dispatch[provider_key](**request_kwargs)


def _normalize_provider(provider: str) -> str:
    provider_key = provider.strip().lower()
    aliases = {
        "cerebris": "cerebras",
        "olama": "ollama",
        "x.ai": "xai",
    }
    return aliases.get(provider_key, provider_key)


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
        raise ValueError(f"{provider}/{model} returned empty assistant content.")
    return text


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

    raise ValueError("Expected valid JSON output but no parseable JSON object was found.")


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
    timeout: float | None,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=_ensure_env("OPENAI_API_KEY"))
    if timeout is not None:
        client = client.with_options(timeout=timeout)

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, query),
    }
    if temperature is not None:
        payload["temperature"] = temperature

    response_format = _build_openai_response_format(json_mode, json_schema)
    if response_format is not None:
        payload["response_format"] = response_format

    response = client.chat.completions.create(**payload)
    text = _extract_content(response.choices[0].message.content)
    return _finalize_text("openai", model, text)


def _query_nvidia(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    timeout: float | None,
) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=_ensure_env("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_BASE_URL", NVIDIA_BASE_URL),
    )
    if timeout is not None:
        client = client.with_options(timeout=timeout)

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, query),
    }
    if temperature is not None:
        payload["temperature"] = temperature

    response_format = _build_openai_response_format(json_mode, json_schema)
    if response_format is not None:
        payload["response_format"] = response_format

    response = client.chat.completions.create(**payload)
    text = _extract_content(response.choices[0].message.content)
    return _finalize_text("nvidia", model, text)


def _query_groq(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
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
    if temperature is not None:
        payload["temperature"] = temperature

    response_format = _build_openai_response_format(json_mode, json_schema)
    if response_format is not None:
        payload["response_format"] = response_format

    response = client.chat.completions.create(**payload)
    text = _extract_content(response.choices[0].message.content)
    return _finalize_text("groq", model, text)


def _query_cerebras(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    timeout: float | None,
) -> str:
    from cerebras.cloud.sdk import Cerebras

    client = Cerebras(api_key=_ensure_env("CEREBRAS_API_KEY"))

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, query),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if timeout is not None:
        payload["timeout"] = timeout

    response_format = _build_openai_response_format(json_mode, json_schema)
    if response_format is not None:
        payload["response_format"] = response_format

    response = client.chat.completions.create(**payload)
    text = _extract_content(response.choices[0].message.content)
    return _finalize_text("cerebras", model, text)


def _query_openrouter(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
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
    text = _extract_content(response.choices[0].message.content)
    return _finalize_text("openrouter", model, text)


def _query_ollama(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
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
    think_value = _ollama_think_value(model=resolved_model, json_mode=json_mode)
    if think_value is not None:
        payload["think"] = think_value

    schema = _resolve_schema(json_schema)
    if json_mode and schema is not None:
        payload["format"] = schema
    elif json_mode:
        payload["format"] = "json"

    options: dict[str, Any] = {"num_predict": OLLAMA_NUM_PREDICT}
    if temperature is not None:
        options["temperature"] = temperature
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
            message = response.get("message", {})
            text = _extract_content(message.get("content"))
            if json_mode:
                if text:
                    return _normalize_json_text(text, schema=schema)
                thinking_text = _extract_content(message.get("thinking"))
                if thinking_text:
                    return _normalize_json_text(thinking_text, schema=schema)
            return _finalize_text("ollama", resolved_model, text)
        if not _is_ollama_not_found_error(error):
            raise
        try:
            _pull_ollama_model(client, resolved_model)
        except Exception as pull_error:
            if _is_ollama_transport_error(pull_error):
                _raise_ollama_connection_error(pull_error)
            raise
        response = client.chat(**payload)
    message = response.get("message", {})
    text = _extract_content(message.get("content"))
    if json_mode:
        if text:
            return _normalize_json_text(text, schema=schema)
        thinking_text = _extract_content(message.get("thinking"))
        if thinking_text:
            return _normalize_json_text(thinking_text, schema=schema)
    return _finalize_text("ollama", resolved_model, text)


def _query_anthropic(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
    timeout: float | None,
) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=_ensure_env("ANTHROPIC_API_KEY"))

    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 2048,
    }
    if temperature is not None:
        payload["temperature"] = temperature
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

    text = _extract_content(response.content)
    return _finalize_text("anthropic", model, text)


def _query_xai(
    *,
    model: str,
    system_prompt: str,
    query: str,
    json_mode: bool,
    json_schema: dict[str, Any] | type[BaseModel] | None,
    temperature: float | None,
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
            raise ValueError(
                "xAI json_mode expected valid JSON output. "
                "Pass a Pydantic schema class for native parsing."
            ) from exc

    return _finalize_text("xai", model, text)
