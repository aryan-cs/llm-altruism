print("[API CALL] Hello, World!")

import json
import os
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_SITE_URL = "https://openrouter.ai"
OPENROUTER_APP_NAME = "llm-altruism"
OLLAMA_BASE_URL = "http://localhost:11434"
XAI_API_HOST = "api.x.ai"


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

    return dispatch[provider_key](
        model=model,
        system_prompt=system_prompt,
        query=query,
        json_mode=json_mode,
        json_schema=json_schema,
        temperature=temperature,
        timeout=timeout,
    )


def _normalize_provider(provider: str) -> str:
    provider_key = provider.strip().lower()
    aliases = {
        "cerebris": "cerebras",
        "olama": "ollama",
        "x.ai": "xai",
    }
    return aliases.get(provider_key, provider_key)


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
) -> str:
    from ollama import Client

    client = Client(host=os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL))

    payload: dict[str, Any] = {
        "model": model,
        "messages": _build_messages(system_prompt, query),
    }

    schema = _resolve_schema(json_schema)
    if json_mode and schema is not None:
        payload["format"] = schema
    elif json_mode:
        payload["format"] = "json"

    options: dict[str, Any] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if options:
        payload["options"] = options

    if timeout is not None:
        payload["keep_alive"] = f"{int(timeout)}s"

    response = client.chat(**payload)
    text = _extract_content(response.get("message", {}).get("content"))
    return _finalize_text("ollama", model, text)


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
