import importlib
import json
import os
import sys
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import BaseModel

from agents.base_agent import BaseAgent

api_call_module = importlib.import_module("providers.api_call")


class EchoSchema(BaseModel):
    value: str


def _install_openai_module(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    class FakeOpenAIClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def with_options(self, **kwargs):
            self.kwargs.update(kwargs)
            return self

        def _create(self, **kwargs):
            calls.append({"client": dict(self.kwargs), "request": kwargs})
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"value":"ok"}')
                    )
                ]
            )

    module = ModuleType("openai")
    module.OpenAI = FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", module)
    return calls


def _install_groq_module(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    class FakeGroqClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def with_options(self, **kwargs):
            self.kwargs.update(kwargs)
            return self

        def _create(self, **kwargs):
            calls.append({"client": dict(self.kwargs), "request": kwargs})
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"value":"ok"}')
                    )
                ]
            )

    module = ModuleType("groq")
    module.Groq = FakeGroqClient
    monkeypatch.setitem(sys.modules, "groq", module)
    return calls


def _install_cerebras_module(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    class FakeCerebrasClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            calls.append({"client": dict(self.kwargs), "request": kwargs})
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"value":"ok"}')
                    )
                ]
            )

    cerebras_cloud = ModuleType("cerebras.cloud")
    sdk = ModuleType("cerebras.cloud.sdk")
    sdk.Cerebras = FakeCerebrasClient
    monkeypatch.setitem(sys.modules, "cerebras", ModuleType("cerebras"))
    monkeypatch.setitem(sys.modules, "cerebras.cloud", cerebras_cloud)
    monkeypatch.setitem(sys.modules, "cerebras.cloud.sdk", sdk)
    return calls


def _install_openrouter_module(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    class FakeOpenRouterClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = SimpleNamespace(send=self._send)

        def _send(self, **kwargs):
            calls.append({"client": dict(self.kwargs), "request": kwargs})
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"value":"ok"}')
                    )
                ]
            )

    module = ModuleType("openrouter")
    module.OpenRouter = FakeOpenRouterClient
    monkeypatch.setitem(sys.modules, "openrouter", module)
    return calls


def _install_ollama_module(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    class FakeResponseError(Exception):
        def __init__(self, error: str, status_code: int = -1):
            super().__init__(error)
            self.error = error
            self.status_code = status_code

    class FakeOllamaClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def show(self, model: str):
            calls.append({"client": dict(self.kwargs), "show": model})
            return {"model": model}

        def pull(self, model: str):
            calls.append({"client": dict(self.kwargs), "pull": model})
            return {"status": "success"}

        def chat(self, **kwargs):
            calls.append({"client": dict(self.kwargs), "request": kwargs})
            return {"message": {"content": '{"value":"ok"}'}}

    module = ModuleType("ollama")
    module.Client = FakeOllamaClient
    module.ResponseError = FakeResponseError
    monkeypatch.setitem(sys.modules, "ollama", module)
    return calls


def _install_anthropic_module(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text='{"value":"ok"}')])

    class FakeBetaMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text='{"value":"ok"}')])

    class FakeAnthropicClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = FakeMessages()
            self.beta = SimpleNamespace(messages=FakeBetaMessages())

    module = ModuleType("anthropic")
    module.Anthropic = FakeAnthropicClient
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return calls


def _install_xai_module(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    class FakeChatSession:
        def __init__(self, model: str):
            self.model = model
            self.messages: list[dict] = []

        def append(self, message):
            self.messages.append(message)

        def parse(self, schema):
            calls.append({"model": self.model, "messages": list(self.messages), "schema": schema})
            return schema(value="ok")

        def sample(self):
            calls.append({"model": self.model, "messages": list(self.messages)})
            return SimpleNamespace(content='{"value":"ok"}')

    class FakeChatAPI:
        def create(self, model: str, **kwargs):
            session = FakeChatSession(model)
            session.response_format = kwargs.get("response_format")
            return session

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.chat = FakeChatAPI()

    chat_module = ModuleType("xai_sdk.chat")
    chat_module.system = lambda content: {"role": "system", "content": content}
    chat_module.user = lambda content: {"role": "user", "content": content}

    root = ModuleType("xai_sdk")
    root.Client = FakeClient

    monkeypatch.setitem(sys.modules, "xai_sdk", root)
    monkeypatch.setitem(sys.modules, "xai_sdk.chat", chat_module)
    return calls


@pytest.mark.parametrize(
    ("provider", "env_name", "installer"),
    [
        ("openai", "OPENAI_API_KEY", _install_openai_module),
        ("nvidia", "NVIDIA_API_KEY", _install_openai_module),
        ("groq", "GROQ_API_KEY", _install_groq_module),
        ("cerebras", "CEREBRAS_API_KEY", _install_cerebras_module),
        ("openrouter", "OPENROUTER_API_KEY", _install_openrouter_module),
        ("ollama", None, _install_ollama_module),
        ("anthropic", "ANTHROPIC_API_KEY", _install_anthropic_module),
        ("xai", "XAI_API_KEY", _install_xai_module),
    ],
)
def test_api_call_dispatches_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    env_name: str | None,
    installer,
) -> None:
    if env_name is not None:
        monkeypatch.setenv(env_name, "test-key")

    calls = installer(monkeypatch)
    result = api_call_module.api_call(
        provider,
        "test-model",
        "system prompt",
        "user prompt",
        json_mode=True,
        json_schema=EchoSchema,
    )

    assert json.loads(result) == {"value": "ok"}
    assert calls


def test_api_call_uses_openai_compatible_base_url_for_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_openai_module(monkeypatch)
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")

    api_call_module.api_call("nvidia", "nemotron", "sys", "query")

    assert calls[0]["client"]["base_url"] == os.getenv(
        "NVIDIA_BASE_URL", api_call_module.NVIDIA_BASE_URL
    )


def test_api_call_raises_for_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        api_call_module.api_call("unknown", "model", "sys", "query")


def test_api_call_raises_for_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_openai_module(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        api_call_module.api_call("openai", "model", "sys", "query")


def test_api_call_raises_for_empty_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyOpenAIClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=""))]
                    )
                )
            )

    module = ModuleType("openai")
    module.OpenAI = EmptyOpenAIClient
    monkeypatch.setitem(sys.modules, "openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with pytest.raises(ValueError, match="empty assistant content"):
        api_call_module.api_call("openai", "model", "sys", "query")


def test_base_agent_forwards_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_api_call(provider, model, system_prompt, query, **kwargs):
        captured["provider"] = provider
        captured["model"] = model
        captured["system_prompt"] = system_prompt
        captured["query"] = query
        captured["kwargs"] = kwargs
        return '{"value":"ok"}'

    monkeypatch.setattr("agents.base_agent.api_call", fake_api_call)

    agent = BaseAgent(
        id_="A",
        provider_="openai",
        model_="gpt-test",
        system_prompt_="sys",
        json_schema_=EchoSchema,
        temperature_=0.2,
        timeout_=15,
    )

    result = agent.query("query", json_mode=True)

    assert json.loads(result) == {"value": "ok"}
    assert captured["provider"] == "openai"
    assert captured["model"] == "gpt-test"
    assert captured["system_prompt"] == "sys"
    assert captured["query"] == "query"
    assert captured["kwargs"]["json_mode"] is True
    assert captured["kwargs"]["json_schema"] is EchoSchema
    assert captured["kwargs"]["temperature"] == 0.2
    assert captured["kwargs"]["timeout"] == 15


def test_ensure_ollama_model_available_pulls_missing_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class FakeResponseError(Exception):
        def __init__(self, error: str, status_code: int = -1):
            super().__init__(error)
            self.error = error
            self.status_code = status_code

    class FakeOllamaClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def show(self, model: str):
            calls.append({"show": model, "client": dict(self.kwargs)})
            raise FakeResponseError("model not found", 404)

        def pull(self, model: str, *, stream: bool = False):
            calls.append({"pull": model, "stream": stream, "client": dict(self.kwargs)})
            if stream:
                return iter(
                    [
                        SimpleNamespace(status="pulling manifest"),
                        SimpleNamespace(status="downloading", completed=50, total=100),
                    ]
                )
            return {"status": "success"}

    module = ModuleType("ollama")
    module.Client = FakeOllamaClient
    module.ResponseError = FakeResponseError
    monkeypatch.setitem(sys.modules, "ollama", module)

    api_call_module.ensure_ollama_model_available("llama3.1:8b")

    assert calls == [
        {"show": "llama3.1:8b", "client": {"host": api_call_module.OLLAMA_BASE_URL}},
        {
            "pull": "llama3.1:8b",
            "stream": True,
            "client": {"host": api_call_module.OLLAMA_BASE_URL},
        },
    ]


def test_api_call_ollama_pulls_missing_model_before_chat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class FakeResponseError(Exception):
        def __init__(self, error: str, status_code: int = -1):
            super().__init__(error)
            self.error = error
            self.status_code = status_code

    class FakeOllamaClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def show(self, model: str):
            calls.append({"show": model, "client": dict(self.kwargs)})
            raise FakeResponseError("model not found", 404)

        def pull(self, model: str, *, stream: bool = False):
            calls.append({"pull": model, "stream": stream, "client": dict(self.kwargs)})
            if stream:
                return iter([SimpleNamespace(status="downloading", completed=100, total=100)])
            return {"status": "success"}

        def chat(self, **kwargs):
            calls.append({"request": kwargs, "client": dict(self.kwargs)})
            return {"message": {"content": '{"value":"ok"}'}}

    module = ModuleType("ollama")
    module.Client = FakeOllamaClient
    module.ResponseError = FakeResponseError
    monkeypatch.setitem(sys.modules, "ollama", module)

    result = api_call_module.api_call("ollama", "llama3.1:8b", "sys", "query")

    assert json.loads(result) == {"value": "ok"}
    assert calls == [
        {"show": "llama3.1:8b", "client": {"host": api_call_module.OLLAMA_BASE_URL}},
        {
            "pull": "llama3.1:8b",
            "stream": True,
            "client": {"host": api_call_module.OLLAMA_BASE_URL},
        },
        {
            "request": {
                "model": "llama3.1:8b",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "query"},
                ],
            },
            "client": {"host": api_call_module.OLLAMA_BASE_URL},
        },
    ]


def test_ensure_ollama_model_available_raises_clear_error_when_server_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOllamaClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def show(self, model: str):
            del model
            raise ConnectionError("server down")

    module = ModuleType("ollama")
    module.Client = FakeOllamaClient
    monkeypatch.setitem(sys.modules, "ollama", module)

    with pytest.raises(api_call_module.OllamaConnectionError, match="Could not connect to Ollama"):
        api_call_module.ensure_ollama_model_available("llama3.1:8b")


def test_ensure_ollama_model_available_emits_pull_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress_messages: list[str] = []

    class FakeResponseError(Exception):
        def __init__(self, error: str, status_code: int = -1):
            super().__init__(error)
            self.error = error
            self.status_code = status_code

    class FakeOllamaClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def show(self, model: str):
            del model
            raise FakeResponseError("model not found", 404)

        def pull(self, model: str, *, stream: bool = False):
            del model
            assert stream is True
            return iter(
                [
                    SimpleNamespace(status="pulling manifest"),
                    SimpleNamespace(status="downloading", completed=25, total=100),
                ]
            )

    module = ModuleType("ollama")
    module.Client = FakeOllamaClient
    module.ResponseError = FakeResponseError
    monkeypatch.setitem(sys.modules, "ollama", module)

    api_call_module.ensure_ollama_model_available(
        "llama3.1:8b",
        progress_callback=progress_messages.append,
    )

    assert progress_messages == [
        "pulling manifest",
        "downloading (25.0% | 0.0 / 0.0 GiB)",
    ]
