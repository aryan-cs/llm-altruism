import json

import pytest

from experiments import part_0


class FakeJudge:
    def __init__(
        self,
        id_: str,
        provider_: str,
        model_: str,
        system_prompt_: str = "",
        *,
        json_schema_=None,
        temperature_=None,
        timeout_=None,
    ) -> None:
        del id_, system_prompt_, json_schema_, temperature_, timeout_
        self.provider = provider_
        self.model = model_

    def query(self, query: str, json_mode: bool = False) -> str:
        del query, json_mode
        if self.provider == "cerebras":
            raise OSError("Missing required environment variable: CEREBRAS_API_KEY")
        return json.dumps({"verdict": "denied", "reason": "fallback judge worked"})


class AlwaysUnavailableJudge(FakeJudge):
    def query(self, query: str, json_mode: bool = False) -> str:
        del query, json_mode
        raise OSError(f"Missing required environment variable: {self.provider.upper()}_API_KEY")


def test_judge_response_falls_back_to_next_configured_judge(monkeypatch) -> None:
    monkeypatch.setattr(part_0, "BaseAgent", FakeJudge)
    monkeypatch.setattr(
        part_0,
        "JUDGE_PROVIDERS",
        [
            {"provider": "cerebras", "model": "llama3.1-8b"},
            {"provider": "ollama", "model": "gpt-oss:20b"},
        ],
    )

    judge = FakeJudge(
        id_="judge",
        provider_="cerebras",
        model_="llama3.1-8b",
        system_prompt_="sys",
    )

    (verdict, reason), selected_judge = part_0.judge_response(
        judge,
        "prompt",
        "response",
        "",
    )

    assert verdict == "denied"
    assert reason == "fallback judge worked"
    assert selected_judge.provider == "ollama"
    assert selected_judge.model == "gpt-oss:20b"


def test_judge_response_exits_when_all_fallbacks_are_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(part_0, "BaseAgent", AlwaysUnavailableJudge)
    monkeypatch.setattr(
        part_0,
        "JUDGE_PROVIDERS",
        [
            {"provider": "cerebras", "model": "llama3.1-8b"},
            {"provider": "openai", "model": "gpt-4.1-mini"},
        ],
    )

    judge = AlwaysUnavailableJudge(
        id_="judge",
        provider_="cerebras",
        model_="llama3.1-8b",
        system_prompt_="sys",
    )

    with pytest.raises(SystemExit) as exc_info:
        part_0.judge_response(
            judge,
            "prompt",
            "response",
            "",
        )

    assert exc_info.value.code == 1
