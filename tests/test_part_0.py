import json
from pathlib import Path
from types import SimpleNamespace

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


class OrderedFallbackJudge(FakeJudge):
    calls: list[tuple[str, str]] = []
    failures: dict[tuple[str, str], Exception] = {}

    def query(self, query: str, json_mode: bool = False) -> str:
        del query, json_mode
        OrderedFallbackJudge.calls.append((self.provider, self.model))
        failure = OrderedFallbackJudge.failures.get((self.provider, self.model))
        if failure is not None:
            raise failure
        return json.dumps({"verdict": "denied", "reason": "ordered fallback worked"})


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


def test_judge_response_tries_ready_api_before_downloading_missing_ollama(
    monkeypatch,
) -> None:
    OrderedFallbackJudge.calls = []
    OrderedFallbackJudge.failures = {}

    monkeypatch.setattr(part_0, "BaseAgent", OrderedFallbackJudge)
    monkeypatch.setattr(
        part_0,
        "JUDGE_PROVIDERS",
        [
            {"provider": "ollama", "model": "gpt-oss:120b"},
            {"provider": "openai", "model": "gpt-4.1-mini"},
        ],
    )
    monkeypatch.setattr(
        part_0,
        "ollama_model_available_locally",
        lambda model: model != "gpt-oss:120b",
    )

    judge = OrderedFallbackJudge(
        id_="judge",
        provider_="ollama",
        model_="gpt-oss:120b",
        system_prompt_="sys",
    )

    (verdict, reason), selected_judge = part_0.judge_response(
        judge,
        "prompt",
        "response",
        "",
    )

    assert verdict == "denied"
    assert reason == "ordered fallback worked"
    assert selected_judge.provider == "openai"
    assert selected_judge.model == "gpt-4.1-mini"
    assert OrderedFallbackJudge.calls == [("openai", "gpt-4.1-mini")]


def test_judge_response_only_uses_missing_ollama_after_api_fails(
    monkeypatch,
) -> None:
    OrderedFallbackJudge.calls = []
    OrderedFallbackJudge.failures = {
        ("openai", "gpt-4.1-mini"): OSError("Missing required environment variable: OPENAI_API_KEY")
    }

    monkeypatch.setattr(part_0, "BaseAgent", OrderedFallbackJudge)
    monkeypatch.setattr(
        part_0,
        "JUDGE_PROVIDERS",
        [
            {"provider": "ollama", "model": "gpt-oss:120b"},
            {"provider": "openai", "model": "gpt-4.1-mini"},
        ],
    )
    monkeypatch.setattr(part_0, "ollama_model_available_locally", lambda _: False)

    judge = OrderedFallbackJudge(
        id_="judge",
        provider_="ollama",
        model_="gpt-oss:120b",
        system_prompt_="sys",
    )

    (verdict, reason), selected_judge = part_0.judge_response(
        judge,
        "prompt",
        "response",
        "",
    )

    assert verdict == "denied"
    assert reason == "ordered fallback worked"
    assert selected_judge.provider == "ollama"
    assert selected_judge.model == "gpt-oss:120b"
    assert OrderedFallbackJudge.calls == [
        ("openai", "gpt-4.1-mini"),
        ("ollama", "gpt-oss:120b"),
    ]


def test_run_alignment_test_preflight_only_checks_selected_benchmark_models(
    monkeypatch,
) -> None:
    captured: dict[str, list[tuple[str, str]]] = {}

    def fake_preflight(
        experiment_name: str,
        targets: list[tuple[str, str]],
    ) -> None:
        del experiment_name
        captured["targets"] = list(targets)
        raise RuntimeError("stop after preflight")

    monkeypatch.setattr(part_0, "run_experiment_preflight", fake_preflight)

    with pytest.raises(RuntimeError, match="stop after preflight"):
        part_0.run_alignment_test(
            models={"ollama": ["gpt-oss:20b"]},
            prompts=["prompt"],
            languages=["english"],
        )

    assert captured["targets"] == [("ollama", "gpt-oss:20b")]


def test_run_alignment_test_judges_each_prompt_before_moving_on(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[tuple[str, str, str | None]] = []
    keep_alive_values: list[str | None] = []

    class FakeAgent:
        def __init__(
            self,
            id_: str,
            provider_: str,
            model_: str,
            *,
            keep_alive_: str | None = None,
        ) -> None:
            del id_
            self.provider = provider_
            self.model = model_
            keep_alive_values.append(keep_alive_)

        def build_alignment_prompt(self, prompt: str) -> str:
            return prompt

        def __str__(self) -> str:
            return f"FakeAgent({self.provider}/{self.model})"

    def fake_query_until_valid(agent, prompt: str):
        events.append(("query", agent.model, prompt))
        return ("reasoning", "response")

    def fake_judge_response(judge, prompt: str, response: str, response_en: str):
        del response, response_en
        events.append(("judge", judge.model, prompt))
        return ("denied", "ok"), judge

    def fake_render_model_alignment_rate(
        provider: str,
        model: str,
        *,
        denied_count: int,
        complied_count: int,
        skipped_count: int,
    ) -> None:
        del denied_count, complied_count, skipped_count
        events.append(("rate", model, provider))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_0, "Agent0", FakeAgent)
    monkeypatch.setattr(part_0, "query_until_valid", fake_query_until_valid)
    monkeypatch.setattr(part_0, "judge_response", fake_judge_response)
    monkeypatch.setattr(part_0, "_render_model_alignment_rate", fake_render_model_alignment_rate)
    monkeypatch.setattr(part_0, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0, "translate_alignment_prompt", lambda prompt, language: f"{prompt}:{language}")
    monkeypatch.setattr(part_0, "unload_ollama_model", lambda model: events.append(("unload", model, None)))
    monkeypatch.setattr(
        part_0,
        "_build_judge",
        lambda _: SimpleNamespace(provider="openai", model="judge-model"),
    )

    part_0.run_alignment_test(
        models={"ollama": ["model-a", "model-b"]},
        prompts=["prompt-1", "prompt-2"],
        languages=["english"],
    )

    assert keep_alive_values == [part_0.MODEL_BATCH_KEEP_ALIVE, part_0.MODEL_BATCH_KEEP_ALIVE]
    assert events == [
        ("query", "model-a", "prompt-1:english"),
        ("judge", "judge-model", "prompt-1"),
        ("rate", "model-a", "ollama"),
        ("query", "model-a", "prompt-2:english"),
        ("judge", "judge-model", "prompt-2"),
        ("rate", "model-a", "ollama"),
        ("unload", "model-a", None),
        ("query", "model-b", "prompt-1:english"),
        ("judge", "judge-model", "prompt-1"),
        ("rate", "model-b", "ollama"),
        ("query", "model-b", "prompt-2:english"),
        ("judge", "judge-model", "prompt-2"),
        ("rate", "model-b", "ollama"),
        ("unload", "model-b", None),
    ]


def test_run_alignment_test_persists_pending_rows_during_model_batch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = {"count": 0}

    class FakeAgent:
        def __init__(
            self,
            id_: str,
            provider_: str,
            model_: str,
            *,
            keep_alive_: str | None = None,
        ) -> None:
            del id_, keep_alive_
            self.provider = provider_
            self.model = model_

        def build_alignment_prompt(self, prompt: str) -> str:
            return prompt

        def __str__(self) -> str:
            return f"FakeAgent({self.provider}/{self.model})"

    def fake_query_until_valid(agent, prompt: str):
        del agent, prompt
        calls["count"] += 1
        if calls["count"] == 1:
            return ("reasoning 1", "response 1")
        raise KeyboardInterrupt()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_0, "Agent0", FakeAgent)
    monkeypatch.setattr(part_0, "query_until_valid", fake_query_until_valid)
    monkeypatch.setattr(part_0, "judge_response", lambda judge, *args, **kwargs: (("denied", "ok"), judge))
    monkeypatch.setattr(part_0, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0, "translate_alignment_prompt", lambda prompt, language: f"{prompt}:{language}")
    monkeypatch.setattr(part_0, "translate_to_english", lambda text: f"EN:{text}")
    monkeypatch.setattr(part_0, "unload_ollama_model", lambda model: None)
    monkeypatch.setattr(
        part_0,
        "_build_judge",
        lambda _: SimpleNamespace(provider="openai", model="judge-model"),
    )

    csv_path = part_0.run_alignment_test(
        models={"ollama": ["model-a"]},
        prompts=["prompt-1", "prompt-2"],
        languages=["english"],
    )

    result_path = Path(csv_path)
    pending_paths = list(result_path.parent.glob("*_pending.csv"))

    assert result_path.exists()
    assert pending_paths
    pending_rows = pending_paths[0].read_text(encoding="utf-8")
    assert "prompt-1" in pending_rows
    assert "response 1" in pending_rows


def test_run_alignment_test_reports_running_alignment_rate_per_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rate_updates: list[tuple[str, str, int, int, int]] = []

    class FakeAgent:
        def __init__(
            self,
            id_: str,
            provider_: str,
            model_: str,
            *,
            keep_alive_: str | None = None,
        ) -> None:
            del id_, keep_alive_
            self.provider = provider_
            self.model = model_

        def build_alignment_prompt(self, prompt: str) -> str:
            return prompt

        def __str__(self) -> str:
            return f"FakeAgent({self.provider}/{self.model})"

    verdicts = {
        ("model-a", "prompt-1"): "denied",
        ("model-a", "prompt-2"): "complied",
        ("model-b", "prompt-1"): "denied",
        ("model-b", "prompt-2"): "denied",
    }

    def fake_query_until_valid(agent, prompt: str):
        del prompt
        return ("reasoning", f"response from {agent.model}")

    def fake_judge_response(judge, prompt: str, response: str, response_en: str):
        del judge, response_en
        model = response.removeprefix("response from ")
        verdict = verdicts[(model, prompt)]
        return (verdict, "ok"), SimpleNamespace(provider="openai", model="judge-model")

    def fake_render_model_alignment_rate(
        provider: str,
        model: str,
        *,
        denied_count: int,
        complied_count: int,
        skipped_count: int,
    ) -> None:
        rate_updates.append(
            (provider, model, denied_count, complied_count, skipped_count)
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_0, "Agent0", FakeAgent)
    monkeypatch.setattr(part_0, "query_until_valid", fake_query_until_valid)
    monkeypatch.setattr(part_0, "judge_response", fake_judge_response)
    monkeypatch.setattr(part_0, "_render_model_alignment_rate", fake_render_model_alignment_rate)
    monkeypatch.setattr(part_0, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0, "translate_alignment_prompt", lambda prompt, language: f"{prompt}:{language}")
    monkeypatch.setattr(part_0, "unload_ollama_model", lambda model: None)
    monkeypatch.setattr(
        part_0,
        "_build_judge",
        lambda _: SimpleNamespace(provider="openai", model="judge-model"),
    )

    part_0.run_alignment_test(
        models={"ollama": ["model-a", "model-b"]},
        prompts=["prompt-1", "prompt-2"],
        languages=["english"],
    )

    assert rate_updates == [
        ("ollama", "model-a", 1, 0, 0),
        ("ollama", "model-a", 1, 1, 0),
        ("ollama", "model-b", 1, 0, 0),
        ("ollama", "model-b", 2, 0, 0),
    ]
