import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

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


def _write_csv_rows(
    path: Path,
    header: list[str],
    rows: list[dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _render_to_text(renderable) -> str:
    capture_console = Console(force_terminal=False, color_system=None, width=100, record=True)
    capture_console.print(renderable)
    return capture_console.export_text()


def test_translate_to_english_reports_failure_status(monkeypatch) -> None:
    class EmptyTranslator:
        def __init__(self, source: str, target: str) -> None:
            del source, target

        def translate(self, text: str) -> str:
            del text
            return ""

    monkeypatch.setattr(part_0, "GoogleTranslator", EmptyTranslator)

    translated, failed = part_0.translate_to_english("hola", return_status=True)

    assert translated == ""
    assert failed is True


def test_build_localized_renderable_keeps_original_text_when_translation_fails() -> None:
    rendered = _render_to_text(
        part_0._build_localized_renderable(
            "respuesta original",
            "",
            language="spanish",
            translation_failed=True,
        )
    )

    assert "respuesta original" in rendered
    assert "[EN] FAIL TO TRANSLATE" in rendered


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


def test_run_alignment_test_can_defer_judging_until_after_collection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

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
        events.append(f"query:{agent.model}:{prompt}")
        return ("reasoning", f"response from {agent.model}:{prompt}")

    def fake_judge_response(judge, prompt: str, response: str, response_en: str):
        del response_en
        events.append(f"judge:{prompt}:{response}")
        return (("denied", "ok"), judge)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_0, "Agent0", FakeAgent)
    monkeypatch.setattr(part_0, "query_until_valid", fake_query_until_valid)
    monkeypatch.setattr(part_0, "judge_response", fake_judge_response)
    monkeypatch.setattr(part_0, "_render_model_alignment_rate", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0, "translate_alignment_prompt", lambda prompt, language: f"{prompt}:{language}")
    monkeypatch.setattr(part_0, "unload_ollama_model", lambda model: events.append(f"unload:{model}"))
    monkeypatch.setattr(
        part_0,
        "_build_judge",
        lambda _: SimpleNamespace(provider="openai", model="judge-model"),
    )

    csv_path = part_0.run_alignment_test(
        models={"ollama": ["model-a", "model-b"]},
        prompts=["prompt-1", "prompt-2"],
        languages=["english"],
        judge_after=True,
    )

    assert events == [
        "query:model-a:prompt-1:english",
        "query:model-a:prompt-2:english",
        "unload:model-a",
        "query:model-b:prompt-1:english",
        "query:model-b:prompt-2:english",
        "unload:model-b",
        "judge:prompt-1:response from model-a:prompt-1:english",
        "judge:prompt-2:response from model-a:prompt-2:english",
        "judge:prompt-1:response from model-b:prompt-1:english",
        "judge:prompt-2:response from model-b:prompt-2:english",
    ]

    metadata_path = Path(csv_path).with_name(f"{Path(csv_path).stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["judge_after"] is True


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


def test_run_alignment_test_resume_uses_latest_interrupted_run_and_pending_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    query_calls: list[tuple[str, str]] = []
    judge_calls: list[tuple[str, str]] = []

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
        query_calls.append((agent.model, prompt))
        return ("fresh reasoning 3", "fresh response 3")

    def fake_judge_response(judge, prompt: str, response: str, response_en: str):
        del response_en
        judge_calls.append((prompt, response))
        return (("denied", "ok"), judge)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_0, "Agent0", FakeAgent)
    monkeypatch.setattr(part_0, "query_until_valid", fake_query_until_valid)
    monkeypatch.setattr(part_0, "judge_response", fake_judge_response)
    monkeypatch.setattr(part_0, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0, "translate_alignment_prompt", lambda prompt, language: f"{prompt}:{language}")
    monkeypatch.setattr(part_0, "unload_ollama_model", lambda model: None)
    monkeypatch.setattr(part_0, "_render_model_alignment_rate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        part_0,
        "_build_judge",
        lambda _: SimpleNamespace(provider="openai", model="judge-model"),
    )

    results_dir = tmp_path / "results" / "alignment"
    older_timestamp = "04-09-2026_12:00:00"
    newer_timestamp = "04-10-2026_23:29:07"
    older_csv = results_dir / f"{older_timestamp}.csv"
    older_pending = results_dir / f"{older_timestamp}_pending.csv"
    older_meta = results_dir / f"{older_timestamp}_meta.json"
    newer_csv = results_dir / f"{newer_timestamp}.csv"
    newer_pending = results_dir / f"{newer_timestamp}_pending.csv"
    newer_meta = results_dir / f"{newer_timestamp}_meta.json"

    _write_csv_rows(
        older_csv,
        part_0.RESULT_HEADERS,
        [],
    )
    _write_csv_rows(
        older_pending,
        part_0.PENDING_RESULT_HEADERS,
        [
            {
                "provider": "ollama",
                "model": "old-model",
                "language": "english",
                "prompt": "old-prompt",
                "reasoning": "old reasoning",
                "response": "old response",
                "reasoning_en": "",
                "response_en": "",
            }
        ],
    )
    older_meta.write_text(
        json.dumps(
            {
                "models": {"ollama": ["old-model"]},
                "prompts": ["old-prompt"],
                "languages": ["english"],
            }
        ),
        encoding="utf-8",
    )

    _write_csv_rows(
        newer_csv,
        part_0.RESULT_HEADERS,
        [
            {
                "provider": "ollama",
                "model": "model-a",
                "language": "english",
                "prompt": "prompt-1",
                "reasoning": "done reasoning 1",
                "response": "done response 1",
                "reasoning_en": "",
                "response_en": "",
                "verdict": "denied",
                "verdict_reason": "done",
            }
        ],
    )
    _write_csv_rows(
        newer_pending,
        part_0.PENDING_RESULT_HEADERS,
        [
            {
                "provider": "ollama",
                "model": "model-a",
                "language": "english",
                "prompt": "prompt-2",
                "reasoning": "saved reasoning 2",
                "response": "saved response 2",
                "reasoning_en": "",
                "response_en": "",
            }
        ],
    )
    newer_meta.write_text(
        json.dumps(
            {
                "models": {"ollama": ["model-a"]},
                "prompts": ["prompt-1", "prompt-2", "prompt-3"],
                "languages": ["english"],
            }
        ),
        encoding="utf-8",
    )

    csv_path = part_0.run_alignment_test(resume=True)

    assert Path(csv_path).resolve() == newer_csv
    assert query_calls == [("model-a", "prompt-3:english")]
    assert judge_calls == [
        ("prompt-2", "saved response 2"),
        ("prompt-3", "fresh response 3"),
    ]

    with newer_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["prompt"] for row in rows] == ["prompt-1", "prompt-2", "prompt-3"]
    assert not newer_pending.exists()


def test_run_alignment_test_resume_can_seed_legacy_run_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    query_calls: list[str] = []
    judge_calls: list[str] = []

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
        del agent
        query_calls.append(prompt)
        return ("fresh reasoning 2", "fresh response 2")

    def fake_judge_response(judge, prompt: str, response: str, response_en: str):
        del response_en
        judge_calls.append(f"{prompt}:{response}")
        return (("denied", "ok"), judge)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_0, "Agent0", FakeAgent)
    monkeypatch.setattr(part_0, "query_until_valid", fake_query_until_valid)
    monkeypatch.setattr(part_0, "judge_response", fake_judge_response)
    monkeypatch.setattr(part_0, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0, "translate_alignment_prompt", lambda prompt, language: f"{prompt}:{language}")
    monkeypatch.setattr(part_0, "unload_ollama_model", lambda model: None)
    monkeypatch.setattr(part_0, "_render_model_alignment_rate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        part_0,
        "_build_judge",
        lambda _: SimpleNamespace(provider="openai", model="judge-model"),
    )

    timestamp = "04-10-2026_23:29:07"
    results_dir = tmp_path / "results" / "alignment"
    csv_path = results_dir / f"{timestamp}.csv"
    pending_path = results_dir / f"{timestamp}_pending.csv"
    metadata_path = results_dir / f"{timestamp}_meta.json"

    _write_csv_rows(
        csv_path,
        part_0.RESULT_HEADERS,
        [
            {
                "provider": "ollama",
                "model": "model-a",
                "language": "english",
                "prompt": "prompt-1",
                "reasoning": "done reasoning 1",
                "response": "done response 1",
                "reasoning_en": "",
                "response_en": "",
                "verdict": "denied",
                "verdict_reason": "done",
            }
        ],
    )
    _write_csv_rows(
        pending_path,
        part_0.PENDING_RESULT_HEADERS,
        [],
    )

    returned_path = part_0.run_alignment_test(
        models={"ollama": ["model-a"]},
        prompts=["prompt-1", "prompt-2"],
        languages=["english"],
        resume=True,
    )

    assert Path(returned_path).resolve() == csv_path
    assert query_calls == ["prompt-2:english"]
    assert judge_calls == ["prompt-2:fresh response 2"]
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {
        "models": {"ollama": ["model-a"]},
        "prompts": ["prompt-1", "prompt-2"],
        "languages": ["english"],
    }


def test_run_alignment_test_resume_preserves_judge_after_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

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
        events.append(f"query:{agent.model}:{prompt}")
        return ("fresh reasoning 3", "fresh response 3")

    def fake_judge_response(judge, prompt: str, response: str, response_en: str):
        del judge, response_en
        events.append(f"judge:{prompt}:{response}")
        return (("denied", "ok"), SimpleNamespace(provider="openai", model="judge-model"))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(part_0, "Agent0", FakeAgent)
    monkeypatch.setattr(part_0, "query_until_valid", fake_query_until_valid)
    monkeypatch.setattr(part_0, "judge_response", fake_judge_response)
    monkeypatch.setattr(part_0, "run_experiment_preflight", lambda *args, **kwargs: None)
    monkeypatch.setattr(part_0, "translate_alignment_prompt", lambda prompt, language: f"{prompt}:{language}")
    monkeypatch.setattr(part_0, "unload_ollama_model", lambda model: None)
    monkeypatch.setattr(part_0, "_render_model_alignment_rate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        part_0,
        "_build_judge",
        lambda _: SimpleNamespace(provider="openai", model="judge-model"),
    )

    timestamp = "04-10-2026_23:29:07"
    results_dir = tmp_path / "results" / "alignment"
    csv_path = results_dir / f"{timestamp}.csv"
    pending_path = results_dir / f"{timestamp}_pending.csv"
    metadata_path = results_dir / f"{timestamp}_meta.json"

    _write_csv_rows(
        csv_path,
        part_0.RESULT_HEADERS,
        [
            {
                "provider": "ollama",
                "model": "model-a",
                "language": "english",
                "prompt": "prompt-1",
                "reasoning": "done reasoning 1",
                "response": "done response 1",
                "reasoning_en": "",
                "response_en": "",
                "verdict": "denied",
                "verdict_reason": "done",
            }
        ],
    )
    _write_csv_rows(
        pending_path,
        part_0.PENDING_RESULT_HEADERS,
        [
            {
                "provider": "ollama",
                "model": "model-a",
                "language": "english",
                "prompt": "prompt-2",
                "reasoning": "saved reasoning 2",
                "response": "saved response 2",
                "reasoning_en": "",
                "response_en": "",
            }
        ],
    )
    metadata_path.write_text(
        json.dumps(
            {
                "models": {"ollama": ["model-a"]},
                "prompts": ["prompt-1", "prompt-2", "prompt-3"],
                "languages": ["english"],
                "judge_after": True,
            }
        ),
        encoding="utf-8",
    )

    returned_path = part_0.run_alignment_test(resume=True)

    assert Path(returned_path).resolve() == csv_path
    assert events == [
        "query:model-a:prompt-3:english",
        "judge:prompt-2:saved response 2",
        "judge:prompt-3:fresh response 3",
    ]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["judge_after"] is True
