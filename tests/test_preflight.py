from types import SimpleNamespace

from experiments import preflight
from providers.api_call import OllamaConnectionError


def test_run_experiment_preflight_runs_tests_and_resolves_ollama(monkeypatch) -> None:
    commands: list[dict] = []
    models: list[str] = []

    monkeypatch.setattr(preflight, "_should_skip_preflight", lambda: False)
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: commands.append(
            {"args": args, "kwargs": kwargs}
        ) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        preflight,
        "ensure_ollama_model_available",
        lambda model, progress_callback=None: models.append(model),
    )

    preflight.run_experiment_preflight(
        "Test Experiment",
        [
            ("openai", "gpt-4.1-mini"),
            ("ollama", "llama3.1:8b"),
            ("ollama", "llama3.1:8b"),
            ("ollama", "gpt-oss:20b"),
        ],
    )

    assert len(commands) == 1
    assert commands[0]["args"][0] == ["uv", "run", "pytest", "-q"]
    assert commands[0]["kwargs"]["env"][preflight.SKIP_PREFLIGHT_ENV_VAR] == "1"
    assert models == ["llama3.1:8b", "gpt-oss:20b"]


def test_run_experiment_preflight_skips_when_requested(monkeypatch) -> None:
    called = {"tests": 0, "ollama": 0}

    monkeypatch.setattr(preflight, "_should_skip_preflight", lambda: True)
    monkeypatch.setattr(
        preflight,
        "_run_tests",
        lambda _: called.__setitem__("tests", called["tests"] + 1),
    )
    monkeypatch.setattr(
        preflight,
        "_ensure_ollama_models",
        lambda _: called.__setitem__("ollama", called["ollama"] + 1),
    )

    preflight.run_experiment_preflight(
        "Test Experiment",
        [("ollama", "llama3.1:8b")],
    )

    assert called == {"tests": 0, "ollama": 0}


def test_run_experiment_preflight_exits_cleanly_when_ollama_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(preflight, "_should_skip_preflight", lambda: False)
    monkeypatch.setattr(preflight, "_run_tests", lambda _: None)
    monkeypatch.setattr(
        preflight,
        "ensure_ollama_model_available",
        lambda _, progress_callback=None: (_ for _ in ()).throw(OllamaConnectionError("down")),
    )

    try:
        preflight.run_experiment_preflight(
            "Test Experiment",
            [("ollama", "llama3.1:8b")],
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected SystemExit when Ollama is unavailable.")
