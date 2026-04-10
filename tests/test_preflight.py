from types import SimpleNamespace

from experiments import preflight


def test_run_experiment_preflight_runs_tests_only(monkeypatch) -> None:
    commands: list[dict] = []

    monkeypatch.setattr(preflight, "_should_skip_preflight", lambda: False)
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: commands.append(
            {"args": args, "kwargs": kwargs}
        ) or SimpleNamespace(returncode=0),
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


def test_run_experiment_preflight_skips_when_requested(monkeypatch) -> None:
    called = {"tests": 0}

    monkeypatch.setattr(preflight, "_should_skip_preflight", lambda: True)
    monkeypatch.setattr(
        preflight,
        "_run_tests",
        lambda _: called.__setitem__("tests", called["tests"] + 1),
    )

    preflight.run_experiment_preflight(
        "Test Experiment",
        [("ollama", "llama3.1:8b")],
    )

    assert called == {"tests": 0}
