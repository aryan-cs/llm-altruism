import signal
from pathlib import Path

import pytest
from experiments.misc import preflight


def test_run_experiment_preflight_runs_tests_only(monkeypatch) -> None:
    commands: list[dict] = []

    monkeypatch.setattr(preflight, "_should_skip_preflight", lambda: False)
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(
        preflight.subprocess,
        "Popen",
        lambda *args, **kwargs: commands.append(
            {"args": args, "kwargs": kwargs}
        ) or type(
            "FakeProcess",
            (),
            {
                "wait": lambda self: 0,
            },
        )(),
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
    assert commands[0]["kwargs"]["cwd"] == Path(__file__).resolve().parents[1]
    assert commands[0]["kwargs"]["env"][preflight.SKIP_PREFLIGHT_ENV_VAR] == "1"
    assert commands[0]["kwargs"]["start_new_session"] is True


def test_run_experiment_preflight_uses_requested_test_paths(monkeypatch) -> None:
    commands: list[dict] = []

    monkeypatch.setattr(preflight, "_should_skip_preflight", lambda: False)
    monkeypatch.setattr(preflight.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(
        preflight.subprocess,
        "Popen",
        lambda *args, **kwargs: commands.append(
            {"args": args, "kwargs": kwargs}
        ) or type(
            "FakeProcess",
            (),
            {
                "wait": lambda self: 0,
            },
        )(),
    )

    preflight.run_experiment_preflight(
        "Test Experiment",
        [("openai", "gpt-4.1-mini")],
        test_paths=["tests/test_preflight.py", "tests/test_part_1.py"],
    )

    assert len(commands) == 1
    assert commands[0]["args"][0] == [
        "uv",
        "run",
        "pytest",
        "-q",
        "tests/test_preflight.py",
        "tests/test_part_1.py",
    ]


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


def test_run_experiment_preflight_skips_tests_on_resume(monkeypatch) -> None:
    called = {"tests": 0}

    monkeypatch.setattr(preflight, "_should_skip_preflight", lambda: False)
    monkeypatch.setattr(
        preflight,
        "_run_tests",
        lambda _: called.__setitem__("tests", called["tests"] + 1),
    )

    preflight.run_experiment_preflight(
        "Test Experiment",
        [("ollama", "llama3.1:8b")],
        resume=True,
    )

    assert called == {"tests": 0}


def test_run_test_command_stops_subprocess_on_keyboard_interrupt(monkeypatch) -> None:
    popen_calls: list[dict] = []
    killpg_calls: list[tuple[int, signal.Signals]] = []

    class FakeProcess:
        pid = 4321

        def __init__(self) -> None:
            self.wait_calls: list[int | None] = []

        def wait(self, timeout: int | None = None) -> int:
            self.wait_calls.append(timeout)
            if timeout is None:
                raise KeyboardInterrupt
            return 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            raise AssertionError("terminate should not be used on POSIX")

        def kill(self) -> None:
            raise AssertionError("kill should not be used on POSIX")

    process = FakeProcess()

    monkeypatch.setattr(
        preflight.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append(
            {"args": args, "kwargs": kwargs}
        ) or process,
    )
    monkeypatch.setattr(preflight.os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))

    with pytest.raises(KeyboardInterrupt):
        preflight._run_test_command(["uv", "run", "pytest", "-q"], env={"A": "B"})

    assert popen_calls == [
        {
            "args": (["uv", "run", "pytest", "-q"],),
            "kwargs": {
                "cwd": preflight._repo_root(),
                "env": {"A": "B"},
                "start_new_session": True,
            },
        }
    ]
    assert killpg_calls == [(4321, signal.SIGINT)]
    assert process.wait_calls == [None, 5]
