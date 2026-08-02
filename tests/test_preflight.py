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


def test_validate_provider_targets_resolves_inference_hub_registry_route(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INFERENCE_HUB_BASE_URL", "https://hub.example.test/v1")
    metadata = preflight.validate_provider_targets(
        [("inference-hub", "claude-opus-5")]
    )

    assert metadata["registry_version"] == "2026-08-01.1"
    assert metadata["targets"][0]["provider"] == "inference_hub"
    assert metadata["targets"][0]["upstream_provider"] == "anthropic"
    assert metadata["targets"][0]["route"] == "claude-opus-5"


def test_validate_provider_targets_rejects_invalid_compatible_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPATIBLE_BASE_URL", "not-a-url")
    with pytest.raises(ValueError, match="absolute HTTP"):
        preflight.validate_provider_targets(
            [("openai-compatible", "local-model")]
        )


def test_validate_provider_targets_strict_mode_checks_credentials(monkeypatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="NVIDIA_API_KEY"):
        preflight.validate_provider_targets(
            [("inference_hub", "gpt-5.6-sol")],
            require_credentials=True,
        )


def test_validate_provider_targets_rejects_unpinned_inference_hub_route() -> None:
    with pytest.raises(ValueError, match="must be pinned"):
        preflight.validate_provider_targets(
            [("inference_hub", "typo-model")]
        )


def test_validate_provider_targets_accepts_explicit_compatible_connection(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    metadata = preflight.validate_provider_targets(
        [("openai-compatible", "custom-model")],
        require_credentials=True,
        connection_overrides={
            "openai_compatible": {
                "base_url": "http://127.0.0.1:8000/v1",
                "api_key": "explicit-key",
            }
        },
    )

    assert metadata["targets"][0]["registered"] is False
    assert metadata["targets"][0]["route"] == "custom-model"
