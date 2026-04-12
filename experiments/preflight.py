print("[EXPERIMENT PREFLIGHT] Hello, World!")

import contextlib
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from rich import box
from rich.console import Console
from rich.panel import Panel

console = Console()
SKIP_PREFLIGHT_ENV_VAR = "LLM_ALTRUISM_SKIP_PREFLIGHT"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _truthy_env(var_name: str) -> bool:
    return os.getenv(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _should_skip_preflight() -> bool:
    return _truthy_env(SKIP_PREFLIGHT_ENV_VAR) or "PYTEST_CURRENT_TEST" in os.environ


def _build_test_command() -> list[str]:
    if shutil.which("uv"):
        return ["uv", "run", "pytest", "-q"]
    return [sys.executable, "-m", "pytest", "-q"]


def _terminate_test_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return

    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return

    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        process.wait()


def _run_test_command(command: list[str], *, env: dict[str, str]) -> int:
    popen_kwargs: dict[str, object] = {}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if creationflags:
            popen_kwargs["creationflags"] = creationflags
    else:
        popen_kwargs["start_new_session"] = True

    process = subprocess.Popen(
        command,
        cwd=_repo_root(),
        env=env,
        **popen_kwargs,
    )

    try:
        return process.wait()
    except KeyboardInterrupt:
        console.print(
            Panel(
                "Interrupted during preflight. Stopping the test subprocess before exiting.",
                title="[bold yellow]Preflight Interrupted[/bold yellow]",
                border_style="yellow",
                expand=True,
            )
        )
        _terminate_test_process(process)
        raise


def _run_tests(experiment_name: str) -> None:
    command = _build_test_command()
    command_label = " ".join(command)

    console.print(
        Panel(
            f"[bold]{experiment_name} Preflight[/bold]\n"
            f"Running test suite before the experiment starts.\n\n"
            f"[cyan]{command_label}[/cyan]",
            title="[bold]Test Gate[/bold]",
            border_style="white",
            box=box.DOUBLE,
            expand=True,
        )
    )

    env = os.environ.copy()
    env[SKIP_PREFLIGHT_ENV_VAR] = "1"
    returncode = _run_test_command(command, env=env)

    if returncode != 0:
        console.print(
            Panel(
                "The test suite failed. The experiment has been aborted.",
                title="[bold red]Preflight Failed[/bold red]",
                border_style="red",
                expand=True,
            )
        )
        raise RuntimeError(
            f"Preflight test suite failed with exit code {returncode}."
        )

    console.print(
        Panel(
            "All tests passed.",
            title="[bold green]Preflight Passed[/bold green]",
            border_style="green",
            expand=True,
        )
    )


def run_experiment_preflight(
    experiment_name: str,
    targets: Iterable[tuple[str, str]],
    *,
    resume: bool = False,
) -> None:
    if _should_skip_preflight():
        return

    del targets
    if resume:
        console.print(
            Panel(
                f"{experiment_name} is resuming from saved artifacts.\n"
                "Skipping the preflight test suite and continuing the interrupted run.",
                title="[bold yellow]Preflight Skipped[/bold yellow]",
                border_style="yellow",
                expand=True,
            )
        )
        return

    _run_tests(experiment_name)
