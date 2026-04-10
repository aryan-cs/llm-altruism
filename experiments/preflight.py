print("[EXPERIMENT PREFLIGHT] Hello, World!")

import os
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
    completed = subprocess.run(
        command,
        cwd=_repo_root(),
        env=env,
        check=False,
    )

    if completed.returncode != 0:
        console.print(
            Panel(
                "The test suite failed. The experiment has been aborted.",
                title="[bold red]Preflight Failed[/bold red]",
                border_style="red",
                expand=True,
            )
        )
        raise RuntimeError(
            f"Preflight test suite failed with exit code {completed.returncode}."
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
) -> None:
    if _should_skip_preflight():
        return

    del targets
    _run_tests(experiment_name)
