# print("[EXPERIMENT PREFLIGHT] Hello, World!")

import contextlib
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from dotenv import load_dotenv

from agents.agent_config import load_endpoint_profile, registry_metadata_for_targets
from rich import box
from rich.console import Console
from rich.panel import Panel

console = Console()
load_dotenv()
SKIP_PREFLIGHT_ENV_VAR = "LLM_ALTRUISM_SKIP_PREFLIGHT"
STRICT_PROVIDER_PREFLIGHT_ENV_VAR = "LLM_ALTRUISM_STRICT_PROVIDER_PREFLIGHT"
SUPPORTED_PROVIDERS = {
    "anthropic",
    "cerebras",
    "groq",
    "inference_hub",
    "nvidia",
    "ollama",
    "openai",
    "openai_compatible",
    "openrouter",
    "xai",
}
PROVIDER_ALIASES = {
    "cerebris": "cerebras",
    "inference-hub": "inference_hub",
    "inferencehub": "inference_hub",
    "olama": "ollama",
    "openai-compatible": "openai_compatible",
    "openaicompatible": "openai_compatible",
    "x.ai": "xai",
}
PROVIDER_CREDENTIAL_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "groq": "GROQ_API_KEY",
    "inference_hub": "NVIDIA_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _truthy_env(var_name: str) -> bool:
    return os.getenv(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _should_skip_preflight() -> bool:
    return _truthy_env(SKIP_PREFLIGHT_ENV_VAR) or "PYTEST_CURRENT_TEST" in os.environ


def _normalized_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


def _validate_http_url(value: str, *, env_name: str) -> None:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{env_name} must be an absolute HTTP(S) URL.")


def validate_provider_targets(
    targets: Iterable[tuple[str, str]],
    *,
    require_credentials: bool = False,
    connection_overrides: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    """Validate provider/model routing before a benchmark spends any requests."""

    normalized_targets: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for provider, model in targets:
        provider_key = _normalized_provider(provider)
        model_id = model.strip()
        if provider_key not in SUPPORTED_PROVIDERS:
            supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ValueError(
                f"Unsupported provider '{provider}'. Supported providers: {supported}."
            )
        if not model_id:
            raise ValueError(f"Model for provider '{provider_key}' must not be empty.")
        target = (provider_key, model_id)
        if target not in seen:
            seen.add(target)
            normalized_targets.append(target)

    if not normalized_targets:
        raise ValueError("Preflight requires at least one provider/model target.")

    target_providers = {provider for provider, _ in normalized_targets}
    overrides = connection_overrides or {}
    for provider in target_providers:
        try:
            profile = load_endpoint_profile(provider)
        except KeyError:
            continue
        env_name = str(profile["base_url_env"])
        override = overrides.get(provider, {})
        value = (
            override.get("base_url", "").strip()
            or os.getenv(env_name, "").strip()
            or str(profile.get("default_base_url", "")).strip()
        )
        if value:
            _validate_http_url(value, env_name=env_name)

    if any(provider == "openai_compatible" for provider, _ in normalized_targets):
        compatible_override = overrides.get("openai_compatible", {})
        if not (
            compatible_override.get("base_url", "").strip()
            or os.getenv("OPENAI_COMPATIBLE_BASE_URL", "").strip()
        ):
            raise EnvironmentError(
                "Missing required environment variable: OPENAI_COMPATIBLE_BASE_URL"
            )

    if require_credentials:
        missing = sorted(
            {
                env_name
                for provider, _ in normalized_targets
                if (env_name := PROVIDER_CREDENTIAL_ENV.get(provider)) is not None
                and not overrides.get(provider, {}).get("api_key", "").strip()
                and not os.getenv(env_name, "").strip()
            }
        )
        if missing:
            raise EnvironmentError(
                "Missing required provider credentials: " + ", ".join(missing)
            )

    registry_metadata = registry_metadata_for_targets(normalized_targets)
    unknown_pinned_routes = [
        str(target["model"])
        for target in registry_metadata["targets"]
        if target.get("provider") == "inference_hub"
        and target.get("registered") is False
    ]
    if unknown_pinned_routes:
        raise ValueError(
            "Inference Hub targets must be pinned in the model registry: "
            + ", ".join(unknown_pinned_routes)
        )
    return registry_metadata


def _build_test_command(test_paths: Iterable[str] | None = None) -> list[str]:
    paths = list(test_paths or [])
    if shutil.which("uv"):
        command = ["uv", "run", "pytest", "-q"]
    else:
        command = [sys.executable, "-m", "pytest", "-q"]
    return [*command, *paths]


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


def _run_tests(
    experiment_name: str,
    *,
    test_paths: Iterable[str] | None = None,
) -> None:
    command = _build_test_command(test_paths)
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
    test_paths: Iterable[str] | None = None,
    strict_provider_checks: bool | None = None,
    connection_overrides: dict[str, dict[str, str]] | None = None,
) -> None:
    if _should_skip_preflight():
        return

    strict_checks = (
        _truthy_env(STRICT_PROVIDER_PREFLIGHT_ENV_VAR)
        if strict_provider_checks is None
        else strict_provider_checks
    )
    validate_provider_targets(
        targets,
        require_credentials=strict_checks and not resume,
        connection_overrides=connection_overrides,
    )

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

    _run_tests(experiment_name, test_paths=test_paths)
