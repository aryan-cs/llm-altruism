from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from agents.agent_config import registry_metadata_for_targets
from providers.api_call import failure_provenance


STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|apikey|authorization|bearer|credential|password|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def _safe_url_snapshot(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "<INVALID_URL>"
    if not parsed.scheme or not hostname:
        return "<INVALID_URL>"
    host = hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def safe_environment_snapshot() -> dict[str, str | bool]:
    keys = [
        "DEFAULT_PROVIDER",
        "DEFAULT_MODEL",
        "PART_1_PROVIDER",
        "PART_1_MODEL",
        "PART_2_PROVIDER",
        "PART_2_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_NUM_PREDICT",
        "OLLAMA_ADMIN_TIMEOUT_SECONDS",
        "OLLAMA_GENERATION_TIMEOUT_SECONDS",
        "OPENAI_BASE_URL",
        "OPENAI_COMPATIBLE_BASE_URL",
        "INFERENCE_HUB_BASE_URL",
        "LLM_ALTRUISM_SKIP_PREFLIGHT",
        "LLM_ALTRUISM_STRICT_PROVIDER_PREFLIGHT",
    ]
    snapshot: dict[str, str | bool] = {}
    for key in keys:
        value = os.getenv(key, "")
        if value == "":
            continue
        snapshot[key] = _safe_url_snapshot(value) if key.endswith("BASE_URL") else value
    return snapshot


def _repo_relative_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except Exception:
        if path.is_absolute():
            return "<ABSOLUTE_PATH>"
        return str(value)


def _command_snapshot() -> list[str]:
    snapshot: list[str] = []
    redact_next = False
    for arg in sys.argv:
        if redact_next:
            snapshot.append("<REDACTED>")
            redact_next = False
            continue
        if "=" in arg:
            option, value = arg.split("=", 1)
            if _SENSITIVE_KEY_PATTERN.search(option):
                snapshot.append(f"{option}=<REDACTED>")
                continue
            del value
        snapshot.append(_repo_relative_path(arg))
        if arg.startswith("-") and _SENSITIVE_KEY_PATTERN.search(arg):
            redact_next = True
    return snapshot


def _redact_sensitive(value: Any, *, key: str = "") -> Any:
    if key and _SENSITIVE_KEY_PATTERN.search(key):
        return "<REDACTED>"
    if isinstance(value, dict):
        return {
            str(child_key): _redact_sensitive(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_sensitive(item) for item in value]
    return value


def safe_error_message(error: BaseException) -> str:
    message = str(error)
    message = re.sub(
        r"https?://[^\s]+",
        lambda match: _safe_url_snapshot(match.group(0)),
        message,
    )
    message = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)\S+",
        r"\1<REDACTED>",
        message,
    )
    message = re.sub(
        r"(?i)((?:api[_-]?key|password|secret|token)\s*[:=]\s*)\S+",
        r"\1<REDACTED>",
        message,
    )
    message = re.sub(r"\bsk-[A-Za-z0-9_-]+", "<REDACTED>", message)
    return message


# Backwards-compatible private alias for callers/tests that imported the old helper.
_safe_error_message = safe_error_message


def base_run_metadata(
    *,
    experiment: str,
    timestamp: str,
    csv_path: str | Path,
    provider: str,
    model: str,
    parameters: dict[str, Any] | None = None,
    targets: list[tuple[str, str]] | None = None,
    generation_config: dict[str, Any] | None = None,
    prompt_config_hash: str | None = None,
    status: str = STATUS_RUNNING,
) -> dict[str, Any]:
    target_list = targets or [(provider, model)]
    registry_metadata = registry_metadata_for_targets(target_list)
    resolved_parameters = _redact_sensitive(parameters or {})
    resolved_generation_config = generation_config
    if resolved_generation_config is None:
        candidate = resolved_parameters.get("generation_config")
        resolved_generation_config = candidate if isinstance(candidate, dict) else {}
    resolved_generation_config = _redact_sensitive(resolved_generation_config)
    credential_env_names = sorted(
        {
            str(target["endpoint"]["credential_env"])
            for target in registry_metadata["targets"]
            if isinstance(target.get("endpoint"), dict)
            and target["endpoint"].get("credential_env")
        }
    )
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "experiment": experiment,
        "status": status,
        "timestamp": timestamp,
        "created_at_utc": utc_now_iso(),
        "csv_path": _repo_relative_path(csv_path),
        "provider": provider,
        "model": model,
        "parameters": resolved_parameters,
        "generation_config": resolved_generation_config,
        "model_registry": registry_metadata,
        "cohort": (
            registry_metadata["cohorts"][0]
            if len(registry_metadata["cohorts"]) == 1
            else None
        ),
        "route": registry_metadata["targets"][0],
        "prompt_config_hash": prompt_config_hash,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "command": _command_snapshot(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "environment": safe_environment_snapshot(),
        "credential_environment": {
            key: bool(os.getenv(key, "").strip())
            for key in credential_env_names
        },
    }
    return metadata


def read_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = metadata_path.with_name(f".{metadata_path.name}.tmp")
    encoded = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    with temporary_path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, metadata_path)
    try:
        directory_fd = os.open(metadata_path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def metadata_is_complete(metadata: dict[str, Any]) -> bool:
    return str(metadata.get("status", "")).strip().lower() == STATUS_COMPLETE


def mark_metadata_complete(
    path: str | Path,
    *,
    completed_rows: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    metadata_path = Path(path)
    metadata = read_metadata(metadata_path)
    metadata["status"] = STATUS_COMPLETE
    metadata["completed_at_utc"] = utc_now_iso()
    if completed_rows is not None:
        metadata["completed_rows"] = completed_rows
    if extra:
        metadata.update(_redact_sensitive(extra))
    write_metadata(metadata_path, metadata)


def mark_metadata_failed(
    path: str | Path,
    *,
    error: Exception,
    provider: str,
    model: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist a typed failure while retaining the original exception semantics."""

    metadata_path = Path(path)
    metadata = read_metadata(metadata_path)
    metadata["status"] = STATUS_FAILED
    metadata["failed_at_utc"] = utc_now_iso()
    metadata["failure"] = {
        "exception_type": type(error).__name__,
        "message": safe_error_message(error),
        "provenance": failure_provenance(
            error,
            provider=provider,
            model=model,
        ),
    }
    if extra:
        metadata.update(_redact_sensitive(extra))
    write_metadata(metadata_path, metadata)
