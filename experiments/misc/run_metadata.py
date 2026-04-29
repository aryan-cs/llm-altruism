from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"


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
        "OLLAMA_TIMEOUT_SECONDS",
        "LLM_ALTRUISM_SKIP_PREFLIGHT",
    ]
    return {
        key: os.getenv(key, "")
        for key in keys
        if os.getenv(key, "") != ""
    }


def _repo_relative_path(value: str | Path) -> str:
    path = Path(value)
    try:
        return str(path.resolve().relative_to(_repo_root()))
    except Exception:
        if path.is_absolute():
            return "<ABSOLUTE_PATH>"
        return str(value)


def _command_snapshot() -> list[str]:
    return [_repo_relative_path(arg) for arg in sys.argv]


def base_run_metadata(
    *,
    experiment: str,
    timestamp: str,
    csv_path: str | Path,
    provider: str,
    model: str,
    parameters: dict[str, Any] | None = None,
    prompt_config_hash: str | None = None,
    status: str = STATUS_RUNNING,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment": experiment,
        "status": status,
        "timestamp": timestamp,
        "created_at_utc": utc_now_iso(),
        "csv_path": _repo_relative_path(csv_path),
        "provider": provider,
        "model": model,
        "parameters": parameters or {},
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
    }
    return metadata


def read_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    metadata_path = Path(path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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
        metadata.update(extra)
    write_metadata(metadata_path, metadata)
