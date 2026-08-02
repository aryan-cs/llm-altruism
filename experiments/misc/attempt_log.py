from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.misc.run_metadata import utc_now_iso


ATTEMPT_LOG_SCHEMA_VERSION = 2
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_OUTCOMES = {
    "success",
    "invalid_response",
    "provider_error",
    "interrupted",
}


@dataclass(frozen=True)
class AttemptLogSummary:
    path: str
    sha256: str
    total_attempts: int
    successful_attempts: int
    invalid_response_attempts: int
    provider_error_attempts: int
    interrupted_attempts: int
    retry_attempts: int
    retried_units: int

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": ATTEMPT_LOG_SCHEMA_VERSION,
            "path": self.path,
            "sha256": self.sha256,
            "total_attempts": self.total_attempts,
            "successful_attempts": self.successful_attempts,
            "invalid_response_attempts": self.invalid_response_attempts,
            "provider_error_attempts": self.provider_error_attempts,
            "interrupted_attempts": self.interrupted_attempts,
            "retry_attempts": self.retry_attempts,
            "retried_units": self.retried_units,
        }


def attempt_log_path_for_csv(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_attempts.jsonl")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return EMPTY_SHA256
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_attempt_log(path: str | Path) -> AttemptLogSummary:
    log_path = Path(path)
    counts = {outcome: 0 for outcome in _OUTCOMES}
    total = 0
    retry_attempts = 0
    retried_units: set[str] = set()

    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in attempt log {log_path} at line {line_number}."
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Attempt log {log_path} line {line_number} is not an object."
                    )
                outcome = str(record.get("outcome", ""))
                if outcome not in _OUTCOMES:
                    raise ValueError(
                        f"Attempt log {log_path} line {line_number} has unknown outcome {outcome!r}."
                    )
                total += 1
                counts[outcome] += 1
                if bool(record.get("is_retry")):
                    retry_attempts += 1
                    retried_units.add(str(record.get("unit_id", "")))

    return AttemptLogSummary(
        path=str(log_path),
        sha256=_sha256_file(log_path),
        total_attempts=total,
        successful_attempts=counts["success"],
        invalid_response_attempts=counts["invalid_response"],
        provider_error_attempts=counts["provider_error"],
        interrupted_attempts=counts["interrupted"],
        retry_attempts=retry_attempts,
        retried_units=len(retried_units),
    )


class DurableAttemptLogger:
    """Append-only, fsync-backed JSONL writer for individual model attempts."""

    def __init__(self, path: str | Path, *, experiment: str) -> None:
        self.path = Path(path)
        self.experiment = experiment
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Validate any existing sidecar before appending during resume.
        self._sequence = summarize_attempt_log(self.path).total_attempts

    def append(
        self,
        *,
        provider: str,
        model: str,
        unit_id: str,
        unit: dict[str, Any],
        attempt: int,
        max_attempts: int,
        prompt_text: str,
        outcome: str,
        raw_response: str | None = None,
        parsed_response: dict[str, Any] | None = None,
        generation_record: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        will_retry: bool = False,
    ) -> None:
        if outcome not in _OUTCOMES:
            raise ValueError(f"Unsupported attempt outcome: {outcome}")
        if attempt < 1 or max_attempts < attempt:
            raise ValueError("attempt must be in the inclusive range 1..max_attempts")

        self._sequence += 1
        record = {
            "schema_version": ATTEMPT_LOG_SCHEMA_VERSION,
            "sequence": self._sequence,
            "recorded_at_utc": utc_now_iso(),
            "experiment": self.experiment,
            "provider": provider,
            "model": model,
            "unit_id": unit_id,
            "unit": unit,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "is_retry": attempt > 1,
            "will_retry": will_retry,
            "prompt_text": prompt_text,
            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            "outcome": outcome,
            "raw_response": raw_response,
            "parsed_response": parsed_response,
            "generation_record": generation_record,
            "error": error,
        }
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self) -> AttemptLogSummary:
        return summarize_attempt_log(self.path)
