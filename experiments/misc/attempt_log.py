from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from experiments.misc.run_metadata import utc_now_iso


ATTEMPT_LOG_SCHEMA_VERSION = 3
LEGACY_ATTEMPT_LOG_SCHEMA_VERSION = 2
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
    schema_versions: tuple[int, ...]
    hash_chain_status: str
    last_record_sha256: str | None

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
            "schema_versions": list(self.schema_versions),
            "hash_chain_status": self.hash_chain_status,
            "last_record_sha256": self.last_record_sha256,
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


def _record_sha256(record: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in record.items()
        if key != "record_sha256"
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def iter_attempt_records(path: str | Path) -> Iterator[dict[str, Any]]:
    log_path = Path(path)
    if not log_path.exists():
        return
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
            yield record


def load_attempt_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_attempt_records(path))


def summarize_attempt_log(path: str | Path) -> AttemptLogSummary:
    log_path = Path(path)
    counts = {outcome: 0 for outcome in _OUTCOMES}
    retry_attempts = 0
    retried_units: set[str] = set()
    schema_versions: set[int] = set()
    previous_record_sha256: str | None = None
    pending_retry: tuple[str, int] | None = None
    total = 0

    for line_number, record in enumerate(iter_attempt_records(log_path), start=1):
        total = line_number
        schema_version = record.get("schema_version")
        if schema_version not in {
            LEGACY_ATTEMPT_LOG_SCHEMA_VERSION,
            ATTEMPT_LOG_SCHEMA_VERSION,
        }:
            raise ValueError(
                f"Attempt log {log_path} line {line_number} has unsupported schema "
                f"version {schema_version!r}."
            )
        schema_versions.add(int(schema_version))
        if len(schema_versions) > 1:
            raise ValueError(
                f"Attempt log {log_path} mixes incompatible schema versions."
            )
        if record.get("sequence") != line_number:
            raise ValueError(
                f"Attempt log {log_path} line {line_number} has a non-contiguous sequence."
            )
        outcome = str(record.get("outcome", ""))
        if outcome not in _OUTCOMES:
            raise ValueError(
                f"Attempt log {log_path} line {line_number} has unknown outcome {outcome!r}."
            )
        for field in ("experiment", "provider", "model", "unit_id", "prompt_text"):
            if not isinstance(record.get(field), str) or not str(record[field]).strip():
                raise ValueError(
                    f"Attempt log {log_path} line {line_number} has invalid {field}."
                )
        if not isinstance(record.get("unit"), dict):
            raise ValueError(
                f"Attempt log {log_path} line {line_number} has invalid unit metadata."
            )
        attempt = record.get("attempt")
        max_attempts = record.get("max_attempts")
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or attempt < 1
            or max_attempts < attempt
        ):
            raise ValueError(
                f"Attempt log {log_path} line {line_number} has invalid attempt bounds."
            )
        if record.get("is_retry") is not (attempt > 1):
            raise ValueError(
                f"Attempt log {log_path} line {line_number} has inconsistent retry state."
            )
        if not isinstance(record.get("will_retry"), bool):
            raise ValueError(
                f"Attempt log {log_path} line {line_number} has invalid will_retry."
            )
        expected_prompt_hash = hashlib.sha256(
            str(record["prompt_text"]).encode("utf-8")
        ).hexdigest()
        if record.get("prompt_sha256") != expected_prompt_hash:
            raise ValueError(
                f"Attempt log {log_path} line {line_number} prompt hash mismatch."
            )
        if pending_retry is not None:
            expected_unit_id, expected_attempt = pending_retry
            if record["unit_id"] != expected_unit_id or attempt != expected_attempt:
                raise ValueError(
                    f"Attempt log {log_path} line {line_number} breaks retry continuity."
                )
            pending_retry = None
        if record["will_retry"]:
            if attempt >= max_attempts:
                raise ValueError(
                    f"Attempt log {log_path} line {line_number} retries past its bound."
                )
            pending_retry = (str(record["unit_id"]), attempt + 1)

        if schema_version == ATTEMPT_LOG_SCHEMA_VERSION:
            if record.get("previous_record_sha256") != previous_record_sha256:
                raise ValueError(
                    f"Attempt log {log_path} line {line_number} hash-chain predecessor mismatch."
                )
            expected_record_hash = _record_sha256(record)
            if record.get("record_sha256") != expected_record_hash:
                raise ValueError(
                    f"Attempt log {log_path} line {line_number} record hash mismatch."
                )
            previous_record_sha256 = expected_record_hash

        counts[outcome] += 1
        if record["is_retry"]:
            retry_attempts += 1
            retried_units.add(str(record["unit_id"]))

    if pending_retry is not None:
        raise ValueError(
            f"Attempt log {log_path} ends with an incomplete retry transition."
        )

    schema_tuple = tuple(sorted(schema_versions))
    hash_chain_status = (
        "empty"
        if total == 0
        else "verified"
        if schema_tuple == (ATTEMPT_LOG_SCHEMA_VERSION,)
        else "legacy_unavailable"
    )

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
        schema_versions=schema_tuple,
        hash_chain_status=hash_chain_status,
        last_record_sha256=previous_record_sha256,
    )


def verify_attempt_log_metadata(
    path: str | Path,
    expected: dict[str, Any],
    *,
    require_hash_chain: bool,
) -> AttemptLogSummary:
    summary = summarize_attempt_log(path)
    actual = summary.to_metadata()
    for key, expected_value in expected.items():
        if key == "coverage":
            continue
        if key == "path":
            matches = Path(str(actual[key])).resolve() == Path(
                str(expected_value)
            ).resolve()
        else:
            matches = key in actual and actual[key] == expected_value
        if not matches:
            raise ValueError(
                f"Attempt log metadata mismatch for {key}: expected "
                f"{expected_value!r}, found {actual.get(key)!r}."
            )
    if require_hash_chain and summary.hash_chain_status not in {"empty", "verified"}:
        raise ValueError("Strict resume requires a hash-chained attempt log.")
    return summary


def validate_terminal_attempt_coverage(
    path: str | Path,
    completed_unit_ids: set[str],
) -> None:
    """Fail closed when semantic attempts and durable result rows diverge."""

    terminal_semantic: list[str] = []
    for record in iter_attempt_records(path):
        if record.get("will_retry") is False and record.get("outcome") in {
            "success",
            "invalid_response",
        }:
            terminal_semantic.append(str(record["unit_id"]))
    terminal_counts = Counter(terminal_semantic)
    duplicates = sorted(
        unit_id for unit_id, count in terminal_counts.items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Attempt log contains repeated terminal semantic units: "
            + ", ".join(duplicates[:5])
        )
    terminal_ids = set(terminal_semantic)
    orphan_attempts = sorted(terminal_ids - completed_unit_ids)
    missing_attempts = sorted(completed_unit_ids - terminal_ids)
    if orphan_attempts or missing_attempts:
        raise ValueError(
            "Attempt/result terminal coverage mismatch "
            f"(terminal_without_result={len(orphan_attempts)}, "
            f"result_without_terminal={len(missing_attempts)})."
        )


class DurableAttemptLogger:
    """Append-only, fsync-backed JSONL writer for individual model attempts."""

    def __init__(self, path: str | Path, *, experiment: str) -> None:
        self.path = Path(path)
        self.experiment = experiment
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Validate any existing sidecar before appending during resume.
        summary = summarize_attempt_log(self.path)
        self._sequence = summary.total_attempts
        self._previous_record_sha256 = summary.last_record_sha256
        self._schema_version = (
            LEGACY_ATTEMPT_LOG_SCHEMA_VERSION
            if summary.schema_versions == (LEGACY_ATTEMPT_LOG_SCHEMA_VERSION,)
            else ATTEMPT_LOG_SCHEMA_VERSION
        )

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
            "schema_version": self._schema_version,
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
        if self._schema_version == ATTEMPT_LOG_SCHEMA_VERSION:
            record["previous_record_sha256"] = self._previous_record_sha256
            record["record_sha256"] = _record_sha256(record)
            self._previous_record_sha256 = str(record["record_sha256"])
        encoded = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

    def summary(self) -> AttemptLogSummary:
        return summarize_attempt_log(self.path)
