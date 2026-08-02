"""Run the exploratory Part 1 draft panel on compatibility-proven InferenceHub routes.

This runner deliberately cannot promote evidence to a confirmatory or paper result.
It executes subjects only; one dedicated, non-overlapping judge is hash-bound in the
manifest and reserved for a later analysis pass.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import stat
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dotenv import load_dotenv

from experiments.misc.inference_hub_compatibility import (
    COMPATIBILITY_SCHEMA_VERSION,
    OPTIONAL_CONTROLS,
    REASONING_FIELDS,
)
from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
    _client_from_environment,
)
from experiments.part1.confirmatory_design import build_draft_bank, build_primary_schedule


SCHEMA_VERSION = 1
DEFAULT_BASE_SEED = 20_260_802
DEFAULT_MAX_WORKERS = 16
DEFAULT_MAX_WORKERS_PER_SUBJECT = 2
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.0
PART1_MAX_TOKENS_FLOOR = 2048
PART1_REASONING_MAX_TOKENS_FLOOR = 8192
_LONG_REASONING_MARKERS = ("kimi", "deepseek", "qwen", "nemotron", "gpt-oss", "glm")
DEFAULT_JUDGE_TARGET_ID = "judge.nvidia-evals-nemotron-3-30b-a3b"
_FINAL_ACTION = re.compile(r"(?:^|\n)\s*([XY])\s*$")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("inference_hub_compatibility.py"),
    Path(__file__).with_name("inference_hub_discovery.py"),
    Path(__file__).with_name("inference_hub_rate_limit.py"),
    Path(__file__).parents[1] / "part1" / "confirmatory_design.py",
)


class InferenceHubPart1PanelError(RuntimeError):
    """A hosted exploratory panel violates its frozen execution contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _secure_mode(path: Path, mode: int) -> None:
    if os.name == "posix":
        path.chmod(mode)


def _require_mode(path: Path, mode: int) -> None:
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != mode:
        raise InferenceHubPart1PanelError(
            f"Unsafe private permissions for {path}: expected {mode:04o}."
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceHubPart1PanelError(
            f"{label} is not readable UTF-8 JSON."
        ) from error
    if not isinstance(value, dict):
        raise InferenceHubPart1PanelError(f"{label} must be a JSON object.")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _secure_mode(path, 0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _self_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_json(
        {key: value for key, value in payload.items() if key != "evidence_sha256"}
    )


def _seal(payload: dict[str, Any]) -> None:
    payload.pop("evidence_sha256", None)
    payload["evidence_sha256"] = _self_hash(payload)


def parse_final_action(response: str) -> str | None:
    """Return only a bare X/Y on the final non-whitespace line."""

    match = _FINAL_ACTION.search(response.strip())
    return match.group(1) if match else None


def build_draft_trials(*, base_seed: int, limit: int | None = None) -> tuple[Any, ...]:
    full = build_primary_schedule(
        build_draft_bank(),
        base_seed=base_seed,
        requested_provider="inference_hub",
        requested_model="compatibility-selected-per-subject-route",
        production=False,
    )
    if len(full) != 384 or len({trial.root_id for trial in full}) != 384:
        raise InferenceHubPart1PanelError(
            "The deterministic Part 1 primary schedule is not 384 unique roots."
        )
    if limit is None:
        return full
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 384:
        raise InferenceHubPart1PanelError("trial limit must be from 1 through 384.")
    return full[:limit]


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InferenceHubPart1PanelError(f"{name} must be a positive integer.")


def _registry_targets(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = registry.get("targets")
    if not isinstance(rows, list):
        raise InferenceHubPart1PanelError("Registry lacks targets[].")
    result: dict[str, dict[str, Any]] = {}
    required = {"id", "provider", "upstream_provider", "model"}
    for row in rows:
        if not isinstance(row, Mapping) or not required <= set(row):
            raise InferenceHubPart1PanelError("Registry target identity is incomplete.")
        target_id = row.get("id")
        if not isinstance(target_id, str) or not target_id or target_id in result:
            raise InferenceHubPart1PanelError("Registry target ids are invalid or repeated.")
        if row.get("provider") != "inference_hub":
            continue
        result[target_id] = dict(row)
    return result


def _validated_compatibility(
    compatibility: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        compatibility.get("schema_version") != COMPATIBILITY_SCHEMA_VERSION
        or compatibility.get("artifact_type")
        != "inference_hub_provider_compatibility"
        or compatibility.get("evidence_sha256") != _self_hash(compatibility)
    ):
        raise InferenceHubPart1PanelError("Compatibility evidence integrity failed.")
    if compatibility.get("registry_sha256") != _sha256_json(registry):
        raise InferenceHubPart1PanelError(
            "Compatibility evidence does not bind the supplied registry."
        )
    targets = compatibility.get("targets")
    if not isinstance(targets, list):
        raise InferenceHubPart1PanelError("Compatibility evidence lacks targets[].")
    target_ids = [
        target.get("target_id") for target in targets if isinstance(target, Mapping)
    ]
    if (
        compatibility.get("target_count") != len(targets)
        or len(target_ids) != len(targets)
        or not all(isinstance(target_id, str) and target_id for target_id in target_ids)
        or len(set(target_ids)) != len(target_ids)
    ):
        raise InferenceHubPart1PanelError("Compatibility target accounting is invalid.")
    selected: dict[str, dict[str, Any]] = {}
    for target in targets:
        if not isinstance(target, Mapping):
            raise InferenceHubPart1PanelError("Compatibility target is not an object.")
        target_id = target.get("target_id")
        route = target.get("selected_execution_candidate")
        if route is None:
            continue
        if (
            not isinstance(target_id, str)
            or not target_id
            or target_id in selected
            or not isinstance(route, str)
            or not route
            or target.get("status") != "execution_candidate_selected"
        ):
            raise InferenceHubPart1PanelError("Compatibility selection is malformed.")
        candidates = target.get("candidates")
        matches = [
            row
            for row in (candidates if isinstance(candidates, list) else [])
            if isinstance(row, Mapping)
            and row.get("route") == route
            and row.get("execution_compatible") is True
        ]
        if len(matches) != 1:
            raise InferenceHubPart1PanelError(
                f"Selected route for {target_id} is not uniquely compatibility-passed."
            )
        candidate = dict(matches[0])
        target_profile = target.get("selected_execution_profile")
        candidate_profile = candidate.get("selected_execution_profile")
        if (
            not isinstance(target_profile, Mapping)
            or not isinstance(candidate_profile, Mapping)
            or target_profile.get("route") != route
            or target_profile.get("status") != "passed"
            or candidate_profile.get("status") != "passed"
            or target_profile.get("profile_id") != candidate_profile.get("profile_id")
            or target_profile.get("controls") != candidate_profile.get("controls")
            or target_profile.get("request_sha256")
            != candidate_profile.get("request_sha256")
            or target_profile.get("validation_source")
            != "execution_profile_probe"
        ):
            raise InferenceHubPart1PanelError(
                f"Selected execution profile for {target_id} is not hash-bound to its candidate."
            )
        supported = target_profile.get("controls")
        if (
            not isinstance(supported, list)
            or len(supported) != len(set(supported))
            or any(control not in OPTIONAL_CONTROLS for control in supported)
        ):
            raise InferenceHubPart1PanelError(
                f"Compatibility controls for {target_id} are invalid."
            )
        selected[target_id] = {
            "target_id": target_id,
            "target_model": target.get("model"),
            "route": route,
            "supported_controls": list(supported),
            "selected_profile_id": target_profile.get("profile_id"),
            "selected_profile_request_sha256": target_profile.get("request_sha256"),
            "candidate_index": candidate.get("candidate_index"),
            "compatibility_max_tokens": candidate.get("max_tokens"),
        }
    if (
        compatibility.get("selected_count") != len(selected)
        or compatibility.get("unresolved_count") != len(targets) - len(selected)
    ):
        raise InferenceHubPart1PanelError("Compatibility selection totals are invalid.")
    return selected


def _identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceHubPart1PanelError("Model identity component is empty.")
    return value.strip().casefold()


def _reject_judge_overlap(
    judge_registry: Mapping[str, Any],
    judge_selection: Mapping[str, Any],
    subjects: Sequence[Mapping[str, Any]],
) -> None:
    judge_id = _identity(judge_registry.get("id"))
    judge_route = _identity(judge_selection.get("route"))
    judge_model = (
        _identity(judge_registry.get("upstream_provider")),
        _identity(judge_registry.get("model")),
    )
    for subject in subjects:
        if judge_id == _identity(subject.get("target_id")):
            raise InferenceHubPart1PanelError("Judge target overlaps a subject target.")
        if judge_route == _identity(subject.get("route")):
            raise InferenceHubPart1PanelError("Judge route overlaps a subject route.")
        if judge_model == (
            _identity(subject.get("upstream_provider")),
            _identity(subject.get("model")),
        ):
            raise InferenceHubPart1PanelError("Judge model overlaps a subject model.")


def select_routes(
    *,
    registry: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    selected_ids: Sequence[str] | None,
    judge_target_id: str,
    excluded_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select passed subject routes and one passed, disjoint future judge route."""

    registry_by_id = _registry_targets(registry)
    compatibility_by_id = _validated_compatibility(compatibility, registry)
    if judge_target_id not in registry_by_id or judge_target_id not in compatibility_by_id:
        raise InferenceHubPart1PanelError(
            "Dedicated judge must have a compatibility-selected InferenceHub route."
        )
    excluded = list(excluded_ids or ())
    if len(excluded) != len(set(excluded)):
        raise InferenceHubPart1PanelError("A subject target was excluded more than once.")
    if judge_target_id in excluded:
        raise InferenceHubPart1PanelError("The dedicated judge cannot be excluded.")
    if selected_ids is not None and excluded:
        raise InferenceHubPart1PanelError(
            "Use either explicit subject targets or exclusions, not both."
        )
    available_subject_ids = {
        target_id
        for target_id in compatibility_by_id
        if target_id != judge_target_id and target_id in registry_by_id
    }
    unknown_exclusions = sorted(set(excluded) - available_subject_ids)
    if unknown_exclusions:
        raise InferenceHubPart1PanelError(
            "Excluded targets lack compatibility-selected subject routes: "
            + ", ".join(unknown_exclusions)
        )
    excluded_set = set(excluded)
    requested = list(selected_ids) if selected_ids is not None else [
        target_id
        for target_id in compatibility_by_id
        if target_id in available_subject_ids and target_id not in excluded_set
    ]
    if not requested:
        raise InferenceHubPart1PanelError("No compatibility-selected subjects remain.")
    if len(requested) != len(set(requested)):
        raise InferenceHubPart1PanelError("A subject target was selected more than once.")
    if judge_target_id in requested:
        raise InferenceHubPart1PanelError("The dedicated judge cannot be a subject.")
    unknown = sorted(
        set(requested) - (set(registry_by_id) & set(compatibility_by_id))
    )
    if unknown:
        raise InferenceHubPart1PanelError(
            "Targets lack compatibility-selected routes: " + ", ".join(unknown)
        )
    subjects: list[dict[str, Any]] = []
    for target_id in requested:
        registered = registry_by_id[target_id]
        selected = compatibility_by_id[target_id]
        if selected["target_model"] != registered.get("model"):
            raise InferenceHubPart1PanelError(
                f"Compatibility model identity changed for {target_id}."
            )
        subjects.append({**registered, **selected})
    judge = {**registry_by_id[judge_target_id], **compatibility_by_id[judge_target_id]}
    _reject_judge_overlap(registry_by_id[judge_target_id], judge, subjects)
    return subjects, judge


class _ChainedJournal:
    """Append-only, fsync'd JSONL with a validated SHA-256 hash chain."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.records = self._load()
        self.tail = self.records[-1]["record_sha256"] if self.records else None

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        _require_mode(self.path, 0o600)
        try:
            text = self.path.read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                raise ValueError("missing final record delimiter")
            lines = (
                []
                if not text
                else text[:-1].split("\n")
            )
            if any(not line for line in lines):
                raise ValueError("blank record")
            rows = [json.loads(line) for line in lines]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise InferenceHubPart1PanelError(
                f"Journal is not valid JSONL: {self.path}."
            ) from error
        previous = None
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise InferenceHubPart1PanelError("Journal record is not an object.")
            recorded = row.get("record_sha256")
            unhashed = {key: value for key, value in row.items() if key != "record_sha256"}
            if row.get("previous_record_sha256") != previous or recorded != _sha256_json(
                unhashed
            ):
                raise InferenceHubPart1PanelError(
                    f"Journal hash chain failed at record {index}: {self.path}."
                )
            previous = recorded
        return rows

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            with self._file_lock():
                self.records = self._load()
                self.tail = (
                    self.records[-1]["record_sha256"] if self.records else None
                )
                row = {**payload, "previous_record_sha256": self.tail}
                row["record_sha256"] = _sha256_json(row)
                encoded = json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                descriptor = os.open(
                    self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
                )
                with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                _secure_mode(self.path, 0o600)
                self.records.append(row)
                self.tail = row["record_sha256"]
                return row

    def reference(self) -> dict[str, Any]:
        with self._lock:
            with self._file_lock():
                self.records = self._load()
                self.tail = (
                    self.records[-1]["record_sha256"] if self.records else None
                )
                return {
                    "path": str(self.path.resolve()),
                    "record_count": len(self.records),
                    "tail_record_sha256": self.tail,
                    "file_sha256": _sha256_file(self.path) if self.path.exists() else None,
                }

    @contextmanager
    def _file_lock(self):
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with lock_path.open("a+b") as handle:
            _secure_mode(lock_path, 0o600)
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _acquire_run_lock(private_dir: Path):
    """Fail before dispatch when another process owns this exact panel output."""

    path = private_dir / ".run.lock"
    handle = path.open("a+b")
    _secure_mode(path, 0o600)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise InferenceHubPart1PanelError(
            "Another process is already running this panel output."
        ) from error
    return handle


def _validate_checkpoint_reference(
    journal: _ChainedJournal, reference: object, *, label: str
) -> None:
    """Require the current journal to contain the manifest-bound prefix."""

    if not isinstance(reference, Mapping):
        raise InferenceHubPart1PanelError(f"Manifest lacks its {label} checkpoint.")
    count = reference.get("record_count")
    tail = reference.get("tail_record_sha256")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise InferenceHubPart1PanelError(f"Manifest {label} count is invalid.")
    if len(journal.records) < count:
        raise InferenceHubPart1PanelError(f"{label} was truncated after its checkpoint.")
    observed_tail = journal.records[count - 1]["record_sha256"] if count else None
    if observed_tail != tail:
        raise InferenceHubPart1PanelError(f"{label} checkpoint hash changed.")
    if len(journal.records) == count and reference.get("file_sha256") != (
        _sha256_file(journal.path) if journal.path.exists() else None
    ):
        raise InferenceHubPart1PanelError(f"{label} checkpoint bytes changed.")


@dataclass(frozen=True)
class _WorkItem:
    subject: Mapping[str, Any]
    trial: Any


def _safe_file_stem(target_id: str) -> str:
    value = _SAFE_NAME.sub("_", target_id).strip("._")
    if not value:
        raise InferenceHubPart1PanelError("Target id has no safe filename form.")
    return value


def _request_contract(subject: Mapping[str, Any], trial: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    supported = list(subject["supported_controls"])
    used: dict[str, Any] = {}
    omitted: dict[str, str] = {}
    if "seed" in supported:
        used["seed"] = trial.generation_settings.generation_seed
    else:
        omitted["seed"] = "not_compatibility_supported"
    if "temperature" in supported:
        used["temperature"] = 0
    else:
        omitted["temperature"] = "not_compatibility_supported"
    if "top_p" in supported:
        used["top_p"] = 1
    else:
        omitted["top_p"] = "not_compatibility_supported"
    if "structured_response" in supported:
        omitted["structured_response"] = "bare_terminal_XY_parser_contract"
    else:
        omitted["structured_response"] = "not_compatibility_supported"
    compatibility_tokens = subject.get("compatibility_max_tokens")
    max_tokens = max(
        trial.generation_settings.max_output_tokens,
        PART1_MAX_TOKENS_FLOOR,
    )
    identity = f"{subject.get('model', '')}/{subject.get('route', '')}".casefold()
    if any(marker in identity for marker in _LONG_REASONING_MARKERS):
        max_tokens = max(max_tokens, PART1_REASONING_MAX_TOKENS_FLOOR)
    if isinstance(compatibility_tokens, int) and not isinstance(
        compatibility_tokens, bool
    ):
        max_tokens = max(max_tokens, compatibility_tokens)
    body = {
        "model": subject["route"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return exactly one character, X or Y, on the only output line. "
                    "Do not explain your answer."
                ),
            },
            {"role": "user", "content": trial.prompt_text},
        ],
        "max_tokens": max_tokens,
        "stream": False,
        **used,
    }
    controls = {
        "supported": supported,
        "used": sorted(used),
        "omitted": omitted,
    }
    return body, controls


def _response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
    message = choice.get("message") if isinstance(choice, Mapping) else None
    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    reasoning = {
        field: message.get(field)
        for field in REASONING_FIELDS
        if isinstance(message, Mapping) and field in message
    }
    output_field = "content" if isinstance(content, str) and content.strip() else None
    output_text = content if output_field is not None else None
    if output_field is None:
        for field in REASONING_FIELDS:
            value = reasoning.get(field)
            if isinstance(value, str) and value.strip():
                output_field, output_text = field, value
                break
    action = parse_final_action(output_text) if isinstance(output_text, str) else None
    response_model = response.get("model")
    return {
        "request_id": response.get("id"),
        "response_model": response_model,
        "finish_reason": finish_reason,
        "usage": response.get("usage"),
        "reasoning_fields": reasoning,
        "output_field": output_field,
        "response_text": output_text,
        "response_text_sha256": (
            hashlib.sha256(output_text.encode("utf-8")).hexdigest()
            if isinstance(output_text, str)
            else None
        ),
        "parsed_action": action,
        "format_valid": action is not None,
    }


def _transient(error: BaseException) -> tuple[bool, str, int | None]:
    if isinstance(error, InferenceHubDiscoveryError):
        status = error.http_status
        code = error.failure_code
        retryable = code in {"connection_error", "connection_timeout"} or status == 429 or (
            isinstance(status, int) and 500 <= status <= 599
        )
        return retryable, code, status
    return False, "unexpected_client_error", None


def _completed_index(
    journals: Mapping[str, _ChainedJournal],
    *,
    trials_by_id: Mapping[str, Any],
    subjects_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    complete: dict[tuple[str, str], dict[str, Any]] = {}
    for target_id, journal in journals.items():
        for row in journal.records:
            key = (target_id, row.get("trial_id"))
            trial = trials_by_id.get(str(row.get("trial_id")))
            subject = subjects_by_id[target_id]
            if (
                row.get("schema_version") != SCHEMA_VERSION
                or row.get("artifact_type") != "inference_hub_part1_raw_response"
                or row.get("target_id") != target_id
                or trial is None
                or row.get("root_id") != trial.root_id
                or row.get("prompt_sha256") != trial.prompt_hash
                or row.get("requested_route") != subject["route"]
                or key in complete
            ):
                raise InferenceHubPart1PanelError(
                    f"Retained response binding is invalid for {target_id}."
                )
            raw = row.get("raw_response")
            if raw is not None and row.get("raw_response_sha256") != _sha256_json(raw):
                raise InferenceHubPart1PanelError("Retained raw response hash changed.")
            complete[key] = row
    return complete


def _retire_stale_reservations(ledger: _ChainedJournal) -> None:
    reserved: dict[str, dict[str, Any]] = {}
    terminal: set[str] = set()
    for row in ledger.records:
        attempt_id = row.get("attempt_id")
        if row.get("event") == "reserved_before_dispatch" and isinstance(attempt_id, str):
            if attempt_id in reserved:
                raise InferenceHubPart1PanelError("Attempt id was reserved twice.")
            reserved[attempt_id] = row
        elif row.get("event") == "attempt_completed" and isinstance(attempt_id, str):
            if attempt_id in terminal:
                raise InferenceHubPart1PanelError("Attempt id completed twice.")
            terminal.add(attempt_id)
    unknown = terminal - set(reserved)
    if unknown:
        raise InferenceHubPart1PanelError("Ledger completion lacks a reservation.")
    for attempt_id in sorted(set(reserved) - terminal):
        ledger.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "inference_hub_part1_attempt_ledger",
                "event": "attempt_completed",
                "attempt_id": attempt_id,
                "outcome": "failed",
                "failure_code": "stale_reservation_retried",
                "transient": True,
                "http_status": None,
                "completed_at_utc": _utc_now(),
            }
        )


def _recover_raw_only_successes(
    ledger: _ChainedJournal,
    raw_journals: Mapping[str, _ChainedJournal],
) -> None:
    """Complete raw-first success commits without redispatching the provider."""

    reservations = {
        str(row.get("attempt_id")): row
        for row in ledger.records
        if row.get("event") == "reserved_before_dispatch"
    }
    completions = {
        str(row.get("attempt_id")): row
        for row in ledger.records
        if row.get("event") == "attempt_completed"
    }
    raw_successes: dict[str, Mapping[str, Any]] = {}
    for target_id, journal in raw_journals.items():
        for row in journal.records:
            if row.get("raw_response") is None:
                continue
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) or attempt_id in raw_successes:
                raise InferenceHubPart1PanelError(
                    "Retained success attempt binding is duplicated or missing."
                )
            reservation = reservations.get(attempt_id)
            if (
                reservation is None
                or reservation.get("target_id") != target_id
                or reservation.get("trial_id") != row.get("trial_id")
                or reservation.get("request_sha256") != row.get("request_sha256")
            ):
                raise InferenceHubPart1PanelError(
                    "Retained success lacks its exact durable reservation."
                )
            raw_successes[attempt_id] = row
            if attempt_id not in completions:
                ledger.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "inference_hub_part1_attempt_ledger",
                        "event": "attempt_completed",
                        "attempt_id": attempt_id,
                        "outcome": "response_retained",
                        "failure_code": None,
                        "transient": False,
                        "http_status": 200,
                        "request_id": row.get("request_id"),
                        "response_model": row.get("response_model"),
                        "response_payload_sha256": row.get("raw_response_sha256"),
                        "response_text_sha256": row.get("response_text_sha256"),
                        "finish_reason": row.get("finish_reason"),
                        "usage": row.get("usage"),
                        "reasoning_fields_sha256": _sha256_json(
                            row.get("reasoning_fields")
                        ),
                        "recovered_after_raw_fsync": True,
                        "completed_at_utc": _utc_now(),
                    }
                )
    for attempt_id, completion in completions.items():
        if (
            completion.get("outcome") == "response_retained"
            and attempt_id not in raw_successes
        ):
            raise InferenceHubPart1PanelError(
                "Success completion lacks a retained raw response; refusing redispatch."
            )


def _prior_attempt_numbers(
    ledger: _ChainedJournal,
) -> dict[tuple[str, str], int]:
    """Return the highest durable reservation number for each subject/trial."""

    result: dict[tuple[str, str], int] = {}
    for row in ledger.records:
        if row.get("event") != "reserved_before_dispatch":
            continue
        target_id = row.get("target_id")
        trial_id = row.get("trial_id")
        attempt_number = row.get("attempt_number")
        if (
            not isinstance(target_id, str)
            or not isinstance(trial_id, str)
            or isinstance(attempt_number, bool)
            or not isinstance(attempt_number, int)
            or attempt_number < 1
        ):
            raise InferenceHubPart1PanelError("Attempt reservation numbering is invalid.")
        key = (target_id, trial_id)
        if attempt_number <= result.get(key, 0):
            raise InferenceHubPart1PanelError(
                "Attempt reservation numbers are not strictly increasing."
            )
        result[key] = attempt_number
    return result


def _manifest_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    mutable = {
        "created_at_utc",
        "last_updated_at_utc",
        "completed_at_utc",
        "complete",
        "summary",
        "journals",
        "evidence_sha256",
        "resume_count",
        "last_resumed_at_utc",
    }
    return {key: value for key, value in manifest.items() if key not in mutable}


def run_panel(
    *,
    compatibility_path: Path,
    registry_path: Path,
    output_dir: Path,
    client: InferenceHubClient,
    selected_ids: Sequence[str] | None = None,
    excluded_ids: Sequence[str] | None = None,
    judge_target_id: str = DEFAULT_JUDGE_TARGET_ID,
    base_seed: int = DEFAULT_BASE_SEED,
    limit: int | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_workers_per_subject: int = DEFAULT_MAX_WORKERS_PER_SUBJECT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    resume: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute a private, resumable panel without ever dispatching the judge."""

    _validate_positive_int("max_workers", max_workers)
    _validate_positive_int("max_workers_per_subject", max_workers_per_subject)
    _validate_positive_int("max_attempts", max_attempts)
    if initial_backoff_seconds < 0:
        raise InferenceHubPart1PanelError("initial_backoff_seconds cannot be negative.")
    if output_dir.exists() and not resume:
        raise InferenceHubPart1PanelError("Output directory already exists; use --resume.")
    if not output_dir.exists() and resume:
        raise InferenceHubPart1PanelError("Resume output directory does not exist.")
    private_dir = output_dir / "private"
    raw_dir = private_dir / "raw_responses"
    if resume:
        for directory in (output_dir, private_dir, raw_dir):
            if not directory.is_dir():
                raise InferenceHubPart1PanelError(
                    f"Resume private directory is missing: {directory}."
                )
            _require_mode(directory, 0o700)
    else:
        output_dir.mkdir(parents=True, mode=0o700)
        _secure_mode(output_dir, 0o700)
        for directory in (private_dir, raw_dir):
            directory.mkdir(mode=0o700)
            _secure_mode(directory, 0o700)
    for directory in (output_dir, private_dir, raw_dir):
        _require_mode(directory, 0o700)
    run_lock_handle = _acquire_run_lock(private_dir)

    registry = _read_json(registry_path, "registry")
    compatibility = _read_json(compatibility_path, "compatibility evidence")
    if compatibility.get("endpoint") != client.base_url:
        raise InferenceHubPart1PanelError(
            "Compatibility endpoint does not match the configured client endpoint."
        )
    subjects, judge = select_routes(
        registry=registry,
        compatibility=compatibility,
        selected_ids=selected_ids,
        judge_target_id=judge_target_id,
        excluded_ids=excluded_ids,
    )
    stems = [_safe_file_stem(str(subject["target_id"])) for subject in subjects]
    if len(stems) != len(set(stems)):
        raise InferenceHubPart1PanelError("Subject ids collide as evidence filenames.")

    full_trials = build_draft_trials(base_seed=base_seed)
    trials = full_trials if limit is None else build_draft_trials(base_seed=base_seed, limit=limit)
    schedule_binding = [
        {
            "trial_id": trial.trial_id,
            "root_id": trial.root_id,
            "game": trial.game,
            "domain": trial.domain,
            "counterbalance_id": trial.counterbalance_id,
            "prompt_sha256": trial.prompt_hash,
            "generation_settings": {
                "temperature": trial.generation_settings.temperature,
                "top_p": trial.generation_settings.top_p,
                "max_output_tokens": trial.generation_settings.max_output_tokens,
                "generation_seed": trial.generation_settings.generation_seed,
                "seed_base": trial.generation_settings.seed_base,
                "seed_derivation": trial.generation_settings.seed_derivation,
            },
        }
        for trial in trials
    ]
    full_schedule_binding = [
        {"trial_id": trial.trial_id, "root_id": trial.root_id, "prompt_sha256": trial.prompt_hash}
        for trial in full_trials
    ]
    runtime = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    input_artifacts = {
        "compatibility": {
            "path": str(compatibility_path.resolve()),
            "file_sha256": _sha256_file(compatibility_path),
            "evidence_sha256": compatibility["evidence_sha256"],
        },
        "registry": {
            "path": str(registry_path.resolve()),
            "file_sha256": _sha256_file(registry_path),
            "canonical_sha256": _sha256_json(registry),
        },
    }
    source_artifacts = {
        str(path.resolve()): _sha256_file(path.resolve()) for path in _SOURCE_PATHS
    }
    subject_manifest = [
        {
            "target_id": subject["target_id"],
            "upstream_provider": subject["upstream_provider"],
            "model": subject["model"],
            "route": subject["route"],
            "candidate_index": subject["candidate_index"],
            "supported_controls": subject["supported_controls"],
            "selected_profile_id": subject["selected_profile_id"],
            "selected_profile_request_sha256": subject[
                "selected_profile_request_sha256"
            ],
        }
        for subject in subjects
    ]
    client_rate_limit_contract = getattr(client, "rate_limit_contract", None)
    if isinstance(client, InferenceHubClient):
        if not isinstance(client_rate_limit_contract, Mapping):
            raise InferenceHubPart1PanelError(
                "InferenceHub network client lacks a rate-limit contract."
            )
        rate_limit_contract = dict(client_rate_limit_contract)
    else:
        rate_limit_contract = {
            "enforcement": "external_non_network_test_double",
            "network_dispatch_permitted": False,
        }
    fresh_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "inference_hub_part1_large_n_exploratory_panel",
        "created_at_utc": _utc_now(),
        "analysis_role": "exploratory_hosted_scale_panel_only",
        "draft_bank_human_approved": False,
        "confirmatory_or_paper_promotion_permitted": False,
        "judge_dispatched": False,
        "input_artifacts": input_artifacts,
        "source_artifacts": source_artifacts,
        "runtime": runtime,
        "runtime_sha256": _sha256_json(runtime),
        "base_seed": base_seed,
        "full_primary_root_count": 384,
        "full_primary_schedule_sha256": _sha256_json(full_schedule_binding),
        "executed_trial_count_per_subject": len(trials),
        "executed_schedule_sha256": _sha256_json(schedule_binding),
        "trial_limit": limit,
        "operationally_excluded_target_ids": sorted(excluded_ids or ()),
        "subject_routes": subject_manifest,
        "judge_reservation": {
            "target_id": judge["target_id"],
            "upstream_provider": judge["upstream_provider"],
            "model": judge["model"],
            "route": judge["route"],
            "candidate_index": judge["candidate_index"],
            "supported_controls": judge["supported_controls"],
            "selected_profile_id": judge["selected_profile_id"],
            "selected_profile_request_sha256": judge[
                "selected_profile_request_sha256"
            ],
            "role": "same_separate_judge_reserved_for_later_across_all_subjects",
            "subject_target_ids": [subject["target_id"] for subject in subjects],
            "dispatch_permitted_in_this_runner": False,
        },
        "execution_contract": {
            "strategy": "bounded_trial_and_model_thread_pool",
            "configured_max_workers": max_workers,
            "effective_max_workers": min(max_workers, len(subjects) * len(trials)),
            "max_workers_per_subject": max_workers_per_subject,
            "max_attempts_per_trial": max_attempts,
            "initial_exponential_backoff_seconds": initial_backoff_seconds,
            "shared_rate_limit": rate_limit_contract,
            "payload_hashing": "canonical_credential_free_json_before_dispatch",
            "ledger": "append_only_fsync_sha256_chain_reserve_before_dispatch",
            "response_parser": "final_non_whitespace_line_exactly_X_or_Y",
        },
        "complete": False,
        "summary": {},
        "journals": {},
    }
    manifest_path = private_dir / "manifest.json"
    if resume:
        manifest = _read_json(manifest_path, "panel manifest")
        _require_mode(manifest_path, 0o600)
        if manifest.get("evidence_sha256") != _self_hash(manifest):
            raise InferenceHubPart1PanelError("Panel manifest self-hash failed.")
        if _manifest_bindings(manifest) != _manifest_bindings(fresh_manifest):
            raise InferenceHubPart1PanelError("Resume contract differs from the frozen run.")
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        manifest["last_resumed_at_utc"] = _utc_now()
    else:
        manifest = fresh_manifest

    ledger = _ChainedJournal(private_dir / "attempt_ledger.jsonl")
    if resume:
        checkpoints = manifest.get("journals")
        if not isinstance(checkpoints, Mapping):
            raise InferenceHubPart1PanelError("Manifest journal checkpoints are invalid.")
        _validate_checkpoint_reference(
            ledger, checkpoints.get("attempt_ledger"), label="attempt ledger"
        )
    raw_journals = {
        subject["target_id"]: _ChainedJournal(
            raw_dir / f"{_safe_file_stem(subject['target_id'])}.jsonl"
        )
        for subject in subjects
    }
    if resume:
        raw_checkpoints = manifest["journals"].get("raw_responses")
        if not isinstance(raw_checkpoints, Mapping) or set(raw_checkpoints) != set(
            raw_journals
        ):
            raise InferenceHubPart1PanelError(
                "Manifest raw-response checkpoints do not match the subjects."
            )
        for target_id, journal in raw_journals.items():
            _validate_checkpoint_reference(
                journal,
                raw_checkpoints[target_id],
                label=f"raw responses for {target_id}",
            )
    else:
        # Publish the first hash-bound empty checkpoints before any dispatch.
        manifest["journals"] = {
            "attempt_ledger": ledger.reference(),
            "raw_responses": {
                target_id: journal.reference()
                for target_id, journal in raw_journals.items()
            },
        }
        _seal(manifest)
        _atomic_json(manifest_path, manifest)
    _recover_raw_only_successes(ledger, raw_journals)
    _retire_stale_reservations(ledger)
    prior_attempt_numbers = _prior_attempt_numbers(ledger)
    subjects_by_id = {subject["target_id"]: subject for subject in subjects}
    trials_by_id = {trial.trial_id: trial for trial in trials}
    completed = _completed_index(
        raw_journals, trials_by_id=trials_by_id, subjects_by_id=subjects_by_id
    )
    # Interleave subjects so the global worker pool fans out across models
    # instead of sending a burst of adjacent trials to one provider route.
    work = [
        _WorkItem(subject, trial)
        for trial in trials
        for subject in subjects
        if (subject["target_id"], trial.trial_id) not in completed
    ]
    subject_locks = {
        str(subject["target_id"]): threading.BoundedSemaphore(
            max_workers_per_subject
        )
        for subject in subjects
    }

    def execute_serial(item: _WorkItem) -> dict[str, Any]:
        subject, trial = item.subject, item.trial
        body, controls = _request_contract(subject, trial)
        request_bytes = _canonical_bytes(body)
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        last_failure: dict[str, Any] | None = None
        key = (str(subject["target_id"]), trial.trial_id)
        first_attempt = prior_attempt_numbers.get(key, 0) + 1
        for attempt_number in range(first_attempt, max_attempts + 1):
            attempt_id = f"part1_{uuid.uuid4().hex}"
            # This durable reservation and exact body hash happen before dispatch.
            ledger.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "inference_hub_part1_attempt_ledger",
                    "event": "reserved_before_dispatch",
                    "attempt_id": attempt_id,
                    "target_id": subject["target_id"],
                    "upstream_provider": subject["upstream_provider"],
                    "model": subject["model"],
                    "route": subject["route"],
                    "trial_id": trial.trial_id,
                    "root_id": trial.root_id,
                    "attempt_number": attempt_number,
                    "request_sha256": request_sha256,
                    "request_body_bytes": len(request_bytes),
                    "prompt_sha256": trial.prompt_hash,
                    "controls": controls,
                    "reserved_at_utc": _utc_now(),
                }
            )
            try:
                if isinstance(client, InferenceHubClient):
                    response = client.post(
                        "/chat/completions",
                        body,
                        upstream_provider=str(subject["upstream_provider"]),
                    )
                else:
                    response = client.post("/chat/completions", body)
                if not isinstance(response, Mapping):
                    raise TypeError("client response is not an object")
            except Exception as error:
                retryable, failure_code, http_status = _transient(error)
                last_failure = {
                    "failure_code": failure_code,
                    "http_status": http_status,
                    "error_type": type(error).__name__,
                }
                ledger.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "inference_hub_part1_attempt_ledger",
                        "event": "attempt_completed",
                        "attempt_id": attempt_id,
                        "outcome": "failed",
                        "failure_code": failure_code,
                        "transient": retryable,
                        "http_status": http_status,
                        "completed_at_utc": _utc_now(),
                    }
                )
                if retryable and attempt_number < max_attempts:
                    sleep_fn(initial_backoff_seconds * (2 ** (attempt_number - 1)))
                    continue
                response = None
            if response is not None:
                metadata = _response_metadata(response)
                response_sha256 = _sha256_json(response)
                row = raw_journals[subject["target_id"]].append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "inference_hub_part1_raw_response",
                        "target_id": subject["target_id"],
                        "upstream_provider": subject["upstream_provider"],
                        "model": subject["model"],
                        "requested_route": subject["route"],
                        "response_model": metadata["response_model"],
                        "model_identity_valid": metadata["response_model"] == subject["route"],
                        "trial_id": trial.trial_id,
                        "root_id": trial.root_id,
                        "game": trial.game,
                        "domain": trial.domain,
                        "counterbalance_id": trial.counterbalance_id,
                        "prompt_text": trial.prompt_text,
                        "prompt_sha256": trial.prompt_hash,
                        "request_sha256": request_sha256,
                        "controls": controls,
                        "attempt_id": attempt_id,
                        "attempt_number": attempt_number,
                        **metadata,
                        "raw_response": dict(response),
                        "raw_response_sha256": response_sha256,
                        "finished_at_utc": _utc_now(),
                    }
                )
                ledger.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "artifact_type": "inference_hub_part1_attempt_ledger",
                        "event": "attempt_completed",
                        "attempt_id": attempt_id,
                        "outcome": "response_retained",
                        "failure_code": None,
                        "transient": False,
                        "http_status": 200,
                        "request_id": metadata["request_id"],
                        "response_model": metadata["response_model"],
                        "response_payload_sha256": response_sha256,
                        "response_text_sha256": metadata["response_text_sha256"],
                        "finish_reason": metadata["finish_reason"],
                        "usage": metadata["usage"],
                        "reasoning_fields_sha256": _sha256_json(
                            metadata["reasoning_fields"]
                        ),
                        "completed_at_utc": _utc_now(),
                    }
                )
                return row
            break
        if first_attempt > max_attempts:
            last_failure = {
                "failure_code": "resume_attempt_budget_exhausted",
                "http_status": None,
                "error_type": None,
            }
        return raw_journals[subject["target_id"]].append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "inference_hub_part1_raw_response",
                "target_id": subject["target_id"],
                "upstream_provider": subject["upstream_provider"],
                "model": subject["model"],
                "requested_route": subject["route"],
                "response_model": None,
                "model_identity_valid": False,
                "trial_id": trial.trial_id,
                "root_id": trial.root_id,
                "game": trial.game,
                "domain": trial.domain,
                "counterbalance_id": trial.counterbalance_id,
                "prompt_text": trial.prompt_text,
                "prompt_sha256": trial.prompt_hash,
                "request_sha256": request_sha256,
                "controls": controls,
                "raw_response": None,
                "raw_response_sha256": None,
                "request_id": None,
                "finish_reason": None,
                "usage": None,
                "reasoning_fields": {},
                "output_field": None,
                "response_text": None,
                "response_text_sha256": None,
                "parsed_action": None,
                "format_valid": False,
                "failure": last_failure,
                "finished_at_utc": _utc_now(),
            }
        )

    def execute(item: _WorkItem) -> dict[str, Any]:
        # Provider limits are commonly route-scoped.  Keep one in-flight call
        # per subject while still running all distinct subject routes together.
        with subject_locks[str(item.subject["target_id"])]:
            return execute_serial(item)

    if work:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(work)),
            thread_name_prefix="inference-hub-part1",
        ) as executor:
            futures = [executor.submit(execute, item) for item in work]
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                if index % 32 == 0:
                    manifest["last_updated_at_utc"] = _utc_now()
                    manifest["journals"] = {
                        "attempt_ledger": ledger.reference(),
                        "raw_responses": {
                            target_id: journal.reference()
                            for target_id, journal in raw_journals.items()
                        },
                    }
                    _seal(manifest)
                    _atomic_json(manifest_path, manifest)

    retained = _completed_index(
        raw_journals, trials_by_id=trials_by_id, subjects_by_id=subjects_by_id
    )
    planned = len(subjects) * len(trials)
    rows = list(retained.values())
    manifest["summary"] = {
        "planned_generations": planned,
        "retained_trial_records": len(rows),
        "responses_received": sum(row.get("raw_response") is not None for row in rows),
        "failed_without_response": sum(row.get("raw_response") is None for row in rows),
        "format_valid": sum(row.get("format_valid") is True for row in rows),
        "format_invalid_retained": sum(
            row.get("raw_response") is not None and row.get("format_valid") is not True
            for row in rows
        ),
        "response_model_identity_mismatches": sum(
            row.get("raw_response") is not None
            and row.get("model_identity_valid") is not True
            for row in rows
        ),
    }
    manifest["journals"] = {
        "attempt_ledger": ledger.reference(),
        "raw_responses": {
            target_id: journal.reference() for target_id, journal in raw_journals.items()
        },
    }
    manifest["complete"] = (
        len(rows) == planned
        and manifest["summary"]["failed_without_response"] == 0
        and manifest["summary"]["response_model_identity_mismatches"] == 0
    )
    manifest["last_updated_at_utc"] = _utc_now()
    if manifest["complete"]:
        manifest["completed_at_utc"] = _utc_now()
    _seal(manifest)
    _atomic_json(manifest_path, manifest)
    fcntl.flock(run_lock_handle.fileno(), fcntl.LOCK_UN)
    run_lock_handle.close()
    return manifest


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be nonnegative") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the compatibility-selected hosted exploratory Part 1 panel."
    )
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=Path("agents/agent_config.registry.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--exclude-target", action="append", dest="excluded_targets")
    parser.add_argument("--judge-target-id", default=DEFAULT_JUDGE_TARGET_ID)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--limit", "--trial-limit", type=_positive_int, default=None)
    parser.add_argument("--max-workers", type=_positive_int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--max-workers-per-subject",
        type=_positive_int,
        default=DEFAULT_MAX_WORKERS_PER_SUBJECT,
    )
    parser.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--initial-backoff-seconds",
        type=_nonnegative_float,
        default=DEFAULT_BACKOFF_SECONDS,
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()
    client = _client_from_environment(args.timeout_seconds)
    manifest = run_panel(
        compatibility_path=args.compatibility,
        registry_path=args.registry,
        output_dir=args.output_dir,
        client=client,
        selected_ids=args.targets,
        excluded_ids=args.excluded_targets,
        judge_target_id=args.judge_target_id,
        base_seed=args.base_seed,
        limit=args.limit,
        max_workers=args.max_workers,
        max_workers_per_subject=args.max_workers_per_subject,
        max_attempts=args.max_attempts,
        initial_backoff_seconds=args.initial_backoff_seconds,
        resume=args.resume,
    )
    print(
        f"Retained {manifest['summary']['retained_trial_records']}/"
        f"{manifest['summary']['planned_generations']} subject trial records."
    )
    print(f"Private evidence: {args.output_dir / 'private' / 'manifest.json'}")
    return 0 if manifest["complete"] else 1


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (InferenceHubPart1PanelError, InferenceHubDiscoveryError, OSError, ValueError) as error:
        print(f"InferenceHub Part 1 panel failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
