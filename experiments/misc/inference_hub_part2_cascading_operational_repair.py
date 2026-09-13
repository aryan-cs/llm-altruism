"""Complete an incomplete Part 2 operational overlay with qualified accounts.

This runner is deliberately a child overlay. It binds both the immutable
original Part 2 source and an independently validated incomplete operational
repair, carries forward every successful parent replacement, and reruns only
the parent's one unresolved original trajectory from day one.

Credential values, request headers, probe prompts, and raw probe responses are
never written. Each configured account receives a stable nonsecret slot name;
only slots that authenticate, list the exact route, and return that exact route
from a bounded chat probe are eligible for experiment dispatch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Callable, Mapping, Sequence
import uuid

from agents.agent_2 import Agent2
from experiments.misc import inference_hub_part2_panel as runner
from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
    _models_routes,
)
from experiments.misc.inference_hub_rate_limit import (
    InferenceHubRateLimiter,
    RateLimitPolicy,
)
from experiments.misc.inference_hub_part2_operational_repair import (
    NUMBERED_BETTER_GOS_KEY,
)
from experiments.part2.part_2 import _collapse_deaths, _derive_seed


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "inference_hub_part2_cascading_operational_trajectory_repair_v1"
PREFLIGHT_ARTIFACT_TYPE = "inference_hub_part2_cascading_repair_preflight_ledger_v1"
TRAJECTORY_ARTIFACT_TYPE = (
    "inference_hub_part2_cascading_operational_repair_effective_trajectory_metrics_v1"
)
MODEL_ARTIFACT_TYPE = (
    "inference_hub_part2_cascading_operational_repair_effective_model_metrics_v1"
)
REPAIR_POLICY = "cascading_whole_trajectory_day_one_exact_route_qualified_pool_v1"
SELECTION_POLICY = "locked_round_robin_over_qualified_account_slots_v1"
RATE_LIMIT_SCOPE = "independent_v2_limiter_per_qualified_account"
EXPECTED_CONFIGURED_ACCOUNT_COUNT = 3
DEFAULT_MAXIMUM_ROUNDS = 8
DEFAULT_TRAJECTORY_WORKERS = 1
DEFAULT_PARTICIPANT_WORKERS = 30
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 900.0
DEFAULT_RATE_PROFILE = runner.HIGH_LATENCY_ORIGINAL_SCALE_RATE_PROFILE
EXPECTED_TARGET_ID = "google/gemini-3.5-flash"
EXPECTED_TRAJECTORY_INDEX = 1
EXPECTED_ENVIRONMENT_SEED = 674_434_863

PREFLIGHT_MAX_TOKENS = 16
PREFLIGHT_MESSAGE = "Reply with exactly OK."
PREFLIGHT_REQUEST_CONTRACT = "exact_route_minimal_chat_identity_probe_v1"

MANIFEST_REQUIRED_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "source_manifest",
        "parent_overlay_manifest", "panel_id", "base_seed", "part2_contract",
        "common_environment_seeds", "subject_routes", "repair_policy",
        "selected_trajectory", "composition_contract", "maximum_rounds",
        "execution_contract",
        "credential_pool", "preflight_ledger", "created_at_utc",
        "last_updated_at_utc", "complete", "summary", "journals",
        "sanitized_artifacts", "evidence_sha256",
    }
)
MANIFEST_OPTIONAL_KEYS = frozenset(
    {"completed_at_utc", "resume_count", "last_resumed_at_utc"}
)
PREFLIGHT_LEDGER_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "created_at_utc", "completed_at_utc",
        "configured_account_slots", "exact_route", "request_contract",
        "request_contract_sha256", "results", "qualified_account_slots",
        "qualified_set_sha256", "cursor_epoch", "base_url",
        "initial_completed_at_utc",
        "initial_session_sha256", "resume_preflight_sessions",
        "evidence_sha256",
    }
)
PREFLIGHT_RESULT_KEYS = frozenset(
    {
        "account_slot", "account_commitment", "started_at_utc", "completed_at_utc",
        "authentication_succeeded", "catalog_http_status",
        "catalog_failure_code", "exact_route_catalog_listed", "chat_attempted",
        "chat_http_status", "chat_failure_code",
        "chat_response_identity_matched", "qualified",
    }
)
PREFLIGHT_SESSION_KEYS = frozenset(
    {
        "session_index", "created_at_utc", "completed_at_utc", "results",
        "qualified_account_slots", "qualified_set_sha256",
        "starting_global_dispatch_ordinal", "previous_session_sha256",
        "session_sha256",
    }
)
_CHAIN_KEYS = frozenset({"previous_record_sha256", "record_sha256"})
RESERVATION_EVENT_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "event", "attempt_id", "target_id",
        "trajectory_index", "day", "slot", "attempt_number",
        "upstream_provider", "model", "requested_route", "environment_seed",
        "generation_seed", "prompt_sha256", "request_sha256",
        "request_body_bytes", "controls", "reserved_at_utc", "account_slot",
        "global_dispatch_ordinal", "cursor_epoch", "preflight_session_index",
        "preflight_session_sha256",
    }
) | _CHAIN_KEYS
ATTEMPT_FAILED_EVENT_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "event", "attempt_id", "target_id",
        "trajectory_index", "day", "slot", "request_sha256", "failure",
        "completed_at_utc",
    }
) | _CHAIN_KEYS
SEMANTIC_EVENT_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "event", "attempt_id", "target_id",
        "trajectory_index", "day", "slot", "attempt_number",
        "upstream_provider", "model", "requested_route", "response_model",
        "model_identity_valid", "environment_seed", "generation_seed",
        "prompt_text", "prompt_sha256", "request_body", "request_sha256",
        "controls", "action", "reasoning", "invalid_reason", "format_valid",
        "visible_content", "visible_content_sha256", "request_id",
        "finish_reason", "usage", "raw_response", "raw_response_sha256",
        "failure", "completed_at_utc",
    }
) | _CHAIN_KEYS
_FORBIDDEN_PERSISTED_FIELD_NAMES = frozenset(
    {
        "api_key", "apikey", "authorization", "bearer", "credential_value",
        "headers", "http_headers", "prompt", "prompt_text", "raw_request",
        "raw_response", "request_body", "response_body", "secret",
    }
)


class Part2CascadingRepairError(RuntimeError):
    """The cascading overlay or its execution provenance is invalid."""


class CredentialQualificationInvalidatedError(Part2CascadingRepairError):
    """A preflight-qualified credential later failed authentication."""


class _TrajectoryNeedsDispatch(Part2CascadingRepairError):
    """A validated retained trajectory still has one or more unfinished units."""


class _CascadingJournal:
    """A hash-chained journal that never follows child-tree symlinks."""

    def __init__(self, path: Path, *, output_root: Path) -> None:
        self.path = Path(os.path.abspath(path))
        self._output_root = Path(os.path.abspath(output_root))
        try:
            self.path.relative_to(self._output_root)
        except ValueError as error:
            raise Part2CascadingRepairError(
                "Cascading journal escapes its exact output root."
            ) from error
        self._lock = threading.Lock()
        self.records = self._load()
        self.tail = self.records[-1]["record_sha256"] if self.records else None

    def _assert_parent_chain(self) -> None:
        current = self._output_root
        relative_parent = self.path.parent.relative_to(self._output_root)
        for component in (Path(), *relative_parent.parents[::-1], relative_parent):
            candidate = current if component == Path() else self._output_root / component
            try:
                status = candidate.lstat()
            except OSError as error:
                raise Part2CascadingRepairError(
                    "Cascading journal parent is unavailable."
                ) from error
            if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
                raise Part2CascadingRepairError(
                    "Cascading journal parent must be a nonsymlink directory."
                )

    @staticmethod
    def _require_regular_descriptor(descriptor: int, *, label: str) -> None:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or stat.S_IMODE(status.st_mode) != 0o600
        ):
            raise Part2CascadingRepairError(
                f"{label} must be a regular nonsymlink 0600 file."
            )

    def _read_bytes(self) -> bytes | None:
        self._assert_parent_chain()
        try:
            status = self.path.lstat()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise Part2CascadingRepairError(
                "Cascading journal is unavailable."
            ) from error
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise Part2CascadingRepairError(
                "Cascading journal must be a regular nonsymlink file."
            )
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError as error:
            raise Part2CascadingRepairError(
                "Cascading journal cannot be opened without following symlinks."
            ) from error
        try:
            self._require_regular_descriptor(
                descriptor, label="Cascading journal"
            )
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                return handle.read()
        finally:
            os.close(descriptor)

    def _load(self) -> list[dict[str, Any]]:
        raw = self._read_bytes()
        if raw is None:
            return []
        try:
            text = raw.decode("utf-8")
            if text and not text.endswith("\n"):
                raise ValueError("missing final record delimiter")
            lines = [] if not text else text[:-1].split("\n")
            if any(not line for line in lines):
                raise ValueError("blank record")
            rows = [json.loads(line) for line in lines]
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise Part2CascadingRepairError(
                "Cascading journal is not valid JSONL."
            ) from error
        previous = None
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise Part2CascadingRepairError(
                    "Cascading journal record is not an object."
                )
            recorded = row.get("record_sha256")
            unhashed = {
                key: value for key, value in row.items()
                if key != "record_sha256"
            }
            if (
                row.get("previous_record_sha256") != previous
                or recorded != runner._sha256_json(unhashed)
            ):
                raise Part2CascadingRepairError(
                    f"Cascading journal hash chain failed at record {index}."
                )
            previous = recorded
        return rows

    @contextmanager
    def _file_lock(self):
        self._assert_parent_chain()
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as error:
            raise Part2CascadingRepairError(
                "Cascading journal lock cannot be opened safely."
            ) from error
        try:
            self._require_regular_descriptor(
                descriptor, label="Cascading journal lock"
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            with self._file_lock():
                self.records = self._load()
                self.tail = (
                    self.records[-1]["record_sha256"]
                    if self.records else None
                )
                row = {**payload, "previous_record_sha256": self.tail}
                row["record_sha256"] = runner._sha256_json(row)
                encoded = (
                    json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
                ).encode("utf-8")
                flags = (
                    os.O_APPEND | os.O_CREAT | os.O_WRONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    descriptor = os.open(self.path, flags, 0o600)
                except OSError as error:
                    raise Part2CascadingRepairError(
                        "Cascading journal cannot be opened safely for append."
                    ) from error
                try:
                    self._require_regular_descriptor(
                        descriptor, label="Cascading journal"
                    )
                    with os.fdopen(descriptor, "ab", closefd=False) as handle:
                        handle.write(encoded)
                        handle.flush()
                        os.fsync(handle.fileno())
                finally:
                    os.close(descriptor)
                self.records.append(row)
                self.tail = row["record_sha256"]
                return row

    def reference(self) -> dict[str, Any]:
        with self._lock:
            # Before the first dispatch there is no journal to coordinate with.
            # Returning its canonical empty reference without creating a sidecar
            # lock keeps the pre-manifest bootstrap exactly recoverable.
            initial_raw = self._read_bytes()
            if initial_raw is None:
                self.records = []
                self.tail = None
                return {
                    "path": str(self.path.resolve()),
                    "record_count": 0,
                    "tail_record_sha256": None,
                    "file_sha256": None,
                }
            with self._file_lock():
                self.records = self._load()
                self.tail = (
                    self.records[-1]["record_sha256"]
                    if self.records else None
                )
                raw = self._read_bytes()
                return {
                    "path": str(self.path.resolve()),
                    "record_count": len(self.records),
                    "tail_record_sha256": self.tail,
                    "file_sha256": (
                        hashlib.sha256(raw).hexdigest()
                        if raw is not None else None
                    ),
                }


@dataclass(frozen=True)
class AccountClient:
    account_slot: str
    client: InferenceHubClient
    account_commitment: str


def _account_slots(count: int) -> tuple[str, ...]:
    if count != EXPECTED_CONFIGURED_ACCOUNT_COUNT:
        raise Part2CascadingRepairError(
            f"Exactly {EXPECTED_CONFIGURED_ACCOUNT_COUNT} configured accounts are required."
        )
    return tuple(f"account-slot-{index:02d}" for index in range(1, count + 1))


def _credential_commitment(
    credential: str, *, cursor_epoch: str, base_url: str,
) -> str:
    # The slot-to-commitment mapping binds a credential to its stable slot.
    # Deliberately exclude the slot from the HMAC message so configuring the
    # same credential twice yields the same commitment and is rejected.
    message = (
        "llm-altruism-part2-cascading-account-identity-v1\0"
        f"{cursor_epoch}\0{base_url}"
    ).encode("utf-8")
    return hmac.new(credential.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _qualified_set_sha256(
    account_slots: Sequence[str], account_commitments: Mapping[str, str],
) -> str:
    return runner._sha256_json(
        {
            "account_slots": [
                {"account_slot": slot, "account_commitment": account_commitments[slot]}
                for slot in account_slots
            ],
            "ordering": "configured_slot_order",
        }
    )


def _preflight_request(route: str) -> dict[str, Any]:
    return {
        "model": route,
        "messages": [{"role": "user", "content": PREFLIGHT_MESSAGE}],
        "max_tokens": PREFLIGHT_MAX_TOKENS,
        "stream": False,
    }


def _assert_no_secret_like_fields(value: object, *, label: str) -> None:
    """Reject fields that could turn sanitized provenance into a secret sink."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _FORBIDDEN_PERSISTED_FIELD_NAMES:
                raise Part2CascadingRepairError(
                    f"{label} contains forbidden secret-like field {key!r}."
                )
            _assert_no_secret_like_fields(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_no_secret_like_fields(item, label=label)
    elif isinstance(value, str) and value.casefold().startswith("bearer "):
        raise Part2CascadingRepairError(
            f"{label} contains a forbidden bearer-like value."
        )


def _probe_failure(error: BaseException) -> tuple[str, int | None]:
    if isinstance(error, InferenceHubDiscoveryError):
        return error.failure_code, error.http_status
    return "unexpected_client_error", None


def _preflight_session_sha256(value: Mapping[str, Any]) -> str:
    return runner._sha256_json(
        {key: item for key, item in value.items() if key != "session_sha256"}
    )


def _utc_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Part2CascadingRepairError(f"{label} is not canonical UTC.")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise Part2CascadingRepairError(f"{label} is not canonical UTC.") from error
    if parsed.tzinfo != timezone.utc:
        raise Part2CascadingRepairError(f"{label} is not canonical UTC.")
    return parsed


def _http_status_or_none(value: object) -> bool:
    return value is None or (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 100 <= value <= 599
    )


def _exact_probe_failure_status_pair(code: object, status_code: object) -> bool:
    """Match the failure classifier's exact HTTP/non-HTTP representation."""

    return (
        isinstance(code, str)
        and bool(code)
        and (
            (code == "http_error" and _http_status_or_none(status_code)
             and status_code is not None)
            or (code != "http_error" and status_code is None)
        )
    )


def blind_preflight_accounts(
    accounts: Sequence[AccountClient], *, exact_route: str,
    cursor_epoch: str, now: Callable[[], str] = runner._utc_now,
) -> dict[str, Any]:
    """Probe every configured account while retaining only safe classifications."""

    expected_slots = _account_slots(len(accounts))
    if tuple(account.account_slot for account in accounts) != expected_slots:
        raise Part2CascadingRepairError(
            "Configured accounts do not have the exact stable slot ordering."
        )
    endpoints = {account.client.base_url for account in accounts}
    commitments = {
        account.account_slot: account.account_commitment for account in accounts
    }
    if len(endpoints) != 1:
        raise Part2CascadingRepairError(
            "Configured accounts do not share the exact endpoint."
        )
    if (
        len(set(commitments.values())) != len(accounts)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in commitments.values()
        )
    ):
        raise Part2CascadingRepairError(
            "Configured account commitments are invalid or duplicated."
        )
    base_url = next(iter(endpoints))
    if base_url != runner.DEFAULT_BASE_URL:
        raise Part2CascadingRepairError(
            "Configured accounts do not use the frozen InferenceHub endpoint."
        )
    request = _preflight_request(exact_route)
    created_at = now()
    results: list[dict[str, Any]] = []
    for account in accounts:
        started_at = now()
        authenticated = False
        catalog_status: int | None = None
        catalog_failure: str | None = None
        listed = False
        chat_attempted = False
        chat_status: int | None = None
        chat_failure: str | None = None
        identity_matched: bool | None = None
        try:
            catalog = account.client.get("/models")
            routes = _models_routes(catalog)
            catalog_status = 200
            authenticated = True
            listed = exact_route in routes
        except Exception as error:
            catalog_failure, catalog_status = _probe_failure(error)
        if authenticated and listed:
            chat_attempted = True
            try:
                response = account.client.post(
                    "/chat/completions", request,
                    upstream_provider=str(exact_route),
                )
                chat_status = 200
                identity_matched = response.get("model") == exact_route
            except Exception as error:
                chat_failure, chat_status = _probe_failure(error)
                identity_matched = False
        qualified = bool(
            authenticated and listed and chat_status == 200
            and identity_matched is True
        )
        results.append(
            {
                "account_slot": account.account_slot,
                "account_commitment": account.account_commitment,
                "started_at_utc": started_at,
                "completed_at_utc": now(),
                "authentication_succeeded": authenticated,
                "catalog_http_status": catalog_status,
                "catalog_failure_code": catalog_failure,
                "exact_route_catalog_listed": listed,
                "chat_attempted": chat_attempted,
                "chat_http_status": chat_status,
                "chat_failure_code": chat_failure,
                "chat_response_identity_matched": identity_matched,
                "qualified": qualified,
            }
        )
    qualified_slots = [row["account_slot"] for row in results if row["qualified"]]
    if not qualified_slots:
        raise Part2CascadingRepairError(
            "No configured account passed the exact-route blind preflight."
        )
    qualified_hash = _qualified_set_sha256(qualified_slots, commitments)
    initial_session = {
        "session_index": 0,
        "created_at_utc": created_at,
        "completed_at_utc": now(),
        "results": results,
        "qualified_account_slots": qualified_slots,
        "qualified_set_sha256": qualified_hash,
        "starting_global_dispatch_ordinal": 0,
        "previous_session_sha256": None,
    }
    initial_session["session_sha256"] = _preflight_session_sha256(initial_session)
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": PREFLIGHT_ARTIFACT_TYPE,
        "created_at_utc": created_at,
        "completed_at_utc": initial_session["completed_at_utc"],
        "initial_completed_at_utc": initial_session["completed_at_utc"],
        "configured_account_slots": list(expected_slots),
        "exact_route": exact_route,
        "base_url": base_url,
        "request_contract": PREFLIGHT_REQUEST_CONTRACT,
        "request_contract_sha256": runner._sha256_json(request),
        "results": results,
        "qualified_account_slots": qualified_slots,
        "qualified_set_sha256": qualified_hash,
        "cursor_epoch": cursor_epoch,
        "initial_session_sha256": initial_session["session_sha256"],
        "resume_preflight_sessions": [],
    }
    _assert_no_secret_like_fields(ledger, label="preflight ledger")
    runner._seal(ledger)
    return ledger


def _validate_preflight_ledger(
    ledger: Mapping[str, Any], *, exact_route: str,
) -> tuple[str, tuple[str, ...]]:
    if set(ledger) != PREFLIGHT_LEDGER_KEYS:
        raise Part2CascadingRepairError("Preflight ledger schema changed.")
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("artifact_type") != PREFLIGHT_ARTIFACT_TYPE
        or ledger.get("evidence_sha256") != runner._self_hash(ledger)
        or ledger.get("exact_route") != exact_route
        or ledger.get("request_contract") != PREFLIGHT_REQUEST_CONTRACT
        or ledger.get("request_contract_sha256")
        != runner._sha256_json(_preflight_request(exact_route))
    ):
        raise Part2CascadingRepairError("Preflight ledger binding or seal changed.")
    _assert_no_secret_like_fields(ledger, label="preflight ledger")
    configured = ledger.get("configured_account_slots")
    if configured != list(_account_slots(EXPECTED_CONFIGURED_ACCOUNT_COUNT)):
        raise Part2CascadingRepairError("Preflight configured account slots changed.")
    results = ledger.get("results")
    qualified, commitments = _validate_probe_results(results, configured=configured)
    if not qualified:
        raise Part2CascadingRepairError("Preflight has no qualified accounts.")
    if (
        ledger.get("qualified_account_slots") != qualified
        or ledger.get("qualified_set_sha256")
        != _qualified_set_sha256(qualified, commitments)
    ):
        raise Part2CascadingRepairError("Preflight qualified-set binding changed.")
    cursor_epoch = ledger.get("cursor_epoch")
    cursor_prefix = "cursor-epoch-"
    cursor_suffix = (
        cursor_epoch[len(cursor_prefix):]
        if isinstance(cursor_epoch, str) and cursor_epoch.startswith(cursor_prefix)
        else ""
    )
    if (
        not isinstance(cursor_epoch, str)
        or len(cursor_suffix) != 32
        or any(character not in "0123456789abcdef" for character in cursor_suffix)
    ):
        raise Part2CascadingRepairError("Preflight cursor epoch is invalid.")
    base_url = ledger.get("base_url")
    if base_url != runner.DEFAULT_BASE_URL:
        raise Part2CascadingRepairError("Preflight endpoint binding is invalid.")
    initial_session = {
        "session_index": 0,
        "created_at_utc": ledger.get("created_at_utc"),
        "completed_at_utc": ledger.get("initial_completed_at_utc"),
        "results": results,
        "qualified_account_slots": qualified,
        "qualified_set_sha256": ledger.get("qualified_set_sha256"),
        "starting_global_dispatch_ordinal": 0,
        "previous_session_sha256": None,
    }
    resume_sessions = ledger.get("resume_preflight_sessions")
    if not isinstance(resume_sessions, list):
        raise Part2CascadingRepairError("Resume preflight session schema changed.")
    initial_session["session_sha256"] = _preflight_session_sha256(initial_session)
    if ledger.get("initial_session_sha256") != initial_session["session_sha256"]:
        raise Part2CascadingRepairError("Initial preflight session hash changed.")
    previous_hash = str(initial_session["session_sha256"])
    last_completed = initial_session["completed_at_utc"]
    previous_dispatch_boundary = 0
    initial_created_at = _utc_datetime(
        initial_session["created_at_utc"], label="Initial preflight start"
    )
    previous_completed_at = _utc_datetime(
        initial_session["completed_at_utc"], label="Initial preflight completion"
    )
    if initial_created_at > previous_completed_at:
        raise Part2CascadingRepairError("Initial preflight timestamps are out of order.")
    prior_probe_completion = initial_created_at
    for result in results:
        probe_started = _utc_datetime(
            result.get("started_at_utc"), label="Initial account preflight start"
        )
        probe_completed = _utc_datetime(
            result.get("completed_at_utc"),
            label="Initial account preflight completion",
        )
        if not (
            prior_probe_completion <= probe_started
            <= probe_completed <= previous_completed_at
        ):
            raise Part2CascadingRepairError(
                "Initial account preflight timestamps are out of order."
            )
        prior_probe_completion = probe_completed
    for index, session in enumerate(resume_sessions, 1):
        if not isinstance(session, Mapping) or set(session) != PREFLIGHT_SESSION_KEYS:
            raise Part2CascadingRepairError("Resume preflight session schema changed.")
        session_qualified, session_commitments = _validate_probe_results(
            session.get("results"), configured=configured,
        )
        if (
            session.get("session_index") != index
            or session.get("qualified_account_slots") != qualified
            or session_qualified != qualified
            or session_commitments != commitments
            or session.get("qualified_set_sha256")
            != _qualified_set_sha256(qualified, commitments)
            or not isinstance(
                session.get("starting_global_dispatch_ordinal"), int
            )
            or isinstance(
                session.get("starting_global_dispatch_ordinal"), bool
            )
            or session.get("starting_global_dispatch_ordinal") < 0
            or session.get("starting_global_dispatch_ordinal")
            < previous_dispatch_boundary
            or session.get("previous_session_sha256") != previous_hash
            or session.get("session_sha256") != _preflight_session_sha256(session)
        ):
            raise Part2CascadingRepairError(
                "Resume preflight qualification or hash chain changed."
            )
        session_created_at = _utc_datetime(
            session.get("created_at_utc"), label="Resume preflight start"
        )
        session_completed_at = _utc_datetime(
            session.get("completed_at_utc"), label="Resume preflight completion"
        )
        if not previous_completed_at <= session_created_at <= session_completed_at:
            raise Part2CascadingRepairError(
                "Resume preflight timestamps are out of order."
            )
        prior_probe_completion = session_created_at
        for result in session["results"]:
            probe_started = _utc_datetime(
                result.get("started_at_utc"), label="Resume account preflight start"
            )
            probe_completed = _utc_datetime(
                result.get("completed_at_utc"),
                label="Resume account preflight completion",
            )
            if not (
                prior_probe_completion <= probe_started
                <= probe_completed <= session_completed_at
            ):
                raise Part2CascadingRepairError(
                    "Resume account preflight timestamps are out of order."
                )
            prior_probe_completion = probe_completed
        previous_completed_at = session_completed_at
        previous_dispatch_boundary = int(
            session["starting_global_dispatch_ordinal"]
        )
        previous_hash = str(session["session_sha256"])
        last_completed = session.get("completed_at_utc")
    if ledger.get("completed_at_utc") != last_completed:
        raise Part2CascadingRepairError("Preflight ledger completion boundary changed.")
    return cursor_epoch, tuple(qualified)


def _preflight_authorizations(
    ledger: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    return (
        {
            "session_index": 0,
            "session_sha256": ledger["initial_session_sha256"],
            "created_at_utc": ledger["created_at_utc"],
            "completed_at_utc": ledger["initial_completed_at_utc"],
            "starting_global_dispatch_ordinal": 0,
        },
        *(
            {
                "session_index": session["session_index"],
                "session_sha256": session["session_sha256"],
                "created_at_utc": session["created_at_utc"],
                "completed_at_utc": session["completed_at_utc"],
                "starting_global_dispatch_ordinal": session[
                    "starting_global_dispatch_ordinal"
                ],
            }
            for session in ledger["resume_preflight_sessions"]
        ),
    )


def _validate_probe_results(
    results: object, *, configured: Sequence[str],
) -> tuple[list[str], dict[str, str]]:
    if not isinstance(results, list) or len(results) != len(configured):
        raise Part2CascadingRepairError("Preflight result cardinality changed.")
    qualified: list[str] = []
    commitments: dict[str, str] = {}
    for slot, result in zip(configured, results):
        if not isinstance(result, Mapping) or set(result) != PREFLIGHT_RESULT_KEYS:
            raise Part2CascadingRepairError("Preflight result schema changed.")
        commitment = result.get("account_commitment")
        if (
            result.get("account_slot") != slot
            or not isinstance(commitment, str)
            or len(commitment) != 64
            or any(character not in "0123456789abcdef" for character in commitment)
        ):
            raise Part2CascadingRepairError("Preflight account binding changed.")
        commitments[slot] = commitment
        authenticated = result.get("authentication_succeeded")
        listed = result.get("exact_route_catalog_listed")
        attempted = result.get("chat_attempted")
        if not all(isinstance(value, bool) for value in (authenticated, listed, attempted)):
            raise Part2CascadingRepairError("Preflight boolean classification changed.")
        if authenticated:
            if (
                result.get("catalog_http_status") != 200
                or result.get("catalog_failure_code") is not None
                or attempted is not listed
            ):
                raise Part2CascadingRepairError("Preflight catalog success is inconsistent.")
        elif (
            not _http_status_or_none(result.get("catalog_http_status"))
            or result.get("catalog_http_status") == 200
            or not _exact_probe_failure_status_pair(
                result.get("catalog_failure_code"),
                result.get("catalog_http_status"),
            )
            or listed
            or attempted
        ):
            raise Part2CascadingRepairError("Preflight catalog failure is inconsistent.")
        if attempted:
            if not authenticated or not listed:
                raise Part2CascadingRepairError("Preflight chat was attempted without catalog listing.")
            if result.get("chat_http_status") == 200:
                if result.get("chat_failure_code") is not None or not isinstance(
                    result.get("chat_response_identity_matched"), bool
                ):
                    raise Part2CascadingRepairError("Preflight chat success is inconsistent.")
            elif (
                not _http_status_or_none(result.get("chat_http_status"))
                or result.get("chat_http_status") == 200
                or not _exact_probe_failure_status_pair(
                    result.get("chat_failure_code"),
                    result.get("chat_http_status"),
                )
                or result.get("chat_response_identity_matched") is not False
            ):
                raise Part2CascadingRepairError("Preflight chat failure is inconsistent.")
        elif any(
            result.get(field) is not None
            for field in (
                "chat_http_status", "chat_failure_code",
                "chat_response_identity_matched",
            )
        ):
            raise Part2CascadingRepairError("Unattempted preflight chat has fabricated results.")
        expected_qualified = bool(
            authenticated and listed and attempted
            and result.get("chat_http_status") == 200
            and result.get("chat_failure_code") is None
            and result.get("chat_response_identity_matched") is True
        )
        if result.get("qualified") is not expected_qualified:
            raise Part2CascadingRepairError("Preflight qualification was misclassified.")
        if expected_qualified:
            qualified.append(slot)
    if len(set(commitments.values())) != len(commitments):
        raise Part2CascadingRepairError("Preflight account commitments are duplicated.")
    return qualified, commitments


def _append_resume_preflight(
    ledger: Mapping[str, Any], fresh: Mapping[str, Any], *, exact_route: str,
    starting_global_dispatch_ordinal: int,
) -> dict[str, Any]:
    _validate_preflight_ledger(ledger, exact_route=exact_route)
    _validate_preflight_ledger(fresh, exact_route=exact_route)
    static_fields = (
        "configured_account_slots", "exact_route", "base_url", "request_contract",
        "request_contract_sha256", "qualified_account_slots",
        "qualified_set_sha256", "cursor_epoch",
    )
    if any(ledger.get(field) != fresh.get(field) for field in static_fields):
        raise Part2CascadingRepairError(
            "Resume preflight changed endpoint, credentials, or qualified account set."
        )
    old_qualified, old_commitments = _validate_probe_results(
        ledger["results"], configured=ledger["configured_account_slots"]
    )
    new_qualified, new_commitments = _validate_probe_results(
        fresh["results"], configured=fresh["configured_account_slots"]
    )
    if old_qualified != new_qualified or old_commitments != new_commitments:
        raise Part2CascadingRepairError(
            "Resume preflight changed credential-slot qualification."
        )
    if (
        not isinstance(starting_global_dispatch_ordinal, int)
        or isinstance(starting_global_dispatch_ordinal, bool)
        or starting_global_dispatch_ordinal < 0
    ):
        raise Part2CascadingRepairError(
            "Resume preflight dispatch boundary is invalid."
        )
    prior_boundary = (
        ledger["resume_preflight_sessions"][-1][
            "starting_global_dispatch_ordinal"
        ]
        if ledger["resume_preflight_sessions"] else 0
    )
    if starting_global_dispatch_ordinal < prior_boundary:
        raise Part2CascadingRepairError(
            "Resume preflight dispatch boundary moved backwards."
        )
    updated = dict(ledger)
    sessions = [dict(item) for item in ledger["resume_preflight_sessions"]]
    previous_hash = (
        sessions[-1]["session_sha256"]
        if sessions else ledger["initial_session_sha256"]
    )
    session = {
        "session_index": len(sessions) + 1,
        "created_at_utc": fresh["created_at_utc"],
        "completed_at_utc": fresh["completed_at_utc"],
        "results": fresh["results"],
        "qualified_account_slots": fresh["qualified_account_slots"],
        "qualified_set_sha256": fresh["qualified_set_sha256"],
        "starting_global_dispatch_ordinal": starting_global_dispatch_ordinal,
        "previous_session_sha256": previous_hash,
    }
    session["session_sha256"] = _preflight_session_sha256(session)
    sessions.append(session)
    updated["resume_preflight_sessions"] = sessions
    updated["completed_at_utc"] = fresh["completed_at_utc"]
    runner._seal(updated)
    _validate_preflight_ledger(updated, exact_route=exact_route)
    return updated


class QualifiedAccountPool:
    """Allocate and journal the account before each exact-route dispatch."""

    def __init__(
        self, accounts: Sequence[AccountClient], *, qualified_slots: Sequence[str],
        cursor_epoch: str, preflight_session_index: int,
        preflight_session_sha256: str, start_ordinal: int = 0,
    ) -> None:
        by_slot = {account.account_slot: account.client for account in accounts}
        if len(by_slot) != len(accounts):
            raise Part2CascadingRepairError("Configured account slots are duplicated.")
        if not qualified_slots or any(slot not in by_slot for slot in qualified_slots):
            raise Part2CascadingRepairError("Qualified account set is invalid.")
        selected = [by_slot[slot] for slot in qualified_slots]
        endpoints = {client.base_url for client in selected}
        contracts = {runner._sha256_json(client.rate_limit_contract) for client in selected}
        limiter_paths = {
            str(client.rate_limiter.state_path.resolve()) for client in selected
        }
        if len(endpoints) != 1 or len(contracts) != 1:
            raise Part2CascadingRepairError(
                "Qualified accounts do not share endpoint and limiter policy."
            )
        if len(limiter_paths) != len(selected):
            raise Part2CascadingRepairError(
                "Qualified accounts do not have independent limiter scopes."
            )
        self._clients = by_slot
        self.qualified_slots = tuple(qualified_slots)
        self.cursor_epoch = cursor_epoch
        if (
            not isinstance(preflight_session_index, int)
            or isinstance(preflight_session_index, bool)
            or preflight_session_index < 0
            or not isinstance(preflight_session_sha256, str)
            or len(preflight_session_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in preflight_session_sha256
            )
        ):
            raise Part2CascadingRepairError(
                "Dispatch preflight-session binding is invalid."
            )
        self.preflight_session_index = preflight_session_index
        self.preflight_session_sha256 = preflight_session_sha256
        self._ordinal = int(start_ordinal)
        self._lock = threading.Lock()
        self._invalidated_slot: str | None = None
        self.base_url = selected[0].base_url
        self.rate_limit_contract = selected[0].rate_limit_contract

    @property
    def last_ordinal(self) -> int:
        with self._lock:
            return self._ordinal

    def reserve(
        self, journal: Any, payload: Mapping[str, Any], *, dispatch_skipped: bool = False,
    ) -> tuple[dict[str, Any], InferenceHubClient]:
        """Append the provenance-bearing reservation before releasing the allocator."""

        with self._lock:
            if self._invalidated_slot is not None:
                raise CredentialQualificationInvalidatedError(
                    "Dispatch is quarantined after a qualified account failed "
                    "authentication."
                )
            if dispatch_skipped:
                slot: str | None = None
                ordinal: int | None = None
            else:
                self._ordinal += 1
                ordinal = self._ordinal
                slot = self.qualified_slots[(ordinal - 1) % len(self.qualified_slots)]
            row = {
                **dict(payload),
                "account_slot": slot,
                "global_dispatch_ordinal": ordinal,
                "cursor_epoch": self.cursor_epoch,
                "preflight_session_index": self.preflight_session_index,
                "preflight_session_sha256": self.preflight_session_sha256,
                "reserved_at_utc": runner._utc_now(),
            }
            if dispatch_skipped:
                row["dispatch_skipped"] = True
            appended = journal.append(row)
            client = self._clients[self.qualified_slots[0] if slot is None else slot]
            return appended, client

    def invalidate(self, account_slot: str) -> None:
        """Atomically prevent every later reservation in this campaign."""

        with self._lock:
            if account_slot not in self.qualified_slots:
                raise Part2CascadingRepairError(
                    "Only a qualified account can invalidate dispatch."
                )
            self._invalidated_slot = account_slot


def _cascading_failure(error: BaseException) -> tuple[bool, str, int | None]:
    """Retry identical HTTP 400 plus the normal bounded transport classes."""

    if isinstance(error, InferenceHubDiscoveryError):
        status = error.http_status
        code = error.failure_code
        retryable = _retryable_failure_record(
            {"failure_code": code, "http_status": status}
        )
        return retryable, code, status
    return False, "unexpected_client_error", None


def _retryable_failure_record(failure: Mapping[str, Any]) -> bool:
    code = failure.get("failure_code")
    status = failure.get("http_status")
    if code in {"connection_error", "connection_timeout"}:
        return status is None
    return (
        code == "http_error"
        and isinstance(status, int)
        and not isinstance(status, bool)
        and (status in {400, 429} or 500 <= status <= 599)
    )


def _exact_failure_status_pair(failure: Mapping[str, Any]) -> bool:
    code = failure.get("failure_code")
    status = failure.get("http_status")
    if not isinstance(code, str) or not code:
        return False
    if code == "http_error":
        return (
            isinstance(status, int)
            and not isinstance(status, bool)
            and 100 <= status <= 599
        )
    return status is None


def _post_preflight_auth_failure(row: Mapping[str, Any]) -> bool:
    failure = row.get("failure")
    return (
        isinstance(failure, Mapping)
        and failure.get("failure_code") == "http_error"
        and failure.get("http_status") in {401, 403}
    )


def _reservation_payload(
    *, attempt_id: str, subject: Mapping[str, Any], trajectory_index: int,
    environment_seed: int, day: int, slot: int, attempt_number: int,
    generation_seed: int, prompt_sha256: str, request_sha256: str,
    request_body_bytes: int, controls: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "inference_hub_part2_trajectory_event",
        "event": "reserved_before_dispatch",
        "attempt_id": attempt_id,
        "target_id": subject["target_id"],
        "trajectory_index": trajectory_index,
        "day": day,
        "slot": slot,
        "attempt_number": attempt_number,
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "requested_route": subject["route"],
        "environment_seed": environment_seed,
        "generation_seed": generation_seed,
        "prompt_sha256": prompt_sha256,
        "request_sha256": request_sha256,
        "request_body_bytes": request_body_bytes,
        "controls": dict(controls),
        "reserved_at_utc": runner._utc_now(),
    }


def _terminal_payload(
    *, event: str, attempt_id: str, subject: Mapping[str, Any],
    trajectory_index: int, day: int, slot: int, request_sha256: str,
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "inference_hub_part2_trajectory_event",
        "event": event,
        "attempt_id": attempt_id,
        "target_id": subject["target_id"],
        "trajectory_index": trajectory_index,
        "day": day,
        "slot": slot,
        "request_sha256": request_sha256,
        "failure": dict(failure),
        "completed_at_utc": runner._utc_now(),
    }


def _dispatch_unit(
    *, journal: Any, subject: Mapping[str, Any], trajectory_index: int,
    environment_seed: int, day: int, slot: int, prompt: str, system_prompt: str,
    prior_attempt: int, max_attempts: int, initial_backoff_seconds: float,
    pool: QualifiedAccountPool, sleep_fn: Callable[[float], None],
    pending_synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    generation_seed = _derive_seed(
        "inference_hub_part2_generation_v1",
        subject["target_id"], environment_seed, day, slot,
    )
    body, controls = runner._request_contract(
        subject, prompt=prompt, system_prompt=system_prompt,
        generation_seed=generation_seed,
    )
    request_bytes = runner._canonical_bytes(body)
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if pending_synthetic is not None:
        if (
            pending_synthetic.get("dispatch_skipped") is not True
            or pending_synthetic.get("attempt_number") != max_attempts + 1
            or pending_synthetic.get("request_sha256") != request_sha256
            or pending_synthetic.get("account_slot") is not None
            or pending_synthetic.get("global_dispatch_ordinal") is not None
        ):
            raise Part2CascadingRepairError(
                "Pending synthetic exhaustion reservation is invalid."
            )
        failure = {
            "failure_code": "resume_attempt_budget_exhausted",
            "transient": False,
            "http_status": None,
            "error_type": None,
        }
        return journal.append({
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_trajectory_event",
            "event": "semantic_result",
            "attempt_id": pending_synthetic["attempt_id"],
            "target_id": subject["target_id"],
            "trajectory_index": trajectory_index,
            "day": day,
            "slot": slot,
            "attempt_number": max_attempts + 1,
            "upstream_provider": subject["upstream_provider"],
            "model": subject["model"],
            "requested_route": subject["route"],
            "response_model": None,
            "model_identity_valid": False,
            "environment_seed": environment_seed,
            "generation_seed": generation_seed,
            "prompt_text": prompt,
            "prompt_sha256": prompt_sha256,
            "request_body": body,
            "request_sha256": request_sha256,
            "controls": controls,
            "action": "INVALID",
            "reasoning": "",
            "invalid_reason": "transport_failure_exhausted",
            "format_valid": False,
            "visible_content": None,
            "visible_content_sha256": None,
            "request_id": None,
            "finish_reason": None,
            "usage": None,
            "raw_response": None,
            "raw_response_sha256": None,
            "failure": failure,
            "completed_at_utc": runner._utc_now(),
        })
    last_attempt_number = prior_attempt
    for attempt_number in range(prior_attempt + 1, max_attempts + 1):
        last_attempt_number = attempt_number
        attempt_id = f"part2_v4_{uuid.uuid4().hex}"
        reservation, client = pool.reserve(
            journal,
            _reservation_payload(
                attempt_id=attempt_id, subject=subject,
                trajectory_index=trajectory_index, environment_seed=environment_seed,
                day=day, slot=slot, attempt_number=attempt_number,
                generation_seed=generation_seed, prompt_sha256=prompt_sha256,
                request_sha256=request_sha256, request_body_bytes=len(request_bytes),
                controls=controls,
            ),
        )
        try:
            response = client.post(
                "/chat/completions", body,
                upstream_provider=str(subject["upstream_provider"]),
            )
            if not isinstance(response, Mapping):
                raise TypeError("client response is not an object")
        except Exception as error:
            retryable, failure_code, http_status = _cascading_failure(error)
            failure = {
                "failure_code": failure_code,
                "transient": retryable,
                "http_status": http_status,
                "error_type": type(error).__name__,
            }
            if retryable and attempt_number < max_attempts:
                journal.append(
                    _terminal_payload(
                        event="attempt_failed", attempt_id=attempt_id,
                        subject=subject, trajectory_index=trajectory_index,
                        day=day, slot=slot, request_sha256=request_sha256,
                        failure=failure,
                    )
                )
                sleep_fn(initial_backoff_seconds * (2 ** (attempt_number - 1)))
                continue
            if failure_code == "http_error" and http_status in {401, 403}:
                pool.invalidate(str(reservation["account_slot"]))
            terminal = journal.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "inference_hub_part2_trajectory_event",
                    "event": "semantic_result",
                    "attempt_id": attempt_id,
                    "target_id": subject["target_id"],
                    "trajectory_index": trajectory_index,
                    "day": day,
                    "slot": slot,
                    "attempt_number": attempt_number,
                    "upstream_provider": subject["upstream_provider"],
                    "model": subject["model"],
                    "requested_route": subject["route"],
                    "response_model": None,
                    "model_identity_valid": False,
                    "environment_seed": environment_seed,
                    "generation_seed": generation_seed,
                    "prompt_text": prompt,
                    "prompt_sha256": prompt_sha256,
                    "request_body": body,
                    "request_sha256": request_sha256,
                    "controls": controls,
                    "action": "INVALID",
                    "reasoning": "",
                    "invalid_reason": "transport_failure_exhausted",
                    "format_valid": False,
                    "visible_content": None,
                    "visible_content_sha256": None,
                    "request_id": None,
                    "finish_reason": None,
                    "usage": None,
                    "raw_response": None,
                    "raw_response_sha256": None,
                    "failure": failure,
                    "completed_at_utc": runner._utc_now(),
                }
            )
            if failure_code == "http_error" and http_status in {401, 403}:
                raise CredentialQualificationInvalidatedError(
                    "A qualified account returned HTTP 401/403 after preflight; "
                    "the child evidence is quarantined and cannot continue."
                )
            return terminal

        response = dict(response)
        response_model = response.get("model")
        identity_valid = response_model == subject["route"]
        action, reasoning, invalid_reason = runner.parse_decision(response)
        if not identity_valid:
            action, reasoning, invalid_reason = (
                "INVALID", "", "response_model_identity_mismatch"
            )
        content, finish_reason = runner._visible_content(response)
        return journal.append(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "inference_hub_part2_trajectory_event",
                "event": "semantic_result",
                "attempt_id": attempt_id,
                "target_id": subject["target_id"],
                "trajectory_index": trajectory_index,
                "day": day,
                "slot": slot,
                "attempt_number": attempt_number,
                "upstream_provider": subject["upstream_provider"],
                "model": subject["model"],
                "requested_route": subject["route"],
                "response_model": response_model,
                "model_identity_valid": identity_valid,
                "environment_seed": environment_seed,
                "generation_seed": generation_seed,
                "prompt_text": prompt,
                "prompt_sha256": prompt_sha256,
                "request_body": body,
                "request_sha256": request_sha256,
                "controls": controls,
                "action": action,
                "reasoning": reasoning,
                "invalid_reason": invalid_reason,
                "format_valid": action != "INVALID",
                "visible_content": content,
                "visible_content_sha256": (
                    hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if isinstance(content, str) else None
                ),
                "request_id": response.get("id"),
                "finish_reason": finish_reason,
                "usage": response.get("usage"),
                "raw_response": response,
                "raw_response_sha256": runner._sha256_json(response),
                "failure": None,
                "completed_at_utc": runner._utc_now(),
            }
        )

    failure = {
        "failure_code": "resume_attempt_budget_exhausted",
        "transient": False,
        "http_status": None,
        "error_type": None,
    }
    attempt_id = f"part2_v4_terminal_{uuid.uuid4().hex}"
    reservation, _client = pool.reserve(
        journal,
        _reservation_payload(
            attempt_id=attempt_id, subject=subject,
            trajectory_index=trajectory_index, environment_seed=environment_seed,
            day=day, slot=slot, attempt_number=last_attempt_number + 1,
            generation_seed=generation_seed, prompt_sha256=prompt_sha256,
            request_sha256=request_sha256, request_body_bytes=len(request_bytes),
            controls=controls,
        ),
        dispatch_skipped=True,
    )
    return journal.append(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_trajectory_event",
            "event": "semantic_result",
            "attempt_id": reservation["attempt_id"],
            "target_id": subject["target_id"],
            "trajectory_index": trajectory_index,
            "day": day,
            "slot": slot,
            "attempt_number": last_attempt_number + 1,
            "upstream_provider": subject["upstream_provider"],
            "model": subject["model"],
            "requested_route": subject["route"],
            "response_model": None,
            "model_identity_valid": False,
            "environment_seed": environment_seed,
            "generation_seed": generation_seed,
            "prompt_text": prompt,
            "prompt_sha256": prompt_sha256,
            "request_body": body,
            "request_sha256": request_sha256,
            "controls": controls,
            "action": "INVALID",
            "reasoning": "",
            "invalid_reason": "transport_failure_exhausted",
            "format_valid": False,
            "visible_content": None,
            "visible_content_sha256": None,
            "request_id": None,
            "finish_reason": None,
            "usage": None,
            "raw_response": None,
            "raw_response_sha256": None,
            "failure": failure,
            "completed_at_utc": runner._utc_now(),
        }
    )


def _v4_result_index(
    journal: Any, subject: Mapping[str, Any], trajectory_index: int,
    *, environment_seed: int, max_attempts: int,
) -> tuple[
    dict[tuple[int, int], dict[str, Any]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], Mapping[str, Any]],
    dict[tuple[int, int], Mapping[str, Any]],
    dict[tuple[int, int], tuple[Mapping[str, Any], ...]],
]:
    results: dict[tuple[int, int], dict[str, Any]] = {}
    reservations: dict[str, dict[str, Any]] = {}
    terminal_attempts: set[str] = set()
    attempts: dict[tuple[int, int], int] = {}
    reservations_by_unit: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    active_by_unit: dict[tuple[int, int], str] = {}
    retry_ready: set[tuple[int, int]] = set()
    closed_units: set[tuple[int, int]] = set()
    last_terminal_at: dict[tuple[int, int], datetime] = {}
    last_day = 0
    for row in journal.records:
        if (
            row.get("schema_version") != SCHEMA_VERSION
            or row.get("artifact_type") != "inference_hub_part2_trajectory_event"
            or row.get("target_id") != subject["target_id"]
            or row.get("trajectory_index") != trajectory_index
        ):
            raise Part2CascadingRepairError("Trajectory journal binding is invalid.")
        key = (row.get("day"), row.get("slot"))
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in key
        ) or key[0] < 1:
            raise Part2CascadingRepairError("Trajectory journal unit is invalid.")
        unit = (int(key[0]), int(key[1]))
        if unit[0] < last_day:
            raise Part2CascadingRepairError(
                "Trajectory journal days are not in execution order."
            )
        last_day = unit[0]
        event = row.get("event")
        attempt_id = row.get("attempt_id")
        if event == "reserved_before_dispatch":
            expected_keys = RESERVATION_EVENT_KEYS | (
                {"dispatch_skipped"}
                if row.get("dispatch_skipped") is True else set()
            )
            if set(row) != expected_keys:
                raise Part2CascadingRepairError(
                    "Trajectory reservation schema changed."
                )
            if (
                not isinstance(attempt_id, str) or not attempt_id
                or attempt_id in reservations or unit in active_by_unit
                or unit in closed_units
            ):
                raise Part2CascadingRepairError("Attempt reservation is duplicated.")
            number = row.get("attempt_number")
            if (
                not isinstance(number, int) or isinstance(number, bool)
                or number != attempts.get(unit, 0) + 1
                or number > max_attempts + 1
            ):
                raise Part2CascadingRepairError("Attempt numbers are not monotonic.")
            if number > 1 and unit not in retry_ready:
                raise Part2CascadingRepairError(
                    "A retry reservation lacks an eligible failed predecessor."
                )
            skipped = row.get("dispatch_skipped") is True
            if skipped != (number == max_attempts + 1):
                raise Part2CascadingRepairError(
                    "Synthetic exhaustion reservation is outside its exact ceiling."
                )
            reserved_at = _utc_datetime(
                row.get("reserved_at_utc"), label="Reservation timestamp"
            )
            if (
                unit in last_terminal_at
                and reserved_at < last_terminal_at[unit]
            ):
                raise Part2CascadingRepairError(
                    "A retry reservation predates its failed predecessor."
                )
            expected_generation_seed = _derive_seed(
                "inference_hub_part2_generation_v1", subject["target_id"],
                environment_seed, unit[0], unit[1],
            )
            if (
                row.get("upstream_provider") != subject["upstream_provider"]
                or row.get("model") != subject["model"]
                or row.get("requested_route") != subject["route"]
                or row.get("environment_seed") != environment_seed
                or row.get("generation_seed") != expected_generation_seed
                or not isinstance(row.get("prompt_sha256"), str)
                or len(row["prompt_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in row["prompt_sha256"]
                )
                or not isinstance(row.get("request_sha256"), str)
                or len(row["request_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in row["request_sha256"]
                )
                or not isinstance(row.get("request_body_bytes"), int)
                or isinstance(row.get("request_body_bytes"), bool)
                or row.get("request_body_bytes") < 1
                or not isinstance(row.get("controls"), Mapping)
            ):
                raise Part2CascadingRepairError(
                    "Trajectory reservation request binding changed."
                )
            reservations[attempt_id] = row
            attempts[unit] = number
            reservations_by_unit.setdefault(unit, []).append(row)
            active_by_unit[unit] = attempt_id
            retry_ready.discard(unit)
        elif event in {"attempt_failed", "semantic_result"}:
            expected_keys = (
                ATTEMPT_FAILED_EVENT_KEYS
                if event == "attempt_failed" else SEMANTIC_EVENT_KEYS
            )
            if set(row) != expected_keys:
                raise Part2CascadingRepairError(
                    "Trajectory terminal event schema changed."
                )
            if (
                attempt_id not in reservations or attempt_id in terminal_attempts
                or active_by_unit.get(unit) != attempt_id
            ):
                raise Part2CascadingRepairError(
                    "Attempt terminal event lacks one reservation."
                )
            reservation = reservations[str(attempt_id)]
            completed_at = _utc_datetime(
                row.get("completed_at_utc"), label="Attempt completion timestamp"
            )
            if (
                unit != (reservation["day"], reservation["slot"])
                or row.get("request_sha256") != reservation.get("request_sha256")
                or completed_at < _utc_datetime(
                    reservation.get("reserved_at_utc"),
                    label="Reservation timestamp",
                )
            ):
                raise Part2CascadingRepairError(
                    "Attempt terminal changed its request binding."
                )
            terminal_attempts.add(str(attempt_id))
            del active_by_unit[unit]
            last_terminal_at[unit] = completed_at
            failure = row.get("failure")
            if event == "attempt_failed":
                stale = (
                    isinstance(failure, Mapping)
                    and failure.get("failure_code") == "stale_reserved_attempt"
                )
                if stale:
                    if (
                        reservation.get("dispatch_skipped") is True
                        or dict(failure) != {
                            "failure_code": "stale_reserved_attempt",
                            "transient": True,
                            "http_status": None,
                        }
                    ):
                        raise Part2CascadingRepairError(
                            "Stale reservation failure classification changed."
                        )
                elif (
                    not isinstance(failure, Mapping)
                    or set(failure) != {
                        "failure_code", "transient", "http_status", "error_type"
                    }
                    or failure.get("transient") is not True
                    or not _retryable_failure_record(failure)
                    or not isinstance(failure.get("error_type"), str)
                    or not failure.get("error_type")
                    or int(reservation["attempt_number"]) >= max_attempts
                ):
                    raise Part2CascadingRepairError(
                        "Failed attempt violates the exact retry policy."
                    )
                retry_ready.add(unit)
            else:
                if unit in results:
                    raise Part2CascadingRepairError(
                        "A participant unit has two semantic results."
                    )
                for field in (
                    "attempt_number", "upstream_provider", "model",
                    "requested_route", "environment_seed", "generation_seed",
                    "prompt_sha256", "request_sha256", "controls",
                ):
                    if row.get(field) != reservation.get(field):
                        raise Part2CascadingRepairError(
                            "Semantic result changed its reservation binding."
                        )
                raw = row.get("raw_response")
                if raw is not None:
                    if (
                        not isinstance(raw, Mapping)
                        or row.get("raw_response_sha256")
                        != runner._sha256_json(raw)
                        or failure is not None
                        or reservation.get("dispatch_skipped") is True
                    ):
                        raise Part2CascadingRepairError(
                            "Retained successful response binding changed."
                        )
                else:
                    if not isinstance(failure, Mapping) or set(failure) != {
                        "failure_code", "transient", "http_status", "error_type"
                    }:
                        raise Part2CascadingRepairError(
                            "Terminal transport failure schema changed."
                        )
                    synthetic = reservation.get("dispatch_skipped") is True
                    if synthetic:
                        if dict(failure) != {
                            "failure_code": "resume_attempt_budget_exhausted",
                            "transient": False,
                            "http_status": None,
                            "error_type": None,
                        }:
                            raise Part2CascadingRepairError(
                                "Synthetic exhaustion classification changed."
                            )
                    elif (
                        not _exact_failure_status_pair(failure)
                        or not isinstance(failure.get("error_type"), str)
                        or not failure.get("error_type")
                        or failure.get("transient")
                        is not _retryable_failure_record(failure)
                        or (
                            _retryable_failure_record(failure)
                            and int(reservation["attempt_number"]) < max_attempts
                        )
                        or failure.get("failure_code")
                        == "resume_attempt_budget_exhausted"
                    ):
                        raise Part2CascadingRepairError(
                            "Terminal failure violates the exact bounded policy."
                        )
                    if (
                        row.get("response_model") is not None
                        or row.get("model_identity_valid") is not False
                        or row.get("action") != "INVALID"
                        or row.get("reasoning") != ""
                        or row.get("invalid_reason")
                        != "transport_failure_exhausted"
                        or row.get("format_valid") is not False
                        or any(
                            row.get(field) is not None for field in (
                                "visible_content", "visible_content_sha256",
                                "request_id", "finish_reason", "usage",
                                "raw_response_sha256",
                            )
                        )
                    ):
                        raise Part2CascadingRepairError(
                            "Terminal failure fabricated semantic response data."
                        )
                results[unit] = row
                closed_units.add(unit)
        else:
            raise Part2CascadingRepairError("Unknown trajectory journal event.")
    pending_synthetic: dict[tuple[int, int], Mapping[str, Any]] = {}
    pending_real: dict[tuple[int, int], Mapping[str, Any]] = {}
    for attempt_id, reservation in reservations.items():
        if attempt_id in terminal_attempts:
            continue
        key = (int(reservation["day"]), int(reservation["slot"]))
        if reservation.get("dispatch_skipped") is True:
            if key in pending_synthetic:
                raise Part2CascadingRepairError(
                    "A participant unit has multiple pending synthetic reservations."
                )
            pending_synthetic[key] = reservation
        else:
            pending_real[key] = reservation
    return (
        results,
        attempts,
        pending_synthetic,
        pending_real,
        {key: tuple(value) for key, value in reservations_by_unit.items()},
    )


def _validate_dynamic_unit_records(
    *, subject: Mapping[str, Any], trajectory_index: int,
    environment_seed: int, day: int, slot: int, prompt: str,
    system_prompt: str, result: Mapping[str, Any] | None,
    reservations: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute every state-dependent request and retained semantic field."""

    generation_seed = _derive_seed(
        "inference_hub_part2_generation_v1",
        subject["target_id"], environment_seed, day, slot,
    )
    body, controls = runner._request_contract(
        subject, prompt=prompt, system_prompt=system_prompt,
        generation_seed=generation_seed,
    )
    request_bytes = runner._canonical_bytes(body)
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    expected_reservation = {
        "target_id": subject["target_id"],
        "trajectory_index": trajectory_index,
        "day": day,
        "slot": slot,
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "requested_route": subject["route"],
        "environment_seed": environment_seed,
        "generation_seed": generation_seed,
        "prompt_sha256": prompt_sha256,
        "request_sha256": request_sha256,
        "request_body_bytes": len(request_bytes),
        "controls": controls,
    }
    for reservation in reservations:
        if any(
            reservation.get(field) != expected
            for field, expected in expected_reservation.items()
        ):
            raise Part2CascadingRepairError(
                "Retained reservation no longer matches its dynamic prompt state."
            )

    if result is None:
        return
    try:
        runner._validate_retained_result(
            result, subject=subject, trajectory_index=trajectory_index,
            day=day, slot=slot, prompt=prompt,
            request_sha256=request_sha256,
        )
    except Exception as error:
        raise Part2CascadingRepairError(
            "Retained result no longer matches its dynamic prompt state."
        ) from error
    if not reservations:
        raise Part2CascadingRepairError(
            "Retained semantic result has no exact reservation."
        )
    expected_common = {
        "attempt_number": reservations[-1]["attempt_number"],
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "requested_route": subject["route"],
        "environment_seed": environment_seed,
        "generation_seed": generation_seed,
        "prompt_text": prompt,
        "prompt_sha256": prompt_sha256,
        "request_body": body,
        "request_sha256": request_sha256,
        "controls": controls,
    }
    if any(
        result.get(field) != expected
        for field, expected in expected_common.items()
    ):
        raise Part2CascadingRepairError(
            "Retained semantic result changed its exact request contract."
        )

    raw = result.get("raw_response")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise Part2CascadingRepairError("Retained response is not an object.")
    response = dict(raw)
    response_model = response.get("model")
    identity_valid = response_model == subject["route"]
    action, reasoning, invalid_reason = runner.parse_decision(response)
    if not identity_valid:
        action, reasoning, invalid_reason = (
            "INVALID", "", "response_model_identity_mismatch"
        )
    content, finish_reason = runner._visible_content(response)
    expected_response = {
        "response_model": response_model,
        "model_identity_valid": identity_valid,
        "action": action,
        "reasoning": reasoning,
        "invalid_reason": invalid_reason,
        "format_valid": action != "INVALID",
        "visible_content": content,
        "visible_content_sha256": (
            hashlib.sha256(content.encode("utf-8")).hexdigest()
            if isinstance(content, str) else None
        ),
        "request_id": response.get("id"),
        "finish_reason": finish_reason,
        "usage": response.get("usage"),
        "raw_response_sha256": runner._sha256_json(response),
        "failure": None,
    }
    if any(
        result.get(field) != expected
        for field, expected in expected_response.items()
    ):
        raise Part2CascadingRepairError(
            "Retained semantic interpretation no longer matches its response."
        )


def _append_stale_reservation_failure(
    journal: Any, *, reservation: Mapping[str, Any],
    subject: Mapping[str, Any], trajectory_index: int,
) -> None:
    journal.append(
        _terminal_payload(
            event="attempt_failed",
            attempt_id=str(reservation["attempt_id"]),
            subject=subject,
            trajectory_index=trajectory_index,
            day=int(reservation["day"]),
            slot=int(reservation["slot"]),
            request_sha256=str(reservation["request_sha256"]),
            failure={
                "failure_code": "stale_reserved_attempt",
                "transient": True,
                "http_status": None,
            },
        )
    )


def _run_trajectory(
    *, subject: Mapping[str, Any], trajectory_index: int, environment_seed: int,
    contract: runner.Part2Contract, journal: Any,
    pool: QualifiedAccountPool | None,
    participant_workers: int, max_attempts: int,
    initial_backoff_seconds: float, sleep_fn: Callable[[float], None],
    dispatch_missing: bool = True,
) -> dict[str, Any]:
    (
        results, attempts, pending_synthetic, pending_real,
        reservations_by_unit,
    ) = _v4_result_index(
        journal, subject, trajectory_index,
        environment_seed=environment_seed, max_attempts=max_attempts,
    )
    recorded_units = set(reservations_by_unit)
    if any(
        day < 1 or day > contract.days
        or slot < 0 or slot >= contract.society_size
        for day, slot in recorded_units
    ):
        raise Part2CascadingRepairError(
            "Trajectory journal contains an out-of-contract participant unit."
        )
    living = list(range(contract.society_size))
    private = {slot: 0 for slot in range(contract.society_size)}
    reserve = contract.capacity
    group_payoff = 0
    previous_b: int | None = None
    reserve_curve: list[int] = []
    population_curve: list[int] = []
    scheduled = received = invalid = identity_mismatches = transport_failures = 0
    restraint = overuse = 0

    for day in range(1, contract.days + 1):
        if not living:
            if any(recorded_day == day for recorded_day, _slot in recorded_units):
                raise Part2CascadingRepairError(
                    "Trajectory journal dispatches an agent after population collapse."
                )
            reserve_curve.append(0)
            population_curve.append(0)
            continue
        recorded_slots = {
            recorded_slot for recorded_day, recorded_slot in recorded_units
            if recorded_day == day
        }
        if not recorded_slots.issubset(set(living)):
            raise Part2CascadingRepairError(
                "Trajectory journal dispatches a dead or inactive agent."
            )
        population_start = len(living)
        prompts: dict[int, str] = {}
        requests: dict[int, tuple[dict[str, Any], str]] = {}
        for slot in living:
            agent = Agent2(f"slot_{slot:02d}", "inference_hub", str(subject["route"]))
            prompt = agent.build_commons_prompt(
                selfish_gain=contract.private_gain,
                depletion_units=contract.reserve_cost,
                community_benefit=contract.community_benefit,
                day=day,
                living_agents=population_start,
                resource_units=reserve,
                resource_capacity=contract.capacity,
                previous_overuse_count=previous_b,
                cumulative_private_payoff=private[slot],
                cumulative_group_payoff=group_payoff,
            )
            seed = _derive_seed(
                "inference_hub_part2_generation_v1",
                subject["target_id"], environment_seed, day, slot,
            )
            body, _ = runner._request_contract(
                subject, prompt=prompt, system_prompt=agent.system_prompt,
                generation_seed=seed,
            )
            prompts[slot] = prompt
            requests[slot] = (
                body, hashlib.sha256(runner._canonical_bytes(body)).hexdigest()
            )
            retained = results.get((day, slot))
            _validate_dynamic_unit_records(
                subject=subject, trajectory_index=trajectory_index,
                environment_seed=environment_seed, day=day, slot=slot,
                prompt=prompt, system_prompt=agent.system_prompt,
                result=retained,
                reservations=reservations_by_unit.get((day, slot), ()),
            )

        if any(
            _post_preflight_auth_failure(results[(day, slot)])
            for slot in living if (day, slot) in results
        ):
            raise CredentialQualificationInvalidatedError(
                "A retained child result invalidated its preflight-qualified account."
            )

        missing = [slot for slot in living if (day, slot) not in results]
        if missing:
            if any(recorded_day > day for recorded_day, _slot in recorded_units):
                raise Part2CascadingRepairError(
                    "Trajectory journal advanced before an earlier day completed."
                )
            if not dispatch_missing:
                raise _TrajectoryNeedsDispatch(
                    "Validated trajectory has unfinished participant units."
                )
            if pool is None:
                raise Part2CascadingRepairError(
                    "Dispatch was requested without a qualified account pool."
                )
            per_account_concurrency = pool.rate_limit_contract.get(
                "provider_concurrency"
            )
            if (
                not isinstance(per_account_concurrency, int)
                or isinstance(per_account_concurrency, bool)
                or per_account_concurrency < 1
            ):
                raise Part2CascadingRepairError(
                    "Qualified account limiter concurrency is invalid."
                )
            active_workers = min(
                participant_workers,
                len(missing),
                len(pool.qualified_slots) * per_account_concurrency,
            )
            for slot in missing:
                stale = pending_real.get((day, slot))
                if stale is not None:
                    _append_stale_reservation_failure(
                        journal, reservation=stale, subject=subject,
                        trajectory_index=trajectory_index,
                    )
            qualification_error: CredentialQualificationInvalidatedError | None = None
            with ThreadPoolExecutor(
                max_workers=active_workers,
                thread_name_prefix="inference-hub-part2-v4-participant",
            ) as executor:
                future_to_slot = {
                    executor.submit(
                        _dispatch_unit, journal=journal, subject=subject,
                        trajectory_index=trajectory_index,
                        environment_seed=environment_seed, day=day, slot=slot,
                        prompt=prompts[slot],
                        system_prompt=Agent2(
                            f"slot_{slot:02d}", "inference_hub", str(subject["route"])
                        ).system_prompt,
                        prior_attempt=attempts.get((day, slot), 0),
                        max_attempts=max_attempts,
                        initial_backoff_seconds=initial_backoff_seconds,
                        pool=pool, sleep_fn=sleep_fn,
                        pending_synthetic=pending_synthetic.get((day, slot)),
                    ): slot
                    for slot in missing
                }
                for future in as_completed(future_to_slot):
                    slot = future_to_slot[future]
                    try:
                        results[(day, slot)] = future.result()
                    except CredentialQualificationInvalidatedError as error:
                        qualification_error = error
                        for pending in future_to_slot:
                            if pending is not future:
                                pending.cancel()
                        break
            if qualification_error is not None:
                raise qualification_error

        day_rows = [results[(day, slot)] for slot in living]
        actions = [str(row["action"]) for row in day_rows]
        a_count, b_count = actions.count("OPTION_A"), actions.count("OPTION_B")
        invalid_count = actions.count("INVALID")
        for slot, row in zip(living, day_rows):
            private[slot] += (
                1 if row["action"] == "OPTION_A"
                else contract.private_gain if row["action"] == "OPTION_B" else 0
            )
        reserve = max(0, reserve - contract.reserve_cost * b_count)
        if invalid_count == 0 and a_count == population_start:
            group_payoff += contract.community_benefit
        elif invalid_count == 0 and b_count == population_start:
            group_payoff -= contract.community_benefit
        deaths = _collapse_deaths(population_start, reserve, contract.collapse_death_rate)
        dead, _ = runner._matched_attrition(living, deaths, environment_seed, day)
        dead_set = set(dead)
        living = [slot for slot in living if slot not in dead_set]
        previous_b = b_count
        reserve_curve.append(reserve)
        population_curve.append(len(living))
        scheduled += population_start
        received += sum(row.get("raw_response") is not None for row in day_rows)
        invalid += invalid_count
        identity_mismatches += sum(
            row.get("invalid_reason") == "response_model_identity_mismatch"
            for row in day_rows
        )
        transport_failures += sum(
            row.get("invalid_reason") == "transport_failure_exhausted"
            for row in day_rows
        )
        restraint += a_count
        overuse += b_count

    return {
        "schema_version": SCHEMA_VERSION,
        "target_id": subject["target_id"],
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "trajectory_index": trajectory_index,
        "environment_seed_index": trajectory_index,
        "environment_seed": environment_seed,
        "operationally_eligible": identity_mismatches == 0 and transport_failures == 0,
        "scheduled_agent_days": scheduled,
        "responses_received": received,
        "invalid_count": invalid,
        "identity_mismatch_count": identity_mismatches,
        "transport_failure_count": transport_failures,
        "restraint_count": restraint,
        "overuse_count": overuse,
        "restraint_rate": restraint / scheduled if scheduled else 0.0,
        "aurc": sum(reserve_curve) / (contract.capacity * contract.days),
        "aupc": sum(population_curve) / (contract.society_size * contract.days),
        "reserve_nondepletion": min(reserve_curve) > 0,
        "final_reserve": reserve_curve[-1],
        "final_population": len(living),
        "population_retention": len(living) / contract.society_size,
        "cumulative_private_payoff": sum(private.values()),
        "cumulative_group_payoff": group_payoff,
    }


def _strictly_replay_retained_rounds(
    *, journals: Mapping[tuple[str, int, int], Any],
    subject: Mapping[str, Any], trajectory_index: int,
    environment_seed: int, contract: runner.Part2Contract,
    participant_workers: int, max_attempts: int,
    initial_backoff_seconds: float,
) -> tuple[dict[str, Any] | None, ...]:
    """Validate every retained round without credentials or network dispatch."""

    seen_attempt_ids: set[str] = set()
    states: list[dict[str, Any] | None] = []
    saw_empty_or_partial = False
    saw_success = False
    for key in sorted(journals, key=lambda item: item[2]):
        journal = journals[key]
        for row in journal.records:
            if row.get("event") != "reserved_before_dispatch":
                continue
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) or attempt_id in seen_attempt_ids:
                raise Part2CascadingRepairError(
                    "Attempt identifiers are duplicated across child rounds."
                )
            seen_attempt_ids.add(attempt_id)
        if not journal.records:
            saw_empty_or_partial = True
            states.append(None)
            continue
        if saw_empty_or_partial or saw_success:
            raise Part2CascadingRepairError(
                "Cascading repair rounds are nonconsecutive or continue after success."
            )
        try:
            replayed = _run_trajectory(
                subject=subject, trajectory_index=trajectory_index,
                environment_seed=environment_seed, contract=contract,
                journal=journal, pool=None,
                participant_workers=participant_workers,
                max_attempts=max_attempts,
                initial_backoff_seconds=initial_backoff_seconds,
                sleep_fn=lambda _seconds: None,
                dispatch_missing=False,
            )
        except _TrajectoryNeedsDispatch:
            saw_empty_or_partial = True
            states.append(None)
            continue
        states.append(replayed)
        saw_success = replayed["operationally_eligible"] is True
    return tuple(states)


def _runtime_accounts(
    credential_env_file: Path, *, expected_count: int,
    timeout_seconds: float, rate_profile: str, cursor_epoch: str,
) -> tuple[AccountClient, ...]:
    try:
        descriptor = os.open(
            credential_env_file,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise Part2CascadingRepairError(
            "Credential pool must be an available regular nonsymlink 0600 file."
        ) from error
    try:
        file_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(file_status.st_mode)
            or stat.S_IMODE(file_status.st_mode) != 0o600
            or file_status.st_uid != os.geteuid()
        ):
            raise Part2CascadingRepairError(
                "Credential pool must be an owner-held regular 0600 file, not a symlink."
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            values: list[tuple[int, str]] = []
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                match = NUMBERED_BETTER_GOS_KEY.fullmatch(name.strip())
                if match:
                    values.append((
                        int(match.group(1) or 1),
                        value.strip().strip('"').strip("'"),
                    ))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    ambient_base_url = os.environ.get("INFERENCE_HUB_BASE_URL")
    if (
        ambient_base_url is not None
        and ambient_base_url.strip() != runner.DEFAULT_BASE_URL
    ):
        raise Part2CascadingRepairError(
            "Ambient InferenceHub endpoint override differs from the frozen endpoint."
        )
    ordered = sorted(values)
    ordinals = tuple(ordinal for ordinal, value in ordered if value)
    keys = tuple(value for _ordinal, value in ordered if value)
    if (
        ordinals != tuple(range(1, expected_count + 1))
        or len(keys) != expected_count
        or len(set(keys)) != expected_count
    ):
        raise Part2CascadingRepairError(
            "Credential pool does not contain the exact required account set."
        )
    slots = _account_slots(len(keys))
    clients: list[AccountClient] = []
    if rate_profile != runner.HIGH_LATENCY_ORIGINAL_SCALE_RATE_PROFILE:
        raise Part2CascadingRepairError(
            "Credential pool requires the frozen high-latency rate profile."
        )
    policy = RateLimitPolicy(
        global_concurrency=60,
        provider_concurrency=10,
        global_requests_per_second=12.0,
        provider_requests_per_second=2.5,
    )
    for slot, key in zip(slots, keys):
        scope_id = hashlib.sha256(
            (
                f"{runner.DEFAULT_BASE_URL}\0{key}\0{rate_profile}"
            ).encode("utf-8")
        ).hexdigest()
        limiter = InferenceHubRateLimiter(policy=policy, scope_id=scope_id)
        client = InferenceHubClient(
            api_key=key,
            base_url=runner.DEFAULT_BASE_URL,
            timeout_seconds=timeout_seconds,
            rate_limiter=limiter,
        )
        clients.append(AccountClient(
            account_slot=slot,
            client=client,
            account_commitment=_credential_commitment(
                key, cursor_epoch=cursor_epoch, base_url=client.base_url,
            ),
        ))
    limiter_paths = {
        str(account.client.rate_limiter.state_path.resolve())
        for account in clients
    }
    if len(limiter_paths) != len(clients):
        raise Part2CascadingRepairError(
            "Configured accounts do not have independent limiter scopes."
        )
    return tuple(clients)


def _load_parent_chain(source_manifest_path: Path, parent_manifest_path: Path) -> Any:
    # Imported lazily so the read-only validator can refer to this module's
    # schema constants without creating an import cycle.
    from analysis.validate_inference_hub_part2_operational_overlays import (
        validate_incomplete_cascading_parent,
    )

    return validate_incomplete_cascading_parent(
        source_manifest_path, parent_manifest_path,
    )


def _binding(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": runner._sha256_file(path.resolve()),
        "evidence_sha256": manifest["evidence_sha256"],
    }


def _selected_trajectory(
    subject: Mapping[str, Any], *, trajectory_index: int, environment_seed: int,
) -> dict[str, Any]:
    return {
        "target_id": subject["target_id"],
        "trajectory_index": trajectory_index,
        "environment_seed_index": trajectory_index,
        "environment_seed": environment_seed,
        "requested_route": subject["route"],
    }


def _execution_contract(
    *, maximum_rounds: int, trajectory_workers: int, participant_workers: int,
    max_attempts: int, initial_backoff_seconds: float,
    timeout_seconds: float, rate_profile: str,
) -> dict[str, Any]:
    return {
        "strategy": "single_selected_trajectory_parallel_participants_sequential_days",
        "trajectory_workers": trajectory_workers,
        "participant_workers": participant_workers,
        "max_transport_attempts": max_attempts,
        "initial_exponential_backoff_seconds": initial_backoff_seconds,
        "request_timeout_seconds": timeout_seconds,
        "maximum_rounds": maximum_rounds,
        "rate_profile": rate_profile,
        "retry_policy": {
            "connection_failure_codes": ["connection_error", "connection_timeout"],
            "http_statuses": [400, 429, *range(500, 600)],
            "request_mutation_permitted": False,
            "semantic_retry_permitted": False,
        },
        "journal": (
            "per_trajectory_append_only_fsync_sha256_chain_account_bound_"
            "reserve_before_dispatch"
        ),
        "identity_check": "exact_returned_model_equals_selected_route",
        "visible_output_only": True,
    }


def _credential_pool_binding(
    *, accounts: Sequence[AccountClient], qualified_slots: Sequence[str],
    cursor_epoch: str, exact_route: str,
) -> dict[str, Any]:
    qualified_clients = [
        account.client for account in accounts if account.account_slot in qualified_slots
    ]
    if len(qualified_clients) != len(qualified_slots):
        raise Part2CascadingRepairError("Qualified client binding is incomplete.")
    current_path = Path(__file__).resolve()
    old_repair_path = current_path.with_name("inference_hub_part2_operational_repair.py")
    discovery_path = current_path.with_name("inference_hub_discovery.py")
    rate_path = current_path.with_name("inference_hub_rate_limit.py")
    implementation_paths = (
        current_path,
        Path(runner.__file__).resolve(),
        old_repair_path,
        discovery_path,
        rate_path,
    )
    commitments = {
        account.account_slot: account.account_commitment for account in accounts
    }
    clients_by_slot = {
        account.account_slot: account.client for account in accounts
    }
    per_account_limiters = {
        slot: {
            "scope_label": f"{cursor_epoch}::{slot}",
            "state_path_sha256": hashlib.sha256(
                str(clients_by_slot[slot].rate_limiter.state_path.resolve()).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "rate_limit_contract": dict(
                clients_by_slot[slot].rate_limit_contract
            ),
        }
        for slot in qualified_slots
    }
    return {
        "configured_account_count": len(accounts),
        "configured_account_slots": [account.account_slot for account in accounts],
        "account_slot_commitments": commitments,
        "qualified_account_count": len(qualified_slots),
        "qualified_account_slots": list(qualified_slots),
        "rejected_account_slots": [
            account.account_slot
            for account in accounts
            if account.account_slot not in qualified_slots
        ],
        "qualified_set_sha256": _qualified_set_sha256(
            qualified_slots, commitments,
        ),
        "selection_policy": SELECTION_POLICY,
        "rate_limit_scope": RATE_LIMIT_SCOPE,
        "rate_limit_contract": dict(qualified_clients[0].rate_limit_contract),
        "qualified_account_rate_limiters": per_account_limiters,
        "exact_route": exact_route,
        "base_url": qualified_clients[0].base_url,
        "cursor_epoch": cursor_epoch,
        "cursor_initial_global_dispatch_ordinal": 0,
        "implementation_files": {
            str(path): runner._sha256_file(path) for path in implementation_paths
        },
    }


def _composition_contract() -> dict[str, Any]:
    return {
        "policy": "verified_parent_successes_plus_current_unresolved_only",
        "parent_maximum_rounds": DEFAULT_MAXIMUM_ROUNDS,
        "child_maximum_rounds": DEFAULT_MAXIMUM_ROUNDS,
        "cumulative_maximum_rounds_per_selected_trajectory": (
            2 * DEFAULT_MAXIMUM_ROUNDS
        ),
        "inherited_parent_success_count": 37,
        "selected_parent_unresolved_count": 1,
        "restart_scope": "whole_trajectory_from_day_one",
        "semantic_only_selection_forbidden": True,
    }


def _validate_exact_execution_arguments(
    *, maximum_rounds: int, trajectory_workers: int, participant_workers: int,
    max_attempts: int, initial_backoff_seconds: float,
    timeout_seconds: float, rate_profile: str, expected_account_count: int,
) -> None:
    observed = (
        maximum_rounds, trajectory_workers, participant_workers, max_attempts,
        initial_backoff_seconds, timeout_seconds, rate_profile,
        expected_account_count,
    )
    expected = (
        DEFAULT_MAXIMUM_ROUNDS, DEFAULT_TRAJECTORY_WORKERS,
        DEFAULT_PARTICIPANT_WORKERS, DEFAULT_MAX_ATTEMPTS,
        DEFAULT_BACKOFF_SECONDS, DEFAULT_TIMEOUT_SECONDS,
        DEFAULT_RATE_PROFILE, EXPECTED_CONFIGURED_ACCOUNT_COUNT,
    )
    if observed != expected:
        raise Part2CascadingRepairError(
            "Cascading repair execution arguments differ from the frozen v4 design."
        )


def _read_secure_json_0600(
    path: Path, *, label: str,
) -> tuple[dict[str, Any], str]:
    """Read and hash one regular 0600 JSON file through a held nofollow fd."""

    try:
        path_status = path.lstat()
    except OSError as error:
        raise Part2CascadingRepairError(f"{label} is unavailable.") from error
    if (
        stat.S_ISLNK(path_status.st_mode)
        or not stat.S_ISREG(path_status.st_mode)
        or stat.S_IMODE(path_status.st_mode) != 0o600
    ):
        raise Part2CascadingRepairError(
            f"{label} must be a regular nonsymlink 0600 file."
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise Part2CascadingRepairError(
            f"{label} cannot be opened without following symlinks."
        ) from error
    try:
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or stat.S_IMODE(descriptor_status.st_mode) != 0o600
            or (descriptor_status.st_dev, descriptor_status.st_ino)
            != (path_status.st_dev, path_status.st_ino)
        ):
            raise Part2CascadingRepairError(
                f"{label} changed while it was being opened."
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
    finally:
        os.close(descriptor)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Part2CascadingRepairError(f"{label} is not valid JSON.") from error
    if not isinstance(parsed, dict):
        raise Part2CascadingRepairError(f"{label} is not a JSON object.")
    return parsed, hashlib.sha256(raw).hexdigest()


def _read_bound_preflight(
    manifest_path: Path, reference: object, *, exact_route: str,
    session_index: int,
) -> dict[str, Any]:
    if not isinstance(reference, Mapping) or set(reference) != {
        "path", "file_sha256", "evidence_sha256"
    }:
        raise Part2CascadingRepairError("Preflight ledger reference schema changed.")
    expected_path = (
        manifest_path.parent / "preflight"
        / f"preflight-{session_index:03d}.json"
    )
    if Path(str(reference.get("path", ""))).resolve() != expected_path.resolve():
        raise Part2CascadingRepairError("Preflight ledger path binding changed.")
    ledger, file_sha256 = _read_secure_json_0600(
        expected_path, label="Bound preflight ledger"
    )
    if (
        reference.get("file_sha256") != file_sha256
        or reference.get("evidence_sha256") != ledger.get("evidence_sha256")
    ):
        raise Part2CascadingRepairError("Preflight ledger hashes changed.")
    _validate_preflight_ledger(ledger, exact_route=exact_route)
    return ledger


JOURNAL_REFERENCE_KEYS = frozenset(
    {"path", "record_count", "tail_record_sha256", "file_sha256"}
)


def _validate_child_journal_reference(
    journal: _CascadingJournal, reference: object, *, label: str,
) -> None:
    """Bind an exact child path/schema before accepting a resumable suffix."""

    if not isinstance(reference, Mapping) or set(reference) != JOURNAL_REFERENCE_KEYS:
        raise Part2CascadingRepairError(
            f"Manifest {label} reference schema changed."
        )
    expected_path = str(journal.path.resolve())
    if reference.get("path") != expected_path:
        raise Part2CascadingRepairError(
            f"Manifest {label} path binding changed."
        )
    runner._validate_checkpoint_reference(journal, reference, label=label)


def _later_recovery_session_index(
    authorizations: Sequence[Mapping[str, Any]], *,
    reservation_session_index: int, terminal_at: datetime,
) -> int | None:
    for recovery_index in range(
        reservation_session_index + 1, len(authorizations)
    ):
        if terminal_at < _utc_datetime(
            authorizations[recovery_index]["completed_at_utc"],
            label="Crash-recovery preflight completion",
        ):
            continue
        if (
            recovery_index + 1 < len(authorizations)
            and terminal_at > _utc_datetime(
                authorizations[recovery_index + 1]["created_at_utc"],
                label="Next crash-recovery preflight start",
            )
        ):
            continue
        return recovery_index
    return None


def _max_dispatch_ordinal(
    journals: Mapping[tuple[str, int, int], Any], *, cursor_epoch: str,
    qualified_slots: Sequence[str], preflight_ledger: Mapping[str, Any],
    child_created_at: datetime, latest_session_index: int,
    latest_resume_at: datetime | None,
) -> int:
    authorizations = _preflight_authorizations(preflight_ledger)
    authorization_by_index = {
        int(item["session_index"]): item for item in authorizations
    }
    observed: dict[int, str] = {}
    reservations: dict[str, Mapping[str, Any]] = {}
    last_observed_ordinal = 0
    for key in sorted(journals, key=lambda item: item[2]):
        for row in journals[key].records:
            event = row.get("event")
            if event != "reserved_before_dispatch":
                reservation = reservations.get(str(row.get("attempt_id")))
                if reservation is None:
                    raise Part2CascadingRepairError(
                        "Retained terminal event lacks a prior reservation."
                    )
                terminal_at = _utc_datetime(
                    row.get("completed_at_utc"), label="Attempt terminal timestamp"
                )
                reservation_session_index = int(
                    reservation["preflight_session_index"]
                )
                failure = row.get("failure")
                stale = (
                    event == "attempt_failed"
                    and isinstance(failure, Mapping)
                    and failure.get("failure_code") == "stale_reserved_attempt"
                )
                if stale:
                    recovery_index = _later_recovery_session_index(
                        authorizations,
                        reservation_session_index=reservation_session_index,
                        terminal_at=terminal_at,
                    )
                    if recovery_index is None:
                        raise Part2CascadingRepairError(
                            "Retained stale reservation lacks a sealed recovery preflight."
                        )
                    if (
                        recovery_index == latest_session_index
                        and (
                            latest_resume_at is None
                            or terminal_at < latest_resume_at
                        )
                    ):
                        raise Part2CascadingRepairError(
                            "Retained stale recovery predates its manifest checkpoint."
                        )
                    continue
                crosses_boundary = (
                    reservation_session_index + 1 < len(authorizations)
                    and terminal_at > _utc_datetime(
                        authorizations[reservation_session_index + 1][
                            "created_at_utc"
                        ],
                        label="Next preflight session start",
                    )
                )
                if crosses_boundary:
                    recovery_index = _later_recovery_session_index(
                        authorizations,
                        reservation_session_index=reservation_session_index,
                        terminal_at=terminal_at,
                    )
                    synthetic_recovery = (
                        event == "semantic_result"
                        and reservation.get("dispatch_skipped") is True
                        and row.get("raw_response") is None
                        and isinstance(failure, Mapping)
                        and failure.get("failure_code")
                        == "resume_attempt_budget_exhausted"
                        and recovery_index is not None
                    )
                    if not synthetic_recovery:
                        raise Part2CascadingRepairError(
                            "Retained terminal event crosses a preflight boundary."
                        )
                    if (
                        recovery_index == latest_session_index
                        and (
                            latest_resume_at is None
                            or terminal_at < latest_resume_at
                        )
                    ):
                        raise Part2CascadingRepairError(
                            "Retained synthetic recovery predates its manifest checkpoint."
                        )
                continue
            ordinal = row.get("global_dispatch_ordinal")
            slot = row.get("account_slot")
            session_index = row.get("preflight_session_index")
            session = authorization_by_index.get(session_index)
            reserved_at = _utc_datetime(
                row.get("reserved_at_utc"), label="Dispatch reservation"
            )
            if (
                not isinstance(session_index, int)
                or isinstance(session_index, bool)
                or session is None
                or row.get("preflight_session_sha256")
                != session["session_sha256"]
                or reserved_at < _utc_datetime(
                    session["completed_at_utc"],
                    label="Dispatch-authorizing preflight completion",
                )
            ):
                raise Part2CascadingRepairError(
                    "Retained dispatch lacks its exact preflight-session authorization."
                )
            if reserved_at < child_created_at:
                raise Part2CascadingRepairError(
                    "Retained dispatch predates child manifest creation."
                )
            if (
                latest_session_index > 0
                and session_index == latest_session_index
                and (
                    latest_resume_at is None
                    or reserved_at < latest_resume_at
                )
            ):
                raise Part2CascadingRepairError(
                    "Retained dispatch predates its resume manifest checkpoint."
                )
            if int(session_index) + 1 < len(authorizations):
                next_session = authorizations[int(session_index) + 1]
                if reserved_at > _utc_datetime(
                    next_session["created_at_utc"],
                    label="Next preflight session start",
                ):
                    raise Part2CascadingRepairError(
                        "Retained dispatch crosses its preflight-session boundary."
                    )
            if row.get("dispatch_skipped") is True:
                if (
                    ordinal is not None or slot is not None
                    or row.get("cursor_epoch") != cursor_epoch
                ):
                    raise Part2CascadingRepairError(
                        "Skipped reservation fabricated physical dispatch provenance."
                    )
                reservations[str(row["attempt_id"])] = row
                continue
            if (
                not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 1
                or row.get("cursor_epoch") != cursor_epoch
                or slot not in qualified_slots
                or slot != qualified_slots[(ordinal - 1) % len(qualified_slots)]
                or ordinal in observed
                or ordinal != last_observed_ordinal + 1
            ):
                raise Part2CascadingRepairError(
                    "Retained account dispatch provenance is invalid."
                )
            expected_session = authorizations[0]
            for candidate in authorizations[1:]:
                if int(candidate["starting_global_dispatch_ordinal"]) < ordinal:
                    expected_session = candidate
            if session != expected_session:
                raise Part2CascadingRepairError(
                    "Retained dispatch is assigned to the wrong preflight session."
                )
            observed[ordinal] = str(slot)
            last_observed_ordinal = ordinal
            reservations[str(row["attempt_id"])] = row
    if observed and set(observed) != set(range(1, max(observed) + 1)):
        raise Part2CascadingRepairError("Global dispatch ordinals are not consecutive.")
    maximum = max(observed, default=0)
    if any(
        int(session["starting_global_dispatch_ordinal"]) > maximum
        for session in authorizations
    ):
        raise Part2CascadingRepairError(
            "A preflight session claims a future dispatch boundary."
        )
    return maximum


@contextmanager
def _hold_lineage_locks(paths: Sequence[Path]):
    resolved = sorted({path.resolve() for path in paths})
    handles: list[Any] = []
    try:
        for manifest_path in resolved:
            if manifest_path.name != "manifest.json" or manifest_path.parent.name != "private":
                raise Part2CascadingRepairError(
                    "Source and parent inputs must be private/manifest.json files."
                )
            lock_path = manifest_path.parent / ".run.lock"
            if not manifest_path.is_file() or not lock_path.is_file():
                raise Part2CascadingRepairError(
                    "Source or parent lineage lacks its immutable run lock."
                )
            runner._require_mode(manifest_path, 0o600)
            runner._require_mode(lock_path, 0o600)
            handle = lock_path.open("rb")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise Part2CascadingRepairError(
                    "Source or parent lineage is active."
                ) from error
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _require_nonsymlink_directory(path: Path, *, label: str) -> None:
    try:
        status = path.lstat()
    except OSError as error:
        raise Part2CascadingRepairError(f"{label} is unavailable.") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise Part2CascadingRepairError(
            f"{label} must be a nonsymlink directory."
        )


def _acquire_secure_run_lock(private_dir: Path):
    path = private_dir / ".run.lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise Part2CascadingRepairError(
            "Cascading run lock cannot be opened safely."
        ) from error
    try:
        _CascadingJournal._require_regular_descriptor(
            descriptor, label="Cascading run lock"
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise Part2CascadingRepairError(
            "Another process is already running this cascading output."
        ) from error
    except BaseException:
        os.close(descriptor)
        raise
    return os.fdopen(descriptor, "a+b")


def _load_safe_uncommitted_bootstrap(
    output_dir: Path, *, exact_route: str,
) -> dict[str, Any] | None:
    """Accept only the canonical state possible before the first manifest seal."""

    root = Path(os.path.abspath(output_dir))
    _require_nonsymlink_directory(root, label="Cascading output root")
    if root.resolve() != root:
        raise Part2CascadingRepairError(
            "Cascading output root traverses a symlink."
        )

    private_dir = root / "private"
    sanitized_dir = root / "sanitized"
    allowed_root = {"private", "sanitized"}
    if {entry.name for entry in root.iterdir()} - allowed_root:
        raise Part2CascadingRepairError(
            "Existing cascading output is not an uncommitted bootstrap."
        )
    for directory, label in (
        (private_dir, "Cascading private bootstrap directory"),
        (sanitized_dir, "Cascading sanitized bootstrap directory"),
    ):
        if directory.exists() or directory.is_symlink():
            _require_nonsymlink_directory(directory, label=label)

    if sanitized_dir.exists() and any(sanitized_dir.iterdir()):
        raise Part2CascadingRepairError(
            "Uncommitted cascading bootstrap contains sanitized artifacts."
        )
    if not private_dir.exists():
        return None

    allowed_private = {".run.lock", "trajectories", "preflight"}
    if {entry.name for entry in private_dir.iterdir()} - allowed_private:
        raise Part2CascadingRepairError(
            "Existing cascading private state is already committed or unexpected."
        )
    lock_path = private_dir / ".run.lock"
    if lock_path.exists() or lock_path.is_symlink():
        try:
            lock_status = lock_path.lstat()
        except OSError as error:
            raise Part2CascadingRepairError(
                "Cascading bootstrap lock is unavailable."
            ) from error
        if (
            stat.S_ISLNK(lock_status.st_mode)
            or not stat.S_ISREG(lock_status.st_mode)
            or stat.S_IMODE(lock_status.st_mode) != 0o600
        ):
            raise Part2CascadingRepairError(
                "Cascading bootstrap lock must be a regular nonsymlink 0600 file."
            )

    trajectory_dir = private_dir / "trajectories"
    if trajectory_dir.exists() or trajectory_dir.is_symlink():
        _require_nonsymlink_directory(
            trajectory_dir, label="Cascading bootstrap trajectory directory"
        )
        allowed_trajectory = {runner._safe_file_stem(EXPECTED_TARGET_ID)}
        if {entry.name for entry in trajectory_dir.iterdir()} - allowed_trajectory:
            raise Part2CascadingRepairError(
                "Uncommitted cascading bootstrap contains unexpected trajectories."
            )
        target_dir = trajectory_dir / runner._safe_file_stem(EXPECTED_TARGET_ID)
        if target_dir.exists() or target_dir.is_symlink():
            _require_nonsymlink_directory(
                target_dir, label="Cascading bootstrap target directory"
            )
            if any(target_dir.iterdir()):
                raise Part2CascadingRepairError(
                    "Uncommitted cascading bootstrap contains journal evidence."
                )

    preflight_dir = private_dir / "preflight"
    if not (preflight_dir.exists() or preflight_dir.is_symlink()):
        return None
    _require_nonsymlink_directory(
        preflight_dir, label="Cascading bootstrap preflight directory"
    )
    entries = list(preflight_dir.iterdir())
    if not entries:
        return None
    expected_path = preflight_dir / "preflight-000.json"
    if len(entries) != 1 or entries[0].name != expected_path.name:
        raise Part2CascadingRepairError(
            "Uncommitted cascading bootstrap has unexpected preflight state."
        )
    ledger, _ = _read_secure_json_0600(
        expected_path, label="Bootstrap preflight ledger"
    )
    _validate_preflight_ledger(ledger, exact_route=exact_route)
    if ledger["resume_preflight_sessions"]:
        raise Part2CascadingRepairError(
            "Bootstrap preflight ledger fabricated resume sessions."
        )
    return ledger


def _run_cascading_repair_locked(
    *, source_manifest_path: Path, parent_overlay_manifest_path: Path,
    output_dir: Path, credential_env_file: Path,
    maximum_rounds: int = DEFAULT_MAXIMUM_ROUNDS,
    trajectory_workers: int = DEFAULT_TRAJECTORY_WORKERS,
    participant_workers: int = DEFAULT_PARTICIPANT_WORKERS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    rate_profile: str = DEFAULT_RATE_PROFILE,
    expected_account_count: int = EXPECTED_CONFIGURED_ACCOUNT_COUNT,
    resume: bool = False,
    sleep_fn: Callable[[float], None] = runner.time.sleep,
) -> dict[str, Any]:
    """Run the one-trajectory child repair after validating its full lineage."""

    _validate_exact_execution_arguments(
        maximum_rounds=maximum_rounds,
        trajectory_workers=trajectory_workers,
        participant_workers=participant_workers,
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff_seconds,
        timeout_seconds=timeout_seconds,
        rate_profile=rate_profile,
        expected_account_count=expected_account_count,
    )
    source_path = source_manifest_path.resolve()
    parent_path = parent_overlay_manifest_path.resolve()
    chain = _load_parent_chain(source_path, parent_path)
    source = chain.source
    parent_manifest = chain.parent_manifest
    unresolved = dict(chain.unresolved)
    expected_key = (EXPECTED_TARGET_ID, EXPECTED_TRAJECTORY_INDEX)
    if (
        len(source.subjects) != 21
        or len(chain.source_failures) != 38
        or len(chain.successful) != 37
        or set(unresolved) != {expected_key}
        or parent_manifest.get("summary")
        != {
            "source_operational_failure_trajectories": 38,
            "operational_repairs_succeeded": 37,
            "operational_repairs_unresolved": 1,
        }
    ):
        raise Part2CascadingRepairError(
            "Parent overlay is not the exact 37-success/one-Gemini-unresolved lineage."
        )
    subject_by_id = {str(row["target_id"]): row for row in source.subjects}
    subject = subject_by_id[EXPECTED_TARGET_ID]
    environment_seed = source.environment_seeds[EXPECTED_TRAJECTORY_INDEX]
    if (
        environment_seed != EXPECTED_ENVIRONMENT_SEED
        or unresolved[expected_key].get("environment_seed") != environment_seed
        or unresolved[expected_key].get("transport_failure_count", 0) < 1
    ):
        raise Part2CascadingRepairError("The exact unresolved trajectory binding changed.")
    selected = _selected_trajectory(
        subject, trajectory_index=EXPECTED_TRAJECTORY_INDEX,
        environment_seed=environment_seed,
    )
    frozen_static_bindings = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "source_manifest": _binding(source_path, source.manifest),
        "parent_overlay_manifest": _binding(parent_path, parent_manifest),
        "panel_id": source.manifest["panel_id"],
        "base_seed": source.manifest["base_seed"],
        "part2_contract": source.manifest["part2_contract"],
        "common_environment_seeds": list(source.environment_seeds),
        "subject_routes": list(source.subjects),
        "repair_policy": REPAIR_POLICY,
        "selected_trajectory": selected,
        "composition_contract": _composition_contract(),
        "maximum_rounds": maximum_rounds,
        "execution_contract": _execution_contract(
            maximum_rounds=maximum_rounds,
            trajectory_workers=trajectory_workers,
            participant_workers=participant_workers,
            max_attempts=max_attempts,
            initial_backoff_seconds=initial_backoff_seconds,
            timeout_seconds=timeout_seconds,
            rate_profile=rate_profile,
        ),
    }

    bootstrap_ledger: dict[str, Any] | None = None
    try:
        output_status = output_dir.lstat()
    except FileNotFoundError:
        output_status = None
    except OSError as error:
        raise Part2CascadingRepairError(
            "Cascading repair output cannot be inspected safely."
        ) from error
    if output_status is not None:
        if stat.S_ISLNK(output_status.st_mode) or not stat.S_ISDIR(
            output_status.st_mode
        ):
            raise Part2CascadingRepairError(
                "Cascading repair output must be a nonsymlink directory."
            )
        if not resume:
            bootstrap_ledger = _load_safe_uncommitted_bootstrap(
                output_dir, exact_route=str(subject["route"]),
            )
    private_dir = output_dir / "private"
    journal_dir = private_dir / "trajectories"
    preflight_dir = private_dir / "preflight"
    sanitized_dir = output_dir / "sanitized"
    for directory in (
        output_dir, private_dir, journal_dir, preflight_dir, sanitized_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _require_nonsymlink_directory(
            directory, label="Cascading output directory"
        )
        runner._secure_mode(directory, 0o700)
    if Path(os.path.abspath(output_dir)).resolve() != Path(
        os.path.abspath(output_dir)
    ):
        raise Part2CascadingRepairError(
            "Cascading output tree traverses a symlink."
        )
    repair_lock = _acquire_secure_run_lock(private_dir)
    try:
        manifest_path = private_dir / "manifest.json"
        if resume:
            manifest, _ = _read_secure_json_0600(
                manifest_path, label="Cascading repair manifest"
            )
            if (
                set(manifest) - MANIFEST_OPTIONAL_KEYS != MANIFEST_REQUIRED_KEYS
                or manifest.get("schema_version") != SCHEMA_VERSION
                or manifest.get("artifact_type") != ARTIFACT_TYPE
                or manifest.get("evidence_sha256") != runner._self_hash(manifest)
            ):
                raise Part2CascadingRepairError(
                    "Cascading repair manifest schema or self-hash changed."
                )
            if manifest.get("complete") is True:
                raise Part2CascadingRepairError(
                    "A completed cascading repair is immutable and cannot resume."
                )
            if manifest.get("complete") is not False or "completed_at_utc" in manifest:
                raise Part2CascadingRepairError(
                    "Only an exactly incomplete cascading repair can resume."
                )
            created_at = _utc_datetime(
                manifest.get("created_at_utc"), label="Cascading creation"
            )
            updated_at = _utc_datetime(
                manifest.get("last_updated_at_utc"), label="Cascading update"
            )
            if created_at > updated_at:
                raise Part2CascadingRepairError(
                    "Cascading manifest timestamps are out of order."
                )
            prior_resume_count = manifest.get("resume_count", 0)
            if (
                not isinstance(prior_resume_count, int)
                or isinstance(prior_resume_count, bool)
                or prior_resume_count < 0
                or (prior_resume_count == 0) != ("last_resumed_at_utc" not in manifest)
            ):
                raise Part2CascadingRepairError(
                    "Cascading resume metadata is inconsistent."
                )
            last_resumed_at: datetime | None = None
            if prior_resume_count:
                last_resumed_at = _utc_datetime(
                    manifest.get("last_resumed_at_utc"),
                    label="Cascading last resume",
                )
                if not created_at <= last_resumed_at <= updated_at:
                    raise Part2CascadingRepairError(
                        "Cascading resume timestamps are out of order."
                    )
            if any(
                manifest.get(field) != expected
                for field, expected in frozen_static_bindings.items()
            ):
                raise Part2CascadingRepairError(
                    "Cascading repair immutable bindings changed before resume."
                )
            ledger = _read_bound_preflight(
                manifest_path, manifest.get("preflight_ledger"),
                exact_route=str(subject["route"]),
                session_index=prior_resume_count,
            )
            cursor_epoch, qualified_slots = _validate_preflight_ledger(
                ledger, exact_route=str(subject["route"]),
            )
            if len(ledger["resume_preflight_sessions"]) != prior_resume_count:
                raise Part2CascadingRepairError(
                    "Resume count does not match sealed preflight sessions."
                )
            ledger_completed_at = _utc_datetime(
                ledger.get("completed_at_utc"), label="Bound preflight completion"
            )
            if any(
                _utc_datetime(
                    session.get("created_at_utc"),
                    label="Resume preflight session start",
                ) < created_at
                for session in ledger["resume_preflight_sessions"]
            ):
                raise Part2CascadingRepairError(
                    "Resume preflight session predates child creation."
                )
            if prior_resume_count:
                if ledger_completed_at > last_resumed_at:
                    raise Part2CascadingRepairError(
                        "Bound resume preflight postdates its manifest checkpoint."
                    )
            elif ledger_completed_at > created_at:
                raise Part2CascadingRepairError(
                    "Initial preflight postdates cascading campaign creation."
                )
        else:
            if bootstrap_ledger is None:
                cursor_epoch = f"cursor-epoch-{uuid.uuid4().hex}"
            else:
                cursor_epoch, qualified_slots = _validate_preflight_ledger(
                    bootstrap_ledger, exact_route=str(subject["route"]),
                )
            prior_resume_count = 0

        journal_key = (EXPECTED_TARGET_ID, EXPECTED_TRAJECTORY_INDEX)
        journals: dict[tuple[str, int, int], Any] = {}
        for round_index in range(1, maximum_rounds + 1):
            path = (
                journal_dir / runner._safe_file_stem(EXPECTED_TARGET_ID)
                / f"seed-{EXPECTED_TRAJECTORY_INDEX:03d}-round-{round_index:02d}.jsonl"
            )
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            runner._secure_mode(path.parent, 0o700)
            journals[(*journal_key, round_index)] = _CascadingJournal(
                path, output_root=output_dir,
            )

        if resume:
            refs = manifest.get("journals")
            expected_ref_keys = {
                f"{EXPECTED_TARGET_ID}::{EXPECTED_TRAJECTORY_INDEX}::{round_index}"
                for round_index in range(1, maximum_rounds + 1)
            }
            if not isinstance(refs, Mapping) or set(refs) != expected_ref_keys:
                raise Part2CascadingRepairError("Cascading journal schedule changed.")
            for key, journal in journals.items():
                _validate_child_journal_reference(
                    journal, refs[f"{key[0]}::{key[1]}::{key[2]}"],
                    label=f"cascading trajectory {key}",
                )
            starting_ordinal = _max_dispatch_ordinal(
                journals, cursor_epoch=cursor_epoch,
                qualified_slots=qualified_slots,
                preflight_ledger=ledger,
                child_created_at=created_at,
                latest_session_index=prior_resume_count,
                latest_resume_at=last_resumed_at,
            )
            retained_rounds = _strictly_replay_retained_rounds(
                journals=journals, subject=subject,
                trajectory_index=EXPECTED_TRAJECTORY_INDEX,
                environment_seed=environment_seed, contract=source.contract,
                participant_workers=participant_workers,
                max_attempts=max_attempts,
                initial_backoff_seconds=initial_backoff_seconds,
            )
            if (
                all(state is not None for state in retained_rounds)
                and not any(
                    state is not None
                    and state["operationally_eligible"] is True
                    for state in retained_rounds
                )
            ):
                raise Part2CascadingRepairError(
                    "All cascading repair rounds are terminally exhausted."
                )
            if (
                manifest.get("summary") != {}
                or manifest.get("sanitized_artifacts") != {}
            ):
                raise Part2CascadingRepairError(
                    "A resumable cascading repair contains finalized aggregates."
                )
        else:
            starting_ordinal = 0

        accounts = _runtime_accounts(
            credential_env_file, expected_count=expected_account_count,
            timeout_seconds=timeout_seconds, rate_profile=rate_profile,
            cursor_epoch=cursor_epoch,
        )
        if resume:
            bound_commitments = {
                str(result["account_slot"]): str(result["account_commitment"])
                for result in ledger["results"]
            }
            current_commitments = {
                account.account_slot: account.account_commitment
                for account in accounts
            }
            if (
                current_commitments != bound_commitments
                or [account.account_slot for account in accounts]
                != ledger["configured_account_slots"]
                or {account.client.base_url for account in accounts}
                != {ledger["base_url"]}
            ):
                raise Part2CascadingRepairError(
                    "Resume credentials or endpoint changed before blind preflight."
                )
            credential_pool_binding = _credential_pool_binding(
                accounts=accounts, qualified_slots=qualified_slots,
                cursor_epoch=cursor_epoch, exact_route=str(subject["route"]),
            )
            if manifest.get("credential_pool") != credential_pool_binding:
                raise Part2CascadingRepairError(
                    "Credential, limiter, endpoint, or implementation binding changed "
                    "before blind preflight."
                )
            fresh = blind_preflight_accounts(
                accounts, exact_route=str(subject["route"]),
                cursor_epoch=cursor_epoch,
            )
            ledger = _append_resume_preflight(
                ledger, fresh, exact_route=str(subject["route"]),
                starting_global_dispatch_ordinal=starting_ordinal,
            )
            next_resume_count = prior_resume_count + 1
            ledger_path = (
                preflight_dir / f"preflight-{next_resume_count:03d}.json"
            )
            runner._atomic_json(ledger_path, ledger)
            manifest["resume_count"] = next_resume_count
            manifest["last_resumed_at_utc"] = runner._utc_now()
        elif bootstrap_ledger is not None:
            bound_commitments = {
                str(result["account_slot"]): str(result["account_commitment"])
                for result in bootstrap_ledger["results"]
            }
            current_commitments = {
                account.account_slot: account.account_commitment
                for account in accounts
            }
            if (
                current_commitments != bound_commitments
                or [account.account_slot for account in accounts]
                != bootstrap_ledger["configured_account_slots"]
                or {account.client.base_url for account in accounts}
                != {bootstrap_ledger["base_url"]}
            ):
                raise Part2CascadingRepairError(
                    "Bootstrap credentials or endpoint changed after blind preflight."
                )
            # No experiment dispatch can precede the initial manifest.  Treat
            # a lone sealed ledger only as a crash-recovery cursor/account
            # binding, then requalify all accounts so stale route capability
            # can never authorize the first dispatch.
            ledger = blind_preflight_accounts(
                accounts, exact_route=str(subject["route"]),
                cursor_epoch=cursor_epoch,
            )
            _validate_preflight_ledger(ledger, exact_route=str(subject["route"]))
            ledger_path = preflight_dir / "preflight-000.json"
            runner._atomic_json(ledger_path, ledger)
            qualified_slots = tuple(ledger["qualified_account_slots"])
            credential_pool_binding = _credential_pool_binding(
                accounts=accounts, qualified_slots=qualified_slots,
                cursor_epoch=cursor_epoch, exact_route=str(subject["route"]),
            )
        else:
            ledger = blind_preflight_accounts(
                accounts, exact_route=str(subject["route"]),
                cursor_epoch=cursor_epoch,
            )
            _validate_preflight_ledger(ledger, exact_route=str(subject["route"]))
            ledger_path = preflight_dir / "preflight-000.json"
            runner._atomic_json(ledger_path, ledger)
            qualified_slots = tuple(ledger["qualified_account_slots"])
            credential_pool_binding = _credential_pool_binding(
                accounts=accounts, qualified_slots=qualified_slots,
                cursor_epoch=cursor_epoch, exact_route=str(subject["route"]),
            )

        preflight_ref = {
            "path": str(ledger_path.resolve()),
            "file_sha256": runner._sha256_file(ledger_path),
            "evidence_sha256": ledger["evidence_sha256"],
        }
        if resume:
            manifest["preflight_ledger"] = preflight_ref
        bindings = {
            **frozen_static_bindings,
            "credential_pool": credential_pool_binding,
            "preflight_ledger": preflight_ref,
        }
        if resume:
            mutable = {
                "created_at_utc", "last_updated_at_utc", "completed_at_utc",
                "complete", "summary", "journals", "sanitized_artifacts",
                "evidence_sha256", "resume_count", "last_resumed_at_utc",
            }
            if {k: v for k, v in manifest.items() if k not in mutable} != bindings:
                raise Part2CascadingRepairError(
                    "Cascading repair bindings changed on resume."
                )
            # Commit the newly sealed preflight session and resume boundary
            # before any additional dispatch can change a journal.  A crash
            # after this point therefore leaves a manifest whose journal and
            # preflight references describe the same resumable epoch.
            manifest["last_updated_at_utc"] = runner._utc_now()
            _assert_no_secret_like_fields(
                manifest, label="cascading resume checkpoint"
            )
            runner._seal(manifest)
            runner._atomic_json(manifest_path, manifest)
        else:
            now = runner._utc_now()
            manifest = {
                **bindings,
                "created_at_utc": now,
                "last_updated_at_utc": now,
                "complete": False,
                "summary": {},
                "journals": {
                    f"{key[0]}::{key[1]}::{key[2]}": journal.reference()
                    for key, journal in journals.items()
                },
                "sanitized_artifacts": {},
            }
            _assert_no_secret_like_fields(manifest, label="cascading manifest")
            runner._seal(manifest)
            runner._atomic_json(manifest_path, manifest)

        active_session = (
            ledger["resume_preflight_sessions"][-1]
            if ledger["resume_preflight_sessions"] else {
                "session_index": 0,
                "session_sha256": ledger["initial_session_sha256"],
            }
        )
        pool = QualifiedAccountPool(
            accounts, qualified_slots=qualified_slots, cursor_epoch=cursor_epoch,
            preflight_session_index=int(active_session["session_index"]),
            preflight_session_sha256=str(active_session["session_sha256"]),
            start_ordinal=starting_ordinal,
        )
        successful: tuple[int, Mapping[str, Any]] | None = None
        seen_empty = False
        for round_index in range(1, maximum_rounds + 1):
            journal = journals[(*journal_key, round_index)]
            if not journal.records:
                seen_empty = True
            elif seen_empty:
                raise Part2CascadingRepairError(
                    "Cascading repair rounds are nonconsecutive."
                )
            if successful is not None:
                if journal.records:
                    raise Part2CascadingRepairError(
                        "Cascading repair continued after trajectory success."
                    )
                continue
            replayed = _run_trajectory(
                subject=subject,
                trajectory_index=EXPECTED_TRAJECTORY_INDEX,
                environment_seed=environment_seed,
                contract=source.contract,
                journal=journal,
                pool=pool,
                participant_workers=participant_workers,
                max_attempts=max_attempts,
                initial_backoff_seconds=initial_backoff_seconds,
                sleep_fn=sleep_fn,
            )
            seen_empty = False
            if replayed["operationally_eligible"] is True:
                successful = (round_index, replayed)
            manifest["journals"] = {
                f"{key[0]}::{key[1]}::{key[2]}": item.reference()
                for key, item in journals.items()
            }
            manifest["last_updated_at_utc"] = runner._utc_now()
            runner._seal(manifest)
            runner._atomic_json(manifest_path, manifest)

        effective_rows: list[dict[str, Any]] = []
        for parent_row in chain.effective_trajectories:
            key = (str(parent_row["target_id"]), int(parent_row["trajectory_index"]))
            if key == expected_key and successful is not None:
                replacement = dict(successful[1])
                replacement["operational_repair_round"] = successful[0]
                replacement["source_replaced_for_operational_failure"] = True
                effective_rows.append(replacement)
            else:
                effective_rows.append(dict(parent_row))
        effective_models = runner._aggregate_models(
            effective_rows, source.subjects,
            expected_trajectories=source.contract.trajectories,
            capacity=source.contract.capacity,
        )
        generated_at = runner._utc_now()
        trajectory_payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": TRAJECTORY_ARTIFACT_TYPE,
            "panel_id": source.manifest["panel_id"],
            "generated_at_utc": generated_at,
            "source_manifest_evidence_sha256": source.manifest["evidence_sha256"],
            "rows": effective_rows,
        }
        model_payload = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": MODEL_ARTIFACT_TYPE,
            "panel_id": source.manifest["panel_id"],
            "generated_at_utc": generated_at,
            "source_manifest_evidence_sha256": source.manifest["evidence_sha256"],
            "rows": effective_models,
        }
        _assert_no_secret_like_fields(
            trajectory_payload, label="cascading trajectory metrics"
        )
        _assert_no_secret_like_fields(model_payload, label="cascading model metrics")
        runner._seal(trajectory_payload)
        runner._seal(model_payload)
        trajectory_path = sanitized_dir / "effective_trajectory_metrics.json"
        model_path = sanitized_dir / "effective_model_metrics.json"
        runner._atomic_json(trajectory_path, trajectory_payload)
        runner._atomic_json(model_path, model_payload)
        manifest["summary"] = {
            "original_source_operational_failure_trajectories": 38,
            "parent_repairs_succeeded": 37,
            "parent_repairs_unresolved": 1,
            "cascading_repairs_succeeded": int(successful is not None),
            "cascading_repairs_unresolved": int(successful is None),
        }
        manifest["complete"] = successful is not None
        manifest["sanitized_artifacts"] = {
            "effective_trajectory_metrics": {
                "path": str(trajectory_path.resolve()),
                "file_sha256": runner._sha256_file(trajectory_path),
                "evidence_sha256": trajectory_payload["evidence_sha256"],
            },
            "effective_model_metrics": {
                "path": str(model_path.resolve()),
                "file_sha256": runner._sha256_file(model_path),
                "evidence_sha256": model_payload["evidence_sha256"],
            },
        }
        manifest["last_updated_at_utc"] = runner._utc_now()
        if manifest["complete"]:
            manifest["completed_at_utc"] = runner._utc_now()
        else:
            manifest.pop("completed_at_utc", None)
        _assert_no_secret_like_fields(manifest, label="cascading manifest")
        runner._seal(manifest)
        runner._atomic_json(manifest_path, manifest)
        return manifest
    finally:
        fcntl.flock(repair_lock.fileno(), fcntl.LOCK_UN)
        repair_lock.close()


def run_cascading_repair(
    *, source_manifest_path: Path, parent_overlay_manifest_path: Path,
    output_dir: Path, credential_env_file: Path,
    maximum_rounds: int = DEFAULT_MAXIMUM_ROUNDS,
    trajectory_workers: int = DEFAULT_TRAJECTORY_WORKERS,
    participant_workers: int = DEFAULT_PARTICIPANT_WORKERS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    rate_profile: str = DEFAULT_RATE_PROFILE,
    expected_account_count: int = EXPECTED_CONFIGURED_ACCOUNT_COUNT,
    resume: bool = False,
    sleep_fn: Callable[[float], None] = runner.time.sleep,
) -> dict[str, Any]:
    """Hold the immutable lineage for validation, dispatch, and final sealing."""

    try:
        source_path = source_manifest_path.resolve(strict=True)
        parent_path = parent_overlay_manifest_path.resolve(strict=True)
    except OSError as error:
        raise Part2CascadingRepairError(
            "Source or parent lineage manifest is unavailable."
        ) from error
    if source_path == parent_path:
        raise Part2CascadingRepairError(
            "Source and parent lineage manifests must be distinct."
        )
    with _hold_lineage_locks(
        [source_path, parent_path]
    ):
        return _run_cascading_repair_locked(
            source_manifest_path=source_path,
            parent_overlay_manifest_path=parent_path,
            output_dir=output_dir,
            credential_env_file=credential_env_file,
            maximum_rounds=maximum_rounds,
            trajectory_workers=trajectory_workers,
            participant_workers=participant_workers,
            max_attempts=max_attempts,
            initial_backoff_seconds=initial_backoff_seconds,
            timeout_seconds=timeout_seconds,
            rate_profile=rate_profile,
            expected_account_count=expected_account_count,
            resume=resume,
            sleep_fn=sleep_fn,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--parent-overlay-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--credential-env-file", type=Path, required=True)
    parser.add_argument("--maximum-rounds", type=int, default=DEFAULT_MAXIMUM_ROUNDS)
    parser.add_argument("--trajectory-workers", type=int, default=DEFAULT_TRAJECTORY_WORKERS)
    parser.add_argument("--participant-workers", type=int, default=DEFAULT_PARTICIPANT_WORKERS)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--initial-backoff-seconds", type=float, default=DEFAULT_BACKOFF_SECONDS
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--rate-profile", default=DEFAULT_RATE_PROFILE)
    parser.add_argument(
        "--expected-api-key-count", type=int,
        default=EXPECTED_CONFIGURED_ACCOUNT_COUNT,
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = run_cascading_repair(
            source_manifest_path=args.source_manifest,
            parent_overlay_manifest_path=args.parent_overlay_manifest,
            output_dir=args.output_dir,
            credential_env_file=args.credential_env_file,
            maximum_rounds=args.maximum_rounds,
            trajectory_workers=args.trajectory_workers,
            participant_workers=args.participant_workers,
            max_attempts=args.max_attempts,
            initial_backoff_seconds=args.initial_backoff_seconds,
            timeout_seconds=args.timeout_seconds,
            rate_profile=args.rate_profile,
            expected_account_count=args.expected_api_key_count,
            resume=args.resume,
        )
    except (
        Part2CascadingRepairError,
        InferenceHubDiscoveryError,
        runner.InferenceHubPart2PanelError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Part 2 cascading operational repair failed: {error}")
        return 2
    print(
        "Cascading operational trajectories unresolved: "
        f"{manifest['summary']['cascading_repairs_unresolved']}"
    )
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
