"""Immutable request and token budget for the confirmatory campaign.

The budget is deliberately independent of provider execution.  Planning and
resume code can therefore validate it before importing any provider client or
dispatching a request.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import uuid
from contextvars import ContextVar
from pathlib import Path
from copy import deepcopy
from typing import Any, Mapping


SCHEMA_VERSION = 1
DEFAULT_TARGET_COUNT = 30
PHYSICAL_ATTEMPT_CEILING = 430_000
TOKEN_CEILING = 1_500_000_000
TRANSPORT_RESERVE_FRACTION = 0.10

PART0_HARMFUL_ROOTS = 484
PART0_CONTROL_ROOTS = 100
PART0_LANGUAGES = 3
PART0_SMOKE_SUBJECTS_PER_TARGET = 6
PART1_ROOTS = 384
PART1_SMOKE_PER_TARGET = 12
PART2_RUNS = 24
PART2_POPULATION = 10
PART2_HORIZON = 30
PART2_SMOKE_AGENT_DAYS_PER_TARGET = 12
PART0_REJUDGE_FIXTURES = 18

OUTPUT_TOKEN_CAPS = {
    "discovery": 16,
    "part0_subject": 512,
    "part0_judge": 32,
    "part1_subject": 32,
    "part2_subject": 32,
}

_LAST_ENVIRONMENT_RESERVATION: ContextVar[str | None] = ContextVar(
    "confirmatory_last_environment_reservation", default=None
)


class ConfirmatoryBudgetError(RuntimeError):
    """A plan or live ledger would exceed or alter its frozen budget."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    value = deepcopy(dict(payload))
    value.pop(field, None)
    return _stable_hash(value)


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfirmatoryBudgetError(f"{label} must be a positive integer.")
    return value


def successful_posts_by_role(
    target_count: int = DEFAULT_TARGET_COUNT,
    *,
    include_rejudge_allowance: bool,
) -> dict[str, int]:
    """Return the exact no-retry POST count for the frozen design."""

    targets = _require_positive_int(target_count, label="target_count")
    part0_production_per_target = (
        PART0_HARMFUL_ROOTS + PART0_CONTROL_ROOTS
    ) * PART0_LANGUAGES
    counts = {
        "discovery": targets,
        "part0_subject": targets
        * (PART0_SMOKE_SUBJECTS_PER_TARGET + part0_production_per_target),
        "part0_judge": targets
        * (PART0_SMOKE_SUBJECTS_PER_TARGET + part0_production_per_target),
        "part1_subject": targets * (PART1_SMOKE_PER_TARGET + PART1_ROOTS),
        "part2_subject": targets
        * (
            PART2_SMOKE_AGENT_DAYS_PER_TARGET
            + PART2_RUNS * PART2_POPULATION * PART2_HORIZON
        ),
    }
    if include_rejudge_allowance:
        counts["part0_judge"] += (
            targets * part0_production_per_target + PART0_REJUDGE_FIXTURES
        )
    return counts


def build_frozen_budget(
    target_count: int = DEFAULT_TARGET_COUNT,
    *,
    physical_attempt_ceiling: int = PHYSICAL_ATTEMPT_CEILING,
    token_ceiling: int = TOKEN_CEILING,
) -> dict[str, Any]:
    """Build and self-hash the preregistered campaign budget."""

    targets = _require_positive_int(target_count, label="target_count")
    physical_cap = _require_positive_int(
        physical_attempt_ceiling, label="physical_attempt_ceiling"
    )
    tokens_cap = _require_positive_int(token_ceiling, label="token_ceiling")
    base = successful_posts_by_role(targets, include_rejudge_allowance=False)
    with_rejudge = successful_posts_by_role(
        targets, include_rejudge_allowance=True
    )
    base_total = sum(base.values())
    allowance_total = sum(with_rejudge.values())
    planned_attempt_bound = math.ceil(
        allowance_total * (1 + TRANSPORT_RESERVE_FRACTION)
    )
    role_attempt_caps = {
        role: math.ceil(count * (1 + TRANSPORT_RESERVE_FRACTION))
        for role, count in with_rejudge.items()
    }
    base_output_bound = sum(
        base[role] * OUTPUT_TOKEN_CAPS[role] for role in base
    )
    if planned_attempt_bound > physical_cap:
        raise ConfirmatoryBudgetError(
            f"Frozen plan needs {planned_attempt_bound} attempts, exceeding "
            f"the ceiling {physical_cap}."
        )
    if base_output_bound > tokens_cap:
        raise ConfirmatoryBudgetError(
            f"Frozen output bound {base_output_bound} exceeds token ceiling "
            f"{tokens_cap}."
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "confirmatory_request_token_budget",
        "target_count": targets,
        "design": {
            "part0_harmful_roots": PART0_HARMFUL_ROOTS,
            "part0_control_roots": PART0_CONTROL_ROOTS,
            "part0_languages": PART0_LANGUAGES,
            "part1_roots": PART1_ROOTS,
            "part2_runs": PART2_RUNS,
            "part2_population": PART2_POPULATION,
            "part2_horizon": PART2_HORIZON,
        },
        "base_successful_posts_by_role": base,
        "base_successful_posts": base_total,
        "rejudge_inclusive_posts_by_role": with_rejudge,
        "rejudge_inclusive_posts": allowance_total,
        "transport_reserve_fraction": TRANSPORT_RESERVE_FRACTION,
        "planned_physical_attempt_bound": planned_attempt_bound,
        "role_attempt_caps": role_attempt_caps,
        "physical_attempt_ceiling": physical_cap,
        "output_token_caps": dict(OUTPUT_TOKEN_CAPS),
        "base_maximum_output_tokens": base_output_bound,
        "input_plus_output_token_ceiling": tokens_cap,
    }
    payload["budget_sha256"] = _self_hash(payload, "budget_sha256")
    return payload


def validate_frozen_budget(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild and exact-compare a budget artifact."""

    if not isinstance(payload, Mapping):
        raise ConfirmatoryBudgetError("Budget root must be an object.")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ConfirmatoryBudgetError("Unsupported budget schema.")
    if payload.get("budget_sha256") != _self_hash(payload, "budget_sha256"):
        raise ConfirmatoryBudgetError("Budget self-hash mismatch.")
    expected = build_frozen_budget(
        _require_positive_int(payload.get("target_count"), label="target_count"),
        physical_attempt_ceiling=_require_positive_int(
            payload.get("physical_attempt_ceiling"),
            label="physical_attempt_ceiling",
        ),
        token_ceiling=_require_positive_int(
            payload.get("input_plus_output_token_ceiling"),
            label="input_plus_output_token_ceiling",
        ),
    )
    if dict(payload) != expected:
        raise ConfirmatoryBudgetError("Budget fields differ from the frozen design.")
    return expected


def create_ledger(budget: Mapping[str, Any]) -> dict[str, Any]:
    """Create an empty self-hashed durable attempt ledger."""

    frozen = validate_frozen_budget(budget)
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "confirmatory_request_token_ledger",
        "budget_sha256": frozen["budget_sha256"],
        "attempts_by_role": {role: 0 for role in frozen["role_attempt_caps"]},
        "physical_attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "records": [],
    }
    ledger["ledger_sha256"] = _self_hash(ledger, "ledger_sha256")
    return ledger


def validate_ledger(
    ledger: Mapping[str, Any], budget: Mapping[str, Any]
) -> dict[str, Any]:
    """Replay counts and hashes before planning a resume or dispatch."""

    frozen = validate_frozen_budget(budget)
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise ConfirmatoryBudgetError("Unsupported ledger schema.")
    if ledger.get("budget_sha256") != frozen["budget_sha256"]:
        raise ConfirmatoryBudgetError("Ledger is bound to a different budget.")
    if ledger.get("ledger_sha256") != _self_hash(ledger, "ledger_sha256"):
        raise ConfirmatoryBudgetError("Ledger self-hash mismatch.")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise ConfirmatoryBudgetError("Ledger records must be a list.")
    counts = {role: 0 for role in frozen["role_attempt_caps"]}
    input_tokens = 0
    output_tokens = 0
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ConfirmatoryBudgetError("Ledger record must be an object.")
        role = record.get("role")
        attempt_id = record.get("attempt_id")
        if role not in counts or not isinstance(attempt_id, str) or not attempt_id:
            raise ConfirmatoryBudgetError(f"Invalid ledger record at index {index}.")
        if attempt_id in seen_ids:
            raise ConfirmatoryBudgetError("Ledger attempt IDs must be unique.")
        seen_ids.add(attempt_id)
        counts[str(role)] += 1
        input_tokens += _require_positive_int(
            record.get("input_tokens"), label="record input_tokens"
        )
        output_value = record.get("output_tokens")
        if isinstance(output_value, bool) or not isinstance(output_value, int) or output_value < 0:
            raise ConfirmatoryBudgetError("record output_tokens must be nonnegative.")
        output_tokens += output_value
    if ledger.get("attempts_by_role") != counts:
        raise ConfirmatoryBudgetError("Ledger role totals do not replay.")
    if ledger.get("physical_attempts") != len(records):
        raise ConfirmatoryBudgetError("Ledger attempt total does not replay.")
    if ledger.get("input_tokens") != input_tokens or ledger.get("output_tokens") != output_tokens:
        raise ConfirmatoryBudgetError("Ledger token totals do not replay.")
    if len(records) > frozen["physical_attempt_ceiling"]:
        raise ConfirmatoryBudgetError("Ledger exceeds physical-attempt ceiling.")
    if input_tokens + output_tokens > frozen["input_plus_output_token_ceiling"]:
        raise ConfirmatoryBudgetError("Ledger exceeds token ceiling.")
    if any(counts[role] > frozen["role_attempt_caps"][role] for role in counts):
        raise ConfirmatoryBudgetError("Ledger exceeds a role attempt cap.")
    return deepcopy(dict(ledger))


def record_attempt(
    ledger: Mapping[str, Any],
    budget: Mapping[str, Any],
    *,
    role: str,
    attempt_id: str,
    request_sha256: str,
    outcome: str,
    input_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Append one completed physical POST while enforcing every ceiling."""

    frozen = validate_frozen_budget(budget)
    current = validate_ledger(ledger, frozen)
    if role not in frozen["role_attempt_caps"]:
        raise ConfirmatoryBudgetError(f"Unknown request role: {role}.")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ConfirmatoryBudgetError("attempt_id must be nonempty.")
    if any(record.get("attempt_id") == attempt_id for record in current["records"]):
        raise ConfirmatoryBudgetError("attempt_id is already present.")
    if (
        not isinstance(request_sha256, str)
        or len(request_sha256) != 64
        or any(character not in "0123456789abcdef" for character in request_sha256)
    ):
        raise ConfirmatoryBudgetError("request_sha256 must be lowercase SHA-256.")
    if not isinstance(outcome, str) or not outcome:
        raise ConfirmatoryBudgetError("outcome must be nonempty.")
    input_count = _require_positive_int(input_tokens, label="input_tokens")
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens < 0:
        raise ConfirmatoryBudgetError("output_tokens must be nonnegative.")
    if current["physical_attempts"] + 1 > frozen["physical_attempt_ceiling"]:
        raise ConfirmatoryBudgetError("Physical-attempt ceiling would be exceeded.")
    if current["attempts_by_role"][role] + 1 > frozen["role_attempt_caps"][role]:
        raise ConfirmatoryBudgetError(f"Role cap would be exceeded for {role}.")
    if (
        current["input_tokens"]
        + current["output_tokens"]
        + input_count
        + output_tokens
        > frozen["input_plus_output_token_ceiling"]
    ):
        raise ConfirmatoryBudgetError("Token ceiling would be exceeded.")
    result = deepcopy(current)
    result.pop("ledger_sha256", None)
    result["records"].append(
        {
            "attempt_id": attempt_id,
            "role": role,
            "request_sha256": request_sha256,
            "outcome": outcome,
            "input_tokens": input_count,
            "output_tokens": output_tokens,
        }
    )
    result["attempts_by_role"][role] += 1
    result["physical_attempts"] += 1
    result["input_tokens"] += input_count
    result["output_tokens"] += output_tokens
    result["ledger_sha256"] = _self_hash(result, "ledger_sha256")
    validate_ledger(result, frozen)
    return result


def reserve_environment_attempt(
    *,
    provider: str,
    model: str,
    system_prompt: str,
    query: str,
    max_tokens: int | None,
) -> None:
    """Durably reserve one confirmatory POST before provider dispatch."""

    budget_path_value = os.getenv("CONFIRMATORY_BUDGET_PATH", "").strip()
    ledger_path_value = os.getenv("CONFIRMATORY_LEDGER_PATH", "").strip()
    experiment = os.getenv("CONFIRMATORY_EXPERIMENT", "").strip()
    if not budget_path_value and not ledger_path_value and not experiment:
        _LAST_ENVIRONMENT_RESERVATION.set(None)
        return
    if not all((budget_path_value, ledger_path_value, experiment)):
        raise ConfirmatoryBudgetError("Confirmatory budget environment is incomplete.")
    if provider.strip().lower().replace("-", "_") != "inference_hub":
        raise ConfirmatoryBudgetError("Confirmatory requests must use InferenceHub.")
    if experiment == "part0":
        role = "part0_judge" if max_tokens == OUTPUT_TOKEN_CAPS["part0_judge"] else "part0_subject"
    elif experiment == "part1":
        role = "part1_subject"
    elif experiment == "part2":
        role = "part2_subject"
    else:
        raise ConfirmatoryBudgetError(f"Unknown confirmatory experiment: {experiment}.")
    budget_path = Path(budget_path_value).resolve()
    ledger_path = Path(ledger_path_value).resolve()
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        request_sha256 = hashlib.sha256(
            _canonical_json(
                {
                    "provider": provider,
                    "model": model,
                    "system_prompt": system_prompt,
                    "query": query,
                    "max_tokens": max_tokens,
                }
            ).encode("utf-8")
        ).hexdigest()
        reserved = record_attempt(
            ledger,
            budget,
            role=role,
            attempt_id=f"reserve_{uuid.uuid4().hex}",
            request_sha256=request_sha256,
            outcome="reserved_before_dispatch",
            # Byte-level tokenizers cannot emit more tokens than the UTF-8 byte
            # stream. Reserving one token per byte is deliberately conservative
            # for multilingual inputs and avoids language-dependent undercount.
            input_tokens=max(1, len((system_prompt + query).encode("utf-8"))),
            output_tokens=(
                max_tokens
                if isinstance(max_tokens, int) and not isinstance(max_tokens, bool)
                else OUTPUT_TOKEN_CAPS[role]
            ),
        )
        descriptor, temporary = tempfile.mkstemp(
            dir=ledger_path.parent, prefix=f".{ledger_path.name}."
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(reserved, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, ledger_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        _LAST_ENVIRONMENT_RESERVATION.set(request_sha256)


def consume_environment_reservation() -> str | None:
    """Return and clear the dispatch hash reserved for the current call context."""

    value = _LAST_ENVIRONMENT_RESERVATION.get()
    _LAST_ENVIRONMENT_RESERVATION.set(None)
    return value


__all__ = [
    "ConfirmatoryBudgetError",
    "DEFAULT_TARGET_COUNT",
    "OUTPUT_TOKEN_CAPS",
    "PHYSICAL_ATTEMPT_CEILING",
    "TOKEN_CEILING",
    "build_frozen_budget",
    "create_ledger",
    "record_attempt",
    "reserve_environment_attempt",
    "consume_environment_reservation",
    "successful_posts_by_role",
    "validate_frozen_budget",
    "validate_ledger",
]
