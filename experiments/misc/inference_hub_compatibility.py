"""Probe exact InferenceHub candidates for the production request contract.

The probe is deliberately separate from registry promotion.  It consumes a
schema-v2 reconciliation, exercises every exact candidate in its frozen order,
and emits sanitized, hash-bound evidence from which a later change can choose a
route.  Generated text and provider error messages are never persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

from analysis.reconcile_inference_hub_routes import reconcile_routes
from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
    _atomic_write_json,
    _bounded_ordered_map,
    _client_from_environment,
    _discovery_ledger_reference,
    _finish_discovery_attempt,
    _reserve_discovery_attempt,
    _resume_discovery_record,
    _sha256_json,
    _utc_now,
)

COMPATIBILITY_SCHEMA_VERSION = 2
RECONCILIATION_SCHEMA_VERSION = 2
DEFAULT_MAX_WORKERS = 16
DEFAULT_MAX_TOKENS_FLOOR = 64
REASONING_MAX_TOKENS_FLOOR = 2048
COMPATIBILITY_SEED = 20_260_801
OPTIONAL_CONTROLS = ("seed", "temperature", "top_p", "structured_response")
# Frozen in descending cardinality so the first passing profile has the maximum
# possible number of validated controls.  Within a cardinality, reproducibility
# and sampling controls precede response shaping: seed, temperature, top_p, then
# structured_response.  Structured JSON is useful, but plain Part 1 action
# extraction does not require it.
EXECUTION_PROFILE_ORDER = (
    ("seed", "temperature", "top_p", "structured_response"),
    ("seed", "temperature", "top_p"),
    ("seed", "temperature", "structured_response"),
    ("seed", "top_p", "structured_response"),
    ("temperature", "top_p", "structured_response"),
    ("seed", "temperature"),
    ("seed", "top_p"),
    ("seed", "structured_response"),
    ("temperature", "top_p"),
    ("temperature", "structured_response"),
    ("top_p", "structured_response"),
    ("seed",),
    ("temperature",),
    ("top_p",),
    ("structured_response",),
)
REASONING_FAMILY_MARKERS = (
    "gpt-5",
    "gpt-5.6",
    "kimi",
    "glm",
    "deepseek",
    "qwen",
    "nemotron",
    "gpt-oss",
)
REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking", "analysis")
STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "string", "enum": ["OK"]}},
    "required": ["ok"],
    "additionalProperties": False,
}


class CompatibilityError(ValueError):
    """An input or provider result cannot support compatibility evidence."""


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be positive") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompatibilityError(f"{label} is not readable UTF-8 JSON.") from error
    if not isinstance(value, dict):
        raise CompatibilityError(f"{label} must be a JSON object.")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_inputs(
    catalog: Mapping[str, Any],
    registry: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        reconciliation.get("schema_version") != RECONCILIATION_SCHEMA_VERSION
        or reconciliation.get("artifact_type")
        != "inference_hub_route_reconciliation"
    ):
        raise CompatibilityError("Reconciliation must be a schema-v2 route report.")
    without_hash = {
        key: value for key, value in reconciliation.items() if key != "report_sha256"
    }
    if reconciliation.get("report_sha256") != _sha256_json(without_hash):
        raise CompatibilityError("Reconciliation report_sha256 is invalid.")
    expected = reconcile_routes(catalog=catalog, registry=registry)
    if reconciliation != expected:
        raise CompatibilityError(
            "Reconciliation does not exactly match the supplied catalog and registry."
        )
    return expected


def output_token_budget(*, target_model: str, route: str, requested_floor: int) -> int:
    """Apply the mandatory reasoning-family output allowance."""

    if requested_floor <= 0:
        raise ValueError("requested_floor must be positive.")
    identity = f"{target_model}/{route}".casefold()
    if any(marker in identity for marker in REASONING_FAMILY_MARKERS):
        return max(requested_floor, REASONING_MAX_TOKENS_FLOOR)
    return requested_floor


def _control_value(control: str) -> dict[str, Any]:
    if control == "seed":
        return {"seed": COMPATIBILITY_SEED}
    if control == "temperature":
        return {"temperature": 0}
    if control == "top_p":
        return {"top_p": 1}
    if control == "structured_response":
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "inference_hub_compatibility",
                    "strict": True,
                    "schema": STRUCTURED_SCHEMA,
                },
            }
        }
    raise ValueError(f"Unknown compatibility control {control!r}.")


def _request_body(
    route: str, max_tokens: int, controls: tuple[str, ...]
) -> dict[str, Any]:
    structured = "structured_response" in controls
    prompt = (
        'Return exactly the JSON object {"ok":"OK"}.'
        if structured
        else "Reply with OK."
    )
    body: dict[str, Any] = {
        "model": route,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    for control in controls:
        body.update(_control_value(control))
    return body


def _safe_response_evidence(response: Mapping[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {"response_sha256": _sha256_json(response)}
    request_id = response.get("id")
    response_model = response.get("model")
    if isinstance(request_id, str) and request_id.strip():
        evidence["request_id"] = request_id.strip()
    if isinstance(response_model, str) and response_model.strip():
        evidence["response_model"] = response_model.strip()
    choices = response.get("choices")
    if (
        isinstance(choices, list)
        and len(choices) == 1
        and isinstance(choices[0], Mapping)
    ):
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason.strip():
            evidence["finish_reason"] = finish_reason.strip()
        message = choice.get("message")
        if isinstance(message, Mapping):
            for field in ("content", *REASONING_FIELDS):
                text = message.get(field)
                if isinstance(text, str) and text.strip():
                    evidence["output_field"] = field
                    evidence["content_sha256"] = hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest()
                    break
    usage = response.get("usage")
    safe_usage = {
        key: value
        for key, value in (usage.items() if isinstance(usage, Mapping) else [])
        if isinstance(key, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
    }
    if safe_usage:
        evidence["usage_sha256"] = _sha256_json(safe_usage)
    return evidence


def _response_error(
    code: str, response: Mapping[str, Any]
) -> InferenceHubDiscoveryError:
    return InferenceHubDiscoveryError(
        "Compatibility response validation failed.",
        failure_code=code,
        evidence=_safe_response_evidence(response),
    )


def _parse_response(
    response: Mapping[str, Any], *, route: str, structured: bool
) -> dict[str, Any]:
    evidence = _safe_response_evidence(response)
    if evidence.get("response_model") != route:
        raise _response_error("response_model_identity_mismatch", response)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise _response_error("invalid_choice_count", response)
    choice = choices[0]
    if not isinstance(choice, Mapping) or not isinstance(
        choice.get("message"), Mapping
    ):
        raise _response_error("missing_message", response)
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        raise _response_error("missing_finish_reason", response)
    if finish_reason.strip().casefold() in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }:
        raise _response_error("truncated_completion", response)
    message = choice["message"]
    output_field = None
    output_text = None
    for field in ("content", *REASONING_FIELDS):
        value = message.get(field)
        if isinstance(value, str) and value.strip():
            output_field, output_text = field, value
            break
    if output_text is None:
        raise _response_error("empty_content_and_reasoning", response)
    if structured:
        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError as error:
            raise _response_error("invalid_structured_json", response) from error
        if parsed != {"ok": "OK"}:
            raise _response_error("structured_schema_mismatch", response)
    return {
        **evidence,
        "output_field": output_field,
        "structured_output_validated": structured,
    }


def _resumed_stage(record: Mapping[str, Any], *, request_sha256: str) -> dict[str, Any]:
    passed = record.get("outcome") == "compatibility_passed"
    safe_keys = (
        "request_id",
        "response_model",
        "response_sha256",
        "content_sha256",
        "finish_reason",
        "usage_sha256",
        "http_status",
        "error_body_sha256",
        "error_body_bytes",
        "error_body_parse_truncated",
        "provider_error_type",
        "provider_error_code",
        "provider_error_param",
        "provider_error_message_sha256",
    )
    evidence = {
        key: record.get(key) for key in safe_keys if record.get(key) is not None
    }
    return {
        "status": "passed" if passed else "failed",
        "failure_code": None if passed else record.get("failure_code"),
        "request_sha256": request_sha256,
        "resumed_from_ledger": True,
        "source_attempt_id": record.get("attempt_id"),
        "evidence": evidence,
    }


def _retryable_terminal_record(record: Mapping[str, Any]) -> bool:
    status = record.get("http_status")
    return (
        record.get("failure_code") in {"connection_error", "connection_timeout"}
        or status == 429
        or (isinstance(status, int) and 500 <= status <= 599)
    )


def _profile_id(index: int, controls: tuple[str, ...]) -> str:
    """Return a stable ledger/evidence identifier for a frozen profile."""

    return f"profile_{index:02d}_{'_'.join(controls)}"


def _selected_profile(
    result: Mapping[str, Any], *, profile_id: str, validation_source: str
) -> dict[str, Any]:
    """Bind a selected profile to the exact successful request evidence."""

    selected = {
        "profile_id": profile_id,
        "controls": list(result["controls"]),
        "status": result["status"],
        "request_sha256": result["request_sha256"],
        "resumed_from_ledger": result["resumed_from_ledger"],
        "validation_source": validation_source,
    }
    for key in ("attempt_id", "source_attempt_id"):
        if result.get(key) is not None:
            selected[key] = result[key]
    return selected


def _run_stage(
    client: InferenceHubClient,
    *,
    ledger_path: Path,
    ledger_target_id: str,
    route: str,
    stage: str,
    max_tokens: int,
    controls: tuple[str, ...],
) -> dict[str, Any]:
    body = _request_body(route, max_tokens, controls)
    request_sha256 = _sha256_json(body)
    resume = _resume_discovery_record(
        ledger_path, target_id=ledger_target_id, route=route, request_body=body
    )
    if resume is not None and not _retryable_terminal_record(resume):
        result = _resumed_stage(resume, request_sha256=request_sha256)
        result["controls"] = list(controls)
        return result
    attempt_id = _reserve_discovery_attempt(
        ledger_path,
        target_id=ledger_target_id,
        route=route,
        request_body=body,
        max_tokens=max_tokens,
    )
    try:
        response = client.post("/chat/completions", body)
        evidence = _parse_response(
            response, route=route, structured="structured_response" in controls
        )
    except InferenceHubDiscoveryError as error:
        failure_code = error.failure_code
        evidence = dict(error.evidence)
        _finish_discovery_attempt(
            ledger_path,
            attempt_id=attempt_id,
            outcome="failed",
            failure_code=failure_code,
            request_id=evidence.get("request_id"),
            http_status=error.http_status,
            response_evidence=evidence,
        )
        return {
            "status": "failed",
            "failure_code": failure_code,
            "request_sha256": request_sha256,
            "resumed_from_ledger": False,
            "attempt_id": attempt_id,
            "controls": list(controls),
            "evidence": {
                **evidence,
                **(
                    {"http_status": error.http_status}
                    if error.http_status is not None
                    else {}
                ),
            },
        }
    except Exception:
        _finish_discovery_attempt(
            ledger_path,
            attempt_id=attempt_id,
            outcome="failed",
            failure_code="unexpected_probe_error",
            request_id=None,
        )
        return {
            "status": "failed",
            "failure_code": "unexpected_probe_error",
            "request_sha256": request_sha256,
            "resumed_from_ledger": False,
            "attempt_id": attempt_id,
            "controls": list(controls),
            "evidence": {},
        }
    _finish_discovery_attempt(
        ledger_path,
        attempt_id=attempt_id,
        outcome="compatibility_passed",
        failure_code=None,
        request_id=evidence.get("request_id"),
        response_evidence=evidence,
    )
    return {
        "status": "passed",
        "failure_code": None,
        "request_sha256": request_sha256,
        "resumed_from_ledger": False,
        "attempt_id": attempt_id,
        "controls": list(controls),
        "evidence": evidence,
    }


def _probe_candidate(
    client: InferenceHubClient,
    *,
    ledger_path: Path,
    target_id: str,
    target_model: str,
    route: str,
    candidate_index: int,
    requested_floor: int,
) -> dict[str, Any]:
    max_tokens = output_token_budget(
        target_model=target_model, route=route, requested_floor=requested_floor
    )
    stages: dict[str, Any] = {}
    minimal_key = f"compatibility:{target_id}:{candidate_index}:minimal"
    stages["minimal"] = _run_stage(
        client,
        ledger_path=ledger_path,
        ledger_target_id=minimal_key,
        route=route,
        stage="minimal",
        max_tokens=max_tokens,
        controls=(),
    )
    if stages["minimal"]["status"] != "passed":
        return {
            "route": route,
            "candidate_index": candidate_index,
            "max_tokens": max_tokens,
            "reasoning_budget_applied": max_tokens >= REASONING_MAX_TOKENS_FLOOR
            and any(
                marker in f"{target_model}/{route}".casefold()
                for marker in REASONING_FAMILY_MARKERS
            ),
            "tested_profiles": [],
            "selected_execution_profile": None,
            "profile_search_status": "skipped_minimal_failed",
            "execution_compatible": False,
            "stages": stages,
        }

    tested_profiles: list[dict[str, Any]] = []
    selected_execution_profile: dict[str, Any] | None = None
    for profile_index, controls in enumerate(EXECUTION_PROFILE_ORDER):
        profile_id = _profile_id(profile_index, controls)
        result = _run_stage(
            client,
            ledger_path=ledger_path,
            ledger_target_id=(
                f"compatibility:{target_id}:{candidate_index}:{profile_id}"
            ),
            route=route,
            stage=profile_id,
            max_tokens=max_tokens,
            controls=controls,
        )
        profile_result = {"profile_id": profile_id, **result}
        tested_profiles.append(profile_result)
        stages[profile_id] = result
        if result["status"] == "passed":
            selected_execution_profile = _selected_profile(
                result,
                profile_id=profile_id,
                validation_source="execution_profile_probe",
            )
            break

    # The minimal request is exactly the empty optional-control profile.  Reuse
    # its successful, hash-bound validation instead of dispatching a duplicate.
    if selected_execution_profile is None:
        selected_execution_profile = _selected_profile(
            stages["minimal"],
            profile_id="minimal",
            validation_source="minimal_stage",
        )
    execution_compatible = (
        stages["minimal"]["status"] == "passed"
        and selected_execution_profile["status"] == "passed"
    )
    return {
        "route": route,
        "candidate_index": candidate_index,
        "max_tokens": max_tokens,
        "reasoning_budget_applied": max_tokens >= REASONING_MAX_TOKENS_FLOOR
        and any(
            marker in f"{target_model}/{route}".casefold()
            for marker in REASONING_FAMILY_MARKERS
        ),
        "tested_profiles": tested_profiles,
        "selected_execution_profile": selected_execution_profile,
        "profile_search_status": (
            "optional_profile_selected"
            if selected_execution_profile["controls"]
            else "minimal_profile_selected_no_optional_controls"
        ),
        "execution_compatible": execution_compatible,
        "stages": stages,
    }


def probe_compatibility(
    client: InferenceHubClient,
    *,
    catalog: Mapping[str, Any],
    registry: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    ledger_path: Path,
    requested_floor: int = DEFAULT_MAX_TOKENS_FLOOR,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Probe every candidate and return deterministic-order sanitized evidence."""

    if requested_floor <= 0 or max_workers <= 0:
        raise ValueError("requested_floor and max_workers must be positive.")
    validated = _validate_inputs(catalog, registry, reconciliation)
    if catalog.get("endpoint") != client.base_url:
        raise CompatibilityError(
            "Catalog endpoint does not match the configured InferenceHub endpoint."
        )
    work: list[tuple[str, str, str, int]] = []
    for resolution in validated["resolutions"]:
        for candidate_index, route in enumerate(
            resolution["all_exact_suffix_candidates"]
        ):
            work.append(
                (resolution["target_id"], resolution["model"], route, candidate_index)
            )

    def worker(item: tuple[str, str, str, int]) -> tuple[str, dict[str, Any]]:
        target_id, model, route, candidate_index = item
        return target_id, _probe_candidate(
            client,
            ledger_path=ledger_path,
            target_id=target_id,
            target_model=model,
            route=route,
            candidate_index=candidate_index,
            requested_floor=requested_floor,
        )

    results = _bounded_ordered_map(worker, work, max_workers=max_workers)
    by_target: dict[str, list[dict[str, Any]]] = {}
    for target_id, result in results:
        by_target.setdefault(target_id, []).append(result)
    targets: list[dict[str, Any]] = []
    for resolution in validated["resolutions"]:
        candidates = by_target.get(resolution["target_id"], [])
        selected_candidate = next(
            (row for row in candidates if row["execution_compatible"]), None
        )
        selected = selected_candidate["route"] if selected_candidate else None
        selected_profile = (
            {
                "route": selected_candidate["route"],
                **selected_candidate["selected_execution_profile"],
            }
            if selected_candidate is not None
            else None
        )
        targets.append(
            {
                "target_id": resolution["target_id"],
                "cohort": resolution["cohort"],
                "model": resolution["model"],
                "frozen_candidate_order": resolution["all_exact_suffix_candidates"],
                "candidate_count": len(candidates),
                "selected_execution_candidate": selected,
                "selected_execution_profile": selected_profile,
                "selection_basis": (
                    "first_execution_compatible_in_reconciliation_frozen_order"
                    if selected is not None
                    else None
                ),
                "status": "execution_candidate_selected" if selected else "unresolved",
                "candidates": candidates,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "artifact_type": "inference_hub_provider_compatibility",
        "completed_at_utc": _utc_now(),
        "endpoint": client.base_url,
        "catalog_sha256": _sha256_json(catalog),
        "catalog_source_payload_sha256": catalog.get("source_payload_sha256"),
        "registry_sha256": _sha256_json(registry),
        "registry_version": registry.get("registry_version"),
        "reconciliation_sha256": _sha256_json(reconciliation),
        "reconciliation_report_sha256": reconciliation.get("report_sha256"),
        "selection_policy": {
            "candidate_order": "reconciliation_frozen_order",
            "minimal_request_required": True,
            "optional_controls": list(OPTIONAL_CONTROLS),
            "execution_profile_order": [
                list(profile) for profile in EXECUTION_PROFILE_ORDER
            ],
            "profile_ordering": (
                "descending_control_count_then_frozen_scientific_priority"
            ),
            "profile_selection_guarantee": (
                "maximum_control_count_then_frozen_priority_among_passing_profiles"
            ),
            "control_priority": list(OPTIONAL_CONTROLS),
            "structured_response_required_for_plain_part1": False,
            "first_passing_profile_selected": True,
            "empty_profile_reuses_successful_minimal_request": True,
            "automatic_registry_promotion": False,
        },
        "request_policy": {
            "requested_max_tokens_floor": requested_floor,
            "reasoning_max_tokens_floor": REASONING_MAX_TOKENS_FLOOR,
            "reasoning_family_markers": list(REASONING_FAMILY_MARKERS),
            "stage_order": [
                "minimal",
                *[
                    _profile_id(index, profile)
                    for index, profile in enumerate(EXECUTION_PROFILE_ORDER)
                ],
            ],
            "structured_schema_sha256": _sha256_json(STRUCTURED_SCHEMA),
        },
        "execution": {
            "strategy": "bounded_thread_pool",
            "configured_max_workers": max_workers,
            "effective_worker_count": min(max_workers, len(work)),
            "result_ordering": "reconciliation_target_then_frozen_candidate_order",
            "ledger_strategy": "locked_reservation_before_each_dispatch",
            "request_hash_resume": True,
        },
        "target_count": len(targets),
        "candidate_count": len(work),
        "selected_count": sum(
            target["selected_execution_candidate"] is not None for target in targets
        ),
        "unresolved_count": sum(
            target["selected_execution_candidate"] is None for target in targets
        ),
        "targets": targets,
        "attempt_ledger": _discovery_ledger_reference(ledger_path),
    }
    payload["evidence_sha256"] = _sha256_json(payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe all exact InferenceHub candidates for production controls."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-workers", type=_positive_int, default=DEFAULT_MAX_WORKERS
    )
    parser.add_argument(
        "--max-tokens-floor", type=_positive_int, default=DEFAULT_MAX_TOKENS_FLOOR
    )
    parser.add_argument("--timeout-seconds", type=_positive_float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_dotenv()
    catalog = _read_json(args.catalog, label="catalog")
    registry = _read_json(args.registry, label="registry")
    reconciliation = _read_json(args.reconciliation, label="reconciliation")
    client = _client_from_environment(args.timeout_seconds)
    evidence = probe_compatibility(
        client,
        catalog=catalog,
        registry=registry,
        reconciliation=reconciliation,
        ledger_path=args.attempt_ledger,
        requested_floor=args.max_tokens_floor,
        max_workers=args.max_workers,
    )
    evidence["inputs"] = {
        "catalog_file_sha256": _file_sha256(args.catalog),
        "registry_file_sha256": _file_sha256(args.registry),
        "reconciliation_file_sha256": _file_sha256(args.reconciliation),
    }
    evidence.pop("evidence_sha256", None)
    evidence["evidence_sha256"] = _sha256_json(evidence)
    _atomic_write_json(args.output, evidence)
    print(
        f"Probed {evidence['candidate_count']} candidates; "
        f"selected {evidence['selected_count']} execution-compatible routes."
    )
    print(f"Evidence: {args.output}")
    return 0 if evidence["unresolved_count"] == 0 else 1


def cli(argv: list[str] | None = None) -> int:
    """Run without emitting tracebacks or provider response objects."""

    try:
        return main(argv)
    except (
        CompatibilityError,
        InferenceHubDiscoveryError,
        OSError,
        ValueError,
    ) as error:
        print(f"InferenceHub compatibility probe failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
