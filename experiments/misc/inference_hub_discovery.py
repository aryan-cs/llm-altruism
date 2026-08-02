"""Discover and verify exact NVIDIA InferenceHub model routes.

This module deliberately does not mutate the experiment model registry. It
captures a sanitized, hash-bound catalog snapshot and can produce a separate
smoke-test evidence record. A registry route should be promoted only after the
evidence record has been reviewed and pinned in version control.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

import certifi
from dotenv import load_dotenv

from agents.agent_config import load_model_cohort, validate_endpoint_base_url
from analysis.reconcile_inference_hub_routes import (
    RouteReconciliationError,
    reconcile_routes,
)

DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
ROUTE_SOURCE = "inference_hub_models_api"
CATALOG_SCHEMA_VERSION = 2
SMOKE_SCHEMA_VERSION = 2
COHORT_EVIDENCE_SCHEMA_VERSION = 2
CANDIDATE_EVIDENCE_SCHEMA_VERSION = 2
CATALOG_PROBE_SCHEMA_VERSION = 2
DISCOVERY_LEDGER_SCHEMA_VERSION = 2
DEFAULT_MAX_WORKERS = 16
SMOKE_SEED = 20_260_801
SMOKE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "string", "enum": ["OK"]}},
    "required": ["ok"],
    "additionalProperties": False,
}


class InferenceHubDiscoveryError(RuntimeError):
    """A safe, credential-free error raised during route discovery."""

    def __init__(
        self,
        message: str,
        *,
        failure_code: str = "inference_hub_discovery_failed",
        http_status: int | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_code = failure_code
        self.http_status = http_status
        self.evidence = dict(evidence or {})


_WorkItem = TypeVar("_WorkItem")
_WorkResult = TypeVar("_WorkResult")


def _validate_max_workers(max_workers: int) -> int:
    if (
        not isinstance(max_workers, int)
        or isinstance(max_workers, bool)
        or max_workers <= 0
    ):
        raise ValueError("max_workers must be a positive integer.")
    return max_workers


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    try:
        return _validate_max_workers(parsed)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error


def _bounded_ordered_map(
    worker: Callable[[_WorkItem], _WorkResult],
    items: list[_WorkItem],
    *,
    max_workers: int,
) -> list[_WorkResult]:
    """Run bounded work concurrently while returning results in input order."""

    configured_workers = _validate_max_workers(max_workers)
    if not items:
        return []
    effective_workers = min(configured_workers, len(items))
    if effective_workers == 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(
        max_workers=effective_workers,
        thread_name_prefix="inference-hub-discovery",
    ) as executor:
        return list(executor.map(worker, items))


def _execution_metadata(
    *,
    max_workers: int,
    item_count: int,
    result_ordering: str,
) -> dict[str, Any]:
    configured_workers = _validate_max_workers(max_workers)
    return {
        "strategy": "bounded_thread_pool",
        "configured_max_workers": configured_workers,
        "effective_worker_count": min(configured_workers, item_count),
        "result_ordering": result_ordering,
        "ledger_strategy": "locked_reservation_before_each_dispatch",
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _discovery_ledger_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_json(
        {key: value for key, value in payload.items() if key != "ledger_sha256"}
    )


@contextmanager
def _discovery_ledger_lock(path: Path):
    """Serialize the complete ledger read/modify/replace transaction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_discovery_ledger_unlocked(path: Path) -> dict[str, Any]:
    if not path.exists():
        payload: dict[str, Any] = {
            "schema_version": DISCOVERY_LEDGER_SCHEMA_VERSION,
            "artifact_type": "inference_hub_discovery_attempt_ledger",
            "records": [],
        }
        payload["ledger_sha256"] = _discovery_ledger_hash(payload)
        _atomic_write_json(path, payload)
        return payload
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceHubDiscoveryError(
            "Discovery attempt ledger is not valid UTF-8 JSON."
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != DISCOVERY_LEDGER_SCHEMA_VERSION
        or payload.get("artifact_type")
        != "inference_hub_discovery_attempt_ledger"
        or not isinstance(payload.get("records"), list)
        or payload.get("ledger_sha256") != _discovery_ledger_hash(payload)
    ):
        raise InferenceHubDiscoveryError("Discovery attempt ledger integrity failed.")
    attempt_ids = [
        record.get("attempt_id")
        for record in payload["records"]
        if isinstance(record, Mapping)
    ]
    if (
        len(attempt_ids) != len(payload["records"])
        or not all(isinstance(value, str) and value for value in attempt_ids)
        or len(set(attempt_ids)) != len(attempt_ids)
    ):
        raise InferenceHubDiscoveryError("Discovery attempt ledger records are invalid.")
    return payload


def _load_discovery_ledger(path: Path) -> dict[str, Any]:
    with _discovery_ledger_lock(path):
        return _load_discovery_ledger_unlocked(path)


def _discovery_ledger_reference(path: Path) -> dict[str, Any]:
    """Hash one validated ledger version while concurrent writers are excluded."""

    with _discovery_ledger_lock(path):
        ledger = _load_discovery_ledger_unlocked(path)
        raw_bytes = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "ledger_sha256": ledger["ledger_sha256"],
        "record_count": len(ledger["records"]),
    }


def _reserve_discovery_attempt(
    path: Path,
    *,
    target_id: str,
    route: str,
    request_body: Mapping[str, Any],
    max_tokens: int,
) -> str:
    with _discovery_ledger_lock(path):
        ledger = _load_discovery_ledger_unlocked(path)
        attempt_id = f"discovery_{uuid.uuid4().hex}"
        encoded = _canonical_bytes(request_body)
        ledger.pop("ledger_sha256", None)
        ledger["records"].append(
            {
                "attempt_id": attempt_id,
                "target_id": target_id,
                "route": route,
                "reserved_at_utc": _utc_now(),
                "request_sha256": hashlib.sha256(encoded).hexdigest(),
                "input_tokens": max(1, len(encoded)),
                "output_tokens": max_tokens,
                "outcome": "reserved_before_dispatch",
                "failure_code": None,
                "http_status": None,
                "request_id": None,
                "response_model": None,
                "response_sha256": None,
                "content_sha256": None,
                "finish_reason": None,
                "usage_sha256": None,
            }
        )
        ledger["ledger_sha256"] = _discovery_ledger_hash(ledger)
        _atomic_write_json(path, ledger)
    return attempt_id


def _finish_discovery_attempt(
    path: Path,
    *,
    attempt_id: str,
    outcome: str,
    failure_code: str | None,
    request_id: str | None,
    http_status: int | None = None,
    response_evidence: Mapping[str, Any] | None = None,
) -> None:
    safe_evidence = dict(response_evidence or {})
    with _discovery_ledger_lock(path):
        ledger = _load_discovery_ledger_unlocked(path)
        matches = [
            record
            for record in ledger["records"]
            if isinstance(record, dict) and record.get("attempt_id") == attempt_id
        ]
        if (
            len(matches) != 1
            or matches[0].get("outcome") != "reserved_before_dispatch"
        ):
            raise InferenceHubDiscoveryError(
                "Discovery attempt completion does not match one pending reservation."
            )
        matches[0]["outcome"] = outcome
        matches[0]["failure_code"] = failure_code
        matches[0]["http_status"] = http_status
        matches[0]["request_id"] = request_id or safe_evidence.get("request_id")
        for key in (
            "response_model",
            "response_sha256",
            "content_sha256",
            "finish_reason",
            "usage_sha256",
        ):
            matches[0][key] = safe_evidence.get(key)
        ledger.pop("ledger_sha256", None)
        ledger["ledger_sha256"] = _discovery_ledger_hash(ledger)
        _atomic_write_json(path, ledger)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Fail closed instead of forwarding a bearer credential to a new URL."""

    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _urlopen_no_redirect(
    request: urllib.request.Request,
    *,
    timeout: float,
    context: ssl.SSLContext,
):
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=context),
        _NoRedirectHandler(),
    )
    return opener.open(request, timeout=timeout)


class InferenceHubClient:
    """Minimal authenticated client with an exact-host trust boundary."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = validate_endpoint_base_url("inference_hub", base_url)
        if not isinstance(api_key, str) or not api_key.strip():
            raise InferenceHubDiscoveryError(
                "NVIDIA_API_KEY is required for InferenceHub discovery."
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self._api_key = api_key.strip()
        self.timeout_seconds = float(timeout_seconds)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not path.startswith("/") or "?" in path or "#" in path:
            raise ValueError("InferenceHub request path must be an absolute clean path.")
        encoded_body = None if body is None else _canonical_bytes(body)
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=encoded_body,
            method=method.upper(),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                **(
                    {"Content-Type": "application/json"}
                    if encoded_body is not None
                    else {}
                ),
            },
        )
        try:
            tls_context = ssl.create_default_context(cafile=certifi.where())
            with _urlopen_no_redirect(
                request,
                timeout=self.timeout_seconds,
                context=tls_context,
            ) as response:
                payload_bytes = response.read()
        except urllib.error.HTTPError as error:
            raise InferenceHubDiscoveryError(
                f"InferenceHub {path} returned HTTP {error.code}.",
                failure_code="http_error",
                http_status=error.code,
            ) from error
        except urllib.error.URLError as error:
            reason = type(error.reason).__name__
            raise InferenceHubDiscoveryError(
                f"InferenceHub {path} connection failed ({reason}).",
                failure_code="connection_error",
            ) from error
        try:
            payload = json.loads(payload_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise InferenceHubDiscoveryError(
                f"InferenceHub {path} returned non-JSON content."
            ) from error
        if not isinstance(payload, dict):
            raise InferenceHubDiscoveryError(
                f"InferenceHub {path} returned a non-object JSON payload."
            )
        return payload

    def get(self, path: str) -> dict[str, Any]:
        return self.request_json("GET", path)

    def post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        return self.request_json("POST", path, body=body)


def _models_routes(payload: Mapping[str, Any]) -> set[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise InferenceHubDiscoveryError("InferenceHub /models payload lacks data[].")
    routes: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise InferenceHubDiscoveryError(
                f"InferenceHub /models data[{index}] is not an object."
            )
        route = item.get("id")
        if not isinstance(route, str) or not route.strip():
            raise InferenceHubDiscoveryError(
                f"InferenceHub /models data[{index}].id is missing."
            )
        normalized = route.strip()
        if normalized in routes:
            raise InferenceHubDiscoveryError(
                f"InferenceHub /models contains duplicate route {normalized!r}."
            )
        routes.add(normalized)
    return routes


def capture_catalog(client: InferenceHubClient) -> dict[str, Any]:
    """Capture the exact catalog exposed to an LLM-route virtual key.

    InferenceHub virtual keys with the ``llm_api_routes`` permission can call
    ``/models`` and ``/chat/completions`` but are forbidden from the portal's
    privileged ``/model/info`` route.  Chat capability and response identity
    are therefore established by the bounded structured smoke, not inferred
    from metadata the credential cannot access.
    """

    models_payload = client.get("/models")
    models_routes = _models_routes(models_payload)
    rows = [
        {
            "route": route,
            "listed_by_models": True,
            "chat_capability": "unverified_until_structured_smoke",
        }
        for route in sorted(models_routes)
    ]
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "captured_at_utc": _utc_now(),
        "endpoint": client.base_url,
        "route_source": ROUTE_SOURCE,
        "source_endpoints": ["/models"],
        "source_payload_sha256": {"models": _sha256_json(models_payload)},
        "route_count": len(rows),
        "routes": rows,
    }


def _catalog_route(catalog: Mapping[str, Any], route: str) -> Mapping[str, Any]:
    routes = catalog.get("routes")
    if not isinstance(routes, list):
        raise InferenceHubDiscoveryError("Catalog snapshot lacks routes[].")
    exact = [row for row in routes if isinstance(row, Mapping) and row.get("route") == route]
    if len(exact) != 1:
        raise InferenceHubDiscoveryError(
            f"Exact route {route!r} does not occur once in the catalog snapshot."
        )
    row = exact[0]
    if row.get("listed_by_models") is not True:
        raise InferenceHubDiscoveryError(f"Route {route!r} is absent from /models.")
    return row


def _smoke_request_body(route: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": route,
        "messages": [
            {
                "role": "user",
                "content": 'Return exactly the JSON object {"ok":"OK"}.',
            }
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": SMOKE_SEED,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "inference_hub_route_smoke",
                "strict": True,
                "schema": SMOKE_SCHEMA,
            },
        },
    }


def _chat_probe_request_body(route: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": route,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": max_tokens,
        "stream": False,
    }


def _sanitized_response_evidence(response: Mapping[str, Any]) -> dict[str, Any]:
    """Retain proof of a rejected response without storing generated text."""

    evidence: dict[str, Any] = {"response_sha256": _sha256_json(response)}
    request_id = response.get("id")
    response_model = response.get("model")
    if isinstance(request_id, str) and request_id.strip():
        evidence["request_id"] = request_id.strip()
    if isinstance(response_model, str) and response_model.strip():
        evidence["response_model"] = response_model.strip()
    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) == 1:
        choice = choices[0]
        if isinstance(choice, Mapping):
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str) and finish_reason.strip():
                evidence["finish_reason"] = finish_reason.strip()
            message = choice.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    evidence["content_sha256"] = hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest()
    usage = response.get("usage")
    safe_usage = {
        key: value
        for key, value in (usage.items() if isinstance(usage, Mapping) else [])
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
    }
    if safe_usage:
        evidence["usage_sha256"] = _sha256_json(safe_usage)
    return evidence


def _chat_response_error(
    message: str,
    *,
    failure_code: str,
    response: Mapping[str, Any],
) -> InferenceHubDiscoveryError:
    return InferenceHubDiscoveryError(
        message,
        failure_code=failure_code,
        evidence=_sanitized_response_evidence(response),
    )


def _failure_details(
    error: InferenceHubDiscoveryError | ValueError,
    *,
    default_code: str,
) -> tuple[str, int | None, dict[str, Any]]:
    if isinstance(error, InferenceHubDiscoveryError):
        code = (
            error.failure_code
            if error.failure_code != "inference_hub_discovery_failed"
            else default_code
        )
        return code, error.http_status, dict(error.evidence)
    return default_code, None, {}


def _successful_response_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    response = evidence.get("response")
    if not isinstance(response, Mapping):
        return {}
    verification = evidence.get("verification_evidence")
    smoke = (
        verification.get("smoke_test")
        if isinstance(verification, Mapping)
        else None
    )
    result = {
        "request_id": response.get("request_id")
        or (smoke.get("request_id") if isinstance(smoke, Mapping) else None),
        "response_model": evidence.get("provider_response_model"),
        "response_sha256": response.get("payload_sha256"),
        "content_sha256": response.get("content_sha256"),
        "finish_reason": response.get("finish_reason"),
        "usage_sha256": response.get("usage_sha256"),
    }
    return {key: value for key, value in result.items() if value is not None}


def _parse_chat_response(
    response: Mapping[str, Any],
    *,
    requested_route: str,
    allow_truncation: bool = False,
    require_usage: bool = True,
) -> dict[str, Any]:
    request_id = response.get("id")
    response_model = response.get("model")
    choices = response.get("choices")
    if not isinstance(request_id, str) or not request_id.strip():
        raise _chat_response_error(
            "Chat response did not report a request/completion id.",
            failure_code="missing_request_id",
            response=response,
        )
    if not isinstance(response_model, str) or not response_model.strip():
        raise _chat_response_error(
            "Chat response did not report a provider response model.",
            failure_code="missing_response_model",
            response=response,
        )
    if response_model.strip() != requested_route:
        raise _chat_response_error(
            "Chat response model identity does not match the exact requested route.",
            failure_code="response_model_identity_mismatch",
            response=response,
        )
    if not isinstance(choices, list) or len(choices) != 1:
        raise _chat_response_error(
            "Chat response must contain exactly one completion choice.",
            failure_code="invalid_choice_count",
            response=response,
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise _chat_response_error(
            "Chat completion choice is not an object.",
            failure_code="invalid_choice",
            response=response,
        )
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise _chat_response_error(
            "Chat completion lacks a message object.",
            failure_code="missing_message",
            response=response,
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _chat_response_error(
            "Chat completion content is empty.",
            failure_code="empty_content",
            response=response,
        )
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        raise _chat_response_error(
            "Chat completion lacks a finish reason.",
            failure_code="missing_finish_reason",
            response=response,
        )
    truncated = finish_reason.strip().lower() in {
        "length",
        "max_tokens",
        "max_output_tokens",
    }
    if truncated and not allow_truncation:
        raise _chat_response_error(
            "Chat completion was truncated.",
            failure_code="truncated_completion",
            response=response,
        )
    usage = response.get("usage")
    if require_usage and (not isinstance(usage, Mapping) or not usage):
        raise _chat_response_error(
            "Chat completion lacks token usage.",
            failure_code="missing_usage",
            response=response,
        )
    safe_usage = {
        key: value
        for key, value in (usage.items() if isinstance(usage, Mapping) else [])
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
    }
    positive_usage = bool(safe_usage) and any(
        safe_usage.get(key, 0) > 0
        for key in ("completion_tokens", "output_tokens", "total_tokens")
    )
    if require_usage and not positive_usage:
        raise _chat_response_error(
            "Chat completion lacks positive output/total token usage.",
            failure_code="nonpositive_usage",
            response=response,
        )
    return {
        "request_id": request_id.strip(),
        "response_model": response_model.strip(),
        "content": content,
        "finish_reason": finish_reason.strip(),
        "truncated": truncated,
        "usage": safe_usage if positive_usage else None,
        "usage_sha256": _sha256_json(safe_usage) if positive_usage else None,
        "response_sha256": _sha256_json(response),
    }


def smoke_verify_route(
    client: InferenceHubClient,
    *,
    catalog: Mapping[str, Any],
    route: str,
    max_tokens: int = 16,
) -> dict[str, Any]:
    """Run one bounded chat request and return sanitized verification evidence."""

    normalized_route = route.strip()
    if not normalized_route:
        raise ValueError("route must be non-empty.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    catalog_row = _catalog_route(catalog, normalized_route)
    schema_sha256 = _sha256_json(SMOKE_SCHEMA)
    request_body = _smoke_request_body(normalized_route, max_tokens)
    response = client.post("/chat/completions", request_body)
    parsed = _parse_chat_response(response, requested_route=normalized_route)
    content = str(parsed["content"])
    try:
        structured_content = json.loads(content)
    except json.JSONDecodeError as error:
        raise _chat_response_error(
            "Smoke completion did not honor structured JSON output.",
            failure_code="invalid_structured_json",
            response=response,
        ) from error
    if structured_content != {"ok": "OK"}:
        raise _chat_response_error(
            "Smoke completion did not satisfy the exact response schema.",
            failure_code="structured_schema_mismatch",
            response=response,
        )
    generation_controls = {
        "temperature": 0,
        "top_p": 1,
        "seed": SMOKE_SEED,
        "max_tokens": max_tokens,
        "stream": False,
        "structured_output": True,
        "json_schema_sha256": schema_sha256,
    }
    completed_at_utc = _utc_now()
    catalog_route_sha256 = _sha256_json(catalog_row)
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "verification_status": "verified",
        "verified_at_utc": completed_at_utc,
        "route_source": ROUTE_SOURCE,
        "endpoint": client.base_url,
        "requested_route": normalized_route,
        "provider_response_model": parsed["response_model"],
        "catalog_captured_at_utc": catalog.get("captured_at_utc"),
        "catalog_source_payload_sha256": catalog.get("source_payload_sha256"),
        "catalog_route_sha256": catalog_route_sha256,
        "verification_evidence": {
            "verified_at_utc": completed_at_utc,
            "discovery_sha256": catalog_route_sha256,
            "smoke_test": {
                "completed_at_utc": completed_at_utc,
                "request_id": parsed["request_id"],
                "response_model": parsed["response_model"],
                "response_sha256": parsed["response_sha256"],
                "finish_reason": parsed["finish_reason"],
                "usage_sha256": parsed["usage_sha256"],
                "generation_controls": generation_controls,
            },
        },
        "request": {
            **generation_controls,
            "request_sha256": _sha256_json(request_body),
            "prompt_sha256": hashlib.sha256(
                request_body["messages"][0]["content"].encode("utf-8")
            ).hexdigest(),
        },
        "response": {
            "payload_sha256": parsed["response_sha256"],
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "finish_reason": parsed["finish_reason"],
            "truncated": parsed["truncated"],
            "usage": parsed["usage"],
            "usage_sha256": parsed["usage_sha256"],
            "structured_output_validated": True,
        },
    }


def chat_probe_route(
    client: InferenceHubClient,
    *,
    catalog: Mapping[str, Any],
    route: str,
    max_tokens: int = 8,
) -> dict[str, Any]:
    """Prove minimal chat callability without assuming optional controls."""

    normalized_route = route.strip()
    if not normalized_route:
        raise ValueError("route must be non-empty.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")
    catalog_row = _catalog_route(catalog, normalized_route)
    request_body = _chat_probe_request_body(normalized_route, max_tokens)
    response = client.post("/chat/completions", request_body)
    parsed = _parse_chat_response(
        response,
        requested_route=normalized_route,
        allow_truncation=True,
        require_usage=False,
    )
    completed_at = _utc_now()
    return {
        "schema_version": CATALOG_PROBE_SCHEMA_VERSION,
        "verification_status": "chat_callable",
        "verified_at_utc": completed_at,
        "route_source": ROUTE_SOURCE,
        "endpoint": client.base_url,
        "requested_route": normalized_route,
        "provider_response_model": parsed["response_model"],
        "catalog_captured_at_utc": catalog.get("captured_at_utc"),
        "catalog_source_payload_sha256": catalog.get("source_payload_sha256"),
        "catalog_route_sha256": _sha256_json(catalog_row),
        "request": {
            "max_tokens": max_tokens,
            "stream": False,
            "request_sha256": _sha256_json(request_body),
            "prompt_sha256": hashlib.sha256(
                request_body["messages"][0]["content"].encode("utf-8")
            ).hexdigest(),
            "optional_generation_controls_asserted": [],
        },
        "response": {
            "request_id": parsed["request_id"],
            "payload_sha256": parsed["response_sha256"],
            "content_sha256": hashlib.sha256(
                str(parsed["content"]).encode("utf-8")
            ).hexdigest(),
            "finish_reason": parsed["finish_reason"],
            "truncated": parsed["truncated"],
            "usage": parsed["usage"],
            "usage_sha256": parsed["usage_sha256"],
        },
    }


def probe_catalog_routes(
    client: InferenceHubClient,
    *,
    catalog: Mapping[str, Any],
    attempt_ledger_path: Path,
    max_tokens: int = 8,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Attempt one minimal chat request against every authorized catalog route."""

    raw_rows = catalog.get("routes")
    if (
        catalog.get("schema_version") != CATALOG_SCHEMA_VERSION
        or catalog.get("source_endpoints") != ["/models"]
        or not isinstance(raw_rows, list)
        or catalog.get("route_count") != len(raw_rows)
    ):
        raise InferenceHubDiscoveryError("Catalog snapshot schema is invalid.")
    routes: list[str] = []
    for row in raw_rows:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("route"), str)
            or row.get("listed_by_models") is not True
        ):
            raise InferenceHubDiscoveryError("Catalog snapshot contains an invalid route.")
        routes.append(str(row["route"]))
    if len(routes) != len(set(routes)):
        raise InferenceHubDiscoveryError("Catalog snapshot contains duplicate routes.")
    ordered_routes = sorted(routes)

    def probe_one(route: str) -> dict[str, Any]:
        request_body = _chat_probe_request_body(route, max_tokens)
        attempt_id = _reserve_discovery_attempt(
            attempt_ledger_path,
            target_id=f"catalog-route:{route}",
            route=route,
            request_body=request_body,
            max_tokens=max_tokens,
        )
        try:
            evidence = chat_probe_route(
                client,
                catalog=catalog,
                route=route,
                max_tokens=max_tokens,
            )
        except (InferenceHubDiscoveryError, ValueError) as error:
            failure_code, http_status, failure_evidence = _failure_details(
                error,
                default_code="minimal_chat_probe_failed",
            )
            _finish_discovery_attempt(
                attempt_ledger_path,
                attempt_id=attempt_id,
                outcome="failed",
                failure_code=failure_code,
                request_id=None,
                http_status=http_status,
                response_evidence=failure_evidence,
            )
            rejection: dict[str, Any] = {
                "route": route,
                "failure_code": failure_code,
            }
            if http_status is not None:
                rejection["http_status"] = http_status
            if failure_evidence:
                rejection["failure_evidence"] = failure_evidence
            return {
                "attempt_id": attempt_id,
                "result_type": "rejected",
                "record": rejection,
            }
        response_evidence = _successful_response_evidence(evidence)
        _finish_discovery_attempt(
            attempt_ledger_path,
            attempt_id=attempt_id,
            outcome="chat_callable",
            failure_code=None,
            request_id=str(evidence["response"]["request_id"]),
            response_evidence=response_evidence,
        )
        return {
            "attempt_id": attempt_id,
            "result_type": "verified",
            "record": {"route": route, "evidence": evidence},
        }

    results = _bounded_ordered_map(
        probe_one,
        ordered_routes,
        max_workers=max_workers,
    )
    verified = [
        result["record"] for result in results if result["result_type"] == "verified"
    ]
    rejected = [
        result["record"] for result in results if result["result_type"] == "rejected"
    ]
    run_attempt_ids = [str(result["attempt_id"]) for result in results]
    payload: dict[str, Any] = {
        "schema_version": CATALOG_PROBE_SCHEMA_VERSION,
        "artifact_type": "inference_hub_full_catalog_chat_probe",
        "status": "complete",
        "completed_at_utc": _utc_now(),
        "endpoint": client.base_url,
        "catalog_captured_at_utc": catalog.get("captured_at_utc"),
        "catalog_source_payload_sha256": catalog.get("source_payload_sha256"),
        "catalog_route_count": len(routes),
        "attempted_route_count": len(routes),
        "chat_callable_route_count": len(verified),
        "rejected_route_count": len(rejected),
        "chat_callable_routes": verified,
        "rejected_routes": rejected,
        "optional_generation_controls_tested": [],
        "execution": _execution_metadata(
            max_workers=max_workers,
            item_count=len(ordered_routes),
            result_ordering="route_lexicographic",
        ),
        "run_attempt_ids": run_attempt_ids,
        "discovery_attempt_ledger": _discovery_ledger_reference(
            attempt_ledger_path
        ),
    }
    payload["bundle_sha256"] = _sha256_json(payload)
    return payload


def smoke_verify_cohorts(
    client: InferenceHubClient,
    *,
    catalog: Mapping[str, Any],
    cohort_ids: list[str],
    max_tokens: int = 16,
    attempt_ledger_path: Path | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Verify every exact route in one or more frozen registry cohorts.

    The returned bundle is created only after every route passes.  It contains
    sanitized, hash-bound evidence and exact cohort membership, but never a
    credential or generated response text.
    """

    if not cohort_ids:
        raise InferenceHubDiscoveryError("At least one cohort is required.")
    cohorts: list[dict[str, Any]] = []
    targets: list[tuple[str, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    seen_routes: set[str] = set()
    registry_versions: set[str] = set()
    registry_hashes: set[str] = set()
    routing_roster_hashes: set[str] = set()
    for cohort_id in cohort_ids:
        cohort = load_model_cohort(cohort_id)
        registry_versions.add(str(cohort["registry_version"]))
        registry_hashes.add(str(cohort["registry_hash"]))
        routing_roster_hashes.add(str(cohort["routing_roster_hash"]))
        cohort_target_ids: list[str] = []
        for target in cohort["targets"]:
            target_id = str(target["id"])
            route = str(target["route"])
            if str(target["provider"]) != "inference_hub":
                raise InferenceHubDiscoveryError(
                    f"Cohort target {target_id} is not routed through InferenceHub."
                )
            if target_id in seen_ids:
                raise InferenceHubDiscoveryError(
                    f"Target {target_id} is duplicated across requested cohorts."
                )
            if route in seen_routes:
                raise InferenceHubDiscoveryError(
                    f"Route {route} is duplicated across requested cohorts."
                )
            seen_ids.add(target_id)
            seen_routes.add(route)
            cohort_target_ids.append(target_id)
            targets.append((target_id, target))
        cohorts.append(
            {
                "id": str(cohort["id"]),
                "version": str(cohort["version"]),
                "target_ids": cohort_target_ids,
            }
        )
    if (
        len(registry_versions) != 1
        or len(registry_hashes) != 1
        or len(routing_roster_hashes) != 1
    ):
        raise InferenceHubDiscoveryError(
            "Requested cohorts do not share one exact registry version, hash, and "
            "routing roster."
        )

    def verify_one(
        target_row: tuple[str, dict[str, Any]],
    ) -> dict[str, Any]:
        target_id, target = target_row
        route = str(target["route"])
        attempt_id: str | None = None
        if attempt_ledger_path is not None:
            attempt_id = _reserve_discovery_attempt(
                attempt_ledger_path,
                target_id=target_id,
                route=route,
                request_body=_smoke_request_body(route, max_tokens),
                max_tokens=max_tokens,
            )
        try:
            evidence = smoke_verify_route(
                client,
                catalog=catalog,
                route=route,
                max_tokens=max_tokens,
            )
        except (InferenceHubDiscoveryError, ValueError) as error:
            default_code = (
                "catalog_route_ineligible"
                if any(
                    marker in str(error).casefold()
                    for marker in ("absent from", "not chat-capable", "does not occur")
                )
                else "route_verification_failed"
            )
            failure_code, http_status, failure_evidence = _failure_details(
                error,
                default_code=default_code,
            )
            if attempt_id is not None:
                _finish_discovery_attempt(
                    attempt_ledger_path,
                    attempt_id=attempt_id,
                    outcome="failed",
                    failure_code=failure_code,
                    request_id=None,
                    http_status=http_status,
                    response_evidence=failure_evidence,
                )
            rejection: dict[str, Any] = {
                "target_id": target_id,
                "route": route,
                "failure_code": failure_code,
            }
            if http_status is not None:
                rejection["http_status"] = http_status
            if failure_evidence:
                rejection["failure_evidence"] = failure_evidence
            return {"result_type": "rejected", "record": rejection}
        if attempt_id is not None:
            response_evidence = _successful_response_evidence(evidence)
            _finish_discovery_attempt(
                attempt_ledger_path,
                attempt_id=attempt_id,
                outcome="verified",
                failure_code=None,
                request_id=str(
                    evidence["verification_evidence"]["smoke_test"]["request_id"]
                ),
                response_evidence=response_evidence,
            )
        return {
            "result_type": "verified",
            "record": {
                "target_id": target_id,
                "upstream_provider": str(target["upstream_provider"]),
                "route": route,
                "evidence": evidence,
            },
        }

    results = _bounded_ordered_map(
        verify_one,
        targets,
        max_workers=max_workers,
    )
    verified_targets = [
        result["record"] for result in results if result["result_type"] == "verified"
    ]
    rejected_targets = [
        result["record"] for result in results if result["result_type"] == "rejected"
    ]
    selected_by_route = {str(target["route"]): target_id for target_id, target in targets}
    catalog_census: list[dict[str, Any]] = []
    for row in catalog.get("routes", []):
        if not isinstance(row, Mapping) or not isinstance(row.get("route"), str):
            raise InferenceHubDiscoveryError("Catalog census contains an invalid route row.")
        route = str(row["route"])
        if route in selected_by_route:
            decision = "included_frozen_panel"
            target_id = selected_by_route[route]
        else:
            decision = "excluded_outside_frozen_panel"
            target_id = None
        catalog_census.append(
            {"route": route, "decision": decision, "target_id": target_id}
        )
    if not set(selected_by_route).issubset(
        {str(row["route"]) for row in catalog_census}
    ):
        raise InferenceHubDiscoveryError(
            "Catalog census does not contain every frozen panel route."
        )
    ledger_reference: dict[str, Any] | None = None
    if attempt_ledger_path is not None:
        ledger_reference = _discovery_ledger_reference(attempt_ledger_path)
    payload: dict[str, Any] = {
        "schema_version": COHORT_EVIDENCE_SCHEMA_VERSION,
        "status": "verified" if not rejected_targets else "incomplete",
        "verified_at_utc": _utc_now(),
        "endpoint": client.base_url,
        "registry_version": next(iter(registry_versions)),
        "registry_hash": next(iter(registry_hashes)),
        "routing_roster_sha256": next(iter(routing_roster_hashes)),
        "catalog_source_payload_sha256": catalog.get("source_payload_sha256"),
        "cohorts": cohorts,
        "target_count": len(targets),
        "verified_target_count": len(verified_targets),
        "targets": verified_targets,
        "rejected_targets": rejected_targets,
        "execution": _execution_metadata(
            max_workers=max_workers,
            item_count=len(targets),
            result_ordering="requested_cohort_then_registry_target",
        ),
        "catalog_census": catalog_census,
        "catalog_census_sha256": _sha256_json(catalog_census),
        "discovery_attempt_ledger": ledger_reference,
    }
    payload["bundle_sha256"] = _sha256_json(payload)
    return payload


def _validate_reconciliation(
    reconciliation: Mapping[str, Any],
    *,
    catalog: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        reconciliation.get("schema_version") != 1
        or reconciliation.get("artifact_type")
        != "inference_hub_route_reconciliation"
    ):
        raise InferenceHubDiscoveryError("Route reconciliation schema is invalid.")
    embedded_hash = reconciliation.get("report_sha256")
    unhashed = {
        key: value for key, value in reconciliation.items() if key != "report_sha256"
    }
    if embedded_hash != _sha256_json(unhashed):
        raise InferenceHubDiscoveryError("Route reconciliation integrity failed.")
    if (
        reconciliation.get("catalog_source_payload_sha256")
        != catalog.get("source_payload_sha256")
        or reconciliation.get("catalog_route_count") != catalog.get("route_count")
    ):
        raise InferenceHubDiscoveryError(
            "Route reconciliation is not bound to the current catalog snapshot."
        )
    try:
        expected_reconciliation = reconcile_routes(
            catalog=catalog,
            registry=registry,
        )
    except RouteReconciliationError as error:
        raise InferenceHubDiscoveryError(
            "Catalog and registry cannot reproduce route reconciliation."
        ) from error
    if _canonical_bytes(reconciliation) != _canonical_bytes(expected_reconciliation):
        raise InferenceHubDiscoveryError(
            "Route reconciliation does not exactly reproduce from the supplied "
            "catalog and registry."
        )
    policy = reconciliation.get("selection_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("automatic_promotion") is not False
        or policy.get("live_smoke_required") is not True
    ):
        raise InferenceHubDiscoveryError("Route reconciliation policy is unsafe.")
    resolutions = reconciliation.get("resolutions")
    if not isinstance(resolutions, list):
        raise InferenceHubDiscoveryError("Route reconciliation lacks resolutions[].")
    selected: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    seen_routes: set[str] = set()
    for index, row in enumerate(resolutions):
        if not isinstance(row, Mapping):
            raise InferenceHubDiscoveryError(
                f"Route reconciliation row {index} is invalid."
            )
        target_id = row.get("target_id")
        status = row.get("status")
        route = row.get("selected_candidate")
        candidates = row.get("all_exact_suffix_candidates")
        if not isinstance(target_id, str) or not target_id or target_id in seen_targets:
            raise InferenceHubDiscoveryError(
                "Route reconciliation target identifiers are invalid."
            )
        seen_targets.add(target_id)
        if status == "candidate_selected_smoke_pending":
            if (
                not isinstance(route, str)
                or not route
                or not isinstance(candidates, list)
                or route not in candidates
                or route in seen_routes
            ):
                raise InferenceHubDiscoveryError(
                    f"Selected reconciliation candidate is invalid for {target_id}."
                )
            _catalog_route(catalog, route)
            seen_routes.add(route)
            selected.append(dict(row))
        elif status == "unresolved_no_exact_suffix":
            if route is not None or candidates != []:
                raise InferenceHubDiscoveryError(
                    f"Unresolved reconciliation row is invalid for {target_id}."
                )
            unresolved.append(dict(row))
        else:
            raise InferenceHubDiscoveryError(
                f"Unknown reconciliation status for {target_id}."
            )
    if (
        reconciliation.get("target_count") != len(resolutions)
        or reconciliation.get("candidate_selected_count") != len(selected)
        or reconciliation.get("unresolved_count") != len(unresolved)
    ):
        raise InferenceHubDiscoveryError("Route reconciliation counts are invalid.")
    return selected, unresolved


def smoke_verify_reconciled_candidates(
    client: InferenceHubClient,
    *,
    catalog: Mapping[str, Any],
    registry: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    attempt_ledger_path: Path,
    max_tokens: int = 16,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Attempt every exact-suffix candidate without mutating the registry."""

    selected, unresolved = _validate_reconciliation(
        reconciliation,
        catalog=catalog,
        registry=registry,
    )
    if not selected:
        raise InferenceHubDiscoveryError(
            "Route reconciliation contains no smoke-pending candidates."
        )
    def verify_one(row: dict[str, Any]) -> dict[str, Any]:
        target_id = str(row["target_id"])
        route = str(row["selected_candidate"])
        attempt_id = _reserve_discovery_attempt(
            attempt_ledger_path,
            target_id=target_id,
            route=route,
            request_body=_smoke_request_body(route, max_tokens),
            max_tokens=max_tokens,
        )
        try:
            evidence = smoke_verify_route(
                client,
                catalog=catalog,
                route=route,
                max_tokens=max_tokens,
            )
        except (InferenceHubDiscoveryError, ValueError) as error:
            failure_code, http_status, failure_evidence = _failure_details(
                error,
                default_code="candidate_route_verification_failed",
            )
            _finish_discovery_attempt(
                attempt_ledger_path,
                attempt_id=attempt_id,
                outcome="failed",
                failure_code=failure_code,
                request_id=None,
                http_status=http_status,
                response_evidence=failure_evidence,
            )
            rejection: dict[str, Any] = {
                "target_id": target_id,
                "route": route,
                "failure_code": failure_code,
            }
            if http_status is not None:
                rejection["http_status"] = http_status
            if failure_evidence:
                rejection["failure_evidence"] = failure_evidence
            return {"result_type": "rejected", "record": rejection}
        request_id = str(evidence["verification_evidence"]["smoke_test"]["request_id"])
        _finish_discovery_attempt(
            attempt_ledger_path,
            attempt_id=attempt_id,
            outcome="verified",
            failure_code=None,
            request_id=request_id,
            response_evidence=_successful_response_evidence(evidence),
        )
        return {
            "result_type": "verified",
            "record": {
                "target_id": target_id,
                "upstream_provider": str(row["upstream_provider"]),
                "planned_route": str(row["planned_route"]),
                "route": route,
                "alternative_exact_suffix_candidates": [
                    candidate
                    for candidate in row["all_exact_suffix_candidates"]
                    if candidate != route
                ],
                "evidence": evidence,
            },
        }

    results = _bounded_ordered_map(
        verify_one,
        selected,
        max_workers=max_workers,
    )
    verified_targets = [
        result["record"] for result in results if result["result_type"] == "verified"
    ]
    rejected_targets = [
        result["record"] for result in results if result["result_type"] == "rejected"
    ]
    ledger_reference = _discovery_ledger_reference(attempt_ledger_path)
    selected_routes = {
        str(row["selected_candidate"]): str(row["target_id"]) for row in selected
    }
    catalog_census = [
        {
            "route": str(row["route"]),
            "decision": (
                "selected_exact_suffix_candidate"
                if str(row["route"]) in selected_routes
                else "not_selected_by_exact_suffix_policy"
            ),
            "target_id": selected_routes.get(str(row["route"])),
        }
        for row in catalog.get("routes", [])
        if isinstance(row, Mapping) and isinstance(row.get("route"), str)
    ]
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_EVIDENCE_SCHEMA_VERSION,
        "artifact_type": "inference_hub_candidate_smoke_evidence",
        "status": "verified" if not rejected_targets else "incomplete",
        "verified_at_utc": _utc_now(),
        "endpoint": client.base_url,
        "catalog_source_payload_sha256": catalog.get("source_payload_sha256"),
        "reconciliation_report_sha256": reconciliation.get("report_sha256"),
        "registry_version": reconciliation.get("registry_version"),
        "automatic_registry_promotion": False,
        "candidate_target_count": len(selected),
        "verified_target_count": len(verified_targets),
        "targets": verified_targets,
        "rejected_targets": rejected_targets,
        "execution": _execution_metadata(
            max_workers=max_workers,
            item_count=len(selected),
            result_ordering="reconciliation_resolution_order",
        ),
        "unresolved_targets": [
            {
                "target_id": str(row["target_id"]),
                "planned_route": str(row["planned_route"]),
                "status": str(row["status"]),
            }
            for row in unresolved
        ],
        "catalog_census": catalog_census,
        "catalog_census_sha256": _sha256_json(catalog_census),
        "discovery_attempt_ledger": ledger_reference,
    }
    payload["bundle_sha256"] = _sha256_json(payload)
    return payload


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InferenceHubDiscoveryError(
            f"{label} is not readable UTF-8 JSON."
        ) from error
    if not isinstance(payload, dict):
        raise InferenceHubDiscoveryError(f"{label} must be a JSON object.")
    return payload


def _client_from_environment(timeout_seconds: float) -> InferenceHubClient:
    load_dotenv()
    return InferenceHubClient(
        api_key=os.getenv("NVIDIA_API_KEY", ""),
        base_url=os.getenv("INFERENCE_HUB_BASE_URL", DEFAULT_BASE_URL),
        timeout_seconds=timeout_seconds,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture and smoke-test exact NVIDIA InferenceHub routes."
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog_parser = subparsers.add_parser("catalog")
    catalog_parser.add_argument("--output", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--route", required=True)
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--catalog-output", type=Path)
    verify_parser.add_argument("--max-tokens", type=int, default=16)

    cohorts_parser = subparsers.add_parser("verify-cohorts")
    cohorts_parser.add_argument("--cohort", action="append", required=True)
    cohorts_parser.add_argument("--output", type=Path, required=True)
    cohorts_parser.add_argument("--catalog-output", type=Path)
    cohorts_parser.add_argument("--max-tokens", type=int, default=16)
    cohorts_parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=DEFAULT_MAX_WORKERS,
    )
    cohorts_parser.add_argument(
        "--attempt-ledger",
        type=Path,
        required=True,
        help="Durable pre-dispatch ledger retained across failed verification runs.",
    )

    candidates_parser = subparsers.add_parser("verify-candidates")
    candidates_parser.add_argument("--catalog-input", type=Path, required=True)
    candidates_parser.add_argument("--registry-input", type=Path, required=True)
    candidates_parser.add_argument("--reconciliation", type=Path, required=True)
    candidates_parser.add_argument("--output", type=Path, required=True)
    candidates_parser.add_argument("--max-tokens", type=int, default=16)
    candidates_parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=DEFAULT_MAX_WORKERS,
    )
    candidates_parser.add_argument(
        "--attempt-ledger",
        type=Path,
        required=True,
        help="Durable pre-dispatch ledger retained across failed verification runs.",
    )

    probe_parser = subparsers.add_parser("probe-catalog")
    probe_parser.add_argument("--catalog-input", type=Path, required=True)
    probe_parser.add_argument("--output", type=Path, required=True)
    probe_parser.add_argument("--max-tokens", type=int, default=8)
    probe_parser.add_argument(
        "--max-workers",
        type=_positive_int,
        default=DEFAULT_MAX_WORKERS,
    )
    probe_parser.add_argument(
        "--attempt-ledger",
        type=Path,
        required=True,
        help="Durable pre-dispatch ledger for every catalog-route probe.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    client = _client_from_environment(args.timeout_seconds)
    catalog = (
        _read_json_object(args.catalog_input, label="catalog snapshot")
        if args.command in {"verify-candidates", "probe-catalog"}
        else capture_catalog(client)
    )
    if args.command == "catalog":
        _atomic_write_json(args.output, catalog)
        print(f"Captured {catalog['route_count']} exact InferenceHub routes.")
        print(f"Snapshot: {args.output}")
        return 0

    if getattr(args, "catalog_output", None) is not None:
        _atomic_write_json(args.catalog_output, catalog)
    if args.command == "verify-cohorts":
        evidence = smoke_verify_cohorts(
            client,
            catalog=catalog,
            cohort_ids=args.cohort,
            max_tokens=args.max_tokens,
            attempt_ledger_path=args.attempt_ledger,
            max_workers=args.max_workers,
        )
        _atomic_write_json(args.output, evidence)
        if evidence["status"] != "verified":
            raise InferenceHubDiscoveryError(
                "One or more frozen panel routes failed verification; incomplete "
                f"evidence was retained at {args.output}."
            )
        print(
            f"Verified {evidence['target_count']} exact InferenceHub routes "
            f"across {len(evidence['cohorts'])} cohort(s)."
        )
        print(f"Evidence bundle: {args.output}")
        return 0
    if args.command == "verify-candidates":
        registry = _read_json_object(
            args.registry_input,
            label="model registry",
        )
        reconciliation = _read_json_object(
            args.reconciliation,
            label="route reconciliation",
        )
        evidence = smoke_verify_reconciled_candidates(
            client,
            catalog=catalog,
            registry=registry,
            reconciliation=reconciliation,
            max_tokens=args.max_tokens,
            attempt_ledger_path=args.attempt_ledger,
            max_workers=args.max_workers,
        )
        _atomic_write_json(args.output, evidence)
        if evidence["status"] != "verified":
            raise InferenceHubDiscoveryError(
                "One or more exact-suffix candidates failed verification; "
                f"incomplete evidence was retained at {args.output}."
            )
        print(
            f"Verified {evidence['candidate_target_count']} exact-suffix "
            "InferenceHub candidates."
        )
        print(f"Evidence bundle: {args.output}")
        return 0
    if args.command == "probe-catalog":
        evidence = probe_catalog_routes(
            client,
            catalog=catalog,
            max_tokens=args.max_tokens,
            attempt_ledger_path=args.attempt_ledger,
            max_workers=args.max_workers,
        )
        _atomic_write_json(args.output, evidence)
        print(
            f"Probed {evidence['attempted_route_count']} catalog routes; "
            f"{evidence['chat_callable_route_count']} were minimally chat-callable."
        )
        print(f"Evidence bundle: {args.output}")
        return 0
    evidence = smoke_verify_route(
        client,
        catalog=catalog,
        route=args.route,
        max_tokens=args.max_tokens,
    )
    _atomic_write_json(args.output, evidence)
    print(f"Verified exact InferenceHub route: {evidence['requested_route']}")
    print(f"Evidence: {args.output}")
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Run the CLI without exposing tracebacks or credential-bearing objects."""

    try:
        return main(argv)
    except (InferenceHubDiscoveryError, ValueError) as error:
        print(f"InferenceHub discovery failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
