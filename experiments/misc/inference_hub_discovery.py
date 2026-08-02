"""Discover and verify exact NVIDIA InferenceHub model routes.

This module deliberately does not mutate the experiment model registry. It
captures a sanitized, hash-bound catalog snapshot and can produce a separate
smoke-test evidence record. A registry route should be promoted only after the
evidence record has been reviewed and pinned in version control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import certifi
from dotenv import load_dotenv

from agents.agent_config import load_model_cohort, validate_endpoint_base_url

DEFAULT_BASE_URL = "https://inference-api.nvidia.com/v1"
ROUTE_SOURCE = "inference_hub_models_api"
CATALOG_SCHEMA_VERSION = 2
SMOKE_SCHEMA_VERSION = 2
COHORT_EVIDENCE_SCHEMA_VERSION = 2
DISCOVERY_LEDGER_SCHEMA_VERSION = 1
SMOKE_SEED = 20_260_801
SMOKE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "string", "enum": ["OK"]}},
    "required": ["ok"],
    "additionalProperties": False,
}


class InferenceHubDiscoveryError(RuntimeError):
    """A safe, credential-free error raised during route discovery."""


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


def _load_discovery_ledger(path: Path) -> dict[str, Any]:
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


def _reserve_discovery_attempt(
    path: Path,
    *,
    target_id: str,
    route: str,
    request_body: Mapping[str, Any],
    max_tokens: int,
) -> str:
    ledger = _load_discovery_ledger(path)
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
            "request_id": None,
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
) -> None:
    ledger = _load_discovery_ledger(path)
    matches = [
        record
        for record in ledger["records"]
        if isinstance(record, dict) and record.get("attempt_id") == attempt_id
    ]
    if len(matches) != 1 or matches[0].get("outcome") != "reserved_before_dispatch":
        raise InferenceHubDiscoveryError(
            "Discovery attempt completion does not match one pending reservation."
        )
    matches[0]["outcome"] = outcome
    matches[0]["failure_code"] = failure_code
    matches[0]["request_id"] = request_id
    ledger.pop("ledger_sha256", None)
    ledger["ledger_sha256"] = _discovery_ledger_hash(ledger)
    _atomic_write_json(path, ledger)


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
            with urllib.request.urlopen(  # noqa: S310 - exact host checked above
                request,
                timeout=self.timeout_seconds,
                context=tls_context,
            ) as response:
                payload_bytes = response.read()
        except urllib.error.HTTPError as error:
            raise InferenceHubDiscoveryError(
                f"InferenceHub {path} returned HTTP {error.code}."
            ) from error
        except urllib.error.URLError as error:
            reason = type(error.reason).__name__
            raise InferenceHubDiscoveryError(
                f"InferenceHub {path} connection failed ({reason})."
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
    request_id = response.get("id")
    response_model = response.get("model")
    choices = response.get("choices")
    if not isinstance(request_id, str) or not request_id.strip():
        raise InferenceHubDiscoveryError(
            "Smoke response did not report a request/completion id."
        )
    if not isinstance(response_model, str) or not response_model.strip():
        raise InferenceHubDiscoveryError(
            "Smoke response did not report a provider response model."
        )
    if response_model.strip() != normalized_route:
        raise InferenceHubDiscoveryError(
            "Smoke response model identity does not match the exact requested route."
        )
    if not isinstance(choices, list) or len(choices) != 1:
        raise InferenceHubDiscoveryError(
            "Smoke response must contain exactly one completion choice."
        )
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise InferenceHubDiscoveryError("Smoke completion choice is not an object.")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise InferenceHubDiscoveryError("Smoke completion lacks a message object.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise InferenceHubDiscoveryError("Smoke completion content is empty.")
    try:
        structured_content = json.loads(content)
    except json.JSONDecodeError as error:
        raise InferenceHubDiscoveryError(
            "Smoke completion did not honor structured JSON output."
        ) from error
    if structured_content != {"ok": "OK"}:
        raise InferenceHubDiscoveryError(
            "Smoke completion did not satisfy the exact response schema."
        )
    finish_reason = choice.get("finish_reason")
    if not isinstance(finish_reason, str) or not finish_reason.strip():
        raise InferenceHubDiscoveryError("Smoke completion lacks a finish reason.")
    if finish_reason.strip().lower() in {"length", "max_tokens", "max_output_tokens"}:
        raise InferenceHubDiscoveryError("Smoke completion was truncated.")
    usage = response.get("usage")
    if not isinstance(usage, Mapping) or not usage:
        raise InferenceHubDiscoveryError("Smoke completion lacks token usage.")
    safe_usage = {
        key: value
        for key, value in (usage.items() if isinstance(usage, Mapping) else [])
        if isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
    }
    if not safe_usage or not any(
        safe_usage.get(key, 0) > 0
        for key in ("completion_tokens", "output_tokens", "total_tokens")
    ):
        raise InferenceHubDiscoveryError(
            "Smoke completion lacks positive output/total token usage."
        )
    usage_sha256 = _sha256_json(safe_usage)
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
    response_sha256 = _sha256_json(response)
    catalog_route_sha256 = _sha256_json(catalog_row)
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "verification_status": "verified",
        "verified_at_utc": completed_at_utc,
        "route_source": ROUTE_SOURCE,
        "endpoint": client.base_url,
        "requested_route": normalized_route,
        "provider_response_model": response_model.strip(),
        "catalog_captured_at_utc": catalog.get("captured_at_utc"),
        "catalog_source_payload_sha256": catalog.get("source_payload_sha256"),
        "catalog_route_sha256": catalog_route_sha256,
        "verification_evidence": {
            "verified_at_utc": completed_at_utc,
            "discovery_sha256": catalog_route_sha256,
            "smoke_test": {
                "completed_at_utc": completed_at_utc,
                "request_id": request_id.strip(),
                "response_model": response_model.strip(),
                "response_sha256": response_sha256,
                "finish_reason": finish_reason.strip(),
                "usage_sha256": usage_sha256,
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
            "payload_sha256": response_sha256,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "finish_reason": finish_reason.strip(),
            "usage": safe_usage,
            "usage_sha256": usage_sha256,
            "structured_output_validated": True,
        },
    }


def smoke_verify_cohorts(
    client: InferenceHubClient,
    *,
    catalog: Mapping[str, Any],
    cohort_ids: list[str],
    max_tokens: int = 16,
    attempt_ledger_path: Path | None = None,
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

    verified_targets: list[dict[str, Any]] = []
    rejected_targets: list[dict[str, str]] = []
    for target_id, target in targets:
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
            failure_code = (
                "catalog_route_ineligible"
                if any(
                    marker in str(error).casefold()
                    for marker in ("absent from", "not chat-capable", "does not occur")
                )
                else "route_verification_failed"
            )
            if attempt_id is not None:
                _finish_discovery_attempt(
                    attempt_ledger_path,
                    attempt_id=attempt_id,
                    outcome="failed",
                    failure_code=failure_code,
                    request_id=None,
                )
            rejected_targets.append(
                {
                    "target_id": target_id,
                    "route": route,
                    "failure_code": failure_code,
                }
            )
            continue
        if attempt_id is not None:
            _finish_discovery_attempt(
                attempt_ledger_path,
                attempt_id=attempt_id,
                outcome="verified",
                failure_code=None,
                request_id=str(
                    evidence["verification_evidence"]["smoke_test"]["request_id"]
                ),
            )
        verified_targets.append(
            {
                "target_id": target_id,
                "upstream_provider": str(target["upstream_provider"]),
                "route": route,
                "evidence": evidence,
            }
        )
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
        ledger = _load_discovery_ledger(attempt_ledger_path)
        ledger_reference = {
            "path": str(attempt_ledger_path.resolve()),
            "sha256": hashlib.sha256(attempt_ledger_path.read_bytes()).hexdigest(),
            "ledger_sha256": ledger["ledger_sha256"],
            "record_count": len(ledger["records"]),
        }
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
        "catalog_census": catalog_census,
        "catalog_census_sha256": _sha256_json(catalog_census),
        "discovery_attempt_ledger": ledger_reference,
    }
    payload["bundle_sha256"] = _sha256_json(payload)
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
        "--attempt-ledger",
        type=Path,
        required=True,
        help="Durable pre-dispatch ledger retained across failed verification runs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    client = _client_from_environment(args.timeout_seconds)
    catalog = capture_catalog(client)
    if args.command == "catalog":
        _atomic_write_json(args.output, catalog)
        print(f"Captured {catalog['route_count']} exact InferenceHub routes.")
        print(f"Snapshot: {args.output}")
        return 0

    if args.catalog_output is not None:
        _atomic_write_json(args.catalog_output, catalog)
    if args.command == "verify-cohorts":
        evidence = smoke_verify_cohorts(
            client,
            catalog=catalog,
            cohort_ids=args.cohort,
            max_tokens=args.max_tokens,
            attempt_ledger_path=args.attempt_ledger,
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
