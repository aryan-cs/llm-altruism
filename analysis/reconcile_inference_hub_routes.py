"""Build an outcome-blind candidate-route report from an authenticated catalog.

This utility does not mutate the model registry and does not call a provider.
It selects a preferred candidate only when an authenticated route has the exact
planned model suffix.  Renamed, versionless, or merely similar routes remain
unresolved for explicit review and live smoke verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


REPORT_SCHEMA_VERSION = 1
BACKEND_PRIORITY = (
    "openai/openai/",
    "gcp/google/",
    "azure/anthropic/",
    "nvidia/",
    "azure/",
    "aws/",
    "nvcf/",
    "us/azure/",
    "switchyard/",
    "nvidia_dynamo/",
)


class RouteReconciliationError(ValueError):
    """An input artifact cannot support deterministic reconciliation."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RouteReconciliationError(f"{label} is not readable UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise RouteReconciliationError(f"{label} must be a JSON object.")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
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


def _backend_rank(route: str) -> tuple[int, str]:
    for index, prefix in enumerate(BACKEND_PRIORITY):
        if route.startswith(prefix):
            return index, route
    return len(BACKEND_PRIORITY), route


def _catalog_routes(catalog: Mapping[str, Any]) -> list[str]:
    if catalog.get("schema_version") != 2:
        raise RouteReconciliationError("Catalog schema_version must be 2.")
    if catalog.get("source_endpoints") != ["/models"]:
        raise RouteReconciliationError("Catalog must come from the authorized /models API.")
    raw_routes = catalog.get("routes")
    if not isinstance(raw_routes, list):
        raise RouteReconciliationError("Catalog routes must be a list.")
    routes: list[str] = []
    for index, row in enumerate(raw_routes):
        if not isinstance(row, Mapping):
            raise RouteReconciliationError(f"Catalog route row {index} is invalid.")
        route = row.get("route")
        if not isinstance(route, str) or not route.strip():
            raise RouteReconciliationError(f"Catalog route row {index} lacks a route.")
        if row.get("listed_by_models") is not True:
            raise RouteReconciliationError(
                f"Catalog route row {index} is not authorized by /models."
            )
        routes.append(route.strip())
    if len(routes) != len(set(routes)):
        raise RouteReconciliationError("Catalog contains duplicate routes.")
    if catalog.get("route_count") != len(routes):
        raise RouteReconciliationError("Catalog route_count does not match routes.")
    return routes


def _registry_targets(registry: Mapping[str, Any]) -> list[dict[str, str]]:
    targets = registry.get("targets")
    cohorts = registry.get("cohorts")
    if not isinstance(targets, list) or not isinstance(cohorts, Mapping):
        raise RouteReconciliationError("Registry lacks targets or cohorts.")
    cohort_by_target: dict[str, str] = {}
    for cohort_id, cohort in cohorts.items():
        if not isinstance(cohort_id, str) or not isinstance(cohort, Mapping):
            raise RouteReconciliationError("Registry contains an invalid cohort.")
        target_ids = cohort.get("targets")
        if not isinstance(target_ids, list):
            raise RouteReconciliationError(f"Cohort {cohort_id} targets must be a list.")
        for target_id in target_ids:
            if not isinstance(target_id, str) or target_id in cohort_by_target:
                raise RouteReconciliationError("Registry cohort membership is invalid.")
            cohort_by_target[target_id] = cohort_id
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise RouteReconciliationError(f"Registry target {index} is invalid.")
        target_id = target.get("id")
        route = target.get("route")
        if not isinstance(target_id, str) or not target_id or target_id in seen:
            raise RouteReconciliationError(f"Registry target {index} has an invalid id.")
        if not isinstance(route, str) or not route.strip():
            raise RouteReconciliationError(f"Registry target {target_id} lacks a route.")
        if target_id not in cohort_by_target:
            raise RouteReconciliationError(f"Registry target {target_id} lacks a cohort.")
        seen.add(target_id)
        rows.append(
            {
                "target_id": target_id,
                "cohort": cohort_by_target[target_id],
                "upstream_provider": str(target.get("upstream_provider", "")),
                "planned_route": route.strip(),
            }
        )
    if set(cohort_by_target) != seen:
        raise RouteReconciliationError("A cohort references an unknown registry target.")
    return rows


def reconcile_routes(
    *, catalog: Mapping[str, Any], registry: Mapping[str, Any]
) -> dict[str, Any]:
    routes = _catalog_routes(catalog)
    targets = _registry_targets(registry)
    resolutions: list[dict[str, Any]] = []
    for target in targets:
        planned = target["planned_route"]
        exact_suffix = sorted(
            (route for route in routes if route == planned or route.endswith(f"/{planned}")),
            key=_backend_rank,
        )
        selected = exact_suffix[0] if exact_suffix else None
        resolutions.append(
            {
                **target,
                "status": (
                    "candidate_selected_smoke_pending"
                    if selected is not None
                    else "unresolved_no_exact_suffix"
                ),
                "selected_candidate": selected,
                "selection_basis": (
                    "exact_model_suffix_then_frozen_backend_priority"
                    if selected is not None
                    else None
                ),
                "all_exact_suffix_candidates": exact_suffix,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_type": "inference_hub_route_reconciliation",
        "catalog_captured_at_utc": catalog.get("captured_at_utc"),
        "catalog_source_payload_sha256": catalog.get("source_payload_sha256"),
        "catalog_route_count": len(routes),
        "registry_version": registry.get("registry_version"),
        "selection_policy": {
            "match": "exact planned route or slash-delimited exact suffix only",
            "backend_priority": list(BACKEND_PRIORITY),
            "automatic_promotion": False,
            "live_smoke_required": True,
            "renamed_or_versionless_aliases": "unresolved",
        },
        "target_count": len(resolutions),
        "candidate_selected_count": sum(
            row["selected_candidate"] is not None for row in resolutions
        ),
        "unresolved_count": sum(
            row["selected_candidate"] is None for row in resolutions
        ),
        "resolutions": resolutions,
    }
    payload["report_sha256"] = _sha256_json(payload)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconcile planned model labels with an authenticated catalog."
    )
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    catalog = _read_json(args.catalog, label="catalog")
    registry = _read_json(args.registry, label="registry")
    report = reconcile_routes(catalog=catalog, registry=registry)
    _atomic_write_json(args.output, report)
    print(
        f"Selected {report['candidate_selected_count']} exact-suffix candidates; "
        f"{report['unresolved_count']} targets remain unresolved."
    )
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
