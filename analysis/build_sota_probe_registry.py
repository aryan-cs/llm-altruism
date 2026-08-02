"""Build an exploratory probe registry from a sealed SOTA roster artifact.

This bridge deliberately does not read or modify the production registry.  Its
output has the small registry surface consumed by route reconciliation and the
Inference Hub compatibility probe, while carrying explicit prohibitions
against confirmatory or paper-result use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ROSTER_SCHEMA_VERSION = 1
ROSTER_ARTIFACT_TYPE = (
    "inference_hub_sota_text_chat_roster_and_reconciliation"
)
ARTIFACT_TYPE = "exploratory_sota_inference_hub_probe_registry"
COHORT_ID = "exploratory_sota"
ALLOWED_ROLES = frozenset({"general_chat", "coding_chat", "search_chat"})
FORBIDDEN_IDENTITY_MARKERS = ("evals", "judge", "guard", "safety")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ProbeRegistryBuildError(ValueError):
    """A roster cannot safely produce an exploratory probe registry."""


# A descriptive alias for callers that prefer the artifact-specific name.
SotaProbeRegistryError = ProbeRegistryBuildError


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ProbeRegistryBuildError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeRegistryBuildError(
            f"{label} is not readable UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise ProbeRegistryBuildError(f"{label} must be a JSON object")
    return value


def _source_hashes(roster: Mapping[str, Any]) -> tuple[str, Any]:
    catalog = roster.get("catalog")
    if not isinstance(catalog, Mapping):
        raise ProbeRegistryBuildError("roster catalog provenance is missing")
    catalog_input = _require_sha256(
        catalog.get("input_sha256"), label="roster catalog.input_sha256"
    )
    source_payload = catalog.get("source_payload_sha256")
    if not isinstance(source_payload, Mapping) or not source_payload:
        raise ProbeRegistryBuildError(
            "roster catalog.source_payload_sha256 must be a non-empty object"
        )
    for name, digest in source_payload.items():
        if not isinstance(name, str) or not name:
            raise ProbeRegistryBuildError(
                "roster catalog source hash names must be non-empty strings"
            )
        _require_sha256(digest, label=f"catalog source payload {name!r}")
    return catalog_input, dict(source_payload)


def _validated_subjects(roster: Mapping[str, Any]) -> list[dict[str, str]]:
    raw_subjects = roster.get("subject_roster")
    inventory = roster.get("route_inventory")
    summary = roster.get("summary")
    if not isinstance(raw_subjects, list) or not isinstance(inventory, list):
        raise ProbeRegistryBuildError(
            "roster must contain subject_roster and route_inventory lists"
        )
    if not isinstance(summary, Mapping):
        raise ProbeRegistryBuildError("roster summary is missing")
    if summary.get("subject_count") != len(raw_subjects):
        raise ProbeRegistryBuildError("roster summary subject_count does not match")

    inventory_by_route: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(inventory):
        if not isinstance(row, Mapping):
            raise ProbeRegistryBuildError(
                f"roster route_inventory row {index} is invalid"
            )
        route = row.get("route")
        if not isinstance(route, str) or not route or route in inventory_by_route:
            raise ProbeRegistryBuildError(
                f"roster route_inventory row {index} has an invalid route"
            )
        inventory_by_route[route] = row

    subjects: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str]] = set()
    selected_routes: set[str] = set()
    for index, subject in enumerate(raw_subjects):
        if not isinstance(subject, Mapping):
            raise ProbeRegistryBuildError(f"roster subject {index} is invalid")
        subject_id = subject.get("subject_id")
        upstream = subject.get("upstream_provider")
        model = subject.get("model_identifier")
        preferred = subject.get("preferred_candidate")
        role = subject.get("role")
        candidates = subject.get("all_exact_candidate_backends")
        values = (subject_id, upstream, model, preferred)
        if not all(
            isinstance(value, str) and value and value == value.strip()
            for value in values
        ):
            raise ProbeRegistryBuildError(
                f"roster subject {index} lacks an exact identity or route"
            )
        assert isinstance(subject_id, str)
        assert isinstance(upstream, str)
        assert isinstance(model, str)
        assert isinstance(preferred, str)
        if subject_id != f"{upstream}/{model}":
            raise ProbeRegistryBuildError(
                f"roster subject {subject_id!r} does not match provider/model identity"
            )
        if subject_id in seen_ids or (upstream, model) in seen_identities:
            raise ProbeRegistryBuildError(f"duplicate roster identity: {subject_id}")
        seen_ids.add(subject_id)
        seen_identities.add((upstream, model))
        if subject.get("modality") != "text" or role not in ALLOWED_ROLES:
            raise ProbeRegistryBuildError(
                f"roster subject {subject_id} is not an allowed text-chat subject"
            )
        identity_text = "/".join(values).casefold()
        if any(marker in identity_text for marker in FORBIDDEN_IDENTITY_MARKERS):
            raise ProbeRegistryBuildError(
                f"roster subject {subject_id} has a forbidden evaluation/safety identity"
            )
        if (
            not isinstance(candidates, list)
            or not candidates
            or len(candidates) != len(set(candidates))
            or preferred != candidates[0]
        ):
            raise ProbeRegistryBuildError(
                f"roster subject {subject_id} has invalid exact candidate backends"
            )
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                raise ProbeRegistryBuildError(
                    f"roster subject {subject_id} has an invalid candidate route"
                )
            parts = candidate.split("/")
            if len(parts) < 3 or parts[-2:] != [upstream, model]:
                raise ProbeRegistryBuildError(
                    f"candidate {candidate!r} does not preserve {subject_id} identity"
                )
            inventory_row = inventory_by_route.get(candidate)
            if (
                inventory_row is None
                or inventory_row.get("upstream_provider") != upstream
                or inventory_row.get("model_identifier") != model
                or inventory_row.get("modality") != "text"
                or inventory_row.get("role") != role
                or inventory_row.get("eligible_text_chat") is not True
            ):
                raise ProbeRegistryBuildError(
                    f"candidate {candidate!r} is not an eligible matching inventory row"
                )
            selected_routes.add(candidate)
        subjects.append(
            {
                "id": subject_id,
                "provider": "inference_hub",
                "upstream_provider": upstream,
                "model": model,
                "route": preferred,
                "endpoint_profile": "inference_hub",
                "verification_status": "unverified",
                "route_source": "sealed_sota_roster_preferred_candidate",
            }
        )

    if summary.get("subject_backend_route_count") != len(selected_routes):
        raise ProbeRegistryBuildError(
            "roster summary subject_backend_route_count does not match"
        )
    return sorted(subjects, key=lambda row: row["id"])


def build_probe_registry(*, roster: Mapping[str, Any]) -> dict[str, Any]:
    """Return a sealed exploratory registry for compatibility probing only."""

    if (
        roster.get("schema_version") != ROSTER_SCHEMA_VERSION
        or roster.get("artifact_type") != ROSTER_ARTIFACT_TYPE
    ):
        raise ProbeRegistryBuildError(
            "roster must be a schema-v1 SOTA text-chat roster artifact"
        )
    roster_hash = _require_sha256(
        roster.get("artifact_sha256"), label="roster artifact_sha256"
    )
    unhashed_roster = {
        key: value for key, value in roster.items() if key != "artifact_sha256"
    }
    if roster_hash != _sha256_json(unhashed_roster):
        raise ProbeRegistryBuildError("roster artifact_sha256 is invalid")
    catalog_input_hash, catalog_payload_hashes = _source_hashes(roster)
    policy = roster.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("automatic_registry_mutation") is not False
        or policy.get("live_smoke_required") is not True
        or policy.get("identity_equivalence_inference") is not False
    ):
        raise ProbeRegistryBuildError("roster exploratory safety policy is invalid")
    targets = _validated_subjects(roster)
    target_ids = [target["id"] for target in targets]
    registry_version = f"exploratory-sota-{roster_hash[:16]}"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "registry_version": registry_version,
        "default_cohort": COHORT_ID,
        "analysis_role": "exploratory_sota_compatibility_probe_only",
        "confirmatory_promotion_permitted": False,
        "paper_result_promotion_permitted": False,
        "production_registry_mutation_permitted": False,
        "source_roster_artifact_sha256": roster_hash,
        "source_catalog_input_sha256": catalog_input_hash,
        "source_catalog_payload_sha256": catalog_payload_hashes,
        "probe_policy": {
            "live_compatibility_probe_required": True,
            "automatic_route_promotion": False,
            "confirmatory_use_permitted": False,
            "paper_result_use_permitted": False,
            "production_registry_mutation_permitted": False,
        },
        "cohorts": {
            COHORT_ID: {
                "version": registry_version,
                "analysis_role": "exploratory_sota_compatibility_probe_only",
                "targets": target_ids,
            }
        },
        "targets": targets,
    }
    payload["artifact_sha256"] = _sha256_json(payload)
    return payload


# Keep the full filename-derived spelling available as a convenient API.
build_sota_probe_registry = build_probe_registry


def validate_probe_registry(registry: Mapping[str, Any]) -> None:
    """Fail closed unless ``registry`` is an intact exploratory bridge output."""

    if (
        registry.get("schema_version") != SCHEMA_VERSION
        or registry.get("artifact_type") != ARTIFACT_TYPE
    ):
        raise ProbeRegistryBuildError("probe registry schema or artifact type is invalid")
    recorded_hash = _require_sha256(
        registry.get("artifact_sha256"), label="probe registry artifact_sha256"
    )
    unhashed = {
        key: value for key, value in registry.items() if key != "artifact_sha256"
    }
    if recorded_hash != _sha256_json(unhashed):
        raise ProbeRegistryBuildError("probe registry artifact_sha256 is invalid")
    if (
        registry.get("default_cohort") != COHORT_ID
        or registry.get("analysis_role")
        != "exploratory_sota_compatibility_probe_only"
        or registry.get("confirmatory_promotion_permitted") is not False
        or registry.get("paper_result_promotion_permitted") is not False
        or registry.get("production_registry_mutation_permitted") is not False
    ):
        raise ProbeRegistryBuildError("probe registry is not exploratory-only")
    targets = registry.get("targets")
    cohorts = registry.get("cohorts")
    if not isinstance(targets, list) or not isinstance(cohorts, Mapping):
        raise ProbeRegistryBuildError("probe registry targets or cohort are missing")
    cohort = cohorts.get(COHORT_ID)
    if not isinstance(cohort, Mapping):
        raise ProbeRegistryBuildError("exploratory_sota cohort is missing")
    ids: list[str] = []
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ProbeRegistryBuildError(f"probe target {index} is invalid")
        target_id = target.get("id")
        if not isinstance(target_id, str) or not target_id or target_id in ids:
            raise ProbeRegistryBuildError(f"probe target {index} has an invalid id")
        ids.append(target_id)
        if (
            target.get("provider") != "inference_hub"
            or target.get("endpoint_profile") != "inference_hub"
            or target.get("verification_status") != "unverified"
        ):
            raise ProbeRegistryBuildError(
                f"probe target {target_id} violates the unverified Inference Hub contract"
            )
    if cohort.get("targets") != ids:
        raise ProbeRegistryBuildError("exploratory_sota cohort membership is invalid")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a sealed exploratory registry for SOTA route probing"
    )
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    registry = build_probe_registry(
        roster=_read_json(args.roster, label="SOTA roster")
    )
    validate_probe_registry(registry)
    _atomic_write_json(args.output, registry)
    print(
        f"Built {len(registry['targets'])} exploratory SOTA probe targets; "
        "confirmatory and paper promotion are prohibited."
    )
    print(f"Registry: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
