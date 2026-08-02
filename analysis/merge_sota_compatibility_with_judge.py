"""Merge exploratory SOTA compatibility evidence with one dedicated judge.

This is an offline, outcome-blind packaging step.  It does not probe a route,
promote a route, or mutate the production registry.  The two returned artifacts
are suitable only for the exploratory hosted Part 1 panel.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REGISTRY_SCHEMA_VERSION = 1
COMPATIBILITY_SCHEMA_VERSION = 2
SOTA_REGISTRY_ARTIFACT_TYPE = (
    "exploratory_sota_inference_hub_probe_registry"
)
COMBINED_REGISTRY_ARTIFACT_TYPE = (
    "exploratory_sota_inference_hub_registry_with_dedicated_judge"
)
COMPATIBILITY_ARTIFACT_TYPE = "inference_hub_provider_compatibility"
SUBJECT_COHORT = "exploratory_sota"
JUDGE_COHORT = "judge_only"
JUDGE_TARGET_ID = "judge.nvidia-evals-nemotron-3-30b-a3b"
DEFAULT_JUDGE_TARGET_ID = JUDGE_TARGET_ID
EXPECTED_SUBJECT_TARGET_COUNT = 84
EXPECTED_SUBJECT_SELECTED_COUNT = 81
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_OPTIONAL_CONTROLS = frozenset(
    {"seed", "temperature", "top_p", "structured_response"}
)


class SotaJudgeMergeError(ValueError):
    """The sealed inputs cannot safely be combined."""


MergeError = SotaJudgeMergeError


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SotaJudgeMergeError(f"{label} must be a lowercase SHA-256 hash")
    return value


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise SotaJudgeMergeError(f"{label} must be a non-empty trimmed string")
    return value


def _identity(value: Any, *, label: str) -> str:
    return _require_text(value, label=label).casefold()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SotaJudgeMergeError(
            f"{label} is not readable UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise SotaJudgeMergeError(f"{label} must be a JSON object")
    return value


def _validate_self_hash(
    artifact: Mapping[str, Any], *, field: str, label: str, required: bool
) -> str | None:
    recorded = artifact.get(field)
    if recorded is None and not required:
        return None
    digest = _require_sha256(recorded, label=f"{label}.{field}")
    unhashed = {key: value for key, value in artifact.items() if key != field}
    if digest != _sha256_json(unhashed):
        raise SotaJudgeMergeError(f"{label}.{field} is invalid")
    return digest


def _validate_hash_map(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise SotaJudgeMergeError(f"{label} must be a non-empty object")
    result: dict[str, str] = {}
    for name, digest in value.items():
        key = _require_text(name, label=f"{label} key")
        result[key] = _require_sha256(digest, label=f"{label}.{key}")
    return result


def _cohort_memberships(
    registry: Mapping[str, Any], *, label: str
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    raw = registry.get("cohorts")
    if not isinstance(raw, Mapping) or not raw:
        raise SotaJudgeMergeError(f"{label}.cohorts must be a non-empty object")
    cohorts: dict[str, Mapping[str, Any]] = {}
    membership: dict[str, str] = {}
    for cohort_name, cohort in raw.items():
        name = _require_text(cohort_name, label=f"{label} cohort name")
        if not isinstance(cohort, Mapping) or not isinstance(
            cohort.get("targets"), list
        ):
            raise SotaJudgeMergeError(f"{label} cohort {name!r} is malformed")
        cohorts[name] = cohort
        for raw_target_id in cohort["targets"]:
            target_id = _require_text(
                raw_target_id, label=f"{label} cohort {name!r} target id"
            )
            normalized = target_id.casefold()
            if normalized in membership:
                raise SotaJudgeMergeError(
                    f"{label} target {target_id!r} belongs to multiple cohorts"
                )
            membership[normalized] = name
    return cohorts, membership


def _validate_registry(
    registry: Mapping[str, Any],
    *,
    label: str,
    require_artifact_seal: bool,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise SotaJudgeMergeError(f"{label} must use registry schema version 1")
    _require_text(registry.get("registry_version"), label=f"{label}.registry_version")
    _validate_self_hash(
        registry,
        field="artifact_sha256",
        label=label,
        required=require_artifact_seal,
    )
    raw_targets = registry.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise SotaJudgeMergeError(f"{label}.targets must be a non-empty list")
    _, memberships = _cohort_memberships(registry, label=label)
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    routes: set[str] = set()
    models: set[tuple[str, str]] = set()
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            raise SotaJudgeMergeError(f"{label} target {index} is not an object")
        row = dict(raw)
        target_id = _identity(row.get("id"), label=f"{label} target {index} id")
        route = _identity(row.get("route"), label=f"{label} target {index} route")
        upstream = _identity(
            row.get("upstream_provider"),
            label=f"{label} target {index} upstream_provider",
        )
        model = _identity(row.get("model"), label=f"{label} target {index} model")
        if row.get("provider") != "inference_hub":
            raise SotaJudgeMergeError(
                f"{label} target {row.get('id')!r} is not an Inference Hub target"
            )
        if target_id in ids or route in routes or (upstream, model) in models:
            raise SotaJudgeMergeError(
                f"{label} target ids, routes, and provider/model identities "
                "must each be unique"
            )
        ids.add(target_id)
        routes.add(route)
        models.add((upstream, model))
        if target_id not in memberships:
            raise SotaJudgeMergeError(
                f"{label} target {row.get('id')!r} has no cohort membership"
            )
        rows.append(row)
    if set(memberships) != ids:
        raise SotaJudgeMergeError(f"{label} cohorts reference unknown targets")
    return rows, memberships


def _selected_profile_is_valid(
    target: Mapping[str, Any], candidate: Mapping[str, Any], *, label: str
) -> bool:
    route = target.get("selected_execution_candidate")
    target_profile = target.get("selected_execution_profile")
    candidate_profile = candidate.get("selected_execution_profile")
    if not isinstance(target_profile, Mapping) or not isinstance(
        candidate_profile, Mapping
    ):
        return False
    controls = target_profile.get("controls")
    return bool(
        target_profile.get("route") == route
        and target_profile.get("status") == "passed"
        and target_profile.get("validation_source") == "execution_profile_probe"
        and candidate_profile.get("status") == "passed"
        and target_profile.get("profile_id") == candidate_profile.get("profile_id")
        and target_profile.get("controls") == candidate_profile.get("controls")
        and target_profile.get("request_sha256")
        == candidate_profile.get("request_sha256")
        and isinstance(controls, list)
        and len(controls) == len(set(controls))
        and all(control in _OPTIONAL_CONTROLS for control in controls)
        and _SHA256.fullmatch(str(target_profile.get("request_sha256"))) is not None
    )


def _validate_compatibility(
    compatibility: Mapping[str, Any],
    registry: Mapping[str, Any],
    registry_rows: Sequence[Mapping[str, Any]],
    memberships: Mapping[str, str],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], int, int]:
    if (
        compatibility.get("schema_version") != COMPATIBILITY_SCHEMA_VERSION
        or compatibility.get("artifact_type") != COMPATIBILITY_ARTIFACT_TYPE
    ):
        raise SotaJudgeMergeError(
            f"{label} must be schema-v2 Inference Hub compatibility evidence"
        )
    _validate_self_hash(
        compatibility,
        field="evidence_sha256",
        label=label,
        required=True,
    )
    if compatibility.get("registry_sha256") != _sha256_json(registry):
        raise SotaJudgeMergeError(
            f"{label} does not bind the canonical supplied registry"
        )
    if compatibility.get("registry_version") != registry.get("registry_version"):
        raise SotaJudgeMergeError(f"{label} registry version is inconsistent")
    _require_text(compatibility.get("endpoint"), label=f"{label}.endpoint")
    _require_sha256(compatibility.get("catalog_sha256"), label=f"{label}.catalog_sha256")
    _validate_hash_map(
        compatibility.get("catalog_source_payload_sha256"),
        label=f"{label}.catalog_source_payload_sha256",
    )
    raw_targets = compatibility.get("targets")
    if not isinstance(raw_targets, list):
        raise SotaJudgeMergeError(f"{label}.targets must be a list")
    registry_by_id = {
        _identity(row.get("id"), label=f"{label} registry target id"): row
        for row in registry_rows
    }
    evidence_rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    selected_routes: set[str] = set()
    selected_count = 0
    candidate_count = 0
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            raise SotaJudgeMergeError(f"{label} target {index} is not an object")
        row = dict(raw)
        target_id = _identity(
            row.get("target_id"), label=f"{label} target {index} target_id"
        )
        if target_id in ids or target_id not in registry_by_id:
            raise SotaJudgeMergeError(f"{label} target ids are repeated or unknown")
        ids.add(target_id)
        registered = registry_by_id[target_id]
        if row.get("model") != registered.get("model"):
            raise SotaJudgeMergeError(
                f"{label} model identity changed for {row.get('target_id')!r}"
            )
        if row.get("cohort") != memberships[target_id]:
            raise SotaJudgeMergeError(
                f"{label} cohort is inconsistent for {row.get('target_id')!r}"
            )
        candidates = row.get("candidates")
        frozen = row.get("frozen_candidate_order")
        if not isinstance(candidates, list) or not isinstance(frozen, list):
            raise SotaJudgeMergeError(f"{label} target candidates are malformed")
        if (
            isinstance(row.get("candidate_count"), bool)
            or row.get("candidate_count") != len(candidates)
            or len(candidates) != len(frozen)
            or len(set(frozen)) != len(frozen)
            or [candidate.get("route") if isinstance(candidate, Mapping) else None for candidate in candidates]
            != frozen
        ):
            raise SotaJudgeMergeError(f"{label} candidate accounting is invalid")
        candidate_count += len(candidates)
        selected = row.get("selected_execution_candidate")
        if selected is None:
            if (
                row.get("status") != "unresolved"
                or row.get("selected_execution_profile") is not None
                or row.get("selection_basis") is not None
                or any(
                    isinstance(candidate, Mapping)
                    and candidate.get("execution_compatible") is True
                    for candidate in candidates
                )
            ):
                raise SotaJudgeMergeError(
                    f"{label} unresolved target accounting is invalid"
                )
        else:
            selected_route = _identity(
                selected, label=f"{label} selected execution route"
            )
            if selected_route in selected_routes:
                raise SotaJudgeMergeError(f"{label} selected routes are repeated")
            selected_routes.add(selected_route)
            matches = [
                candidate
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and candidate.get("route") == selected
                and candidate.get("execution_compatible") is True
            ]
            if (
                row.get("status") != "execution_candidate_selected"
                or len(matches) != 1
                or row.get("selection_basis")
                != "first_execution_compatible_in_reconciliation_frozen_order"
                or not _selected_profile_is_valid(
                    row, matches[0], label=f"{label} selected profile"
                )
            ):
                raise SotaJudgeMergeError(
                    f"{label} selected target evidence is malformed"
                )
            selected_count += 1
        evidence_rows.append(row)
    if ids != set(registry_by_id):
        raise SotaJudgeMergeError(
            f"{label} must contain exactly one evidence row per registry target"
        )
    unresolved_count = len(evidence_rows) - selected_count
    if (
        compatibility.get("target_count") != len(evidence_rows)
        or compatibility.get("candidate_count") != candidate_count
        or compatibility.get("selected_count") != selected_count
        or compatibility.get("unresolved_count") != unresolved_count
    ):
        raise SotaJudgeMergeError(f"{label} summary counts are invalid")
    return evidence_rows, selected_count, unresolved_count


def _bundle_provenance(
    registry: Mapping[str, Any], compatibility: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "registry_sha256": _sha256_json(registry),
        "compatibility_sha256": _sha256_json(compatibility),
        "compatibility_evidence_sha256": compatibility["evidence_sha256"],
    }
    if "artifact_sha256" in registry:
        result["registry_artifact_sha256"] = registry["artifact_sha256"]
    return result


def _route_universe(
    registry_row: Mapping[str, Any], evidence_row: Mapping[str, Any], *, label: str
) -> set[str]:
    routes = {_identity(registry_row.get("route"), label=f"{label} registry route")}
    for value in evidence_row.get("frozen_candidate_order", []):
        routes.add(_identity(value, label=f"{label} compatibility route"))
    selected = evidence_row.get("selected_execution_candidate")
    if selected is not None:
        routes.add(_identity(selected, label=f"{label} selected route"))
    return routes


def merge_sota_compatibility_with_judge(
    *,
    sota_registry: Mapping[str, Any],
    sota_compatibility: Mapping[str, Any],
    production_registry: Mapping[str, Any],
    production_compatibility: Mapping[str, Any],
    judge_target_id: str = JUDGE_TARGET_ID,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a deterministic exploratory registry/evidence pair, without I/O."""

    if judge_target_id != JUDGE_TARGET_ID:
        raise SotaJudgeMergeError(
            f"judge target id must be exactly {JUDGE_TARGET_ID!r}"
        )
    if (
        sota_registry.get("artifact_type") != SOTA_REGISTRY_ARTIFACT_TYPE
        or sota_registry.get("default_cohort") != SUBJECT_COHORT
    ):
        raise SotaJudgeMergeError("SOTA registry is not the sealed exploratory registry")
    for field in (
        "confirmatory_promotion_permitted",
        "paper_result_promotion_permitted",
        "production_registry_mutation_permitted",
    ):
        if sota_registry.get(field) is not False:
            raise SotaJudgeMergeError(f"SOTA registry does not prohibit {field}")

    subject_rows, subject_memberships = _validate_registry(
        sota_registry, label="SOTA registry", require_artifact_seal=True
    )
    production_rows, production_memberships = _validate_registry(
        production_registry,
        label="production registry",
        require_artifact_seal=False,
    )
    subject_evidence, subject_selected, subject_unresolved = _validate_compatibility(
        sota_compatibility,
        sota_registry,
        subject_rows,
        subject_memberships,
        label="SOTA compatibility",
    )
    production_evidence, _, _ = _validate_compatibility(
        production_compatibility,
        production_registry,
        production_rows,
        production_memberships,
        label="production compatibility",
    )
    if (
        len(subject_rows) != EXPECTED_SUBJECT_TARGET_COUNT
        or subject_selected != EXPECTED_SUBJECT_SELECTED_COUNT
        or subject_unresolved
        != EXPECTED_SUBJECT_TARGET_COUNT - EXPECTED_SUBJECT_SELECTED_COUNT
    ):
        raise SotaJudgeMergeError(
            "SOTA bundle must contain exactly 84 targets, 81 selected and 3 unresolved"
        )
    if any(value != SUBJECT_COHORT for value in subject_memberships.values()):
        raise SotaJudgeMergeError("Every SOTA subject must belong only to exploratory_sota")

    endpoint = _require_text(
        sota_compatibility.get("endpoint"), label="SOTA compatibility endpoint"
    )
    if production_compatibility.get("endpoint") != endpoint:
        raise SotaJudgeMergeError("Source compatibility endpoints do not match")
    catalog_sha256 = _require_sha256(
        sota_compatibility.get("catalog_sha256"), label="SOTA catalog_sha256"
    )
    if production_compatibility.get("catalog_sha256") != catalog_sha256:
        raise SotaJudgeMergeError("Source catalog hashes do not match")
    source_payload_hashes = _validate_hash_map(
        sota_compatibility.get("catalog_source_payload_sha256"),
        label="SOTA catalog source hashes",
    )
    if production_compatibility.get("catalog_source_payload_sha256") != source_payload_hashes:
        raise SotaJudgeMergeError("Source catalog payload hashes do not match")
    for policy_name in ("selection_policy", "request_policy"):
        subject_policy = sota_compatibility.get(policy_name)
        judge_policy = production_compatibility.get(policy_name)
        if not isinstance(subject_policy, Mapping) or subject_policy != judge_policy:
            raise SotaJudgeMergeError(
                f"Source compatibility {policy_name} schemas do not match"
            )
    if sota_compatibility["selection_policy"].get(
        "automatic_registry_promotion"
    ) is not False:
        raise SotaJudgeMergeError(
            "Source compatibility policy permits automatic registry promotion"
        )
    if (
        sota_registry.get("source_catalog_input_sha256") != catalog_sha256
        or sota_registry.get("source_catalog_payload_sha256")
        != source_payload_hashes
    ):
        raise SotaJudgeMergeError(
            "SOTA registry catalog provenance does not match its compatibility evidence"
        )

    judge_key = JUDGE_TARGET_ID.casefold()
    production_by_id = {
        _identity(row.get("id"), label="production target id"): row
        for row in production_rows
    }
    production_evidence_by_id = {
        _identity(row.get("target_id"), label="production evidence target id"): row
        for row in production_evidence
    }
    if production_memberships.get(judge_key) != JUDGE_COHORT:
        raise SotaJudgeMergeError("The exact judge is not the production judge_only target")
    judge_members = [
        key for key, cohort in production_memberships.items() if cohort == JUDGE_COHORT
    ]
    if judge_members != [judge_key]:
        raise SotaJudgeMergeError("Production judge_only must contain exactly the judge")
    judge_row = production_by_id.get(judge_key)
    judge_evidence = production_evidence_by_id.get(judge_key)
    if judge_row is None or judge_evidence is None:
        raise SotaJudgeMergeError("The exact judge is absent from a production source")
    if (
        judge_evidence.get("status") != "execution_candidate_selected"
        or judge_evidence.get("selected_execution_candidate") is None
    ):
        raise SotaJudgeMergeError("The exact judge must have selected compatibility evidence")

    subject_by_id = {
        _identity(row.get("id"), label="SOTA target id"): row for row in subject_rows
    }
    subject_evidence_by_id = {
        _identity(row.get("target_id"), label="SOTA evidence target id"): row
        for row in subject_evidence
    }
    if judge_key in subject_by_id:
        raise SotaJudgeMergeError("Judge target id overlaps a SOTA subject")
    judge_model = (
        _identity(judge_row.get("upstream_provider"), label="judge upstream provider"),
        _identity(judge_row.get("model"), label="judge model"),
    )
    subject_models = {
        (
            _identity(row.get("upstream_provider"), label="subject upstream provider"),
            _identity(row.get("model"), label="subject model"),
        )
        for row in subject_rows
    }
    if judge_model in subject_models:
        raise SotaJudgeMergeError("Judge provider/model identity overlaps a SOTA subject")
    judge_routes = _route_universe(
        judge_row, judge_evidence, label="dedicated judge"
    )
    subject_routes: set[str] = set()
    for target_id, row in subject_by_id.items():
        subject_routes.update(
            _route_universe(
                row, subject_evidence_by_id[target_id], label=f"subject {target_id}"
            )
        )
    if judge_routes & subject_routes:
        raise SotaJudgeMergeError("Judge route overlaps a SOTA subject route")

    source_bundles = {
        "exploratory_subjects": _bundle_provenance(
            sota_registry, sota_compatibility
        ),
        "dedicated_judge": _bundle_provenance(
            production_registry, production_compatibility
        ),
    }
    version_material = {
        "subject_registry_sha256": source_bundles["exploratory_subjects"][
            "registry_sha256"
        ],
        "subject_compatibility_sha256": source_bundles["exploratory_subjects"][
            "compatibility_sha256"
        ],
        "judge_registry_sha256": source_bundles["dedicated_judge"][
            "registry_sha256"
        ],
        "judge_compatibility_sha256": source_bundles["dedicated_judge"][
            "compatibility_sha256"
        ],
        "judge_target_id": JUDGE_TARGET_ID,
    }
    registry_version = f"exploratory-sota-with-judge-{_sha256_json(version_material)[:16]}"

    combined_registry = copy.deepcopy(dict(sota_registry))
    combined_registry.pop("artifact_sha256", None)
    combined_registry.update(
        {
            "artifact_type": COMBINED_REGISTRY_ARTIFACT_TYPE,
            "registry_version": registry_version,
            "analysis_role": "exploratory_sota_panel_with_dedicated_judge_only",
            "confirmatory_promotion_permitted": False,
            "paper_result_promotion_permitted": False,
            "production_registry_mutation_permitted": False,
            "source_bundles": copy.deepcopy(source_bundles),
        }
    )
    combined_registry["cohorts"][SUBJECT_COHORT]["version"] = registry_version
    combined_registry["cohorts"][JUDGE_COHORT] = {
        "version": registry_version,
        "analysis_role": "dedicated_judge_only",
        "targets": [JUDGE_TARGET_ID],
    }
    combined_registry["targets"] = [
        *copy.deepcopy(subject_rows),
        copy.deepcopy(judge_row),
    ]
    combined_registry["artifact_sha256"] = _sha256_json(combined_registry)

    combined_compatibility = copy.deepcopy(dict(sota_compatibility))
    combined_compatibility.pop("evidence_sha256", None)
    combined_targets = [
        *copy.deepcopy(subject_evidence),
        copy.deepcopy(judge_evidence),
    ]
    combined_compatibility.update(
        {
            "registry_sha256": _sha256_json(combined_registry),
            "registry_version": registry_version,
            "analysis_role": "exploratory_sota_panel_with_dedicated_judge_only",
            "confirmatory_promotion_permitted": False,
            "paper_result_promotion_permitted": False,
            "production_registry_mutation_permitted": False,
            "source_bundles": copy.deepcopy(source_bundles),
            "target_count": len(combined_targets),
            "candidate_count": sum(row["candidate_count"] for row in combined_targets),
            "selected_count": subject_selected + 1,
            "unresolved_count": subject_unresolved,
            "targets": combined_targets,
        }
    )
    combined_compatibility["evidence_sha256"] = _sha256_json(
        combined_compatibility
    )
    return combined_registry, combined_compatibility


merge_artifacts = merge_sota_compatibility_with_judge


def _stage_json(path: Path, payload: Mapping[str, Any]) -> str:
    parent_existed = path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not parent_existed and os.name == "posix":
        path.parent.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        if os.name == "posix":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_pair(
    registry_path: Path,
    registry: Mapping[str, Any],
    compatibility_path: Path,
    compatibility: Mapping[str, Any],
) -> None:
    staged: list[str] = []
    try:
        staged.append(_stage_json(registry_path, registry))
        staged.append(_stage_json(compatibility_path, compatibility))
        os.replace(staged[0], registry_path)
        staged[0] = ""
        os.replace(staged[1], compatibility_path)
        staged[1] = ""
        if os.name == "posix":
            registry_path.chmod(0o600)
            compatibility_path.chmod(0o600)
    finally:
        for temporary in staged:
            if temporary:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline merge of sealed exploratory SOTA compatibility evidence "
            "with the exact dedicated production judge"
        )
    )
    parser.add_argument(
        "--sota-registry", "--subject-registry", type=Path, required=True
    )
    parser.add_argument(
        "--sota-compatibility", "--subject-compatibility", type=Path, required=True
    )
    parser.add_argument(
        "--production-registry", "--judge-registry", type=Path, required=True
    )
    parser.add_argument(
        "--production-compatibility",
        "--judge-compatibility",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--registry-output", "--output-registry", dest="registry_output", type=Path, required=True
    )
    parser.add_argument(
        "--compatibility-output",
        "--output-compatibility",
        dest="compatibility_output",
        type=Path,
        required=True,
    )
    parser.add_argument("--judge-target-id", default=JUDGE_TARGET_ID)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_paths = (
        args.sota_registry,
        args.sota_compatibility,
        args.production_registry,
        args.production_compatibility,
    )
    output_paths = (args.registry_output, args.compatibility_output)
    resolved_inputs = {path.resolve() for path in input_paths}
    resolved_outputs = [path.resolve() for path in output_paths]
    if len(set(resolved_outputs)) != 2 or any(
        path in resolved_inputs for path in resolved_outputs
    ):
        raise SotaJudgeMergeError(
            "Output paths must be distinct and must not overwrite an input artifact"
        )
    registry, compatibility = merge_sota_compatibility_with_judge(
        sota_registry=_read_json(args.sota_registry, label="SOTA registry"),
        sota_compatibility=_read_json(
            args.sota_compatibility, label="SOTA compatibility"
        ),
        production_registry=_read_json(
            args.production_registry, label="production registry"
        ),
        production_compatibility=_read_json(
            args.production_compatibility, label="production compatibility"
        ),
        judge_target_id=args.judge_target_id,
    )
    _atomic_write_pair(
        args.registry_output,
        registry,
        args.compatibility_output,
        compatibility,
    )
    print(
        "Merged 84 exploratory subjects (81 selected) with one dedicated "
        "selected judge; confirmatory, paper, and production promotion are prohibited."
    )
    print(f"Registry: {args.registry_output}")
    print(f"Compatibility: {args.compatibility_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
