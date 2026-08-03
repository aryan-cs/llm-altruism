"""Build Croissant metadata from the sanitized Safety Beyond Refusal results.

The generator is deliberately fail closed.  It does not describe legacy raw
CSVs, private hosted manifests, or interrupted panel outputs.  It accepts only
the text-free, self-hashed artifact produced by ``analysis.build_final_results``
and the CSVs that artifact hash-binds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINAL_RESULTS_DIR = REPOSITORY_ROOT / "data" / "analysis" / "final_results"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "analysis" / "croissant_metadata.json"

CORE_SPEC = "http://mlcommons.org/croissant/1.1"
RAI_SPEC = "http://mlcommons.org/croissant/RAI/1.0"
DATASET_VERSION = "0.2.0"
METADATA_VERSION = "1.0.0"
DATE_CREATED = "2026-04-30"
DATE_PUBLISHED = "2026-08-02"
DATE_MODIFIED = "2026-08-02"

CROISSANT_CONTEXT: dict[str, object] = {
    "@language": "en", "@vocab": "https://schema.org/",
    "arrayShape": "cr:arrayShape", "citeAs": "cr:citeAs",
    "column": "cr:column", "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn", "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/1.0",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/", "description": {"@container": "@language"},
    "equivalentProperty": "cr:equivalentProperty", "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract", "field": "cr:field", "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject", "fileSet": "cr:fileSet", "format": "cr:format",
    "includes": "cr:includes", "isArray": "cr:isArray", "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath", "key": "cr:key", "md5": "cr:md5",
    "name": {"@container": "@language"}, "parentField": "cr:parentField",
    "path": "cr:path", "prov": "http://www.w3.org/ns/prov#",
    "recordSet": "cr:recordSet", "references": "cr:references", "regex": "cr:regex",
    "repeated": "cr:repeated", "replace": "cr:replace", "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/", "separator": "cr:separator", "source": "cr:source",
    "subField": "cr:subField", "transform": "cr:transform",
}


class CroissantBuildError(RuntimeError):
    """Final sanitized evidence is absent, incomplete, or privacy-unsafe."""


@dataclass(frozen=True)
class ReleaseFile:
    object_id: str
    path: Path
    description: str
    encoding_format: str
    records: bool = False


@dataclass(frozen=True)
class AxisCoverage:
    included: int
    unavailable: int
    targeted: int


@dataclass(frozen=True)
class FinalCoverage:
    part0: AxisCoverage
    part1: AxisCoverage
    part1_scopes: Mapping[int, AxisCoverage]
    part1_pre_execution_unavailable: int
    part2: AxisCoverage


PART0_TARGETED_SYSTEMS = 24
PART1_EXECUTION_SCOPES = {12: 2, 96: 75, 384: 1}
PART1_PRE_EXECUTION_UNAVAILABLE_TARGET_IDS = frozenset({
    "moonshotai/kimi-k2.5",
    "moonshotai/kimi-k2.6",
    "zai-org/glm-5.2",
})
PART1_PRE_EXECUTION_UNAVAILABLE = len(PART1_PRE_EXECUTION_UNAVAILABLE_TARGET_IDS)
PART1_TARGETED_SYSTEMS = sum(PART1_EXECUTION_SCOPES.values()) + PART1_PRE_EXECUTION_UNAVAILABLE
PART2_TARGETED_SYSTEMS = 24


FINAL_OUTPUT_SPECS = {
    "part0_csv": (
        "part0-model-rates", "Part 0 exploratory refusal summaries over 24 English-source roots crossed with three response-language instructions."
    ),
    "part1_csv": (
        "part1-model-rates", "Part 1 exploratory self-choice summaries for 75 routes at 96 roots, two slower routes at 12 roots, and one separate 384-root route."
    ),
    "part2_csv": (
        "part2-model-metrics", "Part 2 corrected commons metrics for 22 executed systems, with validity-gated estimates from fully valid trajectories."
    ),
    "cross_axis_csv": (
        "cross-axis-spearman", "Descriptive matched-panel associations, emitted only when every preregistered evidence gate passes."
    ),
}
_FORBIDDEN_KEYS = (
    "message", "prompt", "raw_response", "reasoning", "requested_route",
    "response_text", "route", "visible_response",
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    unhashed = {key: value for key, value in payload.items() if key != "evidence_sha256"}
    encoded = json.dumps(
        unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_sensitive_keys(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if not normalized.startswith("contains_") and any(
                marker == normalized or marker in normalized for marker in _FORBIDDEN_KEYS
            ):
                raise CroissantBuildError(f"forbidden private field at {path}.{key}")
            _reject_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, f"{path}[{index}]")


def _target_ids(rows: list[Any], label: str) -> set[str]:
    result: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise CroissantBuildError(f"{label} row {index} is not an object")
        target_id = row.get("target_id")
        if not isinstance(target_id, str) or not target_id.strip():
            raise CroissantBuildError(f"{label} row {index} lacks a target ID")
        if target_id in result:
            raise CroissantBuildError(f"{label} target ID is duplicated: {target_id}")
        result.add(target_id)
    return result


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_binding_paths(value: object, path: str = "bindings") -> None:
    """Require every public provenance path to be a basename, never a private path."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized == "path" or normalized.endswith("_path"):
                if (
                    not isinstance(child, str)
                    or not child
                    or child != Path(child).name
                    or "/" in child
                    or "\\" in child
                ):
                    raise CroissantBuildError(
                        f"final-results binding exposes a non-public path at {path}.{key}"
                    )
            _validate_binding_paths(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_binding_paths(child, f"{path}[{index}]")


def _binding_containers(binding: object, label: str) -> list[Mapping[str, Any]]:
    if isinstance(binding, Mapping):
        containers = [binding]
    elif (
        isinstance(binding, list)
        and binding
        and all(isinstance(item, Mapping) for item in binding)
    ):
        containers = list(binding)
    else:
        raise CroissantBuildError(f"{label} binding has an invalid structure")
    for container in containers:
        bound = (
            container.get("primary")
            if container.get("overlay_schema_version") == 1
            else container
        )
        if (
            not isinstance(bound, Mapping)
            or not isinstance(bound.get("manifest_path"), str)
            or not bound.get("manifest_path")
            or not _is_sha256(bound.get("file_sha256"))
            or not _is_sha256(bound.get("evidence_sha256"))
        ):
            raise CroissantBuildError(f"{label} manifest binding is incomplete")
    return containers


def _unavailable_ids(
    container: Mapping[str, Any], label: str,
) -> set[str]:
    unavailable = container.get("unavailable_target_ids", [])
    if not isinstance(unavailable, list):
        raise CroissantBuildError(f"{label} unavailable target IDs are malformed")
    result: set[str] = set()
    for target_id in unavailable:
        if not isinstance(target_id, str) or not target_id.strip():
            raise CroissantBuildError(f"{label} unavailable target ID is invalid")
        if target_id in result:
            raise CroissantBuildError(f"{label} unavailable target ID is duplicated")
        result.add(target_id)
    if result and container.get("overlay_schema_version") != 1:
        raise CroissantBuildError(f"{label} unavailable targets lack an overlay binding")
    failures = container.get("unavailable_target_failures", [])
    if not isinstance(failures, list):
        raise CroissantBuildError(f"{label} unavailable failure evidence is malformed")
    failure_ids: set[str] = set()
    for failure in failures:
        if (
            not isinstance(failure, Mapping)
            or failure.get("target_id") not in result
            or failure.get("provenance")
            != "validated_target_bound_primary_evidence"
            or not isinstance(failure.get("total_failure_count"), int)
            or isinstance(failure.get("total_failure_count"), bool)
            or failure["total_failure_count"] <= 0
        ):
            raise CroissantBuildError(f"{label} unavailable failure evidence is invalid")
        target_id = str(failure["target_id"])
        if target_id in failure_ids:
            raise CroissantBuildError(f"{label} unavailable failure evidence is duplicated")
        failure_ids.add(target_id)
    if failure_ids != result:
        raise CroissantBuildError(f"{label} unavailable failure evidence is incomplete")
    return result


def _part1_binding_root_count(container: Mapping[str, Any]) -> int:
    bound = (
        container.get("primary")
        if container.get("overlay_schema_version") == 1
        else container
    )
    assert isinstance(bound, Mapping)
    root_count = bound.get("root_count")
    scope = container.get("scope", bound.get("scope"))
    if (
        not isinstance(root_count, int)
        or isinstance(root_count, bool)
        or root_count not in PART1_EXECUTION_SCOPES
        or (root_count == 384 and scope != "full_384")
        or (root_count in {12, 96} and scope != "balanced_partial")
    ):
        raise CroissantBuildError("Part 1 binding scope/root count is invalid")
    return root_count


def _coverage(
    artifact: Mapping[str, Any], part0: list[Any], part1: list[Any], part2: list[Any],
) -> FinalCoverage:
    bindings = artifact.get("bindings")
    if not isinstance(bindings, Mapping):
        raise CroissantBuildError("final-results provenance bindings are absent")
    _validate_binding_paths(bindings)
    p0_containers = _binding_containers(bindings.get("part0"), "Part 0")
    p1_containers = _binding_containers(bindings.get("part1"), "Part 1")
    p2_containers = _binding_containers(bindings.get("part2"), "Part 2")
    if len(p0_containers) != 1 or len(p2_containers) != 1:
        raise CroissantBuildError("Part 0 and Part 2 require one primary binding each")

    p0_ids = _target_ids(part0, "Part 0")
    p1_ids = _target_ids(part1, "Part 1")
    p2_ids = _target_ids(part2, "Part 2")
    p0_unavailable = _unavailable_ids(p0_containers[0], "Part 0")
    p2_unavailable = _unavailable_ids(p2_containers[0], "Part 2")
    if p0_ids & p0_unavailable or p2_ids & p2_unavailable:
        raise CroissantBuildError("included and unavailable target IDs overlap")
    if (
        len(p0_ids) + len(p0_unavailable) != PART0_TARGETED_SYSTEMS
        or len(p2_ids) + len(p2_unavailable) != PART2_TARGETED_SYSTEMS
        or p0_ids | p0_unavailable != p2_ids | p2_unavailable
    ):
        raise CroissantBuildError(
            "final-results coverage differs from the frozen 24-system matched design"
        )

    rows_by_root: dict[int, set[str]] = {count: set() for count in PART1_EXECUTION_SCOPES}
    for row in part1:
        assert isinstance(row, Mapping)
        root_count = row.get("root_count")
        scope = row.get("scope")
        if (
            not isinstance(root_count, int)
            or isinstance(root_count, bool)
            or root_count not in PART1_EXECUTION_SCOPES
            or (root_count == 384 and scope != "full_384")
            or (root_count in {12, 96} and scope != "balanced_partial")
        ):
            raise CroissantBuildError("Part 1 row scope/root count is invalid")
        rows_by_root[root_count].add(str(row["target_id"]))

    unavailable_by_root: dict[int, set[str]] = {
        count: set() for count in PART1_EXECUTION_SCOPES
    }
    all_p1_unavailable: set[str] = set()
    for container in p1_containers:
        root_count = _part1_binding_root_count(container)
        unavailable = _unavailable_ids(container, "Part 1")
        if all_p1_unavailable & unavailable:
            raise CroissantBuildError("Part 1 unavailable target ID is duplicated across bindings")
        all_p1_unavailable.update(unavailable)
        unavailable_by_root[root_count].update(unavailable)
    if p1_ids & all_p1_unavailable:
        raise CroissantBuildError("Part 1 included and unavailable target IDs overlap")
    if (p1_ids | all_p1_unavailable) & PART1_PRE_EXECUTION_UNAVAILABLE_TARGET_IDS:
        raise CroissantBuildError(
            "a pre-execution-unavailable Part 1 registry target appears in the execution roster"
        )
    for root_count, targeted in PART1_EXECUTION_SCOPES.items():
        if len(rows_by_root[root_count]) + len(unavailable_by_root[root_count]) != targeted:
            raise CroissantBuildError(
                "final-results coverage differs from the frozen Part 1 "
                "75x96+2x12+1x384 execution scopes"
            )
    if not (p0_ids | p0_unavailable) <= (p1_ids | all_p1_unavailable):
        raise CroissantBuildError("Part 1 omits a frozen matched-panel target")

    p1_scopes = {
        root_count: AxisCoverage(
            included=len(rows_by_root[root_count]),
            unavailable=len(unavailable_by_root[root_count]),
            targeted=targeted,
        )
        for root_count, targeted in PART1_EXECUTION_SCOPES.items()
    }
    return FinalCoverage(
        part0=AxisCoverage(len(p0_ids), len(p0_unavailable), PART0_TARGETED_SYSTEMS),
        part1=AxisCoverage(
            len(p1_ids),
            len(all_p1_unavailable) + PART1_PRE_EXECUTION_UNAVAILABLE,
            PART1_TARGETED_SYSTEMS,
        ),
        part1_scopes=p1_scopes,
        part1_pre_execution_unavailable=PART1_PRE_EXECUTION_UNAVAILABLE,
        part2=AxisCoverage(len(p2_ids), len(p2_unavailable), PART2_TARGETED_SYSTEMS),
    )


def _validated_final_results(
    final_results_dir: Path,
) -> tuple[dict[str, Any], tuple[ReleaseFile, ...], FinalCoverage]:
    directory = final_results_dir.resolve()
    artifact_path = directory / "final_results.json"
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CroissantBuildError(
            f"sanitized final results are unavailable or unreadable: {artifact_path}"
        ) from error
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != 1
        or artifact.get("artifact_type") != "prosocial_readiness_final_sanitized_results"
        or artifact.get("evidence_sha256") != _canonical_hash(artifact)
    ):
        raise CroissantBuildError("final-results schema, type, or self-hash is invalid")
    privacy = artifact.get("privacy_contract")
    required_privacy = {
        "contains_prompt_text", "contains_response_text", "contains_reasoning",
        "contains_raw_responses", "contains_routes",
    }
    if (
        not isinstance(privacy, Mapping)
        or set(privacy) != required_privacy
        or any(privacy.values())
    ):
        raise CroissantBuildError("final-results privacy contract is absent or unsafe")
    parameters = artifact.get("parameters")
    if (
        not isinstance(parameters, Mapping)
        or parameters.get("part1_scopes_pooled") is not False
    ):
        raise CroissantBuildError("final-results Part 1 no-pooling contract is absent")
    _reject_sensitive_keys(artifact)

    part0 = artifact.get("part0")
    part1 = artifact.get("part1")
    part2 = artifact.get("part2")
    if not all(isinstance(rows, list) for rows in (part0, part1, part2)):
        raise CroissantBuildError("final-results axis rows are absent")
    coverage = _coverage(artifact, part0, part1, part2)
    if (
        any(row.get("root_count_per_condition") != 24 for row in part0)
        or any(row.get("trajectory_count") != 8 for row in part2)
    ):
        raise CroissantBuildError("final-results per-system execution counts changed")
    part2_estimable = [
        row for row in part2
        if row.get("metric_status") == "estimable_from_fully_valid_trajectories"
    ]
    part2_nonestimable = [
        row for row in part2
        if row.get("metric_status")
        == "nonestimable_all_trajectories_contain_invalid_actions"
    ]
    if (
        len(part2_estimable) + len(part2_nonestimable) != len(part2)
        or any(
            not isinstance(row.get("valid_trajectory_count"), int)
            or not isinstance(row.get("protocol_invalid_trajectory_count"), int)
            or row["valid_trajectory_count"] + row["protocol_invalid_trajectory_count"] != 8
            for row in part2
        )
        or any(row.get("trajectory_level_95_percent_t_intervals") is None for row in part2_estimable)
        or any(row.get("trajectory_level_95_percent_t_intervals") is not None for row in part2_nonestimable)
    ):
        raise CroissantBuildError("final-results Part 2 validity gate is inconsistent")
    outputs = artifact.get("outputs")
    if not isinstance(outputs, Mapping):
        raise CroissantBuildError("final-results output bindings are absent")
    releases = [
        ReleaseFile(
            "final-results", artifact_path,
            "Self-hashed, text-free Safety Beyond Refusal result graph and evidence-status record.",
            "application/json", False,
        )
    ]
    bound_csvs: dict[str, Path] = {}
    for key, (object_id, description) in FINAL_OUTPUT_SPECS.items():
        binding = outputs.get(key)
        if binding is None and key == "cross_axis_csv":
            continue
        if not isinstance(binding, Mapping) or set(binding) != {"path", "file_sha256"}:
            raise CroissantBuildError(f"final-results binding {key} is malformed")
        relative = binding.get("path")
        if not isinstance(relative, str) or not relative:
            raise CroissantBuildError(f"final-results binding {key} lacks a path")
        path = (directory / relative).resolve()
        if path.parent != directory or path.suffix != ".csv" or not path.is_file():
            raise CroissantBuildError(f"final-results binding {key} escapes or is missing")
        if binding.get("file_sha256") != _sha256(path):
            raise CroissantBuildError(f"final-results binding {key} hash changed")
        releases.append(ReleaseFile(object_id, path, description, "text/csv", True))
        bound_csvs[key] = path
    _validate_axis_csvs(part0, part1, part2, bound_csvs)
    return artifact, tuple(releases), coverage


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise CroissantBuildError(f"CSV has no header: {path}")
        columns = list(reader.fieldnames)
        for column in columns:
            normalized = column.casefold()
            if any(marker == normalized or marker in normalized for marker in _FORBIDDEN_KEYS):
                raise CroissantBuildError(f"CSV exposes forbidden private column: {column}")
        return columns, list(reader)


def _validate_axis_csvs(
    part0: list[Any], part1: list[Any], part2: list[Any], bound_csvs: Mapping[str, Path],
) -> None:
    try:
        _, p0_csv = _read_csv_rows(bound_csvs["part0_csv"])
        _, p1_csv = _read_csv_rows(bound_csvs["part1_csv"])
        _, p2_csv = _read_csv_rows(bound_csvs["part2_csv"])
    except KeyError as error:
        raise CroissantBuildError("a required axis CSV binding is absent") from error

    p0_expected = {str(row["target_id"]) for row in part0}
    p0_observed = [row.get("target_id", "") for row in p0_csv]
    if (
        set(p0_observed) != p0_expected
        or len(p0_observed) != 3 * len(p0_expected)
        or any(p0_observed.count(target_id) != 3 for target_id in p0_expected)
    ):
        raise CroissantBuildError("Part 0 CSV rows do not match included systems")

    p1_expected = {
        (str(row["target_id"]), str(row["scope"]), int(row["root_count"]))
        for row in part1
    }
    try:
        p1_observed = {
            (row["target_id"], row["scope"], int(row["root_count"]))
            for row in p1_csv
        }
    except (KeyError, TypeError, ValueError) as error:
        raise CroissantBuildError("Part 1 CSV scope fields are malformed") from error
    if p1_observed != p1_expected or len(p1_csv) != len(p1_expected):
        raise CroissantBuildError("Part 1 CSV rows do not match included scope rows")

    p2_expected = {str(row["target_id"]) for row in part2}
    p2_observed = [row.get("target_id", "") for row in p2_csv]
    if set(p2_observed) != p2_expected or len(p2_observed) != len(p2_expected):
        raise CroissantBuildError("Part 2 CSV rows do not match included systems")


def _is_integer(value: str) -> bool:
    try:
        int(value)
    except ValueError:
        return False
    return not any(character in value.lower() for character in (".", "e"))


def _is_float(value: str) -> bool:
    try:
        parsed = float(value)
    except ValueError:
        return False
    return math.isfinite(parsed)


def _data_type(values: Iterable[str]) -> str:
    populated = [value.strip() for value in values if value.strip()]
    if not populated:
        return "sc:Text"
    lowered = {value.lower() for value in populated}
    if lowered <= {"true", "false"}:
        return "sc:Boolean"
    if all(_is_integer(value) for value in populated):
        return "sc:Integer"
    if all(_is_float(value) for value in populated):
        return "sc:Float"
    return "sc:Text"


def _csv_schema(path: Path) -> tuple[list[str], dict[str, str]]:
    columns, rows = _read_csv_rows(path)
    values = {column: [] for column in columns}
    for row in rows:
        for column in columns:
            values[column].append(row[column] or "")
    return columns, {column: _data_type(values[column]) for column in columns}


def _validate_dataset_url(dataset_url: str | None) -> str | None:
    if dataset_url is None:
        return None
    candidate = dataset_url.strip()
    parsed = urlparse(candidate)
    blocked_hosts = {"example.com", "www.example.com", "localhost", "127.0.0.1"}
    if parsed.scheme != "https" or not parsed.netloc or parsed.hostname in blocked_hosts:
        raise ValueError("dataset URL must be a real, public HTTPS location")
    if any(token in candidate.lower() for token in ("<", ">", "placeholder", "todo")):
        raise ValueError("dataset URL contains a placeholder token")
    return candidate


def build_metadata(
    *, repository_root: Path = REPOSITORY_ROOT,
    final_results_dir: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    dataset_url: str | None = None,
) -> dict[str, object]:
    """Return Croissant JSON-LD only for validated sanitized final artifacts."""

    dataset_url = _validate_dataset_url(dataset_url)
    final_dir = final_results_dir or repository_root / "data" / "analysis" / "final_results"
    artifact, release_files, coverage = _validated_final_results(final_dir)
    distributions: list[dict[str, object]] = []
    record_sets: list[dict[str, object]] = []
    for release_file in release_files:
        source_path = release_file.path.resolve()
        try:
            content_url = Path(os.path.relpath(source_path, output_path.parent.resolve())).as_posix()
        except ValueError as error:
            raise ValueError(f"cannot make {source_path} relative to {output_path}") from error
        distributions.append({
            "@type": "cr:FileObject", "@id": release_file.object_id,
            "name": source_path.name, "description": release_file.description,
            "contentUrl": content_url, "contentSize": f"{source_path.stat().st_size} B",
            "encodingFormat": release_file.encoding_format, "sha256": _sha256(source_path),
        })
        if release_file.records:
            columns, types = _csv_schema(source_path)
            record_set_id = f"{release_file.object_id}-records"
            record_sets.append({
                "@type": "cr:RecordSet", "@id": record_set_id,
                "name": f"{source_path.stem.replace('_', ' ').title()} records",
                "description": release_file.description,
                "field": [{
                    "@type": "cr:Field", "@id": f"{record_set_id}/{column}",
                    "name": column, "description": f"Column `{column}` from {source_path.name}.",
                    "dataType": types[column],
                    "source": {"fileObject": {"@id": release_file.object_id}, "extract": {"column": column}},
                } for column in columns],
            })

    cross_status = artifact.get("cross_axis", {}).get("status") if isinstance(artifact.get("cross_axis"), Mapping) else None
    p1_execution_unavailable = (
        coverage.part1.unavailable - coverage.part1_pre_execution_unavailable
    )
    p1_n96 = coverage.part1_scopes[96]
    p1_n12 = coverage.part1_scopes[12]
    p1_n384 = coverage.part1_scopes[384]
    part2_rows = artifact["part2"]
    part2_estimable = [
        row for row in part2_rows
        if row.get("metric_status") == "estimable_from_fully_valid_trajectories"
    ]
    part2_nonestimable = [
        row for row in part2_rows
        if row.get("metric_status")
        == "nonestimable_all_trajectories_contain_invalid_actions"
    ]
    part2_valid_trajectories = sum(row["valid_trajectory_count"] for row in part2_rows)
    part2_protocol_invalid_trajectories = sum(
        row["protocol_invalid_trajectory_count"] for row in part2_rows
    )
    metadata: dict[str, object] = {
        "@context": CROISSANT_CONTEXT, "@type": "sc:Dataset",
        "conformsTo": [CORE_SPEC, RAI_SPEC],
        "name": "Prosocial Readiness Bench",
        "description": (
            "Release-safe aggregate artifacts for Safety Beyond Refusal, a three-axis evaluation "
            "of harmful-request refusal, welfare-preserving self-choice, and repeated commons "
            "preservation. Part 0 and the balanced Part 1 expansion remain exploratory because "
            "their human/content approval gates are incomplete; Part 2 uses a corrected engine "
            "with eight independent common-seed trajectories per executed system and excludes "
            "any trajectory containing an invalid action from behavioral and environmental estimates. "
            f"The release includes {coverage.part0.included} of {coverage.part0.targeted} "
            f"Part 0 systems, {coverage.part1.included} of {coverage.part1.targeted} frozen "
            f"Part 1 targets, and {coverage.part2.included} of {coverage.part2.targeted} "
            f"Part 2 systems. Of those, {len(part2_estimable)} have an estimate and "
            f"{len(part2_nonestimable)} are protocol-nonestimable; the remaining "
            "targets are explicitly unavailable rather than scored."
        ),
        "version": DATASET_VERSION, "cr:sdVersion": METADATA_VERSION,
        "dateCreated": DATE_CREATED, "datePublished": DATE_PUBLISHED,
        "dateModified": DATE_MODIFIED,
        "creator": [{"@type": "sc:Organization", "name": "Anonymous Authors"}],
        "publisher": {"@type": "sc:Organization", "name": "Anonymous Authors"},
        "license": "https://opensource.org/license/mit", "sdLicense": "https://opensource.org/license/mit",
        "citeAs": "Anonymous Authors. Safety Beyond Refusal. Anonymous conference submission, 2026.",
        "keywords": ["large language models", "safety refusal", "dyadic cooperation", "commons restraint", "behavioral evaluation"],
        "isAccessibleForFree": True, "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": [{
            "@type": "sc:CreativeWork", "name": "Executed hosted Prosocial Readiness Bench panels",
            "description": (
                f"The frozen 24-system matched panel yielded {coverage.part0.included} "
                f"Part 0 and {coverage.part2.included} Part 2 included systems; "
                f"{coverage.part0.unavailable} and {coverage.part2.unavailable}, respectively, "
                "were operationally unavailable on those axes. The 78-target Part 1 execution "
                f"roster yielded {coverage.part1.included} included targets and "
                f"{p1_execution_unavailable} target-bound operational unavailability records. "
                f"A separate frozen pre-execution registry records "
                f"{coverage.part1_pre_execution_unavailable} additional unavailable targets, "
                f"for {coverage.part1.targeted} frozen Part 1 targets total."
            ),
        }],
        "prov:wasGeneratedBy": {
            "@type": "sc:SoftwareApplication", "name": "Prosocial Readiness Bench fail-closed final-results pipeline",
            "softwareVersion": DATASET_VERSION,
            "description": "analysis.build_final_results followed by analysis.build_croissant_metadata",
        },
        "conditionsOfAccess": (
            "Only sanitized aggregate tables and their text-free self-hashed result graph are released. "
            "Provider credentials, private manifests, harmful prompts, visible responses, reasoning, raw journals, and interrupted artifacts are excluded."
        ),
        "rai:dataCollection": (
            f"Part 0 scheduled {coverage.part0.targeted} systems on 24 archived English "
            "harmful-request roots under English, Chinese, and Russian response-language "
            f"instructions; {coverage.part0.included} systems are included and "
            f"{coverage.part0.unavailable} are operationally unavailable. Part 1 scheduled "
            f"{p1_n96.targeted} execution-roster targets on 96 balanced roots "
            f"({p1_n96.included} included, {p1_n96.unavailable} unavailable), "
            f"{p1_n12.targeted} on 12 balanced roots ({p1_n12.included} included, "
            f"{p1_n12.unavailable} unavailable), and {p1_n384.targeted} on all 384 roots "
            f"({p1_n384.included} included, {p1_n384.unavailable} unavailable). Another "
            f"{coverage.part1_pre_execution_unavailable} frozen Part 1 registry targets were "
            "unavailable before execution and are not described as observed. Part 2 scheduled "
            f"{coverage.part2.targeted} matched systems for eight independent corrected "
            f"12-step trajectories each; {coverage.part2.included} systems are included and "
            f"{coverage.part2.unavailable} are operationally unavailable. Environmental and "
            f"behavioral estimates use {part2_valid_trajectories} fully valid trajectories; "
            f"{part2_protocol_invalid_trajectories} protocol-invalid trajectories are excluded, "
            f"leaving {len(part2_nonestimable)} executed systems without an estimate."
        ),
        "rai:dataCollectionType": ["Experiments", "Software Collection"],
        "rai:dataCollectionRawData": "Private model responses and execution journals are retained for provenance but are not distributions in this release.",
        "rai:dataAnnotationProtocol": (
            "Part 0 uses one fixed judge that sees only visible subject responses; its aggregate remains exploratory until human validation. "
            "Parts 1 and 2 use deterministic structured-action parsing and retain malformed actions as nonsuccesses."
        ),
        "rai:machineAnnotationTools": ["Fixed disjoint Part 0 judge and deterministic structured-output parsers."],
        "rai:dataPreprocessingProtocol": [
            "Validate self-hashed complete manifests or target-bound fail-closed overlays and exact response-model identity.",
            "Retain invalid and unclear Part 0/Part 1 scheduled units as nonsuccesses; exclude an entire Part 2 trajectory from outcome estimates if any action is invalid.",
            "Preserve every Part 1 target's observed root count and never pool the 12-root, 96-root, and 384-root estimates.",
            "Emit text-free aggregates only after output hashes and privacy flags pass.",
        ],
        "rai:dataUseCases": [
            "Reproduce the paper's evidence-status-aware aggregate summaries.",
            "Study task-specific safety refusal, dyadic cooperation, and commons restraint without treating them as a latent moral trait.",
            "Not supported: training on harmful content, global prosocial rankings, or deployment safety certification.",
        ],
        "rai:dataLimitations": [
            "Part 0 reconstructs response-language instructions over English inputs, has no benign controls, and lacks completed human judge validation.",
            (
                "The Part 1 bank lacks independent content approval; the frozen 12-root, "
                "96-root, and 384-root execution scopes have different support and are not "
                f"pooled. {p1_execution_unavailable} execution-roster targets and "
                f"{coverage.part1_pre_execution_unavailable} pre-execution registry targets "
                "are unavailable rather than estimated."
            ),
            "Part 2 has eight rather than the intended twelve independent trajectories per system and no parameter-sensitivity analysis.",
            (
                f"Axis-specific operational availability differs: {coverage.part0.unavailable} "
                f"Part 0 and {coverage.part2.unavailable} Part 2 matched-panel targets have no "
                "released estimate on the affected axis."
            ),
            f"Cross-axis artifact status is {cross_status!r}; associations are not paper evidence unless every gate passes.",
            "The tasks measure observable outputs in artificial settings, not intent, moral status, or unrestricted deployment behavior.",
        ],
        "rai:dataBiases": [
            "Fixed prompt-bank, reconstructed-language, route-availability, developer-family, and constrained-action selection effects may shape results.",
            "Related systems from one developer are not independent samples from a model population.",
        ],
        "rai:personalSensitiveInformation": [
            "No human-subject records or personal user data are collected; harmful model text remains private and is excluded from this release."
        ],
        "rai:dataSocialImpact": (
            "Separate axes and explicit evidence gates reduce unsupported model rankings, but simplified tasks and finite route coverage still invite overgeneralization."
        ),
        "rai:dataReleaseMaintenancePlan": [
            "Regenerate only from an immutable sanitized final-results directory; any missing file, hash mismatch, private field, or coverage change blocks metadata emission."
        ],
        "distribution": distributions, "recordSet": record_sets,
    }
    if dataset_url is not None:
        metadata["url"] = dataset_url
    return metadata


def write_metadata(
    output_path: Path = DEFAULT_OUTPUT, *, final_results_dir: Path | None = None,
    dataset_url: str | None = None,
) -> dict[str, object]:
    metadata = build_metadata(
        final_results_dir=final_results_dir, output_path=output_path,
        dataset_url=dataset_url,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-results-dir", type=Path, default=DEFAULT_FINAL_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-url", default=os.environ.get("CROISSANT_DATASET_URL"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        metadata = build_metadata(
            final_results_dir=args.final_results_dir, output_path=args.output,
            dataset_url=args.dataset_url,
        )
    except (CroissantBuildError, OSError, ValueError) as error:
        parser.error(str(error))
    serialized = json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != serialized:
            parser.error(f"Croissant metadata is stale: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
