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


FINAL_OUTPUT_SPECS = {
    "part0_csv": (
        "part0-model-rates", "Part 0 exploratory refusal summaries over 24 English-source roots crossed with three response-language instructions."
    ),
    "part1_csv": (
        "part1-model-rates", "Part 1 exploratory self-choice summaries for 75 routes at 96 roots, two slower routes at 12 roots, and one separate 384-root route."
    ),
    "part2_csv": (
        "part2-model-metrics", "Part 2 corrected commons metrics for 24 matched systems with eight independent trajectories each."
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


def _validated_final_results(final_results_dir: Path) -> tuple[dict[str, Any], tuple[ReleaseFile, ...]]:
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
    _reject_sensitive_keys(artifact)

    part0 = artifact.get("part0")
    part1 = artifact.get("part1")
    part2 = artifact.get("part2")
    if not all(isinstance(rows, list) for rows in (part0, part1, part2)):
        raise CroissantBuildError("final-results axis rows are absent")
    p0_ids = {row.get("target_id") for row in part0 if isinstance(row, Mapping)}
    p2_ids = {row.get("target_id") for row in part2 if isinstance(row, Mapping)}
    p1_ids = {row.get("target_id") for row in part1 if isinstance(row, Mapping)}
    balanced = [row for row in part1 if isinstance(row, Mapping) and row.get("scope") == "balanced_partial"]
    full = [row for row in part1 if isinstance(row, Mapping) and row.get("scope") == "full_384"]
    balanced_counts = {
        count: sum(row.get("root_count") == count for row in balanced)
        for count in (12, 96)
    }
    if (
        len(part0) != len(p0_ids) != 0
        or len(part2) != len(p2_ids) != 0
        or len(p0_ids) != 24
        or p0_ids != p2_ids
        or len(part1) != 78
        or len(p1_ids) != 78
        or any(row.get("root_count_per_condition") != 24 for row in part0)
        or any(row.get("trajectory_count") != 8 for row in part2)
        or len(balanced) != 77
        or len(full) != 1
        or balanced_counts != {12: 2, 96: 75}
        or full[0].get("root_count") != 384
        or not p0_ids <= p1_ids
    ):
        raise CroissantBuildError(
            "final-results coverage differs from the executed 24-matched/75x96+2x12+1x384 Part 1 design"
        )

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
    return artifact, tuple(releases)


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
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise CroissantBuildError(f"CSV has no header: {path}")
        columns = list(reader.fieldnames)
        values = {column: [] for column in columns}
        for row in reader:
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
    artifact, release_files = _validated_final_results(final_dir)
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
    metadata: dict[str, object] = {
        "@context": CROISSANT_CONTEXT, "@type": "sc:Dataset",
        "conformsTo": [CORE_SPEC, RAI_SPEC],
        "name": "Prosocial Readiness Bench",
        "description": (
            "Release-safe aggregate artifacts for Safety Beyond Refusal, a three-axis evaluation "
            "of harmful-request refusal, welfare-preserving self-choice, and repeated commons "
            "preservation. Part 0 and the balanced Part 1 expansion remain exploratory because "
            "their human/content approval gates are incomplete; Part 2 uses a corrected engine "
            "with eight independent common-seed trajectories per matched system."
        ),
        "version": DATASET_VERSION, "cr:sdVersion": METADATA_VERSION,
        "dateCreated": DATE_CREATED, "datePublished": DATE_PUBLISHED,
        "dateModified": DATE_MODIFIED,
        "creator": [{"@type": "sc:Organization", "name": "Anonymous Authors"}],
        "publisher": {"@type": "sc:Organization", "name": "Anonymous Authors"},
        "license": "https://opensource.org/license/mit", "sdLicense": "https://opensource.org/license/mit",
        "citeAs": "Anonymous Authors. Safety Beyond Refusal. Anonymous conference submission, 2026.",
        "keywords": ["large language models", "safety refusal", "cooperation", "commons preservation", "behavioral evaluation"],
        "isAccessibleForFree": True, "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": [{
            "@type": "sc:CreativeWork", "name": "Executed hosted Prosocial Readiness Bench panels",
            "description": "Twenty-four matched systems for Parts 0 and 2, plus 75 Part 1 routes at 96 roots, two at 12 roots, and one separate 384-root route (78 observed of 81 frozen targets).",
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
            "Part 0 executes 24 archived English harmful-request roots under English, Chinese, and Russian response-language instructions. "
            "Part 1 executes 75 routes on 96 balanced roots, two slower routes on 12 balanced roots each, and one separate route on all 384 roots. "
            "Part 2 executes eight independent corrected 12-step trajectories for each of the 24 matched systems."
        ),
        "rai:dataCollectionType": ["Experiments", "Software Collection"],
        "rai:dataCollectionRawData": "Private model responses and execution journals are retained for provenance but are not distributions in this release.",
        "rai:dataAnnotationProtocol": (
            "Part 0 uses one fixed judge that sees only visible subject responses; its aggregate remains exploratory until human validation. "
            "Parts 1 and 2 use deterministic structured-action parsing and retain malformed actions as nonsuccesses."
        ),
        "rai:machineAnnotationTools": ["Fixed disjoint Part 0 judge and deterministic structured-output parsers."],
        "rai:dataPreprocessingProtocol": [
            "Validate complete self-hashed manifests and exact response-model identity.",
            "Retain invalid and unclear scheduled units in denominators.",
            "Preserve every Part 1 target's observed root count and never pool the 12-root, 96-root, and 384-root estimates.",
            "Emit text-free aggregates only after output hashes and privacy flags pass.",
        ],
        "rai:dataUseCases": [
            "Reproduce the paper's evidence-status-aware aggregate summaries.",
            "Study task-specific refusal, self-choice, and commons preservation without treating them as a latent moral trait.",
            "Not supported: training on harmful content, global prosocial rankings, or deployment safety certification.",
        ],
        "rai:dataLimitations": [
            "Part 0 reconstructs response-language instructions over English inputs, has no benign controls, and lacks completed human judge validation.",
            "The Part 1 bank lacks independent content approval; the two 12-root routes, 75 96-root routes, and single 384-root route have different support and are not pooled.",
            "Part 2 has eight rather than the intended twelve independent trajectories per system and no parameter-sensitivity analysis.",
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
