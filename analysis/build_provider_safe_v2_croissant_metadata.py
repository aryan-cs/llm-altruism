"""Build Croissant 1.1 metadata for the definitive aggregate release.

This exporter accepts only the self-hashed, text-free provider-safe-v2
analysis directory and the paper-assets directory derived from it.  It checks
the complete file inventories and SHA-256 bindings before describing any
distribution.  Private run manifests, prompts, responses, and host paths are
outside this release boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from analysis.build_croissant_metadata import (
    CORE_SPEC,
    CROISSANT_CONTEXT,
    RAI_SPEC,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_DIR = (
    REPOSITORY_ROOT / "data/processed/provider-safe-v2-definitive-analysis"
)
DEFAULT_PAPER_ASSETS_DIR = (
    REPOSITORY_ROOT / "data/processed/provider-safe-v2-paper-assets"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "data/processed/provider-safe-v2-croissant-metadata.json"
)

SCHEMA_VERSION = 1
ANALYSIS_ARTIFACT_TYPE = "provider_safe_v2_definitive_descriptive_analysis"
ASSETS_ARTIFACT_TYPE = "provider_safe_v2_paper_assets"
RELEASE_ASSET_SUFFIXES = frozenset({".png", ".tex", ".md"})
DATE_CREATED = "2026-04-30"
DATE_PUBLISHED = "2026-08-03"
DATASET_VERSION = "1.0.0"
METADATA_VERSION = "1.0.0"

_ABSOLUTE_HOST_PATH = re.compile(
    r"(?:/Users/|/home/(?!anonymous(?:/|$))|/private/|(?<![A-Za-z])[A-Za-z]:[\\/])"
)
_FORBIDDEN_FIELD_MARKERS = (
    "prompt_text",
    "raw_response",
    "response_text",
    "reasoning",
    "private_journal",
    "attempt_ledger",
)


class DefinitiveCroissantError(RuntimeError):
    """A proposed public release is incomplete, unbound, or privacy-unsafe."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _self_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {key: item for key, item in value.items() if key != "evidence_sha256"}
        )
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DefinitiveCroissantError(f"{label} is unavailable: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DefinitiveCroissantError(f"{label} is unreadable JSON") from error
    if not isinstance(value, dict):
        raise DefinitiveCroissantError(f"{label} must be a JSON object")
    return value


def _require_basename(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != Path(value).name
        or "/" in value
        or "\\" in value
    ):
        raise DefinitiveCroissantError(f"{label} is not a portable basename")
    return value


def _reject_sensitive_fields(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if not normalized.startswith("contains_") and any(
                marker in normalized for marker in _FORBIDDEN_FIELD_MARKERS
            ):
                raise DefinitiveCroissantError(
                    f"forbidden private field at {path}.{key}"
                )
            if normalized in {"path", "manifest_path"} or normalized.endswith("_path"):
                _require_basename(child, f"{path}.{key}")
            _reject_sensitive_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_fields(child, f"{path}[{index}]")
    elif isinstance(value, str) and _ABSOLUTE_HOST_PATH.search(value):
        raise DefinitiveCroissantError(f"absolute host path at {path}")


def _scan_public_file(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    if _ABSOLUTE_HOST_PATH.search(text):
        raise DefinitiveCroissantError(
            f"public file contains an absolute/private host path: {path.name}"
        )
    if path.suffix in {".json", ".jsonl"}:
        values: list[object]
        if path.suffix == ".jsonl":
            try:
                values = [json.loads(line) for line in text.splitlines() if line.strip()]
            except json.JSONDecodeError as error:
                raise DefinitiveCroissantError(
                    f"public JSONL is malformed: {path.name}"
                ) from error
        else:
            try:
                values = [json.loads(text)]
            except json.JSONDecodeError as error:
                raise DefinitiveCroissantError(
                    f"public JSON is malformed: {path.name}"
                ) from error
        for value in values:
            _reject_sensitive_fields(value, path.name)
    elif path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as error:
                raise DefinitiveCroissantError(
                    f"public CSV is empty: {path.name}"
                ) from error
        for column in header:
            normalized = column.casefold()
            if any(marker in normalized for marker in _FORBIDDEN_FIELD_MARKERS):
                raise DefinitiveCroissantError(
                    f"forbidden private column in {path.name}: {column}"
                )


def _directory_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise DefinitiveCroissantError(
            f"release directory is unavailable: {directory.name}"
        )
    files = sorted(
        (path for path in directory.iterdir() if path.is_file()),
        key=lambda path: path.name,
    )
    if any(path.name != Path(path.name).name for path in files):  # pragma: no cover
        raise DefinitiveCroissantError("release inventory is not flat")
    return files


def _validate_analysis(directory: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = directory / "analysis_manifest.json"
    manifest = _read_object(manifest_path, "definitive analysis manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != ANALYSIS_ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != _self_hash(manifest)
    ):
        raise DefinitiveCroissantError(
            "definitive analysis manifest type, schema, or self-hash failed"
        )
    if manifest.get("path_policy") != (
        "portable_basenames_only_no_host_absolute_paths_in_public_manifest"
    ):
        raise DefinitiveCroissantError("definitive analysis path policy changed")
    privacy = manifest.get("privacy_policy")
    if not isinstance(privacy, Mapping) or any(
        privacy.get(key) is not expected
        for key, expected in {
            "contains_prompt_text": False,
            "contains_response_text_or_reasoning": False,
            "contains_private_journal_paths": False,
            "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
        }.items()
    ):
        raise DefinitiveCroissantError("definitive analysis privacy policy changed")
    if (
        manifest.get("human_labels_generated") is not False
        or manifest.get("exploratory_only") is not True
        or manifest.get("confirmatory_or_paper_promotion_permitted") is not False
    ):
        raise DefinitiveCroissantError("definitive analysis evidence status changed")
    _reject_sensitive_fields(manifest, "analysis_manifest")

    bindings = manifest.get("public_outputs")
    if not isinstance(bindings, list) or not bindings:
        raise DefinitiveCroissantError("definitive analysis output inventory is absent")
    expected: dict[str, Mapping[str, Any]] = {}
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise DefinitiveCroissantError(
                f"definitive analysis output binding {index} is invalid"
            )
        name = _require_basename(binding.get("basename"), f"public_outputs[{index}]")
        if name in expected or not _is_sha256(binding.get("file_sha256")):
            raise DefinitiveCroissantError(
                "definitive analysis output bindings are duplicate or unhashed"
            )
        expected[name] = binding
    files = _directory_files(directory)
    nonmanifest = [path for path in files if path.name != manifest_path.name]
    if {path.name for path in nonmanifest} != set(expected):
        raise DefinitiveCroissantError(
            "definitive analysis directory differs from its complete public inventory"
        )
    for path in nonmanifest:
        if expected[path.name]["file_sha256"] != _sha256_file(path):
            raise DefinitiveCroissantError(
                f"definitive analysis output hash changed: {path.name}"
            )
        _scan_public_file(path)
    _scan_public_file(manifest_path)

    row_counts = manifest.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise DefinitiveCroissantError("definitive analysis row counts are absent")
    for table_name, declared in row_counts.items():
        path = directory / f"{table_name}.csv"
        if not path.is_file() or not isinstance(declared, int) or declared < 0:
            raise DefinitiveCroissantError(
                f"definitive table binding is incomplete: {table_name}"
            )
        with path.open(newline="", encoding="utf-8") as handle:
            actual = sum(1 for _ in csv.DictReader(handle))
        if actual != declared:
            raise DefinitiveCroissantError(
                f"definitive row count changed: {table_name}"
            )
    return manifest, files


def _validate_assets(
    directory: Path, analysis_evidence_sha256: str
) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = directory / "paper_assets_manifest.json"
    manifest = _read_object(manifest_path, "paper-assets manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != ASSETS_ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != _self_hash(manifest)
        or manifest.get("source_analysis_evidence_sha256")
        != analysis_evidence_sha256
    ):
        raise DefinitiveCroissantError(
            "paper-assets manifest type, schema, self-hash, or source binding failed"
        )
    if (
        manifest.get("human_labels_generated") is not False
        or manifest.get("exploratory_only") is not True
        or manifest.get("confirmatory_or_paper_promotion_permitted") is not False
        or manifest.get("cross_axis_aggregate_or_score_generated") is not False
    ):
        raise DefinitiveCroissantError("paper-assets evidence status changed")
    _reject_sensitive_fields(manifest, "paper_assets_manifest")
    bindings = manifest.get("assets")
    if not isinstance(bindings, list) or not bindings:
        raise DefinitiveCroissantError("paper-assets inventory is absent")
    expected: dict[str, Mapping[str, Any]] = {}
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping):
            raise DefinitiveCroissantError(f"paper asset {index} is invalid")
        name = _require_basename(binding.get("name"), f"assets[{index}]")
        if name in expected or not _is_sha256(binding.get("file_sha256")):
            raise DefinitiveCroissantError("paper assets are duplicate or unhashed")
        expected[name] = binding
    files = _directory_files(directory)
    nonmanifest = [path for path in files if path.name != manifest_path.name]
    actual_names = {path.name for path in nonmanifest}
    required_names = {
        name for name in expected if Path(name).suffix in RELEASE_ASSET_SUFFIXES
    }
    if not required_names <= actual_names or not actual_names <= set(expected):
        raise DefinitiveCroissantError(
            "paper-assets directory differs from its complete asset inventory"
        )
    for path in nonmanifest:
        binding = expected[path.name]
        if (
            binding["file_sha256"] != _sha256_file(path)
            or binding.get("byte_count") != path.stat().st_size
        ):
            raise DefinitiveCroissantError(f"paper asset hash changed: {path.name}")
        _scan_public_file(path)
    _scan_public_file(manifest_path)
    return manifest, files


def validate_release_sources(
    analysis_dir: Path, paper_assets_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], list[Path], list[Path]]:
    """Validate both release directories and return their complete inventories."""

    analysis_dir = analysis_dir.resolve()
    paper_assets_dir = paper_assets_dir.resolve()
    analysis, analysis_files = _validate_analysis(analysis_dir)
    assets, asset_files = _validate_assets(
        paper_assets_dir, str(analysis["evidence_sha256"])
    )
    return analysis, assets, analysis_files, asset_files


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _integer(row: Mapping[str, str], key: str, label: str) -> int:
    try:
        value = int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise DefinitiveCroissantError(f"{label} lacks integer field {key}") from error
    if value < 0:
        raise DefinitiveCroissantError(f"{label}.{key} is negative")
    return value


def _coverage(analysis_dir: Path) -> dict[str, int]:
    part0 = _csv_rows(analysis_dir / "part0_models.csv")
    part1 = _csv_rows(analysis_dir / "part1_models.csv")
    part2 = _csv_rows(analysis_dir / "part2_models.csv")
    role = _csv_rows(analysis_dir / "role_calibration_model_frames.csv")
    sensitivity_models = _csv_rows(analysis_dir / "sensitivity_models.csv")
    sensitivity_effects = _csv_rows(analysis_dir / "sensitivity_main_effects.csv")
    coverage = {
        "part0_model_count": len(part0),
        "part0_scheduled_response_count": sum(
            _integer(row, "scheduled_units", "Part 0 row") for row in part0
        ),
        "part1_model_count": len(part1),
        "part1_scheduled_choice_count": sum(
            _integer(row, "scheduled_units", "Part 1 row") for row in part1
        ),
        "part2_model_count": len(part2),
        "part2_trajectory_count": sum(
            _integer(row, "trajectory_count", "Part 2 row") for row in part2
        ),
        "part2_environmentally_estimable_trajectory_count": sum(
            _integer(
                row, "environmentally_estimable_trajectory_count", "Part 2 row"
            )
            for row in part2
        ),
        "part2_semantic_invalid_trajectory_count": sum(
            _integer(row, "semantic_invalid_trajectory_count", "Part 2 row")
            for row in part2
        ),
        "part2_first_attempt_invalid_action_count": sum(
            _integer(row, "first_attempt_invalid_count", "Part 2 row")
            for row in part2
        ),
        "role_calibration_model_frame_count": len(role),
        "sensitivity_model_count": len(sensitivity_models),
        "sensitivity_main_effect_count": len(sensitivity_effects),
    }
    return coverage


def _infer_csv_type(values: Sequence[str]) -> str:
    nonempty = [value for value in values if value != ""]
    if not nonempty:
        return "sc:Text"
    try:
        integers = [int(value) for value in nonempty]
    except ValueError:
        integers = []
    if len(integers) == len(nonempty):
        return "sc:Integer"
    try:
        floats = [float(value) for value in nonempty]
    except ValueError:
        floats = []
    if len(floats) == len(nonempty) and all(math.isfinite(value) for value in floats):
        return "sc:Float"
    if all(value.casefold() in {"true", "false"} for value in nonempty):
        return "sc:Boolean"
    return "sc:Text"


def _record_set(path: Path, file_id: str) -> dict[str, Any]:
    rows = _csv_rows(path)
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    record_id = f"{file_id}-records"
    return {
        "@type": "cr:RecordSet",
        "@id": record_id,
        "name": f"{path.stem.replace('_', ' ').title()} records",
        "description": f"Rows from the sanitized aggregate table {path.name}.",
        "field": [
            {
                "@type": "cr:Field",
                "@id": f"{record_id}/{column}",
                "name": column,
                "description": f"Column `{column}` from {path.name}.",
                "dataType": _infer_csv_type([row.get(column, "") for row in rows]),
                "source": {
                    "fileObject": {"@id": file_id},
                    "extract": {"column": column},
                },
            }
            for column in header
        ],
    }


def _relative_distribution(
    path: Path, directory: Path, output_path: Path, prefix: str
) -> dict[str, Any]:
    if directory.parent.resolve() != output_path.parent.resolve():
        raise DefinitiveCroissantError(
            "analysis, paper assets, and Croissant metadata must share one parent directory"
        )
    name = _require_basename(path.name, "distribution filename")
    content_url = f"{directory.name}/{name}"
    if Path(content_url).is_absolute() or _ABSOLUTE_HOST_PATH.search(content_url):
        raise DefinitiveCroissantError("Croissant content URL is not portable")
    mime = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".jsonl": "application/x-ndjson",
        ".png": "image/png",
        ".tex": "application/x-tex",
        ".md": "text/markdown",
    }.get(path.suffix, "application/octet-stream")
    return {
        "@type": "cr:FileObject",
        "@id": (
            f"{prefix}-{path.stem.replace('_', '-')}-"
            f"{path.suffix.lstrip('.').replace('_', '-') or 'file'}"
        ),
        "name": name,
        "description": (
            "Hash-bound definitive aggregate analysis artifact."
            if prefix == "analysis"
            else "Hash-bound paper-facing table or raster figure derived from the definitive aggregates."
        ),
        "contentUrl": content_url,
        "contentSize": f"{path.stat().st_size} B",
        "encodingFormat": mime,
        "sha256": _sha256_file(path),
    }


def build_metadata(
    *,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    paper_assets_dir: Path = DEFAULT_PAPER_ASSETS_DIR,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    """Validate the release and return portable Croissant 1.1 JSON-LD."""

    analysis_dir = analysis_dir.resolve()
    paper_assets_dir = paper_assets_dir.resolve()
    output_path = output_path.resolve()
    analysis, assets, analysis_files, asset_files = validate_release_sources(
        analysis_dir, paper_assets_dir
    )
    coverage = _coverage(analysis_dir)
    release_asset_files = [
        path
        for path in asset_files
        if path.name == "paper_assets_manifest.json"
        or path.suffix in RELEASE_ASSET_SUFFIXES
    ]
    distributions = [
        *(
            _relative_distribution(path, analysis_dir, output_path, "analysis")
            for path in analysis_files
        ),
        *(
            _relative_distribution(path, paper_assets_dir, output_path, "asset")
            for path in release_asset_files
        ),
    ]
    ids = [item["@id"] for item in distributions]
    if len(ids) != len(set(ids)):
        raise DefinitiveCroissantError("Croissant distribution IDs are not unique")
    by_content_url = {item["contentUrl"]: item for item in distributions}
    record_sets = []
    for path in analysis_files:
        if path.suffix != ".csv":
            continue
        content_url = f"{analysis_dir.name}/{path.name}"
        record_sets.append(_record_set(path, by_content_url[content_url]["@id"]))

    metadata: dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": [CORE_SPEC, RAI_SPEC],
        "name": "Prosocial Readiness Bench definitive aggregate release",
        "description": (
            "Release-safe aggregates for Safety Beyond Refusal. The three axes remain "
            "separate: harmful-request refusal, welfare-preserving one-shot choice, "
            "and repeated-commons restraint. Counts are computed from the hash-bound "
            f"release: {coverage['part0_model_count']} Part 0 models over "
            f"{coverage['part0_scheduled_response_count']} scheduled responses; "
            f"{coverage['part1_model_count']} Part 1 models over "
            f"{coverage['part1_scheduled_choice_count']} scheduled choices; and "
            f"{coverage['part2_model_count']} Part 2 models over "
            f"{coverage['part2_trajectory_count']} trajectories, of which "
            f"{coverage['part2_environmentally_estimable_trajectory_count']} are "
            "environmentally estimable. No cross-axis composite score is released."
        ),
        "version": DATASET_VERSION,
        "cr:sdVersion": METADATA_VERSION,
        "dateCreated": DATE_CREATED,
        "datePublished": DATE_PUBLISHED,
        "dateModified": DATE_PUBLISHED,
        "creator": [{"@type": "sc:Organization", "name": "Anonymous Authors"}],
        "publisher": {"@type": "sc:Organization", "name": "Anonymous Authors"},
        "license": "https://opensource.org/license/mit",
        "sdLicense": "https://opensource.org/license/mit",
        "citeAs": (
            "Anonymous Authors. Safety Beyond Refusal. Anonymous conference submission, 2026."
        ),
        "keywords": [
            "large language models",
            "safety refusal",
            "cooperation",
            "commons restraint",
            "behavioral evaluation",
        ],
        "isAccessibleForFree": True,
        "rai:hasSyntheticData": True,
        "prov:wasGeneratedBy": {
            "@type": "sc:SoftwareApplication",
            "name": "Provider-safe-v2 fail-closed aggregate release pipeline",
            "softwareVersion": DATASET_VERSION,
            "description": (
                "analysis.analyze_provider_safe_v2_definitive, "
                "analysis.build_provider_safe_v2_paper_assets, and "
                "analysis.build_provider_safe_v2_croissant_metadata"
            ),
        },
        "conditionsOfAccess": (
            "The release contains identifiers, hash bindings, derived aggregate tables, "
            "paper tables, and raster figures only. Credentials, private manifests, "
            "prompts, responses, reasoning, journals, and interrupted runs are excluded."
        ),
        "rai:dataCollection": (
            f"The definitive panels contain {coverage['part0_model_count']} Part 0, "
            f"{coverage['part1_model_count']} Part 1, and "
            f"{coverage['part2_model_count']} Part 2 model routes. Part 2 contains "
            f"{coverage['part2_trajectory_count']} scheduled trajectories, "
            f"{coverage['part2_environmentally_estimable_trajectory_count']} "
            "environmentally estimable trajectories, "
            f"{coverage['part2_semantic_invalid_trajectory_count']} semantic-invalid "
            f"trajectories, and {coverage['part2_first_attempt_invalid_action_count']} "
            "first-attempt invalid actions. The separate role-calibration and deadline "
            f"sensitivity releases contain {coverage['role_calibration_model_frame_count']} "
            f"model-frame rows, {coverage['sensitivity_model_count']} sentinel models, "
            f"and {coverage['sensitivity_main_effect_count']} prespecified effects."
        ),
        "rai:dataCollectionType": ["Experiments", "Software Collection"],
        "rai:dataAnnotationProtocol": (
            "Part 0 uses one fixed judge disjoint from every evaluated subject. Parts 1 "
            "and 2 use deterministic structured-action parsing. Primary denominators "
            "retain first-attempt invalids; semantic repairs are isolated supplements."
        ),
        "rai:dataUseCases": [
            "Reproduce axis-specific descriptive tables, intervals, and figures.",
            "Audit model-route coverage, invalid handling, and evidence status.",
            "Not supported: a latent altruism score, global safety ranking, or deployment certification.",
        ],
        "rai:dataLimitations": [
            "Part 0 remains exploratory pending completed human judge validation and benign controls.",
            "Part 1 remains exploratory pending independent prompt-bank content approval.",
            "Part 2 is an artificial repeated-commons environment; it does not identify intent.",
            "Finite-bank intervals for Parts 0 and 1 do not claim population sampling uncertainty.",
            "Availability and semantic-repair panels are reported separately and never replace primary denominators.",
        ],
        "rai:personalSensitiveInformation": [
            "No human-subject records or personal user data are released."
        ],
        "rai:dataReleaseMaintenancePlan": [
            "Regenerate only from self-hashed complete aggregate directories; any inventory, hash, privacy, or source-binding mismatch blocks publication."
        ],
        "releaseCoverage": coverage,
        "sourceAnalysisEvidenceSha256": analysis["evidence_sha256"],
        "sourcePaperAssetsEvidenceSha256": assets["evidence_sha256"],
        "distribution": distributions,
        "recordSet": record_sets,
    }
    _reject_sensitive_fields(metadata, "croissant")
    return metadata


def serialized_metadata(metadata: Mapping[str, Any]) -> str:
    return json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_metadata(
    *,
    analysis_dir: Path = DEFAULT_ANALYSIS_DIR,
    paper_assets_dir: Path = DEFAULT_PAPER_ASSETS_DIR,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    metadata = build_metadata(
        analysis_dir=analysis_dir,
        paper_assets_dir=paper_assets_dir,
        output_path=output_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized_metadata(metadata), encoding="utf-8")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--paper-assets-dir", type=Path, default=DEFAULT_PAPER_ASSETS_DIR
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        metadata = build_metadata(
            analysis_dir=args.analysis_dir,
            paper_assets_dir=args.paper_assets_dir,
            output_path=args.output,
        )
        serialized = serialized_metadata(metadata)
        if args.check:
            if not args.output.is_file() or args.output.read_text(
                encoding="utf-8"
            ) != serialized:
                raise DefinitiveCroissantError(
                    "Croissant metadata is absent or stale relative to release sources"
                )
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
    except (DefinitiveCroissantError, OSError, ValueError) as error:
        parser.error(str(error))
    if not args.check:
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
