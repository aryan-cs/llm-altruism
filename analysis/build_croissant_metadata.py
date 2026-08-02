"""Build the paper-facing MLCommons Croissant metadata.

The graph describes release-safe raw Part 1/Part 2 traces and their derived
analysis files. Raw Part 0 prompts and completions remain intentionally absent.
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
from typing import Iterable
from urllib.parse import urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "analysis" / "croissant_metadata.json"

CORE_SPEC = "http://mlcommons.org/croissant/1.1"
RAI_SPEC = "http://mlcommons.org/croissant/RAI/1.0"
DATASET_VERSION = "0.1.0"
METADATA_VERSION = "1.0.0"
DATE_CREATED = "2026-04-30"
DATE_PUBLISHED = "2026-08-02"
DATE_MODIFIED = "2026-08-02"

# This is the official Croissant 1.1 context emitted by mlcroissant 1.1.0.  The
# aliases matter: spelling ``recordSet`` without mapping it to ``cr:recordSet``
# silently produces a schema.org property and mlcroissant loads zero records.
CROISSANT_CONTEXT: dict[str, object] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "arrayShape": "cr:arrayShape",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "description": {"@container": "@language"},
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isArray": "cr:isArray",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "name": {"@container": "@language"},
    "parentField": "cr:parentField",
    "path": "cr:path",
    "prov": "http://www.w3.org/ns/prov#",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


@dataclass(frozen=True)
class ReleaseFile:
    object_id: str
    path: str
    description: str
    encoding_format: str
    records: bool = False


DERIVED_RELEASE_FILES = (
    ReleaseFile(
        "part1-dimension-summary",
        "data/analysis/tables/part1_dimension_summary.csv",
        "Part 1 cooperation rates by model and experimental dimension.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part1-factor-decomposition",
        "data/analysis/tables/part1_factor_decomposition.csv",
        "Descriptive balanced main-effect variance decomposition for Part 1 binary cooperation.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part1-frame-effects",
        "data/analysis/tables/part1_frame_effects.csv",
        "Part 1 frame-level cooperation rates and percentage-point effects.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part1-model-summary",
        "data/analysis/tables/part1_model_summary.csv",
        "Model-level focal-dilemma cooperation summary table.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part1-prompt-sensitivity",
        "data/analysis/tables/part1_prompt_sensitivity.csv",
        "Per-model prompt-sensitivity ranges across Part 1 dimensions.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part2-model-summary",
        "data/analysis/tables/part2_model_summary.csv",
        "Model-level summary of stored OPTION_A tokens and mechanically downstream state in contract-mismatched Part 2 traces.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part2-run-summary",
        "data/analysis/tables/part2_run_summary.csv",
        "Run-level summary of stored OPTION_A tokens and mechanically downstream state in contract-mismatched Part 2 traces.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "validation-report",
        "data/analysis/validation/validation_report.json",
        "Graph-independent validation report for paper-used data files.",
        "application/json",
    ),
    ReleaseFile(
        "run-manifest",
        "data/analysis/run_manifest.jsonl",
        "Manifest linking raw runs, metadata sidecars, validation status, and derived analysis artifacts.",
        "application/x-ndjson",
    ),
)


def _raw_release_files(part: int) -> tuple[ReleaseFile, ...]:
    raw_directory = REPOSITORY_ROOT / "data" / "raw" / f"part_{part}"
    paths = sorted(raw_directory.glob("*.csv"))
    expected = 13
    if len(paths) != expected:
        raise RuntimeError(
            f"expected {expected} release-safe Part {part} raw CSVs, found {len(paths)}"
        )
    if part == 1:
        description = (
            "Raw Part 1 focal-dilemma prompt, action-token, and justification trace."
        )
    elif part == 2:
        description = (
            "Raw Part 2 prompt--engine contract-audit token and state trace; not a "
            "commons-preference measurement."
        )
    else:  # pragma: no cover - construction is fixed above
        raise ValueError("only release-safe Parts 1 and 2 may enter Croissant metadata")
    return tuple(
        ReleaseFile(
            object_id=f"part{part}-raw-{index:02d}",
            path=path.relative_to(REPOSITORY_ROOT).as_posix(),
            description=description,
            encoding_format="text/csv",
            records=True,
        )
        for index, path in enumerate(paths, start=1)
    )


RAW_RELEASE_FILES = _raw_release_files(1) + _raw_release_files(2)
RELEASE_FILES = RAW_RELEASE_FILES + DERIVED_RELEASE_FILES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            raise ValueError(f"CSV has no header: {path}")
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
    *,
    repository_root: Path = REPOSITORY_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    dataset_url: str | None = None,
) -> dict[str, object]:
    """Return deterministic Croissant JSON-LD for the current release files."""

    dataset_url = _validate_dataset_url(dataset_url)
    distributions: list[dict[str, object]] = []
    record_sets: list[dict[str, object]] = []

    for release_file in RELEASE_FILES:
        source_path = (repository_root / release_file.path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"missing Croissant distribution: {source_path}")
        try:
            content_url = Path(
                os.path.relpath(source_path, output_path.parent.resolve())
            ).as_posix()
        except ValueError as error:
            raise ValueError(f"cannot make {source_path} relative to {output_path}") from error
        distributions.append(
            {
                "@type": "cr:FileObject",
                "@id": release_file.object_id,
                "name": source_path.name,
                "description": release_file.description,
                "contentUrl": content_url,
                "contentSize": f"{source_path.stat().st_size} B",
                "encodingFormat": release_file.encoding_format,
                "sha256": _sha256(source_path),
            }
        )
        if release_file.records:
            columns, types = _csv_schema(source_path)
            record_set_id = f"{release_file.object_id}-records"
            fields = [
                {
                    "@type": "cr:Field",
                    "@id": f"{record_set_id}/{column}",
                    "name": column,
                    "description": f"Column `{column}` from {source_path.name}.",
                    "dataType": types[column],
                    "source": {
                        "fileObject": {"@id": release_file.object_id},
                        "extract": {"column": column},
                    },
                }
                for column in columns
            ]
            record_sets.append(
                {
                    "@type": "cr:RecordSet",
                    "@id": record_set_id,
                    "name": f"{source_path.stem.replace('_', ' ').title()} records",
                    "description": release_file.description,
                    "field": fields,
                }
            )

    metadata: dict[str, object] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": [CORE_SPEC, RAI_SPEC],
        "name": "Prosocial Cost-Shifting Bench",
        "description": (
            "A negative benchmark-audit artifact. Part 1 contains descriptive action-token "
            "profiles for hypothetical dilemmas. Part 2 contains traces from a prompt--engine "
            "contract mismatch and supports protocol diagnosis and transition replay, not "
            "commons-preference measurement. The legacy Part 0 label audit is documented in "
            "the paper, but harmful raw content, invalid refusal rates, and their cross-part "
            "derivatives are not distributed."
        ),
        "version": DATASET_VERSION,
        "cr:sdVersion": METADATA_VERSION,
        "dateCreated": DATE_CREATED,
        "datePublished": DATE_PUBLISHED,
        "dateModified": DATE_MODIFIED,
        "creator": [{"@type": "sc:Organization", "name": "Anonymous Authors"}],
        "publisher": {"@type": "sc:Organization", "name": "Anonymous Authors"},
        "license": "https://opensource.org/license/mit",
        "sdLicense": "https://opensource.org/license/mit",
        "citeAs": (
            "Anonymous Authors. When a Benchmark Fails Its Audit. Anonymous conference submission, 2026."
        ),
        "keywords": [
            "large language models",
            "behavioral evaluation",
            "safety refusal",
            "cooperation",
            "prompt-engine contract audit",
            "agent simulation",
        ],
        "isAccessibleForFree": True,
        "rai:hasSyntheticData": True,
        "prov:wasDerivedFrom": [
            {
                "@type": "sc:CreativeWork",
                "name": "April 2026 Part 1 and Part 2 model-generated pilot traces",
                "description": (
                    "The exact release-safe raw CSV distributions and metadata sidecars "
                    "bound by the repository run manifest and provenance checks."
                ),
            }
        ],
        "prov:wasGeneratedBy": {
            "@type": "sc:SoftwareApplication",
            "name": "Prosocial Cost-Shifting Bench deterministic analysis pipeline",
            "softwareVersion": DATASET_VERSION,
            "description": (
                "analysis.validation, analysis.summarize_results, analysis.build_manifest, "
                "and analysis.build_croissant_metadata"
            ),
        },
        "conditionsOfAccess": (
            "The anonymous release exposes derived tables, validation reports, the "
            "run manifest, metadata, code, figures, and raw Part 1/Part 2 traces. Raw "
            "Part 0 harmful content and every invalid legacy-label rate/correlation "
            "table or plot are excluded."
        ),
        "rai:dataCollection": (
            "Rows are generated by executable experiments and summarized by the repository "
            "analysis pipeline. Part 0's legacy labels are excluded after a response-only "
            "audit found material instability; Part 1 uses constrained choices in hypothetical "
            "dilemmas; Part 2 uses repeated stateless calls under a prompt whose stated score "
            "contract was never implemented by the engine."
        ),
        "rai:dataCollectionType": ["Experiments", "Software Collection"],
        "rai:dataCollectionRawData": (
            "Stored model responses, structured decisions, simulation state, and "
            "metadata sidecars. Raw Part 0 harmful content is not distributed in the "
            "anonymous release."
        ),
        "rai:dataAnnotationProtocol": (
            "No Part 0 model-level label distribution is released. Part 1 action labels and "
            "Part 2 stored OPTION_A/OPTION_B tokens are deterministic mappings from constrained "
            "outputs. Part 2 state columns are mechanically downstream of those tokens under "
            "the recorded engine, not behavioral annotations. Human labels are never imputed."
        ),
        "rai:machineAnnotationTools": [
            "Deterministic Part 1/Part 2 analysis code."
        ],
        "rai:dataPreprocessingProtocol": [
            "Exclude invalid legacy Part 0 labels and all dependent model-level summaries.",
            "Aggregate Part 1 self-direct choices as the primary cooperation measure while retaining role-conditioned diagnostics.",
            "Summarize each Part 2 contract-mismatched trajectory and replay its deterministic resource and population transitions without treating it as a commons-preference estimate.",
            "Exclude raw Part 0 harmful prompts and completions from the anonymous supplement.",
        ],
        "rai:dataUseCases": [
            "Protocol diagnosis, trace validation, and reproduction of Part 1 descriptive action-label summaries.",
            "Inspection of the Part 2 prompt--engine mismatch and deterministic transition replay; not behavioral comparison.",
            "Validation and reproduction of the released paper's aggregate analysis.",
            "Not supported: training harmful-compliance models, globally ranking models as prosocial, or treating pilot results as production safety certification.",
        ],
        "rai:dataLimitations": [
            "Descriptive pilot over a limited local/open-weight model cohort; results do not estimate all models or deployment settings.",
            "Part 0 uses a fixed multilingual harmful-prompt set; response-only refusal claims still require the documented human audit.",
            "Part 1 uses fixed one-shot focal and role-conditioned prompts.",
            "Part 2's prompt described private and group scores that the engine never computed, stored, or fed back; its traces cannot estimate commons preference.",
            "Part 2 also uses homogeneous same-model policies, simplified reserve dynamics, stateless calls, and one pilot trajectory per model.",
            "The artifact does not measure moral agency, general human prosociality, or deployment safety outside the stated protocols.",
        ],
        "rai:dataBiases": [
            "Selection bias can arise from the fixed harmful-prompt sample, English/Chinese/Russian coverage, benchmark-derived sources, and locally runnable model cohort.",
            "Label bias can arise from the automated Part 0 judge and deterministic mappings from constrained action tokens.",
            "The contract-mismatched repeated-resource protocol abstracts away cultural, institutional, heterogeneous-agent, and deployment-context effects.",
        ],
        "rai:personalSensitiveInformation": [
            "No human-subject records or personally identifying user data are collected. Raw Part 0 model outputs may contain unsafe generated content and are excluded from the anonymous release."
        ],
        "rai:dataSocialImpact": (
            "The artifact improves traceability of multi-axis LLM behavior, but limited "
            "coverage and simplified games can support overclaiming or unfair model "
            "comparisons. Excluding raw harmful content and documenting uncertainty "
            "reduces, but does not eliminate, these risks."
        ),
        "rai:dataReleaseMaintenancePlan": [
            "The checked metadata is regenerated with analysis.build_croissant_metadata whenever a released distribution changes; checksums, sizes, schemas, metadata version, and modification date must be reviewed before publication."
        ],
        "distribution": distributions,
        "recordSet": record_sets,
    }
    if dataset_url is not None:
        metadata["url"] = dataset_url
    return metadata


def write_metadata(
    output_path: Path = DEFAULT_OUTPUT, *, dataset_url: str | None = None
) -> dict[str, object]:
    metadata = build_metadata(output_path=output_path, dataset_url=dataset_url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset-url",
        default=os.environ.get("CROISSANT_DATASET_URL"),
        help=(
            "Public HTTPS dataset landing page. Omitted for the checked anonymous-review "
            "artifact until the review platform provides a stable non-identifying URL."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the output differs from freshly generated metadata.",
    )
    args = parser.parse_args()
    metadata = build_metadata(output_path=args.output, dataset_url=args.dataset_url)
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
