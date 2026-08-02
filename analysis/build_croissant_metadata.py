"""Build the paper-facing MLCommons Croissant metadata.

The checked-in metadata describes only the release-safe, derived analysis files.
Raw Part 0 prompts and completions are intentionally outside this metadata graph.
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


RELEASE_FILES = (
    ReleaseFile(
        "cross-part-correlations",
        "data/analysis/tables/cross_part_correlations.csv",
        "Model-level Pearson, Spearman, and Kendall cross-part correlations with uncertainty and sensitivity summaries.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "cross-part-model-summary",
        "data/analysis/tables/cross_part_model_summary.csv",
        "Joined model-level summary table for cross-part diagnostics.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part0-language-robustness",
        "data/analysis/tables/part0_language_robustness.csv",
        "Model-level multilingual refusal robustness summary.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part0-model-summary",
        "data/analysis/tables/part0_model_summary.csv",
        "Model-level safety-refusal summary table.",
        "text/csv",
        True,
    ),
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
        "Model-level commons-restraint, survival, reserve, and trajectory summary table.",
        "text/csv",
        True,
    ),
    ReleaseFile(
        "part2-run-summary",
        "data/analysis/tables/part2_run_summary.csv",
        "Run-level commons-restraint, survival, reserve, and trajectory summary table.",
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
        "name": "Prosocial Readiness Bench",
        "description": (
            "A three-part behavioral evaluation artifact measuring observable safety "
            "refusal, focal and role-conditioned choices in hypothetical dilemmas, "
            "and commons restraint in a controlled population microworld. This "
            "anonymous-review release contains derived analysis tables, validation "
            "output, and a run manifest for the pilot model cohort."
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
            "Anonymous Authors. Safety Beyond Refusal: A Multi-Axis Benchmark for "
            "Prosocial Readiness in Language Models. Anonymous conference submission, 2026."
        ),
        "keywords": [
            "large language models",
            "behavioral evaluation",
            "safety refusal",
            "cooperation",
            "commons restraint",
            "agent simulation",
        ],
        "isAccessibleForFree": True,
        "conditionsOfAccess": (
            "The anonymous release exposes derived tables, validation reports, the "
            "run manifest, metadata, code, figures, and Part 1/Part 2 outputs. Raw "
            "Part 0 harmful prompts, source CSVs, metadata sidecars, and completions "
            "are excluded pending separate safety review and access controls."
        ),
        "rai:dataCollection": (
            "Rows are generated by executable experiments in experiments/part0, "
            "experiments/part1, and experiments/part2 and summarized by the repository "
            "analysis pipeline. Part 0 uses harmful-request prompts and automated "
            "response-only judgments; Part 1 uses constrained choices in hypothetical "
            "dilemmas; Part 2 uses repeated stateless calls in a common-resource simulation."
        ),
        "rai:dataCollectionType": ["Experiments", "Software Collection"],
        "rai:dataCollectionRawData": (
            "Stored model responses, structured decisions, simulation state, and "
            "metadata sidecars. Raw Part 0 harmful content is not distributed in the "
            "anonymous release."
        ),
        "rai:dataAnnotationProtocol": (
            "Part 0 final-response labels come from a response-only automated judge "
            "with structured output and explicit unjudged failures. Part 1 cooperation "
            "labels and Part 2 restraint/outcome measures are deterministic mappings "
            "from constrained actions and simulation logs. Human labels are never imputed."
        ),
        "rai:machineAnnotationTools": [
            "Repository response-only Part 0 judge and deterministic Part 1/Part 2 analysis code."
        ],
        "rai:dataPreprocessingProtocol": [
            "Normalize constrained action labels and use response-only Part 0 labels when present.",
            "Aggregate Part 1 self-direct choices as the primary cooperation measure while retaining role-conditioned diagnostics.",
            "Aggregate Part 2 trajectories with equal run weighting and compute survival, resource, and restraint summaries.",
            "Exclude raw Part 0 harmful prompts and completions from the anonymous supplement.",
        ],
        "rai:dataUseCases": [
            "Behavioral auditing and fixed-protocol comparison of refusal, cooperation, and commons-restraint metrics.",
            "Validation and reproduction of the released paper's aggregate analysis.",
            "Not supported: training harmful-compliance models, globally ranking models as prosocial, or treating pilot results as production safety certification.",
        ],
        "rai:dataLimitations": [
            "Descriptive pilot over a limited local/open-weight model cohort; results do not estimate all models or deployment settings.",
            "Part 0 uses a fixed multilingual harmful-prompt set; response-only refusal claims still require the documented human audit.",
            "Part 1 uses fixed one-shot focal and role-conditioned prompts.",
            "Part 2 uses homogeneous same-model societies, simplified reserve dynamics, stateless calls, and one pilot trajectory per model.",
            "The artifact does not measure moral agency, general human prosociality, or deployment safety outside the stated protocols.",
        ],
        "rai:dataBiases": [
            "Selection bias can arise from the fixed harmful-prompt sample, English/Chinese/Russian coverage, benchmark-derived sources, and locally runnable model cohort.",
            "Label bias can arise from the automated Part 0 judge and deterministic mappings from constrained action tokens.",
            "The commons simulation abstracts away cultural, institutional, heterogeneous-agent, and deployment-context effects.",
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
