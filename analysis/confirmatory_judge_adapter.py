"""Integrity-preserving Part 0 confirmatory input for the blinded judge audit.

The adapter deliberately retains only the harmful/control request, the visible
response, and the frozen automated decision.  Provider reasoning and judge
rationales are neither read into nor written to the audit population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.misc.attempt_log import attempt_log_path_for_csv, verify_attempt_log_metadata
from experiments.misc.run_metadata import sha256_file, stable_json_hash, validate_metadata_integrity
from experiments.part0 import confirmatory_runner as part0_runner


SCHEMA_VERSION = 2
ARTIFACT_TYPE = "part0_confirmatory_judge_audit_input"
RECORD_TYPE = "part0_confirmatory_judge_audit_record"
LANGUAGES = ("english", "chinese", "russian")
LABELS = ("complied", "denied", "unclear")
MANIFEST_NAME = "confirmatory_audit_input.manifest.json"
RECORDS_NAME = "confirmatory_audit_input.jsonl"
_RECORD_KEYS = {
    "schema_version", "record_type", "item_id", "language", "automated_label",
    "provider", "model", "request_original", "request_english",
    "response_original", "response_english", "source_run", "source_row",
    "source_trial_id", "source_record_sha256", "record_sha256",
    "target_id", "arm", "source", "category", "semantic_cluster_id",
    "generation_block",
}
_SOURCE_KEYS = {
    "run_directory", "plan_path", "plan_sha256", "plan_file_sha256",
    "results_path", "results_sha256", "result_count", "attempts_path",
    "attempts_sha256", "metadata_path", "metadata_sha256",
    "metadata_file_sha256", "registry_path", "registry_sha256", "provider",
    "model",
}
_MANIFEST_KEYS = {
    "schema_version", "artifact_type", "created_at_utc", "records_path",
    "records_sha256", "record_count", "native_unclear_count", "languages",
    "automated_labels", "strata", "source_runs", "manifest_sha256",
}


class ConfirmatoryJudgeAdapterError(ValueError):
    """A private source or adapted audit bundle is incomplete or tampered."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sealed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result[field] = stable_json_hash(result)
    return result


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryJudgeAdapterError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ConfirmatoryJudgeAdapterError(f"{label} root must be an object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ConfirmatoryJudgeAdapterError(f"records are not readable UTF-8: {path}") from error
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ConfirmatoryJudgeAdapterError(f"records contain a blank line at {number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ConfirmatoryJudgeAdapterError(f"invalid record JSON at line {number}") from error
        if not isinstance(row, dict):
            raise ConfirmatoryJudgeAdapterError(f"record {number} is not an object")
        rows.append(row)
    return rows


def _native_run(run_directory: Path, registry_path: Path, registry_sha256: str) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    directory = run_directory.resolve()
    plan_path = directory / "part0_confirmatory_plan.json"
    results_path = directory / "part0_confirmatory_results.jsonl"
    metadata_path = directory / "part0_confirmatory_meta.json"
    attempts_path = attempt_log_path_for_csv(results_path)
    for path in (plan_path, results_path, metadata_path, attempts_path, registry_path):
        if not path.is_file():
            raise ConfirmatoryJudgeAdapterError(f"required Part 0 artifact is missing: {path}")
    plan = _load_object(plan_path, "Part 0 plan")
    metadata = _load_object(metadata_path, "Part 0 metadata")
    try:
        loaded = part0_runner.load_production_registry(
            registry_path, expected_sha256=registry_sha256
        )
        part0_runner.validate_execution_plan(plan, loaded)
        summary = part0_runner.summarize_results(results_path)
        validate_metadata_integrity(metadata, required=True)
        verify_attempt_log_metadata(
            attempts_path, metadata.get("attempt_log", {}), require_hash_chain=True
        )
        part0_runner._validate_attempt_result_reconciliation(
            attempts_path=attempts_path,
            results_path=results_path,
            schedule=plan["schedule"],
            extractor_route=plan["extractor_route"],
            judge_route=plan["judge_route"],
        )
    except Exception as error:
        raise ConfirmatoryJudgeAdapterError(
            f"Part 0 native integrity validation failed for {directory}: {error}"
        ) from error
    if (
        plan.get("execution_mode") != "production"
        or plan.get("analysis_eligibility", {}).get("eligible") is not True
        or metadata.get("status") != "complete"
        or metadata.get("run_contract") != part0_runner._run_contract(plan)
        or metadata.get("results") != summary
        or summary["total_results"] != len(plan.get("schedule", []))
    ):
        raise ConfirmatoryJudgeAdapterError("Part 0 run is not complete production evidence")
    if plan.get("languages") != list(LANGUAGES):
        raise ConfirmatoryJudgeAdapterError("Part 0 plan does not use the exact three languages")
    records = part0_runner._load_result_records(results_path)
    if len(records) != len(plan["schedule"]):
        raise ConfirmatoryJudgeAdapterError("Part 0 result count differs from the frozen plan")
    item_by_id = {
        str(item["base_prompt_id"]): item for item in loaded.payload["stimuli"]
    }
    output: list[dict[str, Any]] = []
    unclear = 0
    model = str(plan["subject_route"]["route"])
    provider = str(plan["subject_route"]["provider"])
    target_id = str(plan["subject_route"].get("identity", {}).get("id", ""))
    if not target_id:
        raise ConfirmatoryJudgeAdapterError("Part 0 subject route lacks its registry target ID")
    for source_row, (record, trial) in enumerate(zip(records, plan["schedule"], strict=True), start=1):
        if record.get("trial_id") != trial.get("trial_id"):
            raise ConfirmatoryJudgeAdapterError("Part 0 result order differs from its plan")
        label = record.get("judge_label")
        if label == "UNCLEAR":
            unclear += 1
        mapped = {
            "COMPLIANCE": "complied",
            "REFUSAL": "denied",
            "UNCLEAR": "unclear",
        }.get(label)
        visible = record.get("visible_response")
        language = record.get("language")
        if record.get("status") != "SCORED" or mapped is None or not isinstance(visible, str) or not visible.strip():
            raise ConfirmatoryJudgeAdapterError(
                f"Part 0 record {source_row} is not a scored auditable decision"
            )
        if language not in LANGUAGES:
            raise ConfirmatoryJudgeAdapterError("Part 0 result has an invalid language")
        source_item = item_by_id.get(str(record.get("base_prompt_id")))
        if source_item is None:
            raise ConfirmatoryJudgeAdapterError("Part 0 result references an unknown stimulus")
        english = source_item["localized_stimuli"]["english"]["text"]
        provenance = record.get("source_provenance")
        if not isinstance(provenance, list) or not provenance:
            raise ConfirmatoryJudgeAdapterError("Part 0 result lacks source provenance")
        sources = sorted({str(item.get("source_name", "")).strip() for item in provenance})
        categories = sorted(
            {
                str(
                    item.get("source_category")
                    or item.get("functional_category")
                    or item.get("semantic_category")
                    or "uncategorized"
                ).strip()
                for item in provenance
            }
        )
        if any(not value for value in (*sources, *categories)):
            raise ConfirmatoryJudgeAdapterError("Part 0 result has empty source/category provenance")
        item_id = "ca_" + hashlib.sha256(
            f"{provider}\0{model}\0{record['trial_id']}".encode("utf-8")
        ).hexdigest()[:24]
        adapted = {
            "schema_version": SCHEMA_VERSION,
            "record_type": RECORD_TYPE,
            "item_id": item_id,
            "language": language,
            "automated_label": mapped,
            "provider": provider,
            "model": model,
            "target_id": target_id,
            "arm": record["arm"],
            "source": "+".join(sources),
            "category": "+".join(categories),
            "semantic_cluster_id": record["semantic_cluster_id"],
            "generation_block": record["block"],
            "request_original": record["prompt_sent"],
            "request_english": english,
            "response_original": visible,
            "response_english": "",
            "source_run": str(directory),
            "source_row": source_row,
            "source_trial_id": record["trial_id"],
            "source_record_sha256": record["record_sha256"],
        }
        output.append(_sealed(adapted, "record_sha256"))
    source = {
        "run_directory": str(directory),
        "plan_path": str(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": sha256_file(plan_path),
        "results_path": str(results_path),
        "results_sha256": summary["sha256"],
        "result_count": summary["total_results"],
        "attempts_path": str(attempts_path),
        "attempts_sha256": metadata["attempt_log"]["sha256"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata["metadata_sha256"],
        "metadata_file_sha256": sha256_file(metadata_path),
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": registry_sha256,
        "provider": provider,
        "model": model,
    }
    return output, source, unclear


def build_confirmatory_audit_input(
    run_directories: Sequence[str | Path],
    *,
    registry_path: str | Path,
    registry_sha256: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Validate production runs and atomically write one private audit population."""

    if not run_directories:
        raise ConfirmatoryJudgeAdapterError("at least one Part 0 run directory is required")
    resolved_runs = [Path(value).resolve() for value in run_directories]
    if len(set(resolved_runs)) != len(resolved_runs):
        raise ConfirmatoryJudgeAdapterError("Part 0 run directories contain duplicates")
    registry = Path(registry_path).resolve()
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    unclear = 0
    seen_models: set[tuple[str, str]] = set()
    seen_items: set[str] = set()
    for directory in resolved_runs:
        run_rows, source, run_unclear = _native_run(directory, registry, registry_sha256)
        identity = (source["provider"], source["model"])
        if identity in seen_models:
            raise ConfirmatoryJudgeAdapterError(f"duplicate subject route: {identity[1]}")
        seen_models.add(identity)
        for row in run_rows:
            if row["item_id"] in seen_items:
                raise ConfirmatoryJudgeAdapterError(f"duplicate adapted item: {row['item_id']}")
            seen_items.add(row["item_id"])
        rows.extend(run_rows)
        sources.append(source)
        unclear += run_unclear
    counts = Counter((row["language"], row["automated_label"]) for row in rows)
    retained_unclear = sum(counts[(language, "unclear")] for language in LANGUAGES)
    if unclear != retained_unclear:
        raise ConfirmatoryJudgeAdapterError(
            "native UNCLEAR count differs from the retained audit population"
        )
    expected = {(language, label) for language in LANGUAGES for label in LABELS}
    if set(counts) != expected or any(counts[key] <= 0 for key in expected):
        missing = sorted(expected - set(counts))
        raise ConfirmatoryJudgeAdapterError(
            f"adapted population lacks exact three-language three-label strata: {missing}"
        )
    output = Path(output_directory).resolve()
    if output.exists():
        raise ConfirmatoryJudgeAdapterError(f"output directory already exists: {output}")
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    records_path = output / RECORDS_NAME
    records_bytes = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    _atomic_write(records_path, records_bytes)
    strata = {
        language: {label: counts[(language, label)] for label in LABELS}
        for language in LANGUAGES
    }
    manifest = _sealed(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "created_at_utc": _utc_now(),
            "records_path": str(records_path),
            "records_sha256": sha256_file(records_path),
            "record_count": len(rows),
            "native_unclear_count": unclear,
            "languages": list(LANGUAGES),
            "automated_labels": list(LABELS),
            "strata": strata,
            "source_runs": sources,
        },
        "manifest_sha256",
    )
    _atomic_write(
        output / MANIFEST_NAME,
        (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    return manifest


def load_confirmatory_audit_input(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Revalidate an adapter manifest, its records, and all native source bytes."""

    manifest_path = Path(path).resolve()
    manifest = _load_object(manifest_path, "confirmatory audit manifest")
    if set(manifest) != _MANIFEST_KEYS or manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ConfirmatoryJudgeAdapterError("confirmatory audit manifest schema is not exact")
    if manifest.get("manifest_sha256") != stable_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise ConfirmatoryJudgeAdapterError("confirmatory audit manifest hash mismatch")
    if manifest.get("languages") != list(LANGUAGES) or manifest.get("automated_labels") != list(LABELS):
        raise ConfirmatoryJudgeAdapterError("confirmatory audit strata labels changed")
    records_path = Path(str(manifest["records_path"])).resolve()
    if records_path.parent != manifest_path.parent or records_path.name != RECORDS_NAME:
        raise ConfirmatoryJudgeAdapterError("confirmatory audit records path escaped its bundle")
    if not records_path.is_file() or sha256_file(records_path) != manifest["records_sha256"]:
        raise ConfirmatoryJudgeAdapterError("confirmatory audit records hash mismatch")
    records = _load_jsonl(records_path)
    if len(records) != manifest.get("record_count"):
        raise ConfirmatoryJudgeAdapterError("confirmatory audit record count mismatch")
    ids: set[str] = set()
    counts: Counter[tuple[str, str]] = Counter()
    for row in records:
        if set(row) != _RECORD_KEYS or row.get("schema_version") != SCHEMA_VERSION or row.get("record_type") != RECORD_TYPE:
            raise ConfirmatoryJudgeAdapterError("confirmatory audit record schema is not exact")
        if row.get("record_sha256") != stable_json_hash(
            {key: value for key, value in row.items() if key != "record_sha256"}
        ):
            raise ConfirmatoryJudgeAdapterError("confirmatory audit record hash mismatch")
        if row["item_id"] in ids:
            raise ConfirmatoryJudgeAdapterError("confirmatory audit item IDs are duplicated")
        ids.add(row["item_id"])
        if row["language"] not in LANGUAGES or row["automated_label"] not in LABELS:
            raise ConfirmatoryJudgeAdapterError("confirmatory audit record has an invalid stratum")
        for field in ("provider", "model", "request_original", "request_english", "response_original"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ConfirmatoryJudgeAdapterError(f"confirmatory audit record has empty {field}")
        for field in ("target_id", "source", "category", "semantic_cluster_id"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ConfirmatoryJudgeAdapterError(f"confirmatory audit record has empty {field}")
        if row.get("arm") not in {"harmful", "control"} or row.get("generation_block") not in {1, 2, 3}:
            raise ConfirmatoryJudgeAdapterError("confirmatory audit record has invalid arm/block")
        counts[(row["language"], row["automated_label"])] += 1
    expected_strata = {
        language: {label: counts[(language, label)] for label in LABELS}
        for language in LANGUAGES
    }
    if manifest.get("strata") != expected_strata or any(
        counts[(language, label)] <= 0 for language in LANGUAGES for label in LABELS
    ):
        raise ConfirmatoryJudgeAdapterError("confirmatory audit manifest strata are incomplete")
    sources = manifest.get("source_runs")
    if not isinstance(sources, list) or not sources:
        raise ConfirmatoryJudgeAdapterError("confirmatory audit source runs are missing")
    source_models: set[tuple[str, str]] = set()
    regenerated_records: list[dict[str, Any]] = []
    regenerated_sources: list[dict[str, Any]] = []
    regenerated_unclear = 0
    for source in sources:
        if not isinstance(source, dict) or set(source) != _SOURCE_KEYS:
            raise ConfirmatoryJudgeAdapterError("confirmatory audit source schema is not exact")
        identity = (source["provider"], source["model"])
        if identity in source_models:
            raise ConfirmatoryJudgeAdapterError("confirmatory audit source routes are duplicated")
        source_models.add(identity)
        for path_key, hash_key in (
            ("plan_path", "plan_file_sha256"), ("results_path", "results_sha256"),
            ("attempts_path", "attempts_sha256"), ("metadata_path", "metadata_file_sha256"),
            ("registry_path", "registry_sha256"),
        ):
            source_path = Path(str(source[path_key]))
            if not source_path.is_file() or sha256_file(source_path) != source[hash_key]:
                raise ConfirmatoryJudgeAdapterError(f"confirmatory audit source changed: {path_key}")
        current_rows, current_source, current_unclear = _native_run(
            Path(str(source["run_directory"])),
            Path(str(source["registry_path"])),
            str(source["registry_sha256"]),
        )
        regenerated_records.extend(current_rows)
        regenerated_sources.append(current_source)
        regenerated_unclear += current_unclear
    if (
        records != regenerated_records
        or sources != regenerated_sources
        or manifest.get("native_unclear_count") != regenerated_unclear
        or manifest.get("native_unclear_count")
        != sum(row["automated_label"] == "unclear" for row in records)
    ):
        raise ConfirmatoryJudgeAdapterError(
            "confirmatory audit bundle differs from regenerated native evidence"
        )
    return records, manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adapt completed private Part 0 confirmatory runs for blinded audit.")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_confirmatory_audit_input(
            args.run_dir,
            registry_path=args.registry,
            registry_sha256=args.registry_sha256,
            output_directory=args.output_dir,
        )
    except ConfirmatoryJudgeAdapterError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"manifest": str(Path(args.output_dir).resolve() / MANIFEST_NAME), "record_count": manifest["record_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
