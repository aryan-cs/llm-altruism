"""Validate and summarize the two bounded Part 1 semantic-repair campaigns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "inference_hub_semantic_invalid_repair_analysis_v1"
MAX_SEMANTIC_ROUNDS = 8
KINDS = {
    "part1": {
        "manifest_type": "inference_hub_part1_semantic_invalid_repair_v1",
        "payload_type": "inference_hub_part1_semantic_invalid_repair_outcomes_v1",
        "source_count_key": "source_scheduled_unit_count",
        "source_invalid_key": "source_first_attempt_invalid_count",
    },
    "role": {
        "manifest_type": "inference_hub_part1_role_semantic_invalid_repair_v1",
        "payload_type": "inference_hub_part1_role_semantic_invalid_repair_outcomes_v1",
        "source_count_key": "source_scheduled_draw_count",
        "source_invalid_key": "source_first_response_invalid_count",
    },
}


class SemanticRepairAnalysisError(RuntimeError):
    """A repair artifact is incomplete, inconsistent, or unsafe to publish."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _self_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in value.items() if k != "evidence_sha256"})).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticRepairAnalysisError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise SemanticRepairAnalysisError(f"{label} must be an object.")
    return value


def _manifest_path(value: Path) -> Path:
    return (value / "private/manifest.json" if value.is_dir() else value).resolve()


def _load(value: Path, kind: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = KINDS[kind]
    manifest_path = _manifest_path(value)
    manifest = _read_object(manifest_path, f"{kind} repair manifest")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != contract["manifest_type"]
        or manifest.get("evidence_sha256") != _self_hash(manifest)
        or manifest.get("complete") is not True
        or not manifest.get("completed_at_utc")
        or manifest.get("primary_records_mutated") is not False
        or manifest.get("primary_denominators_changed") is not False
        or manifest.get("promotion_permitted") is not False
    ):
        raise SemanticRepairAnalysisError(f"{kind} repair manifest contract failed.")
    max_rounds = manifest.get("max_semantic_rounds")
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or not 1 <= max_rounds <= MAX_SEMANTIC_ROUNDS
    ):
        raise SemanticRepairAnalysisError(
            f"{kind} repair round budget is invalid."
        )

    summary = manifest.get("summary")
    if not isinstance(summary, Mapping):
        raise SemanticRepairAnalysisError(f"{kind} repair summary is missing.")
    source_count = summary.get(contract["source_count_key"])
    source_invalid = summary.get(contract["source_invalid_key"])
    repaired = summary.get("repaired_valid_separate_count")
    unrepaired = summary.get("unrepaired_after_bounded_rounds_count")
    if (
        any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (source_count, source_invalid, repaired, unrepaired))
        or summary.get("primary_denominator") != source_count
        or summary.get("primary_denominator_changed") is not False
        or repaired + unrepaired != source_invalid
    ):
        raise SemanticRepairAnalysisError(f"{kind} repair summary does not reconcile.")

    reference = manifest.get("sanitized_artifact")
    if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
        raise SemanticRepairAnalysisError(f"{kind} sanitized repair reference is missing.")
    payload_path = Path(reference["path"]).resolve()
    run_root = manifest_path.parent.parent.resolve()
    try:
        payload_path.relative_to(run_root / "sanitized")
    except ValueError as error:
        raise SemanticRepairAnalysisError(f"{kind} sanitized repair escaped its run.") from error
    payload = _read_object(payload_path, f"{kind} repair outcomes")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_type") != contract["payload_type"]
        or payload.get("evidence_sha256") != _self_hash(payload)
        or reference.get("file_sha256") != _sha_file(payload_path)
        or reference.get("evidence_sha256") != payload.get("evidence_sha256")
        or payload.get("raw_text_included") is not False
        or payload.get("original_records_mutated") is not False
        or payload.get("primary_denominators_changed") is not False
        or payload.get("repaired_estimates_separate_only") is not True
        or payload.get("promotion_permitted") is not False
        or not isinstance(payload.get("rows"), list)
        or len(payload["rows"]) != source_invalid
    ):
        raise SemanticRepairAnalysisError(f"{kind} sanitized repair contract failed.")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for index, raw in enumerate(payload["rows"]):
        if not isinstance(raw, Mapping):
            raise SemanticRepairAnalysisError(f"{kind} row {index} is not an object.")
        row = dict(raw)
        required = ("target_id", "upstream_provider", "model", "trial_id", "rounds_reserved", "rounds_with_retained_response", "repair_status", "repaired_format_valid")
        if any(field not in row for field in required):
            raise SemanticRepairAnalysisError(f"{kind} row {index} lacks required fields.")
        identity = tuple(str(row[field]) for field in ("target_id", "trial_id"))
        if kind == "role":
            if not isinstance(row.get("frame_id"), str) or row.get("frame_pooled") is not False or row.get("model_pooled") is not False:
                raise SemanticRepairAnalysisError(f"role row {index} changed its frame contract.")
            identity += (str(row["frame_id"]),)
        if identity in seen:
            raise SemanticRepairAnalysisError(f"{kind} repair rows duplicate a source unit.")
        seen.add(identity)
        reserved = row["rounds_reserved"]
        retained = row["rounds_with_retained_response"]
        valid = row["repaired_format_valid"]
        if (
            isinstance(reserved, bool) or not isinstance(reserved, int) or not 0 <= reserved <= max_rounds
            or isinstance(retained, bool) or not isinstance(retained, int) or not 0 <= retained <= reserved
            or not isinstance(valid, bool)
            or row.get("original_format_valid") is not False
            or row.get("primary_record_mutated") is not False
            or row.get("primary_denominator_changed") is not False
            or row.get("repair_status") != ("repaired_valid_separate" if valid else "unrepaired_after_bounded_rounds")
        ):
            raise SemanticRepairAnalysisError(f"{kind} row {index} does not reconcile.")
        rows.append(row)
    if sum(bool(row["repaired_format_valid"]) for row in rows) != repaired:
        raise SemanticRepairAnalysisError(f"{kind} repaired count differs from rows.")
    return manifest, rows


def _aggregate(rows: Sequence[Mapping[str, Any]], *, role: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["target_id"]), str(row["upstream_provider"]), str(row["model"]))
        if role:
            key += (str(row["frame_id"]),)
        item = groups.setdefault(key, {"source_invalid_count": 0, "units_retried_count": 0, "repair_attempt_count": 0, "repaired_valid_count": 0})
        item["source_invalid_count"] += 1
        item["units_retried_count"] += int(int(row["rounds_reserved"]) > 0)
        item["repair_attempt_count"] += int(row["rounds_reserved"])
        item["repaired_valid_count"] += int(bool(row["repaired_format_valid"]))
    output = []
    for key, counts in sorted(groups.items()):
        source_invalid = counts["source_invalid_count"]
        repaired = counts["repaired_valid_count"]
        record = {
            "target_id": key[0],
            "upstream_provider": key[1],
            "model": key[2],
            **counts,
            "still_invalid_count": source_invalid - repaired,
            "repair_rate_among_source_invalid": repaired / source_invalid,
            "primary_records_mutated": False,
            "primary_denominator_changed": False,
        }
        if role:
            record["frame_id"] = key[3]
        output.append(record)
    return output


def _escape(value: object) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(character, character) for character in str(value))


def _table(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: bool,
    max_rounds: int,
) -> str:
    caption = (
        f"Bounded Part 1 role-frame semantic-repair diagnostics. Each row is one exact target route and one distinct role frame with at least one format-invalid source response; Source invalid is the original first-response invalid count, Units retried is the number receiving at least one repair dispatch, Attempts is the total bounded repair dispatch count, Repaired valid is the number producing a valid separate response within at most {max_rounds} rounds, Still invalid is the number without a valid repair, and Repair rate is Repaired valid divided by Source invalid. Higher repair rate and lower Still invalid indicate better recoverability of the output contract, not better cooperation or general safety. Source invalids remain nonsuccesses in primary denominators; frames and models are not pooled."
        if role else
        f"Bounded Part 1 semantic-repair diagnostics. Each row is one exact target route with at least one format-invalid source response; Source invalid is the original first-attempt invalid count, Units retried is the number receiving at least one repair dispatch, Attempts is the total bounded repair dispatch count, Repaired valid is the number producing a valid separate response within at most {max_rounds} rounds, Still invalid is the number without a valid repair, and Repair rate is Repaired valid divided by Source invalid. Higher repair rate and lower Still invalid indicate better recoverability of the output contract, not better welfare preservation or general safety. Source invalids remain nonsuccesses in the primary 384-root denominator and repaired responses never replace them."
    )
    headers = ["Target route ID", "Provider", "Model ID"] + (["Frame"] if role else []) + ["Source invalid", "Units retried", "Attempts", "Repaired valid", "Still invalid", "Repair rate"]
    body = []
    for row in rows:
        values = [_escape(row["target_id"]), _escape(row["upstream_provider"]), _escape(row["model"])]
        if role:
            values.append(_escape(row["frame_id"]))
        values.extend(str(row[field]) for field in ("source_invalid_count", "units_retried_count", "repair_attempt_count", "repaired_valid_count", "still_invalid_count"))
        values.append(f"{100 * float(row['repair_rate_among_source_invalid']):.1f}\\%")
        body.append(" & ".join(values) + r" \\")
    columns = "llllrrrrrr" if role else "lllrrrrrr"
    return "\n".join([
        r"\par\addvspace{15pt}", r"\begin{table*}[tbp]", r"\centering",
        f"\\caption{{{caption}}}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{%", f"\\begin{{tabular}}{{{columns}}}", r"\toprule",
        " & ".join(headers) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}%", "}",
        r"\end{table*}", r"\par\addvspace{15pt}", "",
    ])


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(*, part1: Path, role: Path, output_dir: Path) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    tables: dict[str, list[dict[str, Any]]] = {}
    for kind, value in (("part1", part1), ("role", role)):
        manifest, rows = _load(value, kind)
        manifests[kind] = manifest
        tables[kind] = _aggregate(rows, role=kind == "role")
    if output_dir.exists():
        raise SemanticRepairAnalysisError("Output directory exists; refusing overwrite.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        assets: dict[str, dict[str, Any]] = {}
        for kind, rows in tables.items():
            stem = "part1_role_semantic_repair" if kind == "role" else "part1_semantic_repair"
            jsonl = temporary / f"{stem}.jsonl"
            csv_path = temporary / f"{stem}.csv"
            tex = temporary / f"{stem}.tex"
            _write_jsonl(jsonl, rows)
            _write_csv(csv_path, rows)
            tex.write_text(
                _table(
                    rows,
                    role=kind == "role",
                    max_rounds=int(
                        manifests[kind]["max_semantic_rounds"]
                    ),
                ),
                encoding="utf-8",
            )
            assets[kind] = {
                "row_count": len(rows),
                "jsonl": {"filename": jsonl.name, "file_sha256": _sha_file(jsonl)},
                "csv": {"filename": csv_path.name, "file_sha256": _sha_file(csv_path)},
                "latex": {"filename": tex.name, "file_sha256": _sha_file(tex)},
            }
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "input_evidence_sha256": {kind: manifest["evidence_sha256"] for kind, manifest in manifests.items()},
            "published_outputs": assets,
            "primary_records_mutated": False,
            "primary_denominators_changed": False,
            "repaired_estimates_separate_only": True,
            "promotion_permitted": False,
            "max_semantic_rounds": {
                kind: manifest["max_semantic_rounds"]
                for kind, manifest in manifests.items()
            },
            "table_outer_spacing_pt": 15,
        }
        result["evidence_sha256"] = _self_hash(result)
        (temporary / "analysis_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, output_dir)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, required=True)
    parser.add_argument("--role", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = analyze(part1=args.part1, role=args.role, output_dir=args.output_dir)
    except (SemanticRepairAnalysisError, OSError) as error:
        print(f"Semantic-repair analysis failed: {error}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
