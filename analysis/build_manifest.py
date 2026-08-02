from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.misc.run_metadata import read_metadata

RAW_DIR = Path("data") / "raw"
ANALYSIS_DIR = Path("data") / "analysis"
DEFAULT_OUTPUT = ANALYSIS_DIR / "run_manifest.jsonl"
IDENTIFIER_COLUMNS = (
    "provider",
    "model",
    "requested_model",
    "response_model",
    "provider_response_model",
    "run_id",
    "trajectory_id",
    "structural_cell_id",
    "generation_seed",
    "environment_seed",
)
RUN_CONTRACT_FIELDS = (
    "experiment",
    "parameters",
    "generation_config",
    "prompt_config_hash",
    "protocol_contract",
    "resume_contract",
    "model_registry",
    "registry",
    "code_bundle",
    "source_bundle",
    "judge",
    "extractor",
    "attempt_log",
)


def _metadata_path_for_csv(path: Path) -> Path:
    return path.with_name(f"{path.stem}_meta.json")


def _load_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_metadata(path)
    except Exception as error:
        return {"metadata_error": f"{type(error).__name__}: {error}"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _csv_integrity(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV manifest input has no header: {path}")
        columns = list(reader.fieldnames)
        if len(columns) != len(set(columns)):
            raise ValueError(f"CSV manifest input has duplicate columns: {path}")
        identifier_values: dict[str, set[str]] = {
            column: set() for column in IDENTIFIER_COLUMNS if column in columns
        }
        row_count = 0
        for row in reader:
            row_count += 1
            for column, values in identifier_values.items():
                value = (row.get(column) or "").strip()
                if value:
                    values.add(value)
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "row_count": row_count,
        "columns": columns,
        "schema_sha256": _stable_json_hash(columns),
        "identifier_values": {
            column: sorted(values)
            for column, values in identifier_values.items()
        },
    }


def _metadata_integrity(
    path: Path,
    metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if metadata is None or not path.is_file():
        return None
    return {
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "canonical_sha256": _stable_json_hash(metadata),
    }


def _run_contract(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    if metadata is None:
        return None
    contract = {
        field: metadata[field]
        for field in RUN_CONTRACT_FIELDS
        if field in metadata
    }
    return contract or None


def _part_from_path(path: Path) -> str:
    for part in ("part_0", "part_1", "part_2"):
        if part in path.parts:
            return part
    return "unknown"


def build_manifest(raw_dir: Path = RAW_DIR) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for csv_path in sorted(raw_dir.glob("part_*/*.csv")):
        if csv_path.name.endswith("_pending.csv"):
            continue
        metadata_path = _metadata_path_for_csv(csv_path)
        metadata = _load_metadata(metadata_path)
        csv_integrity = _csv_integrity(csv_path)
        metadata_integrity = _metadata_integrity(metadata_path, metadata)
        run_contract = _run_contract(metadata)
        entries.append(
            {
                "part": _part_from_path(csv_path),
                "csv_path": str(csv_path),
                "metadata_path": str(metadata_path) if metadata_path.exists() else None,
                "metadata_status": None if metadata is None else metadata.get("status", "unknown"),
                "provider": None if metadata is None else metadata.get("provider"),
                "model": None if metadata is None else metadata.get("model"),
                "timestamp": None if metadata is None else metadata.get("timestamp"),
                "git_commit": None if metadata is None else metadata.get("git_commit"),
                "prompt_config_hash": None if metadata is None else metadata.get("prompt_config_hash"),
                "csv_integrity": csv_integrity,
                "metadata_integrity": metadata_integrity,
                "run_contract": run_contract,
                "run_contract_sha256": (
                    None if run_contract is None else _stable_json_hash(run_contract)
                ),
            }
        )
    return entries


def write_manifest(
    *,
    raw_dir: Path = RAW_DIR,
    output_path: Path = DEFAULT_OUTPUT,
) -> Path:
    entries = build_manifest(raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CSV-to-metadata manifest for raw experiment outputs.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_path = write_manifest(raw_dir=Path(args.raw_dir), output_path=Path(args.output))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
