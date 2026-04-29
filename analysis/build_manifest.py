from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.misc.run_metadata import read_metadata

RAW_DIR = Path("data") / "raw"
ANALYSIS_DIR = Path("data") / "analysis"
DEFAULT_OUTPUT = ANALYSIS_DIR / "run_manifest.jsonl"


def _metadata_path_for_csv(path: Path) -> Path:
    return path.with_name(f"{path.stem}_meta.json")


def _load_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return read_metadata(path)
    except Exception as error:
        return {"metadata_error": f"{type(error).__name__}: {error}"}


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
