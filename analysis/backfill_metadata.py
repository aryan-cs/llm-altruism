from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.misc.run_metadata import STATUS_COMPLETE, base_run_metadata, write_metadata
from experiments.part0 import part_0
from experiments.part1 import part_1
from experiments.part2 import part_2

RAW_DIR = Path("data") / "raw"


def _metadata_path_for_csv(path: Path) -> Path:
    return path.with_name(f"{path.stem}_meta.json")


def _row_count(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _split_filename(path: Path) -> list[str]:
    return path.stem.split("__")


def _part0_metadata(path: Path) -> dict[str, object]:
    legacy_metadata_path = _metadata_path_for_csv(path)
    legacy: dict[str, object] = {}
    if legacy_metadata_path.exists():
        try:
            legacy = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            legacy = {}
    models = legacy.get("models", {})
    prompts = legacy.get("prompts", [])
    languages = legacy.get("languages", [])
    parameters = {
        "models": models if isinstance(models, dict) else {},
        "prompts": prompts if isinstance(prompts, list) else [],
        "languages": languages if isinstance(languages, list) else [],
    }
    metadata = base_run_metadata(
        experiment="part_0",
        timestamp=path.stem,
        csv_path=path,
        provider="unknown",
        model="unknown",
        parameters=parameters,
        prompt_config_hash=None,
        status=STATUS_COMPLETE,
    )
    if "judge_after" in legacy:
        metadata["judge_after"] = bool(legacy.get("judge_after"))
    return metadata


def _part1_metadata(path: Path) -> dict[str, object]:
    parts = _split_filename(path)
    provider = parts[1] if len(parts) > 1 else "unknown"
    model = parts[2] if len(parts) > 2 else "unknown"
    scope = parts[3] if len(parts) > 3 else "unknown"
    timestamp = parts[4] if len(parts) > 4 else path.stem
    parameters = {
        "scope": scope,
        "games": list(part_1.PART_1_PROMPTS["defaults"]["games"]) if scope == "full" else [],
        "frames": list(part_1.PART_1_PROMPTS["defaults"]["frames"]) if scope == "full" else [],
        "domains": list(part_1.PART_1_PROMPTS["defaults"]["domains"]) if scope == "full" else [],
        "presentations": list(part_1.PART_1_PROMPTS["defaults"]["presentations"]) if scope == "full" else [],
        "limit": None,
        "total_prompts": _row_count(path),
    }
    metadata = base_run_metadata(
        experiment="part_1",
        timestamp=timestamp,
        csv_path=path,
        provider=provider,
        model=model,
        parameters=parameters,
        status=STATUS_COMPLETE,
    )
    metadata.update(parameters)
    return metadata


def _part2_metadata(path: Path) -> dict[str, object]:
    parts = _split_filename(path)
    provider = parts[1] if len(parts) > 1 else "unknown"
    model = parts[2] if len(parts) > 2 else "unknown"
    society_size = int(parts[3].removeprefix("n")) if len(parts) > 3 and parts[3].startswith("n") else None
    raw_days = parts[4] if len(parts) > 4 else ""
    days = 0 if raw_days == "open" else int(raw_days.removeprefix("d")) if raw_days.startswith("d") else None
    resource = parts[5] if len(parts) > 5 else "unknown"
    timestamp = parts[6] if len(parts) > 6 else path.stem
    config = {
        "society_size": society_size,
        "days": days,
        "resource": resource,
        "selfish_gain": None,
        "depletion_units": None,
        "community_benefit": None,
    }
    metadata = base_run_metadata(
        experiment="part_2",
        timestamp=timestamp,
        csv_path=path,
        provider=provider,
        model=model,
        parameters={"society_config": config},
        prompt_config_hash=part_2.PROMPT_CONFIG_HASH,
        status=STATUS_COMPLETE,
    )
    metadata.update(
        {
            "society_config": config,
            "resource_capacity": None,
            "completed_rows": _row_count(path),
        }
    )
    return metadata


def backfill_metadata(raw_dir: Path = RAW_DIR, *, overwrite: bool = False) -> list[Path]:
    written: list[Path] = []
    builders = {
        "part_0": _part0_metadata,
        "part_1": _part1_metadata,
        "part_2": _part2_metadata,
    }
    for part, builder in builders.items():
        for csv_path in sorted((raw_dir / part).glob("*.csv")):
            if csv_path.name.endswith("_pending.csv"):
                continue
            metadata_path = _metadata_path_for_csv(csv_path)
            if metadata_path.exists() and not overwrite:
                if part != "part_0":
                    continue
                try:
                    current = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    current = {}
                if current.get("status"):
                    continue
            metadata = builder(csv_path)
            if part == "part_0":
                parameters = metadata.get("parameters", {})
                if isinstance(parameters, dict):
                    metadata.update(parameters)
            metadata["backfilled"] = True
            metadata["backfill_note"] = (
                "Created from filename and available CSV row count. "
                "Provider runtime settings not captured by the original run remain unknown."
            )
            write_metadata(metadata_path, metadata)
            written.append(metadata_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill metadata sidecars for legacy raw CSVs.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    written = backfill_metadata(Path(args.raw_dir), overwrite=args.overwrite)
    print(f"Wrote {len(written)} metadata file(s).")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
