from __future__ import annotations

import csv
from pathlib import Path
import json

import pytest

from analysis import backfill_metadata


def _write_part1(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["provider", "model"])
        writer.writeheader()
        for provider, model in rows:
            writer.writerow({"provider": provider, "model": model})


def test_part1_backfill_uses_exact_csv_identity_not_filename_slug(tmp_path: Path) -> None:
    path = tmp_path / "part1__ollama__vendor-model-20b__full__20260424_000000.csv"
    _write_part1(path, [("ollama", "vendor/model:20b")])

    metadata = backfill_metadata._part1_metadata(path)

    assert metadata["provider"] == "ollama"
    assert metadata["model"] == "vendor/model:20b"


def test_part1_backfill_rejects_ambiguous_csv_identity(tmp_path: Path) -> None:
    path = tmp_path / "part1__ollama__model__full__20260424_000000.csv"
    _write_part1(path, [("ollama", "model:a"), ("ollama", "model:b")])

    with pytest.raises(ValueError, match="Ambiguous provider/model identity"):
        backfill_metadata._part1_metadata(path)


def test_identity_repair_preserves_legacy_provenance(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = raw / "part_1" / "part1__ollama__vendor-model__full__20260424_000000.csv"
    _write_part1(path, [("ollama", "vendor/model:20b")])
    sidecar = path.with_name(f"{path.stem}_meta.json")
    original = {
        "schema_version": 1,
        "created_at_utc": "2026-04-29T00:00:00+00:00",
        "git_commit": "abc123",
        "provider": "ollama",
        "model": "vendor-model-20b",
    }
    sidecar.write_text(json.dumps(original), encoding="utf-8")

    assert backfill_metadata.repair_part1_identities(raw) == [sidecar]
    repaired = json.loads(sidecar.read_text(encoding="utf-8"))

    assert repaired["model"] == "vendor/model:20b"
    assert repaired["created_at_utc"] == original["created_at_utc"]
    assert repaired["git_commit"] == original["git_commit"]
    assert repaired["identity_repair"]["method"] == "unique_csv_provider_model"
    assert len(repaired["identity_repair"]["csv_sha256"]) == 64
