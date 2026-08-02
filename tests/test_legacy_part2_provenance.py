import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from analysis.build_legacy_part2_provenance import (
    ARCHIVE_DIRNAME,
    ARCHIVE_MANIFEST_FILENAME,
    ARCHIVE_PROMPT_FILENAME,
    ARCHIVE_SOURCE_FILENAME,
    PROMPT_CONFIG_SHA256,
    PROMPT_SHA256,
    SOURCE_SHA256,
    build_provenance,
    materialize_execution_archive,
)
from analysis.build_supplement import build_supplement
from analysis.validation import validate_part2_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "part_2"
PROVENANCE_PATH = RAW_DIR / "legacy_structural_provenance.json"
EXAMPLE_CSV = (
    RAW_DIR
    / "part2__ollama__gpt-oss-20b__n50__d100__water__20260429_051434.csv"
)


def test_checked_in_legacy_part2_provenance_is_exactly_reproducible() -> None:
    expected = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))

    assert build_provenance(raw_dir=RAW_DIR, project_root=PROJECT_ROOT) == expected
    assert len(expected["entries"]) == 13
    assert {
        entry["collapse_death_rate"] for entry in expected["entries"]
    } == {0.2}


def test_checked_in_execution_archive_hash_binds_source_and_prompt() -> None:
    archive_dir = RAW_DIR / ARCHIVE_DIRNAME
    manifest = json.loads(
        (archive_dir / ARCHIVE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    assets = {item["archive_filename"]: item for item in manifest["assets"]}
    assert {filename: item["sha256"] for filename, item in assets.items()} == {
        ARCHIVE_SOURCE_FILENAME: SOURCE_SHA256,
        ARCHIVE_PROMPT_FILENAME: PROMPT_SHA256,
    }
    assert assets[ARCHIVE_PROMPT_FILENAME]["canonical_json_sha256"] == (
        PROMPT_CONFIG_SHA256
    )
    assert build_provenance(raw_dir=RAW_DIR) == json.loads(
        PROVENANCE_PATH.read_text(encoding="utf-8")
    )


def test_execution_archive_materializes_exact_historical_bytes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "part_2"
    archive_dir = materialize_execution_archive(
        raw_dir=raw_dir,
        project_root=PROJECT_ROOT,
    )

    assert (archive_dir / ARCHIVE_SOURCE_FILENAME).read_bytes() == (
        RAW_DIR / ARCHIVE_DIRNAME / ARCHIVE_SOURCE_FILENAME
    ).read_bytes()
    assert (archive_dir / ARCHIVE_PROMPT_FILENAME).read_bytes() == (
        RAW_DIR / ARCHIVE_DIRNAME / ARCHIVE_PROMPT_FILENAME
    ).read_bytes()
    assert (archive_dir / ARCHIVE_MANIFEST_FILENAME).read_text(encoding="utf-8") == (
        RAW_DIR / ARCHIVE_DIRNAME / ARCHIVE_MANIFEST_FILENAME
    ).read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        materialize_execution_archive(raw_dir=raw_dir, project_root=PROJECT_ROOT)


@pytest.mark.parametrize(
    "filename, expected_message",
    [
        (ARCHIVE_SOURCE_FILENAME, "source hash disagrees"),
        (ARCHIVE_PROMPT_FILENAME, "prompt hash disagrees"),
    ],
)
def test_legacy_part2_provenance_rejects_tampered_portable_archive(
    tmp_path: Path,
    filename: str,
    expected_message: str,
) -> None:
    raw_dir = tmp_path / "part_2"
    shutil.copytree(RAW_DIR, raw_dir)
    archive_path = raw_dir / ARCHIVE_DIRNAME / filename
    archive_path.write_bytes(archive_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=expected_message):
        build_provenance(raw_dir=raw_dir)


def test_provenance_check_succeeds_in_extracted_supplement_without_git(
    tmp_path: Path,
) -> None:
    supplement_path = tmp_path / "supplement.zip"
    build_supplement(PROJECT_ROOT, supplement_path)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(supplement_path) as archive:
        archive.extractall(extracted)

    assert not (extracted / ".git").exists()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.build_legacy_part2_provenance",
            "--check",
        ],
        cwd=extracted,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Verified data/raw/part_2/legacy_structural_provenance.json" in completed.stdout


def test_checked_in_legacy_part2_trajectory_replays_under_sealed_rate() -> None:
    report = validate_part2_file(EXAMPLE_CSV)

    assert report.status == "warn"
    assert report.errors == []
    assert report.metrics["structural_metadata_valid"] is True
    assert report.metrics["recorded_collapse_death_rate"] == 0.2
    assert report.metrics["transition_errors"] == 0


def _copy_example_bundle(tmp_path: Path) -> Path:
    csv_path = tmp_path / EXAMPLE_CSV.name
    metadata_path = EXAMPLE_CSV.with_name(f"{EXAMPLE_CSV.stem}_meta.json")
    shutil.copyfile(EXAMPLE_CSV, csv_path)
    shutil.copyfile(metadata_path, tmp_path / metadata_path.name)
    shutil.copyfile(PROVENANCE_PATH, tmp_path / PROVENANCE_PATH.name)
    return csv_path


def test_legacy_part2_provenance_rejects_tampered_csv_bytes(tmp_path: Path) -> None:
    csv_path = _copy_example_bundle(tmp_path)
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")

    report = validate_part2_file(csv_path)

    assert report.status == "fail"
    assert any("provenance bytes" in error for error in report.errors)


def test_legacy_part2_provenance_rejects_rehashed_source_substitution(
    tmp_path: Path,
) -> None:
    csv_path = _copy_example_bundle(tmp_path)
    provenance_path = tmp_path / PROVENANCE_PATH.name
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["source_contract"]["sha256"] = "0" * 64
    payload = {
        key: value for key, value in provenance.items() if key != "artifact_sha256"
    }
    from analysis.part2_dynamics import stable_json_sha256

    provenance["artifact_sha256"] = stable_json_sha256(payload)
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    report = validate_part2_file(csv_path)

    assert report.status == "fail"
    assert any("archived rule" in error for error in report.errors)
