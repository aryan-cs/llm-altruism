import json
import shutil
from pathlib import Path

from analysis.build_legacy_part2_provenance import build_provenance
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
