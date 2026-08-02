from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pytest

from analysis.build_croissant_metadata import (
    CORE_SPEC,
    CROISSANT_CONTEXT,
    DEFAULT_OUTPUT,
    RAI_SPEC,
    RELEASE_FILES,
    build_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _checked_metadata() -> dict[str, object]:
    return json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))


def test_checked_metadata_is_current_generator_output() -> None:
    assert _checked_metadata() == build_metadata()


def test_context_versions_dates_and_anonymous_url_policy() -> None:
    metadata = _checked_metadata()
    assert metadata["@context"] == CROISSANT_CONTEXT
    assert metadata["@type"] == "sc:Dataset"
    assert metadata["conformsTo"] == [CORE_SPEC, RAI_SPEC]
    assert SEMVER.fullmatch(metadata["version"])
    assert SEMVER.fullmatch(metadata["cr:sdVersion"])

    created = date.fromisoformat(metadata["dateCreated"])
    published = date.fromisoformat(metadata["datePublished"])
    modified = date.fromisoformat(metadata["dateModified"])
    assert created <= published <= modified

    # Do not leak the author-identifying repository or invent a placeholder review URL.
    if "url" in metadata:
        assert metadata["url"].startswith("https://")
        assert "example.com" not in metadata["url"]
        assert "aryan-cs" not in metadata["url"]


def test_distributions_resolve_and_match_released_bytes() -> None:
    metadata = _checked_metadata()
    distributions = metadata["distribution"]
    assert len(distributions) == len(RELEASE_FILES)
    assert len({item["@id"] for item in distributions}) == len(distributions)

    for item in distributions:
        assert item["@type"] == "cr:FileObject"
        relative = Path(item["contentUrl"])
        assert not relative.is_absolute()
        assert not relative.as_posix().startswith("data/analysis/")
        resolved = (DEFAULT_OUTPUT.parent / relative).resolve()
        assert resolved.is_file()
        assert item["name"] == resolved.name
        assert item["contentSize"] == f"{resolved.stat().st_size} B"
        assert item["sha256"] == hashlib.sha256(resolved.read_bytes()).hexdigest()


def test_every_released_csv_has_a_complete_record_set() -> None:
    metadata = _checked_metadata()
    distributions = {item["@id"]: item for item in metadata["distribution"]}
    record_sets = metadata["recordSet"]
    csv_specs = [item for item in RELEASE_FILES if item.records]
    assert len(record_sets) == len(csv_specs)

    for spec, record_set in zip(csv_specs, record_sets, strict=True):
        assert record_set["@type"] == "cr:RecordSet"
        assert record_set["@id"] == f"{spec.object_id}-records"
        with (ROOT / spec.path).open(newline="", encoding="utf-8") as handle:
            columns = next(csv.reader(handle))
        fields = record_set["field"]
        assert [field["name"] for field in fields] == columns
        for field in fields:
            assert field["@type"] == "cr:Field"
            assert field["@id"] == f'{record_set["@id"]}/{field["name"]}'
            assert field["dataType"] in {
                "sc:Boolean",
                "sc:Float",
                "sc:Integer",
                "sc:Text",
            }
            assert field["source"] == {
                "fileObject": {"@id": spec.object_id},
                "extract": {"column": field["name"]},
            }
        assert distributions[spec.object_id]["encodingFormat"] == "text/csv"


def test_dataset_url_rejects_placeholders_and_accepts_review_host(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="public HTTPS"):
        build_metadata(dataset_url="./")
    with pytest.raises(ValueError, match="public HTTPS"):
        build_metadata(dataset_url="https://example.com/dataset")
    metadata = build_metadata(dataset_url="https://openreview.net/")
    assert metadata["url"] == "https://openreview.net/"

    relocated = build_metadata(output_path=tmp_path / "metadata.json")
    for distribution in relocated["distribution"]:
        assert (tmp_path / distribution["contentUrl"]).resolve().is_file()


def test_mlcroissant_loads_all_record_sets_and_reads_real_rows() -> None:
    mlcroissant = pytest.importorskip("mlcroissant")
    dataset = mlcroissant.Dataset(str(DEFAULT_OUTPUT))
    expected = [
        f"{release_file.object_id}-records"
        for release_file in RELEASE_FILES
        if release_file.records
    ]
    assert [record_set.uuid for record_set in dataset.metadata.record_sets] == expected
    assert len(dataset.metadata.file_objects) == len(RELEASE_FILES)

    for record_set_id in expected:
        record = next(iter(dataset.records(record_set=record_set_id)))
        declared_fields = next(
            item["field"]
            for item in _checked_metadata()["recordSet"]
            if item["@id"] == record_set_id
        )
        assert set(record) == {field["@id"] for field in declared_fields}
