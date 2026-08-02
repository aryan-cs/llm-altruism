import zipfile
import hashlib
import json
from pathlib import Path

import pytest

from analysis import build_supplement


def test_anonymous_supplement_excludes_author_identity_metadata(tmp_path: Path) -> None:
    files = build_supplement.collect_supplement_files(
        project_root=build_supplement.PROJECT_ROOT,
        output_path=tmp_path / "supplement.zip",
    )

    assert Path("docs/release/research-proposal-metadata.json") not in files


def test_supplement_has_no_declared_missing_include_roots() -> None:
    missing = [
        path
        for path in build_supplement.INCLUDE_PATHS
        if not (build_supplement.PROJECT_ROOT / path).exists()
    ]

    assert missing == []


def test_policy_manifest_discloses_identity_exclusion() -> None:
    exclusions = {
        item["path"]: item["reason"]
        for item in build_supplement.POLICY_EXCLUSIONS
    }

    assert "docs/release/research-proposal-metadata.json" in exclusions
    assert "author-identifying" in exclusions[
        "docs/release/research-proposal-metadata.json"
    ]


def test_anonymous_supplement_preserves_api_contract_and_rewrites_author_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    source = root / "experiments" / "inference_hub_probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "InferenceHub https://inference-api.nvidia.com/v1 "
        "NVIDIA_API_KEY private-handle /Users/private-login\n",
        encoding="utf-8",
    )
    output = tmp_path / "anonymous.zip"

    original_paths = build_supplement.INCLUDE_PATHS
    try:
        build_supplement.INCLUDE_PATHS = (Path("experiments"),)
        monkeypatch.setattr(
            build_supplement,
            "_identity_replacements",
            lambda project_root: (("private-handle", "anonymous-author"),),
        )
        build_supplement.build_supplement(root, output)
    finally:
        build_supplement.INCLUDE_PATHS = original_paths

    assert build_supplement.audit_anonymous_archive(output) == []
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        assert names == [
            build_supplement.MANIFEST_NAME,
            "experiments/inference_hub_probe.py",
        ]
        payload = zf.read(names[1]).decode("utf-8")
    assert "InferenceHub" in payload
    assert "https://inference-api.nvidia.com/v1" in payload
    assert "NVIDIA_API_KEY" in payload
    assert "anonymous-author" in payload
    assert "/home/anonymous" in payload


def test_supplement_excludes_every_withdrawn_part0_dependent_plot() -> None:
    files = build_supplement.collect_supplement_files()
    names = {path.as_posix() for path in files}
    assert not any(name.startswith("data/graphs/part_0/") for name in names)
    assert not any(name.startswith("data/graphs/cross_part/") for name in names)
    assert not any(
        name.endswith(
            (
                "part0_refusal_rate_by_model.png",
                "part0_refusal_by_language_heatmap.png",
                "behavioral_fingerprint_heatmap.png",
                "model_behavior_pca.png",
            )
        )
        for name in names
    )


def test_supplement_builder_does_not_embed_local_identity_literals() -> None:
    source = Path(build_supplement.__file__).read_text(encoding="utf-8").lower()
    for marker, _ in build_supplement._identity_replacements(
        build_supplement.PROJECT_ROOT
    ):
        assert marker.lower() not in source


def test_supplement_manifest_hashes_every_included_payload(tmp_path: Path) -> None:
    output, _ = build_supplement.build_supplement(
        build_supplement.PROJECT_ROOT, tmp_path / "supplement.zip"
    )
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(
            archive.read(build_supplement.MANIFEST_NAME).decode("utf-8")
        )
        names = set(archive.namelist()) - {build_supplement.MANIFEST_NAME}
        assert set(manifest["file_sha256s"]) == names == set(manifest["files"])
        for name in names:
            assert hashlib.sha256(archive.read(name)).hexdigest() == manifest[
                "file_sha256s"
            ][name]
