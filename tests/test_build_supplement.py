import zipfile
from pathlib import Path

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


def test_anonymous_supplement_rewrites_private_gateway_and_author_markers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    source = root / "experiments" / "inference_hub_probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "InferenceHub https://inference-api.nvidia.com/v1 "
        "NVIDIA_API_KEY aryan-cs /Users/aryagupta\n",
        encoding="utf-8",
    )
    output = tmp_path / "anonymous.zip"

    original_paths = build_supplement.INCLUDE_PATHS
    try:
        build_supplement.INCLUDE_PATHS = (Path("experiments"),)
        build_supplement.build_supplement(root, output)
    finally:
        build_supplement.INCLUDE_PATHS = original_paths

    assert build_supplement.audit_anonymous_archive(output) == []
    with zipfile.ZipFile(output) as zf:
        names = zf.namelist()
        assert names == [
            build_supplement.MANIFEST_NAME,
            "experiments/hosted_gateway_probe.py",
        ]
        payload = zf.read(names[1]).decode("utf-8")
    assert "HostedGateway" in payload
    assert "YOUR_HOSTED_GATEWAY.example" in payload
    assert "HOSTED_GATEWAY_API_KEY" in payload
    assert "anonymous-author" in payload
    assert "/home/anonymous" in payload
