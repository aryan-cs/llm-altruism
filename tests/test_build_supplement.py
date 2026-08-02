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
        if build_supplement.resolve_include_path(
            build_supplement.PROJECT_ROOT, path
        ) is None
    ]

    assert missing == []
    policy_path = (
        build_supplement.PROJECT_ROOT / build_supplement.ANONYMIZATION_POLICY_NAME
    )
    if (build_supplement.PROJECT_ROOT / ".git").exists():
        assert policy_path.is_file()


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    source = root / "experiments" / "gateway_probe.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "InferenceHub https://employer.internal/v1 "
        "EMPLOYER_API_KEY private-handle /Users/private-login\n",
        encoding="utf-8",
    )
    (root / build_supplement.ANONYMIZATION_POLICY_NAME).write_text(
        json.dumps(
            {
                "replacements": [
                    {
                        "source": "https://employer.internal/v1",
                        "replacement": "https://inference-gateway.example.invalid/v1",
                    },
                    {
                        "source": "EMPLOYER_API_KEY",
                        "replacement": "INFERENCE_HUB_API_KEY",
                    },
                ]
            }
        ),
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
            "experiments/gateway_probe.py",
        ]
        payload = zf.read(names[1]).decode("utf-8")
    assert "InferenceHub" in payload
    assert "https://employer.internal/v1" not in payload
    assert "EMPLOYER_API_KEY" not in payload
    assert "https://inference-gateway.example.invalid/v1" in payload
    assert "INFERENCE_HUB_API_KEY" in payload
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


def test_supplement_includes_exact_hosted_reproducibility_surface_only() -> None:
    files = build_supplement.collect_supplement_files()
    names = {path.as_posix() for path in files}
    expected = {
        path.as_posix() for path in build_supplement.HOSTED_REPRODUCIBILITY_ALLOWLIST
    }

    assert expected <= names
    assert "agents/agent_config.registry.json" in names
    assert "agents/local_control.registry.json" in names
    assert not any("local_hf" in name for name in names)
    assert "analysis/build_sota_probe_registry.py" not in names
    assert "analysis/build_sota_inference_hub_roster.py" not in names
    assert "analysis/merge_sota_compatibility_with_judge.py" not in names
    assert "analysis/analyze_inference_hub_part1_panel.py" not in names


def test_strict_denylist_excludes_credentials_private_and_interrupted_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    fixtures = {
        ".env": "NVIDIA_API_KEY=nvapi-real-secret\n",
        ".env.secret": "NVIDIA_API_KEY=nvapi-real-secret\n",
        ".env.example": "NVIDIA_API_KEY=your-key-here\n",
        "credentials.key": "secret\n",
        "nvidia_api_key.txt": "nvapi-real-secret\n",
        "safe.py": "print('safe')\n",
        "data/private/inference_hub/run/private/raw_responses/model.jsonl": "{}\n",
        "data/raw/part_2/interrupted-results.csv": "private\n",
        "data/raw/part_2/pending-results.csv": "private\n",
        "data/raw/part_2/legacy_structural_provenance.json": "{}\n",
        "data/raw/part_2/legacy_execution_archive/manifest.json": "{}\n",
    }
    for relative, content in fixtures.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(build_supplement, "INCLUDE_PATHS", (Path("."),))

    names = {
        path.as_posix()
        for path in build_supplement.collect_supplement_files(
            project_root=root, output_path=tmp_path / "supplement.zip"
        )
    }

    assert names == {".env.example", "safe.py"}


def _write_sealed_json(path: Path, payload: dict[str, object]) -> bytes:
    payload["evidence_sha256"] = build_supplement._json_self_hash(payload)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


def test_only_complete_hash_bound_sanitized_aggregates_are_remapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    run = root / "data/private/inference_hub/part2-complete"
    sanitized = run / "sanitized"
    payloads = {
        "trajectory_metrics.json": {
            "schema_version": 1,
            "artifact_type": "inference_hub_part2_sanitized_trajectory_metrics",
            "rows": [{"target_id": "subject.alpha", "restraint_rate": 0.5}],
        },
        "model_metrics.json": {
            "schema_version": 1,
            "artifact_type": "inference_hub_part2_sanitized_model_metrics",
            "rows": [{"target_id": "subject.alpha", "mean_restraint_rate": 0.5}],
        },
    }
    bindings: dict[str, object] = {}
    for filename, payload in payloads.items():
        path = sanitized / filename
        encoded = _write_sealed_json(path, payload)
        bindings[filename.removesuffix(".json")] = {
            "path": str(path.resolve()),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
            "evidence_sha256": payload["evidence_sha256"],
        }
    manifest = {
        "schema_version": 1,
        "artifact_type": "inference_hub_part2_private_manifest",
        "complete": True,
        "sanitized_artifacts": {
            "trajectory_metrics": bindings["trajectory_metrics"],
            "model_metrics": bindings["model_metrics"],
        },
    }
    _write_sealed_json(run / "private/manifest.json", manifest)
    monkeypatch.setattr(build_supplement, "INCLUDE_PATHS", ())

    output, files = build_supplement.build_supplement(
        project_root=root, output_path=tmp_path / "supplement.zip"
    )

    assert {path.name for path in files} == set(payloads)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert not any(name.startswith("data/private/") for name in names)
    assert {
        "data/analysis/inference_hub_sanitized/part2-complete/model_metrics.json",
        "data/analysis/inference_hub_sanitized/part2-complete/trajectory_metrics.json",
    } <= names


def test_incomplete_or_sensitive_sanitized_artifacts_are_not_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    run = root / "data/private/inference_hub/part2-incomplete"
    sanitized = run / "sanitized"
    payloads = {
        "trajectory_metrics.json": {
            "artifact_type": "inference_hub_part2_sanitized_trajectory_metrics",
            "rows": [],
        },
        "model_metrics.json": {
            "artifact_type": "inference_hub_part2_sanitized_model_metrics",
            "rows": [{"raw_response": "must never ship"}],
        },
    }
    bindings: dict[str, object] = {}
    for filename, payload in payloads.items():
        path = sanitized / filename
        encoded = _write_sealed_json(path, payload)
        bindings[filename.removesuffix(".json")] = {
            "path": str(path.resolve()),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
            "evidence_sha256": payload["evidence_sha256"],
        }
    manifest = {
        "complete": False,
        "sanitized_artifacts": {
            "trajectory_metrics": bindings["trajectory_metrics"],
            "model_metrics": bindings["model_metrics"],
        },
    }
    _write_sealed_json(run / "private/manifest.json", manifest)
    monkeypatch.setattr(build_supplement, "INCLUDE_PATHS", ())

    assert build_supplement.collect_supplement_files(
        project_root=root, output_path=tmp_path / "supplement.zip"
    ) == []


def test_deprecated_legacy_evidence_is_excluded() -> None:
    names = {
        path.as_posix() for path in build_supplement.collect_supplement_files()
    }
    assert "data/raw/part_2/legacy_structural_provenance.json" not in names
    assert not any(
        name.startswith("data/raw/part_2/legacy_execution_archive/")
        for name in names
    )


def test_supplement_builder_does_not_embed_local_identity_literals() -> None:
    source = Path(build_supplement.__file__).read_text(encoding="utf-8").lower()
    for marker, _ in build_supplement._identity_replacements(
        build_supplement.PROJECT_ROOT
    ):
        assert marker.lower() not in source


def test_anonymity_audit_rejects_private_gateway_affiliation_marker(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / build_supplement.ANONYMIZATION_POLICY_NAME).write_text(
        json.dumps(
            {
                "replacements": [
                    {
                        "source": "EMPLOYER_API_KEY",
                        "replacement": "INFERENCE_HUB_API_KEY",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "leaky.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("README.md", "Use EMPLOYER_API_KEY for the private route.")

    findings = build_supplement.audit_anonymous_archive(output, project_root)

    assert findings
    assert "employer_api_key" in findings[0]


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
