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
    assert "CHECKPOINT.md" in names
    assert "agents/agent_config.registry.json" in names
    assert "agents/local_control.registry.json" in names
    assert {
        name for name in names if "local_hf" in name
    } == {"data/analysis/local_hf_part1_controls.json"}
    assert "analysis/build_sota_probe_registry.py" not in names
    assert "analysis/build_sota_inference_hub_roster.py" not in names
    assert "analysis/analyze_availability_retry_panels.py" in names
    assert "tests/test_analyze_availability_retry_panels.py" in names
    assert "docs/AVAILABILITY_RETRY_ANALYSIS.md" in names
    assert "analysis/merge_sota_compatibility_with_judge.py" in names
    assert "analysis/analyze_inference_hub_part1_panel.py" in names
    assert "analysis/finalize_inference_hub_part2_offline.py" in names
    assert "experiments/misc/inference_hub_retire_target.py" in names
    assert "tests/test_finalize_inference_hub_part2_offline.py" in names
    assert "tests/test_inference_hub_retire_target.py" in names
    assert {
        "analysis/accelerated_part0_human_validation.py",
        "analysis/analyze_availability_retry_panels.py",
        "analysis/analyze_provider_safe_v2_definitive.py",
        "analysis/build_provider_safe_v2_paper_assets.py",
        "analysis/part2_confirmatory.py",
        "docs/ACCELERATED_PART0_HUMAN_VALIDATION.md",
        "docs/AVAILABILITY_RETRY_ANALYSIS.md",
        "docs/PART1_ROLE_CALIBRATION_V1.md",
        "docs/PART1_SEMANTIC_INVALID_REPAIR.md",
        "docs/PART1_ROLE_SEMANTIC_INVALID_REPAIR.md",
        "docs/PART2_SENSITIVITY_V1.md",
        "docs/PROVIDER_SAFE_V2_DEFINITIVE_ANALYSIS.md",
        "docs/PROVIDER_SAFE_V2_PAPER_ASSETS.md",
        "docs/LOCAL_MODEL_CONTROLS.md",
        "experiments/misc/inference_hub_compatibility_provider_safe.py",
        "experiments/misc/inference_hub_exploratory_accelerated.py",
        "experiments/misc/inference_hub_main_accelerated.py",
        "experiments/misc/inference_hub_part0_deadline_retry.py",
        "experiments/misc/inference_hub_part1_deadline_accelerated.py",
        "experiments/misc/inference_hub_part1_role_calibration_v1.py",
        "experiments/misc/inference_hub_part1_semantic_invalid_repair.py",
        "experiments/misc/inference_hub_part1_role_semantic_invalid_repair.py",
        "experiments/misc/inference_hub_part2_sensitivity_v1.py",
        "experiments/misc/inference_hub_provider_safe.py",
        "experiments/misc/inference_hub_provider_safe_v2.py",
        "experiments/misc/inference_hub_sensitivity_deadline_accelerated.py",
        "experiments/misc/inference_hub_visible_compatibility.py",
        "experiments/part1/role_calibration_panel_v1.json",
        "experiments/part2/part2_sensitivity_v1.json",
        "experiments/part2/part2_sensitivity_deadline_exploratory_v1.json",
        "tests/test_accelerated_part0_human_validation.py",
        "tests/test_analyze_availability_retry_panels.py",
        "tests/test_analyze_provider_safe_v2_definitive.py",
        "tests/test_build_provider_safe_v2_paper_assets.py",
        "tests/test_inference_hub_compatibility_provider_safe.py",
        "tests/test_inference_hub_exploratory_accelerated.py",
        "tests/test_inference_hub_main_accelerated.py",
        "tests/test_inference_hub_part0_deadline_retry.py",
        "tests/test_inference_hub_part1_deadline_accelerated.py",
        "tests/test_inference_hub_part1_role_calibration_v1.py",
        "tests/test_inference_hub_part1_semantic_invalid_repair.py",
        "tests/test_inference_hub_part1_role_semantic_invalid_repair.py",
        "tests/test_inference_hub_part2_sensitivity_v1.py",
        "tests/test_inference_hub_provider_safe.py",
        "tests/test_inference_hub_provider_safe_v2.py",
        "tests/test_inference_hub_sensitivity_deadline_accelerated.py",
        "tests/test_inference_hub_visible_compatibility.py",
        "tests/test_merge_sota_compatibility_with_judge.py",
        "tests/test_part2_confirmatory_cli.py",
        "tests/test_part2_confirmatory_statistics.py",
    } <= names

    # b94e65a adds deterministic scalar manuscript macros to the already
    # exact-allowlisted paper-assets generator and sealed final-result surface.
    assert "data/analysis/final_results/paper_headlines.tex" in names
    assert "data/analysis/final_results/paper_macros.tex" in names


def test_unreviewed_availability_retry_source_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "analyze_availability_retry_panels_v2.py").write_text(
        "# unreviewed hosted retry analyzer\n", encoding="utf-8"
    )
    (analysis_dir / "ordinary_release_helper.py").write_text(
        "# ordinary packaged helper\n", encoding="utf-8"
    )
    monkeypatch.setattr(build_supplement, "INCLUDE_PATHS", (Path("analysis"),))

    names = {
        path.as_posix()
        for path in build_supplement.collect_supplement_files(
            project_root=root, output_path=tmp_path / "supplement.zip"
        )
    }

    assert "analysis/analyze_availability_retry_panels_v2.py" not in names
    assert "analysis/ordinary_release_helper.py" in names


def test_supplement_release_boundary_is_aggregate_only_and_current() -> None:
    names = {
        path.as_posix() for path in build_supplement.collect_supplement_files()
    }

    assert not any(name.startswith("data/raw/") for name in names)
    assert not any(name.startswith("data/private/") for name in names)
    assert not any(name.startswith("artifacts/availability_retry") for name in names)
    assert not any(name.startswith("data/analysis/tables/") for name in names)
    assert not any(name.startswith("data/analysis/validation/") for name in names)
    assert "data/analysis/part0_rejudge_audit_checkpoint.json" not in names
    assert "tests/test_part0_audit_checkpoint.py" not in names
    assert not any(
        name.startswith(
            (
                "data/graphs/part_0/",
                "data/graphs/part_1/",
                "data/graphs/part_2/",
                "data/graphs/cross_part/",
                "data/graphs/paper_visuals/",
            )
        )
        for name in names
    )
    assert {
        "data/analysis/final_results/final_results.json",
        "data/analysis/final_results/part0_model_rates.csv",
        "data/analysis/final_results/part1_model_rates.csv",
        "data/analysis/final_results/part2_model_metrics.csv",
        "data/analysis/croissant_metadata.json",
        "data/analysis/local_hf_part1_controls.json",
    } <= names
    assert {
        name
        for name in names
        if name.startswith("docs/conference_submission/figures/")
    } == {
        "docs/conference_submission/figures/part0_response_language_conditions.png",
        "docs/conference_submission/figures/part1_scope_distributions.png",
        "docs/conference_submission/figures/part2_corrected_outcomes.png",
    }


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


def test_private_run_local_aggregates_are_never_implicitly_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    run = root / "data/private/inference_hub/part2-complete"
    sanitized = run / "sanitized"
    for filename in ("trajectory_metrics.json", "model_metrics.json"):
        path = sanitized / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"rows": [{"reasoning": "must never ship"}]}\n', encoding="utf-8")
    private_manifest = run / "private/manifest.json"
    private_manifest.parent.mkdir(parents=True, exist_ok=True)
    private_manifest.write_text('{"complete": true}\n', encoding="utf-8")
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


def test_rebuild_preserves_anonymous_model_registry_archive_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "extracted-supplement"
    registry = root / "docs" / "release" / "MODEL_REGISTRY.md"
    registry.parent.mkdir(parents=True)
    registry.write_text("# Anonymous model registry\n", encoding="utf-8")
    monkeypatch.setattr(
        build_supplement,
        "INCLUDE_PATHS",
        (Path("docs/conference_submission/SUPPLEMENT_MODEL_REGISTRY.md"),),
    )

    output, files = build_supplement.build_supplement(
        project_root=root,
        output_path=tmp_path / "rebuilt.zip",
    )

    assert files == [Path("docs/release/MODEL_REGISTRY.md")]
    with zipfile.ZipFile(output) as archive:
        assert "docs/release/MODEL_REGISTRY.md" in archive.namelist()


def test_clean_extraction_rebuild_preserves_payload_manifest(tmp_path: Path) -> None:
    first_zip = tmp_path / "first.zip"
    first_output, _ = build_supplement.build_supplement(
        project_root=build_supplement.PROJECT_ROOT,
        output_path=first_zip,
    )
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(first_output) as archive:
        archive.extractall(extracted)

    second_output, _ = build_supplement.build_supplement(
        project_root=extracted,
        output_path=tmp_path / "second.zip",
    )
    with zipfile.ZipFile(first_output) as first_archive:
        first_manifest = json.loads(
            first_archive.read(build_supplement.MANIFEST_NAME).decode("utf-8")
        )
    with zipfile.ZipFile(second_output) as second_archive:
        second_manifest = json.loads(
            second_archive.read(build_supplement.MANIFEST_NAME).decode("utf-8")
        )

    assert first_manifest == second_manifest
    assert first_output.read_bytes() == second_output.read_bytes()


def test_repeated_build_is_byte_reproducible(tmp_path: Path) -> None:
    first_output, _ = build_supplement.build_supplement(
        project_root=build_supplement.PROJECT_ROOT,
        output_path=tmp_path / "first.zip",
    )
    second_output, _ = build_supplement.build_supplement(
        project_root=build_supplement.PROJECT_ROOT,
        output_path=tmp_path / "second.zip",
    )

    assert first_output.read_bytes() == second_output.read_bytes()
    with zipfile.ZipFile(first_output) as archive:
        manifest = json.loads(
            archive.read(build_supplement.MANIFEST_NAME).decode("utf-8")
        )
    assert manifest["created_utc"] == build_supplement.REPRODUCIBLE_CREATED_UTC


def test_packaged_reproducibility_note_matches_frozen_sensitivity_counts() -> None:
    text = (
        build_supplement.PROJECT_ROOT / "docs/release/REPRODUCIBILITY.md"
    ).read_text(encoding="utf-8")

    assert "17,280-post" in text
    assert "32 trajectories per sentinel" in text
    assert "192 total" in text
    assert "four exact paired" in text
    assert "103,680-post" not in text
