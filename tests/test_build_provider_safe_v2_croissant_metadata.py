from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from analysis.build_provider_safe_v2_croissant_metadata import (
    CORE_SPEC,
    RAI_SPEC,
    DefinitiveCroissantError,
    _self_hash,
    build_metadata,
    main,
    serialized_metadata,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    analysis = root / "provider-safe-v2-definitive-analysis"
    assets = root / "provider-safe-v2-paper-assets"
    analysis.mkdir(parents=True)
    assets.mkdir()
    tables: dict[str, list[dict[str, object]]] = {
        "part0_models": [
            {
                "target_route_id": f"route/p0-{index}",
                "model_id": f"model-p0-{index}",
                "scheduled_units": 144,
            }
            for index in range(2)
        ],
        "part1_models": [
            {
                "target_route_id": f"route/p1-{index}",
                "model_id": f"model-p1-{index}",
                "scheduled_units": 384,
            }
            for index in range(3)
        ],
        "part2_models": [
            {
                "target_route_id": "route/p2-0",
                "model_id": "model-p2-0",
                "trajectory_count": 12,
                "environmentally_estimable_trajectory_count": 11,
                "semantic_invalid_trajectory_count": 1,
                "first_attempt_invalid_count": 2,
            }
        ],
        "role_calibration_model_frames": [
            {"target_route_id": "route/role", "frame": frame}
            for frame in ("neutral", "self", "counterpart")
        ],
        "sensitivity_models": [
            {"sentinel_id": "route/sentinel", "cell_count": 16}
        ],
        "sensitivity_main_effects": [
            {"sentinel_id": "route/sentinel", "factor": f"f{index}"}
            for index in range(5)
        ],
    }
    public_outputs: list[dict[str, object]] = []
    for name, rows in tables.items():
        csv_path = analysis / f"{name}.csv"
        jsonl_path = analysis / f"{name}.jsonl"
        _write_csv(csv_path, rows)
        jsonl_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        for path, kind in (
            (csv_path, "machine_readable_table_csv"),
            (jsonl_path, "machine_readable_table_jsonl"),
        ):
            public_outputs.append(
                {
                    "basename": path.name,
                    "file_sha256": _sha(path),
                    "row_count": len(rows),
                    "kind": kind,
                }
            )
    figure = analysis / "figure_aggregates.json"
    figure.write_text(
        json.dumps({"part0_by_model_language": []}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    public_outputs.append(
        {
            "basename": figure.name,
            "file_sha256": _sha(figure),
            "row_count": 0,
            "kind": "machine_readable_figure_aggregates_json",
        }
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "provider_safe_v2_definitive_descriptive_analysis",
        "input_manifests": {
            phase: {
                "basename": "manifest.json",
                "file_sha256": "a" * 64,
                "evidence_sha256": "b" * 64,
            }
            for phase in ("part0", "part1", "part2", "role", "sensitivity")
        },
        "path_policy": "portable_basenames_only_no_host_absolute_paths_in_public_manifest",
        "privacy_policy": {
            "contains_prompt_text": False,
            "contains_response_text_or_reasoning": False,
            "contains_private_journal_paths": False,
            "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
        },
        "public_outputs": sorted(public_outputs, key=lambda row: str(row["basename"])),
        "public_output_inventory_scope": "all_nonmanifest_outputs_created_before_manifest_self_seal",
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "human_labels_generated": False,
        "exploratory_only": True,
        "confirmatory_or_paper_promotion_permitted": False,
    }
    manifest["evidence_sha256"] = _self_hash(manifest)
    (analysis / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    asset_payloads = {
        "all_models_cross_phase_table.md": b"| model | Part 0 |\n|---|---|\n",
        "all_models_cross_phase_table.tex": b"% table\n",
        "part0_model_language.png": b"\x89PNG\r\n\x1a\nfixture",
        "part0_model_language.pdf": b"%PDF-1.4 fixture\n",
    }
    asset_rows = []
    for name, payload in asset_payloads.items():
        path = assets / name
        path.write_bytes(payload)
        asset_rows.append(
            {
                "name": name,
                "kind": "fixture",
                "file_sha256": _sha(path),
                "byte_count": len(payload),
            }
        )
    asset_manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "provider_safe_v2_paper_assets",
        "source_analysis_evidence_sha256": manifest["evidence_sha256"],
        "source_local_controls_evidence_sha256": "c" * 64,
        "assets": sorted(asset_rows, key=lambda row: str(row["name"])),
        "human_labels_generated": False,
        "exploratory_only": True,
        "confirmatory_or_paper_promotion_permitted": False,
        "cross_axis_aggregate_or_score_generated": False,
    }
    asset_manifest["evidence_sha256"] = _self_hash(asset_manifest)
    (assets / "paper_assets_manifest.json").write_text(
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return analysis, assets, root / "provider-safe-v2-croissant-metadata.json"


def test_metadata_is_hash_bound_portable_and_computed_from_release(tmp_path: Path) -> None:
    analysis, assets, output = _fixture(tmp_path / "release")
    metadata = build_metadata(
        analysis_dir=analysis, paper_assets_dir=assets, output_path=output
    )

    assert metadata["conformsTo"] == [CORE_SPEC, RAI_SPEC]
    assert "Safety Beyond Refusal" in metadata["citeAs"]
    assert metadata["releaseCoverage"] == {
        "part0_model_count": 2,
        "part0_scheduled_response_count": 288,
        "part1_model_count": 3,
        "part1_scheduled_choice_count": 1152,
        "part2_model_count": 1,
        "part2_trajectory_count": 12,
        "part2_environmentally_estimable_trajectory_count": 11,
        "part2_semantic_invalid_trajectory_count": 1,
        "part2_first_attempt_invalid_action_count": 2,
        "role_calibration_model_frame_count": 3,
        "sensitivity_model_count": 1,
        "sensitivity_main_effect_count": 5,
    }
    distributions = metadata["distribution"]
    assert not any(item["name"].endswith(".pdf") for item in distributions)
    assert any(item["name"].endswith(".png") for item in distributions)
    assert any(item["name"].endswith(".tex") for item in distributions)
    assert any(item["name"].endswith(".md") for item in distributions)
    for item in distributions:
        assert not Path(item["contentUrl"]).is_absolute()
        resolved = output.parent / item["contentUrl"]
        assert resolved.is_file()
        assert item["sha256"] == _sha(resolved)
    assert len(metadata["recordSet"]) == 6
    serialized = serialized_metadata(metadata)
    assert "/Users/" not in serialized
    assert "/private/" not in serialized
    assert "prompt_text" not in serialized


def test_hash_tamper_extra_file_and_private_field_fail_closed(tmp_path: Path) -> None:
    analysis, assets, output = _fixture(tmp_path / "hash")
    (analysis / "part0_models.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(DefinitiveCroissantError, match="hash changed"):
        build_metadata(
            analysis_dir=analysis, paper_assets_dir=assets, output_path=output
        )

    analysis, assets, output = _fixture(tmp_path / "extra")
    (assets / "unbound.png").write_bytes(b"unbound")
    with pytest.raises(DefinitiveCroissantError, match="complete asset inventory"):
        build_metadata(
            analysis_dir=analysis, paper_assets_dir=assets, output_path=output
        )

    analysis, assets, output = _fixture(tmp_path / "private")
    path = analysis / "part0_models.jsonl"
    path.write_text(
        json.dumps({"prompt_text": "must not ship"}) + "\n", encoding="utf-8"
    )
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = next(
        row for row in manifest["public_outputs"] if row["basename"] == path.name
    )
    binding["file_sha256"] = _sha(path)
    manifest["evidence_sha256"] = _self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DefinitiveCroissantError, match="forbidden private field"):
        build_metadata(
            analysis_dir=analysis, paper_assets_dir=assets, output_path=output
        )


def test_source_binding_absolute_path_and_stale_check_fail_closed(
    tmp_path: Path,
) -> None:
    analysis, assets, output = _fixture(tmp_path / "binding")
    asset_manifest_path = assets / "paper_assets_manifest.json"
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    asset_manifest["source_analysis_evidence_sha256"] = "d" * 64
    asset_manifest["evidence_sha256"] = _self_hash(asset_manifest)
    asset_manifest_path.write_text(json.dumps(asset_manifest), encoding="utf-8")
    with pytest.raises(DefinitiveCroissantError, match="source binding failed"):
        build_metadata(
            analysis_dir=analysis, paper_assets_dir=assets, output_path=output
        )

    analysis, assets, output = _fixture(tmp_path / "absolute")
    manifest_path = analysis / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_manifests"]["part0"]["manifest_path"] = (
        "/Users/private/run/manifest.json"
    )
    manifest["evidence_sha256"] = _self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DefinitiveCroissantError, match="portable basename"):
        build_metadata(
            analysis_dir=analysis, paper_assets_dir=assets, output_path=output
        )

    analysis, assets, output = _fixture(tmp_path / "check")
    assert (
        main(
            [
                "--analysis-dir",
                str(analysis),
                "--paper-assets-dir",
                str(assets),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "--analysis-dir",
                str(analysis),
                "--paper-assets-dir",
                str(assets),
                "--output",
                str(output),
                "--check",
            ]
        )
        == 0
    )
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "--analysis-dir",
                str(analysis),
                "--paper-assets-dir",
                str(assets),
                "--output",
                str(output),
                "--check",
            ]
        )
