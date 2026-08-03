from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from analysis.build_provider_safe_v2_paper_assets import (
    DEFAULT_LOCAL_CONTROLS_PATH,
    EXPECTED_ROW_COUNTS,
    INVALID_POLICY,
    PaperAssetsError,
    SENSITIVITY_FACTORS,
    SENSITIVITY_LEVELS,
    _load_and_validate,
    _self_hash,
    _validate_local_controls,
    _write_headlines,
    build_paper_assets,
)


_HEADLINE_MACRO = re.compile(
    r"^\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}$", re.MULTILINE
)


def _headline_macros(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    pairs = _HEADLINE_MACRO.findall(text)
    assert pairs
    assert len(pairs) == len({name for name, _ in pairs})
    return dict(pairs)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_payload() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    tables: dict[str, list[dict[str, Any]]] = {}

    part0_models: list[dict[str, Any]] = []
    part0_language: list[dict[str, Any]] = []
    languages = ("english", "chinese", "russian")
    for model_index in range(22):
        target = f"route/p0_{model_index:02d}:exact"
        model = f"model/p0_{model_index:02d}_exact"
        language_rows = []
        for language_index, language in enumerate(languages):
            refusal = 14 + ((model_index + language_index) % 10)
            invalid = 1
            unclear = 1
            compliance = 48 - refusal - invalid - unclear
            row = {
                "target_id": target,
                "model": model,
                "language": language,
                "scheduled_units": 48,
                "refusal_count": refusal,
                "compliance_count": compliance,
                "unclear_count": unclear,
                "invalid_count": invalid,
                "refusal_rate_all_scheduled": refusal / 48,
            }
            language_rows.append(row)
            part0_language.append(row)
        refusal = sum(row["refusal_count"] for row in language_rows)
        compliance = sum(row["compliance_count"] for row in language_rows)
        unclear = sum(row["unclear_count"] for row in language_rows)
        invalid = sum(row["invalid_count"] for row in language_rows)
        part0_models.append(
            {
                "phase": "part0",
                "target_id": target,
                "upstream_provider": f"provider_{model_index % 5}",
                "model": model,
                "scheduled_units": 144,
                "refusal_count": refusal,
                "compliance_count": compliance,
                "unclear_count": unclear,
                "first_attempt_invalid_count": invalid,
                "repaired_invalid_count": 0,
                "refusal_rate_all_scheduled": refusal / 144,
                "primary_denominator": "all_scheduled_units",
                "exploratory_only": True,
            }
        )
    tables["part0_models"] = part0_models

    part1_models: list[dict[str, Any]] = []
    for model_index in range(75):
        scheduled = 384
        invalid = scheduled if model_index == 0 else model_index % 4
        valid = scheduled - invalid
        welfare = min(valid, 190 + model_index)
        part1_models.append(
            {
                "phase": "part1",
                "target_id": f"route/p1_{model_index:02d}:exact",
                "upstream_provider": f"provider_{model_index % 9}",
                "model": f"model/p1_{model_index:02d}_exact",
                "scheduled_units": scheduled,
                "format_valid_first_attempt_count": valid,
                "first_attempt_invalid_count": invalid,
                "repaired_invalid_count": 0,
                "welfare_preserving_count_first_attempt": welfare,
                "welfare_preserving_rate_all_scheduled": welfare / scheduled,
                "welfare_preserving_rate_among_first_attempt_valid": None if valid == 0 else welfare / valid,
                "primary_denominator": "all_scheduled_units",
                "exploratory_only": True,
            }
        )
    tables["part1_models"] = part1_models

    part2_models: list[dict[str, Any]] = []
    for model_index in range(19):
        scheduled = 120
        invalid = scheduled if model_index == 0 else model_index % 3
        restraint = 0 if model_index == 0 else 58 + model_index
        overuse = scheduled - invalid - restraint
        part2_models.append(
            {
                "phase": "part2",
                "target_id": f"route/p2_{model_index:02d}:exact",
                "upstream_provider": f"provider_{model_index % 7}",
                "model": f"model/p2_{model_index:02d}_exact",
                "trajectory_count": 12,
                "operationally_eligible_trajectory_count": 0 if model_index == 0 else 12,
                "environmentally_estimable_trajectory_count": (
                    0
                    if model_index == 0
                    else 11 if invalid > 0 else 12
                ),
                "semantic_invalid_trajectory_count": (
                    0
                    if model_index == 0
                    else 1 if invalid > 0 else 0
                ),
                "scheduled_agent_days": scheduled,
                "restraint_count": restraint,
                "overuse_count": overuse,
                "first_attempt_invalid_count": invalid,
                "repaired_invalid_count": 0,
                "restraint_rate_all_scheduled": restraint / scheduled,
                "restraint_rate_among_valid": None if invalid == scheduled else restraint / (scheduled - invalid),
                "mean_aurc_eligible": None if model_index == 0 else 0.31 + model_index * 0.025,
                "mean_aupc_eligible": None if model_index == 0 else 0.7,
                "reserve_nondepletion_rate_eligible": None if model_index == 0 else 0.75,
                "mean_population_retention_eligible": None if model_index == 0 else 0.8,
                "primary_denominator": "all_scheduled_agent_days",
                "exploratory_only": True,
            }
        )
    tables["part2_models"] = part2_models

    role_rows: list[dict[str, Any]] = []
    sensitivity_models: list[dict[str, Any]] = []
    frames = ("advice", "observer_evaluation", "prediction")
    sentinel_targets = [f"route/sentinel_{index:02d}:exact" for index in range(6)]
    for model_index, target in enumerate(sentinel_targets):
        provider = f"sentinel_provider_{model_index}"
        model = f"sentinel/model_{model_index:02d}_exact"
        for frame_index, frame in enumerate(frames):
            scheduled = 384
            invalid = scheduled if model_index == 0 and frame_index == 0 else frame_index + model_index % 2
            valid = scheduled - invalid
            welfare = 0 if valid == 0 else 160 + model_index * 12 + frame_index * 9
            role_rows.append(
                {
                    "phase": "part1_role_calibration",
                    "target_id": target,
                    "upstream_provider": provider,
                    "model": model,
                    "frame_id": frame,
                    "scheduled_draws": scheduled,
                    "format_valid_first_attempt_count": valid,
                    "first_attempt_invalid_count": invalid,
                    "repaired_invalid_count": 0,
                    "welfare_preserving_count_first_attempt": welfare,
                    "welfare_preserving_rate_all_scheduled": welfare / scheduled,
                    "welfare_preserving_rate_among_first_attempt_valid": None if valid == 0 else welfare / valid,
                    "primary_denominator": "all_scheduled_draws",
                    "ancillary_only": True,
                    "frames_pooled": False,
                }
            )
        sensitivity_models.append(
            {
                "phase": "part2_sensitivity",
                "target_id": target,
                "upstream_provider": provider,
                "model": model,
                "trajectory_count": 32,
                "scheduled_agent_days": 2880,
                "first_attempt_invalid_count": model_index,
                "repaired_invalid_count": 0,
                "inference_scope": "deadline_exploratory",
                "confirmatory": False,
                "exploratory_only": True,
            }
        )
    tables["role_calibration_model_frames"] = role_rows
    tables["sensitivity_models"] = sensitivity_models

    sensitivity_rows: list[dict[str, Any]] = []
    for model_index, target in enumerate(sentinel_targets):
        for factor_index, factor in enumerate(SENSITIVITY_FACTORS):
            low, high = SENSITIVITY_LEVELS[factor]
            effect = (factor_index - 2) * 0.018 + model_index * 0.001
            holm = 0.04 if (model_index + factor_index) % 7 == 0 else 0.24
            sensitivity_rows.append(
                {
                    "sentinel_id": target,
                    "factor": factor,
                    "low_level": low,
                    "high_level": high,
                    "effect_high_minus_low": effect,
                    "standard_error": 0.01,
                    "t_statistic": effect / 0.01,
                    "raw_exact_p": 0.01,
                    "holm_adjusted_p": holm,
                    "max_t_adjusted_p": 0.08,
                    "within_sentinel_holm_adjusted_p": 0.05,
                    "within_sentinel_max_t_adjusted_p": 0.08,
                    "cell_count": 16,
                    "common_seed_count": 2,
                    "permutation_count": 4,
                    "design": "2^(5-1)_resolution_V_I=ABCDE",
                    "analysis_unit": "environment_seed_block",
                    "holm_family": "30_prespecified_sentinel_by_factor_main_effects",
                    "holm_family_size": 30,
                    "max_t_family": "five_main_effects_within_sentinel_diagnostic",
                    "inference_scope": "deadline_exploratory",
                    "confirmatory": False,
                }
            )
    tables["sensitivity_main_effects"] = sensitivity_rows

    aggregates = {
        "part0_by_model_language": part0_language,
        "part1_by_model_game_domain": [],
        "part2_trajectories": [],
        "role_calibration_by_model_frame": role_rows,
        "sensitivity_main_effects": sensitivity_rows,
    }
    return tables, aggregates


def _write_source(root: Path, *, mutate=None) -> Path:
    source = root / "definitive-analysis"
    source.mkdir()
    tables, aggregates = _source_payload()
    if mutate is not None:
        mutate(tables, aggregates)
    for name, rows in tables.items():
        _write_jsonl(source / f"{name}.jsonl", rows)
    (source / "figure_aggregates.json").write_text(
        json.dumps(aggregates, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "provider_safe_v2_definitive_descriptive_analysis",
        "generated_at_utc": "2026-08-03T12:00:00Z",
        "input_manifests": {"intentionally": "not dereferenced by paper generator"},
        "judge_disjointness": [],
        "row_counts": copy.deepcopy(EXPECTED_ROW_COUNTS),
        "invalid_policy": INVALID_POLICY,
        "human_labels_generated": False,
        "exploratory_only": True,
        "confirmatory_or_paper_promotion_permitted": False,
    }
    manifest["evidence_sha256"] = _self_hash(manifest)
    (source / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_local_controls(root / "local_controls.json")
    return source


def _write_local_controls(path: Path) -> Path:
    models = []
    scales = ("135M", "360M", "0.5B", "1.7B")
    for index, scale in enumerate(scales):
        scheduled = 384
        valid = (3, 27, 384, 362)[index]
        invalid = scheduled - valid
        welfare_count = (2, 8, 205, 179)[index]
        models.append(
            {
                "model_id": f"hf.local-model-{index}_exact",
                "parameter_scale": scale,
                "inference_status": "exploratory_descriptive_fixed_local_model",
                "counts": {
                    "planned_and_retained_count": scheduled,
                    "format_valid_count": valid,
                    "format_invalid_count": invalid,
                    "format_valid_rate": valid / scheduled,
                    "primary_cooperation_rate_format_invalid_retained_as_noncooperation": welfare_count / scheduled,
                },
            }
        )
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "local_hf_part1_exploratory_private_panel_analysis",
        "analysis_role": "exploratory_only_no_confirmatory_or_paper_promotion",
        "draft_bank_human_approved": False,
        "confirmatory_or_paper_promotion_permitted": False,
        "frontier_route_substitution_permitted": False,
        "coverage": {
            "all_selected_models_complete": True,
            "game_domain_strata": 12,
            "model_count": 4,
            "retained_model_trial_count": 1536,
            "roots_per_model": 384,
            "roots_per_stratum_per_model": 32,
        },
        "privacy_contract": {
            "contains_prompt_text": False,
            "contains_raw_response_or_response_text": False,
            "contains_reasoning": False,
            "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
        },
        "parameters": {
            "invalid_handling": "format_invalid_responses_retained_in_primary_denominator_as_non_x_and_noncooperation"
        },
        "bindings": {
            "path_policy": "portable_input_basename_only_no_host_absolute_paths",
            "panel_manifest_path": "manifest.json",
            "registry_path": "local_control.registry.json",
        },
        "models": models,
    }
    artifact["evidence_sha256"] = _self_hash(artifact)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_builds_full_production_shaped_vector_png_and_latex_assets(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    output = tmp_path / "paper-assets"
    result = build_paper_assets(source, output, tmp_path / "local_controls.json")

    assert result["human_labels_generated"] is False
    assert result["cross_axis_assets_generated"] is True
    assert result["cross_axis_aggregate_or_score_generated"] is False
    assert result["confirmatory_or_paper_promotion_permitted"] is False
    assert result["table_outer_spacing_pt"] == 15
    assert result["figure_palette"].startswith("matplotlib_turbo")
    assert result["figure_semantic_redundancy"] == (
        "directional_caption_position_and_printed_values"
    )
    assert result["route_and_model_ids_preserved_exactly"] is True
    assert len(result["assets"]) == 20
    assert result["local_controls_pooled_with_hosted_routes"] is False
    assert {path.suffix for path in output.iterdir()} >= {".pdf", ".png", ".tex", ".json"}

    expected_stems = {
        "part0_model_language", "part1_all_models", "part2_all_models",
        "part1_role_calibration", "part2_sensitivity_effects", "part1_local_controls",
    }
    assert {path.stem for path in output.glob("*.pdf")} == expected_stems
    assert {path.stem for path in output.glob("*.png")} == expected_stems
    for path in output.glob("*.pdf"):
        content = path.read_bytes()
        assert content.startswith(b"%PDF")
        assert b"/Subtype /Image" not in content, f"{path.name} unexpectedly embeds raster marks"
        assert len(content) > 8_000
    for path in output.glob("*.png"):
        with Image.open(path) as image:
            assert image.width >= 3_000
            assert image.height >= 1_000
            assert image.mode in {"RGB", "RGBA"}

    manifest = json.loads((output / "paper_assets_manifest.json").read_text())
    assert manifest["evidence_sha256"] == _self_hash(manifest)
    assert {row["name"] for row in manifest["assets"]} == {
        path.name for path in output.iterdir() if path.name != "paper_assets_manifest.json"
    }
    for row in manifest["assets"]:
        digest = hashlib.sha256((output / row["name"]).read_bytes()).hexdigest()
        assert row["file_sha256"] == digest
    headline_asset = next(row for row in manifest["assets"] if row["name"] == "paper_headlines.tex")
    assert headline_asset["kind"] == "latex_macros"


def test_deterministic_headline_macros_match_full_production_fixture_exactly(
    tmp_path: Path,
) -> None:
    source = _write_source(tmp_path)
    local_path = tmp_path / "local_controls.json"
    data = _load_and_validate(source)
    _, local_rows = _validate_local_controls(local_path)
    data["local_controls"] = local_rows
    first = tmp_path / "headlines-first"
    second = tmp_path / "headlines-second"
    first.mkdir()
    second.mkdir()
    first_path = _write_headlines(data, first)
    second_path = _write_headlines(data, second)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.read_bytes().endswith(b"\n")

    assert _headline_macros(first_path) == {
        "ProviderSafePartZeroModelCount": "22",
        "ProviderSafePartZeroScheduledResponseCount": "3168",
        "ProviderSafePartZeroRefusalCount": "1203",
        "ProviderSafePartZeroComplianceCount": "1833",
        "ProviderSafePartZeroUnclearCount": "66",
        "ProviderSafePartZeroInvalidCount": "66",
        "ProviderSafePartZeroModelRefusalRatePctMinimum": "31.2",
        "ProviderSafePartZeroModelRefusalRatePctMedian": "37.5",
        "ProviderSafePartZeroModelRefusalRatePctMaximum": "45.8",
        "ProviderSafePartOneModelCount": "75",
        "ProviderSafePartOneScheduledUnitCount": "28800",
        "ProviderSafePartOneWelfarePreservingCount": "16835",
        "ProviderSafePartOneInvalidCount": "495",
        "ProviderSafePartOneModelWelfareRatePctMinimum": "0.0",
        "ProviderSafePartOneModelWelfareRatePctMedian": "59.1",
        "ProviderSafePartOneModelWelfareRatePctMaximum": "68.8",
        "ProviderSafePartTwoModelCount": "19",
        "ProviderSafePartTwoTrajectoryCount": "228",
        "ProviderSafePartTwoOperationallyEligibleTrajectoryCount": "216",
        "ProviderSafePartTwoOperationallyIneligibleTrajectoryCount": "12",
        "ProviderSafePartTwoEnvironmentallyEstimableTrajectoryCount": "204",
        "ProviderSafePartTwoSemanticInvalidTrajectoryCount": "12",
        "ProviderSafePartTwoScheduledAgentDayCount": "2280",
        "ProviderSafePartTwoValidAgentDayCount": "2142",
        "ProviderSafePartTwoInvalidAgentDayCount": "138",
        "ProviderSafePartTwoNonestimableModelCount": "1",
        "ProviderSafePartTwoModelNormalizedAURCMinimum": "0.335",
        "ProviderSafePartTwoModelNormalizedAURCMedian": "0.548",
        "ProviderSafePartTwoModelNormalizedAURCMaximum": "0.760",
        "ProviderSafePartTwoModelNormalizedAUPCMinimum": "0.700",
        "ProviderSafePartTwoModelNormalizedAUPCMedian": "0.700",
        "ProviderSafePartTwoModelNormalizedAUPCMaximum": "0.700",
        "ProviderSafePartTwoModelReserveNondepletionRatePctMinimum": "75.0",
        "ProviderSafePartTwoModelReserveNondepletionRatePctMedian": "75.0",
        "ProviderSafePartTwoModelReserveNondepletionRatePctMaximum": "75.0",
        "ProviderSafePartTwoModelPopulationRetentionPctMinimum": "80.0",
        "ProviderSafePartTwoModelPopulationRetentionPctMedian": "80.0",
        "ProviderSafePartTwoModelPopulationRetentionPctMaximum": "80.0",
        "ProviderSafePartTwoModelRestraintRatePctMinimum": "0.0",
        "ProviderSafePartTwoModelRestraintRatePctMedian": "55.8",
        "ProviderSafePartTwoModelRestraintRatePctMaximum": "63.3",
        "ProviderSafeRoleAdviceModelCount": "6",
        "ProviderSafeRoleAdviceWelfareRatePctMinimum": "0.0",
        "ProviderSafeRoleAdviceWelfareRatePctMedian": "49.5",
        "ProviderSafeRoleAdviceWelfareRatePctMaximum": "57.3",
        "ProviderSafeRoleAdviceValidCoveragePctMinimum": "0.0",
        "ProviderSafeRoleAdviceValidCoveragePctMedian": "99.7",
        "ProviderSafeRoleAdviceValidCoveragePctMaximum": "100.0",
        "ProviderSafeRoleObserverEvaluationModelCount": "6",
        "ProviderSafeRoleObserverEvaluationWelfareRatePctMinimum": "44.0",
        "ProviderSafeRoleObserverEvaluationWelfareRatePctMedian": "51.8",
        "ProviderSafeRoleObserverEvaluationWelfareRatePctMaximum": "59.6",
        "ProviderSafeRoleObserverEvaluationValidCoveragePctMinimum": "99.5",
        "ProviderSafeRoleObserverEvaluationValidCoveragePctMedian": "99.6",
        "ProviderSafeRoleObserverEvaluationValidCoveragePctMaximum": "99.7",
        "ProviderSafeRolePredictionModelCount": "6",
        "ProviderSafeRolePredictionWelfareRatePctMinimum": "46.4",
        "ProviderSafeRolePredictionWelfareRatePctMedian": "54.2",
        "ProviderSafeRolePredictionWelfareRatePctMaximum": "62.0",
        "ProviderSafeRolePredictionValidCoveragePctMinimum": "99.2",
        "ProviderSafeRolePredictionValidCoveragePctMedian": "99.3",
        "ProviderSafeRolePredictionValidCoveragePctMaximum": "99.5",
        "ProviderSafeSensitivitySentinelCount": "6",
        "ProviderSafeSensitivityTrajectoryCount": "192",
        "ProviderSafeSensitivityCommonSeedCount": "2",
        "ProviderSafeSensitivityEffectCount": "30",
        "ProviderSafeSensitivityMaximumAbsoluteEffect": "0.0410",
        "ProviderSafeSensitivityHolmSignificantCount": "4",
    }
    text = first_path.read_text(encoding="utf-8")
    assert "no cross-axis aggregate or promotion is defined" in text
    assert "CrossAxis" not in text and "Composite" not in text


def test_latex_tables_preserve_ids_define_directions_and_space_every_float(tmp_path: Path) -> None:
    output = tmp_path / "paper-assets"
    build_paper_assets(
        _write_source(tmp_path), output, tmp_path / "local_controls.json"
    )

    expected_table_counts = {
        "all_models_cross_phase_table.tex": 5,
        "part0_model_language_table.tex": 1,
        "part1_all_models_table.tex": 3,
        "part2_all_models_table.tex": 1,
        "part1_role_calibration_table.tex": 1,
        "part2_sensitivity_effects_table.tex": 2,
        "part1_local_controls_table.tex": 1,
    }
    for name, table_count in expected_table_counts.items():
        text = (output / name).read_text()
        assert text.count("\\begin{table*}") == table_count
        assert text.count("\\par\\addvspace{15pt}") == table_count * 2
        assert not re.search(r"\\vspace\s*\{\s*-", text)
        assert not re.search(r"(?<!\\)_", text), f"{name} contains an unescaped underscore"
        assert "general safety" in text
        assert "Model ID" in text
        if name != "part1_local_controls_table.tex":
            assert "Target route ID" in text

    part0 = (output / "part0_model_language_table.tex").read_text()
    assert r"route/p0\_00:exact" in part0
    assert r"model/p0\_00\_exact" in part0
    assert "english" in part0 and "chinese" in part0 and "russian" in part0
    part1 = (output / "part1_all_models_table.tex").read_text()
    assert sum(line.rstrip().endswith(r"\\") for line in part1.splitlines()) >= 75
    role = (output / "part1_role_calibration_table.tex").read_text()
    assert "frames ask different questions" in role
    sensitivity = (output / "part2_sensitivity_effects_table.tex").read_text()
    assert "Positive/negative is not automatically good/bad" in sensitivity
    assert "Holm p" in sensitivity and "high-low AURC" in sensitivity
    local = (output / "part1_local_controls_table.tex").read_text()
    assert "exploratory local execution-scale controls" in local
    assert "not substitutes for hosted routes" in local
    assert "format-invalid outputs retained as nonsuccesses" in local
    assert "135M" in local and "1.7B" in local
    cross_phase = (output / "all_models_cross_phase_table.tex").read_text()
    assert "Each row is one exact target route" in cross_phase
    assert "Part 0: R; V" in cross_phase
    assert "Part 1: W; V" in cross_phase
    assert "Part 2: R; A; V" in cross_phase
    assert "environmentally estimable-trajectory normalized AURC" in cross_phase
    assert "no environmentally estimable AURC trajectory" in cross_phase
    assert "no composite or general safety ranking is computed" in cross_phase
    assert "-- (not in panel)" in cross_phase


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda tables, aggregates: tables["part1_models"].pop(),
            "row count",
        ),
        (
            lambda tables, aggregates: tables["part2_models"][0].pop("mean_aurc_eligible"),
            "required fields",
        ),
        (
            lambda tables, aggregates: tables["part1_models"].__setitem__(
                1, {**tables["part1_models"][1], "target_id": tables["part1_models"][0]["target_id"]}
            ),
            "duplicates exact target_id",
        ),
        (
            lambda tables, aggregates: aggregates["part0_by_model_language"].pop(),
            "66 rows",
        ),
        (
            lambda tables, aggregates: tables["sensitivity_main_effects"][0].__setitem__("holm_family_size", 5),
            "Holm contract",
        ),
        (
            lambda tables, aggregates: tables["role_calibration_model_frames"][0].__setitem__("model", "changed-model"),
            "identity",
        ),
    ],
)
def test_missing_or_inconsistent_schema_rows_fail_closed_before_output(
    tmp_path: Path, mutation, message: str,
) -> None:
    source = _write_source(tmp_path, mutate=mutation)
    output = tmp_path / "must-not-exist"
    with pytest.raises(PaperAssetsError, match=message):
        build_paper_assets(source, output, tmp_path / "local_controls.json")
    assert not output.exists()


def test_tampered_manifest_and_existing_output_fail_closed(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    manifest_path = source / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["human_labels_generated"] = True
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(PaperAssetsError, match="self-hash"):
        build_paper_assets(
            source, tmp_path / "tampered-output", tmp_path / "local_controls.json"
        )
    assert not (tmp_path / "tampered-output").exists()

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean_source = _write_source(clean_root)
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "owner-data.txt"
    marker.write_text("preserve")
    with pytest.raises(PaperAssetsError, match="refusing overwrite"):
        build_paper_assets(
            clean_source, existing, clean_root / "local_controls.json"
        )
    assert marker.read_text() == "preserve"


def test_local_control_schema_and_repository_sanitized_artifact(tmp_path: Path) -> None:
    artifact, rows = _validate_local_controls(DEFAULT_LOCAL_CONTROLS_PATH)
    assert artifact["evidence_sha256"] == _self_hash(artifact)
    assert len(rows) == 4
    assert [row["parameter_scale"] for row in rows] == ["135M", "360M", "0.5B", "1.7B"]

    source = _write_source(tmp_path)
    local_path = tmp_path / "local_controls.json"
    local = json.loads(local_path.read_text())
    local["models"].pop()
    local["evidence_sha256"] = _self_hash(local)
    local_path.write_text(json.dumps(local) + "\n", encoding="utf-8")
    output = tmp_path / "local-missing-output"
    with pytest.raises(PaperAssetsError, match="exactly four model rows"):
        build_paper_assets(source, output, local_path)
    assert not output.exists()
