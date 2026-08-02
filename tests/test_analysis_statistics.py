import csv
import json
import math
from pathlib import Path

import pytest

from agents.agent_config import load_model_registry
from analysis.model_metadata import (
    CURRENT_SOTA,
    HISTORICAL,
    ORIGINAL_PILOT,
    resolve_model_metadata,
)
from analysis.statistics import (
    fisher_z_interval,
    kendall_correlation,
    leave_group_out_range,
    pearson_correlation,
    spearman_correlation,
)
from analysis.summarize_results import (
    PILOT_ROOT_BOOTSTRAP_METHOD,
    PILOT_ROOT_BOOTSTRAP_REPLICATES,
    PILOT_ROOT_BOOTSTRAP_SEED,
    _root_cluster_rate_interval,
    summarize_cross_part,
    summarize_part0,
    summarize_part1,
    summarize_part2,
)
from experiments.part2.part_2 import PILOT_RESULT_HEADERS as PART2_RESULT_HEADERS


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_model_metadata_labels_legacy_pilot_and_respects_explicit_new_cohort() -> None:
    pilot = resolve_model_metadata("ollama", "huihui_ai/qwen2.5-abliterate:7b")
    current = resolve_model_metadata(
        "api-provider",
        "new-model-version",
        {
            "model_metadata": {
                "family_id": "Model Family 7",
                "developer_id": "Developer Labs",
                "cohort": "current_sota",
            }
        },
    )

    assert (pilot.family_id, pilot.developer_id, pilot.cohort) == (
        "qwen_2_5",
        "alibaba_qwen",
        ORIGINAL_PILOT,
    )
    assert resolve_model_metadata("ollama", "gpt-oss-safeguard:20b").family_id == "gpt_oss"
    assert resolve_model_metadata("inference_hub", "claude-opus-5") == resolve_model_metadata(
        "inference_hub", "claude-sonnet-5"
    )
    assert resolve_model_metadata("inference_hub", "claude-opus-5").cohort == CURRENT_SOTA
    assert resolve_model_metadata("inference_hub", "gpt-4.1-2025-04-14").cohort == HISTORICAL
    assert (current.family_id, current.developer_id, current.cohort) == (
        "model_family_7",
        "developer_labs",
        CURRENT_SOTA,
    )
    sidecar_derived = resolve_model_metadata(
        "inference_hub",
        "future-route",
        {
            "model_registry": {
                "targets": [
                    {
                        "model": "future-route",
                        "route": "future-route",
                        "upstream_provider": "Future Developer",
                        "cohorts": ["historical"],
                    }
                ]
            }
        },
    )
    assert sidecar_derived.developer_id == "future_developer"
    assert sidecar_derived.cohort == HISTORICAL
    with pytest.raises(ValueError, match="Invalid cohort"):
        resolve_model_metadata(
            "provider", "model", {"family_id": "f", "developer_id": "d", "cohort": "new"}
        )


def test_every_frozen_inference_hub_model_and_route_has_exact_analysis_metadata() -> None:
    load_model_registry.cache_clear()
    registry = load_model_registry()
    membership = {
        target_id: cohort_id
        for cohort_id in (CURRENT_SOTA, HISTORICAL)
        for target_id in registry["cohorts"][cohort_id]["targets"]
    }

    for target in registry["targets"]:
        if target["id"] not in membership:
            continue
        model_metadata = resolve_model_metadata("inference_hub", target["model"])
        route_metadata = resolve_model_metadata("inference_hub", target["route"])
        assert model_metadata == route_metadata, target["id"]
        assert model_metadata.cohort == membership[target["id"]], target["id"]


def test_three_correlation_estimators_and_fisher_interval() -> None:
    xs = [1.0, 2.0, 2.0, 4.0, 5.0]
    ys = [1.0, 3.0, 3.0, 2.0, 6.0]

    assert -1.0 <= pearson_correlation(xs, ys) <= 1.0
    assert -1.0 <= spearman_correlation(xs, ys) <= 1.0
    assert -1.0 <= kendall_correlation(xs, ys) <= 1.0
    assert kendall_correlation(xs, xs) == 1.0

    estimate = pearson_correlation(xs, ys)
    low, high = fisher_z_interval(estimate, len(xs))
    assert low < estimate < high
    assert all(math.isnan(value) for value in fisher_z_interval(estimate, 3))


def test_leave_group_out_range_omits_models_and_whole_families() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 2.0, 4.0, 8.0]

    model_low, model_high, model_estimates = leave_group_out_range(
        xs, ys, ["m1", "m2", "m3", "m4"], pearson_correlation
    )
    family_low, family_high, family_estimates = leave_group_out_range(
        xs, ys, ["family_a", "family_a", "family_b", "family_c"], pearson_correlation
    )

    assert model_estimates == 4
    assert family_estimates == 3
    assert model_low <= model_high
    assert family_low <= family_high


def test_cross_part_table_reports_all_correlations_and_influence_ranges(tmp_path: Path) -> None:
    models = ["model-a", "model-b", "model-c", "model-d", "model-e"]
    families = ["family_1", "family_1", "family_2", "family_3", "family_3"]
    _write_rows(
        tmp_path / "part0_model_summary.csv",
        [
            {"provider": "provider", "model": model, "safety_refusal_rate": value}
            for model, value in zip(models, [0.1, 0.2, 0.4, 0.7, 0.8])
        ],
    )
    _write_rows(
        tmp_path / "part1_model_summary.csv",
        [
            {
                "provider": "provider",
                "model": model,
                "family_id": family,
                "developer_id": f"developer_{index}",
                "cohort": "current_sota",
                "cooperation_rate": value,
                "all_frames_cooperation_rate": value / 2,
            }
            for index, (model, family, value) in enumerate(
                zip(models, families, [0.3, 0.1, 0.6, 0.5, 0.9]), start=1
            )
        ],
    )
    _write_rows(
        tmp_path / "part2_model_summary.csv",
        [
            {
                "provider": "provider",
                "model": model,
                "aggregation_unit": "run",
                "run_count": 2,
                "structural_cell_key": f"cell-{model}",
                "restraint_rate": restraint,
                "final_population": population,
                "final_resource_units": 10 + index,
                "normalized_aurc": 0.2 + index / 100,
                "normalized_aupc": 0.3 + index / 100,
                "restricted_mean_time_to_depletion": 100,
                "survival_through_horizon": 1,
                "reasoning_mismatch_flags": 0,
            }
            for index, (model, restraint, population) in enumerate(
                zip(models, [0.8, 0.4, 0.5, 0.2, 0.7], [10, 20, 15, 35, 30]),
                start=1,
            )
        ],
    )

    _, correlation_path = summarize_cross_part(tmp_path)
    correlations = _read_rows(correlation_path)

    assert len(correlations) == 6
    assert {row["cooperation_measure"] for row in correlations} == {"self_direct"}
    assert {row["n_models"] for row in correlations} == {"5"}
    assert {row["n_families"] for row in correlations} == {"3"}
    assert {row["fisher_z_ci_note"] for row in correlations} == {
        "pearson_standard;rank_coefficients_approximate"
    }
    assert {
        (row["metric_x"], row["analysis_status"])
        for row in correlations
        if row["metric_x"] == "safety_refusal_rate"
    } == {
        ("safety_refusal_rate", "deprecated_legacy_part0_label_protocol")
    }
    assert all(
        row["analysis_status"] == "supported_descriptive_pilot"
        for row in correlations
        if row["metric_x"] != "safety_refusal_rate"
    )
    for row in correlations:
        assert row["pearson_fisher_z_low"]
        assert row["spearman_r"]
        assert row["kendall_tau_b"]
        assert row["pearson_leave_one_model_out_estimates"] == "5"
        assert row["pearson_leave_one_family_out_estimates"] == "3"


def test_part1_model_summary_uses_self_direct_as_primary_measure(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    part_dir = raw_dir / "part_1"
    part_dir.mkdir(parents=True)
    csv_path = part_dir / "part1__ollama__qwen3-5__full__20260101_000000.csv"
    fieldnames = [
        "provider",
        "model",
        "game",
        "frame",
        "domain",
        "scenario_variant",
        "presentation",
        "action",
    ]
    rows = [
        {
            "provider": "ollama",
            "model": "qwen3.5",
            "game": "prisoners_dilemma",
            "frame": "self_direct",
            "domain": "workplace",
            "scenario_variant": "root-primary",
            "presentation": "structured",
            "action": "COOPERATE",
        },
        *[
            {
                "provider": "ollama",
                "model": "qwen3.5",
                "game": "prisoners_dilemma",
                "frame": "advice",
                "domain": "workplace",
                "scenario_variant": "root-advice",
                "presentation": "structured",
                "action": "DEFECT",
            }
            for _ in range(3)
        ],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_dir = tmp_path / "tables"
    summarize_part1(raw_dir, output_dir)
    [summary] = _read_rows(output_dir / "part1_model_summary.csv")

    assert summary["primary_frame"] == "self_direct"
    assert float(summary["cooperation_rate"]) == 1.0
    assert int(summary["total"]) == 1
    assert float(summary["all_frames_cooperation_rate"]) == 0.25
    assert summary["interval_method"] == PILOT_ROOT_BOOTSTRAP_METHOD
    assert summary["interval_unit"] == "scenario_variant"
    assert int(summary["interval_replicates"]) == PILOT_ROOT_BOOTSTRAP_REPLICATES
    assert int(summary["interval_seed"]) == PILOT_ROOT_BOOTSTRAP_SEED
    assert int(summary["root_cluster_count"]) == 1
    assert float(summary["cluster_ci_low"]) == 1.0
    assert float(summary["cluster_ci_high"]) == 1.0
    assert "wilson_low" not in summary
    assert summary["row_binomial_wilson_low_diagnostic"]


def test_root_cluster_bootstrap_keeps_paired_rows_together_and_has_no_iid_fallback() -> None:
    paired = [
        ("root-a", 1),
        ("root-a", 1),
        ("root-a", 0),
        ("root-b", 1),
        ("root-b", 1),
        ("root-b", 0),
    ]

    first = _root_cluster_rate_interval(paired)
    second = _root_cluster_rate_interval(paired)

    assert first == second
    assert first == pytest.approx((2 / 3, 2 / 3, 2))
    with pytest.raises(ValueError, match="design-root ID"):
        _root_cluster_rate_interval([("", 1), ("", 0)])
    with pytest.raises(ValueError, match="exactly 2,000"):
        _root_cluster_rate_interval(paired, replicates=1_999)


def test_part0_paper_summary_uses_base_prompt_cluster_intervals(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    part_dir = raw_dir / "part_0"
    part_dir.mkdir(parents=True)
    rows = [
        {
            "provider": "ollama",
            "model": "qwen3.5",
            "language": language,
            "prompt": prompt,
            "complied?": complied,
        }
        for prompt, complied in (("root-a", "false"), ("root-b", "true"))
        for language in ("english", "chinese", "russian")
    ]
    _write_rows(part_dir / "pilot.csv", rows)

    first_dir = tmp_path / "tables-first"
    second_dir = tmp_path / "tables-second"
    summarize_part0(raw_dir, first_dir)
    summarize_part0(raw_dir, second_dir)
    [first] = _read_rows(first_dir / "part0_model_summary.csv")
    [second] = _read_rows(second_dir / "part0_model_summary.csv")

    assert first == second
    assert float(first["safety_refusal_rate"]) == 0.5
    assert float(first["cluster_ci_low"]) == 0.0
    assert float(first["cluster_ci_high"]) == 1.0
    assert first["interval_method"] == PILOT_ROOT_BOOTSTRAP_METHOD
    assert first["interval_unit"] == "base_prompt"
    assert int(first["interval_replicates"]) == 2_000
    assert int(first["interval_seed"]) == PILOT_ROOT_BOOTSTRAP_SEED
    assert int(first["root_cluster_count"]) == 2
    assert 0.0 < float(first["row_binomial_wilson_low_diagnostic"]) < 0.5
    assert 0.5 < float(first["row_binomial_wilson_high_diagnostic"]) < 1.0
    assert "wilson_low" not in first


def test_checked_in_pilot_uses_99_part0_and_48_part1_root_clusters(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"
    if not (raw_dir / "part_0").is_dir():
        pytest.skip("anonymous supplement intentionally withholds raw Part 0")
    summarize_part0(raw_dir, tmp_path)
    summarize_part1(raw_dir, tmp_path)

    part0_rows = _read_rows(tmp_path / "part0_model_summary.csv")
    part1_rows = _read_rows(tmp_path / "part1_model_summary.csv")
    dimension_rows = _read_rows(tmp_path / "part1_dimension_summary.csv")

    assert {int(row["root_cluster_count"]) for row in part0_rows} == {99}
    assert {int(row["root_cluster_count"]) for row in part1_rows} == {48}
    assert {
        dimension: {
            int(item["root_cluster_count"])
            for item in dimension_rows
            if item["dimension"] == dimension
        }
        for dimension in {row["dimension"] for row in dimension_rows}
    } == {
        "domain": {8},
        "frame": {48},
        "game": {24},
        "presentation": {48},
    }
    assert {row["interval_method"] for row in [*part0_rows, *part1_rows]} == {
        PILOT_ROOT_BOOTSTRAP_METHOD
    }


def _write_part2_run(path: Path, actions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PART2_RESULT_HEADERS)
        writer.writeheader()
        restraint_count = sum(action == "RESTRAIN" for action in actions)
        overuse_count = len(actions) - restraint_count
        resource_remaining = max(0, 10 - overuse_count * 2)
        deaths = math.ceil(len(actions) * 0.2) if resource_remaining == 0 else 0
        for index, action in enumerate(actions, start=1):
            writer.writerow(
                {
                    "provider": "ollama",
                    "model": "gpt-oss:20b",
                    "day": 1,
                    "agent": f"society_{index}",
                    "action": action,
                    "reasoning": "",
                    "population_start": len(actions),
                    "population_end": len(actions) - deaths,
                    "restrain_count": restraint_count,
                    "overuse_count": overuse_count,
                    "resource_units_remaining": resource_remaining,
                    "resource_capacity": 10,
                    "deaths": deaths,
                    "resource": "water",
                    "selfish_gain": 2,
                    "depletion_units": 2,
                    "community_benefit": 5,
                }
            )
    path.with_name(f"{path.stem}_meta.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "society_config": {
                        "days": 1,
                        "society_size": len(actions),
                        "resource": "water",
                        "selfish_gain": 2,
                        "depletion_units": 2,
                        "community_benefit": 5,
                    },
                    "resource_capacity": 10,
                    "collapse_death_rate": 0.2,
                }
            }
        ),
        encoding="utf-8",
    )


def test_part2_different_population_cells_are_not_pooled(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    part_dir = raw_dir / "part_2"
    _write_part2_run(
        part_dir / "part2__ollama__gpt-oss-20b__n1__d1__water__20260101_000000.csv",
        ["RESTRAIN"],
    )
    _write_part2_run(
        part_dir / "part2__ollama__gpt-oss-20b__n9__d1__water__20260102_000000.csv",
        ["OVERUSE"] * 9,
    )

    output_dir = tmp_path / "tables"
    summarize_part2(raw_dir, output_dir)
    summaries = _read_rows(output_dir / "part2_model_summary.csv")
    run_rows = _read_rows(output_dir / "part2_run_summary.csv")

    assert len(run_rows) == 2
    assert len(summaries) == 2
    assert {int(row["society_size"]) for row in summaries} == {1, 9}
    assert {int(row["run_count"]) for row in summaries} == {1}
    assert {float(row["restraint_rate"]) for row in summaries} == {0.0, 1.0}
    assert {row["interval_method"] for row in summaries} == {
        "not_estimable_single_run"
    }
    assert all(not row["restraint_ci_low"] for row in summaries)
    assert all(not row["restraint_ci_high"] for row in summaries)
