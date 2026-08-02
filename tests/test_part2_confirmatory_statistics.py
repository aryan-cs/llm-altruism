import json
import math

import pytest

from analysis.part2_confirmatory import (
    DEFAULT_BCA_REPLICATES,
    SENSITIVITY_FACTORS,
    analyze_resolution_v_main_effects,
    analyze_sentinel_sensitivity,
    bca_mean_interval,
    resolution_v_half_fraction,
    select_blinded_variance_run_count,
)
from analysis.part2_dynamics import (
    Part2StructuralCell,
    bernoulli_policy_baseline,
    mechanical_survival_threshold,
    no_call_baseline_suite,
    threshold_policy_baseline,
)


def _baseline_cell() -> Part2StructuralCell:
    return Part2StructuralCell(
        provider="no-call",
        model="baseline",
        society_size=50,
        horizon_days=100,
        resource="water",
        resource_capacity=2500,
        selfish_gain=2,
        depletion_units=2,
        community_benefit=5,
        collapse_death_rate=0.2,
    )


def _values_with_sample_sd(mean: float, standard_deviation: float) -> list[float]:
    offset = standard_deviation * math.sqrt(7 / 8)
    return [mean - offset] * 4 + [mean + offset] * 4


def test_blinded_variance_gate_selects_from_variance_only() -> None:
    seeds = range(8100, 8108)
    lower_mean = _values_with_sample_sd(0.3, 0.1)
    higher_mean = _values_with_sample_sd(0.7, 0.15)
    artifact = select_blinded_variance_run_count(
        {
            "blind-alpha": dict(zip(seeds, lower_mean, strict=True)),
            "blind-beta": dict(zip(seeds, higher_mean, strict=True)),
        },
        expected_blinded_groups=("blind-alpha", "blind-beta"),
    )

    assert artifact["pilot_runs_per_group"] == 8
    assert artifact["selected_common_run_count"] == 38
    assert artifact["achieved_half_width"] <= 0.05
    assert artifact["identity_masked"] is True
    serialized = json.dumps(artifact, sort_keys=True)
    assert "blind-alpha" not in serialized
    assert "blind-beta" not in serialized
    assert "0.3" not in serialized
    assert "0.7" not in serialized
    assert not any("mean" in key for key in artifact)


def test_blinded_variance_gate_caps_and_rejects_incomplete_designs() -> None:
    seeds = range(8)
    high_variance = _values_with_sample_sd(0.5, 0.3)
    artifact = select_blinded_variance_run_count(
        {"blind": dict(zip(seeds, high_variance, strict=True))},
        expected_blinded_groups=("blind",),
    )
    assert artifact["selected_common_run_count"] == 40
    assert artifact["capped_at_maximum"] is True

    with pytest.raises(ValueError, match="exactly 8"):
        select_blinded_variance_run_count(
            {"blind": {seed: 0.5 for seed in range(7)}},
            expected_blinded_groups=("blind",),
        )
    with pytest.raises(ValueError, match="same eight"):
        select_blinded_variance_run_count(
            {
                "one": {seed: 0.5 for seed in range(8)},
                "two": {seed: 0.5 for seed in range(1, 9)},
            },
            expected_blinded_groups=("one", "two"),
        )
    with pytest.raises(ValueError, match="frozen blinded groups"):
        select_blinded_variance_run_count(
            {"one": {seed: 0.5 for seed in range(8)}},
            expected_blinded_groups=("one", "missing"),
        )


def test_bca_mean_interval_is_deterministic_and_run_level() -> None:
    values = [0.1, 0.2, 0.4, 0.8, 0.9]
    first = bca_mean_interval(values, replicates=500, seed=71)
    second = bca_mean_interval(values, replicates=500, seed=71)
    assert first == second
    assert first[0] < sum(values) / len(values) < first[1]
    assert bca_mean_interval([0.4, 0.4, 0.4], replicates=500) == (0.4, 0.4)
    assert all(math.isnan(value) for value in bca_mean_interval([0.4]))
    with pytest.raises(ValueError, match="at least 200"):
        bca_mean_interval(values, replicates=199)


def test_exact_no_call_baselines_and_mechanical_survival_threshold() -> None:
    cell = _baseline_cell()
    threshold = mechanical_survival_threshold(cell)

    assert threshold.total_agent_days == 5000
    assert threshold.max_safe_overuse_actions == 1249
    assert threshold.max_safe_overuse_rate == pytest.approx(0.2498)
    assert threshold.min_restraint_actions == 3751
    assert threshold.min_restraint_rate == pytest.approx(0.7502)
    assert threshold.resource_floor_units == 2

    policy = threshold_policy_baseline(cell)
    assert policy.depletion_day is None
    assert policy.final_resource_units == threshold.resource_floor_units
    assert policy.restraint_rate == threshold.min_restraint_rate
    assert policy.final_population == 50
    assert policy.normalized_aupc == 1.0


def test_bernoulli_no_call_simulation_is_exactly_seeded() -> None:
    cell = _baseline_cell()
    first = bernoulli_policy_baseline(cell, 0.5, environment_seed=912)
    repeated = bernoulli_policy_baseline(cell, 0.5, environment_seed=912)
    different_seed = bernoulli_policy_baseline(cell, 0.5, environment_seed=913)

    assert first == repeated
    assert first != different_seed
    assert first.policy == "bernoulli-overuse-p0.50-seed-912"
    assert 0.0 <= first.normalized_aurc <= 1.0
    assert 0.0 <= first.normalized_aupc <= 1.0
    with pytest.raises(ValueError, match="0.25, 0.50, or 0.75"):
        bernoulli_policy_baseline(cell, 0.4, environment_seed=912)

    suite = no_call_baseline_suite(cell, environment_seeds=[912, 913])
    assert len(suite) == 9
    assert {row.policy for row in suite[:3]} == {
        "always-restrain",
        "always-overuse",
        "safe-reserve-threshold",
    }
    assert sum(row.policy.startswith("bernoulli") for row in suite) == 6
    with pytest.raises(ValueError, match="unique"):
        no_call_baseline_suite(cell, environment_seeds=[912, 912])


def _sensitivity_observations(
    *,
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105, 106),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cell in resolution_v_half_fraction():
        coded = cell["coded_levels"]
        for seed_index, seed in enumerate(seeds):
            # Capacity has a stable positive effect with small seed-blocked
            # variation; the common seed offset cancels from every contrast.
            value = (
                0.5
                + (0.04 + seed_index * 0.001) * coded["capacity_per_initial_agent"]
                + 0.01 * coded["depletion_units"]
                + seed_index * 0.0002
            )
            rows.append(
                {
                    **{
                        factor: cell[factor]
                        for factor in SENSITIVITY_FACTORS
                    },
                    "cell_id": cell["cell_id"],
                    "resource_capacity": cell["resource_capacity"],
                    "environment_seed": seed,
                    "normalized_aurc": value,
                }
            )
    return rows


def test_resolution_v_design_is_exact_and_main_effects_use_holm_max_t() -> None:
    design = resolution_v_half_fraction()
    assert len(design) == 16
    assert len({cell["cell_id"] for cell in design}) == 16
    for cell in design:
        coded = cell["coded_levels"]
        assert math.prod(coded.values()) == 1
        assert cell["resolution"] == 5
        assert cell["resource_capacity"] == (
            cell["capacity_per_initial_agent"] * cell["society_size"]
        )
    for factor in SENSITIVITY_FACTORS:
        assert sum(cell["coded_levels"][factor] for cell in design) == 0
    for left_index, left in enumerate(SENSITIVITY_FACTORS):
        for right in SENSITIVITY_FACTORS[left_index + 1 :]:
            assert sum(
                cell["coded_levels"][left] * cell["coded_levels"][right]
                for cell in design
            ) == 0

    results = analyze_resolution_v_main_effects(_sensitivity_observations())
    assert {row["factor"] for row in results} == set(SENSITIVITY_FACTORS)
    assert all(row["cell_count"] == 16 for row in results)
    assert all(row["common_seed_count"] == 6 for row in results)
    assert all(row["permutation_count"] == 64 for row in results)
    capacity = next(
        row for row in results if row["factor"] == "capacity_per_initial_agent"
    )
    assert capacity["effect_high_minus_low"] == pytest.approx(0.085)
    assert capacity["raw_exact_p"] <= capacity["holm_adjusted_p"]
    assert capacity["raw_exact_p"] <= capacity["max_t_adjusted_p"]


def test_resolution_v_analysis_fails_closed_on_any_design_defect() -> None:
    complete = _sensitivity_observations()
    with pytest.raises(ValueError, match="incomplete"):
        analyze_resolution_v_main_effects(complete[:-6])

    noncommon = [dict(row) for row in complete]
    noncommon[0]["environment_seed"] = 999
    with pytest.raises(ValueError, match="common environment seeds"):
        analyze_resolution_v_main_effects(noncommon)

    wrong_level = [dict(row) for row in complete]
    wrong_level[0]["depletion_units"] = 2
    with pytest.raises(ValueError, match="incorrect depletion_units"):
        analyze_resolution_v_main_effects(wrong_level)


def test_sentinel_analysis_requires_six_systems_and_common_seeds() -> None:
    complete = _sensitivity_observations()
    six = {f"sentinel-{index}": complete for index in range(6)}
    frozen = tuple(six)
    results = analyze_sentinel_sensitivity(six, expected_sentinel_ids=frozen)
    assert len(results) == 30
    assert len({row["sentinel_id"] for row in results}) == 6
    assert {row["holm_family_size"] for row in results} == {30}
    assert {row["holm_family"] for row in results} == {
        "30_prespecified_sentinel_by_factor_main_effects"
    }
    assert {row["max_t_family"] for row in results} == {
        "five_main_effects_within_sentinel_diagnostic"
    }
    assert all(row["raw_exact_p"] <= row["holm_adjusted_p"] for row in results)
    assert all(
        row["within_sentinel_holm_adjusted_p"] <= row["holm_adjusted_p"]
        for row in results
    )
    assert all(
        row["within_sentinel_max_t_adjusted_p"] == row["max_t_adjusted_p"]
        for row in results
    )

    with pytest.raises(ValueError, match="exactly 6"):
        analyze_sentinel_sensitivity(
            {"only-one": complete},
            expected_sentinel_ids=("only-one",),
        )

    with pytest.raises(ValueError, match="frozen sentinels"):
        analyze_sentinel_sensitivity(
            six,
            expected_sentinel_ids=(*frozen[:-1], "unrun-sentinel"),
        )

    mixed = dict(six)
    mixed["sentinel-5"] = _sensitivity_observations(
        seeds=(201, 202, 203, 204, 205, 206)
    )
    with pytest.raises(ValueError, match="same common environment seeds"):
        analyze_sentinel_sensitivity(mixed, expected_sentinel_ids=frozen)


def test_default_bca_replicates_is_confirmatory_scale() -> None:
    assert DEFAULT_BCA_REPLICATES >= 2000
