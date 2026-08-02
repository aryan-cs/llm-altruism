"""Prespecified fixed-stage confirmatory statistics for Part 2.

The functions in this module are deliberately provider-independent.  They
operate on completed run-level quantities, never on agent-days. Historical
variance-pilot helpers remain private solely to revalidate archived two-stage
data-lock lineage; the CLI cannot create or select a two-stage campaign.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import stat
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

from analysis.part2_dynamics import (
    Part2StructuralCell,
    load_part2_run_identity,
    load_part2_structural_cell,
    mechanical_survival_threshold,
    no_call_baseline_suite,
    normalized_part2_auc,
)


# Archived two-stage constants support historical data-lock validation only.
VARIANCE_PILOT_SEEDS = 8
MIN_BASELINE_RUNS = 20
MAX_BASELINE_RUNS = 40
TARGET_HALF_WIDTH = 0.05
DEFAULT_BCA_REPLICATES = 2_000

SENSITIVITY_FACTORS = (
    "capacity_per_initial_agent",
    "depletion_units",
    "collapse_death_rate",
    "society_size",
    "horizon_days",
)
SENSITIVITY_LEVELS: dict[str, tuple[int | float, int | float]] = {
    "capacity_per_initial_agent": (25, 75),
    "depletion_units": (1, 4),
    "collapse_death_rate": (0.1, 0.4),
    "society_size": (25, 50),
    "horizon_days": (50, 100),
}
SENSITIVITY_CELL_COUNT = 16
# Twelve common seeds give 2^12 exact sign patterns. For a two-sided absolute
# statistic the minimum attainable p-value is 2/2^12, which remains below
# 0.05/30 for the frozen global Holm family. Six seeds could never reject even
# an infinite effect after the 30-test adjustment.
SENSITIVITY_SEEDS_PER_CELL = 12
SENSITIVITY_SENTINEL_COUNT = 6


# Two-sided 95% Student critical values used by the prespecified n=20..40
# fixed-width gate.  Recording the finite-df values avoids a scipy dependency
# and, importantly, avoids silently substituting the normal quantile at n=40.
_T_975_BY_DF = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
    10: 2.228139,
    11: 2.200985,
    12: 2.178813,
    13: 2.160369,
    14: 2.144787,
    15: 2.131450,
    16: 2.119905,
    17: 2.109816,
    18: 2.100922,
    19: 2.093024,
    20: 2.085963,
    21: 2.079614,
    22: 2.073873,
    23: 2.068658,
    24: 2.063899,
    25: 2.059539,
    26: 2.055529,
    27: 2.051831,
    28: 2.048407,
    29: 2.045230,
    30: 2.042272,
    31: 2.039513,
    32: 2.036933,
    33: 2.034515,
    34: 2.032245,
    35: 2.030108,
    36: 2.028094,
    37: 2.026192,
    38: 2.024394,
    39: 2.022691,
}


def student_t_975(df: int) -> float:
    """Return the recorded 0.975 Student-t critical value."""

    if df not in _T_975_BY_DF:
        raise ValueError("the recorded Student-t table supports df=1..39")
    return _T_975_BY_DF[df]


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("a sample standard deviation requires at least two runs")
    mean = math.fsum(values) / len(values)
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def select_blinded_variance_run_count(
    pilot_by_blinded_group: Mapping[str, Mapping[int, float]],
    *,
    expected_blinded_groups: Sequence[str],
) -> dict[str, object]:
    """Recompute the archived two-stage run count for lineage validation.

    Group labels exist solely to separate the pilot runs during calculation.
    The returned artifact contains neither labels nor group means.  Its input
    digest is calculated from sorted, mean-centered within-group vectors, so it
    also cannot encode the means or the input group order.
    """

    expected_groups = [str(group).strip() for group in expected_blinded_groups]
    if not expected_groups or any(not group for group in expected_groups):
        raise ValueError("the frozen blinded group list must be nonempty")
    if len(set(expected_groups)) != len(expected_groups):
        raise ValueError("the frozen blinded group list contains duplicates")
    if set(pilot_by_blinded_group) != set(expected_groups):
        raise ValueError("variance pilot does not exactly match the frozen blinded groups")
    if not pilot_by_blinded_group:
        raise ValueError("the blinded variance pilot contains no groups")
    seed_sets: set[tuple[int, ...]] = set()
    centered_groups: list[list[float]] = []
    standard_deviations: list[float] = []
    for blinded_group, observations in pilot_by_blinded_group.items():
        if not str(blinded_group).strip():
            raise ValueError("every variance-pilot group needs a nonempty blinded label")
        if len(observations) != VARIANCE_PILOT_SEEDS:
            raise ValueError(
                f"each blinded group must contain exactly {VARIANCE_PILOT_SEEDS} seeds"
            )
        seeds = tuple(sorted(observations))
        seed_sets.add(seeds)
        values = [float(observations[seed]) for seed in seeds]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("variance-pilot normalized AURC values must be finite in [0, 1]")
        mean = math.fsum(values) / len(values)
        centered = [value - mean for value in values]
        centered_groups.append(centered)
        standard_deviations.append(_sample_sd(values))
    if len(seed_sets) != 1:
        raise ValueError("all blinded groups must use the same eight environment seeds")
    [common_environment_seeds] = seed_sets

    s_max = max(standard_deviations)
    selected_n = MAX_BASELINE_RUNS
    for candidate in range(MIN_BASELINE_RUNS, MAX_BASELINE_RUNS + 1):
        half_width = student_t_975(candidate - 1) * s_max / math.sqrt(candidate)
        if half_width <= TARGET_HALF_WIDTH:
            selected_n = candidate
            break
    achieved_half_width = (
        student_t_975(selected_n - 1) * s_max / math.sqrt(selected_n)
    )
    centered_payload = sorted(
        [[format(value, ".17g") for value in group] for group in centered_groups]
    )
    centered_sha256 = hashlib.sha256(
        json.dumps(centered_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    seed_sha256 = hashlib.sha256(
        json.dumps(common_environment_seeds, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "selection_rule": "smallest_n_with_t95_half_width_at_most_0.05_capped_20_40",
        "identity_masked": True,
        "group_count": len(pilot_by_blinded_group),
        "pilot_runs_per_group": VARIANCE_PILOT_SEEDS,
        "common_environment_seed_count": VARIANCE_PILOT_SEEDS,
        "common_environment_seeds_sha256": seed_sha256,
        "target_half_width": TARGET_HALF_WIDTH,
        "s_max": s_max,
        "selected_common_run_count": selected_n,
        "t_critical": student_t_975(selected_n - 1),
        "achieved_half_width": achieved_half_width,
        "capped_at_maximum": (
            selected_n == MAX_BASELINE_RUNS and achieved_half_width > TARGET_HALF_WIDTH
        ),
        "location_removed_masked_input_sha256": centered_sha256,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return float("nan")
    probability = min(1.0, max(0.0, probability))
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def bca_interval(
    values: Sequence[float],
    statistic,
    *,
    replicates: int = DEFAULT_BCA_REPLICATES,
    seed: int = 20260802,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Deterministic nonparametric BCa interval over independent runs."""

    numeric = [float(value) for value in values]
    if len(numeric) < 2:
        return float("nan"), float("nan")
    if any(not math.isfinite(value) for value in numeric):
        raise ValueError("BCa input values must be finite")
    if replicates < 200:
        raise ValueError("BCa intervals require at least 200 bootstrap replicates")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if min(numeric) == max(numeric):
        return numeric[0], numeric[0]

    observed = float(statistic(numeric))
    if not math.isfinite(observed):
        raise ValueError("BCa statistic must be finite")
    rng = random.Random(seed)
    bootstrap = sorted(
        float(statistic([numeric[rng.randrange(len(numeric))] for _ in numeric]))
        for _ in range(replicates)
    )
    if any(not math.isfinite(value) for value in bootstrap):
        raise ValueError("BCa bootstrap statistic returned a nonfinite value")

    less = sum(value < observed for value in bootstrap)
    equal = sum(value == observed for value in bootstrap)
    bias_probability = (less + 0.5 * equal) / replicates
    epsilon = 0.5 / replicates
    bias_probability = min(1.0 - epsilon, max(epsilon, bias_probability))
    normal = NormalDist()
    z0 = normal.inv_cdf(bias_probability)

    jackknife = [
        float(statistic(numeric[:index] + numeric[index + 1 :]))
        for index in range(len(numeric))
    ]
    if any(not math.isfinite(value) for value in jackknife):
        raise ValueError("BCa jackknife statistic returned a nonfinite value")
    jackknife_mean = sum(jackknife) / len(jackknife)
    deltas = [jackknife_mean - estimate for estimate in jackknife]
    denominator = 6.0 * sum(delta * delta for delta in deltas) ** 1.5
    acceleration = (
        sum(delta**3 for delta in deltas) / denominator if denominator else 0.0
    )

    def adjusted_probability(probability: float) -> float:
        z_alpha = normal.inv_cdf(probability)
        adjusted_denominator = 1.0 - acceleration * (z0 + z_alpha)
        if adjusted_denominator == 0.0:
            return 0.0 if z0 + z_alpha < 0 else 1.0
        return normal.cdf(z0 + (z0 + z_alpha) / adjusted_denominator)

    low_probability = adjusted_probability(alpha / 2.0)
    high_probability = adjusted_probability(1.0 - alpha / 2.0)
    if not (
        math.isfinite(low_probability)
        and math.isfinite(high_probability)
        and low_probability <= high_probability
    ):
        raise ValueError("BCa adjusted quantiles are not ordered finite probabilities")
    return _quantile(bootstrap, low_probability), _quantile(
        bootstrap, high_probability
    )


def bca_mean_interval(
    values: Sequence[float],
    *,
    replicates: int = DEFAULT_BCA_REPLICATES,
    seed: int = 20260802,
) -> tuple[float, float]:
    """Convenience BCa interval for a run-level arithmetic mean."""

    return bca_interval(
        values,
        lambda sample: sum(sample) / len(sample),
        replicates=replicates,
        seed=seed,
    )


def resolution_v_half_fraction() -> list[dict[str, object]]:
    """Return the prespecified 2^(5-1) design with defining relation I=ABCDE."""

    cells: list[dict[str, object]] = []
    for first_four in itertools.product((-1, 1), repeat=4):
        signs = (*first_four, math.prod(first_four))
        coded = dict(zip(SENSITIVITY_FACTORS, signs, strict=True))
        levels = {
            factor: SENSITIVITY_LEVELS[factor][int((sign + 1) / 2)]
            for factor, sign in coded.items()
        }
        cell_id = "p2sens_" + "_".join(
            f"{factor}={'hi' if coded[factor] == 1 else 'lo'}"
            for factor in SENSITIVITY_FACTORS
        )
        cells.append(
            {
                "cell_id": cell_id,
                **levels,
                "resource_capacity": int(
                    levels["capacity_per_initial_agent"] * levels["society_size"]
                ),
                "coded_levels": coded,
                "defining_relation": "I=ABCDE",
                "resolution": 5,
            }
        )
    return cells


def _signed_t(values: Sequence[float]) -> float:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    numerical_scale = max(1.0, *(abs(value) for value in values))
    if abs(mean) <= 1e-15 * numerical_scale and variance <= (
        1e-15 * numerical_scale
    ) ** 2:
        return 0.0
    if variance == 0.0:
        if mean == 0.0:
            return 0.0
        return math.copysign(float("inf"), mean)
    return mean / math.sqrt(variance / len(values))


def _holm_adjust(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, probability) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * probability))
        adjusted[name] = running
    return adjusted


def analyze_resolution_v_main_effects(
    observations: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Analyze five blocked main effects with exact sign-flip Holm/max-T tests.

    Exactly one normalized-AURC observation is required for every one of the 16
    cells under each of twelve common environment seeds.  Anything else is an
    incomplete or off-protocol design and is rejected before calculation.
    """

    design = resolution_v_half_fraction()
    expected = {str(cell["cell_id"]): cell for cell in design}
    by_cell_seed: dict[tuple[str, int], float] = {}
    seeds_by_cell: dict[str, set[int]] = defaultdict(set)
    for row in observations:
        cell_id = str(row.get("cell_id", ""))
        if cell_id not in expected:
            raise ValueError(f"unrecognized sensitivity cell_id {cell_id!r}")
        cell = expected[cell_id]
        for factor in SENSITIVITY_FACTORS:
            if factor not in row or float(row[factor]) != float(cell[factor]):
                raise ValueError(
                    f"sensitivity cell {cell_id} has an incorrect {factor} level"
                )
        if "resource_capacity" in row and int(row["resource_capacity"]) != int(
            cell["resource_capacity"]
        ):
            raise ValueError(f"sensitivity cell {cell_id} has an incorrect resource capacity")
        try:
            seed = int(row["environment_seed"])
            value = float(row["normalized_aurc"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "sensitivity observations require environment_seed and normalized_aurc"
            ) from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("sensitivity normalized AURC must be finite in [0, 1]")
        key = (cell_id, seed)
        if key in by_cell_seed:
            raise ValueError(f"duplicate sensitivity cell/seed observation: {key}")
        by_cell_seed[key] = value
        seeds_by_cell[cell_id].add(seed)

    if set(seeds_by_cell) != set(expected):
        raise ValueError("sensitivity design is incomplete: all 16 cells are required")
    seed_sets = {tuple(sorted(seeds)) for seeds in seeds_by_cell.values()}
    if len(seed_sets) != 1:
        raise ValueError(
            f"sensitivity cells must use {SENSITIVITY_SEEDS_PER_CELL} common "
            "environment seeds"
        )
    [common_seeds] = seed_sets
    if len(common_seeds) != SENSITIVITY_SEEDS_PER_CELL:
        raise ValueError(
            f"each sensitivity cell must use exactly {SENSITIVITY_SEEDS_PER_CELL} seeds"
        )
    if len(by_cell_seed) != SENSITIVITY_CELL_COUNT * SENSITIVITY_SEEDS_PER_CELL:
        raise ValueError("sensitivity design has an incorrect observation count")

    effects_by_factor: dict[str, list[float]] = {}
    for factor in SENSITIVITY_FACTORS:
        seed_effects: list[float] = []
        for seed in common_seeds:
            high = [
                by_cell_seed[(cell_id, seed)]
                for cell_id, cell in expected.items()
                if cell["coded_levels"][factor] == 1
            ]
            low = [
                by_cell_seed[(cell_id, seed)]
                for cell_id, cell in expected.items()
                if cell["coded_levels"][factor] == -1
            ]
            seed_effects.append(
                math.fsum(high) / len(high) - math.fsum(low) / len(low)
            )
        effects_by_factor[factor] = seed_effects

    observed_t = {
        factor: abs(_signed_t(seed_effects))
        for factor, seed_effects in effects_by_factor.items()
    }
    raw_exceedances = {factor: 0 for factor in SENSITIVITY_FACTORS}
    max_exceedances = {factor: 0 for factor in SENSITIVITY_FACTORS}
    permutations = list(itertools.product((-1, 1), repeat=len(common_seeds)))
    for signs in permutations:
        permuted_t = {
            factor: abs(
                _signed_t(
                    [effect * sign for effect, sign in zip(seed_effects, signs, strict=True)]
                )
            )
            for factor, seed_effects in effects_by_factor.items()
        }
        maximum = max(permuted_t.values())
        for factor in SENSITIVITY_FACTORS:
            if permuted_t[factor] >= observed_t[factor] - 1e-15:
                raw_exceedances[factor] += 1
            if maximum >= observed_t[factor] - 1e-15:
                max_exceedances[factor] += 1
    permutation_count = len(permutations)
    raw_p = {
        factor: raw_exceedances[factor] / permutation_count
        for factor in SENSITIVITY_FACTORS
    }
    holm_p = _holm_adjust(raw_p)

    rows: list[dict[str, object]] = []
    for factor in SENSITIVITY_FACTORS:
        seed_effects = effects_by_factor[factor]
        effect = math.fsum(seed_effects) / len(seed_effects)
        sd = _sample_sd(seed_effects)
        rows.append(
            {
                "factor": factor,
                "low_level": SENSITIVITY_LEVELS[factor][0],
                "high_level": SENSITIVITY_LEVELS[factor][1],
                "effect_high_minus_low": effect,
                "standard_error": sd / math.sqrt(len(seed_effects)),
                "t_statistic": _signed_t(seed_effects),
                "raw_exact_p": raw_p[factor],
                "holm_adjusted_p": holm_p[factor],
                "max_t_adjusted_p": max_exceedances[factor] / permutation_count,
                "cell_count": SENSITIVITY_CELL_COUNT,
                "common_seed_count": SENSITIVITY_SEEDS_PER_CELL,
                "permutation_count": permutation_count,
                "design": "2^(5-1)_resolution_V_I=ABCDE",
                "analysis_unit": "environment_seed_block",
            }
        )
    return rows


def analyze_sentinel_sensitivity(
    observations_by_sentinel: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    expected_sentinel_ids: Sequence[str],
) -> list[dict[str, object]]:
    """Analyze six sentinels with one global 30-hypothesis Holm family.

    The five-factor Holm and max-T values produced for an individual sentinel
    remain useful diagnostics, but they do not control selection across all six
    sentinels.  This wrapper therefore makes the global Holm value the published
    ``holm_adjusted_p`` and retains the narrower adjustments under explicitly
    scoped field names.
    """

    expected = [str(sentinel).strip() for sentinel in expected_sentinel_ids]
    if len(expected) != SENSITIVITY_SENTINEL_COUNT or any(
        not sentinel for sentinel in expected
    ):
        raise ValueError("the frozen sensitivity list must contain exactly 6 sentinels")
    if len(set(expected)) != len(expected):
        raise ValueError("the frozen sensitivity sentinel list contains duplicates")
    if len(observations_by_sentinel) != SENSITIVITY_SENTINEL_COUNT:
        raise ValueError(
            f"sensitivity requires exactly {SENSITIVITY_SENTINEL_COUNT} sentinel systems"
        )
    if set(observations_by_sentinel) != set(expected):
        raise ValueError("sensitivity observations do not match the frozen sentinels")
    output: list[dict[str, object]] = []
    sentinel_seed_sets: set[tuple[int, ...]] = set()
    for sentinel_id, observations in sorted(observations_by_sentinel.items()):
        if not str(sentinel_id).strip():
            raise ValueError("sentinel identifiers must be nonempty")
        seeds = tuple(sorted({int(row["environment_seed"]) for row in observations}))
        sentinel_seed_sets.add(seeds)
        for row in analyze_resolution_v_main_effects(observations):
            output.append({"sentinel_id": sentinel_id, **row})
    if len(sentinel_seed_sets) != 1:
        raise ValueError("all six sentinels must use the same common environment seeds")

    raw_global = {
        f"{row['sentinel_id']}|{row['factor']}": float(row["raw_exact_p"])
        for row in output
    }
    if len(raw_global) != SENSITIVITY_SENTINEL_COUNT * len(SENSITIVITY_FACTORS):
        raise ValueError("sensitivity analysis does not contain exactly 30 hypotheses")
    global_holm = _holm_adjust(raw_global)
    for row in output:
        key = f"{row['sentinel_id']}|{row['factor']}"
        row["within_sentinel_holm_adjusted_p"] = row["holm_adjusted_p"]
        row["within_sentinel_max_t_adjusted_p"] = row["max_t_adjusted_p"]
        row["holm_adjusted_p"] = global_holm[key]
        row["holm_family"] = "30_prespecified_sentinel_by_factor_main_effects"
        row["holm_family_size"] = len(raw_global)
        row["max_t_family"] = "five_main_effects_within_sentinel_diagnostic"
    return output


CLI_SCHEMA_VERSION = 1
_SENSITIVITY_OBSERVATION_FIELDS = frozenset(
    {
        "sentinel_id",
        "cell_id",
        *SENSITIVITY_FACTORS,
        "resource_capacity",
        "environment_seed",
        "normalized_aurc",
    }
)
_STRUCTURAL_CELL_INPUT_FIELDS = frozenset(
    {
        "provider",
        "model",
        "society_size",
        "horizon_days",
        "resource",
        "resource_capacity",
        "selfish_gain",
        "depletion_units",
        "community_benefit",
        "collapse_death_rate",
    }
)
_SENSITIVITY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "design",
        "frozen_sentinel_ids",
        "common_environment_seeds",
        "cells",
        "cell_count",
        "seeds_per_cell",
        "sentinel_count",
        "expected_observations_per_sentinel",
        "expected_observation_count",
        "manifest_sha256",
    }
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON number {value} is not allowed")


def _load_json_object(path: Path) -> tuple[dict[str, object], str]:
    raw = path.read_bytes()
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON input: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("JSON input root must be an object")
    return parsed, hashlib.sha256(raw).hexdigest()


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], location: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ValueError(f"{location} has an invalid schema: {'; '.join(details)}")


def _require_schema_version(value: Mapping[str, object], location: str) -> None:
    version = value.get("schema_version")
    if isinstance(version, bool) or version != CLI_SCHEMA_VERSION:
        raise ValueError(f"{location}.schema_version must equal {CLI_SCHEMA_VERSION}")


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sealed_artifact(
    payload: Mapping[str, object], *, hash_field: str = "artifact_sha256"
) -> dict[str, object]:
    safe_payload = _json_safe(dict(payload))
    assert isinstance(safe_payload, dict)
    digest = hashlib.sha256(_canonical_json_bytes(safe_payload)).hexdigest()
    return {**safe_payload, hash_field: digest}


def _verify_sealed_artifact(
    artifact: Mapping[str, object], *, hash_field: str
) -> str:
    recorded = artifact.get(hash_field)
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(f"artifact is missing a valid {hash_field}")
    payload = dict(artifact)
    del payload[hash_field]
    expected = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if recorded != expected:
        raise ValueError(f"artifact {hash_field} verification failed")
    return recorded


def _atomic_write_fresh_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically introduce a mode-0600 JSON file without overwriting a target."""

    parent = path.parent
    if not parent.is_dir():
        raise ValueError(f"output parent directory does not exist: {parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    temporary = parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        data = _canonical_json_bytes(value)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # link() atomically creates the final name and fails if it appeared
        # after the preflight existence check; unlike replace(), it never
        # overwrites a concurrent writer's output.
        os.link(temporary, path)
        temporary.unlink()
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _require_private_input(path: Path) -> None:
    """Require private permissions for any outcome-bearing analysis input."""

    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise ValueError("analysis input must have private permissions (mode 0600)")


def _parse_blinded_variance_input(
    document: Mapping[str, object],
) -> tuple[str, list[str], dict[str, dict[int, float]]]:
    _require_exact_keys(
        document,
        frozenset(
            {
                "schema_version",
                "artifact_type",
                "artifact_sha256",
                "pilot_campaign_path",
                "pilot_campaign_manifest_sha256",
                "pilot_campaign_payload_sha256",
                "native_artifact_set_sha256",
                "identity_masking_scheme",
                "frozen_blinded_group_ids",
                "pilot_by_blinded_group",
            }
        ),
        "variance input",
    )
    _require_schema_version(document, "variance input")
    if document["artifact_type"] != "part2_native_identity_masked_variance_input":
        raise ValueError("variance input is not a native identity-masked campaign adapter artifact")
    _verify_sealed_artifact(document, hash_field="artifact_sha256")
    if document["identity_masking_scheme"] != "automated_plan_bound_location_insensitive_v1":
        raise ValueError("variance input uses an unsupported identity-masking scheme")
    campaign_path = document["pilot_campaign_path"]
    if not isinstance(campaign_path, str) or not Path(campaign_path).is_absolute():
        raise ValueError("pilot_campaign_path must be an absolute path")
    campaign_manifest_sha256 = document["pilot_campaign_manifest_sha256"]
    if (
        not isinstance(campaign_manifest_sha256, str)
        or len(campaign_manifest_sha256) != 64
        or any(character not in "0123456789abcdef" for character in campaign_manifest_sha256)
    ):
        raise ValueError(
            "pilot_campaign_manifest_sha256 must be a lowercase 64-character SHA-256"
        )
    for name in ("pilot_campaign_payload_sha256", "native_artifact_set_sha256"):
        digest = document[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    frozen = document["frozen_blinded_group_ids"]
    groups = document["pilot_by_blinded_group"]
    if not isinstance(frozen, list) or not frozen:
        raise ValueError("frozen_blinded_group_ids must be a nonempty array")
    if any(
        not isinstance(group, str) or not group or group != group.strip()
        for group in frozen
    ):
        raise ValueError("frozen blinded group IDs must be nonempty trimmed strings")
    if len(set(frozen)) != len(frozen):
        raise ValueError("frozen blinded group IDs must be unique")
    if not isinstance(groups, dict) or set(groups) != set(frozen):
        raise ValueError("pilot groups do not exactly match the frozen blinded groups")

    parsed: dict[str, dict[int, float]] = {}
    for group in frozen:
        observations = groups[group]
        if not isinstance(observations, dict) or len(observations) != VARIANCE_PILOT_SEEDS:
            raise ValueError(
                f"each pilot group must contain exactly {VARIANCE_PILOT_SEEDS} seed values"
            )
        parsed_observations: dict[int, float] = {}
        for seed_text, value in observations.items():
            if not isinstance(seed_text, str):
                raise ValueError("variance-pilot seed keys must be canonical integer strings")
            try:
                seed = int(seed_text)
            except ValueError as exc:
                raise ValueError(
                    "variance-pilot seed keys must be canonical integer strings"
                ) from exc
            if str(seed) != seed_text:
                raise ValueError("variance-pilot seed keys must be canonical integer strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("variance-pilot AURC values must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError("variance-pilot AURC values must be finite in [0, 1]")
            parsed_observations[seed] = numeric
        parsed[group] = parsed_observations
    return campaign_manifest_sha256, list(frozen), parsed


def _read_part2_aurc(
    csv_path: Path, *, expected_environment_seed: int, expected_generation_seed: int
) -> float:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"could not read native Part 2 CSV: {csv_path}") from error
    if not rows:
        raise ValueError("native Part 2 variance-pilot CSV is empty")
    cell = load_part2_structural_cell(csv_path, rows)
    identity = load_part2_run_identity(csv_path, rows)
    if not identity.strict_schema:
        raise ValueError("native Part 2 variance pilot must use the strict run schema")
    if (
        identity.environment_seed != expected_environment_seed
        or identity.generation_seed != expected_generation_seed
    ):
        raise ValueError("native Part 2 CSV seeds differ from the frozen pilot job")
    day_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            day_rows[int(row.get("day", ""))].append(row)
        except (TypeError, ValueError):
            raise ValueError("native Part 2 CSV contains a noninteger day") from None
    aurc, _aupc = normalized_part2_auc(
        day_rows,
        horizon=cell.horizon_days,
        society_size=cell.society_size,
        resource_capacity=cell.resource_capacity,
    )
    if not math.isfinite(aurc) or not 0.0 <= aurc <= 1.0:
        raise ValueError("native Part 2 trajectory has a nonestimable normalized AURC")
    return aurc


def _derive_native_blinded_variance_input(
    pilot_campaign_path: Path, *, expected_sha256: str
) -> dict[str, object]:
    """Revalidate archived native pilot lineage and derive its blinded AURCs."""

    from experiments import confirmatory_campaign

    resolved = pilot_campaign_path.resolve()
    if not resolved.is_file():
        raise ValueError(f"pilot campaign manifest does not exist: {resolved}")
    actual_sha256 = confirmatory_campaign.sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "pilot campaign manifest hash mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    try:
        pilot = confirmatory_campaign._validate_completed_variance_pilot_manifest(
            resolved, expected_sha256=expected_sha256
        )
    except Exception as error:
        raise ValueError(f"native variance-pilot campaign validation failed: {error}") from error
    plan_sha256 = pilot.get("plan_sha256")
    payload_sha256 = pilot.get("manifest_sha256")
    if not isinstance(plan_sha256, str) or not isinstance(payload_sha256, str):
        raise ValueError("pilot campaign lacks immutable plan/payload hashes")
    targets = [str(target["id"]) for target in pilot["targets"]]
    blind_by_target = {
        target_id: "blind-"
        + hashlib.sha256(
            f"part2-variance-blind-v1|{plan_sha256}|{target_id}".encode("utf-8")
        ).hexdigest()[:24]
        for target_id in targets
    }
    if len(set(blind_by_target.values())) != len(targets):
        raise ValueError("blinded target pseudonym collision")
    by_blind: dict[str, dict[str, float]] = {
        blind_by_target[target_id]: {} for target_id in targets
    }
    native_refs: list[dict[str, object]] = []
    scientific_jobs = [
        job
        for job in pilot["jobs"]
        if job["experiment"] == "part2" and job["stage"] == "part2_variance_pilot"
    ]
    for job in scientific_jobs:
        target_id = str(job["target_id"])
        expected = job["expected"]
        environment_seed = expected.get("environment_seed")
        generation_seed = expected.get("generation_seed")
        if (
            isinstance(environment_seed, bool)
            or not isinstance(environment_seed, int)
            or isinstance(generation_seed, bool)
            or not isinstance(generation_seed, int)
        ):
            raise ValueError("native variance-pilot job has invalid frozen seeds")
        seed_offset = environment_seed - confirmatory_campaign.DEFAULT_PART2_ENVIRONMENT_SEED_BASE
        if (
            seed_offset not in range(VARIANCE_PILOT_SEEDS)
            or generation_seed
            != confirmatory_campaign.DEFAULT_PART2_GENERATION_SEED_BASE + seed_offset
        ):
            raise ValueError("native variance-pilot jobs do not use the exact common seed panel")
        verified = confirmatory_campaign.resolve_job_artifact(job, None)
        csv_path = (confirmatory_campaign.REPO_ROOT / str(verified["csv_path"])).resolve()
        metadata_path = (
            confirmatory_campaign.REPO_ROOT / str(verified["metadata_path"])
        ).resolve()
        seed_key = str(environment_seed)
        blind_id = blind_by_target[target_id]
        if seed_key in by_blind[blind_id]:
            raise ValueError("native variance-pilot campaign duplicates a target/seed")
        by_blind[blind_id][seed_key] = _read_part2_aurc(
            csv_path,
            expected_environment_seed=environment_seed,
            expected_generation_seed=generation_seed,
        )
        native_refs.append(
            {
                "job_id": job["id"],
                "blinded_group_id": blind_id,
                "environment_seed": environment_seed,
                "generation_seed": generation_seed,
                "csv_path": str(csv_path),
                "csv_sha256": confirmatory_campaign.sha256_file(csv_path),
                "metadata_path": str(metadata_path),
                "metadata_sha256": confirmatory_campaign.sha256_file(metadata_path),
            }
        )
    expected_seeds = {
        str(confirmatory_campaign.DEFAULT_PART2_ENVIRONMENT_SEED_BASE + offset)
        for offset in range(VARIANCE_PILOT_SEEDS)
    }
    if any(set(seed_values) != expected_seeds for seed_values in by_blind.values()):
        raise ValueError("native variance pilot lacks the exact common eight-seed panel")
    native_refs.sort(key=lambda row: str(row["job_id"]))
    native_artifact_set_sha256 = hashlib.sha256(
        _canonical_json_bytes(native_refs)
    ).hexdigest()
    return _sealed_artifact(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "artifact_type": "part2_native_identity_masked_variance_input",
            "pilot_campaign_path": str(resolved),
            "pilot_campaign_manifest_sha256": expected_sha256,
            "pilot_campaign_payload_sha256": payload_sha256,
            "native_artifact_set_sha256": native_artifact_set_sha256,
            "identity_masking_scheme": "automated_plan_bound_location_insensitive_v1",
            "frozen_blinded_group_ids": sorted(by_blind),
            "pilot_by_blinded_group": {
                blind_id: dict(sorted(values.items(), key=lambda item: int(item[0])))
                for blind_id, values in sorted(by_blind.items())
            },
        }
    )


def _validated_identifier_list(
    values: Sequence[str], *, expected_count: int, name: str
) -> list[str]:
    normalized = [str(value).strip() for value in values]
    if len(normalized) != expected_count or any(not value for value in normalized):
        raise ValueError(f"{name} requires exactly {expected_count} nonempty values")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique")
    return normalized


def _validated_seed_list(
    values: Sequence[int], *, expected_count: int | None = None
) -> list[int]:
    seeds = list(values)
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("environment seeds must be integers")
    if expected_count is not None and len(seeds) != expected_count:
        raise ValueError(f"exactly {expected_count} environment seeds are required")
    if not seeds:
        raise ValueError("at least one environment seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("environment seeds must be unique")
    return seeds


def _build_sensitivity_manifest(
    sentinel_ids: Sequence[str], environment_seeds: Sequence[int]
) -> dict[str, object]:
    sentinels = _validated_identifier_list(
        sentinel_ids,
        expected_count=SENSITIVITY_SENTINEL_COUNT,
        name="sentinel IDs",
    )
    seeds = _validated_seed_list(
        environment_seeds,
        expected_count=SENSITIVITY_SEEDS_PER_CELL,
    )
    cells = resolution_v_half_fraction()
    payload = {
        "schema_version": CLI_SCHEMA_VERSION,
        "artifact_type": "part2_resolution_v_sensitivity_design",
        "design": "2^(5-1)_resolution_V_I=ABCDE",
        "frozen_sentinel_ids": sentinels,
        "common_environment_seeds": seeds,
        "cells": cells,
        "cell_count": SENSITIVITY_CELL_COUNT,
        "seeds_per_cell": SENSITIVITY_SEEDS_PER_CELL,
        "sentinel_count": SENSITIVITY_SENTINEL_COUNT,
        "expected_observations_per_sentinel": (
            SENSITIVITY_CELL_COUNT * SENSITIVITY_SEEDS_PER_CELL
        ),
        "expected_observation_count": (
            SENSITIVITY_SENTINEL_COUNT
            * SENSITIVITY_CELL_COUNT
            * SENSITIVITY_SEEDS_PER_CELL
        ),
    }
    return _sealed_artifact(payload, hash_field="manifest_sha256")


def _validate_sensitivity_manifest(
    manifest: Mapping[str, object],
) -> tuple[str, list[str], list[int]]:
    _require_exact_keys(manifest, _SENSITIVITY_MANIFEST_FIELDS, "design manifest")
    _require_schema_version(manifest, "design manifest")
    if manifest["artifact_type"] != "part2_resolution_v_sensitivity_design":
        raise ValueError("design manifest has an incorrect artifact_type")
    manifest_sha256 = _verify_sealed_artifact(manifest, hash_field="manifest_sha256")
    expected_cells = resolution_v_half_fraction()
    if manifest["design"] != "2^(5-1)_resolution_V_I=ABCDE":
        raise ValueError("design manifest has an incorrect defining relation")
    if manifest["cells"] != expected_cells:
        raise ValueError("design manifest cells do not match the prespecified design")
    expected_constants = {
        "cell_count": SENSITIVITY_CELL_COUNT,
        "seeds_per_cell": SENSITIVITY_SEEDS_PER_CELL,
        "sentinel_count": SENSITIVITY_SENTINEL_COUNT,
        "expected_observations_per_sentinel": (
            SENSITIVITY_CELL_COUNT * SENSITIVITY_SEEDS_PER_CELL
        ),
        "expected_observation_count": (
            SENSITIVITY_SENTINEL_COUNT
            * SENSITIVITY_CELL_COUNT
            * SENSITIVITY_SEEDS_PER_CELL
        ),
    }
    if any(manifest.get(key) != value for key, value in expected_constants.items()):
        raise ValueError("design manifest contains incorrect design counts")
    sentinel_value = manifest["frozen_sentinel_ids"]
    seed_value = manifest["common_environment_seeds"]
    if not isinstance(sentinel_value, list):
        raise ValueError("design manifest frozen_sentinel_ids must be an array")
    if not isinstance(seed_value, list):
        raise ValueError("design manifest common_environment_seeds must be an array")
    sentinels = _validated_identifier_list(
        sentinel_value,
        expected_count=SENSITIVITY_SENTINEL_COUNT,
        name="manifest sentinel IDs",
    )
    seeds = _validated_seed_list(
        seed_value,
        expected_count=SENSITIVITY_SEEDS_PER_CELL,
    )
    return manifest_sha256, sentinels, seeds


def _parse_sensitivity_observations(
    document: Mapping[str, object],
    *,
    manifest_sha256: str,
    sentinel_ids: Sequence[str],
    environment_seeds: Sequence[int],
) -> dict[str, list[dict[str, object]]]:
    _require_exact_keys(
        document,
        frozenset(
            {
                "schema_version",
                "design_manifest_sha256",
                "observations",
            }
        ),
        "sensitivity input",
    )
    _require_schema_version(document, "sensitivity input")
    if document["design_manifest_sha256"] != manifest_sha256:
        raise ValueError("sensitivity input does not pin the supplied design manifest")
    observations = document["observations"]
    expected_count = (
        SENSITIVITY_SENTINEL_COUNT
        * SENSITIVITY_CELL_COUNT
        * SENSITIVITY_SEEDS_PER_CELL
    )
    if not isinstance(observations, list) or len(observations) != expected_count:
        raise ValueError(
            f"sensitivity input requires exactly {expected_count} run-level observations"
        )
    grouped: dict[str, list[dict[str, object]]] = {
        sentinel: [] for sentinel in sentinel_ids
    }
    allowed_seeds = set(environment_seeds)
    for index, row in enumerate(observations):
        if not isinstance(row, dict):
            raise ValueError(f"sensitivity observation {index} must be an object")
        _require_exact_keys(row, _SENSITIVITY_OBSERVATION_FIELDS, "sensitivity observation")
        sentinel_id = row["sentinel_id"]
        if not isinstance(sentinel_id, str) or sentinel_id not in grouped:
            raise ValueError("sensitivity observation has an unfrozen sentinel_id")
        seed = row["environment_seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in allowed_seeds:
            raise ValueError("sensitivity observation has an unplanned environment seed")
        value = row["normalized_aurc"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("sensitivity normalized_aurc must be numeric")
        grouped[sentinel_id].append(dict(row))
    return grouped


def _parse_structural_cell(document: Mapping[str, object]) -> Part2StructuralCell:
    _require_exact_keys(
        document,
        frozenset({"schema_version", "structural_cell"}),
        "baseline input",
    )
    _require_schema_version(document, "baseline input")
    value = document["structural_cell"]
    if not isinstance(value, dict):
        raise ValueError("baseline structural_cell must be an object")
    _require_exact_keys(value, _STRUCTURAL_CELL_INPUT_FIELDS, "structural_cell")
    for field in ("provider", "model", "resource"):
        if not isinstance(value[field], str) or not str(value[field]).strip():
            raise ValueError(f"structural_cell.{field} must be a nonempty string")
    for field in (
        "society_size",
        "horizon_days",
        "resource_capacity",
        "selfish_gain",
        "depletion_units",
        "community_benefit",
    ):
        if isinstance(value[field], bool) or not isinstance(value[field], int):
            raise ValueError(f"structural_cell.{field} must be an integer")
    for field in ("society_size", "horizon_days", "resource_capacity", "depletion_units"):
        if int(value[field]) <= 0:
            raise ValueError(f"structural_cell.{field} must be positive")
    death_rate = value["collapse_death_rate"]
    if isinstance(death_rate, bool) or not isinstance(death_rate, (int, float)):
        raise ValueError("structural_cell.collapse_death_rate must be numeric")
    death_rate = float(death_rate)
    if not math.isfinite(death_rate) or not 0.0 < death_rate <= 1.0:
        raise ValueError("structural_cell.collapse_death_rate must be in (0, 1]")
    return Part2StructuralCell(
        provider=str(value["provider"]),
        model=str(value["model"]),
        society_size=int(value["society_size"]),
        horizon_days=int(value["horizon_days"]),
        resource=str(value["resource"]),
        resource_capacity=int(value["resource_capacity"]),
        selfish_gain=int(value["selfish_gain"]),
        depletion_units=int(value["depletion_units"]),
        community_benefit=int(value["community_benefit"]),
        collapse_death_rate=death_rate,
    )


def _run_sensitivity_design(args: argparse.Namespace) -> dict[str, object]:
    manifest = _build_sensitivity_manifest(args.sentinel_id, args.environment_seed)
    _atomic_write_fresh_json(args.output, manifest)
    return manifest


def _run_analyze_sensitivity(args: argparse.Namespace) -> dict[str, object]:
    manifest, _manifest_input_sha256 = _load_json_object(args.design_manifest)
    manifest_sha256, sentinel_ids, environment_seeds = _validate_sensitivity_manifest(
        manifest
    )
    document, observation_input_sha256 = _load_json_object(args.input)
    grouped = _parse_sensitivity_observations(
        document,
        manifest_sha256=manifest_sha256,
        sentinel_ids=sentinel_ids,
        environment_seeds=environment_seeds,
    )
    results = analyze_sentinel_sensitivity(
        grouped,
        expected_sentinel_ids=sentinel_ids,
    )
    artifact = _sealed_artifact(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "artifact_type": "part2_resolution_v_sensitivity_analysis",
            "design_manifest_sha256": manifest_sha256,
            "observation_input_sha256": observation_input_sha256,
            "observation_count": sum(len(rows) for rows in grouped.values()),
            "analysis_unit": "environment_seed_block",
            "main_effect_results": results,
        }
    )
    _atomic_write_fresh_json(args.output, artifact)
    return artifact


def _run_baselines(args: argparse.Namespace) -> dict[str, object]:
    document, input_sha256 = _load_json_object(args.input)
    cell = _parse_structural_cell(document)
    seeds = _validated_seed_list(args.environment_seed)
    baselines = no_call_baseline_suite(cell, environment_seeds=seeds)
    artifact = _sealed_artifact(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "artifact_type": "part2_no_call_baselines",
            "structural_cell_input_sha256": input_sha256,
            "structural_cell": {
                field: getattr(cell, field) for field in _STRUCTURAL_CELL_INPUT_FIELDS
            },
            "environment_seeds": seeds,
            "mechanical_survival_threshold": asdict(
                mechanical_survival_threshold(cell)
            ),
            "baselines": [asdict(baseline) for baseline in baselines],
        }
    )
    _atomic_write_fresh_json(args.output, artifact)
    return artifact


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fail-closed Part 2 confirmatory statistics and baselines."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    design = subparsers.add_parser(
        "sensitivity-design",
        help="write the frozen 16-cell resolution-V sensitivity manifest",
    )
    design.add_argument("--sentinel-id", required=True, action="append")
    design.add_argument(
        "--environment-seed", required=True, action="append", type=int
    )
    design.add_argument("--output", required=True, type=Path)
    design.set_defaults(handler=_run_sensitivity_design)

    sensitivity = subparsers.add_parser(
        "analyze-sensitivity",
        help="analyze completed observations against a hash-pinned design manifest",
    )
    sensitivity.add_argument("--input", required=True, type=Path)
    sensitivity.add_argument("--design-manifest", required=True, type=Path)
    sensitivity.add_argument("--output", required=True, type=Path)
    sensitivity.set_defaults(handler=_run_analyze_sensitivity)

    baselines = subparsers.add_parser(
        "baselines", help="emit deterministic no-call Part 2 baseline trajectories"
    )
    baselines.add_argument("--input", required=True, type=Path)
    baselines.add_argument(
        "--environment-seed", required=True, action="append", type=int
    )
    baselines.add_argument("--output", required=True, type=Path)
    baselines.set_defaults(handler=_run_baselines)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    try:
        artifact = args.handler(args)
    except Exception as exc:  # noqa: BLE001 - CLI must not print private tracebacks.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    hash_value = artifact.get("artifact_sha256", artifact.get("manifest_sha256", ""))
    print(f"wrote {args.output} sha256={hash_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
