from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from experiments.part0 import part_0
from experiments.part1 import part_1
from experiments.part2 import part_2
from analysis.model_metadata import ModelMetadata, resolve_model_metadata
from analysis.statistics import (
    fisher_z_interval,
    kendall_correlation,
    leave_group_out_range,
    pearson_correlation,
    rank_values,
    spearman_correlation,
)
from analysis.validation import _csv_paths, _flag_reasoning_misunderstanding

RAW_DIR = Path("data") / "raw"
TABLES_DIR = Path("data") / "analysis" / "tables"
PART0_LANGUAGE_ORDER = ("english", "chinese", "russian")
PART1_FACTORS = ("model", "frame", "game", "domain", "presentation")
PART1_SENSITIVITY_DIMENSIONS = ("frame", "game", "domain", "presentation")
PART1_FRAME_COMPARISONS = ("observer_evaluation", "self_direct", "advice")
PART1_FRAME_BASELINE = "prediction"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def _model_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("provider", "unknown"), row.get("model", "unknown")


def _part0_compliance_value(row: dict[str, str]) -> str:
    """Prefer a response-only rejudgment over the retained legacy label."""

    if "new_complied" in row:
        return row.get("new_complied", "")
    return row.get("complied?", "")


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5 / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
    return pearson_correlation(xs, ys)


def _rank_values(values: list[float]) -> list[float]:
    return rank_values(values)


def _spearman_correlation(xs: list[float], ys: list[float]) -> float:
    return spearman_correlation(xs, ys)


def _kendall_correlation(xs: list[float], ys: list[float]) -> float:
    return kendall_correlation(xs, ys)


def _fisher_z_interval(correlation: float, n: int) -> tuple[float, float]:
    return fisher_z_interval(correlation, n)


def _bootstrap_ci(
    xs: list[float],
    ys: list[float],
    statistic,
    *,
    samples: int = 2000,
    seed: int = 20260429,
) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    estimates: list[float] = []
    indices = list(range(len(xs)))
    for _ in range(samples):
        draw = [rng.choice(indices) for _ in indices]
        estimate = statistic([xs[i] for i in draw], [ys[i] for i in draw])
        if not math.isnan(estimate):
            estimates.append(estimate)
    if not estimates:
        return float("nan"), float("nan")
    estimates.sort()
    low_index = int(0.025 * (len(estimates) - 1))
    high_index = int(0.975 * (len(estimates) - 1))
    return estimates[low_index], estimates[high_index]


def _rate(counter: Counter[str], positive_key: str, negative_key: str) -> float:
    total = counter[positive_key] + counter[negative_key]
    return counter[positive_key] / total if total else float("nan")


def _part0_language_robustness_rows(
    language_counts: dict[tuple[str, str, str], Counter[str]],
) -> list[dict[str, object]]:
    by_model: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (provider, model, language), counter in language_counts.items():
        total = counter["denied"] + counter["complied"]
        if total:
            by_model[(provider, model)][language] = counter["denied"] / total

    rows: list[dict[str, object]] = []
    for (provider, model), rates_by_language in sorted(by_model.items()):
        rates = list(rates_by_language.values())
        min_language, min_rate = min(rates_by_language.items(), key=lambda item: item[1])
        max_language, max_rate = max(rates_by_language.items(), key=lambda item: item[1])
        row: dict[str, object] = {
            "provider": provider,
            "model": model,
            "languages_evaluated": len(rates),
            "mean_language_refusal_rate": sum(rates) / len(rates),
            "minimum_language_refusal_rate": min_rate,
            "minimum_language": min_language,
            "maximum_language_refusal_rate": max_rate,
            "maximum_language": max_language,
            "language_refusal_gap": max_rate - min_rate,
        }
        for language in PART0_LANGUAGE_ORDER:
            row[f"{language}_refusal_rate"] = rates_by_language.get(language, "")
        rows.append(row)
    return rows


def summarize_part0(raw_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    language_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    model_metadata: dict[tuple[str, str], ModelMetadata] = {}
    for path in _csv_paths(raw_dir / "part_0"):
        sidecar = _metadata_for_csv(path)
        for row in _read_rows(path):
            verdict = part_0._verdict_from_complied_value(_part0_compliance_value(row))
            if "new_complied" not in row and not verdict and row.get("verdict"):
                verdict = row["verdict"].strip().lower()
            key = _model_key(row)
            model_metadata[key] = resolve_model_metadata(key[0], key[1], sidecar)
            counts[key][verdict or "skipped"] += 1
            language = row.get("language", "").strip().lower()
            if language and verdict in {"denied", "complied"}:
                language_counts[(key[0], key[1], language)][verdict] += 1

    rows: list[dict[str, object]] = []
    for (provider, model), counter in sorted(counts.items()):
        denied = counter["denied"]
        complied = counter["complied"]
        skipped = counter["skipped"]
        total = denied + complied
        low, high = _wilson_interval(denied, total)
        grouping = model_metadata[(provider, model)]
        rows.append(
            {
                "provider": provider,
                "model": model,
                "family_id": grouping.family_id,
                "developer_id": grouping.developer_id,
                "cohort": grouping.cohort,
                "judged": total,
                "denied": denied,
                "complied": complied,
                "skipped": skipped,
                "safety_refusal_rate": denied / total if total else "",
                "wilson_low": low if total else "",
                "wilson_high": high if total else "",
            }
        )

    path = output_dir / "part0_model_summary.csv"
    robustness_path = output_dir / "part0_language_robustness.csv"
    _write_csv(
        path,
        [
            "provider",
            "model",
            "family_id",
            "developer_id",
            "cohort",
            "judged",
            "denied",
            "complied",
            "skipped",
            "safety_refusal_rate",
            "wilson_low",
            "wilson_high",
        ],
        rows,
    )
    _write_csv(
        robustness_path,
        [
            "provider",
            "model",
            "languages_evaluated",
            "mean_language_refusal_rate",
            "minimum_language_refusal_rate",
            "minimum_language",
            "maximum_language_refusal_rate",
            "maximum_language",
            "language_refusal_gap",
            "english_refusal_rate",
            "chinese_refusal_rate",
            "russian_refusal_rate",
        ],
        _part0_language_robustness_rows(language_counts),
    )
    return path, robustness_path


def _part1_is_cooperative(row: dict[str, str]) -> bool:
    return row.get("action", "") in {"COOPERATE", "RESTRAIN"}


def _part1_frame_effect_rows(
    observations: list[dict[str, object]],
    model_providers: dict[str, str],
) -> list[dict[str, object]]:
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    counts["ALL_MODELS"]
    for observation in observations:
        outcome = "cooperate" if observation["cooperate"] else "defect_or_overuse"
        frame = str(observation["frame"])
        model = str(observation["model"])
        counts[model][frame][outcome] += 1
        counts["ALL_MODELS"][frame][outcome] += 1

    rows: list[dict[str, object]] = []
    for model in ["ALL_MODELS", *sorted(model_providers)]:
        frame_rates = {
            frame: _rate(counter, "cooperate", "defect_or_overuse")
            for frame, counter in counts[model].items()
        }
        baseline_rate = frame_rates.get(PART1_FRAME_BASELINE, float("nan"))
        row: dict[str, object] = {
            "provider": "aggregate" if model == "ALL_MODELS" else model_providers[model],
            "model": model,
            "prediction_rate": baseline_rate,
        }
        for frame in PART1_FRAME_COMPARISONS:
            comparison_rate = frame_rates.get(frame, float("nan"))
            row[f"{frame}_rate"] = comparison_rate
            row[f"{frame}_minus_prediction_pp"] = (comparison_rate - baseline_rate) * 100.0
        rows.append(row)
    return rows


def _part1_prompt_sensitivity_rows(
    observations: list[dict[str, object]],
    model_providers: dict[str, str],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for observation in observations:
        outcome = "cooperate" if observation["cooperate"] else "defect_or_overuse"
        model = str(observation["model"])
        for dimension in PART1_SENSITIVITY_DIMENSIONS:
            grouped[(model, dimension, str(observation[dimension]))][outcome] += 1

    rows: list[dict[str, object]] = []
    for model in sorted(model_providers):
        sensitivity_by_dimension: dict[str, float] = {}
        for dimension in PART1_SENSITIVITY_DIMENSIONS:
            rates = [
                _rate(counter, "cooperate", "defect_or_overuse")
                for (group_model, group_dimension, _value), counter in grouped.items()
                if group_model == model and group_dimension == dimension
            ]
            sensitivity_by_dimension[dimension] = (max(rates) - min(rates)) * 100.0 if rates else float("nan")
        max_dimension, max_sensitivity = max(
            sensitivity_by_dimension.items(),
            key=lambda item: -1.0 if math.isnan(item[1]) else item[1],
        )
        rows.append(
            {
                "provider": model_providers[model],
                "model": model,
                "frame_sensitivity_pp": sensitivity_by_dimension["frame"],
                "game_sensitivity_pp": sensitivity_by_dimension["game"],
                "domain_sensitivity_pp": sensitivity_by_dimension["domain"],
                "presentation_sensitivity_pp": sensitivity_by_dimension["presentation"],
                "maximum_prompt_sensitivity_pp": max_sensitivity,
                "maximum_prompt_sensitivity_dimension": max_dimension,
            }
        )
    return rows


def _part1_factor_decomposition_rows(observations: list[dict[str, object]]) -> list[dict[str, object]]:
    outcomes = [float(observation["cooperate"]) for observation in observations]
    if not outcomes:
        return []

    grand_mean = sum(outcomes) / len(outcomes)
    total_ss = sum((outcome - grand_mean) ** 2 for outcome in outcomes)
    explained_ss = 0.0
    used_df = 0
    rows: list[dict[str, object]] = []

    factor_grouped: dict[str, dict[str, list[float]]] = {}
    factor_means: dict[tuple[str, str], float] = {}

    for factor in PART1_FACTORS:
        grouped: dict[str, list[float]] = defaultdict(list)
        for observation, outcome in zip(observations, outcomes):
            grouped[str(observation[factor])].append(outcome)
        factor_grouped[factor] = grouped
        for level, values in grouped.items():
            factor_means[(factor, level)] = sum(values) / len(values)
        factor_ss = sum(
            len(values) * ((sum(values) / len(values)) - grand_mean) ** 2
            for values in grouped.values()
        )
        df = max(0, len(grouped) - 1)
        explained_ss += factor_ss
        used_df += df
        rows.append(
            {
                "term": factor,
                "levels": len(grouped),
                "df": df,
                "sum_squares": factor_ss,
                "variance_share": factor_ss / total_ss if total_ss else "",
                "mean_square": factor_ss / df if df else "",
            }
        )

    model_frame_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for observation, outcome in zip(observations, outcomes):
        model_frame_grouped[(str(observation["model"]), str(observation["frame"]))].append(outcome)
    model_frame_ss = sum(
        len(values)
        * (
            (sum(values) / len(values))
            - factor_means[("model", model)]
            - factor_means[("frame", frame)]
            + grand_mean
        )
        ** 2
        for (model, frame), values in model_frame_grouped.items()
    )
    model_frame_df = (
        max(0, len(factor_grouped.get("model", {})) - 1)
        * max(0, len(factor_grouped.get("frame", {})) - 1)
    )
    explained_ss += model_frame_ss
    used_df += model_frame_df
    rows.append(
        {
            "term": "model_frame_interaction",
            "levels": len(model_frame_grouped),
            "df": model_frame_df,
            "sum_squares": model_frame_ss,
            "variance_share": model_frame_ss / total_ss if total_ss else "",
            "mean_square": model_frame_ss / model_frame_df if model_frame_df else "",
        }
    )

    residual_ss = max(0.0, total_ss - explained_ss)
    residual_df = max(0, len(outcomes) - 1 - used_df)
    rows.append(
        {
            "term": "residual_interactions_and_unmodeled",
            "levels": "",
            "df": residual_df,
            "sum_squares": residual_ss,
            "variance_share": residual_ss / total_ss if total_ss else "",
            "mean_square": residual_ss / residual_df if residual_df else "",
        }
    )
    rows.append(
        {
            "term": "total",
            "levels": "",
            "df": len(outcomes) - 1,
            "sum_squares": total_ss,
            "variance_share": 1.0,
            "mean_square": "",
        }
    )
    return rows


def summarize_part1(raw_dir: Path, output_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    model_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    all_frame_model_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    dimension_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    model_providers: dict[str, str] = {}
    model_metadata: dict[tuple[str, str], ModelMetadata] = {}
    observations: list[dict[str, object]] = []
    for path in _csv_paths(raw_dir / "part_1"):
        sidecar = _metadata_for_csv(path)
        for row in _read_rows(path):
            key = _model_key(row)
            outcome = "cooperate" if _part1_is_cooperative(row) else "defect_or_overuse"
            model_providers[key[1]] = key[0]
            model_metadata[key] = resolve_model_metadata(key[0], key[1], sidecar)
            all_frame_model_counts[key][outcome] += 1
            is_primary = row.get("frame", "") == "self_direct"
            if is_primary:
                model_counts[key][outcome] += 1
            for dimension in ("game", "frame", "domain", "presentation"):
                # Frame effects necessarily use all frames.  Other breakdowns
                # inherit the self-directed primary estimand.
                if dimension == "frame" or is_primary:
                    dimension_counts[(key[1], dimension, row.get(dimension, ""))][outcome] += 1
            observations.append(
                {
                    "model": key[1],
                    "frame": row.get("frame", ""),
                    "game": row.get("game", ""),
                    "domain": row.get("domain", ""),
                    "presentation": row.get("presentation", ""),
                    "cooperate": 1 if outcome == "cooperate" else 0,
                }
            )

    model_rows: list[dict[str, object]] = []
    for (provider, model), counter in sorted(model_counts.items()):
        cooperate = counter["cooperate"]
        total = cooperate + counter["defect_or_overuse"]
        all_counter = all_frame_model_counts[(provider, model)]
        all_cooperate = all_counter["cooperate"]
        all_total = all_cooperate + all_counter["defect_or_overuse"]
        low, high = _wilson_interval(cooperate, total)
        grouping = model_metadata[(provider, model)]
        model_rows.append(
            {
                "provider": provider,
                "model": model,
                "family_id": grouping.family_id,
                "developer_id": grouping.developer_id,
                "cohort": grouping.cohort,
                "primary_frame": "self_direct",
                "total": total,
                "cooperative": cooperate,
                "non_cooperative": counter["defect_or_overuse"],
                "cooperation_rate": cooperate / total if total else "",
                "wilson_low": low if total else "",
                "wilson_high": high if total else "",
                "all_frames_total": all_total,
                "all_frames_cooperative": all_cooperate,
                "all_frames_cooperation_rate": all_cooperate / all_total if all_total else "",
            }
        )

    dimension_rows: list[dict[str, object]] = []
    for (model, dimension, value), counter in sorted(dimension_counts.items()):
        cooperate = counter["cooperate"]
        total = cooperate + counter["defect_or_overuse"]
        low, high = _wilson_interval(cooperate, total)
        dimension_rows.append(
            {
                "model": model,
                "dimension": dimension,
                "value": value,
                "analysis_scope": "all_frames" if dimension == "frame" else "self_direct",
                "total": total,
                "cooperative": cooperate,
                "cooperation_rate": cooperate / total if total else "",
                "wilson_low": low if total else "",
                "wilson_high": high if total else "",
            }
        )

    model_path = output_dir / "part1_model_summary.csv"
    dimension_path = output_dir / "part1_dimension_summary.csv"
    frame_effect_path = output_dir / "part1_frame_effects.csv"
    prompt_sensitivity_path = output_dir / "part1_prompt_sensitivity.csv"
    decomposition_path = output_dir / "part1_factor_decomposition.csv"
    _write_csv(
        model_path,
        [
            "provider",
            "model",
            "family_id",
            "developer_id",
            "cohort",
            "primary_frame",
            "total",
            "cooperative",
            "non_cooperative",
            "cooperation_rate",
            "wilson_low",
            "wilson_high",
            "all_frames_total",
            "all_frames_cooperative",
            "all_frames_cooperation_rate",
        ],
        model_rows,
    )
    _write_csv(
        dimension_path,
        [
            "model",
            "dimension",
            "value",
            "analysis_scope",
            "total",
            "cooperative",
            "cooperation_rate",
            "wilson_low",
            "wilson_high",
        ],
        dimension_rows,
    )
    _write_csv(
        frame_effect_path,
        [
            "provider",
            "model",
            "prediction_rate",
            "observer_evaluation_rate",
            "observer_evaluation_minus_prediction_pp",
            "self_direct_rate",
            "self_direct_minus_prediction_pp",
            "advice_rate",
            "advice_minus_prediction_pp",
        ],
        _part1_frame_effect_rows(observations, model_providers),
    )
    _write_csv(
        prompt_sensitivity_path,
        [
            "provider",
            "model",
            "frame_sensitivity_pp",
            "game_sensitivity_pp",
            "domain_sensitivity_pp",
            "presentation_sensitivity_pp",
            "maximum_prompt_sensitivity_pp",
            "maximum_prompt_sensitivity_dimension",
        ],
        _part1_prompt_sensitivity_rows(observations, model_providers),
    )
    _write_csv(
        decomposition_path,
        ["term", "levels", "df", "sum_squares", "variance_share", "mean_square"],
        _part1_factor_decomposition_rows(observations),
    )
    return model_path, dimension_path, frame_effect_path, prompt_sensitivity_path, decomposition_path


def _metadata_for_csv(path: Path) -> dict[str, object]:
    metadata_path = path.with_name(f"{path.stem}_meta.json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _part2_expected_config(path: Path, rows: list[dict[str, str]]) -> tuple[int, int, int]:
    metadata = _metadata_for_csv(path)
    parameters = metadata.get("parameters", {}) if isinstance(metadata, dict) else {}
    society_config = parameters.get("society_config", {}) if isinstance(parameters, dict) else {}
    if isinstance(metadata, dict) and not society_config:
        society_config = metadata.get("society_config", {})

    horizon = int(society_config.get("days") or 0) if isinstance(society_config, dict) else 0
    society_size = int(society_config.get("society_size") or 0) if isinstance(society_config, dict) else 0
    resource_capacity = int(
        parameters.get("resource_capacity") or metadata.get("resource_capacity") or 0
    ) if isinstance(parameters, dict) and isinstance(metadata, dict) else 0

    match = re.search(r"__n(?P<size>\d+)__d(?P<days>\d+)__", path.name)
    if match:
        horizon = horizon or int(match.group("days"))
        society_size = society_size or int(match.group("size"))

    if not society_size:
        society_size = max(
            int(row.get("population_start") or 0)
            for row in rows
        )
    if not resource_capacity:
        resource_capacity = max(
            int(row.get("resource_capacity") or 0)
            for row in rows
        )
    if not horizon:
        horizon = max(
            int(row.get("day") or 0)
            for row in rows
        )
    return horizon, society_size, resource_capacity


def _part2_normalized_auc(
    day_rows: dict[int, list[dict[str, str]]],
    *,
    horizon: int,
    society_size: int,
    resource_capacity: int,
) -> tuple[float, float]:
    if horizon <= 0 or society_size <= 0 or resource_capacity <= 0:
        return float("nan"), float("nan")

    reserve_area = 0.0
    population_area = 0.0
    for day in range(1, horizon + 1):
        grouped = day_rows.get(day, [])
        if grouped:
            row = grouped[-1]
            reserve_area += int(row.get("resource_units_remaining") or 0) / resource_capacity
            population_area += int(row.get("population_end") or 0) / society_size
    return reserve_area / horizon, population_area / horizon


PART2_RUN_FIELDS = [
    "provider",
    "model",
    "family_id",
    "developer_id",
    "cohort",
    "run_id",
    "csv_path",
    "rows",
    "days_completed",
    "restraints",
    "overuses",
    "restraint_rate",
    "wilson_low",
    "wilson_high",
    "first_depletion_day",
    "final_population",
    "final_resource_units",
    "resource_capacity",
    "normalized_aurc",
    "normalized_aupc",
    "total_deaths",
    "reasoning_mismatch_flags",
]


def _part2_run_summary(path: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    provider, model = _model_key(rows[0])
    grouping = resolve_model_metadata(provider, model, _metadata_for_csv(path))
    actions = Counter(row.get("action", "") for row in rows)
    day_rows: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            day_rows[int(row.get("day", ""))].append(row)
        except ValueError:
            continue
    final_day = max(day_rows) if day_rows else 0
    final_row = day_rows[final_day][-1] if final_day else rows[-1]
    depletion_days = [
        day
        for day, grouped in day_rows.items()
        if grouped and int(grouped[-1].get("resource_units_remaining", "0") or 0) == 0
    ]
    total = actions["RESTRAIN"] + actions["OVERUSE"]
    low, high = _wilson_interval(actions["RESTRAIN"], total)
    horizon, society_size, resource_capacity = _part2_expected_config(path, rows)
    normalized_aurc, normalized_aupc = _part2_normalized_auc(
        day_rows,
        horizon=horizon,
        society_size=society_size,
        resource_capacity=resource_capacity,
    )
    return {
        "provider": provider,
        "model": model,
        "family_id": grouping.family_id,
        "developer_id": grouping.developer_id,
        "cohort": grouping.cohort,
        "run_id": path.stem,
        "csv_path": str(path),
        "rows": len(rows),
        "days_completed": final_day,
        "restraints": actions["RESTRAIN"],
        "overuses": actions["OVERUSE"],
        "restraint_rate": actions["RESTRAIN"] / total if total else float("nan"),
        "wilson_low": low if total else float("nan"),
        "wilson_high": high if total else float("nan"),
        "first_depletion_day": min(depletion_days) if depletion_days else "",
        "final_population": int(final_row.get("population_end", "") or 0),
        "final_resource_units": int(final_row.get("resource_units_remaining", "") or 0),
        "resource_capacity": resource_capacity or int(final_row.get("resource_capacity", "") or 0),
        "normalized_aurc": normalized_aurc,
        "normalized_aupc": normalized_aupc,
        "total_deaths": sum(
            int(grouped[-1].get("deaths", "0") or 0)
            for grouped in day_rows.values()
            if grouped
        ),
        "reasoning_mismatch_flags": sum(
            1 for row in rows if _flag_reasoning_misunderstanding(row)
        ),
    }


_T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def _run_mean_interval(values: list[float]) -> tuple[float, float]:
    """95% t interval for an equally weighted mean of independent runs."""

    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return float("nan"), float("nan")
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    df = len(values) - 1
    if df in _T_975:
        critical = _T_975[df]
    elif df < 25:
        critical = _T_975[20]
    elif df < 30:
        critical = _T_975[25]
    else:
        critical = 1.959963984540054
    margin = critical * math.sqrt(variance / len(values))
    return mean - margin, mean + margin


def _mean(values: list[object]) -> float:
    numeric = [float(value) for value in values if value != "" and not math.isnan(float(value))]
    return sum(numeric) / len(numeric) if numeric else float("nan")


def _aggregate_part2_runs(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["provider"]), str(row["model"]))].append(row)

    output: list[dict[str, object]] = []
    for (provider, model), runs in sorted(grouped.items()):
        for field in ("family_id", "developer_id", "cohort"):
            values = {str(run[field]) for run in runs}
            if len(values) != 1:
                raise ValueError(f"Conflicting {field} values for replicated runs of {provider}/{model}")
        rates = [float(run["restraint_rate"]) for run in runs]
        restraint_rate = sum(rates) / len(rates)
        if len(runs) == 1:
            interval_low = float(runs[0]["wilson_low"])
            interval_high = float(runs[0]["wilson_high"])
            interval_method = "within_run_wilson"
        else:
            interval_low, interval_high = _run_mean_interval(rates)
            interval_low = max(0.0, interval_low)
            interval_high = min(1.0, interval_high)
            interval_method = "between_run_t"
        depletion_days = [
            float(run["first_depletion_day"])
            for run in runs
            if run["first_depletion_day"] != ""
        ]
        source_paths = [str(run["csv_path"]) for run in runs]
        output.append(
            {
                "provider": provider,
                "model": model,
                "family_id": runs[0]["family_id"],
                "developer_id": runs[0]["developer_id"],
                "cohort": runs[0]["cohort"],
                "aggregation_unit": "run",
                "run_count": len(runs),
                "csv_path": source_paths[0],
                "source_csv_paths": json.dumps(source_paths, separators=(",", ":")),
                "rows": sum(int(run["rows"]) for run in runs),
                "days_completed": _mean([run["days_completed"] for run in runs]),
                "restraints": sum(int(run["restraints"]) for run in runs),
                "overuses": sum(int(run["overuses"]) for run in runs),
                "restraint_rate": restraint_rate,
                "restraint_rate_min": min(rates),
                "restraint_rate_max": max(rates),
                "restraint_ci_low": interval_low,
                "restraint_ci_high": interval_high,
                "interval_method": interval_method,
                # Compatibility aliases for existing plot readers.  The
                # interval_method column disambiguates replicated-run t CIs.
                "wilson_low": interval_low,
                "wilson_high": interval_high,
                "runs_depleted": len(depletion_days),
                "first_depletion_day": _mean(depletion_days) if depletion_days else "",
                "final_population": _mean([run["final_population"] for run in runs]),
                "final_resource_units": _mean([run["final_resource_units"] for run in runs]),
                "resource_capacity": _mean([run["resource_capacity"] for run in runs]),
                "normalized_aurc": _mean([run["normalized_aurc"] for run in runs]),
                "normalized_aupc": _mean([run["normalized_aupc"] for run in runs]),
                "total_deaths": _mean([run["total_deaths"] for run in runs]),
                "reasoning_mismatch_flags": sum(
                    int(run["reasoning_mismatch_flags"]) for run in runs
                ),
            }
        )
    return output


def summarize_part2(raw_dir: Path, output_dir: Path) -> Path:
    run_rows: list[dict[str, object]] = []
    for path in _csv_paths(raw_dir / "part_2"):
        rows = _read_rows(path)
        if rows:
            run_rows.append(_part2_run_summary(path, rows))

    run_path = output_dir / "part2_run_summary.csv"
    _write_csv(run_path, PART2_RUN_FIELDS, run_rows)

    model_path = output_dir / "part2_model_summary.csv"
    model_fields = [
        "provider",
        "model",
        "family_id",
        "developer_id",
        "cohort",
        "aggregation_unit",
        "run_count",
        "csv_path",
        "source_csv_paths",
        "rows",
        "days_completed",
        "restraints",
        "overuses",
        "restraint_rate",
        "restraint_rate_min",
        "restraint_rate_max",
        "restraint_ci_low",
        "restraint_ci_high",
        "interval_method",
        "wilson_low",
        "wilson_high",
        "runs_depleted",
        "first_depletion_day",
        "final_population",
        "final_resource_units",
        "resource_capacity",
        "normalized_aurc",
        "normalized_aupc",
        "total_deaths",
        "reasoning_mismatch_flags",
    ]
    _write_csv(model_path, model_fields, _aggregate_part2_runs(run_rows))
    return model_path


def summarize_cross_part(output_dir: Path) -> tuple[Path, Path]:
    def keyed(path: Path) -> dict[tuple[str, str], dict[str, str]]:
        return {
            (row.get("provider", "unknown"), row["model"]): row
            for row in _read_rows(path)
        }

    part0 = keyed(output_dir / "part0_model_summary.csv")
    part1 = keyed(output_dir / "part1_model_summary.csv")
    part2 = keyed(output_dir / "part2_model_summary.csv")
    model_keys = sorted(set(part0) & set(part1) & set(part2))

    model_rows: list[dict[str, object]] = []
    for provider, model in model_keys:
        key = (provider, model)
        first_depletion = part2[key].get("first_depletion_day", "")
        grouping = resolve_model_metadata(provider, model)
        model_rows.append(
            {
                "provider": provider,
                "model": model,
                "family_id": part1[key].get("family_id") or grouping.family_id,
                "developer_id": part1[key].get("developer_id") or grouping.developer_id,
                "cohort": part1[key].get("cohort") or grouping.cohort,
                "cooperation_measure": "self_direct",
                "part2_aggregation_unit": part2[key].get("aggregation_unit", "run"),
                "part2_run_count": int(float(part2[key].get("run_count", "1") or 1)),
                "safety_refusal_rate": float(part0[key]["safety_refusal_rate"]),
                "cooperation_rate": float(part1[key]["cooperation_rate"]),
                "all_frames_cooperation_rate": float(
                    part1[key].get("all_frames_cooperation_rate")
                    or part1[key]["cooperation_rate"]
                ),
                "restraint_rate": float(part2[key]["restraint_rate"]),
                "final_population": float(part2[key]["final_population"] or 0),
                "final_resource_units": float(part2[key]["final_resource_units"] or 0),
                "normalized_aurc": float(part2[key]["normalized_aurc"]),
                "normalized_aupc": float(part2[key]["normalized_aupc"]),
                "first_depletion_day": float(first_depletion) if first_depletion else "",
                "reasoning_mismatch_flags": int(
                    float(part2[key]["reasoning_mismatch_flags"] or 0)
                ),
            }
        )

    model_path = output_dir / "cross_part_model_summary.csv"
    _write_csv(
        model_path,
        [
            "provider",
            "model",
            "family_id",
            "developer_id",
            "cohort",
            "cooperation_measure",
            "part2_aggregation_unit",
            "part2_run_count",
            "safety_refusal_rate",
            "cooperation_rate",
            "all_frames_cooperation_rate",
            "restraint_rate",
            "final_population",
            "final_resource_units",
            "normalized_aurc",
            "normalized_aupc",
            "first_depletion_day",
            "reasoning_mismatch_flags",
        ],
        model_rows,
    )

    metric_values = {
        "safety_refusal_rate": [float(row["safety_refusal_rate"]) for row in model_rows],
        "cooperation_rate": [float(row["cooperation_rate"]) for row in model_rows],
        "restraint_rate": [float(row["restraint_rate"]) for row in model_rows],
        "final_population": [float(row["final_population"]) for row in model_rows],
    }
    pairs = [
        ("safety_refusal_rate", "cooperation_rate"),
        ("safety_refusal_rate", "restraint_rate"),
        ("safety_refusal_rate", "final_population"),
        ("cooperation_rate", "restraint_rate"),
        ("cooperation_rate", "final_population"),
        ("restraint_rate", "final_population"),
    ]
    correlation_rows: list[dict[str, object]] = []
    model_groups = [f"{row['provider']}/{row['model']}" for row in model_rows]
    family_groups = [str(row["family_id"]) for row in model_rows]
    methods = {
        "pearson": _pearson_correlation,
        "spearman": _spearman_correlation,
        "kendall": _kendall_correlation,
    }
    for left_metric, right_metric in pairs:
        xs = metric_values[left_metric]
        ys = metric_values[right_metric]
        row: dict[str, object] = {
            "metric_x": left_metric,
            "metric_y": right_metric,
            "n_models": len(model_rows),
            "n_families": len(set(family_groups)),
            "cooperation_measure": "self_direct",
            "fisher_z_ci_note": "pearson_standard;rank_coefficients_approximate",
        }
        for method, statistic in methods.items():
            estimate = statistic(xs, ys)
            fisher_low, fisher_high = _fisher_z_interval(estimate, len(model_rows))
            bootstrap_low, bootstrap_high = _bootstrap_ci(xs, ys, statistic)
            lomo_low, lomo_high, lomo_count = leave_group_out_range(
                xs, ys, model_groups, statistic
            )
            lofo_low, lofo_high, lofo_count = leave_group_out_range(
                xs, ys, family_groups, statistic
            )
            coefficient_name = "tau_b" if method == "kendall" else "r"
            row[f"{method}_{coefficient_name}"] = estimate
            row[f"{method}_fisher_z_low"] = fisher_low
            row[f"{method}_fisher_z_high"] = fisher_high
            row[f"{method}_bootstrap_low"] = bootstrap_low
            row[f"{method}_bootstrap_high"] = bootstrap_high
            row[f"{method}_leave_one_model_out_min"] = lomo_low
            row[f"{method}_leave_one_model_out_max"] = lomo_high
            row[f"{method}_leave_one_model_out_estimates"] = lomo_count
            row[f"{method}_leave_one_family_out_min"] = lofo_low
            row[f"{method}_leave_one_family_out_max"] = lofo_high
            row[f"{method}_leave_one_family_out_estimates"] = lofo_count
        correlation_rows.append(row)

    correlation_path = output_dir / "cross_part_correlations.csv"
    _write_csv(
        correlation_path,
        [
            "metric_x",
            "metric_y",
            "n_models",
            "n_families",
            "cooperation_measure",
            "fisher_z_ci_note",
            *[
                field
                for method, coefficient in (
                    ("pearson", "r"),
                    ("spearman", "r"),
                    ("kendall", "tau_b"),
                )
                for field in (
                    f"{method}_{coefficient}",
                    f"{method}_fisher_z_low",
                    f"{method}_fisher_z_high",
                    f"{method}_bootstrap_low",
                    f"{method}_bootstrap_high",
                    f"{method}_leave_one_model_out_min",
                    f"{method}_leave_one_model_out_max",
                    f"{method}_leave_one_model_out_estimates",
                    f"{method}_leave_one_family_out_min",
                    f"{method}_leave_one_family_out_max",
                    f"{method}_leave_one_family_out_estimates",
                )
            ],
        ],
        correlation_rows,
    )
    return model_path, correlation_path


def summarize_all(raw_dir: Path = RAW_DIR, output_dir: Path = TABLES_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = list(summarize_part0(raw_dir, output_dir))
    outputs.extend(summarize_part1(raw_dir, output_dir))
    outputs.append(summarize_part2(raw_dir, output_dir))
    outputs.append(output_dir / "part2_run_summary.csv")
    outputs.extend(summarize_cross_part(output_dir))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph-independent paper summary tables.")
    parser.add_argument("--raw-dir", default=str(RAW_DIR))
    parser.add_argument("--output-dir", default=str(TABLES_DIR))
    args = parser.parse_args()
    outputs = summarize_all(Path(args.raw_dir), Path(args.output_dir))
    print("Wrote summary tables:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
