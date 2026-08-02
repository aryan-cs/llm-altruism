from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from experiments.part0 import part_0
from experiments.part1 import part_1
from experiments.part2 import part_2
from analysis.model_metadata import ModelMetadata, resolve_model_metadata
from analysis.part2_dynamics import (
    STRUCTURAL_OUTPUT_FIELDS,
    load_part2_structural_cell,
    load_part2_run_identity,
    normalized_part2_auc,
)
from analysis.part2_confirmatory import (
    DEFAULT_BCA_REPLICATES,
    bca_mean_interval,
    student_t_975,
)
from analysis.statistics import (
    fisher_z_interval,
    kendall_correlation,
    leave_group_out_range,
    pearson_correlation,
    rank_values,
    spearman_correlation,
)
from analysis.validation import (
    _csv_paths,
    _flag_reasoning_misunderstanding,
    validate_part2_file,
)

RAW_DIR = Path("data") / "raw"
TABLES_DIR = Path("data") / "analysis" / "tables"
PART0_LANGUAGE_ORDER = ("english", "chinese", "russian")
PART1_FACTORS = ("model", "frame", "game", "domain", "presentation")
PART1_SENSITIVITY_DIMENSIONS = ("frame", "game", "domain", "presentation")
PART1_FRAME_COMPARISONS = ("observer_evaluation", "self_direct", "advice")
PART1_FRAME_BASELINE = "prediction"
PILOT_ROOT_BOOTSTRAP_REPLICATES = 2_000
PILOT_ROOT_BOOTSTRAP_SEED = 20260802
PILOT_ROOT_BOOTSTRAP_METHOD = "root_cluster_bootstrap_percentile_95"


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


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _root_cluster_rate_interval(
    observations: list[tuple[str, int]],
    *,
    replicates: int = PILOT_ROOT_BOOTSTRAP_REPLICATES,
    seed: int = PILOT_ROOT_BOOTSTRAP_SEED,
) -> tuple[float, float, int]:
    """Percentile interval resampling roots while retaining all rows per root."""

    if replicates != PILOT_ROOT_BOOTSTRAP_REPLICATES:
        raise ValueError("pilot root-cluster intervals require exactly 2,000 replicates")
    if not observations:
        raise ValueError("root-cluster interval requires judged observations")
    clusters: dict[str, list[int]] = defaultdict(list)
    for root_id, outcome in observations:
        if not isinstance(root_id, str) or not root_id.strip():
            raise ValueError(
                "root-cluster interval requires a nonempty independent design-root ID"
            )
        if outcome not in {0, 1}:
            raise ValueError("root-cluster interval outcomes must equal zero or one")
        clusters[root_id].append(outcome)
    root_ids = sorted(clusters)
    rng = random.Random(seed)
    draws: list[float] = []
    for _replicate in range(replicates):
        selected = [rng.choice(root_ids) for _ in root_ids]
        successes = sum(sum(clusters[root_id]) for root_id in selected)
        total = sum(len(clusters[root_id]) for root_id in selected)
        draws.append(successes / total)
    return _percentile(draws, 0.025), _percentile(draws, 0.975), len(root_ids)


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
            "evidence_status": "deprecated_invalid_legacy_label_protocol",
            "evidence_note": "Not a paper result; complete response-only rejudgment and human validation are required.",
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
    root_observations: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
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
            if verdict in {"denied", "complied"}:
                root_id = row.get("base_prompt_id", "").strip() or row.get(
                    "prompt", ""
                ).strip()
                root_observations[key].append((root_id, int(verdict == "denied")))
            language = row.get("language", "").strip().lower()
            if language and verdict in {"denied", "complied"}:
                language_counts[(key[0], key[1], language)][verdict] += 1

    rows: list[dict[str, object]] = []
    for (provider, model), counter in sorted(counts.items()):
        denied = counter["denied"]
        complied = counter["complied"]
        skipped = counter["skipped"]
        total = denied + complied
        cluster_low, cluster_high, root_count = _root_cluster_rate_interval(
            root_observations[(provider, model)]
        )
        row_low, row_high = _wilson_interval(denied, total)
        grouping = model_metadata[(provider, model)]
        rows.append(
            {
                "evidence_status": "deprecated_invalid_legacy_label_protocol",
                "evidence_note": "Not a paper result; complete response-only rejudgment and human validation are required.",
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
                "cluster_ci_low": cluster_low,
                "cluster_ci_high": cluster_high,
                "interval_method": PILOT_ROOT_BOOTSTRAP_METHOD,
                "interval_unit": "base_prompt",
                "interval_replicates": PILOT_ROOT_BOOTSTRAP_REPLICATES,
                "interval_seed": PILOT_ROOT_BOOTSTRAP_SEED,
                "root_cluster_count": root_count,
                "row_binomial_wilson_low_diagnostic": row_low,
                "row_binomial_wilson_high_diagnostic": row_high,
            }
        )

    path = output_dir / "part0_model_summary.csv"
    robustness_path = output_dir / "part0_language_robustness.csv"
    _write_csv(
        path,
        [
            "evidence_status",
            "evidence_note",
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
            "cluster_ci_low",
            "cluster_ci_high",
            "interval_method",
            "interval_unit",
            "interval_replicates",
            "interval_seed",
            "root_cluster_count",
            "row_binomial_wilson_low_diagnostic",
            "row_binomial_wilson_high_diagnostic",
        ],
        rows,
    )
    _write_csv(
        robustness_path,
        [
            "evidence_status",
            "evidence_note",
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
                    "provider": key[0],
                    "model": key[1],
                    "scenario_variant": row.get("scenario_variant", "").strip(),
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
        primary_observations = [
            (str(observation["scenario_variant"]), int(observation["cooperate"]))
            for observation in observations
            if observation["provider"] == provider
            and observation["model"] == model
            and observation["frame"] == "self_direct"
        ]
        all_observations = [
            (str(observation["scenario_variant"]), int(observation["cooperate"]))
            for observation in observations
            if observation["provider"] == provider and observation["model"] == model
        ]
        cluster_low, cluster_high, root_count = _root_cluster_rate_interval(
            primary_observations
        )
        all_cluster_low, all_cluster_high, all_root_count = (
            _root_cluster_rate_interval(all_observations)
        )
        row_low, row_high = _wilson_interval(cooperate, total)
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
                "cluster_ci_low": cluster_low,
                "cluster_ci_high": cluster_high,
                "interval_method": PILOT_ROOT_BOOTSTRAP_METHOD,
                "interval_unit": "scenario_variant",
                "interval_replicates": PILOT_ROOT_BOOTSTRAP_REPLICATES,
                "interval_seed": PILOT_ROOT_BOOTSTRAP_SEED,
                "root_cluster_count": root_count,
                "row_binomial_wilson_low_diagnostic": row_low,
                "row_binomial_wilson_high_diagnostic": row_high,
                "all_frames_total": all_total,
                "all_frames_cooperative": all_cooperate,
                "all_frames_cooperation_rate": all_cooperate / all_total if all_total else "",
                "all_frames_cluster_ci_low": all_cluster_low,
                "all_frames_cluster_ci_high": all_cluster_high,
                "all_frames_root_cluster_count": all_root_count,
            }
        )

    dimension_rows: list[dict[str, object]] = []
    for (model, dimension, value), counter in sorted(dimension_counts.items()):
        cooperate = counter["cooperate"]
        total = cooperate + counter["defect_or_overuse"]
        scoped_observations = [
            (str(observation["scenario_variant"]), int(observation["cooperate"]))
            for observation in observations
            if observation["model"] == model
            and (
                observation[dimension] == value
                and (dimension == "frame" or observation["frame"] == "self_direct")
            )
        ]
        cluster_low, cluster_high, root_count = _root_cluster_rate_interval(
            scoped_observations
        )
        row_low, row_high = _wilson_interval(cooperate, total)
        dimension_rows.append(
            {
                "model": model,
                "dimension": dimension,
                "value": value,
                "analysis_scope": "all_frames" if dimension == "frame" else "self_direct",
                "total": total,
                "cooperative": cooperate,
                "cooperation_rate": cooperate / total if total else "",
                "cluster_ci_low": cluster_low,
                "cluster_ci_high": cluster_high,
                "interval_method": PILOT_ROOT_BOOTSTRAP_METHOD,
                "interval_unit": "scenario_variant",
                "interval_replicates": PILOT_ROOT_BOOTSTRAP_REPLICATES,
                "interval_seed": PILOT_ROOT_BOOTSTRAP_SEED,
                "root_cluster_count": root_count,
                "row_binomial_wilson_low_diagnostic": row_low,
                "row_binomial_wilson_high_diagnostic": row_high,
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
            "cluster_ci_low",
            "cluster_ci_high",
            "interval_method",
            "interval_unit",
            "interval_replicates",
            "interval_seed",
            "root_cluster_count",
            "row_binomial_wilson_low_diagnostic",
            "row_binomial_wilson_high_diagnostic",
            "all_frames_total",
            "all_frames_cooperative",
            "all_frames_cooperation_rate",
            "all_frames_cluster_ci_low",
            "all_frames_cluster_ci_high",
            "all_frames_root_cluster_count",
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
            "cluster_ci_low",
            "cluster_ci_high",
            "interval_method",
            "interval_unit",
            "interval_replicates",
            "interval_seed",
            "root_cluster_count",
            "row_binomial_wilson_low_diagnostic",
            "row_binomial_wilson_high_diagnostic",
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


def _part2_normalized_auc(
    day_rows: dict[int, list[dict[str, str]]],
    *,
    horizon: int,
    society_size: int,
    resource_capacity: int,
) -> tuple[float, float]:
    return normalized_part2_auc(
        day_rows,
        horizon=horizon,
        society_size=society_size,
        resource_capacity=resource_capacity,
    )


PART2_RUN_FIELDS = [
    "provider",
    "model",
    "family_id",
    "developer_id",
    "cohort",
    *STRUCTURAL_OUTPUT_FIELDS,
    "run_id",
    "trajectory_id",
    "environment_seed",
    "generation_seed",
    "strict_schema",
    "csv_path",
    "rows",
    "days_completed",
    "restraints",
    "overuses",
    "invalid_decisions",
    "restraint_rate",
    "restraint_rate_scorable",
    "invalid_rate",
    "first_depletion_day",
    "depletion_observed",
    "time_to_depletion_or_censoring",
    "censoring_day",
    "survived_through_horizon",
    "final_population",
    "final_resource_units",
    "normalized_aurc",
    "normalized_aupc",
    "total_deaths",
    "reasoning_mismatch_flags",
]


def _part2_run_summary(path: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    cell = load_part2_structural_cell(path, rows)
    identity = load_part2_run_identity(path, rows)
    provider, model = cell.provider, cell.model
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
    scorable_total = actions["RESTRAIN"] + actions["OVERUSE"]
    total = scorable_total + actions["INVALID"]
    normalized_aurc, normalized_aupc = _part2_normalized_auc(
        day_rows,
        horizon=cell.horizon_days,
        society_size=cell.society_size,
        resource_capacity=cell.resource_capacity,
    )
    first_depletion_day = min(depletion_days) if depletion_days else None
    return {
        "provider": provider,
        "model": model,
        "family_id": grouping.family_id,
        "developer_id": grouping.developer_id,
        "cohort": grouping.cohort,
        **cell.output_fields(),
        "run_id": identity.run_id,
        "trajectory_id": identity.trajectory_id,
        "environment_seed": (
            identity.environment_seed if identity.environment_seed is not None else ""
        ),
        "generation_seed": (
            identity.generation_seed if identity.generation_seed is not None else ""
        ),
        "strict_schema": int(identity.strict_schema),
        "csv_path": str(path),
        "rows": len(rows),
        "days_completed": final_day,
        "restraints": actions["RESTRAIN"],
        "overuses": actions["OVERUSE"],
        "invalid_decisions": actions["INVALID"],
        "restraint_rate": actions["RESTRAIN"] / total if total else float("nan"),
        "restraint_rate_scorable": (
            actions["RESTRAIN"] / scorable_total
            if scorable_total
            else float("nan")
        ),
        "invalid_rate": actions["INVALID"] / total if total else float("nan"),
        "first_depletion_day": first_depletion_day if first_depletion_day is not None else "",
        "depletion_observed": int(first_depletion_day is not None),
        "time_to_depletion_or_censoring": first_depletion_day or final_day,
        "censoring_day": "" if first_depletion_day is not None else final_day,
        "survived_through_horizon": (
            int(first_depletion_day is None)
            if first_depletion_day is not None or final_day >= cell.horizon_days
            else ""
        ),
        "final_population": int(final_row.get("population_end", "") or 0),
        "final_resource_units": int(final_row.get("resource_units_remaining", "") or 0),
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


def _run_mean_interval(values: list[float]) -> tuple[float, float]:
    """95% t interval for an equally weighted mean of independent runs."""

    if not values:
        return float("nan"), float("nan")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return float("nan"), float("nan")
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    df = len(values) - 1
    critical = student_t_975(df) if df <= 39 else 1.959963984540054
    margin = critical * math.sqrt(variance / len(values))
    return mean - margin, mean + margin


def _mean(values: list[object]) -> float:
    numeric = [float(value) for value in values if value != "" and not math.isnan(float(value))]
    return sum(numeric) / len(numeric) if numeric else float("nan")


def _part2_interval_fields(
    runs: list[dict[str, object]],
    *,
    metric: str,
    output_metric: str | None = None,
    lower_bound: float,
    upper_bound: float,
    seed_material: str,
) -> dict[str, object]:
    """Return t and deterministic BCa intervals over independent runs."""

    values = [float(run[metric]) for run in runs if run[metric] != ""]
    output_metric = output_metric or metric
    fields: dict[str, object] = {
        f"{output_metric}_t_ci_low": "",
        f"{output_metric}_t_ci_high": "",
        f"{output_metric}_bca_ci_low": "",
        f"{output_metric}_bca_ci_high": "",
    }
    if len(values) != len(runs):
        return fields
    if len(values) < 2:
        return fields
    t_low, t_high = _run_mean_interval(values)
    seed = int.from_bytes(
        hashlib.sha256(f"{seed_material}:{metric}".encode("utf-8")).digest()[:8],
        "big",
    )
    bca_low, bca_high = bca_mean_interval(
        values,
        replicates=DEFAULT_BCA_REPLICATES,
        seed=seed,
    )
    fields.update(
        {
            f"{output_metric}_t_ci_low": max(lower_bound, t_low),
            f"{output_metric}_t_ci_high": min(upper_bound, t_high),
            f"{output_metric}_bca_ci_low": max(lower_bound, bca_low),
            f"{output_metric}_bca_ci_high": min(upper_bound, bca_high),
        }
    )
    return fields


def _restricted_depletion_estimates(
    runs: list[dict[str, object]],
    horizon: int,
) -> tuple[float, float]:
    """Kaplan-Meier RMST and survival through a common finite horizon.

    Event times are first depletion days; undepleted runs are right-censored at
    their last observed day.  Event processing precedes censoring at tied times.
    RMST is not extrapolated after all surviving trajectories have been censored.
    """

    observations = [
        (
            min(int(run["time_to_depletion_or_censoring"]), horizon),
            bool(int(run["depletion_observed"])),
        )
        for run in runs
    ]
    at_risk = len(observations)
    survival = 1.0
    rmst = 0.0
    previous_time = 0
    for time in sorted({time for time, _event in observations}):
        if time > horizon:
            break
        if time > previous_time:
            if at_risk == 0 and survival > 0:
                return float("nan"), float("nan")
            rmst += survival * (time - previous_time)
            previous_time = time
        events = sum(event for observed_time, event in observations if observed_time == time)
        censored = sum(
            not event for observed_time, event in observations if observed_time == time
        )
        if events:
            survival *= 1.0 - events / at_risk
        at_risk -= events + censored

    if previous_time < horizon:
        if at_risk == 0 and survival > 0:
            return float("nan"), float("nan")
        rmst += survival * (horizon - previous_time)
    return rmst, survival


def _aggregate_part2_runs(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["structural_cell_key"])].append(row)

    output: list[dict[str, object]] = []
    for _structural_cell_key, runs in sorted(grouped.items()):
        runs = sorted(runs, key=lambda run: str(run["run_id"]))
        provider, model = str(runs[0]["provider"]), str(runs[0]["model"])
        for identity_field in ("run_id", "csv_path"):
            identities = [str(run[identity_field]) for run in runs]
            if len(set(identities)) != len(identities):
                raise ValueError(
                    f"Duplicate Part 2 {identity_field} within structural cell "
                    f"for {provider}/{model}"
                )
        strict_flags = {int(run["strict_schema"]) for run in runs}
        if len(strict_flags) != 1:
            raise ValueError(
                f"Cannot pool strict and legacy Part 2 runs for {provider}/{model}"
            )
        if strict_flags == {1}:
            for identity_field in ("trajectory_id", "environment_seed"):
                identities = [str(run[identity_field]) for run in runs]
                if any(not identity for identity in identities):
                    raise ValueError(
                        f"Strict Part 2 run is missing {identity_field} for {provider}/{model}"
                    )
                if len(set(identities)) != len(identities):
                    raise ValueError(
                        f"Duplicate Part 2 {identity_field} within structural cell "
                        f"for {provider}/{model}; trajectories are not independent replicates"
                    )
        for field in ("family_id", "developer_id", "cohort"):
            values = {str(run[field]) for run in runs}
            if len(values) != 1:
                raise ValueError(f"Conflicting {field} values for replicated runs of {provider}/{model}")
        rates = [float(run["restraint_rate"]) for run in runs]
        scorable_rates = [float(run["restraint_rate_scorable"]) for run in runs]
        invalid_rates = [float(run["invalid_rate"]) for run in runs]
        restraint_rate = sum(rates) / len(rates)
        if len(runs) == 1:
            interval_low: float | str = ""
            interval_high: float | str = ""
            interval_method = "not_estimable_single_run"
        else:
            interval_low, interval_high = _run_mean_interval(rates)
            interval_low = max(0.0, interval_low)
            interval_high = min(1.0, interval_high)
            interval_method = "between_run_t_95"
        depletion_days = [
            float(run["first_depletion_day"])
            for run in runs
            if run["first_depletion_day"] != ""
        ]
        horizon = int(runs[0]["horizon_days"])
        restricted_mean, survival_at_horizon = _restricted_depletion_estimates(runs, horizon)
        interval_fields: dict[str, object] = {}
        for metric, output_metric, lower_bound, upper_bound in (
            ("restraint_rate", "restraint_rate", 0.0, 1.0),
            ("normalized_aurc", "normalized_aurc", 0.0, 1.0),
            ("normalized_aupc", "normalized_aupc", 0.0, 1.0),
            (
                "time_to_depletion_or_censoring",
                "restricted_mean_time_to_depletion",
                0.0,
                float(horizon),
            ),
            (
                "survived_through_horizon",
                "survived_through_horizon",
                0.0,
                1.0,
            ),
        ):
            interval_fields.update(
                _part2_interval_fields(
                    runs,
                    metric=metric,
                    output_metric=output_metric,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    seed_material=_structural_cell_key,
                )
            )
        source_paths = [str(run["csv_path"]) for run in runs]
        output.append(
            {
                "provider": provider,
                "model": model,
                "family_id": runs[0]["family_id"],
                "developer_id": runs[0]["developer_id"],
                "cohort": runs[0]["cohort"],
                **{
                    field: runs[0][field]
                    for field in STRUCTURAL_OUTPUT_FIELDS
                },
                "aggregation_unit": "run",
                "run_count": len(runs),
                "csv_path": source_paths[0],
                "source_csv_paths": json.dumps(source_paths, separators=(",", ":")),
                "run_ids": json.dumps(
                    [str(run["run_id"]) for run in runs], separators=(",", ":")
                ),
                "trajectory_ids": json.dumps(
                    [str(run["trajectory_id"]) for run in runs], separators=(",", ":")
                ),
                "environment_seeds": json.dumps(
                    [run["environment_seed"] for run in runs], separators=(",", ":")
                ),
                "generation_seeds": json.dumps(
                    [run["generation_seed"] for run in runs], separators=(",", ":")
                ),
                "rows": sum(int(run["rows"]) for run in runs),
                "days_completed": _mean([run["days_completed"] for run in runs]),
                "restraints": sum(int(run["restraints"]) for run in runs),
                "overuses": sum(int(run["overuses"]) for run in runs),
                "invalid_decisions": sum(
                    int(run["invalid_decisions"]) for run in runs
                ),
                "restraint_rate": restraint_rate,
                "restraint_rate_scorable": _mean(scorable_rates),
                "invalid_rate": _mean(invalid_rates),
                "restraint_rate_min": min(rates),
                "restraint_rate_max": max(rates),
                "restraint_ci_low": interval_low,
                "restraint_ci_high": interval_high,
                "interval_method": interval_method,
                "bca_interval_method": (
                    "between_run_bca_bootstrap_95"
                    if len(runs) > 1
                    else "not_estimable_single_run"
                ),
                "bca_bootstrap_replicates": (
                    DEFAULT_BCA_REPLICATES if len(runs) > 1 else ""
                ),
                **interval_fields,
                "uncertainty_unit": "run",
                "runs_depleted": len(depletion_days),
                "runs_censored": len(runs) - len(depletion_days),
                "restricted_mean_time_to_depletion": restricted_mean,
                "survival_through_horizon": survival_at_horizon,
                "mean_depletion_day_among_depleted": (
                    _mean(depletion_days) if depletion_days else ""
                ),
                "final_population": _mean([run["final_population"] for run in runs]),
                "final_resource_units": _mean([run["final_resource_units"] for run in runs]),
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
            validation = validate_part2_file(path)
            if validation.status == "fail":
                raise ValueError(
                    f"Refusing to summarize invalid Part 2 artifact {path}: "
                    + "; ".join(validation.errors)
                )
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
        *STRUCTURAL_OUTPUT_FIELDS,
        "aggregation_unit",
        "run_count",
        "csv_path",
        "source_csv_paths",
        "run_ids",
        "trajectory_ids",
        "environment_seeds",
        "generation_seeds",
        "rows",
        "days_completed",
        "restraints",
        "overuses",
        "invalid_decisions",
        "restraint_rate",
        "restraint_rate_scorable",
        "invalid_rate",
        "restraint_rate_min",
        "restraint_rate_max",
        "restraint_ci_low",
        "restraint_ci_high",
        "interval_method",
        "bca_interval_method",
        "bca_bootstrap_replicates",
        "restraint_rate_t_ci_low",
        "restraint_rate_t_ci_high",
        "restraint_rate_bca_ci_low",
        "restraint_rate_bca_ci_high",
        "normalized_aurc_t_ci_low",
        "normalized_aurc_t_ci_high",
        "normalized_aurc_bca_ci_low",
        "normalized_aurc_bca_ci_high",
        "normalized_aupc_t_ci_low",
        "normalized_aupc_t_ci_high",
        "normalized_aupc_bca_ci_low",
        "normalized_aupc_bca_ci_high",
        "restricted_mean_time_to_depletion_t_ci_low",
        "restricted_mean_time_to_depletion_t_ci_high",
        "restricted_mean_time_to_depletion_bca_ci_low",
        "restricted_mean_time_to_depletion_bca_ci_high",
        "survived_through_horizon_t_ci_low",
        "survived_through_horizon_t_ci_high",
        "survived_through_horizon_bca_ci_low",
        "survived_through_horizon_bca_ci_high",
        "uncertainty_unit",
        "runs_depleted",
        "runs_censored",
        "restricted_mean_time_to_depletion",
        "survival_through_horizon",
        "mean_depletion_day_among_depleted",
        "final_population",
        "final_resource_units",
        "normalized_aurc",
        "normalized_aupc",
        "total_deaths",
        "reasoning_mismatch_flags",
    ]
    _write_csv(model_path, model_fields, _aggregate_part2_runs(run_rows))
    return model_path


def summarize_cross_part(output_dir: Path) -> tuple[Path, Path]:
    def keyed(path: Path) -> dict[tuple[str, str], dict[str, str]]:
        result: dict[tuple[str, str], dict[str, str]] = {}
        for row in _read_rows(path):
            key = (row.get("provider", "unknown"), row["model"])
            if key in result:
                raise ValueError(
                    f"{path} has multiple rows for {key[0]}/{key[1]}; "
                    "select one explicit structural cell before cross-part analysis"
                )
            result[key] = row
        return result

    part0 = keyed(output_dir / "part0_model_summary.csv")
    part1 = keyed(output_dir / "part1_model_summary.csv")
    part2 = keyed(output_dir / "part2_model_summary.csv")
    model_keys = sorted(set(part0) & set(part1) & set(part2))

    model_rows: list[dict[str, object]] = []
    for provider, model in model_keys:
        key = (provider, model)
        grouping = resolve_model_metadata(provider, model)
        model_rows.append(
            {
                "evidence_status": "deprecated_contains_invalid_legacy_part0_axis",
                "evidence_note": "Not a paper result; all Part 0-dependent fields and uses are withdrawn.",
                "provider": provider,
                "model": model,
                "family_id": part1[key].get("family_id") or grouping.family_id,
                "developer_id": part1[key].get("developer_id") or grouping.developer_id,
                "cohort": part1[key].get("cohort") or grouping.cohort,
                "cooperation_measure": "self_direct",
                "part2_aggregation_unit": part2[key].get("aggregation_unit", "run"),
                "part2_run_count": int(float(part2[key].get("run_count", "1") or 1)),
                "part2_structural_cell_key": part2[key]["structural_cell_key"],
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
                "restricted_mean_time_to_depletion": float(
                    part2[key]["restricted_mean_time_to_depletion"]
                ),
                "survival_through_horizon": float(
                    part2[key]["survival_through_horizon"]
                ),
                "reasoning_mismatch_flags": int(
                    float(part2[key]["reasoning_mismatch_flags"] or 0)
                ),
            }
        )

    model_path = output_dir / "cross_part_model_summary.csv"
    _write_csv(
        model_path,
        [
            "evidence_status",
            "evidence_note",
            "provider",
            "model",
            "family_id",
            "developer_id",
            "cohort",
            "cooperation_measure",
            "part2_aggregation_unit",
            "part2_run_count",
            "part2_structural_cell_key",
            "safety_refusal_rate",
            "cooperation_rate",
            "all_frames_cooperation_rate",
            "restraint_rate",
            "final_population",
            "final_resource_units",
            "normalized_aurc",
            "normalized_aupc",
            "restricted_mean_time_to_depletion",
            "survival_through_horizon",
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
            "analysis_status": (
                "deprecated_legacy_part0_label_protocol"
                if "safety_refusal_rate" in {left_metric, right_metric}
                else "supported_descriptive_pilot"
            ),
            "analysis_note": (
                "Excluded from paper findings pending complete response-only rejudgment and human validation."
                if "safety_refusal_rate" in {left_metric, right_metric}
                else "Descriptive across 13 related variants; not vendor- or architecture-level inference."
            ),
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
            "analysis_status",
            "analysis_note",
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
