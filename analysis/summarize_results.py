from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from experiments.part0 import part_0
from experiments.part1 import part_1
from experiments.part2 import part_2
from analysis.validation import _csv_paths, _flag_reasoning_misunderstanding

RAW_DIR = Path("data") / "raw"
TABLES_DIR = Path("data") / "analysis" / "tables"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [{key: value or "" for key, value in row.items()} for row in csv.DictReader(handle)]


def _model_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("provider", "unknown"), row.get("model", "unknown")


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
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return float("nan")
    return numerator / math.sqrt(x_var * y_var)


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(indexed):
        end = start + 1
        while end < len(indexed) and indexed[end][0] == indexed[start][0]:
            end += 1
        average_rank = (start + end - 1) / 2 + 1
        for _, index in indexed[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _spearman_correlation(xs: list[float], ys: list[float]) -> float:
    return _pearson_correlation(_rank_values(xs), _rank_values(ys))


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


def summarize_part0(raw_dir: Path, output_dir: Path) -> Path:
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for path in _csv_paths(raw_dir / "part_0"):
        for row in _read_rows(path):
            verdict = part_0._verdict_from_complied_value(row.get("complied?", ""))
            if not verdict and row.get("verdict"):
                verdict = row["verdict"].strip().lower()
            key = _model_key(row)
            counts[key][verdict or "skipped"] += 1

    rows: list[dict[str, object]] = []
    for (provider, model), counter in sorted(counts.items()):
        denied = counter["denied"]
        complied = counter["complied"]
        skipped = counter["skipped"]
        total = denied + complied
        low, high = _wilson_interval(denied, total)
        rows.append(
            {
                "provider": provider,
                "model": model,
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
    _write_csv(
        path,
        [
            "provider",
            "model",
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
    return path


def _part1_is_cooperative(row: dict[str, str]) -> bool:
    return row.get("action", "") in {"COOPERATE", "RESTRAIN"}


def summarize_part1(raw_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    model_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    dimension_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for path in _csv_paths(raw_dir / "part_1"):
        for row in _read_rows(path):
            key = _model_key(row)
            outcome = "cooperate" if _part1_is_cooperative(row) else "defect_or_overuse"
            model_counts[key][outcome] += 1
            for dimension in ("game", "frame", "domain", "presentation"):
                dimension_counts[(key[1], dimension, row.get(dimension, ""))][outcome] += 1

    model_rows: list[dict[str, object]] = []
    for (provider, model), counter in sorted(model_counts.items()):
        cooperate = counter["cooperate"]
        total = cooperate + counter["defect_or_overuse"]
        low, high = _wilson_interval(cooperate, total)
        model_rows.append(
            {
                "provider": provider,
                "model": model,
                "total": total,
                "cooperative": cooperate,
                "non_cooperative": counter["defect_or_overuse"],
                "cooperation_rate": cooperate / total if total else "",
                "wilson_low": low if total else "",
                "wilson_high": high if total else "",
            }
        )

    dimension_rows: list[dict[str, object]] = []
    for (model, dimension, value), counter in sorted(dimension_counts.items()):
        cooperate = counter["cooperate"]
        total = cooperate + counter["defect_or_overuse"]
        dimension_rows.append(
            {
                "model": model,
                "dimension": dimension,
                "value": value,
                "total": total,
                "cooperative": cooperate,
                "cooperation_rate": cooperate / total if total else "",
            }
        )

    model_path = output_dir / "part1_model_summary.csv"
    dimension_path = output_dir / "part1_dimension_summary.csv"
    _write_csv(
        model_path,
        [
            "provider",
            "model",
            "total",
            "cooperative",
            "non_cooperative",
            "cooperation_rate",
            "wilson_low",
            "wilson_high",
        ],
        model_rows,
    )
    _write_csv(
        dimension_path,
        ["model", "dimension", "value", "total", "cooperative", "cooperation_rate"],
        dimension_rows,
    )
    return model_path, dimension_path


def summarize_part2(raw_dir: Path, output_dir: Path) -> Path:
    rows_out: list[dict[str, object]] = []
    for path in _csv_paths(raw_dir / "part_2"):
        rows = _read_rows(path)
        if not rows:
            continue
        provider, model = _model_key(rows[0])
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
        rows_out.append(
            {
                "provider": provider,
                "model": model,
                "csv_path": str(path),
                "rows": len(rows),
                "days_completed": final_day,
                "restraints": actions["RESTRAIN"],
                "overuses": actions["OVERUSE"],
                "restraint_rate": actions["RESTRAIN"] / total if total else "",
                "wilson_low": low if total else "",
                "wilson_high": high if total else "",
                "first_depletion_day": min(depletion_days) if depletion_days else "",
                "final_population": final_row.get("population_end", ""),
                "final_resource_units": final_row.get("resource_units_remaining", ""),
                "resource_capacity": final_row.get("resource_capacity", ""),
                "total_deaths": sum(
                    int(grouped[-1].get("deaths", "0") or 0)
                    for grouped in day_rows.values()
                    if grouped
                ),
                "reasoning_mismatch_flags": sum(1 for row in rows if _flag_reasoning_misunderstanding(row)),
            }
        )

    path = output_dir / "part2_model_summary.csv"
    _write_csv(
        path,
        [
            "provider",
            "model",
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
            "total_deaths",
            "reasoning_mismatch_flags",
        ],
        rows_out,
    )
    return path


def summarize_cross_part(output_dir: Path) -> tuple[Path, Path]:
    part0 = {row["model"]: row for row in _read_rows(output_dir / "part0_model_summary.csv")}
    part1 = {row["model"]: row for row in _read_rows(output_dir / "part1_model_summary.csv")}
    part2 = {row["model"]: row for row in _read_rows(output_dir / "part2_model_summary.csv")}
    models = sorted(set(part0) & set(part1) & set(part2))

    model_rows: list[dict[str, object]] = []
    for model in models:
        first_depletion = part2[model].get("first_depletion_day", "")
        model_rows.append(
            {
                "model": model,
                "safety_refusal_rate": float(part0[model]["safety_refusal_rate"]),
                "cooperation_rate": float(part1[model]["cooperation_rate"]),
                "restraint_rate": float(part2[model]["restraint_rate"]),
                "final_population": int(part2[model]["final_population"] or 0),
                "final_resource_units": int(part2[model]["final_resource_units"] or 0),
                "first_depletion_day": int(first_depletion) if first_depletion else "",
                "reasoning_mismatch_flags": int(part2[model]["reasoning_mismatch_flags"] or 0),
            }
        )

    model_path = output_dir / "cross_part_model_summary.csv"
    _write_csv(
        model_path,
        [
            "model",
            "safety_refusal_rate",
            "cooperation_rate",
            "restraint_rate",
            "final_population",
            "final_resource_units",
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
    for left_metric, right_metric in pairs:
        xs = metric_values[left_metric]
        ys = metric_values[right_metric]
        pearson_low, pearson_high = _bootstrap_ci(xs, ys, _pearson_correlation)
        spearman_low, spearman_high = _bootstrap_ci(xs, ys, _spearman_correlation)
        correlation_rows.append(
            {
                "metric_x": left_metric,
                "metric_y": right_metric,
                "n_models": len(model_rows),
                "pearson_r": _pearson_correlation(xs, ys),
                "pearson_bootstrap_low": pearson_low,
                "pearson_bootstrap_high": pearson_high,
                "spearman_r": _spearman_correlation(xs, ys),
                "spearman_bootstrap_low": spearman_low,
                "spearman_bootstrap_high": spearman_high,
            }
        )

    correlation_path = output_dir / "cross_part_correlations.csv"
    _write_csv(
        correlation_path,
        [
            "metric_x",
            "metric_y",
            "n_models",
            "pearson_r",
            "pearson_bootstrap_low",
            "pearson_bootstrap_high",
            "spearman_r",
            "spearman_bootstrap_low",
            "spearman_bootstrap_high",
        ],
        correlation_rows,
    )
    return model_path, correlation_path


def summarize_all(raw_dir: Path = RAW_DIR, output_dir: Path = TABLES_DIR) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [summarize_part0(raw_dir, output_dir)]
    outputs.extend(summarize_part1(raw_dir, output_dir))
    outputs.append(summarize_part2(raw_dir, output_dir))
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
