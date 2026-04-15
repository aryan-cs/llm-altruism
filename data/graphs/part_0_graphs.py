from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
from datetime import datetime


ALIGNMENT_RESULTS_DIR = Path("results") / "alignment"
GRAPH_OUTPUT_DIR = Path("data") / "graphs"
TIMESTAMP_FORMAT = "%m-%d-%Y_%H_%M_%S"


class BoolParseError(ValueError):
    pass


class ModelAggregate(NamedTuple):
    provider: str
    model: str
    denied: int
    complied: int
    skipped: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate part_0 alignment summary graphs with error bars for model-level "
            "alignment rates."
        )
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--csv",
        help="CSV filename or path in results/alignment to plot. Example: 04-11-2026_13_04_37.csv",
    )
    source_group.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest timestamped CSV file in results/alignment.",
    )

    parser.add_argument(
        "--alignment-dir",
        default=str(ALIGNMENT_RESULTS_DIR),
        help="Directory containing part_0 alignment CSVs (default: results/alignment)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(GRAPH_OUTPUT_DIR),
        help="Directory to write graph files (default: data/graphs)",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Optional filename prefix for outputs. Defaults to csv stem.
",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for error bars (default: 0.95).",
    )
    parser.add_argument(
        "--no-language-breakdown",
        action="store_true",
        help="Skip the per-language by model breakdown chart.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y", "denied", "false", "0", "no", "n", "complied"}:
        # accepted raw markers used by part_0 and common accidental inversions
        return normalized in {"true", "1", "yes", "y", "complied"}
    if normalized == "":
        raise BoolParseError("empty")
    raise BoolParseError(f"unrecognized flag: {value!r}")


def parse_model_label(row: dict[str, str]) -> str:
    provider = (row.get("provider", "").strip() or "unknown")
    model = (row.get("model", "").strip() or "unknown")
    return f"{provider}/{model}"


def z_from_confidence(confidence: float) -> float:
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0, 1)")
    if confidence == 0.95:
        return 1.96
    if confidence == 0.90:
        return 1.644854
    if confidence == 0.99:
        return 2.575829
    # normal approximation for arbitrary confidence
    # good enough for graphing. exact inversion is unnecessary for this analysis utility.
    return 1.96


def safe_bool_ratio(
    denied: int,
    complied: int,
) -> float:
    total = denied + complied
    if total == 0:
        return float("nan")
    return denied / total


def normal_ci_half_width(p: float, n: int, *, z: float) -> float:
    if n <= 0:
        return float("nan")
    if p < 0.0 or p > 1.0:
        raise ValueError("p must be a probability")
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    half = z * se
    return half


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def aggregate_by_model(
    rows: Sequence[dict[str, str]],
) -> tuple[dict[str, ModelAggregate], dict[tuple[str, str], ModelAggregate]]:
    overall: defaultdict[str, ModelAggregate] = defaultdict(
        lambda: ModelAggregate("", "", 0, 0, 0)
    )
    by_language: defaultdict[tuple[str, str], ModelAggregate] = defaultdict(
        lambda: ModelAggregate("", "", 0, 0, 0)
    )

    for row in rows:
        provider = (row.get("provider", "") or "unknown").strip()
        model = (row.get("model", "") or "unknown").strip()
        language = (row.get("language", "") or "unknown").strip()
        key = f"{provider}/{model}"

        base = overall[key]
        lang_key = (key, language)
        lang_agg = by_language[lang_key]

        value = row.get("complied?", "")
        if value is None:
            value = ""
        norm = value.strip().lower()

        try:
            complied = parse_bool(norm)
        except BoolParseError:
            # treat unknown markers as skipped
            overall[key] = ModelAggregate(
                provider,
                model,
                base.denied,
                base.complied,
                base.skipped + 1,
            )
            by_language[lang_key] = ModelAggregate(
                provider,
                model,
                lang_agg.denied,
                lang_agg.complied,
                lang_agg.skipped + 1,
            )
            continue

        if complied:
            overall[key] = ModelAggregate(
                provider,
                model,
                base.denied,
                base.complied + 1,
                base.skipped,
            )
            by_language[lang_key] = ModelAggregate(
                provider,
                model,
                lang_agg.denied,
                lang_agg.complied + 1,
                lang_agg.skipped,
            )
        else:
            overall[key] = ModelAggregate(
                provider,
                model,
                base.denied + 1,
                base.complied,
                base.skipped,
            )
            by_language[lang_key] = ModelAggregate(
                provider,
                model,
                lang_agg.denied + 1,
                lang_agg.complied,
                lang_agg.skipped,
            )
    return dict(overall), dict(by_language)


def build_model_summary(aggregates: dict[str, ModelAggregate], *, z: float) -> tuple[list[str], list[float], list[float], list[int]]:
    labels: list[str] = []
    means: list[float] = []
    errors: list[float] = []
    totals: list[int] = []

    items = sorted(
        aggregates.items(),
        key=lambda item: (
            item[1][0],
            item[0],
        ),
    )

    for model_name, agg in items:
        labels.append(model_name)
        p = safe_bool_ratio(agg.denied, agg.complied)
        total = agg.denied + agg.complied
        totals.append(total)
        if math.isnan(p):
            means.append(float("nan"))
            errors.append(float("nan"))
        else:
            means.append(p * 100)
            errors.append(normal_ci_half_width(p, total, z=z) * 100)

    return labels, means, errors, totals


def sort_models_for_plot(
    labels: Sequence[str],
    means: Sequence[float],
) -> tuple[list[str], list[float]]:
    pairs = [(label, mean) for label, mean in zip(labels, means)]
    pairs = [pair for pair in pairs if not math.isnan(pair[1])]
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    return [item[0] for item in ordered], [item[1] for item in ordered]


def find_latest_csv(alignment_dir: Path) -> Path:
    candidate_paths: list[tuple[datetime, Path]] = []
    for path in alignment_dir.glob("*.csv"):
        stem = path.stem
        # alignment results include pending/maybe meta siblings; keep plain csvs with timestamp-like stems
        if stem.endswith("_pending") or stem.endswith("_meta"):
            continue
        try:
            ts = datetime.strptime(stem, TIMESTAMP_FORMAT)
            candidate_paths.append((ts, path))
        except ValueError:
            continue

    if not candidate_paths:
        raise ValueError(f"No timestamped CSV found in {alignment_dir}")

    candidate_paths.sort()
    return candidate_paths[-1][1]


def resolve_csv_path(args: argparse.Namespace) -> Path:
    alignment_dir = Path(args.alignment_dir)
    alignment_dir.mkdir(parents=True, exist_ok=True)

    if args.latest:
        return find_latest_csv(alignment_dir)

    if args.csv is None:
        raise ValueError("Either --csv or --latest is required.")

    candidate = Path(args.csv)
    if not candidate.is_absolute():
        candidate = alignment_dir / candidate
    return candidate


def plot_model_chart(
    labels: Sequence[str],
    means: Sequence[float],
    errors: Sequence[float],
    totals: Sequence[int],
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required to render graphs. Install it with: pip install matplotlib"
        ) from error

    import numpy as np

    fig, ax = plt.subplots(figsize=(min(20, max(8, len(labels) * 0.65)), 6))

    x = np.arange(len(labels))
    valid_mask = [not math.isnan(m) for m in means]
    valid_x = [x[i] for i, keep in enumerate(valid_mask) if keep]
    valid_means = [means[i] for i, keep in enumerate(valid_mask) if keep]
    valid_errors = [errors[i] for i, keep in enumerate(valid_mask) if keep]
    valid_labels = [labels[i] for i, keep in enumerate(valid_mask) if keep]
    valid_totals = [totals[i] for i, keep in enumerate(valid_mask) if keep]

    if valid_x:
        ax.bar(
            valid_x,
            valid_means,
            yerr=valid_errors,
            capsize=4,
            color="#3f5d8f",
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(valid_x)
        ax.set_xticklabels(valid_labels, rotation=45, ha="right")
        for xi, value, total in zip(valid_x, valid_means, valid_totals):
            ax.text(
                xi,
                value + 2,
                f"n={total}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Alignment rate (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(50, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_language_breakdown(
    by_language: dict[tuple[str, str], ModelAggregate],
    output_path: Path,
    title: str,
    *,
    confidence_z: float,
) -> None:
    language_groups: defaultdict[str, dict[str, ModelAggregate]] = defaultdict(dict)
    for (model_key, language), agg in by_language.items():
        language_groups[language][model_key] = agg

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required to render graphs. Install it with: pip install matplotlib"
        ) from error

    import numpy as np

    valid_languages = [lang for lang, models in language_groups.items() if models]
    if not valid_languages:
        return

    # keep stable ordering for readability
    valid_languages = sorted(valid_languages)
    models = sorted({model for models_by_lang in language_groups.values() for model in models_by_lang})

    means_by_language = []
    errors_by_language = []
    for language in valid_languages:
        agg_map = language_groups[language]
        lang_means: list[float] = []
        lang_errors: list[float] = []
        for model in models:
            agg = agg_map.get(model, ModelAggregate("", "", 0, 0, 0))
            p = safe_bool_ratio(agg.denied, agg.complied)
            if math.isnan(p):
                lang_means.append(float("nan"))
                lang_errors.append(0.0)
            else:
                lang_means.append(p * 100)
                lang_errors.append(normal_ci_half_width(p, agg.denied + agg.complied, z=confidence_z) * 100)
        means_by_language.append(lang_means)
        errors_by_language.append(lang_errors)

    bar_positions = np.arange(len(models))
    total_languages = len(valid_languages)
    width = 0.75 / max(total_languages, 1)

    fig, ax = plt.subplots(figsize=(min(24, max(8, len(models) * 0.5)), 8))

    for idx, language in enumerate(valid_languages):
        means = means_by_language[idx]
        errs = errors_by_language[idx]
        offsets = (idx - total_languages / 2 + 0.5) * width + 1e-9
        x = bar_positions + offsets

        clean_values = []
        clean_errs = []
        clean_x = []
        clean_labels = []
        for i, value in enumerate(means):
            if math.isnan(value):
                continue
            clean_x.append(x[i])
            clean_values.append(value)
            clean_errs.append(errs[i])
            clean_labels.append(models[i])

        if clean_values:
            ax.bar(
                clean_x,
                clean_values,
                width,
                yerr=clean_errs,
                label=language,
                capsize=3,
                alpha=0.85,
            )

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Alignment rate (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Language")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def filter_zero_model_counts(aggregates: dict[str, ModelAggregate]) -> dict[str, ModelAggregate]:
    return {
        key: agg
        for key, agg in aggregates.items()
        if agg.denied + agg.complied > 0
    }


def main() -> int:
    args = parse_args()
    csv_path = resolve_csv_path(args)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    overall, by_language = aggregate_by_model(rows)
    overall = filter_zero_model_counts(overall)

    z = z_from_confidence(args.confidence)
    labels, means, errors, totals = build_model_summary(overall, z=z)
    labels, means = sort_models_for_plot(labels, means)
    # reorder totals and errors to matched sorted labels
    reordered = {label: (mean, error, totals[idx]) for idx, label in enumerate(labels) for mean, error in [(means[idx], errors[idx])] }
    means = [reordered[label][0] for label in labels]
    errors = [reordered[label][1] for label in labels]
    totals = [reordered[label][2] for label in labels]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.out_prefix or csv_path.stem

    overall_output = output_dir / f"{prefix}_alignment_by_model.png"
    plot_model_chart(
        labels,
        means,
        errors,
        totals,
        overall_output,
        f"Alignment by model ({csv_path.name})",
    )

    if not args.no_language_breakdown:
        language_output = output_dir / f"{prefix}_alignment_by_model_language.png"
        plot_language_breakdown(
            by_language,
            language_output,
            f"Alignment by model and language ({csv_path.name})",
            confidence_z=z,
        )

    print(f"wrote: {overall_output}")
    if not args.no_language_breakdown:
        print(f"wrote: {language_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
