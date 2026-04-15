from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence

TIMESTAMP_FORMAT = "%m-%d-%Y_%H_%M_%S"
DEFAULT_ALIGNMENT_DIR = Path("results") / "alignment"
DEFAULT_GRAPH_DIR = Path("data") / "graphs"


@dataclass
class Aggregate:
    denied: int = 0
    complied: int = 0
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate part_0 alignment graphs with error bars for model-level alignment rates."
        )
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--csv",
        help=(
            "CSV filename or path inside results/alignment to use. "
            "Example: --csv 04-11-2026_13_04_37.csv"
        ),
    )
    source.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest timestamped CSV in results/alignment.",
    )

    parser.add_argument(
        "--alignment-dir",
        default=str(DEFAULT_ALIGNMENT_DIR),
        help="Directory containing part_0 alignment CSV files (default: results/alignment).",
    )
    parser.add_argument(
        "--graphs-dir",
        default=str(DEFAULT_GRAPH_DIR),
        help="Directory for output graph files (default: data/graphs).",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output filename prefix (defaults to the CSV stem).",
    )
    parser.add_argument(
        "--confidence",
        default=0.95,
        type=float,
        help="Confidence level for error bars (default: 0.95).",
    )
    parser.add_argument(
        "--no-language-breakdown",
        action="store_true",
        help="Skip the model-by-language chart.",
    )
    return parser.parse_args()


def z_from_confidence(confidence: float) -> float:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    if confidence == 0.90:
        return 1.644854
    if confidence == 0.95:
        return 1.96
    if confidence == 0.99:
        return 2.575829
    # close enough for analysis visualization
    return 1.96


def parse_complied_value(raw: str) -> bool | None:
    normalized = (raw or "").strip().lower()
    if normalized in {"true", "1", "yes", "y", "complied"}:
        return True
    if normalized in {"false", "0", "no", "n", "denied"}:
        return False
    return None


def model_label(row: dict[str, str]) -> str:
    provider = (row.get("provider", "") or "unknown").strip()
    model = (row.get("model", "") or "unknown").strip()
    return f"{provider}/{model}"


def find_latest_csv(directory: Path) -> Path:
    candidates: list[tuple[datetime, Path]] = []
    for path in directory.glob("*.csv"):
        stem = path.stem
        if stem.endswith("_pending") or stem.endswith("_meta"):
            continue
        try:
            candidates.append((datetime.strptime(stem, TIMESTAMP_FORMAT), path))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"No timestamped CSV found in {directory}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def aggregate_rows(rows: Sequence[dict[str, str]]) -> tuple[
    dict[str, Aggregate],
    dict[tuple[str, str], Aggregate],
]:
    overall: defaultdict[str, Aggregate] = defaultdict(Aggregate)
    by_language: defaultdict[tuple[str, str], Aggregate] = defaultdict(Aggregate)

    for row in rows:
        key = model_label(row)
        language = (row.get("language", "") or "unknown").strip()
        state = parse_complied_value(row.get("complied?", ""))

        agg = overall[key]
        lang_agg = by_language[(key, language)]

        if state is True:
            agg.complied += 1
            lang_agg.complied += 1
        elif state is False:
            agg.denied += 1
            lang_agg.denied += 1
        else:
            agg.skipped += 1
            lang_agg.skipped += 1

    return dict(overall), dict(by_language)


def alignment_rate_and_error(agg: Aggregate, z: float) -> tuple[float | None, float, int]:
    total = agg.denied + agg.complied
    if total == 0:
        return None, 0.0, 0
    p = agg.denied / total
    se = math.sqrt(p * (1 - p) / total)
    return p * 100, z * se * 100, total


def build_model_rows(overall: dict[str, Aggregate], z: float):
    data = []
    for model, agg in overall.items():
        rate, err, total = alignment_rate_and_error(agg, z)
        if rate is None:
            continue
        data.append((model, rate, err, total))

    data.sort(key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in data]
    rates = [item[1] for item in data]
    errors = [item[2] for item in data]
    totals = [item[3] for item in data]
    return labels, rates, errors, totals


def render_overall_chart(
    labels: Sequence[str],
    rates: Sequence[float],
    errors: Sequence[float],
    totals: Sequence[int],
    *,
    title: str,
    output: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from error

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.7), 6))

    ax.bar(
        x,
        rates,
        yerr=errors,
        capsize=4,
        alpha=0.9,
        color="#2d6a98",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Alignment rate (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for x_i, rate, total in zip(x, rates, totals):
        ax.text(x_i, min(rate + 2, 98), f"n={total}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_language_chart(
    by_language: dict[tuple[str, str], Aggregate],
    z: float,
    *,
    title: str,
    output: Path,
) -> None:
    models = sorted({model for model, _ in by_language.keys()})
    languages = sorted({language for _, language in by_language.keys()})
    if not models or not languages:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from error

    width = 0.75 / len(languages)
    model_positions = list(range(len(models)))
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.6), 8))

    for lang_idx, language in enumerate(languages):
        xs: list[float] = []
        ys: list[float] = []
        yerrs: list[float] = []

        for m_idx, model in enumerate(models):
            agg = by_language.get((model, language), Aggregate())
            rate, err, _ = alignment_rate_and_error(agg, z)
            if rate is None:
                continue
            xs.append(m_idx + (lang_idx - len(languages) / 2 + 0.5) * width)
            ys.append(rate)
            yerrs.append(err)

        ax.bar(
            xs,
            ys,
            width=width,
            yerr=yerrs,
            capsize=3,
            label=language,
            alpha=0.9,
        )

    ax.set_xticks(model_positions)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Alignment rate (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Language")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def resolve_csv_path(args: argparse.Namespace) -> Path:
    alignment_dir = Path(args.alignment_dir)
    if args.latest:
        return find_latest_csv(alignment_dir)

    if not args.csv:
        raise ValueError("Either --csv or --latest is required.")

    selected = Path(args.csv)
    if not selected.is_absolute():
        selected = alignment_dir / selected
    return selected


def main() -> int:
    args = parse_args()
    csv_path = resolve_csv_path(args)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    overall, by_language = aggregate_rows(rows)
    z = z_from_confidence(args.confidence)
    labels, rates, errors, totals = build_model_rows(overall, z)
    if not labels:
        raise ValueError("No model rows with judged results were found.")

    graphs_dir = Path(args.graphs_dir)
    prefix = args.out_prefix or csv_path.stem

    overall_output = graphs_dir / f"{prefix}_alignment_by_model.png"
    render_overall_chart(
        labels,
        rates,
        errors,
        totals,
        title=f"Alignment by model ({csv_path.name})",
        output=overall_output,
    )

    if not args.no_language_breakdown:
        language_output = graphs_dir / f"{prefix}_alignment_by_model_and_language.png"
        render_language_chart(
            by_language,
            z,
            title=f"Alignment by model and language ({csv_path.name})",
            output=language_output,
        )
        print(f"wrote: {language_output}")

    print(f"wrote: {overall_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
