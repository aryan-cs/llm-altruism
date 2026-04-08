#!/usr/bin/env python3
"""Refresh the full local evidence packet for an in-progress ecology run."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the local summary, figures, casebook, and status JSON "
            "for a live ecology results directory."
        )
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing the live ecology JSONL artifact.",
    )
    parser.add_argument(
        "--log",
        help="Optional explicit JSONL log path. Defaults to the newest JSONL in the results directory.",
    )
    return parser.parse_args()


def resolve_log_path(results_dir: Path, explicit_log: str | None = None) -> Path:
    if explicit_log:
        return Path(explicit_log)
    candidates = sorted(results_dir.glob("*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No JSONL logs found in {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def packet_paths(results_dir: Path, log_path: Path) -> dict[str, Path]:
    logical_name = log_path.stem.rsplit("-", 1)[0] if "-" in log_path.stem else log_path.stem
    return {
        "summary_markdown": results_dir / "interim_summary.md",
        "summary_csv": results_dir / "interim_summary.csv",
        "figures_dir": results_dir / "monitoring_figures",
        "casebook_markdown": results_dir / f"{logical_name}-casebook.md",
        "status_json": results_dir / "live_status.json",
        "trial_snapshot_markdown": results_dir / "live_trial_snapshot.md",
        "trial_snapshot_csv": results_dir / "live_trial_snapshot.csv",
        "trial_snapshot_figure": results_dir / "live_trial_snapshot.png",
        "trial_comparison_markdown": results_dir / "live_trial_comparison.md",
    }


def load_live_status_module():
    module_path = Path(__file__).with_name("live_run_status.py")
    spec = importlib.util.spec_from_file_location("live_run_status_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load live status helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_command(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def write_trial_snapshot(outputs: dict[str, Path], live_status: dict[str, object]) -> None:
    trial_rows = list(live_status.get("trial_status_rows") or [])
    if not trial_rows:
        outputs["trial_snapshot_markdown"].write_text(
            "# Live Trial Snapshot\n\nNo trial rows observed.\n",
            encoding="utf-8",
        )
        outputs["trial_snapshot_csv"].write_text("", encoding="utf-8")
        return

    csv_columns = [
        "trial_id",
        "prompt_variant",
        "repetition",
        "completed",
        "latest_round_num",
        "alive_count",
        "total_agents",
        "alive_fraction",
        "population_regime",
        "first_loss_round_num",
        "first_death_round_num",
        "last_death_round_num",
        "stability_start_round_num",
        "rounds_since_last_death",
        "collapse_duration_rounds",
        "collapse_death_count",
        "plateau_duration_rounds",
        "final_survival_rate",
        "survival_rate",
        "alive_models",
    ]
    with outputs["trial_snapshot_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns)
        writer.writeheader()
        for row in trial_rows:
            csv_row = {column: row.get(column) for column in csv_columns}
            csv_row["alive_models"] = ", ".join(
                f"{model}: {count}" for model, count in (row.get("alive_models") or {}).items()
            )
            writer.writerow(csv_row)

    lines = [
        "# Live Trial Snapshot",
        "",
        "| trial | prompt_variant | status | latest_round | alive | regime | last_death | plateau_rounds | final_survival_rate | alive_models |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in trial_rows:
        status = "completed" if row.get("completed") else "active"
        alive_text = "n/a"
        if row.get("alive_count") is not None and row.get("total_agents") is not None:
            alive_text = f"{row['alive_count']}/{row['total_agents']}"
        final_survival_rate = row.get("final_survival_rate")
        final_survival_text = (
            f"{float(final_survival_rate):.4f}"
            if isinstance(final_survival_rate, (int, float))
            else ""
        )
        alive_models_text = ", ".join(
            f"{model}: {count}" for model, count in (row.get("alive_models") or {}).items()
        )
        lines.append(
            "| {trial_id} | {prompt_variant} | {status} | {latest_round_num} | {alive_text} | "
            "{population_regime} | {last_death_round_num} | {plateau_duration_rounds} | "
            "{final_survival_text} | {alive_models_text} |".format(
                trial_id=row.get("trial_id"),
                prompt_variant=row.get("prompt_variant"),
                status=status,
                latest_round_num=row.get("latest_round_num"),
                alive_text=alive_text,
                population_regime=row.get("population_regime"),
                last_death_round_num=row.get("last_death_round_num"),
                plateau_duration_rounds=row.get("plateau_duration_rounds"),
                final_survival_text=final_survival_text,
                alive_models_text=alive_models_text,
            )
        )
    outputs["trial_snapshot_markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_trial_snapshot_figure(outputs: dict[str, Path], live_status: dict[str, object]) -> None:
    trial_rows = list(live_status.get("trial_status_rows") or [])
    if not trial_rows:
        return

    ordered_rows = sorted(trial_rows, key=lambda row: (row.get("trial_id", 0)))
    labels = [f"{row.get('trial_id')}: {row.get('prompt_variant')}" for row in ordered_rows]
    alive_fraction = [
        float(row.get("alive_fraction"))
        if isinstance(row.get("alive_fraction"), (int, float))
        else 0.0
        for row in ordered_rows
    ]
    plateau_rounds = [
        float(row.get("plateau_duration_rounds"))
        if isinstance(row.get("plateau_duration_rounds"), (int, float))
        else 0.0
        for row in ordered_rows
    ]
    colors = ["#2a9d8f" if row.get("completed") else "#577590" for row in ordered_rows]
    status_labels = ["completed" if row.get("completed") else "active" for row in ordered_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    y_positions = list(range(len(ordered_rows)))

    axes[0].barh(y_positions, alive_fraction, color=colors, zorder=3)
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Alive fraction")
    axes[0].set_title("Latest alive fraction")
    axes[0].grid(axis="x", alpha=0.25, zorder=0)
    for index, row in enumerate(ordered_rows):
        alive_count = row.get("alive_count")
        total_agents = row.get("total_agents")
        alive_text = (
            f"{alive_count}/{total_agents}"
            if alive_count is not None and total_agents is not None
            else "n/a"
        )
        axes[0].text(
            min(0.98, alive_fraction[index] + 0.02),
            index,
            f"{alive_text} ({status_labels[index]})",
            va="center",
            ha="left",
            fontsize=10,
        )

    axes[1].barh(y_positions, plateau_rounds, color=colors, zorder=3)
    axes[1].set_yticks(y_positions)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Rounds")
    axes[1].set_title("Stable plateau duration")
    axes[1].grid(axis="x", alpha=0.25, zorder=0)
    upper = max(1.0, max(plateau_rounds) * 1.15)
    axes[1].set_xlim(0, upper)
    for index, row in enumerate(ordered_rows):
        last_death = row.get("last_death_round_num")
        last_death_text = "" if last_death is None else f"last death {last_death}"
        axes[1].text(
            plateau_rounds[index] + upper * 0.02,
            index,
            last_death_text,
            va="center",
            ha="left",
            fontsize=10,
        )

    fig.suptitle("Live trial comparison snapshot", fontsize=16)
    fig.tight_layout()
    fig.savefig(outputs["trial_snapshot_figure"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_trial_comparison(outputs: dict[str, Path], live_status: dict[str, object]) -> None:
    trial_rows = sorted(
        list(live_status.get("trial_status_rows") or []),
        key=lambda row: (row.get("trial_id", 0)),
    )
    if len(trial_rows) < 2:
        outputs["trial_comparison_markdown"].write_text(
            "# Live Trial Comparison\n\nNot enough trial rows observed for a comparison.\n",
            encoding="utf-8",
        )
        return

    completed_rows = [row for row in trial_rows if row.get("completed")]
    active_rows = [row for row in trial_rows if not row.get("completed")]
    reference = completed_rows[-1] if completed_rows else trial_rows[0]
    candidate = active_rows[0] if active_rows else trial_rows[-1]

    def fmt_delta(current: object, baseline: object, *, digits: int = 4) -> str:
        if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
            return ""
        delta = float(current) - float(baseline)
        return f"{delta:+.{digits}f}"

    lines = [
        "# Live Trial Comparison",
        "",
        f"- reference trial: `{reference.get('trial_id')}` `{reference.get('prompt_variant')}`",
        f"- comparison trial: `{candidate.get('trial_id')}` `{candidate.get('prompt_variant')}`",
        "",
        "| metric | reference | comparison | delta |",
        "| --- | --- | --- | --- |",
    ]

    metrics = [
        ("alive_fraction", "Alive fraction", 4),
        ("alive_count", "Alive count", 0),
        ("plateau_duration_rounds", "Plateau rounds", 0),
        ("collapse_death_count", "Collapse deaths", 0),
        ("public_food", "Latest public food", 1),
        ("public_water", "Latest public water", 1),
        ("average_health", "Latest average health", 4),
        ("average_energy", "Latest average energy", 4),
        ("trade_volume", "Latest trade volume", 0),
    ]
    for key, label, digits in metrics:
        reference_value = reference.get(key)
        comparison_value = candidate.get(key)
        reference_text = (
            f"{float(reference_value):.{digits}f}" if isinstance(reference_value, (int, float)) else ""
        )
        comparison_text = (
            f"{float(comparison_value):.{digits}f}" if isinstance(comparison_value, (int, float)) else ""
        )
        lines.append(
            f"| {label} | {reference_text} | {comparison_text} | {fmt_delta(comparison_value, reference_value, digits=digits)} |"
        )

    outputs["trial_comparison_markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolve_log_path(results_dir, args.log)
    outputs = packet_paths(results_dir, log_path)

    run_command(
        [
            sys.executable,
            "scripts/paper_summary.py",
            str(results_dir),
            "--markdown",
            str(outputs["summary_markdown"]),
            "--csv",
            str(outputs["summary_csv"]),
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/paper_figures.py",
            str(results_dir),
            "--output-dir",
            str(outputs["figures_dir"]),
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/ecology_casebook.py",
            str(log_path),
            "--output",
            str(outputs["casebook_markdown"]),
        ]
    )

    live_status = load_live_status_module().summarize_jsonl_log(log_path)
    outputs["status_json"].write_text(json.dumps(live_status, indent=2), encoding="utf-8")
    write_trial_snapshot(outputs, live_status)
    write_trial_snapshot_figure(outputs, live_status)
    write_trial_comparison(outputs, live_status)

    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
