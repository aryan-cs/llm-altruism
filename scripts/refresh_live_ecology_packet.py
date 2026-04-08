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

    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
