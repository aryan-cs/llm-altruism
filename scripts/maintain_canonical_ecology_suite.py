#!/usr/bin/env python3
"""Check the canonical ecology baseline and trigger recovery if it has stalled."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a canonical ecology baseline run, write a machine-readable "
            "maintenance status file, and trigger stale-run recovery when needed."
        )
    )
    parser.add_argument(
        "baseline_results",
        help="Directory or JSONL file for the baseline run to supervise.",
    )
    parser.add_argument(
        "--followon-root",
        default="results/canonical_ecology_followon",
        help="Results root for queued reputation/event-stress runs.",
    )
    parser.add_argument(
        "--results-parent",
        default="results",
        help="Parent directory for fresh continuation runs if recovery is needed.",
    )
    parser.add_argument(
        "--config",
        default="configs/part2/society_baseline.yaml",
        help="Baseline config path forwarded to the recovery script.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help="Repeatable model selector override forwarded to recovery if needed.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=15.0,
        help="Treat the baseline as stale after this many quiet minutes.",
    )
    parser.add_argument(
        "--status-file",
        help="Optional explicit maintenance status JSON path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forward dry-run mode to the recovery script.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the recovery command without launching it.",
    )
    return parser.parse_args()


def _load_module(filename: str, module_name: str):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_baseline(path: str | Path, *, stale_minutes: float) -> dict[str, Any]:
    live_status = _load_module("live_run_status.py", "live_run_status_module")
    raw_path = Path(path)
    if raw_path.is_dir():
        candidates = sorted(raw_path.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No JSONL logs found in {raw_path}")
        log_path = max(candidates, key=lambda item: item.stat().st_mtime)
    else:
        log_path = raw_path
    summary = live_status.summarize_jsonl_log(log_path, stale_minutes=stale_minutes)
    if summary is None:
        raise RuntimeError(f"Unable to summarize baseline results at {log_path}")
    return summary


def build_recovery_command(
    *,
    baseline_results: str | Path,
    followon_root: str | Path,
    results_parent: str | Path,
    config_path: str,
    models: list[str],
    stale_minutes: float,
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/recover_canonical_ecology_baseline.py",
        str(baseline_results),
        "--followon-root",
        str(followon_root),
        "--results-parent",
        str(results_parent),
        "--config",
        config_path,
        "--stale-minutes",
        str(stale_minutes),
    ]
    for selector in models:
        command.extend(["--model", selector])
    if dry_run:
        command.append("--dry-run")
    return command


def status_path(args: argparse.Namespace) -> Path:
    if args.status_file:
        return Path(args.status_file)
    return Path(args.followon_root) / "maintenance_status.json"


def write_status(
    *,
    path: Path,
    baseline_summary: dict[str, Any],
    recovery_needed: bool,
    recovery_command: list[str],
    recovery_returncode: int | None,
) -> None:
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "recovery_needed": recovery_needed,
        "recovery_command": recovery_command,
        "recovery_returncode": recovery_returncode,
        "baseline_summary": baseline_summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary = summarize_baseline(args.baseline_results, stale_minutes=args.stale_minutes)

    recover_module = _load_module(
        "recover_canonical_ecology_baseline.py",
        "recover_canonical_ecology_baseline_module",
    )
    recovery_needed = bool(recover_module.needs_recovery(summary))
    command = build_recovery_command(
        baseline_results=args.baseline_results,
        followon_root=args.followon_root,
        results_parent=args.results_parent,
        config_path=args.config,
        models=args.model,
        stale_minutes=args.stale_minutes,
        dry_run=args.dry_run,
    )

    recovery_returncode: int | None = None
    if recovery_needed:
        print(f"[recover] {' '.join(command)}")
        if not args.print_only:
            completed = subprocess.run(command, cwd=ROOT, check=False)
            recovery_returncode = int(completed.returncode)
    else:
        print(
            "no recovery needed: "
            f"state={summary.get('state')} "
            f"completed={summary.get('completed_trials')}/{summary.get('total_expected_trials')}"
        )

    write_status(
        path=status_path(args),
        baseline_summary=summary,
        recovery_needed=recovery_needed,
        recovery_command=command,
        recovery_returncode=recovery_returncode,
    )
    return 0 if recovery_returncode in {None, 0} else recovery_returncode


if __name__ == "__main__":
    raise SystemExit(main())
