#!/usr/bin/env python3
"""Recover a stale canonical ecology baseline into a fresh continuation directory."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_SELECTORS = [
    "cerebras:llama3.1-8b",
    "nvidia:deepseek-ai/deepseek-v3.2",
    "nvidia:moonshotai/kimi-k2-instruct-0905",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "If a canonical ecology baseline JSONL has gone stale before "
            "completion, build and optionally launch a fresh continuation run "
            "plus follow-on watcher."
        )
    )
    parser.add_argument(
        "baseline_results",
        help="Directory or JSONL file for the stale/incomplete baseline run.",
    )
    parser.add_argument(
        "--config",
        default="configs/part2/society_baseline.yaml",
        help="Baseline config to relaunch.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help="Repeatable model selector override. Defaults to the stable triplet cohort.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=15.0,
        help="Treat the monitored baseline as stale after this many quiet minutes.",
    )
    parser.add_argument(
        "--results-parent",
        default="results",
        help="Parent directory where the fresh continuation directory should be created.",
    )
    parser.add_argument(
        "--followon-root",
        default="results/canonical_ecology_followon",
        help="Results root for the queued reputation/event-stress continuation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forward dry-run mode to the recovered baseline and watcher.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the recovery commands without launching them.",
    )
    return parser.parse_args()


def load_live_status_module():
    module_path = Path(__file__).with_name("live_run_status.py")
    import importlib.util

    spec = importlib.util.spec_from_file_location("live_run_status_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load live status helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_baseline(path: str | Path, *, stale_minutes: float) -> dict[str, Any]:
    module = load_live_status_module()
    raw_path = Path(path)
    if raw_path.is_dir():
        candidates = sorted(raw_path.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No JSONL logs found in {raw_path}")
        log_path = max(candidates, key=lambda item: item.stat().st_mtime)
    else:
        log_path = raw_path
    summary = module.summarize_jsonl_log(log_path, stale_minutes=stale_minutes)
    if summary is None:
        raise RuntimeError(f"Unable to summarize baseline results at {log_path}")
    return summary


def needs_recovery(summary: dict[str, Any]) -> bool:
    total = summary.get("total_expected_trials")
    completed = summary.get("completed_trials")
    state = summary.get("state")
    return (
        state == "stale"
        and isinstance(total, int)
        and isinstance(completed, int)
        and completed < total
    )


def make_rollover_results_dir(results_parent: str | Path, *, name: str, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return Path(results_parent) / f"{name}_{timestamp}"


def build_resume_command(
    *,
    config_path: str,
    results_dir: str | Path,
    resume_log: str | Path,
    models: list[str],
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        config_path,
        "--results-dir",
        str(results_dir),
        "--resume-log",
        str(resume_log),
    ]
    for selector in models:
        command.extend(["--model", selector])
    if dry_run:
        command.append("--dry-run")
    return command


def build_followon_watcher_command(
    *,
    baseline_results: str | Path,
    followon_root: str | Path,
    models: list[str],
    dry_run: bool,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/continue_canonical_ecology_suite.py",
        str(baseline_results),
        "--results-root",
        str(followon_root),
        "--from-run",
        "reputation",
        "--poll-seconds",
        "120",
        "--refresh-packet",
    ]
    for selector in models:
        command.extend(["--model", selector])
    if dry_run:
        command.append("--dry-run")
    return command


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def spawn_background(command: list[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return int(process.pid)


def main() -> int:
    args = parse_args()
    models = args.model or list(DEFAULT_MODEL_SELECTORS)
    summary = summarize_baseline(args.baseline_results, stale_minutes=args.stale_minutes)

    if not needs_recovery(summary):
        print(
            "No recovery action taken: "
            f"state={summary.get('state')} "
            f"completed={summary.get('completed_trials')}/{summary.get('total_expected_trials')}"
        )
        return 0

    resume_log = summary["path"]
    results_dir = make_rollover_results_dir(args.results_parent, name=Path(summary["name"]).name)
    resume_command = build_resume_command(
        config_path=args.config,
        results_dir=results_dir,
        resume_log=resume_log,
        models=models,
        dry_run=args.dry_run,
    )
    watcher_command = build_followon_watcher_command(
        baseline_results=results_dir,
        followon_root=args.followon_root,
        models=models,
        dry_run=args.dry_run,
    )

    print(f"[resume] {format_command(resume_command)}")
    print(f"[watcher] {format_command(watcher_command)}")
    if args.print_only:
        return 0

    runner_pid = spawn_background(resume_command, log_path=results_dir / "launcher.log")
    watcher_pid = spawn_background(
        watcher_command,
        log_path=Path(args.followon_root) / "watch.log",
    )
    print(
        "launched: "
        f"runner_pid={runner_pid} watcher_pid={watcher_pid} "
        f"results_dir={results_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
