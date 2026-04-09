#!/usr/bin/env python3
"""Wait for a live baseline run to finish, then launch the remaining ecology suite."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def project_python() -> str:
    candidate = ROOT / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor a live baseline ecology results directory and launch the "
            "remaining canonical suite stages once the baseline is complete."
        )
    )
    parser.add_argument(
        "baseline_results",
        help="Directory or JSONL file for the active baseline run to monitor.",
    )
    parser.add_argument(
        "--results-root",
        default="results/canonical_ecology_followon",
        help="Root directory for the follow-on reputation and event-stress outputs.",
    )
    parser.add_argument(
        "--from-run",
        choices=["reputation", "event-stress"],
        default="reputation",
        help="Suite stage to launch once the monitored baseline is complete.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help="Repeatable model selector override forwarded to the suite runner.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=180.0,
        help="How long to sleep between baseline status checks.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=15.0,
        help="Treat the monitored baseline as stale after this many quiet minutes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Forward dry-run mode to the suite runner.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the command that would be launched and exit.",
    )
    parser.add_argument(
        "--refresh-packet",
        action="store_true",
        help=(
            "Refresh the monitored baseline's local summary/figure packet on each poll "
            "using scripts/refresh_live_ecology_packet.py."
        ),
    )
    return parser.parse_args()


def load_live_status_module():
    module_path = Path(__file__).with_name("live_run_status.py")
    spec = importlib.util.spec_from_file_location("live_run_status_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load live status helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_baseline(
    baseline_results: str | Path,
    *,
    stale_minutes: float,
) -> dict[str, Any]:
    module = load_live_status_module()
    path = Path(baseline_results)
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No JSONL logs found in {path}")
        log_path = max(candidates, key=lambda item: item.stat().st_mtime)
    else:
        log_path = path
    summary = module.summarize_jsonl_log(log_path, stale_minutes=stale_minutes)
    if summary is None:
        raise RuntimeError(f"Unable to summarize baseline results at {log_path}")
    return summary


def baseline_is_complete(summary: dict[str, Any]) -> bool:
    total = summary.get("total_expected_trials")
    completed = summary.get("completed_trials")
    return isinstance(total, int) and total > 0 and isinstance(completed, int) and completed >= total


def build_followon_command(
    *,
    results_root: str,
    from_run: str,
    models: list[str],
    dry_run: bool,
) -> list[str]:
    command = [
        project_python(),
        "scripts/run_canonical_ecology_suite.py",
        "--from-run",
        from_run,
        "--results-root",
        results_root,
    ]
    for selector in models:
        command.extend(["--model", selector])
    if dry_run:
        command.append("--dry-run")
    return command


def build_refresh_command(results_dir: str | Path) -> list[str]:
    return [
        project_python(),
        "scripts/refresh_live_ecology_packet.py",
        str(results_dir),
    ]


def watcher_status_path(results_root: str | Path) -> Path:
    return Path(results_root) / "watch_status.json"


def write_watcher_status(
    *,
    results_root: str | Path,
    baseline_results: str | Path,
    summary: dict[str, Any],
    followon_command: list[str],
    watcher_state: str,
) -> Path:
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "watcher_state": watcher_state,
        "baseline_results": str(baseline_results),
        "followon_command": followon_command,
        "baseline_summary": summary,
    }
    output_path = watcher_status_path(results_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def refresh_packet(results_dir: str | Path) -> None:
    completed = subprocess.run(build_refresh_command(results_dir), cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Packet refresh failed for {results_dir} with exit code {completed.returncode}"
        )


def wait_for_baseline_completion(
    baseline_results: str | Path,
    *,
    results_root: str | Path,
    followon_command: list[str],
    poll_seconds: float,
    stale_minutes: float,
    refresh_results_packet: bool,
) -> dict[str, Any]:
    while True:
        summary = summarize_baseline(baseline_results, stale_minutes=stale_minutes)
        if refresh_results_packet:
            refresh_packet(baseline_results)
        watcher_state = "waiting"
        if baseline_is_complete(summary):
            watcher_state = "baseline_complete"
        elif summary.get("state") == "stale":
            watcher_state = "baseline_stale"
        write_watcher_status(
            results_root=results_root,
            baseline_results=baseline_results,
            summary=summary,
            followon_command=followon_command,
            watcher_state=watcher_state,
        )
        if baseline_is_complete(summary):
            return summary
        if summary.get("state") == "stale":
            latest = summary.get("latest_event_timestamp")
            raise RuntimeError(
                f"Monitored baseline became stale before completion: {baseline_results} "
                f"(latest event {latest})"
            )
        print(
            "waiting: "
            f"completed={summary.get('completed_trials')}/{summary.get('total_expected_trials')} "
            f"trial={summary.get('latest_trial_id')} "
            f"prompt={summary.get('prompt_variant')} "
            f"round={summary.get('latest_round_num')}",
            flush=True,
        )
        time.sleep(max(1.0, poll_seconds))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    command = build_followon_command(
        results_root=args.results_root,
        from_run=args.from_run,
        models=args.model,
        dry_run=args.dry_run,
    )
    print(f"[followon] {' '.join(command)}", flush=True)
    if args.print_only:
        return 0

    summary = wait_for_baseline_completion(
        args.baseline_results,
        results_root=args.results_root,
        followon_command=command,
        poll_seconds=args.poll_seconds,
        stale_minutes=args.stale_minutes,
        refresh_results_packet=args.refresh_packet,
    )
    print(
        "baseline complete: "
        f"completed={summary.get('completed_trials')}/{summary.get('total_expected_trials')} "
        f"latest_trial={summary.get('latest_trial_id')}",
        flush=True,
    )
    write_watcher_status(
        results_root=args.results_root,
        baseline_results=args.baseline_results,
        summary=summary,
        followon_command=command,
        watcher_state="launching_followon",
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
