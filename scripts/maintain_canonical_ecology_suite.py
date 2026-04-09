#!/usr/bin/env python3
"""Check the canonical ecology baseline and trigger recovery if it has stalled."""

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
    parser.add_argument(
        "--loop",
        action="store_true",
        help=(
            "Keep supervising the baseline instead of performing a single check. "
            "If recovery is launched, the supervisor exits after that handoff."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=180.0,
        help="Sleep interval between maintenance checks when --loop is enabled.",
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
        project_python(),
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


def build_ops_refresh_command(
    *,
    baseline_results: str | Path,
    followon_root: str | Path,
) -> list[str]:
    return [
        project_python(),
        "scripts/refresh_canonical_ecology_ops_status.py",
        str(baseline_results),
        "--followon-root",
        str(followon_root),
    ]


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
    watcher_status: dict[str, Any],
) -> None:
    payload = {
        "updated_at": datetime.now(UTC).isoformat(),
        "recovery_needed": recovery_needed,
        "recovery_command": recovery_command,
        "recovery_returncode": recovery_returncode,
        "baseline_summary": baseline_summary,
        "watcher": watcher_status,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def refresh_ops_snapshot(*, baseline_results: str | Path, followon_root: str | Path) -> int:
    command = build_ops_refresh_command(
        baseline_results=baseline_results,
        followon_root=followon_root,
    )
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


def configured_models(args: argparse.Namespace, recover_module: Any) -> list[str]:
    return list(args.model or getattr(recover_module, "DEFAULT_MODEL_SELECTORS", []))


def build_watcher_command(
    *,
    baseline_results: str | Path,
    followon_root: str | Path,
    models: list[str],
    dry_run: bool,
    recover_module: Any,
) -> list[str]:
    return recover_module.build_followon_watcher_command(
        baseline_results=baseline_results,
        followon_root=followon_root,
        models=models,
        dry_run=dry_run,
    )


def followon_has_started(followon_root: str | Path) -> bool:
    root = Path(followon_root)
    for stage in ("reputation", "event_stress"):
        stage_dir = root / stage
        if any(stage_dir.glob("*.jsonl")):
            return True
    return False


def find_running_command_pid(required_fragments: list[str]) -> int | None:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, argv_text = parts
        if all(fragment in argv_text for fragment in required_fragments):
            try:
                return int(pid_text)
            except ValueError:
                continue
    return None


def watcher_match_fragments(
    baseline_results: str | Path,
    followon_root: str | Path,
) -> list[str]:
    return [
        "scripts/continue_canonical_ecology_suite.py",
        str(baseline_results),
        "--results-root",
        str(followon_root),
        "--from-run",
        "reputation",
    ]


def ensure_watcher(
    *,
    baseline_results: str | Path,
    followon_root: str | Path,
    watcher_command: list[str],
    dry_run: bool,
    print_only: bool,
    recover_module: Any,
) -> dict[str, Any]:
    followon_started = followon_has_started(followon_root)
    watcher_needed = not followon_started
    watcher_pid = None
    watcher_started_now = False

    if watcher_needed:
        watcher_pid = find_running_command_pid(
            watcher_match_fragments(baseline_results, followon_root)
        )
        if watcher_pid is None and not (dry_run or print_only):
            watcher_pid = int(
                recover_module.spawn_background(
                    watcher_command,
                    log_path=Path(followon_root) / "watch.log",
                )
            )
            watcher_started_now = True
            print(f"[watcher] started pid={watcher_pid}", flush=True)
        elif watcher_pid is None and print_only:
            print(f"[watcher] {' '.join(watcher_command)}", flush=True)

    return {
        "needed": watcher_needed,
        "running": watcher_pid is not None,
        "pid": watcher_pid,
        "started_now": watcher_started_now,
        "followon_started": followon_started,
        "watcher_command": watcher_command,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    recover_module = _load_module(
        "recover_canonical_ecology_baseline.py",
        "recover_canonical_ecology_baseline_module",
    )
    models = configured_models(args, recover_module)
    while True:
        summary = summarize_baseline(args.baseline_results, stale_minutes=args.stale_minutes)
        recovery_needed = bool(recover_module.needs_recovery(summary))
        command = build_recovery_command(
            baseline_results=args.baseline_results,
            followon_root=args.followon_root,
            results_parent=args.results_parent,
            config_path=args.config,
            models=models,
            stale_minutes=args.stale_minutes,
            dry_run=args.dry_run,
        )
        watcher_command = build_watcher_command(
            baseline_results=args.baseline_results,
            followon_root=args.followon_root,
            models=models,
            dry_run=args.dry_run,
            recover_module=recover_module,
        )
        watcher_status = ensure_watcher(
            baseline_results=args.baseline_results,
            followon_root=args.followon_root,
            watcher_command=watcher_command,
            dry_run=args.dry_run,
            print_only=args.print_only,
            recover_module=recover_module,
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
            watcher_status=watcher_status,
        )
        ops_returncode = refresh_ops_snapshot(
            baseline_results=args.baseline_results,
            followon_root=args.followon_root,
        )
        if ops_returncode != 0:
            return ops_returncode
        if recovery_needed or not args.loop:
            return 0 if recovery_returncode in {None, 0} else recovery_returncode
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
