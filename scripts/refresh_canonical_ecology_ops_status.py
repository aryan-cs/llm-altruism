#!/usr/bin/env python3
"""Build a consolidated canonical ecology operations status snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh a combined operations snapshot for the canonical ecology "
            "baseline, follow-on watcher, and maintenance supervisor."
        )
    )
    parser.add_argument(
        "baseline_results",
        help="Directory or JSONL file for the active baseline run.",
    )
    parser.add_argument(
        "--followon-root",
        default="results/canonical_ecology_followon",
        help="Directory containing watch_status.json and maintenance_status.json.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional explicit JSON output path. Defaults to <followon-root>/ops_status.json.",
    )
    parser.add_argument(
        "--markdown-output",
        help="Optional explicit markdown output path. Defaults to <followon-root>/ops_status.md.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=15.0,
        help="Staleness threshold forwarded to live_run_status.py helpers.",
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


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def default_output_paths(followon_root: str | Path) -> tuple[Path, Path]:
    root = Path(followon_root)
    return root / "ops_status.json", root / "ops_status.md"


def build_payload(
    *,
    baseline_summary: dict[str, Any],
    watch_status: dict[str, Any] | None,
    maintenance_status: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "updated_at": datetime.now(UTC).isoformat(),
        "baseline": baseline_summary,
        "watch_status": watch_status,
        "maintenance_status": maintenance_status,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    baseline = payload.get("baseline") or {}
    watch_status = payload.get("watch_status") or {}
    maintenance_status = payload.get("maintenance_status") or {}
    watcher = maintenance_status.get("watcher") or {}
    baseline_alive = ""
    if baseline.get("alive_count") is not None and baseline.get("total_agents") is not None:
        baseline_alive = f"{baseline['alive_count']}/{baseline['total_agents']}"
    trial_eta = baseline.get("estimated_minutes_remaining_in_trial")
    trial_eta_text = f"{trial_eta:.1f}" if isinstance(trial_eta, (int, float)) else "n/a"
    suite_eta = baseline.get("naive_minutes_remaining_in_baseline_suite")
    suite_eta_text = f"{suite_eta:.1f}" if isinstance(suite_eta, (int, float)) else "n/a"
    lines = [
        "# Canonical Ecology Ops Status",
        "",
        f"- updated_at: `{payload.get('updated_at')}`",
        "",
        "## Baseline",
        "",
        f"- state: `{baseline.get('state')}`",
        f"- prompt_variant: `{baseline.get('prompt_variant')}`",
        f"- latest_round: `{baseline.get('latest_round_num')}`",
        f"- alive: `{baseline_alive}`",
        f"- progress: `{baseline.get('completed_trials')}/{baseline.get('total_expected_trials')}` completed",
        f"- provider_retries: `{baseline.get('provider_retry_count')}`",
        f"- trial_eta_minutes: `{trial_eta_text}`",
        f"- baseline_suite_eta_minutes: `{suite_eta_text}`",
        "",
        "## Follow-on Watcher",
        "",
        f"- watcher_state: `{watch_status.get('watcher_state')}`",
        f"- baseline_results: `{watch_status.get('baseline_results')}`",
        f"- watcher_needed: `{watcher.get('needed')}`",
        f"- watcher_running: `{watcher.get('running')}`",
        f"- watcher_pid: `{watcher.get('pid')}`",
        f"- followon_started: `{watcher.get('followon_started')}`",
        "",
        "## Maintenance",
        "",
        f"- recovery_needed: `{maintenance_status.get('recovery_needed')}`",
        f"- recovery_returncode: `{maintenance_status.get('recovery_returncode')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    baseline_summary = summarize_baseline(args.baseline_results, stale_minutes=args.stale_minutes)
    followon_root = Path(args.followon_root)
    json_output, markdown_output = default_output_paths(followon_root)
    if args.json_output:
        json_output = Path(args.json_output)
    if args.markdown_output:
        markdown_output = Path(args.markdown_output)

    watch_status = read_json(followon_root / "watch_status.json")
    maintenance_status = read_json(followon_root / "maintenance_status.json")
    payload = build_payload(
        baseline_summary=baseline_summary,
        watch_status=watch_status,
        maintenance_status=maintenance_status,
    )
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_output.write_text(build_markdown(payload), encoding="utf-8")
    print(f"json_output: {json_output}")
    print(f"markdown_output: {markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
