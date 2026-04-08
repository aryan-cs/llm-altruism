#!/usr/bin/env python3
"""Summarize active JSONL experiment logs for heartbeat reporting."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize live JSONL experiment logs and flag stale runs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="JSONL files or directories containing JSONL experiment logs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a text summary.",
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=15.0,
        help="Mark a run stale if its latest event is older than this many minutes.",
    )
    return parser.parse_args()


def expand_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    return paths


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def infer_track(experiment: dict[str, Any], run_metadata: dict[str, Any]) -> str | None:
    track = run_metadata.get("track")
    if isinstance(track, str) and track:
        return track
    part = experiment.get("part")
    if part == 2:
        return "society"
    if part == 3:
        return "reputation"
    if part == 1:
        return "part1"
    return None


def decode_trial_metadata(experiment: dict[str, Any], trial_id: int | None) -> dict[str, Any]:
    if trial_id is None:
        return {"prompt_variant": None, "repetition": None}
    prompt_variants = experiment.get("prompt_variants", []) or []
    temperatures = experiment.get("parameters", {}).get("temperature", [0.0]) or [0.0]
    repetitions = int(experiment.get("repetitions", 1) or 1)
    prompt_count = max(1, len(prompt_variants))
    temp_count = max(1, len(temperatures))
    conditions_per_group = prompt_count * temp_count * repetitions
    inner = trial_id % conditions_per_group
    prompt_variant_index = inner // (temp_count * repetitions)
    prompt_variant = None
    if prompt_variants:
        prompt_variant = prompt_variants[prompt_variant_index].get("name")
    repetition = inner % repetitions
    return {
        "prompt_variant": prompt_variant,
        "repetition": repetition,
    }


def alive_models_from_round(round_payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for vital in (round_payload.get("agent_vitals") or {}).values():
        if not isinstance(vital, dict) or not vital.get("alive", False):
            continue
        model = str(vital.get("model") or "unknown")
        counts[model] = counts.get(model, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def summarize_jsonl_log(
    path: Path,
    *,
    stale_minutes: float = 15.0,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    start: dict[str, Any] | None = None
    latest_round: dict[str, Any] | None = None
    latest_event_type: str | None = None
    latest_event_time: datetime | None = None
    latest_event_timestamp: str | None = None
    last_retry: dict[str, Any] | None = None
    trial_summary_count = 0
    provider_retry_count = 0

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            record_type = record.get("type")
            timestamp = parse_timestamp(record.get("timestamp"))
            if record_type == "experiment_start":
                start = record
            if record_type == "round":
                latest_round = record
            if record_type == "trial_summary":
                trial_summary_count += 1
            if record_type == "provider_retry":
                provider_retry_count += 1
                last_retry = record
            if timestamp is not None and (latest_event_time is None or timestamp >= latest_event_time):
                latest_event_time = timestamp
                latest_event_timestamp = record.get("timestamp")
                latest_event_type = record_type

    if start is None:
        return None

    config = start.get("config", {})
    experiment = config.get("experiment", config)
    run_metadata = config.get("run_metadata", {})
    track = infer_track(experiment, run_metadata)

    latest_trial_id = None
    if latest_round is not None:
        latest_trial_id = latest_round.get("trial_id")
    prompt_info = decode_trial_metadata(experiment, latest_trial_id)

    round_payload = latest_round.get("data", {}) if latest_round is not None else {}
    alive_count = round_payload.get("alive_count")
    total_agents = round_payload.get("total_agents")
    alive_fraction = None
    if isinstance(alive_count, (int, float)) and isinstance(total_agents, (int, float)) and total_agents:
        alive_fraction = float(alive_count) / float(total_agents)

    if now is None:
        now = datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    minutes_since_latest_event = None
    if latest_event_time is not None:
        comparison_time = latest_event_time
        if comparison_time.tzinfo is None:
            comparison_time = comparison_time.replace(tzinfo=UTC)
        minutes_since_latest_event = (now - comparison_time).total_seconds() / 60.0

    state = "unknown"
    if minutes_since_latest_event is not None:
        state = "active" if minutes_since_latest_event <= stale_minutes else "stale"

    return {
        "path": str(path),
        "experiment_id": start.get("experiment_id"),
        "name": experiment.get("name"),
        "track": track,
        "part": experiment.get("part"),
        "latest_trial_id": latest_trial_id,
        "prompt_variant": prompt_info["prompt_variant"],
        "repetition": prompt_info["repetition"],
        "latest_round_num": latest_round.get("round_num") if latest_round is not None else None,
        "alive_count": alive_count,
        "total_agents": total_agents,
        "alive_fraction": alive_fraction,
        "public_food": round_payload.get("public_food"),
        "public_water": round_payload.get("public_water"),
        "average_health": round_payload.get("average_health"),
        "average_energy": round_payload.get("average_energy"),
        "trade_volume": round_payload.get("trade_volume"),
        "alive_models": alive_models_from_round(round_payload),
        "trial_summary_count": trial_summary_count,
        "provider_retry_count": provider_retry_count,
        "last_retry_model": (last_retry.get("data") or {}).get("model") if last_retry else None,
        "latest_event_type": latest_event_type,
        "latest_event_timestamp": latest_event_timestamp,
        "minutes_since_latest_event": minutes_since_latest_event,
        "state": state,
    }


def format_status(summary: dict[str, Any]) -> str:
    alive = ""
    if summary.get("alive_count") is not None and summary.get("total_agents") is not None:
        alive = f"{summary['alive_count']}/{summary['total_agents']}"
    alive_models = summary.get("alive_models") or {}
    alive_models_text = ", ".join(f"{model}: {count}" for model, count in alive_models.items())
    minutes = summary.get("minutes_since_latest_event")
    minutes_text = f"{minutes:.1f}" if isinstance(minutes, (int, float)) else "n/a"

    lines = [
        str(summary["path"]),
        (
            f"  state={summary.get('state')} track={summary.get('track')} "
            f"trial={summary.get('latest_trial_id')} prompt={summary.get('prompt_variant')}"
        ),
        (
            f"  latest_round={summary.get('latest_round_num')} alive={alive} "
            f"food={summary.get('public_food')} water={summary.get('public_water')} "
            f"health={summary.get('average_health')} energy={summary.get('average_energy')} "
            f"trade={summary.get('trade_volume')}"
        ),
        (
            f"  summaries={summary.get('trial_summary_count')} "
            f"provider_retries={summary.get('provider_retry_count')} "
            f"latest_event={summary.get('latest_event_type')}@{summary.get('latest_event_timestamp')} "
            f"minutes_since_latest={minutes_text}"
        ),
    ]
    if alive_models_text:
        lines.append(f"  alive_models={alive_models_text}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summaries = [
        summary
        for summary in (
            summarize_jsonl_log(path, stale_minutes=args.stale_minutes) for path in expand_inputs(args.inputs)
        )
        if summary is not None
    ]
    if not summaries:
        print("No JSONL experiment logs found.")
        return 1

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        for index, summary in enumerate(summaries):
            if index:
                print()
            print(format_status(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
