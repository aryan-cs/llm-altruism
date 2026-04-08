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


def expected_trial_count(experiment: dict[str, Any]) -> int:
    prompt_variants = experiment.get("prompt_variants", []) or []
    temperatures = experiment.get("parameters", {}).get("temperature", [0.0]) or [0.0]
    repetitions = int(experiment.get("repetitions", 1) or 1)
    prompt_count = max(1, len(prompt_variants))
    temp_count = max(1, len(temperatures))
    return prompt_count * temp_count * repetitions


def alive_models_from_round(round_payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for vital in (round_payload.get("agent_vitals") or {}).values():
        if not isinstance(vital, dict) or not vital.get("alive", False):
            continue
        model = str(vital.get("model") or "unknown")
        counts[model] = counts.get(model, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def summarize_trial_records(
    experiment: dict[str, Any],
    round_records: list[dict[str, Any]],
    trial_summary_records: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped_rounds: dict[int, list[dict[str, Any]]] = {}
    for record in round_records:
        trial_id = int(record.get("trial_id", 0))
        grouped_rounds.setdefault(trial_id, []).append(record)

    observed_trial_ids = sorted(set(grouped_rounds) | set(trial_summary_records))
    for trial_id in observed_trial_ids:
        records = grouped_rounds.get(trial_id, [])
        latest_round = records[-1] if records else None
        latest_payload = latest_round.get("data", {}) if latest_round is not None else {}
        diagnostics = collapse_diagnostics(records)
        phase_metrics = phase_diagnostics(records, diagnostics)
        prompt_info = decode_trial_metadata(experiment, trial_id)
        summary_payload = trial_summary_records.get(trial_id, {})
        alive_count = latest_payload.get("alive_count")
        total_agents = latest_payload.get("total_agents")
        alive_fraction = None
        if isinstance(alive_count, (int, float)) and isinstance(total_agents, (int, float)) and total_agents:
            alive_fraction = float(alive_count) / float(total_agents)

        row = {
            "trial_id": trial_id,
            "prompt_variant": prompt_info["prompt_variant"],
            "repetition": prompt_info["repetition"],
            "completed": trial_id in trial_summary_records,
            "latest_round_num": latest_round.get("round_num") if latest_round is not None else None,
            "alive_count": alive_count,
            "total_agents": total_agents,
            "alive_fraction": alive_fraction,
            "public_food": latest_payload.get("public_food"),
            "public_water": latest_payload.get("public_water"),
            "average_health": latest_payload.get("average_health"),
            "average_energy": latest_payload.get("average_energy"),
            "trade_volume": latest_payload.get("trade_volume"),
            "alive_models": alive_models_from_round(latest_payload),
            "final_survival_rate": summary_payload.get("final_survival_rate"),
            "survival_rate": summary_payload.get("survival_rate"),
            **diagnostics,
            **phase_metrics,
        }
        rows.append(row)
    return rows


def collapse_diagnostics(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    if not rounds:
        return {
            "first_loss_round_num": None,
            "first_death_round_num": None,
            "last_death_round_num": None,
            "stability_start_round_num": None,
            "rounds_since_last_death": None,
            "stabilized_post_collapse": False,
            "population_regime": "no_rounds",
        }

    latest_round = rounds[-1]
    latest_payload = latest_round.get("data", {}) or {}
    latest_round_num = latest_round.get("round_num")
    latest_alive = latest_payload.get("alive_count")
    total_agents = latest_payload.get("total_agents")
    first_loss_round_num = next(
        (
            record.get("round_num")
            for record in rounds
            if isinstance((record.get("data") or {}).get("alive_count"), (int, float))
            and isinstance((record.get("data") or {}).get("total_agents"), (int, float))
            and (record.get("data") or {})["alive_count"] < (record.get("data") or {})["total_agents"]
        ),
        None,
    )
    first_death_round_num = next(
        (
            record.get("round_num")
            for record in rounds
            if ((record.get("data") or {}).get("newly_dead") or [])
        ),
        None,
    )
    last_death_round_num = next(
        (
            record.get("round_num")
            for record in reversed(rounds)
            if ((record.get("data") or {}).get("newly_dead") or [])
        ),
        None,
    )
    stability_start_round_num = next(
        (
            record.get("round_num")
            for record in reversed(rounds)
            if (record.get("data") or {}).get("alive_count") != latest_alive
        ),
        None,
    )
    if stability_start_round_num is None and rounds:
        stability_start_round_num = rounds[0].get("round_num")
    elif stability_start_round_num is not None:
        stability_start_round_num = int(stability_start_round_num) + 1

    rounds_since_last_death = None
    if last_death_round_num is not None and latest_round_num is not None:
        rounds_since_last_death = int(latest_round_num) - int(last_death_round_num)

    stabilized_post_collapse = bool(
        rounds_since_last_death is not None
        and rounds_since_last_death >= 5
        and stability_start_round_num == last_death_round_num
    )

    population_regime = "unknown"
    if first_loss_round_num is None and isinstance(latest_alive, (int, float)) and isinstance(total_agents, (int, float)):
        population_regime = "no_losses_yet" if int(latest_alive) == int(total_agents) else "losses_without_detection"
    elif stabilized_post_collapse:
        population_regime = "stable_post_collapse"
    elif first_loss_round_num is not None and last_death_round_num is None:
        population_regime = "losses_observed"
    elif last_death_round_num is not None and latest_round_num is not None and int(last_death_round_num) == int(latest_round_num):
        population_regime = "active_collapse"
    elif last_death_round_num is not None:
        population_regime = "post_loss_unsettled"

    return {
        "first_loss_round_num": first_loss_round_num,
        "first_death_round_num": first_death_round_num,
        "last_death_round_num": last_death_round_num,
        "stability_start_round_num": stability_start_round_num,
        "rounds_since_last_death": rounds_since_last_death,
        "stabilized_post_collapse": stabilized_post_collapse,
        "population_regime": population_regime,
    }


def phase_mean(rounds: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float((record.get("data") or {}).get(key))
        for record in rounds
        if isinstance((record.get("data") or {}).get(key), (int, float))
    ]
    if not values:
        return None
    return sum(values) / len(values)


def phase_diagnostics(rounds: list[dict[str, Any]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    first_loss = diagnostics.get("first_loss_round_num")
    last_death = diagnostics.get("last_death_round_num")
    stability_start = diagnostics.get("stability_start_round_num")
    latest_round_num = rounds[-1].get("round_num") if rounds else None

    result: dict[str, Any] = {
        "collapse_start_round_num": first_loss,
        "collapse_end_round_num": last_death,
        "collapse_duration_rounds": None,
        "collapse_death_count": None,
        "collapse_mean_public_food": None,
        "collapse_mean_public_water": None,
        "collapse_mean_health": None,
        "collapse_mean_energy": None,
        "plateau_start_round_num": stability_start if diagnostics.get("stabilized_post_collapse") else None,
        "plateau_end_round_num": latest_round_num if diagnostics.get("stabilized_post_collapse") else None,
        "plateau_duration_rounds": None,
        "plateau_mean_public_food": None,
        "plateau_mean_public_water": None,
        "plateau_mean_health": None,
        "plateau_mean_energy": None,
    }

    if isinstance(first_loss, (int, float)) and isinstance(last_death, (int, float)):
        collapse_rounds = [
            record
            for record in rounds
            if isinstance(record.get("round_num"), (int, float))
            and int(first_loss) <= int(record.get("round_num")) <= int(last_death)
        ]
        if collapse_rounds:
            result["collapse_duration_rounds"] = len(collapse_rounds)
            result["collapse_death_count"] = sum(
                len(((record.get("data") or {}).get("newly_dead") or []))
                for record in collapse_rounds
            )
            result["collapse_mean_public_food"] = phase_mean(collapse_rounds, "public_food")
            result["collapse_mean_public_water"] = phase_mean(collapse_rounds, "public_water")
            result["collapse_mean_health"] = phase_mean(collapse_rounds, "average_health")
            result["collapse_mean_energy"] = phase_mean(collapse_rounds, "average_energy")

    if diagnostics.get("stabilized_post_collapse") and isinstance(stability_start, (int, float)):
        plateau_rounds = [
            record
            for record in rounds
            if isinstance(record.get("round_num"), (int, float))
            and int(stability_start) <= int(record.get("round_num")) <= int(latest_round_num)
        ]
        if plateau_rounds:
            result["plateau_duration_rounds"] = len(plateau_rounds)
            result["plateau_mean_public_food"] = phase_mean(plateau_rounds, "public_food")
            result["plateau_mean_public_water"] = phase_mean(plateau_rounds, "public_water")
            result["plateau_mean_health"] = phase_mean(plateau_rounds, "average_health")
            result["plateau_mean_energy"] = phase_mean(plateau_rounds, "average_energy")

    return result


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
    round_records: list[dict[str, Any]] = []
    trial_summary_records: dict[int, dict[str, Any]] = {}

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
                round_records.append(record)
            if record_type == "trial_summary":
                trial_summary_count += 1
                trial_summary_records[int(record.get("trial_id", 0))] = record.get("summary", {}) or {}
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

    active_round_records = round_records
    if latest_trial_id is not None:
        active_round_records = [
            record for record in round_records if record.get("trial_id") == latest_trial_id
        ]
        if not active_round_records and latest_round is not None:
            active_round_records = [latest_round]

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

    diagnostics = collapse_diagnostics(active_round_records)
    phase_metrics = phase_diagnostics(active_round_records, diagnostics)
    trial_status_rows = summarize_trial_records(experiment, round_records, trial_summary_records)
    total_expected_trials = expected_trial_count(experiment)
    completed_trials = sum(1 for row in trial_status_rows if row.get("completed"))
    active_trials = sum(
        1
        for row in trial_status_rows
        if not row.get("completed") and row.get("latest_round_num") is not None
    )
    remaining_trials = max(0, int(total_expected_trials) - int(completed_trials))
    completion_fraction = (
        float(completed_trials) / float(total_expected_trials)
        if total_expected_trials
        else None
    )

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
        "population_loss_fraction": None if alive_fraction is None else 1.0 - float(alive_fraction),
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
        "total_expected_trials": total_expected_trials,
        "completed_trials": completed_trials,
        "active_trials": active_trials,
        "remaining_trials": remaining_trials,
        "completion_fraction": completion_fraction,
        "trial_status_rows": trial_status_rows,
        **diagnostics,
        **phase_metrics,
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
        (
            f"  progress=completed {summary.get('completed_trials')}/{summary.get('total_expected_trials')} "
            f"active={summary.get('active_trials')} remaining={summary.get('remaining_trials')}"
        ),
        (
            f"  regime={summary.get('population_regime')} "
            f"first_loss={summary.get('first_loss_round_num')} "
            f"last_death={summary.get('last_death_round_num')} "
            f"stability_start={summary.get('stability_start_round_num')} "
            f"rounds_since_last_death={summary.get('rounds_since_last_death')}"
        ),
    ]
    if alive_models_text:
        lines.append(f"  alive_models={alive_models_text}")
    if summary.get("plateau_duration_rounds") is not None:
        lines.append(
            (
                f"  phases=collapse[{summary.get('collapse_start_round_num')}..{summary.get('collapse_end_round_num')}] "
                f"plateau[{summary.get('plateau_start_round_num')}..{summary.get('plateau_end_round_num')}] "
                f"plateau_rounds={summary.get('plateau_duration_rounds')}"
            )
        )
    trial_rows = summary.get("trial_status_rows") or []
    if trial_rows:
        trial_parts = []
        for row in trial_rows:
            completed_flag = "done" if row.get("completed") else "live"
            alive_text = "n/a"
            if row.get("alive_count") is not None and row.get("total_agents") is not None:
                alive_text = f"{row['alive_count']}/{row['total_agents']}"
            trial_parts.append(
                f"{row.get('trial_id')}:{row.get('prompt_variant')}:{completed_flag}:"
                f"round={row.get('latest_round_num')}:alive={alive_text}"
            )
        lines.append(f"  trials={' ; '.join(trial_parts)}")
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
