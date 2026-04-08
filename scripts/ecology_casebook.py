#!/usr/bin/env python3
"""Build a qualitative casebook from a society-simulation JSONL log."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a markdown casebook from a live ecology JSONL artifact."
    )
    parser.add_argument("log", help="Path to a JSONL experiment log.")
    parser.add_argument(
        "--output",
        help="Optional markdown output path. Defaults to <log stem>_casebook.md",
    )
    return parser.parse_args()


def load_log(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    start: dict[str, Any] | None = None
    rounds: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            if record.get("type") == "experiment_start":
                start = record
            elif record.get("type") == "round":
                rounds.append(record)
    if start is None:
        raise ValueError(f"No experiment_start found in {path}")
    if not rounds:
        raise ValueError(f"No round records found in {path}")
    return start, rounds


def round_payload(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("data", {})


def event_counter(rounds: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in rounds:
        for event in round_payload(record).get("events", []) or []:
            counts[str(event.get("kind") or "unknown")] += 1
    return counts


def top_events_text(rounds: list[dict[str, Any]], *, limit: int = 3) -> str:
    counts = event_counter(rounds)
    if not counts:
        return "none"
    return ", ".join(f"{kind}: {count}" for kind, count in counts.most_common(limit))


def alive_models_text(record: dict[str, Any]) -> str:
    counts: Counter[str] = Counter()
    for vital in (round_payload(record).get("agent_vitals") or {}).values():
        if not isinstance(vital, dict) or not vital.get("alive", False):
            continue
        counts[str(vital.get("model") or "unknown")] += 1
    if not counts:
        return "none"
    return ", ".join(f"{model}: {count}" for model, count in counts.most_common())


def fragile_agents_text(record: dict[str, Any], *, limit: int = 3) -> str:
    candidates: list[tuple[float, float, float, str]] = []
    for vital in (round_payload(record).get("agent_vitals") or {}).values():
        if not isinstance(vital, dict) or not vital.get("alive", False):
            continue
        health = float(vital.get("health", 0.0))
        energy = float(vital.get("energy", 0.0))
        resources_total = float(vital.get("resources_total", 0.0))
        agent_id = str(vital.get("agent_id") or "unknown")
        candidates.append((health, energy, resources_total, agent_id))
    if not candidates:
        return "none"
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return ", ".join(
        f"{agent_id} (health={health:.0f}, energy={energy:.0f}, total={resources_total:.0f})"
        for health, energy, resources_total, agent_id in candidates[:limit]
    )


def stable_plateau_summary(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return a stable post-collapse plateau summary when one exists."""
    if not rounds:
        return None

    latest = rounds[-1]
    latest_alive = round_payload(latest).get("alive_count")
    if latest_alive is None:
        return None

    last_death_index = next(
        (
            index
            for index in range(len(rounds) - 1, -1, -1)
            if round_payload(rounds[index]).get("newly_dead")
        ),
        None,
    )
    if last_death_index is None:
        return None

    plateau_start_index = next(
        (
            index + 1
            for index in range(len(rounds) - 1, -1, -1)
            if round_payload(rounds[index]).get("alive_count") != latest_alive
        ),
        0,
    )
    if plateau_start_index >= len(rounds):
        return None

    if plateau_start_index != last_death_index:
        return None

    plateau_start_round = rounds[plateau_start_index]
    plateau_rounds = rounds[plateau_start_index:]
    if len(plateau_rounds) < 5:
        return None

    if any(round_payload(record).get("alive_count") != latest_alive for record in plateau_rounds):
        return None
    if any(round_payload(record).get("newly_dead") for record in plateau_rounds[1:]):
        return None

    mean_public_food = sum(float(round_payload(record).get("public_food", 0.0)) for record in plateau_rounds) / len(plateau_rounds)
    mean_public_water = sum(float(round_payload(record).get("public_water", 0.0)) for record in plateau_rounds) / len(plateau_rounds)
    mean_health = sum(float(round_payload(record).get("average_health", 0.0)) for record in plateau_rounds) / len(plateau_rounds)
    mean_energy = sum(float(round_payload(record).get("average_energy", 0.0)) for record in plateau_rounds) / len(plateau_rounds)

    return {
        "start_round": plateau_start_round,
        "latest_round": latest,
        "duration_rounds": len(plateau_rounds),
        "alive": latest_alive,
        "alive_models": alive_models_text(latest),
        "mean_public_food": mean_public_food,
        "mean_public_water": mean_public_water,
        "mean_health": mean_health,
        "mean_energy": mean_energy,
        "action_mix": top_events_text(plateau_rounds, limit=5),
        "fragile_agents": fragile_agents_text(latest),
        "deaths_on_start": ", ".join(round_payload(plateau_start_round).get("newly_dead", []) or []) or "none",
    }


def milestone_records(rounds: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    milestones: list[tuple[str, dict[str, Any]]] = [("start", rounds[0])]

    first_loss = next(
        (
            record
            for record in rounds
            if isinstance(round_payload(record).get("alive_count"), (int, float))
            and isinstance(round_payload(record).get("total_agents"), (int, float))
            and round_payload(record)["alive_count"] < round_payload(record)["total_agents"]
        ),
        None,
    )
    if first_loss is not None:
        milestones.append(("first_loss", first_loss))

    largest_death_shock = max(rounds, key=lambda record: len(round_payload(record).get("newly_dead", []) or []))
    if len(round_payload(largest_death_shock).get("newly_dead", []) or []) > 0:
        milestones.append(("largest_death_shock", largest_death_shock))

    latest = rounds[-1]
    if latest not in [record for _, record in milestones]:
        milestones.append(("latest", latest))

    return milestones


def milestone_label(name: str) -> str:
    return name.replace("_", " ").title()


def summary_table_rows(milestones: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, record in milestones:
        payload = round_payload(record)
        rows.append(
            {
                "milestone": milestone_label(name),
                "round": record.get("round_num"),
                "alive": f"{payload.get('alive_count')}/{payload.get('total_agents')}",
                "public_food": payload.get("public_food"),
                "public_water": payload.get("public_water"),
                "avg_health": payload.get("average_health"),
                "avg_energy": payload.get("average_energy"),
                "deaths": len(payload.get("newly_dead", []) or []),
                "top_events": top_events_text([record]),
            }
        )
    return rows


def render_markdown(start: dict[str, Any], rounds: list[dict[str, Any]]) -> str:
    config = start.get("config", {})
    experiment = config.get("experiment", config)
    milestones = milestone_records(rounds)
    plateau = stable_plateau_summary(rounds)
    lines = [
        "# Ecology Casebook",
        "",
        f"- experiment: `{start.get('experiment_id')}`",
        f"- name: `{experiment.get('name')}`",
        f"- part: `{experiment.get('part')}`",
        f"- rounds observed: `{len(rounds)}`",
        f"- overall action mix: `{top_events_text(rounds, limit=5)}`",
        "",
        "## Milestones",
        "",
        "| milestone | round | alive | public_food | public_water | avg_health | avg_energy | deaths | top_events |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summary_table_rows(milestones):
        lines.append(
            "| {milestone} | {round} | {alive} | {public_food} | {public_water} | "
            "{avg_health:.1f} | {avg_energy:.1f} | {deaths} | {top_events} |".format(
                **row
            )
        )

    for name, record in milestones:
        payload = round_payload(record)
        lines.extend(
            [
                "",
                f"## {milestone_label(name)}",
                "",
                f"- round: `{record.get('round_num')}`",
                f"- alive: `{payload.get('alive_count')}/{payload.get('total_agents')}`",
                f"- public food / water: `{payload.get('public_food')}` / `{payload.get('public_water')}`",
                f"- average health / energy: `{payload.get('average_health')}` / `{payload.get('average_energy')}`",
                f"- newly dead: `{', '.join(payload.get('newly_dead', []) or []) or 'none'}`",
                f"- alive models: `{alive_models_text(record)}`",
                f"- dominant events in this round: `{top_events_text([record], limit=5)}`",
                f"- most fragile surviving agents: `{fragile_agents_text(record)}`",
            ]
        )

    if plateau is not None:
        lines.extend(
            [
                "",
                "## Stable Plateau",
                "",
                f"- plateau start round: `{plateau['start_round'].get('round_num')}`",
                f"- latest observed round: `{plateau['latest_round'].get('round_num')}`",
                f"- duration: `{plateau['duration_rounds']}` rounds",
                f"- stable alive population: `{int(plateau['alive'])}/{round_payload(plateau['latest_round']).get('total_agents')}`",
                f"- alive models: `{plateau['alive_models']}`",
                f"- mean public food / water during plateau: `{plateau['mean_public_food']:.1f}` / `{plateau['mean_public_water']:.1f}`",
                f"- mean health / energy during plateau: `{plateau['mean_health']:.1f}` / `{plateau['mean_energy']:.1f}`",
                f"- deaths on plateau start round: `{plateau['deaths_on_start']}`",
                f"- plateau action mix: `{plateau['action_mix']}`",
                f"- most fragile agents at latest observed state: `{plateau['fragile_agents']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    log_path = Path(args.log)
    output_path = (
        Path(args.output)
        if args.output
        else log_path.with_name(f"{log_path.stem}_casebook.md")
    )
    start, rounds = load_log(log_path)
    output_path.write_text(render_markdown(start, rounds), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
