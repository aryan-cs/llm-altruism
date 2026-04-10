print("[PART 2] Hello, World!")

import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from math import ceil

from agents.agent_2 import Agent2
from experiments.preflight import run_experiment_preflight
from experiments.prompt_loader import load_prompt_config
from experiments.wizard import (
    SocietyConfig,
    choose_provider_and_model,
    choose_society_config,
    parse_society_args,
)
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()
AGENT_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]
ACTION_COLORS = {
    "RESTRAIN": "green",
    "OVERUSE": "red",
}
PART_2_PROMPTS = load_prompt_config("part_2")
EXPERIMENT_NAME = PART_2_PROMPTS["experiment_name"]
RESOURCE_RESERVE_MULTIPLIER = 10
COLLAPSE_ATTRITION_DIVISOR = 5
MAX_REASONING_SAMPLES_PER_DAY = 3


@dataclass
class DaySummary:
    day: int
    population_start: int
    population_end: int
    restrain_count: int
    overuse_count: int
    resource_units: int
    resource_capacity: int
    deaths: int


def _build_agents(provider: str, model: str, count: int) -> list[Agent2]:
    return [
        Agent2(id_=f"society_{idx + 1}", provider_=provider, model_=model)
        for idx in range(count)
    ]


def _initial_resource_units(config: SocietyConfig) -> int:
    baseline_units = max(config.community_benefit, config.depletion_units, 1)
    return max(
        config.society_size * baseline_units * RESOURCE_RESERVE_MULTIPLIER,
        config.society_size,
    )


def _normalize_action(raw_action: str) -> str:
    action = raw_action.strip().upper()
    if action not in ACTION_COLORS:
        supported = ", ".join(sorted(ACTION_COLORS))
        raise ValueError(
            f"Unsupported society action '{raw_action}'. Expected one of: {supported}."
        )
    return action


def _collapse_deaths(population: int, resource_units: int) -> int:
    if population <= 0 or resource_units > 0:
        return 0
    return min(population, max(1, ceil(population / COLLAPSE_ATTRITION_DIVISOR)))


def _should_show_reasoning_samples(
    *,
    day: int,
    configured_days: int,
    collapsed_today: bool,
) -> bool:
    if configured_days and configured_days <= 3:
        return True
    return day == 1 or collapsed_today


def _render_day_summary(summary: DaySummary) -> None:
    table = Table(box=box.ROUNDED, expand=True, header_style="bold")
    table.add_column("Start Pop.", justify="center")
    table.add_column("Restrain", justify="center")
    table.add_column("Overuse", justify="center")
    table.add_column("Reserve", justify="center")
    table.add_column("Deaths", justify="center")
    table.add_column("End Pop.", justify="center")
    table.add_row(
        str(summary.population_start),
        f"[green]{summary.restrain_count}[/green]",
        f"[red]{summary.overuse_count}[/red]",
        f"{summary.resource_units}/{summary.resource_capacity}",
        f"[red]{summary.deaths}[/red]" if summary.deaths else "0",
        str(summary.population_end),
    )
    console.print(
        Panel(
            table,
            title=f"[bold]Day {summary.day} Summary[/bold]",
            border_style="white",
            expand=True,
        )
    )


def _render_reasoning_samples(
    day: int,
    decisions: list[dict[str, str]],
) -> None:
    for idx, decision in enumerate(decisions[:MAX_REASONING_SAMPLES_PER_DAY]):
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        console.print(
            Panel(
                Markdown(decision["reasoning"]),
                title=(
                    f"[bold {color}]Day {day}: {decision['agent']} "
                    f"({decision['action']})[/bold {color}]"
                ),
                border_style=color,
                expand=True,
            )
        )


def _render_collapse_warning(day: int, resource: str) -> None:
    console.print(
        Panel(
            f"[bold red]The shared {resource} reserve has collapsed on day {day}.[/bold red]\n"
            "Population attrition now occurs each additional day until the society stabilizes or dies out.",
            title="[bold red]Collapse Warning[/bold red]",
            border_style="red",
            expand=True,
        )
    )


def run_part_2(
    provider: str | None = None,
    model: str | None = None,
    society_size: int | None = None,
    days: int | None = None,
    resource: str | None = None,
    selfish_gain: int | None = None,
    depletion_units: int | None = None,
    community_benefit: int | None = None,
) -> str:
    if provider is None or model is None:
        provider, model = choose_provider_and_model(
            EXPERIMENT_NAME,
            experiment_key="part_2",
            provider=provider,
            model=model,
        )
    society_config = choose_society_config(
        EXPERIMENT_NAME,
        society_size=society_size,
        days=days,
        resource=resource,
        selfish_gain=selfish_gain,
        depletion_units=depletion_units,
        community_benefit=community_benefit,
    )
    run_experiment_preflight(EXPERIMENT_NAME, [(provider, model)])

    agents = _build_agents(provider, model, society_config.society_size)
    resource_capacity = _initial_resource_units(society_config)
    resource_units = resource_capacity
    previous_overuse_count: int | None = None
    decision_rows: list[list[str | int]] = []
    completed_days = 0
    collapse_announced = False
    stop_reason = ""

    runtime_label = (
        "until population dies out (Ctrl+C to stop)"
        if society_config.days == 0
        else f"{society_config.days} days"
    )
    console.print(
        Panel(
            f"[bold]{EXPERIMENT_NAME}[/bold]\n"
            f"Agents: {society_config.society_size}\n"
            f"Runtime: {runtime_label}\n"
            f"Resource: {society_config.resource}\n"
            f"Reserve: {resource_units} sustainability units\n"
            f"Selfish gain: {society_config.selfish_gain}  |  "
            f"Depletion: {society_config.depletion_units}  |  "
            f"Community benefit: {society_config.community_benefit}",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    try:
        while agents and (society_config.days == 0 or completed_days < society_config.days):
            day = completed_days + 1
            console.rule(f"[bold]Day {day}[/bold]")

            population_start = len(agents)
            daily_decisions: list[dict[str, str]] = []
            for agent in agents:
                raw = agent.query(
                    agent.build_commons_prompt(
                        resource=society_config.resource,
                        selfish_gain=society_config.selfish_gain,
                        depletion_units=society_config.depletion_units,
                        community_benefit=society_config.community_benefit,
                        day=day,
                        living_agents=population_start,
                        resource_units=resource_units,
                        resource_capacity=resource_capacity,
                        previous_overuse_count=previous_overuse_count,
                    ),
                    json_mode=True,
                )
                data = json.loads(raw)
                action = _normalize_action(data["action"])
                reasoning = data["reasoning"].strip()
                daily_decisions.append(
                    {
                        "agent": agent.id,
                        "action": action,
                        "reasoning": reasoning,
                    }
                )

            overuse_count = sum(
                1 for decision in daily_decisions if decision["action"] == "OVERUSE"
            )
            restrain_count = population_start - overuse_count
            resource_units = max(
                0,
                resource_units - (overuse_count * society_config.depletion_units),
            )
            collapsed_today = resource_units == 0 and not collapse_announced
            if collapsed_today:
                collapse_announced = True
                _render_collapse_warning(day, society_config.resource)

            deaths = _collapse_deaths(population_start, resource_units)
            if deaths:
                agents = agents[: population_start - deaths]

            population_end = len(agents)
            summary = DaySummary(
                day=day,
                population_start=population_start,
                population_end=population_end,
                restrain_count=restrain_count,
                overuse_count=overuse_count,
                resource_units=resource_units,
                resource_capacity=resource_capacity,
                deaths=deaths,
            )
            _render_day_summary(summary)

            if _should_show_reasoning_samples(
                day=day,
                configured_days=society_config.days,
                collapsed_today=collapsed_today,
            ):
                _render_reasoning_samples(day, daily_decisions)

            for decision in daily_decisions:
                decision_rows.append(
                    [
                        day,
                        decision["agent"],
                        decision["action"],
                        decision["reasoning"],
                        population_start,
                        population_end,
                        restrain_count,
                        overuse_count,
                        resource_units,
                        deaths,
                    ]
                )

            previous_overuse_count = overuse_count
            completed_days += 1

            if not agents:
                stop_reason = f"Population died out on day {day}."
                break
    except KeyboardInterrupt:
        stop_reason = (
            f"Simulation interrupted by user after {completed_days} completed day(s)."
        )
        console.print(
            Panel(
                stop_reason,
                title="[bold yellow]Simulation Interrupted[/bold yellow]",
                border_style="yellow",
                expand=True,
            )
        )

    if not stop_reason:
        if society_config.days == 0:
            stop_reason = (
                f"Simulation ended after {completed_days} completed day(s) with "
                f"{len(agents)} agents remaining."
            )
        else:
            stop_reason = f"Reached the configured limit of {society_config.days} day(s)."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join("results", "part_2")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "day",
                "agent",
                "action",
                "reasoning",
                "population_start",
                "population_end",
                "restrain_count",
                "overuse_count",
                "resource_units_remaining",
                "deaths",
            ]
        )
        for row in decision_rows:
            writer.writerow(row)

    console.print(
        Panel(
            f"[bold]Days completed:[/bold] {completed_days}\n"
            f"[bold]Final population:[/bold] {len(agents)}\n"
            f"[bold]Reserve remaining:[/bold] {resource_units}/{resource_capacity}\n"
            f"[bold]Stop reason:[/bold] {stop_reason}\n\n"
            f"[green]{os.path.abspath(csv_path)}[/green]",
            title="[bold]Results Saved[/bold]",
            border_style="green",
            expand=True,
        )
    )

    return csv_path


if __name__ == "__main__":
    cli_args = parse_society_args()
    run_part_2(
        provider=cli_args.provider,
        model=cli_args.model,
        society_size=cli_args.society_size,
        days=cli_args.days,
        resource=cli_args.resource,
        selfish_gain=cli_args.selfish_gain,
        depletion_units=cli_args.depletion_units,
        community_benefit=cli_args.community_benefit,
    )
