# print("[PART 1] Hello, World!")

import os
import json
from datetime import datetime

from agents.agent_1 import Agent1
from experiments.misc.preflight import run_experiment_preflight
from experiments.misc.prompt_loader import load_prompt_config
from experiments.misc.result_writer import IncrementalCsvWriter
from experiments.misc.wizard import (
    choose_prompt_style,
    choose_provider_and_model,
    parse_game_theory_args,
)
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()
ACTION_COLORS = {
    "SNITCH": "red",
    "STAY_SILENT": "green",
    "TAKE_DRUG": "red",
    "DECLINE_DRUG": "green",
}
PART_1_PROMPTS = load_prompt_config("part_1")
EXPERIMENT_NAME = PART_1_PROMPTS["experiment_name"]
SCENARIO_LABELS = {
    name: config["label"]
    for name, config in PART_1_PROMPTS["agent"]["scenarios"].items()
}


def run_part_1(
    provider: str | None = None,
    model: str | None = None,
    prompt_style: str | None = None,
) -> str:
    if provider is None or model is None:
        provider, model = choose_provider_and_model(
            EXPERIMENT_NAME,
            experiment_key="part_1",
            provider=provider,
            model=model,
        )
    prompt_style = choose_prompt_style(
        EXPERIMENT_NAME,
        prompt_style=prompt_style,
    )
    run_experiment_preflight(EXPERIMENT_NAME, [(provider, model)])

    agent_a = Agent1(id_="A", provider_=provider, model_=model)
    agent_b = Agent1(id_="B", provider_=provider, model_=model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join("results", "part_1")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{timestamp}.csv")

    console.print(
        Panel(
            f"[bold]{EXPERIMENT_NAME}[/bold]\n"
            f"{agent_a} vs {agent_b}\n"
            f"Scenario: {SCENARIO_LABELS[prompt_style]}",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    if prompt_style == "direct":
        prompt_a = agent_a.build_prisoners_dilemma_prompt()
        prompt_b = agent_b.build_prisoners_dilemma_prompt()
    else:
        prompt_a = agent_a.build_indirect_competition_prompt()
        prompt_b = agent_b.build_indirect_competition_prompt()

    try:
        with IncrementalCsvWriter(
            csv_path,
            ["agent", "action", "reasoning"],
        ) as writer:
            raw_a = agent_a.query(prompt_a, json_mode=True)
            data_a = json.loads(raw_a)
            writer.write_row(["A", data_a["action"], data_a["reasoning"]])

            raw_b = agent_b.query(prompt_b, json_mode=True)
            data_b = json.loads(raw_b)
            writer.write_row(["B", data_b["action"], data_b["reasoning"]])
    except KeyboardInterrupt:
        console.print(
            Panel(
                f"Saved partial results to:\n[green]{os.path.abspath(csv_path)}[/green]",
                title="[bold yellow]Experiment Interrupted[/bold yellow]",
                border_style="yellow",
                expand=True,
            )
        )
        return csv_path

    console.rule("[bold]Round 1 / 1[/bold]")
    console.print(
        Panel(
            Markdown(data_a["reasoning"]),
            title="[bold cyan]Agent A Reasoning[/bold cyan]",
            border_style="cyan",
            expand=True,
        )
    )
    console.print(
        Panel(
            Markdown(data_b["reasoning"]),
            title="[bold magenta]Agent B Reasoning[/bold magenta]",
            border_style="magenta",
            expand=True,
        )
    )

    table = Table(box=box.ROUNDED, expand=True, header_style="bold")
    table.add_column("Agent", justify="center")
    table.add_column("Action", justify="center")
    table.add_row("A", f"[{ACTION_COLORS.get(data_a['action'], 'white')}][bold]{data_a['action']}[/bold][/{ACTION_COLORS.get(data_a['action'], 'white')}]")
    table.add_row("B", f"[{ACTION_COLORS.get(data_b['action'], 'white')}][bold]{data_b['action']}[/bold][/{ACTION_COLORS.get(data_b['action'], 'white')}]")
    console.print(
        Panel(
            table,
            title="[bold]Decision[/bold]",
            border_style="white",
            expand=True,
        )
    )

    console.print(
        Panel(
            f"[green]{os.path.abspath(csv_path)}[/green]",
            title="[bold]Results Saved[/bold]",
            border_style="green",
            expand=True,
        )
    )

    return csv_path


if __name__ == "__main__":
    cli_args = parse_game_theory_args()
    run_part_1(
        provider=cli_args.provider,
        model=cli_args.model,
        prompt_style=cli_args.prompt_style,
    )
