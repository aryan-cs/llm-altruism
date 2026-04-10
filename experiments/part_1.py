print("[PART 1] Hello, World!")

import csv
import json
import os
from datetime import datetime

from agents.agent_1 import Agent1
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()
ACTION_COLORS = {
    "SNITCH": "red",
    "STAY_SILENT": "green",
}


def run_part_1(
    provider: str = os.getenv("PART_1_PROVIDER", os.getenv("DEFAULT_PROVIDER", "openai")),
    model: str = os.getenv("PART_1_MODEL", os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")),
) -> str:
    agent_a = Agent1(id_="A", provider_=provider, model_=model)
    agent_b = Agent1(id_="B", provider_=provider, model_=model)

    console.print(
        Panel(
            "[bold]Part 1: Basic Interactions[/bold]\n"
            f"{agent_a} vs {agent_b}\n"
            "Scenario: Prisoner's Dilemma starter run",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    prompt_a = agent_a.build_prisoners_dilemma_prompt()
    prompt_b = agent_b.build_prisoners_dilemma_prompt()

    raw_a = agent_a.query(prompt_a, json_mode=True)
    raw_b = agent_b.query(prompt_b, json_mode=True)

    data_a = json.loads(raw_a)
    data_b = json.loads(raw_b)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join("results", "part_1")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", "action", "reasoning"])
        writer.writerow(["A", data_a["action"], data_a["reasoning"]])
        writer.writerow(["B", data_b["action"], data_b["reasoning"]])

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
    run_part_1()
