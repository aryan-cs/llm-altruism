print("[PART 2] Hello, World!")

import csv
import json
import os
from datetime import datetime

from agents.agent_2 import Agent2
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


def run_part_2(
    provider: str = os.getenv("PART_2_PROVIDER", os.getenv("DEFAULT_PROVIDER", "openai")),
    model: str = os.getenv("PART_2_MODEL", os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")),
    society_size: int = 3,
) -> str:
    agents = [
        Agent2(id_=f"society_{idx + 1}", provider_=provider, model_=model)
        for idx in range(society_size)
    ]

    console.print(
        Panel(
            "[bold]Part 2: Society Starter[/bold]\n"
            f"Agents: {society_size}\n"
            "Scenario: Shared-resource commons",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    decisions: list[tuple[str, str, str]] = []
    for idx, agent in enumerate(agents):
        raw = agent.query(agent.build_commons_prompt(), json_mode=True)
        data = json.loads(raw)
        decisions.append((agent.id, data["action"], data["reasoning"]))
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        console.print(
            Panel(
                Markdown(data["reasoning"]),
                title=f"[bold {color}]{agent.id} Reasoning[/bold {color}]",
                border_style=color,
                expand=True,
            )
        )

    table = Table(box=box.ROUNDED, expand=True, header_style="bold")
    table.add_column("Agent", justify="center")
    table.add_column("Action", justify="center")
    for agent_id, action, reasoning in decisions:
        table.add_row(
            agent_id,
            f"[{ACTION_COLORS.get(action, 'white')}][bold]{action}[/bold][/{ACTION_COLORS.get(action, 'white')}]",
        )
    console.print(
        Panel(
            table,
            title="[bold]Decision[/bold]",
            border_style="white",
            expand=True,
        )
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join("results", "part_2")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"{timestamp}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["agent", "action", "reasoning"])
        for row in decisions:
            writer.writerow(row)

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
    run_part_2()
