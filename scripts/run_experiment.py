#!/usr/bin/env python3
"""Interactive CLI entrypoint for society and precursor-game experiments."""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import questionary
from dotenv import load_dotenv
from questionary import Choice
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.traceback import install

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments import (  # noqa: E402
    ModelAccessResult,
    ModelSpec,
    apply_model_selection,
    apply_runtime_overrides,
    estimate_trial_conditions,
    known_model_specs,
    list_experiment_templates,
    probe_model_access_results,
    load_experiment_config,
    models_from_config,
    parse_model_selectors,
    run_experiment_config,
    spec_selector,
    template_description,
    template_label,
    wrap_picker_description,
)
from src.experiments.runner import API_KEY_ENV, ENDPOINT_ENV  # noqa: E402

console = Console()
install(show_locals=False)

WIZARD_STEPS = 4


@dataclass
class PreparedRun:
    """Concrete run request resolved before the async experiment starts."""

    config_path: Path
    selected_model_values: list[str]
    runtime_config: object
    run_metadata: dict[str, object]
    dry_run: bool
    results_dir: str
    resume_log: str | None = None
    access_results: dict[str, ModelAccessResult] | None = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an llm-altruism experiment. The main study is society survival "
            "under scarcity and reputation, while Part 1 configs provide "
            "precursor repeated-game diagnostics. With no arguments in a "
            "terminal, an interactive wizard guides you through experiment, "
            "model, and run settings."
        )
    )
    parser.add_argument(
        "--config",
        help="Optional experiment template YAML path. If omitted, choose one interactively.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help="Repeatable model selector override, for example openrouter:openai/gpt-oss-20b:free.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Force the full interactive wizard even if config/model arguments are supplied.",
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="List available experiment templates and exit.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List currently accessible catalog models and exit.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override the number of rounds/timesteps for the run.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help="Override the number of repetitions per condition.",
    )
    parser.add_argument(
        "--temperature",
        action="append",
        default=[],
        metavar="TEMP[,TEMP...]",
        help="Override one or more temperatures, for example --temperature 0.0,0.7.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Override concurrent model calls, mainly used in society simulations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without provider API calls using deterministic mock responses.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where JSON/JSONL outputs should be written.",
    )
    parser.add_argument(
        "--resume-log",
        help=(
            "Optional prior JSONL log to reuse completed trial summaries from before "
            "continuing the remaining trial slots."
        ),
    )
    return parser.parse_args()


def is_interactive_terminal() -> bool:
    """Return True when stdin/stdout are attached to a terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def provider_status(provider: str) -> str:
    """Summarize local env readiness for a provider."""
    key_env = API_KEY_ENV.get(provider, "")
    endpoint_env = ENDPOINT_ENV.get(provider, "")
    missing = []

    if key_env and not os.getenv(key_env):
        missing.append(key_env)
    if endpoint_env and not os.getenv(endpoint_env):
        missing.append(endpoint_env)

    if not missing:
        return "ready"
    return "missing " + ", ".join(missing)


def load_accessible_catalog(
    candidate_specs: list[ModelSpec] | None = None,
) -> tuple[list[ModelSpec], dict[str, ModelAccessResult]]:
    """Run the model-access test sweep for this invocation."""
    probe_specs = list(candidate_specs or known_model_specs())
    with console.status(
        "[bold cyan]Running model access tests for this session...[/bold cyan]",
        spinner="dots",
    ):
        access_results = asyncio.run(probe_model_access_results(probe_specs))

    accessible_specs = [
        access_results[spec_selector(spec)].spec
        for spec in probe_specs
        if spec_selector(spec) in access_results and access_results[spec_selector(spec)].accessible
    ]
    return accessible_specs, access_results


def render_access_test_summary(
    accessible_specs: list[ModelSpec],
    access_results: dict[str, ModelAccessResult],
) -> None:
    """Show a compact summary of the startup access-test sweep."""
    passed_count = len(accessible_specs)
    failed_count = sum(1 for result in access_results.values() if not result.accessible)

    details = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    details.add_column("Label", style="bold cyan", no_wrap=True)
    details.add_column("Value", style="white")
    details.add_row("Checked models", str(len(access_results)))
    details.add_row("Passed", str(passed_count))
    details.add_row("Hidden", str(failed_count))
    details.add_row(
        "Rule",
        "Only models that passed this startup access test are shown as options in this run.",
    )

    console.print(Panel(details, title="Startup Access Tests", border_style="cyan", box=box.ROUNDED))
    console.print()


def display_path(path: Path) -> str:
    """Return a user-friendly path relative to the repo when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_temperature_values(raw_values: list[str]) -> list[float] | None:
    """Parse one or more CLI temperature strings into a float list."""
    if not raw_values:
        return None

    values: list[float] = []
    for raw in raw_values:
        for chunk in raw.split(","):
            item = chunk.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except ValueError as exc:
                raise ValueError(f"Invalid temperature value: {item!r}") from exc

    if not values:
        raise ValueError("At least one valid temperature value is required.")
    return values


def format_temperature_values(values: list[float]) -> str:
    """Render temperature values for prompts and tables."""
    return ", ".join(f"{value:g}" for value in values)


def summarize_models(selected_models: list[str]) -> tuple[int, int]:
    """Return ready and missing-config counts for selected models."""
    ready = 0
    missing = 0
    for selector in selected_models:
        provider, _model = selector.split(":", 1)
        if provider_status(provider) == "ready":
            ready += 1
        else:
            missing += 1
    return ready, missing


def verify_selected_models_access(
    selected_model_values: list[str],
    *,
    dry_run: bool,
    access_results: dict[str, ModelAccessResult],
) -> dict[str, ModelAccessResult] | None:
    """Verify that live-run selected models passed the startup access tests."""
    if dry_run:
        return None

    parsed_specs = {
        spec_selector(spec): spec
        for spec in parse_model_selectors(selected_model_values)
    }
    inaccessible = []
    for selector in selected_model_values:
        result = access_results.get(selector)
        if result is None:
            inaccessible.append(
                ModelAccessResult(
                    spec=parsed_specs[selector],
                    accessible=False,
                    status="not present in this run's startup access-test results",
                )
            )
            continue
        if not result.accessible:
            inaccessible.append(result)
    if inaccessible:
        lines = [
            f"- {spec_selector(result.spec)}: {result.status}"
            for result in inaccessible
        ]
        raise ValueError(
            "These selected models did not pass the startup access tests for this run:\n"
            + "\n".join(lines)
        )

    return {selector: access_results[selector] for selector in selected_model_values}


def run_summary_sentence(config, selected_models: list[str], dry_run: bool) -> str:
    """Describe the planned run in plain English."""
    trial_conditions = estimate_trial_conditions(config)
    run_mode = "dry run with mock responses" if dry_run else "live run against configured providers"

    if config.part == 1:
        matchup_rule = (
            "With one selected model, Part 1 uses self-play. "
            "With multiple selected models, it runs pairwise matchups only."
        )
        return (
            f"This will run {config.game} using {len(selected_models)} selected model(s). "
            f"That expands to {len(config.pairings)} matchup(s) and about {trial_conditions} "
            f"top-level trial condition(s) once prompt variants, temperatures, and repetitions are included. "
            f"{matchup_rule} It will execute as a {run_mode}."
        )

    total_agents = sum(agent.count for agent in config.agents)
    return (
        f"This will simulate a Part {config.part} society with {total_agents} total agent slot(s) "
        f"drawn from {len(selected_models)} selected model(s), over {config.rounds} round(s) and "
        f"about {trial_conditions} top-level trial condition(s). It will execute as a {run_mode}."
    )


def render_section_intro(step: int, title: str, body: str, *, tip: str | None = None) -> None:
    """Render a clear wizard step header before a questionary prompt."""
    console.clear()
    content = Text()
    content.append(f"{title}\n", style="bold white")
    content.append(body, style="white")
    if tip:
        content.append(f"\n\nTip: {tip}", style="dim")

    console.print(
        Panel(
            content,
            title=f"Step {step} of {WIZARD_STEPS}",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    console.print()


def render_prompt_help(title: str, body: str, *, border_style: str = "blue") -> None:
    """Render a small helper panel before an individual question."""
    content = Text()
    content.append(f"{title}\n", style="bold white")
    content.append(body, style="white")
    console.print(
        Panel(
            content,
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    console.print()


def render_experiment_overview(path: Path) -> None:
    """Render a compact experiment overview panel inside the wizard."""
    config = load_experiment_config(path)
    details = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    details.add_column("Label", style="bold cyan", no_wrap=True)
    details.add_column("Value", style="white")
    details.add_row("Template", display_path(path))
    details.add_row("Experiment", config.name)
    details.add_row("Description", template_description(path))
    details.add_row("Part", str(config.part))
    if config.game:
        details.add_row("Game", config.game)
    details.add_row("Default rounds", str(config.rounds))
    details.add_row("Default repetitions", str(config.repetitions))
    details.add_row("Prompt variants", str(len(config.prompt_variants)))
    console.print(Panel(details, title="Selected Experiment", border_style="magenta"))
    console.print()


def list_experiments() -> None:
    """Print the available experiment templates."""
    table = Table(title="Experiment Templates", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Path", style="bold white")
    table.add_column("Template", style="green")
    table.add_column("Description", style="white")

    for path in list_experiment_templates():
        table.add_row(str(path.relative_to(ROOT)), template_label(path), template_description(path))

    console.print(table)


def list_models(
    accessible_specs: list[ModelSpec],
    results: dict[str, ModelAccessResult],
) -> None:
    """Print only the models that are currently reachable."""
    if not accessible_specs:
        console.print(
            Panel(
                "No models passed the startup access tests. Verify your `.env` and run "
                "`uv run pytest -q tests/test_readiness.py -k live_smoke` for details.",
                title="No Accessible Models",
                border_style="red",
                box=box.ROUNDED,
            )
        )
        return

    table = Table(title="Accessible Model Catalog", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Provider", style="bold white")
    table.add_column("Model", style="green")
    table.add_column("Access", style="yellow")

    for spec in accessible_specs:
        result = results[spec_selector(spec)]
        style = "green" if result.status == "verified" else "yellow"
        table.add_row(spec.provider or "unknown", spec.model, f"[{style}]{result.status}[/{style}]")

    console.print(table)


def choose_template_interactively() -> Path:
    """Prompt the user to choose an experiment template with arrow keys."""
    render_section_intro(
        1,
        "Choose an experiment",
        "Start by picking the kind of experiment you want to run. "
        "Each template is a starting point, and you can customize the run after this.",
        tip="Use the arrow keys to move. The preview under the list explains the highlighted option.",
    )
    template_paths = list_experiment_templates()
    terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
    answer = questionary.select(
        "What experiment would you like to run?",
        choices=[
            Choice(
                title=template_label(path),
                value=path,
                description=wrap_picker_description(
                    template_description(path),
                    columns=terminal_width,
                ),
            )
            for path in template_paths
        ],
        instruction="(Use the arrow keys, then press Enter)",
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return Path(answer)


def choose_models_interactively(
    config_path: Path,
    *,
    accessible_specs: list[ModelSpec],
    access_results: dict[str, ModelAccessResult],
) -> list[str]:
    """Prompt the user to choose the models to test with arrow-key checkboxes."""
    if not accessible_specs:
        raise ValueError(
            "No models passed the startup access tests. Verify your credentials and try "
            "`uv run pytest -q tests/test_readiness.py -k live_smoke`."
        )

    render_section_intro(
        2,
        "Choose the models",
        "Now pick which models should participate in the run. "
        "This list has already been filtered to models that passed the startup access tests for this run.",
        tip="Use Space to toggle a model on or off, and Enter when the list looks right.",
    )
    render_experiment_overview(config_path)

    template_config = load_experiment_config(config_path)
    accessible_selectors = {spec_selector(spec) for spec in accessible_specs}
    template_defaults = {
        spec_selector(spec)
        for spec in models_from_config(template_config)
        if spec_selector(spec) in accessible_selectors
    }
    all_accessible_defaults = {spec_selector(spec) for spec in accessible_specs}

    hidden_count = len(known_model_specs()) - len(accessible_specs)
    verification_note = (
        f"We hid {hidden_count} model(s) that are missing credentials or failed the startup access tests."
        if hidden_count
        else "Every known model in the catalog passed the startup access tests."
    )

    render_prompt_help(
        "Access filter",
        verification_note,
        border_style="green",
    )
    render_prompt_help(
        "Model selection style",
        "Choose whether you want a single-model run or a multi-model comparison. "
        "Single-model runs are useful for self-play or one-model society simulations.",
    )
    selection_mode = questionary.select(
        "How would you like to choose models?",
        choices=[
            Choice(
                title="One model",
                value="single",
                description="Pick exactly one model. Part 1 will run self-play only; Parts 2 and 3 will build the population from that model.",
            ),
            Choice(
                title="Multiple models",
                value="multiple",
                description="Pick any number of models. Part 1 will run pairwise matchups only; Parts 2 and 3 will distribute the population across them.",
            ),
        ],
        default="multiple",
        instruction="(Use the arrow keys, then press Enter)",
    ).ask()
    if selection_mode is None:
        raise KeyboardInterrupt

    if selection_mode == "single":
        render_prompt_help(
            "Pick one model",
            "Only one model will be selected for this run. In Part 1, that means self-play only. You can still change rounds, temperatures, and other settings in the next step.",
        )
        terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
        answer = questionary.select(
            "Which model would you like to use?",
            choices=[
                Choice(
                    title=f"{spec.provider:<11} {spec.model}",
                    value=spec_selector(spec),
                    description=wrap_picker_description(
                        f"Access: {access_results[spec_selector(spec)].status}. "
                        f"Part 1 uses self-play with this model; Parts 2 and 3 build the society from it.",
                        columns=terminal_width,
                    ),
                )
                for spec in accessible_specs
            ],
            instruction="(Use the arrow keys, then press Enter)",
        ).ask()
        if answer is None:
            raise KeyboardInterrupt
        return [answer]

    default_strategy = "template" if len(template_defaults) <= 4 else "empty"
    render_prompt_help(
        "Choose how to start the model list",
        "This controls which boxes are already checked before you edit the model list.",
    )
    preselection_strategy = questionary.select(
        "How should the multi-model list start?",
        choices=[
            Choice(
                title="Start empty (Recommended for large templates)",
                value="empty",
                description="Nothing is preselected. Best if you want to build a small custom set by hand.",
            ),
            Choice(
                title="Use this template's accessible defaults",
                value="template",
                description="Preselects the template's default models, but only if they passed the startup access tests.",
            ),
            Choice(
                title="Select all accessible models",
                value="accessible",
                description="Preselects every model that passed the startup access tests for this run.",
            ),
        ],
        default=default_strategy,
        instruction="(Use the arrow keys, then press Enter)",
    ).ask()
    if preselection_strategy is None:
        raise KeyboardInterrupt

    if preselection_strategy == "template":
        preselected = template_defaults
    elif preselection_strategy == "accessible":
        preselected = all_accessible_defaults
    else:
        preselected = set()

    render_prompt_help(
        "Pick the models",
        "Now edit the actual model list. Use Space to toggle entries on or off. "
        "Every item below already passed the startup access tests. "
        "For Part 1, if you leave one model selected it will self-play; if you select multiple models, the run will use pairwise matchups only.",
    )

    choices = []
    for spec in accessible_specs:
        key = spec_selector(spec)
        access_status = access_results[key].status
        title = f"{spec.provider:<11} {spec.model}  [{access_status}]"
        choices.append(
            Choice(
                title=title,
                value=key,
                checked=key in preselected,
            )
        )

    answer = questionary.checkbox(
        "Which models would you like to include?",
        choices=choices,
        instruction="(Use arrows, Space to toggle, Enter to continue)",
        validate=lambda selected: True if selected else "Select at least one model.",
    ).ask()

    if answer is None:
        raise KeyboardInterrupt
    return list(answer)


def prompt_positive_int(message: str, *, default: int) -> int:
    """Prompt for a positive integer with validation."""
    answer = questionary.text(
        message,
        default=str(default),
        validate=lambda value: (
            True
            if value.strip().isdigit() and int(value.strip()) > 0
            else "Enter a whole number greater than 0."
        ),
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return int(answer.strip())


def prompt_temperature_list(defaults: list[float]) -> list[float]:
    """Prompt for a comma-separated temperature list."""
    answer = questionary.text(
        "Which temperatures should we use? You can enter one value or a comma-separated list.",
        default=format_temperature_values(defaults),
        validate=lambda value: (
            True
            if _is_valid_temperature_input(value)
            else "Enter one or more numbers, for example 0.0 or 0.0,0.7."
        ),
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    parsed = parse_temperature_values([answer])
    if parsed is None:
        raise ValueError("Temperature input unexpectedly resolved to no values.")
    return parsed


def _is_valid_temperature_input(value: str) -> bool:
    """Return True when a temperature input string can be parsed."""
    try:
        return bool(parse_temperature_values([value]))
    except ValueError:
        return False


def choose_run_settings_interactively(
    config,
    *,
    initial_dry_run: bool,
    initial_results_dir: str,
):
    """Prompt for runtime settings such as rounds and repetitions."""
    render_section_intro(
        3,
        "Customize the run",
        "These settings let you make the run smaller, larger, faster, or safer. "
        "If you like the defaults, just press Enter to keep them.",
        tip="Changing rounds, repetitions, or temperatures can change both runtime and cost.",
    )

    rounds_label = (
        "How many rounds should each matchup run for?"
        if config.part == 1
        else "How many timesteps should each society simulation run for?"
    )
    render_prompt_help(
        "Rounds / timesteps",
        "This controls how long each trial runs. More rounds give models more chances to adapt, cooperate, defect, or recover from earlier moves.",
    )
    rounds = prompt_positive_int(rounds_label, default=config.rounds)

    render_prompt_help(
        "Repetitions",
        "Repetitions rerun the same condition multiple times. This helps smooth out randomness from sampling and gives you more stable averages.",
    )
    repetitions = prompt_positive_int(
        "How many repetitions should we run for each condition?",
        default=config.repetitions,
    )

    render_prompt_help(
        "Temperatures",
        "Temperature controls how deterministic or creative the model outputs are. "
        "Lower values are steadier; higher values can reveal more varied behavior.",
    )
    temperatures = prompt_temperature_list(config.parameters.temperature)

    concurrency = None
    if config.part in {2, 3}:
        render_prompt_help(
            "Concurrency",
            "Concurrency sets how many model calls can happen at the same time. "
            "Higher values can speed things up, but they can also increase rate-limit pressure.",
        )
        concurrency = prompt_positive_int(
            "How many model calls should run in parallel?",
            default=config.parameters.concurrency,
        )

    render_prompt_help(
        "Run mode",
        "Choose whether to execute a real run or a dry run. Dry runs are great for testing the setup before spending time or API quota.",
    )
    run_mode = questionary.select(
        "How should this run execute?",
        choices=[
            Choice(
                title="Live run",
                value="live",
                description="Uses your configured providers and can incur cost or rate limits.",
            ),
            Choice(
                title="Dry run",
                value="dry",
                description="Uses deterministic mock responses so you can test the pipeline quickly.",
            ),
        ],
        default="dry" if initial_dry_run else "live",
        instruction="(Use the arrow keys, then press Enter)",
    ).ask()
    if run_mode is None:
        raise KeyboardInterrupt

    render_prompt_help(
        "Results directory",
        "This is where the JSON and JSONL artifacts will be written. "
        "Keeping separate directories can help organize trial runs and overnight batches.",
    )
    results_dir = questionary.text(
        "Where should the result files be written?",
        default=initial_results_dir,
        validate=lambda value: True if value.strip() else "Enter a directory path.",
    ).ask()
    if results_dir is None:
        raise KeyboardInterrupt

    updated = apply_runtime_overrides(
        config,
        rounds=rounds,
        repetitions=repetitions,
        temperatures=temperatures,
        concurrency=concurrency,
    )
    return updated, run_mode == "dry", results_dir.strip()


def render_run_plan(
    config_path: Path,
    config,
    selected_models: list[str],
    dry_run: bool,
    results_dir: str,
    *,
    resume_log: str | None = None,
    access_results: dict[str, ModelAccessResult] | None = None,
) -> None:
    """Render a colored summary of the planned run."""
    ready_count, missing_count = summarize_models(selected_models)

    summary_text = Text(run_summary_sentence(config, selected_models, dry_run), style="white")
    console.print(Panel(summary_text, title="What This Run Will Do", border_style="cyan", box=box.ROUNDED))
    console.print()

    details = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    details.add_column("Label", style="bold cyan", no_wrap=True)
    details.add_column("Value", style="white")
    details.add_row("Template", display_path(config_path))
    details.add_row("Experiment", config.name)
    details.add_row("Description", template_description(config_path))
    details.add_row("Part", str(config.part))
    if config.game:
        details.add_row("Game", str(config.game))
    details.add_row("Rounds", str(config.rounds))
    details.add_row("Repetitions", str(config.repetitions))
    details.add_row("Temperatures", format_temperature_values(config.parameters.temperature))
    if config.part in {2, 3}:
        details.add_row("Concurrency", str(config.parameters.concurrency))
    details.add_row("Prompt variants", str(len(config.prompt_variants)))
    details.add_row("Estimated trial conditions", str(estimate_trial_conditions(config)))
    details.add_row("Run mode", "dry run" if dry_run else "live run")
    details.add_row("Results directory", results_dir)
    if resume_log:
        details.add_row("Resume log", resume_log)
    details.add_row("Selected models", str(len(selected_models)))
    details.add_row("Ready models", str(ready_count))
    if missing_count:
        details.add_row("Missing-config models", str(missing_count))

    console.print(Panel(details, title="Run Settings", border_style="green", box=box.ROUNDED))
    console.print()

    model_table = Table(title="Chosen Models", box=box.ROUNDED, header_style="bold magenta")
    model_table.add_column("Provider", style="bold white")
    model_table.add_column("Model", style="green")
    model_table.add_column("Status", style="yellow")

    for selector in selected_models:
        provider, model = selector.split(":", 1)
        status = access_results[selector].status if access_results and selector in access_results else provider_status(provider)
        status_style = "green" if status in {"ready", "verified"} else "yellow"
        model_table.add_row(provider, model, f"[{status_style}]{status}[/{status_style}]")

    console.print(model_table)
    console.print()


def confirm_run(
    config_path: Path,
    config,
    selected_models: list[str],
    dry_run: bool,
    results_dir: str,
    *,
    resume_log: str | None = None,
    access_results: dict[str, ModelAccessResult] | None = None,
) -> None:
    """Show a review screen and ask the user to confirm the run."""
    render_section_intro(
        4,
        "Review and start",
        "Here is the final plan. Take a quick look to make sure the experiment, models, and settings match what you intended.",
        tip="If this looks good, choose Start. If not, cancel and rerun the wizard.",
    )
    render_run_plan(
        config_path,
        config,
        selected_models,
        dry_run,
        results_dir,
        resume_log=resume_log,
        access_results=access_results,
    )
    start = questionary.confirm(
        "Start this experiment now?",
        default=True,
    ).ask()
    if start is None or not start:
        raise KeyboardInterrupt


def resolve_config_path(args: argparse.Namespace) -> Path:
    """Resolve the template config path from CLI args or interactive selection."""
    if args.config:
        return Path(args.config)
    if args.interactive or is_interactive_terminal():
        return choose_template_interactively()
    raise SystemExit("Non-interactive mode requires --config.")


def resolve_selected_models(
    args: argparse.Namespace,
    config_path: Path,
    *,
    accessible_specs: list[ModelSpec],
    access_results: dict[str, ModelAccessResult],
) -> list[str]:
    """Resolve model selectors from CLI args or interactive selection."""
    if args.model:
        return args.model
    if args.interactive or is_interactive_terminal():
        return choose_models_interactively(
            config_path,
            accessible_specs=accessible_specs,
            access_results=access_results,
        )
    raise SystemExit(
        "Non-interactive mode requires at least one --model provider:model-id selector."
    )


def startup_access_probe_specs(args: argparse.Namespace) -> list[ModelSpec]:
    """Choose which models to probe before the run starts."""
    if args.list_models:
        return known_model_specs()
    if args.model:
        return parse_model_selectors(args.model)
    if args.interactive or is_interactive_terminal():
        return known_model_specs()
    return []


def should_use_full_wizard(args: argparse.Namespace) -> bool:
    """Return True when the full staged wizard should run."""
    return args.interactive or (is_interactive_terminal() and not args.config and not args.model)


def build_run_metadata(
    *,
    config_path: Path,
    selected_models: list[str],
    selection_mode: str,
    config,
    dry_run: bool,
    results_dir: str,
    resume_log: str | None,
) -> dict[str, object]:
    """Build persisted metadata describing how the run was configured."""
    return {
        "template_path": display_path(config_path),
        "selected_models": selected_models,
        "selection_mode": selection_mode,
        "runtime_overrides": {
            "rounds": config.rounds,
            "repetitions": config.repetitions,
            "temperatures": config.parameters.temperature,
            "concurrency": config.parameters.concurrency if config.part in {2, 3} else None,
            "dry_run": dry_run,
            "results_dir": results_dir,
            "resume_log": resume_log,
        },
    }


def prepare_run_via_wizard(
    args: argparse.Namespace,
    *,
    accessible_specs: list[ModelSpec],
    access_results: dict[str, ModelAccessResult],
) -> PreparedRun:
    """Walk the user through the full interactive wizard."""
    config_path = choose_template_interactively()
    template_config = load_experiment_config(config_path)
    selected_model_values = choose_models_interactively(
        config_path,
        accessible_specs=accessible_specs,
        access_results=access_results,
    )
    selected_models = parse_model_selectors(selected_model_values)
    runtime_config = apply_model_selection(template_config, selected_models)
    runtime_config, dry_run, results_dir = choose_run_settings_interactively(
        runtime_config,
        initial_dry_run=args.dry_run,
        initial_results_dir=args.results_dir,
    )
    selected_access_results = verify_selected_models_access(
        selected_model_values,
        dry_run=dry_run,
        access_results=access_results,
    )
    confirm_run(
        config_path,
        runtime_config,
        selected_model_values,
        dry_run,
        results_dir,
        resume_log=args.resume_log,
        access_results=selected_access_results,
    )
    run_metadata = build_run_metadata(
        config_path=config_path,
        selected_models=selected_model_values,
        selection_mode="wizard",
        config=runtime_config,
        dry_run=dry_run,
        results_dir=results_dir,
        resume_log=args.resume_log,
    )
    return PreparedRun(
        config_path=config_path,
        selected_model_values=selected_model_values,
        runtime_config=runtime_config,
        run_metadata=run_metadata,
        dry_run=dry_run,
        results_dir=results_dir,
        resume_log=args.resume_log,
        access_results=selected_access_results,
    )


def prepare_run_non_interactive(
    args: argparse.Namespace,
    *,
    accessible_specs: list[ModelSpec],
    access_results: dict[str, ModelAccessResult],
) -> PreparedRun:
    """Resolve the run using CLI flags and targeted prompts for missing pieces."""
    config_path = resolve_config_path(args)
    template_config = load_experiment_config(config_path)
    selected_model_values = resolve_selected_models(
        args,
        config_path,
        accessible_specs=accessible_specs,
        access_results=access_results,
    )
    selected_models = parse_model_selectors(selected_model_values)
    runtime_config = apply_model_selection(template_config, selected_models)
    runtime_config = apply_runtime_overrides(
        runtime_config,
        rounds=args.rounds,
        repetitions=args.repetitions,
        temperatures=parse_temperature_values(args.temperature),
        concurrency=args.concurrency,
    )
    selected_access_results = verify_selected_models_access(
        selected_model_values,
        dry_run=args.dry_run,
        access_results=access_results,
    )
    run_metadata = build_run_metadata(
        config_path=config_path,
        selected_models=selected_model_values,
        selection_mode="cli" if args.model else "mixed",
        config=runtime_config,
        dry_run=args.dry_run,
        results_dir=args.results_dir,
        resume_log=args.resume_log,
    )
    return PreparedRun(
        config_path=config_path,
        selected_model_values=selected_model_values,
        runtime_config=runtime_config,
        run_metadata=run_metadata,
        dry_run=args.dry_run,
        results_dir=args.results_dir,
        resume_log=args.resume_log,
        access_results=selected_access_results,
    )


def prepare_run(
    args: argparse.Namespace,
    *,
    accessible_specs: list[ModelSpec],
    access_results: dict[str, ModelAccessResult],
) -> PreparedRun:
    """Resolve template, models, and settings before entering asyncio."""
    if should_use_full_wizard(args):
        return prepare_run_via_wizard(
            args,
            accessible_specs=accessible_specs,
            access_results=access_results,
        )
    return prepare_run_non_interactive(
        args,
        accessible_specs=accessible_specs,
        access_results=access_results,
    )


async def _run(prepared: PreparedRun) -> int:
    load_dotenv(ROOT / ".env", override=False)

    console.clear()
    render_run_plan(
        config_path=prepared.config_path,
        config=prepared.runtime_config,
        selected_models=prepared.selected_model_values,
        dry_run=prepared.dry_run,
        results_dir=prepared.results_dir,
        resume_log=prepared.resume_log,
        access_results=prepared.access_results,
    )

    with console.status("[bold cyan]Running experiment...[/bold cyan]", spinner="dots"):
        result = await run_experiment_config(
            prepared.runtime_config,
            dry_run=prepared.dry_run,
            results_dir=prepared.results_dir,
            run_metadata=prepared.run_metadata,
            resume_log=prepared.resume_log,
        )

    console.print(Panel(Text("Experiment complete.", style="bold green"), border_style="green"))
    console.print()
    render_result_summary(result)
    return 0


def render_result_summary(summary: dict) -> None:
    """Render a colored experiment result summary."""
    config = summary.get("config", {})
    experiment = config.get("experiment", config)
    stats = Table(box=box.SIMPLE_HEAVY, show_header=False, pad_edge=False)
    stats.add_column("Label", style="bold cyan", no_wrap=True)
    stats.add_column("Value", style="white")
    stats.add_row("Experiment ID", summary.get("experiment_id", "unknown"))
    stats.add_row("Name", experiment.get("name", "n/a"))
    stats.add_row("Part", str(experiment.get("part", "n/a")))
    if experiment.get("game"):
        stats.add_row("Game", str(experiment["game"]))
    stats.add_row("Trials", str(summary.get("trial_count", len(summary.get("trials", [])))))
    stats.add_row("Cost (USD)", f"{summary.get('total_cost_usd', 0.0):.4f}")
    stats.add_row("Duration (s)", f"{summary.get('total_duration_seconds', 0.0):.2f}")
    stats.add_row("Skipped Models", str(len(summary.get("skipped_models", []))))
    stats.add_row("Skipped Trials", str(len(summary.get("skipped_trials", []))))
    console.print(Panel(stats, title="Run Summary", border_style="green", box=box.ROUNDED))
    console.print()

    aggregate = summary.get("aggregate_summary", {})
    if aggregate:
        metrics = Table(title="Aggregate Metrics", box=box.ROUNDED, header_style="bold cyan")
        metrics.add_column("Metric", style="bold white")
        metrics.add_column("Value", justify="right", style="green")
        for key, value in sorted(aggregate.items()):
            rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
            metrics.add_row(key, rendered)
        console.print(metrics)
        console.print()

    skipped_models = summary.get("skipped_models", [])
    if skipped_models:
        skipped = Table(title="Skipped Models", box=box.ROUNDED, header_style="bold yellow")
        skipped.add_column("Model", style="bold white")
        skipped.add_column("Reason", style="yellow")
        for item in skipped_models:
            skipped.add_row(item.get("model", "unknown"), item.get("reason", "n/a"))
        console.print(skipped)


def main() -> int:
    """Run the CLI."""
    args = parse_args()
    try:
        load_dotenv(ROOT / ".env", override=False)

        if args.list_experiments:
            list_experiments()
            return 0

        if not args.list_models and not args.model and not (args.interactive or is_interactive_terminal()):
            raise SystemExit(
                "Non-interactive mode requires at least one --model provider:model-id selector."
            )

        accessible_specs, access_results = load_accessible_catalog(startup_access_probe_specs(args))
        if args.list_models:
            render_access_test_summary(accessible_specs, access_results)
            list_models(accessible_specs, access_results)
            return 0

        render_access_test_summary(accessible_specs, access_results)
        prepared = prepare_run(
            args,
            accessible_specs=accessible_specs,
            access_results=access_results,
        )
        return asyncio.run(_run(prepared))
    except KeyboardInterrupt:
        console.clear()
        console.print(Panel("Experiment setup cancelled.", border_style="yellow", box=box.ROUNDED))
        return 1
    except ValueError as exc:
        console.print(Panel(str(exc), title="Input Error", border_style="red", box=box.ROUNDED))
        return 2
    except Exception as exc:
        console.print(Panel(str(exc), title="Run Failed", border_style="red", box=box.ROUNDED))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
