print("[EXPERIMENT WIZARD] Hello, World!")

import argparse
from dataclasses import dataclass
from typing import Iterable

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

console = Console()
DEFAULT_SOCIETY_SIZE = 50
DEFAULT_SOCIETY_DAYS = 100
DEFAULT_RESOURCE = "water"
DEFAULT_SELFISH_GAIN = 2
DEFAULT_DEPLETION_UNITS = 2
DEFAULT_COMMUNITY_BENEFIT = 5
PROMPT_STYLES = {
    "direct": "Present the game directly using the standard game-theory framing.",
    "indirect": "Present the game indirectly using an analogous real-world scenario.",
}

PROVIDER_MODEL_OPTIONS = {
    "anthropic": [
        "claude-sonnet-4-5",
        "claude-opus-4-1",
    ],
    "openai": [
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-5-mini",
    ],
    "nvidia": [
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/nemotron-3-nano-30b-a3b",
    ],
    "cerebras": [
        "llama3.1-8b",
        "gpt-oss-120b",
    ],
    "ollama": [
        "gpt-oss:20b",
        "gpt-oss-safeguard:20b",
        "llama3.1:8b",
        "llama3.2:3b",
    ],
    "openrouter": [
        "openai/gpt-4.1-mini",
        "anthropic/claude-sonnet-4.5",
        "meta-llama/llama-3.1-70b-instruct",
    ],
    "groq": [
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ],
    "xai": [
        "grok-4-fast-non-reasoning",
        "grok-4.20",
        "grok-3-fast",
    ],
}


@dataclass
class SocietyConfig:
    society_size: int = DEFAULT_SOCIETY_SIZE
    days: int = DEFAULT_SOCIETY_DAYS
    resource: str = DEFAULT_RESOURCE
    selfish_gain: int = DEFAULT_SELFISH_GAIN
    depletion_units: int = DEFAULT_DEPLETION_UNITS
    community_benefit: int = DEFAULT_COMMUNITY_BENEFIT


def _require_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return value


def _require_non_negative(name: str, value: int) -> int:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return value


def _require_non_empty(name: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string.")
    return normalized


def _build_base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--provider", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    return parser


def parse_experiment_args() -> argparse.Namespace:
    return _build_base_parser().parse_args()


def parse_alignment_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_base_parser()
    parser.add_argument(
        "--benchmark",
        action="append",
        default=None,
        help="Repeatable provider:model benchmark target for part 0.",
    )
    return parser.parse_args(argv)


def parse_game_theory_args() -> argparse.Namespace:
    parser = _build_base_parser()
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=sorted(PROMPT_STYLES),
        default=None,
    )
    return parser.parse_args()


def parse_society_args() -> argparse.Namespace:
    parser = _build_base_parser()
    parser.add_argument("--society-size", type=int, default=None)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--resource", type=str, default=None)
    parser.add_argument("--selfish-gain", type=int, default=None)
    parser.add_argument("--depletion-units", type=int, default=None)
    parser.add_argument("--community-benefit", type=int, default=None)
    return parser.parse_args()


def _validate_provider(provider: str) -> str:
    normalized = _require_non_empty("provider", provider).lower()
    if normalized not in PROVIDER_MODEL_OPTIONS:
        supported = ", ".join(PROVIDER_MODEL_OPTIONS)
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers: {supported}."
        )
    return normalized


def _validate_model(provider: str, model: str) -> str:
    return _require_non_empty(f"model for provider {provider}", model)


def _parse_benchmark_spec(spec: str) -> tuple[str, str]:
    raw_spec = _require_non_empty("benchmark", spec)
    if ":" not in raw_spec:
        raise ValueError(
            "Benchmark entries must use the format provider:model."
        )
    provider, model = raw_spec.split(":", 1)
    normalized_provider = _validate_provider(provider)
    normalized_model = _validate_model(normalized_provider, model)
    return normalized_provider, normalized_model


def _dedupe_benchmarks(
    benchmarks: Iterable[tuple[str, str]],
) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for provider, model in benchmarks:
        item = (provider, model)
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _group_benchmarks(
    benchmarks: Iterable[tuple[str, str]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for provider, model in benchmarks:
        grouped.setdefault(provider, []).append(model)
    return grouped


def _prompt_for_provider() -> str:
    providers = list(PROVIDER_MODEL_OPTIONS)
    provider_table = Table(box=box.ROUNDED, expand=True, header_style="bold")
    provider_table.add_column("#", justify="center")
    provider_table.add_column("Provider", justify="center")
    for idx, provider_option in enumerate(providers, start=1):
        provider_table.add_row(str(idx), provider_option)
    console.print(
        Panel(
            provider_table,
            title="[bold]Provider Wizard[/bold]",
            border_style="white",
            expand=True,
        )
    )
    provider_idx = IntPrompt.ask(
        "[bold cyan]Select provider number[/bold cyan]",
        choices=[str(idx) for idx in range(1, len(providers) + 1)],
        show_choices=False,
    )
    return providers[provider_idx - 1]


def _prompt_for_model(provider: str) -> str:
    models = PROVIDER_MODEL_OPTIONS[provider]
    model_table = Table(box=box.ROUNDED, expand=True, header_style="bold")
    model_table.add_column("#", justify="center")
    model_table.add_column("Model", justify="center")
    for idx, model_option in enumerate(models, start=1):
        model_table.add_row(str(idx), model_option)
    model_table.add_row(str(len(models) + 1), "[italic]Custom model...[/italic]")
    console.print(
        Panel(
            model_table,
            title=f"[bold]Model Wizard: {provider}[/bold]",
            border_style="white",
            expand=True,
        )
    )

    model_idx = IntPrompt.ask(
        "[bold magenta]Select model number[/bold magenta]",
        choices=[str(idx) for idx in range(1, len(models) + 2)],
        show_choices=False,
    )

    if model_idx == len(models) + 1:
        return Prompt.ask("[bold green]Enter custom model id[/bold green]").strip()
    return models[model_idx - 1]


def _resolve_provider_and_model(
    *,
    provider: str | None,
    model: str | None,
) -> tuple[str, str]:
    if model is not None and provider is None:
        raise ValueError("A model cannot be provided without also providing a provider.")

    if provider is None:
        provider = _prompt_for_provider()
    else:
        provider = _validate_provider(provider)

    if model is None:
        model = _prompt_for_model(provider)

    return provider, _validate_model(provider, model)


def choose_provider_and_model(
    experiment_name: str,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    console.print(
        Panel(
            f"[bold]{experiment_name} Setup[/bold]\n"
            "Choose the provider and model for this run.",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    provider, model = _resolve_provider_and_model(
        provider=provider,
        model=model,
    )

    console.print(
        Panel(
            f"[bold]Provider:[/bold] {provider}\n"
            f"[bold]Model:[/bold] {model}",
            title="[bold]Selected Configuration[/bold]",
            border_style="green",
            expand=True,
        )
    )

    return provider, model


def choose_benchmark_models(
    experiment_name: str,
    benchmarks: list[str] | list[tuple[str, str]] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, list[str]]:
    if benchmarks is not None and (provider is not None or model is not None):
        raise ValueError(
            "Use either benchmark entries or provider/model, not both."
        )

    normalized: list[tuple[str, str]] = []
    if benchmarks is not None:
        for item in benchmarks:
            if isinstance(item, tuple):
                normalized.append(
                    (
                        _validate_provider(item[0]),
                        _validate_model(item[0], item[1]),
                    )
                )
            else:
                normalized.append(_parse_benchmark_spec(item))
    elif provider is not None or model is not None:
        normalized.append(_resolve_provider_and_model(provider=provider, model=model))
    else:
        console.print(
            Panel(
                f"[bold]{experiment_name} Benchmark Setup[/bold]\n"
                "Choose one or more provider/model pairs to benchmark.",
                box=box.DOUBLE,
                border_style="white",
                expand=True,
            )
        )

        while True:
            provider_choice, model_choice = _resolve_provider_and_model(
                provider=None,
                model=None,
            )
            normalized = _dedupe_benchmarks(
                [*normalized, (provider_choice, model_choice)]
            )

            selected_table = Table(box=box.ROUNDED, expand=True, header_style="bold")
            selected_table.add_column("#", justify="center")
            selected_table.add_column("Provider", justify="center")
            selected_table.add_column("Model", justify="center")
            for idx, (selected_provider, selected_model) in enumerate(
                normalized,
                start=1,
            ):
                selected_table.add_row(
                    str(idx),
                    selected_provider,
                    selected_model,
                )
            console.print(
                Panel(
                    selected_table,
                    title="[bold]Benchmark Targets[/bold]",
                    border_style="green",
                    expand=True,
                )
            )

            add_another = Prompt.ask(
                "[bold cyan]Add another benchmark model?[/bold cyan]",
                choices=["y", "n"],
                default="n",
                show_choices=False,
            )
            if add_another == "n":
                break

    deduped = _dedupe_benchmarks(normalized)
    if not deduped:
        raise ValueError("At least one benchmark model must be selected.")

    grouped = _group_benchmarks(deduped)
    selected_lines = [
        f"[bold]{provider_name}:[/bold] {', '.join(models)}"
        for provider_name, models in grouped.items()
    ]
    console.print(
        Panel(
            "\n".join(selected_lines),
            title="[bold]Selected Benchmark Models[/bold]",
            border_style="green",
            expand=True,
        )
    )
    return grouped


def choose_prompt_style(
    experiment_name: str,
    prompt_style: str | None = None,
) -> str:
    if prompt_style is not None and prompt_style not in PROMPT_STYLES:
        supported = ", ".join(sorted(PROMPT_STYLES))
        raise ValueError(
            f"Unsupported prompt style '{prompt_style}'. Supported prompt styles: {supported}."
        )

    console.print(
        Panel(
            f"[bold]{experiment_name} Prompt Style[/bold]\n"
            "Choose whether to use the direct or indirect framing.",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    if prompt_style is None:
        prompt_table = Table(box=box.ROUNDED, expand=True, header_style="bold")
        prompt_table.add_column("#", justify="center")
        prompt_table.add_column("Style", justify="center")
        prompt_table.add_column("Description", justify="center")
        prompt_options = list(PROMPT_STYLES)
        for idx, style in enumerate(prompt_options, start=1):
            prompt_table.add_row(str(idx), style, PROMPT_STYLES[style])
        console.print(
            Panel(
                prompt_table,
                title="[bold]Prompt Wizard[/bold]",
                border_style="white",
                expand=True,
            )
        )
        prompt_idx = IntPrompt.ask(
            "[bold cyan]Select prompt style number[/bold cyan]",
            choices=[str(idx) for idx in range(1, len(prompt_options) + 1)],
            show_choices=False,
        )
        prompt_style = prompt_options[prompt_idx - 1]

    console.print(
        Panel(
            f"[bold]Prompt style:[/bold] {prompt_style}",
            title="[bold]Selected Prompt Style[/bold]",
            border_style="green",
            expand=True,
        )
    )

    return prompt_style


def choose_society_config(
    experiment_name: str,
    society_size: int | None = None,
    days: int | None = None,
    resource: str | None = None,
    selfish_gain: int | None = None,
    depletion_units: int | None = None,
    community_benefit: int | None = None,
) -> SocietyConfig:
    console.print(
        Panel(
            f"[bold]{experiment_name} Society Setup[/bold]\n"
            "Choose the starting society size, runtime, and resource-sharing parameters.",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    if society_size is None:
        society_size = IntPrompt.ask(
            "[bold cyan]Starting number of agents[/bold cyan]",
            default=DEFAULT_SOCIETY_SIZE,
            show_default=True,
        )
    society_size = _require_positive("society_size", society_size)

    if days is None:
        days = IntPrompt.ask(
            "[bold cyan]Simulation length in days (0 = until population dies out)[/bold cyan]",
            default=DEFAULT_SOCIETY_DAYS,
            show_default=True,
        )
    days = _require_non_negative("days", days)

    if resource is None:
        resource = Prompt.ask(
            "[bold cyan]Shared resource[/bold cyan]",
            default=DEFAULT_RESOURCE,
            show_default=True,
        ).strip()
    resource = _require_non_empty("resource", resource)

    if selfish_gain is None:
        selfish_gain = IntPrompt.ask(
            "[bold cyan]Selfish gain multiplier[/bold cyan]",
            default=DEFAULT_SELFISH_GAIN,
            show_default=True,
        )
    selfish_gain = _require_positive("selfish_gain", selfish_gain)

    if depletion_units is None:
        depletion_units = IntPrompt.ask(
            "[bold cyan]Depletion units per overuse[/bold cyan]",
            default=DEFAULT_DEPLETION_UNITS,
            show_default=True,
        )
    depletion_units = _require_positive("depletion_units", depletion_units)

    if community_benefit is None:
        community_benefit = IntPrompt.ask(
            "[bold cyan]Community benefit units[/bold cyan]",
            default=DEFAULT_COMMUNITY_BENEFIT,
            show_default=True,
        )
    community_benefit = _require_positive("community_benefit", community_benefit)

    config = SocietyConfig(
        society_size=society_size,
        days=days,
        resource=resource,
        selfish_gain=selfish_gain,
        depletion_units=depletion_units,
        community_benefit=community_benefit,
    )

    console.print(
        Panel(
            f"[bold]Agents:[/bold] {config.society_size}\n"
            f"[bold]Days:[/bold] {config.days if config.days else 'until population dies out'}\n"
            f"[bold]Resource:[/bold] {config.resource}\n"
            f"[bold]Selfish gain:[/bold] {config.selfish_gain}\n"
            f"[bold]Depletion units:[/bold] {config.depletion_units}\n"
            f"[bold]Community benefit:[/bold] {config.community_benefit}",
            title="[bold]Selected Society Configuration[/bold]",
            border_style="green",
            expand=True,
        )
    )

    return config
