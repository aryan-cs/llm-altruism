print("[EXPERIMENT WIZARD] Hello, World!")

import argparse
from dataclasses import dataclass
from typing import Iterable, TypeVar

from agents.agent_config import load_all_model_options, load_experiment_model_options
from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalMargin, ScrollbarMargin, Window
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.mouse_events import MouseEventType
from prompt_toolkit.shortcuts.dialogs import _create_app, _return_none
from prompt_toolkit.widgets import Button, Dialog, Label
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt

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
T = TypeVar("T")


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
    parser.add_argument(
        "--language",
        action="append",
        default=None,
        help="Repeatable language selection for part 0.",
    )
    parser.add_argument(
        "--resume",
        "--pick-up-where-we-left-off",
        action="store_true",
        default=False,
        help="Resume the latest interrupted part 0 run from its timestamped artifacts.",
    )
    parser.add_argument(
        "--judge-after",
        action="store_true",
        default=False,
        help="Collect every model response first, then run the judge over the saved responses afterwards.",
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


def _load_provider_model_options(
    experiment_key: str | None,
) -> dict[str, list[str]]:
    if experiment_key is None:
        return load_all_model_options()
    return load_experiment_model_options(experiment_key)


def _validate_provider(
    provider: str,
    provider_model_options: dict[str, list[str]],
) -> str:
    normalized = _require_non_empty("provider", provider).lower()
    if normalized not in provider_model_options:
        supported = ", ".join(provider_model_options)
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers: {supported}."
        )
    return normalized


def _validate_model(provider: str, model: str) -> str:
    return _require_non_empty(f"model for provider {provider}", model)


def _parse_benchmark_spec(
    spec: str,
    provider_model_options: dict[str, list[str]],
) -> tuple[str, str]:
    raw_spec = _require_non_empty("benchmark", spec)
    if ":" not in raw_spec:
        raise ValueError(
            "Benchmark entries must use the format provider:model."
        )
    provider, model = raw_spec.split(":", 1)
    normalized_provider = _validate_provider(provider, provider_model_options)
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


def _dedupe_values[T](values: Iterable[T]) -> list[T]:
    deduped: list[T] = []
    seen: set[T] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _validate_language(
    language: str,
    available_languages: list[str],
) -> str:
    normalized = _require_non_empty("language", language).lower()
    if normalized not in available_languages:
        supported = ", ".join(available_languages)
        raise ValueError(
            f"Unsupported language '{language}'. Supported languages: {supported}."
        )
    return normalized


class _ConfirmCheckboxList[T]:
    def __init__(
        self,
        values: list[tuple[T, str]],
        *,
        initial_values: Iterable[T] | None = None,
    ) -> None:
        self.values = values
        self.current_values: list[T] = list(initial_values or [])
        self._selected_index = 0

        kb = KeyBindings()

        @kb.add("up")
        @kb.add("k")
        def _up(event) -> None:
            del event
            self._selected_index = max(0, self._selected_index - 1)

        @kb.add("down")
        @kb.add("j")
        def _down(event) -> None:
            del event
            self._selected_index = min(len(self.values) - 1, self._selected_index + 1)

        @kb.add("pageup")
        def _pageup(event) -> None:
            window = event.app.layout.current_window
            if window.render_info:
                self._selected_index = max(
                    0,
                    self._selected_index - len(window.render_info.displayed_lines),
                )

        @kb.add("pagedown")
        def _pagedown(event) -> None:
            window = event.app.layout.current_window
            if window.render_info:
                self._selected_index = min(
                    len(self.values) - 1,
                    self._selected_index + len(window.render_info.displayed_lines),
                )

        @kb.add(" ")
        def _toggle(event) -> None:
            del event
            value = self.values[self._selected_index][0]
            if value in self.current_values:
                self.current_values.remove(value)
            else:
                self.current_values.append(value)

        @kb.add("enter")
        def _confirm(event) -> None:
            del event
            get_app().exit(result=self._ordered_current_values())

        self.control = FormattedTextControl(
            self._get_text_fragments,
            key_bindings=kb,
            focusable=True,
            show_cursor=True,
        )
        self.window = Window(
            content=self.control,
            right_margins=[
                ConditionalMargin(
                    margin=ScrollbarMargin(display_arrows=True),
                    filter=True,
                )
            ],
            dont_extend_height=True,
        )

    def _get_text_fragments(self):
        def mouse_handler(mouse_event) -> None:
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                self._selected_index = mouse_event.position.y

        fragments = []
        for index, value in enumerate(self.values):
            checked = value[0] in self.current_values
            selected = index == self._selected_index
            style = "class:checkbox"
            if checked:
                style += " class:checkbox-checked"
            if selected:
                style += " class:checkbox-selected"

            fragments.append((style, "["))
            if selected:
                fragments.append(("[SetCursorPosition]", ""))
            fragments.append((style, "*" if checked else " "))
            fragments.append((style, "] "))
            fragments.extend(to_formatted_text(value[1], style=style))
            fragments.append(("", "\n"))

        for index in range(len(fragments)):
            fragments[index] = (
                fragments[index][0],
                fragments[index][1],
                mouse_handler,
            )

        if fragments:
            fragments.pop()
        return fragments

    def _ordered_current_values(self) -> list[T]:
        return [value for value, _ in self.values if value in self.current_values]

    def __pt_container__(self):
        return self.window


def _arrow_select_one(
    *,
    title: str,
    text: str,
    options: list[tuple[str, str]],
) -> str:
    from prompt_toolkit.shortcuts import radiolist_dialog

    result = radiolist_dialog(
        title=title,
        text=text,
        values=options,
    ).run()
    if result is None:
        raise KeyboardInterrupt("Selection cancelled.")
    return result


def _arrow_select_many[T](
    *,
    title: str,
    text: str,
    options: list[tuple[T, str]],
    initial_values: Iterable[T] | None = None,
) -> list[T]:
    checkbox_list = _ConfirmCheckboxList(
        options,
        initial_values=initial_values,
    )

    def ok_handler() -> None:
        get_app().exit(result=checkbox_list._ordered_current_values())

    dialog = Dialog(
        title=title,
        body=HSplit(
            [Label(text=text, dont_extend_height=True), checkbox_list],
            padding=1,
        ),
        buttons=[
            Button(text="Ok", handler=ok_handler),
            Button(text="Cancel", handler=_return_none),
        ],
        with_background=True,
    )

    result = _create_app(dialog, style=None).run()
    if result is None:
        raise KeyboardInterrupt("Selection cancelled.")
    return result


def _prompt_for_provider(provider_model_options: dict[str, list[str]]) -> str:
    providers = list(provider_model_options)
    return _arrow_select_one(
        title="Provider Wizard",
        text="Use ↑/↓ to move and Enter to select a provider.",
        options=[(provider, provider) for provider in providers],
    )


def _prompt_for_model(
    provider: str,
    provider_model_options: dict[str, list[str]],
) -> str:
    models = provider_model_options[provider]
    custom_option = "__custom_model__"
    model = _arrow_select_one(
        title=f"Model Wizard: {provider}",
        text="Use ↑/↓ to move and Enter to select a model.",
        options=[
            *((model_option, model_option) for model_option in models),
            (custom_option, "Custom model..."),
        ],
    )

    if model == custom_option:
        return Prompt.ask("[bold green]Enter custom model id[/bold green]").strip()
    return model


def _prompt_for_benchmark_models(
    experiment_name: str,
    provider_model_options: dict[str, list[str]],
) -> list[tuple[str, str]]:
    options: list[tuple[tuple[str, str], str]] = []
    for provider, models in provider_model_options.items():
        for model in models:
            options.append(((provider, model), f"{provider} / {model}"))

    return _arrow_select_many(
        title=f"{experiment_name} Benchmark Setup",
        text="Use ↑/↓ to move, Space to toggle models, and Enter to confirm.",
        options=options,
    )


def _prompt_for_languages(
    experiment_name: str,
    available_languages: list[str],
) -> list[str]:
    return _arrow_select_many(
        title=f"{experiment_name} Language Setup",
        text="Use ↑/↓ to move, Space to toggle languages, and Enter to confirm.",
        options=[(language, language) for language in available_languages],
        initial_values=available_languages,
    )


def _resolve_provider_and_model(
    *,
    provider: str | None,
    model: str | None,
    provider_model_options: dict[str, list[str]],
) -> tuple[str, str]:
    if model is not None and provider is None:
        raise ValueError("A model cannot be provided without also providing a provider.")

    if provider is None:
        provider = _prompt_for_provider(provider_model_options)
    else:
        provider = _validate_provider(provider, provider_model_options)

    if model is None:
        model = _prompt_for_model(provider, provider_model_options)

    return provider, _validate_model(provider, model)


def choose_provider_and_model(
    experiment_name: str,
    experiment_key: str | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[str, str]:
    provider_model_options = _load_provider_model_options(experiment_key)
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
        provider_model_options=provider_model_options,
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
    experiment_key: str | None = None,
    benchmarks: list[str] | list[tuple[str, str]] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, list[str]]:
    provider_model_options = _load_provider_model_options(experiment_key)
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
                        _validate_provider(item[0], provider_model_options),
                        _validate_model(item[0], item[1]),
                    )
                )
            else:
                normalized.append(_parse_benchmark_spec(item, provider_model_options))
    elif provider is not None or model is not None:
        normalized.append(
            _resolve_provider_and_model(
                provider=provider,
                model=model,
                provider_model_options=provider_model_options,
            )
        )
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
        normalized = _prompt_for_benchmark_models(
            experiment_name,
            provider_model_options,
        )

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


def choose_languages(
    experiment_name: str,
    available_languages: list[str],
    languages: list[str] | None = None,
) -> list[str]:
    if not available_languages:
        raise ValueError("At least one available language must be configured.")

    console.print(
        Panel(
            f"[bold]{experiment_name} Language Setup[/bold]\n"
            "Choose one or more languages to run for each prompt.",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    if languages is None:
        normalized = _prompt_for_languages(experiment_name, available_languages)
    else:
        normalized = [
            _validate_language(language, available_languages)
            for language in languages
        ]

    selected_languages = _dedupe_values(normalized)
    if not selected_languages:
        raise ValueError("At least one language must be selected.")

    console.print(
        Panel(
            "\n".join(
                f"[bold]{index}.[/bold] {language}"
                for index, language in enumerate(selected_languages, start=1)
            ),
            title="[bold]Selected Languages[/bold]",
            border_style="green",
            expand=True,
        )
    )
    return selected_languages


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
        prompt_options = list(PROMPT_STYLES)
        prompt_style = _arrow_select_one(
            title="Prompt Wizard",
            text="Use ↑/↓ to move and Enter to select the prompt style.",
            options=[
                (style, f"{style}: {PROMPT_STYLES[style]}")
                for style in prompt_options
            ],
        )

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
