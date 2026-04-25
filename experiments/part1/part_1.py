# print("[PART 1] Hello, World!")

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.agent_1 import Agent1
from experiments.part1.scenario_variants import list_scenario_variants
from experiments.misc.preflight import run_experiment_preflight
from experiments.misc.prompt_loader import load_prompt_config
from experiments.misc.result_writer import IncrementalCsvWriter
from experiments.misc.wizard import (
    choose_part_1_matrix,
    choose_provider_and_model,
    parse_game_theory_args,
)
from providers.api_call import (
    OllamaConnectionError,
    delete_other_ollama_models,
    unload_all_ollama_models,
    unload_ollama_model,
)
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()
PART_1_PROMPTS = load_prompt_config("part_1")
EXPERIMENT_NAME = PART_1_PROMPTS["experiment_name"]
PREFLIGHT_TEST_PATHS = [
    "tests/test_preflight.py",
    "tests/test_part_1.py",
]
RESULT_HEADERS = [
    "provider",
    "model",
    "game",
    "frame",
    "domain",
    "scenario_variant",
    "presentation",
    "prompt_id",
    "action",
    "justification",
    "prompt_text",
]
PART_1_RESULTS_DIR = Path("results") / "part_1"
PART_1_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
MODEL_BATCH_KEEP_ALIVE = "30m"
INITIAL_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 30.0


@dataclass(frozen=True)
class PromptVariant:
    game: str
    frame: str
    domain: str
    scenario_variant: str
    presentation: str
    prompt_id: str
    prompt_text: str
    allowed_actions: tuple[str, ...]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-")


def _is_default_selection(
    *,
    values: list[str],
    default_values: list[str],
) -> bool:
    return values == default_values


def _result_scope_label(
    *,
    games: list[str],
    frames: list[str],
    domains: list[str],
    presentations: list[str],
    limit: int | None,
    prompt_count: int,
) -> str:
    if limit is not None:
        return f"smoke-{prompt_count}prompts"

    defaults = PART_1_PROMPTS["defaults"]
    is_full = (
        _is_default_selection(values=games, default_values=defaults["games"])
        and _is_default_selection(values=frames, default_values=defaults["frames"])
        and _is_default_selection(values=domains, default_values=defaults["domains"])
        and _is_default_selection(
            values=presentations,
            default_values=defaults["presentations"],
        )
    )
    if is_full:
        return "full"
    return f"subset-{prompt_count}prompts"


def _build_result_filename(
    *,
    provider: str,
    model: str,
    games: list[str],
    frames: list[str],
    domains: list[str],
    presentations: list[str],
    limit: int | None,
    prompt_count: int,
    timestamp: str,
) -> str:
    scope_label = _result_scope_label(
        games=games,
        frames=frames,
        domains=domains,
        presentations=presentations,
        limit=limit,
        prompt_count=prompt_count,
    )
    segments = [
        "part1",
        _slugify(provider),
        _slugify(model),
        scope_label,
        timestamp,
    ]
    return "__".join(segments) + ".csv"


def _prompt_variant_count(
    *,
    games: list[str],
    frames: list[str],
    domains: list[str],
    presentations: list[str],
    limit: int | None = None,
) -> int:
    if limit is not None:
        return limit

    total = 0
    for game in games:
        for domain in domains:
            domain_game = PART_1_PROMPTS["domains"][domain]["games"][game]
            total += (
                len(frames)
                * len(presentations)
                * len(
                    list_scenario_variants(
                        domain_id=domain,
                        game_id=game,
                        fallback=domain_game,
                    )
                )
            )
    return total


def _build_prompt_variants(
    *,
    agent: Agent1,
    games: list[str],
    frames: list[str],
    domains: list[str],
    presentations: list[str],
    limit: int | None = None,
) -> list[PromptVariant]:
    variants: list[PromptVariant] = []
    for game in games:
        action_labels = tuple(PART_1_PROMPTS["games"][game]["action_descriptions"])
        for frame in frames:
            for domain in domains:
                for scenario_variant in agent.list_scenario_variant_ids(game, domain):
                    for presentation in presentations:
                        variants.append(
                            PromptVariant(
                                game=game,
                                frame=frame,
                                domain=domain,
                                scenario_variant=scenario_variant,
                                presentation=presentation,
                                prompt_id="__".join(
                                    [game, frame, domain, scenario_variant, presentation]
                                ),
                                prompt_text=agent.build_prompt(
                                    game,
                                    frame,
                                    domain,
                                    presentation,
                                    scenario_variant_id=scenario_variant,
                                ),
                                allowed_actions=action_labels,
                            )
                        )
    if limit is not None:
        return variants[:limit]
    return variants


def _resolve_headless_matrix_defaults(
    *,
    games: list[str] | None,
    frames: list[str] | None,
    domains: list[str] | None,
    presentations: list[str] | None,
) -> tuple[list[str] | None, list[str] | None, list[str] | None, list[str] | None]:
    defaults = PART_1_PROMPTS["defaults"]
    return (
        list(defaults["games"]) if games is None else games,
        list(defaults["frames"]) if frames is None else frames,
        list(defaults["domains"]) if domains is None else domains,
        list(defaults["presentations"]) if presentations is None else presentations,
    )


def _parse_agent_response(
    raw_response: str,
    *,
    allowed_actions: tuple[str, ...],
) -> tuple[str, str]:
    data = json.loads(raw_response)
    action = str(data.get("action", "")).strip()
    justification = str(data.get("justification", "")).strip()
    if action not in allowed_actions:
        raise ValueError(
            f"Invalid action '{action}'. Expected one of: {', '.join(allowed_actions)}."
        )
    if not justification:
        raise ValueError("Missing justification in part 1 response.")
    return action, justification


def _render_prompt_variant(
    variant: PromptVariant,
    *,
    index: int,
    total: int,
    action: str,
    justification: str,
) -> None:
    console.rule(f"[bold]Prompt {index} / {total}[/bold]")
    console.print(
        Panel(
            f"[bold]Game:[/bold] {variant.game}\n"
            f"[bold]Frame:[/bold] {variant.frame}\n"
            f"[bold]Domain:[/bold] {variant.domain}\n"
            f"[bold]Scenario Variant:[/bold] {variant.scenario_variant}\n"
            f"[bold]Presentation:[/bold] {variant.presentation}\n"
            f"[bold]Prompt ID:[/bold] {variant.prompt_id}",
            title="[bold]Prompt Metadata[/bold]",
            border_style="white",
            expand=True,
        )
    )
    console.print(
        Panel(
            Markdown(variant.prompt_text),
            title="[bold]Prompt[/bold]",
            border_style="cyan",
            expand=True,
        )
    )
    console.print(
        Panel(
            f"[bold]Action:[/bold] {action}\n\n[bold]Justification:[/bold]\n{justification}",
            title="[bold]Decision[/bold]",
            border_style="green",
            expand=True,
        )
    )


def _render_summary(
    rows: list[dict[str, str]],
) -> None:
    for dimension in ("game", "frame", "domain"):
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            counts[row[dimension]][row["action"]] += 1

        table = Table(box=box.ROUNDED, expand=True, header_style="bold")
        table.add_column(dimension.capitalize())
        table.add_column("Action Counts")
        for value, action_counts in counts.items():
            summary = ", ".join(
                f"{action}={count}"
                for action, count in sorted(action_counts.items())
            )
            table.add_row(value, summary)

        console.print(
            Panel(
                table,
                title=f"[bold]Summary By {dimension.capitalize()}[/bold]",
                border_style="white",
                expand=True,
            )
        )


def _render_headless_progress(
    *,
    processed_count: int,
    total_prompts: int,
    variant: PromptVariant,
    action: str,
) -> None:
    progress_bar = _build_headless_progress_bar(
        completed_count=processed_count,
        total_prompts=total_prompts,
    )
    console.print(
        f"[{processed_count}/{total_prompts}] "
        f"{progress_bar} {variant.prompt_id} -> {action}",
        markup=False,
    )


def _build_headless_progress_bar(
    *,
    completed_count: int,
    total_prompts: int,
    width: int = 20,
) -> str:
    if total_prompts < 1:
        raise ValueError("total_prompts must be >= 1")
    if completed_count < 0:
        raise ValueError("completed_count must be >= 0")

    normalized_completed = min(completed_count, total_prompts)
    filled = round((normalized_completed / total_prompts) * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _render_headless_prompt_start(
    *,
    provider: str,
    model: str,
    next_index: int,
    total_prompts: int,
    variant: PromptVariant,
) -> None:
    progress_bar = _build_headless_progress_bar(
        completed_count=max(0, next_index - 1),
        total_prompts=total_prompts,
    )
    console.print(
        f"Model {provider}/{model} "
        f"[prompt {next_index}/{total_prompts}] "
        f"{progress_bar} RUNNING {variant.prompt_id}",
        markup=False,
    )


def _metadata_path_for_csv(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_meta.json")


def _emit_retry_status_line(message: str, *, finalize: bool = False) -> None:
    message = message.replace("\r", " ").replace("\n", " ")
    line = f"\r\x1b[2K{message}"
    console.print(line, end="\n" if finalize else "\r", overflow="ignore")


def _retry_delay_seconds(attempt: int) -> float:
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    return min(MAX_RETRY_DELAY_SECONDS, INITIAL_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)))


def _is_ollama_resource_error(error: Exception) -> bool:
    error_text = str(error).lower()
    return (
        "requires more system memory" in error_text
        or "runner process has terminated" in error_text
        or "llama runner process has terminated" in error_text
    )


def _unload_agent_if_needed(agent: Agent1) -> None:
    if agent.provider.strip().lower() != "ollama":
        return
    try:
        unload_ollama_model(agent.model)
    except OllamaConnectionError as error:
        console.print(
            f"  [yellow][WARN] Could not unload Ollama model {agent.model}: {error}[/yellow]"
        )


def _prepare_ollama_model_for_run(
    *,
    provider: str,
    model: str,
) -> None:
    if provider.strip().lower() != "ollama":
        return

    try:
        unload_all_ollama_models()
    except OllamaConnectionError as error:
        console.print(
            f"  [yellow][WARN] Could not stop loaded Ollama models before running {model}: {error}[/yellow]"
        )

    try:
        delete_other_ollama_models(model)
    except OllamaConnectionError as error:
        console.print(
            f"  [yellow][WARN] Could not delete old Ollama models before running {model}: {error}[/yellow]"
        )


def _recover_agent_after_error(agent: Agent1, error: Exception) -> None:
    if agent.provider.strip().lower() != "ollama":
        return
    if not isinstance(error, OllamaConnectionError) and not _is_ollama_resource_error(error):
        return

    _unload_agent_if_needed(agent)
    _prepare_ollama_model_for_run(provider=agent.provider, model=agent.model)


def _query_variant_until_valid(
    agent: Agent1,
    variant: PromptVariant,
) -> tuple[str, str]:
    had_retry_status = False
    attempt = 0

    while True:
        attempt += 1
        try:
            raw_response = agent.query(variant.prompt_text, json_mode=True)
            action, justification = _parse_agent_response(
                raw_response,
                allowed_actions=variant.allowed_actions,
            )
            if had_retry_status:
                _emit_retry_status_line("", finalize=True)
            return action, justification
        except KeyboardInterrupt:
            if had_retry_status:
                _emit_retry_status_line("", finalize=True)
            raise
        except Exception as error:
            had_retry_status = True
            delay_seconds = _retry_delay_seconds(attempt)
            _emit_retry_status_line(
                f"  [yellow][WARN] Agent {agent.id} attempt {attempt} raised "
                f"{type(error).__name__}: {error}. Retrying in {delay_seconds:.0f}s...[/yellow]"
            )
            _recover_agent_after_error(agent, error)
            time.sleep(delay_seconds)


LEGACY_RESULT_HEADERS = [
    "provider",
    "model",
    "game",
    "frame",
    "domain",
    "presentation",
    "prompt_id",
    "action",
    "justification",
    "prompt_text",
]


def _load_part_1_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        source_header = list(reader.fieldnames)
        valid_headers = {tuple(RESULT_HEADERS), tuple(LEGACY_RESULT_HEADERS)}
        if tuple(source_header) not in valid_headers:
            raise ValueError(
                f"Unexpected CSV header for {csv_path}: expected {RESULT_HEADERS}, found {source_header}."
            )
        return [
            {column: row.get(column, "") or "" for column in RESULT_HEADERS}
            for row in reader
        ]


def _write_part_1_metadata(
    path: str | Path,
    *,
    timestamp: str,
    csv_path: str | Path,
    provider: str,
    model: str,
    games: list[str],
    frames: list[str],
    domains: list[str],
    presentations: list[str],
    limit: int | None,
    total_prompts: int,
) -> None:
    metadata = {
        "timestamp": timestamp,
        "csv_path": str(csv_path),
        "provider": provider,
        "model": model,
        "games": games,
        "frames": frames,
        "domains": domains,
        "presentations": presentations,
        "limit": limit,
        "total_prompts": total_prompts,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_part_1_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latest_interrupted_part_1_metadata_path() -> Path:
    if not PART_1_RESULTS_DIR.exists():
        raise ValueError(
            f"No interrupted part 1 run metadata was found in {PART_1_RESULTS_DIR}."
        )

    candidates: list[tuple[datetime, Path]] = []
    for metadata_path in PART_1_RESULTS_DIR.glob("*_meta.json"):
        try:
            metadata = _load_part_1_metadata(metadata_path)
            timestamp = str(metadata.get("timestamp", "")).strip()
            parsed = datetime.strptime(timestamp, PART_1_TIMESTAMP_FORMAT)
        except Exception:
            continue
        candidates.append((parsed, metadata_path))

    if not candidates:
        raise ValueError(
            f"No interrupted part 1 run metadata was found in {PART_1_RESULTS_DIR}."
        )

    candidates.sort()
    return candidates[-1][1]


def _render_resume_panel(
    *,
    timestamp: str,
    csv_path: Path,
    total_prompts: int,
    completed_count: int,
) -> None:
    console.print(
        Panel(
            f"[bold]Timestamp:[/bold] {timestamp}\n"
            f"[bold]Results:[/bold] [green]{csv_path.resolve()}[/green]\n"
            f"[bold]Completed prompts:[/bold] {completed_count} / {total_prompts}\n"
            f"[bold]Remaining prompts:[/bold] {total_prompts - completed_count}",
            title="[bold cyan]Resuming Part 1 Run[/bold cyan]",
            border_style="cyan",
            expand=True,
        )
    )


def _render_pause_panel(
    *,
    csv_path: Path,
    error: Exception,
) -> None:
    console.print(
        Panel(
            f"[bold]Saved partial results:[/bold] [green]{csv_path.resolve()}[/green]\n"
            f"[bold]Reason:[/bold] {type(error).__name__}: {error}\n\n"
            "Free system memory, then rerun with `--resume` to continue this model.",
            title="[bold yellow]Run Paused[/bold yellow]",
            border_style="yellow",
            expand=True,
        )
    )


def run_part_1(
    provider: str | None = None,
    model: str | None = None,
    games: list[str] | None = None,
    frames: list[str] | None = None,
    domains: list[str] | None = None,
    presentations: list[str] | None = None,
    limit: int | None = None,
    *,
    resume: bool = False,
    resume_metadata_path: str | Path | None = None,
    headless: bool = False,
    suppress_keyboard_interrupt: bool = True,
) -> str:
    completed_rows: list[dict[str, str]] = []

    if resume or resume_metadata_path is not None:
        metadata_path = (
            Path(resume_metadata_path)
            if resume_metadata_path is not None
            else _latest_interrupted_part_1_metadata_path()
        )
        metadata = _load_part_1_metadata(metadata_path)
        resumed_provider = str(metadata["provider"])
        resumed_model = str(metadata["model"])
        resumed_games = [str(value) for value in metadata["games"]]
        resumed_frames = [str(value) for value in metadata["frames"]]
        resumed_domains = [str(value) for value in metadata["domains"]]
        resumed_presentations = [str(value) for value in metadata["presentations"]]
        resumed_limit = metadata.get("limit")
        total_prompts = int(metadata["total_prompts"])
        timestamp = str(metadata["timestamp"])
        csv_path = Path(str(metadata["csv_path"]))

        if provider is not None and provider != resumed_provider:
            raise ValueError(
                f"Resume provider mismatch: expected {resumed_provider}, received {provider}."
            )
        if model is not None and model != resumed_model:
            raise ValueError(
                f"Resume model mismatch: expected {resumed_model}, received {model}."
            )

        provider = resumed_provider
        model = resumed_model
        games = resumed_games
        frames = resumed_frames
        domains = resumed_domains
        presentations = resumed_presentations
        limit = resumed_limit if isinstance(resumed_limit, int) else None

        completed_rows = _load_part_1_rows(csv_path)
        _render_resume_panel(
            timestamp=timestamp,
            csv_path=csv_path,
            total_prompts=total_prompts,
            completed_count=len(completed_rows),
        )
    else:
        if provider is None or model is None:
            provider, model = choose_provider_and_model(
                EXPERIMENT_NAME,
                experiment_key="part_1",
                provider=provider,
                model=model,
            )

        if headless:
            games, frames, domains, presentations = _resolve_headless_matrix_defaults(
                games=games,
                frames=frames,
                domains=domains,
                presentations=presentations,
            )

        matrix = choose_part_1_matrix(
            EXPERIMENT_NAME,
            available_games=PART_1_PROMPTS["defaults"]["games"],
            available_frames=PART_1_PROMPTS["defaults"]["frames"],
            available_domains=PART_1_PROMPTS["defaults"]["domains"],
            available_presentations=PART_1_PROMPTS["defaults"]["presentations"],
            games=games,
            frames=frames,
            domains=domains,
            presentations=presentations,
            limit=limit,
        )
        games = matrix.games
        frames = matrix.frames
        domains = matrix.domains
        presentations = matrix.presentations
        limit = matrix.limit
        timestamp = datetime.now().strftime(PART_1_TIMESTAMP_FORMAT)

        csv_path = PART_1_RESULTS_DIR / _build_result_filename(
            provider=provider,
            model=model,
            games=games,
            frames=frames,
            domains=domains,
            presentations=presentations,
            limit=limit,
            prompt_count=_prompt_variant_count(
                games=games,
                frames=frames,
                domains=domains,
                presentations=presentations,
                limit=limit,
            ),
            timestamp=timestamp,
        )
        metadata_path = csv_path.with_name(f"{csv_path.stem}_meta.json")

    run_experiment_preflight(
        EXPERIMENT_NAME,
        [(provider, model)],
        resume=resume,
        test_paths=PREFLIGHT_TEST_PATHS,
    )

    _prepare_ollama_model_for_run(provider=provider, model=model)

    keep_alive = MODEL_BATCH_KEEP_ALIVE if provider.strip().lower() == "ollama" else None
    agent = Agent1(
        id_="P1",
        provider_=provider,
        model_=model,
        keep_alive_=keep_alive,
    )
    prompt_variants = _build_prompt_variants(
        agent=agent,
        games=games,
        frames=frames,
        domains=domains,
        presentations=presentations,
        limit=limit,
    )
    total_prompts = len(prompt_variants)

    if not resume:
        _write_part_1_metadata(
            metadata_path,
            timestamp=timestamp,
            csv_path=csv_path,
            provider=provider,
            model=model,
            games=games,
            frames=frames,
            domains=domains,
            presentations=presentations,
            limit=limit,
            total_prompts=total_prompts,
        )

    completed_prompt_ids = {
        row["prompt_id"]
        for row in completed_rows
    }
    remaining_prompt_variants = [
        variant
        for variant in prompt_variants
        if variant.prompt_id not in completed_prompt_ids
    ]

    if not remaining_prompt_variants:
        if metadata_path.exists():
            metadata_path.unlink()
        console.print(
            Panel(
                f"[green]{csv_path.resolve()}[/green]",
                title="[bold]Run Already Complete[/bold]",
                border_style="green",
                expand=True,
            )
        )
        return str(csv_path)

    console.print(
        Panel(
            f"[bold]{EXPERIMENT_NAME}[/bold]\n"
            f"Model: {provider}/{model}\n"
            f"Prompt variants: {total_prompts}\n"
            f"Remaining this run: {len(remaining_prompt_variants)}",
            box=box.DOUBLE,
            border_style="white",
            expand=True,
        )
    )

    all_rows = list(completed_rows)
    paused_error: Exception | None = None
    interrupted = False

    try:
        with IncrementalCsvWriter(csv_path, RESULT_HEADERS, append=resume) as writer:
            processed_count = len(completed_rows)
            for variant in remaining_prompt_variants:
                next_index = processed_count + 1
                if headless:
                    _render_headless_prompt_start(
                        provider=provider,
                        model=model,
                        next_index=next_index,
                        total_prompts=total_prompts,
                        variant=variant,
                    )
                action, justification = _query_variant_until_valid(agent, variant)
                processed_count = next_index
                row = {
                    "provider": provider,
                    "model": model,
                    "game": variant.game,
                    "frame": variant.frame,
                    "domain": variant.domain,
                    "scenario_variant": variant.scenario_variant,
                    "presentation": variant.presentation,
                    "prompt_id": variant.prompt_id,
                    "action": action,
                    "justification": justification,
                    "prompt_text": variant.prompt_text,
                }
                writer.write_row([row[column] for column in RESULT_HEADERS])
                all_rows.append(row)
                if headless:
                    _render_headless_progress(
                        processed_count=processed_count,
                        total_prompts=total_prompts,
                        variant=variant,
                        action=action,
                    )
                else:
                    _render_prompt_variant(
                        variant,
                        index=processed_count,
                        total=total_prompts,
                        action=action,
                        justification=justification,
                    )
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        if _is_ollama_resource_error(error):
            paused_error = error
        else:
            raise
    finally:
        _unload_agent_if_needed(agent)

    if interrupted:
        if not suppress_keyboard_interrupt:
            raise KeyboardInterrupt()
        console.print(
            Panel(
                f"Saved partial results to:\n[green]{csv_path.resolve()}[/green]",
                title="[bold yellow]Experiment Interrupted[/bold yellow]",
                border_style="yellow",
                expand=True,
            )
        )
        return str(csv_path)

    if paused_error is not None:
        _render_pause_panel(csv_path=csv_path, error=paused_error)
        return str(csv_path)

    if metadata_path.exists():
        metadata_path.unlink()

    _render_summary(all_rows)
    console.print(
        Panel(
            f"[green]{csv_path.resolve()}[/green]",
            title="[bold]Results Saved[/bold]",
            border_style="green",
            expand=True,
        )
    )

    return str(csv_path)


def run_part_1_until_complete(
    provider: str | None = None,
    model: str | None = None,
    games: list[str] | None = None,
    frames: list[str] | None = None,
    domains: list[str] | None = None,
    presentations: list[str] | None = None,
    limit: int | None = None,
    *,
    resume: bool = False,
    headless: bool = False,
) -> str:
    resume_metadata_path: Path | None = None
    attempt = 0

    while True:
        attempt += 1
        console.print(
            f"[cyan]Part 1 run attempt {attempt} for {provider}/{model}[/cyan]"
        )
        csv_path = Path(
            run_part_1(
                provider=provider,
                model=model,
                games=games,
                frames=frames,
                domains=domains,
                presentations=presentations,
                limit=limit,
                resume=resume or resume_metadata_path is not None,
                resume_metadata_path=resume_metadata_path,
                headless=headless,
                suppress_keyboard_interrupt=False,
            )
        )

        rows = _load_part_1_rows(csv_path)
        metadata_path = _metadata_path_for_csv(csv_path)
        if rows and not metadata_path.exists():
            console.print(
                f"[green]Completed non-empty part 1 run for {provider}/{model}: {csv_path}[/green]"
            )
            return str(csv_path)

        resume_metadata_path = metadata_path if metadata_path.exists() else None
        if resume_metadata_path is None:
            console.print(
                f"[yellow][WARN] {provider}/{model} produced no usable rows. Retrying from scratch.[/yellow]"
            )
        else:
            console.print(
                f"[yellow][WARN] {provider}/{model} is still incomplete. Retrying via resume.[/yellow]"
            )


if __name__ == "__main__":
    cli_args = parse_game_theory_args()
    run_part_1_until_complete(
        provider=cli_args.provider,
        model=cli_args.model,
        games=cli_args.game,
        frames=cli_args.frame,
        domains=cli_args.domain,
        presentations=cli_args.presentation,
        limit=cli_args.limit,
        resume=cli_args.resume,
        headless=cli_args.headless,
    )
