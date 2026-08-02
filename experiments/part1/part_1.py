# print("[PART 1] Hello, World!")

import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.agent_1 import Agent1
from experiments.part1.scenario_variants import list_scenario_variants
from experiments.misc.attempt_log import (
    DurableAttemptLogger,
    attempt_log_path_for_csv,
    validate_terminal_attempt_coverage,
    verify_attempt_log_metadata,
)
from experiments.misc.final_answer import (
    DEFAULT_EXTRACTOR_MAX_TOKENS,
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_PROVIDER,
    DEFAULT_OUTPUT_TOKEN_CAP,
    ExtractionConfig,
    extraction_record_from_error,
)
from experiments.misc.preflight import run_experiment_preflight
from experiments.misc.prompt_loader import load_prompt_config
from experiments.misc.result_writer import IncrementalCsvWriter
from experiments.misc.run_metadata import (
    base_run_metadata,
    file_integrity_metadata,
    mark_metadata_complete,
    mark_metadata_failed,
    metadata_is_complete,
    read_metadata,
    registry_identity_metadata,
    safe_error_message,
    source_bundle_metadata,
    stable_json_hash,
    validate_resume_contract,
    validate_file_integrity,
    validate_metadata_integrity,
    write_metadata,
)
from experiments.misc.wizard import (
    choose_part_1_matrix,
    choose_provider_and_model,
    parse_game_theory_args,
)
from providers.api_call import (
    OllamaConnectionError,
    ResponseParseError,
    delete_other_ollama_models,
    failure_provenance,
    is_retryable_api_failure,
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
    "order_position",
    "order_seed",
    "order_strategy",
    "counterbalance_index",
    "action",
    "justification",
    "prompt_text",
]
PART_1_RESULTS_DIR = Path("data") / "raw" / "part_1"
PART_1_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
MODEL_BATCH_KEEP_ALIVE = "30m"
INITIAL_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 30.0
MAX_AGENT_ATTEMPTS = 3
MAX_RUN_ATTEMPTS = 3
DEFAULT_PART_1_ORDER_SEED = 20260801
DEFAULT_PART_1_ORDER_STRATEGY = "counterbalanced"
PART_1_ORDER_STRATEGIES = ("seeded_shuffle", "counterbalanced")


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
    return variants


def _order_prompt_variants(
    variants: list[PromptVariant],
    *,
    seed: int | None,
    strategy: str,
    counterbalance_index: int,
    limit: int | None = None,
) -> list[PromptVariant]:
    """Return a reproducible randomized order with optional cyclic counterbalance."""

    if strategy == "legacy_canonical":
        ordered = list(variants)
    else:
        if strategy not in PART_1_ORDER_STRATEGIES:
            supported = ", ".join(PART_1_ORDER_STRATEGIES)
            raise ValueError(f"order_strategy must be one of: {supported}.")
        if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("order_seed must be an integer.")
        if counterbalance_index < 0:
            raise ValueError("counterbalance_index must be >= 0.")
        ordered = list(variants)
        random.Random(seed).shuffle(ordered)
        if strategy == "counterbalanced" and ordered:
            offset = counterbalance_index % len(ordered)
            ordered = ordered[offset:] + ordered[:offset]
    if limit is not None:
        return ordered[:limit]
    return ordered


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
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ResponseParseError("Invalid JSON in part 1 response.") from error
    action = str(data.get("action", "")).strip()
    justification = str(data.get("justification", "")).strip()
    if action not in allowed_actions:
        raise ResponseParseError(
            f"Invalid action '{action}'. Expected one of: {', '.join(allowed_actions)}."
        )
    if not justification:
        raise ResponseParseError("Missing justification in part 1 response.")
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


def _emit_headless_status_line(message: str, *, finalize: bool = False) -> None:
    sanitized = message.replace("\r", " ").replace("\n", " ")
    suffix = "\n" if finalize else ""
    console.file.write(f"\r\x1b[2K{sanitized}{suffix}")
    console.file.flush()


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
    _emit_headless_status_line(
        f"[{processed_count}/{total_prompts}] "
        f"{progress_bar} {variant.prompt_id} -> {action}",
        finalize=True,
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
    _emit_headless_status_line(
        f"Model {provider}/{model} "
        f"[prompt {next_index}/{total_prompts}] "
        f"{progress_bar} RUNNING {variant.prompt_id}",
    )


def _metadata_path_for_csv(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_meta.json")


def _emit_retry_status_line(message: str, *, finalize: bool = False) -> None:
    sanitized = message.replace("\r", " ").replace("\n", " ")
    plain_message = console.render_str(sanitized).plain
    _emit_headless_status_line(plain_message, finalize=finalize)


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
    if not _is_ollama_resource_error(error):
        return

    _unload_agent_if_needed(agent)
    _prepare_ollama_model_for_run(provider=agent.provider, model=agent.model)


def _query_variant_until_valid(
    agent: Agent1,
    variant: PromptVariant,
    *,
    attempt_logger: DurableAttemptLogger | None = None,
    order_position: int | None = None,
    extraction_config: ExtractionConfig | None = None,
) -> tuple[str, str]:
    had_retry_status = False
    attempt = 0

    while attempt < MAX_AGENT_ATTEMPTS:
        attempt += 1
        try:
            generation_record = None
            if extraction_config is None:
                raw_response = agent.query(variant.prompt_text, json_mode=True)
            else:
                raw_response, generated = agent.query_for_grading(
                    variant.prompt_text,
                    extraction_config=extraction_config,
                    extraction_kind="part_1_binary_game_decision",
                )
                generation_record = generated.to_dict()
        except KeyboardInterrupt as error:
            if attempt_logger is not None:
                attempt_logger.append(
                    provider=agent.provider,
                    model=agent.model,
                    unit_id=variant.prompt_id,
                    unit={"prompt_id": variant.prompt_id, "order_position": order_position},
                    attempt=attempt,
                    max_attempts=MAX_AGENT_ATTEMPTS,
                    prompt_text=variant.prompt_text,
                    outcome="interrupted",
                    error={
                        "exception_type": type(error).__name__,
                        "message": safe_error_message(error),
                        "provenance": None,
                    },
                )
            if had_retry_status:
                _emit_retry_status_line("", finalize=True)
            raise
        except Exception as error:
            retryable = (
                bool(getattr(error, "retryable", is_retryable_api_failure(error)))
                or _is_ollama_resource_error(error)
            ) and not isinstance(error, OllamaConnectionError)
            will_retry = retryable and attempt < MAX_AGENT_ATTEMPTS
            if attempt_logger is not None:
                attempt_logger.append(
                    provider=agent.provider,
                    model=agent.model,
                    unit_id=variant.prompt_id,
                    unit={"prompt_id": variant.prompt_id, "order_position": order_position},
                    attempt=attempt,
                    max_attempts=MAX_AGENT_ATTEMPTS,
                    prompt_text=variant.prompt_text,
                    outcome="provider_error",
                    error={
                        "exception_type": type(error).__name__,
                        "message": safe_error_message(error),
                        "provenance": failure_provenance(
                            error,
                            provider=agent.provider,
                            model=agent.model,
                        ),
                    },
                    generation_record=extraction_record_from_error(error),
                    will_retry=will_retry,
                )
            if isinstance(error, OllamaConnectionError):
                if had_retry_status:
                    _emit_retry_status_line("", finalize=True)
                raise
            if not retryable or not will_retry:
                if had_retry_status:
                    _emit_retry_status_line("", finalize=True)
                raise
            had_retry_status = True
            delay_seconds = _retry_delay_seconds(attempt)
            _emit_retry_status_line(
                f"  [yellow][WARN] Agent {agent.id} attempt {attempt} raised "
                f"{type(error).__name__}: {error}. Retrying in {delay_seconds:.0f}s...[/yellow]"
            )
            _recover_agent_after_error(agent, error)
            time.sleep(delay_seconds)
            continue

        try:
            action, justification = _parse_agent_response(
                raw_response,
                allowed_actions=variant.allowed_actions,
            )
        except Exception as error:
            retryable = (
                extraction_config is None
                and is_retryable_api_failure(error)
            )
            will_retry = retryable and attempt < MAX_AGENT_ATTEMPTS
            if attempt_logger is not None:
                attempt_logger.append(
                    provider=agent.provider,
                    model=agent.model,
                    unit_id=variant.prompt_id,
                    unit={"prompt_id": variant.prompt_id, "order_position": order_position},
                    attempt=attempt,
                    max_attempts=MAX_AGENT_ATTEMPTS,
                    prompt_text=variant.prompt_text,
                    outcome="invalid_response",
                    raw_response=raw_response,
                    generation_record=generation_record,
                    error={
                        "exception_type": type(error).__name__,
                        "message": safe_error_message(error),
                        "provenance": failure_provenance(
                            error,
                            provider=agent.provider,
                            model=agent.model,
                        ),
                    },
                    will_retry=will_retry,
                )
            if not retryable or not will_retry:
                if had_retry_status:
                    _emit_retry_status_line("", finalize=True)
                raise
            had_retry_status = True
            delay_seconds = _retry_delay_seconds(attempt)
            _emit_retry_status_line(
                f"  [yellow][WARN] Agent {agent.id} attempt {attempt} raised "
                f"{type(error).__name__}: {error}. Retrying in {delay_seconds:.0f}s...[/yellow]"
            )
            time.sleep(delay_seconds)
            continue

        if attempt_logger is not None:
            attempt_logger.append(
                provider=agent.provider,
                model=agent.model,
                unit_id=variant.prompt_id,
                unit={"prompt_id": variant.prompt_id, "order_position": order_position},
                attempt=attempt,
                max_attempts=MAX_AGENT_ATTEMPTS,
                prompt_text=variant.prompt_text,
                outcome="success",
                raw_response=raw_response,
                parsed_response={"action": action, "justification": justification},
                generation_record=generation_record,
            )
        if had_retry_status:
            _emit_retry_status_line("", finalize=True)
        return action, justification


PRE_ORDERING_RESULT_HEADERS = [
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
        valid_headers = {
            tuple(RESULT_HEADERS),
            tuple(PRE_ORDERING_RESULT_HEADERS),
            tuple(LEGACY_RESULT_HEADERS),
        }
        if tuple(source_header) not in valid_headers:
            raise ValueError(
                f"Unexpected CSV header for {csv_path}: expected {RESULT_HEADERS}, found {source_header}."
            )
        return [
            {column: row.get(column, "") or "" for column in RESULT_HEADERS}
            for row in reader
        ]


def _part_1_output_headers(path: str | Path, *, resume: bool) -> list[str]:
    csv_path = Path(path)
    if not resume or not csv_path.exists() or csv_path.stat().st_size == 0:
        return list(RESULT_HEADERS)
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        existing = next(csv.reader(handle), [])
    valid_headers = {
        tuple(RESULT_HEADERS),
        tuple(PRE_ORDERING_RESULT_HEADERS),
        tuple(LEGACY_RESULT_HEADERS),
    }
    if tuple(existing) not in valid_headers:
        raise ValueError(f"Unexpected CSV header for {csv_path}: {existing}")
    return existing


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
    order_seed: int,
    order_strategy: str,
    counterbalance_index: int,
    attempt_logger: DurableAttemptLogger,
    extraction_config: ExtractionConfig | None,
) -> None:
    ordering = {
        "seed": order_seed,
        "strategy": order_strategy,
        "counterbalance_index": counterbalance_index,
        "algorithm": "python_random_mt19937_shuffle_then_cyclic_rotation_v1",
    }
    parameters = {
        "games": games,
        "frames": frames,
        "domains": domains,
        "presentations": presentations,
        "limit": limit,
        "total_prompts": total_prompts,
        "ordering": ordering,
        "grading_protocol": (
            extraction_config.to_metadata() if extraction_config is not None else None
        ),
    }
    metadata = base_run_metadata(
        experiment="part_1",
        timestamp=timestamp,
        csv_path=csv_path,
        provider=provider,
        model=model,
        parameters=parameters,
        prompt_config_hash=stable_json_hash(PART_1_PROMPTS),
    )
    metadata.update(parameters)
    if extraction_config is not None:
        metadata["resume_contract"] = _strict_resume_contract(
            provider=provider,
            model=model,
            run_parameters=parameters,
            extraction_config=extraction_config,
        )
    attempt_log = attempt_logger.summary().to_metadata()
    attempt_log["coverage"] = "full_run"
    metadata["attempt_log"] = attempt_log
    write_metadata(path, metadata)


def _load_part_1_metadata(path: str | Path) -> dict[str, Any]:
    return read_metadata(path)


def _fresh_extraction_config(
    *,
    output_token_cap: int | None,
    extractor_provider: str | None,
    extractor_model: str | None,
    extractor_max_tokens: int | None,
) -> ExtractionConfig | None:
    if all(
        value is None
        for value in (
            output_token_cap,
            extractor_provider,
            extractor_model,
            extractor_max_tokens,
        )
    ):
        return None
    return ExtractionConfig(
        subject_output_token_cap=(
            DEFAULT_OUTPUT_TOKEN_CAP
            if output_token_cap is None
            else output_token_cap
        ),
        provider=(
            DEFAULT_EXTRACTOR_PROVIDER
            if extractor_provider is None
            else extractor_provider
        ),
        model=(
            DEFAULT_EXTRACTOR_MODEL
            if extractor_model is None
            else extractor_model
        ),
        extractor_max_tokens=(
            DEFAULT_EXTRACTOR_MAX_TOKENS
            if extractor_max_tokens is None
            else extractor_max_tokens
        ),
    )


def _strict_resume_contract(
    *,
    provider: str,
    model: str,
    run_parameters: dict[str, Any],
    extraction_config: ExtractionConfig,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    grading_protocol = extraction_config.to_metadata()
    return {
        "schema_version": 1,
        "experiment": "part_1",
        "source_bundle": source_bundle_metadata(
            [
                Path(__file__),
                repo_root / "agents" / "agent_1.py",
                repo_root / "agents" / "base_agent.py",
                repo_root / "experiments" / "misc" / "attempt_log.py",
                repo_root / "experiments" / "misc" / "final_answer.py",
                repo_root / "experiments" / "misc" / "preflight.py",
                repo_root / "experiments" / "misc" / "prompt_loader.py",
                repo_root / "experiments" / "misc" / "result_writer.py",
                repo_root / "experiments" / "misc" / "run_metadata.py",
                repo_root / "experiments" / "part1" / "part_1_prompt.json",
                repo_root / "experiments" / "part1" / "scenario_variants.py",
                repo_root / "agents" / "agent_config.py",
                repo_root / "agents" / "agent_config.registry.json",
                repo_root / "providers" / "api_call.py",
                repo_root / "pyproject.toml",
                repo_root / "uv.lock",
            ]
        ),
        "prompt_config_hash": stable_json_hash(PART_1_PROMPTS),
        "grading_protocol": grading_protocol,
        "model_roles": {
            "subject": registry_identity_metadata([(provider, model)]),
            "extractor": grading_protocol["extractor"],
        },
        "run_parameters": run_parameters,
        "retry_policy": {
            "agent_max_attempts": MAX_AGENT_ATTEMPTS,
            "run_max_attempts": MAX_RUN_ATTEMPTS,
            "initial_delay_seconds": INITIAL_RETRY_DELAY_SECONDS,
            "max_delay_seconds": MAX_RETRY_DELAY_SECONDS,
            "classification": "typed_provider_retryability_v1",
        },
    }


def _resume_extraction_config(metadata: dict[str, Any]) -> ExtractionConfig | None:
    value = metadata.get("grading_protocol")
    # Legacy runs did not use independent extraction. Continue those runs under
    # their original protocol rather than silently mixing row semantics.
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Resume metadata contains an invalid grading_protocol.")
    return ExtractionConfig.from_metadata(value)


def _attempt_log_metadata(
    logger: DurableAttemptLogger,
    *,
    coverage: str,
) -> dict[str, Any]:
    attempt_log = logger.summary().to_metadata()
    attempt_log["coverage"] = coverage
    return {
        "attempt_log": attempt_log,
        "invalid_attempts": attempt_log["invalid_response_attempts"],
        "retry_attempts": attempt_log["retry_attempts"],
        "provider_error_attempts": attempt_log["provider_error_attempts"],
    }


def _sync_part_1_attempt_metadata(
    metadata_path: str | Path,
    logger: DurableAttemptLogger,
    *,
    coverage: str,
    csv_path: str | Path,
) -> None:
    path = Path(metadata_path)
    if not path.exists():
        return
    metadata = read_metadata(path)
    metadata.update(_attempt_log_metadata(logger, coverage=coverage))
    if Path(csv_path).is_file():
        metadata["artifact_integrity"] = {
            "results": file_integrity_metadata(csv_path)
        }
    write_metadata(path, metadata)


def _latest_interrupted_part_1_metadata_path() -> Path:
    if not PART_1_RESULTS_DIR.exists():
        raise ValueError(
            f"No interrupted part 1 run metadata was found in {PART_1_RESULTS_DIR}."
        )

    candidates: list[tuple[datetime, Path]] = []
    for metadata_path in PART_1_RESULTS_DIR.glob("*_meta.json"):
        try:
            metadata = _load_part_1_metadata(metadata_path)
            if metadata_is_complete(metadata):
                continue
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


def _part_1_metadata_complete(path: str | Path) -> bool:
    try:
        return metadata_is_complete(_load_part_1_metadata(path))
    except Exception:
        return False


def run_part_1(
    provider: str | None = None,
    model: str | None = None,
    games: list[str] | None = None,
    frames: list[str] | None = None,
    domains: list[str] | None = None,
    presentations: list[str] | None = None,
    limit: int | None = None,
    order_seed: int | None = None,
    order_strategy: str | None = None,
    counterbalance_index: int | None = None,
    output_token_cap: int | None = None,
    extractor_provider: str | None = None,
    extractor_model: str | None = None,
    extractor_max_tokens: int | None = None,
    *,
    resume: bool = False,
    resume_metadata_path: str | Path | None = None,
    headless: bool = False,
    suppress_keyboard_interrupt: bool = True,
) -> str:
    completed_rows: list[dict[str, str]] = []
    is_resuming = resume or resume_metadata_path is not None
    requested_order_seed = order_seed
    requested_order_strategy = order_strategy
    requested_counterbalance_index = counterbalance_index
    attempt_log_coverage = "full_run"

    if is_resuming:
        metadata_path = (
            Path(resume_metadata_path)
            if resume_metadata_path is not None
            else _latest_interrupted_part_1_metadata_path()
        )
        metadata = _load_part_1_metadata(metadata_path)
        if metadata.get("grading_protocol") is not None:
            validate_metadata_integrity(metadata, required=True)
        extraction_config = _resume_extraction_config(metadata)
        if extraction_config is None:
            if any(
                value is not None
                for value in (
                    output_token_cap,
                    extractor_provider,
                    extractor_model,
                    extractor_max_tokens,
                )
            ):
                raise ValueError(
                    "Cannot add final-answer extraction while resuming a legacy Part 1 run."
                )
        else:
            requested_extraction = ExtractionConfig(
                subject_output_token_cap=(
                    extraction_config.subject_output_token_cap
                    if output_token_cap is None
                    else output_token_cap
                ),
                provider=(
                    extraction_config.provider
                    if extractor_provider is None
                    else extractor_provider
                ),
                model=(
                    extraction_config.model
                    if extractor_model is None
                    else extractor_model
                ),
                extractor_max_tokens=(
                    extraction_config.extractor_max_tokens
                    if extractor_max_tokens is None
                    else extractor_max_tokens
                ),
                timeout_seconds=extraction_config.timeout_seconds,
            )
            if requested_extraction != extraction_config:
                raise ValueError(
                    "Resume final-answer extraction configuration does not match metadata."
                )
        if extraction_config is not None:
            validate_resume_contract(
                metadata.get("resume_contract"),
                _strict_resume_contract(
                    provider=str(metadata["provider"]),
                    model=str(metadata["model"]),
                    run_parameters={
                        key: metadata[key]
                        for key in (
                            "games",
                            "frames",
                            "domains",
                            "presentations",
                            "limit",
                            "total_prompts",
                            "ordering",
                            "grading_protocol",
                        )
                    },
                    extraction_config=extraction_config,
                ),
                experiment="Part 1",
            )
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
        ordering = metadata.get("ordering")
        if isinstance(ordering, dict):
            resumed_order_seed = int(ordering["seed"])
            resumed_order_strategy = str(ordering["strategy"])
            resumed_counterbalance_index = int(ordering["counterbalance_index"])
        else:
            resumed_order_seed = None
            resumed_order_strategy = "legacy_canonical"
            resumed_counterbalance_index = 0
            attempt_log_coverage = "resume_segment_only"

        if provider is not None and provider != resumed_provider:
            raise ValueError(
                f"Resume provider mismatch: expected {resumed_provider}, received {provider}."
            )
        if model is not None and model != resumed_model:
            raise ValueError(
                f"Resume model mismatch: expected {resumed_model}, received {model}."
            )
        if (
            requested_order_seed is not None
            and requested_order_seed != resumed_order_seed
        ):
            raise ValueError(
                f"Resume order seed mismatch: expected {resumed_order_seed}, "
                f"received {requested_order_seed}."
            )
        if (
            requested_order_strategy is not None
            and requested_order_strategy != resumed_order_strategy
        ):
            raise ValueError(
                f"Resume order strategy mismatch: expected {resumed_order_strategy}, "
                f"received {requested_order_strategy}."
            )
        if (
            requested_counterbalance_index is not None
            and requested_counterbalance_index != resumed_counterbalance_index
        ):
            raise ValueError(
                "Resume counterbalance index mismatch: expected "
                f"{resumed_counterbalance_index}, received {requested_counterbalance_index}."
            )

        provider = resumed_provider
        model = resumed_model
        games = resumed_games
        frames = resumed_frames
        domains = resumed_domains
        presentations = resumed_presentations
        limit = resumed_limit if isinstance(resumed_limit, int) else None
        order_seed = resumed_order_seed
        order_strategy = resumed_order_strategy
        counterbalance_index = resumed_counterbalance_index

        if extraction_config is not None:
            artifact_integrity = metadata.get("artifact_integrity")
            if not isinstance(artifact_integrity, dict):
                raise ValueError(
                    "Strict Part 1 resume metadata is missing result integrity."
                )
            validate_file_integrity(
                csv_path,
                artifact_integrity.get("results"),
                label="Part 1 result CSV",
            )

        completed_rows = _load_part_1_rows(csv_path)
        _render_resume_panel(
            timestamp=timestamp,
            csv_path=csv_path,
            total_prompts=total_prompts,
            completed_count=len(completed_rows),
        )
    else:
        extraction_config = _fresh_extraction_config(
            output_token_cap=output_token_cap,
            extractor_provider=extractor_provider,
            extractor_model=extractor_model,
            extractor_max_tokens=extractor_max_tokens,
        )
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
        order_seed = DEFAULT_PART_1_ORDER_SEED if order_seed is None else order_seed
        order_strategy = (
            DEFAULT_PART_1_ORDER_STRATEGY
            if order_strategy is None
            else order_strategy
        )
        counterbalance_index = (
            0 if counterbalance_index is None else counterbalance_index
        )

    if order_strategy != "legacy_canonical":
        if (
            order_seed is None
            or isinstance(order_seed, bool)
            or not isinstance(order_seed, int)
        ):
            raise TypeError("order_seed must be an integer.")
        if order_strategy not in PART_1_ORDER_STRATEGIES:
            supported = ", ".join(PART_1_ORDER_STRATEGIES)
            raise ValueError(f"order_strategy must be one of: {supported}.")
    if counterbalance_index is None or counterbalance_index < 0:
        raise ValueError("counterbalance_index must be >= 0.")

    attempt_log_metadata = metadata.get("attempt_log") if is_resuming else None
    configured_attempt_path = (
        attempt_log_metadata.get("path")
        if isinstance(attempt_log_metadata, dict)
        else None
    )
    attempt_path = (
        Path(str(configured_attempt_path))
        if configured_attempt_path
        else attempt_log_path_for_csv(csv_path)
    )
    if is_resuming and not configured_attempt_path:
        attempt_log_coverage = "resume_segment_only"
    if is_resuming and extraction_config is not None:
        if not isinstance(attempt_log_metadata, dict):
            raise ValueError(
                "Strict Part 1 resume metadata is missing attempt-log integrity."
            )
        verify_attempt_log_metadata(
            attempt_path,
            attempt_log_metadata,
            require_hash_chain=True,
        )
        validate_terminal_attempt_coverage(
            attempt_path,
            {row["prompt_id"] for row in completed_rows},
        )
    attempt_logger = DurableAttemptLogger(attempt_path, experiment="part_1")

    run_experiment_preflight(
        EXPERIMENT_NAME,
        [
            (provider, model),
            *(
                [(extraction_config.provider, extraction_config.model)]
                if extraction_config is not None
                else []
            ),
        ],
        resume=is_resuming,
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
    prompt_variants = _order_prompt_variants(
        _build_prompt_variants(
            agent=agent,
            games=games,
            frames=frames,
            domains=domains,
            presentations=presentations,
            limit=None,
        ),
        seed=order_seed,
        strategy=order_strategy,
        counterbalance_index=counterbalance_index,
        limit=limit,
    )
    total_prompts = len(prompt_variants)

    if not is_resuming:
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
            order_seed=order_seed,
            order_strategy=order_strategy,
            counterbalance_index=counterbalance_index,
            attempt_logger=attempt_logger,
            extraction_config=extraction_config,
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
            mark_metadata_complete(
                metadata_path,
                completed_rows=len(completed_rows),
                extra={
                    "total_prompts": total_prompts,
                    **_attempt_log_metadata(
                        attempt_logger,
                        coverage=attempt_log_coverage,
                    ),
                    "artifact_integrity": {
                        "results": file_integrity_metadata(csv_path)
                    },
                },
            )
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
        output_headers = _part_1_output_headers(
            csv_path,
            resume=is_resuming,
        )
        prompt_positions = {
            variant.prompt_id: index
            for index, variant in enumerate(prompt_variants, start=1)
        }
        with IncrementalCsvWriter(
            csv_path,
            output_headers,
            append=is_resuming,
        ) as writer:
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
                order_position = prompt_positions[variant.prompt_id]
                action, justification = _query_variant_until_valid(
                    agent,
                    variant,
                    attempt_logger=attempt_logger,
                    order_position=order_position,
                    extraction_config=extraction_config,
                )
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
                    "order_position": order_position,
                    "order_seed": "" if order_seed is None else order_seed,
                    "order_strategy": order_strategy,
                    "counterbalance_index": counterbalance_index,
                    "action": action,
                    "justification": justification,
                    "prompt_text": variant.prompt_text,
                }
                writer.write_row([row[column] for column in output_headers])
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
        if metadata_path.exists():
            mark_metadata_failed(
                metadata_path,
                error=error,
                provider=provider,
                model=model,
                extra={
                    "completed_rows": len(all_rows),
                    **_attempt_log_metadata(
                        attempt_logger,
                        coverage=attempt_log_coverage,
                    ),
                    **(
                        {
                            "artifact_integrity": {
                                "results": file_integrity_metadata(csv_path)
                            }
                        }
                        if Path(csv_path).is_file()
                        else {}
                    ),
                },
            )
        if isinstance(error, OllamaConnectionError) or _is_ollama_resource_error(error):
            paused_error = error
        else:
            raise
    finally:
        _unload_agent_if_needed(agent)

    if interrupted:
        _sync_part_1_attempt_metadata(
            metadata_path,
            attempt_logger,
            coverage=attempt_log_coverage,
            csv_path=csv_path,
        )
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
        if not suppress_keyboard_interrupt:
            raise paused_error
        return str(csv_path)

    if metadata_path.exists():
        mark_metadata_complete(
            metadata_path,
            completed_rows=len(all_rows),
            extra={
                "total_prompts": total_prompts,
                **_attempt_log_metadata(
                    attempt_logger,
                    coverage=attempt_log_coverage,
                ),
                "artifact_integrity": {
                    "results": file_integrity_metadata(csv_path)
                },
            },
        )

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
    order_seed: int | None = None,
    order_strategy: str | None = None,
    counterbalance_index: int | None = None,
    output_token_cap: int | None = None,
    extractor_provider: str | None = None,
    extractor_model: str | None = None,
    extractor_max_tokens: int | None = None,
    *,
    resume: bool = False,
    headless: bool = False,
) -> str:
    resume_metadata_path: Path | None = None
    attempt = 0

    while attempt < MAX_RUN_ATTEMPTS:
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
                order_seed=order_seed,
                order_strategy=order_strategy,
                counterbalance_index=counterbalance_index,
                output_token_cap=output_token_cap,
                extractor_provider=extractor_provider,
                extractor_model=extractor_model,
                extractor_max_tokens=extractor_max_tokens,
                resume=resume or resume_metadata_path is not None,
                resume_metadata_path=resume_metadata_path,
                headless=headless,
                suppress_keyboard_interrupt=False,
            )
        )

        rows = _load_part_1_rows(csv_path)
        metadata_path = _metadata_path_for_csv(csv_path)
        if rows and (
            not metadata_path.exists()
            or _part_1_metadata_complete(metadata_path)
        ):
            console.print(
                f"[green]Completed non-empty part 1 run for {provider}/{model}: {csv_path}[/green]"
            )
            return str(csv_path)

        resume_metadata_path = (
            metadata_path
            if metadata_path.exists() and not _part_1_metadata_complete(metadata_path)
            else None
        )
        if resume_metadata_path is None:
            console.print(
                f"[yellow][WARN] {provider}/{model} produced no usable rows. Retrying from scratch.[/yellow]"
            )
        else:
            console.print(
                f"[yellow][WARN] {provider}/{model} is still incomplete. Retrying via resume.[/yellow]"
            )

    raise RuntimeError(
        f"Part 1 did not complete after {MAX_RUN_ATTEMPTS} run attempts for "
        f"{provider}/{model}."
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
        order_seed=cli_args.order_seed,
        order_strategy=cli_args.order_strategy,
        counterbalance_index=cli_args.counterbalance_index,
        output_token_cap=cli_args.output_token_cap,
        extractor_provider=cli_args.extractor_provider,
        extractor_model=cli_args.extractor_model,
        extractor_max_tokens=cli_args.extractor_max_tokens,
        resume=cli_args.resume,
        headless=cli_args.headless,
    )
