# print("[PART 2] Hello, World!")

import hashlib
import json
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

from agents.agent_2 import Agent2
from experiments.misc.attempt_log import (
    DurableAttemptLogger,
    attempt_log_path_for_csv,
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
    mark_metadata_complete,
    mark_metadata_failed,
    metadata_is_complete,
    read_metadata,
    safe_error_message,
    write_metadata,
)
from experiments.misc.wizard import (
    SocietyConfig,
    choose_provider_and_model,
    choose_society_config,
    parse_society_args,
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
PREFLIGHT_TEST_PATHS = [
    "tests/test_preflight.py",
    "tests/test_part_2.py",
]
AGENT_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]
ACTION_COLORS = {
    "RESTRAIN": "green",
    "OVERUSE": "red",
}
PROMPT_ACTION_ALIASES = {
    "A": "RESTRAIN",
    "OPTION_A": "RESTRAIN",
    "ACTION_A": "RESTRAIN",
    "RESTRAIN": "RESTRAIN",
    "B": "OVERUSE",
    "OPTION_B": "OVERUSE",
    "ACTION_B": "OVERUSE",
    "OVERUSE": "OVERUSE",
}
PART_2_PROMPTS = load_prompt_config("part_2")
EXPERIMENT_NAME = PART_2_PROMPTS["experiment_name"]
PROMPT_CONFIG_HASH = hashlib.sha256(
    json.dumps(
        PART_2_PROMPTS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
RESOURCE_RESERVE_MULTIPLIER = 10
COLLAPSE_ATTRITION_DIVISOR = 5
DEFAULT_COLLAPSE_DEATH_RATE = 1 / COLLAPSE_ATTRITION_DIVISOR
MAX_REASONING_SAMPLES_PER_DAY = 3
MODEL_BATCH_KEEP_ALIVE = "30m"
INITIAL_RETRY_DELAY_SECONDS = 1.0
MAX_RETRY_DELAY_SECONDS = 30.0
MAX_AGENT_ATTEMPTS = 3
MAX_RUN_ATTEMPTS = 3
PART_2_RESULTS_DIR = Path("data") / "raw" / "part_2"
PART_2_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
RESULT_HEADERS = [
    "provider",
    "model",
    "day",
    "agent",
    "action",
    "reasoning",
    "population_start",
    "population_end",
    "restrain_count",
    "overuse_count",
    "resource_units_remaining",
    "resource_capacity",
    "deaths",
    "resource",
    "selfish_gain",
    "depletion_units",
    "community_benefit",
]
LEGACY_RESULT_HEADERS = [
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


@dataclass(frozen=True)
class ResumeState:
    completed_days: int
    resource_units: int
    previous_overuse_count: int | None
    population: int
    collapse_announced: bool
    rows: list[dict[str, str]]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-")


def _build_result_filename(
    *,
    provider: str,
    model: str,
    config: SocietyConfig,
    timestamp: str,
    resource_capacity: int | None = None,
    collapse_death_rate: float = DEFAULT_COLLAPSE_DEATH_RATE,
    seed: int | None = None,
) -> str:
    day_label = "open" if config.days == 0 else f"d{config.days}"
    segments = [
        "part2",
        _slugify(provider),
        _slugify(model),
        f"n{config.society_size}",
        day_label,
        _slugify(config.resource),
    ]
    if resource_capacity is not None:
        segments.append(f"c{resource_capacity}")
        segments.append(f"du{config.depletion_units}")
    if collapse_death_rate != DEFAULT_COLLAPSE_DEATH_RATE:
        segments.append(f"dr{repr(collapse_death_rate).replace('.', 'p')}")
    if seed is not None:
        segments.append(f"s{seed}")
    segments.append(timestamp)
    return "__".join(segments) + ".csv"


def _metadata_path_for_csv(csv_path: str | Path) -> Path:
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_meta.json")


def _config_to_metadata(config: SocietyConfig) -> dict[str, Any]:
    return {
        "society_size": config.society_size,
        "days": config.days,
        "resource": config.resource,
        "selfish_gain": config.selfish_gain,
        "depletion_units": config.depletion_units,
        "community_benefit": config.community_benefit,
    }


def _config_from_metadata(metadata: dict[str, Any]) -> SocietyConfig:
    config = metadata.get("society_config", {})
    if not isinstance(config, dict):
        raise ValueError("Part 2 metadata is missing society_config.")
    return SocietyConfig(
        society_size=int(config["society_size"]),
        days=int(config["days"]),
        resource=str(config["resource"]),
        selfish_gain=int(config["selfish_gain"]),
        depletion_units=int(config["depletion_units"]),
        community_benefit=int(config["community_benefit"]),
    )


def _write_part_2_metadata(
    path: str | Path,
    *,
    timestamp: str,
    csv_path: str | Path,
    provider: str,
    model: str,
    config: SocietyConfig,
    resource_capacity: int,
    collapse_death_rate: float,
    seed: int | None,
    attempt_logger: DurableAttemptLogger,
    extraction_config: ExtractionConfig | None,
) -> None:
    parameters = {
        "society_config": _config_to_metadata(config),
        "resource_capacity": resource_capacity,
        "collapse_death_rate": collapse_death_rate,
        "generation_seed": seed,
        "grading_protocol": (
            extraction_config.to_metadata() if extraction_config is not None else None
        ),
    }
    metadata = base_run_metadata(
        experiment="part_2",
        timestamp=timestamp,
        csv_path=csv_path,
        provider=provider,
        model=model,
        parameters=parameters,
        prompt_config_hash=PROMPT_CONFIG_HASH,
    )
    metadata.update(parameters)
    attempt_log = attempt_logger.summary().to_metadata()
    attempt_log["coverage"] = "full_run"
    metadata["attempt_log"] = attempt_log
    write_metadata(path, metadata)


def _load_part_2_metadata(path: str | Path) -> dict[str, Any]:
    return read_metadata(path)


def _fresh_extraction_config(
    *,
    output_token_cap: int | None,
    extractor_provider: str | None,
    extractor_model: str | None,
    extractor_max_tokens: int | None,
) -> ExtractionConfig:
    return ExtractionConfig(
        subject_output_token_cap=(output_token_cap or DEFAULT_OUTPUT_TOKEN_CAP),
        provider=(extractor_provider or DEFAULT_EXTRACTOR_PROVIDER),
        model=(extractor_model or DEFAULT_EXTRACTOR_MODEL),
        extractor_max_tokens=(extractor_max_tokens or DEFAULT_EXTRACTOR_MAX_TOKENS),
    )


def _resume_extraction_config(metadata: dict[str, Any]) -> ExtractionConfig | None:
    value = metadata.get("grading_protocol")
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


def _sync_part_2_attempt_metadata(
    metadata_path: str | Path,
    logger: DurableAttemptLogger,
    *,
    coverage: str,
) -> None:
    path = Path(metadata_path)
    if not path.exists():
        return
    metadata = read_metadata(path)
    metadata.update(_attempt_log_metadata(logger, coverage=coverage))
    write_metadata(path, metadata)


def _metadata_uses_current_part_2_prompt(metadata: dict[str, Any]) -> bool:
    return metadata.get("prompt_config_hash") == PROMPT_CONFIG_HASH


def _latest_interrupted_part_2_metadata_path() -> Path:
    if not PART_2_RESULTS_DIR.exists():
        raise ValueError(
            f"No interrupted part 2 run metadata was found in {PART_2_RESULTS_DIR}."
        )

    candidates: list[tuple[datetime, Path]] = []
    for metadata_path in PART_2_RESULTS_DIR.glob("*_meta.json"):
        try:
            metadata = _load_part_2_metadata(metadata_path)
            if metadata_is_complete(metadata):
                continue
            timestamp = str(metadata.get("timestamp", "")).strip()
            parsed = datetime.strptime(timestamp, PART_2_TIMESTAMP_FORMAT)
        except Exception:
            continue
        candidates.append((parsed, metadata_path))

    if not candidates:
        raise ValueError(
            f"No interrupted part 2 run metadata was found in {PART_2_RESULTS_DIR}."
        )

    candidates.sort()
    return candidates[-1][1]


def _matching_part_2_metadata_path(
    *,
    provider: str,
    model: str,
    config: SocietyConfig,
    resource_capacity: int | None = None,
    collapse_death_rate: float = DEFAULT_COLLAPSE_DEATH_RATE,
    seed: int | None = None,
) -> Path | None:
    if not PART_2_RESULTS_DIR.exists():
        return None

    candidates: list[tuple[datetime, Path]] = []
    expected_config = _config_to_metadata(config)
    for metadata_path in PART_2_RESULTS_DIR.glob("*_meta.json"):
        try:
            metadata = _load_part_2_metadata(metadata_path)
            if metadata_is_complete(metadata):
                continue
            if str(metadata.get("provider", "")) != provider:
                continue
            if str(metadata.get("model", "")) != model:
                continue
            if metadata.get("society_config") != expected_config:
                continue
            expected_capacity = (
                _initial_resource_units(config)
                if resource_capacity is None
                else resource_capacity
            )
            if metadata.get("resource_capacity") != expected_capacity:
                continue
            if float(
                metadata.get("collapse_death_rate", DEFAULT_COLLAPSE_DEATH_RATE)
            ) != collapse_death_rate:
                continue
            if metadata.get("generation_seed") != seed:
                continue
            if not _metadata_uses_current_part_2_prompt(metadata):
                continue
            timestamp = str(metadata.get("timestamp", "")).strip()
            candidates.append((datetime.strptime(timestamp, PART_2_TIMESTAMP_FORMAT), metadata_path))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def _load_part_2_rows(path: str | Path) -> list[dict[str, str]]:
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


def _complete_day_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_day: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        try:
            day = int(row["day"])
        except ValueError:
            continue
        by_day[day].append(row)

    complete_rows: list[dict[str, str]] = []
    for day in sorted(by_day):
        day_rows = by_day[day]
        try:
            population_start = int(day_rows[0]["population_start"])
        except ValueError:
            break
        if len(day_rows) != population_start:
            break
        complete_rows.extend(day_rows)
    return complete_rows


def _rewrite_part_2_rows(path: str | Path, rows: list[dict[str, str]]) -> None:
    with IncrementalCsvWriter(path, RESULT_HEADERS, append=False) as writer:
        writer.write_rows(
            [[row.get(column, "") for column in RESULT_HEADERS] for row in rows]
        )


def _resume_state_from_rows(
    *,
    rows: list[dict[str, str]],
    config: SocietyConfig,
    resource_capacity: int,
) -> ResumeState:
    complete_rows = _complete_day_rows(rows)
    if len(complete_rows) != len(rows):
        complete_rows = list(complete_rows)

    if not complete_rows:
        return ResumeState(
            completed_days=0,
            resource_units=resource_capacity,
            previous_overuse_count=None,
            population=config.society_size,
            collapse_announced=False,
            rows=[],
        )

    last_row = complete_rows[-1]
    return ResumeState(
        completed_days=int(last_row["day"]),
        resource_units=int(last_row["resource_units_remaining"]),
        previous_overuse_count=int(last_row["overuse_count"]),
        population=int(last_row["population_end"]),
        collapse_announced=any(
            int(row["resource_units_remaining"]) == 0 for row in complete_rows
        ),
        rows=complete_rows,
    )


def _build_agents(
    provider: str,
    model: str,
    count: int,
    *,
    seed: int | None = None,
    keep_alive: float | str | None = None,
) -> list[Agent2]:
    return [
        Agent2(
            id_=f"society_{idx + 1}",
            provider_=provider,
            model_=model,
            seed_=seed,
            keep_alive_=keep_alive,
        )
        for idx in range(count)
    ]


def _initial_resource_units(config: SocietyConfig) -> int:
    baseline_units = max(config.community_benefit, config.depletion_units, 1)
    return max(
        config.society_size * baseline_units * RESOURCE_RESERVE_MULTIPLIER,
        config.society_size,
    )


def _normalize_action(raw_action: str) -> str:
    action = raw_action.strip().upper().replace("-", "_").replace(" ", "_")
    action = PROMPT_ACTION_ALIASES.get(action, action)
    if action not in ACTION_COLORS:
        supported = ", ".join(sorted(PROMPT_ACTION_ALIASES))
        raise ResponseParseError(
            f"Unsupported society action '{raw_action}'. Expected one of: {supported}."
        )
    return action


def _collapse_deaths(
    population: int,
    resource_units: int,
    death_rate: float = DEFAULT_COLLAPSE_DEATH_RATE,
) -> int:
    if population <= 0 or resource_units > 0:
        return 0
    return min(population, max(1, ceil(population * death_rate)))


def _should_show_reasoning_samples(
    *,
    day: int,
    configured_days: int,
    collapsed_today: bool,
) -> bool:
    if configured_days and configured_days <= 3:
        return True
    return day == 1 or collapsed_today


def _build_headless_progress_bar(
    *,
    completed_count: int,
    total_count: int,
    width: int = 20,
) -> str:
    if total_count < 1:
        return "[" + ("-" * width) + "]"
    normalized_completed = min(max(completed_count, 0), total_count)
    filled = round((normalized_completed / total_count) * width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _emit_headless_status_line(message: str, *, finalize: bool = False) -> None:
    sanitized = message.replace("\r", " ").replace("\n", " ")
    suffix = "\n" if finalize else ""
    console.file.write(f"\r\x1b[2K{sanitized}{suffix}")
    console.file.flush()


def _render_headless_day_start(
    *,
    provider: str,
    model: str,
    day: int,
    total_days: int,
    population: int,
    resource_units: int,
    resource_capacity: int,
) -> None:
    progress_bar = _build_headless_progress_bar(
        completed_count=day - 1,
        total_count=total_days,
    )
    total_label = "open" if total_days == 0 else str(total_days)
    _emit_headless_status_line(
        f"Model {provider}/{model} [day {day}/{total_label}] {progress_bar} "
        f"RUNNING pop={population} reserve={resource_units}/{resource_capacity}",
    )


def _render_headless_day_complete(
    *,
    provider: str,
    model: str,
    summary: DaySummary,
    total_days: int,
) -> None:
    progress_bar = _build_headless_progress_bar(
        completed_count=summary.day,
        total_count=total_days,
    )
    total_label = "open" if total_days == 0 else str(total_days)
    _emit_headless_status_line(
        f"Model {provider}/{model} [day {summary.day}/{total_label}] {progress_bar} "
        f"done pop={summary.population_end} restrain={summary.restrain_count} "
        f"overuse={summary.overuse_count} reserve={summary.resource_units}/"
        f"{summary.resource_capacity} deaths={summary.deaths}",
        finalize=True,
    )


def _emit_retry_status_line(message: str, *, finalize: bool = False) -> None:
    sanitized = message.replace("\r", " ").replace("\n", " ")
    plain_message = console.render_str(sanitized).plain
    width = max(20, min(console.width or shutil.get_terminal_size((120, 24)).columns, 240))
    max_message_width = max(1, width - 1)
    if len(plain_message) > max_message_width:
        suffix = "..."
        if max_message_width <= len(suffix):
            plain_message = suffix[:max_message_width]
        else:
            plain_message = plain_message[: max_message_width - len(suffix)] + suffix
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


def _unload_agent_if_needed(agent: Agent2) -> None:
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


def _recover_agent_after_error(agent: Agent2, error: Exception) -> None:
    if agent.provider.strip().lower() != "ollama":
        return
    if not _is_ollama_resource_error(error):
        return

    _unload_agent_if_needed(agent)
    _prepare_ollama_model_for_run(provider=agent.provider, model=agent.model)


def _parse_agent_response(raw_response: str) -> tuple[str, str]:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as error:
        raise ResponseParseError("Invalid JSON in part 2 response.") from error
    action = _normalize_action(str(data.get("action", "")))
    reasoning = str(data.get("reasoning", "")).strip()
    if not reasoning:
        raise ResponseParseError("Missing reasoning in part 2 response.")
    return action, reasoning


def _query_agent_until_valid(
    agent: Agent2,
    prompt: str,
    *,
    attempt_logger: DurableAttemptLogger | None = None,
    unit: dict[str, Any] | None = None,
    extraction_config: ExtractionConfig | None = None,
) -> tuple[str, str]:
    had_retry_status = False
    attempt = 0
    resolved_unit = dict(unit or {"agent": agent.id})
    unit_id = "__".join(
        str(resolved_unit.get(key, ""))
        for key in ("day", "agent")
    ).strip("_") or agent.id

    while attempt < MAX_AGENT_ATTEMPTS:
        attempt += 1
        try:
            generation_record = None
            if extraction_config is None:
                raw = agent.query(prompt, json_mode=True)
            else:
                raw, generated = agent.query_for_grading(
                    prompt,
                    extraction_config=extraction_config,
                    extraction_kind="part_2_commons_decision",
                )
                generation_record = generated.to_dict()
        except KeyboardInterrupt as error:
            if attempt_logger is not None:
                attempt_logger.append(
                    provider=agent.provider,
                    model=agent.model,
                    unit_id=unit_id,
                    unit=resolved_unit,
                    attempt=attempt,
                    max_attempts=MAX_AGENT_ATTEMPTS,
                    prompt_text=prompt,
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
                is_retryable_api_failure(error)
                or _is_ollama_resource_error(error)
            ) and not isinstance(error, OllamaConnectionError)
            will_retry = retryable and attempt < MAX_AGENT_ATTEMPTS
            if attempt_logger is not None:
                attempt_logger.append(
                    provider=agent.provider,
                    model=agent.model,
                    unit_id=unit_id,
                    unit=resolved_unit,
                    attempt=attempt,
                    max_attempts=MAX_AGENT_ATTEMPTS,
                    prompt_text=prompt,
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
            action, reasoning = _parse_agent_response(raw)
        except Exception as error:
            retryable = is_retryable_api_failure(error)
            will_retry = retryable and attempt < MAX_AGENT_ATTEMPTS
            if attempt_logger is not None:
                attempt_logger.append(
                    provider=agent.provider,
                    model=agent.model,
                    unit_id=unit_id,
                    unit=resolved_unit,
                    attempt=attempt,
                    max_attempts=MAX_AGENT_ATTEMPTS,
                    prompt_text=prompt,
                    outcome="invalid_response",
                    raw_response=raw,
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
                unit_id=unit_id,
                unit=resolved_unit,
                attempt=attempt,
                max_attempts=MAX_AGENT_ATTEMPTS,
                prompt_text=prompt,
                outcome="success",
                raw_response=raw,
                parsed_response={"action": action, "reasoning": reasoning},
                generation_record=generation_record,
            )
        if had_retry_status:
            _emit_retry_status_line("")
        return action, reasoning


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


def _render_resume_panel(
    *,
    timestamp: str,
    csv_path: Path,
    config: SocietyConfig,
    completed_days: int,
    population: int,
    resource_units: int,
    resource_capacity: int,
) -> None:
    total_days = (
        "until population dies out"
        if config.days == 0
        else str(config.days)
    )
    remaining_days = (
        "open-ended"
        if config.days == 0 and population > 0
        else str(max(0, config.days - completed_days))
    )
    console.print(
        Panel(
            f"[bold]Timestamp:[/bold] {timestamp}\n"
            f"[bold]Results:[/bold] [green]{csv_path.resolve()}[/green]\n"
            f"[bold]Completed days:[/bold] {completed_days} / {total_days}\n"
            f"[bold]Remaining days:[/bold] {remaining_days}\n"
            f"[bold]Current population:[/bold] {population}\n"
            f"[bold]Reserve:[/bold] {resource_units}/{resource_capacity}",
            title="[bold cyan]Resuming Part 2 Run[/bold cyan]",
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


def run_part_2(
    provider: str | None = None,
    model: str | None = None,
    society_size: int | None = None,
    days: int | None = None,
    resource: str | None = None,
    selfish_gain: int | None = None,
    depletion_units: int | None = None,
    community_benefit: int | None = None,
    resource_capacity: int | None = None,
    collapse_death_rate: float | None = None,
    seed: int | None = None,
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
    attempt_log_coverage = "full_run"
    requested_resource_capacity = resource_capacity
    requested_death_rate = collapse_death_rate
    requested_seed = seed

    if is_resuming:
        metadata_path = (
            Path(resume_metadata_path)
            if resume_metadata_path is not None
            else _latest_interrupted_part_2_metadata_path()
        )
        metadata = _load_part_2_metadata(metadata_path)
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
                    "Cannot add final-answer extraction while resuming a legacy Part 2 run."
                )
        else:
            requested_extraction = ExtractionConfig(
                subject_output_token_cap=(
                    output_token_cap or extraction_config.subject_output_token_cap
                ),
                provider=extractor_provider or extraction_config.provider,
                model=extractor_model or extraction_config.model,
                extractor_max_tokens=(
                    extractor_max_tokens or extraction_config.extractor_max_tokens
                ),
                timeout_seconds=extraction_config.timeout_seconds,
            )
            if requested_extraction != extraction_config:
                raise ValueError(
                    "Resume final-answer extraction configuration does not match metadata."
                )
        if not _metadata_uses_current_part_2_prompt(metadata):
            raise ValueError(
                "Cannot resume this part 2 run because its metadata was created "
                "with an older prompt configuration. Start a fresh run for this "
                "model/configuration so results do not mix prompt versions."
            )
        resumed_provider = str(metadata["provider"])
        resumed_model = str(metadata["model"])
        resumed_config = _config_from_metadata(metadata)
        timestamp = str(metadata["timestamp"])
        csv_path = Path(str(metadata["csv_path"]))
        resource_capacity = int(metadata["resource_capacity"])
        resumed_death_rate = float(
            metadata.get("collapse_death_rate", DEFAULT_COLLAPSE_DEATH_RATE)
        )
        resumed_seed = metadata.get("generation_seed")

        if provider is not None and provider != resumed_provider:
            raise ValueError(
                f"Resume provider mismatch: expected {resumed_provider}, received {provider}."
            )
        if model is not None and model != resumed_model:
            raise ValueError(
                f"Resume model mismatch: expected {resumed_model}, received {model}."
            )
        if (
            requested_resource_capacity is not None
            and requested_resource_capacity != int(metadata["resource_capacity"])
        ):
            raise ValueError(
                "Resume resource capacity mismatch: expected "
                f"{metadata['resource_capacity']}, received {requested_resource_capacity}."
            )
        if requested_death_rate is not None and requested_death_rate != resumed_death_rate:
            raise ValueError(
                "Resume collapse death rate mismatch: expected "
                f"{resumed_death_rate}, received {requested_death_rate}."
            )
        if requested_seed is not None and requested_seed != resumed_seed:
            raise ValueError(
                f"Resume generation seed mismatch: expected {resumed_seed}, received {requested_seed}."
            )

        provider = resumed_provider
        model = resumed_model
        society_config = resumed_config
        collapse_death_rate = resumed_death_rate
        seed = resumed_seed
        completed_rows = _load_part_2_rows(csv_path)
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
                experiment_key="part_2",
                provider=provider,
                model=model,
            )

        if headless:
            defaults = SocietyConfig()
            society_size = defaults.society_size if society_size is None else society_size
            days = defaults.days if days is None else days
            resource = defaults.resource if resource is None else resource
            selfish_gain = defaults.selfish_gain if selfish_gain is None else selfish_gain
            depletion_units = (
                defaults.depletion_units
                if depletion_units is None
                else depletion_units
            )
            community_benefit = (
                defaults.community_benefit
                if community_benefit is None
                else community_benefit
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
        if resource_capacity is not None and resource_capacity <= 0:
            raise ValueError("resource_capacity must be greater than 0.")
        if collapse_death_rate is None:
            collapse_death_rate = DEFAULT_COLLAPSE_DEATH_RATE
        if not 0 < collapse_death_rate <= 1:
            raise ValueError("collapse_death_rate must be greater than 0 and at most 1.")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer.")
        timestamp = datetime.now().strftime(PART_2_TIMESTAMP_FORMAT)
        csv_path = PART_2_RESULTS_DIR / _build_result_filename(
            provider=provider,
            model=model,
            config=society_config,
            timestamp=timestamp,
            resource_capacity=resource_capacity,
            collapse_death_rate=collapse_death_rate,
            seed=seed,
        )
        metadata_path = _metadata_path_for_csv(csv_path)
        resource_capacity = (
            _initial_resource_units(society_config)
            if resource_capacity is None
            else resource_capacity
        )

    assert collapse_death_rate is not None

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
    attempt_logger = DurableAttemptLogger(attempt_path, experiment="part_2")

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

    if not is_resuming:
        _write_part_2_metadata(
            metadata_path,
            timestamp=timestamp,
            csv_path=csv_path,
            provider=provider,
            model=model,
            config=society_config,
            resource_capacity=resource_capacity,
            collapse_death_rate=collapse_death_rate,
            seed=seed,
            attempt_logger=attempt_logger,
            extraction_config=extraction_config,
        )

    resume_state = _resume_state_from_rows(
        rows=completed_rows,
        config=society_config,
        resource_capacity=resource_capacity,
    )
    if len(resume_state.rows) != len(completed_rows):
        _rewrite_part_2_rows(csv_path, resume_state.rows)
    completed_days = resume_state.completed_days
    resource_units = resume_state.resource_units
    previous_overuse_count = resume_state.previous_overuse_count
    collapse_announced = resume_state.collapse_announced
    keep_alive = MODEL_BATCH_KEEP_ALIVE if provider.strip().lower() == "ollama" else None
    agents = _build_agents(
        provider,
        model,
        resume_state.population,
        seed=seed,
        keep_alive=keep_alive,
    )
    stop_reason = ""
    interrupted = False
    paused_error: Exception | None = None

    runtime_label = (
        "until population dies out (Ctrl+C to stop)"
        if society_config.days == 0
        else f"{society_config.days} days"
    )

    if is_resuming:
        _render_resume_panel(
            timestamp=timestamp,
            csv_path=csv_path,
            config=society_config,
            completed_days=completed_days,
            population=len(agents),
            resource_units=resource_units,
            resource_capacity=resource_capacity,
        )

    if not agents or (
        society_config.days > 0 and completed_days >= society_config.days
    ):
        if metadata_path.exists():
            mark_metadata_complete(
                metadata_path,
                completed_rows=len(completed_rows),
                extra={
                    "completed_days": completed_days,
                    **_attempt_log_metadata(
                        attempt_logger,
                        coverage=attempt_log_coverage,
                    ),
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

    if not headless:
        console.print(
            Panel(
                f"[bold]{EXPERIMENT_NAME}[/bold]\n"
                f"Model: {provider}/{model}\n"
                f"Agents: {society_config.society_size}\n"
                f"Runtime: {runtime_label}\n"
                f"Completed days: {completed_days}\n"
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
        with IncrementalCsvWriter(csv_path, RESULT_HEADERS, append=is_resuming) as writer:
            while agents and (society_config.days == 0 or completed_days < society_config.days):
                day = completed_days + 1
                if headless:
                    _render_headless_day_start(
                        provider=provider,
                        model=model,
                        day=day,
                        total_days=society_config.days,
                        population=len(agents),
                        resource_units=resource_units,
                        resource_capacity=resource_capacity,
                    )
                else:
                    console.rule(f"[bold]Day {day}[/bold]")

                population_start = len(agents)
                daily_decisions: list[dict[str, str]] = []
                for agent_index, agent in enumerate(agents, start=1):
                    if not headless:
                        _emit_retry_status_line(
                            f"  Agent {agent.id} {agent_index}/{population_start}: querying..."
                        )
                    prompt = agent.build_commons_prompt(
                        resource=society_config.resource,
                        selfish_gain=society_config.selfish_gain,
                        depletion_units=society_config.depletion_units,
                        community_benefit=society_config.community_benefit,
                        day=day,
                        living_agents=population_start,
                        resource_units=resource_units,
                        resource_capacity=resource_capacity,
                        previous_overuse_count=previous_overuse_count,
                    )
                    action, reasoning = _query_agent_until_valid(
                        agent,
                        prompt,
                        attempt_logger=attempt_logger,
                        unit={
                            "day": day,
                            "agent": agent.id,
                            "agent_index": agent_index,
                            "population_start": population_start,
                        },
                        extraction_config=extraction_config,
                    )
                    if not headless:
                        _emit_retry_status_line("")
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
                    if not headless:
                        _render_collapse_warning(day, society_config.resource)

                deaths = _collapse_deaths(
                    population_start,
                    resource_units,
                    collapse_death_rate,
                )
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
                if headless:
                    _render_headless_day_complete(
                        provider=provider,
                        model=model,
                        summary=summary,
                        total_days=society_config.days,
                    )
                else:
                    _render_day_summary(summary)

                if not headless and _should_show_reasoning_samples(
                    day=day,
                    configured_days=society_config.days,
                    collapsed_today=collapsed_today,
                ):
                    _render_reasoning_samples(day, daily_decisions)

                writer.write_rows(
                    [
                        [
                            provider,
                            model,
                            day,
                            decision["agent"],
                            decision["action"],
                            decision["reasoning"],
                            population_start,
                            population_end,
                            restrain_count,
                            overuse_count,
                            resource_units,
                            resource_capacity,
                            deaths,
                            society_config.resource,
                            society_config.selfish_gain,
                            society_config.depletion_units,
                            society_config.community_benefit,
                        ]
                        for decision in daily_decisions
                    ]
                )

                previous_overuse_count = overuse_count
                completed_days += 1

                if not agents:
                    stop_reason = f"Population died out on day {day}."
                    break
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = (
            f"Simulation interrupted by user after {completed_days} completed day(s)."
        )
    except Exception as error:
        if metadata_path.exists():
            mark_metadata_failed(
                metadata_path,
                error=error,
                provider=provider,
                model=model,
                extra={
                    "completed_rows": len(_load_part_2_rows(csv_path)),
                    "completed_days": completed_days,
                    **_attempt_log_metadata(
                        attempt_logger,
                        coverage=attempt_log_coverage,
                    ),
                },
            )
        if isinstance(error, OllamaConnectionError) or _is_ollama_resource_error(error):
            paused_error = error
        else:
            raise
    finally:
        if provider.strip().lower() == "ollama":
            try:
                unload_ollama_model(model)
            except OllamaConnectionError as error:
                console.print(
                    f"  [yellow][WARN] Could not unload Ollama model {model}: {error}[/yellow]"
                )

    if interrupted:
        _sync_part_2_attempt_metadata(
            metadata_path,
            attempt_logger,
            coverage=attempt_log_coverage,
        )
        if not suppress_keyboard_interrupt:
            raise KeyboardInterrupt()
        console.print(
            Panel(
                f"{stop_reason}\n\n"
                f"Saved partial results to:\n[green]{csv_path.resolve()}[/green]\n\n"
                "Rerun with `--resume` to continue this model.",
                title="[bold yellow]Simulation Interrupted[/bold yellow]",
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

    if not stop_reason:
        if society_config.days == 0:
            stop_reason = (
                f"Simulation ended after {completed_days} completed day(s) with "
                f"{len(agents)} agents remaining."
            )
        else:
            stop_reason = f"Reached the configured limit of {society_config.days} day(s)."

    run_complete = (
        not interrupted
        and (
            not agents
            or (society_config.days > 0 and completed_days >= society_config.days)
        )
    )
    if run_complete and metadata_path.exists():
        mark_metadata_complete(
            metadata_path,
            completed_rows=sum(1 for _ in _load_part_2_rows(csv_path)),
            extra={
                "completed_days": completed_days,
                "final_population": len(agents),
                "final_resource_units": resource_units,
                "stop_reason": stop_reason,
                **_attempt_log_metadata(
                    attempt_logger,
                    coverage=attempt_log_coverage,
                ),
            },
        )

    if not headless:
        console.print(
            Panel(
                f"[bold]Days completed:[/bold] {completed_days}\n"
                f"[bold]Final population:[/bold] {len(agents)}\n"
                f"[bold]Reserve remaining:[/bold] {resource_units}/{resource_capacity}\n"
                f"[bold]Stop reason:[/bold] {stop_reason}\n\n"
                f"[green]{Path(csv_path).resolve()}[/green]",
                title="[bold]Results Saved[/bold]",
                border_style="green",
                expand=True,
            )
        )

    return str(csv_path)


def run_part_2_until_complete(
    provider: str | None = None,
    model: str | None = None,
    society_size: int | None = None,
    days: int | None = None,
    resource: str | None = None,
    selfish_gain: int | None = None,
    depletion_units: int | None = None,
    community_benefit: int | None = None,
    resource_capacity: int | None = None,
    collapse_death_rate: float | None = None,
    seed: int | None = None,
    output_token_cap: int | None = None,
    extractor_provider: str | None = None,
    extractor_model: str | None = None,
    extractor_max_tokens: int | None = None,
    *,
    resume: bool = False,
    headless: bool = False,
) -> str:
    resume_metadata_path: Path | None = None
    if not resume and provider is not None and model is not None:
        default_config = SocietyConfig()
        candidate_config = SocietyConfig(
            society_size=(
                default_config.society_size
                if society_size is None and headless
                else society_size
            ),
            days=default_config.days if days is None and headless else days,
            resource=(
                default_config.resource
                if resource is None and headless
                else resource
            ),
            selfish_gain=(
                default_config.selfish_gain
                if selfish_gain is None and headless
                else selfish_gain
            ),
            depletion_units=(
                default_config.depletion_units
                if depletion_units is None and headless
                else depletion_units
            ),
            community_benefit=(
                default_config.community_benefit
                if community_benefit is None and headless
                else community_benefit
            ),
        )
        if all(
            value is not None
            for value in (
                candidate_config.society_size,
                candidate_config.days,
                candidate_config.resource,
                candidate_config.selfish_gain,
                candidate_config.depletion_units,
                candidate_config.community_benefit,
            )
        ):
            resume_metadata_path = _matching_part_2_metadata_path(
                provider=provider,
                model=model,
                config=candidate_config,
                resource_capacity=resource_capacity,
                collapse_death_rate=(
                    DEFAULT_COLLAPSE_DEATH_RATE
                    if collapse_death_rate is None
                    else collapse_death_rate
                ),
                seed=seed,
                output_token_cap=output_token_cap,
                extractor_provider=extractor_provider,
                extractor_model=extractor_model,
                extractor_max_tokens=extractor_max_tokens,
            )

    attempt = 0
    while attempt < MAX_RUN_ATTEMPTS:
        attempt += 1
        console.print(
            f"[cyan]Part 2 run attempt {attempt} for {provider}/{model}[/cyan]"
        )
        csv_path = Path(
            run_part_2(
                provider=provider,
                model=model,
                society_size=society_size,
                days=days,
                resource=resource,
                selfish_gain=selfish_gain,
                depletion_units=depletion_units,
                community_benefit=community_benefit,
                resource_capacity=resource_capacity,
                collapse_death_rate=collapse_death_rate,
                seed=seed,
                resume=resume or resume_metadata_path is not None,
                resume_metadata_path=resume_metadata_path,
                headless=headless,
                suppress_keyboard_interrupt=False,
            )
        )

        metadata_path = _metadata_path_for_csv(csv_path)
        metadata_complete = False
        if metadata_path.exists():
            try:
                metadata_complete = metadata_is_complete(_load_part_2_metadata(metadata_path))
            except Exception:
                metadata_complete = False
        if not metadata_path.exists() or metadata_complete:
            console.print(
                f"[green]Completed part 2 run for {provider}/{model}: {csv_path}[/green]"
            )
            return str(csv_path)

        resume_metadata_path = metadata_path
        console.print(
            f"[yellow][WARN] {provider}/{model} is still incomplete. Retrying via resume.[/yellow]"
        )

    raise RuntimeError(
        f"Part 2 did not complete after {MAX_RUN_ATTEMPTS} run attempts for "
        f"{provider}/{model}."
    )


if __name__ == "__main__":
    cli_args = parse_society_args()
    run_part_2_until_complete(
        provider=cli_args.provider,
        model=cli_args.model,
        society_size=cli_args.society_size,
        days=cli_args.days,
        resource=cli_args.resource,
        selfish_gain=cli_args.selfish_gain,
        depletion_units=cli_args.depletion_units,
        community_benefit=cli_args.community_benefit,
        resource_capacity=cli_args.resource_capacity,
        collapse_death_rate=cli_args.collapse_death_rate,
        seed=cli_args.seed,
        output_token_cap=cli_args.output_token_cap,
        extractor_provider=cli_args.extractor_provider,
        extractor_model=cli_args.extractor_model,
        extractor_max_tokens=cli_args.extractor_max_tokens,
        resume=cli_args.resume,
        headless=cli_args.headless,
    )
