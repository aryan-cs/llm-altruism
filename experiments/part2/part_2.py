# print("[PART 2] Hello, World!")

import hashlib
import json
import random
import re
import shutil
import time
import uuid
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any, Mapping

from agents.agent_2 import Agent2, OPTION_A_PRIVATE_PAYOFF
from experiments.misc.attempt_log import (
    DurableAttemptLogger,
    attempt_log_path_for_csv,
    iter_attempt_records,
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
    validate_resume_contract,
    validate_file_integrity,
    validate_metadata_integrity,
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
    unload_all_ollama_models,
    unload_ollama_model,
)
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from analysis.part2_dynamics import (
    structural_cell_id_for_contract,
    trajectory_id_for_contract,
)

console = Console()
PREFLIGHT_TEST_PATHS = [
    "tests/test_preflight.py",
    "tests/test_part_2.py",
]
AGENT_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red"]
ACTION_COLORS = {
    "RESTRAIN": "green",
    "OVERUSE": "red",
    "INVALID": "yellow",
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
PART_2_SCHEMA_VERSION = 4
SEED_DERIVATION_POLICY = "sha256_run_day_anonymous_slot_v1"
ATTRITION_POLICY = "python_random_sample_living_slots_per_day_v1"
INVALID_ACTION_POLICY = "no_reserve_action_count_separately_v1"
INCENTIVE_POLICY = "recorded_private_and_collective_payoffs_v2"
DEFAULT_DIRECT_OUTPUT_TOKEN_CAP = 32
DIRECT_GENERATION_PROTOCOL_VERSION = "part_2_direct_generation_v2"
RESULT_HEADERS = [
    "run_id",
    "trajectory_id",
    "structural_cell_id",
    "provider",
    "model",
    "day",
    "agent",
    "anonymous_agent_slot",
    "action",
    "private_payoff_delta",
    "cumulative_private_payoff",
    "group_payoff_delta",
    "cumulative_group_payoff",
    "reasoning",
    "invalid_reason",
    "attempt_outcome",
    "attempt_count",
    "environment_seed",
    "generation_seed",
    "call_seed",
    "requested_model",
    "returned_model",
    "request_id",
    "finish_reason",
    "usage_json",
    "raw_response_sha256",
    "population_start",
    "population_end",
    "restrain_count",
    "overuse_count",
    "invalid_count",
    "resource_units_remaining",
    "resource_capacity",
    "deaths",
    "died_today",
    "attrition_rank",
    "attrition_seed",
    "death_selected_slots_json",
    "resource",
    "selfish_gain",
    "depletion_units",
    "community_benefit",
]
PILOT_RESULT_HEADERS = [
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
    invalid_count: int
    resource_units: int
    resource_capacity: int
    deaths: int
    group_payoff_delta: int
    cumulative_group_payoff: int


@dataclass(frozen=True)
class ResumeState:
    completed_days: int
    resource_units: int
    previous_overuse_count: int | None
    population: int
    living_slots: tuple[int, ...]
    private_payoff_by_slot: tuple[tuple[int, int], ...]
    cumulative_group_payoff: int
    collapse_announced: bool
    rows: list[dict[str, str]]


@dataclass(frozen=True)
class DecisionResult:
    action: str
    reasoning: str
    invalid_reason: str
    attempt_outcome: str
    attempt_count: int
    call_seed: int
    requested_model: str
    returned_model: str
    request_id: str
    finish_reason: str
    usage_json: str
    raw_response_sha256: str


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


def _stable_json_sha256(value: Any) -> str:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _derive_seed(*parts: Any) -> int:
    digest = hashlib.sha256(
        json.dumps(
            list(parts),
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 2_147_483_647


def _derive_call_seed(
    *,
    run_id: str,
    day: int,
    anonymous_agent_slot: int,
    generation_seed: int,
) -> int:
    return _derive_seed(
        SEED_DERIVATION_POLICY,
        generation_seed,
        run_id,
        day,
        anonymous_agent_slot,
    )


def _part_2_dynamics_metadata() -> dict[str, Any]:
    return {
        "schema_version": PART_2_SCHEMA_VERSION,
        "incentive_policy": {
            "id": INCENTIVE_POLICY,
            "individual_score": {
                "option_a_delta": OPTION_A_PRIVATE_PAYOFF,
                "option_b_delta_field": "society_config.selfish_gain",
                "invalid_delta": 0,
                "recording": "per_slot_daily_delta_and_cumulative_total",
            },
            "group_score": {
                "unanimous_option_a_delta_field": (
                    "society_config.community_benefit"
                ),
                "unanimous_option_b_delta_field": (
                    "negative_society_config.community_benefit"
                ),
                "mixed_or_invalid_delta": 0,
                "recording": "daily_delta_and_cumulative_total",
            },
            "shared_reserve": {
                "option_a_delta": 0,
                "option_b_delta_field": (
                    "negative_society_config.depletion_units"
                ),
                "invalid_delta": 0,
            },
            "prompt_engine_parity": "exact_v2",
        },
        "invalid_action_policy": {
            "id": INVALID_ACTION_POLICY,
            "action": "INVALID",
            "reserve_delta": 0,
            "private_payoff_delta": 0,
            "group_unanimity": "prevents_unanimous_delta",
            "counts_as_restrain": False,
            "counts_as_overuse": False,
            "semantic_retry": False,
        },
        "seed_policy": {
            "id": SEED_DERIVATION_POLICY,
            "environment_and_generation_seeds_separate": True,
            "call_seed_inputs": [
                "generation_seed",
                "run_id",
                "day",
                "anonymous_agent_slot",
            ],
            "transport_retries_reuse_call_seed": True,
        },
        "attrition_policy": {
            "id": ATTRITION_POLICY,
            "eligible_population": "living_agents_at_end_of_day",
            "selection": "uniform_sample_without_replacement",
            "environment_rng": "python_random_mt19937",
        },
    }


def _structural_cell_id(
    *,
    config: SocietyConfig,
    resource_capacity: int,
    collapse_death_rate: float,
) -> str:
    return structural_cell_id_for_contract(
        society_config=_config_to_metadata(config),
        resource_capacity=resource_capacity,
        collapse_death_rate=collapse_death_rate,
        dynamics=_part_2_dynamics_metadata(),
    )


def _trajectory_id(*, structural_cell_id: str, environment_seed: int) -> str:
    return trajectory_id_for_contract(
        structural_cell_id=structural_cell_id,
        environment_seed=environment_seed,
    )


def _new_run_id() -> str:
    return f"p2run_{uuid.uuid4().hex}"


def _resolved_generation_seed(
    *,
    seed: int | None,
    generation_seed: int | None,
) -> int:
    if seed is not None and generation_seed is not None and seed != generation_seed:
        raise ValueError(
            "seed is a legacy alias for generation_seed and cannot specify a different value."
        )
    resolved = generation_seed if generation_seed is not None else seed
    if resolved is None:
        return 0
    if not isinstance(resolved, int) or isinstance(resolved, bool):
        raise TypeError("generation_seed must be an integer.")
    return resolved


def _resolved_environment_seed(
    *,
    environment_seed: int | None,
    generation_seed: int,
) -> tuple[int, str]:
    if environment_seed is None:
        return (
            _derive_seed("legacy_default_environment_seed_v1", generation_seed),
            "derived_legacy_default",
        )
    if not isinstance(environment_seed, int) or isinstance(environment_seed, bool):
        raise TypeError("environment_seed must be an integer.")
    return environment_seed, "explicit"


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
    environment_seed: int,
    environment_seed_origin: str,
    generation_seed: int,
    run_id: str,
    trajectory_id: str,
    structural_cell_id: str,
    attempt_logger: DurableAttemptLogger,
    extraction_config: ExtractionConfig | None,
    direct_output_token_cap: int,
) -> None:
    generation_protocol = _generation_protocol_metadata(
        extraction_config=extraction_config,
        direct_output_token_cap=direct_output_token_cap,
    )
    parameters = {
        "part_2_schema_version": PART_2_SCHEMA_VERSION,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "structural_cell_id": structural_cell_id,
        "society_config": _config_to_metadata(config),
        "resource_capacity": resource_capacity,
        "collapse_death_rate": collapse_death_rate,
        "environment_seed": environment_seed,
        "environment_seed_origin": environment_seed_origin,
        "generation_seed": generation_seed,
        "dynamics": _part_2_dynamics_metadata(),
        "result_schema": list(RESULT_HEADERS),
        "grading_protocol": (
            extraction_config.to_metadata() if extraction_config is not None else None
        ),
        "generation_protocol": generation_protocol,
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
    metadata["resume_contract"] = _strict_resume_contract(
        provider=provider,
        model=model,
        run_parameters=parameters,
        extraction_config=extraction_config,
        direct_output_token_cap=direct_output_token_cap,
    )
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
    extraction_config: ExtractionConfig | None,
    direct_output_token_cap: int = DEFAULT_DIRECT_OUTPUT_TOKEN_CAP,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    grading_protocol = (
        extraction_config.to_metadata() if extraction_config is not None else None
    )
    generation_protocol = _generation_protocol_metadata(
        extraction_config=extraction_config,
        direct_output_token_cap=direct_output_token_cap,
    )
    return {
        "schema_version": 2,
        "experiment": "part_2",
        "source_bundle": source_bundle_metadata(
            [
                Path(__file__),
                repo_root / "agents" / "agent_2.py",
                repo_root / "agents" / "base_agent.py",
                repo_root / "experiments" / "misc" / "attempt_log.py",
                repo_root / "experiments" / "misc" / "final_answer.py",
                repo_root / "experiments" / "misc" / "preflight.py",
                repo_root / "experiments" / "misc" / "prompt_loader.py",
                repo_root / "experiments" / "misc" / "result_writer.py",
                repo_root / "experiments" / "misc" / "run_metadata.py",
                repo_root / "experiments" / "part2" / "part_2_prompt.json",
                repo_root / "analysis" / "part2_dynamics.py",
                repo_root / "agents" / "agent_config.py",
                repo_root / "agents" / "agent_config.registry.json",
                repo_root / "providers" / "api_call.py",
                repo_root / "pyproject.toml",
                repo_root / "uv.lock",
            ]
        ),
        "prompt_config_hash": PROMPT_CONFIG_HASH,
        "grading_protocol": grading_protocol,
        "generation_protocol": generation_protocol,
        "model_roles": {
            "subject": registry_identity_metadata([(provider, model)]),
            **(
                {"extractor": grading_protocol["extractor"]}
                if grading_protocol is not None
                else {}
            ),
        },
        "run_parameters": run_parameters,
        "retry_policy": {
            "agent_max_attempts": MAX_AGENT_ATTEMPTS,
            "run_max_attempts": MAX_RUN_ATTEMPTS,
            "initial_delay_seconds": INITIAL_RETRY_DELAY_SECONDS,
            "max_delay_seconds": MAX_RETRY_DELAY_SECONDS,
            "classification": "transport_only_no_semantic_retry_v1",
        },
    }


def _generation_protocol_metadata(
    *,
    extraction_config: ExtractionConfig | None,
    direct_output_token_cap: int,
) -> dict[str, Any]:
    if extraction_config is not None:
        return {
            "schema_version": 1,
            "mode": "independent_final_answer_extraction",
            "subject_output_token_cap": extraction_config.subject_output_token_cap,
        }
    if (
        not isinstance(direct_output_token_cap, int)
        or isinstance(direct_output_token_cap, bool)
        or direct_output_token_cap <= 0
    ):
        raise ValueError("direct_output_token_cap must be a positive integer.")
    return {
        "schema_version": 1,
        "mode": "direct_provider_structured_output",
        "protocol": DIRECT_GENERATION_PROTOCOL_VERSION,
        "output_schema": "SocietyDecision(extra=forbid)",
        "parser": "exact_json_object_action_reasoning_no_semantic_retry_v1",
        "output_token_cap": direct_output_token_cap,
        "identity_policy": "exact_requested_returned_model_required_v1",
    }


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
    generation_seed: int | None = None,
    environment_seed: int | None = None,
    output_token_cap: int | None = None,
    extractor_provider: str | None = None,
    extractor_model: str | None = None,
    extractor_max_tokens: int | None = None,
    direct_output_token_cap: int = DEFAULT_DIRECT_OUTPUT_TOKEN_CAP,
) -> Path | None:
    if not PART_2_RESULTS_DIR.exists():
        return None

    candidates: list[tuple[datetime, Path]] = []
    expected_config = _config_to_metadata(config)
    expected_extraction = _fresh_extraction_config(
        output_token_cap=output_token_cap,
        extractor_provider=extractor_provider,
        extractor_model=extractor_model,
        extractor_max_tokens=extractor_max_tokens,
    )
    expected_grading_protocol = (
        expected_extraction.to_metadata()
        if expected_extraction is not None
        else None
    )
    expected_generation_protocol = _generation_protocol_metadata(
        extraction_config=expected_extraction,
        direct_output_token_cap=direct_output_token_cap,
    )
    expected_generation_seed = _resolved_generation_seed(
        seed=seed,
        generation_seed=generation_seed,
    )
    expected_environment_seed, _ = _resolved_environment_seed(
        environment_seed=environment_seed,
        generation_seed=expected_generation_seed,
    )
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
            legacy_unseeded = (
                expected_grading_protocol is None
                and seed is None
                and generation_seed is None
            )
            if legacy_unseeded:
                if metadata.get("generation_seed") not in {None, 0}:
                    continue
            elif metadata.get("generation_seed") != expected_generation_seed:
                continue
            if environment_seed is None and expected_grading_protocol is None:
                if metadata.get("environment_seed") not in {
                    None,
                    expected_environment_seed,
                }:
                    continue
            elif metadata.get("environment_seed") != expected_environment_seed:
                continue
            if metadata.get("grading_protocol") != expected_grading_protocol:
                continue
            if metadata.get("generation_protocol") != expected_generation_protocol:
                continue
            if (
                expected_grading_protocol is not None
                and metadata.get("part_2_schema_version") != PART_2_SCHEMA_VERSION
            ):
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
        valid_headers = {
            tuple(RESULT_HEADERS),
            tuple(PILOT_RESULT_HEADERS),
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


def _part_2_csv_header(path: str | Path) -> list[str]:
    import csv

    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle).__next__())


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
            living_slots=tuple(range(1, config.society_size + 1)),
            private_payoff_by_slot=tuple(
                (slot, 0) for slot in range(1, config.society_size + 1)
            ),
            cumulative_group_payoff=0,
            collapse_announced=False,
            rows=[],
        )

    last_row = complete_rows[-1]
    last_day = int(last_row["day"])
    last_day_rows = [row for row in complete_rows if int(row["day"]) == last_day]
    if all(row.get("anonymous_agent_slot", "").strip() for row in last_day_rows):
        living_slots = tuple(
            int(row["anonymous_agent_slot"])
            for row in last_day_rows
            if row.get("died_today", "").strip().lower() != "true"
        )
    else:
        living_slots = tuple(range(1, int(last_row["population_end"]) + 1))
    latest_private_payoff_by_slot: dict[int, int] = {}
    for row in complete_rows:
        if row.get("anonymous_agent_slot", "").strip():
            latest_private_payoff_by_slot[int(row["anonymous_agent_slot"])] = int(
                row["cumulative_private_payoff"]
            )
    private_payoff_by_slot = tuple(sorted(latest_private_payoff_by_slot.items()))
    if not set(living_slots).issubset(latest_private_payoff_by_slot):
        raise ValueError(
            "Part 2 resume rows do not contain one cumulative private payoff "
            "for every living anonymous slot."
        )
    return ResumeState(
        completed_days=last_day,
        resource_units=int(last_row["resource_units_remaining"]),
        previous_overuse_count=int(last_row["overuse_count"]),
        population=int(last_row["population_end"]),
        living_slots=living_slots,
        private_payoff_by_slot=private_payoff_by_slot,
        cumulative_group_payoff=int(last_row["cumulative_group_payoff"]),
        collapse_announced=any(
            int(row["resource_units_remaining"]) == 0 for row in complete_rows
        ),
        rows=complete_rows,
    )


def _build_agents(
    provider: str,
    model: str,
    slots: int | tuple[int, ...] | list[int],
    *,
    seed: int | None = None,
    max_tokens: int | None = None,
    keep_alive: float | str | None = None,
) -> list[Agent2]:
    resolved_slots = (
        list(range(1, slots + 1)) if isinstance(slots, int) else list(slots)
    )
    agents: list[Agent2] = []
    for slot in resolved_slots:
        agent = Agent2(
            id_="anonymous_participant",
            provider_=provider,
            model_=model,
            seed_=None,
            max_tokens_=max_tokens,
            keep_alive_=keep_alive,
        )
        agent.part_2_anonymous_slot = slot
        agents.append(agent)
    return agents


def _agent_slot(agent: Agent2) -> int:
    slot = getattr(agent, "part_2_anonymous_slot", None)
    if not isinstance(slot, int) or isinstance(slot, bool) or slot < 1:
        raise ValueError("Part 2 agent is missing a valid anonymous slot.")
    return slot


def _agent_row_id(agent: Agent2) -> str:
    return f"slot_{_agent_slot(agent):06d}"


def _initial_resource_units(config: SocietyConfig) -> int:
    baseline_units = max(config.community_benefit, config.depletion_units, 1)
    return max(
        config.society_size * baseline_units * RESOURCE_RESERVE_MULTIPLIER,
        config.society_size,
    )


def _validate_incentive_config(config: SocietyConfig) -> None:
    if config.selfish_gain <= OPTION_A_PRIVATE_PAYOFF:
        raise ValueError(
            "selfish_gain must exceed OPTION_A's private payoff so OPTION_B "
            "has a real individual incentive."
        )
    if config.depletion_units <= 0:
        raise ValueError("depletion_units must be positive.")
    if config.community_benefit <= 0:
        raise ValueError("community_benefit must be positive.")


def _private_payoff_delta(action: str, config: SocietyConfig) -> int:
    normalized = _normalize_action(action)
    if normalized == "RESTRAIN":
        return OPTION_A_PRIVATE_PAYOFF
    if normalized == "OVERUSE":
        return config.selfish_gain
    if normalized == "INVALID":
        return 0
    raise AssertionError(f"Unhandled Part 2 action: {normalized}")


def _group_payoff_delta(
    *,
    population_start: int,
    restrain_count: int,
    overuse_count: int,
    invalid_count: int,
    community_benefit: int,
) -> int:
    if restrain_count + overuse_count + invalid_count != population_start:
        raise ValueError("Part 2 daily outcome counts do not partition the population.")
    if population_start <= 0 or invalid_count:
        return 0
    if restrain_count == population_start:
        return community_benefit
    if overuse_count == population_start:
        return -community_benefit
    return 0


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


def _select_attrition_slots(
    *,
    living_slots: list[int],
    deaths: int,
    environment_seed: int,
    trajectory_id: str,
    day: int,
) -> tuple[list[int], int | None]:
    if deaths <= 0:
        return [], None
    if deaths > len(living_slots):
        raise ValueError("Attrition deaths cannot exceed the living population.")
    attrition_seed = _derive_seed(
        ATTRITION_POLICY,
        environment_seed,
        trajectory_id,
        day,
    )
    rng = random.Random(attrition_seed)
    return rng.sample(list(living_slots), deaths), attrition_seed


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
        f"overuse={summary.overuse_count} invalid={summary.invalid_count} "
        f"reserve={summary.resource_units}/"
        f"{summary.resource_capacity} group_score={summary.cumulative_group_payoff} "
        f"deaths={summary.deaths}",
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
    if not isinstance(data, dict):
        raise ResponseParseError("Part 2 response must be one JSON object.")
    if set(data) != {"action", "reasoning"}:
        raise ResponseParseError(
            "Part 2 response must contain exactly action and reasoning."
        )
    if not isinstance(data["action"], str) or not isinstance(data["reasoning"], str):
        raise ResponseParseError("Part 2 action and reasoning must be strings.")
    action = _normalize_action(data["action"])
    reasoning = data["reasoning"].strip()
    if not reasoning:
        raise ResponseParseError("Missing reasoning in part 2 response.")
    return action, reasoning


def _direct_generation_record(raw: str) -> dict[str, Any] | None:
    details = getattr(raw, "details", None)
    if details is None or not callable(getattr(details, "to_dict", None)):
        return None
    subject = details.to_dict()
    subject["raw_response_sha256"] = _stable_json_sha256(
        subject.get("raw_response")
    )
    subject["content_sha256"] = _stable_json_sha256(str(subject.get("content", "")))
    subject["reasoning_sha256"] = _stable_json_sha256(
        str(subject.get("reasoning", ""))
    )
    return {
        "protocol": DIRECT_GENERATION_PROTOCOL_VERSION,
        "status": "success",
        "kind": "part_2_commons_decision",
        "subject": subject,
    }


def validate_direct_attempt_provenance(
    attempt_path: str | Path,
    rows: list[dict[str, str]],
    *,
    provider: str,
    model: str,
) -> None:
    """Bind each direct semantic result to an exact, auditable provider response."""

    by_unit = {
        f"{row['day']}__{row['agent']}": row
        for row in rows
    }
    terminal: dict[str, dict[str, Any]] = {}
    for record in iter_attempt_records(attempt_path):
        if record.get("will_retry") is False and record.get("outcome") in {
            "success",
            "invalid_response",
        }:
            terminal[str(record["unit_id"])] = record
    if set(terminal) != set(by_unit):
        raise ValueError("Direct Part 2 attempt/result coverage mismatch.")
    for unit_id, row in by_unit.items():
        record = terminal[unit_id]
        if record.get("provider") != provider or record.get("model") != model:
            raise ValueError(f"Direct Part 2 route mismatch for {unit_id}.")
        generation = record.get("generation_record")
        subject = generation.get("subject") if isinstance(generation, dict) else None
        if (
            not isinstance(generation, dict)
            or generation.get("protocol") != DIRECT_GENERATION_PROTOCOL_VERSION
            or not isinstance(subject, dict)
        ):
            raise ValueError(f"Direct Part 2 provenance is missing for {unit_id}.")
        if (
            subject.get("requested_model") != model
            or subject.get("response_model") != model
            or subject.get("model_identity_match") is not True
        ):
            raise ValueError(f"Direct Part 2 exact model identity failed for {unit_id}.")
        if not str(subject.get("request_id") or "").strip():
            raise ValueError(f"Direct Part 2 request ID is missing for {unit_id}.")
        if not isinstance(subject.get("usage"), dict):
            raise ValueError(f"Direct Part 2 token usage is missing for {unit_id}.")
        if not str(subject.get("raw_response_sha256") or "").strip():
            raise ValueError(f"Direct Part 2 raw response hash is missing for {unit_id}.")
        if record.get("outcome") == "success":
            if subject.get("truncated") is not False:
                raise ValueError(f"Direct Part 2 success was truncated for {unit_id}.")
            parsed = record.get("parsed_response")
            parsed_action = parsed.get("action") if isinstance(parsed, dict) else None
            if _normalize_action(str(parsed_action)) != row["action"]:
                raise ValueError(f"Direct Part 2 parsed action mismatch for {unit_id}.")


def _pending_terminal_attempts(
    attempt_path: str | Path,
    *,
    completed_rows: list[dict[str, str]],
    next_day: int,
    living_agent_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Return the durable prefix of a partially completed day for replay."""

    terminal = [
        record
        for record in iter_attempt_records(attempt_path)
        if record.get("will_retry") is False
        and record.get("outcome") in {"success", "invalid_response"}
    ]
    counts = Counter(str(record["unit_id"]) for record in terminal)
    if any(count != 1 for count in counts.values()):
        raise ValueError("Part 2 attempt log repeats a terminal semantic unit.")
    by_id = {str(record["unit_id"]): record for record in terminal}
    completed_ids = {f"{row['day']}__{row['agent']}" for row in completed_rows}
    missing = completed_ids - set(by_id)
    if missing:
        raise ValueError("Part 2 completed rows are missing terminal attempts.")
    pending = {key: value for key, value in by_id.items() if key not in completed_ids}
    ordered_pending_ids: list[str] = []
    for record in terminal:
        unit_id = str(record["unit_id"])
        if unit_id not in pending:
            continue
        unit = record.get("unit")
        if not isinstance(unit, dict) or unit.get("day") != next_day:
            raise ValueError("Part 2 orphan terminal attempt is outside the next day.")
        ordered_pending_ids.append(str(unit.get("agent")))
    expected_prefix = living_agent_ids[: len(ordered_pending_ids)]
    if ordered_pending_ids != expected_prefix:
        raise ValueError("Part 2 orphan attempts are not a contiguous living-agent prefix.")
    return pending


def _decision_from_terminal_attempt(
    agent: Agent2, record: Mapping[str, Any]
) -> DecisionResult:
    unit = record.get("unit")
    if not isinstance(unit, Mapping) or not isinstance(unit.get("call_seed"), int):
        raise ValueError("Part 2 replay attempt lacks its exact call seed.")
    generation = record.get("generation_record")
    outcome = str(record.get("outcome"))
    if outcome == "success":
        parsed = record.get("parsed_response")
        if not isinstance(parsed, Mapping):
            raise ValueError("Part 2 successful replay attempt lacks parsed response.")
        action = _normalize_action(str(parsed.get("action")))
        reasoning = str(parsed.get("reasoning") or "").strip()
        if not reasoning:
            raise ValueError("Part 2 successful replay attempt lacks reasoning.")
        invalid_reason = ""
    else:
        error = record.get("error")
        action = "INVALID"
        reasoning = ""
        invalid_reason = (
            f"{error.get('exception_type')}: {error.get('message')}"
            if isinstance(error, Mapping)
            else "invalid_response"
        )
    return _decision_result(
        agent=agent,
        action=action,
        reasoning=reasoning,
        invalid_reason=invalid_reason,
        attempt_outcome=outcome,
        attempt_count=int(record.get("attempt") or 1),
        call_seed=int(unit["call_seed"]),
        raw_response=(
            str(record["raw_response"])
            if record.get("raw_response") is not None
            else None
        ),
        generation_record=(generation if isinstance(generation, dict) else None),
    )


def _decision_result(
    *,
    agent: Agent2,
    action: str,
    reasoning: str,
    invalid_reason: str,
    attempt_outcome: str,
    attempt_count: int,
    call_seed: int,
    raw_response: str | None,
    generation_record: dict[str, Any] | None,
) -> DecisionResult:
    subject = (
        generation_record.get("subject")
        if isinstance(generation_record, dict)
        and isinstance(generation_record.get("subject"), dict)
        else {}
    )
    raw_payload = subject.get("raw_response")
    if raw_payload is None:
        raw_payload = raw_response
    supplied_raw_hash = subject.get("raw_response_sha256")
    usage = subject.get("usage")
    return DecisionResult(
        action=action,
        reasoning=reasoning,
        invalid_reason=invalid_reason,
        attempt_outcome=attempt_outcome,
        attempt_count=attempt_count,
        call_seed=call_seed,
        requested_model=str(
            subject.get("requested_model") or subject.get("model") or agent.model
        ),
        returned_model=str(subject.get("response_model") or ""),
        request_id=str(subject.get("request_id") or ""),
        finish_reason=str(subject.get("finish_reason") or ""),
        usage_json=(
            json.dumps(usage, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if usage is not None
            else ""
        ),
        raw_response_sha256=(
            str(supplied_raw_hash)
            if isinstance(supplied_raw_hash, str) and supplied_raw_hash.strip()
            else _stable_json_sha256(raw_payload)
            if raw_payload is not None
            else ""
        ),
    )


def _is_transport_retry(error: Exception, *, provider: str, model: str) -> bool:
    provenance = failure_provenance(error, provider=provider, model=model)
    status_code = provenance.get("status_code") if provenance else None
    return bool(
        provenance
        and (
            provenance.get("category") in {"gateway", "transport"}
            or (
                isinstance(status_code, int)
                and (status_code in {408, 429} or status_code >= 500)
            )
        )
        and not isinstance(error, OllamaConnectionError)
    )


def _query_agent_until_valid(
    agent: Agent2,
    prompt: str,
    *,
    attempt_logger: DurableAttemptLogger | None = None,
    unit: dict[str, Any] | None = None,
    extraction_config: ExtractionConfig | None = None,
    call_seed: int | None = None,
) -> DecisionResult:
    had_retry_status = False
    attempt = 0
    resolved_unit = dict(unit or {"agent": agent.id})
    unit_id = "__".join(
        str(resolved_unit.get(key, ""))
        for key in ("day", "agent")
    ).strip("_") or agent.id
    resolved_call_seed = getattr(agent, "seed", None) if call_seed is None else call_seed
    if resolved_call_seed is None:
        resolved_call_seed = 0
    agent.seed = resolved_call_seed
    resolved_unit["call_seed"] = resolved_call_seed

    while attempt < MAX_AGENT_ATTEMPTS:
        attempt += 1
        try:
            generation_record = None
            try:
                if extraction_config is None:
                    raw = agent.query(prompt, json_mode=True)
                    generation_record = _direct_generation_record(raw)
                else:
                    raw, generated = agent.query_for_grading(
                        prompt,
                        extraction_config=extraction_config,
                        extraction_kind="part_2_commons_decision",
                    )
                    generation_record = generated.to_dict()
            finally:
                from experiments.confirmatory_budget import (
                    consume_environment_reservation,
                )

                dispatch_hash = consume_environment_reservation()
                if dispatch_hash is not None:
                    resolved_unit["dispatch_request_sha256"] = dispatch_hash
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
            generation_record = extraction_record_from_error(error)
            retryable = _is_transport_retry(
                error,
                provider=agent.provider,
                model=agent.model,
            )
            will_retry = retryable and attempt < MAX_AGENT_ATTEMPTS
            subject_was_returned = bool(
                isinstance(generation_record, dict)
                and isinstance(generation_record.get("subject"), dict)
            )
            if subject_was_returned and not retryable:
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
                        raw_response=None,
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
                        will_retry=False,
                    )
                if had_retry_status:
                    _emit_retry_status_line("", finalize=True)
                return _decision_result(
                    agent=agent,
                    action="INVALID",
                    reasoning="",
                    invalid_reason=f"{type(error).__name__}: {safe_error_message(error)}",
                    attempt_outcome="invalid_response",
                    attempt_count=attempt,
                    call_seed=resolved_call_seed,
                    raw_response=None,
                    generation_record=generation_record,
                )
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
                    generation_record=generation_record,
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
            details = getattr(raw, "details", None)
            if details is not None and getattr(details, "truncated", None) is True:
                raise ResponseParseError(
                    "Provider marked the subject response as truncated."
                )
            action, reasoning = _parse_agent_response(raw)
        except Exception as error:
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
                    will_retry=False,
                )
            if had_retry_status:
                _emit_retry_status_line("", finalize=True)
            return _decision_result(
                agent=agent,
                action="INVALID",
                reasoning="",
                invalid_reason=f"{type(error).__name__}: {safe_error_message(error)}",
                attempt_outcome="invalid_response",
                attempt_count=attempt,
                call_seed=resolved_call_seed,
                raw_response=str(raw),
                generation_record=generation_record,
            )

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
        return _decision_result(
            agent=agent,
            action=action,
            reasoning=reasoning,
            invalid_reason="",
            attempt_outcome="success",
            attempt_count=attempt,
            call_seed=resolved_call_seed,
            raw_response=str(raw),
            generation_record=generation_record,
        )

    raise RuntimeError("Part 2 query loop exhausted without an outcome.")


def _render_day_summary(summary: DaySummary) -> None:
    table = Table(box=box.ROUNDED, expand=True, header_style="bold")
    table.add_column("Start Pop.", justify="center")
    table.add_column("Restrain", justify="center")
    table.add_column("Overuse", justify="center")
    table.add_column("Invalid", justify="center")
    table.add_column("Reserve", justify="center")
    table.add_column("Group score", justify="center")
    table.add_column("Deaths", justify="center")
    table.add_column("End Pop.", justify="center")
    table.add_row(
        str(summary.population_start),
        f"[green]{summary.restrain_count}[/green]",
        f"[red]{summary.overuse_count}[/red]",
        f"[yellow]{summary.invalid_count}[/yellow]",
        f"{summary.resource_units}/{summary.resource_capacity}",
        f"{summary.cumulative_group_payoff} ({summary.group_payoff_delta:+d})",
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
                Markdown(decision["reasoning"] or decision.get("invalid_reason", "INVALID")),
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
    cumulative_group_payoff: int,
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
            f"[bold]Reserve:[/bold] {resource_units}/{resource_capacity}\n"
            f"[bold]Group score:[/bold] {cumulative_group_payoff}",
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
    generation_seed: int | None = None,
    environment_seed: int | None = None,
    output_token_cap: int | None = None,
    extractor_provider: str | None = None,
    extractor_model: str | None = None,
    extractor_max_tokens: int | None = None,
    direct_output_token_cap: int = DEFAULT_DIRECT_OUTPUT_TOKEN_CAP,
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
    requested_generation_seed = generation_seed if generation_seed is not None else seed
    requested_environment_seed = environment_seed
    strict_generation = False
    if (
        not isinstance(direct_output_token_cap, int)
        or isinstance(direct_output_token_cap, bool)
        or direct_output_token_cap <= 0
    ):
        raise ValueError("direct_output_token_cap must be a positive integer.")

    if is_resuming:
        metadata_path = (
            Path(resume_metadata_path)
            if resume_metadata_path is not None
            else _latest_interrupted_part_2_metadata_path()
        )
        metadata = _load_part_2_metadata(metadata_path)
        strict_generation = metadata.get("generation_protocol") is not None
        if metadata.get("grading_protocol") is not None or strict_generation:
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
                    "Cannot add final-answer extraction while resuming a legacy Part 2 run."
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
        if extraction_config is not None or strict_generation:
            required_strict_fields = {
                "part_2_schema_version": PART_2_SCHEMA_VERSION,
                "result_schema": RESULT_HEADERS,
                "dynamics": _part_2_dynamics_metadata(),
            }
            for field, expected_value in required_strict_fields.items():
                if metadata.get(field) != expected_value:
                    raise ValueError(
                        "Strict Part 2 resume metadata is missing or mismatches "
                        f"{field}; old-schema confirmatory runs cannot be resumed."
                    )
            validate_resume_contract(
                metadata.get("resume_contract"),
                _strict_resume_contract(
                    provider=str(metadata["provider"]),
                    model=str(metadata["model"]),
                    run_parameters={
                        key: metadata[key]
                        for key in (
                            "society_config",
                            "part_2_schema_version",
                            "run_id",
                            "trajectory_id",
                            "structural_cell_id",
                            "resource_capacity",
                            "collapse_death_rate",
                            "environment_seed",
                            "environment_seed_origin",
                            "generation_seed",
                            "dynamics",
                            "result_schema",
                            "grading_protocol",
                            "generation_protocol",
                        )
                    },
                    extraction_config=extraction_config,
                    direct_output_token_cap=int(
                        metadata.get("generation_protocol", {}).get(
                            "output_token_cap", direct_output_token_cap
                        )
                    ),
                ),
                experiment="Part 2",
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
        resumed_generation_seed = int(metadata.get("generation_seed") or 0)
        resumed_environment_seed, legacy_environment_origin = _resolved_environment_seed(
            environment_seed=metadata.get("environment_seed"),
            generation_seed=resumed_generation_seed,
        )
        environment_seed_origin = str(
            metadata.get("environment_seed_origin") or legacy_environment_origin
        )
        structural_cell_id = str(
            metadata.get("structural_cell_id")
            or _structural_cell_id(
                config=resumed_config,
                resource_capacity=resource_capacity,
                collapse_death_rate=resumed_death_rate,
            )
        )
        trajectory_id = str(
            metadata.get("trajectory_id")
            or _trajectory_id(
                structural_cell_id=structural_cell_id,
                environment_seed=resumed_environment_seed,
            )
        )
        run_id = str(
            metadata.get("run_id")
            or f"legacy_{_stable_json_sha256([timestamp, resumed_provider, resumed_model])[:20]}"
        )

        if extraction_config is not None or strict_generation:
            artifact_integrity = metadata.get("artifact_integrity")
            if not isinstance(artifact_integrity, dict):
                raise ValueError(
                    "Strict Part 2 resume metadata is missing result integrity."
                )
            validate_file_integrity(
                csv_path,
                artifact_integrity.get("results"),
                label="Part 2 result CSV",
            )

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
        if (
            requested_generation_seed is not None
            and requested_generation_seed != resumed_generation_seed
        ):
            raise ValueError(
                "Resume generation seed mismatch: expected "
                f"{resumed_generation_seed}, received {requested_generation_seed}."
            )
        if (
            requested_environment_seed is not None
            and requested_environment_seed != resumed_environment_seed
        ):
            raise ValueError(
                "Resume environment seed mismatch: expected "
                f"{resumed_environment_seed}, received {requested_environment_seed}."
            )

        provider = resumed_provider
        model = resumed_model
        society_config = resumed_config
        collapse_death_rate = resumed_death_rate
        generation_seed = resumed_generation_seed
        environment_seed = resumed_environment_seed
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
        _validate_incentive_config(society_config)
        if resource_capacity is not None and resource_capacity <= 0:
            raise ValueError("resource_capacity must be greater than 0.")
        if collapse_death_rate is None:
            collapse_death_rate = DEFAULT_COLLAPSE_DEATH_RATE
        if not 0 < collapse_death_rate <= 1:
            raise ValueError("collapse_death_rate must be greater than 0 and at most 1.")
        generation_seed = _resolved_generation_seed(
            seed=seed,
            generation_seed=generation_seed,
        )
        environment_seed, environment_seed_origin = _resolved_environment_seed(
            environment_seed=environment_seed,
            generation_seed=generation_seed,
        )
        timestamp = datetime.now().strftime(PART_2_TIMESTAMP_FORMAT)
        csv_path = PART_2_RESULTS_DIR / _build_result_filename(
            provider=provider,
            model=model,
            config=society_config,
            timestamp=timestamp,
            resource_capacity=resource_capacity,
            collapse_death_rate=collapse_death_rate,
            seed=generation_seed,
        )
        metadata_path = _metadata_path_for_csv(csv_path)
        resource_capacity = (
            _initial_resource_units(society_config)
            if resource_capacity is None
            else resource_capacity
        )
        structural_cell_id = _structural_cell_id(
            config=society_config,
            resource_capacity=resource_capacity,
            collapse_death_rate=collapse_death_rate,
        )
        trajectory_id = _trajectory_id(
            structural_cell_id=structural_cell_id,
            environment_seed=environment_seed,
        )
        run_id = _new_run_id()

    _validate_incentive_config(society_config)
    assert collapse_death_rate is not None
    assert generation_seed is not None
    assert environment_seed is not None

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
    if is_resuming and (extraction_config is not None or strict_generation):
        if not isinstance(attempt_log_metadata, dict):
            raise ValueError(
                "Strict Part 2 resume metadata is missing attempt-log integrity."
            )
        verify_attempt_log_metadata(
            attempt_path,
            attempt_log_metadata,
            require_hash_chain=True,
        )
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
            environment_seed=environment_seed,
            environment_seed_origin=environment_seed_origin,
            generation_seed=generation_seed,
            run_id=run_id,
            trajectory_id=trajectory_id,
            structural_cell_id=structural_cell_id,
            attempt_logger=attempt_logger,
            extraction_config=extraction_config,
            direct_output_token_cap=direct_output_token_cap,
        )

    resume_state = _resume_state_from_rows(
        rows=completed_rows,
        config=society_config,
        resource_capacity=resource_capacity,
    )
    if (
        len(resume_state.rows) != len(completed_rows)
        or (is_resuming and _part_2_csv_header(csv_path) != RESULT_HEADERS)
    ):
        _rewrite_part_2_rows(csv_path, resume_state.rows)
    completed_days = resume_state.completed_days
    resource_units = resume_state.resource_units
    previous_overuse_count = resume_state.previous_overuse_count
    private_payoff_by_slot = dict(resume_state.private_payoff_by_slot)
    cumulative_group_payoff = resume_state.cumulative_group_payoff
    collapse_announced = resume_state.collapse_announced
    keep_alive = MODEL_BATCH_KEEP_ALIVE if provider.strip().lower() == "ollama" else None
    agents = _build_agents(
        provider,
        model,
        resume_state.living_slots,
        seed=None,
        max_tokens=(
            None
            if extraction_config is not None
            else int(
                metadata.get("generation_protocol", {}).get(
                    "output_token_cap", direct_output_token_cap
                )
                if is_resuming
                else direct_output_token_cap
            )
        ),
        keep_alive=keep_alive,
    )
    pending_terminal_attempts = (
        _pending_terminal_attempts(
            attempt_path,
            completed_rows=completed_rows,
            next_day=completed_days + 1,
            living_agent_ids=[_agent_row_id(agent) for agent in agents],
        )
        if is_resuming and (extraction_config is not None or strict_generation)
        else {}
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
            cumulative_group_payoff=cumulative_group_payoff,
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
                f"OPTION_B depletion: {society_config.depletion_units} units\n"
                f"OPTION_A private payoff: {OPTION_A_PRIVATE_PAYOFF} point\n"
                f"OPTION_B private payoff: {society_config.selfish_gain} points\n"
                f"Unanimous group payoff: +/-{society_config.community_benefit} points",
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
                daily_decisions: list[dict[str, Any]] = []
                for agent_index, agent in enumerate(agents, start=1):
                    anonymous_slot = _agent_slot(agent)
                    agent_row_id = _agent_row_id(agent)
                    call_seed = _derive_call_seed(
                        run_id=run_id,
                        day=day,
                        anonymous_agent_slot=anonymous_slot,
                        generation_seed=generation_seed,
                    )
                    if not headless:
                        _emit_retry_status_line(
                            f"  Anonymous slot {anonymous_slot} "
                            f"{agent_index}/{population_start}: querying..."
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
                        cumulative_private_payoff=private_payoff_by_slot[
                            anonymous_slot
                        ],
                        cumulative_group_payoff=cumulative_group_payoff,
                    )
                    unit_id = f"{day}__{agent_row_id}"
                    replay = pending_terminal_attempts.pop(unit_id, None)
                    decision_result = (
                        _decision_from_terminal_attempt(agent, replay)
                        if replay is not None
                        else _query_agent_until_valid(
                            agent,
                            prompt,
                            attempt_logger=attempt_logger,
                            unit={
                                "day": day,
                                "agent": agent_row_id,
                                "anonymous_agent_slot": anonymous_slot,
                                "agent_index": agent_index,
                                "population_start": population_start,
                                "run_id": run_id,
                                "trajectory_id": trajectory_id,
                                "structural_cell_id": structural_cell_id,
                            },
                            extraction_config=extraction_config,
                            call_seed=call_seed,
                        )
                    )
                    if not headless:
                        _emit_retry_status_line("")
                    daily_decisions.append(
                        {
                            "agent": agent_row_id,
                            "anonymous_agent_slot": anonymous_slot,
                            "action": decision_result.action,
                            "reasoning": decision_result.reasoning,
                            "invalid_reason": decision_result.invalid_reason,
                            "attempt_outcome": decision_result.attempt_outcome,
                            "attempt_count": decision_result.attempt_count,
                            "call_seed": decision_result.call_seed,
                            "requested_model": decision_result.requested_model,
                            "returned_model": decision_result.returned_model,
                            "request_id": decision_result.request_id,
                            "finish_reason": decision_result.finish_reason,
                            "usage_json": decision_result.usage_json,
                            "raw_response_sha256": decision_result.raw_response_sha256,
                        }
                    )

                overuse_count = sum(
                    1 for decision in daily_decisions if decision["action"] == "OVERUSE"
                )
                restrain_count = sum(
                    1 for decision in daily_decisions if decision["action"] == "RESTRAIN"
                )
                invalid_count = sum(
                    1 for decision in daily_decisions if decision["action"] == "INVALID"
                )
                if restrain_count + overuse_count + invalid_count != population_start:
                    raise RuntimeError("Part 2 daily outcome counts do not partition the population.")
                group_payoff_delta = _group_payoff_delta(
                    population_start=population_start,
                    restrain_count=restrain_count,
                    overuse_count=overuse_count,
                    invalid_count=invalid_count,
                    community_benefit=society_config.community_benefit,
                )
                cumulative_group_payoff += group_payoff_delta
                for decision in daily_decisions:
                    slot = int(decision["anonymous_agent_slot"])
                    private_delta = _private_payoff_delta(
                        str(decision["action"]), society_config
                    )
                    private_payoff_by_slot[slot] += private_delta
                    decision["private_payoff_delta"] = private_delta
                    decision["cumulative_private_payoff"] = (
                        private_payoff_by_slot[slot]
                    )
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
                death_slots, attrition_seed = _select_attrition_slots(
                    living_slots=[_agent_slot(agent) for agent in agents],
                    deaths=deaths,
                    environment_seed=environment_seed,
                    trajectory_id=trajectory_id,
                    day=day,
                )
                death_slot_set = set(death_slots)
                death_rank = {
                    slot: rank for rank, slot in enumerate(death_slots, start=1)
                }
                if deaths:
                    agents = [
                        agent
                        for agent in agents
                        if _agent_slot(agent) not in death_slot_set
                    ]

                population_end = len(agents)
                summary = DaySummary(
                    day=day,
                    population_start=population_start,
                    population_end=population_end,
                    restrain_count=restrain_count,
                    overuse_count=overuse_count,
                    invalid_count=invalid_count,
                    resource_units=resource_units,
                    resource_capacity=resource_capacity,
                    deaths=deaths,
                    group_payoff_delta=group_payoff_delta,
                    cumulative_group_payoff=cumulative_group_payoff,
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

                row_dicts: list[dict[str, Any]] = []
                selected_slots_json = json.dumps(death_slots, separators=(",", ":"))
                for decision in daily_decisions:
                    slot = int(decision["anonymous_agent_slot"])
                    row_dicts.append(
                        {
                            "run_id": run_id,
                            "trajectory_id": trajectory_id,
                            "structural_cell_id": structural_cell_id,
                            "provider": provider,
                            "model": model,
                            "day": day,
                            **decision,
                            "environment_seed": environment_seed,
                            "generation_seed": generation_seed,
                            "group_payoff_delta": group_payoff_delta,
                            "cumulative_group_payoff": cumulative_group_payoff,
                            "population_start": population_start,
                            "population_end": population_end,
                            "restrain_count": restrain_count,
                            "overuse_count": overuse_count,
                            "invalid_count": invalid_count,
                            "resource_units_remaining": resource_units,
                            "resource_capacity": resource_capacity,
                            "deaths": deaths,
                            "died_today": str(slot in death_slot_set).lower(),
                            "attrition_rank": death_rank.get(slot, ""),
                            "attrition_seed": attrition_seed if attrition_seed is not None else "",
                            "death_selected_slots_json": selected_slots_json,
                            "resource": society_config.resource,
                            # These values define the visible and executed incentive
                            # contract for this trajectory.
                            "selfish_gain": society_config.selfish_gain,
                            "depletion_units": society_config.depletion_units,
                            "community_benefit": society_config.community_benefit,
                        }
                    )
                writer.write_rows(
                    [[row.get(column, "") for column in RESULT_HEADERS] for row in row_dicts]
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
            csv_path=csv_path,
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
                "final_cumulative_group_payoff": cumulative_group_payoff,
                "final_total_private_payoff": sum(
                    private_payoff_by_slot.values()
                ),
                "stop_reason": stop_reason,
                **_attempt_log_metadata(
                    attempt_logger,
                    coverage=attempt_log_coverage,
                ),
                "artifact_integrity": {
                    "results": file_integrity_metadata(csv_path)
                },
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
    generation_seed: int | None = None,
    environment_seed: int | None = None,
    output_token_cap: int | None = None,
    extractor_provider: str | None = None,
    extractor_model: str | None = None,
    extractor_max_tokens: int | None = None,
    direct_output_token_cap: int = DEFAULT_DIRECT_OUTPUT_TOKEN_CAP,
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
                generation_seed=generation_seed,
                environment_seed=environment_seed,
                output_token_cap=output_token_cap,
                extractor_provider=extractor_provider,
                extractor_model=extractor_model,
                extractor_max_tokens=extractor_max_tokens,
                direct_output_token_cap=direct_output_token_cap,
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
                generation_seed=generation_seed,
                environment_seed=environment_seed,
                output_token_cap=output_token_cap,
                extractor_provider=extractor_provider,
                extractor_model=extractor_model,
                extractor_max_tokens=extractor_max_tokens,
                direct_output_token_cap=direct_output_token_cap,
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


def parse_part_2_args(argv: list[str] | None = None) -> Any:
    """Parse new split seeds while preserving the legacy society CLI parser."""

    import argparse

    seed_parser = argparse.ArgumentParser(add_help=False)
    seed_parser.add_argument("--generation-seed", type=int, default=None)
    seed_parser.add_argument("--environment-seed", type=int, default=None)
    seed_parser.add_argument(
        "--direct-output-token-cap",
        type=int,
        default=DEFAULT_DIRECT_OUTPUT_TOKEN_CAP,
    )
    seed_args, remaining = seed_parser.parse_known_args(argv)
    parsed = parse_society_args(remaining)
    parsed.generation_seed = seed_args.generation_seed
    parsed.environment_seed = seed_args.environment_seed
    parsed.direct_output_token_cap = seed_args.direct_output_token_cap
    return parsed


if __name__ == "__main__":
    cli_args = parse_part_2_args()
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
        generation_seed=cli_args.generation_seed,
        environment_seed=cli_args.environment_seed,
        output_token_cap=cli_args.output_token_cap,
        extractor_provider=cli_args.extractor_provider,
        extractor_model=cli_args.extractor_model,
        extractor_max_tokens=cli_args.extractor_max_tokens,
        direct_output_token_cap=cli_args.direct_output_token_cap,
        resume=cli_args.resume,
        headless=cli_args.headless,
    )
