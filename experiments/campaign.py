"""Resumable orchestration for the model-registry experiment campaign.

The experiment entrypoints remain the source of truth for result generation.
This module plans pinned commands, runs a strict campaign-wide preflight, records
every subprocess attempt, and only marks a job complete after verifying the
entrypoint's native metadata and CSV artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from agents.agent_config import load_model_cohort
from experiments.misc.preflight import run_experiment_preflight
from experiments.misc.final_answer import (
    DEFAULT_EXTRACTOR_MAX_TOKENS,
    DEFAULT_EXTRACTOR_MODEL,
    DEFAULT_EXTRACTOR_PROVIDER,
    DEFAULT_OUTPUT_TOKEN_CAP,
    ExtractionConfig,
)
from experiments.misc.prompt_loader import load_part_0_raw_prompts, load_prompt_config
from experiments.part1.part_1 import DEFAULT_PART_1_ORDER_SEED
from experiments.part1.scenario_variants import list_scenario_variants
from experiments.part2.part_2 import (
    DEFAULT_COLLAPSE_DEATH_RATE,
    _initial_resource_units,
)
from experiments.misc.wizard import SocietyConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "data" / "campaigns"
REGISTRY_COHORTS = ("current_sota", "historical")
PHASE_ORDER = ("smoke", "part0", "part1", "part2")
DEFAULT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
MANIFEST_SCHEMA_VERSION = 1
MAX_PART2_SENSITIVITY_CELLS = 4_096
MAX_PART2_SENSITIVITY_REQUESTS = 10_000_000
_CAMPAIGN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


class CampaignError(RuntimeError):
    """Raised for a campaign planning, execution, or verification failure."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    timed_out: bool = False
    error: str | None = None


ProcessRunner = Callable[[Sequence[str], Path, dict[str, str], Path, int], ProcessResult]
PreflightRunner = Callable[..., None]
JudgeProbe = Callable[[str], bool]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _default_campaign_id() -> str:
    return datetime.now(timezone.utc).strftime("campaign-%Y%m%dT%H%M%SZ")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _death_rate(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("death rate must be greater than 0 and at most 1")
    return parsed


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _campaign_dir(campaign_id: str) -> Path:
    if not _CAMPAIGN_ID_PATTERN.fullmatch(campaign_id):
        raise CampaignError(
            "campaign id must be 1-80 characters using letters, digits, '.', '-', or '_'"
        )
    root = CAMPAIGN_ROOT.resolve()
    candidate = (root / campaign_id).resolve()
    if candidate.parent != root:
        raise CampaignError("campaign output must be a direct child of data/campaigns")
    return candidate


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _plan_hash(manifest: dict[str, Any]) -> str:
    immutable = {
        "schema_version": manifest["schema_version"],
        "campaign_id": manifest["campaign_id"],
        "cohort": manifest["cohort"],
        "phases": manifest["phases"],
        "timeout_seconds": manifest["timeout_seconds"],
        "part2_design": manifest.get("part2_design"),
        "jobs": [
            {
                key: job[key]
                for key in (
                    "id",
                    "phase",
                    "experiment",
                    "target_ids",
                    "command",
                    "expected",
                    "counts",
                )
            }
            for job in manifest["jobs"]
        ],
    }
    if "grading_protocol" in manifest:
        immutable["grading_protocol"] = manifest["grading_protocol"]
    encoded = json.dumps(
        immutable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CampaignError(f"campaign manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise CampaignError(f"campaign manifest is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"campaign manifest must be a JSON object: {path}")
    return value


def _part_1_count(
    config: dict[str, Any],
    *,
    games: list[str],
    frames: list[str],
    domains: list[str],
    presentations: list[str],
    limit: int | None,
) -> int:
    total = 0
    for game in games:
        if game not in config["games"]:
            raise CampaignError(f"unknown part 1 game: {game}")
        for domain in domains:
            try:
                fallback = config["domains"][domain]["games"][game]
            except KeyError as error:
                raise CampaignError(
                    f"unknown part 1 domain/game combination: {domain}/{game}"
                ) from error
            total += (
                len(frames)
                * len(presentations)
                * len(
                    list_scenario_variants(
                        domain_id=domain,
                        game_id=game,
                        fallback=fallback,
                    )
                )
            )
    if not set(frames).issubset(config["frames"]):
        unknown = sorted(set(frames) - set(config["frames"]))
        raise CampaignError(f"unknown part 1 frame(s): {', '.join(unknown)}")
    if not set(presentations).issubset(config["presentations"]):
        unknown = sorted(set(presentations) - set(config["presentations"]))
        raise CampaignError(f"unknown part 1 presentation(s): {', '.join(unknown)}")
    if limit is not None:
        if limit > total:
            raise CampaignError(f"part 1 limit {limit} exceeds the selected matrix ({total})")
        return limit
    return total


def _base_job(
    *,
    job_id: str,
    phase: str,
    experiment: str,
    command: list[str],
    expected: dict[str, Any],
    counts: dict[str, int],
    target_ids: list[str],
) -> dict[str, Any]:
    return {
        "id": job_id,
        "phase": phase,
        "experiment": experiment,
        "target_ids": target_ids,
        "command": command,
        "command_display": shlex.join(command),
        "expected": expected,
        "counts": counts,
        "status": "pending",
        "artifact": None,
        "attempts": [],
    }


def _target_command_args(target: dict[str, Any]) -> list[str]:
    return ["--provider", str(target["provider"]), "--model", str(target["route"])]


def _extraction_command_args(config: ExtractionConfig) -> list[str]:
    return [
        "--output-token-cap",
        str(config.subject_output_token_cap),
        "--extractor-provider",
        config.provider,
        "--extractor-model",
        config.model,
        "--extractor-max-tokens",
        str(config.extractor_max_tokens),
    ]


PART2_CELL_FIELDS = (
    "resource_capacity",
    "depletion_units",
    "collapse_death_rate",
    "society_size",
    "days",
    "seed",
)


def _validate_part2_cell(value: Any, *, label: str) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise CampaignError(f"{label} must be a JSON object")
    missing = sorted(set(PART2_CELL_FIELDS) - set(value))
    extra = sorted(set(value) - set(PART2_CELL_FIELDS))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise CampaignError(f"{label} has invalid fields ({'; '.join(details)})")

    integer_fields = (
        "resource_capacity",
        "depletion_units",
        "society_size",
        "days",
        "seed",
    )
    cell: dict[str, int | float] = {}
    for field in integer_fields:
        raw = value[field]
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise CampaignError(f"{label}.{field} must be an integer")
        if field == "seed":
            if raw < 0:
                raise CampaignError(f"{label}.seed must be non-negative")
        elif raw <= 0:
            raise CampaignError(f"{label}.{field} must be greater than zero")
        cell[field] = raw
    rate = value["collapse_death_rate"]
    if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not 0 < rate <= 1:
        raise CampaignError(
            f"{label}.collapse_death_rate must be numeric, greater than zero, and at most one"
        )
    cell["collapse_death_rate"] = float(rate)
    return cell


def _part2_sensitivity_cells(
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, int | float]]] | None:
    factorial_values = (
        args.part2_grid_capacity,
        args.part2_grid_depletion_units,
        args.part2_grid_death_rate,
        args.part2_grid_population,
        args.part2_grid_horizon,
        args.part2_grid_seed,
    )
    has_factorial = any(values for values in factorial_values)
    if args.part2_cell and has_factorial:
        raise CampaignError(
            "--part2-cell cannot be combined with --part2-grid-* factorial axes"
        )
    if not args.part2_cell and not has_factorial:
        return None

    if args.part2_cell:
        cells: list[dict[str, int | float]] = []
        for index, raw in enumerate(args.part2_cell, start=1):
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as error:
                raise CampaignError(
                    f"part 2 cell {index} is invalid JSON: {error.msg}"
                ) from error
            cells.append(_validate_part2_cell(decoded, label=f"part 2 cell {index}"))
        mode = "specified"
    else:
        if not args.part2_grid_seed:
            raise CampaignError(
                "factorial Part 2 sensitivity plans require at least one explicit "
                "--part2-grid-seed"
            )
        default_capacity = _initial_resource_units(
            SocietyConfig(
                society_size=args.part2_society_size,
                days=args.part2_days,
                resource=args.part2_resource,
                selfish_gain=args.part2_selfish_gain,
                depletion_units=args.part2_depletion_units,
                community_benefit=args.part2_community_benefit,
            )
        )
        axes = (
            _unique(args.part2_grid_capacity or [default_capacity]),
            _unique(args.part2_grid_depletion_units or [args.part2_depletion_units]),
            _unique(args.part2_grid_death_rate or [DEFAULT_COLLAPSE_DEATH_RATE]),
            _unique(args.part2_grid_population or [args.part2_society_size]),
            _unique(args.part2_grid_horizon or [args.part2_days]),
            _unique(args.part2_grid_seed),
        )
        cells = [
            {
                "resource_capacity": capacity,
                "depletion_units": depletion,
                "collapse_death_rate": death_rate,
                "society_size": population,
                "days": horizon,
                "seed": seed,
            }
            for capacity, depletion, death_rate, population, horizon, seed in itertools.product(
                *axes
            )
        ]
        mode = "factorial"

    exact_keys = [tuple(cell[field] for field in PART2_CELL_FIELDS) for cell in cells]
    if len(set(exact_keys)) != len(exact_keys):
        raise CampaignError("Part 2 sensitivity cells must be unique")
    structural_fields = PART2_CELL_FIELDS[:-1]
    seeds_by_structure: dict[tuple[int | float, ...], set[int | float]] = {}
    for cell in cells:
        structure = tuple(cell[field] for field in structural_fields)
        seeds_by_structure.setdefault(structure, set()).add(cell["seed"])
    replicate_counts = {len(seeds) for seeds in seeds_by_structure.values()}
    if len(replicate_counts) != 1:
        raise CampaignError(
            "every Part 2 structural cell must have the same number of replicate seeds"
        )
    return mode, cells


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build a deterministic plan from an exact registry cohort."""

    cohort = load_model_cohort(args.cohort)
    targets = list(cohort["targets"])
    if not targets:
        raise CampaignError(f"registry cohort is empty: {args.cohort}")
    target_by_id = {str(target["id"]): target for target in targets}
    extraction_config = ExtractionConfig(
        subject_output_token_cap=args.output_token_cap,
        provider=args.extractor_provider,
        model=args.extractor_model,
        extractor_max_tokens=args.extractor_max_tokens,
    )
    extraction_args = _extraction_command_args(extraction_config)

    phases = _unique(args.phase or PHASE_ORDER)
    phases = [phase for phase in PHASE_ORDER if phase in phases]
    if not phases:
        raise CampaignError("at least one phase must be selected")

    part_0_config = load_prompt_config("part_0")
    available_languages = list(part_0_config["languages"])
    languages = _unique(args.part0_language or available_languages)
    unknown_languages = sorted(set(languages) - set(available_languages))
    if unknown_languages:
        raise CampaignError(f"unknown part 0 language(s): {', '.join(unknown_languages)}")
    total_part_0_prompts = len(load_part_0_raw_prompts())
    part_0_prompt_count = args.part0_prompt_count or total_part_0_prompts
    if part_0_prompt_count > total_part_0_prompts:
        raise CampaignError(
            f"part 0 prompt count {part_0_prompt_count} exceeds available prompts "
            f"({total_part_0_prompts})"
        )
    if args.smoke_part0_prompts > total_part_0_prompts:
        raise CampaignError(
            f"smoke part 0 prompt count {args.smoke_part0_prompts} exceeds available "
            f"prompts ({total_part_0_prompts})"
        )
    if not args.part2_resource.strip():
        raise CampaignError("part 2 resource must be a non-empty string")
    args.part2_resource = args.part2_resource.strip()

    part_1_config = load_prompt_config("part_1")
    defaults = part_1_config["defaults"]
    games = _unique(args.part1_game or list(defaults["games"]))
    frames = _unique(args.part1_frame or list(defaults["frames"]))
    domains = _unique(args.part1_domain or list(defaults["domains"]))
    presentations = _unique(
        args.part1_presentation or list(defaults["presentations"])
    )
    part_1_decisions = _part_1_count(
        part_1_config,
        games=games,
        frames=frames,
        domains=domains,
        presentations=presentations,
        limit=args.part1_limit,
    )

    selected_part_2_ids = _unique(args.part2_target or list(target_by_id))
    unknown_part_2_ids = sorted(set(selected_part_2_ids) - set(target_by_id))
    if unknown_part_2_ids:
        raise CampaignError(
            "part 2 targets must be exact members of the selected cohort; unknown: "
            + ", ".join(unknown_part_2_ids)
        )
    selected_part_2_targets = [target_by_id[target_id] for target_id in selected_part_2_ids]
    sensitivity_plan = _part2_sensitivity_cells(args)
    if sensitivity_plan is not None and "part2" not in phases:
        raise CampaignError("Part 2 sensitivity controls require --phase part2")
    if sensitivity_plan is not None and args.part2_replicates != 1:
        raise CampaignError(
            "--part2-replicates is a legacy unseeded repeat control and cannot be "
            "combined with sensitivity cells; enumerate exact --part2-grid-seed values instead"
        )
    if sensitivity_plan is not None:
        _, sensitivity_cells = sensitivity_plan
        sensitivity_jobs = len(sensitivity_cells) * len(selected_part_2_targets)
        sensitivity_requests = sum(
            int(cell["society_size"]) * int(cell["days"])
            for cell in sensitivity_cells
        ) * len(selected_part_2_targets)
        if sensitivity_jobs > MAX_PART2_SENSITIVITY_CELLS:
            raise CampaignError(
                "Part 2 sensitivity plan exceeds the bounded job limit "
                f"({sensitivity_jobs} > {MAX_PART2_SENSITIVITY_CELLS})"
            )
        if sensitivity_requests > MAX_PART2_SENSITIVITY_REQUESTS:
            raise CampaignError(
                "Part 2 sensitivity plan exceeds the bounded request upper limit "
                f"({sensitivity_requests} > {MAX_PART2_SENSITIVITY_REQUESTS})"
            )

    python = sys.executable
    jobs: list[dict[str, Any]] = []

    def add_part_0(*, phase: str, prompt_count: int, selected_languages: list[str]) -> None:
        decisions = len(targets) * prompt_count * len(selected_languages)
        non_english = sum(language != "english" for language in selected_languages)
        models: dict[str, list[str]] = {}
        command = [python, "-m", "experiments.part0.part_0"]
        command.extend(extraction_args)
        for target in targets:
            models.setdefault(str(target["provider"]), []).append(str(target["route"]))
            command.extend(["--benchmark", f"{target['provider']}:{target['route']}"])
        for language in selected_languages:
            command.extend(["--language", language])
        command.extend(["--prompt-count", str(prompt_count), "--judge-after", "--headless"])
        jobs.append(
            _base_job(
                job_id=f"{phase}-part0-cohort",
                phase=phase,
                experiment="part_0",
                command=command,
                expected={
                    "models": models,
                    "languages": selected_languages,
                    "prompt_count": prompt_count,
                    "row_count": decisions,
                },
                counts={
                    "decisions": decisions,
                    "target_model_requests_baseline_estimate": decisions,
                    "extractor_requests_baseline_estimate": decisions,
                    "subject_output_tokens_upper_bound": (
                        decisions * extraction_config.subject_output_token_cap
                    ),
                    "extractor_output_tokens_upper_bound": (
                        decisions * extraction_config.extractor_max_tokens
                    ),
                    "judge_requests_baseline_estimate": decisions,
                    "translation_operations_baseline_estimate": (
                        len(targets) * prompt_count * non_english * 2
                    ),
                },
                target_ids=list(target_by_id),
            )
        )

    def add_part_1(*, phase: str, limit: int | None, decisions: int) -> None:
        for counterbalance_index, target in enumerate(targets):
            command = [python, "-m", "experiments.part1.part_1"]
            command.extend(extraction_args)
            command.extend(_target_command_args(target))
            for value in games:
                command.extend(["--game", value])
            for value in frames:
                command.extend(["--frame", value])
            for value in domains:
                command.extend(["--domain", value])
            for value in presentations:
                command.extend(["--presentation", value])
            if limit is not None:
                command.extend(["--limit", str(limit)])
            command.extend(
                [
                    "--order-seed",
                    str(args.part1_order_seed),
                    "--order-strategy",
                    "counterbalanced",
                    "--counterbalance-index",
                    str(counterbalance_index),
                ]
            )
            command.append("--headless")
            jobs.append(
                _base_job(
                    job_id=f"{phase}-part1-{target['id']}",
                    phase=phase,
                    experiment="part_1",
                    command=command,
                    expected={
                        "provider": target["provider"],
                        "model": target["route"],
                        "games": games,
                        "frames": frames,
                        "domains": domains,
                        "presentations": presentations,
                        "limit": limit,
                        "ordering": {
                            "seed": args.part1_order_seed,
                            "strategy": "counterbalanced",
                            "counterbalance_index": counterbalance_index,
                        },
                        "row_count": decisions,
                    },
                    counts={
                        "decisions": decisions,
                        "target_model_requests_baseline_estimate": decisions,
                        "extractor_requests_baseline_estimate": decisions,
                        "subject_output_tokens_upper_bound": (
                            decisions * extraction_config.subject_output_token_cap
                        ),
                        "extractor_output_tokens_upper_bound": (
                            decisions * extraction_config.extractor_max_tokens
                        ),
                    },
                    target_ids=[str(target["id"])],
                )
            )

    def add_part_2(
        *,
        phase: str,
        phase_targets: list[dict[str, Any]],
        replicates: int,
        society_size: int,
        days: int,
        sensitivity: tuple[str, list[dict[str, int | float]]] | None = None,
    ) -> None:
        if sensitivity is None:
            config = SocietyConfig(
                society_size=society_size,
                days=days,
                resource=args.part2_resource,
                selfish_gain=args.part2_selfish_gain,
                depletion_units=args.part2_depletion_units,
                community_benefit=args.part2_community_benefit,
            )
            capacity = _initial_resource_units(config)
            trajectory_cells: list[dict[str, int | float | None]] = [
                {
                    "resource_capacity": capacity,
                    "depletion_units": args.part2_depletion_units,
                    "collapse_death_rate": DEFAULT_COLLAPSE_DEATH_RATE,
                    "society_size": society_size,
                    "days": days,
                    "seed": None,
                    "replicate": replicate,
                }
                for replicate in range(1, replicates + 1)
            ]
        else:
            _, cells = sensitivity
            trajectory_cells = [
                {**cell, "replicate": None}
                for cell in cells
            ]
        for target in phase_targets:
            for cell_index, cell in enumerate(trajectory_cells, start=1):
                cell_society_size = int(cell["society_size"])
                cell_days = int(cell["days"])
                decisions_upper = cell_society_size * cell_days
                replicate = cell["replicate"]
                seed = cell["seed"]
                command = [python, "-m", "experiments.part2.part_2"]
                command.extend(extraction_args)
                command.extend(_target_command_args(target))
                command.extend(
                    [
                        "--society-size",
                        str(cell_society_size),
                        "--days",
                        str(cell_days),
                        "--resource",
                        args.part2_resource,
                        "--selfish-gain",
                        str(args.part2_selfish_gain),
                        "--depletion-units",
                        str(cell["depletion_units"]),
                        "--community-benefit",
                        str(args.part2_community_benefit),
                    ]
                )
                if sensitivity is not None:
                    command.extend(
                        [
                            "--resource-capacity",
                            str(cell["resource_capacity"]),
                            "--collapse-death-rate",
                            str(cell["collapse_death_rate"]),
                            "--seed",
                            str(seed),
                        ]
                    )
                command.append("--headless")
                job_suffix = (
                    f"r{int(replicate):03d}"
                    if replicate is not None
                    else f"cell{cell_index:04d}-s{int(seed)}"
                )
                jobs.append(
                    _base_job(
                        job_id=f"{phase}-part2-{target['id']}-{job_suffix}",
                        phase=phase,
                        experiment="part_2",
                        command=command,
                        expected={
                            "provider": target["provider"],
                            "model": target["route"],
                            "society_config": {
                                "society_size": cell_society_size,
                                "days": cell_days,
                                "resource": args.part2_resource,
                                "selfish_gain": args.part2_selfish_gain,
                                "depletion_units": int(cell["depletion_units"]),
                                "community_benefit": args.part2_community_benefit,
                            },
                            "resource_capacity": int(cell["resource_capacity"]),
                            "collapse_death_rate": float(cell["collapse_death_rate"]),
                            "generation_seed": seed,
                            "row_count_upper_bound": decisions_upper,
                            "replicate": replicate,
                            "sensitivity_cell": (
                                {
                                    field: cell[field]
                                    for field in PART2_CELL_FIELDS
                                }
                                if sensitivity is not None
                                else None
                            ),
                        },
                        counts={
                            "decisions_upper_bound": decisions_upper,
                            "target_model_requests_baseline_estimate": decisions_upper,
                            "extractor_requests_baseline_estimate": decisions_upper,
                            "subject_output_tokens_upper_bound": (
                                decisions_upper * extraction_config.subject_output_token_cap
                            ),
                            "extractor_output_tokens_upper_bound": (
                                decisions_upper * extraction_config.extractor_max_tokens
                            ),
                        },
                        target_ids=[str(target["id"])],
                    )
                )

    if "smoke" in phases:
        add_part_0(
            phase="smoke",
            prompt_count=args.smoke_part0_prompts,
            selected_languages=["english"],
        )
        smoke_part_1_count = min(args.smoke_part1_limit, part_1_decisions)
        add_part_1(phase="smoke", limit=smoke_part_1_count, decisions=smoke_part_1_count)
        add_part_2(
            phase="smoke",
            phase_targets=targets,
            replicates=1,
            society_size=args.smoke_part2_society_size,
            days=args.smoke_part2_days,
        )
    if "part0" in phases:
        add_part_0(
            phase="part0",
            prompt_count=part_0_prompt_count,
            selected_languages=languages,
        )
    if "part1" in phases:
        add_part_1(
            phase="part1",
            limit=args.part1_limit,
            decisions=part_1_decisions,
        )
    if "part2" in phases:
        add_part_2(
            phase="part2",
            phase_targets=selected_part_2_targets,
            replicates=args.part2_replicates,
            society_size=args.part2_society_size,
            days=args.part2_days,
            sensitivity=sensitivity_plan,
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign_id": args.campaign_id,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "status": "planned",
        "cohort": {
            "id": cohort["id"],
            "version": cohort["version"],
            "registry_version": cohort["registry_version"],
            "registry_hash": cohort["registry_hash"],
            "target_ids": list(target_by_id),
        },
        "phases": phases,
        "timeout_seconds": args.timeout_seconds,
        "grading_protocol": extraction_config.to_metadata(),
        "jobs": jobs,
    }
    if sensitivity_plan is not None and "part2" in phases:
        sensitivity_mode, cells = sensitivity_plan
        structural_fields = PART2_CELL_FIELDS[:-1]
        structural_cells = {
            tuple(cell[field] for field in structural_fields)
            for cell in cells
        }
        manifest["part2_design"] = {
            "mode": sensitivity_mode,
            "cell_fields": list(PART2_CELL_FIELDS),
            "trajectory_cells": cells,
            "structural_cell_count": len(structural_cells),
            "replicates_per_structural_cell": len(cells) // len(structural_cells),
            "trajectory_cell_count": len(cells),
            "target_count": len(selected_part_2_targets),
            "job_count": len(cells) * len(selected_part_2_targets),
            "target_model_requests_upper_bound": sum(
                int(cell["society_size"]) * int(cell["days"])
                for cell in cells
            ) * len(selected_part_2_targets),
        }
    manifest["plan_hash"] = _plan_hash(manifest)
    return manifest


def _metadata_snapshot(experiment: str) -> dict[str, tuple[int, int]]:
    directory = {
        "part_0": REPO_ROOT / "data" / "raw" / "part_0",
        "part_1": REPO_ROOT / "data" / "raw" / "part_1",
        "part_2": REPO_ROOT / "data" / "raw" / "part_2",
    }[experiment]
    snapshot: dict[str, tuple[int, int]] = {}
    for path in directory.glob("*_meta.json"):
        with contextlib.suppress(OSError):
            stat = path.stat()
            snapshot[str(path.resolve())] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise CampaignError(f"result CSV is empty: {path}")
        if not header or any(not column for column in header):
            raise CampaignError(f"result CSV has an invalid header: {path}")
        return sum(1 for row in reader if any(cell.strip() for cell in row))


def _safe_result_path(metadata_path: Path, raw_value: Any, experiment: str) -> Path:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise CampaignError(f"metadata is missing csv_path: {metadata_path}")
    path = Path(raw_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    expected_root = (REPO_ROOT / "data" / "raw" / experiment).resolve()
    try:
        resolved.relative_to(expected_root)
    except ValueError as error:
        raise CampaignError(
            f"metadata result path escapes {expected_root}: {resolved}"
        ) from error
    return resolved


def verify_artifact(job: dict[str, Any], metadata_path: Path) -> dict[str, Any]:
    """Verify native run metadata and its CSV against a planned job."""

    metadata_path = metadata_path.resolve()
    experiment = str(job["experiment"])
    expected_directory = REPO_ROOT / "data" / "raw" / experiment
    try:
        metadata_path.relative_to(expected_directory.resolve())
    except ValueError as error:
        raise CampaignError(f"metadata path is outside the expected result directory: {metadata_path}") from error

    metadata = _load_json(metadata_path)
    if metadata.get("experiment") != experiment:
        raise CampaignError(f"artifact experiment mismatch: {metadata_path}")
    if str(metadata.get("status", "")).lower() != "complete":
        raise CampaignError(f"artifact is not marked complete: {metadata_path}")

    expected = job["expected"]
    if experiment == "part_0":
        if metadata.get("models") != expected["models"]:
            raise CampaignError(f"part 0 artifact model cohort mismatch: {metadata_path}")
        if metadata.get("languages") != expected["languages"]:
            raise CampaignError(f"part 0 artifact language mismatch: {metadata_path}")
        prompts = metadata.get("prompts")
        if not isinstance(prompts, list) or len(prompts) != expected["prompt_count"]:
            raise CampaignError(f"part 0 artifact prompt count mismatch: {metadata_path}")
    elif experiment == "part_1":
        for key in ("provider", "model", "games", "frames", "domains", "presentations", "limit"):
            if metadata.get(key) != expected[key]:
                raise CampaignError(f"part 1 artifact {key} mismatch: {metadata_path}")
        if metadata.get("total_prompts") != expected["row_count"]:
            raise CampaignError(f"part 1 artifact prompt count mismatch: {metadata_path}")
        expected_ordering = expected.get("ordering")
        if expected_ordering is not None and metadata.get("ordering") != expected_ordering | {
            "algorithm": "python_random_mt19937_shuffle_then_cyclic_rotation_v1"
        }:
            raise CampaignError(f"part 1 artifact ordering mismatch: {metadata_path}")
    elif experiment == "part_2":
        for key in ("provider", "model"):
            if metadata.get(key) != expected[key]:
                raise CampaignError(f"part 2 artifact {key} mismatch: {metadata_path}")
        if metadata.get("society_config") != expected["society_config"]:
            raise CampaignError(f"part 2 artifact society config mismatch: {metadata_path}")
        for key in ("resource_capacity", "collapse_death_rate", "generation_seed"):
            if metadata.get(key) != expected[key]:
                raise CampaignError(f"part 2 artifact {key} mismatch: {metadata_path}")
        completed_days = metadata.get("completed_days")
        final_population = metadata.get("final_population")
        configured_days = expected["society_config"]["days"]
        if (
            not isinstance(completed_days, int)
            or completed_days <= 0
            or completed_days > configured_days
            or not isinstance(final_population, int)
            or final_population < 0
            or (final_population > 0 and completed_days != configured_days)
        ):
            raise CampaignError(f"part 2 artifact completion state mismatch: {metadata_path}")
    else:
        raise CampaignError(f"unsupported experiment in job: {experiment}")

    csv_path = _safe_result_path(metadata_path, metadata.get("csv_path"), experiment)
    if not csv_path.is_file():
        raise CampaignError(f"result CSV does not exist: {csv_path}")
    rows = _row_count(csv_path)
    if experiment in {"part_0", "part_1"}:
        if rows != expected["row_count"]:
            raise CampaignError(
                f"result row count mismatch for {job['id']}: expected "
                f"{expected['row_count']}, found {rows}"
            )
    elif rows <= 0 or rows > expected["row_count_upper_bound"]:
        raise CampaignError(
            f"part 2 result rows must be within 1..{expected['row_count_upper_bound']}; found {rows}"
        )
    completed_rows = metadata.get("completed_rows")
    if completed_rows is not None and completed_rows != rows:
        raise CampaignError(
            f"metadata completed_rows mismatch: expected {completed_rows}, found {rows}"
        )
    return {
        "metadata_path": str(metadata_path.relative_to(REPO_ROOT)),
        "csv_path": str(csv_path.relative_to(REPO_ROOT)),
        "rows": rows,
        "verified_at_utc": _utc_now(),
    }


def _find_new_artifact(
    job: dict[str, Any], before: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    after = _metadata_snapshot(str(job["experiment"]))
    changed = [
        Path(path)
        for path, fingerprint in after.items()
        if before.get(path) != fingerprint
    ]
    changed.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    errors: list[str] = []
    for path in changed:
        try:
            return verify_artifact(job, path)
        except CampaignError as error:
            errors.append(str(error))
    detail = f" ({'; '.join(errors[:3])})" if errors else ""
    raise CampaignError(
        f"subprocess exited successfully but produced no verified artifact for {job['id']}{detail}"
    )


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name != "nt":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return
    with contextlib.suppress(ProcessLookupError):
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        process.wait()


def run_subprocess(
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> ProcessResult:
    """Run one command without a shell, with a finite timeout and durable log."""

    popen_options: dict[str, Any] = {}
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if creation_flags:
            popen_options["creationflags"] = creation_flags
    else:
        popen_options["start_new_session"] = True
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                **popen_options,
            )
            try:
                return ProcessResult(returncode=process.wait(timeout=timeout_seconds))
            except subprocess.TimeoutExpired:
                _terminate_process_group(process)
                return ProcessResult(
                    returncode=None,
                    timed_out=True,
                    error=f"timed out after {timeout_seconds} seconds",
                )
            except KeyboardInterrupt:
                _terminate_process_group(process)
                raise
    except OSError as error:
        return ProcessResult(
            returncode=None,
            error=f"{type(error).__name__}: {error}",
        )


def _load_part_0_judges() -> list[tuple[str, str]]:
    path = REPO_ROOT / "experiments" / "part0" / "part_0_config.json"
    raw_lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith(("//", "#"))
    ]
    config = json.loads(re.sub(r",(\s*[}\]])", r"\1", "\n".join(raw_lines)))
    judges = config.get("judge", {}).get("providers", [])
    result = [
        (str(item["provider"]).strip().lower(), str(item["model"]).strip())
        for item in judges
        if isinstance(item, dict) and item.get("provider") and item.get("model")
    ]
    if not result:
        raise CampaignError("part 0 has no configured compliance judge")
    return result


def perform_strict_preflight(
    manifest: dict[str, Any],
    *,
    preflight_runner: PreflightRunner = run_experiment_preflight,
    judge_probe: JudgeProbe | None = None,
) -> None:
    target_pairs: list[tuple[str, str]] = []
    for job in manifest["jobs"]:
        expected = job["expected"]
        if job["experiment"] == "part_0":
            for provider, models in expected["models"].items():
                target_pairs.extend((provider, model) for model in models)
        else:
            target_pairs.append((expected["provider"], expected["model"]))
    includes_part_0 = any(job["experiment"] == "part_0" for job in manifest["jobs"])
    judges = _load_part_0_judges() if includes_part_0 else []
    skip_value = os.environ.pop("LLM_ALTRUISM_SKIP_PREFLIGHT", None)
    try:
        preflight_runner(
            "Registry campaign",
            _unique_pairs([*target_pairs, *judges]),
            resume=False,
            test_paths=[
                "tests/test_preflight.py",
                "tests/test_part_0.py",
                "tests/test_part_1.py",
                "tests/test_part_2.py",
                "tests/test_campaign.py",
            ],
            strict_provider_checks=True,
        )
    finally:
        if skip_value is not None:
            os.environ["LLM_ALTRUISM_SKIP_PREFLIGHT"] = skip_value
    ollama_judges = [model for provider, model in judges if provider == "ollama"]
    if ollama_judges:
        if judge_probe is None:
            from providers.api_call import ollama_model_available_locally

            judge_probe = ollama_model_available_locally
        available = False
        failures: list[str] = []
        for model in ollama_judges:
            try:
                if judge_probe(model):
                    available = True
                    break
                failures.append(f"{model} is not installed locally")
            except Exception as error:
                failures.append(f"{model}: {type(error).__name__}: {error}")
        if not available:
            raise CampaignError(
                "strict preflight found no available Ollama compliance judge: "
                + "; ".join(failures)
            )


def _unique_pairs(values: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _resume_manifest(campaign_id: str) -> tuple[Path, dict[str, Any]]:
    campaign_dir = _campaign_dir(campaign_id)
    manifest_path = campaign_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CampaignError("unsupported campaign manifest schema")
    if manifest.get("campaign_id") != campaign_id:
        raise CampaignError("campaign id does not match the manifest")
    if manifest.get("plan_hash") != _plan_hash(manifest):
        raise CampaignError("campaign plan hash mismatch; the manifest was modified")
    entrypoints = {
        "part_0": "experiments.part0.part_0",
        "part_1": "experiments.part1.part_1",
        "part_2": "experiments.part2.part_2",
    }
    for job in manifest.get("jobs", []):
        command = job.get("command")
        experiment = job.get("experiment")
        if (
            experiment not in entrypoints
            or not isinstance(command, list)
            or len(command) < 3
            or command[0] != sys.executable
            or command[1:3] != ["-m", entrypoints.get(experiment)]
        ):
            raise CampaignError(
                f"campaign command is not pinned to the current Python/entrypoint: {job.get('id')}"
            )
    cohort = load_model_cohort(str(manifest.get("cohort", {}).get("id", "")))
    pinned = manifest["cohort"]
    for key in ("version", "registry_version", "registry_hash"):
        if pinned.get(key) != cohort.get(key):
            raise CampaignError(
                f"cannot resume after registry drift ({key} changed); start a new campaign"
            )
    return campaign_dir, manifest


def execute_manifest(
    campaign_dir: Path,
    manifest: dict[str, Any],
    *,
    fail_fast: bool,
    process_runner: ProcessRunner = run_subprocess,
    preflight_runner: PreflightRunner = run_experiment_preflight,
    judge_probe: JudgeProbe | None = None,
) -> int:
    """Execute or resume a manifest, returning a process-style exit code."""

    manifest_path = campaign_dir / "manifest.json"
    logs_dir = campaign_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Completion is never trusted solely because the campaign manifest says so.
    for job in manifest["jobs"]:
        if job.get("status") != "complete":
            continue
        artifact = job.get("artifact")
        try:
            if not isinstance(artifact, dict) or not artifact.get("metadata_path"):
                raise CampaignError("missing recorded artifact")
            job["artifact"] = verify_artifact(
                job, REPO_ROOT / str(artifact["metadata_path"])
            )
        except CampaignError as error:
            job["status"] = "pending"
            job["artifact"] = None
            job["resume_verification_error"] = str(error)

    manifest["status"] = "preflight"
    manifest["updated_at_utc"] = _utc_now()
    _atomic_write_json(manifest_path, manifest)
    try:
        perform_strict_preflight(
            manifest,
            preflight_runner=preflight_runner,
            judge_probe=judge_probe,
        )
    except Exception as error:
        manifest["status"] = "preflight_failed"
        manifest["preflight_failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "at_utc": _utc_now(),
        }
        manifest["updated_at_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        raise

    manifest["preflight"] = {"status": "complete", "completed_at_utc": _utc_now()}
    manifest["status"] = "running"
    manifest["updated_at_utc"] = _utc_now()
    _atomic_write_json(manifest_path, manifest)

    env = os.environ.copy()
    # The stricter campaign-wide gate above has already run. Child gates are
    # disabled to prevent dozens of redundant test-suite invocations.
    env["LLM_ALTRUISM_SKIP_PREFLIGHT"] = "1"
    failures = 0
    for job in manifest["jobs"]:
        if job.get("status") == "complete":
            continue
        attempt_number = len(job.get("attempts", [])) + 1
        log_path = logs_dir / f"{job['id']}.attempt-{attempt_number:03d}.log"
        before = _metadata_snapshot(str(job["experiment"]))
        attempt: dict[str, Any] = {
            "number": attempt_number,
            "started_at_utc": _utc_now(),
            "log_path": str(log_path.relative_to(REPO_ROOT)),
        }
        job.setdefault("attempts", []).append(attempt)
        job["status"] = "running"
        manifest["updated_at_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)

        try:
            result = process_runner(
                list(job["command"]),
                REPO_ROOT,
                env,
                log_path,
                int(manifest["timeout_seconds"]),
            )
        except KeyboardInterrupt:
            attempt["ended_at_utc"] = _utc_now()
            attempt["error"] = "interrupted by operator"
            job["status"] = "interrupted"
            manifest["status"] = "interrupted"
            manifest["updated_at_utc"] = _utc_now()
            _atomic_write_json(manifest_path, manifest)
            raise
        except Exception as error:
            result = ProcessResult(
                returncode=None,
                error=f"{type(error).__name__}: {error}",
            )
        attempt["ended_at_utc"] = _utc_now()
        attempt["returncode"] = result.returncode
        attempt["timed_out"] = result.timed_out
        if result.error:
            attempt["error"] = result.error

        try:
            if result.timed_out:
                raise CampaignError(result.error or "subprocess timed out")
            if result.returncode != 0:
                raise CampaignError(
                    result.error or f"subprocess exited with code {result.returncode}"
                )
            artifact = _find_new_artifact(job, before)
        except CampaignError as error:
            failures += 1
            job["status"] = "timed_out" if result.timed_out else "failed"
            job["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "at_utc": _utc_now(),
            }
            manifest["updated_at_utc"] = _utc_now()
            _atomic_write_json(manifest_path, manifest)
            if fail_fast:
                break
            continue

        job["artifact"] = artifact
        job["status"] = "complete"
        job.pop("failure", None)
        manifest["updated_at_utc"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)

    incomplete = [job for job in manifest["jobs"] if job.get("status") != "complete"]
    manifest["status"] = "complete" if not incomplete else "failed"
    manifest["summary"] = {
        "complete_jobs": len(manifest["jobs"]) - len(incomplete),
        "incomplete_jobs": len(incomplete),
        "failures_this_attempt": failures,
    }
    manifest["updated_at_utc"] = _utc_now()
    _atomic_write_json(manifest_path, manifest)
    return 0 if not incomplete else 1


def _plan_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    for job in manifest["jobs"]:
        for key, value in job["counts"].items():
            totals[key] = totals.get(key, 0) + int(value)
    summary = {
        "campaign_id": manifest["campaign_id"],
        "cohort": manifest["cohort"],
        "phases": manifest["phases"],
        "job_count": len(manifest["jobs"]),
        "totals": totals,
        "jobs": [
            {
                "id": job["id"],
                "phase": job["phase"],
                "counts": job["counts"],
                "command": job["command_display"],
            }
            for job in manifest["jobs"]
        ],
    }
    if "part2_design" in manifest:
        summary["part2_design"] = manifest["part2_design"]
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, execute, and resume the registry-pinned model campaign."
    )
    parser.add_argument("--cohort", choices=REGISTRY_COHORTS, default="current_sota")
    parser.add_argument("--phase", action="append", choices=PHASE_ORDER)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument(
        "--resume",
        metavar="CAMPAIGN_ID",
        help="Resume this campaign from data/campaigns after re-verifying completed jobs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the exact plan; make no requests or files.")
    parser.add_argument("--json", action="store_true", help="Print plan/summary as JSON.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop scheduling jobs after the first failure.")
    parser.add_argument("--timeout-seconds", type=_positive_int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--output-token-cap",
        type=_positive_int,
        default=DEFAULT_OUTPUT_TOKEN_CAP,
        help="Per-subject total output cap (reasoning plus visible response).",
    )
    parser.add_argument(
        "--extractor-provider",
        default=DEFAULT_EXTRACTOR_PROVIDER,
        help="Provider for the independent final-answer extractor.",
    )
    parser.add_argument(
        "--extractor-model",
        default=DEFAULT_EXTRACTOR_MODEL,
        help="Exact model route for the independent final-answer extractor.",
    )
    parser.add_argument(
        "--extractor-max-tokens",
        type=_positive_int,
        default=DEFAULT_EXTRACTOR_MAX_TOKENS,
        help="Output cap for the extractor's schema-constrained JSON.",
    )

    parser.add_argument("--part0-language", action="append")
    parser.add_argument("--part0-prompt-count", type=_positive_int)
    parser.add_argument("--part1-game", action="append")
    parser.add_argument("--part1-frame", action="append")
    parser.add_argument("--part1-domain", action="append")
    parser.add_argument("--part1-presentation", action="append")
    parser.add_argument("--part1-limit", type=_positive_int)
    parser.add_argument(
        "--part1-order-seed",
        type=int,
        default=DEFAULT_PART_1_ORDER_SEED,
        help="Shared seed for the Part 1 cohort permutation and cyclic rotations.",
    )

    parser.add_argument(
        "--part2-target",
        action="append",
        help="Repeatable exact registry target id; defaults to every cohort member.",
    )
    parser.add_argument("--part2-replicates", type=_positive_int, default=1)
    parser.add_argument("--part2-society-size", type=_positive_int, default=50)
    parser.add_argument("--part2-days", type=_positive_int, default=100)
    parser.add_argument("--part2-resource", default="water")
    parser.add_argument("--part2-selfish-gain", type=_positive_int, default=2)
    parser.add_argument("--part2-depletion-units", type=_positive_int, default=2)
    parser.add_argument("--part2-community-benefit", type=_non_negative_int, default=5)
    parser.add_argument(
        "--part2-grid-capacity",
        action="append",
        type=_positive_int,
        help="Repeatable commons-capacity axis for a factorial Part 2 sensitivity plan.",
    )
    parser.add_argument(
        "--part2-grid-depletion-units",
        action="append",
        type=_positive_int,
        help="Repeatable overuse-depletion axis for a factorial Part 2 sensitivity plan.",
    )
    parser.add_argument(
        "--part2-grid-death-rate",
        action="append",
        type=_death_rate,
        help="Repeatable depleted-day population death-rate axis in (0, 1].",
    )
    parser.add_argument(
        "--part2-grid-population",
        action="append",
        type=_positive_int,
        help="Repeatable starting-population axis for a factorial Part 2 sensitivity plan.",
    )
    parser.add_argument(
        "--part2-grid-horizon",
        action="append",
        type=_positive_int,
        help="Repeatable day-horizon axis for a factorial Part 2 sensitivity plan.",
    )
    parser.add_argument(
        "--part2-grid-seed",
        action="append",
        type=_non_negative_int,
        help="Repeatable generation-seed axis; required for factorial sensitivity plans.",
    )
    parser.add_argument(
        "--part2-cell",
        action="append",
        help=(
            "Repeatable exact JSON trajectory cell with resource_capacity, depletion_units, "
            "collapse_death_rate, society_size, days, and seed."
        ),
    )

    parser.add_argument("--smoke-part0-prompts", type=_positive_int, default=1)
    parser.add_argument("--smoke-part1-limit", type=_positive_int, default=1)
    parser.add_argument("--smoke-part2-society-size", type=_positive_int, default=2)
    parser.add_argument("--smoke-part2-days", type=_positive_int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.campaign_id:
        parser.error("--resume and --campaign-id cannot be used together")

    if args.resume:
        campaign_dir, manifest = _resume_manifest(args.resume)
        if args.dry_run:
            summary = _plan_summary(manifest)
            print(json.dumps(summary, indent=2) if args.json else _format_summary(summary))
            return 0
    else:
        args.campaign_id = args.campaign_id or _default_campaign_id()
        manifest = build_plan(args)
        if args.dry_run:
            summary = _plan_summary(manifest)
            print(json.dumps(summary, indent=2) if args.json else _format_summary(summary))
            return 0
        campaign_dir = _campaign_dir(args.campaign_id)
        if campaign_dir.exists():
            raise CampaignError(
                f"campaign directory already exists: {campaign_dir}; use --resume {args.campaign_id}"
            )
        campaign_dir.mkdir(parents=True)
        _atomic_write_json(campaign_dir / "manifest.json", manifest)

    return execute_manifest(campaign_dir, manifest, fail_fast=args.fail_fast)


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Campaign: {summary['campaign_id']}",
        f"Cohort: {summary['cohort']['id']} ({len(summary['cohort']['target_ids'])} targets)",
        f"Phases: {', '.join(summary['phases'])}",
        f"Jobs: {summary['job_count']}",
        "Estimated counts: "
        + ", ".join(f"{key}={value}" for key, value in sorted(summary["totals"].items())),
        "Commands:",
    ]
    design = summary.get("part2_design")
    if isinstance(design, dict):
        lines.insert(
            -1,
            "Part 2 design: "
            f"mode={design['mode']}, structural_cells={design['structural_cell_count']}, "
            f"replicates_per_cell={design['replicates_per_structural_cell']}, "
            f"trajectory_cells={design['trajectory_cell_count']}, "
            f"jobs={design['job_count']}, "
            f"request_upper_bound={design['target_model_requests_upper_bound']}",
        )
    for job in summary["jobs"]:
        count_label = ", ".join(
            f"{key}={value}" for key, value in sorted(job["counts"].items())
        )
        lines.append(f"  [{job['id']}] {count_label}")
        lines.append(f"    {job['command']}")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as error:
        print(f"campaign error: {error}", file=sys.stderr)
        raise SystemExit(2)
