"""Run the preregistered Part 2 resolution-V sentinel sensitivity campaign.

This module intentionally leaves the source-bound corrected matched-panel runner
unchanged.  It reuses that runner's strict request, simulator, and journal
primitives while binding every raw record to one immutable sensitivity cell.
The design is the frozen 2^(5-1) resolution-V half-fraction (I=ABCDE): sixteen
cells vary capacity per initial agent, depletion, collapse death rate, population,
and horizon. Six outcome-blind developer/capability sentinels receive the same
twelve common environment seeds in every cell.

Raw prompts, responses, reasoning, routes, and request bodies remain in private,
append-only journals.  Sanitized outputs contain only trajectory metrics and
per-sentinel/per-cell estimates plus exactly five AURC main effects per sentinel.
The 30 sentinel-by-factor exact tests form one global Holm family; within-sentinel
max-T values remain explicitly diagnostic. INVALID actions are retained as
nonrestraints under the authoritative confirmatory contract and are never retried.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import platform
import random
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agents.agent_2 import Agent2
from analysis.part2_confirmatory import (
    SENSITIVITY_CELL_COUNT,
    SENSITIVITY_FACTORS,
    SENSITIVITY_LEVELS,
    SENSITIVITY_SEEDS_PER_CELL,
    SENSITIVITY_SENTINEL_COUNT,
    analyze_sentinel_sensitivity,
    resolution_v_half_fraction,
)

from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
)
from experiments.misc.inference_hub_part1_panel import (
    _ChainedJournal,
    _acquire_run_lock,
    _atomic_json,
    _canonical_bytes,
    _read_json,
    _require_mode,
    _safe_file_stem,
    _seal,
    _self_hash,
    _sha256_file,
    _sha256_json,
    _secure_mode,
    _validate_checkpoint_reference,
    _transient,
    select_routes,
)
from experiments.misc.inference_hub_part2_panel import (
    DEFAULT_BACKOFF_SECONDS,
    DEFAULT_BASE_SEED,
    DEFAULT_COMPATIBILITY,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_PANEL,
    DEFAULT_PARTICIPANT_WORKERS,
    DEFAULT_REGISTRY,
    InferenceHubPart2PanelError,
    Part2Contract,
    SCHEMA_VERSION as PART2_JOURNAL_SCHEMA_VERSION,
    _load_panel,
    _matched_attrition,
    _result_index,
    _structured_schema,
    _t_interval,
    _utc_now,
    _validate_retained_result,
    _visible_content,
    _wilson_interval,
    parse_decision,
)
from experiments.misc.inference_hub_provider_safe import _provider_safe_client
from experiments.misc.inference_hub_provider_safe_v2 import provider_round_robin
from experiments.part2.part_2 import _collapse_deaths, _derive_seed


SENSITIVITY_SCHEMA_VERSION = 1
DEFAULT_DESIGN = Path("experiments/part2/part2_sensitivity_v1.json")
DEFAULT_EXPLORATORY_DESIGN = Path(
    "experiments/part2/part2_sensitivity_deadline_exploratory_v1.json"
)
DEFAULT_CAMPAIGN_WORKERS = 24
SENSITIVITY_OUTPUT_TOKENS = 8192
_SEED_NAMESPACE = "inference_hub_part2_sensitivity_common_environment_v1"
_SOURCE_PATHS = (
    Path(__file__),
    DEFAULT_DESIGN,
    DEFAULT_EXPLORATORY_DESIGN,
    Path(__file__).with_name("inference_hub_part2_panel.py"),
    Path(__file__).with_name("inference_hub_provider_safe.py"),
    Path(__file__).with_name("inference_hub_provider_safe_v2.py"),
    Path(__file__).with_name("inference_hub_rate_limit.py"),
    Path(__file__).parents[1] / "part2" / "part_2.py",
    Path(__file__).parents[2] / "agents" / "agent_2.py",
    Path(__file__).parents[2] / "analysis" / "part2_confirmatory.py",
    Path(__file__).parents[1] / "part2" / "part_2_prompt.json",
)


class InferenceHubPart2SensitivityError(RuntimeError):
    """The versioned Part 2 sensitivity contract was violated."""


@dataclass(frozen=True)
class SensitivityCondition:
    cell_id: str
    capacity_per_initial_agent: int
    depletion_units: int
    collapse_death_rate: float
    society_size: int
    horizon_days: int
    resource_capacity: int
    coded_levels: Mapping[str, int]

    def contract(
        self, frozen: Part2Contract, *, trajectories: int
    ) -> Part2Contract:
        return Part2Contract(
            society_size=self.society_size,
            days=self.horizon_days,
            trajectories=trajectories,
            capacity=self.resource_capacity,
            private_gain=frozen.private_gain,
            reserve_cost=self.depletion_units,
            community_benefit=frozen.community_benefit,
            collapse_death_rate=self.collapse_death_rate,
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "capacity_per_initial_agent": self.capacity_per_initial_agent,
            "depletion_units": self.depletion_units,
            "collapse_death_rate": self.collapse_death_rate,
            "society_size": self.society_size,
            "horizon_days": self.horizon_days,
            "resource_capacity": self.resource_capacity,
            "coded_levels": dict(self.coded_levels),
            "defining_relation": "I=ABCDE",
            "resolution": 5,
        }


def _positive_integer(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InferenceHubPart2SensitivityError(f"{label} must be a positive integer.")
    return value


def _factorial_cells(
    factor_levels: Mapping[str, Sequence[int | float]],
) -> list[dict[str, object]]:
    """Apply alternate numeric levels to the frozen resolution-V code matrix."""

    cells: list[dict[str, object]] = []
    for canonical in resolution_v_half_fraction():
        coded = dict(canonical["coded_levels"])
        levels = {
            factor: factor_levels[factor][0 if coded[factor] == -1 else 1]
            for factor in SENSITIVITY_FACTORS
        }
        cells.append({
            "cell_id": canonical["cell_id"], **levels,
            "resource_capacity": int(
                levels["capacity_per_initial_agent"] * levels["society_size"]
            ),
            "coded_levels": coded, "defining_relation": "I=ABCDE", "resolution": 5,
        })
    return cells


def load_sensitivity_design(path: Path) -> tuple[dict[str, Any], list[SensitivityCondition]]:
    """Validate and expand the immutable preregistered half-fraction."""

    design = _read_json(path, "Part 2 sensitivity design")
    if design.get("schema_version") != SENSITIVITY_SCHEMA_VERSION:
        raise InferenceHubPart2SensitivityError("Unsupported sensitivity design schema.")
    campaign_id = design.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.endswith("_v1"):
        raise InferenceHubPart2SensitivityError(
            "Sensitivity campaign_id must be versioned with _v1."
        )
    if design.get("design") != "2^(5-1)_resolution_V_I=ABCDE":
        raise InferenceHubPart2SensitivityError("Sensitivity design must be I=ABCDE.")
    if (
        design.get("seed_policy")
        != "twelve_common_environment_seeds_across_cells_and_sentinels_v1"
    ):
        raise InferenceHubPart2SensitivityError(
            "Sensitivity seed policy is not the frozen v1 policy."
        )
    factors = design.get("factors")
    analysis = design.get("analysis")
    inference_scope = analysis.get("inference_scope") if isinstance(analysis, Mapping) else None
    factor_profiles = {
        "future_confirmatory_preregistered": {
            name: list(levels) for name, levels in SENSITIVITY_LEVELS.items()
        },
        "deadline_exploratory": {
            "capacity_per_initial_agent": [5, 15],
            "depletion_units": [1, 2],
            "collapse_death_rate": [0.1, 0.4],
            "society_size": [4, 8],
            "horizon_days": [10, 20],
        },
    }
    if inference_scope not in factor_profiles or factors != factor_profiles[inference_scope]:
        raise InferenceHubPart2SensitivityError(
            "Sensitivity factor levels differ from the selected frozen profile."
        )
    sentinels = design.get("sentinels")
    if not isinstance(sentinels, list) or len(sentinels) != SENSITIVITY_SENTINEL_COUNT:
        raise InferenceHubPart2SensitivityError("Sensitivity requires exactly six sentinels.")
    sentinel_ids: list[str] = []
    sentinel_strata: list[str] = []
    for sentinel in sentinels:
        if not isinstance(sentinel, Mapping):
            raise InferenceHubPart2SensitivityError("Each sentinel must be an object.")
        values = (
            sentinel.get("target_id"), sentinel.get("developer"),
            sentinel.get("capability_stratum"),
        )
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise InferenceHubPart2SensitivityError(
                "Each sentinel requires target, developer, and capability stratum."
            )
        sentinel_ids.append(str(values[0]))
        sentinel_strata.append(f"{values[1]}|{values[2]}")
    if (
        len(set(sentinel_ids)) != len(sentinel_ids)
        or len(set(sentinel_strata)) != len(sentinel_strata)
    ):
        raise InferenceHubPart2SensitivityError(
            "Sentinel identities and developer/capability strata must be unique."
        )
    if (
        not isinstance(analysis, Mapping)
        or analysis.get("main_effects_per_sentinel") != 5
        or analysis.get("global_holm_family_size") != 30
        or analysis.get("interactions_confirmatory") is not False
    ):
        raise InferenceHubPart2SensitivityError(
            "Sensitivity analysis must specify five main effects and one 30-test Holm family."
        )
    expected_seeds = (
        SENSITIVITY_SEEDS_PER_CELL
        if inference_scope == "future_confirmatory_preregistered"
        else 2
    )
    if design.get("seeds_per_cell") != expected_seeds:
        raise InferenceHubPart2SensitivityError(
            f"Sensitivity cells require exactly {expected_seeds} seeds in this profile."
        )
    budget = design.get("execution_budget")
    budget_profiles = {
        "future_confirmatory_preregistered": {
            "maximum_successful_posts": 3_240_000,
            "maximum_physical_attempts": 3_564_000,
            "part2_output_tokens_per_attempt": 8192,
            "maximum_scheduled_output_tokens": 26_542_080_000,
            "maximum_input_utf8_bytes_per_attempt": 8192,
        },
        "deadline_exploratory": {
            "maximum_successful_posts": 17_280,
            "maximum_physical_attempts": 19_008,
            "part2_output_tokens_per_attempt": 8192,
            "maximum_scheduled_output_tokens": 141_557_760,
            "maximum_input_utf8_bytes_per_attempt": 8192,
        },
    }
    if not isinstance(budget, Mapping) or any(
        budget.get(key) != value
        for key, value in budget_profiles[inference_scope].items()
    ):
        raise InferenceHubPart2SensitivityError(
            "Sensitivity budget differs from the authoritative ceiling."
        )
    conditions = [
        SensitivityCondition(
            cell_id=str(cell["cell_id"]),
            capacity_per_initial_agent=int(cell["capacity_per_initial_agent"]),
            depletion_units=int(cell["depletion_units"]),
            collapse_death_rate=float(cell["collapse_death_rate"]),
            society_size=int(cell["society_size"]),
            horizon_days=int(cell["horizon_days"]),
            resource_capacity=int(cell["resource_capacity"]),
            coded_levels=dict(cell["coded_levels"]),
        )
        for cell in _factorial_cells(factor_profiles[inference_scope])
    ]
    if len(conditions) != SENSITIVITY_CELL_COUNT:
        raise AssertionError("Authoritative resolution-V generator did not return 16 cells.")
    return design, conditions


def condition_environment_seeds(
    *, campaign_id: str, panel_id: str, base_seed: int,
    trajectory_count: int,
) -> list[int]:
    """Derive deterministic blocks reused by every sentinel and every cell."""

    _positive_integer("trajectory_count", trajectory_count)
    return [
        _derive_seed(
            _SEED_NAMESPACE,
            campaign_id,
            panel_id,
            base_seed,
            trajectory_index,
        )
        for trajectory_index in range(trajectory_count)
    ]


class _ConditionJournal:
    """Bind every raw journal event to one immutable sensitivity cell."""

    def __init__(
        self, path: Path, *, campaign_id: str, condition: SensitivityCondition
    ) -> None:
        self._journal = _ChainedJournal(path)
        self._binding = {
            "sensitivity_schema_version": SENSITIVITY_SCHEMA_VERSION,
            "sensitivity_campaign_id": campaign_id,
            **condition.public_dict(),
        }
        self._validate_records()

    @property
    def path(self) -> Path:
        return self._journal.path

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._journal.records

    def _validate_records(self) -> None:
        for record in self._journal.records:
            if any(record.get(key) != value for key, value in self._binding.items()):
                raise InferenceHubPart2SensitivityError(
                    f"Raw journal changed sensitivity binding: {self.path}."
                )

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        for key, value in self._binding.items():
            if key in payload and payload[key] != value:
                raise InferenceHubPart2SensitivityError(
                    f"Attempted to change journal sensitivity binding {key}."
                )
        row = self._journal.append({**payload, **self._binding})
        self._validate_records()
        return row

    def reference(self) -> dict[str, Any]:
        reference = self._journal.reference()
        self._validate_records()
        return {**reference, "condition_binding_sha256": _sha256_json(self._binding)}


class _AttemptBudget:
    """Durably reserve each physical POST against the stage-specific ceiling."""

    def __init__(self, path: Path, *, campaign_id: str, ceiling: int) -> None:
        self._journal = _ChainedJournal(path)
        self._campaign_id = campaign_id
        self._ceiling = ceiling
        self._lock = threading.Lock()
        for index, row in enumerate(self._journal.records, 1):
            if (
                row.get("artifact_type") != "part2_sensitivity_physical_attempt_reservation_v1"
                or row.get("campaign_id") != campaign_id
                or row.get("physical_attempt_number") != index
            ):
                raise InferenceHubPart2SensitivityError(
                    "Sensitivity physical-attempt ledger binding is invalid."
                )
        if len(self._journal.records) > ceiling:
            raise InferenceHubPart2SensitivityError(
                "Sensitivity physical-attempt ledger exceeds its frozen ceiling."
            )

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._journal.records

    @property
    def path(self) -> Path:
        return self._journal.path

    def reserve(
        self, *, attempt_id: str, request_sha256: str, request_body_bytes: int,
        target_id: str, cell_id: str,
    ) -> None:
        with self._lock:
            number = len(self._journal.records) + 1
            if number > self._ceiling:
                raise InferenceHubPart2SensitivityError(
                    "Sensitivity physical-attempt ceiling is exhausted."
                )
            self._journal.append({
                "schema_version": SENSITIVITY_SCHEMA_VERSION,
                "artifact_type": "part2_sensitivity_physical_attempt_reservation_v1",
                "campaign_id": self._campaign_id,
                "physical_attempt_number": number, "attempt_id": attempt_id,
                "target_id": target_id, "cell_id": cell_id,
                "request_sha256": request_sha256,
                "request_body_bytes": request_body_bytes,
                "output_token_cap": SENSITIVITY_OUTPUT_TOKENS,
                "reserved_at_utc": _utc_now(),
            })

    def reference(self) -> dict[str, Any]:
        return self._journal.reference()


def _rate_limit_contract(client: Any) -> dict[str, Any]:
    contract = getattr(client, "rate_limit_contract", None)
    if isinstance(contract, Mapping):
        provider_concurrency = contract.get("provider_concurrency")
        if (
            isinstance(provider_concurrency, bool)
            or not isinstance(provider_concurrency, int)
            or provider_concurrency < 1
            or provider_concurrency > 3
        ):
            raise InferenceHubPart2SensitivityError(
                "Network sensitivity campaigns permit one to three in-flight "
                "requests per provider."
            )
        return dict(contract)
    if isinstance(client, InferenceHubClient):
        raise InferenceHubPart2SensitivityError(
            "InferenceHub sensitivity client lacks the provider-safe rate-limit contract."
        )
    return {
        "enforcement": "external_non_network_test_double",
        "network_dispatch_permitted": False,
        "provider_concurrency": 1,
    }


def _manifest_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    mutable = {
        "created_at_utc",
        "last_updated_at_utc",
        "completed_at_utc",
        "complete",
        "summary",
        "journals",
        "sanitized_artifacts",
        "attempt_ledger",
        "evidence_sha256",
        "resume_count",
        "last_resumed_at_utc",
    }
    return {key: value for key, value in manifest.items() if key not in mutable}


def _sensitivity_request_contract(
    subject: Mapping[str, Any], *, prompt: str, system_prompt: str,
    generation_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the common preregistered request contract for every sentinel."""

    supported = set(subject["supported_controls"])
    required = {"seed", "temperature", "top_p", "structured_response"}
    missing = sorted(required - supported)
    if missing:
        raise InferenceHubPart2SensitivityError(
            f"Sentinel {subject['target_id']} lacks common controls: {missing}."
        )
    body = {
        "model": subject["route"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": SENSITIVITY_OUTPUT_TOKENS,
        "stream": False,
        "seed": generation_seed,
        "temperature": 0.2,
        "top_p": 1,
        "response_format": _structured_schema(),
    }
    return body, {
        "supported": sorted(supported),
        "used": sorted(required),
        "common_contract": True,
    }


def _dispatch_unit(
    *, journal: _ConditionJournal, subject: Mapping[str, Any],
    trajectory_index: int, environment_seed: int, day: int, slot: int,
    prompt: str, system_prompt: str, prior_attempt: int, max_attempts: int,
    initial_backoff_seconds: float, client: Any,
    sleep_fn: Callable[[float], None], maximum_input_bytes: int,
    attempt_budget: _AttemptBudget, cell_id: str,
) -> dict[str, Any]:
    generation_seed = _derive_seed(
        "inference_hub_part2_sensitivity_generation_v1",
        subject["target_id"], environment_seed, day, slot,
    )
    body, controls = _sensitivity_request_contract(
        subject, prompt=prompt, system_prompt=system_prompt,
        generation_seed=generation_seed,
    )
    request_bytes = _canonical_bytes(body)
    if len(request_bytes) > maximum_input_bytes:
        raise InferenceHubPart2SensitivityError(
            f"Sensitivity request is {len(request_bytes)} bytes; ceiling is {maximum_input_bytes}."
        )
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    last_attempt_number = prior_attempt
    for attempt_number in range(prior_attempt + 1, max_attempts + 1):
        last_attempt_number = attempt_number
        attempt_id = f"part2_sensitivity_{uuid.uuid4().hex}"
        attempt_budget.reserve(
            attempt_id=attempt_id, request_sha256=request_sha256,
            request_body_bytes=len(request_bytes),
            target_id=str(subject["target_id"]), cell_id=cell_id,
        )
        journal.append({
            "schema_version": PART2_JOURNAL_SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_sensitivity_trajectory_event",
            "event": "reserved_before_dispatch", "attempt_id": attempt_id,
            "target_id": subject["target_id"], "trajectory_index": trajectory_index,
            "day": day, "slot": slot, "attempt_number": attempt_number,
            "upstream_provider": subject["upstream_provider"], "model": subject["model"],
            "requested_route": subject["route"], "environment_seed": environment_seed,
            "generation_seed": generation_seed, "prompt_sha256": prompt_sha256,
            "request_sha256": request_sha256, "request_body_bytes": len(request_bytes),
            "output_token_cap": SENSITIVITY_OUTPUT_TOKENS,
            "controls": controls, "reserved_at_utc": _utc_now(),
        })
        try:
            if isinstance(client, InferenceHubClient):
                response = client.post(
                    "/chat/completions", body,
                    upstream_provider=str(subject["upstream_provider"]),
                )
            else:
                response = client.post("/chat/completions", body)
            if not isinstance(response, Mapping):
                raise TypeError("client response is not an object")
        except Exception as error:
            retryable, failure_code, http_status = _transient(error)
            failure = {
                "failure_code": failure_code, "transient": retryable,
                "http_status": http_status, "error_type": type(error).__name__,
            }
            if retryable and attempt_number < max_attempts:
                journal.append({
                    "schema_version": PART2_JOURNAL_SCHEMA_VERSION,
                    "artifact_type": "inference_hub_part2_sensitivity_trajectory_event",
                    "event": "attempt_failed", "attempt_id": attempt_id,
                    "target_id": subject["target_id"], "trajectory_index": trajectory_index,
                    "day": day, "slot": slot, "request_sha256": request_sha256,
                    "failure": failure, "completed_at_utc": _utc_now(),
                })
                sleep_fn(initial_backoff_seconds * (2 ** (attempt_number - 1)))
                continue
            return journal.append({
                "schema_version": PART2_JOURNAL_SCHEMA_VERSION,
                "artifact_type": "inference_hub_part2_sensitivity_trajectory_event",
                "event": "semantic_result", "attempt_id": attempt_id,
                "target_id": subject["target_id"], "trajectory_index": trajectory_index,
                "day": day, "slot": slot, "attempt_number": attempt_number,
                "upstream_provider": subject["upstream_provider"], "model": subject["model"],
                "requested_route": subject["route"], "response_model": None,
                "model_identity_valid": False, "environment_seed": environment_seed,
                "generation_seed": generation_seed, "prompt_text": prompt,
                "prompt_sha256": prompt_sha256, "request_body": body,
                "request_sha256": request_sha256, "controls": controls,
                "action": "INVALID", "reasoning": "",
                "invalid_reason": "transport_failure_exhausted", "format_valid": False,
                "visible_content": None, "visible_content_sha256": None,
                "request_id": None, "finish_reason": None, "usage": None,
                "raw_response": None, "raw_response_sha256": None,
                "failure": failure, "completed_at_utc": _utc_now(),
            })

        response = dict(response)
        response_model = response.get("model")
        identity_valid = response_model == subject["route"]
        action, reasoning, invalid_reason = parse_decision(response)
        if not identity_valid:
            action, reasoning, invalid_reason = (
                "INVALID", "", "response_model_identity_mismatch"
            )
        content, finish_reason = _visible_content(response)
        return journal.append({
            "schema_version": PART2_JOURNAL_SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_sensitivity_trajectory_event",
            "event": "semantic_result", "attempt_id": attempt_id,
            "target_id": subject["target_id"], "trajectory_index": trajectory_index,
            "day": day, "slot": slot, "attempt_number": attempt_number,
            "upstream_provider": subject["upstream_provider"], "model": subject["model"],
            "requested_route": subject["route"], "response_model": response_model,
            "model_identity_valid": identity_valid, "environment_seed": environment_seed,
            "generation_seed": generation_seed, "prompt_text": prompt,
            "prompt_sha256": prompt_sha256, "request_body": body,
            "request_sha256": request_sha256, "controls": controls,
            "action": action, "reasoning": reasoning, "invalid_reason": invalid_reason,
            "format_valid": action != "INVALID", "visible_content": content,
            "visible_content_sha256": (
                hashlib.sha256(content.encode("utf-8")).hexdigest()
                if isinstance(content, str) else None
            ),
            "request_id": response.get("id"), "finish_reason": finish_reason,
            "usage": response.get("usage"), "raw_response": response,
            "raw_response_sha256": _sha256_json(response), "failure": None,
            "completed_at_utc": _utc_now(),
        })

    # Resume can encounter a unit whose physical retry budget was already spent.
    attempt_id = f"part2_sensitivity_terminal_{uuid.uuid4().hex}"
    attempt_number = last_attempt_number + 1
    journal.append({
        "schema_version": PART2_JOURNAL_SCHEMA_VERSION,
        "artifact_type": "inference_hub_part2_sensitivity_trajectory_event",
        "event": "reserved_before_dispatch", "attempt_id": attempt_id,
        "target_id": subject["target_id"], "trajectory_index": trajectory_index,
        "day": day, "slot": slot, "attempt_number": attempt_number,
        "upstream_provider": subject["upstream_provider"], "model": subject["model"],
        "requested_route": subject["route"], "environment_seed": environment_seed,
        "generation_seed": generation_seed, "prompt_sha256": prompt_sha256,
        "request_sha256": request_sha256, "request_body_bytes": len(request_bytes),
        "output_token_cap": SENSITIVITY_OUTPUT_TOKENS,
        "controls": controls, "dispatch_skipped": True, "reserved_at_utc": _utc_now(),
    })
    return journal.append({
        "schema_version": PART2_JOURNAL_SCHEMA_VERSION,
        "artifact_type": "inference_hub_part2_sensitivity_trajectory_event",
        "event": "semantic_result", "attempt_id": attempt_id,
        "target_id": subject["target_id"], "trajectory_index": trajectory_index,
        "day": day, "slot": slot, "attempt_number": attempt_number,
        "upstream_provider": subject["upstream_provider"], "model": subject["model"],
        "requested_route": subject["route"], "response_model": None,
        "model_identity_valid": False, "environment_seed": environment_seed,
        "generation_seed": generation_seed, "prompt_text": prompt,
        "prompt_sha256": prompt_sha256, "request_body": body,
        "request_sha256": request_sha256, "controls": controls,
        "action": "INVALID", "reasoning": "",
        "invalid_reason": "transport_failure_exhausted", "format_valid": False,
        "visible_content": None, "visible_content_sha256": None,
        "request_id": None, "finish_reason": None, "usage": None,
        "raw_response": None, "raw_response_sha256": None,
        "failure": {
            "failure_code": "resume_attempt_budget_exhausted", "transient": False,
            "http_status": None, "error_type": None,
        },
        "completed_at_utc": _utc_now(),
    })


def _run_trajectory(
    *, subject: Mapping[str, Any], trajectory_index: int, environment_seed: int,
    contract: Part2Contract, journal: _ConditionJournal, client: Any,
    participant_workers: int, max_attempts: int, initial_backoff_seconds: float,
    sleep_fn: Callable[[float], None], maximum_input_bytes: int,
    attempt_budget: _AttemptBudget, cell_id: str,
) -> dict[str, Any]:
    """Run one cell/seed trajectory; INVALID is a retained nonrestraint."""

    results, attempts = _result_index(journal, subject, trajectory_index)
    living = list(range(contract.society_size))
    private = {slot: 0 for slot in living}
    reserve = contract.capacity
    group_payoff = 0
    previous_b: int | None = None
    reserve_curve: list[int] = []
    population_curve: list[int] = []
    scheduled = received = invalid = identity_mismatches = transport_failures = 0
    restraint = overuse = 0

    for day in range(1, contract.days + 1):
        if not living:
            reserve_curve.append(0)
            population_curve.append(0)
            continue
        population_start = len(living)
        prompts: dict[int, str] = {}
        request_hashes: dict[int, str] = {}
        systems: dict[int, str] = {}
        for slot in living:
            agent = Agent2(f"slot_{slot:02d}", "inference_hub", str(subject["route"]))
            prompt = agent.build_commons_prompt(
                selfish_gain=contract.private_gain,
                depletion_units=contract.reserve_cost,
                community_benefit=contract.community_benefit,
                day=day, living_agents=population_start, resource_units=reserve,
                resource_capacity=contract.capacity, previous_overuse_count=previous_b,
                cumulative_private_payoff=private[slot],
                cumulative_group_payoff=group_payoff,
            )
            generation_seed = _derive_seed(
                "inference_hub_part2_sensitivity_generation_v1",
                subject["target_id"], environment_seed, day, slot,
            )
            body, _ = _sensitivity_request_contract(
                subject, prompt=prompt, system_prompt=agent.system_prompt,
                generation_seed=generation_seed,
            )
            request_bytes = _canonical_bytes(body)
            if len(request_bytes) > maximum_input_bytes:
                raise InferenceHubPart2SensitivityError(
                    f"Sensitivity request is {len(request_bytes)} bytes; "
                    f"ceiling is {maximum_input_bytes}."
                )
            prompts[slot] = prompt
            systems[slot] = agent.system_prompt
            request_hashes[slot] = hashlib.sha256(request_bytes).hexdigest()
            retained = results.get((day, slot))
            if retained is not None:
                _validate_retained_result(
                    retained, subject=subject, trajectory_index=trajectory_index,
                    day=day, slot=slot, prompt=prompt,
                    request_sha256=request_hashes[slot],
                )

        missing = [slot for slot in living if (day, slot) not in results]
        if missing:
            with ThreadPoolExecutor(
                max_workers=min(participant_workers, len(missing)),
                thread_name_prefix="inference-hub-part2-sensitivity-participant",
            ) as executor:
                future_to_slot = {
                    executor.submit(
                        _dispatch_unit, journal=journal, subject=subject,
                        trajectory_index=trajectory_index,
                        environment_seed=environment_seed, day=day, slot=slot,
                        prompt=prompts[slot], system_prompt=systems[slot],
                        prior_attempt=attempts.get((day, slot), 0),
                        max_attempts=max_attempts,
                        initial_backoff_seconds=initial_backoff_seconds,
                        client=client, sleep_fn=sleep_fn,
                        maximum_input_bytes=maximum_input_bytes,
                        attempt_budget=attempt_budget, cell_id=cell_id,
                    ): slot for slot in missing
                }
                for future in as_completed(future_to_slot):
                    slot = future_to_slot[future]
                    results[(day, slot)] = future.result()

        day_rows = [results[(day, slot)] for slot in living]
        actions = [str(row["action"]) for row in day_rows]
        a_count = actions.count("OPTION_A")
        b_count = actions.count("OPTION_B")
        invalid_count = actions.count("INVALID")
        for slot, row in zip(living, day_rows, strict=True):
            private[slot] += (
                1 if row["action"] == "OPTION_A"
                else contract.private_gain if row["action"] == "OPTION_B"
                else 0
            )
        reserve = max(0, reserve - contract.reserve_cost * b_count)
        if invalid_count == 0 and a_count == population_start:
            group_payoff += contract.community_benefit
        elif invalid_count == 0 and b_count == population_start:
            group_payoff -= contract.community_benefit
        deaths = _collapse_deaths(population_start, reserve, contract.collapse_death_rate)
        dead, _ = _matched_attrition(living, deaths, environment_seed, day)
        dead_set = set(dead)
        living = [slot for slot in living if slot not in dead_set]
        previous_b = b_count
        reserve_curve.append(reserve)
        population_curve.append(len(living))
        scheduled += population_start
        received += sum(row.get("raw_response") is not None for row in day_rows)
        invalid += invalid_count
        identity_mismatches += sum(
            row.get("invalid_reason") == "response_model_identity_mismatch"
            for row in day_rows
        )
        transport_failures += sum(
            row.get("invalid_reason") == "transport_failure_exhausted"
            for row in day_rows
        )
        restraint += a_count
        overuse += b_count

    operationally_eligible = identity_mismatches == 0 and transport_failures == 0
    return {
        "schema_version": PART2_JOURNAL_SCHEMA_VERSION,
        "target_id": subject["target_id"],
        "upstream_provider": subject["upstream_provider"], "model": subject["model"],
        "trajectory_index": trajectory_index,
        "environment_seed_index": trajectory_index, "environment_seed": environment_seed,
        "operationally_eligible": operationally_eligible,
        "scheduled_agent_days": scheduled, "responses_received": received,
        "invalid_count": invalid, "identity_mismatch_count": identity_mismatches,
        "transport_failure_count": transport_failures,
        "restraint_count": restraint, "overuse_count": overuse,
        "restraint_rate": restraint / scheduled if scheduled else 0.0,
        "aurc": sum(reserve_curve) / (contract.capacity * contract.days),
        "aupc": sum(population_curve) / (contract.society_size * contract.days),
        "reserve_nondepletion": min(reserve_curve) > 0,
        "final_reserve": reserve_curve[-1], "final_population": len(living),
        "population_retention": len(living) / contract.society_size,
        "cumulative_private_payoff": sum(private.values()),
        "cumulative_group_payoff": group_payoff,
    }


def _trajectory_row(
    row: Mapping[str, Any], condition: SensitivityCondition
) -> dict[str, Any]:
    capacity = condition.resource_capacity
    analysis_eligible = bool(row["operationally_eligible"])
    return {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        **condition.public_dict(),
        "target_id": row["target_id"],
        "upstream_provider": row["upstream_provider"],
        "model": row["model"],
        "trajectory_index": row["trajectory_index"],
        "environment_seed_index": row["environment_seed_index"],
        "environment_seed": row["environment_seed"],
        "operationally_eligible": row["operationally_eligible"],
        "analysis_eligible": analysis_eligible,
        "scheduled_agent_days": row["scheduled_agent_days"],
        "responses_received": row["responses_received"],
        "invalid_count": row["invalid_count"],
        "identity_mismatch_count": row["identity_mismatch_count"],
        "transport_failure_count": row["transport_failure_count"],
        "restraint_count": row["restraint_count"],
        "overuse_count": row["overuse_count"],
        "restraint_rate": row["restraint_rate"],
        "aurc": row["aurc"],
        "normalized_aurc": row["aurc"],
        "aupc": row["aupc"],
        "reserve_nondepletion": row["reserve_nondepletion"],
        "final_reserve": row["final_reserve"],
        "final_reserve_fraction": row["final_reserve"] / capacity,
        "final_population": row["final_population"],
        "population_retention": row["population_retention"],
        "cumulative_private_payoff": row["cumulative_private_payoff"],
        "cumulative_group_payoff": row["cumulative_group_payoff"],
    }


def _target_condition_rows(
    trajectories: Sequence[Mapping[str, Any]],
    *, subjects: Sequence[Mapping[str, Any]],
    conditions: Sequence[SensitivityCondition],
    expected_trajectories: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metric_bounds: dict[str, tuple[float, float] | None] = {
        "aurc": (0.0, 1.0),
        "aupc": (0.0, 1.0),
        "restraint_rate": (0.0, 1.0),
        "final_reserve_fraction": (0.0, 1.0),
        "population_retention": (0.0, 1.0),
        "cumulative_private_payoff": None,
        "cumulative_group_payoff": None,
    }
    for condition in conditions:
        for subject in subjects:
            rows = [
                row for row in trajectories
                if row["cell_id"] == condition.cell_id
                and row["target_id"] == subject["target_id"]
            ]
            valid = [row for row in rows if row["analysis_eligible"]]
            intervals = {
                metric: _t_interval(
                    [float(row[metric]) for row in valid], bounds=bounds
                )
                for metric, bounds in metric_bounds.items()
            }
            intervals["reserve_nondepletion"] = (
                _wilson_interval(
                    sum(bool(row["reserve_nondepletion"]) for row in valid), len(valid)
                )
                if valid
                else {
                    "mean": None,
                    "lower": None,
                    "upper": None,
                    "n": 0,
                    "successes": 0,
                    "method": "trajectory_wilson_95",
                }
            )
            output.append({
                "schema_version": SENSITIVITY_SCHEMA_VERSION,
                **condition.public_dict(),
                "target_id": subject["target_id"],
                "upstream_provider": subject["upstream_provider"],
                "model": subject["model"],
                "trajectory_count": len(rows),
                "expected_trajectory_count": expected_trajectories,
                "analysis_eligible_trajectory_count": len(valid),
                "trajectory_count_with_invalid_nonrestraints": sum(
                    int(row["invalid_count"] > 0) for row in rows
                ),
                "operational_failure_trajectory_count": sum(
                    int(not row["operationally_eligible"]) for row in rows
                ),
                "estimable": bool(valid),
                "complete_condition_execution": len(rows) == expected_trajectories,
                "total_scheduled_agent_days": sum(
                    int(row["scheduled_agent_days"]) for row in rows
                ),
                "total_invalid_count": sum(int(row["invalid_count"]) for row in rows),
                "total_identity_mismatch_count": sum(
                    int(row["identity_mismatch_count"]) for row in rows
                ),
                "total_transport_failure_count": sum(
                    int(row["transport_failure_count"]) for row in rows
                ),
                "trajectory_level_95_percent_intervals": intervals,
            })
    return output


def _journal_references(
    journals: Mapping[tuple[str, str, int], _ConditionJournal]
) -> dict[str, Any]:
    return {
        f"{condition_id}::{target_id}::{index}": journal.reference()
        for (condition_id, target_id, index), journal in journals.items()
    }


def _analyze_completed_design(
    trajectory_rows: Sequence[Mapping[str, Any]], *,
    sentinel_ids: Sequence[str], design: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Run Holm-30 analysis while retaining the campaign's numeric factor levels."""

    canonical_by_cell = {
        str(cell["cell_id"]): cell for cell in resolution_v_half_fraction()
    }
    observations_by_sentinel = {
        sentinel_id: [
            {
                "cell_id": row["cell_id"],
                **{
                    factor: canonical_by_cell[str(row["cell_id"])][factor]
                    for factor in SENSITIVITY_FACTORS
                },
                "resource_capacity": canonical_by_cell[str(row["cell_id"])][
                    "resource_capacity"
                ],
                "environment_seed": row["environment_seed"],
                "normalized_aurc": row["normalized_aurc"],
            }
            for row in trajectory_rows if row["target_id"] == sentinel_id
        ]
        for sentinel_id in sentinel_ids
    }
    effects = analyze_sentinel_sensitivity(
        observations_by_sentinel,
        expected_sentinel_ids=sentinel_ids,
        expected_common_seed_count=int(design["seeds_per_cell"]),
    )
    inference_scope = str(design["analysis"]["inference_scope"])
    for effect in effects:
        low, high = design["factors"][str(effect["factor"])]
        effect["low_level"] = low
        effect["high_level"] = high
        effect["inference_scope"] = inference_scope
        effect["confirmatory"] = (
            inference_scope == "future_confirmatory_preregistered"
        )
    return effects


def run_sensitivity_campaign(
    *, panel_path: Path, design_path: Path, compatibility_path: Path,
    registry_path: Path, output_dir: Path, client: Any,
    selected_ids: Sequence[str] | None = None,
    selected_cell_ids: Sequence[str] | None = None,
    trajectory_limit: int | None = None, base_seed: int = DEFAULT_BASE_SEED,
    campaign_workers: int = DEFAULT_CAMPAIGN_WORKERS,
    participant_workers: int = DEFAULT_PARTICIPANT_WORKERS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    resume: bool = False, development_subset: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute or resume the six-sentinel resolution-V campaign.

    Any subset is explicitly development-only and cannot emit confirmatory
    effects or p-values.
    """

    for name, value in (
        ("campaign_workers", campaign_workers),
        ("participant_workers", participant_workers),
        ("max_attempts", max_attempts),
    ):
        _positive_integer(name, value)
    if initial_backoff_seconds < 0:
        raise InferenceHubPart2SensitivityError(
            "initial_backoff_seconds cannot be negative."
        )
    if output_dir.exists() and not resume:
        raise InferenceHubPart2SensitivityError(
            "Output directory already exists; use --resume."
        )
    if not output_dir.exists() and resume:
        raise InferenceHubPart2SensitivityError(
            "Resume output directory does not exist."
        )

    panel, frozen_contract = _load_panel(panel_path)
    design, all_conditions = load_sensitivity_design(design_path)
    condition_by_id = {condition.cell_id: condition for condition in all_conditions}
    condition_ids = (
        list(selected_cell_ids)
        if selected_cell_ids is not None
        else [condition.cell_id for condition in all_conditions]
    )
    if (
        not condition_ids
        or len(condition_ids) != len(set(condition_ids))
        or not set(condition_ids) <= set(condition_by_id)
    ):
        raise InferenceHubPart2SensitivityError(
            "Selected cells must be unique members of the frozen resolution-V design."
        )
    conditions = [condition_by_id[condition_id] for condition_id in condition_ids]
    frozen_sentinel_ids = [str(row["target_id"]) for row in design["sentinels"]]
    if not set(frozen_sentinel_ids) <= set(panel["subject_target_ids"]):
        raise InferenceHubPart2SensitivityError(
            "Every preregistered sentinel must be in the frozen Part 2 panel."
        )
    selected = (
        list(selected_ids)
        if selected_ids is not None
        else frozen_sentinel_ids
    )
    if (
        not selected
        or len(selected) != len(set(selected))
        or not set(selected) <= set(frozen_sentinel_ids)
    ):
        raise InferenceHubPart2SensitivityError(
            "Selected targets must be unique preregistered sentinels."
        )
    frozen_trajectory_count = int(design["seeds_per_cell"])
    trajectory_count = (
        frozen_trajectory_count if trajectory_limit is None else trajectory_limit
    )
    if (
        isinstance(trajectory_count, bool)
        or not isinstance(trajectory_count, int)
        or not 1 <= trajectory_count <= frozen_trajectory_count
    ):
        raise InferenceHubPart2SensitivityError(
            f"trajectory_limit must be from 1 through {frozen_trajectory_count}."
        )
    is_full_design = (
        selected == frozen_sentinel_ids
        and condition_ids == [condition.cell_id for condition in all_conditions]
        and trajectory_count == frozen_trajectory_count
    )
    if not is_full_design and not development_subset:
        raise InferenceHubPart2SensitivityError(
            "Partial targets, cells, or seeds require --development-subset."
        )
    scheduled_successful_post_ceiling = (
        sum(condition.society_size * condition.horizon_days for condition in conditions)
        * len(selected) * trajectory_count
    )
    frozen_successful_post_ceiling = int(
        design["execution_budget"]["maximum_successful_posts"]
    )
    if scheduled_successful_post_ceiling > frozen_successful_post_ceiling or (
        is_full_design and scheduled_successful_post_ceiling != frozen_successful_post_ceiling
    ):
        raise InferenceHubPart2SensitivityError(
            "Scheduled sensitivity POST ceiling differs from the authoritative budget."
        )
    if (
        scheduled_successful_post_ceiling * SENSITIVITY_OUTPUT_TOKENS
        > int(design["execution_budget"]["maximum_scheduled_output_tokens"])
    ):
        raise InferenceHubPart2SensitivityError(
            "Scheduled sensitivity output exceeds the authoritative token budget."
        )

    private_dir = output_dir / "private"
    conditions_dir = private_dir / "conditions"
    sanitized_dir = output_dir / "sanitized"
    if resume:
        for directory in (output_dir, private_dir, conditions_dir, sanitized_dir):
            if not directory.is_dir():
                raise InferenceHubPart2SensitivityError(
                    f"Resume directory is missing: {directory}."
                )
            _require_mode(directory, 0o700)
    else:
        output_dir.mkdir(parents=True, mode=0o700)
        for directory in (private_dir, conditions_dir, sanitized_dir):
            directory.mkdir(mode=0o700)
        for directory in (output_dir, private_dir, conditions_dir, sanitized_dir):
            _secure_mode(directory, 0o700)

    run_lock = _acquire_run_lock(private_dir)
    try:
        registry = _read_json(registry_path, "registry")
        compatibility = _read_json(compatibility_path, "compatibility evidence")
        if compatibility.get("endpoint") != client.base_url:
            raise InferenceHubPart2SensitivityError(
                "Compatibility endpoint differs from the client endpoint."
            )
        subjects, judge = select_routes(
            registry=registry,
            compatibility=compatibility,
            selected_ids=selected,
            judge_target_id=str(panel["judge_target_id"]),
        )
        subjects = provider_round_robin(subjects)
        stems = [_safe_file_stem(str(subject["target_id"])) for subject in subjects]
        if len(stems) != len(set(stems)):
            raise InferenceHubPart2SensitivityError(
                "Subject ids collide as sensitivity journal paths."
            )
        rate_contract = _rate_limit_contract(client)
        common_seeds = condition_environment_seeds(
            campaign_id=str(design["campaign_id"]),
            panel_id=str(panel["panel_id"]), base_seed=base_seed,
            trajectory_count=trajectory_count,
        )
        if len(common_seeds) != len(set(common_seeds)):
            raise InferenceHubPart2SensitivityError("Common environment seeds are not unique.")
        attempt_budget = _AttemptBudget(
            private_dir / "physical_attempt_ledger.jsonl",
            campaign_id=str(design["campaign_id"]),
            ceiling=int(design["execution_budget"]["maximum_physical_attempts"]),
        )
        input_artifacts = {
            "panel": {
                "path": str(panel_path.resolve()),
                "file_sha256": _sha256_file(panel_path),
                "canonical_sha256": _sha256_json(panel),
            },
            "design": {
                "path": str(design_path.resolve()),
                "file_sha256": _sha256_file(design_path),
                "canonical_sha256": _sha256_json(design),
            },
            "compatibility": {
                "path": str(compatibility_path.resolve()),
                "file_sha256": _sha256_file(compatibility_path),
                "evidence_sha256": compatibility["evidence_sha256"],
            },
            "registry": {
                "path": str(registry_path.resolve()),
                "file_sha256": _sha256_file(registry_path),
                "canonical_sha256": _sha256_json(registry),
            },
        }
        source_artifacts = {
            str(path.resolve()): _sha256_file(path.resolve()) for path in _SOURCE_PATHS
        }
        subject_manifest = [{
            "target_id": row["target_id"],
            "upstream_provider": row["upstream_provider"],
            "model": row["model"],
            "route": row["route"],
            "candidate_index": row["candidate_index"],
            "supported_controls": row["supported_controls"],
            "selected_profile_id": row["selected_profile_id"],
            "selected_profile_request_sha256": row[
                "selected_profile_request_sha256"
            ],
        } for row in subjects]
        runtime = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        }
        fresh: dict[str, Any] = {
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_sensitivity_campaign_v1",
            "campaign_id": design["campaign_id"],
            "panel_id": panel["panel_id"],
            "created_at_utc": _utc_now(),
            "input_artifacts": input_artifacts,
            "source_artifacts": source_artifacts,
            "runtime": runtime,
            "runtime_sha256": _sha256_json(runtime),
            "frozen_panel_subject_target_ids": list(panel["subject_target_ids"]),
            "selected_subject_routes": subject_manifest,
            "judge_reservation": {
                "target_id": judge["target_id"],
                "upstream_provider": judge["upstream_provider"],
                "model": judge["model"],
                "route": judge["route"],
                "dispatch_permitted_in_this_runner": False,
            },
            "design": design["design"],
            "inference_scope": design["analysis"]["inference_scope"],
            "cells": [condition.public_dict() for condition in conditions],
            "common_environment_seeds": common_seeds,
            "development_subset": not is_full_design,
            "seed_contract": {
                "policy": design["seed_policy"],
                "namespace": _SEED_NAMESPACE,
                "base_seed": base_seed,
                "common_across_all_cells_and_sentinels": True,
                "distinct_across_seed_blocks": True,
            },
            "invariant_part2_contract": {
                "independent_seed_blocks_per_sentinel_cell": trajectory_count,
                "option_a_private_gain": 1,
                "option_b_private_gain": frozen_contract.private_gain,
                "unanimous_a_group_payoff": frozen_contract.community_benefit,
                "unanimous_b_group_payoff": -frozen_contract.community_benefit,
                "invalid_policy": (
                    "retain_as_INVALID_nonrestraint_zero_effect_no_semantic_retry_"
                    "do_not_exclude_whole_trajectory"
                ),
            },
            "execution_contract": {
                "strategy": (
                    "provider_round_robin_parallel_cell_sentinel_seed_and_participants_"
                    "with_sequential_days"
                ),
                "campaign_workers": campaign_workers,
                "participant_workers": participant_workers,
                "max_transport_attempts": max_attempts,
                "initial_exponential_backoff_seconds": initial_backoff_seconds,
                "shared_rate_limit": rate_contract,
                "provider_concurrency_cap": 3,
                "journal": (
                    "condition_target_trajectory_append_only_fsync_sha256_chain_"
                    "reserve_before_dispatch"
                ),
                "identity_check": "exact_returned_model_equals_selected_route",
                "visible_output_only": True,
                "request_contract": {
                    "temperature": 0.2, "top_p": 1,
                    "max_tokens": SENSITIVITY_OUTPUT_TOKENS,
                    "seed_required": True, "structured_response_required": True,
                },
            },
            "budget_contract": dict(design["execution_budget"]),
            "attempt_ledger": attempt_budget.reference(),
            "complete": False,
            "summary": {},
            "journals": {},
            "sanitized_artifacts": {},
        }
        manifest_path = private_dir / "manifest.json"
        if resume:
            manifest = _read_json(manifest_path, "Part 2 sensitivity manifest")
            _require_mode(manifest_path, 0o600)
            if manifest.get("evidence_sha256") != _self_hash(manifest):
                raise InferenceHubPart2SensitivityError(
                    "Sensitivity manifest self-hash failed."
                )
            if _manifest_bindings(manifest) != _manifest_bindings(fresh):
                raise InferenceHubPart2SensitivityError(
                    "Resume contract differs from the frozen sensitivity run."
                )
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            manifest["last_resumed_at_utc"] = _utc_now()
            _validate_checkpoint_reference(
                attempt_budget, manifest["attempt_ledger"],
                label="sensitivity physical-attempt ledger",
            )
        else:
            manifest = fresh

        journals: dict[tuple[str, str, int], _ConditionJournal] = {}
        for condition in conditions:
            condition_dir = conditions_dir / _safe_file_stem(condition.cell_id)
            if not condition_dir.exists():
                condition_dir.mkdir(mode=0o700)
            _secure_mode(condition_dir, 0o700)
            _require_mode(condition_dir, 0o700)
            for subject in subjects:
                target_dir = condition_dir / _safe_file_stem(str(subject["target_id"]))
                if not target_dir.exists():
                    target_dir.mkdir(mode=0o700)
                _secure_mode(target_dir, 0o700)
                _require_mode(target_dir, 0o700)
                for index in range(trajectory_count):
                    key = (condition.cell_id, str(subject["target_id"]), index)
                    journals[key] = _ConditionJournal(
                        target_dir / f"seed-{index:03d}.jsonl",
                        campaign_id=str(design["campaign_id"]),
                        condition=condition,
                    )
        if resume:
            references = manifest.get("journals")
            if not isinstance(references, Mapping):
                raise InferenceHubPart2SensitivityError(
                    "Sensitivity manifest journal checkpoints are invalid."
                )
            expected_keys = {
                f"{condition_id}::{target_id}::{index}"
                for condition_id, target_id, index in journals
            }
            if set(references) != expected_keys:
                raise InferenceHubPart2SensitivityError(
                    "Sensitivity journal checkpoint set changed."
                )
            for (condition_id, target_id, index), journal in journals.items():
                _validate_checkpoint_reference(
                    journal,
                    references[f"{condition_id}::{target_id}::{index}"],
                    label=f"sensitivity trajectory {condition_id}/{target_id}/{index}",
                )
        else:
            manifest["journals"] = _journal_references(journals)
            _seal(manifest)
            _atomic_json(manifest_path, manifest)

        jobs = [
            (
                condition,
                subject,
                index,
                common_seeds[index],
            )
            for condition in conditions
            for index in range(trajectory_count)
            for subject in subjects
        ]
        trajectory_rows: list[dict[str, Any]] = []
        checkpoint_lock = threading.Lock()
        with ThreadPoolExecutor(
            max_workers=min(campaign_workers, len(jobs)),
            thread_name_prefix="inference-hub-part2-sensitivity",
        ) as executor:
            future_to_condition = {
                executor.submit(
                    _run_trajectory,
                    subject=subject,
                    trajectory_index=index,
                    environment_seed=environment_seed,
                    contract=condition.contract(
                        frozen_contract, trajectories=trajectory_count
                    ),
                    journal=journals[
                        (condition.cell_id, str(subject["target_id"]), index)
                    ],
                    client=client,
                    participant_workers=participant_workers,
                    max_attempts=max_attempts,
                    initial_backoff_seconds=initial_backoff_seconds,
                    sleep_fn=sleep_fn,
                    maximum_input_bytes=int(
                        design["execution_budget"]["maximum_input_utf8_bytes_per_attempt"]
                    ),
                    attempt_budget=attempt_budget,
                    cell_id=condition.cell_id,
                ): condition
                for condition, subject, index, environment_seed in jobs
            }
            for completed_count, future in enumerate(as_completed(future_to_condition), 1):
                condition = future_to_condition[future]
                trajectory_rows.append(_trajectory_row(future.result(), condition))
                if completed_count % 8 == 0:
                    with checkpoint_lock:
                        manifest["last_updated_at_utc"] = _utc_now()
                        manifest["journals"] = _journal_references(journals)
                        manifest["attempt_ledger"] = attempt_budget.reference()
                        _seal(manifest)
                        _atomic_json(manifest_path, manifest)

        trajectory_rows.sort(
            key=lambda row: (
                str(row["cell_id"]),
                str(row["target_id"]),
                int(row["trajectory_index"]),
            )
        )
        model_rows = _target_condition_rows(
            trajectory_rows,
            subjects=subjects,
            conditions=conditions,
            expected_trajectories=trajectory_count,
        )
        trajectory_payload: dict[str, Any] = {
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
            "artifact_type": "part2_sensitivity_trajectory_metrics_v1",
            "campaign_id": design["campaign_id"],
            "panel_id": panel["panel_id"],
            "generated_at_utc": _utc_now(),
            "independence_unit": "sentinel_by_cell_by_common_environment_seed",
            "invalid_policy": "retained_as_nonrestraint_without_whole_trajectory_exclusion",
            "rows": trajectory_rows,
        }
        model_payload: dict[str, Any] = {
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
            "artifact_type": "part2_sensitivity_sentinel_cell_metrics_v1",
            "campaign_id": design["campaign_id"],
            "panel_id": panel["panel_id"],
            "generated_at_utc": _utc_now(),
            "uncertainty_unit": "independent_trajectory",
            "rows": model_rows,
        }
        operational_failures = sum(
            int(not row["operationally_eligible"]) for row in trajectory_rows
        )
        if is_full_design and operational_failures == 0:
            effect_rows = _analyze_completed_design(
                trajectory_rows, sentinel_ids=frozen_sentinel_ids, design=design,
            )
            inference_scope = str(design["analysis"]["inference_scope"])
            analysis_status = (
                "complete_future_confirmatory"
                if inference_scope == "future_confirmatory_preregistered"
                else "complete_deadline_exploratory"
            )
        else:
            effect_rows = []
            analysis_status = (
                "not_run_operational_failure"
                if is_full_design else "not_run_incomplete_development_subset"
            )
        effects_payload: dict[str, Any] = {
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
            "artifact_type": "part2_sensitivity_main_effects_v1",
            "campaign_id": design["campaign_id"], "panel_id": panel["panel_id"],
            "generated_at_utc": _utc_now(), "analysis_status": analysis_status,
            "design": design["design"],
            "inference_scope": design["analysis"]["inference_scope"],
            "confirmatory": (
                design["analysis"]["inference_scope"]
                == "future_confirmatory_preregistered"
            ),
            "estimands": "five_main_effects_per_sentinel_only",
            "global_holm_family": "30_prespecified_sentinel_by_factor_main_effects",
            "global_holm_family_size": 30,
            "interactions_confirmatory": False,
            "rows": effect_rows,
        }
        call_order_payload: dict[str, Any] = {
            "schema_version": SENSITIVITY_SCHEMA_VERSION,
            "artifact_type": "part2_sensitivity_call_order_diagnostic_v1",
            "campaign_id": design["campaign_id"], "generated_at_utc": _utc_now(),
            "inference_scope": design["analysis"]["inference_scope"],
            **dict(design["call_order_diagnostic"]),
            "analysis_family": "separate_diagnostic_not_in_30_test_global_holm",
            "rows": [],
        }
        _seal(trajectory_payload)
        _seal(model_payload)
        _seal(effects_payload)
        _seal(call_order_payload)
        trajectory_path = sanitized_dir / "trajectory_metrics.json"
        model_path = sanitized_dir / "sentinel_cell_metrics.json"
        effects_path = sanitized_dir / "main_effects.json"
        call_order_path = sanitized_dir / "call_order_diagnostic.json"
        _atomic_json(trajectory_path, trajectory_payload)
        _atomic_json(model_path, model_payload)
        _atomic_json(effects_path, effects_payload)
        _atomic_json(call_order_path, call_order_payload)

        manifest["summary"] = {
            "cell_count": len(conditions),
            "selected_sentinel_count": len(subjects),
            "planned_trajectories": len(jobs),
            "completed_trajectories": len(trajectory_rows),
            "analysis_eligible_trajectories": sum(
                int(row["analysis_eligible"]) for row in trajectory_rows
            ),
            "trajectories_with_invalid_nonrestraints": sum(
                int(row["operationally_eligible"] and row["invalid_count"] > 0)
                for row in trajectory_rows
            ),
            "operational_failure_trajectories": sum(
                int(not row["operationally_eligible"]) for row in trajectory_rows
            ),
            "scheduled_agent_days": sum(
                int(row["scheduled_agent_days"]) for row in trajectory_rows
            ),
            "responses_received": sum(
                int(row["responses_received"]) for row in trajectory_rows
            ),
            "invalid_count": sum(int(row["invalid_count"]) for row in trajectory_rows),
            "identity_mismatch_count": sum(
                int(row["identity_mismatch_count"]) for row in trajectory_rows
            ),
            "transport_failure_count": sum(
                int(row["transport_failure_count"]) for row in trajectory_rows
            ),
            "main_effect_test_count": len(effect_rows),
            "analysis_status": analysis_status,
            "physical_attempts_reserved": len(attempt_budget.records),
        }
        manifest["complete"] = (
            len(trajectory_rows) == len(jobs)
            and manifest["summary"]["identity_mismatch_count"] == 0
            and manifest["summary"]["transport_failure_count"] == 0
        )
        manifest["summary"]["confirmatory_analysis_complete"] = (
            analysis_status == "complete_future_confirmatory"
            and len(effect_rows) == 30
        )
        manifest["journals"] = _journal_references(journals)
        manifest["attempt_ledger"] = attempt_budget.reference()
        manifest["sanitized_artifacts"] = {
            "trajectory_metrics": {
                "path": str(trajectory_path.resolve()),
                "file_sha256": _sha256_file(trajectory_path),
                "evidence_sha256": trajectory_payload["evidence_sha256"],
            },
            "sentinel_cell_metrics": {
                "path": str(model_path.resolve()),
                "file_sha256": _sha256_file(model_path),
                "evidence_sha256": model_payload["evidence_sha256"],
            },
            "main_effects": {
                "path": str(effects_path.resolve()),
                "file_sha256": _sha256_file(effects_path),
                "evidence_sha256": effects_payload["evidence_sha256"],
            },
            "call_order_diagnostic": {
                "path": str(call_order_path.resolve()),
                "file_sha256": _sha256_file(call_order_path),
                "evidence_sha256": call_order_payload["evidence_sha256"],
            },
        }
        manifest["last_updated_at_utc"] = _utc_now()
        if manifest["complete"]:
            manifest["completed_at_utc"] = _utc_now()
        _seal(manifest)
        _atomic_json(manifest_path, manifest)
        return manifest
    finally:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_UN)
        run_lock.close()


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be nonnegative") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the provider-safe six-sentinel Part 2 resolution-V "
            "sensitivity campaign v1."
        )
    )
    parser.add_argument("--panel-config", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--compatibility", type=Path, default=DEFAULT_COMPATIBILITY)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--cell", action="append", dest="cells")
    parser.add_argument("--trajectory-limit", type=_positive_int, default=None)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument(
        "--campaign-workers", type=_positive_int, default=DEFAULT_CAMPAIGN_WORKERS
    )
    parser.add_argument(
        "--participant-workers",
        type=_positive_int,
        default=DEFAULT_PARTICIPANT_WORKERS,
    )
    parser.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--initial-backoff-seconds",
        type=_nonnegative_float,
        default=DEFAULT_BACKOFF_SECONDS,
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--development-subset", action="store_true",
        help="Allow an explicitly non-confirmatory subset; no p-values are emitted.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    client = _provider_safe_client(args.timeout_seconds)
    manifest = run_sensitivity_campaign(
        panel_path=args.panel_config,
        design_path=args.design,
        compatibility_path=args.compatibility,
        registry_path=args.registry,
        output_dir=args.output_dir,
        client=client,
        selected_ids=args.targets,
        selected_cell_ids=args.cells,
        trajectory_limit=args.trajectory_limit,
        base_seed=args.base_seed,
        campaign_workers=args.campaign_workers,
        participant_workers=args.participant_workers,
        max_attempts=args.max_attempts,
        initial_backoff_seconds=args.initial_backoff_seconds,
        resume=args.resume,
        development_subset=args.development_subset,
    )
    print(
        f"Completed {manifest['summary']['completed_trajectories']}/"
        f"{manifest['summary']['planned_trajectories']} Part 2 sensitivity trajectories."
    )
    print(f"Private evidence: {args.output_dir / 'private' / 'manifest.json'}")
    print(
        "Sanitized metrics: "
        f"{args.output_dir / 'sanitized' / 'main_effects.json'}"
    )
    return 0 if manifest["complete"] else 1


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (
        InferenceHubPart2SensitivityError,
        InferenceHubPart2PanelError,
        InferenceHubDiscoveryError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"InferenceHub Part 2 sensitivity v1 failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
