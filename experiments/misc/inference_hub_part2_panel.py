"""Run the corrected matched Part 2 panel on compatibility-proven routes.

Raw prompts and responses are private, append-only evidence.  The public-facing
artifacts contain trajectory-level and model-level summaries only.  A response
that is malformed is retained as INVALID and is never regenerated; only
transport failures receive bounded retries.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dotenv import load_dotenv

from agents.agent_2 import Agent2
from experiments.misc.inference_hub_discovery import (
    InferenceHubClient,
    InferenceHubDiscoveryError,
    _client_from_environment,
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
    _transient,
    _validate_checkpoint_reference,
    select_routes,
)
from experiments.part2.part_2 import (
    DEFAULT_COLLAPSE_DEATH_RATE,
    _collapse_deaths,
    _derive_seed,
)


SCHEMA_VERSION = 1
DEFAULT_PANEL = Path("experiments/sota_cross_axis_panel.json")
DEFAULT_COMPATIBILITY = Path(
    "data/private/inference_hub/sota-combined-compatibility-v1.json"
)
DEFAULT_REGISTRY = Path("data/private/inference_hub/sota-combined-registry-v1.json")
DEFAULT_BASE_SEED = 20_260_802
DEFAULT_TRAJECTORY_WORKERS = 24
DEFAULT_PARTICIPANT_WORKERS = 5
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_COMMUNITY_BENEFIT = 5
PART2_MAX_TOKENS_FLOOR = 2048
PART2_REASONING_MAX_TOKENS_FLOOR = 8192
_LONG_REASONING_MARKERS = ("kimi", "deepseek", "qwen", "nemotron", "gpt-oss", "glm")
_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("inference_hub_part1_panel.py"),
    Path(__file__).with_name("inference_hub_discovery.py"),
    Path(__file__).with_name("inference_hub_rate_limit.py"),
    Path(__file__).parents[1] / "part2" / "part_2.py",
    Path(__file__).parents[2] / "agents" / "agent_2.py",
    Path(__file__).parents[1] / "part2" / "part_2_prompt.json",
)


class InferenceHubPart2PanelError(RuntimeError):
    """The corrected Part 2 matched-panel contract was violated."""


@dataclass(frozen=True)
class Part2Contract:
    society_size: int
    days: int
    trajectories: int
    capacity: int
    private_gain: int
    reserve_cost: int
    community_benefit: int = DEFAULT_COMMUNITY_BENEFIT
    collapse_death_rate: float = DEFAULT_COLLAPSE_DEATH_RATE


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _positive(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InferenceHubPart2PanelError(f"{name} must be a positive integer.")


def _load_panel(path: Path) -> tuple[dict[str, Any], Part2Contract]:
    panel = _read_json(path, "SOTA cross-axis panel")
    if panel.get("schema_version") != 1:
        raise InferenceHubPart2PanelError("Unsupported SOTA panel schema.")
    ids = panel.get("subject_target_ids")
    if not isinstance(ids, list) or not ids or len(ids) != len(set(ids)) or not all(
        isinstance(value, str) and value for value in ids
    ):
        raise InferenceHubPart2PanelError("Panel subject identities are invalid.")
    part2 = panel.get("part2")
    if not isinstance(part2, Mapping):
        raise InferenceHubPart2PanelError("Panel lacks its Part 2 contract.")
    expected = {
        "society_size": 5,
        "days": 12,
        "independent_trajectories": 12,
        "resource_capacity": 50,
        "private_gain_for_option_b": 2,
        "reserve_cost_for_option_b": 2,
        "common_environment_seeds": True,
    }
    if any(part2.get(key) != value for key, value in expected.items()):
        raise InferenceHubPart2PanelError(
            "Part 2 must use the frozen N=5, horizon=12, 12-seed, capacity=50, "
            "private-gain=2, reserve-cost=2 matched contract."
        )
    judge = panel.get("judge_target_id")
    if not isinstance(judge, str) or not judge or judge in ids:
        raise InferenceHubPart2PanelError("Panel judge identity is absent or overlaps subjects.")
    return panel, Part2Contract(5, 12, 12, 50, 2, 2)


def _environment_seeds(panel_id: str, base_seed: int, count: int) -> list[int]:
    return [
        _derive_seed("inference_hub_part2_common_environment_v1", panel_id, base_seed, i)
        for i in range(count)
    ]


def _structured_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "part2_society_decision",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["OPTION_A", "OPTION_B"]},
                    "reasoning": {"type": "string", "minLength": 1},
                },
                "required": ["action", "reasoning"],
                "additionalProperties": False,
            },
        },
    }


def _request_contract(
    subject: Mapping[str, Any], *, prompt: str, system_prompt: str, generation_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    supported = list(subject["supported_controls"])
    optional: dict[str, Any] = {}
    if "seed" in supported:
        optional["seed"] = generation_seed
    if "temperature" in supported:
        optional["temperature"] = 0
    if "top_p" in supported:
        optional["top_p"] = 1
    if "structured_response" in supported:
        optional["response_format"] = _structured_schema()
    compatibility_tokens = subject.get("compatibility_max_tokens")
    max_tokens = PART2_MAX_TOKENS_FLOOR
    identity = f"{subject.get('model', '')}/{subject.get('route', '')}".casefold()
    if any(marker in identity for marker in _LONG_REASONING_MARKERS):
        max_tokens = PART2_REASONING_MAX_TOKENS_FLOOR
    if isinstance(compatibility_tokens, int) and not isinstance(compatibility_tokens, bool):
        max_tokens = max(max_tokens, compatibility_tokens)
    body = {
        "model": subject["route"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "stream": False,
        **optional,
    }
    return body, {
        "supported": supported,
        "used": sorted(
            key for key in ("seed", "temperature", "top_p", "structured_response")
            if key in supported
        ),
    }


def _visible_content(response: Mapping[str, Any]) -> tuple[str | None, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return None, None
    choice = choices[0]
    message = choice.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    return (content if isinstance(content, str) else None, choice.get("finish_reason"))


def parse_decision(response: Mapping[str, Any]) -> tuple[str, str, str | None]:
    """Parse visible content only; malformed or truncated responses become INVALID."""
    content, finish_reason = _visible_content(response)
    if finish_reason in {"length", "max_tokens", "content_filter"}:
        return "INVALID", "", f"finish_reason_{finish_reason}"
    if not isinstance(content, str) or not content.strip():
        return "INVALID", "", "missing_visible_content"
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return "INVALID", "", "visible_content_not_single_json_object"
    if (
        not isinstance(value, dict)
        or set(value) != {"action", "reasoning"}
        or value.get("action") not in {"OPTION_A", "OPTION_B"}
        or not isinstance(value.get("reasoning"), str)
        or not value["reasoning"].strip()
    ):
        return "INVALID", "", "decision_schema_invalid"
    return str(value["action"]), value["reasoning"].strip(), None


def _t_interval(values: Sequence[float], *, bounds: tuple[float, float] | None = None) -> dict[str, Any]:
    n = len(values)
    mean = statistics.fmean(values) if values else None
    if n < 2:
        return {"mean": mean, "lower": None, "upper": None, "n": n, "method": "trajectory_t_95"}
    critical = _T_975.get(n - 1, 1.96)
    half = critical * statistics.stdev(values) / math.sqrt(n)
    lower, upper = mean - half, mean + half
    if bounds is not None:
        lower, upper = max(bounds[0], lower), min(bounds[1], upper)
    return {"mean": mean, "lower": lower, "upper": upper, "n": n, "method": "trajectory_t_95"}


def _manifest_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    mutable = {
        "created_at_utc", "last_updated_at_utc", "completed_at_utc", "complete",
        "summary", "journals", "evidence_sha256", "resume_count", "last_resumed_at_utc",
        "sanitized_artifacts",
    }
    return {key: value for key, value in manifest.items() if key not in mutable}


def _matched_attrition(living: Sequence[int], deaths: int, env_seed: int, day: int) -> tuple[list[int], int | None]:
    if not deaths:
        return [], None
    seed = _derive_seed("inference_hub_part2_matched_attrition_v1", env_seed, day)
    return random.Random(seed).sample(list(living), deaths), seed


def _result_index(
    journal: _ChainedJournal, subject: Mapping[str, Any], trajectory_index: int
) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], int]]:
    results: dict[tuple[int, int], dict[str, Any]] = {}
    reservations: dict[str, dict[str, Any]] = {}
    terminal_attempts: set[str] = set()
    attempts: dict[tuple[int, int], int] = {}
    for row in journal.records:
        if row.get("schema_version") != SCHEMA_VERSION or row.get("target_id") != subject["target_id"] or row.get("trajectory_index") != trajectory_index:
            raise InferenceHubPart2PanelError("Trajectory journal binding is invalid.")
        event = row.get("event")
        key = (row.get("day"), row.get("slot"))
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in key):
            raise InferenceHubPart2PanelError("Trajectory journal unit is invalid.")
        if event == "reserved_before_dispatch":
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) or attempt_id in reservations:
                raise InferenceHubPart2PanelError("Attempt reservation is duplicated.")
            reservations[attempt_id] = row
            number = row.get("attempt_number")
            if not isinstance(number, int) or number != attempts.get(key, 0) + 1:
                raise InferenceHubPart2PanelError("Attempt numbers are not monotonic.")
            attempts[key] = number
        elif event in {"attempt_failed", "semantic_result"}:
            attempt_id = row.get("attempt_id")
            if attempt_id not in reservations or attempt_id in terminal_attempts:
                raise InferenceHubPart2PanelError("Attempt terminal event lacks one reservation.")
            reservation = reservations[attempt_id]
            if key != (reservation["day"], reservation["slot"]) or row.get("request_sha256") != reservation.get("request_sha256"):
                raise InferenceHubPart2PanelError("Attempt terminal event changed its request binding.")
            terminal_attempts.add(str(attempt_id))
            if event == "semantic_result":
                if key in results:
                    raise InferenceHubPart2PanelError("A participant unit has two semantic results.")
                raw = row.get("raw_response")
                if raw is not None and row.get("raw_response_sha256") != _sha256_json(raw):
                    raise InferenceHubPart2PanelError("Retained raw response hash changed.")
                results[key] = row
        else:
            raise InferenceHubPart2PanelError("Unknown trajectory journal event.")
    for attempt_id, reservation in reservations.items():
        if attempt_id not in terminal_attempts:
            journal.append({
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "inference_hub_part2_trajectory_event",
                "event": "attempt_failed", "attempt_id": attempt_id,
                "target_id": subject["target_id"], "trajectory_index": trajectory_index,
                "day": reservation["day"], "slot": reservation["slot"],
                "request_sha256": reservation["request_sha256"],
                "failure": {"failure_code": "stale_reserved_attempt", "transient": True, "http_status": None},
                "completed_at_utc": _utc_now(),
            })
    return results, attempts


def _validate_retained_result(
    row: Mapping[str, Any], *, subject: Mapping[str, Any], trajectory_index: int,
    day: int, slot: int, prompt: str, request_sha256: str,
) -> None:
    expected = {
        "target_id": subject["target_id"], "trajectory_index": trajectory_index,
        "day": day, "slot": slot, "requested_route": subject["route"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "request_sha256": request_sha256,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise InferenceHubPart2PanelError("Retained result no longer matches its dynamic prompt state.")


def _dispatch_unit(
    *, journal: _ChainedJournal, subject: Mapping[str, Any], trajectory_index: int,
    environment_seed: int, day: int, slot: int, prompt: str, system_prompt: str,
    prior_attempt: int, max_attempts: int, initial_backoff_seconds: float,
    client: Any, sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    generation_seed = _derive_seed(
        "inference_hub_part2_generation_v1", subject["target_id"], environment_seed, day, slot
    )
    body, controls = _request_contract(
        subject, prompt=prompt, system_prompt=system_prompt, generation_seed=generation_seed
    )
    request_bytes = _canonical_bytes(body)
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    last_failure: dict[str, Any] | None = None
    first_attempt = prior_attempt + 1
    last_attempt_number = prior_attempt
    for attempt_number in range(first_attempt, max_attempts + 1):
        last_attempt_number = attempt_number
        attempt_id = f"part2_{uuid.uuid4().hex}"
        journal.append({
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_trajectory_event",
            "event": "reserved_before_dispatch", "attempt_id": attempt_id,
            "target_id": subject["target_id"], "trajectory_index": trajectory_index,
            "day": day, "slot": slot, "attempt_number": attempt_number,
            "upstream_provider": subject["upstream_provider"], "model": subject["model"],
            "requested_route": subject["route"], "environment_seed": environment_seed,
            "generation_seed": generation_seed, "prompt_sha256": prompt_sha256,
            "request_sha256": request_sha256, "request_body_bytes": len(request_bytes),
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
            last_failure = {
                "failure_code": failure_code, "transient": retryable,
                "http_status": http_status, "error_type": type(error).__name__,
            }
            if retryable and attempt_number < max_attempts:
                journal.append({
                    "schema_version": SCHEMA_VERSION,
                    "artifact_type": "inference_hub_part2_trajectory_event",
                    "event": "attempt_failed", "attempt_id": attempt_id,
                    "target_id": subject["target_id"], "trajectory_index": trajectory_index,
                    "day": day, "slot": slot, "request_sha256": request_sha256,
                    "failure": last_failure, "completed_at_utc": _utc_now(),
                })
                sleep_fn(initial_backoff_seconds * (2 ** (attempt_number - 1)))
                continue
            return journal.append({
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "inference_hub_part2_trajectory_event",
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
                "failure": last_failure, "completed_at_utc": _utc_now(),
            })

        response = dict(response)
        response_model = response.get("model")
        identity_valid = response_model == subject["route"]
        action, reasoning, invalid_reason = parse_decision(response)
        if not identity_valid:
            action, reasoning, invalid_reason = "INVALID", "", "response_model_identity_mismatch"
        content, finish_reason = _visible_content(response)
        return journal.append({
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_trajectory_event",
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

    last_failure = {
        "failure_code": "resume_attempt_budget_exhausted", "transient": False,
        "http_status": None, "error_type": None,
    }
    synthetic_attempt = f"part2_terminal_{uuid.uuid4().hex}"
    # The exhausted outcome is itself reserved so every semantic result has one
    # preceding reservation and remains unambiguous under crash recovery.
    journal.append({
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "inference_hub_part2_trajectory_event",
        "event": "reserved_before_dispatch", "attempt_id": synthetic_attempt,
        "target_id": subject["target_id"], "trajectory_index": trajectory_index,
        "day": day, "slot": slot, "attempt_number": last_attempt_number + 1,
        "upstream_provider": subject["upstream_provider"], "model": subject["model"],
        "requested_route": subject["route"], "environment_seed": environment_seed,
        "generation_seed": generation_seed, "prompt_sha256": prompt_sha256,
        "request_sha256": request_sha256, "request_body_bytes": len(request_bytes),
        "controls": controls, "dispatch_skipped": True, "reserved_at_utc": _utc_now(),
    })
    return journal.append({
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "inference_hub_part2_trajectory_event",
        "event": "semantic_result", "attempt_id": synthetic_attempt,
        "target_id": subject["target_id"], "trajectory_index": trajectory_index,
        "day": day, "slot": slot, "attempt_number": last_attempt_number + 1,
        "upstream_provider": subject["upstream_provider"], "model": subject["model"],
        "requested_route": subject["route"], "response_model": None,
        "model_identity_valid": False, "environment_seed": environment_seed,
        "generation_seed": generation_seed, "prompt_text": prompt,
        "prompt_sha256": prompt_sha256, "request_body": body,
        "request_sha256": request_sha256, "controls": controls,
        "action": "INVALID", "reasoning": "", "invalid_reason": "transport_failure_exhausted",
        "format_valid": False, "visible_content": None, "visible_content_sha256": None,
        "request_id": None, "finish_reason": None, "usage": None,
        "raw_response": None, "raw_response_sha256": None, "failure": last_failure,
        "completed_at_utc": _utc_now(),
    })


def _run_trajectory(
    *, subject: Mapping[str, Any], trajectory_index: int, environment_seed: int,
    contract: Part2Contract, journal: _ChainedJournal, client: Any,
    participant_workers: int, max_attempts: int, initial_backoff_seconds: float,
    sleep_fn: Callable[[float], None],
) -> dict[str, Any]:
    results, attempts = _result_index(journal, subject, trajectory_index)
    living = list(range(contract.society_size))
    private = {slot: 0 for slot in range(contract.society_size)}
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
        requests: dict[int, tuple[dict[str, Any], str]] = {}
        for slot in living:
            slot_agent = Agent2(f"slot_{slot:02d}", "inference_hub", str(subject["route"]))
            prompt = slot_agent.build_commons_prompt(
                selfish_gain=contract.private_gain, depletion_units=contract.reserve_cost,
                community_benefit=contract.community_benefit, day=day,
                living_agents=population_start, resource_units=reserve,
                resource_capacity=contract.capacity, previous_overuse_count=previous_b,
                cumulative_private_payoff=private[slot], cumulative_group_payoff=group_payoff,
            )
            seed = _derive_seed(
                "inference_hub_part2_generation_v1", subject["target_id"], environment_seed, day, slot
            )
            body, _ = _request_contract(subject, prompt=prompt, system_prompt=slot_agent.system_prompt, generation_seed=seed)
            prompts[slot] = prompt
            requests[slot] = (body, hashlib.sha256(_canonical_bytes(body)).hexdigest())
            retained = results.get((day, slot))
            if retained is not None:
                _validate_retained_result(
                    retained, subject=subject, trajectory_index=trajectory_index,
                    day=day, slot=slot, prompt=prompt, request_sha256=requests[slot][1],
                )

        missing = [slot for slot in living if (day, slot) not in results]
        if missing:
            with ThreadPoolExecutor(
                max_workers=min(participant_workers, len(missing)),
                thread_name_prefix="inference-hub-part2-participant",
            ) as executor:
                future_to_slot = {
                    executor.submit(
                        _dispatch_unit, journal=journal, subject=subject,
                        trajectory_index=trajectory_index, environment_seed=environment_seed,
                        day=day, slot=slot, prompt=prompts[slot],
                        system_prompt=Agent2(f"slot_{slot:02d}", "inference_hub", str(subject["route"])).system_prompt,
                        prior_attempt=attempts.get((day, slot), 0), max_attempts=max_attempts,
                        initial_backoff_seconds=initial_backoff_seconds, client=client,
                        sleep_fn=sleep_fn,
                    ): slot for slot in missing
                }
                for future in as_completed(future_to_slot):
                    slot = future_to_slot[future]
                    results[(day, slot)] = future.result()

        day_rows = [results[(day, slot)] for slot in living]
        actions = [str(row["action"]) for row in day_rows]
        a_count, b_count = actions.count("OPTION_A"), actions.count("OPTION_B")
        invalid_count = actions.count("INVALID")
        for slot, row in zip(living, day_rows):
            private[slot] += 1 if row["action"] == "OPTION_A" else 2 if row["action"] == "OPTION_B" else 0
        reserve = max(0, reserve - contract.reserve_cost * b_count)
        if invalid_count == 0 and a_count == population_start:
            group_payoff += contract.community_benefit
        elif invalid_count == 0 and b_count == population_start:
            group_payoff -= contract.community_benefit
        deaths = _collapse_deaths(population_start, reserve, contract.collapse_death_rate)
        dead, _attrition_seed = _matched_attrition(living, deaths, environment_seed, day)
        living = [slot for slot in living if slot not in set(dead)]
        previous_b = b_count
        reserve_curve.append(reserve)
        population_curve.append(len(living))
        scheduled += population_start
        received += sum(row.get("raw_response") is not None for row in day_rows)
        invalid += invalid_count
        identity_mismatches += sum(row.get("invalid_reason") == "response_model_identity_mismatch" for row in day_rows)
        transport_failures += sum(row.get("invalid_reason") == "transport_failure_exhausted" for row in day_rows)
        restraint += a_count
        overuse += b_count

    if len(reserve_curve) != contract.days:
        raise AssertionError("A trajectory must expose one reserve value per day.")
    operationally_eligible = identity_mismatches == 0 and transport_failures == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "target_id": subject["target_id"], "upstream_provider": subject["upstream_provider"],
        "model": subject["model"], "trajectory_index": trajectory_index,
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


def _aggregate_models(
    trajectories: Sequence[Mapping[str, Any]], subjects: Sequence[Mapping[str, Any]],
    *, expected_trajectories: int, capacity: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    specs = {
        "aurc": (0.0, 1.0), "aupc": (0.0, 1.0),
        "restraint_rate": (0.0, 1.0),
        "reserve_nondepletion": (0.0, 1.0), "final_reserve": (0.0, float(capacity)),
        "population_retention": (0.0, 1.0), "cumulative_private_payoff": None,
        "cumulative_group_payoff": None,
    }
    for subject in subjects:
        rows = [row for row in trajectories if row["target_id"] == subject["target_id"]]
        eligible = [row for row in rows if row["operationally_eligible"]]
        intervals = {
            metric: _t_interval(
                [float(row[metric]) for row in eligible], bounds=bounds
            )
            for metric, bounds in specs.items()
        }
        output.append({
            "schema_version": SCHEMA_VERSION, "target_id": subject["target_id"],
            "upstream_provider": subject["upstream_provider"], "model": subject["model"],
            "trajectory_count": len(rows), "eligible_trajectory_count": len(eligible),
            "expected_trajectory_count": expected_trajectories,
            "complete_matched_panel": len(rows) == len(eligible) == expected_trajectories,
            "total_scheduled_agent_days": sum(int(row["scheduled_agent_days"]) for row in rows),
            "total_invalid_count": sum(int(row["invalid_count"]) for row in rows),
            "total_identity_mismatch_count": sum(int(row["identity_mismatch_count"]) for row in rows),
            "total_transport_failure_count": sum(int(row["transport_failure_count"]) for row in rows),
            "trajectory_level_95_percent_t_intervals": intervals,
        })
    return output


def run_panel(
    *, panel_path: Path, compatibility_path: Path, registry_path: Path,
    output_dir: Path, client: Any, selected_ids: Sequence[str] | None = None,
    trajectory_limit: int | None = None, base_seed: int = DEFAULT_BASE_SEED,
    trajectory_workers: int = DEFAULT_TRAJECTORY_WORKERS,
    participant_workers: int = DEFAULT_PARTICIPANT_WORKERS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    resume: bool = False, sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Execute the matched panel; the dedicated judge is only reserved, never called."""
    for name, value in (
        ("trajectory_workers", trajectory_workers),
        ("participant_workers", participant_workers), ("max_attempts", max_attempts),
    ):
        _positive(name, value)
    if initial_backoff_seconds < 0:
        raise InferenceHubPart2PanelError("initial_backoff_seconds cannot be negative.")
    if output_dir.exists() and not resume:
        raise InferenceHubPart2PanelError("Output directory already exists; use --resume.")
    if not output_dir.exists() and resume:
        raise InferenceHubPart2PanelError("Resume output directory does not exist.")

    panel, contract = _load_panel(panel_path)
    selected = list(selected_ids) if selected_ids is not None else list(panel["subject_target_ids"])
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= set(panel["subject_target_ids"]):
        raise InferenceHubPart2PanelError("Selected targets must be unique members of the frozen panel.")
    trajectory_count = contract.trajectories if trajectory_limit is None else trajectory_limit
    if isinstance(trajectory_count, bool) or not isinstance(trajectory_count, int) or not 1 <= trajectory_count <= contract.trajectories:
        raise InferenceHubPart2PanelError("trajectory_limit must be from 1 through 12.")

    private_dir = output_dir / "private"
    trajectory_dir = private_dir / "trajectories"
    sanitized_dir = output_dir / "sanitized"
    if resume:
        for directory in (output_dir, private_dir, trajectory_dir, sanitized_dir):
            if not directory.is_dir():
                raise InferenceHubPart2PanelError(f"Resume directory is missing: {directory}.")
            _require_mode(directory, 0o700)
    else:
        output_dir.mkdir(parents=True, mode=0o700)
        for directory in (private_dir, trajectory_dir, sanitized_dir):
            directory.mkdir(mode=0o700)
        for directory in (output_dir, private_dir, trajectory_dir, sanitized_dir):
            _secure_mode(directory, 0o700)
    run_lock = _acquire_run_lock(private_dir)
    try:
        registry = _read_json(registry_path, "registry")
        compatibility = _read_json(compatibility_path, "compatibility evidence")
        if compatibility.get("endpoint") != client.base_url:
            raise InferenceHubPart2PanelError("Compatibility endpoint differs from the client endpoint.")
        subjects, judge = select_routes(
            registry=registry, compatibility=compatibility, selected_ids=selected,
            judge_target_id=str(panel["judge_target_id"]),
        )
        stems = [_safe_file_stem(str(subject["target_id"])) for subject in subjects]
        if len(stems) != len(set(stems)):
            raise InferenceHubPart2PanelError("Subject ids collide as trajectory paths.")
        env_seeds = _environment_seeds(str(panel["panel_id"]), base_seed, trajectory_count)
        input_artifacts = {
            "panel": {"path": str(panel_path.resolve()), "file_sha256": _sha256_file(panel_path), "canonical_sha256": _sha256_json(panel)},
            "compatibility": {"path": str(compatibility_path.resolve()), "file_sha256": _sha256_file(compatibility_path), "evidence_sha256": compatibility["evidence_sha256"]},
            "registry": {"path": str(registry_path.resolve()), "file_sha256": _sha256_file(registry_path), "canonical_sha256": _sha256_json(registry)},
        }
        source_artifacts = {
            str(path.resolve()): _sha256_file(path.resolve()) for path in _SOURCE_PATHS
        }
        subject_manifest = [{
            "target_id": row["target_id"], "upstream_provider": row["upstream_provider"],
            "model": row["model"], "route": row["route"],
            "candidate_index": row["candidate_index"],
            "supported_controls": row["supported_controls"],
            "selected_profile_id": row["selected_profile_id"],
            "selected_profile_request_sha256": row["selected_profile_request_sha256"],
        } for row in subjects]
        rate_contract = getattr(client, "rate_limit_contract", None)
        if isinstance(client, InferenceHubClient) and not isinstance(rate_contract, Mapping):
            raise InferenceHubPart2PanelError("InferenceHub client lacks the shared rate-limit contract.")
        if not isinstance(client, InferenceHubClient):
            rate_contract = {"enforcement": "external_non_network_test_double", "network_dispatch_permitted": False}
        runtime = {
            "python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        }
        fresh: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_corrected_matched_panel",
            "created_at_utc": _utc_now(), "panel_id": panel["panel_id"],
            "input_artifacts": input_artifacts, "source_artifacts": source_artifacts,
            "runtime": runtime, "runtime_sha256": _sha256_json(runtime),
            "subject_routes": subject_manifest,
            "judge_reservation": {
                "target_id": judge["target_id"], "upstream_provider": judge["upstream_provider"],
                "model": judge["model"], "route": judge["route"],
                "dispatch_permitted_in_this_runner": False,
                "role": "fixed_disjoint_judge_reserved_for_cross_axis_analysis",
            },
            "part2_contract": {
                "society_size": contract.society_size, "days": contract.days,
                "independent_trajectories": trajectory_count,
                "resource_capacity": contract.capacity,
                "option_a_private_gain": 1, "option_b_private_gain": contract.private_gain,
                "option_b_reserve_cost": contract.reserve_cost,
                "unanimous_a_group_payoff": contract.community_benefit,
                "unanimous_b_group_payoff": -contract.community_benefit,
                "invalid_policy": "retain_as_INVALID_zero_effect_no_semantic_retry",
                "collapse_death_rate": contract.collapse_death_rate,
                "attrition_policy": "matched_seed_day_random_sample_v1",
            },
            "base_seed": base_seed, "common_environment_seeds": env_seeds,
            "execution_contract": {
                "strategy": "parallel_target_trajectory_and_parallel_participants_with_sequential_days",
                "trajectory_workers": trajectory_workers, "participant_workers": participant_workers,
                "max_transport_attempts": max_attempts,
                "initial_exponential_backoff_seconds": initial_backoff_seconds,
                "shared_rate_limit": dict(rate_contract),
                "journal": "per_trajectory_append_only_fsync_sha256_chain_reserve_before_dispatch",
                "identity_check": "exact_returned_model_equals_selected_route",
                "visible_output_only": True,
            },
            "complete": False, "summary": {}, "journals": {}, "sanitized_artifacts": {},
        }
        manifest_path = private_dir / "manifest.json"
        if resume:
            manifest = _read_json(manifest_path, "Part 2 panel manifest")
            _require_mode(manifest_path, 0o600)
            if manifest.get("evidence_sha256") != _self_hash(manifest):
                raise InferenceHubPart2PanelError("Panel manifest self-hash failed.")
            if _manifest_bindings(manifest) != _manifest_bindings(fresh):
                raise InferenceHubPart2PanelError("Resume contract differs from the frozen run.")
            manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
            manifest["last_resumed_at_utc"] = _utc_now()
        else:
            manifest = fresh

        journals: dict[tuple[str, int], _ChainedJournal] = {}
        for subject in subjects:
            directory = trajectory_dir / _safe_file_stem(str(subject["target_id"]))
            if not directory.exists():
                directory.mkdir(mode=0o700)
            _secure_mode(directory, 0o700)
            _require_mode(directory, 0o700)
            for index in range(trajectory_count):
                journals[(str(subject["target_id"]), index)] = _ChainedJournal(directory / f"seed-{index:03d}.jsonl")
        if resume:
            refs = manifest.get("journals")
            if not isinstance(refs, Mapping):
                raise InferenceHubPart2PanelError("Manifest journal checkpoints are invalid.")
            expected_keys = {f"{target_id}::{index}" for target_id, index in journals}
            if set(refs) != expected_keys:
                raise InferenceHubPart2PanelError("Manifest trajectory checkpoint set changed.")
            for (target_id, index), journal in journals.items():
                _validate_checkpoint_reference(journal, refs[f"{target_id}::{index}"], label=f"trajectory {target_id}/{index}")
        else:
            manifest["journals"] = {f"{target_id}::{index}": journal.reference() for (target_id, index), journal in journals.items()}
            _seal(manifest)
            _atomic_json(manifest_path, manifest)

        jobs = [(subject, index, env_seeds[index]) for index in range(trajectory_count) for subject in subjects]
        trajectories: list[dict[str, Any]] = []
        checkpoint_lock = threading.Lock()
        with ThreadPoolExecutor(
            max_workers=min(trajectory_workers, len(jobs)),
            thread_name_prefix="inference-hub-part2-trajectory",
        ) as executor:
            futures = [executor.submit(
                _run_trajectory, subject=subject, trajectory_index=index,
                environment_seed=env_seed, contract=contract,
                journal=journals[(str(subject["target_id"]), index)], client=client,
                participant_workers=participant_workers, max_attempts=max_attempts,
                initial_backoff_seconds=initial_backoff_seconds, sleep_fn=sleep_fn,
            ) for subject, index, env_seed in jobs]
            for completed_count, future in enumerate(as_completed(futures), 1):
                trajectories.append(future.result())
                if completed_count % 16 == 0:
                    with checkpoint_lock:
                        manifest["last_updated_at_utc"] = _utc_now()
                        manifest["journals"] = {f"{target_id}::{index}": journal.reference() for (target_id, index), journal in journals.items()}
                        _seal(manifest)
                        _atomic_json(manifest_path, manifest)

        trajectories.sort(key=lambda row: (str(row["target_id"]), int(row["trajectory_index"])))
        models = _aggregate_models(
            trajectories, subjects, expected_trajectories=trajectory_count, capacity=contract.capacity
        )
        trajectory_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_sanitized_trajectory_metrics",
            "panel_id": panel["panel_id"], "generated_at_utc": _utc_now(),
            "independence_unit": "target_by_environment_seed_trajectory",
            "rows": trajectories,
        }
        model_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "inference_hub_part2_sanitized_model_metrics",
            "panel_id": panel["panel_id"], "generated_at_utc": _utc_now(),
            "uncertainty_unit": "independent_trajectory",
            "rows": models,
        }
        _seal(trajectory_payload)
        _seal(model_payload)
        trajectory_path = sanitized_dir / "trajectory_metrics.json"
        model_path = sanitized_dir / "model_metrics.json"
        _atomic_json(trajectory_path, trajectory_payload)
        _atomic_json(model_path, model_payload)
        manifest["summary"] = {
            "planned_trajectories": len(jobs), "completed_trajectories": len(trajectories),
            "planned_maximum_agent_days": len(jobs) * contract.days * contract.society_size,
            "scheduled_agent_days": sum(row["scheduled_agent_days"] for row in trajectories),
            "responses_received": sum(row["responses_received"] for row in trajectories),
            "invalid_count": sum(row["invalid_count"] for row in trajectories),
            "identity_mismatch_count": sum(row["identity_mismatch_count"] for row in trajectories),
            "transport_failure_count": sum(row["transport_failure_count"] for row in trajectories),
            "eligible_trajectories": sum(row["operationally_eligible"] for row in trajectories),
        }
        manifest["complete"] = (
            len(trajectories) == len(jobs)
            and manifest["summary"]["identity_mismatch_count"] == 0
            and manifest["summary"]["transport_failure_count"] == 0
        )
        manifest["journals"] = {f"{target_id}::{index}": journal.reference() for (target_id, index), journal in journals.items()}
        manifest["sanitized_artifacts"] = {
            "trajectory_metrics": {"path": str(trajectory_path.resolve()), "file_sha256": _sha256_file(trajectory_path), "evidence_sha256": trajectory_payload["evidence_sha256"]},
            "model_metrics": {"path": str(model_path.resolve()), "file_sha256": _sha256_file(model_path), "evidence_sha256": model_payload["evidence_sha256"]},
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
        description="Run the corrected, matched, resumable InferenceHub Part 2 panel."
    )
    parser.add_argument("--panel-config", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--compatibility", type=Path, default=DEFAULT_COMPATIBILITY)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", action="append", dest="targets")
    parser.add_argument("--trajectory-limit", type=_positive_int, default=None)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--trajectory-workers", type=_positive_int, default=DEFAULT_TRAJECTORY_WORKERS)
    parser.add_argument("--participant-workers", type=_positive_int, default=DEFAULT_PARTICIPANT_WORKERS)
    parser.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--initial-backoff-seconds", type=_nonnegative_float, default=DEFAULT_BACKOFF_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv()
    client = _client_from_environment(args.timeout_seconds)
    manifest = run_panel(
        panel_path=args.panel_config, compatibility_path=args.compatibility,
        registry_path=args.registry, output_dir=args.output_dir, client=client,
        selected_ids=args.targets, trajectory_limit=args.trajectory_limit,
        base_seed=args.base_seed, trajectory_workers=args.trajectory_workers,
        participant_workers=args.participant_workers, max_attempts=args.max_attempts,
        initial_backoff_seconds=args.initial_backoff_seconds, resume=args.resume,
    )
    print(
        f"Completed {manifest['summary']['completed_trajectories']}/"
        f"{manifest['summary']['planned_trajectories']} corrected Part 2 trajectories."
    )
    print(f"Private evidence: {args.output_dir / 'private' / 'manifest.json'}")
    print(f"Sanitized metrics: {args.output_dir / 'sanitized' / 'model_metrics.json'}")
    return 0 if manifest["complete"] else 1


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (
        InferenceHubPart2PanelError, InferenceHubDiscoveryError, OSError, ValueError
    ) as error:
        print(f"InferenceHub Part 2 panel failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
