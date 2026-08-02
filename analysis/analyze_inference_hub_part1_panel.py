"""Fail-closed analysis of the private hosted exploratory Part 1 panel.

The analyzer deliberately emits aggregates and cryptographic bindings only.  It
never copies prompts, response text, raw provider payloads, or reasoning fields
into the analysis artifact, and the resulting artifact is not promotable to a
confirmatory or paper result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import stat
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.misc.inference_hub_part1_panel import (
    InferenceHubPart1PanelError,
    SCHEMA_VERSION as PANEL_SCHEMA_VERSION,
    _response_metadata,
    build_draft_trials,
    parse_final_action,
    select_routes,
)
from experiments.part1.confirmatory_design import (
    COUNTERBALANCE_BY_ID,
    DOMAINS,
    GAMES,
    WELFARE_PRESERVING,
)


SCHEMA_VERSION = 1
DEFAULT_BOOTSTRAP_SEED = 20_260_804
BOOTSTRAP_REPLICATES = 5_000
EXPECTED_ROOTS = 384
EXPECTED_CELLS = 12
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class HostedPart1AnalysisError(RuntimeError):
    """Private panel evidence cannot safely support the requested analysis."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise HostedPart1AnalysisError(f"Cannot hash bound file: {path}.") from error
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HostedPart1AnalysisError(f"{label} is not readable UTF-8 JSON.") from error
    if not isinstance(value, dict):
        raise HostedPart1AnalysisError(f"{label} must be a JSON object.")
    return value


def _require_mode(path: Path, mode: int) -> None:
    if os.name == "posix":
        try:
            observed = stat.S_IMODE(path.stat().st_mode)
        except OSError as error:
            raise HostedPart1AnalysisError(f"Cannot inspect private path: {path}.") from error
        if observed != mode:
            raise HostedPart1AnalysisError(
                f"Unsafe private permissions for {path}: expected {mode:04o}, "
                f"observed {observed:04o}."
            )


def _self_hash(payload: Mapping[str, Any]) -> str:
    return _sha256_json(
        {key: value for key, value in payload.items() if key != "evidence_sha256"}
    )


def _seal(payload: dict[str, Any]) -> None:
    payload.pop("evidence_sha256", None)
    payload["evidence_sha256"] = _self_hash(payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            path.chmod(0o600)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_file_stem(target_id: str) -> str:
    value = _SAFE_NAME.sub("_", target_id).strip("._")
    if not value:
        raise HostedPart1AnalysisError("Subject id has no safe evidence filename.")
    return value


def _read_chained_journal(
    path: Path, reference: object, *, label: str
) -> list[dict[str, Any]]:
    _require_mode(path, 0o600)
    if not isinstance(reference, Mapping):
        raise HostedPart1AnalysisError(f"Manifest lacks the {label} reference.")
    if reference.get("path") != str(path.resolve()):
        raise HostedPart1AnalysisError(f"Manifest {label} path binding changed.")
    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
        lines = text.splitlines()
        if not text.endswith("\n") or any(not line for line in lines):
            raise ValueError("non-canonical JSONL framing")
        rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise HostedPart1AnalysisError(f"{label} is not canonical UTF-8 JSONL.") from error
    previous: str | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise HostedPart1AnalysisError(f"{label} record {index} is not an object.")
        recorded = row.get("record_sha256")
        unhashed = {key: value for key, value in row.items() if key != "record_sha256"}
        if (
            row.get("previous_record_sha256") != previous
            or not _hex_digest(recorded)
            or recorded != _sha256_json(unhashed)
        ):
            raise HostedPart1AnalysisError(f"{label} hash chain failed at record {index}.")
        previous = recorded
    expected_count = reference.get("record_count")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int):
        raise HostedPart1AnalysisError(f"Manifest {label} record count is invalid.")
    if (
        expected_count != len(rows)
        or reference.get("tail_record_sha256") != previous
        or reference.get("file_sha256") != hashlib.sha256(raw_bytes).hexdigest()
    ):
        raise HostedPart1AnalysisError(f"Manifest {label} file/count/tail binding failed.")
    return rows


def _validate_bound_files(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inputs = manifest.get("input_artifacts")
    sources = manifest.get("source_artifacts")
    if not isinstance(inputs, Mapping) or set(inputs) != {"compatibility", "registry"}:
        raise HostedPart1AnalysisError("Manifest input artifact bindings are incomplete.")
    if not isinstance(sources, Mapping) or not sources:
        raise HostedPart1AnalysisError("Manifest source artifact bindings are incomplete.")

    compatibility_ref = inputs["compatibility"]
    registry_ref = inputs["registry"]
    if not isinstance(compatibility_ref, Mapping) or not isinstance(registry_ref, Mapping):
        raise HostedPart1AnalysisError("Manifest input artifact reference is malformed.")
    compatibility_path = Path(str(compatibility_ref.get("path", "")))
    registry_path = Path(str(registry_ref.get("path", "")))
    if not compatibility_path.is_absolute() or not registry_path.is_absolute():
        raise HostedPart1AnalysisError("Manifest input artifact paths must be absolute.")
    if compatibility_ref.get("file_sha256") != _sha256_file(compatibility_path):
        raise HostedPart1AnalysisError("Compatibility input file hash changed.")
    if registry_ref.get("file_sha256") != _sha256_file(registry_path):
        raise HostedPart1AnalysisError("Registry input file hash changed.")
    compatibility = _read_json(compatibility_path, "compatibility input")
    registry = _read_json(registry_path, "registry input")
    if (
        compatibility_ref.get("evidence_sha256") != compatibility.get("evidence_sha256")
        or compatibility.get("evidence_sha256") != _self_hash(compatibility)
    ):
        raise HostedPart1AnalysisError("Compatibility input evidence self-hash failed.")
    if registry_ref.get("canonical_sha256") != _sha256_json(registry):
        raise HostedPart1AnalysisError("Registry input canonical hash changed.")

    normalized_sources: dict[str, str] = {}
    for raw_path, expected_hash in sources.items():
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise HostedPart1AnalysisError("Manifest source paths must be absolute.")
        if not _hex_digest(expected_hash) or expected_hash != _sha256_file(Path(raw_path)):
            raise HostedPart1AnalysisError(f"Source artifact hash changed: {raw_path}.")
        normalized_sources[raw_path] = expected_hash
    return (
        {
            "input_artifacts": {key: dict(value) for key, value in inputs.items()},
            "source_artifacts": dict(sorted(normalized_sources.items())),
        },
        registry,
        compatibility,
    )


def _validate_selected_identities(
    *,
    registry: Mapping[str, Any],
    compatibility: Mapping[str, Any],
    subjects: Mapping[str, Mapping[str, Any]],
    judge: Mapping[str, Any],
) -> None:
    try:
        selected, selected_judge = select_routes(
            registry=registry,
            compatibility=compatibility,
            selected_ids=list(subjects),
            judge_target_id=str(judge["target_id"]),
        )
    except InferenceHubPart1PanelError as error:
        raise HostedPart1AnalysisError(
            "Bound compatibility/registry identity validation failed."
        ) from error
    keys = (
        "target_id",
        "upstream_provider",
        "model",
        "route",
        "candidate_index",
        "supported_controls",
        "selected_profile_id",
        "selected_profile_request_sha256",
    )
    expected_subjects = {
        str(row["target_id"]): {key: row.get(key) for key in keys} for row in selected
    }
    observed_subjects = {
        target_id: {key: row.get(key) for key in keys} for target_id, row in subjects.items()
    }
    if observed_subjects != expected_subjects or {
        key: judge.get(key) for key in keys
    } != {key: selected_judge.get(key) for key in keys}:
        raise HostedPart1AnalysisError(
            "Manifest subject/judge identities do not replay from bound compatibility inputs."
        )


def _schedule_bindings(trials: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detailed = [
        {
            "trial_id": trial.trial_id,
            "root_id": trial.root_id,
            "game": trial.game,
            "domain": trial.domain,
            "counterbalance_id": trial.counterbalance_id,
            "prompt_sha256": trial.prompt_hash,
            "generation_settings": {
                "temperature": trial.generation_settings.temperature,
                "top_p": trial.generation_settings.top_p,
                "max_output_tokens": trial.generation_settings.max_output_tokens,
                "generation_seed": trial.generation_settings.generation_seed,
                "seed_base": trial.generation_settings.seed_base,
                "seed_derivation": trial.generation_settings.seed_derivation,
            },
        }
        for trial in trials
    ]
    compact = [
        {
            "trial_id": trial.trial_id,
            "root_id": trial.root_id,
            "prompt_sha256": trial.prompt_hash,
        }
        for trial in trials
    ]
    return detailed, compact


def _subject_routes(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("subject_routes")
    if not isinstance(rows, list) or not rows:
        raise HostedPart1AnalysisError("Manifest has no subject routes.")
    result: dict[str, dict[str, Any]] = {}
    identities: set[tuple[str, str]] = set()
    routes: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise HostedPart1AnalysisError("Manifest subject route is malformed.")
        target_id = row.get("target_id")
        provider = row.get("upstream_provider")
        model = row.get("model")
        route = row.get("route")
        if not all(isinstance(value, str) and value for value in (target_id, provider, model, route)):
            raise HostedPart1AnalysisError("Manifest subject identity is incomplete.")
        identity = (provider.casefold(), model.casefold())
        if target_id in result or identity in identities or route.casefold() in routes:
            raise HostedPart1AnalysisError("Manifest subject identities/routes are repeated.")
        result[target_id] = dict(row)
        identities.add(identity)
        routes.add(route.casefold())
    return result


def _validate_judge(manifest: Mapping[str, Any], subjects: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    judge = manifest.get("judge_reservation")
    if not isinstance(judge, Mapping):
        raise HostedPart1AnalysisError("Manifest judge reservation is missing.")
    required = ("target_id", "upstream_provider", "model", "route")
    if not all(isinstance(judge.get(key), str) and judge.get(key) for key in required):
        raise HostedPart1AnalysisError("Manifest judge identity is incomplete.")
    if (
        judge.get("dispatch_permitted_in_this_runner") is not False
        or judge.get("subject_target_ids") != list(subjects)
    ):
        raise HostedPart1AnalysisError("Manifest judge reservation contract changed.")
    judge_id = str(judge["target_id"]).casefold()
    judge_route = str(judge["route"]).casefold()
    judge_identity = (
        str(judge["upstream_provider"]).casefold(),
        str(judge["model"]).casefold(),
    )
    for subject in subjects.values():
        if (
            str(subject["target_id"]).casefold() == judge_id
            or str(subject["route"]).casefold() == judge_route
            or (
                str(subject["upstream_provider"]).casefold(),
                str(subject["model"]).casefold(),
            )
            == judge_identity
        ):
            raise HostedPart1AnalysisError("Judge identity overlaps a subject.")
    return dict(judge)


def _is_judge_row(row: Mapping[str, Any], judge: Mapping[str, Any]) -> bool:
    target = row.get("target_id")
    route = row.get("route", row.get("requested_route"))
    return (
        isinstance(target, str)
        and target.casefold() == str(judge["target_id"]).casefold()
    ) or (
        isinstance(route, str)
        and route.casefold() == str(judge["route"]).casefold()
    )


def _validate_ledger(
    rows: Sequence[Mapping[str, Any]],
    *,
    subjects: Mapping[str, Mapping[str, Any]],
    trials: Mapping[str, Any],
    judge: Mapping[str, Any],
    max_attempts: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    reservations: dict[str, dict[str, Any]] = {}
    completions: dict[str, dict[str, Any]] = {}
    attempts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    reservation_positions: dict[str, int] = {}
    for position, row in enumerate(rows):
        if (
            row.get("schema_version") != PANEL_SCHEMA_VERSION
            or row.get("artifact_type") != "inference_hub_part1_attempt_ledger"
            or _is_judge_row(row, judge)
        ):
            raise HostedPart1AnalysisError("Ledger contains an invalid or judge record.")
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise HostedPart1AnalysisError("Ledger attempt id is invalid.")
        if row.get("event") == "reserved_before_dispatch":
            target_id = row.get("target_id")
            trial_id = row.get("trial_id")
            subject = subjects.get(str(target_id))
            trial = trials.get(str(trial_id))
            number = row.get("attempt_number")
            if (
                attempt_id in reservations
                or subject is None
                or trial is None
                or isinstance(number, bool)
                or not isinstance(number, int)
                or not 1 <= number <= max_attempts
                or row.get("upstream_provider") != subject["upstream_provider"]
                or row.get("model") != subject["model"]
                or row.get("route") != subject["route"]
                or row.get("root_id") != trial.root_id
                or row.get("prompt_sha256") != trial.prompt_hash
                or not _hex_digest(row.get("request_sha256"))
            ):
                raise HostedPart1AnalysisError("Ledger reservation binding is invalid.")
            reservations[attempt_id] = dict(row)
            reservation_positions[attempt_id] = position
            attempts[(str(target_id), str(trial_id))].append(dict(row))
        elif row.get("event") == "attempt_completed":
            if (
                attempt_id in completions
                or attempt_id not in reservations
                or reservation_positions[attempt_id] >= position
                or row.get("outcome") not in {"failed", "response_retained"}
            ):
                raise HostedPart1AnalysisError("Ledger completion binding is invalid.")
            if row.get("outcome") == "response_retained" and (
                not _hex_digest(row.get("response_payload_sha256"))
                or not _hex_digest(row.get("response_text_sha256"))
            ):
                raise HostedPart1AnalysisError("Ledger retained-response hashes are invalid.")
            completions[attempt_id] = dict(row)
        else:
            raise HostedPart1AnalysisError("Ledger event is not recognized.")
    if set(reservations) != set(completions):
        raise HostedPart1AnalysisError("Ledger has a non-terminal reservation/completion.")
    for key, values in attempts.items():
        numbers = [int(row["attempt_number"]) for row in values]
        if numbers != list(range(1, len(numbers) + 1)):
            raise HostedPart1AnalysisError(f"Ledger retry sequence is invalid for {key!r}.")
        if any(
            completions[row["attempt_id"]]["outcome"] == "response_retained"
            for row in values[:-1]
        ):
            raise HostedPart1AnalysisError("Ledger retried after retaining a response.")
    return reservations, completions, attempts


def _validate_raw_row(
    row: Mapping[str, Any],
    *,
    subject: Mapping[str, Any],
    trial: Any,
    judge: Mapping[str, Any],
    reservations: Mapping[str, Mapping[str, Any]],
    completions: Mapping[str, Mapping[str, Any]],
    attempts: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if _is_judge_row(row, judge):
        raise HostedPart1AnalysisError("Raw-response journal contains a judge record.")
    if (
        row.get("schema_version") != PANEL_SCHEMA_VERSION
        or row.get("artifact_type") != "inference_hub_part1_raw_response"
        or row.get("target_id") != subject["target_id"]
        or row.get("upstream_provider") != subject["upstream_provider"]
        or row.get("model") != subject["model"]
        or row.get("requested_route") != subject["route"]
        or row.get("trial_id") != trial.trial_id
        or row.get("root_id") != trial.root_id
        or row.get("game") != trial.game
        or row.get("domain") != trial.domain
        or row.get("counterbalance_id") != trial.counterbalance_id
        or row.get("prompt_sha256") != trial.prompt_hash
        or row.get("prompt_text") != trial.prompt_text
    ):
        raise HostedPart1AnalysisError("Raw response subject/schedule identity is invalid.")
    action = row.get("parsed_action")
    response_text = row.get("response_text")
    expected_action = parse_final_action(response_text) if isinstance(response_text, str) else None
    expected_text_hash = (
        hashlib.sha256(response_text.encode("utf-8")).hexdigest()
        if isinstance(response_text, str)
        else None
    )
    if (
        action != expected_action
        or row.get("format_valid") is not (action in {"X", "Y"})
        or row.get("response_text_sha256") != expected_text_hash
    ):
        raise HostedPart1AnalysisError("Raw response parser/text-hash binding is invalid.")

    key = (str(subject["target_id"]), trial.trial_id)
    relevant = list(attempts.get(key, ()))
    if not relevant:
        raise HostedPart1AnalysisError("Raw response has no corresponding ledger attempt.")
    raw_response = row.get("raw_response")
    if raw_response is None:
        failure = row.get("failure")
        final_completion = completions[relevant[-1]["attempt_id"]]
        if (
            row.get("raw_response_sha256") is not None
            or row.get("response_model") is not None
            or row.get("model_identity_valid") is not False
            or row.get("format_valid") is not False
            or not isinstance(failure, Mapping)
            or row.get("reasoning_fields") != {}
            or row.get("output_field") is not None
            or final_completion.get("outcome") != "failed"
            or final_completion.get("failure_code") != failure.get("failure_code")
            or final_completion.get("http_status") != failure.get("http_status")
        ):
            raise HostedPart1AnalysisError("Provider-failure row is not safely retained.")
        provider_failure = True
    else:
        attempt_id = row.get("attempt_id")
        reservation = reservations.get(str(attempt_id))
        completion = completions.get(str(attempt_id))
        if not isinstance(raw_response, Mapping):
            raise HostedPart1AnalysisError("Raw response payload is not an object.")
        expected_metadata = _response_metadata(raw_response)
        if (
            row.get("raw_response_sha256") != _sha256_json(raw_response)
            or reservation is None
            or completion is None
            or reservation.get("target_id") != subject["target_id"]
            or reservation.get("trial_id") != trial.trial_id
            or reservation.get("request_sha256") != row.get("request_sha256")
            or completion.get("outcome") != "response_retained"
            or completion.get("response_payload_sha256") != row.get("raw_response_sha256")
            or completion.get("response_text_sha256") != row.get("response_text_sha256")
            or completion.get("reasoning_fields_sha256")
            != _sha256_json(row.get("reasoning_fields"))
            or any(row.get(key) != expected_metadata[key] for key in (
                "request_id", "response_model", "finish_reason", "usage",
                "reasoning_fields", "output_field", "response_text",
                "response_text_sha256", "parsed_action", "format_valid",
            ))
            or not isinstance(row.get("response_model"), str)
            or raw_response.get("model") != row.get("response_model")
            or completion.get("response_model") != row.get("response_model")
            or row.get("model_identity_valid")
            is not (row.get("response_model") == subject["route"])
        ):
            raise HostedPart1AnalysisError("Raw response hash/route/model identity is invalid.")
        provider_failure = False

    counterbalance = COUNTERBALANCE_BY_ID.get(trial.counterbalance_id)
    if counterbalance is None:
        raise HostedPart1AnalysisError("Trial counterbalance is unknown.")
    cooperation_label = counterbalance.label_for(WELFARE_PRESERVING)
    return {
        "target_id": subject["target_id"],
        "game": trial.game,
        "domain": trial.domain,
        "root_id": trial.root_id,
        "action": action,
        "format_valid": action in {"X", "Y"},
        "provider_failure": provider_failure,
        "model_identity_mismatch": (
            raw_response is not None and row.get("model_identity_valid") is not True
        ),
        "action_x": action == "X",
        "cooperation": action == cooperation_label,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise HostedPart1AnalysisError("Bootstrap distribution is empty.")
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(point: float, draws: Sequence[float]) -> dict[str, Any]:
    return {
        "estimate": point,
        "lower": _percentile(draws, 0.025),
        "upper": _percentile(draws, 0.975),
        "confidence_level": 0.95,
        "method": "within_game_domain_stratified_root_bootstrap_percentile",
    }


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid = sum(bool(row["format_valid"]) for row in rows)
    x_count = sum(row["action"] == "X" for row in rows)
    y_count = sum(row["action"] == "Y" for row in rows)
    cooperation = sum(bool(row["cooperation"]) for row in rows)
    failures = sum(bool(row["provider_failure"]) for row in rows)
    return {
        "planned_and_retained_count": total,
        "format_valid_count": valid,
        "format_invalid_count": total - valid,
        "provider_failure_count": failures,
        "action_x_count": x_count,
        "action_y_count": y_count,
        "format_valid_rate": valid / total,
        "primary_action_x_rate_format_invalid_retained_as_non_x": x_count / total,
        "primary_cooperation_rate_format_invalid_retained_as_noncooperation": cooperation / total,
        "valid_only_action_x_rate_descriptive": x_count / valid if valid else None,
        "valid_only_cooperation_rate_descriptive": cooperation / valid if valid else None,
        "valid_only_denominator": valid,
        "valid_only_label": "descriptive_only_conditioned_on_format_valid_response",
    }


def _family(model: str) -> str:
    value = model.casefold()
    rules = (
        ("claude", "claude"), ("gemini", "gemini"), ("deepseek", "deepseek"),
        ("gemma", "gemma"), ("glm", "glm"), ("kimi", "kimi"),
        ("llama", "llama"), ("minimax", "minimax"), ("mistral", "mistral"),
        ("mixtral", "mistral"), ("nemotron", "nemotron"), ("sonar", "perplexity_sonar"),
        ("qwen", "qwen"), ("gpt-3", "openai_gpt_3"), ("gpt-4", "openai_gpt_4"),
        ("gpt-5", "openai_gpt_5"), ("gpt-oss", "openai_gpt_oss"),
    )
    return next((family for marker, family in rules if marker in value), "other")


def _descriptive_groups(
    subject_results: Sequence[Mapping[str, Any]], key: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for subject in subject_results:
        groups[str(subject[key])].append(subject)
    result: list[dict[str, Any]] = []
    for name in sorted(groups):
        members = groups[name]
        result.append(
            {
                key: name,
                "subject_count": len(members),
                "subject_target_ids": sorted(str(row["target_id"]) for row in members),
                "equal_subject_mean_action_x_rate": sum(
                    float(row["counts"]["primary_action_x_rate_format_invalid_retained_as_non_x"])
                    for row in members
                ) / len(members),
                "equal_subject_mean_cooperation_rate": sum(
                    float(row["counts"]["primary_cooperation_rate_format_invalid_retained_as_noncooperation"])
                    for row in members
                ) / len(members),
                "inference_status": "descriptive_only_no_independent_model_or_provider_replication_claim",
            }
        )
    return result


def analyze_panel(
    manifest_path: Path,
    *,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """Validate a private panel and return a privacy-safe aggregate artifact."""

    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise HostedPart1AnalysisError("Bootstrap seed must be an integer.")
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise HostedPart1AnalysisError("Hosted panel analysis requires exactly 5,000 bootstrap replicates.")
    manifest_path = manifest_path.resolve()
    private_dir = manifest_path.parent
    raw_dir = private_dir / "raw_responses"
    if manifest_path.name != "manifest.json" or private_dir.name != "private":
        raise HostedPart1AnalysisError("Expected a private/manifest.json panel manifest.")
    _require_mode(private_dir.parent, 0o700)
    _require_mode(private_dir, 0o700)
    _require_mode(raw_dir, 0o700)
    _require_mode(manifest_path, 0o600)
    manifest = _read_json(manifest_path, "panel manifest")
    if (
        manifest.get("schema_version") != PANEL_SCHEMA_VERSION
        or manifest.get("artifact_type") != "inference_hub_part1_large_n_exploratory_panel"
        or manifest.get("evidence_sha256") != _self_hash(manifest)
    ):
        raise HostedPart1AnalysisError("Panel manifest schema/type/self-hash failed.")
    if (
        manifest.get("analysis_role") != "exploratory_hosted_scale_panel_only"
        or manifest.get("draft_bank_human_approved") is not False
        or manifest.get("confirmatory_or_paper_promotion_permitted") is not False
        or manifest.get("judge_dispatched") is not False
    ):
        raise HostedPart1AnalysisError("Manifest exploratory nonpromotion flags changed.")
    bindings, registry, compatibility = _validate_bound_files(manifest)
    subjects = _subject_routes(manifest)
    judge = _validate_judge(manifest, subjects)
    _validate_selected_identities(
        registry=registry,
        compatibility=compatibility,
        subjects=subjects,
        judge=judge,
    )

    if (
        manifest.get("full_primary_root_count") != EXPECTED_ROOTS
        or manifest.get("executed_trial_count_per_subject") != EXPECTED_ROOTS
        or manifest.get("trial_limit") is not None
    ):
        raise HostedPart1AnalysisError("Analysis requires the complete 384-root schedule per subject.")
    base_seed = manifest.get("base_seed")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise HostedPart1AnalysisError("Manifest base seed is invalid.")
    trial_list = list(build_draft_trials(base_seed=base_seed))
    detailed_schedule, compact_schedule = _schedule_bindings(trial_list)
    if (
        len(trial_list) != EXPECTED_ROOTS
        or len({trial.root_id for trial in trial_list}) != EXPECTED_ROOTS
        or manifest.get("executed_schedule_sha256") != _sha256_json(detailed_schedule)
        or manifest.get("full_primary_schedule_sha256") != _sha256_json(compact_schedule)
    ):
        raise HostedPart1AnalysisError("Manifest schedule hash/root binding failed.")
    cell_counts = Counter((trial.game, trial.domain) for trial in trial_list)
    expected_cells = {(game, domain) for game in GAMES for domain in DOMAINS}
    if set(cell_counts) != expected_cells or len(cell_counts) != EXPECTED_CELLS or set(cell_counts.values()) != {32}:
        raise HostedPart1AnalysisError("The planned schedule is not 12 balanced game/domain cells.")
    trials = {trial.trial_id: trial for trial in trial_list}

    journals = manifest.get("journals")
    if not isinstance(journals, Mapping) or set(journals) != {"attempt_ledger", "raw_responses"}:
        raise HostedPart1AnalysisError("Manifest journal bindings are incomplete.")
    raw_refs = journals.get("raw_responses")
    if not isinstance(raw_refs, Mapping) or set(raw_refs) != set(subjects):
        raise HostedPart1AnalysisError("Raw-response journal subjects do not match the manifest.")
    ledger_path = private_dir / "attempt_ledger.jsonl"
    ledger_rows = _read_chained_journal(
        ledger_path, journals["attempt_ledger"], label="attempt ledger"
    )
    contract = manifest.get("execution_contract")
    max_attempts = contract.get("max_attempts_per_trial") if isinstance(contract, Mapping) else None
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise HostedPart1AnalysisError("Manifest maximum attempt count is invalid.")
    reservations, completions, attempts = _validate_ledger(
        ledger_rows,
        subjects=subjects,
        trials=trials,
        judge=judge,
        max_attempts=max_attempts,
    )

    observations: dict[str, list[dict[str, Any]]] = {}
    journal_bindings: dict[str, Any] = {
        "attempt_ledger": dict(journals["attempt_ledger"]), "raw_responses": {}
    }
    for target_id, subject in subjects.items():
        raw_path = raw_dir / f"{_safe_file_stem(target_id)}.jsonl"
        raw_rows = _read_chained_journal(raw_path, raw_refs[target_id], label=f"raw responses for {target_id}")
        by_trial: dict[str, Mapping[str, Any]] = {}
        derived: list[dict[str, Any]] = []
        for row in raw_rows:
            trial_id = row.get("trial_id")
            if not isinstance(trial_id, str) or trial_id in by_trial or trial_id not in trials:
                raise HostedPart1AnalysisError(f"Raw response coverage is duplicated/unknown for {target_id}.")
            by_trial[trial_id] = row
            derived.append(
                _validate_raw_row(
                    row,
                    subject=subject,
                    trial=trials[trial_id],
                    judge=judge,
                    reservations=reservations,
                    completions=completions,
                    attempts=attempts,
                )
            )
        if set(by_trial) != set(trials) or len({row["root_id"] for row in derived}) != EXPECTED_ROOTS:
            raise HostedPart1AnalysisError(f"Subject {target_id} lacks exact 384-root coverage.")
        observations[target_id] = derived
        journal_bindings["raw_responses"][target_id] = dict(raw_refs[target_id])

    expected_attempt_keys = {(target_id, trial_id) for target_id in subjects for trial_id in trials}
    if set(attempts) != expected_attempt_keys:
        raise HostedPart1AnalysisError("Ledger does not cover exactly the planned subject-by-trial cells.")

    all_rows = [row for values in observations.values() for row in values]
    summary_expected = {
        "planned_generations": len(subjects) * EXPECTED_ROOTS,
        "retained_trial_records": len(all_rows),
        "responses_received": sum(not row["provider_failure"] for row in all_rows),
        "failed_without_response": sum(row["provider_failure"] for row in all_rows),
        "format_valid": sum(row["format_valid"] for row in all_rows),
        "format_invalid_retained": sum(
            (not row["provider_failure"]) and (not row["format_valid"]) for row in all_rows
        ),
        "response_model_identity_mismatches": sum(
            row["model_identity_mismatch"] for row in all_rows
        ),
    }
    if manifest.get("summary") != summary_expected:
        raise HostedPart1AnalysisError("Manifest summary does not match the validated evidence.")
    expected_complete = (
        summary_expected["failed_without_response"] == 0
        and summary_expected["response_model_identity_mismatches"] == 0
    )
    if manifest.get("complete") is not expected_complete:
        raise HostedPart1AnalysisError("Manifest completion flag does not match retained evidence.")

    quarantined_targets: list[dict[str, Any]] = []
    eligible_observations: dict[str, list[dict[str, Any]]] = {}
    for target_id, rows in observations.items():
        provider_failure_count = sum(row["provider_failure"] for row in rows)
        identity_mismatch_count = sum(row["model_identity_mismatch"] for row in rows)
        if provider_failure_count or identity_mismatch_count:
            reasons = []
            if identity_mismatch_count:
                reasons.append("response_model_identity_mismatch")
            if provider_failure_count:
                reasons.append("failed_without_response")
            quarantined_targets.append(
                {
                    "target_id": target_id,
                    "operational_reasons": reasons,
                    "response_model_identity_mismatch_count": identity_mismatch_count,
                    "failed_without_response_count": provider_failure_count,
                    "excluded_row_count": EXPECTED_ROOTS,
                    "exclusion_scope": "entire_target_excluded_from_all_aggregates_and_bootstrap",
                    "exclusion_basis": "protocol_and_model_identity_only_not_observed_action",
                }
            )
        else:
            eligible_observations[target_id] = rows

    ordered_cells = [(game, domain) for game in GAMES for domain in DOMAINS]
    rows_by_subject_root = {
        target_id: {str(row["root_id"]): row for row in rows}
        for target_id, rows in eligible_observations.items()
    }
    roots_by_cell = {
        cell: [trial.root_id for trial in trial_list if (trial.game, trial.domain) == cell]
        for cell in ordered_cells
    }
    subject_ids = list(eligible_observations)
    action_draws = {target_id: [] for target_id in subject_ids}
    cooperation_draws = {target_id: [] for target_id in subject_ids}
    panel_action_draws: list[float] = []
    panel_cooperation_draws: list[float] = []
    rng = random.Random(bootstrap_seed)
    for _ in range(bootstrap_replicates if subject_ids else 0):
        action_totals = dict.fromkeys(subject_ids, 0)
        cooperation_totals = dict.fromkeys(subject_ids, 0)
        for cell in ordered_cells:
            roots = roots_by_cell[cell]
            sampled = rng.choices(roots, k=len(roots))
            sampled_counts = Counter(sampled)
            for target_id in subject_ids:
                lookup = rows_by_subject_root[target_id]
                action_totals[target_id] += sum(
                    count * int(lookup[root]["action_x"]) for root, count in sampled_counts.items()
                )
                cooperation_totals[target_id] += sum(
                    count * int(lookup[root]["cooperation"]) for root, count in sampled_counts.items()
                )
        for target_id in subject_ids:
            action_draws[target_id].append(action_totals[target_id] / EXPECTED_ROOTS)
            cooperation_draws[target_id].append(cooperation_totals[target_id] / EXPECTED_ROOTS)
        panel_action_draws.append(sum(action_totals.values()) / (EXPECTED_ROOTS * len(subject_ids)))
        panel_cooperation_draws.append(sum(cooperation_totals.values()) / (EXPECTED_ROOTS * len(subject_ids)))

    subject_results: list[dict[str, Any]] = []
    for target_id in subject_ids:
        subject = subjects[target_id]
        rows = eligible_observations[target_id]
        counts = _counts(rows)
        cell_summaries = []
        for game, domain in ordered_cells:
            cell_rows = [row for row in rows if row["game"] == game and row["domain"] == domain]
            cell_summaries.append({"game": game, "domain": domain, **_counts(cell_rows)})
        subject_results.append(
            {
                "target_id": target_id,
                "upstream_provider": subject["upstream_provider"],
                "model": subject["model"],
                "route": subject["route"],
                "family": _family(str(subject["model"])),
                "counts": counts,
                "primary_action_x_rate_95_ci": _interval(
                    counts["primary_action_x_rate_format_invalid_retained_as_non_x"],
                    action_draws[target_id],
                ),
                "primary_cooperation_rate_95_ci": _interval(
                    counts["primary_cooperation_rate_format_invalid_retained_as_noncooperation"],
                    cooperation_draws[target_id],
                ),
                "per_game_domain": cell_summaries,
            }
        )

    panel_action = (
        sum(
            row["counts"]["primary_action_x_rate_format_invalid_retained_as_non_x"]
            for row in subject_results
        ) / len(subject_results)
        if subject_results else None
    )
    panel_cooperation = (
        sum(
            row["counts"]["primary_cooperation_rate_format_invalid_retained_as_noncooperation"]
            for row in subject_results
        ) / len(subject_results)
        if subject_results else None
    )
    panel_cells = [
        {
            "game": game,
            "domain": domain,
            "equal_subject_mean_format_valid_rate": sum(
                subject["per_game_domain"][index]["format_valid_rate"]
                for subject in subject_results
            ) / len(subject_results),
            "equal_subject_mean_action_x_rate_format_invalid_retained_as_non_x": sum(
                subject["per_game_domain"][index][
                    "primary_action_x_rate_format_invalid_retained_as_non_x"
                ]
                for subject in subject_results
            ) / len(subject_results),
            "equal_subject_mean_cooperation_rate_format_invalid_retained_as_noncooperation": sum(
                subject["per_game_domain"][index][
                    "primary_cooperation_rate_format_invalid_retained_as_noncooperation"
                ]
                for subject in subject_results
            ) / len(subject_results),
        }
        for index, (game, domain) in enumerate(ordered_cells)
    ] if subject_results else []
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "inference_hub_part1_exploratory_private_panel_analysis",
        "analysis_role": "exploratory_only_no_confirmatory_or_paper_promotion",
        "confirmatory_or_paper_promotion_permitted": False,
        "draft_bank_human_approved": False,
        "privacy_contract": {
            "contains_prompt_text": False,
            "contains_raw_response_or_response_text": False,
            "contains_reasoning": False,
            "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
        },
        "parameters": {
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_method": "shared_root_resampling_with_replacement_within_each_of_12_game_by_domain_strata",
            "confidence_interval": "deterministic_nonparametric_percentile_95",
            "invalid_handling": "format_invalid_responses_retained_in_primary_denominator_as_non_x_and_noncooperation",
            "operational_failure_handling": "any_provider_failure_or_response_model_identity_mismatch_quarantines_the_entire_target_before_aggregation",
        },
        "bindings": {
            "panel_manifest_path": str(manifest_path),
            "panel_manifest_file_sha256": _sha256_file(manifest_path),
            "panel_manifest_evidence_sha256": manifest["evidence_sha256"],
            **bindings,
            "journals": journal_bindings,
        },
        "coverage": {
            "planned_subject_count": len(subjects),
            "aggregate_eligible_subject_count": len(subject_ids),
            "quarantined_subject_count": len(quarantined_targets),
            "roots_per_subject": EXPECTED_ROOTS,
            "planned_subject_trial_count": len(subjects) * EXPECTED_ROOTS,
            "retained_subject_trial_count": len(all_rows),
            "game_domain_strata": EXPECTED_CELLS,
            "roots_per_stratum_per_subject": 32,
            "panel_wide_complete_for_all_subjects": not quarantined_targets,
            "runner_complete_flag": manifest["complete"],
            "judge_rows": 0,
        },
        "quarantined_targets": quarantined_targets,
        "subjects": subject_results,
        "overall_equal_subject": (
            {
                "weighting": "finite_panel_equal_subject_weight",
                "subject_count": len(subject_ids),
                "primary_action_x_rate_95_ci": _interval(panel_action, panel_action_draws),
                "primary_cooperation_rate_95_ci": _interval(panel_cooperation, panel_cooperation_draws),
                "per_game_domain": panel_cells,
                "inference_status": "exploratory_finite_panel_summary_over_nonquarantined_targets_only",
            }
            if panel_action is not None and panel_cooperation is not None
            else None
        ),
        "provider_descriptive": _descriptive_groups(subject_results, "upstream_provider"),
        "family_descriptive": _descriptive_groups(subject_results, "family"),
        "family_definition": "deterministic_casefolded_model_identifier_marker_v1; unmatched models are other",
        "limitations": [
            "Exploratory draft scenarios were not human-approved.",
            "Provider and family group summaries are descriptive and do not establish independent replication.",
            "Valid-only rates condition on format validity and are not primary estimands.",
            "Operationally quarantined targets contribute no rows to any aggregate.",
            "This artifact cannot be promoted to confirmatory or paper evidence.",
        ],
    }
    _seal(artifact)
    return artifact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and safely aggregate a private hosted Part 1 panel."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = analyze_panel(args.manifest, bootstrap_seed=args.bootstrap_seed)
    _atomic_json(args.output, artifact)
    print(
        f"Validated {artifact['coverage']['planned_subject_count']} subjects with 384 roots each; "
        f"{artifact['coverage']['aggregate_eligible_subject_count']} are aggregate-eligible."
    )
    print(f"Exploratory aggregate artifact: {args.output}")
    return 0


def cli(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (HostedPart1AnalysisError, OSError, ValueError) as error:
        print(f"Hosted Part 1 analysis failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(cli())
