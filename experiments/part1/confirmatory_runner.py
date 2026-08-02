"""Production execution path for the frozen Part 1 confirmatory design.

The exploratory Part 1 runner is intentionally not imported.  This module
accepts only a byte-pinned 384-root registry that passes every production
content-review gate in :mod:`experiments.part1.confirmatory_design`, freezes
the complete primary and secondary schedules, and writes private, hash-chained
execution artifacts.  Semantic responses are never regenerated.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError

from agents.agent_config import resolve_model_registry_entry
from experiments.misc.attempt_log import (
    DurableAttemptLogger,
    attempt_log_path_for_csv,
    load_attempt_records,
    verify_attempt_log_metadata,
)
from experiments.misc.final_answer import (
    EXTRACTOR_PROMPT_TEMPLATE,
    EXTRACTOR_SYSTEM_PROMPT,
    FINAL_ANSWER_PROTOCOL,
)
from experiments.misc.run_metadata import (
    git_commit,
    git_dirty,
    metadata_payload_sha256,
    safe_error_message,
    sha256_file,
    source_bundle_metadata,
    stable_json_hash,
    utc_now_iso,
    validate_metadata_integrity,
)
from experiments.part1.confirmatory_design import (
    EXPECTED_ROOT_COUNT,
    ActionSpec,
    ConfirmatoryTrial,
    HumanContentApproval,
    MappingReview,
    ObjectiveOutcome,
    ParsedOutcome,
    PayoffOrdering,
    ProductionGateError,
    ScenarioRoot,
    WelfareMapping,
    build_primary_schedule,
    build_role_schedule,
    parse_exact_final_token,
    validate_bank,
    validate_primary_schedule,
    validate_role_schedule,
    welfare_preserving_choice,
)
from providers.api_call import (
    ProviderResponse,
    ResponseParseError,
    api_call_detailed,
    failure_provenance,
)


BANK_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
RUN_METADATA_SCHEMA_VERSION = 1
EXPECTED_PRIMARY_TRIALS = 3_072
EXPECTED_SECONDARY_TRIALS = 1_152
EXPECTED_TRIALS_PER_MODEL = EXPECTED_PRIMARY_TRIALS + EXPECTED_SECONDARY_TRIALS
DEFAULT_PRIMARY_SEED = 20_260_801
DEFAULT_SECONDARY_SEED = 20_260_802
DEFAULT_EXTRACTOR_SEED = 20_260_803
SUBJECT_SYSTEM_PROMPT = ""
EXTRACTOR_MAX_TOKENS = 1_024
MAX_TRANSPORT_ATTEMPTS = 3
SEED_DERIVATION = "sha256-first-63-bits-v1"
SCHEDULE_PROTOCOL = "part1-primary-secondary-exact-order-v1"
PRIVATE_RESULTS_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "private" / "part1_confirmatory"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ConfirmatoryPart1Error(RuntimeError):
    """A Part 1 bank, plan, request, or artifact violated its frozen contract."""


class TransportExhaustedError(ConfirmatoryPart1Error):
    """A frozen route exhausted its bounded transport-only retries."""


class RouteIdentityError(ConfirmatoryPart1Error):
    """A configured or returned route identity was not exact and verified."""


class VisibleAnswer(BaseModel):
    """Exact visible-response copy returned by the independent extractor."""

    model_config = ConfigDict(extra="forbid")
    response: str


@dataclass(frozen=True)
class LoadedScenarioBank:
    path: Path
    file_sha256: str
    roots: tuple[ScenarioRoot, ...]
    payload: dict[str, Any]


@dataclass(frozen=True)
class FrozenRoute:
    provider: str
    route: str
    registry_version: str
    registry_hash: str
    identity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DetailedCall = Callable[..., ProviderResponse]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _record_hash(value: Mapping[str, Any]) -> str:
    return stable_json_hash(
        {key: item for key, item in value.items() if key != "record_sha256"}
    )


def _derived_seed(seed_base: int, *parts: object) -> int:
    material = "|".join([str(seed_base), *(str(part) for part in parts)])
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16) & (
        (1 << 63) - 1
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ConfirmatoryPart1Error(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ConfirmatoryPart1Error(
            f"{label} fields are not exact (missing={missing}, extra={extra})."
        )


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _secure_file(path: Path) -> None:
    if path.exists():
        os.chmod(path, 0o600)


def _private_output_directory(path: str | Path, *, create: bool) -> Path:
    private_root = PRIVATE_RESULTS_ROOT.resolve()
    output = Path(path).resolve()
    try:
        output.relative_to(private_root)
    except ValueError as error:
        raise ConfirmatoryPart1Error(
            f"Production Part 1 output must stay under {private_root}."
        ) from error
    if create:
        private_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(private_root, 0o700)
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(output, 0o700)
    elif not output.is_dir():
        raise ConfirmatoryPart1Error("Strict resume private output directory is missing.")
    return output


def _require_private_permissions(directory: Path, files: Sequence[Path]) -> None:
    if directory.stat().st_mode & 0o077:
        raise ConfirmatoryPart1Error("Private Part 1 output directory is too permissive.")
    for path in files:
        if not path.is_file():
            raise ConfirmatoryPart1Error(f"Strict resume is missing artifact {path}.")
        if path.stat().st_mode & 0o077:
            raise ConfirmatoryPart1Error(
                f"Private Part 1 artifact has permissive mode bits: {path}."
            )


def _production_source_bundle() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    return source_bundle_metadata(
        [
            Path(__file__),
            repo_root / "experiments" / "part1" / "confirmatory_design.py",
            repo_root / "experiments" / "part1" / "scenario_variants.py",
            repo_root / "experiments" / "part1" / "part_1_prompt.json",
            repo_root / "experiments" / "misc" / "attempt_log.py",
            repo_root / "experiments" / "misc" / "final_answer.py",
            repo_root / "experiments" / "misc" / "preflight.py",
            repo_root / "experiments" / "misc" / "prompt_loader.py",
            repo_root / "experiments" / "misc" / "result_writer.py",
            repo_root / "experiments" / "misc" / "run_metadata.py",
            repo_root / "providers" / "api_call.py",
            repo_root / "agents" / "agent_config.py",
            repo_root / "agents" / "agent_config.registry.json",
            repo_root / "pyproject.toml",
            repo_root / "uv.lock",
        ]
    )


def _environment_lock() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    return source_bundle_metadata(
        [repo_root / "pyproject.toml", repo_root / "uv.lock"]
    )


def _require_clean_git_state() -> str:
    commit = git_commit()
    if not isinstance(commit, str) or _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ConfirmatoryPart1Error(
            "Confirmatory Part 1 requires an exact 40-character Git commit."
        )
    if git_dirty() is not False:
        raise ConfirmatoryPart1Error(
            "Confirmatory Part 1 requires a clean Git worktree."
        )
    return commit


def _scenario_root_from_dict(value: Any, *, index: int) -> ScenarioRoot:
    if not isinstance(value, Mapping):
        raise ConfirmatoryPart1Error(f"Scenario root {index} is not an object.")
    expected_root_fields = {
        "root_id",
        "semantic_cluster_id",
        "game",
        "domain",
        "scenario_index_in_cell",
        "context",
        "focal_actor",
        "counterpart_actor",
        "actions",
        "objective_outcomes",
        "welfare_mapping",
        "payoff_ordering",
        "semantic_facets",
        "wording_family_id",
        "authorship_method",
        "mapping_review",
        "human_content_approvals",
        "content_hash",
    }
    _require_exact_keys(value, expected_root_fields, label=f"scenario root {index}")
    try:
        actions_raw = value["actions"]
        outcomes_raw = value["objective_outcomes"]
        approvals_raw = value["human_content_approvals"]
        if not isinstance(actions_raw, list) or not isinstance(outcomes_raw, list):
            raise TypeError("actions and objective_outcomes must be arrays")
        if not isinstance(approvals_raw, list):
            raise TypeError("human_content_approvals must be an array")
        actions = tuple(ActionSpec(**item) for item in actions_raw)
        outcomes = tuple(ObjectiveOutcome(**item) for item in outcomes_raw)
        approvals = tuple(HumanContentApproval(**item) for item in approvals_raw)
        facets = value["semantic_facets"]
        if not isinstance(facets, list) or not all(isinstance(item, str) for item in facets):
            raise TypeError("semantic_facets must be a string array")
        return ScenarioRoot(
            root_id=value["root_id"],
            semantic_cluster_id=value["semantic_cluster_id"],
            game=value["game"],
            domain=value["domain"],
            scenario_index_in_cell=value["scenario_index_in_cell"],
            context=value["context"],
            focal_actor=value["focal_actor"],
            counterpart_actor=value["counterpart_actor"],
            actions=actions,
            objective_outcomes=outcomes,
            welfare_mapping=WelfareMapping(**value["welfare_mapping"]),
            payoff_ordering=PayoffOrdering(**value["payoff_ordering"]),
            semantic_facets=tuple(facets),
            wording_family_id=value["wording_family_id"],
            authorship_method=value["authorship_method"],
            mapping_review=MappingReview(**value["mapping_review"]),
            human_content_approvals=approvals,
            content_hash=value["content_hash"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfirmatoryPart1Error(
            f"Scenario root {index} does not match the frozen schema."
        ) from error


def load_production_bank(
    path: str | Path, *, expected_sha256: str
) -> LoadedScenarioBank:
    """Load and validate a human-approved bank bound to exact file bytes."""

    expected = _require_sha256(expected_sha256, label="expected scenario-bank hash")
    bank_path = Path(path)
    if not bank_path.is_file():
        raise FileNotFoundError(f"Pinned Part 1 bank does not exist: {bank_path}")
    actual = sha256_file(bank_path)
    if actual != expected:
        raise ConfirmatoryPart1Error(
            f"Part 1 scenario-bank hash mismatch: expected {expected}, found {actual}."
        )
    try:
        payload = json.loads(bank_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryPart1Error("Part 1 bank is not valid UTF-8 JSON.") from error
    if not isinstance(payload, dict):
        raise ConfirmatoryPart1Error("Part 1 bank root must be an object.")
    _require_exact_keys(payload, {"schema_version", "roots"}, label="scenario bank")
    if payload["schema_version"] != BANK_SCHEMA_VERSION:
        raise ConfirmatoryPart1Error("Unsupported Part 1 scenario-bank schema.")
    roots_raw = payload["roots"]
    if not isinstance(roots_raw, list) or len(roots_raw) != EXPECTED_ROOT_COUNT:
        raise ConfirmatoryPart1Error(
            f"Production Part 1 requires exactly {EXPECTED_ROOT_COUNT} roots."
        )
    roots = tuple(
        _scenario_root_from_dict(item, index=index)
        for index, item in enumerate(roots_raw)
    )
    try:
        validate_bank(roots, production=True).require_valid()
    except ProductionGateError as error:
        raise ConfirmatoryPart1Error(
            "Part 1 bank failed human-approval or content-integrity gates: " + str(error)
        ) from error
    return LoadedScenarioBank(bank_path.resolve(), actual, roots, payload)


def freeze_verified_route(provider: str, model: str) -> FrozenRoute:
    entry = resolve_model_registry_entry(provider, model)
    if not isinstance(entry, dict):
        raise RouteIdentityError(f"Route is absent from registry: {provider}/{model}.")
    if entry.get("verification_status") != "verified" or not isinstance(
        entry.get("verification_evidence"), Mapping
    ):
        raise RouteIdentityError(f"Route is not verified: {provider}/{model}.")
    route = FrozenRoute(
        provider=str(entry["provider"]),
        route=str(entry["route"]),
        registry_version=str(entry["registry_version"]),
        registry_hash=str(entry["registry_hash"]),
        identity=deepcopy(entry),
    )
    validate_frozen_route(route)
    return route


def validate_frozen_route(route: FrozenRoute) -> None:
    identity = route.identity
    if not all((route.provider.strip(), route.route.strip(), route.registry_version.strip())):
        raise RouteIdentityError("Frozen route fields must be non-empty.")
    _require_sha256(route.registry_hash, label="model registry hash")
    if (
        identity.get("provider") != route.provider
        or identity.get("route") != route.route
        or identity.get("registry_version") != route.registry_version
        or identity.get("registry_hash") != route.registry_hash
        or identity.get("verification_status") != "verified"
        or not isinstance(identity.get("verification_evidence"), Mapping)
    ):
        raise RouteIdentityError("Frozen route does not match its verified identity.")
    if resolve_model_registry_entry(route.provider, route.route) != identity:
        raise RouteIdentityError(
            "Frozen route identity does not exactly match the current model registry."
        )


def _route_from_dict(value: Mapping[str, Any]) -> FrozenRoute:
    try:
        route = FrozenRoute(
            provider=str(value["provider"]),
            route=str(value["route"]),
            registry_version=str(value["registry_version"]),
            registry_hash=str(value["registry_hash"]),
            identity=deepcopy(value["identity"]),
        )
    except (KeyError, TypeError) as error:
        raise RouteIdentityError("Frozen plan contains an invalid route object.") from error
    validate_frozen_route(route)
    return route


def _serialize_trial(
    trial: ConfirmatoryTrial,
    *,
    phase: str,
    execution_index: int,
    subject_route: FrozenRoute,
    extractor_seed_base: int,
) -> dict[str, Any]:
    return {
        **asdict(trial),
        "phase": phase,
        "execution_index": execution_index,
        "subject_route": subject_route.to_dict(),
        "extractor_seed": _derived_seed(
            extractor_seed_base, "part1", trial.trial_id, "extractor"
        ),
        "extractor_seed_base": extractor_seed_base,
        "extractor_seed_derivation": SEED_DERIVATION,
    }


def build_confirmatory_schedule(
    loaded_bank: LoadedScenarioBank,
    *,
    subject_route: FrozenRoute,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
    secondary_seed: int = DEFAULT_SECONDARY_SEED,
    extractor_seed: int = DEFAULT_EXTRACTOR_SEED,
) -> list[dict[str, Any]]:
    """Construct the exact 3,072 primary plus 1,152 secondary calls."""

    validate_bank(loaded_bank.roots, production=True).require_valid()
    validate_frozen_route(subject_route)
    primary = build_primary_schedule(
        loaded_bank.roots,
        base_seed=primary_seed,
        requested_provider=subject_route.provider,
        requested_model=subject_route.route,
        production=True,
    )
    secondary = build_role_schedule(
        loaded_bank.roots,
        base_seed=secondary_seed,
        requested_provider=subject_route.provider,
        requested_model=subject_route.route,
        production=True,
    )
    validate_primary_schedule(primary, loaded_bank.roots, production=True).require_valid()
    validate_role_schedule(secondary, loaded_bank.roots, production=True).require_valid()
    schedule: list[dict[str, Any]] = []
    for phase, trials in (("primary", primary), ("secondary", secondary)):
        for trial in trials:
            schedule.append(
                _serialize_trial(
                    trial,
                    phase=phase,
                    execution_index=len(schedule),
                    subject_route=subject_route,
                    extractor_seed_base=extractor_seed,
                )
            )
    if len(schedule) != EXPECTED_TRIALS_PER_MODEL:
        raise ConfirmatoryPart1Error("Part 1 schedule cardinality is not exact.")
    subject_seeds = [item["generation_settings"]["generation_seed"] for item in schedule]
    extractor_seeds = [item["extractor_seed"] for item in schedule]
    if len(set(subject_seeds)) != len(subject_seeds):
        raise ConfirmatoryPart1Error("Subject per-call generation seeds are not unique.")
    if len(set(extractor_seeds)) != len(extractor_seeds):
        raise ConfirmatoryPart1Error("Extractor per-call generation seeds are not unique.")
    return schedule


def freeze_execution_plan(
    loaded_bank: LoadedScenarioBank,
    *,
    subject_route: FrozenRoute,
    extractor_route: FrozenRoute,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
    secondary_seed: int = DEFAULT_SECONDARY_SEED,
    extractor_seed: int = DEFAULT_EXTRACTOR_SEED,
) -> dict[str, Any]:
    """Freeze content, complete schedules, routes, code, and dependencies."""

    for route in (subject_route, extractor_route):
        validate_frozen_route(route)
    clean_commit = _require_clean_git_state()
    schedule = build_confirmatory_schedule(
        loaded_bank,
        subject_route=subject_route,
        primary_seed=primary_seed,
        secondary_seed=secondary_seed,
        extractor_seed=extractor_seed,
    )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment": "part_1_confirmatory",
        "bank_file_sha256": loaded_bank.file_sha256,
        "bank_manifest_sha256": stable_json_hash(
            [asdict(root) for root in loaded_bank.roots]
        ),
        "root_count": EXPECTED_ROOT_COUNT,
        "primary_trial_count": EXPECTED_PRIMARY_TRIALS,
        "secondary_trial_count": EXPECTED_SECONDARY_TRIALS,
        "schedule_protocol": SCHEDULE_PROTOCOL,
        "primary_seed_base": primary_seed,
        "secondary_seed_base": secondary_seed,
        "extractor_seed_base": extractor_seed,
        "freeze_state": {
            "git_commit": clean_commit,
            "git_dirty": False,
            "source_bundle": _production_source_bundle(),
            "environment_lock": _environment_lock(),
        },
        "subject_route": subject_route.to_dict(),
        "extractor_route": extractor_route.to_dict(),
        "protocol": {
            "subject_system_prompt": SUBJECT_SYSTEM_PROMPT,
            "subject_system_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
            "subject_settings_source": "per_trial_generation_settings",
            "extraction_protocol": FINAL_ANSWER_PROTOCOL,
            "extractor_system_prompt_sha256": _sha256_text(EXTRACTOR_SYSTEM_PROMPT),
            "extractor_prompt_template_sha256": _sha256_text(EXTRACTOR_PROMPT_TEMPLATE),
            "extractor_input_policy": "base64_visible_subject_content_only_exact_copy",
            "extractor_settings": {
                "temperature": 0,
                "top_p": 1.0,
                "max_tokens": EXTRACTOR_MAX_TOKENS,
                "schema": VisibleAnswer.model_json_schema(),
            },
            "terminal_parser": "parse_exact_final_token",
            "allowed_outcomes": ["X", "Y", "INVALID"],
            "hidden_reasoning_input": False,
            "semantic_retry": False,
            "transport_retry": "same_frozen_route_only",
            "max_transport_attempts": MAX_TRANSPORT_ATTEMPTS,
        },
        "schedule": schedule,
    }
    payload["plan_sha256"] = stable_json_hash(payload)
    validate_execution_plan(payload, loaded_bank)
    return payload


def validate_execution_plan(
    plan: Mapping[str, Any], loaded_bank: LoadedScenarioBank
) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ConfirmatoryPart1Error("Unsupported Part 1 plan schema.")
    expected_hash = stable_json_hash(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected_hash:
        raise ConfirmatoryPart1Error("Part 1 plan hash mismatch.")
    if plan.get("bank_file_sha256") != loaded_bank.file_sha256:
        raise ConfirmatoryPart1Error("Part 1 plan is bound to a different bank.")
    expected_bank_hash = stable_json_hash([asdict(root) for root in loaded_bank.roots])
    if plan.get("bank_manifest_sha256") != expected_bank_hash:
        raise ConfirmatoryPart1Error("Part 1 plan bank manifest hash mismatch.")
    expected_counts = {
        "root_count": EXPECTED_ROOT_COUNT,
        "primary_trial_count": EXPECTED_PRIMARY_TRIALS,
        "secondary_trial_count": EXPECTED_SECONDARY_TRIALS,
        "schedule_protocol": SCHEDULE_PROTOCOL,
    }
    if any(plan.get(key) != value for key, value in expected_counts.items()):
        raise ConfirmatoryPart1Error("Part 1 frozen design counts or protocol changed.")
    current_commit = _require_clean_git_state()
    expected_freeze = {
        "git_commit": current_commit,
        "git_dirty": False,
        "source_bundle": _production_source_bundle(),
        "environment_lock": _environment_lock(),
    }
    if plan.get("freeze_state") != expected_freeze:
        raise ConfirmatoryPart1Error(
            "Part 1 freeze state changed: commit, sources, or environment differ."
        )
    subject_route = _route_from_dict(plan["subject_route"])
    extractor_route = _route_from_dict(plan["extractor_route"])
    del extractor_route
    expected_protocol = freeze_protocol()
    if plan.get("protocol") != expected_protocol:
        raise ConfirmatoryPart1Error("Part 1 frozen execution protocol changed.")
    expected_schedule = build_confirmatory_schedule(
        loaded_bank,
        subject_route=subject_route,
        primary_seed=int(plan["primary_seed_base"]),
        secondary_seed=int(plan["secondary_seed_base"]),
        extractor_seed=int(plan["extractor_seed_base"]),
    )
    if plan.get("schedule") != expected_schedule:
        raise ConfirmatoryPart1Error("Part 1 schedule does not reconstruct exactly.")


def freeze_protocol() -> dict[str, Any]:
    """Return the exact execution protocol independently of a plan."""

    return {
        "subject_system_prompt": SUBJECT_SYSTEM_PROMPT,
        "subject_system_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
        "subject_settings_source": "per_trial_generation_settings",
        "extraction_protocol": FINAL_ANSWER_PROTOCOL,
        "extractor_system_prompt_sha256": _sha256_text(EXTRACTOR_SYSTEM_PROMPT),
        "extractor_prompt_template_sha256": _sha256_text(EXTRACTOR_PROMPT_TEMPLATE),
        "extractor_input_policy": "base64_visible_subject_content_only_exact_copy",
        "extractor_settings": {
            "temperature": 0,
            "top_p": 1.0,
            "max_tokens": EXTRACTOR_MAX_TOKENS,
            "schema": VisibleAnswer.model_json_schema(),
        },
        "terminal_parser": "parse_exact_final_token",
        "allowed_outcomes": ["X", "Y", "INVALID"],
        "hidden_reasoning_input": False,
        "semantic_retry": False,
        "transport_retry": "same_frozen_route_only",
        "max_transport_attempts": MAX_TRANSPORT_ATTEMPTS,
    }


def _response_audit(response: ProviderResponse, route: FrozenRoute) -> dict[str, Any]:
    audit = {
        "provider": response.provider,
        "model": response.model,
        "requested_model": response.requested_model,
        "response_model": response.response_model,
        "model_identity_match": response.model_identity_match,
        "request_id": response.request_id,
        "finish_reason": response.finish_reason,
        "truncated": response.truncated,
        "usage": response.usage,
        "visible_response_sha256": _sha256_text(response.content),
        "hidden_reasoning_present": bool(response.reasoning),
        "raw_response_sha256": _sha256_text(_canonical_json(response.raw_response)),
        "frozen_route": route.to_dict(),
    }
    if (
        response.provider != route.provider
        or response.model != route.route
        or response.requested_model != route.route
        or response.response_model != route.route
        or response.model_identity_match is not True
    ):
        raise RouteIdentityError(
            f"Returned identity does not exactly match {route.provider}/{route.route}."
        )
    if not isinstance(response.request_id, str) or not response.request_id.strip():
        raise RouteIdentityError(f"{route.provider}/{route.route} omitted request ID.")
    if not isinstance(response.finish_reason, str) or not response.finish_reason.strip():
        raise RouteIdentityError(f"{route.provider}/{route.route} omitted finish reason.")
    if not isinstance(response.usage, Mapping):
        raise RouteIdentityError(f"{route.provider}/{route.route} omitted token usage.")
    return audit


def _transport_retryable(
    error: Exception, route: FrozenRoute
) -> tuple[bool, dict[str, Any] | None]:
    provenance = failure_provenance(error, provider=route.provider, model=route.route)
    status_code = provenance.get("status_code") if provenance else None
    retryable = bool(
        not isinstance(error, (ResponseParseError, TypeError, ValueError, ValidationError))
        and provenance
        and (
            provenance.get("category") in {"gateway", "transport"}
            or (
                isinstance(status_code, int)
                and (status_code in {408, 409, 425, 429} or status_code >= 500)
            )
        )
    )
    return retryable, provenance


def _extractor_prompt(visible_response: str) -> str:
    return EXTRACTOR_PROMPT_TEMPLATE.format(
        kind="part_1_exact_visible_response_copy",
        schema=json.dumps(
            VisibleAnswer.model_json_schema(), sort_keys=True, ensure_ascii=False
        ),
        content_sha256=_sha256_text(visible_response),
        content_base64=base64.b64encode(visible_response.encode("utf-8")).decode("ascii"),
    )


def _attempt_unit(
    trial: Mapping[str, Any], *, stage: str, request_sha256: str
) -> dict[str, Any]:
    return {
        "stage": stage,
        "trial_id": trial["trial_id"],
        "phase": trial["phase"],
        "execution_index": trial["execution_index"],
        "root_id": trial["root_id"],
        "semantic_cluster_id": trial["semantic_cluster_id"],
        "game": trial["game"],
        "domain": trial["domain"],
        "generation_block": trial["generation_block"],
        "frame_id": trial["frame_id"],
        "counterbalance_id": trial["counterbalance_id"],
        "prompt_sha256": trial["prompt_hash"],
        "root_content_hash": trial["root_content_hash"],
        "generation_seed": trial["generation_settings"]["generation_seed"],
        "extractor_seed": trial["extractor_seed"],
        "stage_request_sha256": request_sha256,
    }


def _append_private_attempt(
    attempt_logger: DurableAttemptLogger, **kwargs: Any
) -> None:
    attempt_logger.append(**kwargs)
    _secure_file(attempt_logger.path)


def _call_stage(
    *,
    stage: str,
    trial: Mapping[str, Any],
    route: FrozenRoute,
    system_prompt: str,
    query: str,
    attempt_logger: DurableAttemptLogger,
    detailed_call: DetailedCall,
    json_mode: bool,
    json_schema: type[BaseModel] | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    expected_visible_copy: str | None = None,
) -> tuple[ProviderResponse | None, dict[str, Any] | None, str | None]:
    request_sha256 = _sha256_text(query)
    unit = _attempt_unit(trial, stage=stage, request_sha256=request_sha256)
    unit_id = f"{trial['trial_id']}::{stage}"
    redacted_prompt = f"[REDACTED {stage} request sha256={request_sha256}]"
    for attempt in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
        try:
            response = detailed_call(
                route.provider,
                route.route,
                system_prompt,
                query,
                json_mode=json_mode,
                json_schema=json_schema,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed,
                reasoning_effort=None,
            )
            audit = _response_audit(response, route)
        except KeyboardInterrupt:
            _append_private_attempt(
                attempt_logger,
                provider=route.provider,
                model=route.route,
                unit_id=unit_id,
                unit=unit,
                attempt=attempt,
                max_attempts=MAX_TRANSPORT_ATTEMPTS,
                prompt_text=redacted_prompt,
                outcome="interrupted",
                error={
                    "exception_type": "KeyboardInterrupt",
                    "message": "Request interrupted before a response was retained.",
                },
            )
            raise
        except RouteIdentityError as error:
            _append_private_attempt(
                attempt_logger,
                provider=route.provider,
                model=route.route,
                unit_id=unit_id,
                unit=unit,
                attempt=attempt,
                max_attempts=MAX_TRANSPORT_ATTEMPTS,
                prompt_text=redacted_prompt,
                outcome="invalid_response",
                error={"exception_type": type(error).__name__, "message": str(error)},
            )
            raise
        except Exception as error:
            retryable, provenance = _transport_retryable(error, route)
            will_retry = retryable and attempt < MAX_TRANSPORT_ATTEMPTS
            _append_private_attempt(
                attempt_logger,
                provider=route.provider,
                model=route.route,
                unit_id=unit_id,
                unit=unit,
                attempt=attempt,
                max_attempts=MAX_TRANSPORT_ATTEMPTS,
                prompt_text=redacted_prompt,
                outcome="provider_error" if retryable else "invalid_response",
                error={
                    "exception_type": type(error).__name__,
                    "message": safe_error_message(error),
                    "provenance": provenance,
                },
                will_retry=will_retry,
            )
            if will_retry:
                continue
            if retryable:
                raise TransportExhaustedError(
                    f"{stage} exhausted same-route transport retries for {unit_id}."
                ) from error
            return None, None, f"{stage}:{type(error).__name__}"

        if response.truncated is not False:
            reason = f"{stage}:truncation_status_{response.truncated!r}"
            _append_private_attempt(
                attempt_logger,
                provider=route.provider,
                model=route.route,
                unit_id=unit_id,
                unit=unit,
                attempt=attempt,
                max_attempts=MAX_TRANSPORT_ATTEMPTS,
                prompt_text=redacted_prompt,
                outcome="invalid_response",
                generation_record=audit,
                error={"exception_type": "SemanticTruncation", "message": reason},
            )
            return response, audit, reason
        if not response.content.strip():
            reason = f"{stage}:empty_visible_response"
            _append_private_attempt(
                attempt_logger,
                provider=route.provider,
                model=route.route,
                unit_id=unit_id,
                unit=unit,
                attempt=attempt,
                max_attempts=MAX_TRANSPORT_ATTEMPTS,
                prompt_text=redacted_prompt,
                outcome="invalid_response",
                generation_record=audit,
                error={"exception_type": "EmptyVisibleResponse", "message": reason},
            )
            return response, audit, reason
        if json_schema is not None:
            try:
                parsed = json_schema.model_validate_json(response.content)
                if not isinstance(parsed, VisibleAnswer) or not parsed.response.strip():
                    raise ValueError("extractor response is empty")
                if expected_visible_copy is None or parsed.response != expected_visible_copy:
                    raise ValueError("extractor did not return an exact visible copy")
            except (ValidationError, ValueError, json.JSONDecodeError) as error:
                reason = f"{stage}:semantic_invalid:{type(error).__name__}"
                _append_private_attempt(
                    attempt_logger,
                    provider=route.provider,
                    model=route.route,
                    unit_id=unit_id,
                    unit=unit,
                    attempt=attempt,
                    max_attempts=MAX_TRANSPORT_ATTEMPTS,
                    prompt_text=redacted_prompt,
                    outcome="invalid_response",
                    generation_record=audit,
                    error={"exception_type": type(error).__name__, "message": reason},
                )
                return response, audit, reason
        _append_private_attempt(
            attempt_logger,
            provider=route.provider,
            model=route.route,
            unit_id=unit_id,
            unit=unit,
            attempt=attempt,
            max_attempts=MAX_TRANSPORT_ATTEMPTS,
            prompt_text=redacted_prompt,
            outcome="success",
            generation_record=audit,
        )
        return response, audit, None
    raise AssertionError("unreachable transport retry state")


def _base_result(trial: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        **deepcopy(dict(trial)),
        "subject": None,
        "extractor": None,
        "visible_response": None,
        "extracted_visible_response": None,
        "choice": None,
        "parse_reason": None,
        "final_line": None,
        "welfare_preserving": None,
        "status": "RUNNING",
        "scorable": False,
        "unscorable_reason": None,
    }


def execute_trial(
    trial: Mapping[str, Any],
    *,
    extractor_route: FrozenRoute,
    attempt_logger: DurableAttemptLogger,
    detailed_call: DetailedCall = api_call_detailed,
) -> dict[str, Any]:
    """Execute one trial; only transport failures can retry."""

    subject_route = _route_from_dict(trial["subject_route"])
    validate_frozen_route(extractor_route)
    result = _base_result(trial)
    settings = trial["generation_settings"]
    subject, subject_audit, invalid = _call_stage(
        stage="subject",
        trial=trial,
        route=subject_route,
        system_prompt=SUBJECT_SYSTEM_PROMPT,
        query=str(trial["prompt_text"]),
        attempt_logger=attempt_logger,
        detailed_call=detailed_call,
        json_mode=False,
        json_schema=None,
        temperature=float(settings["temperature"]),
        top_p=float(settings["top_p"]),
        max_tokens=int(settings["max_output_tokens"]),
        seed=int(settings["generation_seed"]),
    )
    result["subject"] = subject_audit
    result["visible_response"] = subject.content if subject is not None else None
    if invalid is not None or subject is None:
        result.update(
            status="INVALID",
            choice="INVALID",
            unscorable_reason=invalid or "subject:unavailable",
        )
        return result

    extractor_query = _extractor_prompt(subject.content)
    extractor, extractor_audit, invalid = _call_stage(
        stage="extractor",
        trial=trial,
        route=extractor_route,
        system_prompt=EXTRACTOR_SYSTEM_PROMPT,
        query=extractor_query,
        attempt_logger=attempt_logger,
        detailed_call=detailed_call,
        json_mode=True,
        json_schema=VisibleAnswer,
        temperature=0,
        top_p=1.0,
        max_tokens=EXTRACTOR_MAX_TOKENS,
        seed=int(trial["extractor_seed"]),
        expected_visible_copy=subject.content,
    )
    result["extractor"] = extractor_audit
    if invalid is not None or extractor is None:
        result.update(
            status="INVALID",
            choice="INVALID",
            unscorable_reason=invalid or "extractor:unavailable",
        )
        return result
    extracted = VisibleAnswer.model_validate_json(extractor.content)
    result["extracted_visible_response"] = extracted.response
    parsed = parse_exact_final_token(
        extracted.response,
        finish_reason=subject.finish_reason,
        truncated=subject.truncated,
    )
    result.update(
        choice=parsed.outcome.value,
        parse_reason=parsed.reason.value,
        final_line=parsed.final_line,
        welfare_preserving=welfare_preserving_choice(
            parsed, counterbalance_id=str(trial["counterbalance_id"])
        ),
    )
    if parsed.outcome == ParsedOutcome.INVALID:
        result.update(
            status="INVALID",
            unscorable_reason=f"parser:{parsed.reason.value}",
        )
        return result
    result.update(status="SCORED", scorable=True)
    return result


def summarize_results(path: str | Path) -> dict[str, Any]:
    results_path = Path(path)
    previous: str | None = None
    total = 0
    status_counts: dict[str, int] = {}
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ConfirmatoryPart1Error(
                        f"Invalid result JSON at line {line_number}."
                    ) from error
                if not isinstance(record, dict) or record.get("sequence") != line_number:
                    raise ConfirmatoryPart1Error("Result sequence is not contiguous.")
                if record.get("previous_record_sha256") != previous:
                    raise ConfirmatoryPart1Error("Result predecessor hash mismatch.")
                expected_hash = _record_hash(record)
                if record.get("record_sha256") != expected_hash:
                    raise ConfirmatoryPart1Error("Result record hash mismatch.")
                if record.get("schema_version") != RESULT_SCHEMA_VERSION:
                    raise ConfirmatoryPart1Error("Unsupported result schema version.")
                status = record.get("status")
                if status not in {"SCORED", "INVALID"}:
                    raise ConfirmatoryPart1Error("Result has an invalid status.")
                if (status == "SCORED") is not (record.get("scorable") is True):
                    raise ConfirmatoryPart1Error("Result status/scorable fields disagree.")
                choice = record.get("choice")
                if (status == "SCORED" and choice not in {"X", "Y"}) or (
                    status == "INVALID" and choice != "INVALID"
                ):
                    raise ConfirmatoryPart1Error(
                        "Completed result choice must be exact X, Y, or INVALID."
                    )
                previous = expected_hash
                total = line_number
                status_counts[str(status)] = status_counts.get(str(status), 0) + 1
    return {
        "path": str(results_path),
        "sha256": sha256_file(results_path) if results_path.exists() else _sha256_text(""),
        "total_results": total,
        "status_counts": status_counts,
        "hash_chain_status": "empty" if total == 0 else "verified",
        "last_record_sha256": previous,
    }


def _append_result(path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    summary = summarize_results(path)
    record = {
        **deepcopy(dict(result)),
        "sequence": summary["total_results"] + 1,
        "previous_record_sha256": summary["last_record_sha256"],
    }
    record["record_sha256"] = _record_hash(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _secure_file(path)
    return record


def _load_result_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_result_prefix(
    path: Path, schedule: Sequence[Mapping[str, Any]]
) -> int:
    summary = summarize_results(path)
    if summary["total_results"] > len(schedule):
        raise ConfirmatoryPart1Error("Result artifact exceeds the frozen schedule.")
    records = _load_result_records(path)
    for index, record in enumerate(records):
        trial = schedule[index]
        if record.get("trial_id") != trial["trial_id"]:
            raise ConfirmatoryPart1Error("Resume results are not a schedule prefix.")
        for field in (
            "phase",
            "execution_index",
            "root_id",
            "semantic_cluster_id",
            "game",
            "domain",
            "generation_block",
            "frame_id",
            "counterbalance_id",
            "prompt_hash",
            "root_content_hash",
            "generation_settings",
            "extractor_seed",
            "subject_route",
        ):
            if record.get(field) != trial[field]:
                raise ConfirmatoryPart1Error(
                    f"Resume result {record['trial_id']} has tampered {field}."
                )
        visible = record.get("visible_response")
        subject_audit = record.get("subject")
        if isinstance(visible, str) and (
            not isinstance(subject_audit, Mapping)
            or subject_audit.get("visible_response_sha256") != _sha256_text(visible)
        ):
            raise ConfirmatoryPart1Error("Visible response is not bound to subject audit.")
        extracted = record.get("extracted_visible_response")
        if extracted is not None and extracted != visible:
            raise ConfirmatoryPart1Error("Extractor output is not the exact visible response.")
        if record.get("choice") is not None and isinstance(extracted, str):
            parsed = parse_exact_final_token(
                extracted,
                finish_reason=subject_audit.get("finish_reason"),
                truncated=subject_audit.get("truncated"),
            )
            expected_mapping = welfare_preserving_choice(
                parsed, counterbalance_id=str(trial["counterbalance_id"])
            )
            if (
                record.get("choice") != parsed.outcome.value
                or record.get("parse_reason") != parsed.reason.value
                or record.get("final_line") != parsed.final_line
                or record.get("welfare_preserving") != expected_mapping
            ):
                raise ConfirmatoryPart1Error("Stored X/Y/INVALID parse was tampered.")
    return int(summary["total_results"])


def _expected_terminal_stages(result: Mapping[str, Any]) -> dict[str, str]:
    status = result.get("status")
    reason = result.get("unscorable_reason")
    if status == "SCORED" or (
        status == "INVALID" and isinstance(reason, str) and reason.startswith("parser:")
    ):
        return {"subject": "success", "extractor": "success"}
    if status == "INVALID" and isinstance(reason, str) and reason.startswith("subject:"):
        return {"subject": "invalid_response"}
    if status == "INVALID" and isinstance(reason, str) and reason.startswith("extractor:"):
        return {"subject": "success", "extractor": "invalid_response"}
    raise ConfirmatoryPart1Error("Result does not identify an exact terminal stage.")


def _stage_request_hash(
    *, stage: str, trial: Mapping[str, Any], result: Mapping[str, Any]
) -> str:
    if stage == "subject":
        return str(trial["prompt_hash"])
    visible = result.get("visible_response")
    if not isinstance(visible, str):
        raise ConfirmatoryPart1Error("Completed extractor stage lacks visible response.")
    return _sha256_text(_extractor_prompt(visible))


def _validate_attempt_result_reconciliation(
    *,
    attempts_path: Path,
    results_path: Path,
    schedule: Sequence[Mapping[str, Any]],
    extractor_route: Mapping[str, Any],
) -> None:
    results = _load_result_records(results_path)
    attempts = load_attempt_records(attempts_path)
    schedule_index = {
        str(trial["trial_id"]): index for index, trial in enumerate(schedule)
    }
    attempts_by_trial: dict[str, list[dict[str, Any]]] = {}
    immutable_unit_fields = (
        "phase",
        "execution_index",
        "root_id",
        "semantic_cluster_id",
        "game",
        "domain",
        "generation_block",
        "frame_id",
        "counterbalance_id",
        "prompt_sha256",
        "root_content_hash",
        "generation_seed",
        "extractor_seed",
    )
    for record in attempts:
        if record.get("experiment") != "part_1_confirmatory":
            raise ConfirmatoryPart1Error("Attempt log mixes experiments.")
        if record.get("raw_response") is not None or record.get("parsed_response") is not None:
            raise ConfirmatoryPart1Error("Attempt log contains prohibited raw response data.")
        unit = record.get("unit")
        if not isinstance(unit, dict):
            raise ConfirmatoryPart1Error("Attempt record lacks Part 1 unit metadata.")
        trial_id, stage = unit.get("trial_id"), unit.get("stage")
        if trial_id not in schedule_index or stage not in {"subject", "extractor"}:
            raise ConfirmatoryPart1Error("Attempt references unknown trial or stage.")
        trial = schedule[schedule_index[str(trial_id)]]
        expected_unit = _attempt_unit(
            trial,
            stage=str(stage),
            request_sha256=str(unit.get("stage_request_sha256")),
        )
        if any(unit.get(field) != expected_unit[field] for field in immutable_unit_fields):
            raise ConfirmatoryPart1Error(f"Attempt unit {trial_id} was tampered.")
        request_hash = _require_sha256(
            unit.get("stage_request_sha256"), label=f"{trial_id} {stage} request hash"
        )
        if record.get("prompt_text") != f"[REDACTED {stage} request sha256={request_hash}]":
            raise ConfirmatoryPart1Error("Attempt log contains unredacted request content.")
        attempts_by_trial.setdefault(str(trial_id), []).append(record)

    for index, result in enumerate(results):
        trial = schedule[index]
        trial_id = str(trial["trial_id"])
        expected = _expected_terminal_stages(result)
        by_stage: dict[str, list[dict[str, Any]]] = {}
        for record in attempts_by_trial.get(trial_id, []):
            by_stage.setdefault(str(record["unit"]["stage"]), []).append(record)
        if set(by_stage) != set(expected):
            raise ConfirmatoryPart1Error(
                f"Result {trial_id} lacks exact attempt-stage coverage."
            )
        for stage, expected_outcome in expected.items():
            stage_attempts = by_stage[stage]
            terminal = stage_attempts[-1]
            if terminal.get("outcome") != expected_outcome:
                raise ConfirmatoryPart1Error(
                    f"Result {trial_id} disagrees with terminal {stage} attempt."
                )
            request_hash = _stage_request_hash(stage=stage, trial=trial, result=result)
            if any(
                record["unit"].get("stage_request_sha256") != request_hash
                for record in stage_attempts
            ):
                raise ConfirmatoryPart1Error(f"{trial_id} has a mismatched {stage} request.")
            result_audit = result.get(stage)
            if terminal.get("generation_record") != result_audit:
                raise ConfirmatoryPart1Error(
                    f"Result {trial_id} disagrees with {stage} response audit."
                )
            expected_route = trial["subject_route"] if stage == "subject" else extractor_route
            if isinstance(result_audit, Mapping):
                if result_audit.get("frozen_route") != expected_route:
                    raise ConfirmatoryPart1Error(f"{trial_id} has mixed {stage} route.")
            elif expected_outcome == "success":
                raise ConfirmatoryPart1Error(f"Successful {stage} lacks provenance.")
            if any(
                record.get("provider") != expected_route["provider"]
                or record.get("model") != expected_route["route"]
                for record in stage_attempts
            ):
                raise ConfirmatoryPart1Error(f"{trial_id} has mixed {stage} attempts.")

    completed_count = len(results)
    unfinished = [
        record
        for trial_id, records in attempts_by_trial.items()
        if schedule_index[trial_id] >= completed_count
        for record in records
    ]
    if any(
        schedule_index[str(record["unit"]["trial_id"])] > completed_count
        for record in unfinished
    ):
        raise ConfirmatoryPart1Error("Attempt log skips beyond the result prefix.")
    if unfinished:
        if any(record["unit"].get("stage") != "subject" for record in unfinished):
            raise ConfirmatoryPart1Error(
                "Unfinished trial reached extraction; semantic replay is forbidden."
            )
        terminal = unfinished[-1]
        if terminal.get("outcome") != "interrupted" or terminal.get("generation_record") is not None:
            raise ConfirmatoryPart1Error(
                "Unfinished trial retained a semantic result; refusing replay."
            )
        if any(
            record.get("outcome") not in {"provider_error", "interrupted"}
            for record in unfinished
        ):
            raise ConfirmatoryPart1Error("Unfinished trial has a semantic attempt.")


def _run_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "part_1_confirmatory",
        "plan_sha256": plan["plan_sha256"],
        "bank_file_sha256": plan["bank_file_sha256"],
        "bank_manifest_sha256": plan["bank_manifest_sha256"],
        "freeze_state": deepcopy(plan["freeze_state"]),
        "routes": {
            "subject": deepcopy(plan["subject_route"]),
            "extractor": deepcopy(plan["extractor_route"]),
        },
        "protocol": deepcopy(plan["protocol"]),
        "resume_policy": "exact-prefix-hash-chain-crash-boundary-v1",
    }


def _write_run_metadata(
    path: Path,
    *,
    plan: Mapping[str, Any],
    results_path: Path,
    attempt_logger: DurableAttemptLogger,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_METADATA_SCHEMA_VERSION,
        "experiment": "part_1_confirmatory",
        "updated_at_utc": utc_now_iso(),
        "status": status,
        "run_contract": _run_contract(plan),
        "results": summarize_results(results_path),
        "attempt_log": attempt_logger.summary().to_metadata(),
        "error": error,
    }
    payload["metadata_sha256"] = metadata_payload_sha256(payload)
    _atomic_json_write(path, payload)
    _secure_file(results_path)
    _secure_file(attempt_logger.path)
    return payload


def run_frozen_plan(
    *,
    plan: Mapping[str, Any],
    bank_path: str | Path,
    output_directory: str | Path,
    resume: bool = False,
    detailed_call: DetailedCall = api_call_detailed,
) -> Path:
    """Run or strictly resume the complete frozen Part 1 plan."""

    loaded = load_production_bank(
        bank_path, expected_sha256=str(plan.get("bank_file_sha256", ""))
    )
    validate_execution_plan(plan, loaded)
    output_dir = _private_output_directory(output_directory, create=not resume)
    plan_path = output_dir / "part1_confirmatory_plan.json"
    results_path = output_dir / "part1_confirmatory_results.jsonl"
    metadata_path = output_dir / "part1_confirmatory_meta.json"
    attempts_path = attempt_log_path_for_csv(results_path)
    artifacts = (plan_path, results_path, metadata_path, attempts_path)

    if resume:
        if not all(path.is_file() for path in artifacts):
            raise ConfirmatoryPart1Error("Strict resume requires every run artifact.")
        _require_private_permissions(output_dir, artifacts)
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ConfirmatoryPart1Error("Strict resume plan differs from supplied plan.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_metadata_integrity(metadata, required=True)
        if metadata.get("run_contract") != _run_contract(plan):
            raise ConfirmatoryPart1Error("Strict resume contract mismatch.")
        if metadata.get("results") != summarize_results(results_path):
            raise ConfirmatoryPart1Error("Strict resume result integrity mismatch.")
        verify_attempt_log_metadata(
            attempts_path,
            metadata.get("attempt_log", {}),
            require_hash_chain=True,
        )
    else:
        existing = [path for path in artifacts if path.exists()]
        if existing:
            raise ConfirmatoryPart1Error("Fresh execution refuses to overwrite artifacts.")
        _atomic_json_write(plan_path, plan)
        results_path.touch(mode=0o600)
        attempts_path.touch(mode=0o600)
        _secure_file(results_path)
        _secure_file(attempts_path)

    attempt_logger = DurableAttemptLogger(
        attempts_path, experiment="part_1_confirmatory"
    )
    start_index = _validate_result_prefix(results_path, plan["schedule"])
    _validate_attempt_result_reconciliation(
        attempts_path=attempts_path,
        results_path=results_path,
        schedule=plan["schedule"],
        extractor_route=plan["extractor_route"],
    )
    _write_run_metadata(
        metadata_path,
        plan=plan,
        results_path=results_path,
        attempt_logger=attempt_logger,
        status="running",
    )
    extractor_route = _route_from_dict(plan["extractor_route"])
    try:
        for trial in plan["schedule"][start_index:]:
            result = execute_trial(
                trial,
                extractor_route=extractor_route,
                attempt_logger=attempt_logger,
                detailed_call=detailed_call,
            )
            _append_result(results_path, result)
            _write_run_metadata(
                metadata_path,
                plan=plan,
                results_path=results_path,
                attempt_logger=attempt_logger,
                status="running",
            )
    except BaseException as error:
        _write_run_metadata(
            metadata_path,
            plan=plan,
            results_path=results_path,
            attempt_logger=attempt_logger,
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {safe_error_message(error)}",
        )
        raise
    _write_run_metadata(
        metadata_path,
        plan=plan,
        results_path=results_path,
        attempt_logger=attempt_logger,
        status="complete",
    )
    return results_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the production Part 1 confirmatory protocol."
    )
    parser.add_argument("--bank", required=True, help="Approved 384-root JSON bank.")
    parser.add_argument("--bank-sha256", required=True, help="Exact bank file SHA-256.")
    parser.add_argument("--subject-provider", required=True)
    parser.add_argument("--subject-model", required=True, help="Exact verified model route.")
    parser.add_argument("--extractor-provider", required=True)
    parser.add_argument("--extractor-model", required=True, help="Exact verified route.")
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--primary-seed", type=int, default=DEFAULT_PRIMARY_SEED)
    parser.add_argument("--secondary-seed", type=int, default=DEFAULT_SECONDARY_SEED)
    parser.add_argument("--extractor-seed", type=int, default=DEFAULT_EXTRACTOR_SEED)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loaded = load_production_bank(args.bank, expected_sha256=args.bank_sha256)
    subject_route = freeze_verified_route(args.subject_provider, args.subject_model)
    extractor_route = freeze_verified_route(args.extractor_provider, args.extractor_model)
    plan = freeze_execution_plan(
        loaded,
        subject_route=subject_route,
        extractor_route=extractor_route,
        primary_seed=args.primary_seed,
        secondary_seed=args.secondary_seed,
        extractor_seed=args.extractor_seed,
    )
    results = run_frozen_plan(
        plan=plan,
        bank_path=args.bank,
        output_directory=args.output_directory,
        resume=args.resume,
    )
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BANK_SCHEMA_VERSION",
    "EXPECTED_PRIMARY_TRIALS",
    "EXPECTED_SECONDARY_TRIALS",
    "EXPECTED_TRIALS_PER_MODEL",
    "ConfirmatoryPart1Error",
    "FrozenRoute",
    "LoadedScenarioBank",
    "RouteIdentityError",
    "VisibleAnswer",
    "build_confirmatory_schedule",
    "execute_trial",
    "freeze_execution_plan",
    "freeze_verified_route",
    "load_production_bank",
    "main",
    "run_frozen_plan",
    "summarize_results",
    "validate_execution_plan",
]
