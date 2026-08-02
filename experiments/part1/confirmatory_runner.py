"""Production execution path for the frozen Part 1 confirmatory design.

The exploratory Part 1 runner is intentionally not imported.  This module
accepts only a byte-pinned 384-root registry that passes every production
content-review gate in :mod:`experiments.part1.confirmatory_design`, freezes
the complete self-directed primary schedule, and writes private, hash-chained
execution artifacts.  Semantic responses are never regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agents.agent_config import (
    require_fresh_route_verification,
    resolve_model_registry_entry,
)
from experiments.misc.attempt_log import (
    DurableAttemptLogger,
    attempt_log_path_for_csv,
    load_attempt_records,
    verify_attempt_log_metadata,
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
    COUNTERBALANCES,
    DOMAINS,
    EXPECTED_ROOT_COUNT,
    GAMES,
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
    parse_exact_final_token,
    validate_bank,
    validate_primary_schedule,
    welfare_preserving_choice,
)
from providers.api_call import (
    ProviderResponse,
    ResponseParseError,
    api_call_detailed,
    failure_provenance,
)


BANK_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 2
RESULT_SCHEMA_VERSION = 2
RUN_METADATA_SCHEMA_VERSION = 1
EXPECTED_PRIMARY_TRIALS = EXPECTED_ROOT_COUNT
EXPECTED_SECONDARY_TRIALS = 0
EXPECTED_TRIALS_PER_MODEL = EXPECTED_PRIMARY_TRIALS + EXPECTED_SECONDARY_TRIALS
EXECUTION_MODES = ("production", "sacrificial_smoke")
SMOKE_SELECTION_METHOD = (
    "sha256-rank-one-per-game-domain-balanced-counterbalance-v2"
)
SMOKE_EXCLUSION_REASON = "sacrificial_part1_full_path_smoke"
EXPECTED_SMOKE_TRIALS = len(GAMES) * len(DOMAINS)
DEFAULT_PRIMARY_SEED = 20_260_801
SUBJECT_SYSTEM_PROMPT = ""
MAX_TRANSPORT_ATTEMPTS = 3
SCHEDULE_PROTOCOL = "part1-self-direct-once-per-root-v2"
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
    try:
        require_fresh_route_verification(entry)
    except ValueError as error:
        raise RouteIdentityError(
            f"Route verification is not production-current: {provider}/{model}."
        ) from error
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
) -> dict[str, Any]:
    return {
        **asdict(trial),
        "phase": phase,
        "execution_mode": "production",
        "analysis_eligible": True,
        "execution_index": execution_index,
        "subject_route": subject_route.to_dict(),
    }


def build_confirmatory_schedule(
    loaded_bank: LoadedScenarioBank,
    *,
    subject_route: FrozenRoute,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
) -> list[dict[str, Any]]:
    """Construct exactly one self-directed trial for each approved root."""

    validate_bank(loaded_bank.roots, production=True).require_valid()
    validate_frozen_route(subject_route)
    primary = build_primary_schedule(
        loaded_bank.roots,
        base_seed=primary_seed,
        requested_provider=subject_route.provider,
        requested_model=subject_route.route,
        production=True,
    )
    validate_primary_schedule(primary, loaded_bank.roots, production=True).require_valid()
    schedule: list[dict[str, Any]] = []
    for trial in primary:
        schedule.append(
            _serialize_trial(
                trial,
                phase="primary",
                execution_index=len(schedule),
                subject_route=subject_route,
            )
        )
    if len(schedule) != EXPECTED_TRIALS_PER_MODEL:
        raise ConfirmatoryPart1Error("Part 1 schedule cardinality is not exact.")
    subject_seeds = [item["generation_settings"]["generation_seed"] for item in schedule]
    if len(set(subject_seeds)) != len(subject_seeds):
        raise ConfirmatoryPart1Error("Subject per-call generation seeds are not unique.")
    if any(item["frame_id"] != "self_direct" for item in schedule):
        raise ConfirmatoryPart1Error("Part 1 production contains a secondary role frame.")
    return schedule


def build_sacrificial_smoke_schedule(
    production_schedule: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select a deterministic, balanced, analysis-ineligible full-path smoke.

    One self-directed root is selected for every game/domain cell.  The frozen
    assignment covers each of the four counterbalances exactly three times.
    Purpose-bound SHA-256 ranking chooses among eligible frozen trials without
    observing any model response.
    """

    if len(production_schedule) != EXPECTED_TRIALS_PER_MODEL or len(
        {str(trial.get("trial_id")) for trial in production_schedule}
    ) != EXPECTED_TRIALS_PER_MODEL:
        raise ConfirmatoryPart1Error(
            "Sacrificial smoke selection requires the exact unique production schedule."
        )
    if {trial.get("execution_index") for trial in production_schedule} != set(
        range(EXPECTED_TRIALS_PER_MODEL)
    ):
        raise ConfirmatoryPart1Error(
            "Sacrificial smoke selection requires exact production execution indices."
        )

    game_index = {game: index for index, game in enumerate(GAMES)}
    domain_index = {domain: index for index, domain in enumerate(DOMAINS)}
    counterbalance_ids = tuple(item.counterbalance_id for item in COUNTERBALANCES)
    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for trial in production_schedule:
        game = str(trial.get("game"))
        domain = str(trial.get("domain"))
        counterbalance = str(trial.get("counterbalance_id"))
        if (
            game not in game_index
            or domain not in domain_index
            or counterbalance not in counterbalance_ids
            or trial.get("frame_id") != "self_direct"
            or trial.get("execution_mode") != "production"
            or trial.get("analysis_eligible") is not True
        ):
            raise ConfirmatoryPart1Error(
                "Production schedule contains an invalid sacrificial-smoke cell."
            )
        buckets.setdefault((game, domain, counterbalance), []).append(trial)

    selected: list[Mapping[str, Any]] = []
    for game in GAMES:
        for domain in DOMAINS:
            desired_counterbalance = counterbalance_ids[
                (game_index[game] * len(DOMAINS) + domain_index[domain])
                % len(counterbalance_ids)
            ]
            candidates = buckets.get((game, domain, desired_counterbalance), [])
            if not candidates:
                raise ConfirmatoryPart1Error(
                    "Sacrificial smoke schedule lacks exact balanced coverage."
                )
            ranked = sorted(
                candidates,
                key=lambda trial: (
                    _sha256_text(
                        f"{SMOKE_EXCLUSION_REASON}|{SMOKE_SELECTION_METHOD}|"
                        f"{trial['trial_id']}"
                    ),
                    str(trial["trial_id"]),
                ),
            )
            selected.append(ranked[0])

    selected.sort(key=lambda trial: int(trial["execution_index"]))
    result: list[dict[str, Any]] = []
    for trial in selected:
        smoke_trial = deepcopy(dict(trial))
        smoke_trial["execution_mode"] = "sacrificial_smoke"
        smoke_trial["analysis_eligible"] = False
        result.append(smoke_trial)
    if len(result) != EXPECTED_SMOKE_TRIALS or len(
        {str(trial["trial_id"]) for trial in result}
    ) != EXPECTED_SMOKE_TRIALS:
        raise ConfirmatoryPart1Error(
            "Sacrificial smoke schedule is not exact and unique."
        )
    cell_counts = Counter((str(row["game"]), str(row["domain"])) for row in result)
    counterbalance_counts = Counter(str(row["counterbalance_id"]) for row in result)
    if set(cell_counts.values()) != {1} or set(counterbalance_counts.values()) != {3}:
        raise ConfirmatoryPart1Error(
            "Sacrificial smoke cell/counterbalance strata are not exactly balanced."
        )
    return result


def _freeze_execution_plan(
    loaded_bank: LoadedScenarioBank,
    *,
    subject_route: FrozenRoute,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
    execution_mode: str,
    completed_smoke_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze one mode with exact content, routes, sources, and schedules."""

    if execution_mode not in EXECUTION_MODES:
        raise ConfirmatoryPart1Error(
            f"Unsupported Part 1 execution mode: {execution_mode}."
        )
    validate_frozen_route(subject_route)
    clean_commit = _require_clean_git_state()
    production_schedule = build_confirmatory_schedule(
        loaded_bank,
        subject_route=subject_route,
        primary_seed=primary_seed,
    )
    schedule = (
        production_schedule
        if execution_mode == "production"
        else build_sacrificial_smoke_schedule(production_schedule)
    )
    analysis_eligibility = (
        {
            "eligible": True,
            "purpose": "confirmatory_part1_inference",
            "exclusion_required": False,
        }
        if execution_mode == "production"
        else {
            "eligible": False,
            "purpose": SMOKE_EXCLUSION_REASON,
            "exclusion_required": True,
        }
    )
    payload: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "experiment": "part_1_confirmatory",
        "execution_mode": execution_mode,
        "analysis_eligibility": analysis_eligibility,
        "bank_file_sha256": loaded_bank.file_sha256,
        "bank_manifest_sha256": stable_json_hash(
            [asdict(root) for root in loaded_bank.roots]
        ),
        "root_count": EXPECTED_ROOT_COUNT,
        "primary_trial_count": EXPECTED_PRIMARY_TRIALS,
        "secondary_trial_count": EXPECTED_SECONDARY_TRIALS,
        "planned_trial_count": len(schedule),
        "schedule_protocol": SCHEDULE_PROTOCOL,
        "primary_seed_base": primary_seed,
        "freeze_state": {
            "git_commit": clean_commit,
            "git_dirty": False,
            "source_bundle": _production_source_bundle(),
            "environment_lock": _environment_lock(),
        },
        "subject_route": subject_route.to_dict(),
        "protocol": {
            "subject_system_prompt": SUBJECT_SYSTEM_PROMPT,
            "subject_system_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
            "subject_settings_source": "per_trial_generation_settings",
            "terminal_parser": "parse_exact_final_token",
            "parser_input_policy": "raw_visible_subject_response_only",
            "allowed_outcomes": ["X", "Y", "INVALID"],
            "hidden_reasoning_input": False,
            "semantic_retry": False,
            "transport_retry": "same_frozen_route_only",
            "max_transport_attempts": MAX_TRANSPORT_ATTEMPTS,
        },
        "schedule": schedule,
    }
    if execution_mode == "sacrificial_smoke":
        payload["smoke_design"] = {
            "selection_method": SMOKE_SELECTION_METHOD,
            "exclusion_reason": SMOKE_EXCLUSION_REASON,
            "expected_trials": EXPECTED_SMOKE_TRIALS,
            "frame": "self_direct",
            "games": list(GAMES),
            "domains": list(DOMAINS),
            "counterbalance_assignment": (
                "(game_index * 6 + domain_index) mod 4"
            ),
            "full_call_path": ["subject", "exact_xy_parser"],
        }
    elif completed_smoke_gate is not None:
        payload["completed_smoke_gate"] = deepcopy(dict(completed_smoke_gate))
    else:
        raise ConfirmatoryPart1Error(
            "Production Part 1 requires a revalidated completed full-path smoke gate."
        )
    payload["plan_sha256"] = stable_json_hash(payload)
    validate_execution_plan(payload, loaded_bank)
    return payload


def freeze_execution_plan(
    loaded_bank: LoadedScenarioBank,
    *,
    subject_route: FrozenRoute,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
    completed_smoke_directory: str | Path,
) -> dict[str, Any]:
    """Freeze the complete analysis-eligible Part 1 production plan."""

    try:
        require_fresh_route_verification(subject_route.identity)
    except ValueError as error:
        raise RouteIdentityError(
            "A new Part 1 production freeze requires fresh route evidence."
        ) from error
    smoke_gate = validate_completed_smoke_directory(
        loaded_bank,
        subject_route=subject_route,
        primary_seed=primary_seed,
        smoke_directory=completed_smoke_directory,
    )
    return _freeze_execution_plan(
        loaded_bank,
        subject_route=subject_route,
        primary_seed=primary_seed,
        execution_mode="production",
        completed_smoke_gate=smoke_gate,
    )


def freeze_sacrificial_smoke_plan(
    loaded_bank: LoadedScenarioBank,
    *,
    subject_route: FrozenRoute,
    primary_seed: int = DEFAULT_PRIMARY_SEED,
) -> dict[str, Any]:
    """Freeze a balanced full-path plan that can never be production data."""

    try:
        require_fresh_route_verification(subject_route.identity)
    except ValueError as error:
        raise RouteIdentityError(
            "A new Part 1 smoke freeze requires fresh route evidence."
        ) from error
    return _freeze_execution_plan(
        loaded_bank,
        subject_route=subject_route,
        primary_seed=primary_seed,
        execution_mode="sacrificial_smoke",
    )


def validate_execution_plan(
    plan: Mapping[str, Any], loaded_bank: LoadedScenarioBank
) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ConfirmatoryPart1Error("Unsupported Part 1 plan schema.")
    execution_mode = plan.get("execution_mode")
    if execution_mode not in EXECUTION_MODES:
        raise ConfirmatoryPart1Error("Part 1 plan lacks an exact execution mode.")
    expected_analysis_eligibility = (
        {
            "eligible": True,
            "purpose": "confirmatory_part1_inference",
            "exclusion_required": False,
        }
        if execution_mode == "production"
        else {
            "eligible": False,
            "purpose": SMOKE_EXCLUSION_REASON,
            "exclusion_required": True,
        }
    )
    if plan.get("analysis_eligibility") != expected_analysis_eligibility:
        raise ConfirmatoryPart1Error(
            "Part 1 analysis eligibility is missing or changed."
        )
    expected_smoke_design = {
        "selection_method": SMOKE_SELECTION_METHOD,
        "exclusion_reason": SMOKE_EXCLUSION_REASON,
        "expected_trials": EXPECTED_SMOKE_TRIALS,
        "frame": "self_direct",
        "games": list(GAMES),
        "domains": list(DOMAINS),
        "counterbalance_assignment": (
            "(game_index * 6 + domain_index) mod 4"
        ),
        "full_call_path": ["subject", "exact_xy_parser"],
    }
    if execution_mode == "sacrificial_smoke":
        if plan.get("smoke_design") != expected_smoke_design:
            raise ConfirmatoryPart1Error("Part 1 sacrificial smoke design changed.")
    elif "smoke_design" in plan:
        raise ConfirmatoryPart1Error("Production Part 1 plan contains smoke metadata.")
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
        "planned_trial_count": (
            EXPECTED_TRIALS_PER_MODEL
            if execution_mode == "production"
            else EXPECTED_SMOKE_TRIALS
        ),
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
    if execution_mode == "production":
        gate = plan.get("completed_smoke_gate")
        if not isinstance(gate, Mapping):
            raise ConfirmatoryPart1Error(
                "Production Part 1 plan lacks its completed smoke gate."
            )
        current_gate = validate_completed_smoke_directory(
            loaded_bank,
            subject_route=subject_route,
            primary_seed=int(plan["primary_seed_base"]),
            smoke_directory=str(gate.get("smoke_directory", "")),
        )
        if dict(gate) != current_gate:
            raise ConfirmatoryPart1Error(
                "Production Part 1 smoke gate changed or no longer validates."
            )
    elif "completed_smoke_gate" in plan:
        raise ConfirmatoryPart1Error(
            "Sacrificial Part 1 plan cannot carry a production smoke gate."
        )
    expected_protocol = freeze_protocol()
    if plan.get("protocol") != expected_protocol:
        raise ConfirmatoryPart1Error("Part 1 frozen execution protocol changed.")
    production_schedule = build_confirmatory_schedule(
        loaded_bank,
        subject_route=subject_route,
        primary_seed=int(plan["primary_seed_base"]),
    )
    expected_schedule = (
        production_schedule
        if execution_mode == "production"
        else build_sacrificial_smoke_schedule(production_schedule)
    )
    if plan.get("schedule") != expected_schedule:
        raise ConfirmatoryPart1Error("Part 1 schedule does not reconstruct exactly.")


def freeze_protocol() -> dict[str, Any]:
    """Return the exact execution protocol independently of a plan."""

    return {
        "subject_system_prompt": SUBJECT_SYSTEM_PROMPT,
        "subject_system_prompt_sha256": _sha256_text(SUBJECT_SYSTEM_PROMPT),
        "subject_settings_source": "per_trial_generation_settings",
        "terminal_parser": "parse_exact_final_token",
        "parser_input_policy": "raw_visible_subject_response_only",
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
        not isinstance(error, (ResponseParseError, TypeError, ValueError))
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


def _attempt_unit(
    trial: Mapping[str, Any], *, stage: str, request_sha256: str
) -> dict[str, Any]:
    return {
        "stage": stage,
        "trial_id": trial["trial_id"],
        "phase": trial["phase"],
        "execution_mode": trial["execution_mode"],
        "analysis_eligible": trial["analysis_eligible"],
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
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
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
                json_schema=None,
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
        "visible_response": None,
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
    attempt_logger: DurableAttemptLogger,
    detailed_call: DetailedCall = api_call_detailed,
) -> dict[str, Any]:
    """Execute one trial; only transport failures can retry."""

    subject_route = _route_from_dict(trial["subject_route"])
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

    parsed = parse_exact_final_token(
        subject.content,
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
            "execution_mode",
            "analysis_eligible",
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
        if record.get("choice") is not None and isinstance(visible, str):
            parsed = parse_exact_final_token(
                visible,
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
        return {"subject": "success"}
    if status == "INVALID" and isinstance(reason, str) and reason.startswith("subject:"):
        return {"subject": "invalid_response"}
    raise ConfirmatoryPart1Error("Result does not identify an exact terminal stage.")


def _stage_request_hash(
    *, stage: str, trial: Mapping[str, Any], result: Mapping[str, Any]
) -> str:
    if stage == "subject":
        return str(trial["prompt_hash"])
    raise ConfirmatoryPart1Error(f"Unknown Part 1 attempt stage: {stage}.")


def _validate_attempt_result_reconciliation(
    *,
    attempts_path: Path,
    results_path: Path,
    schedule: Sequence[Mapping[str, Any]],
) -> None:
    results = _load_result_records(results_path)
    attempts = load_attempt_records(attempts_path)
    schedule_index = {
        str(trial["trial_id"]): index for index, trial in enumerate(schedule)
    }
    attempts_by_trial: dict[str, list[dict[str, Any]]] = {}
    immutable_unit_fields = (
        "phase",
        "execution_mode",
        "analysis_eligible",
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
        if trial_id not in schedule_index or stage != "subject":
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
            expected_route = trial["subject_route"]
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
            raise ConfirmatoryPart1Error("Unfinished trial has an unknown stage.")
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
    contract = {
        "schema_version": 1,
        "experiment": "part_1_confirmatory",
        "execution_mode": plan["execution_mode"],
        "analysis_eligibility": deepcopy(plan["analysis_eligibility"]),
        "plan_sha256": plan["plan_sha256"],
        "bank_file_sha256": plan["bank_file_sha256"],
        "bank_manifest_sha256": plan["bank_manifest_sha256"],
        "freeze_state": deepcopy(plan["freeze_state"]),
        "routes": {
            "subject": deepcopy(plan["subject_route"]),
        },
        "protocol": deepcopy(plan["protocol"]),
        "resume_policy": "exact-prefix-hash-chain-crash-boundary-v1",
    }
    if plan["execution_mode"] == "sacrificial_smoke":
        contract["smoke_design"] = deepcopy(plan["smoke_design"])
    return contract


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


def _smoke_exclusion_payload(
    *, plan: Mapping[str, Any], results_path: Path, metadata_path: Path
) -> dict[str, Any]:
    if plan.get("execution_mode") != "sacrificial_smoke":
        raise ConfirmatoryPart1Error(
            "Smoke exclusion marker cannot bind a production plan."
        )
    results = summarize_results(results_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata_integrity(metadata, required=True)
    if metadata.get("status") != "complete":
        raise ConfirmatoryPart1Error(
            "Sacrificial smoke exclusion can only bind a complete run."
        )
    if results["total_results"] != EXPECTED_SMOKE_TRIALS:
        raise ConfirmatoryPart1Error(
            "Sacrificial smoke exclusion requires the complete exact schedule."
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scope": "canonical_analysis_and_validation",
        "reason": SMOKE_EXCLUSION_REASON,
        "execution_mode": "sacrificial_smoke",
        "analysis_eligible": False,
        "plan_sha256": plan["plan_sha256"],
        "schedule_sha256": stable_json_hash(plan["schedule"]),
        "bank_file_sha256": plan["bank_file_sha256"],
        "results_sha256": results["sha256"],
        "total_results": results["total_results"],
        "metadata_sha256": metadata["metadata_sha256"],
    }
    payload["marker_payload_sha256"] = stable_json_hash(payload)
    return payload


def _write_or_verify_smoke_exclusion(
    marker_path: Path,
    *,
    plan: Mapping[str, Any],
    results_path: Path,
    metadata_path: Path,
) -> None:
    expected = _smoke_exclusion_payload(
        plan=plan, results_path=results_path, metadata_path=metadata_path
    )
    if marker_path.exists():
        try:
            actual = json.loads(marker_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConfirmatoryPart1Error(
                "Sacrificial smoke exclusion marker is invalid."
            ) from error
        if actual != expected:
            raise ConfirmatoryPart1Error(
                "Sacrificial smoke exclusion marker does not match its artifacts."
            )
    else:
        _atomic_json_write(marker_path, expected)
    _secure_file(marker_path)


def validate_completed_smoke_directory(
    loaded_bank: LoadedScenarioBank,
    *,
    subject_route: FrozenRoute,
    primary_seed: int,
    smoke_directory: str | Path,
) -> dict[str, Any]:
    """Revalidate the exact operational smoke required by a production freeze."""

    output_dir = _private_output_directory(smoke_directory, create=False)
    plan_path = output_dir / "part1_confirmatory_plan.json"
    results_path = output_dir / "part1_confirmatory_results.jsonl"
    metadata_path = output_dir / "part1_confirmatory_meta.json"
    attempts_path = attempt_log_path_for_csv(results_path)
    exclusion_path = output_dir / "part1_confirmatory_analysis_exclude.json"
    artifacts = (
        plan_path,
        results_path,
        metadata_path,
        attempts_path,
        exclusion_path,
    )
    _require_private_permissions(output_dir, artifacts)
    expected_plan = _freeze_execution_plan(
        loaded_bank,
        subject_route=subject_route,
        primary_seed=primary_seed,
        execution_mode="sacrificial_smoke",
    )
    try:
        persisted_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryPart1Error(
            "Completed Part 1 smoke contains invalid JSON."
        ) from error
    if persisted_plan != expected_plan:
        raise ConfirmatoryPart1Error(
            "Completed Part 1 smoke does not match the exact production inputs."
        )
    validate_execution_plan(persisted_plan, loaded_bank)
    validate_metadata_integrity(metadata, required=True)
    if metadata.get("status") != "complete" or metadata.get(
        "run_contract"
    ) != _run_contract(expected_plan):
        raise ConfirmatoryPart1Error(
            "Completed Part 1 smoke metadata is incomplete or mismatched."
        )
    results = summarize_results(results_path)
    records = _load_result_records(results_path)
    if (
        results["total_results"] != EXPECTED_SMOKE_TRIALS
        or metadata.get("results") != results
        or len(records) != EXPECTED_SMOKE_TRIALS
        or any(
            _expected_terminal_stages(record) != {"subject": "success"}
            for record in records
        )
    ):
        raise ConfirmatoryPart1Error(
            "Part 1 production requires every smoke trial to complete the subject call."
        )
    attempt_summary = verify_attempt_log_metadata(
        attempts_path,
        metadata.get("attempt_log", {}),
        require_hash_chain=True,
    )
    _validate_attempt_result_reconciliation(
        attempts_path=attempts_path,
        results_path=results_path,
        schedule=expected_plan["schedule"],
    )
    expected_exclusion = _smoke_exclusion_payload(
        plan=expected_plan,
        results_path=results_path,
        metadata_path=metadata_path,
    )
    if exclusion != expected_exclusion:
        raise ConfirmatoryPart1Error(
            "Completed Part 1 smoke exclusion marker is stale or tampered."
        )
    gate: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "part_1_confirmatory",
        "status": "validated_complete_full_path_smoke",
        "smoke_directory": str(output_dir),
        "smoke_plan_sha256": expected_plan["plan_sha256"],
        "smoke_schedule_sha256": stable_json_hash(expected_plan["schedule"]),
        "bank_file_sha256": loaded_bank.file_sha256,
        "bank_manifest_sha256": expected_plan["bank_manifest_sha256"],
        "route_identity_sha256": {
            "subject": stable_json_hash(expected_plan["subject_route"])
        },
        "freeze_state": deepcopy(expected_plan["freeze_state"]),
        "protocol_sha256": stable_json_hash(expected_plan["protocol"]),
        "result_count": EXPECTED_SMOKE_TRIALS,
        "results_sha256": results["sha256"],
        "metadata_sha256": metadata["metadata_sha256"],
        "attempt_log_sha256": attempt_summary.sha256,
        "exclusion_marker_sha256": sha256_file(exclusion_path),
    }
    gate["smoke_gate_sha256"] = stable_json_hash(gate)
    return gate


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
    exclusion_path = output_dir / "part1_confirmatory_analysis_exclude.json"
    artifacts = (plan_path, results_path, metadata_path, attempts_path)
    persisted_metadata: Mapping[str, Any] | None = None

    if resume:
        if not all(path.is_file() for path in artifacts):
            raise ConfirmatoryPart1Error("Strict resume requires every run artifact.")
        _require_private_permissions(output_dir, artifacts)
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ConfirmatoryPart1Error("Strict resume plan differs from supplied plan.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        persisted_metadata = metadata
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
        if plan["execution_mode"] == "production":
            if exclusion_path.exists():
                raise ConfirmatoryPart1Error(
                    "Production Part 1 output contains a smoke exclusion marker."
                )
        elif metadata.get("status") == "complete":
            if not exclusion_path.is_file():
                raise ConfirmatoryPart1Error(
                    "Completed sacrificial smoke is missing its exclusion marker."
                )
            _require_private_permissions(output_dir, (exclusion_path,))
            _write_or_verify_smoke_exclusion(
                exclusion_path,
                plan=plan,
                results_path=results_path,
                metadata_path=metadata_path,
            )
        elif exclusion_path.exists():
            raise ConfirmatoryPart1Error(
                "Incomplete sacrificial smoke contains a premature exclusion marker."
            )
    else:
        existing = [path for path in (*artifacts, exclusion_path) if path.exists()]
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
    )
    if (
        resume
        and start_index == len(plan["schedule"])
        and persisted_metadata is not None
        and persisted_metadata.get("status") == "complete"
    ):
        return results_path
    _write_run_metadata(
        metadata_path,
        plan=plan,
        results_path=results_path,
        attempt_logger=attempt_logger,
        status="running",
    )
    try:
        for trial in plan["schedule"][start_index:]:
            result = execute_trial(
                trial,
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
    if plan["execution_mode"] == "sacrificial_smoke":
        _write_or_verify_smoke_exclusion(
            exclusion_path,
            plan=plan,
            results_path=results_path,
            metadata_path=metadata_path,
        )
    return results_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the production Part 1 confirmatory protocol or its explicitly "
            "analysis-ineligible sacrificial full-path smoke."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("production", "sacrificial-smoke"),
        default="production",
    )
    parser.add_argument("--bank", required=True, help="Approved 384-root JSON bank.")
    parser.add_argument("--bank-sha256", required=True, help="Exact bank file SHA-256.")
    parser.add_argument("--subject-provider", required=True)
    parser.add_argument("--subject-model", required=True, help="Exact verified model route.")
    parser.add_argument(
        "--completed-smoke-directory",
        help=(
            "Required for production: exact completed sacrificial-smoke directory "
            "to revalidate and bind into the production freeze."
        ),
    )
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--primary-seed", type=int, default=DEFAULT_PRIMARY_SEED)
    parser.add_argument("--resume", action="store_true")
    return parser


def _load_strict_resume_plan(
    *,
    loaded_bank: LoadedScenarioBank,
    output_directory: str | Path,
    mode: str,
    subject_provider: str,
    subject_model: str,
    primary_seed: int,
    completed_smoke_directory: str | Path | None,
) -> dict[str, Any]:
    """Reconstruct an immutable run without reapplying wall-clock freshness.

    Freshness is a gate on creating a new freeze.  Once frozen, resume remains
    bound to the exact registry identity embedded in the hash-checked plan and
    still revalidates the current registry bytes, sources, bank, and smoke.
    """

    output = _private_output_directory(output_directory, create=False)
    plan_path = output / "part1_confirmatory_plan.json"
    if not plan_path.is_file():
        raise ConfirmatoryPart1Error("Strict resume is missing its frozen plan.")
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryPart1Error("Strict resume plan is not valid JSON.") from error
    if not isinstance(plan, dict):
        raise ConfirmatoryPart1Error("Strict resume plan root must be an object.")
    validate_execution_plan(plan, loaded_bank)
    expected_mode = "sacrificial_smoke" if mode == "sacrificial-smoke" else "production"
    expected_cli = {
        "execution_mode": expected_mode,
        "subject_provider": subject_provider,
        "subject_model": subject_model,
        "primary_seed": primary_seed,
    }
    actual_cli = {
        "execution_mode": plan.get("execution_mode"),
        "subject_provider": plan.get("subject_route", {}).get("provider"),
        "subject_model": plan.get("subject_route", {}).get("route"),
        "primary_seed": plan.get("primary_seed_base"),
    }
    if actual_cli != expected_cli:
        raise ConfirmatoryPart1Error(
            "Strict resume CLI identity, mode, or seeds differ from the frozen plan."
        )
    if expected_mode == "production":
        frozen_smoke = plan.get("completed_smoke_gate", {}).get("smoke_directory")
        if (
            completed_smoke_directory is None
            or Path(str(frozen_smoke)).resolve()
            != Path(completed_smoke_directory).resolve()
        ):
            raise ConfirmatoryPart1Error(
                "Strict resume smoke directory differs from the frozen plan."
            )
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode == "production" and not args.completed_smoke_directory:
        parser.error("--completed-smoke-directory is required in production mode")
    if args.mode == "sacrificial-smoke" and args.completed_smoke_directory:
        parser.error(
            "--completed-smoke-directory is not valid in sacrificial-smoke mode"
        )
    loaded = load_production_bank(args.bank, expected_sha256=args.bank_sha256)
    if args.resume:
        plan = _load_strict_resume_plan(
            loaded_bank=loaded,
            output_directory=args.output_directory,
            mode=args.mode,
            subject_provider=args.subject_provider,
            subject_model=args.subject_model,
            primary_seed=args.primary_seed,
            completed_smoke_directory=args.completed_smoke_directory,
        )
    elif args.mode == "production":
        subject_route = freeze_verified_route(args.subject_provider, args.subject_model)
        plan = freeze_execution_plan(
            loaded,
            subject_route=subject_route,
            primary_seed=args.primary_seed,
            completed_smoke_directory=args.completed_smoke_directory,
        )
    else:
        subject_route = freeze_verified_route(args.subject_provider, args.subject_model)
        plan = freeze_sacrificial_smoke_plan(
            loaded,
            subject_route=subject_route,
            primary_seed=args.primary_seed,
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
    "EXPECTED_SMOKE_TRIALS",
    "EXPECTED_TRIALS_PER_MODEL",
    "ConfirmatoryPart1Error",
    "FrozenRoute",
    "LoadedScenarioBank",
    "RouteIdentityError",
    "build_confirmatory_schedule",
    "build_sacrificial_smoke_schedule",
    "execute_trial",
    "freeze_execution_plan",
    "freeze_sacrificial_smoke_plan",
    "freeze_verified_route",
    "load_production_bank",
    "main",
    "run_frozen_plan",
    "summarize_results",
    "validate_execution_plan",
]
