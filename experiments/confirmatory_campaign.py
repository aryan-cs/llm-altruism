"""Strict, isolated orchestration for the confirmatory three-part campaign.

This module does not use the legacy campaign planner.  It treats the native
confirmatory Part 0/1 runners and the production-shaped Part 2 runner as sealed
entrypoints, freezes every input before writing a manifest, and releases no
scientific job until its target's matching sacrificial smoke has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from agents.agent_config import load_model_cohort, require_fresh_route_verification
from experiments.confirmatory_budget import (
    build_frozen_budget,
    create_ledger,
    record_attempt,
    validate_frozen_budget,
    validate_ledger,
)
from experiments.misc.attempt_log import attempt_log_path_for_csv, verify_attempt_log_metadata
from experiments.misc.run_metadata import (
    git_commit,
    git_dirty,
    metadata_payload_sha256,
    sha256_file,
    source_bundle_metadata,
    stable_json_hash,
    validate_metadata_integrity,
)
from experiments.part0 import confirmatory_runner as part0_runner
from experiments.part1 import confirmatory_runner as part1_runner
from experiments.part2 import part_2
from experiments.part2.part_2 import DEFAULT_COLLAPSE_DEATH_RATE, _initial_resource_units
from experiments.misc.wizard import SocietyConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = REPO_ROOT / "data" / "private" / "confirmatory_campaigns"
DEFAULT_COHORTS = ("current_sota", "historical")
MANIFEST_SCHEMA_VERSION = 1
ENDPOINT_EVIDENCE_SCHEMA_VERSION = 2
VARIANCE_GATE_SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
DEFAULT_PART2_GENERATION_SEED_BASE = 2_026_080_100
DEFAULT_PART2_ENVIRONMENT_SEED_BASE = 1_026_080_100
PART2_SMOKE_GENERATION_SEED = 3_026_080_100
PART2_SMOKE_ENVIRONMENT_SEED = 3_026_080_200
# Archived two-stage constants are retained only so historical manifests can be
# revalidated for legacy data locks. New campaign planning and execution reject
# every two-stage scientific stage.
VARIANCE_PILOT_REPLICATES = 8
MIN_BASELINE_REPLICATES = 20
MAX_BASELINE_REPLICATES = 40
FIXED_PART2_REPLICATES = 24
CAMPAIGN_SCHEDULING_SEED = 2_026_080_202
CAMPAIGN_SCHEDULING_PROTOCOL = (
    "sha256_seeded_target_then_part_blocks_smoke_before_science_v1"
)
_CAMPAIGN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_TARGET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_JOB_STATUSES = {
    "pending",
    "running",
    "complete",
    "failed",
    "blocked_smoke",
    "blocked_route_health",
}


class ConfirmatoryCampaignError(RuntimeError):
    """A pinned input, immutable plan, execution, or artifact failed closed."""


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    timed_out: bool = False
    error: str | None = None


ProcessRunner = Callable[[Sequence[str], Path, Mapping[str, str], Path, int], ProcessResult]
ArtifactResolver = Callable[[Mapping[str, Any], Mapping[str, Any] | None], dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ConfirmatoryCampaignError(f"{label} must be a lowercase SHA-256.")
    return value


def _require_campaign_id(value: str) -> str:
    if _CAMPAIGN_ID_RE.fullmatch(value) is None:
        raise ConfirmatoryCampaignError(
            "campaign id must use 1-80 letters, digits, '.', '-', or '_'"
        )
    return value


def _campaign_directory(campaign_id: str) -> Path:
    root = CAMPAIGN_ROOT.resolve()
    result = (root / _require_campaign_id(campaign_id)).resolve()
    if result.parent != root:
        raise ConfirmatoryCampaignError("campaign path escaped its private root.")
    return result


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
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


def _load_json(path: str | Path, *, label: str) -> dict[str, Any]:
    artifact = Path(path)
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key: {key}")
            value[key] = item
        return value

    def reject_nonfinite_constant(constant: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {constant}")

    try:
        value = json.loads(
            artifact.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except FileNotFoundError as error:
        raise ConfirmatoryCampaignError(f"{label} does not exist: {artifact}") from error
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ConfirmatoryCampaignError(f"{label} is not valid strict UTF-8 JSON.") from error
    if not isinstance(value, dict):
        raise ConfirmatoryCampaignError(f"{label} root must be an object.")
    return value


def _hash_pinned_json(
    path: str | Path, *, expected_sha256: str, label: str
) -> tuple[Path, dict[str, Any], str]:
    expected = _require_sha256(expected_sha256, label=f"{label} hash")
    artifact = Path(path).resolve()
    if not artifact.is_file():
        raise ConfirmatoryCampaignError(f"{label} does not exist: {artifact}")
    actual = sha256_file(artifact)
    if actual != expected:
        raise ConfirmatoryCampaignError(
            f"{label} hash mismatch: expected {expected}, found {actual}."
        )
    return artifact, _load_json(artifact, label=label), actual


def _source_paths() -> list[Path]:
    paths = [
        Path(__file__),
        REPO_ROOT / "experiments" / "part0" / "confirmatory_runner.py",
        REPO_ROOT / "experiments" / "part0" / "stimulus_registry.py",
        REPO_ROOT / "experiments" / "part1" / "confirmatory_runner.py",
        REPO_ROOT / "experiments" / "part1" / "confirmatory_design.py",
        REPO_ROOT / "experiments" / "part2" / "part_2.py",
        REPO_ROOT / "experiments" / "part2" / "part_2_prompt.json",
        REPO_ROOT / "experiments" / "misc" / "attempt_log.py",
        REPO_ROOT / "experiments" / "misc" / "final_answer.py",
        REPO_ROOT / "experiments" / "misc" / "inference_hub_discovery.py",
        REPO_ROOT / "experiments" / "misc" / "preflight.py",
        REPO_ROOT / "experiments" / "misc" / "prompt_loader.py",
        REPO_ROOT / "experiments" / "misc" / "result_writer.py",
        REPO_ROOT / "experiments" / "misc" / "run_metadata.py",
        REPO_ROOT / "analysis" / "part2_dynamics.py",
        REPO_ROOT / "analysis" / "part2_confirmatory.py",
        REPO_ROOT / "analysis" / "confirmatory_estimators.py",
        REPO_ROOT / "analysis" / "confirmatory_judge_adapter.py",
        REPO_ROOT / "analysis" / "confirmatory_data_lock.py",
        REPO_ROOT / "analysis" / "summarize_results.py",
        REPO_ROOT / "analysis" / "judge_audit.py",
        REPO_ROOT / "analysis" / "validation.py",
        REPO_ROOT / "docs" / "CONFIRMATORY_PROTOCOL.md",
        REPO_ROOT / "docs" / "CONFIRMATORY_CAMPAIGN.md",
        REPO_ROOT / "agents" / "agent_2.py",
        REPO_ROOT / "agents" / "base_agent.py",
        REPO_ROOT / "agents" / "agent_config.py",
        REPO_ROOT / "agents" / "agent_config.registry.json",
        REPO_ROOT / "providers" / "api_call.py",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "uv.lock",
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ConfirmatoryCampaignError(
            "confirmatory campaign source bundle is incomplete: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _execution_freeze() -> dict[str, Any]:
    commit = git_commit()
    dirty = git_dirty()
    if not isinstance(commit, str) or _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise ConfirmatoryCampaignError("Campaign requires an exact Git commit.")
    if dirty is not False:
        raise ConfirmatoryCampaignError("Campaign requires a clean Git worktree.")
    return {
        "schema_version": 1,
        "git_commit": commit,
        "git_dirty": False,
        "python_executable": str(Path(sys.executable).resolve()),
        "source_bundle": source_bundle_metadata(_source_paths()),
        "dependency_lock": source_bundle_metadata(
            [REPO_ROOT / "pyproject.toml", REPO_ROOT / "uv.lock"]
        ),
    }


def _load_exact_cohort_union(
    cohort_ids: Sequence[str],
    *,
    enforce_freshness: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, str, str]:
    if not cohort_ids or len(set(cohort_ids)) != len(cohort_ids):
        raise ConfirmatoryCampaignError("Requested cohorts must be nonempty and unique.")
    cohorts: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_routes: set[str] = set()
    registry_versions: set[str] = set()
    registry_hashes: set[str] = set()
    routing_roster_hashes: set[str] = set()
    policy_hashes: set[str] = set()
    bundle_hashes: set[str] = set()
    for cohort_id in cohort_ids:
        try:
            cohort = load_model_cohort(cohort_id)
        except (KeyError, ValueError) as error:
            raise ConfirmatoryCampaignError(f"Invalid model cohort {cohort_id!r}: {error}") from error
        registry_versions.add(str(cohort["registry_version"]))
        registry_hashes.add(str(cohort["registry_hash"]))
        routing_roster_hashes.add(str(cohort.get("routing_roster_hash", "")))
        policy = cohort.get("route_verification_policy")
        bundle = cohort.get("verification_bundle")
        if not isinstance(policy, Mapping) or not isinstance(bundle, Mapping):
            raise ConfirmatoryCampaignError(
                f"Cohort {cohort_id} lacks verification policy/bundle attestation."
            )
        policy_hashes.add(stable_json_hash(policy))
        bundle_hashes.add(stable_json_hash(bundle))
        member_ids: list[str] = []
        for raw_target in cohort["targets"]:
            target = deepcopy(dict(raw_target))
            target_id = str(target.get("id", ""))
            route = str(target.get("route", ""))
            if target.get("provider") != "inference_hub":
                raise ConfirmatoryCampaignError(
                    f"Target {target_id} is not routed through inference_hub."
                )
            if target.get("verification_status") != "verified" or not isinstance(
                target.get("verification_evidence"), Mapping
            ):
                raise ConfirmatoryCampaignError(f"Target {target_id} is not verified.")
            if enforce_freshness:
                try:
                    require_fresh_route_verification(
                        {
                            **target,
                            "routing_roster_hash": cohort.get("routing_roster_hash"),
                            "route_verification_policy": deepcopy(dict(policy)),
                            "verification_bundle": deepcopy(dict(bundle)),
                        }
                    )
                except ValueError as error:
                    raise ConfirmatoryCampaignError(
                        f"Target {target_id} route verification is not fresh: {error}"
                    ) from error
            if not target_id or not route:
                raise ConfirmatoryCampaignError("Cohort contains an empty target or route.")
            if _TARGET_ID_RE.fullmatch(target_id) is None:
                raise ConfirmatoryCampaignError(
                    f"Target id is unsafe for deterministic artifacts: {target_id!r}."
                )
            if target_id in seen_ids:
                raise ConfirmatoryCampaignError(
                    f"Target {target_id} is duplicated across requested cohorts."
                )
            if route in seen_routes:
                raise ConfirmatoryCampaignError(
                    f"Route {route} is duplicated across requested cohorts."
                )
            seen_ids.add(target_id)
            seen_routes.add(route)
            member_ids.append(target_id)
            targets.append(target)
        cohorts.append(
            {
                "id": str(cohort["id"]),
                "version": str(cohort["version"]),
                "target_ids": member_ids,
            }
        )
    if (
        len(registry_versions) != 1
        or len(registry_hashes) != 1
        or routing_roster_hashes == {""}
        or len(routing_roster_hashes) != 1
        or len(policy_hashes) != 1
        or len(bundle_hashes) != 1
    ):
        raise ConfirmatoryCampaignError(
            "Requested cohorts do not share one registry, routing roster, policy, "
            "and verification bundle."
        )
    return (
        cohorts,
        targets,
        next(iter(registry_versions)),
        next(iter(registry_hashes)),
        next(iter(routing_roster_hashes)),
    )


def _validate_endpoint_evidence(
    payload: Mapping[str, Any],
    *,
    cohorts: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    registry_version: str,
    routing_roster_hash: str,
) -> None:
    recorded_hash = payload.get("bundle_sha256")
    expected_hash = stable_json_hash(
        {key: value for key, value in payload.items() if key != "bundle_sha256"}
    )
    if payload.get("schema_version") != ENDPOINT_EVIDENCE_SCHEMA_VERSION:
        raise ConfirmatoryCampaignError("Unsupported endpoint-evidence schema.")
    if (
        payload.get("status") != "verified"
        or payload.get("verified_target_count") != len(targets)
        or payload.get("rejected_targets") != []
    ):
        raise ConfirmatoryCampaignError(
            "Endpoint evidence is incomplete or contains a failed frozen-panel route."
        )
    if recorded_hash != expected_hash:
        raise ConfirmatoryCampaignError("Endpoint-evidence bundle hash is invalid.")
    if payload.get("registry_version") != registry_version:
        raise ConfirmatoryCampaignError(
            "Endpoint evidence is bound to a different registry version."
        )
    if payload.get("routing_roster_sha256") != routing_roster_hash:
        raise ConfirmatoryCampaignError(
            "Endpoint evidence is bound to a different routing roster."
        )
    if payload.get("cohorts") != list(cohorts):
        raise ConfirmatoryCampaignError(
            "Endpoint evidence does not exactly cover requested cohorts."
        )
    census = payload.get("catalog_census")
    if (
        not isinstance(census, list)
        or payload.get("catalog_census_sha256") != stable_json_hash(census)
        or not census
    ):
        raise ConfirmatoryCampaignError("Endpoint evidence lacks a hash-bound catalog census.")
    allowed_census_decisions = {
        "included_frozen_panel",
        "excluded_catalog_api_disagreement",
        "excluded_non_chat_mode",
        "excluded_outside_frozen_panel",
    }
    census_by_route: dict[str, Mapping[str, Any]] = {}
    for row in census:
        if (
            not isinstance(row, Mapping)
            or set(row) != {"route", "decision", "target_id"}
            or not isinstance(row.get("route"), str)
            or row.get("decision") not in allowed_census_decisions
            or row["route"] in census_by_route
        ):
            raise ConfirmatoryCampaignError("Catalog census row is invalid or duplicated.")
        census_by_route[str(row["route"])] = row
    ledger_reference = payload.get("discovery_attempt_ledger")
    if not isinstance(ledger_reference, Mapping) or set(ledger_reference) != {
        "path", "sha256", "ledger_sha256", "record_count"
    }:
        raise ConfirmatoryCampaignError("Endpoint evidence lacks its discovery ledger.")
    discovery_ledger_path = Path(str(ledger_reference["path"]))
    discovery_ledger = _load_json(
        discovery_ledger_path, label="InferenceHub discovery attempt ledger"
    )
    if (
        not discovery_ledger_path.is_file()
        or sha256_file(discovery_ledger_path) != ledger_reference["sha256"]
        or discovery_ledger.get("ledger_sha256") != ledger_reference["ledger_sha256"]
        or discovery_ledger.get("ledger_sha256")
        != stable_json_hash(
            {
                key: value
                for key, value in discovery_ledger.items()
                if key != "ledger_sha256"
            }
        )
        or len(discovery_ledger.get("records", [])) != ledger_reference["record_count"]
    ):
        raise ConfirmatoryCampaignError("Discovery attempt ledger changed.")
    discovery_records = discovery_ledger.get("records")
    if not isinstance(discovery_records, list) or not discovery_records:
        raise ConfirmatoryCampaignError("Discovery attempt ledger has no physical POSTs.")
    evidence_targets = payload.get("targets")
    if not isinstance(evidence_targets, list) or payload.get("target_count") != len(
        targets
    ):
        raise ConfirmatoryCampaignError("Endpoint evidence target count is not exact.")
    if [item.get("target_id") for item in evidence_targets if isinstance(item, dict)] != [
        target["id"] for target in targets
    ]:
        raise ConfirmatoryCampaignError(
            "Endpoint evidence target order/coverage is not exact."
        )
    endpoint = payload.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith("https://"):
        raise ConfirmatoryCampaignError("Endpoint evidence lacks an HTTPS endpoint.")
    for target, record in zip(targets, evidence_targets, strict=True):
        if not isinstance(record, Mapping):
            raise ConfirmatoryCampaignError("Endpoint evidence target is not an object.")
        route = str(target["route"])
        census_row = census_by_route.get(route)
        evidence = record.get("evidence")
        if (
            record.get("target_id") != target["id"]
            or record.get("upstream_provider") != target.get("upstream_provider")
            or record.get("route") != route
            or not isinstance(evidence, Mapping)
            or evidence.get("verification_status") != "verified"
            or evidence.get("requested_route") != route
            or evidence.get("provider_response_model") != route
            or evidence.get("endpoint") != endpoint
            or evidence.get("verification_evidence")
            != target.get("verification_evidence")
            or census_row
            != {
                "route": route,
                "decision": "included_frozen_panel",
                "target_id": target["id"],
            }
        ):
            raise ConfirmatoryCampaignError(
                f"Endpoint evidence does not prove exact verified route {route}."
            )
        smoke = target["verification_evidence"].get("smoke_test")
        controls = smoke.get("generation_controls") if isinstance(smoke, Mapping) else None
        if (
            not isinstance(smoke, Mapping)
            or smoke.get("response_model") != route
            or not isinstance(smoke.get("request_id"), str)
            or not smoke.get("request_id")
            or not isinstance(controls, Mapping)
            or controls.get("temperature") != 0
            or controls.get("top_p") != 1
            or controls.get("stream") is not False
            or controls.get("structured_output") is not True
            or not isinstance(controls.get("seed"), int)
            or not isinstance(controls.get("max_tokens"), int)
            or controls.get("max_tokens", 0) <= 0
        ):
            raise ConfirmatoryCampaignError(
                f"Endpoint evidence lacks frozen smoke controls for {route}."
            )
        request = evidence.get("request")
        request_sha256 = request.get("request_sha256") if isinstance(request, Mapping) else None
        if not any(
            isinstance(attempt, Mapping)
            and attempt.get("target_id") == target["id"]
            and attempt.get("route") == route
            and attempt.get("request_sha256") == request_sha256
            and attempt.get("outcome") == "verified"
            and attempt.get("request_id") == smoke["request_id"]
            for attempt in discovery_records
        ):
            raise ConfirmatoryCampaignError(
                f"Discovery ledger lacks the successful physical POST for {route}."
            )


def _validate_variance_gate(
    payload: Mapping[str, Any],
    *,
    pilot_manifest_path: Path,
    expected_pilot_manifest_sha256: str,
) -> int:
    expected_keys = {
        "schema_version",
        "artifact_type",
        "private_input_sha256",
        "pilot_campaign_manifest_sha256",
        "selection",
        "artifact_sha256",
    }
    if set(payload) != expected_keys:
        raise ConfirmatoryCampaignError("Variance-selection artifact fields are not exact.")
    if (
        payload.get("schema_version") != VARIANCE_GATE_SCHEMA_VERSION
        or payload.get("artifact_type") != "part2_identity_masked_variance_selection"
    ):
        raise ConfirmatoryCampaignError(
            "Variance-selection artifact is not the native identity-masked gate."
        )
    _require_sha256(payload.get("private_input_sha256"), label="variance private input hash")
    pilot_hash = _require_sha256(
        payload.get("pilot_campaign_manifest_sha256"),
        label="variance pilot campaign manifest hash",
    )
    if pilot_hash != expected_pilot_manifest_sha256:
        raise ConfirmatoryCampaignError(
            "Variance selection does not descend from the supplied completed pilot campaign."
        )
    recorded_artifact_hash = _require_sha256(
        payload.get("artifact_sha256"), label="variance artifact self hash"
    )
    artifact_payload = {
        key: deepcopy(value)
        for key, value in payload.items()
        if key != "artifact_sha256"
    }
    canonical_bytes = (
        json.dumps(
            artifact_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(canonical_bytes).hexdigest() != recorded_artifact_hash:
        raise ConfirmatoryCampaignError("Variance-selection artifact self hash is invalid.")
    selection = payload.get("selection")
    expected_selection_keys = {
        "schema_version",
        "selection_rule",
        "identity_masked",
        "group_count",
        "pilot_runs_per_group",
        "common_environment_seed_count",
        "common_environment_seeds_sha256",
        "target_half_width",
        "s_max",
        "selected_common_run_count",
        "t_critical",
        "achieved_half_width",
        "capped_at_maximum",
        "location_removed_masked_input_sha256",
    }
    if not isinstance(selection, Mapping) or set(selection) != expected_selection_keys:
        raise ConfirmatoryCampaignError("Variance-selection calculation fields are not exact.")
    if (
        selection.get("schema_version") != 1
        or selection.get("selection_rule")
        != "smallest_n_with_t95_half_width_at_most_0.05_capped_20_40"
        or selection.get("identity_masked") is not True
        or selection.get("pilot_runs_per_group") != VARIANCE_PILOT_REPLICATES
        or selection.get("common_environment_seed_count")
        != VARIANCE_PILOT_REPLICATES
        or selection.get("target_half_width") != 0.05
    ):
        raise ConfirmatoryCampaignError(
            "Variance-selection artifact does not use the prespecified identity-masked rule."
        )
    for key in (
        "common_environment_seeds_sha256",
        "location_removed_masked_input_sha256",
    ):
        _require_sha256(selection.get(key), label=f"variance selection {key}")
    group_count = selection.get("group_count")
    if not isinstance(group_count, int) or isinstance(group_count, bool) or group_count <= 0:
        raise ConfirmatoryCampaignError("Variance selection group count must be positive.")
    for key in ("s_max", "t_critical", "achieved_half_width"):
        value = selection.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ConfirmatoryCampaignError(f"Variance selection {key} must be numeric.")
        if not float("-inf") < float(value) < float("inf") or float(value) < 0:
            raise ConfirmatoryCampaignError(f"Variance selection {key} must be finite and nonnegative.")
    if not isinstance(selection.get("capped_at_maximum"), bool):
        raise ConfirmatoryCampaignError("Variance selection cap flag must be boolean.")
    selected_n = selection.get("selected_common_run_count")
    if (
        not isinstance(selected_n, int)
        or isinstance(selected_n, bool)
        or not MIN_BASELINE_REPLICATES <= selected_n <= MAX_BASELINE_REPLICATES
    ):
        raise ConfirmatoryCampaignError("Variance-selected n must be in 20..40.")
    try:
        from analysis import part2_confirmatory

        native_input = part2_confirmatory._derive_native_blinded_variance_input(
            pilot_manifest_path,
            expected_sha256=expected_pilot_manifest_sha256,
        )
        native_input_sha256 = hashlib.sha256(
            part2_confirmatory._canonical_json_bytes(native_input)
        ).hexdigest()
        native_pilot_hash, frozen_groups, pilot_values = (
            part2_confirmatory._parse_blinded_variance_input(native_input)
        )
        expected_selection = part2_confirmatory.select_blinded_variance_run_count(
            pilot_values,
            expected_blinded_groups=frozen_groups,
        )
    except Exception as error:
        raise ConfirmatoryCampaignError(
            f"Native variance-pilot rederivation failed: {error}"
        ) from error
    if (
        native_pilot_hash != expected_pilot_manifest_sha256
        or payload.get("private_input_sha256") != native_input_sha256
        or dict(selection) != expected_selection
    ):
        raise ConfirmatoryCampaignError(
            "Variance selection does not exactly equal the native rederived pilot rule."
        )
    return selected_n


def _validate_completed_variance_pilot_manifest(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    """Revalidate the complete-union pilot and every native artifact it pins."""

    expected = _require_sha256(expected_sha256, label="variance pilot manifest hash")
    if not path.is_file() or sha256_file(path) != expected:
        raise ConfirmatoryCampaignError("Variance pilot campaign manifest hash mismatch.")
    pilot = load_manifest(path)
    if (
        pilot.get("status") != "complete"
        or pilot.get("target_selection", {}).get("mode") != "complete_union"
        or pilot.get("part2_design", {}).get("scientific_stage")
        != "part2_variance_pilot"
        or pilot.get("part2_design", {}).get("replicates_per_target")
        != VARIANCE_PILOT_REPLICATES
    ):
        raise ConfirmatoryCampaignError(
            "Variance selection requires a completed complete-union variance-pilot campaign."
        )
    target_ids = {str(target["id"]) for target in pilot.get("targets", [])}
    variance_jobs = [
        job
        for job in pilot.get("jobs", [])
        if isinstance(job, Mapping)
        and job.get("stage") == "part2_variance_pilot"
        and job.get("experiment") == "part2"
    ]
    expected_pairs = {
        (target_id, seed_index, seed_index)
        for target_id in target_ids
        for seed_index in range(VARIANCE_PILOT_REPLICATES)
    }
    observed_pairs: set[tuple[str, int, int]] = set()
    for job in variance_jobs:
        generation_seed = job.get("expected", {}).get("generation_seed")
        environment_seed = job.get("expected", {}).get("environment_seed")
        if not isinstance(generation_seed, int) or not isinstance(environment_seed, int):
            raise ConfirmatoryCampaignError("Variance-pilot job seeds are invalid.")
        observed_pairs.add(
            (
                str(job.get("target_id")),
                generation_seed - DEFAULT_PART2_GENERATION_SEED_BASE,
                environment_seed - DEFAULT_PART2_ENVIRONMENT_SEED_BASE,
            )
        )
    if observed_pairs != expected_pairs or any(
        not isinstance(job, Mapping) or job.get("status") != "complete"
        for job in pilot.get("jobs", [])
    ):
        raise ConfirmatoryCampaignError(
            "Variance-pilot manifest is incomplete or has a nonexact seed panel."
        )
    for job in pilot["jobs"]:
        resolve_job_artifact(job, None)
    return pilot


def _route_role(target_by_id: Mapping[str, Mapping[str, Any]], target_id: str, role: str) -> dict[str, Any]:
    target = target_by_id.get(target_id)
    if target is None:
        raise ConfirmatoryCampaignError(
            f"{role} target {target_id!r} is not in the exact requested cohort union."
        )
    if target.get("provider") != "inference_hub" or target.get(
        "verification_status"
    ) != "verified":
        raise ConfirmatoryCampaignError(f"{role} route is not verified InferenceHub.")
    return deepcopy(dict(target))


def _part2_config(*, smoke: bool, args: argparse.Namespace) -> dict[str, Any]:
    config = SocietyConfig(
        society_size=4 if smoke else args.part2_society_size,
        days=3 if smoke else args.part2_days,
        resource=args.part2_resource,
        selfish_gain=args.part2_selfish_gain,
        depletion_units=args.part2_depletion_units,
        community_benefit=args.part2_community_benefit,
    )
    # The sacrificial gate is a compact mechanics stress test.  A deliberately
    # small reserve permits collapse/attrition when agents overuse, while three
    # days always exercise state feedback and day-specific seed derivation.
    resource_capacity = (
        config.society_size * config.depletion_units
        if smoke
        else int(
            getattr(args, "part2_resource_capacity", 0)
            or _initial_resource_units(config)
        )
    )
    return {
        "society_size": config.society_size,
        "days": config.days,
        "resource": config.resource,
        "selfish_gain": config.selfish_gain,
        "depletion_units": config.depletion_units,
        "community_benefit": config.community_benefit,
        "resource_capacity": resource_capacity,
        "collapse_death_rate": DEFAULT_COLLAPSE_DEATH_RATE,
    }


def _part2_argv(
    *,
    python: str,
    target: Mapping[str, Any],
    config: Mapping[str, Any],
    generation_seed: int,
    environment_seed: int,
    resume: bool,
) -> list[str]:
    argv = [
        python,
        "-m",
        "experiments.part2.part_2",
        "--provider",
        "inference_hub",
        "--model",
        str(target["route"]),
        "--direct-output-token-cap",
        "32",
        "--society-size",
        str(config["society_size"]),
        "--days",
        str(config["days"]),
        "--resource",
        str(config["resource"]),
        "--selfish-gain",
        str(config["selfish_gain"]),
        "--depletion-units",
        str(config["depletion_units"]),
        "--community-benefit",
        str(config["community_benefit"]),
        "--resource-capacity",
        str(config["resource_capacity"]),
        "--collapse-death-rate",
        str(config["collapse_death_rate"]),
        "--generation-seed",
        str(generation_seed),
        "--environment-seed",
        str(environment_seed),
        "--headless",
    ]
    if resume:
        argv.append("--resume")
    return argv


def _p0_argv(
    *,
    python: str,
    mode: str,
    target: Mapping[str, Any],
    judge: Mapping[str, Any],
    registry_path: str,
    registry_hash: str,
    output_dir: str,
    completed_smoke_dir: str | None,
    resume: bool,
) -> list[str]:
    argv = [
        python,
        "-m",
        "experiments.part0.confirmatory_runner",
        "--mode",
        mode,
        "--registry",
        registry_path,
        "--registry-sha256",
        registry_hash,
        "--subject-provider",
        "inference_hub",
        "--subject-route",
        str(target["route"]),
        "--judge-provider",
        "inference_hub",
        "--judge-route",
        str(judge["route"]),
        "--master-seed",
        "20260801",
        "--output-dir",
        output_dir,
        "--allow-frozen-campaign-route",
    ]
    if completed_smoke_dir is not None:
        argv.extend(["--completed-smoke-dir", completed_smoke_dir])
    argv.append("--resume" if resume else "--fresh")
    return argv


def _p1_argv(
    *,
    python: str,
    mode: str,
    target: Mapping[str, Any],
    bank_path: str,
    bank_hash: str,
    output_dir: str,
    completed_smoke_dir: str | None,
    resume: bool,
) -> list[str]:
    argv = [
        python,
        "-m",
        "experiments.part1.confirmatory_runner",
        "--mode",
        mode,
        "--bank",
        bank_path,
        "--bank-sha256",
        bank_hash,
        "--subject-provider",
        "inference_hub",
        "--subject-model",
        str(target["route"]),
        "--output-directory",
        output_dir,
        "--allow-frozen-campaign-route",
    ]
    if completed_smoke_dir is not None:
        argv.extend(["--completed-smoke-directory", completed_smoke_dir])
    if resume:
        argv.append("--resume")
    return argv


def _new_job(
    *,
    job_id: str,
    stage: str,
    experiment: str,
    target: Mapping[str, Any],
    smoke_job_id: str | None,
    argv_fresh: list[str],
    argv_resume: list[str],
    output_dir: str | None,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": job_id,
        "stage": stage,
        "experiment": experiment,
        "target_id": target["id"],
        "provider": "inference_hub",
        "route": target["route"],
        "smoke_job_id": smoke_job_id,
        "argv_fresh": argv_fresh,
        "argv_resume": argv_resume,
        "output_dir": output_dir,
        "expected": deepcopy(dict(expected)),
        "status": "pending",
        "attempts": [],
        "artifact": None,
        "last_error": None,
    }


def _block_randomized_jobs(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Order target/part blocks reproducibly while preserving smoke dependencies."""

    def rank(*parts: object) -> str:
        material = "|".join(
            [CAMPAIGN_SCHEDULING_PROTOCOL, str(CAMPAIGN_SCHEDULING_SEED)]
            + [str(part) for part in parts]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def key(job: Mapping[str, Any]) -> tuple[str, str, int, str]:
        target_id = str(job["target_id"])
        experiment = str(job["experiment"])
        return (
            rank("target", target_id),
            rank("part", target_id, experiment),
            0 if job["stage"] == "smoke" else 1,
            rank("within-part", target_id, experiment, job["id"]),
        )

    ordered = [deepcopy(dict(job)) for job in sorted(jobs, key=key)]
    position = {str(job["id"]): index for index, job in enumerate(ordered)}
    for job in ordered:
        smoke_id = job.get("smoke_job_id")
        if smoke_id is not None and position[str(smoke_id)] >= position[str(job["id"])]:
            raise ConfirmatoryCampaignError(
                "Block-randomized schedule violated a same-target smoke dependency."
            )
    return ordered


def _build_plan_from_arguments(
    args: argparse.Namespace,
    *,
    enforce_route_freshness: bool = True,
    archived_validation: bool = False,
) -> dict[str, Any]:
    """Reconstruct a plan, including archived two-stage plans for validation."""

    part2_stage = getattr(args, "part2_stage", "fixed-production")
    if part2_stage != "fixed-production" and not archived_validation:
        raise ConfirmatoryCampaignError(
            "New confirmatory campaigns require the fixed one-stage Part 2 design."
        )

    cohort_ids = tuple(args.cohort or DEFAULT_COHORTS)
    (
        cohorts,
        complete_union_targets,
        registry_version,
        registry_hash,
        routing_roster_hash,
    ) = _load_exact_cohort_union(
        cohort_ids,
        # Archived baseline manifests are reconstructed only for compatibility
        # validation. Their elapsed route-freshness window is not execution
        # authority and therefore cannot invalidate their historical lineage.
        enforce_freshness=(
            enforce_route_freshness and part2_stage != "baseline-production"
        ),
    )
    target_by_id = {
        str(target["id"]): target for target in complete_union_targets
    }
    judge = _route_role(target_by_id, args.judge_target_id, "judge")
    requested_target_ids = list(args.target_id or [])
    if len(set(requested_target_ids)) != len(requested_target_ids):
        raise ConfirmatoryCampaignError("Requested target shard contains duplicates.")
    unknown_target_ids = sorted(set(requested_target_ids) - set(target_by_id))
    if unknown_target_ids:
        raise ConfirmatoryCampaignError(
            "Requested target shard is outside the exact cohort union: "
            + ", ".join(unknown_target_ids)
        )
    selected_id_set = set(requested_target_ids)
    targets = (
        [
            target
            for target in complete_union_targets
            if str(target["id"]) in selected_id_set
        ]
        if requested_target_ids
        else list(complete_union_targets)
    )

    p0_path = Path(args.part0_registry).resolve()
    p0_hash = _require_sha256(args.part0_registry_sha256, label="Part 0 registry hash")
    try:
        part0_runner.load_production_registry(p0_path, expected_sha256=p0_hash)
    except Exception as error:
        raise ConfirmatoryCampaignError(f"Part 0 registry failed production validation: {error}") from error
    p1_path = Path(args.part1_bank).resolve()
    p1_hash = _require_sha256(args.part1_bank_sha256, label="Part 1 bank hash")
    try:
        part1_runner.load_production_bank(p1_path, expected_sha256=p1_hash)
    except Exception as error:
        raise ConfirmatoryCampaignError(f"Part 1 bank failed production validation: {error}") from error

    evidence_path, evidence, evidence_hash = _hash_pinned_json(
        args.endpoint_evidence,
        expected_sha256=args.endpoint_evidence_sha256,
        label="endpoint-evidence bundle",
    )
    _validate_endpoint_evidence(
        evidence,
        cohorts=cohorts,
        targets=complete_union_targets,
        registry_version=registry_version,
        routing_roster_hash=routing_roster_hash,
    )

    selected_n: int | None = None
    gate_integrity: dict[str, Any] | None = None
    if part2_stage == "fixed-production":
        if (
            args.variance_selection
            or args.variance_selection_sha256
            or args.variance_pilot_manifest
            or args.variance_pilot_manifest_sha256
        ):
            raise ConfirmatoryCampaignError(
                "Fixed production does not consume a variance-pilot artifact."
            )
        if (
            args.part2_society_size != 10
            or args.part2_days != 30
            or getattr(args, "part2_resource_capacity", 150) != 150
        ):
            raise ConfirmatoryCampaignError(
                "Fixed production requires Part 2 N=10, horizon=30, capacity=150."
            )
        part2_seeds = list(range(FIXED_PART2_REPLICATES))
        scientific_stage = "part2_fixed_production"
    elif part2_stage == "variance-pilot":
        if (
            args.variance_selection
            or args.variance_selection_sha256
            or args.variance_pilot_manifest
            or args.variance_pilot_manifest_sha256
        ):
            raise ConfirmatoryCampaignError(
                "Variance pilot must not consume a prior pilot or variance-selection gate."
            )
        part2_seeds = list(range(VARIANCE_PILOT_REPLICATES))
        scientific_stage = "part2_variance_pilot"
    elif part2_stage == "baseline-production":
        if (
            not args.variance_selection
            or not args.variance_selection_sha256
            or not args.variance_pilot_manifest
            or not args.variance_pilot_manifest_sha256
        ):
            raise ConfirmatoryCampaignError(
                "Baseline production requires a hash-pinned completed pilot manifest "
                "and its native identity-masked variance gate."
            )
        pilot_manifest_path = Path(args.variance_pilot_manifest).resolve()
        pilot_manifest_hash = _require_sha256(
            args.variance_pilot_manifest_sha256,
            label="variance pilot campaign manifest hash",
        )
        pilot_manifest = _validate_completed_variance_pilot_manifest(
            pilot_manifest_path,
            expected_sha256=pilot_manifest_hash,
        )
        if (
            pilot_manifest.get("cohorts") != cohorts
            or pilot_manifest.get("registry")
            != {
                "version": registry_version,
                "hash": registry_hash,
                "routing_roster_sha256": routing_roster_hash,
            }
            or pilot_manifest.get("roles")
            != {
                "judge_target_id": judge["id"],
                "judge_route": judge["route"],
            }
            or pilot_manifest.get("inputs", {}).get("endpoint_evidence")
            != {
                "path": str(evidence_path),
                "sha256": evidence_hash,
                "bundle_sha256": evidence["bundle_sha256"],
            }
        ):
            raise ConfirmatoryCampaignError(
                "Baseline production must continue the pilot's exact cohort, "
                "registry, roles, and endpoint attestation."
            )
        gate_path, gate, gate_hash = _hash_pinned_json(
            args.variance_selection,
            expected_sha256=args.variance_selection_sha256,
            label="variance-selection artifact",
        )
        selected_n = _validate_variance_gate(
            gate,
            pilot_manifest_path=pilot_manifest_path,
            expected_pilot_manifest_sha256=pilot_manifest_hash,
        )
        gate_integrity = {
            "path": str(gate_path),
            "sha256": gate_hash,
            "selected_n": selected_n,
            "artifact_sha256": gate["artifact_sha256"],
            "private_input_sha256": gate["private_input_sha256"],
            "pilot_campaign": {
                "path": str(pilot_manifest_path),
                "sha256": pilot_manifest_hash,
                "campaign_id": pilot_manifest["campaign_id"],
                "plan_sha256": pilot_manifest["plan_sha256"],
                "manifest_payload_sha256": pilot_manifest["manifest_sha256"],
            },
        }
        part2_seeds = list(range(selected_n))
        scientific_stage = "part2_baseline_production"
    else:
        raise ConfirmatoryCampaignError("Unsupported Part 2 scientific stage.")

    freeze = _execution_freeze()
    python = freeze["python_executable"]
    campaign_id = _require_campaign_id(args.campaign_id)
    production_config = _part2_config(smoke=False, args=args)
    smoke_config = _part2_config(smoke=True, args=args)
    jobs: list[dict[str, Any]] = []
    smoke_ids: dict[tuple[str, str], str] = {}

    smoke_experiments = (
        ("part0", "part1", "part2")
        if part2_stage in {"fixed-production", "variance-pilot"}
        else ("part2",)
    )
    for experiment in smoke_experiments:
        for target in targets:
            target_id = str(target["id"])
            job_id = f"smoke-{experiment}-{target_id}"
            smoke_ids[(experiment, target_id)] = job_id
            if experiment == "part0":
                output = (
                    part0_runner.PRIVATE_RESULTS_ROOT
                    / campaign_id
                    / target_id
                    / "smoke"
                ).resolve()
                fresh = _p0_argv(
                    python=python,
                    mode="sacrificial-smoke",
                    target=target,
                    judge=judge,
                    registry_path=str(p0_path),
                    registry_hash=p0_hash,
                    output_dir=str(output),
                    completed_smoke_dir=None,
                    resume=False,
                )
                resume_argv = _p0_argv(
                    python=python,
                    mode="sacrificial-smoke",
                    target=target,
                    judge=judge,
                    registry_path=str(p0_path),
                    registry_hash=p0_hash,
                    output_dir=str(output),
                    completed_smoke_dir=None,
                    resume=True,
                )
                expected = {
                    "execution_mode": "sacrificial_smoke",
                    "registry_path": str(p0_path),
                    "registry_sha256": p0_hash,
                    "judge_route": judge["route"],
                }
            elif experiment == "part1":
                output = (
                    part1_runner.PRIVATE_RESULTS_ROOT
                    / campaign_id
                    / target_id
                    / "smoke"
                ).resolve()
                fresh = _p1_argv(
                    python=python,
                    mode="sacrificial-smoke",
                    target=target,
                    bank_path=str(p1_path),
                    bank_hash=p1_hash,
                    output_dir=str(output),
                    completed_smoke_dir=None,
                    resume=False,
                )
                resume_argv = _p1_argv(
                    python=python,
                    mode="sacrificial-smoke",
                    target=target,
                    bank_path=str(p1_path),
                    bank_hash=p1_hash,
                    output_dir=str(output),
                    completed_smoke_dir=None,
                    resume=True,
                )
                expected = {
                    "execution_mode": "sacrificial_smoke",
                    "bank_path": str(p1_path),
                    "bank_sha256": p1_hash,
                }
            else:
                output = None
                fresh = _part2_argv(
                    python=python,
                    target=target,
                    config=smoke_config,
                    generation_seed=PART2_SMOKE_GENERATION_SEED,
                    environment_seed=PART2_SMOKE_ENVIRONMENT_SEED,
                    resume=False,
                )
                resume_argv = _part2_argv(
                    python=python,
                    target=target,
                    config=smoke_config,
                    generation_seed=PART2_SMOKE_GENERATION_SEED,
                    environment_seed=PART2_SMOKE_ENVIRONMENT_SEED,
                    resume=True,
                )
                expected = _part2_expected(
                    target=target,
                    config=smoke_config,
                    generation_seed=PART2_SMOKE_GENERATION_SEED,
                    environment_seed=PART2_SMOKE_ENVIRONMENT_SEED,
                    git_commit=str(freeze["git_commit"]),
                )
            jobs.append(
                _new_job(
                    job_id=job_id,
                    stage="smoke",
                    experiment=experiment,
                    target=target,
                    smoke_job_id=None,
                    argv_fresh=fresh,
                    argv_resume=resume_argv,
                    output_dir=str(output) if output is not None else None,
                    expected=expected,
                )
            )

    production_experiments = (
        ("part0", "part1")
        if part2_stage in {"fixed-production", "variance-pilot"}
        else ()
    )
    for experiment in production_experiments:
        for target in targets:
            target_id = str(target["id"])
            job_id = f"production-{experiment}-{target_id}"
            if experiment == "part0":
                completed_smoke = (
                    part0_runner.PRIVATE_RESULTS_ROOT
                    / campaign_id
                    / target_id
                    / "smoke"
                ).resolve()
                output = (
                    part0_runner.PRIVATE_RESULTS_ROOT
                    / campaign_id
                    / target_id
                    / "production"
                ).resolve()
                fresh = _p0_argv(
                    python=python,
                    mode="production",
                    target=target,
                    judge=judge,
                    registry_path=str(p0_path),
                    registry_hash=p0_hash,
                    output_dir=str(output),
                    completed_smoke_dir=str(completed_smoke),
                    resume=False,
                )
                resume_argv = _p0_argv(
                    python=python,
                    mode="production",
                    target=target,
                    judge=judge,
                    registry_path=str(p0_path),
                    registry_hash=p0_hash,
                    output_dir=str(output),
                    completed_smoke_dir=str(completed_smoke),
                    resume=True,
                )
            else:
                completed_smoke = (
                    part1_runner.PRIVATE_RESULTS_ROOT
                    / campaign_id
                    / target_id
                    / "smoke"
                ).resolve()
                output = (
                    part1_runner.PRIVATE_RESULTS_ROOT
                    / campaign_id
                    / target_id
                    / "production"
                ).resolve()
                fresh = _p1_argv(
                    python=python,
                    mode="production",
                    target=target,
                    bank_path=str(p1_path),
                    bank_hash=p1_hash,
                    output_dir=str(output),
                    completed_smoke_dir=str(completed_smoke),
                    resume=False,
                )
                resume_argv = _p1_argv(
                    python=python,
                    mode="production",
                    target=target,
                    bank_path=str(p1_path),
                    bank_hash=p1_hash,
                    output_dir=str(output),
                    completed_smoke_dir=str(completed_smoke),
                    resume=True,
                )
            jobs.append(
                _new_job(
                    job_id=job_id,
                    stage="production",
                    experiment=experiment,
                    target=target,
                    smoke_job_id=smoke_ids[(experiment, target_id)],
                    argv_fresh=fresh,
                    argv_resume=resume_argv,
                    output_dir=str(output),
                    expected=(
                        {
                            "execution_mode": "production",
                            "registry_path": str(p0_path),
                            "registry_sha256": p0_hash,
                            "judge_route": judge["route"],
                        }
                        if experiment == "part0"
                        else {
                            "execution_mode": "production",
                            "bank_path": str(p1_path),
                            "bank_sha256": p1_hash,
                        }
                    ),
                )
            )

    for target in targets:
        target_id = str(target["id"])
        for seed_offset in part2_seeds:
            generation_seed = DEFAULT_PART2_GENERATION_SEED_BASE + seed_offset
            environment_seed = DEFAULT_PART2_ENVIRONMENT_SEED_BASE + seed_offset
            job_id = f"{scientific_stage}-{target_id}-s{seed_offset + 1:02d}"
            fresh = _part2_argv(
                python=python,
                target=target,
                config=production_config,
                generation_seed=generation_seed,
                environment_seed=environment_seed,
                resume=False,
            )
            resume_argv = _part2_argv(
                python=python,
                target=target,
                config=production_config,
                generation_seed=generation_seed,
                environment_seed=environment_seed,
                resume=True,
            )
            jobs.append(
                _new_job(
                    job_id=job_id,
                    stage=scientific_stage,
                    experiment="part2",
                    target=target,
                    smoke_job_id=smoke_ids[("part2", target_id)],
                    argv_fresh=fresh,
                    argv_resume=resume_argv,
                    output_dir=None,
                    expected=_part2_expected(
                        target=target,
                        config=production_config,
                        generation_seed=generation_seed,
                        environment_seed=environment_seed,
                        git_commit=str(freeze["git_commit"]),
                    ),
                )
            )

    if len({job["id"] for job in jobs}) != len(jobs):
        raise ConfirmatoryCampaignError("Planned job IDs are not unique.")
    jobs = _block_randomized_jobs(jobs)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign_id": campaign_id,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "status": "planned",
        "cohorts": list(cohorts),
        "target_selection": {
            "mode": (
                "complete_union"
                if len(targets) == len(complete_union_targets)
                else "shard"
            ),
            "selected_target_ids": [str(target["id"]) for target in targets],
            "complete_union_target_count": len(complete_union_targets),
        },
        "targets": [
            {
                "id": target["id"],
                "provider": target["provider"],
                "route": target["route"],
                "upstream_provider": target.get("upstream_provider"),
                "verification_evidence_sha256": stable_json_hash(
                    target["verification_evidence"]
                ),
            }
            for target in targets
        ],
        "roles": {
            "judge_target_id": judge["id"],
            "judge_route": judge["route"],
        },
        "registry": {
            "version": registry_version,
            "hash": registry_hash,
            "routing_roster_sha256": routing_roster_hash,
        },
        "inputs": {
            "part0_registry": {"path": str(p0_path), "sha256": p0_hash},
            "part1_bank": {"path": str(p1_path), "sha256": p1_hash},
            "endpoint_evidence": {
                "path": str(evidence_path),
                "sha256": evidence_hash,
                "bundle_sha256": evidence["bundle_sha256"],
            },
            "variance_selection": gate_integrity,
        },
        "part2_design": {
            "scientific_stage": scientific_stage,
            "replicates_per_target": len(part2_seeds),
            "common_generation_seeds": [
                DEFAULT_PART2_GENERATION_SEED_BASE + value for value in part2_seeds
            ],
            "common_environment_seeds": [
                DEFAULT_PART2_ENVIRONMENT_SEED_BASE + value for value in part2_seeds
            ],
            "variance_pilot_replicates": (
                VARIANCE_PILOT_REPLICATES
                if part2_stage == "variance-pilot"
                else None
            ),
            "variance_selected_n": selected_n,
            "fixed_replicates": (
                FIXED_PART2_REPLICATES
                if part2_stage == "fixed-production"
                else None
            ),
            "production_config": production_config,
            "smoke_config": smoke_config,
        },
        "scheduling": {
            "protocol": CAMPAIGN_SCHEDULING_PROTOCOL,
            "seed": CAMPAIGN_SCHEDULING_SEED,
            "unit": "target_then_part_dependency_block",
            "smoke_dependency": "same_target_same_part_smoke_precedes_science",
        },
        "timeout_seconds": int(args.timeout_seconds),
        "request_budget": build_frozen_budget(len(targets)),
        "execution_freeze": freeze,
        "jobs": jobs,
    }
    manifest["plan_sha256"] = _plan_hash(manifest)
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    return manifest


def build_plan(
    args: argparse.Namespace, *, enforce_route_freshness: bool = True
) -> dict[str, Any]:
    """Build the sole authorized fixed one-stage confirmatory campaign plan."""

    return _build_plan_from_arguments(
        args,
        enforce_route_freshness=enforce_route_freshness,
        archived_validation=False,
    )


def _part2_expected(
    *,
    target: Mapping[str, Any],
    config: Mapping[str, Any],
    generation_seed: int,
    environment_seed: int,
    git_commit: str,
) -> dict[str, Any]:
    return {
        "provider": "inference_hub",
        "model": target["route"],
        "society_config": {
            key: config[key]
            for key in (
                "society_size",
                "days",
                "resource",
                "selfish_gain",
                "depletion_units",
                "community_benefit",
            )
        },
        "resource_capacity": config["resource_capacity"],
        "collapse_death_rate": config["collapse_death_rate"],
        "generation_seed": generation_seed,
        "environment_seed": environment_seed,
        "row_count_upper_bound": int(config["society_size"]) * int(config["days"]),
        "grading_protocol": None,
        "generation_protocol": {
            "schema_version": 1,
            "mode": "direct_provider_structured_output",
            "protocol": part_2.DIRECT_GENERATION_PROTOCOL_VERSION,
            "output_schema": "SocietyDecision(extra=forbid)",
            "parser": "exact_json_object_action_reasoning_no_semantic_retry_v1",
            "output_token_cap": part_2.DEFAULT_DIRECT_OUTPUT_TOKEN_CAP,
            "identity_policy": "exact_requested_returned_model_required_v1",
        },
        "campaign_git_commit": git_commit,
    }


def _job_plan(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(job[key])
        for key in (
            "id",
            "stage",
            "experiment",
            "target_id",
            "provider",
            "route",
            "smoke_job_id",
            "argv_fresh",
            "argv_resume",
            "output_dir",
            "expected",
        )
    }


def _plan_hash(manifest: Mapping[str, Any]) -> str:
    immutable = {
        key: deepcopy(manifest[key])
        for key in (
            "schema_version",
            "campaign_id",
            "cohorts",
            "target_selection",
            "targets",
            "roles",
            "registry",
            "inputs",
            "part2_design",
            "scheduling",
            "timeout_seconds",
            "request_budget",
            "execution_freeze",
        )
    }
    immutable["jobs"] = [_job_plan(job) for job in manifest["jobs"]]
    return stable_json_hash(immutable)


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    return stable_json_hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def _require_fixed_execution_manifest(
    manifest: Mapping[str, Any], *, action: str
) -> None:
    """Reject creation or execution of archived two-stage campaign plans."""

    design = manifest.get("part2_design")
    inputs = manifest.get("inputs")
    if (
        not isinstance(design, Mapping)
        or design.get("scientific_stage") != "part2_fixed_production"
        or design.get("replicates_per_target") != FIXED_PART2_REPLICATES
        or design.get("fixed_replicates") != FIXED_PART2_REPLICATES
        or design.get("variance_pilot_replicates") is not None
        or design.get("variance_selected_n") is not None
        or not isinstance(inputs, Mapping)
        or inputs.get("variance_selection") is not None
    ):
        raise ConfirmatoryCampaignError(
            f"Cannot {action} an archived two-stage campaign; only the fixed "
            "one-stage Part 2 design is executable."
        )


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at_utc"] = _utc_now()
    manifest["manifest_sha256"] = _manifest_hash(manifest)
    _atomic_json_write(path, manifest)


def _rebuild_arguments_from_manifest(manifest: Mapping[str, Any]) -> argparse.Namespace:
    """Recover the sole planning arguments represented by an immutable manifest."""

    inputs = manifest["inputs"]
    roles = manifest["roles"]
    design = manifest["part2_design"]
    production = design["production_config"]
    scientific_stage = design["scientific_stage"]
    gate = inputs.get("variance_selection")
    is_baseline = scientific_stage == "part2_baseline_production"
    pilot = gate.get("pilot_campaign") if isinstance(gate, Mapping) else None
    selection = manifest["target_selection"]
    target_ids = (
        list(selection["selected_target_ids"])
        if selection.get("mode") == "shard"
        else None
    )
    return argparse.Namespace(
        campaign_id=manifest["campaign_id"],
        cohort=[row["id"] for row in manifest["cohorts"]],
        target_id=target_ids,
        judge_target_id=roles["judge_target_id"],
        part0_registry=inputs["part0_registry"]["path"],
        part0_registry_sha256=inputs["part0_registry"]["sha256"],
        part1_bank=inputs["part1_bank"]["path"],
        part1_bank_sha256=inputs["part1_bank"]["sha256"],
        endpoint_evidence=inputs["endpoint_evidence"]["path"],
        endpoint_evidence_sha256=inputs["endpoint_evidence"]["sha256"],
        part2_stage=(
            "baseline-production"
            if is_baseline
            else "fixed-production"
            if scientific_stage == "part2_fixed_production"
            else "variance-pilot"
        ),
        variance_selection=(gate.get("path") if is_baseline else None),
        variance_selection_sha256=(gate.get("sha256") if is_baseline else None),
        variance_pilot_manifest=(pilot.get("path") if is_baseline else None),
        variance_pilot_manifest_sha256=(pilot.get("sha256") if is_baseline else None),
        part2_society_size=production["society_size"],
        part2_days=production["days"],
        part2_resource=production["resource"],
        part2_selfish_gain=production["selfish_gain"],
        part2_depletion_units=production["depletion_units"],
        part2_community_benefit=production["community_benefit"],
        part2_resource_capacity=production["resource_capacity"],
        timeout_seconds=manifest["timeout_seconds"],
    )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ConfirmatoryCampaignError("Unsupported confirmatory manifest schema.")
    try:
        validate_frozen_budget(manifest.get("request_budget", {}))
    except Exception as error:
        raise ConfirmatoryCampaignError(f"Manifest request budget is invalid: {error}") from error
    if manifest.get("plan_sha256") != _plan_hash(manifest):
        raise ConfirmatoryCampaignError("Confirmatory manifest plan hash mismatch.")
    if manifest.get("manifest_sha256") != _manifest_hash(manifest):
        raise ConfirmatoryCampaignError("Confirmatory manifest payload hash mismatch.")
    if manifest.get("execution_freeze") != _execution_freeze():
        raise ConfirmatoryCampaignError(
            "Campaign commit, sources, interpreter, or dependencies drifted."
        )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ConfirmatoryCampaignError("Manifest pinned inputs are missing.")
    for key in ("part0_registry", "part1_bank", "endpoint_evidence"):
        value = inputs.get(key)
        if not isinstance(value, Mapping):
            raise ConfirmatoryCampaignError(f"Manifest input {key} is missing.")
        path = Path(str(value.get("path", "")))
        if not path.is_file() or sha256_file(path) != value.get("sha256"):
            raise ConfirmatoryCampaignError(f"Manifest input {key} changed.")
    cohort_ids = [str(value["id"]) for value in manifest.get("cohorts", [])]
    (
        cohorts,
        complete_union_targets,
        registry_version,
        registry_hash,
        routing_roster_hash,
    ) = _load_exact_cohort_union(cohort_ids, enforce_freshness=False)
    if cohorts != manifest.get("cohorts") or manifest.get("registry") != {
        "version": registry_version,
        "hash": registry_hash,
        "routing_roster_sha256": routing_roster_hash,
    }:
        raise ConfirmatoryCampaignError("Current cohort union differs from the manifest.")
    selection = manifest.get("target_selection")
    if not isinstance(selection, Mapping):
        raise ConfirmatoryCampaignError("Manifest target selection is missing.")
    selected_ids = selection.get("selected_target_ids")
    if (
        not isinstance(selected_ids, list)
        or not selected_ids
        or len(selected_ids) != len(set(selected_ids))
        or not all(isinstance(target_id, str) for target_id in selected_ids)
    ):
        raise ConfirmatoryCampaignError("Manifest target shard is invalid.")
    complete_target_by_id = {
        str(target["id"]): target for target in complete_union_targets
    }
    if selection.get("complete_union_target_count") != len(
        complete_union_targets
    ) or not set(selected_ids).issubset(complete_target_by_id):
        raise ConfirmatoryCampaignError("Manifest target shard left the cohort union.")
    expected_mode = (
        "complete_union"
        if len(selected_ids) == len(complete_union_targets)
        else "shard"
    )
    if selection.get("mode") != expected_mode:
        raise ConfirmatoryCampaignError("Manifest target selection mode is invalid.")
    targets = [
        target
        for target in complete_union_targets
        if str(target["id"]) in set(selected_ids)
    ]
    if [str(target["id"]) for target in targets] != selected_ids:
        raise ConfirmatoryCampaignError(
            "Manifest target shard order differs from the cohort union."
        )
    current_targets = [
        {
            "id": target["id"],
            "provider": target["provider"],
            "route": target["route"],
            "upstream_provider": target.get("upstream_provider"),
            "verification_evidence_sha256": stable_json_hash(
                target["verification_evidence"]
            ),
        }
        for target in targets
    ]
    if manifest.get("targets") != current_targets:
        raise ConfirmatoryCampaignError("Current verified target union changed.")
    target_by_id = complete_target_by_id
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise ConfirmatoryCampaignError("Manifest model roles are missing.")
    judge = _route_role(
        target_by_id, str(roles.get("judge_target_id", "")), "judge"
    )
    if set(roles) != {"judge_target_id", "judge_route"} or roles.get(
        "judge_route"
    ) != judge["route"]:
        raise ConfirmatoryCampaignError("Manifest role routes changed.")
    evidence_record = inputs["endpoint_evidence"]
    endpoint_payload = _load_json(
        evidence_record["path"], label="endpoint-evidence bundle"
    )
    _validate_endpoint_evidence(
        endpoint_payload,
        cohorts=cohorts,
        targets=complete_union_targets,
        registry_version=registry_version,
        routing_roster_hash=routing_roster_hash,
    )
    if endpoint_payload.get("bundle_sha256") != evidence_record.get("bundle_sha256"):
        raise ConfirmatoryCampaignError("Endpoint bundle identity changed.")
    try:
        part0_runner.load_production_registry(
            inputs["part0_registry"]["path"],
            expected_sha256=inputs["part0_registry"]["sha256"],
        )
        part1_runner.load_production_bank(
            inputs["part1_bank"]["path"],
            expected_sha256=inputs["part1_bank"]["sha256"],
        )
    except Exception as error:
        raise ConfirmatoryCampaignError(
            f"Pinned confirmatory inputs no longer pass production validation: {error}"
        ) from error
    gate = inputs.get("variance_selection")
    if gate is not None:
        if not isinstance(gate, Mapping):
            raise ConfirmatoryCampaignError("Variance gate integrity is invalid.")
        path = Path(str(gate.get("path", "")))
        if not path.is_file() or sha256_file(path) != gate.get("sha256"):
            raise ConfirmatoryCampaignError("Variance-selection artifact changed.")
        pilot_record = gate.get("pilot_campaign")
        if not isinstance(pilot_record, Mapping):
            raise ConfirmatoryCampaignError("Variance pilot lineage is missing.")
        pilot_path = Path(str(pilot_record.get("path", "")))
        pilot = _validate_completed_variance_pilot_manifest(
            pilot_path,
            expected_sha256=str(pilot_record.get("sha256", "")),
        )
        if (
            pilot.get("campaign_id") != pilot_record.get("campaign_id")
            or pilot.get("plan_sha256") != pilot_record.get("plan_sha256")
            or pilot.get("manifest_sha256")
            != pilot_record.get("manifest_payload_sha256")
        ):
            raise ConfirmatoryCampaignError("Variance pilot lineage changed.")
        payload = _load_json(path, label="variance-selection artifact")
        if (
            _validate_variance_gate(
                payload,
                pilot_manifest_path=pilot_path,
                expected_pilot_manifest_sha256=str(pilot_record.get("sha256", "")),
            )
            != gate.get("selected_n")
            or payload.get("artifact_sha256") != gate.get("artifact_sha256")
            or payload.get("private_input_sha256")
            != gate.get("private_input_sha256")
        ):
            raise ConfirmatoryCampaignError("Variance-selection n changed.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ConfirmatoryCampaignError("Manifest jobs are missing.")
    if len({job.get("id") for job in jobs if isinstance(job, Mapping)}) != len(jobs):
        raise ConfirmatoryCampaignError("Manifest job IDs are duplicated.")
    job_ids = {str(job["id"]) for job in jobs}
    for job in jobs:
        if not isinstance(job, Mapping) or job.get("status") not in _JOB_STATUSES:
            raise ConfirmatoryCampaignError("Manifest job state is invalid.")
        for field in ("argv_fresh", "argv_resume"):
            argv = job.get(field)
            if (
                not isinstance(argv, list)
                or not argv
                or not all(isinstance(item, str) and item for item in argv)
            ):
                raise ConfirmatoryCampaignError(f"Job {job.get('id')} argv is invalid.")
        if job.get("provider") != "inference_hub":
            raise ConfirmatoryCampaignError("Manifest contains a non-InferenceHub job.")
        smoke_id = job.get("smoke_job_id")
        if job.get("stage") != "smoke" and smoke_id not in job_ids:
            raise ConfirmatoryCampaignError("Scientific job lacks its matching smoke.")
    rebuilt = _build_plan_from_arguments(
        _rebuild_arguments_from_manifest(manifest),
        enforce_route_freshness=False,
        archived_validation=True,
    )
    if rebuilt.get("plan_sha256") != manifest.get("plan_sha256"):
        raise ConfirmatoryCampaignError(
            "Manifest job matrix or immutable planning fields differ from the sole "
            "plan reconstructed from pinned inputs."
        )


def create_manifest(manifest: dict[str, Any]) -> Path:
    """Persist a fresh validated plan beneath the private campaign root."""

    _require_fixed_execution_manifest(manifest, action="create")
    validate_manifest(manifest)
    directory = _campaign_directory(str(manifest["campaign_id"]))
    path = directory / "manifest.json"
    if directory.exists() or path.exists():
        raise ConfirmatoryCampaignError("Fresh campaign refuses an existing directory.")
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(CAMPAIGN_ROOT, 0o700)
    directory.mkdir(parents=True, mode=0o700)
    os.chmod(directory, 0o700)
    (directory / "logs").mkdir(mode=0o700)
    _atomic_json_write(directory / "request_budget.json", manifest["request_budget"])
    _atomic_json_write(
        directory / "request_ledger.json",
        _initial_request_ledger(manifest),
    )
    _write_manifest(path, manifest)
    return path


def _initial_request_ledger(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Seed the live ledger with every reserved route-smoke physical POST."""

    budget = manifest["request_budget"]
    ledger = create_ledger(budget)
    evidence = _load_json(
        manifest["inputs"]["endpoint_evidence"]["path"],
        label="endpoint evidence for budget seeding",
    )
    reference = evidence["discovery_attempt_ledger"]
    discovery_ledger = _load_json(
        reference["path"], label="discovery attempt ledger for budget seeding"
    )
    for record in discovery_ledger["records"]:
        ledger = record_attempt(
            ledger,
            budget,
            role="discovery",
            attempt_id=str(record["attempt_id"]),
            request_sha256=str(record["request_sha256"]),
            outcome=str(record["outcome"]),
            input_tokens=int(record["input_tokens"]),
            output_tokens=int(record["output_tokens"]),
        )
    return ledger


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = _load_json(manifest_path, label="confirmatory manifest")
    validate_manifest(manifest)
    if manifest_path != _campaign_directory(str(manifest["campaign_id"])) / "manifest.json":
        raise ConfirmatoryCampaignError("Manifest is outside its deterministic directory.")
    if manifest_path.stat().st_mode & 0o077:
        raise ConfirmatoryCampaignError("Manifest permissions are too permissive.")
    return manifest


def run_process(
    argv: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    timeout_seconds: int,
) -> ProcessResult:
    """Execute an argv vector directly; a shell is never involved."""

    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ConfirmatoryCampaignError("Subprocess argv must be nonempty strings.")
    log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            os.chmod(log_path, 0o600)
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=os.name != "nt",
            )
            try:
                return ProcessResult(process.wait(timeout=timeout_seconds))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return ProcessResult(
                    None,
                    timed_out=True,
                    error=f"timed out after {timeout_seconds} seconds",
                )
    except OSError as error:
        return ProcessResult(None, error=f"{type(error).__name__}: {error}")


def _part2_snapshot() -> dict[str, tuple[int, int, str]]:
    root = REPO_ROOT / "data" / "raw" / "part_2"
    if not root.exists():
        return {}
    return {
        str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns, sha256_file(path))
        for path in root.glob("*_meta.json")
        if path.is_file()
    }


def _verify_private_runner_artifact(job: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(str(job.get("output_dir", ""))).resolve()
    if job["experiment"] == "part0":
        plan_path = output / "part0_confirmatory_plan.json"
        results_path = output / "part0_confirmatory_results.jsonl"
        metadata_path = output / "part0_confirmatory_meta.json"
        attempts_path = attempt_log_path_for_csv(results_path)
        plan = _load_json(plan_path, label="Part 0 plan")
        loaded = part0_runner.load_production_registry(
            job["expected"]["registry_path"],
            expected_sha256=job["expected"]["registry_sha256"],
        )
        part0_runner.validate_execution_plan(plan, loaded)
        if (
            plan.get("judge_route", {}).get("route")
            != job["expected"]["judge_route"]
        ):
            raise ConfirmatoryCampaignError("Native Part 0 role routes changed.")
    else:
        plan_path = output / "part1_confirmatory_plan.json"
        results_path = output / "part1_confirmatory_results.jsonl"
        metadata_path = output / "part1_confirmatory_meta.json"
        attempts_path = attempt_log_path_for_csv(results_path)
        plan = _load_json(plan_path, label="Part 1 plan")
        loaded = part1_runner.load_production_bank(
            job["expected"]["bank_path"],
            expected_sha256=job["expected"]["bank_sha256"],
        )
        part1_runner.validate_execution_plan(plan, loaded)
    metadata = _load_json(metadata_path, label="confirmatory run metadata")
    validate_metadata_integrity(metadata, required=True)
    if metadata.get("status") != "complete":
        raise ConfirmatoryCampaignError("Native confirmatory artifact is incomplete.")
    if plan.get("execution_mode") != job["expected"]["execution_mode"]:
        raise ConfirmatoryCampaignError("Native execution mode does not match its job.")
    if plan.get("subject_route", {}).get("provider") != "inference_hub" or plan.get(
        "subject_route", {}
    ).get("route") != job["route"]:
        raise ConfirmatoryCampaignError("Native subject route does not match its job.")
    if metadata.get("run_contract", {}).get("plan_sha256") != plan.get("plan_sha256"):
        raise ConfirmatoryCampaignError("Native metadata/plan binding is invalid.")
    if not all(path.is_file() for path in (results_path, attempts_path)):
        raise ConfirmatoryCampaignError("Native confirmatory artifacts are incomplete.")
    verify_attempt_log_metadata(
        attempts_path,
        metadata.get("attempt_log", {}),
        require_hash_chain=True,
    )
    summary = (
        part0_runner.summarize_results(results_path)
        if job["experiment"] == "part0"
        else part1_runner.summarize_results(results_path)
    )
    if metadata.get("results") != summary:
        raise ConfirmatoryCampaignError("Native result hash chain differs from metadata.")
    marker = output / f"{job['experiment']}_confirmatory_analysis_exclude.json"
    if job["stage"] == "smoke":
        if not marker.is_file():
            raise ConfirmatoryCampaignError("Smoke artifact lacks exclusion marker.")
        expected_marker = (
            part0_runner._smoke_exclusion_payload(
                plan=plan, results_path=results_path, metadata_path=metadata_path
            )
            if job["experiment"] == "part0"
            else part1_runner._smoke_exclusion_payload(
                plan=plan, results_path=results_path, metadata_path=metadata_path
            )
        )
        if _load_json(marker, label="native smoke exclusion") != expected_marker:
            raise ConfirmatoryCampaignError("Native smoke exclusion marker is stale.")
    elif marker.exists():
        raise ConfirmatoryCampaignError("Production artifact contains smoke marker.")
    artifacts = [plan_path, results_path, metadata_path, attempts_path]
    if marker.exists():
        artifacts.append(marker)
    return {
        "kind": "private_confirmatory_run",
        "output_dir": str(output),
        "files": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in artifacts
        ],
    }


def _verify_part2_artifact(
    job: Mapping[str, Any], before: Mapping[str, Any] | None
) -> dict[str, Any]:
    from experiments import campaign as legacy_campaign

    if before is None:
        artifact = job.get("artifact")
        if not isinstance(artifact, Mapping) or not artifact.get("metadata_path"):
            raise ConfirmatoryCampaignError("Completed Part 2 job lacks its artifact.")
        metadata_path = REPO_ROOT / str(artifact["metadata_path"])
    else:
        after = _part2_snapshot()
        changed = sorted(path for path, fingerprint in after.items() if before.get(path) != fingerprint)
        if len(changed) != 1:
            raise ConfirmatoryCampaignError(
                f"Part 2 job must produce exactly one target+seed artifact; found {len(changed)}."
            )
        metadata_path = Path(changed[0])
    legacy_job = {
        "id": job["id"],
        "experiment": "part_2",
        "expected": deepcopy(job["expected"]),
    }
    try:
        verified = legacy_campaign.verify_artifact(legacy_job, metadata_path)
    except Exception as error:
        raise ConfirmatoryCampaignError(f"Part 2 artifact failed native verification: {error}") from error
    if job["stage"] == "smoke":
        csv_path = REPO_ROOT / str(verified["csv_path"])
        marker = csv_path.with_name(f"{csv_path.stem}.analysis_exclude.json")
        expected_marker = {
            "schema_version": 1,
            "scope": "canonical_analysis_and_validation",
            "reason": "sacrificial_campaign_smoke",
            "csv_sha256": sha256_file(csv_path),
        }
        if marker.exists():
            if _load_json(marker, label="Part 2 smoke exclusion") != expected_marker:
                raise ConfirmatoryCampaignError("Part 2 smoke exclusion marker is stale.")
        else:
            _atomic_json_write(marker, expected_marker)
        verified["analysis_exclusion"] = {
            "path": str(marker.relative_to(REPO_ROOT)),
            "sha256": sha256_file(marker),
        }
    return verified


def resolve_job_artifact(
    job: Mapping[str, Any], before: Mapping[str, Any] | None
) -> dict[str, Any]:
    if job["experiment"] in {"part0", "part1"}:
        return _verify_private_runner_artifact(job)
    return _verify_part2_artifact(job, before)


def execute_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    *,
    process_runner: ProcessRunner = run_process,
    artifact_resolver: ArtifactResolver = resolve_job_artifact,
) -> dict[str, Any]:
    """Execute/resume jobs sequentially while enforcing matching-smoke gates."""

    _require_fixed_execution_manifest(manifest, action="execute")
    validate_manifest(manifest)
    manifest["status"] = "running"
    _write_manifest(manifest_path, manifest)
    job_by_id = {str(job["id"]): job for job in manifest["jobs"]}
    logs = manifest_path.parent / "logs"
    for job in manifest["jobs"]:
        if job["status"] == "complete":
            job["artifact"] = artifact_resolver(job, None)
            _write_manifest(manifest_path, manifest)
            continue
        if job["status"] in {"failed", "blocked_route_health"}:
            # Retry exhaustion and route-health quarantine are permanent for
            # this immutable campaign. A new dispatch requires a new freeze.
            continue
        smoke_id = job.get("smoke_job_id")
        if smoke_id is not None and job_by_id[smoke_id]["status"] != "complete":
            job["status"] = "blocked_smoke"
            job["last_error"] = f"matching smoke incomplete: {smoke_id}"
            _write_manifest(manifest_path, manifest)
            continue
        prior_route_failure = next(
            (
                prior
                for prior in manifest["jobs"]
                if prior is not job
                and prior.get("target_id") == job.get("target_id")
                and prior.get("experiment") == job.get("experiment")
                and prior.get("status") == "failed"
            ),
            None,
        )
        if prior_route_failure is not None:
            job["status"] = "blocked_route_health"
            job["last_error"] = (
                "same target/part quarantined after failed job: "
                f"{prior_route_failure['id']}"
            )
            _write_manifest(manifest_path, manifest)
            continue
        if job["status"] in {
            "blocked_smoke",
            "running",
        }:
            job["status"] = "pending"
        # Part 0/1 expose deterministic private directories and require an
        # explicit resume flag only after those artifacts exist.  Part 2's
        # fresh argv already locates an exact incomplete target/config/seed
        # artifact and resumes it internally; passing a generic --resume with
        # no known artifact could select unrelated state.
        use_resume = bool(
            job.get("output_dir")
            and Path(str(job["output_dir"])).exists()
        )
        argv = list(job["argv_resume"] if use_resume else job["argv_fresh"])
        if argv != (job["argv_resume"] if use_resume else job["argv_fresh"]):
            raise ConfirmatoryCampaignError("Job argv changed before execution.")
        before = _part2_snapshot() if job["experiment"] == "part2" else None
        attempt = {
            "attempt": len(job["attempts"]) + 1,
            "started_at_utc": _utc_now(),
            "argv_sha256": stable_json_hash(argv),
            "resume": use_resume,
            "finished_at_utc": None,
            "returncode": None,
            "timed_out": False,
            "error": None,
        }
        job["attempts"].append(attempt)
        job["status"] = "running"
        job["last_error"] = None
        _write_manifest(manifest_path, manifest)
        log_path = logs / f"{job['id']}.log"
        process_env = os.environ.copy()
        process_env.update(
            {
                "CONFIRMATORY_BUDGET_PATH": str(manifest_path.parent / "request_budget.json"),
                "CONFIRMATORY_LEDGER_PATH": str(manifest_path.parent / "request_ledger.json"),
                "CONFIRMATORY_EXPERIMENT": str(job["experiment"]),
            }
        )
        validate_frozen_budget(manifest["request_budget"])
        validate_ledger(
            _load_json(manifest_path.parent / "request_ledger.json", label="request ledger"),
            manifest["request_budget"],
        )
        result = process_runner(
            argv,
            REPO_ROOT,
            process_env,
            log_path,
            int(manifest["timeout_seconds"]),
        )
        attempt.update(
            finished_at_utc=_utc_now(),
            returncode=result.returncode,
            timed_out=result.timed_out,
            error=result.error,
        )
        if result.returncode != 0:
            job["status"] = "failed"
            job["last_error"] = result.error or f"subprocess exit {result.returncode}"
            _write_manifest(manifest_path, manifest)
            continue
        try:
            job["artifact"] = artifact_resolver(job, before)
        except Exception as error:
            job["status"] = "failed"
            job["last_error"] = f"{type(error).__name__}: {error}"
            _write_manifest(manifest_path, manifest)
            continue
        job["status"] = "complete"
        _write_manifest(manifest_path, manifest)
    statuses = {str(job["status"]) for job in manifest["jobs"]}
    manifest["status"] = "complete" if statuses == {"complete"} else "incomplete"
    _write_manifest(manifest_path, manifest)
    return manifest


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the isolated strict confirmatory campaign."
    )
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--cohort", action="append", default=None)
    parser.add_argument(
        "--target-id",
        action="append",
        help=(
            "Repeatable exact target shard; defaults to the full requested cohort "
            "union. Use disjoint shards for parallel execution within route freshness."
        ),
    )
    parser.add_argument("--judge-target-id")
    parser.add_argument("--part0-registry")
    parser.add_argument("--part0-registry-sha256")
    parser.add_argument("--part1-bank")
    parser.add_argument("--part1-bank-sha256")
    parser.add_argument("--endpoint-evidence")
    parser.add_argument("--endpoint-evidence-sha256")
    parser.set_defaults(
        part2_stage="fixed-production",
        variance_selection=None,
        variance_selection_sha256=None,
        variance_pilot_manifest=None,
        variance_pilot_manifest_sha256=None,
    )
    parser.add_argument("--part2-society-size", type=_positive_int, default=10)
    parser.add_argument("--part2-days", type=_positive_int, default=30)
    parser.add_argument("--part2-resource-capacity", type=_positive_int, default=150)
    parser.add_argument("--part2-resource", default="water")
    parser.add_argument("--part2-selfish-gain", type=_positive_int, default=2)
    parser.add_argument("--part2-depletion-units", type=_positive_int, default=2)
    parser.add_argument("--part2-community-benefit", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=_positive_int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def _require_planning_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    required = (
        "judge_target_id",
        "part0_registry",
        "part0_registry_sha256",
        "part1_bank",
        "part1_bank_sha256",
        "endpoint_evidence",
        "endpoint_evidence_sha256",
    )
    missing = [f"--{field.replace('_', '-')}" for field in required if not getattr(args, field)]
    if missing:
        parser.error("planning requires " + ", ".join(missing))


def _summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for job in manifest["jobs"]:
        key = f"{job['stage']}:{job['experiment']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "campaign_id": manifest["campaign_id"],
        "status": manifest["status"],
        "target_count": len(manifest["targets"]),
        "job_count": len(manifest["jobs"]),
        "job_counts": counts,
        "part2_design": manifest["part2_design"],
        "plan_sha256": manifest["plan_sha256"],
        "dry_run": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resume and args.dry_run:
        parser.error("--resume and --dry-run cannot be combined")
    if args.resume:
        manifest_path = _campaign_directory(args.campaign_id) / "manifest.json"
        manifest = load_manifest(manifest_path)
        execute_manifest(manifest, manifest_path)
    else:
        _require_planning_arguments(parser, args)
        manifest = build_plan(args)
        if args.dry_run:
            summary = _summary(manifest)
            summary["dry_run"] = True
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        manifest_path = create_manifest(manifest)
        execute_manifest(manifest, manifest_path)
    print(json.dumps(_summary(manifest), indent=2, sort_keys=True))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ConfirmatoryCampaignError",
    "ProcessResult",
    "build_parser",
    "build_plan",
    "create_manifest",
    "execute_manifest",
    "load_manifest",
    "main",
    "resolve_job_artifact",
    "run_process",
    "validate_manifest",
]
