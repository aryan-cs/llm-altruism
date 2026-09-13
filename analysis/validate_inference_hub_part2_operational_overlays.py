"""Read-only validation for the definitive 100-day Part 2 evidence union.

The definitive campaign is deliberately sharded into three source runs and
three source-bound operational-repair overlays.  This module validates those
three pairs without dispatching a request or modifying any evidence file.  It
holds shared run locks for the complete snapshot, validates every manifest,
artifact, and journal hash, replays every source and repair round through the
frozen simulator, and reconciles the exact 23-route, 12-seed effective union.

No raw prompt, response, reasoning, credential, or route is emitted by the
command-line interface.  A live or incomplete overlay is rejected before any
trajectory journal is read.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator, Mapping, Sequence

from agents.agent_2 import Agent2
from analysis import finalize_inference_hub_part2_offline as offline
from experiments.misc import inference_hub_part2_cascading_operational_repair as cascading
from experiments.misc import inference_hub_part2_operational_repair as repair
from experiments.misc import inference_hub_part2_panel as runner
from experiments.misc.inference_hub_part1_panel import (
    _canonical_bytes,
    _require_mode,
    _safe_file_stem,
    _self_hash,
    _sha256_file,
    _sha256_json,
    InferenceHubPart1PanelError,
)
from experiments.part2.part_2 import _collapse_deaths, _derive_seed


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "inference_hub_part2_operational_overlay_validation_v1"
EXPECTED_PANEL_ID = "sota_cross_axis_part2_corrected_original_scale_100d_v1"
EXPECTED_BASE_SEED = 20_260_802
EXPECTED_ROUTE_COUNT = 23
EXPECTED_TRAJECTORIES_PER_ROUTE = 12
EXPECTED_TRAJECTORY_COUNT = EXPECTED_ROUTE_COUNT * EXPECTED_TRAJECTORIES_PER_ROUTE
EXPECTED_PANEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/sota_cross_axis_part2_100day_panel.json"
)
EXPECTED_PANEL_FILE_SHA256 = (
    "262af694302601ebab6c7362becb9c92f0aac7fb9735987c91878d1f91a43455"
)
EXPECTED_PANEL_CANONICAL_SHA256 = (
    "9afacf612abbf0dd931481dec1588af6ff3b305a46fe0fe5dade728d012919ce"
)
EXPECTED_COMMON_ENVIRONMENT_SEEDS = (
    945_353_965,
    674_434_863,
    373_620_026,
    161_949_049,
    305_447_851,
    1_694_051_603,
    827_330_313,
    526_445_251,
    1_853_673_051,
    1_941_677_986,
    1_517_370_141,
    597_333_924,
)
EXPECTED_EXCLUDED_TARGET_IDS = frozenset({"anthropic/claude-opus-4-5"})
EXPECTED_SINGLETON_SHARDS = frozenset(
    {
        "deepseek-ai/deepseek-v4-flash",
        "nvidia/nemotron-3-ultra",
    }
)

EXPECTED_SOURCE_TRAJECTORY_WORKERS = 48
EXPECTED_SOURCE_PARTICIPANT_WORKERS = 16
EXPECTED_SOURCE_MAX_TRANSPORT_ATTEMPTS = 8
EXPECTED_SOURCE_INITIAL_BACKOFF_SECONDS = 1.0
EXPECTED_REPAIR_MAXIMUM_ROUNDS = 8
EXPECTED_SHARED_RATE_LIMIT_POLICY = {
    "schema_version": 2,
    "algorithm": (
        "cross_process_provider_aware_leaky_bucket_with_leases_all_http_5xx_"
        "full_throttle_cooldown"
    ),
    "global_concurrency": 60,
    "provider_concurrency": 10,
    "global_requests_per_second": 12.0,
    "provider_requests_per_second": 2.5,
    "lease_seconds": 900.0,
    "poll_seconds": 0.05,
    "throttle_cooldown_seconds": 30.0,
    "transient_cooldown_seconds": 5.0,
}
EXPECTED_SHARED_RATE_LIMIT_CONTRACT = {
    **EXPECTED_SHARED_RATE_LIMIT_POLICY,
    "policy_sha256": _sha256_json(EXPECTED_SHARED_RATE_LIMIT_POLICY),
}
EXPECTED_SOURCE_EXECUTION_CONTRACT = {
    "strategy": (
        "parallel_target_trajectory_and_parallel_participants_with_sequential_days"
    ),
    "trajectory_workers": EXPECTED_SOURCE_TRAJECTORY_WORKERS,
    "participant_workers": EXPECTED_SOURCE_PARTICIPANT_WORKERS,
    "max_transport_attempts": EXPECTED_SOURCE_MAX_TRANSPORT_ATTEMPTS,
    "initial_exponential_backoff_seconds": EXPECTED_SOURCE_INITIAL_BACKOFF_SECONDS,
    "shared_rate_limit": EXPECTED_SHARED_RATE_LIMIT_CONTRACT,
    "journal": "per_trajectory_append_only_fsync_sha256_chain_reserve_before_dispatch",
    "identity_check": "exact_returned_model_equals_selected_route",
    "visible_output_only": True,
}
EXPECTED_RUNNER_IMPLEMENTATION_SHA256 = (
    "28c05c1c730390c61577b3874b74f5a3b2252f6bdc4badac8d9e393b70eaa138"
)
EXPECTED_REPAIR_IMPLEMENTATION_SHA256 = (
    "4a8e314ab02e0ade62ebccb3d4b171d905b0c252edd9e44e660c4dbe23b27a73"
)
EXPECTED_CASCADING_REPAIR_IMPLEMENTATION_SHA256 = (
    "3d18d7a0ab11cc43e5648c60abc24b243e18e5195437cc345cd7479e43301dec"
)
EXPECTED_DISCOVERY_IMPLEMENTATION_SHA256 = (
    "f88b0641bf7c8676c97439ebb354d735d7000bd2ab9b759c94dc64c48e29bd56"
)
EXPECTED_RATE_LIMIT_IMPLEMENTATION_SHA256 = (
    "abec4df6fc66b9e044d2e79ff5b38c9af3f030af74e552f039276b246033497e"
)
EXPECTED_MAIN_PARENT_SOURCE_FAILURES = 38
EXPECTED_MAIN_PARENT_SUCCESSES = 37
EXPECTED_CASCADING_TARGET_ID = "google/gemini-3.5-flash"
EXPECTED_CASCADING_TRAJECTORY_INDEX = 1
EXPECTED_CASCADING_ENVIRONMENT_SEED = 674_434_863
EXPECTED_CASCADING_REPAIR_POLICY = (
    "cascading_whole_trajectory_day_one_exact_route_qualified_pool_v1"
)
EXPECTED_CASCADING_SELECTION_POLICY = (
    "locked_round_robin_over_qualified_account_slots_v1"
)
EXPECTED_CASCADING_RATE_LIMIT_SCOPE = (
    "independent_v2_limiter_per_qualified_account"
)
EXPECTED_CASCADING_CONFIGURED_ACCOUNT_COUNT = 3
EXPECTED_CASCADING_MAXIMUM_ROUNDS = 8
EXPECTED_CASCADING_TRAJECTORY_WORKERS = 1
EXPECTED_CASCADING_PARTICIPANT_WORKERS = 30
EXPECTED_CASCADING_MAX_ATTEMPTS = 8
EXPECTED_CASCADING_INITIAL_BACKOFF_SECONDS = 1.0
EXPECTED_CASCADING_TIMEOUT_SECONDS = 900.0
EXPECTED_CASCADING_RATE_PROFILE = "accelerated_original_scale_high_latency_v2"
EXPECTED_CASCADING_COMPOSITION_CONTRACT = {
    "policy": "verified_parent_successes_plus_current_unresolved_only",
    "parent_maximum_rounds": 8,
    "child_maximum_rounds": 8,
    "cumulative_maximum_rounds_per_selected_trajectory": 16,
    "inherited_parent_success_count": 37,
    "selected_parent_unresolved_count": 1,
    "restart_scope": "whole_trajectory_from_day_one",
    "semantic_only_selection_forbidden": True,
}
EXPECTED_CASCADING_EXECUTION_CONTRACT = {
    "strategy": "single_selected_trajectory_parallel_participants_sequential_days",
    "trajectory_workers": EXPECTED_CASCADING_TRAJECTORY_WORKERS,
    "participant_workers": EXPECTED_CASCADING_PARTICIPANT_WORKERS,
    "max_transport_attempts": EXPECTED_CASCADING_MAX_ATTEMPTS,
    "initial_exponential_backoff_seconds": (
        EXPECTED_CASCADING_INITIAL_BACKOFF_SECONDS
    ),
    "request_timeout_seconds": EXPECTED_CASCADING_TIMEOUT_SECONDS,
    "maximum_rounds": EXPECTED_CASCADING_MAXIMUM_ROUNDS,
    "rate_profile": EXPECTED_CASCADING_RATE_PROFILE,
    "retry_policy": {
        "connection_failure_codes": ["connection_error", "connection_timeout"],
        "http_statuses": [400, 429, *range(500, 600)],
        "request_mutation_permitted": False,
        "semantic_retry_permitted": False,
    },
    "journal": (
        "per_trajectory_append_only_fsync_sha256_chain_account_bound_"
        "reserve_before_dispatch"
    ),
    "identity_check": "exact_returned_model_equals_selected_route",
    "visible_output_only": True,
}

# The repair manifest deliberately binds maximum_rounds and, for the main
# multikey overlay, its credential-pool implementation and rate contract.  The
# repair CLI did not persist these local scheduling knobs.  They do not alter a
# prompt, seed, decoding control, simulator transition, or semantic retry
# policy, so this validator never fabricates configured values for them.
OVERLAY_UNBOUND_NONSEMANTIC_LOCAL_SETTINGS = (
    "trajectory_workers",
    "participant_workers",
    "initial_exponential_backoff_seconds",
)

SOURCE_MANIFEST_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "created_at_utc",
        "panel_id",
        "input_artifacts",
        "source_artifacts",
        "runtime",
        "runtime_sha256",
        "subject_routes",
        "judge_reservation",
        "part2_contract",
        "base_seed",
        "common_environment_seeds",
        "execution_contract",
        "complete",
        "summary",
        "journals",
        "sanitized_artifacts",
        "last_updated_at_utc",
        "evidence_sha256",
    }
)
SOURCE_MANIFEST_OPTIONAL_KEYS = frozenset(
    {"completed_at_utc", "resume_count", "last_resumed_at_utc"}
)
OVERLAY_MANIFEST_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_manifest",
        "panel_id",
        "part2_contract",
        "common_environment_seeds",
        "subject_routes",
        "repair_policy",
        "maximum_rounds",
        "created_at_utc",
        "last_updated_at_utc",
        "complete",
        "summary",
        "journals",
        "sanitized_artifacts",
        "evidence_sha256",
    }
)
OVERLAY_MANIFEST_OPTIONAL_KEYS = frozenset({"completed_at_utc", "credential_pool"})

SOURCE_TRAJECTORY_TOP_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "panel_id",
        "generated_at_utc",
        "independence_unit",
        "rows",
        "evidence_sha256",
    }
)
SOURCE_MODEL_TOP_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "panel_id",
        "generated_at_utc",
        "uncertainty_unit",
        "rows",
        "evidence_sha256",
    }
)
OVERLAY_TOP_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "panel_id",
        "generated_at_utc",
        "source_manifest_evidence_sha256",
        "rows",
        "evidence_sha256",
    }
)
TRAJECTORY_ROW_KEYS = frozenset(
    {
        "schema_version",
        "target_id",
        "upstream_provider",
        "model",
        "trajectory_index",
        "environment_seed_index",
        "environment_seed",
        "operationally_eligible",
        "scheduled_agent_days",
        "responses_received",
        "invalid_count",
        "identity_mismatch_count",
        "transport_failure_count",
        "restraint_count",
        "overuse_count",
        "restraint_rate",
        "aurc",
        "aupc",
        "reserve_nondepletion",
        "final_reserve",
        "final_population",
        "population_retention",
        "cumulative_private_payoff",
        "cumulative_group_payoff",
    }
)
OVERLAY_TRAJECTORY_ROW_KEYS = TRAJECTORY_ROW_KEYS | frozenset(
    {"operational_repair_round", "source_replaced_for_operational_failure"}
)
MODEL_ROW_KEYS = frozenset(
    {
        "schema_version",
        "target_id",
        "upstream_provider",
        "model",
        "trajectory_count",
        "eligible_trajectory_count",
        "expected_trajectory_count",
        "complete_matched_panel",
        "total_scheduled_agent_days",
        "total_invalid_count",
        "total_identity_mismatch_count",
        "total_transport_failure_count",
        "trajectory_level_95_percent_t_intervals",
    }
)


class Part2OperationalOverlayValidationError(RuntimeError):
    """The definitive Part 2 source/overlay union is not independently valid."""


@dataclass
class _FileTracker:
    """Track byte bindings so the validated snapshot can be rechecked at exit."""

    digests: dict[Path, str] = field(default_factory=dict)
    absent: set[Path] = field(default_factory=set)
    secure_0600: set[Path] = field(default_factory=set)

    def add(
        self, path: Path, digest: str | None = None, *, secure_0600: bool = False,
    ) -> str:
        resolved = path.resolve()
        observed = _sha256_file(resolved) if digest is None else digest
        previous = self.digests.get(resolved)
        if previous is not None and previous != observed:
            raise Part2OperationalOverlayValidationError(
                "One evidence file acquired two different snapshot hashes."
            )
        self.digests[resolved] = observed
        self.absent.discard(resolved)
        if secure_0600:
            self.secure_0600.add(resolved)
        return observed

    def add_absent(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved in self.digests:
            raise Part2OperationalOverlayValidationError(
                "An evidence path is both present and absent in the snapshot."
            )
        self.absent.add(resolved)

    def verify(self) -> None:
        for path, digest in self.digests.items():
            if path in self.secure_0600:
                observed = hashlib.sha256(
                    _read_regular_nonsymlink_0600(
                        path, label="Tracked secure evidence file"
                    )
                ).hexdigest()
            else:
                observed = _sha256_file(path) if path.is_file() else None
            if observed != digest:
                raise Part2OperationalOverlayValidationError(
                    "An evidence file changed during read-only validation."
                )
        if any(path.exists() for path in self.absent):
            raise Part2OperationalOverlayValidationError(
                "An unused repair journal appeared during read-only validation."
            )


@dataclass(frozen=True)
class _SourceEvidence:
    manifest_path: Path
    manifest_file_sha256: str
    manifest: Mapping[str, Any]
    subjects: tuple[Mapping[str, Any], ...]
    contract: runner.Part2Contract
    environment_seeds: tuple[int, ...]
    trajectories: tuple[Mapping[str, Any], ...]
    models: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class _ValidatedPair:
    source_manifest_path: Path
    overlay_manifest_path: Path
    source_manifest: Mapping[str, Any]
    overlay_manifest: Mapping[str, Any]
    subjects: tuple[Mapping[str, Any], ...]
    contract: runner.Part2Contract
    environment_seeds: tuple[int, ...]
    effective_trajectories: tuple[Mapping[str, Any], ...]
    effective_models: tuple[Mapping[str, Any], ...]
    source_operational_failure_count: int
    repair_round_count: int


@dataclass(frozen=True)
class _ValidatedPartialOverlay:
    source: _SourceEvidence
    parent_manifest_path: Path
    parent_manifest_file_sha256: str
    parent_manifest: Mapping[str, Any]
    effective_trajectories: tuple[Mapping[str, Any], ...]
    effective_models: tuple[Mapping[str, Any], ...]
    source_failures: Mapping[tuple[str, int], Mapping[str, Any]]
    successful: Mapping[tuple[str, int], tuple[int, Mapping[str, Any]]]
    unresolved: Mapping[tuple[str, int], Mapping[str, Any]]
    repair_round_count: int


def _integer(value: object, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _utc_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise Part2OperationalOverlayValidationError(
            f"{label} is not a canonical UTC timestamp."
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise Part2OperationalOverlayValidationError(
            f"{label} is not a canonical UTC timestamp."
        ) from error
    if parsed.tzinfo != timezone.utc:
        raise Part2OperationalOverlayValidationError(
            f"{label} is not a canonical UTC timestamp."
        )
    return parsed


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        raise Part2OperationalOverlayValidationError(f"{label} schema changed.")


def _read_regular_nonsymlink_0600(path: Path, *, label: str) -> bytes:
    """Read one exact secure file without following a replaceable leaf symlink."""

    try:
        path_status = path.lstat()
    except OSError as error:
        raise Part2OperationalOverlayValidationError(
            f"{label} is unavailable."
        ) from error
    if (
        stat.S_ISLNK(path_status.st_mode)
        or not stat.S_ISREG(path_status.st_mode)
        or stat.S_IMODE(path_status.st_mode) != 0o600
    ):
        raise Part2OperationalOverlayValidationError(
            f"{label} must be a regular nonsymlink 0600 file."
        )
    descriptor = -1
    try:
        descriptor = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        descriptor_status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_status.st_mode)
            or stat.S_IMODE(descriptor_status.st_mode) != 0o600
            or (descriptor_status.st_dev, descriptor_status.st_ino)
            != (path_status.st_dev, path_status.st_ino)
        ):
            raise Part2OperationalOverlayValidationError(
                f"{label} changed while it was being opened."
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    except OSError as error:
        raise Part2OperationalOverlayValidationError(
            f"{label} cannot be opened without following symlinks."
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _safe_json_0600(
    path: Path, *, label: str, tracker: _FileTracker,
) -> dict[str, Any]:
    raw = _read_regular_nonsymlink_0600(path, label=label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Part2OperationalOverlayValidationError(
            f"{label} is not a readable JSON object."
        ) from error
    if not isinstance(value, dict):
        raise Part2OperationalOverlayValidationError(
            f"{label} is not a JSON object."
        )
    tracker.add(
        path, hashlib.sha256(raw).hexdigest(), secure_0600=True,
    )
    return value


def _safe_json(path: Path, *, label: str, tracker: _FileTracker) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Part2OperationalOverlayValidationError(
            f"{label} is not a readable JSON object."
        ) from error
    if not isinstance(value, dict):
        raise Part2OperationalOverlayValidationError(f"{label} is not a JSON object.")
    tracker.add(path, hashlib.sha256(raw).hexdigest())
    return value


def _private_manifest_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.name != "manifest.json" or resolved.parent.name != "private":
        raise Part2OperationalOverlayValidationError(
            "Each input must be an existing private/manifest.json file."
        )
    if not resolved.is_file():
        raise Part2OperationalOverlayValidationError("A required private manifest is absent.")
    _require_mode(resolved.parent, 0o700)
    _require_mode(resolved, 0o600)
    return resolved


@contextmanager
def _hold_run_locks(manifest_paths: Sequence[Path]) -> Iterator[None]:
    """Hold shared locks without creating or changing any lock file."""

    resolved = [_private_manifest_path(path) for path in manifest_paths]
    if len(set(resolved)) != len(resolved):
        raise Part2OperationalOverlayValidationError("A manifest input is duplicated.")
    handles: list[Any] = []
    try:
        for manifest_path in sorted(resolved):
            lock_path = manifest_path.parent / ".run.lock"
            if not lock_path.is_file():
                raise Part2OperationalOverlayValidationError(
                    "A required run lock file is absent; refusing an unlocked snapshot."
                )
            _require_mode(lock_path, 0o600)
            handle = lock_path.open("rb")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise Part2OperationalOverlayValidationError(
                    "A Part 2 source or operational overlay is active."
                ) from error
            handles.append(handle)
        yield
    except (OSError, offline.OfflinePart2FinalizationError, InferenceHubPart1PanelError) as error:
        if isinstance(error, Part2OperationalOverlayValidationError):
            raise
        raise Part2OperationalOverlayValidationError(str(error)) from error
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def _preflight_overlay(
    manifest_path: Path, tracker: _FileTracker,
) -> dict[str, Any]:
    manifest = _safe_json(manifest_path, label="operational overlay manifest", tracker=tracker)
    allowed = OVERLAY_MANIFEST_REQUIRED_KEYS | OVERLAY_MANIFEST_OPTIONAL_KEYS
    if not OVERLAY_MANIFEST_REQUIRED_KEYS <= set(manifest) or not set(manifest) <= allowed:
        raise Part2OperationalOverlayValidationError("Operational overlay manifest schema changed.")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != repair.ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != _self_hash(manifest)
    ):
        raise Part2OperationalOverlayValidationError(
            "Operational overlay manifest type or self-hash failed."
        )
    if manifest.get("complete") is not True:
        raise Part2OperationalOverlayValidationError(
            "An operational overlay is incomplete; no definitive union exists."
        )
    created = _utc_datetime(
        manifest.get("created_at_utc"), label="Operational overlay creation timestamp"
    )
    updated = _utc_datetime(
        manifest.get("last_updated_at_utc"),
        label="Operational overlay update timestamp",
    )
    completed = _utc_datetime(
        manifest.get("completed_at_utc"),
        label="Operational overlay completion timestamp",
    )
    if not created <= updated <= completed:
        raise Part2OperationalOverlayValidationError(
            "Operational overlay timestamps are out of order."
        )
    return manifest


def _preflight_partial_parent_overlay(
    manifest_path: Path, tracker: _FileTracker,
) -> dict[str, Any]:
    manifest = _safe_json(
        manifest_path, label="incomplete parent operational overlay manifest",
        tracker=tracker,
    )
    allowed = OVERLAY_MANIFEST_REQUIRED_KEYS | OVERLAY_MANIFEST_OPTIONAL_KEYS
    if not OVERLAY_MANIFEST_REQUIRED_KEYS <= set(manifest) or not set(manifest) <= allowed:
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent operational overlay manifest schema changed."
        )
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != repair.ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != _self_hash(manifest)
    ):
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent operational overlay type or self-hash failed."
        )
    if manifest.get("complete") is not False or "completed_at_utc" in manifest:
        raise Part2OperationalOverlayValidationError(
            "Cascading parent must be incomplete and lack a completion timestamp."
        )
    created = _utc_datetime(
        manifest.get("created_at_utc"), label="Parent overlay creation timestamp"
    )
    updated = _utc_datetime(
        manifest.get("last_updated_at_utc"), label="Parent overlay update timestamp"
    )
    if created > updated:
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent overlay timestamps are out of order."
        )
    return manifest


def _preflight_cascading_overlay(
    manifest_path: Path, tracker: _FileTracker,
) -> dict[str, Any]:
    manifest = _safe_json(
        manifest_path, label="cascading operational overlay manifest", tracker=tracker,
    )
    allowed = cascading.MANIFEST_REQUIRED_KEYS | cascading.MANIFEST_OPTIONAL_KEYS
    if (
        not cascading.MANIFEST_REQUIRED_KEYS <= set(manifest)
        or not set(manifest) <= allowed
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading operational overlay manifest schema changed."
        )
    if (
        manifest.get("schema_version") != cascading.SCHEMA_VERSION
        or manifest.get("artifact_type") != cascading.ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != _self_hash(manifest)
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading operational overlay type or self-hash failed."
        )
    if manifest.get("complete") is not True:
        raise Part2OperationalOverlayValidationError(
            "A cascading operational overlay is incomplete; no definitive union exists."
        )
    try:
        cascading._assert_no_secret_like_fields(
            manifest, label="cascading operational overlay manifest"
        )
    except cascading.Part2CascadingRepairError as error:
        raise Part2OperationalOverlayValidationError(str(error)) from error
    created = _utc_datetime(
        manifest.get("created_at_utc"), label="Cascading overlay creation timestamp"
    )
    updated = _utc_datetime(
        manifest.get("last_updated_at_utc"), label="Cascading overlay update timestamp"
    )
    completed = _utc_datetime(
        manifest.get("completed_at_utc"), label="Cascading overlay completion timestamp"
    )
    if not created <= updated <= completed:
        raise Part2OperationalOverlayValidationError(
            "Cascading operational overlay timestamps are out of order."
        )
    return manifest


def _read_journal_reference(
    reference: object, *, expected_path: Path, label: str, tracker: _FileTracker,
) -> list[dict[str, Any]]:
    if not isinstance(reference, Mapping) or set(reference) != {
        "path", "record_count", "tail_record_sha256", "file_sha256"
    }:
        raise Part2OperationalOverlayValidationError(f"{label} reference schema changed.")
    path_value = reference.get("path")
    if not isinstance(path_value, str) or Path(path_value).resolve() != expected_path.resolve():
        raise Part2OperationalOverlayValidationError(f"{label} path binding changed.")
    count = reference.get("record_count")
    if not _integer(count):
        raise Part2OperationalOverlayValidationError(f"{label} record count is invalid.")
    path = expected_path.resolve()
    if not path.exists():
        if count != 0 or reference.get("tail_record_sha256") is not None or reference.get("file_sha256") is not None:
            raise Part2OperationalOverlayValidationError(f"{label} missing-file binding is invalid.")
        tracker.add_absent(path)
        return []
    if not path.is_file():
        raise Part2OperationalOverlayValidationError(f"{label} is not a regular file.")
    _require_mode(path, 0o600)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if text and not text.endswith("\n"):
            raise ValueError("missing final delimiter")
        lines = [] if not text else text[:-1].split("\n")
        if any(not line for line in lines):
            raise ValueError("blank journal record")
        rows = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise Part2OperationalOverlayValidationError(f"{label} is not valid JSONL.") from error
    previous = None
    for row in rows:
        if not isinstance(row, dict):
            raise Part2OperationalOverlayValidationError(f"{label} contains a non-object record.")
        recorded = row.get("record_sha256")
        unhashed = {key: item for key, item in row.items() if key != "record_sha256"}
        if row.get("previous_record_sha256") != previous or recorded != _sha256_json(unhashed):
            raise Part2OperationalOverlayValidationError(f"{label} hash chain failed.")
        previous = recorded
    digest = hashlib.sha256(raw).hexdigest()
    if (
        len(rows) != count
        or reference.get("tail_record_sha256") != previous
        or reference.get("file_sha256") != digest
    ):
        raise Part2OperationalOverlayValidationError(f"{label} checkpoint binding changed.")
    tracker.add(path, digest)
    return rows


def _validate_failure(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get("failure_code"), str):
        raise Part2OperationalOverlayValidationError(f"{label} failure metadata is invalid.")
    if not isinstance(value.get("transient"), bool) or value.get("http_status") is not None and not _integer(value.get("http_status"), minimum=100):
        raise Part2OperationalOverlayValidationError(f"{label} failure metadata is invalid.")
    return value


def _validate_semantic_payload(
    row: Mapping[str, Any], *, reservation: Mapping[str, Any],
    subject: Mapping[str, Any], trajectory_index: int, environment_seed: int,
) -> None:
    prompt = row.get("prompt_text")
    body = row.get("request_body")
    if not isinstance(prompt, str) or not isinstance(body, Mapping):
        raise Part2OperationalOverlayValidationError("A semantic result lacks its private request binding.")
    slot = int(row["slot"])
    expected_generation_seed = _derive_seed(
        "inference_hub_part2_generation_v1",
        str(subject["target_id"]), environment_seed, int(row["day"]), slot,
    )
    agent = Agent2(f"slot_{slot:02d}", "inference_hub", str(subject["route"]))
    expected_body, expected_controls = runner._request_contract(
        subject, prompt=prompt, system_prompt=agent.system_prompt,
        generation_seed=expected_generation_seed,
    )
    request_bytes = _canonical_bytes(expected_body)
    request_hash = hashlib.sha256(request_bytes).hexdigest()
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    expected_shared = {
        "attempt_number": reservation.get("attempt_number"),
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "requested_route": subject["route"],
        "environment_seed": environment_seed,
        "generation_seed": expected_generation_seed,
        "prompt_sha256": prompt_hash,
        "request_sha256": request_hash,
        "controls": expected_controls,
    }
    if dict(body) != expected_body or any(row.get(key) != value for key, value in expected_shared.items()):
        raise Part2OperationalOverlayValidationError("A semantic result changed its request or route binding.")
    if any(reservation.get(key) != value for key, value in expected_shared.items() if key != "attempt_number"):
        raise Part2OperationalOverlayValidationError("An attempt reservation changed its request or route binding.")
    if reservation.get("request_body_bytes") != len(request_bytes):
        raise Part2OperationalOverlayValidationError("An attempt reservation byte count changed.")

    raw = row.get("raw_response")
    if raw is None:
        failure = _validate_failure(row.get("failure"), label="transport result")
        if (
            row.get("invalid_reason") != "transport_failure_exhausted"
            or row.get("model_identity_valid") is not False
            or row.get("response_model") is not None
            or row.get("raw_response_sha256") is not None
            or row.get("visible_content") is not None
            or row.get("visible_content_sha256") is not None
            or row.get("request_id") is not None
            or row.get("finish_reason") is not None
            or row.get("usage") is not None
            or row.get("action") != "INVALID"
            or row.get("reasoning") != ""
            or row.get("format_valid") is not False
            or failure.get("failure_code") == ""
        ):
            raise Part2OperationalOverlayValidationError("Transport-failure evidence is inconsistent.")
        return
    if not isinstance(raw, Mapping):
        raise Part2OperationalOverlayValidationError("A retained raw response is not an object.")
    response = dict(raw)
    if row.get("raw_response_sha256") != _sha256_json(response):
        raise Part2OperationalOverlayValidationError("A retained raw response hash changed.")
    response_model = response.get("model")
    identity_valid = response_model == subject["route"]
    action, reasoning, invalid_reason = runner.parse_decision(response)
    if not identity_valid:
        action, reasoning, invalid_reason = "INVALID", "", "response_model_identity_mismatch"
    visible_content, finish_reason = runner._visible_content(response)
    visible_hash = (
        hashlib.sha256(visible_content.encode("utf-8")).hexdigest()
        if isinstance(visible_content, str) else None
    )
    expected_response = {
        "response_model": response_model,
        "model_identity_valid": identity_valid,
        "action": action,
        "reasoning": reasoning,
        "invalid_reason": invalid_reason,
        "format_valid": action != "INVALID",
        "visible_content": visible_content,
        "visible_content_sha256": visible_hash,
        "request_id": response.get("id"),
        "finish_reason": finish_reason,
        "usage": response.get("usage"),
        "failure": None,
    }
    if any(row.get(key) != value for key, value in expected_response.items()):
        raise Part2OperationalOverlayValidationError("Response parsing or identity metadata changed.")


def _validate_attempt_bindings(
    records: Sequence[Mapping[str, Any]], *, subject: Mapping[str, Any],
    trajectory_index: int, environment_seed: int,
    global_attempt_ids: set[str], observable_dispatch_attempt_ceiling: int,
) -> dict[tuple[int, int], Mapping[str, Any]]:
    if not _integer(observable_dispatch_attempt_ceiling, minimum=1):
        raise Part2OperationalOverlayValidationError(
            "The observable dispatch-attempt ceiling is invalid."
        )
    try:
        offline._validate_journal_identity(
            records, subject=subject, trajectory_index=trajectory_index,
            environment_seed=environment_seed,
        )
    except (offline.OfflinePart2FinalizationError, KeyError, TypeError, ValueError) as error:
        raise Part2OperationalOverlayValidationError(str(error)) from error

    reservations: dict[str, Mapping[str, Any]] = {}
    terminals: set[str] = set()
    attempts_by_unit: dict[tuple[int, int], int] = {}
    reservations_by_unit: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    semantics: dict[tuple[int, int], Mapping[str, Any]] = {}
    closed_units: set[tuple[int, int]] = set()
    active_attempt_by_unit: dict[tuple[int, int], str] = {}
    retry_ready_units: set[tuple[int, int]] = set()
    last_terminal_at: dict[tuple[int, int], datetime] = {}
    last_day = 0
    for row in records:
        day, slot = row.get("day"), row.get("slot")
        if not _integer(day, minimum=1) or not _integer(slot):
            raise Part2OperationalOverlayValidationError("A journal unit index is invalid.")
        key = (int(day), int(slot))
        if key[0] < last_day:
            raise Part2OperationalOverlayValidationError(
                "Trajectory journal days are not in execution order."
            )
        last_day = key[0]
        event = row.get("event")
        attempt_id = row.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise Part2OperationalOverlayValidationError("A journal attempt id is invalid.")
        if event == "reserved_before_dispatch":
            if (
                attempt_id in reservations
                or attempt_id in global_attempt_ids
                or key in closed_units
                or key in active_attempt_by_unit
            ):
                raise Part2OperationalOverlayValidationError(
                    "An attempt reservation is duplicated, overlaps an active attempt, "
                    "or follows completion."
                )
            number = row.get("attempt_number")
            if not _integer(number, minimum=1) or number != attempts_by_unit.get(key, 0) + 1:
                raise Part2OperationalOverlayValidationError("Attempt numbers are not consecutive per participant unit.")
            if int(number) > 1 and key not in retry_ready_units:
                raise Part2OperationalOverlayValidationError(
                    "A retry reservation does not follow its prior failed attempt."
                )
            dispatch_skipped = row.get("dispatch_skipped")
            if int(number) <= observable_dispatch_attempt_ceiling:
                if "dispatch_skipped" in row:
                    raise Part2OperationalOverlayValidationError(
                        "A within-budget attempt was incorrectly marked dispatch-skipped."
                    )
            elif dispatch_skipped is not True:
                raise Part2OperationalOverlayValidationError(
                    "A journal exceeds its observable dispatch-attempt ceiling."
                )
            reserved_at = _utc_datetime(
                row.get("reserved_at_utc"), label="Attempt reservation timestamp"
            )
            if key in last_terminal_at and reserved_at < last_terminal_at[key]:
                raise Part2OperationalOverlayValidationError(
                    "A retry reservation predates its failed predecessor."
                )
            expected_generation_seed = _derive_seed(
                "inference_hub_part2_generation_v1",
                str(subject["target_id"]), environment_seed, key[0], key[1],
            )
            if (
                row.get("generation_seed") != expected_generation_seed
                or not isinstance(row.get("prompt_sha256"), str)
                or not isinstance(row.get("request_sha256"), str)
                or not _integer(row.get("request_body_bytes"), minimum=1)
                or not isinstance(row.get("controls"), Mapping)
            ):
                raise Part2OperationalOverlayValidationError("An attempt reservation binding is invalid.")
            reservations[attempt_id] = row
            global_attempt_ids.add(attempt_id)
            attempts_by_unit[key] = int(number)
            reservations_by_unit.setdefault(key, []).append(row)
            active_attempt_by_unit[key] = attempt_id
            retry_ready_units.discard(key)
            continue
        if event not in {"attempt_failed", "semantic_result"}:
            raise Part2OperationalOverlayValidationError("A trajectory journal event type is unknown.")
        if (
            attempt_id not in reservations
            or attempt_id in terminals
            or active_attempt_by_unit.get(key) != attempt_id
        ):
            raise Part2OperationalOverlayValidationError("An attempt terminal lacks exactly one reservation.")
        reservation = reservations[attempt_id]
        completed_at = _utc_datetime(
            row.get("completed_at_utc"), label="Attempt terminal timestamp"
        )
        if (
            key != (reservation.get("day"), reservation.get("slot"))
            or row.get("request_sha256") != reservation.get("request_sha256")
            or completed_at < _utc_datetime(
                reservation.get("reserved_at_utc"),
                label="Attempt reservation timestamp",
            )
        ):
            raise Part2OperationalOverlayValidationError("An attempt terminal changed its unit or request binding.")
        terminals.add(attempt_id)
        del active_attempt_by_unit[key]
        last_terminal_at[key] = completed_at
        if event == "attempt_failed":
            if reservation.get("dispatch_skipped") is True:
                failure = _validate_failure(
                    row.get("failure"), label="stale skipped reservation"
                )
                if dict(failure) != {
                    "failure_code": "stale_reserved_attempt",
                    "transient": True,
                    "http_status": None,
                }:
                    raise Part2OperationalOverlayValidationError(
                        "A dispatch-skipped failed event is not an exact stale reservation."
                    )
                retry_ready_units.add(key)
                continue
            failure = _validate_failure(row.get("failure"), label="failed attempt")
            if failure.get("transient") is not True:
                raise Part2OperationalOverlayValidationError("A retried attempt was not transient.")
            retry_ready_units.add(key)
            continue
        if key in semantics:
            raise Part2OperationalOverlayValidationError("A participant unit has multiple semantic results.")
        if row.get("attempt_number") != reservation.get("attempt_number"):
            raise Part2OperationalOverlayValidationError("A semantic result changed its attempt number.")
        _validate_semantic_payload(
            row, reservation=reservation, subject=subject,
            trajectory_index=trajectory_index, environment_seed=environment_seed,
        )
        if reservation.get("dispatch_skipped") is True:
            failure = row.get("failure")
            if (
                int(row.get("attempt_number", 0))
                <= observable_dispatch_attempt_ceiling
                or row.get("invalid_reason") != "transport_failure_exhausted"
                or not isinstance(failure, Mapping)
                or failure.get("failure_code") != "resume_attempt_budget_exhausted"
                or failure.get("transient") is not False
                or failure.get("http_status") is not None
                or failure.get("error_type") is not None
            ):
                raise Part2OperationalOverlayValidationError(
                    "A synthetic exhausted-attempt terminal is inconsistent."
                )
        semantics[key] = row
        closed_units.add(key)

    if set(reservations) != terminals or active_attempt_by_unit:
        raise Part2OperationalOverlayValidationError("A reserved attempt is unterminated.")
    if set(reservations_by_unit) != set(semantics):
        raise Part2OperationalOverlayValidationError("A reserved participant unit lacks one semantic result.")
    for key, unit_reservations in reservations_by_unit.items():
        semantic = semantics[key]
        for reservation in unit_reservations:
            for field_name in (
                "target_id", "trajectory_index", "day", "slot",
                "upstream_provider", "model", "requested_route",
                "environment_seed", "generation_seed", "prompt_sha256",
                "request_sha256", "controls",
            ):
                if reservation.get(field_name) != semantic.get(field_name):
                    raise Part2OperationalOverlayValidationError(
                        "Retries changed a participant unit's frozen request binding."
                    )
            body = semantic["request_body"]
            if reservation.get("request_body_bytes") != len(_canonical_bytes(body)):
                raise Part2OperationalOverlayValidationError("A retry byte count changed.")
    return semantics


def _validate_scheduled_units(
    semantics: Mapping[tuple[int, int], Mapping[str, Any]], *,
    contract: runner.Part2Contract, environment_seed: int,
) -> None:
    living = list(range(contract.society_size))
    reserve = contract.capacity
    expected: set[tuple[int, int]] = set()
    for day in range(1, contract.days + 1):
        if not living:
            continue
        population_start = len(living)
        day_keys = {(day, slot) for slot in living}
        expected.update(day_keys)
        if not day_keys <= set(semantics):
            raise Part2OperationalOverlayValidationError("A trajectory is incomplete before extinction or horizon.")
        actions = [semantics[(day, slot)].get("action") for slot in living]
        if any(action not in {"OPTION_A", "OPTION_B", "INVALID"} for action in actions):
            raise Part2OperationalOverlayValidationError("A trajectory action is invalid.")
        reserve = max(0, reserve - contract.reserve_cost * actions.count("OPTION_B"))
        deaths = _collapse_deaths(population_start, reserve, contract.collapse_death_rate)
        dead, _ = runner._matched_attrition(living, deaths, environment_seed, day)
        dead_set = set(dead)
        living = [slot for slot in living if slot not in dead_set]
    if set(semantics) != expected:
        raise Part2OperationalOverlayValidationError("A trajectory contains unscheduled participant results.")


def _replay_trajectory(
    records: Sequence[Mapping[str, Any]], *, subject: Mapping[str, Any],
    trajectory_index: int, environment_seed: int,
    contract: runner.Part2Contract, execution_contract: Mapping[str, Any],
    global_attempt_ids: set[str],
) -> dict[str, Any]:
    attempt_ceiling = execution_contract.get("max_transport_attempts")
    if not _integer(attempt_ceiling, minimum=1):
        raise Part2OperationalOverlayValidationError(
            "The replay contract lacks an observable transport-attempt ceiling."
        )
    semantics = _validate_attempt_bindings(
        records, subject=subject, trajectory_index=trajectory_index,
        environment_seed=environment_seed, global_attempt_ids=global_attempt_ids,
        observable_dispatch_attempt_ceiling=int(attempt_ceiling),
    )
    _validate_scheduled_units(
        semantics, contract=contract, environment_seed=environment_seed,
    )
    try:
        replayed = offline._offline_replay(
            subject=subject, trajectory_index=trajectory_index,
            environment_seed=environment_seed, contract=contract,
            records=records, execution_contract=execution_contract,
        )
    except (runner.InferenceHubPart2PanelError, KeyError, TypeError, ValueError) as error:
        raise Part2OperationalOverlayValidationError(str(error)) from error
    if replayed is None:
        raise Part2OperationalOverlayValidationError("A trajectory is incomplete under simulator replay.")
    return replayed


def _validate_trajectory_row(row: Mapping[str, Any], *, overlay: bool) -> None:
    _exact_keys(
        row, OVERLAY_TRAJECTORY_ROW_KEYS if overlay else TRAJECTORY_ROW_KEYS,
        "sanitized trajectory row",
    )
    integer_fields = (
        "trajectory_index", "environment_seed_index", "environment_seed",
        "scheduled_agent_days", "responses_received", "invalid_count",
        "identity_mismatch_count", "transport_failure_count", "restraint_count",
        "overuse_count", "final_reserve", "final_population",
        "cumulative_private_payoff", "cumulative_group_payoff",
    )
    if any(not _integer(row.get(key)) for key in integer_fields[:-1]):
        raise Part2OperationalOverlayValidationError("A sanitized trajectory integer field is invalid.")
    if not isinstance(row.get("cumulative_group_payoff"), int) or isinstance(row.get("cumulative_group_payoff"), bool):
        raise Part2OperationalOverlayValidationError("A sanitized group payoff is invalid.")
    if row.get("schema_version") != SCHEMA_VERSION or any(
        not isinstance(row.get(key), str) or not row.get(key)
        for key in ("target_id", "upstream_provider", "model")
    ):
        raise Part2OperationalOverlayValidationError("A sanitized trajectory identity is invalid.")
    if not isinstance(row.get("operationally_eligible"), bool) or not isinstance(row.get("reserve_nondepletion"), bool):
        raise Part2OperationalOverlayValidationError("A sanitized trajectory eligibility field is invalid.")
    if any(not _number(row.get(key)) for key in ("restraint_rate", "aurc", "aupc", "population_retention")):
        raise Part2OperationalOverlayValidationError("A sanitized trajectory metric is invalid.")
    scheduled = int(row["scheduled_agent_days"])
    restraint = int(row["restraint_count"])
    overuse = int(row["overuse_count"])
    invalid = int(row["invalid_count"])
    if scheduled != restraint + overuse + invalid:
        raise Part2OperationalOverlayValidationError("A behavioral denominator does not reconcile.")
    expected_rate = restraint / scheduled if scheduled else 0.0
    if float(row["restraint_rate"]) != expected_rate:
        raise Part2OperationalOverlayValidationError("A trajectory restraint denominator changed.")
    expected_operational = int(row["identity_mismatch_count"]) == int(row["transport_failure_count"]) == 0
    if row["operationally_eligible"] is not expected_operational:
        raise Part2OperationalOverlayValidationError("Operational eligibility does not match failure counts.")
    if overlay:
        round_index = row.get("operational_repair_round")
        replaced = row.get("source_replaced_for_operational_failure")
        if not isinstance(replaced, bool) or (replaced and not _integer(round_index, minimum=1)) or (not replaced and round_index is not None):
            raise Part2OperationalOverlayValidationError("An operational repair marker is invalid.")


def _validate_model_row(row: Mapping[str, Any]) -> None:
    _exact_keys(row, MODEL_ROW_KEYS, "sanitized model row")
    if row.get("schema_version") != SCHEMA_VERSION or any(
        not isinstance(row.get(key), str) or not row.get(key)
        for key in ("target_id", "upstream_provider", "model")
    ):
        raise Part2OperationalOverlayValidationError("A sanitized model identity is invalid.")


def _validate_trajectory_payload_schema(payload: Mapping[str, Any], *, overlay: bool) -> None:
    _exact_keys(payload, OVERLAY_TOP_KEYS if overlay else SOURCE_TRAJECTORY_TOP_KEYS, "trajectory payload")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise Part2OperationalOverlayValidationError("Trajectory payload rows are invalid.")
    for row in rows:
        _validate_trajectory_row(row, overlay=overlay)


def _validate_model_payload_schema(payload: Mapping[str, Any], *, overlay: bool) -> None:
    _exact_keys(payload, OVERLAY_TOP_KEYS if overlay else SOURCE_MODEL_TOP_KEYS, "model payload")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise Part2OperationalOverlayValidationError("Model payload rows are invalid.")
    for row in rows:
        _validate_model_row(row)


def _bound_sanitized_payload(
    manifest_path: Path, manifest: Mapping[str, Any], *, key: str,
    filename: str, artifact_type: str, tracker: _FileTracker,
    is_model: bool, overlay: bool,
) -> dict[str, Any]:
    refs = manifest.get("sanitized_artifacts")
    if not isinstance(refs, Mapping) or set(refs) != (
        {"effective_trajectory_metrics", "effective_model_metrics"}
        if overlay else {"trajectory_metrics", "model_metrics"}
    ):
        raise Part2OperationalOverlayValidationError("Sanitized artifact reference set changed.")
    reference = refs.get(key)
    if not isinstance(reference, Mapping) or set(reference) != {"path", "file_sha256", "evidence_sha256"}:
        raise Part2OperationalOverlayValidationError("A sanitized artifact binding schema changed.")
    expected_path = manifest_path.parents[1] / "sanitized" / filename
    path_value = reference.get("path")
    if not isinstance(path_value, str) or Path(path_value).resolve() != expected_path.resolve():
        raise Part2OperationalOverlayValidationError("A sanitized artifact path binding changed.")
    _require_mode(expected_path, 0o600)
    payload = _safe_json(expected_path, label="sanitized artifact", tracker=tracker)
    if (
        reference.get("file_sha256") != tracker.digests[expected_path.resolve()]
        or payload.get("evidence_sha256") != _self_hash(payload)
        or reference.get("evidence_sha256") != payload.get("evidence_sha256")
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("artifact_type") != artifact_type
        or payload.get("panel_id") != manifest.get("panel_id")
    ):
        raise Part2OperationalOverlayValidationError("A sanitized artifact hash or identity binding failed.")
    if overlay and payload.get("source_manifest_evidence_sha256") != manifest.get("source_manifest", {}).get("evidence_sha256"):
        raise Part2OperationalOverlayValidationError("An effective artifact changed its source binding.")
    if is_model:
        _validate_model_payload_schema(payload, overlay=overlay)
    else:
        _validate_trajectory_payload_schema(payload, overlay=overlay)
    return payload


def _validate_source_input_artifacts(manifest: Mapping[str, Any]) -> None:
    inputs = manifest.get("input_artifacts")
    expected_reference_keys = {
        "panel": {"path", "file_sha256", "canonical_sha256"},
        "compatibility": {"path", "file_sha256", "evidence_sha256"},
        "registry": {"path", "file_sha256", "canonical_sha256"},
    }
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_reference_keys):
        raise Part2OperationalOverlayValidationError(
            "Source input-artifact provenance set changed."
        )
    for key, expected_keys in expected_reference_keys.items():
        reference = inputs.get(key)
        if (
            not isinstance(reference, Mapping)
            or set(reference) != expected_keys
            or not isinstance(reference.get("path"), str)
            or any(
                not isinstance(reference.get(hash_key), str)
                or len(str(reference[hash_key])) != 64
                for hash_key in expected_keys - {"path"}
            )
        ):
            raise Part2OperationalOverlayValidationError(
                f"Source {key} input binding schema changed."
            )
    panel_ref = inputs["panel"]
    if (
        Path(str(panel_ref["path"])).resolve() != EXPECTED_PANEL_PATH
        or panel_ref.get("file_sha256") != EXPECTED_PANEL_FILE_SHA256
        or panel_ref.get("canonical_sha256") != EXPECTED_PANEL_CANONICAL_SHA256
    ):
        raise Part2OperationalOverlayValidationError(
            "Source panel binding changed from the frozen 100-day artifact."
        )


def _validate_execution_contract(execution: object) -> Mapping[str, Any]:
    if not isinstance(execution, Mapping):
        raise Part2OperationalOverlayValidationError("Source execution contract is absent.")
    if set(execution) != set(EXPECTED_SOURCE_EXECUTION_CONTRACT):
        raise Part2OperationalOverlayValidationError(
            "Source execution contract schema changed."
        )
    rate = execution.get("shared_rate_limit")
    if not isinstance(rate, Mapping) or dict(rate) != EXPECTED_SHARED_RATE_LIMIT_CONTRACT:
        raise Part2OperationalOverlayValidationError(
            "Source shared rate-limit policy changed."
        )
    if rate.get("policy_sha256") != _sha256_json(
        {key: value for key, value in rate.items() if key != "policy_sha256"}
    ):
        raise Part2OperationalOverlayValidationError(
            "Source shared rate-limit policy hash changed."
        )
    if (
        dict(execution) != EXPECTED_SOURCE_EXECUTION_CONTRACT
        or _sha256_json(execution)
        != _sha256_json(EXPECTED_SOURCE_EXECUTION_CONTRACT)
    ):
        raise Part2OperationalOverlayValidationError(
            "Source frozen original-scale execution design changed."
        )
    return execution


def _validate_frozen_source_design(
    manifest: Mapping[str, Any], *, contract: runner.Part2Contract,
    environment_seeds: Sequence[int],
) -> Mapping[str, Any]:
    expected_contract = runner.Part2Contract(
        society_size=50,
        days=100,
        trajectories=EXPECTED_TRAJECTORIES_PER_ROUTE,
        capacity=2500,
        private_gain=2,
        reserve_cost=2,
        community_benefit=5,
        collapse_death_rate=0.2,
    )
    if contract != expected_contract:
        raise Part2OperationalOverlayValidationError(
            "Source is not the frozen 50-agent/100-day/12-seed contract."
        )
    if (
        not _integer(manifest.get("base_seed"))
        or manifest.get("base_seed") != EXPECTED_BASE_SEED
    ):
        raise Part2OperationalOverlayValidationError(
            "Source base seed changed from the frozen original-scale design."
        )
    if (
        tuple(environment_seeds) != EXPECTED_COMMON_ENVIRONMENT_SEEDS
        or len(set(environment_seeds)) != EXPECTED_TRAJECTORIES_PER_ROUTE
        or any(not _integer(seed) for seed in environment_seeds)
    ):
        raise Part2OperationalOverlayValidationError(
            "Source common-environment seed vector changed from the frozen panel."
        )
    derived = tuple(
        runner._environment_seeds(
            EXPECTED_PANEL_ID,
            EXPECTED_BASE_SEED,
            EXPECTED_TRAJECTORIES_PER_ROUTE,
        )
    )
    if derived != EXPECTED_COMMON_ENVIRONMENT_SEEDS:
        raise Part2OperationalOverlayValidationError(
            "Loaded runner no longer derives the frozen common-environment seeds."
        )
    return _validate_execution_contract(manifest.get("execution_contract"))


def _source_summary(rows: Sequence[Mapping[str, Any]], *, route_count: int, contract: runner.Part2Contract) -> dict[str, Any]:
    return {
        "planned_trajectories": route_count * contract.trajectories,
        "completed_trajectories": len(rows),
        "planned_maximum_agent_days": route_count * contract.trajectories * contract.days * contract.society_size,
        "scheduled_agent_days": sum(int(row["scheduled_agent_days"]) for row in rows),
        "responses_received": sum(int(row["responses_received"]) for row in rows),
        "invalid_count": sum(int(row["invalid_count"]) for row in rows),
        "identity_mismatch_count": sum(int(row["identity_mismatch_count"]) for row in rows),
        "transport_failure_count": sum(int(row["transport_failure_count"]) for row in rows),
        "eligible_trajectories": sum(row["operationally_eligible"] is True for row in rows),
    }


def _validate_source(
    manifest_path: Path, manifest: Mapping[str, Any], *, tracker: _FileTracker,
    global_attempt_ids: set[str], source_verification_root: Path | None,
) -> _SourceEvidence:
    if not SOURCE_MANIFEST_REQUIRED_KEYS <= set(manifest) or not set(manifest) <= SOURCE_MANIFEST_REQUIRED_KEYS | SOURCE_MANIFEST_OPTIONAL_KEYS:
        raise Part2OperationalOverlayValidationError("Source manifest schema changed.")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("artifact_type") != offline.ARTIFACT_TYPE
        or manifest.get("evidence_sha256") != _self_hash(manifest)
        or manifest.get("panel_id") != EXPECTED_PANEL_ID
    ):
        raise Part2OperationalOverlayValidationError("Source manifest identity or self-hash failed.")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or manifest.get("runtime_sha256") != _sha256_json(runtime):
        raise Part2OperationalOverlayValidationError("Source runtime binding changed.")
    _validate_source_input_artifacts(manifest)
    try:
        offline._validate_source_bindings(
            manifest, source_verification_root=source_verification_root,
        )
        subjects, contract, seeds = offline._validate_frozen_inputs(manifest)
    except (offline.OfflinePart2FinalizationError, runner.InferenceHubPart2PanelError, KeyError, TypeError, ValueError) as error:
        raise Part2OperationalOverlayValidationError(str(error)) from error
    execution = _validate_frozen_source_design(
        manifest, contract=contract, environment_seeds=seeds,
    )
    for reference in manifest["input_artifacts"].values():
        if isinstance(reference, Mapping) and isinstance(reference.get("path"), str):
            tracker.add(Path(reference["path"]).resolve())
    repository_root = Path(__file__).resolve().parents[1]
    verification_root = (
        repository_root
        if source_verification_root is None
        else source_verification_root.resolve()
    )
    runner_path = Path(runner.__file__).resolve()
    if manifest["source_artifacts"].get(str(runner_path)) != EXPECTED_RUNNER_IMPLEMENTATION_SHA256:
        raise Part2OperationalOverlayValidationError(
            "Source runner implementation hash changed from the frozen campaign."
        )
    for path_value, digest in manifest["source_artifacts"].items():
        bound_path = Path(path_value).resolve()
        try:
            relative = bound_path.relative_to(repository_root)
        except ValueError as error:
            raise Part2OperationalOverlayValidationError(
                "A manifest-bound source lies outside the repository root."
            ) from error
        tracker.add(verification_root / relative, str(digest))

    subject_by_id = {str(subject["target_id"]): subject for subject in subjects}
    refs = manifest.get("journals")
    expected_keys = {
        f"{target_id}::{index}"
        for target_id in subject_by_id for index in range(contract.trajectories)
    }
    if not isinstance(refs, Mapping) or set(refs) != expected_keys:
        raise Part2OperationalOverlayValidationError("Source trajectory journal set is incomplete or duplicated.")
    replayed_rows: list[dict[str, Any]] = []
    for target_id, subject in subject_by_id.items():
        journal_dir = manifest_path.parent / "trajectories" / _safe_file_stem(target_id)
        for index in range(contract.trajectories):
            key = f"{target_id}::{index}"
            records = _read_journal_reference(
                refs[key], expected_path=journal_dir / f"seed-{index:03d}.jsonl",
                label="source trajectory journal", tracker=tracker,
            )
            replayed_rows.append(_replay_trajectory(
                records, subject=subject, trajectory_index=index,
                environment_seed=seeds[index], contract=contract,
                execution_contract=execution, global_attempt_ids=global_attempt_ids,
            ))
    replayed_rows.sort(key=lambda row: (str(row["target_id"]), int(row["trajectory_index"])))
    trajectory_payload = _bound_sanitized_payload(
        manifest_path, manifest, key="trajectory_metrics", filename="trajectory_metrics.json",
        artifact_type=offline.TRAJECTORY_ARTIFACT_TYPE, tracker=tracker,
        is_model=False, overlay=False,
    )
    sanitized_rows = sorted(
        (dict(row) for row in trajectory_payload["rows"]),
        key=lambda row: (str(row["target_id"]), int(row["trajectory_index"])),
    )
    if replayed_rows != sanitized_rows:
        raise Part2OperationalOverlayValidationError("Source trajectory metrics differ from simulator replay.")
    expected_models = runner._aggregate_models(
        replayed_rows, subjects, expected_trajectories=contract.trajectories,
        capacity=contract.capacity,
    )
    model_payload = _bound_sanitized_payload(
        manifest_path, manifest, key="model_metrics", filename="model_metrics.json",
        artifact_type=offline.MODEL_ARTIFACT_TYPE, tracker=tracker,
        is_model=True, overlay=False,
    )
    if model_payload["rows"] != expected_models:
        raise Part2OperationalOverlayValidationError("Source model aggregates do not reconcile.")
    expected_summary = _source_summary(replayed_rows, route_count=len(subjects), contract=contract)
    if manifest.get("summary") != expected_summary:
        raise Part2OperationalOverlayValidationError("Source manifest summary does not reconcile.")
    expected_complete = (
        expected_summary["identity_mismatch_count"] == 0
        and expected_summary["transport_failure_count"] == 0
    )
    if manifest.get("complete") is not expected_complete:
        raise Part2OperationalOverlayValidationError("Source completion flag does not match operational evidence.")
    return _SourceEvidence(
        manifest_path=manifest_path,
        manifest_file_sha256=tracker.digests[manifest_path], manifest=manifest,
        subjects=tuple(subjects), contract=contract,
        environment_seeds=tuple(seeds), trajectories=tuple(replayed_rows),
        models=tuple(expected_models),
    )


def _validate_credential_pool(
    value: object, tracker: _FileTracker, *, source_manifest: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "account_count", "selection_policy", "rate_limit_scope",
        "rate_limit_contract", "implementation_files",
    }:
        raise Part2OperationalOverlayValidationError("Credential-pool provenance schema changed.")
    if (
        not _integer(value.get("account_count"), minimum=1)
        or value.get("account_count") != 3
        or value.get("selection_policy") != "thread_safe_round_robin"
        or value.get("rate_limit_scope") != "independent_per_account"
        or not isinstance(value.get("rate_limit_contract"), Mapping)
    ):
        raise Part2OperationalOverlayValidationError("Credential-pool provenance changed.")
    rate_contract = value["rate_limit_contract"]
    if (
        dict(rate_contract) != EXPECTED_SHARED_RATE_LIMIT_CONTRACT
        or rate_contract.get("policy_sha256")
        != _sha256_json(
            {key: item for key, item in rate_contract.items() if key != "policy_sha256"}
        )
    ):
        raise Part2OperationalOverlayValidationError(
            "Credential-pool rate-limit contract changed."
        )
    files = value.get("implementation_files")
    repair_path = Path(repair.__file__).resolve()
    runner_path = Path(runner.__file__).resolve()
    expected_files = {
        str(repair_path): EXPECTED_REPAIR_IMPLEMENTATION_SHA256,
        str(runner_path): EXPECTED_RUNNER_IMPLEMENTATION_SHA256,
    }
    if not isinstance(files, Mapping) or dict(files) != expected_files:
        raise Part2OperationalOverlayValidationError(
            "Operational repair implementation bindings changed."
        )
    source_artifacts = source_manifest.get("source_artifacts")
    if (
        not isinstance(source_artifacts, Mapping)
        or source_artifacts.get(str(runner_path))
        != EXPECTED_RUNNER_IMPLEMENTATION_SHA256
    ):
        raise Part2OperationalOverlayValidationError(
            "Credential-pool runner does not match the source campaign runner."
        )
    for path_value, digest in files.items():
        path = Path(str(path_value)).resolve()
        if str(digest) != _sha256_file(path):
            raise Part2OperationalOverlayValidationError("Operational repair implementation bytes changed.")
        tracker.add(path, str(digest))


def _validate_partial_parent_overlay(
    manifest_path: Path, manifest: Mapping[str, Any], *, source: _SourceEvidence,
    tracker: _FileTracker, global_attempt_ids: set[str],
) -> _ValidatedPartialOverlay:
    """Replay the exact incomplete v3 parent without modifying its journals."""

    source_binding = manifest.get("source_manifest")
    if (
        not isinstance(source_binding, Mapping)
        or set(source_binding) != {"path", "file_sha256", "evidence_sha256"}
    ):
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent source binding schema changed."
        )
    if (
        not isinstance(source_binding.get("path"), str)
        or Path(str(source_binding["path"])).resolve() != source.manifest_path
        or source_binding.get("file_sha256") != source.manifest_file_sha256
        or source_binding.get("evidence_sha256") != source.manifest.get("evidence_sha256")
        or manifest.get("panel_id") != source.manifest.get("panel_id")
        or manifest.get("part2_contract") != source.manifest.get("part2_contract")
        or manifest.get("common_environment_seeds")
        != source.manifest.get("common_environment_seeds")
        or manifest.get("subject_routes") != source.manifest.get("subject_routes")
        or manifest.get("repair_policy")
        != "whole_trajectory_day_one_exact_route_separate_overlay"
        or manifest.get("maximum_rounds") != EXPECTED_REPAIR_MAXIMUM_ROUNDS
    ):
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent changed its source or repair binding."
        )
    _validate_credential_pool(
        manifest.get("credential_pool"), tracker, source_manifest=source.manifest,
    )
    source_rows = {
        (str(row["target_id"]), int(row["trajectory_index"])): dict(row)
        for row in source.trajectories
    }
    failures = {
        key: row for key, row in source_rows.items()
        if row["operationally_eligible"] is False
    }
    if len(failures) != EXPECTED_MAIN_PARENT_SOURCE_FAILURES or any(
        int(row["identity_mismatch_count"])
        == int(row["transport_failure_count"])
        == 0
        for row in failures.values()
    ):
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent did not select the exact original operational failures."
        )
    refs = manifest.get("journals")
    expected_keys = {
        f"{target_id}::{index}::{round_index}"
        for target_id, index in failures
        for round_index in range(1, EXPECTED_REPAIR_MAXIMUM_ROUNDS + 1)
    }
    if not isinstance(refs, Mapping) or set(refs) != expected_keys:
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent journal schedule changed."
        )
    subjects = {str(row["target_id"]): row for row in source.subjects}
    execution = _validate_execution_contract(source.manifest.get("execution_contract"))
    successful: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    nonempty_round_counts: dict[tuple[str, int], int] = {}
    executed_rounds = 0
    for key in sorted(failures):
        target_id, index = key
        seen_empty = False
        for round_index in range(1, EXPECTED_REPAIR_MAXIMUM_ROUNDS + 1):
            ref_key = f"{target_id}::{index}::{round_index}"
            expected_path = (
                manifest_path.parent / "trajectories" / _safe_file_stem(target_id)
                / f"seed-{index:03d}-round-{round_index:02d}.jsonl"
            )
            records = _read_journal_reference(
                refs[ref_key], expected_path=expected_path,
                label="incomplete parent trajectory journal", tracker=tracker,
            )
            if not records:
                seen_empty = True
                continue
            if seen_empty or key in successful:
                raise Part2OperationalOverlayValidationError(
                    "Incomplete parent rounds are nonconsecutive or continue after success."
                )
            replayed = _replay_trajectory(
                records, subject=subjects[target_id], trajectory_index=index,
                environment_seed=source.environment_seeds[index],
                contract=source.contract, execution_contract=execution,
                global_attempt_ids=global_attempt_ids,
            )
            executed_rounds += 1
            nonempty_round_counts[key] = nonempty_round_counts.get(key, 0) + 1
            if replayed["operationally_eligible"] is True:
                successful[key] = (round_index, replayed)
    unresolved_keys = set(failures) - set(successful)
    expected_unresolved = {
        (EXPECTED_CASCADING_TARGET_ID, EXPECTED_CASCADING_TRAJECTORY_INDEX)
    }
    if (
        len(successful) != EXPECTED_MAIN_PARENT_SUCCESSES
        or unresolved_keys != expected_unresolved
        or nonempty_round_counts.get(next(iter(expected_unresolved)))
        != EXPECTED_REPAIR_MAXIMUM_ROUNDS
    ):
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent is not the exact 37-success/one-Gemini-unresolved overlay."
        )
    unresolved = {key: failures[key] for key in unresolved_keys}
    unresolved_row = unresolved[next(iter(expected_unresolved))]
    if (
        unresolved_row.get("environment_seed_index")
        != EXPECTED_CASCADING_TRAJECTORY_INDEX
        or unresolved_row.get("environment_seed")
        != EXPECTED_CASCADING_ENVIRONMENT_SEED
        or int(unresolved_row.get("transport_failure_count", 0)) < 1
    ):
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent's exact unresolved trajectory changed."
        )

    effective_rows: list[dict[str, Any]] = []
    for key, source_row in sorted(source_rows.items()):
        if key in successful:
            round_index, replayed = successful[key]
            row = dict(replayed)
            row["operational_repair_round"] = round_index
            row["source_replaced_for_operational_failure"] = True
        else:
            row = dict(source_row)
            row["operational_repair_round"] = None
            row["source_replaced_for_operational_failure"] = False
        effective_rows.append(row)
    trajectory_payload = _bound_sanitized_payload(
        manifest_path, manifest, key="effective_trajectory_metrics",
        filename="effective_trajectory_metrics.json",
        artifact_type=(
            "inference_hub_part2_operational_repair_effective_trajectory_metrics_v1"
        ),
        tracker=tracker, is_model=False, overlay=True,
    )
    if trajectory_payload["rows"] != effective_rows:
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent trajectory metrics do not match recursive replay."
        )
    effective_models = runner._aggregate_models(
        effective_rows, source.subjects,
        expected_trajectories=source.contract.trajectories,
        capacity=source.contract.capacity,
    )
    model_payload = _bound_sanitized_payload(
        manifest_path, manifest, key="effective_model_metrics",
        filename="effective_model_metrics.json",
        artifact_type=(
            "inference_hub_part2_operational_repair_effective_model_metrics_v1"
        ),
        tracker=tracker, is_model=True, overlay=True,
    )
    if model_payload["rows"] != effective_models:
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent model aggregates do not reconcile."
        )
    expected_summary = {
        "source_operational_failure_trajectories": len(failures),
        "operational_repairs_succeeded": len(successful),
        "operational_repairs_unresolved": len(unresolved),
    }
    if manifest.get("summary") != expected_summary:
        raise Part2OperationalOverlayValidationError(
            "Incomplete parent summary does not reconcile."
        )
    return _ValidatedPartialOverlay(
        source=source,
        parent_manifest_path=manifest_path,
        parent_manifest_file_sha256=tracker.digests[manifest_path],
        parent_manifest=manifest,
        effective_trajectories=tuple(effective_rows),
        effective_models=tuple(effective_models),
        source_failures=failures,
        successful=successful,
        unresolved=unresolved,
        repair_round_count=executed_rounds,
    )


def _validate_cascading_cursor_epoch(value: object) -> str:
    suffix = value.removeprefix("cursor-epoch-") if isinstance(value, str) else ""
    if (
        not isinstance(value, str)
        or not value.startswith("cursor-epoch-")
        or len(suffix) != 32
        or any(character not in "0123456789abcdef" for character in suffix)
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading cursor epoch is not the exact generated format."
        )
    return value


def _validate_cascading_preflight_ledger(
    manifest_path: Path, reference: object, *, exact_route: str,
    tracker: _FileTracker, resume_count: int,
) -> tuple[Mapping[str, Any], tuple[str, ...], str]:
    if not isinstance(reference, Mapping) or set(reference) != {
        "path", "file_sha256", "evidence_sha256"
    }:
        raise Part2OperationalOverlayValidationError(
            "Cascading preflight ledger reference schema changed."
        )
    expected_path = (
        manifest_path.parent / "preflight"
        / f"preflight-{resume_count:03d}.json"
    )
    if (
        not isinstance(reference.get("path"), str)
        or Path(str(reference["path"])).resolve() != expected_path.resolve()
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading preflight ledger path binding changed."
        )
    ledger = _safe_json_0600(
        expected_path, label="cascading preflight ledger", tracker=tracker,
    )
    if (
        reference.get("file_sha256") != tracker.digests[expected_path.resolve()]
        or reference.get("evidence_sha256") != ledger.get("evidence_sha256")
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading preflight ledger hashes changed."
        )
    try:
        cursor_epoch, qualified_slots = cascading._validate_preflight_ledger(
            ledger, exact_route=exact_route,
        )
    except cascading.Part2CascadingRepairError as error:
        raise Part2OperationalOverlayValidationError(str(error)) from error
    cursor_epoch = _validate_cascading_cursor_epoch(cursor_epoch)
    created = _utc_datetime(
        ledger.get("created_at_utc"), label="Preflight ledger creation timestamp"
    )
    completed = _utc_datetime(
        ledger.get("completed_at_utc"), label="Preflight ledger completion timestamp"
    )
    initial_completed = _utc_datetime(
        ledger.get("initial_completed_at_utc"),
        label="Initial preflight ledger completion timestamp",
    )
    if not created <= initial_completed <= completed:
        raise Part2OperationalOverlayValidationError(
            "Preflight ledger timestamps are out of order."
        )
    if ledger.get("base_url") != runner.DEFAULT_BASE_URL:
        raise Part2OperationalOverlayValidationError(
            "Preflight ledger endpoint differs from the frozen InferenceHub endpoint."
        )
    results = ledger["results"]
    previous_probe_end = created
    for result in results:
        started = _utc_datetime(
            result.get("started_at_utc"), label="Account preflight start timestamp"
        )
        ended = _utc_datetime(
            result.get("completed_at_utc"), label="Account preflight completion timestamp"
        )
        if not previous_probe_end <= started <= ended <= initial_completed:
            raise Part2OperationalOverlayValidationError(
                "Account preflight timestamps are outside the sealed ledger interval."
            )
        previous_probe_end = ended
        authenticated = result["authentication_succeeded"]
        catalog_status = result["catalog_http_status"]
        catalog_failure = result["catalog_failure_code"]
        listed = result["exact_route_catalog_listed"]
        chat_attempted = result["chat_attempted"]
        chat_status = result["chat_http_status"]
        chat_failure = result["chat_failure_code"]
        identity = result["chat_response_identity_matched"]
        if not all(isinstance(value, bool) for value in (authenticated, listed, chat_attempted)):
            raise Part2OperationalOverlayValidationError(
                "Account preflight boolean classifications are invalid."
            )
        _validate_cascading_preflight_control_flow(
            authenticated, listed, chat_attempted, label="account preflight"
        )
        if authenticated:
            if catalog_status != 200 or catalog_failure is not None:
                raise Part2OperationalOverlayValidationError(
                    "Authenticated catalog preflight lacks its exact success status."
                )
        elif (
            catalog_status == 200
            or catalog_status is not None
            and (
                not _integer(catalog_status, minimum=100)
                or int(catalog_status) > 599
            )
            or not isinstance(catalog_failure, str)
            or not catalog_failure
            or listed
            or chat_attempted
        ):
            raise Part2OperationalOverlayValidationError(
                "Failed authentication/catalog preflight is inconsistent."
            )
        if chat_attempted:
            if not authenticated or not listed:
                raise Part2OperationalOverlayValidationError(
                    "Chat preflight was attempted without catalog qualification."
                )
            if chat_status == 200:
                if chat_failure is not None or not isinstance(identity, bool):
                    raise Part2OperationalOverlayValidationError(
                        "Successful chat preflight classification is inconsistent."
                    )
            elif (
                chat_status == 200
                or chat_status is not None
                and (
                    not _integer(chat_status, minimum=100)
                    or int(chat_status) > 599
                )
                or not isinstance(chat_failure, str)
                or not chat_failure
                or identity is not False
            ):
                raise Part2OperationalOverlayValidationError(
                    "Failed chat preflight classification is inconsistent."
                )
        elif chat_status is not None or chat_failure is not None or identity is not None:
            raise Part2OperationalOverlayValidationError(
                "Unattempted chat preflight contains fabricated results."
            )
        if not authenticated:
            _validate_cascading_preflight_failure_pair(
                catalog_status, catalog_failure, label="catalog preflight"
            )
        if chat_attempted and chat_status != 200:
            _validate_cascading_preflight_failure_pair(
                chat_status, chat_failure, label="chat preflight"
            )
    previous_completed = initial_completed
    for session in ledger["resume_preflight_sessions"]:
        session_created = _utc_datetime(
            session.get("created_at_utc"), label="Resume preflight session start"
        )
        session_completed = _utc_datetime(
            session.get("completed_at_utc"), label="Resume preflight session completion"
        )
        if not previous_completed <= session_created <= session_completed:
            raise Part2OperationalOverlayValidationError(
                "Resume preflight session timestamps are out of order."
            )
        previous_probe_end = session_created
        for result in session["results"]:
            started = _utc_datetime(
                result.get("started_at_utc"), label="Resume account preflight start"
            )
            ended = _utc_datetime(
                result.get("completed_at_utc"),
                label="Resume account preflight completion",
            )
            if not previous_probe_end <= started <= ended <= session_completed:
                raise Part2OperationalOverlayValidationError(
                    "Resume account probe timestamps are outside its session."
                )
            _validate_cascading_preflight_control_flow(
                result.get("authentication_succeeded"),
                result.get("exact_route_catalog_listed"),
                result.get("chat_attempted"),
                label="resume account preflight",
            )
            if result.get("authentication_succeeded") is False:
                _validate_cascading_preflight_failure_pair(
                    result.get("catalog_http_status"),
                    result.get("catalog_failure_code"),
                    label="resume catalog preflight",
                )
            if (
                result.get("chat_attempted") is True
                and result.get("chat_http_status") != 200
            ):
                _validate_cascading_preflight_failure_pair(
                    result.get("chat_http_status"),
                    result.get("chat_failure_code"),
                    label="resume chat preflight",
                )
            previous_probe_end = ended
        previous_completed = session_completed
    return ledger, qualified_slots, cursor_epoch


def _validate_cascading_preflight_failure_pair(
    status: object, failure_code: object, *, label: str,
) -> None:
    exact_pair = (
        failure_code == "http_error"
        and _integer(status, minimum=100)
        and int(status) <= 599
        and status != 200
    ) or (
        isinstance(failure_code, str)
        and bool(failure_code)
        and failure_code != "http_error"
        and status is None
    )
    if not exact_pair:
        raise Part2OperationalOverlayValidationError(
            f"{label.capitalize()} failure code/status pairing changed."
        )


def _validate_cascading_preflight_control_flow(
    authenticated: object, catalog_listed: object, chat_attempted: object, *,
    label: str,
) -> None:
    if (
        not all(
            isinstance(value, bool)
            for value in (authenticated, catalog_listed, chat_attempted)
        )
        or chat_attempted is not (authenticated and catalog_listed)
    ):
        raise Part2OperationalOverlayValidationError(
            f"{label.capitalize()} chat-attempt control flow changed."
        )


def _validate_cascading_resume_session_times(
    ledger: Mapping[str, Any], *, child_created: datetime,
) -> None:
    if any(
        _utc_datetime(
            session.get("created_at_utc"),
            label="Resume preflight session start",
        ) < child_created
        for session in ledger["resume_preflight_sessions"]
    ):
        raise Part2OperationalOverlayValidationError(
            "A cascading resume preflight predates child campaign creation."
        )


def _validate_cascading_credential_pool(
    value: object, *, qualified_slots: Sequence[str], cursor_epoch: str,
    exact_route: str, account_commitments: Mapping[str, str],
    source_manifest: Mapping[str, Any], tracker: _FileTracker,
) -> None:
    expected_keys = {
        "configured_account_count", "configured_account_slots",
        "account_slot_commitments",
        "qualified_account_count", "qualified_account_slots",
        "rejected_account_slots", "qualified_set_sha256", "selection_policy",
        "rate_limit_scope", "rate_limit_contract",
        "qualified_account_rate_limiters", "exact_route", "base_url",
        "cursor_epoch",
        "cursor_initial_global_dispatch_ordinal", "implementation_files",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise Part2OperationalOverlayValidationError(
            "Cascading credential-pool provenance schema changed."
        )
    if (
        cascading.EXPECTED_CONFIGURED_ACCOUNT_COUNT
        != EXPECTED_CASCADING_CONFIGURED_ACCOUNT_COUNT
        or cascading.SELECTION_POLICY != EXPECTED_CASCADING_SELECTION_POLICY
        or cascading.RATE_LIMIT_SCOPE != EXPECTED_CASCADING_RATE_LIMIT_SCOPE
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading credential-pool implementation constants changed."
        )
    configured = list(
        cascading._account_slots(EXPECTED_CASCADING_CONFIGURED_ACCOUNT_COUNT)
    )
    qualified = list(qualified_slots)
    rejected = [slot for slot in configured if slot not in qualified]
    commitments = value.get("account_slot_commitments")
    if (
        value.get("configured_account_count") != len(configured)
        or value.get("configured_account_slots") != configured
        or not isinstance(commitments, Mapping)
        or set(commitments) != set(configured)
        or dict(commitments) != dict(account_commitments)
        or any(
            not isinstance(commitment, str)
            or len(commitment) != 64
            or any(character not in "0123456789abcdef" for character in commitment)
            for commitment in commitments.values()
        )
        or len(set(commitments.values())) != len(configured)
        or value.get("qualified_account_count") != len(qualified)
        or value.get("qualified_account_slots") != qualified
        or value.get("rejected_account_slots") != rejected
        or value.get("qualified_set_sha256")
        != cascading._qualified_set_sha256(qualified, commitments)
        or value.get("selection_policy") != EXPECTED_CASCADING_SELECTION_POLICY
        or value.get("rate_limit_scope") != EXPECTED_CASCADING_RATE_LIMIT_SCOPE
        or value.get("exact_route") != exact_route
        or value.get("base_url") != "https://inference-api.nvidia.com/v1"
        or value.get("cursor_epoch") != cursor_epoch
        or value.get("cursor_initial_global_dispatch_ordinal") != 0
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading credential-pool provenance changed."
        )
    rate_contract = value.get("rate_limit_contract")
    if not isinstance(rate_contract, Mapping) or dict(rate_contract) != EXPECTED_SHARED_RATE_LIMIT_CONTRACT:
        raise Part2OperationalOverlayValidationError(
            "Cascading qualified-account rate-limit contract changed."
        )
    per_account_limiters = value.get("qualified_account_rate_limiters")
    if (
        not isinstance(per_account_limiters, Mapping)
        or list(per_account_limiters) != qualified
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading per-account limiter scopes changed."
        )
    state_scope_hashes: list[str] = []
    for slot in qualified:
        limiter = per_account_limiters[slot]
        if not isinstance(limiter, Mapping) or set(limiter) != {
            "scope_label", "state_path_sha256", "rate_limit_contract"
        }:
            raise Part2OperationalOverlayValidationError(
                "Cascading per-account limiter schema changed."
            )
        state_hash = limiter.get("state_path_sha256")
        if (
            limiter.get("scope_label") != f"{cursor_epoch}::{slot}"
            or not isinstance(state_hash, str)
            or len(state_hash) != 64
            or any(character not in "0123456789abcdef" for character in state_hash)
            or limiter.get("rate_limit_contract") != EXPECTED_SHARED_RATE_LIMIT_CONTRACT
        ):
            raise Part2OperationalOverlayValidationError(
                "Cascading per-account limiter binding changed."
            )
        state_scope_hashes.append(state_hash)
    if len(set(state_scope_hashes)) != len(state_scope_hashes):
        raise Part2OperationalOverlayValidationError(
            "Cascading qualified accounts do not have independent limiter scopes."
        )
    cascading_path = Path(cascading.__file__).resolve()
    runner_path = Path(runner.__file__).resolve()
    repair_path = Path(repair.__file__).resolve()
    discovery_path = cascading_path.with_name("inference_hub_discovery.py")
    rate_path = cascading_path.with_name("inference_hub_rate_limit.py")
    expected_files = {
        str(cascading_path): EXPECTED_CASCADING_REPAIR_IMPLEMENTATION_SHA256,
        str(runner_path): EXPECTED_RUNNER_IMPLEMENTATION_SHA256,
        str(repair_path): EXPECTED_REPAIR_IMPLEMENTATION_SHA256,
        str(discovery_path): EXPECTED_DISCOVERY_IMPLEMENTATION_SHA256,
        str(rate_path): EXPECTED_RATE_LIMIT_IMPLEMENTATION_SHA256,
    }
    files = value.get("implementation_files")
    if not isinstance(files, Mapping) or dict(files) != expected_files:
        raise Part2OperationalOverlayValidationError(
            "Cascading repair implementation bindings changed."
        )
    source_artifacts = source_manifest.get("source_artifacts")
    if (
        not isinstance(source_artifacts, Mapping)
        or source_artifacts.get(str(runner_path)) != EXPECTED_RUNNER_IMPLEMENTATION_SHA256
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading runner does not match the frozen source campaign runner."
        )
    for path_value, digest in expected_files.items():
        path = Path(path_value)
        if _sha256_file(path) != digest:
            raise Part2OperationalOverlayValidationError(
                "Cascading repair implementation bytes changed."
            )
        tracker.add(path, digest)


_CASCADING_RESERVATION_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "event", "attempt_id", "target_id",
        "trajectory_index", "day", "slot", "attempt_number",
        "upstream_provider", "model", "requested_route", "environment_seed",
        "generation_seed", "prompt_sha256", "request_sha256",
        "request_body_bytes", "controls", "reserved_at_utc", "account_slot",
        "global_dispatch_ordinal", "cursor_epoch", "previous_record_sha256",
        "preflight_session_index", "preflight_session_sha256", "record_sha256",
    }
)
_CASCADING_ATTEMPT_FAILED_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "event", "attempt_id", "target_id",
        "trajectory_index", "day", "slot", "request_sha256", "failure",
        "completed_at_utc", "previous_record_sha256", "record_sha256",
    }
)
_CASCADING_SEMANTIC_KEYS = frozenset(
    {
        "schema_version", "artifact_type", "event", "attempt_id", "target_id",
        "trajectory_index", "day", "slot", "attempt_number",
        "upstream_provider", "model", "requested_route", "response_model",
        "model_identity_valid", "environment_seed", "generation_seed",
        "prompt_text", "prompt_sha256", "request_body", "request_sha256",
        "controls", "action", "reasoning", "invalid_reason", "format_valid",
        "visible_content", "visible_content_sha256", "request_id",
        "finish_reason", "usage", "raw_response", "raw_response_sha256",
        "failure", "completed_at_utc", "previous_record_sha256", "record_sha256",
    }
)


def _cascading_retryable_failure(failure: Mapping[str, Any]) -> bool:
    status = failure.get("http_status")
    code = failure.get("failure_code")
    if code in {"connection_error", "connection_timeout"}:
        return status is None
    if code != "http_error" or isinstance(status, bool) or not isinstance(status, int):
        return False
    return status in {400, 429} or 500 <= status <= 599


def _cascading_preflight_authorizations(
    ledger: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    initial_hash = ledger.get("initial_session_sha256")
    if (
        not isinstance(initial_hash, str)
        or len(initial_hash) != 64
        or any(character not in "0123456789abcdef" for character in initial_hash)
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading initial preflight-session hash is invalid."
        )
    authorizations = [{
        "session_index": 0,
        "session_sha256": initial_hash,
        "created_at_utc": ledger["created_at_utc"],
        "completed_at_utc": ledger["initial_completed_at_utc"],
        "starting_global_dispatch_ordinal": 0,
    }]
    previous_boundary = 0
    for expected_index, session in enumerate(
        ledger["resume_preflight_sessions"], 1,
    ):
        authorization = {
            "session_index": session["session_index"],
            "session_sha256": session["session_sha256"],
            "created_at_utc": session["created_at_utc"],
            "completed_at_utc": session["completed_at_utc"],
            "starting_global_dispatch_ordinal": session[
                "starting_global_dispatch_ordinal"
            ],
        }
        session_hash = authorization["session_sha256"]
        boundary = authorization["starting_global_dispatch_ordinal"]
        if (
            authorization["session_index"] != expected_index
            or not isinstance(session_hash, str)
            or len(session_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in session_hash
            )
            or not _integer(boundary)
            or int(boundary) < previous_boundary
        ):
            raise Part2OperationalOverlayValidationError(
                "Cascading preflight session authorization is invalid."
            )
        previous_boundary = int(boundary)
        authorizations.append(authorization)
    return tuple(authorizations)


def _validate_cascading_dispatch_boundaries(
    ledger: Mapping[str, Any], *, final_global_ordinal: int,
) -> None:
    if not _integer(final_global_ordinal) or any(
        int(session["starting_global_dispatch_ordinal"]) > final_global_ordinal
        for session in _cascading_preflight_authorizations(ledger)
    ):
        raise Part2OperationalOverlayValidationError(
            "A cascading preflight session claims a future dispatch boundary."
        )


def _validate_cascading_manifest_dispatch_times(
    records: Sequence[Mapping[str, Any]], *, child_created: datetime,
    latest_session_index: int, latest_resume_at: datetime | None,
) -> None:
    for row in records:
        if row.get("event") != "reserved_before_dispatch":
            continue
        reserved_at = _utc_datetime(
            row.get("reserved_at_utc"), label="Cascading reservation timestamp"
        )
        if reserved_at < child_created:
            raise Part2OperationalOverlayValidationError(
                "A cascading dispatch predates child manifest creation."
            )
        if (
            latest_session_index > 0
            and row.get("preflight_session_index") == latest_session_index
            and (
                latest_resume_at is None or reserved_at < latest_resume_at
            )
        ):
            raise Part2OperationalOverlayValidationError(
                "A cascading dispatch predates its resume manifest checkpoint."
            )


def _later_cascading_recovery_session_index(
    authorizations: Sequence[Mapping[str, Any]], *,
    reservation_session_index: int, terminal_at: datetime,
) -> int | None:
    for recovery_index in range(
        reservation_session_index + 1, len(authorizations)
    ):
        if terminal_at < _utc_datetime(
            authorizations[recovery_index]["completed_at_utc"],
            label="Crash-recovery preflight completion",
        ):
            continue
        if recovery_index + 1 < len(authorizations) and terminal_at > _utc_datetime(
            authorizations[recovery_index + 1]["created_at_utc"],
            label="Next crash-recovery preflight start",
        ):
            continue
        return recovery_index
    return None


def _validate_cascading_dispatch_provenance(
    records: Sequence[Mapping[str, Any]], *, qualified_slots: Sequence[str],
    cursor_epoch: str, next_global_ordinal: int, max_attempts: int,
    preflight_ledger: Mapping[str, Any], latest_resume_at: datetime | None = None,
) -> int:
    authorizations = _cascading_preflight_authorizations(preflight_ledger)
    authorization_by_index = {
        int(item["session_index"]): item for item in authorizations
    }
    reservations: dict[str, Mapping[str, Any]] = {}
    last_terminal_at: dict[tuple[int, int], datetime] = {}
    for row in records:
        event = row.get("event")
        if event == "reserved_before_dispatch":
            expected_keys = _CASCADING_RESERVATION_KEYS | (
                {"dispatch_skipped"} if row.get("dispatch_skipped") is True else set()
            )
            if set(row) != expected_keys:
                raise Part2OperationalOverlayValidationError(
                    "Cascading dispatch reservation schema changed."
                )
            session_index = row.get("preflight_session_index")
            session = authorization_by_index.get(session_index)
            if (
                not _integer(session_index)
                or session is None
                or row.get("preflight_session_sha256")
                != session["session_sha256"]
            ):
                raise Part2OperationalOverlayValidationError(
                    "Cascading reservation lacks its exact preflight session."
                )
            reserved_at = _utc_datetime(
                row.get("reserved_at_utc"), label="Cascading reservation timestamp"
            )
            session_completed = _utc_datetime(
                session["completed_at_utc"],
                label="Dispatch-authorizing preflight completion",
            )
            if reserved_at < session_completed:
                raise Part2OperationalOverlayValidationError(
                    "Cascading reservation predates its authorizing preflight."
                )
            unit = (int(row["day"]), int(row["slot"]))
            if unit in last_terminal_at and reserved_at < last_terminal_at[unit]:
                raise Part2OperationalOverlayValidationError(
                    "Cascading retry reservation predates its failed predecessor."
                )
            if int(session_index) + 1 < len(authorizations):
                next_session_created = _utc_datetime(
                    authorizations[int(session_index) + 1]["created_at_utc"],
                    label="Next cascading preflight start",
                )
                if reserved_at > next_session_created:
                    raise Part2OperationalOverlayValidationError(
                        "Cascading reservation crosses its preflight session boundary."
                    )
            attempt_number = row.get("attempt_number")
            dispatch_skipped = row.get("dispatch_skipped") is True
            if (
                not _integer(attempt_number, minimum=1)
                or dispatch_skipped
                and int(attempt_number) != max_attempts + 1
                or not dispatch_skipped
                and int(attempt_number) > max_attempts
            ):
                raise Part2OperationalOverlayValidationError(
                    "Cascading reservation exceeds its exact dispatch-attempt ceiling."
                )
            if dispatch_skipped:
                if (
                    row.get("global_dispatch_ordinal") is not None
                    or row.get("account_slot") is not None
                    or row.get("cursor_epoch") != cursor_epoch
                ):
                    raise Part2OperationalOverlayValidationError(
                        "Cascading skipped reservation fabricated dispatch provenance."
                    )
            else:
                next_global_ordinal += 1
                expected_slot = qualified_slots[
                    (next_global_ordinal - 1) % len(qualified_slots)
                ]
                if (
                    row.get("global_dispatch_ordinal") != next_global_ordinal
                    or row.get("account_slot") != expected_slot
                    or row.get("cursor_epoch") != cursor_epoch
                ):
                    raise Part2OperationalOverlayValidationError(
                        "Cascading round-robin account dispatch provenance changed."
                    )
                expected_session = authorizations[0]
                for candidate in authorizations[1:]:
                    if (
                        int(candidate["starting_global_dispatch_ordinal"])
                        < next_global_ordinal
                    ):
                        expected_session = candidate
                if session != expected_session:
                    raise Part2OperationalOverlayValidationError(
                        "Cascading dispatch is assigned to the wrong preflight session."
                    )
            attempt_id = str(row.get("attempt_id"))
            if attempt_id in reservations:
                raise Part2OperationalOverlayValidationError(
                    "Cascading reservation attempt id is duplicated."
                )
            reservations[attempt_id] = row
            continue
        if event == "attempt_failed":
            if set(row) != _CASCADING_ATTEMPT_FAILED_KEYS:
                raise Part2OperationalOverlayValidationError(
                    "Cascading failed-attempt schema changed."
                )
            reservation = reservations.get(str(row.get("attempt_id")))
            failure = row.get("failure")
            if not isinstance(reservation, Mapping) or not isinstance(failure, Mapping):
                raise Part2OperationalOverlayValidationError(
                    "Cascading failed attempt lacks its account-bound reservation."
                )
            completed_at = _utc_datetime(
                row.get("completed_at_utc"), label="Cascading failure timestamp"
            )
            if completed_at < _utc_datetime(
                reservation.get("reserved_at_utc"),
                label="Cascading reservation timestamp",
            ):
                raise Part2OperationalOverlayValidationError(
                    "Cascading attempt completed before its reservation."
                )
            failure_keys = set(failure)
            stale = failure.get("failure_code") == "stale_reserved_attempt"
            if failure_keys != ({"failure_code", "transient", "http_status"} if stale else {
                "failure_code", "transient", "http_status", "error_type"
            }):
                raise Part2OperationalOverlayValidationError(
                    "Cascading failure metadata schema changed."
                )
            if stale:
                if reservation.get("dispatch_skipped") is True or failure != {
                    "failure_code": "stale_reserved_attempt",
                    "transient": True,
                    "http_status": None,
                }:
                    raise Part2OperationalOverlayValidationError(
                        "Cascading stale reservation classification changed."
                    )
                reservation_session_index = int(
                    reservation["preflight_session_index"]
                )
                recovery_index = _later_cascading_recovery_session_index(
                    authorizations,
                    reservation_session_index=reservation_session_index,
                    terminal_at=completed_at,
                )
                if recovery_index is None:
                    raise Part2OperationalOverlayValidationError(
                        "Cascading stale reservation lacks a later recovery preflight."
                    )
                if (
                    recovery_index == len(authorizations) - 1
                    and (
                        latest_resume_at is None
                        or completed_at < latest_resume_at
                    )
                ):
                    raise Part2OperationalOverlayValidationError(
                        "Cascading stale recovery predates its manifest checkpoint."
                    )
            elif (
                failure.get("transient") is not True
                or not _cascading_retryable_failure(failure)
                or not isinstance(failure.get("error_type"), str)
                or not failure.get("error_type")
                or int(reservation.get("attempt_number", 0)) >= max_attempts
            ):
                raise Part2OperationalOverlayValidationError(
                    "Cascading retry does not follow the exact bounded retry policy."
                )
            if not stale:
                reservation_session_index = int(
                    reservation["preflight_session_index"]
                )
                if (
                    reservation_session_index + 1 < len(authorizations)
                    and completed_at > _utc_datetime(
                        authorizations[reservation_session_index + 1][
                            "created_at_utc"
                        ],
                        label="Next preflight session start",
                    )
                ):
                    raise Part2OperationalOverlayValidationError(
                        "Cascading failed attempt crosses a resume boundary."
                    )
        elif event == "semantic_result":
            if set(row) != _CASCADING_SEMANTIC_KEYS:
                raise Part2OperationalOverlayValidationError(
                    "Cascading semantic-result schema changed."
                )
            reservation = reservations.get(str(row.get("attempt_id")))
            failure = row.get("failure")
            if not isinstance(reservation, Mapping):
                raise Part2OperationalOverlayValidationError(
                    "Cascading semantic result lacks its account-bound reservation."
                )
            completed_at = _utc_datetime(
                row.get("completed_at_utc"), label="Cascading result timestamp"
            )
            if completed_at < _utc_datetime(
                reservation.get("reserved_at_utc"),
                label="Cascading reservation timestamp",
            ):
                raise Part2OperationalOverlayValidationError(
                    "Cascading result completed before its reservation."
                )
            reservation_session_index = int(
                reservation["preflight_session_index"]
            )
            crosses_resume_boundary = (
                reservation_session_index + 1 < len(authorizations)
                and completed_at > _utc_datetime(
                    authorizations[reservation_session_index + 1][
                        "created_at_utc"
                    ],
                    label="Next preflight session start",
                )
            )
            if crosses_resume_boundary:
                recovery_index = (
                    _later_cascading_recovery_session_index(
                        authorizations,
                        reservation_session_index=reservation_session_index,
                        terminal_at=completed_at,
                    )
                )
                synthetic_recovery = (
                    reservation.get("dispatch_skipped") is True
                    and row.get("raw_response") is None
                    and isinstance(failure, Mapping)
                    and failure.get("failure_code")
                    == "resume_attempt_budget_exhausted"
                    and recovery_index is not None
                )
                if not synthetic_recovery:
                    raise Part2OperationalOverlayValidationError(
                        "Cascading semantic result crosses a resume boundary."
                    )
                if (
                    recovery_index == len(authorizations) - 1
                    and (
                        latest_resume_at is None
                        or completed_at < latest_resume_at
                    )
                ):
                    raise Part2OperationalOverlayValidationError(
                        "Cascading synthetic recovery predates its manifest checkpoint."
                    )
            if row.get("raw_response") is None:
                if not isinstance(failure, Mapping) or set(failure) != {
                    "failure_code", "transient", "http_status", "error_type"
                }:
                    raise Part2OperationalOverlayValidationError(
                        "Cascading terminal failure metadata schema changed."
                    )
                retryable = _cascading_retryable_failure(failure)
                synthetic = failure.get("failure_code") == "resume_attempt_budget_exhausted"
                failure_code = failure.get("failure_code")
                http_status = failure.get("http_status")
                exact_status_pair = (
                    failure_code == "http_error"
                    and _integer(http_status, minimum=100)
                    and int(http_status) <= 599
                ) or (
                    failure_code != "http_error"
                    and isinstance(failure_code, str)
                    and bool(failure_code)
                    and http_status is None
                )
                if (
                    not exact_status_pair
                    or not synthetic
                    and (
                        not isinstance(failure.get("error_type"), str)
                        or not failure.get("error_type")
                    )
                    or
                    synthetic
                    and failure != {
                        "failure_code": "resume_attempt_budget_exhausted",
                        "transient": False,
                        "http_status": None,
                        "error_type": None,
                    }
                    or not synthetic and failure.get("transient") is not retryable
                ):
                    raise Part2OperationalOverlayValidationError(
                        "Cascading terminal failure classification changed."
                    )
                if (
                    failure.get("failure_code") == "http_error"
                    and failure.get("http_status") in {401, 403}
                ):
                    raise Part2OperationalOverlayValidationError(
                        "Cascading dispatch invalidated a preflight-qualified account."
                    )
            if (
                row.get("raw_response") is None
                and isinstance(failure, Mapping)
                and reservation.get("dispatch_skipped") is not True
                and _cascading_retryable_failure(failure)
                and int(reservation.get("attempt_number", 0)) < max_attempts
            ):
                raise Part2OperationalOverlayValidationError(
                    "Cascading retryable transport failure exhausted before attempt ceiling."
                )
        else:
            raise Part2OperationalOverlayValidationError(
                "Cascading journal event type changed."
            )
        _utc_datetime(
            row.get("completed_at_utc"), label="Cascading terminal timestamp"
        )
        reservation = reservations.get(str(row.get("attempt_id")))
        if reservation is None:
            raise Part2OperationalOverlayValidationError(
                "Cascading terminal lacks its provenance-bearing reservation."
            )
        last_terminal_at[
            (int(reservation["day"]), int(reservation["slot"]))
        ] = _utc_datetime(
            row.get("completed_at_utc"), label="Cascading terminal timestamp"
        )
    return next_global_ordinal


def _validate_cascading_execution_contract(value: object) -> Mapping[str, Any]:
    implementation_defaults = (
        cascading.DEFAULT_MAXIMUM_ROUNDS,
        cascading.DEFAULT_TRAJECTORY_WORKERS,
        cascading.DEFAULT_PARTICIPANT_WORKERS,
        cascading.DEFAULT_MAX_ATTEMPTS,
        cascading.DEFAULT_BACKOFF_SECONDS,
        cascading.DEFAULT_TIMEOUT_SECONDS,
        cascading.DEFAULT_RATE_PROFILE,
    )
    frozen_defaults = (
        EXPECTED_CASCADING_MAXIMUM_ROUNDS,
        EXPECTED_CASCADING_TRAJECTORY_WORKERS,
        EXPECTED_CASCADING_PARTICIPANT_WORKERS,
        EXPECTED_CASCADING_MAX_ATTEMPTS,
        EXPECTED_CASCADING_INITIAL_BACKOFF_SECONDS,
        EXPECTED_CASCADING_TIMEOUT_SECONDS,
        EXPECTED_CASCADING_RATE_PROFILE,
    )
    if (
        implementation_defaults != frozen_defaults
        or not isinstance(value, Mapping)
        or dict(value) != EXPECTED_CASCADING_EXECUTION_CONTRACT
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading execution contract differs from the frozen exact design."
        )
    return value


def _validate_cascading_overlay(
    manifest_path: Path, manifest: Mapping[str, Any], *,
    parent: _ValidatedPartialOverlay, tracker: _FileTracker,
    global_attempt_ids: set[str],
) -> _ValidatedPair:
    source = parent.source
    source_binding = manifest.get("source_manifest")
    parent_binding = manifest.get("parent_overlay_manifest")
    if (
        not isinstance(source_binding, Mapping)
        or set(source_binding) != {"path", "file_sha256", "evidence_sha256"}
        or not isinstance(parent_binding, Mapping)
        or set(parent_binding) != {"path", "file_sha256", "evidence_sha256"}
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading source/parent binding schema changed."
        )
    expected_key = (
        EXPECTED_CASCADING_TARGET_ID, EXPECTED_CASCADING_TRAJECTORY_INDEX,
    )
    if (
        Path(str(source_binding.get("path", ""))).resolve() != source.manifest_path
        or source_binding.get("file_sha256") != source.manifest_file_sha256
        or source_binding.get("evidence_sha256") != source.manifest["evidence_sha256"]
        or Path(str(parent_binding.get("path", ""))).resolve()
        != parent.parent_manifest_path
        or parent_binding.get("file_sha256") != parent.parent_manifest_file_sha256
        or parent_binding.get("evidence_sha256")
        != parent.parent_manifest["evidence_sha256"]
        or manifest.get("panel_id") != source.manifest.get("panel_id")
        or manifest.get("base_seed") != EXPECTED_BASE_SEED
        or manifest.get("part2_contract") != source.manifest.get("part2_contract")
        or manifest.get("common_environment_seeds")
        != source.manifest.get("common_environment_seeds")
        or manifest.get("subject_routes") != source.manifest.get("subject_routes")
        or cascading.REPAIR_POLICY != EXPECTED_CASCADING_REPAIR_POLICY
        or manifest.get("repair_policy") != EXPECTED_CASCADING_REPAIR_POLICY
        or manifest.get("maximum_rounds") != EXPECTED_CASCADING_MAXIMUM_ROUNDS
        or manifest.get("composition_contract")
        != EXPECTED_CASCADING_COMPOSITION_CONTRACT
    ):
        raise Part2OperationalOverlayValidationError(
            "Cascading overlay changed its frozen lineage or composition binding."
        )
    if set(parent.unresolved) != {expected_key}:
        raise Part2OperationalOverlayValidationError(
            "Cascading parent no longer has the exact selected unresolved trajectory."
        )
    subjects = {str(row["target_id"]): row for row in source.subjects}
    subject = subjects[EXPECTED_CASCADING_TARGET_ID]
    expected_selected = {
        "target_id": EXPECTED_CASCADING_TARGET_ID,
        "trajectory_index": EXPECTED_CASCADING_TRAJECTORY_INDEX,
        "environment_seed_index": EXPECTED_CASCADING_TRAJECTORY_INDEX,
        "environment_seed": EXPECTED_CASCADING_ENVIRONMENT_SEED,
        "requested_route": subject["route"],
    }
    if manifest.get("selected_trajectory") != expected_selected:
        raise Part2OperationalOverlayValidationError(
            "Cascading selected-trajectory binding changed."
        )
    execution = _validate_cascading_execution_contract(
        manifest.get("execution_contract")
    )
    resume_count = manifest.get("resume_count", 0)
    if not _integer(resume_count):
        raise Part2OperationalOverlayValidationError(
            "Cascading resume count is invalid."
        )
    ledger, qualified_slots, cursor_epoch = _validate_cascading_preflight_ledger(
        manifest_path, manifest.get("preflight_ledger"),
        exact_route=str(subject["route"]), tracker=tracker,
        resume_count=int(resume_count),
    )
    initial_preflight_completed = _utc_datetime(
        ledger["initial_completed_at_utc"],
        label="Initial preflight completion timestamp",
    )
    child_created = _utc_datetime(
        manifest["created_at_utc"], label="Cascading creation timestamp"
    )
    if initial_preflight_completed > child_created:
        raise Part2OperationalOverlayValidationError(
            "Cascading preflight was not sealed before campaign creation."
        )
    _validate_cascading_resume_session_times(
        ledger, child_created=child_created,
    )
    ledger_completed = _utc_datetime(
        ledger["completed_at_utc"], label="Latest preflight completion timestamp"
    )
    child_updated = _utc_datetime(
        manifest["last_updated_at_utc"], label="Cascading update timestamp"
    )
    if len(ledger["resume_preflight_sessions"]) != int(resume_count):
        raise Part2OperationalOverlayValidationError(
            "Cascading resume count does not match sealed preflight sessions."
        )
    if int(resume_count) == 0:
        last_resumed: datetime | None = None
        if "last_resumed_at_utc" in manifest:
            raise Part2OperationalOverlayValidationError(
                "Fresh cascading evidence contains fabricated resume metadata."
            )
    else:
        last_resumed = _utc_datetime(
            manifest.get("last_resumed_at_utc"),
            label="Cascading last-resumed timestamp",
        )
        if not ledger_completed <= last_resumed <= child_updated:
            raise Part2OperationalOverlayValidationError(
                "Cascading resume/preflight timestamps are out of order."
            )
    _validate_cascading_credential_pool(
        manifest.get("credential_pool"), qualified_slots=qualified_slots,
        cursor_epoch=cursor_epoch, exact_route=str(subject["route"]),
        account_commitments={
            str(result["account_slot"]): str(result["account_commitment"])
            for result in ledger["results"]
        },
        source_manifest=source.manifest, tracker=tracker,
    )

    refs = manifest.get("journals")
    expected_ref_keys = {
        f"{EXPECTED_CASCADING_TARGET_ID}::{EXPECTED_CASCADING_TRAJECTORY_INDEX}::{round_index}"
        for round_index in range(1, EXPECTED_CASCADING_MAXIMUM_ROUNDS + 1)
    }
    if not isinstance(refs, Mapping) or set(refs) != expected_ref_keys:
        raise Part2OperationalOverlayValidationError(
            "Cascading child journal schedule changed."
        )
    successful: tuple[int, dict[str, Any]] | None = None
    seen_empty = False
    executed_rounds = 0
    global_ordinal = 0
    for round_index in range(1, EXPECTED_CASCADING_MAXIMUM_ROUNDS + 1):
        ref_key = (
            f"{EXPECTED_CASCADING_TARGET_ID}::"
            f"{EXPECTED_CASCADING_TRAJECTORY_INDEX}::{round_index}"
        )
        expected_path = (
            manifest_path.parent / "trajectories"
            / _safe_file_stem(EXPECTED_CASCADING_TARGET_ID)
            / f"seed-{EXPECTED_CASCADING_TRAJECTORY_INDEX:03d}-round-{round_index:02d}.jsonl"
        )
        records = _read_journal_reference(
            refs[ref_key], expected_path=expected_path,
            label="cascading child trajectory journal", tracker=tracker,
        )
        if not records:
            seen_empty = True
            continue
        if seen_empty or successful is not None:
            raise Part2OperationalOverlayValidationError(
                "Cascading rounds are nonconsecutive or continue after success."
            )
        _validate_cascading_manifest_dispatch_times(
            records, child_created=child_created,
            latest_session_index=int(resume_count),
            latest_resume_at=last_resumed,
        )
        global_ordinal = _validate_cascading_dispatch_provenance(
            records, qualified_slots=qualified_slots, cursor_epoch=cursor_epoch,
            next_global_ordinal=global_ordinal,
            max_attempts=EXPECTED_CASCADING_MAX_ATTEMPTS,
            preflight_ledger=ledger, latest_resume_at=last_resumed,
        )
        replayed = _replay_trajectory(
            records, subject=subject,
            trajectory_index=EXPECTED_CASCADING_TRAJECTORY_INDEX,
            environment_seed=EXPECTED_CASCADING_ENVIRONMENT_SEED,
            contract=source.contract, execution_contract=execution,
            global_attempt_ids=global_attempt_ids,
        )
        executed_rounds += 1
        if replayed["operationally_eligible"] is True:
            successful = (round_index, replayed)
    _validate_cascading_dispatch_boundaries(
        ledger, final_global_ordinal=global_ordinal,
    )
    if successful is None:
        raise Part2OperationalOverlayValidationError(
            "A complete cascading overlay has an unresolved child trajectory."
        )

    effective_rows: list[dict[str, Any]] = []
    for parent_row in parent.effective_trajectories:
        key = (str(parent_row["target_id"]), int(parent_row["trajectory_index"]))
        if key == expected_key:
            replacement = dict(successful[1])
            replacement["operational_repair_round"] = successful[0]
            replacement["source_replaced_for_operational_failure"] = True
            effective_rows.append(replacement)
        else:
            effective_rows.append(dict(parent_row))
    trajectory_payload = _bound_sanitized_payload(
        manifest_path, manifest, key="effective_trajectory_metrics",
        filename="effective_trajectory_metrics.json",
        artifact_type=cascading.TRAJECTORY_ARTIFACT_TYPE,
        tracker=tracker, is_model=False, overlay=True,
    )
    if trajectory_payload["rows"] != effective_rows:
        raise Part2OperationalOverlayValidationError(
            "Cascading effective trajectories changed inherited or replayed rows."
        )
    try:
        cascading._assert_no_secret_like_fields(
            trajectory_payload, label="cascading effective trajectories"
        )
    except cascading.Part2CascadingRepairError as error:
        raise Part2OperationalOverlayValidationError(str(error)) from error
    effective_models = runner._aggregate_models(
        effective_rows, source.subjects,
        expected_trajectories=source.contract.trajectories,
        capacity=source.contract.capacity,
    )
    model_payload = _bound_sanitized_payload(
        manifest_path, manifest, key="effective_model_metrics",
        filename="effective_model_metrics.json",
        artifact_type=cascading.MODEL_ARTIFACT_TYPE,
        tracker=tracker, is_model=True, overlay=True,
    )
    if model_payload["rows"] != effective_models:
        raise Part2OperationalOverlayValidationError(
            "Cascading effective model aggregates do not reconcile."
        )
    expected_summary = {
        "original_source_operational_failure_trajectories": 38,
        "parent_repairs_succeeded": 37,
        "parent_repairs_unresolved": 1,
        "cascading_repairs_succeeded": 1,
        "cascading_repairs_unresolved": 0,
    }
    if manifest.get("summary") != expected_summary:
        raise Part2OperationalOverlayValidationError(
            "Cascading cumulative summary does not reconcile."
        )
    return _ValidatedPair(
        source_manifest_path=source.manifest_path,
        overlay_manifest_path=manifest_path,
        source_manifest=source.manifest,
        overlay_manifest=manifest,
        subjects=source.subjects,
        contract=source.contract,
        environment_seeds=source.environment_seeds,
        effective_trajectories=tuple(effective_rows),
        effective_models=tuple(effective_models),
        source_operational_failure_count=len(parent.source_failures),
        repair_round_count=parent.repair_round_count + executed_rounds,
    )


def _validate_overlay(
    manifest_path: Path, manifest: Mapping[str, Any], *, source: _SourceEvidence,
    tracker: _FileTracker, global_attempt_ids: set[str],
) -> _ValidatedPair:
    source_binding = manifest.get("source_manifest")
    if not isinstance(source_binding, Mapping) or set(source_binding) != {"path", "file_sha256", "evidence_sha256"}:
        raise Part2OperationalOverlayValidationError("Operational overlay source binding schema changed.")
    if (
        not isinstance(source_binding.get("path"), str)
        or Path(str(source_binding["path"])).resolve() != source.manifest_path
        or source_binding.get("file_sha256") != source.manifest_file_sha256
        or source_binding.get("evidence_sha256") != source.manifest.get("evidence_sha256")
        or manifest.get("panel_id") != source.manifest.get("panel_id")
        or manifest.get("part2_contract") != source.manifest.get("part2_contract")
        or manifest.get("common_environment_seeds") != source.manifest.get("common_environment_seeds")
        or manifest.get("subject_routes") != source.manifest.get("subject_routes")
        or manifest.get("repair_policy") != "whole_trajectory_day_one_exact_route_separate_overlay"
    ):
        raise Part2OperationalOverlayValidationError("Operational overlay changed its source or repair binding.")
    maximum_rounds = manifest.get("maximum_rounds")
    if (
        not _integer(maximum_rounds, minimum=1)
        or maximum_rounds != EXPECTED_REPAIR_MAXIMUM_ROUNDS
    ):
        raise Part2OperationalOverlayValidationError(
            "Operational overlay maximum-round binding changed."
        )
    has_credential_pool = "credential_pool" in manifest
    if len(source.subjects) == 21 and not has_credential_pool:
        raise Part2OperationalOverlayValidationError(
            "The main 21-route overlay lacks its bound credential-pool provenance."
        )
    if has_credential_pool:
        _validate_credential_pool(
            manifest["credential_pool"], tracker, source_manifest=source.manifest,
        )

    source_rows = {
        (str(row["target_id"]), int(row["trajectory_index"])): dict(row)
        for row in source.trajectories
    }
    if len(source_rows) != len(source.trajectories):
        raise Part2OperationalOverlayValidationError("Source trajectory identities are duplicated.")
    failures = {
        key: row for key, row in source_rows.items()
        if row["operationally_eligible"] is False
    }
    if not failures or any(
        int(row["identity_mismatch_count"]) == int(row["transport_failure_count"]) == 0
        for row in failures.values()
    ):
        raise Part2OperationalOverlayValidationError("Operational repair selection includes semantic-only evidence.")
    refs = manifest.get("journals")
    expected_keys = {
        f"{target_id}::{index}::{round_index}"
        for target_id, index in failures
        for round_index in range(1, int(maximum_rounds) + 1)
    }
    if not isinstance(refs, Mapping) or set(refs) != expected_keys:
        raise Part2OperationalOverlayValidationError("Operational repair journal schedule changed.")
    subjects = {str(row["target_id"]): row for row in source.subjects}
    execution = _validate_execution_contract(source.manifest.get("execution_contract"))
    successful: dict[tuple[str, int], tuple[int, dict[str, Any]]] = {}
    executed_rounds = 0
    for key in sorted(failures):
        target_id, index = key
        seen_empty = False
        for round_index in range(1, int(maximum_rounds) + 1):
            ref_key = f"{target_id}::{index}::{round_index}"
            expected_path = (
                manifest_path.parent / "trajectories" / _safe_file_stem(target_id)
                / f"seed-{index:03d}-round-{round_index:02d}.jsonl"
            )
            records = _read_journal_reference(
                refs[ref_key], expected_path=expected_path,
                label="operational repair trajectory journal", tracker=tracker,
            )
            if not records:
                seen_empty = True
                continue
            if seen_empty or key in successful:
                raise Part2OperationalOverlayValidationError("Operational repair rounds are nonconsecutive or continue after success.")
            # Repair worker counts and backoff were local CLI scheduling settings,
            # not overlay-manifest bindings.  Exact prompts, seeds, requests, and
            # semantic results are replayed below; the source's frozen retry
            # ceiling is used only to enforce what the journal observably did.
            replayed = _replay_trajectory(
                records, subject=subjects[target_id], trajectory_index=index,
                environment_seed=source.environment_seeds[index],
                contract=source.contract, execution_contract=execution,
                global_attempt_ids=global_attempt_ids,
            )
            executed_rounds += 1
            if replayed["operationally_eligible"] is True:
                successful[key] = (round_index, replayed)
    if set(successful) != set(failures):
        raise Part2OperationalOverlayValidationError("A complete operational overlay has unresolved source failures.")

    effective_rows: list[dict[str, Any]] = []
    for key, source_row in sorted(source_rows.items()):
        if key in successful:
            round_index, replayed = successful[key]
            row = dict(replayed)
            row["operational_repair_round"] = round_index
            row["source_replaced_for_operational_failure"] = True
        else:
            row = dict(source_row)
            row["operational_repair_round"] = None
            row["source_replaced_for_operational_failure"] = False
        effective_rows.append(row)
    trajectory_payload = _bound_sanitized_payload(
        manifest_path, manifest, key="effective_trajectory_metrics",
        filename="effective_trajectory_metrics.json",
        artifact_type="inference_hub_part2_operational_repair_effective_trajectory_metrics_v1",
        tracker=tracker, is_model=False, overlay=True,
    )
    if trajectory_payload["rows"] != effective_rows:
        raise Part2OperationalOverlayValidationError("Effective trajectory metrics do not match source plus repair replay.")
    effective_models = runner._aggregate_models(
        effective_rows, source.subjects,
        expected_trajectories=source.contract.trajectories,
        capacity=source.contract.capacity,
    )
    model_payload = _bound_sanitized_payload(
        manifest_path, manifest, key="effective_model_metrics",
        filename="effective_model_metrics.json",
        artifact_type="inference_hub_part2_operational_repair_effective_model_metrics_v1",
        tracker=tracker, is_model=True, overlay=True,
    )
    if model_payload["rows"] != effective_models:
        raise Part2OperationalOverlayValidationError("Effective model aggregates do not reconcile.")
    expected_summary = {
        "source_operational_failure_trajectories": len(failures),
        "operational_repairs_succeeded": len(successful),
        "operational_repairs_unresolved": 0,
    }
    if manifest.get("summary") != expected_summary:
        raise Part2OperationalOverlayValidationError("Operational overlay summary does not reconcile.")
    return _ValidatedPair(
        source_manifest_path=source.manifest_path,
        overlay_manifest_path=manifest_path,
        source_manifest=source.manifest,
        overlay_manifest=manifest,
        subjects=source.subjects,
        contract=source.contract,
        environment_seeds=source.environment_seeds,
        effective_trajectories=tuple(effective_rows),
        effective_models=tuple(effective_models),
        source_operational_failure_count=len(failures),
        repair_round_count=executed_rounds,
    )


def _validate_union(pairs: Sequence[_ValidatedPair]) -> dict[str, Any]:
    if len(pairs) != 3:
        raise Part2OperationalOverlayValidationError("Exactly three source/overlay pairs are required.")
    expected_contract = runner.Part2Contract(50, 100, 12, 2500, 2, 2, 5, 0.2)
    if any(pair.contract != expected_contract for pair in pairs):
        raise Part2OperationalOverlayValidationError("The source shards do not share the frozen 100-day contract.")
    seeds = pairs[0].environment_seeds
    if (
        seeds != EXPECTED_COMMON_ENVIRONMENT_SEEDS
        or any(pair.environment_seeds != EXPECTED_COMMON_ENVIRONMENT_SEEDS for pair in pairs)
        or any(
            not _integer(pair.source_manifest.get("base_seed"))
            or pair.source_manifest.get("base_seed") != EXPECTED_BASE_SEED
            for pair in pairs
        )
    ):
        raise Part2OperationalOverlayValidationError(
            "The source shards do not share the exact frozen base seed and common-seed vector."
        )
    subject_sets = [frozenset(str(row["target_id"]) for row in pair.subjects) for pair in pairs]
    singleton_ids = {next(iter(values)) for values in subject_sets if len(values) == 1}
    if sorted(len(values) for values in subject_sets) != [1, 1, 21] or singleton_ids != set(EXPECTED_SINGLETON_SHARDS):
        raise Part2OperationalOverlayValidationError("The definitive 21+1+1 source shard identity split changed.")
    all_subjects = [dict(subject) for pair in pairs for subject in pair.subjects]
    target_ids = [str(row["target_id"]) for row in all_subjects]
    routes = [str(row["route"]).strip().casefold() for row in all_subjects]
    provider_models = [
        (str(row["upstream_provider"]).strip().casefold(), str(row["model"]).strip().casefold())
        for row in all_subjects
    ]
    if (
        len(target_ids) != EXPECTED_ROUTE_COUNT
        or len(set(target_ids)) != EXPECTED_ROUTE_COUNT
        or len(set(routes)) != EXPECTED_ROUTE_COUNT
        or len(set(provider_models)) != EXPECTED_ROUTE_COUNT
    ):
        raise Part2OperationalOverlayValidationError("Definitive route identities overlap or are incomplete.")
    panels = []
    judges = []
    for pair in pairs:
        inputs = pair.source_manifest.get("input_artifacts", {})
        panel_ref = inputs.get("panel", {}) if isinstance(inputs, Mapping) else {}
        panels.append(
            (
                panel_ref.get("path"),
                panel_ref.get("file_sha256"),
                panel_ref.get("canonical_sha256"),
            )
        )
        judges.append(pair.source_manifest.get("judge_reservation"))
    expected_panel_binding = (
        str(EXPECTED_PANEL_PATH),
        EXPECTED_PANEL_FILE_SHA256,
        EXPECTED_PANEL_CANONICAL_SHA256,
    )
    if (
        any(panel != expected_panel_binding for panel in panels)
        or any(judge != judges[0] for judge in judges)
    ):
        raise Part2OperationalOverlayValidationError("Frozen panel or judge identity differs across source shards.")
    try:
        panel, _ = runner._load_panel(EXPECTED_PANEL_PATH)
    except (runner.InferenceHubPart2PanelError, OSError, ValueError, KeyError, TypeError) as error:
        raise Part2OperationalOverlayValidationError(str(error)) from error
    if (
        _sha256_file(EXPECTED_PANEL_PATH) != EXPECTED_PANEL_FILE_SHA256
        or _sha256_json(panel) != EXPECTED_PANEL_CANONICAL_SHA256
        or not isinstance(judges[0], Mapping)
        or judges[0].get("target_id") != panel.get("judge_target_id")
    ):
        raise Part2OperationalOverlayValidationError(
            "The repository panel or reserved judge changed from the frozen design."
        )
    expected_targets = set(panel["subject_target_ids"]) - set(EXPECTED_EXCLUDED_TARGET_IDS)
    if set(target_ids) != expected_targets or set(panel["subject_target_ids"]) - set(target_ids) != set(EXPECTED_EXCLUDED_TARGET_IDS):
        raise Part2OperationalOverlayValidationError("The exact 23-route execution roster or exclusion changed.")

    rows = [dict(row) for pair in pairs for row in pair.effective_trajectories]
    row_keys = [(str(row["target_id"]), int(row["trajectory_index"])) for row in rows]
    expected_keys = {
        (target_id, index)
        for target_id in expected_targets
        for index in range(EXPECTED_TRAJECTORIES_PER_ROUTE)
    }
    if len(rows) != EXPECTED_TRAJECTORY_COUNT or len(set(row_keys)) != len(row_keys) or set(row_keys) != expected_keys:
        raise Part2OperationalOverlayValidationError("The effective union is not exactly 276 unique trajectories.")
    for row in rows:
        _validate_trajectory_row(row, overlay=True)
        index = int(row["trajectory_index"])
        if (
            row["environment_seed_index"] != index
            or row["environment_seed"] != seeds[index]
            or row["operationally_eligible"] is not True
        ):
            raise Part2OperationalOverlayValidationError("An effective trajectory changed its seed or remains operationally ineligible.")
    expected_models = runner._aggregate_models(
        rows, all_subjects, expected_trajectories=EXPECTED_TRAJECTORIES_PER_ROUTE,
        capacity=expected_contract.capacity,
    )
    observed_models = [dict(row) for pair in pairs for row in pair.effective_models]
    expected_by_target = {str(row["target_id"]): row for row in expected_models}
    observed_by_target = {str(row["target_id"]): row for row in observed_models}
    if len(observed_models) != EXPECTED_ROUTE_COUNT or observed_by_target != expected_by_target:
        raise Part2OperationalOverlayValidationError("The 23-route model aggregate union does not reconcile.")

    scheduled = sum(int(row["scheduled_agent_days"]) for row in rows)
    restraint = sum(int(row["restraint_count"]) for row in rows)
    overuse = sum(int(row["overuse_count"]) for row in rows)
    invalid = sum(int(row["invalid_count"]) for row in rows)
    valid_actions = restraint + overuse
    if scheduled != valid_actions + invalid or scheduled < 1:
        raise Part2OperationalOverlayValidationError("Union behavioral denominators do not reconcile.")
    environmental_eligible = sum(int(row["invalid_count"]) == 0 for row in rows)
    cascading_main_present = any(
        pair.overlay_manifest.get("artifact_type") == cascading.ARTIFACT_TYPE
        for pair in pairs
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "status": "passed",
        "panel_id": EXPECTED_PANEL_ID,
        "source_overlay_pair_count": 3,
        "route_count": EXPECTED_ROUTE_COUNT,
        "trajectory_count": EXPECTED_TRAJECTORY_COUNT,
        "common_environment_seed_count": EXPECTED_TRAJECTORIES_PER_ROUTE,
        "base_seed": EXPECTED_BASE_SEED,
        "common_environment_seeds": list(EXPECTED_COMMON_ENVIRONMENT_SEEDS),
        "excluded_target_ids": sorted(EXPECTED_EXCLUDED_TARGET_IDS),
        "source_operational_failure_trajectories": sum(
            pair.source_operational_failure_count for pair in pairs
        ),
        "executed_operational_repair_rounds": sum(pair.repair_round_count for pair in pairs),
        "overlay_execution_provenance": {
            "manifest_bound_maximum_rounds": EXPECTED_REPAIR_MAXIMUM_ROUNDS,
            "main_overlay_manifest_bound_credential_pool": True,
            "journal_observable_dispatch_attempt_ceiling": (
                EXPECTED_SOURCE_MAX_TRANSPORT_ATTEMPTS
            ),
            "configured_attempt_ceiling_manifest_bound_for_overlay": False,
            "nonsemantic_local_scheduling_settings_not_manifest_bound": list(
                OVERLAY_UNBOUND_NONSEMANTIC_LOCAL_SETTINGS
            ),
            "semantic_retry_permitted": False,
            "cascading_main_overlay_present": cascading_main_present,
            "cascading_main_overlay_all_execution_settings_manifest_bound": (
                cascading_main_present
            ),
            "cascading_http_400_identical_request_retry_permitted": (
                cascading_main_present
            ),
        },
        "denominators": {
            "behavioral_all_scheduled_living_agent_days": scheduled,
            "behavioral_valid_actions": valid_actions,
            "restraint_actions": restraint,
            "overuse_actions": overuse,
            "semantic_invalid_actions": invalid,
            "restraint_rate_all_scheduled": restraint / scheduled,
            "restraint_rate_valid_actions": restraint / valid_actions if valid_actions else None,
            "environmental_total_trajectories": EXPECTED_TRAJECTORY_COUNT,
            "environmental_eligible_zero_invalid_trajectories": environmental_eligible,
            "environmental_policy": "operationally_eligible_and_zero_semantic_invalid",
        },
    }


def validate_incomplete_cascading_parent(
    source_manifest_path: Path, parent_overlay_manifest_path: Path,
    *, source_verification_root: Path | None = None,
) -> _ValidatedPartialOverlay:
    """Validate and replay the exact immutable 37/38 v3 parent snapshot."""

    tracker = _FileTracker()
    attempt_ids: set[str] = set()
    source_path = _private_manifest_path(source_manifest_path)
    parent_path = _private_manifest_path(parent_overlay_manifest_path)
    try:
        with _hold_run_locks([source_path, parent_path]):
            parent_manifest = _preflight_partial_parent_overlay(parent_path, tracker)
            source_manifest = _safe_json(
                source_path, label="source manifest", tracker=tracker,
            )
            source = _validate_source(
                source_path, source_manifest, tracker=tracker,
                global_attempt_ids=attempt_ids,
                source_verification_root=source_verification_root,
            )
            validated = _validate_partial_parent_overlay(
                parent_path, parent_manifest, source=source, tracker=tracker,
                global_attempt_ids=attempt_ids,
            )
            tracker.verify()
            return validated
    except (
        Part2OperationalOverlayValidationError,
        offline.OfflinePart2FinalizationError,
        InferenceHubPart1PanelError,
        runner.InferenceHubPart2PanelError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, Part2OperationalOverlayValidationError):
            raise
        raise Part2OperationalOverlayValidationError(str(error)) from error


def _discover_cascading_parent_path(overlay_path: Path) -> Path | None:
    try:
        value = json.loads(overlay_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Part2OperationalOverlayValidationError(
            "Operational overlay manifest is not readable during lock discovery."
        ) from error
    if not isinstance(value, Mapping) or value.get("artifact_type") != cascading.ARTIFACT_TYPE:
        return None
    reference = value.get("parent_overlay_manifest")
    if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
        raise Part2OperationalOverlayValidationError(
            "Cascading overlay lacks a discoverable parent path binding."
        )
    return _private_manifest_path(Path(str(reference["path"])))


def validate_operational_overlay_pairs(
    pairs: Sequence[tuple[Path, Path]], *,
    source_verification_root: Path | None = None,
) -> dict[str, Any]:
    """Validate exactly three source/overlay pairs without modifying evidence."""

    if len(pairs) != 3:
        raise Part2OperationalOverlayValidationError("Exactly three source/overlay pairs are required.")
    tracker = _FileTracker()
    global_attempt_ids: set[str] = set()
    try:
        normalized = [
            (_private_manifest_path(source), _private_manifest_path(overlay))
            for source, overlay in pairs
        ]
        discovered_parents = {
            overlay: _discover_cascading_parent_path(overlay)
            for _, overlay in normalized
        }
        lock_paths = {
            path for pair in normalized for path in pair
        } | {
            path for path in discovered_parents.values() if path is not None
        }
        # Overlay locks and completion are checked before source journals so a
        # live production campaign fails quickly and never validates a partial union.
        with _hold_run_locks(sorted(lock_paths)):
            overlay_manifests: dict[Path, dict[str, Any]] = {}
            for _, overlay in normalized:
                if discovered_parents[overlay] is None:
                    overlay_manifests[overlay] = _preflight_overlay(overlay, tracker)
                else:
                    overlay_manifests[overlay] = _preflight_cascading_overlay(
                        overlay, tracker,
                    )
            source_manifests = {
                source: _safe_json(source, label="source manifest", tracker=tracker)
                for source, _ in normalized
            }
            validated: list[_ValidatedPair] = []
            for source_path, overlay_path in normalized:
                source = _validate_source(
                    source_path, source_manifests[source_path], tracker=tracker,
                    global_attempt_ids=global_attempt_ids,
                    source_verification_root=source_verification_root,
                )
                parent_path = discovered_parents[overlay_path]
                if parent_path is None:
                    validated.append(_validate_overlay(
                        overlay_path, overlay_manifests[overlay_path], source=source,
                        tracker=tracker, global_attempt_ids=global_attempt_ids,
                    ))
                else:
                    actual_parent = overlay_manifests[overlay_path].get(
                        "parent_overlay_manifest"
                    )
                    if (
                        not isinstance(actual_parent, Mapping)
                        or Path(str(actual_parent.get("path", ""))).resolve()
                        != parent_path
                    ):
                        raise Part2OperationalOverlayValidationError(
                            "Cascading parent path changed during lock acquisition."
                        )
                    parent_manifest = _preflight_partial_parent_overlay(
                        parent_path, tracker,
                    )
                    partial = _validate_partial_parent_overlay(
                        parent_path, parent_manifest, source=source, tracker=tracker,
                        global_attempt_ids=global_attempt_ids,
                    )
                    validated.append(_validate_cascading_overlay(
                        overlay_path, overlay_manifests[overlay_path], parent=partial,
                        tracker=tracker, global_attempt_ids=global_attempt_ids,
                    ))
            result = _validate_union(validated)
            tracker.verify()
            return result
    except (
        Part2OperationalOverlayValidationError,
        offline.OfflinePart2FinalizationError,
        InferenceHubPart1PanelError,
        runner.InferenceHubPart2PanelError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        if isinstance(error, Part2OperationalOverlayValidationError):
            raise
        raise Part2OperationalOverlayValidationError(str(error)) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair", action="append", nargs=2, metavar=("SOURCE_MANIFEST", "OVERLAY_MANIFEST"),
        type=Path, required=True,
        help="Source private/manifest.json followed by its repair private/manifest.json; repeat exactly three times.",
    )
    parser.add_argument(
        "--source-verification-root", type=Path,
        help="Optional clean repository tree containing the source bytes recorded by the source manifests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate_operational_overlay_pairs(
            [(source, overlay) for source, overlay in args.pair],
            source_verification_root=args.source_verification_root,
        )
    except Part2OperationalOverlayValidationError as error:
        print(f"Part 2 operational overlay validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
