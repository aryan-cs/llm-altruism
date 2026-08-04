"""Repair terminal Part 2 sensitivity operational failures by full reruns.

Only source trajectories marked operationally ineligible because of transport
or identity failures are eligible. Every repair replicate starts from day one
with the exact frozen target route, cell, common environment seed, simulator,
prompt, and request-control contract. Source evidence and genuine semantic
INVALID observations are immutable; all repair evidence lives in a separate
hash-chained overlay.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments.misc import inference_hub_part2_sensitivity_v1 as runner
from experiments.misc.inference_hub_discovery import (
    InferenceHubDiscoveryError,
)
from experiments.misc.inference_hub_sensitivity_deadline_accelerated import (
    _deadline_client,
)


SCHEMA_VERSION = 1
MANIFEST_ARTIFACT_TYPE = (
    "inference_hub_part2_sensitivity_operational_trajectory_repair_v1"
)
TRAJECTORY_ARTIFACT_TYPE = (
    "part2_sensitivity_operational_repair_effective_trajectory_metrics_v1"
)
CELL_ARTIFACT_TYPE = (
    "part2_sensitivity_operational_repair_sentinel_cell_metrics_v1"
)
EFFECT_ARTIFACT_TYPE = (
    "part2_sensitivity_operational_repair_main_effects_v1"
)
OUTCOME_ARTIFACT_TYPE = (
    "part2_sensitivity_operational_repair_outcomes_v1"
)
DEFAULT_MAX_ROUNDS = 8
DEFAULT_CAMPAIGN_WORKERS = 4
DEFAULT_PARTICIPANT_WORKERS = 8
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_BACKOFF_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 900.0
_SOURCE_PATHS = (
    Path(__file__),
    Path(runner.__file__),
    Path(__file__).with_name("inference_hub_sensitivity_deadline_accelerated.py"),
    Path(__file__).with_name("inference_hub_part2_panel.py"),
    Path(__file__).with_name("inference_hub_rate_limit.py"),
    Path(__file__).parents[1] / "part2" / "part_2.py",
    Path(__file__).parents[2] / "agents" / "agent_2.py",
    Path(__file__).parents[2] / "analysis" / "part2_confirmatory.py",
)


class Part2SensitivityOperationalRepairError(RuntimeError):
    """The source or repair overlay violates the immutable repair contract."""


class _NoDispatch:
    def post(self, *_: Any, **__: Any) -> Any:  # pragma: no cover - failure guard
        raise Part2SensitivityOperationalRepairError(
            "Source trajectory reconstruction attempted a provider dispatch."
        )


class _NoBudget:
    def reserve(self, **_: Any) -> None:  # pragma: no cover - failure guard
        raise Part2SensitivityOperationalRepairError(
            "Source trajectory reconstruction attempted a physical reservation."
        )


def _require_private_file(path: Path) -> None:
    if not path.is_file():
        raise Part2SensitivityOperationalRepairError(
            f"Private evidence is missing: {path}."
        )
    if os.name == "posix" and stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise Part2SensitivityOperationalRepairError(
            f"Private evidence must have mode 0600: {path}."
        )


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _load_bound_json(
    reference: Mapping[str, Any], *, label: str, parent: Path | None = None
) -> tuple[Path, dict[str, Any]]:
    path_value = reference.get("path")
    if not isinstance(path_value, str):
        raise Part2SensitivityOperationalRepairError(f"{label} path is missing.")
    path = Path(path_value).resolve()
    if parent is not None and not _within(path, parent):
        raise Part2SensitivityOperationalRepairError(f"{label} escaped its root.")
    payload = runner._read_json(path, label)
    if reference.get("file_sha256") != runner._sha256_file(path):
        raise Part2SensitivityOperationalRepairError(f"{label} file hash changed.")
    evidence = reference.get("evidence_sha256")
    if evidence is not None and payload.get("evidence_sha256") != evidence:
        raise Part2SensitivityOperationalRepairError(f"{label} evidence hash changed.")
    canonical = reference.get("canonical_sha256")
    if canonical is not None and runner._sha256_json(payload) != canonical:
        raise Part2SensitivityOperationalRepairError(f"{label} canonical hash changed.")
    return path, payload


def _source_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("cell_id")),
        str(row.get("target_id")),
        int(row.get("trajectory_index")),
    )


def _load_source(
    manifest_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[runner.SensitivityCondition],
    dict[str, Mapping[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
    Any,
]:
    """Validate all 160 source trajectories from their immutable raw journals."""

    manifest_path = manifest_path.resolve()
    _require_private_file(manifest_path)
    manifest = runner._read_json(manifest_path, "source sensitivity manifest")
    if (
        manifest.get("schema_version") != runner.SENSITIVITY_SCHEMA_VERSION
        or manifest.get("artifact_type")
        != "inference_hub_part2_sensitivity_campaign_v1"
        or manifest.get("evidence_sha256") != runner._self_hash(manifest)
        or manifest.get("campaign_id")
        != "part2_resolution_v_deadline_exploratory_v2"
    ):
        raise Part2SensitivityOperationalRepairError(
            "Source sensitivity manifest identity or integrity failed."
        )
    summary = manifest.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("planned_trajectories") != 160
        or summary.get("completed_trajectories") != 160
        or summary.get("identity_mismatch_count") != 0
    ):
        raise Part2SensitivityOperationalRepairError(
            "Source sensitivity campaign is not terminal with 160 trajectories."
        )
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or not source_artifacts:
        raise Part2SensitivityOperationalRepairError(
            "Source sensitivity provenance is missing."
        )
    inputs = manifest.get("input_artifacts")
    if not isinstance(inputs, Mapping):
        raise Part2SensitivityOperationalRepairError("Source inputs are missing.")
    panel_path, _ = _load_bound_json(inputs["panel"], label="source panel")
    design_path, design_document = _load_bound_json(
        inputs["design"], label="source sensitivity design"
    )
    design, conditions = runner.load_sensitivity_design(design_path)
    panel, frozen_contract = runner._load_panel(panel_path)
    if design != design_document:
        raise Part2SensitivityOperationalRepairError(
            "Source design loader changed the frozen document."
        )
    if manifest.get("cells") != [condition.public_dict() for condition in conditions]:
        raise Part2SensitivityOperationalRepairError("Source cell bindings changed.")
    subjects_list = manifest.get("selected_subject_routes")
    if not isinstance(subjects_list, list):
        raise Part2SensitivityOperationalRepairError("Source routes are missing.")
    subjects = {
        str(row.get("target_id")): row
        for row in subjects_list
        if isinstance(row, Mapping) and row.get("target_id")
    }
    frozen_ids = [str(row["target_id"]) for row in design["sentinels"]]
    if list(subjects) != frozen_ids or len(subjects) != 5:
        raise Part2SensitivityOperationalRepairError(
            "Source exact five-sentinel route panel changed."
        )
    common_seeds = runner.condition_environment_seeds(
        campaign_id=str(design["campaign_id"]),
        panel_id=str(panel["panel_id"]),
        base_seed=int(manifest["seed_contract"]["base_seed"]),
        trajectory_count=int(design["seeds_per_cell"]),
    )
    if manifest.get("common_environment_seeds") != common_seeds:
        raise Part2SensitivityOperationalRepairError(
            "Source common environment seeds changed."
        )
    refs = manifest.get("journals")
    if not isinstance(refs, Mapping) or len(refs) != 160:
        raise Part2SensitivityOperationalRepairError(
            "Source trajectory journal set is not exactly 160."
        )
    trajectory_ref = manifest.get("sanitized_artifacts", {}).get("trajectory_metrics")
    if not isinstance(trajectory_ref, Mapping):
        raise Part2SensitivityOperationalRepairError(
            "Source trajectory metrics reference is missing."
        )
    _, trajectory_payload = _load_bound_json(
        trajectory_ref,
        label="source trajectory metrics",
        parent=manifest_path.parents[1] / "sanitized",
    )
    if (
        trajectory_payload.get("artifact_type")
        != "part2_sensitivity_trajectory_metrics_v1"
        or trajectory_payload.get("evidence_sha256")
        != runner._self_hash(trajectory_payload)
        or not isinstance(trajectory_payload.get("rows"), list)
        or len(trajectory_payload["rows"]) != 160
    ):
        raise Part2SensitivityOperationalRepairError(
            "Source trajectory metric artifact is invalid."
        )
    public_rows = {_source_key(row): dict(row) for row in trajectory_payload["rows"]}
    if len(public_rows) != 160:
        raise Part2SensitivityOperationalRepairError(
            "Source trajectory identities are duplicated."
        )
    condition_by_id = {condition.cell_id: condition for condition in conditions}
    reconstructed: dict[tuple[str, str, int], dict[str, Any]] = {}
    private_dir = manifest_path.parent
    for key_string, reference in refs.items():
        parts = str(key_string).rsplit("::", 2)
        if len(parts) != 3:
            raise Part2SensitivityOperationalRepairError(
                "Source trajectory journal key is malformed."
            )
        cell_id, target_id, index_text = parts
        index = int(index_text)
        key = (cell_id, target_id, index)
        if key not in public_rows or target_id not in subjects or cell_id not in condition_by_id:
            raise Part2SensitivityOperationalRepairError(
                "Source trajectory journal identity is outside the frozen design."
            )
        if not isinstance(reference, Mapping) or not isinstance(reference.get("path"), str):
            raise Part2SensitivityOperationalRepairError(
                "Source trajectory checkpoint is malformed."
            )
        path = Path(reference["path"]).resolve()
        if not _within(path, private_dir):
            raise Part2SensitivityOperationalRepairError(
                "Source trajectory journal escaped private evidence."
            )
        journal = runner._ConditionJournal(
            path,
            campaign_id=str(manifest["campaign_id"]),
            condition=condition_by_id[cell_id],
        )
        runner._validate_checkpoint_reference(
            journal._journal, reference, label=f"source trajectory {key_string}"
        )
        if len(journal.records) != reference.get("record_count"):
            raise Part2SensitivityOperationalRepairError(
                "Source trajectory grew beyond its terminal checkpoint."
            )
        rebuilt = runner._trajectory_row(
            runner._run_trajectory(
                subject=subjects[target_id],
                trajectory_index=index,
                environment_seed=common_seeds[index],
                contract=condition_by_id[cell_id].contract(
                    frozen_contract, trajectories=len(common_seeds)
                ),
                journal=journal,
                client=_NoDispatch(),
                participant_workers=1,
                max_attempts=int(
                    manifest["execution_contract"]["max_transport_attempts"]
                ),
                initial_backoff_seconds=0,
                sleep_fn=lambda _: None,
                maximum_input_bytes=int(
                    design["execution_budget"]["maximum_input_utf8_bytes_per_attempt"]
                ),
                attempt_budget=_NoBudget(),
                cell_id=cell_id,
            ),
            condition_by_id[cell_id],
        )
        if runner._canonical_bytes(rebuilt) != runner._canonical_bytes(public_rows[key]):
            raise Part2SensitivityOperationalRepairError(
                f"Source trajectory metrics do not reproduce for {key_string}."
            )
        reconstructed[key] = rebuilt
    if set(reconstructed) != set(public_rows):
        raise Part2SensitivityOperationalRepairError(
            "Source trajectory journal coverage is incomplete."
        )
    failures = [row for row in reconstructed.values() if not row["operationally_eligible"]]
    if (
        len(failures) != int(summary["operational_failure_trajectories"])
        or sum(int(row["transport_failure_count"]) for row in failures)
        != int(summary["transport_failure_count"])
        or any(int(row["identity_mismatch_count"]) for row in failures)
    ):
        raise Part2SensitivityOperationalRepairError(
            "Source operational-failure accounting does not reproduce."
        )
    return manifest, design, conditions, subjects, reconstructed, frozen_contract


def _load_complete_subset(
    subset_manifest_path: Path,
    *,
    source: Mapping[str, Any],
    design: Mapping[str, Any],
    conditions: Sequence[runner.SensitivityCondition],
    subjects: Mapping[str, Mapping[str, Any]],
    frozen_contract: Any,
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, Any]]]:
    """Validate a COMPLETE production subset and reconstruct every trajectory."""

    path = subset_manifest_path.resolve()
    _require_private_file(path)
    subset = runner._read_json(path, "complete sensitivity repair subset")
    if (
        subset.get("schema_version") != runner.SENSITIVITY_SCHEMA_VERSION
        or subset.get("artifact_type")
        != "inference_hub_part2_sensitivity_campaign_v1"
        or subset.get("evidence_sha256") != runner._self_hash(subset)
        or subset.get("campaign_id") != source.get("campaign_id")
        or subset.get("complete") is not True
        or not subset.get("completed_at_utc")
        or subset.get("input_artifacts") != source.get("input_artifacts")
        or subset.get("source_artifacts") != source.get("source_artifacts")
        or subset.get("seed_contract") != source.get("seed_contract")
        or subset.get("common_environment_seeds")
        != source.get("common_environment_seeds")
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset is not COMPLETE under the exact frozen source contract."
        )
    source_artifacts = subset.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or any(
        not Path(str(source_path)).is_file()
        or runner._sha256_file(Path(str(source_path))) != digest
        for source_path, digest in source_artifacts.items()
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset implementation provenance changed."
        )
    source_subject_rows = {
        target: runner._canonical_bytes(row) for target, row in subjects.items()
    }
    subset_subjects = {
        str(row.get("target_id")): row
        for row in subset.get("selected_subject_routes", [])
        if isinstance(row, Mapping) and row.get("target_id")
    }
    if (
        not subset_subjects
        or any(
            target not in source_subject_rows
            or runner._canonical_bytes(row) != source_subject_rows[target]
            for target, row in subset_subjects.items()
        )
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset route identities changed."
        )
    condition_by_id = {condition.cell_id: condition for condition in conditions}
    subset_cells = subset.get("cells")
    if (
        not isinstance(subset_cells, list)
        or not subset_cells
        or any(
            not isinstance(row, Mapping)
            or str(row.get("cell_id")) not in condition_by_id
            or runner._canonical_bytes(row)
            != runner._canonical_bytes(
                condition_by_id[str(row.get("cell_id"))].public_dict()
            )
            for row in subset_cells
        )
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset cells changed the frozen design."
        )
    execution = subset.get("execution_contract")
    source_execution = source.get("execution_contract")
    if (
        not isinstance(execution, Mapping)
        or not isinstance(source_execution, Mapping)
        or any(
            execution.get(key) != source_execution.get(key)
            for key in (
                "identity_check",
                "journal",
                "max_transport_attempts",
                "request_contract",
                "visible_output_only",
            )
        )
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset request or identity contract changed."
        )
    trajectory_ref = subset.get("sanitized_artifacts", {}).get("trajectory_metrics")
    if not isinstance(trajectory_ref, Mapping):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset trajectory metrics are missing."
        )
    _, payload = _load_bound_json(
        trajectory_ref,
        label="repair subset trajectory metrics",
        parent=path.parents[1] / "sanitized",
    )
    if (
        payload.get("artifact_type") != "part2_sensitivity_trajectory_metrics_v1"
        or payload.get("evidence_sha256") != runner._self_hash(payload)
        or not isinstance(payload.get("rows"), list)
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset trajectory metrics are invalid."
        )
    public_rows = {_source_key(row): dict(row) for row in payload["rows"]}
    refs = subset.get("journals")
    if (
        not isinstance(refs, Mapping)
        or len(public_rows) != len(payload["rows"])
        or len(refs) != len(public_rows)
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset trajectory membership is invalid."
        )
    common_seeds = list(source["common_environment_seeds"])
    rebuilt: dict[tuple[str, str, int], dict[str, Any]] = {}
    reserved_ids: set[str] = set()
    for key_text, reference in refs.items():
        parts = str(key_text).rsplit("::", 2)
        if len(parts) != 3:
            raise Part2SensitivityOperationalRepairError(
                "Repair subset journal key is malformed."
            )
        key = (parts[0], parts[1], int(parts[2]))
        cell_id, target_id, trajectory_index = key
        if (
            key not in public_rows
            or cell_id not in condition_by_id
            or target_id not in subset_subjects
            or not 0 <= trajectory_index < len(common_seeds)
            or not isinstance(reference, Mapping)
            or not isinstance(reference.get("path"), str)
        ):
            raise Part2SensitivityOperationalRepairError(
                "Repair subset trajectory is outside the frozen schedule."
            )
        journal_path = Path(reference["path"]).resolve()
        if not _within(journal_path, path.parent):
            raise Part2SensitivityOperationalRepairError(
                "Repair subset journal escaped its private run."
            )
        journal = runner._ConditionJournal(
            journal_path,
            campaign_id=str(subset["campaign_id"]),
            condition=condition_by_id[cell_id],
        )
        runner._validate_checkpoint_reference(
            journal._journal, reference, label=f"repair subset trajectory {key_text}"
        )
        reserved_ids.update(
            str(row.get("attempt_id"))
            for row in journal.records
            if row.get("event") == "reserved_before_dispatch"
            and row.get("dispatch_skipped") is not True
        )
        row = runner._trajectory_row(
            runner._run_trajectory(
                subject=subset_subjects[target_id],
                trajectory_index=trajectory_index,
                environment_seed=common_seeds[trajectory_index],
                contract=condition_by_id[cell_id].contract(
                    frozen_contract, trajectories=len(common_seeds)
                ),
                journal=journal,
                client=_NoDispatch(),
                participant_workers=1,
                max_attempts=int(execution["max_transport_attempts"]),
                initial_backoff_seconds=0,
                sleep_fn=lambda _: None,
                maximum_input_bytes=int(
                    design["execution_budget"][
                        "maximum_input_utf8_bytes_per_attempt"
                    ]
                ),
                attempt_budget=_NoBudget(),
                cell_id=cell_id,
            ),
            condition_by_id[cell_id],
        )
        if runner._canonical_bytes(row) != runner._canonical_bytes(public_rows[key]):
            raise Part2SensitivityOperationalRepairError(
                "Repair subset trajectory does not reproduce."
            )
        rebuilt[key] = row
    attempt_ref = subset.get("attempt_ledger")
    if not isinstance(attempt_ref, Mapping) or not isinstance(
        attempt_ref.get("path"), str
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset attempt ledger is missing."
        )
    ledger = runner._ChainedJournal(Path(attempt_ref["path"]))
    runner._validate_checkpoint_reference(
        ledger, attempt_ref, label="repair subset physical attempt ledger"
    )
    ledger_ids = {str(row.get("attempt_id")) for row in ledger.records}
    summary = subset.get("summary")
    if (
        len(ledger_ids) != len(ledger.records)
        or ledger_ids != reserved_ids
        or not isinstance(summary, Mapping)
        or summary.get("planned_trajectories") != len(rebuilt)
        or summary.get("completed_trajectories") != len(rebuilt)
        or summary.get("operational_failure_trajectories") != 0
        or summary.get("transport_failure_count") != 0
        or summary.get("identity_mismatch_count") != 0
        or any(row.get("operationally_eligible") is not True for row in rebuilt.values())
    ):
        raise Part2SensitivityOperationalRepairError(
            "Repair subset terminal/attempt accounting failed."
        )
    return subset, rebuilt


def _bindings(
    source_path: Path,
    source: Mapping[str, Any],
    max_rounds: int,
    max_attempts: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "source_manifest": {
            "path": str(source_path.resolve()),
            "file_sha256": runner._sha256_file(source_path),
            "evidence_sha256": source["evidence_sha256"],
        },
        "source_campaign_id": source["campaign_id"],
        "source_journal_references_sha256": runner._sha256_json(source["journals"]),
        "source_trajectory_metrics_evidence_sha256": source["sanitized_artifacts"][
            "trajectory_metrics"
        ]["evidence_sha256"],
        "repair_source_artifacts": {
            str(path.resolve()): runner._sha256_file(path.resolve())
            for path in _SOURCE_PATHS
        },
        "max_full_trajectory_rounds": max_rounds,
        "max_physical_attempts_per_agent_day": max_attempts,
        "eligibility_policy": "source_operationally_ineligible_trajectory_only_v1",
        "rerun_policy": "whole_trajectory_from_day_1_exact_frozen_contract_v1",
        "semantic_invalid_retry_permitted": False,
        "source_mutated": False,
        "substitution_permitted": False,
    }


def _seal_payload(payload: dict[str, Any]) -> None:
    runner._seal(payload)


def _artifacts(
    *,
    source: Mapping[str, Any],
    design: Mapping[str, Any],
    conditions: Sequence[runner.SensitivityCondition],
    subjects: Mapping[str, Mapping[str, Any]],
    source_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    successful: Mapping[tuple[str, str, int], tuple[int, Mapping[str, Any]]],
    round_rows: Mapping[tuple[str, str, int, int], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    effective = dict(source_rows)
    for key, (_, row) in successful.items():
        effective[key] = row
    rows = [effective[key] for key in sorted(effective)]
    cell_rows = runner._target_condition_rows(
        rows,
        subjects=list(subjects.values()),
        conditions=conditions,
        expected_trajectories=int(design["seeds_per_cell"]),
    )
    effects = runner._analyze_completed_design(
        rows, sentinel_ids=list(subjects), design=design
    )
    trajectory_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": TRAJECTORY_ARTIFACT_TYPE,
        "source_manifest_evidence_sha256": source["evidence_sha256"],
        "source_mutated": False,
        "semantic_invalid_retry_permitted": False,
        "replacement_unit": "complete_trajectory_only",
        "rows": rows,
    }
    cell_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": CELL_ARTIFACT_TYPE,
        "source_manifest_evidence_sha256": source["evidence_sha256"],
        "rows": cell_rows,
    }
    effect_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": EFFECT_ARTIFACT_TYPE,
        "source_manifest_evidence_sha256": source["evidence_sha256"],
        "analysis_status": "complete_deadline_exploratory_operational_overlay",
        "inference_scope": design["analysis"]["inference_scope"],
        "confirmatory": False,
        "global_holm_family": "25_prespecified_sentinel_by_factor_main_effects",
        "global_holm_family_size": 25,
        "rows": effects,
    }
    eligible = {key for key, row in source_rows.items() if not row["operationally_eligible"]}
    outcomes = []
    for key in sorted(eligible):
        success = successful.get(key)
        attempts = [
            row for round_key, row in sorted(round_rows.items()) if round_key[:3] == key
        ]
        outcomes.append(
            {
                "cell_id": key[0],
                "target_id": key[1],
                "trajectory_index": key[2],
                "environment_seed": source_rows[key]["environment_seed"],
                "source_operationally_eligible": False,
                "source_transport_failure_count": source_rows[key][
                    "transport_failure_count"
                ],
                "source_identity_mismatch_count": source_rows[key][
                    "identity_mismatch_count"
                ],
                "full_trajectory_rounds_completed": len(attempts),
                "successful_round": success[0] if success else None,
                "repair_operationally_eligible": bool(success),
                "repair_semantic_invalid_count": (
                    int(success[1]["invalid_count"]) if success else None
                ),
            }
        )
    outcome_payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": OUTCOME_ARTIFACT_TYPE,
        "source_manifest_evidence_sha256": source["evidence_sha256"],
        "source_trajectory_count": len(source_rows),
        "eligible_operational_failure_trajectory_count": len(eligible),
        "successful_full_trajectory_repair_count": len(successful),
        "unresolved_operational_failure_trajectory_count": len(eligible - set(successful)),
        "source_semantic_invalid_trajectories_retried": 0,
        "source_mutated": False,
        "rows": outcomes,
    }
    for payload in (trajectory_payload, cell_payload, effect_payload, outcome_payload):
        _seal_payload(payload)
    return trajectory_payload, cell_payload, effect_payload, outcome_payload


def build_overlay_from_complete_subset(
    *,
    source_manifest_path: Path,
    subset_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build an immutable overlay by selecting exact failures from a full subset."""

    source_path = source_manifest_path.resolve()
    subset_path = subset_manifest_path.resolve()
    if output_dir.exists():
        raise Part2SensitivityOperationalRepairError(
            "Repair overlay output already exists; refusing overwrite."
        )
    source_lock = runner._acquire_run_lock(source_path.parent)
    try:
        subset_lock = runner._acquire_run_lock(subset_path.parent)
        try:
            source, design, conditions, subjects, source_rows, frozen_contract = (
                _load_source(source_path)
            )
            subset, subset_rows = _load_complete_subset(
                subset_path,
                source=source,
                design=design,
                conditions=conditions,
                subjects=subjects,
                frozen_contract=frozen_contract,
            )
            eligible = {
                key
                for key, row in source_rows.items()
                if row.get("operationally_eligible") is not True
            }
            if not eligible or not eligible <= set(subset_rows):
                raise Part2SensitivityOperationalRepairError(
                    "Complete subset does not contain every exact source failure."
                )
            selected = {key: subset_rows[key] for key in eligible}
            if any(
                row.get("operationally_eligible") is not True
                or int(row.get("identity_mismatch_count", -1)) != 0
                or int(row.get("transport_failure_count", -1)) != 0
                for row in selected.values()
            ):
                raise Part2SensitivityOperationalRepairError(
                    "Complete subset did not operationally resolve every source failure."
                )
            successful = {key: (1, row) for key, row in selected.items()}
            round_rows = {(*key, 1): row for key, row in selected.items()}
            trajectory, cells, effects, outcomes = _artifacts(
                source=source,
                design=design,
                conditions=conditions,
                subjects=subjects,
                source_rows=source_rows,
                successful=successful,
                round_rows=round_rows,
            )
            private_dir = output_dir / "private"
            sanitized_dir = output_dir / "sanitized"
            output_dir.mkdir(parents=True, mode=0o700)
            private_dir.mkdir(mode=0o700)
            sanitized_dir.mkdir(mode=0o700)
            for directory in (output_dir, private_dir, sanitized_dir):
                runner._secure_mode(directory, 0o700)
                runner._require_mode(directory, 0o700)
            repair_lock = runner._acquire_run_lock(private_dir)
            fcntl.flock(repair_lock.fileno(), fcntl.LOCK_UN)
            repair_lock.close()
            payloads = {
                "effective_trajectory_metrics": trajectory,
                "effective_sentinel_cell_metrics": cells,
                "effective_main_effects": effects,
                "repair_outcomes": outcomes,
            }
            sanitized_refs = {}
            for name, payload in payloads.items():
                artifact_path = sanitized_dir / f"{name}.json"
                runner._atomic_json(artifact_path, payload)
                sanitized_refs[name] = {
                    "path": str(artifact_path.resolve()),
                    "file_sha256": runner._sha256_file(artifact_path),
                    "evidence_sha256": payload["evidence_sha256"],
                }
            subset_refs = subset["journals"]
            selected_refs = {
                f"{key[0]}::{key[1]}::{key[2]}::1": subset_refs[
                    f"{key[0]}::{key[1]}::{key[2]}"
                ]
                for key in sorted(eligible)
            }
            manifest = {
                **_bindings(source_path, source, 1, int(
                    subset["execution_contract"]["max_transport_attempts"]
                )),
                "repair_source_kind": "complete_frozen_contract_subset_manifest_v1",
                "repair_subset_manifest": {
                    "path": str(subset_path),
                    "file_sha256": runner._sha256_file(subset_path),
                    "evidence_sha256": subset["evidence_sha256"],
                },
                "subset_full_journal_references_sha256": runner._sha256_json(
                    subset["journals"]
                ),
                "selected_subset_journal_references_sha256": runner._sha256_json(
                    selected_refs
                ),
                "subset_trajectory_count": len(subset_rows),
                "selected_subset_trajectory_count": len(selected),
                "excluded_subset_trajectory_count": len(subset_rows) - len(selected),
                "created_at_utc": runner._utc_now(),
                "last_updated_at_utc": runner._utc_now(),
                "complete": True,
                "completed_at_utc": runner._utc_now(),
                "summary": {
                    "source_trajectory_count": len(source_rows),
                    "eligible_operational_failure_trajectories": len(eligible),
                    "successful_full_trajectory_repairs": len(selected),
                    "unresolved_operational_failure_trajectories": 0,
                    "effective_trajectory_count": len(trajectory["rows"]),
                    "effective_operational_failure_trajectories": 0,
                    "effective_identity_mismatch_count": 0,
                    "effective_transport_failure_count": 0,
                    "source_semantic_invalid_trajectories_retried": 0,
                },
                "journals": selected_refs,
                "attempt_ledger": subset["attempt_ledger"],
                "sanitized_artifacts": sanitized_refs,
            }
            runner._seal(manifest)
            runner._atomic_json(private_dir / "manifest.json", manifest)
            return manifest
        finally:
            fcntl.flock(subset_lock.fileno(), fcntl.LOCK_UN)
            subset_lock.close()
    finally:
        fcntl.flock(source_lock.fileno(), fcntl.LOCK_UN)
        source_lock.close()


def run_repair(
    *,
    source_manifest_path: Path,
    output_dir: Path,
    client: Any,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    campaign_workers: int = DEFAULT_CAMPAIGN_WORKERS,
    participant_workers: int = DEFAULT_PARTICIPANT_WORKERS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    resume: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run whole-trajectory operational repairs in a separate immutable overlay."""

    for name, value in (
        ("max_rounds", max_rounds),
        ("campaign_workers", campaign_workers),
        ("participant_workers", participant_workers),
        ("max_attempts", max_attempts),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise Part2SensitivityOperationalRepairError(
                f"{name} must be a positive integer."
            )
    if initial_backoff_seconds < 0:
        raise Part2SensitivityOperationalRepairError(
            "initial_backoff_seconds must be nonnegative."
        )
    runner._rate_limit_contract(client)
    source_path = source_manifest_path.resolve()
    try:
        source_lock = runner._acquire_run_lock(source_path.parent)
    except Exception as error:
        raise Part2SensitivityOperationalRepairError(
            "Source sensitivity writer is active; repair requires terminal evidence."
        ) from error
    try:
        try:
            (
                source,
                design,
                conditions,
                subjects,
                source_rows,
                frozen_contract,
            ) = _load_source(source_path)
        except (
            runner.InferenceHubPart2SensitivityError,
            runner.InferenceHubPart2PanelError,
        ) as error:
            raise Part2SensitivityOperationalRepairError(
                "Source trajectory route/request/state binding failed."
            ) from error
        eligible = {
            key: row for key, row in source_rows.items() if not row["operationally_eligible"]
        }
        if any(
            row["operationally_eligible"] and row["invalid_count"] > 0
            for row in source_rows.values()
            if _source_key(row) in eligible
        ):
            raise AssertionError("Semantic-only INVALID entered operational eligibility.")
        bindings = _bindings(source_path, source, max_rounds, max_attempts)
        if output_dir.exists() and not resume:
            raise Part2SensitivityOperationalRepairError(
                "Repair output exists; use --resume explicitly."
            )
        private_dir = output_dir / "private"
        conditions_dir = private_dir / "conditions"
        sanitized_dir = output_dir / "sanitized"
        if resume:
            for directory in (output_dir, private_dir, conditions_dir, sanitized_dir):
                if not directory.is_dir():
                    raise Part2SensitivityOperationalRepairError(
                        f"Repair resume directory is missing: {directory}."
                    )
        else:
            output_dir.mkdir(parents=True, mode=0o700)
            for directory in (private_dir, conditions_dir, sanitized_dir):
                directory.mkdir(mode=0o700)
        for directory in (output_dir, private_dir, conditions_dir, sanitized_dir):
            runner._secure_mode(directory, 0o700)
            runner._require_mode(directory, 0o700)
        repair_lock = runner._acquire_run_lock(private_dir)
        try:
            manifest_path = private_dir / "manifest.json"
            condition_by_id = {condition.cell_id: condition for condition in conditions}
            journal_map: dict[
                tuple[str, str, int, int], runner._ConditionJournal
            ] = {}
            for cell_id, target_id, trajectory_index in eligible:
                condition = condition_by_id[cell_id]
                for round_index in range(1, max_rounds + 1):
                    path = (
                        conditions_dir
                        / runner._safe_file_stem(cell_id)
                        / runner._safe_file_stem(target_id)
                        / f"seed-{trajectory_index:03d}-round-{round_index:02d}.jsonl"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    runner._secure_mode(path.parent, 0o700)
                    journal_map[(cell_id, target_id, trajectory_index, round_index)] = (
                        runner._ConditionJournal(
                            path,
                            campaign_id=(
                                f"{source['campaign_id']}_operational_repair_overlay_v1"
                            ),
                            condition=condition,
                        )
                    )
            ceiling = (
                sum(
                    condition_by_id[key[0]].society_size
                    * condition_by_id[key[0]].horizon_days
                    for key in eligible
                )
                * max_rounds
                * max_attempts
            )
            attempt_budget = runner._AttemptBudget(
                private_dir / "physical_attempt_ledger.jsonl",
                campaign_id=f"{source['campaign_id']}_operational_repair_overlay_v1",
                ceiling=ceiling,
            )
            if resume:
                manifest = runner._read_json(manifest_path, "repair manifest")
                _require_private_file(manifest_path)
                if manifest.get("evidence_sha256") != runner._self_hash(manifest):
                    raise Part2SensitivityOperationalRepairError(
                        "Repair manifest self-hash failed."
                    )
                if {key: manifest.get(key) for key in bindings} != bindings:
                    raise Part2SensitivityOperationalRepairError(
                        "Repair/source bindings changed on resume."
                    )
                refs = manifest.get("journals")
                expected_names = {
                    f"{key[0]}::{key[1]}::{key[2]}::{key[3]}" for key in journal_map
                }
                if not isinstance(refs, Mapping) or set(refs) != expected_names:
                    raise Part2SensitivityOperationalRepairError(
                        "Repair trajectory journal set changed."
                    )
                for key, journal in journal_map.items():
                    runner._validate_checkpoint_reference(
                        journal._journal,
                        refs[f"{key[0]}::{key[1]}::{key[2]}::{key[3]}"],
                        label=f"repair trajectory {key}",
                    )
                runner._validate_checkpoint_reference(
                    attempt_budget._journal,
                    manifest.get("attempt_ledger"),
                    label="repair physical attempt ledger",
                )
            else:
                manifest = {
                    **bindings,
                    "created_at_utc": runner._utc_now(),
                    "last_updated_at_utc": runner._utc_now(),
                    "complete": False,
                    "summary": {},
                    "journals": {
                        f"{key[0]}::{key[1]}::{key[2]}::{key[3]}": journal.reference()
                        for key, journal in journal_map.items()
                    },
                    "attempt_ledger": attempt_budget.reference(),
                }
                runner._seal(manifest)
                runner._atomic_json(manifest_path, manifest)

            round_rows: dict[tuple[str, str, int, int], Mapping[str, Any]] = {}
            successful: dict[
                tuple[str, str, int], tuple[int, Mapping[str, Any]]
            ] = {}
            common_seeds = list(source["common_environment_seeds"])
            maximum_input_bytes = int(
                design["execution_budget"]["maximum_input_utf8_bytes_per_attempt"]
            )
            for round_index in range(1, max_rounds + 1):
                work = [key for key in eligible if key not in successful]
                if work:
                    with ThreadPoolExecutor(
                        max_workers=min(campaign_workers, len(work)),
                        thread_name_prefix=f"p2-sensitivity-repair-r{round_index}",
                    ) as executor:
                        future_map = {}
                        for key in work:
                            cell_id, target_id, trajectory_index = key
                            condition = condition_by_id[cell_id]
                            future = executor.submit(
                                runner._run_trajectory,
                                subject=subjects[target_id],
                                trajectory_index=trajectory_index,
                                environment_seed=common_seeds[trajectory_index],
                                contract=condition.contract(
                                    frozen_contract, trajectories=len(common_seeds)
                                ),
                                journal=journal_map[(*key, round_index)],
                                client=client,
                                participant_workers=participant_workers,
                                max_attempts=max_attempts,
                                initial_backoff_seconds=initial_backoff_seconds,
                                sleep_fn=sleep_fn,
                                maximum_input_bytes=maximum_input_bytes,
                                attempt_budget=attempt_budget,
                                cell_id=cell_id,
                            )
                            future_map[future] = key
                        for future in as_completed(future_map):
                            key = future_map[future]
                            row = runner._trajectory_row(
                                future.result(), condition_by_id[key[0]]
                            )
                            round_rows[(*key, round_index)] = row
                            if row["operationally_eligible"]:
                                successful[key] = (round_index, row)
                manifest["last_updated_at_utc"] = runner._utc_now()
                manifest["summary"] = {
                    "source_trajectory_count": len(source_rows),
                    "eligible_operational_failure_trajectories": len(eligible),
                    "successful_full_trajectory_repairs": len(successful),
                    "unresolved_operational_failure_trajectories": len(
                        set(eligible) - set(successful)
                    ),
                    "full_trajectory_rounds_started": round_index,
                    "source_semantic_invalid_trajectories_retried": 0,
                }
                manifest["journals"] = {
                    f"{key[0]}::{key[1]}::{key[2]}::{key[3]}": journal.reference()
                    for key, journal in journal_map.items()
                }
                manifest["attempt_ledger"] = attempt_budget.reference()
                runner._seal(manifest)
                runner._atomic_json(manifest_path, manifest)
                if len(successful) == len(eligible):
                    break

            trajectory_payload, cell_payload, effect_payload, outcome_payload = _artifacts(
                source=source,
                design=design,
                conditions=conditions,
                subjects=subjects,
                source_rows=source_rows,
                successful=successful,
                round_rows=round_rows,
            )
            payloads = {
                "effective_trajectory_metrics": trajectory_payload,
                "effective_sentinel_cell_metrics": cell_payload,
                "effective_main_effects": effect_payload,
                "repair_outcomes": outcome_payload,
            }
            sanitized_refs = {}
            for name, payload in payloads.items():
                path = sanitized_dir / f"{name}.json"
                runner._atomic_json(path, payload)
                sanitized_refs[name] = {
                    "path": str(path.resolve()),
                    "file_sha256": runner._sha256_file(path),
                    "evidence_sha256": payload["evidence_sha256"],
                }
            manifest["complete"] = len(successful) == len(eligible)
            manifest["summary"] = {
                "source_trajectory_count": len(source_rows),
                "eligible_operational_failure_trajectories": len(eligible),
                "successful_full_trajectory_repairs": len(successful),
                "unresolved_operational_failure_trajectories": len(
                    set(eligible) - set(successful)
                ),
                "effective_trajectory_count": len(trajectory_payload["rows"]),
                "effective_operational_failure_trajectories": sum(
                    not row["operationally_eligible"]
                    for row in trajectory_payload["rows"]
                ),
                "effective_identity_mismatch_count": sum(
                    int(row["identity_mismatch_count"])
                    for row in trajectory_payload["rows"]
                ),
                "effective_transport_failure_count": sum(
                    int(row["transport_failure_count"])
                    for row in trajectory_payload["rows"]
                ),
                "source_semantic_invalid_trajectories_retried": 0,
            }
            manifest["last_updated_at_utc"] = runner._utc_now()
            if manifest["complete"]:
                manifest["completed_at_utc"] = runner._utc_now()
            manifest["journals"] = {
                f"{key[0]}::{key[1]}::{key[2]}::{key[3]}": journal.reference()
                for key, journal in journal_map.items()
            }
            manifest["attempt_ledger"] = attempt_budget.reference()
            manifest["sanitized_artifacts"] = sanitized_refs
            runner._seal(manifest)
            runner._atomic_json(manifest_path, manifest)
            return manifest
        finally:
            fcntl.flock(repair_lock.fileno(), fcntl.LOCK_UN)
            repair_lock.close()
    finally:
        fcntl.flock(source_lock.fileno(), fcntl.LOCK_UN)
        source_lock.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--complete-subset-manifest",
        type=Path,
        help=(
            "Build without network calls from a COMPLETE exact-contract subset; "
            "only source operational-failure keys are selected."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=_positive_int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument(
        "--campaign-workers", type=_positive_int, default=DEFAULT_CAMPAIGN_WORKERS
    )
    parser.add_argument(
        "--participant-workers", type=_positive_int, default=DEFAULT_PARTICIPANT_WORKERS
    )
    parser.add_argument("--max-attempts", type=_positive_int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--initial-backoff-seconds",
        type=_nonnegative_float,
        default=DEFAULT_BACKOFF_SECONDS,
    )
    parser.add_argument(
        "--timeout-seconds", type=_nonnegative_float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.complete_subset_manifest is not None:
            if args.resume:
                raise Part2SensitivityOperationalRepairError(
                    "Complete-subset overlay construction never overwrites or resumes."
                )
            manifest = build_overlay_from_complete_subset(
                source_manifest_path=args.source_manifest,
                subset_manifest_path=args.complete_subset_manifest,
                output_dir=args.output_dir,
            )
        else:
            manifest = run_repair(
                source_manifest_path=args.source_manifest,
                output_dir=args.output_dir,
                client=_deadline_client(args.timeout_seconds),
                max_rounds=args.max_rounds,
                campaign_workers=args.campaign_workers,
                participant_workers=args.participant_workers,
                max_attempts=args.max_attempts,
                initial_backoff_seconds=args.initial_backoff_seconds,
                resume=args.resume,
            )
    except (
        Part2SensitivityOperationalRepairError,
        runner.InferenceHubPart2SensitivityError,
        InferenceHubDiscoveryError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(f"Part 2 sensitivity operational repair failed: {error}")
        return 2
    print(
        "Operational sensitivity trajectories unresolved: "
        f"{manifest['summary']['unresolved_operational_failure_trajectories']}"
    )
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
