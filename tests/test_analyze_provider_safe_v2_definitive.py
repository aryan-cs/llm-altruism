from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import pytest

from analysis.analyze_provider_safe_v2_definitive import (
    DefinitiveAnalysisError,
    SENSITIVITY_DIAGNOSTIC_FAMILY,
    SENSITIVITY_HOLM_FAMILY,
    SENSITIVITY_HOLM_FAMILY_SIZE,
    _analyze_deadline_sensitivity,
    _provider_safe_contract,
    _mean_t_95,
    _root_cluster_bootstrap_95,
    _self_hash,
    _stratified_root_bootstrap_95,
    _validate_role_journals,
    _wilson_95,
    analyze,
)
from experiments.misc.inference_hub_part2_sensitivity_v1 import load_sensitivity_design
from experiments.part1.confirmatory_design import COUNTERBALANCES, DOMAINS, GAMES


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "experiments/part2/part2_sensitivity_deadline_exploratory_v2.json"
JUDGE = {
    "target_id": "judge.nemotron", "route": "judge/nemotron",
    "upstream_provider": "judge-provider", "model": "nemotron-judge",
    "dispatch_permitted_in_this_runner": False,
}


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = _self_hash(value)
    return value


def _route(index: int, prefix: str) -> dict[str, Any]:
    return {
        "target_id": f"{prefix}/model-{index:03d}",
        "route": f"route/{prefix}/model-{index:03d}",
        "upstream_provider": f"provider-{index:03d}",
        "model": f"model-{index:03d}",
    }


def _write_json(path: Path, value: Mapping[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    if private:
        os.chmod(path, 0o600)


def _journal(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    encoded = []
    for payload in rows:
        row = {**payload, "previous_record_sha256": previous}
        row["record_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        previous = row["record_sha256"]
        encoded.append(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    path.write_text("".join(encoded), encoding="utf-8")
    os.chmod(path, 0o600)
    return {
        "path": str(path.resolve()), "record_count": len(rows),
        "tail_record_sha256": previous, "file_sha256": _sha_file(path),
    }


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _role_overlay_fixture(
    run: Path, *, replace_visible_semantic_invalid: bool = False
) -> tuple[dict[str, Any], Path, Path]:
    target = "role/model"
    route = "route/role/model"
    subject = {
        "target_id": target,
        "route": route,
        "upstream_provider": "provider",
        "model": "model",
    }
    visible_raw = {"model": route, "choices": [{"message": {"content": "OPTION_X"}}]}
    original_rows = [
        {
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_role_calibration_raw_response_v1",
            "target_id": target,
            "requested_route": route,
            "trial_id": "visible-invalid",
            "root_id": "root-visible",
            "frame_id": "advice",
            "generation_block": "block-0",
            "counterbalance_id": "cb-0",
            "prompt_sha256": "a" * 64,
            "request_sha256": "b" * 64,
            "raw_response": visible_raw,
            "raw_response_sha256": _canonical_sha(visible_raw),
            "model_identity_valid": True,
            "format_valid": False,
            "welfare_preserving": None,
        },
        {
            "schema_version": 1,
            "artifact_type": "inference_hub_part1_role_calibration_raw_response_v1",
            "target_id": target,
            "requested_route": route,
            "trial_id": "transport-null",
            "root_id": "root-null",
            "frame_id": "prediction",
            "generation_block": "block-0",
            "counterbalance_id": "cb-1",
            "prompt_sha256": "c" * 64,
            "request_sha256": "d" * 64,
            "raw_response": None,
            "raw_response_sha256": None,
            "model_identity_valid": False,
            "format_valid": False,
            "welfare_preserving": None,
        },
    ]
    original_path = run / "private/raw/role.jsonl"
    original_ref = _journal(original_path, original_rows)
    retained = [json.loads(line) for line in original_path.read_text().splitlines()]
    selected = retained[0 if replace_visible_semantic_invalid else 1]

    repaired_raw = {
        "model": route,
        "choices": [{"message": {"content": "OPTION_A"}}],
    }
    attempt_id = "repair-attempt-1"
    repair_binding = {
        "repair_reason": "original_retained_raw_response_null",
        "replaces_original_record_sha256": selected["record_sha256"],
    }
    repair_row = {
        "schema_version": 1,
        "artifact_type": (
            "inference_hub_part1_role_calibration_operational_repair_response_v1"
        ),
        "target_id": target,
        "requested_route": route,
        "response_model": route,
        "model_identity_valid": True,
        "trial_id": selected["trial_id"],
        "root_id": selected["root_id"],
        "frame_id": selected["frame_id"],
        "generation_block": selected["generation_block"],
        "counterbalance_id": selected["counterbalance_id"],
        "prompt_sha256": selected["prompt_sha256"],
        "request_sha256": selected["request_sha256"],
        "attempt_id": attempt_id,
        **repair_binding,
        "raw_response": repaired_raw,
        "raw_response_sha256": _canonical_sha(repaired_raw),
        "response_text_sha256": "e" * 64,
        "format_valid": True,
        "welfare_preserving": True,
    }
    repair_path = run / "private/operational_repairs/raw/role.jsonl"
    repair_ref = _journal(repair_path, [repair_row])
    reservation = {
        "schema_version": 1,
        "artifact_type": (
            "inference_hub_part1_role_calibration_operational_repair_attempt_v1"
        ),
        "event": "reserved_before_dispatch",
        "attempt_id": attempt_id,
        "target_id": target,
        "trial_id": selected["trial_id"],
        "request_sha256": selected["request_sha256"],
        **repair_binding,
    }
    completion = {
        "schema_version": 1,
        "artifact_type": (
            "inference_hub_part1_role_calibration_operational_repair_attempt_v1"
        ),
        "event": "attempt_completed",
        "attempt_id": attempt_id,
        "outcome": "response_retained",
        "response_payload_sha256": repair_row["raw_response_sha256"],
        "response_text_sha256": repair_row["response_text_sha256"],
        "response_model": route,
        **repair_binding,
    }
    repair_ledger = _journal(
        run / "private/operational_repairs/attempt_ledger.jsonl",
        [reservation, completion],
    )
    original_ledger = _journal(run / "private/attempts.jsonl", [])
    repair_source = ROOT / "experiments/misc/inference_hub_part1_role_calibration_v1.py"
    manifest = {
        "subject_routes": [subject],
        "summary": {
            "operational_repair_eligible_originals": 1,
            "operational_repairs_succeeded": 1,
            "operational_repairs_unresolved": 0,
            "failed_without_response": 0,
        },
        "journals": {
            "attempt_ledger": original_ledger,
            "raw_responses": {target: original_ref},
            "operational_repair_overlay": {
                "policy": (
                    "overlay_only_original_retained_raw_response_null_"
                    "no_semantic_retry_v1"
                ),
                "implementation_source_artifacts": {
                    str(repair_source.resolve()): _sha_file(repair_source)
                },
                "attempt_ledger": repair_ledger,
                "raw_responses": {target: repair_ref},
            },
        },
    }
    return manifest, original_path, repair_path


def _base_manifest(kind: str, subjects: list[dict[str, Any]]) -> dict[str, Any]:
    shared = {
        "schema_version": 2,
        "algorithm": "cross_process_provider_aware_leaky_bucket_with_leases_all_http_5xx_full_throttle_cooldown",
        "global_concurrency": 16, "provider_concurrency": 1,
        "global_requests_per_second": 8.0, "provider_requests_per_second": 1.0,
        "lease_seconds": 900.0, "poll_seconds": 0.05,
        "throttle_cooldown_seconds": 30.0, "transient_cooldown_seconds": 5.0,
    }
    shared["policy_sha256"] = hashlib.sha256(
        json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": 1, "artifact_type": kind, "complete": True,
        "completed_at_utc": "2026-08-03T12:00:00Z", "subject_routes": subjects,
        "judge_reservation": dict(JUDGE),
        "execution_contract": {"shared_rate_limit": shared},
        "source_artifacts": {str(ROOT / "experiments/misc/inference_hub_provider_safe_v2.py"): "a" * 64},
    }


def _artifact(run: Path, name: str, artifact_type: str, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    payload = _seal({"schema_version": 1, "artifact_type": artifact_type, "rows": rows})
    path = run / "sanitized" / f"{name}.json"
    _write_json(path, payload)
    manifest.setdefault("sanitized_artifacts", {})[name] = {
        "path": str(path.resolve()), "file_sha256": _sha_file(path),
        "evidence_sha256": payload["evidence_sha256"],
    }


@pytest.fixture(scope="module")
def production_bundle(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("provider-safe-v2-definitive")
    outputs: dict[str, Path] = {}

    # Part 0: the exact production 22 x (48 roots x 3 languages) shape.
    run = root / "part0"
    subjects = [_route(index, "p0") for index in range(22)]
    manifest = _base_manifest("inference_hub_part0_accelerated_private_panel", subjects)
    manifest["judge"] = manifest.pop("judge_reservation")
    manifest["summary"] = {"planned_units": 22 * 144}
    raw = {}
    for subject in subjects:
        rows = []
        for root_index in range(48):
            for language_index, language in enumerate(("english", "chinese", "russian")):
                unit = root_index * 3 + language_index
                outcome = "INVALID" if unit == 0 else ("REFUSAL" if unit % 2 else "COMPLIANCE")
                rows.append({
                    "event": "unit_completed", "target_id": subject["target_id"],
                    "trial_id": f"trial-{root_index}-{language}",
                    "root_id": f"root-{root_index}", "language": language,
                    "outcome": outcome,
                })
        raw[subject["target_id"]] = _journal(run / "private/raw" / f"{subject['model']}.jsonl", rows)
    manifest["journals"] = {"attempt_ledger": _journal(run / "private/attempts.jsonl", []), "raw_responses": raw}
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["part0"] = path

    # Part 1: the exact production 75 x 384 matrix.
    run = root / "part1"
    subjects = [_route(index, "p1") for index in range(75)]
    manifest = _base_manifest("inference_hub_part1_large_n_exploratory_panel", subjects)
    manifest.update({"trial_limit": 384, "judge_dispatched": False})
    raw = {}
    for subject in subjects:
        rows = []
        unit = 0
        for game in GAMES:
            for domain in DOMAINS:
                for root_index in range(32):
                    counterbalance = COUNTERBALANCES[root_index % len(COUNTERBALANCES)]
                    rows.append({
                        "target_id": subject["target_id"], "trial_id": f"trial-{unit}",
                        "root_id": f"{game}-{domain}-{root_index}",
                        "counterbalance_id": counterbalance.counterbalance_id,
                        "game": game, "domain": domain,
                        "format_valid": unit != 0,
                        "parsed_action": counterbalance.welfare_preserving_label,
                    })
                    unit += 1
        raw[subject["target_id"]] = _journal(run / "private/raw" / f"{subject['model']}.jsonl", rows)
    manifest["journals"] = {"attempt_ledger": _journal(run / "private/attempts.jsonl", []), "raw_responses": raw}
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["part1"] = path

    # Part 2: the exact production 19 x 12 matched trajectories.
    run = root / "part2"
    subjects = [_route(index, "p2") for index in range(19)]
    manifest = _base_manifest("inference_hub_part2_corrected_matched_panel", subjects)
    refs, trajectory_rows = {}, []
    for subject in subjects:
        for index in range(12):
            key = f"{subject['target_id']}::{index}"
            refs[key] = _journal(run / "private/trajectories" / f"{subject['model']}-{index}.jsonl", [])
            trajectory_rows.append({
                "target_id": subject["target_id"], "trajectory_index": index,
                "scheduled_agent_days": 10, "responses_received": 10,
                "invalid_count": 1 if index == 0 else 0,
                "identity_mismatch_count": 0,
                "transport_failure_count": 0, "restraint_count": 7,
                "overuse_count": 2 if index == 0 else 3,
                "operationally_eligible": True,
                "aurc": 0.8, "aupc": 0.9,
                "reserve_nondepletion": True,
                "population_retention": 0.8,
            })
    manifest["journals"] = refs
    _artifact(run, "trajectory_metrics", "inference_hub_part2_sanitized_trajectory_metrics", trajectory_rows, manifest)
    _artifact(run, "model_metrics", "inference_hub_part2_sanitized_model_metrics", [{"target_id": row["target_id"]} for row in subjects], manifest)
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["part2"] = path

    # Role calibration: exact 6 x 96 roots x 3 frames x 4 blocks.
    run = root / "role"
    subjects = [_route(index, "role") for index in range(6)]
    manifest = _base_manifest("inference_hub_part1_role_calibration_private_v1", subjects)
    manifest.update({"frames": ["advice", "observer_evaluation", "prediction"], "trials_per_subject": 1152})
    raw = {}
    for subject in subjects:
        rows = []
        for frame in manifest["frames"]:
            for root_index in range(96):
                for block in range(4):
                    rows.append({
                        "target_id": subject["target_id"], "trial_id": f"{frame}-{root_index}-{block}",
                        "frame_id": frame, "format_valid": not (root_index == 0 and block == 0),
                        "welfare_preserving": True,
                    })
        raw[subject["target_id"]] = _journal(run / "private/raw" / f"{subject['model']}.jsonl", rows)
    manifest["journals"] = {"attempt_ledger": _journal(run / "private/attempts.jsonl", []), "raw_responses": raw}
    role_summary = _seal({"schema_version": 1, "artifact_type": "part1_role_calibration_sanitized_summary_v1", "estimates": []})
    _write_json(run / "sanitized/summary.json", role_summary)
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["role"] = path

    # Sensitivity: five exact compatible sentinels and one real Holm-25 family.
    run = root / "sensitivity"
    subjects = [_route(index, "sens") for index in range(5)]
    manifest = _base_manifest("inference_hub_part2_sensitivity_campaign_v1", subjects)
    manifest["selected_subject_routes"] = manifest.pop("subject_routes")
    manifest["execution_contract"]["provider_concurrency_required"] = 1
    manifest["source_artifacts"][str(DESIGN_PATH)] = _sha_file(DESIGN_PATH)
    design, cells = load_sensitivity_design(DESIGN_PATH)
    refs, trajectory_rows = {}, []
    for subject_index, subject in enumerate(subjects):
        for cell in cells:
            for seed in range(int(design["seeds_per_cell"])):
                key = f"{cell.cell_id}::{subject['target_id']}::{seed}"
                refs[key] = _journal(run / "private/trajectories" / f"{subject_index}-{cell.cell_id}-{seed}.jsonl", [])
                coded_signal = sum(cell.coded_levels.values()) * 0.01
                scheduled = cell.society_size * cell.horizon_days
                invalid = 1 if seed == 0 else 0
                trajectory_rows.append({
                    "cell_id": cell.cell_id, "target_id": subject["target_id"],
                    "trajectory_index": seed, "environment_seed_index": seed,
                    "environment_seed": seed,
                    "normalized_aurc": 0.5 + coded_signal + seed * 0.0001,
                    "society_size": cell.society_size,
                    "horizon_days": cell.horizon_days,
                    "scheduled_agent_days": scheduled,
                    "responses_received": scheduled,
                    "invalid_count": invalid,
                    "identity_mismatch_count": 0,
                    "transport_failure_count": 0,
                    "restraint_count": scheduled - invalid,
                    "overuse_count": 0,
                    "operationally_eligible": True,
                })
    manifest["journals"] = refs
    manifest["attempt_ledger"] = _journal(run / "private/attempts.jsonl", [])
    effects = _analyze_deadline_sensitivity(
        trajectory_rows,
        sentinel_ids=[row["target_id"] for row in subjects],
        design=design,
    )
    _artifact(run, "trajectory_metrics", "part2_sensitivity_trajectory_metrics_v1", trajectory_rows, manifest)
    _artifact(run, "sentinel_cell_metrics", "part2_sensitivity_sentinel_cell_metrics_v1", [], manifest)
    effect_payload = _seal({
        "schema_version": 1, "artifact_type": "part2_sensitivity_main_effects_v1",
        "analysis_status": "complete_deadline_exploratory", "confirmatory": False,
        "inference_scope": "deadline_exploratory",
        "global_holm_family": SENSITIVITY_HOLM_FAMILY,
        "global_holm_family_size": SENSITIVITY_HOLM_FAMILY_SIZE,
        "rows": effects,
    })
    effect_path = run / "sanitized/main_effects.json"
    _write_json(effect_path, effect_payload)
    manifest["sanitized_artifacts"]["main_effects"] = {"path": str(effect_path.resolve()), "file_sha256": _sha_file(effect_path), "evidence_sha256": effect_payload["evidence_sha256"]}
    diagnostic = _seal({"schema_version": 1, "artifact_type": "part2_sensitivity_call_order_diagnostic_v1", "analysis_family": SENSITIVITY_DIAGNOSTIC_FAMILY, "rows": []})
    diagnostic_path = run / "sanitized/call_order_diagnostic.json"
    _write_json(diagnostic_path, diagnostic)
    manifest["sanitized_artifacts"]["call_order_diagnostic"] = {"path": str(diagnostic_path.resolve()), "file_sha256": _sha_file(diagnostic_path), "evidence_sha256": diagnostic["evidence_sha256"]}
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    outputs["sensitivity"] = path
    return outputs


def _run(bundle: Mapping[str, Path], output: Path) -> dict[str, Any]:
    return analyze(
        part0=bundle["part0"], part1=bundle["part1"], part2=bundle["part2"],
        role_calibration=bundle["role"], sensitivity=bundle["sensitivity"],
        output_dir=output,
    )


def _terminalized_part0_run(root: Path, *, drift_request_hash: bool = False) -> Path:
    run = root / "terminalized-part0"
    subjects = [_route(index, "p0") for index in range(22)]
    manifest = _base_manifest(
        "inference_hub_part0_accelerated_private_panel", subjects
    )
    manifest["judge"] = manifest.pop("judge_reservation")
    manifest["complete"] = False
    manifest.pop("completed_at_utc", None)
    manifest["schedule"] = [
        {
            "trial_id": f"trial-{root_index}-{language}",
            "root_id": f"root-{root_index}",
            "language": language,
        }
        for root_index in range(48)
        for language in ("english", "chinese", "russian")
    ]
    safe = ROOT / "experiments/misc/inference_hub_provider_safe_v2.py"
    deadline = ROOT / "experiments/misc/inference_hub_part0_deadline_retry.py"
    manifest["source_artifacts"] = {
        str(safe): _sha_file(safe),
        str(deadline): _sha_file(deadline),
    }
    shared = {
        "schema_version": 2,
        "algorithm": "cross_process_provider_aware_leaky_bucket_with_leases_all_http_5xx_full_throttle_cooldown",
        "global_concurrency": 16,
        "provider_concurrency": 3,
        "global_requests_per_second": 10.0,
        "provider_requests_per_second": 2.0,
        "lease_seconds": 900.0,
        "poll_seconds": 0.05,
        "throttle_cooldown_seconds": 30.0,
        "transient_cooldown_seconds": 5.0,
    }
    shared["policy_sha256"] = hashlib.sha256(
        json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["execution_contract"] = {
        "shared_rate_limit": shared,
        "max_attempts_per_request": 8,
    }

    failure_counts = (3, 4, 9, 3, 2, 9, 6, 5)
    raw_refs = {}
    ledger_rows = []
    outcomes = {"REFUSAL": 0, "COMPLIANCE": 0, "UNCLEAR": 0, "INVALID": 0}
    for subject_index, subject in enumerate(subjects):
        failed_count = failure_counts[subject_index] if subject_index < 8 else 0
        rows = []
        visible_ids = []
        for unit, scheduled in enumerate(manifest["schedule"]):
            trial_id = scheduled["trial_id"]
            if unit < failed_count:
                for attempt_number in range(1, 9):
                    attempt_id = f"attempt-{subject_index}-{unit}-{attempt_number}"
                    request_hash = (
                        f"{'b' if drift_request_hash and subject_index == 0 and unit == 0 and attempt_number == 8 else 'a'}"
                        * 64
                    )
                    ledger_rows.extend(
                        [
                            {
                                "event": "reserved_before_dispatch",
                                "attempt_id": attempt_id,
                                "attempt_number": attempt_number,
                                "request_sha256": request_hash,
                                "role": "subject",
                                "route": subject["route"],
                                "upstream_provider": subject["upstream_provider"],
                                "target_id": subject["target_id"],
                                "work_id": trial_id,
                            },
                            {
                                "event": "attempt_completed",
                                "attempt_id": attempt_id,
                                "outcome": "failed",
                                "transient": True,
                                "failure_code": "http_400_periodic_retry",
                                "http_status": 400,
                            },
                        ]
                    )
                rows.append(
                    {
                        "event": "unit_completed",
                        "target_id": subject["target_id"],
                        "trial_id": trial_id,
                        "root_id": scheduled["root_id"],
                        "language": scheduled["language"],
                        "outcome": "INVALID",
                        "judge_label": None,
                        "judge_dispatched": False,
                        "operational_failure": {
                            "failure_code": "http_400_periodic_retry",
                            "http_status": 400,
                        },
                    }
                )
                outcomes["INVALID"] += 1
                continue
            visible_ids.append(trial_id)
            rows.append(
                {
                    "event": "subject_response_retained",
                    "target_id": subject["target_id"],
                    "trial_id": trial_id,
                    "model_identity_valid": True,
                }
            )
            outcome = "REFUSAL" if unit % 2 else "COMPLIANCE"
            rows.append(
                {
                    "event": "unit_completed",
                    "target_id": subject["target_id"],
                    "trial_id": trial_id,
                    "root_id": scheduled["root_id"],
                    "language": scheduled["language"],
                    "outcome": outcome,
                }
            )
            outcomes[outcome] += 1
        for start in range(0, len(visible_ids), 8):
            rows.append(
                {
                    "event": "judge_batch_retained",
                    "target_id": subject["target_id"],
                    "trial_ids": visible_ids[start : start + 8],
                    "model_identity_valid": True,
                }
            )
        raw_refs[subject["target_id"]] = _journal(
            run / "private/raw" / f"{subject_index}.jsonl", rows
        )
    manifest["journals"] = {
        "attempt_ledger": _journal(run / "private/attempts.jsonl", ledger_rows),
        "raw_responses": raw_refs,
    }
    manifest["summary"] = {
        "planned_units": 22 * 144,
        "retained_terminal_units": 22 * 144,
        "outcomes": outcomes,
        "subject_transport_failures": 41,
        "judge_failed_units": 0,
        "subject_model_identity_mismatches": 0,
        "judge_model_identity_mismatches": 0,
        "operationally_retired_units": 0,
    }
    (run / "private/.run.lock").touch()
    path = run / "private/manifest.json"
    _write_json(path, _seal(manifest), private=True)
    return path


def test_uncertainty_helpers_are_deterministic_and_use_independent_units() -> None:
    clusters = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] * 12
    first = _root_cluster_bootstrap_95(
        clusters, namespace="part0-overall-root-cluster", target_id="route/exact"
    )
    second = _root_cluster_bootstrap_95(
        clusters, namespace="part0-overall-root-cluster", target_id="route/exact"
    )
    assert first == second
    assert first[0] < 0.5 < first[1]

    strata = [[float((root + stratum) % 3 == 0) for root in range(32)] for stratum in range(12)]
    stratified_first = _stratified_root_bootstrap_95(strata, target_id="route/exact")
    stratified_second = _stratified_root_bootstrap_95(strata, target_id="route/exact")
    assert stratified_first == stratified_second
    assert stratified_first[0] < sum(map(sum, strata)) / 384 < stratified_first[1]

    zero_low, zero_high = _wilson_95(0, 48)
    full_low, full_high = _wilson_95(48, 48)
    assert zero_low == 0.0 < zero_high < 1.0
    assert 0.0 < full_low < full_high == 1.0
    bounded_low, bounded_high = _mean_t_95(
        [1.0, 1.0, 1.0, 0.95], bounds=(0.0, 1.0)
    )
    assert bounded_low is not None and 0.0 <= bounded_low <= 1.0
    assert bounded_high == 1.0


def test_full_production_shaped_analysis_and_invalid_denominators(
    production_bundle: Mapping[str, Path], tmp_path: Path,
) -> None:
    result = _run(production_bundle, tmp_path / "out")

    assert result["row_counts"] == {
        "part0_models": 22, "part1_models": 75, "part2_models": 19,
        "role_calibration_model_frames": 18, "sensitivity_models": 5,
        "sensitivity_main_effects": 25,
    }
    assert result["human_labels_generated"] is False
    assert result["confirmatory_or_paper_promotion_permitted"] is False
    p0 = json.loads((tmp_path / "out/part0_models.jsonl").read_text().splitlines()[0])
    assert p0["scheduled_units"] == 144
    assert p0["first_attempt_invalid_count"] == 1
    assert p0["repaired_invalid_count"] == 0
    assert p0["refusal_rate_all_scheduled"] == p0["refusal_count"] / 144
    assert p0["finite_bank_sensitivity_replicates"] == 5_000
    assert p0["refusal_rate_all_scheduled_finite_bank_sensitivity_low"] <= p0["refusal_rate_all_scheduled"] <= p0["refusal_rate_all_scheduled_finite_bank_sensitivity_high"]
    p1 = json.loads((tmp_path / "out/part1_models.jsonl").read_text().splitlines()[0])
    assert p1["welfare_preserving_rate_all_scheduled"] == 383 / 384
    assert p1["finite_bank_sensitivity_strata"] == 12
    assert p1["finite_bank_sensitivity_replicates"] == 5_000
    p2 = json.loads((tmp_path / "out/part2_models.jsonl").read_text().splitlines()[0])
    assert p2["restraint_rate_all_scheduled"] == 84 / 120
    assert p2["restraint_rate_among_valid"] == 84 / 119
    assert p2["operationally_eligible_trajectory_count"] == 12
    assert p2["environmentally_estimable_trajectory_count"] == 11
    assert p2["semantic_invalid_trajectory_count"] == 1
    assert p2["mean_aurc_eligible"] == pytest.approx(0.8)
    assert p2["mean_aupc_eligible"] == pytest.approx(0.9)
    assert p2["reserve_nondepletion_rate_eligible"] == 1.0
    assert p2["mean_population_retention_eligible"] == pytest.approx(0.8)
    assert p2["mean_aurc_eligible_t95_low"] == pytest.approx(0.8)
    assert p2["mean_aurc_eligible_t95_high"] == pytest.approx(0.8)
    assert p2["reserve_nondepletion_rate_eligible_wilson95_low"] < 1.0
    assert p2["reserve_nondepletion_rate_eligible_wilson95_high"] == 1.0
    assert p2["restraint_interval_trajectory_count"] == 12

    manifest = json.loads((tmp_path / "out/analysis_manifest.json").read_text())
    assert manifest["path_policy"] == "portable_basenames_only_no_host_absolute_paths_in_public_manifest"
    assert all("path" not in binding for binding in manifest["input_manifests"].values())
    assert all(not Path(binding["basename"]).is_absolute() for binding in manifest["input_manifests"].values())
    assert len(manifest["public_outputs"]) == 13
    for binding in manifest["public_outputs"]:
        public_path = tmp_path / "out" / binding["basename"]
        assert binding["file_sha256"] == _sha_file(public_path)
        assert binding["kind"].startswith("machine_readable_")
    serialized_manifest = json.dumps(manifest, sort_keys=True)
    assert "/Users/" not in serialized_manifest
    assert "/private/" not in serialized_manifest


def test_incomplete_manifest_fails_before_output(tmp_path: Path) -> None:
    incomplete = ROOT / "data/private/inference_hub/provider-safe-v2-part0-large-n48-main22-v1/private/manifest.json"
    if not incomplete.exists():
        pytest.skip("Private running campaign is not available.")
    output = tmp_path / "must-not-exist"
    with pytest.raises(DefinitiveAnalysisError, match="not COMPLETE"):
        analyze(part0=incomplete, part1=incomplete, part2=incomplete, role_calibration=incomplete, sensitivity=incomplete, output_dir=output)
    assert not output.exists()


def test_role_operational_overlay_replaces_only_transport_nulls(tmp_path: Path) -> None:
    run = tmp_path / "role-overlay"
    manifest, original_path, _ = _role_overlay_fixture(run)
    original_sha = _sha_file(original_path)

    effective = _validate_role_journals(run, manifest)
    rows = effective["role/model"]
    assert len(rows) == 2
    assert rows[0]["trial_id"] == "visible-invalid"
    assert rows[0]["raw_response"] is not None
    assert rows[0]["format_valid"] is False
    assert rows[1]["trial_id"] == "transport-null"
    assert rows[1]["raw_response"] is not None
    assert rows[1]["format_valid"] is True
    assert rows[1]["welfare_preserving"] is True
    assert _sha_file(original_path) == original_sha


def test_role_operational_overlay_rejects_visible_semantic_replacement(
    tmp_path: Path,
) -> None:
    run = tmp_path / "role-overlay-semantic"
    manifest, _, _ = _role_overlay_fixture(
        run, replace_visible_semantic_invalid=True
    )
    with pytest.raises(
        DefinitiveAnalysisError, match="operational-repair binding failed"
    ):
        _validate_role_journals(run, manifest)


def test_terminalized_part0_policy_is_explicit_ledger_bound_and_all_scheduled(
    production_bundle: Mapping[str, Path], tmp_path: Path,
) -> None:
    terminalized = _terminalized_part0_run(tmp_path / "valid")
    changed = dict(production_bundle)
    changed["part0"] = terminalized
    with pytest.raises(DefinitiveAnalysisError, match="not COMPLETE"):
        _run(changed, tmp_path / "strict-out")

    result = analyze(
        part0=terminalized,
        part1=changed["part1"],
        part2=changed["part2"],
        role_calibration=changed["role"],
        sensitivity=changed["sensitivity"],
        output_dir=tmp_path / "terminalized-out",
        allow_terminalized_part0_operational_invalids=True,
    )
    audit = result["part0_terminalized_operational_audit"]
    assert audit["scheduled_units"] == 3_168
    assert audit["operational_failure_units"] == 41
    assert audit["subject_transport_failure_units"] == 41
    assert audit["judge_transport_failure_units"] == 0
    assert audit["visible_subject_response_units"] == 3_127
    assert audit["affected_target_count"] == 8
    assert audit["semantics"] == (
        "operational_nonsuccess_not_refusal_not_semantic_model_output"
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "terminalized-out/part0_models.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 22
    assert sum(row["scheduled_units"] for row in rows) == 3_168
    assert sum(row["operational_invalid_count"] for row in rows) == 41
    assert sum(row["visible_subject_response_count"] for row in rows) == 3_127

    drifted = _terminalized_part0_run(
        tmp_path / "drifted", drift_request_hash=True
    )
    with pytest.raises(DefinitiveAnalysisError, match="retry lineage changed"):
        analyze(
            part0=drifted,
            part1=changed["part1"],
            part2=changed["part2"],
            role_calibration=changed["role"],
            sensitivity=changed["sensitivity"],
            output_dir=tmp_path / "drifted-out",
            allow_terminalized_part0_operational_invalids=True,
        )


def test_hash_tamper_and_judge_overlap_fail_closed(
    production_bundle: Mapping[str, Path], tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered.json"
    shutil.copyfile(production_bundle["part0"], tampered)
    os.chmod(tampered, 0o600)
    value = json.loads(tampered.read_text())
    value["summary"]["planned_units"] += 1
    _write_json(tampered, value, private=True)
    changed = dict(production_bundle)
    changed["part0"] = tampered
    with pytest.raises(DefinitiveAnalysisError, match="self-hash failed"):
        _run(changed, tmp_path / "hash-out")
    assert not (tmp_path / "hash-out").exists()

    overlap_run = tmp_path / "overlap"
    shutil.copytree(production_bundle["part1"].parents[1], overlap_run)
    overlap_manifest = overlap_run / "private/manifest.json"
    value = json.loads(overlap_manifest.read_text())
    value["judge_reservation"]["target_id"] = value["subject_routes"][0]["target_id"]
    _write_json(overlap_manifest, _seal(value), private=True)
    changed = dict(production_bundle)
    changed["part1"] = overlap_manifest
    with pytest.raises(DefinitiveAnalysisError, match="judge overlaps"):
        _run(changed, tmp_path / "overlap-out")
    assert not (tmp_path / "overlap-out").exists()


def test_campaign_specific_accelerated_launcher_and_policy_bindings() -> None:
    safe = ROOT / "experiments/misc/inference_hub_provider_safe_v2.py"
    main = ROOT / "experiments/misc/inference_hub_main_accelerated.py"
    deadline = ROOT / "experiments/misc/inference_hub_part1_deadline_accelerated.py"
    part0_deadline = ROOT / "experiments/misc/inference_hub_part0_deadline_retry.py"
    sensitivity_deadline = (
        ROOT / "experiments/misc/inference_hub_sensitivity_deadline_accelerated.py"
    )
    exploratory = ROOT / "experiments/misc/inference_hub_exploratory_accelerated.py"

    def manifest(
        launcher: Path,
        concurrency: int,
        rps: float,
        *,
        global_concurrency: int = 12,
        global_rps: float = 8.0,
    ) -> dict[str, Any]:
        shared = {
            "schema_version": 2,
            "algorithm": "cross_process_provider_aware_leaky_bucket_with_leases_all_http_5xx_full_throttle_cooldown",
            "global_concurrency": global_concurrency, "provider_concurrency": concurrency,
            "global_requests_per_second": global_rps, "provider_requests_per_second": rps,
            "lease_seconds": 900.0, "poll_seconds": 0.05,
            "throttle_cooldown_seconds": 30.0, "transient_cooldown_seconds": 5.0,
        }
        shared["policy_sha256"] = hashlib.sha256(
            json.dumps(shared, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "source_artifacts": {str(safe): _sha_file(safe), str(launcher): _sha_file(launcher)},
            "execution_contract": {"shared_rate_limit": shared},
        }

    main_manifest = manifest(main, 2, 1.5)
    exploratory_manifest = manifest(exploratory, 3, 2.0)
    deadline_manifest = manifest(
        deadline, 4, 2.5, global_concurrency=24, global_rps=12.0
    )
    part0_deadline_manifest = manifest(
        part0_deadline, 3, 2.0, global_concurrency=16, global_rps=10.0
    )
    sensitivity_deadline_manifest = manifest(
        sensitivity_deadline, 3, 2.5, global_concurrency=24, global_rps=12.0
    )
    _provider_safe_contract(main_manifest, "part0")
    _provider_safe_contract(exploratory_manifest, "role")
    _provider_safe_contract(deadline_manifest, "part1")
    _provider_safe_contract(part0_deadline_manifest, "part0")
    _provider_safe_contract(sensitivity_deadline_manifest, "sensitivity")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(main_manifest, "sensitivity")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(deadline_manifest, "part0")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(part0_deadline_manifest, "part1")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(sensitivity_deadline_manifest, "role")
    exploratory_manifest["execution_contract"]["shared_rate_limit"]["provider_requests_per_second"] = 2.1
    with pytest.raises(DefinitiveAnalysisError, match="policy hash failed"):
        _provider_safe_contract(exploratory_manifest, "sensitivity")
