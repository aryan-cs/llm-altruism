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
    _provider_safe_contract,
    _self_hash,
    analyze,
)
from experiments.misc.inference_hub_part2_sensitivity_v1 import (
    _analyze_completed_design,
    load_sensitivity_design,
)
from experiments.part1.confirmatory_design import COUNTERBALANCES


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "experiments/part2/part2_sensitivity_deadline_exploratory_v1.json"
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
        for unit in range(144):
            outcome = "INVALID" if unit == 0 else ("REFUSAL" if unit % 2 else "COMPLIANCE")
            rows.append({"event": "unit_completed", "target_id": subject["target_id"], "trial_id": f"trial-{unit}", "language": ("english", "chinese", "russian")[unit % 3], "outcome": outcome})
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
        for unit in range(384):
            counterbalance = COUNTERBALANCES[unit % len(COUNTERBALANCES)]
            rows.append({
                "target_id": subject["target_id"], "trial_id": f"trial-{unit}",
                "counterbalance_id": counterbalance.counterbalance_id,
                "game": ("prisoners_dilemma", "stag_hunt")[unit % 2],
                "domain": ("education", "health")[unit % 2],
                "format_valid": unit != 0, "parsed_action": counterbalance.welfare_preserving_label,
            })
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
                "invalid_count": 1, "identity_mismatch_count": 0,
                "transport_failure_count": 0, "restraint_count": 7,
                "overuse_count": 2, "operationally_eligible": True,
                "aurc": 0.8, "aupc": 0.9,
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

    # Sensitivity: exact frozen deadline design and real Holm-30 analysis.
    run = root / "sensitivity"
    subjects = [_route(index, "sens") for index in range(6)]
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
                trajectory_rows.append({
                    "cell_id": cell.cell_id, "target_id": subject["target_id"],
                    "trajectory_index": seed, "environment_seed": seed,
                    "normalized_aurc": 0.5 + coded_signal + seed * 0.0001,
                    "scheduled_agent_days": cell.society_size * cell.horizon_days,
                    "invalid_count": 1 if seed == 0 else 0,
                    "operationally_eligible": True,
                })
    manifest["journals"] = refs
    manifest["attempt_ledger"] = _journal(run / "private/attempts.jsonl", [])
    effects = _analyze_completed_design(trajectory_rows, sentinel_ids=[row["target_id"] for row in subjects], design=design)
    _artifact(run, "trajectory_metrics", "part2_sensitivity_trajectory_metrics_v1", trajectory_rows, manifest)
    _artifact(run, "sentinel_cell_metrics", "part2_sensitivity_sentinel_cell_metrics_v1", [], manifest)
    effect_payload = _seal({
        "schema_version": 1, "artifact_type": "part2_sensitivity_main_effects_v1",
        "analysis_status": "complete_deadline_exploratory", "confirmatory": False,
        "inference_scope": "deadline_exploratory", "global_holm_family_size": 30,
        "rows": effects,
    })
    effect_path = run / "sanitized/main_effects.json"
    _write_json(effect_path, effect_payload)
    manifest["sanitized_artifacts"]["main_effects"] = {"path": str(effect_path.resolve()), "file_sha256": _sha_file(effect_path), "evidence_sha256": effect_payload["evidence_sha256"]}
    diagnostic = _seal({"schema_version": 1, "artifact_type": "part2_sensitivity_call_order_diagnostic_v1", "analysis_family": "separate_diagnostic_not_in_30_test_global_holm", "rows": []})
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


def test_full_production_shaped_analysis_and_invalid_denominators(
    production_bundle: Mapping[str, Path], tmp_path: Path,
) -> None:
    result = _run(production_bundle, tmp_path / "out")

    assert result["row_counts"] == {
        "part0_models": 22, "part1_models": 75, "part2_models": 19,
        "role_calibration_model_frames": 18, "sensitivity_models": 6,
        "sensitivity_main_effects": 30,
    }
    assert result["human_labels_generated"] is False
    assert result["confirmatory_or_paper_promotion_permitted"] is False
    p0 = json.loads((tmp_path / "out/part0_models.jsonl").read_text().splitlines()[0])
    assert p0["scheduled_units"] == 144
    assert p0["first_attempt_invalid_count"] == 1
    assert p0["repaired_invalid_count"] == 0
    assert p0["refusal_rate_all_scheduled"] == p0["refusal_count"] / 144
    p1 = json.loads((tmp_path / "out/part1_models.jsonl").read_text().splitlines()[0])
    assert p1["welfare_preserving_rate_all_scheduled"] == 383 / 384
    p2 = json.loads((tmp_path / "out/part2_models.jsonl").read_text().splitlines()[0])
    assert p2["restraint_rate_all_scheduled"] == 84 / 120
    assert p2["restraint_rate_among_valid"] == 84 / 108


def test_incomplete_manifest_fails_before_output(tmp_path: Path) -> None:
    incomplete = ROOT / "data/private/inference_hub/provider-safe-v2-part0-large-n48-main22-v1/private/manifest.json"
    if not incomplete.exists():
        pytest.skip("Private running campaign is not available.")
    output = tmp_path / "must-not-exist"
    with pytest.raises(DefinitiveAnalysisError, match="not COMPLETE"):
        analyze(part0=incomplete, part1=incomplete, part2=incomplete, role_calibration=incomplete, sensitivity=incomplete, output_dir=output)
    assert not output.exists()


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
    _provider_safe_contract(main_manifest, "part0")
    _provider_safe_contract(exploratory_manifest, "role")
    _provider_safe_contract(deadline_manifest, "part1")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(main_manifest, "sensitivity")
    with pytest.raises(DefinitiveAnalysisError, match="Wrong accelerated launcher"):
        _provider_safe_contract(deadline_manifest, "part0")
    exploratory_manifest["execution_contract"]["shared_rate_limit"]["provider_requests_per_second"] = 2.1
    with pytest.raises(DefinitiveAnalysisError, match="policy hash failed"):
        _provider_safe_contract(exploratory_manifest, "sensitivity")
