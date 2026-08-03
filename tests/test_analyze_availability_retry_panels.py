from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from analysis.analyze_availability_retry_panels import (
    EXPECTED_IDENTITIES,
    EXPECTED_TARGETS,
    JUDGE_ID,
    MAIN_LAUNCHER,
    PROVIDER_SAFE_V2,
    AvailabilityRetryAnalysisError,
    _schedule_hash,
    _self_hash,
    analyze,
)
from experiments.misc import (
    inference_hub_part1_panel as p1base,
    inference_hub_part1_stratified_panel as p1strat,
)
from experiments.part1.confirmatory_design import (
    COUNTERBALANCES,
)


def _journal(
    path: Path,
    payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    rows = []
    encoded = []
    for payload in payloads:
        row = {
            **payload,
            "previous_record_sha256": previous,
        }
        row["record_sha256"] = (
            p1base._sha256_json(row)
        )
        previous = row["record_sha256"]
        rows.append(row)
        encoded.append(
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
            ) + "\n"
        )
    path.write_text(
        "".join(encoded), encoding="utf-8"
    )
    os.chmod(path, 0o600)
    return ({
        "path": str(path.resolve()),
        "record_count": len(rows),
        "tail_record_sha256": previous,
        "file_sha256":
            p1base._sha256_file(path),
    }, rows)


def _write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    private: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    if private:
        os.chmod(path, 0o600)


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = (
        _self_hash(value)
    )
    return value


def _policy() -> dict[str, Any]:
    policy = {
        "schema_version": 2,
        "algorithm":
            "cross_process_provider_aware_leaky_"
            "bucket_with_leases_all_http_5xx_full_"
            "throttle_cooldown",
        "global_concurrency": 12,
        "provider_concurrency": 2,
        "global_requests_per_second": 8.0,
        "provider_requests_per_second": 1.5,
        "lease_seconds": 900.0,
        "poll_seconds": 0.05,
        "throttle_cooldown_seconds": 30.0,
        "transient_cooldown_seconds": 5.0,
    }
    policy["policy_sha256"] = (
        p1base._sha256_json(policy)
    )
    return policy


def _route(target: str) -> str:
    return f"retry-route/{target}"


def _subject(target: str) -> dict[str, Any]:
    upstream, model = EXPECTED_IDENTITIES[target]
    return {
        "target_id": target,
        "upstream_provider": upstream,
        "model": model,
        "route": _route(target),
        "supported_controls": [
            "seed", "temperature", "top_p"
        ],
    }


def _combined_inputs(
    root: Path,
) -> dict[str, dict[str, Any]]:
    required = sorted({
        target
        for targets in EXPECTED_TARGETS.values()
        for target in targets
    } | {JUDGE_ID})
    registry_targets = []
    compatibility_targets = []
    for target in required:
        if target == JUDGE_ID:
            upstream, model = (
                "nvidia",
                "nvidia/evals-nemotron-3-30b-a3b",
            )
            route = (
                "nvidia/nvidia/"
                "evals-nemotron-3-30b-a3b"
            )
        else:
            upstream, model = (
                EXPECTED_IDENTITIES[target]
            )
            route = _route(target)
        registry_targets.append({
            "id": target,
            "provider": "inference_hub",
            "upstream_provider": upstream,
            "model": model,
        })
        compatibility_targets.append({
            "target_id": target,
            "status":
                "execution_candidate_selected",
            "selected_execution_candidate":
                route,
        })
    registry = {
        "schema_version": 1,
        "artifact_type":
            "exploratory_sota_inference_hub_registry_"
            "with_dedicated_judge",
        "analysis_role":
            "exploratory_sota_panel_with_dedicated_"
            "judge_only",
        "paper_result_promotion_permitted": False,
        "confirmatory_promotion_permitted": False,
        "production_registry_mutation_permitted": False,
        "targets": registry_targets,
    }
    registry["artifact_sha256"] = (
        p1base._sha256_json(registry)
    )
    registry_path = (
        root
        / "combined-visible-retry-registry-test.json"
    )
    _write_json(registry_path, registry)
    compatibility = _seal({
        "schema_version": 2,
        "artifact_type":
            "inference_hub_provider_compatibility",
        "analysis_role":
            "exploratory_sota_panel_with_dedicated_"
            "judge_only",
        "paper_result_promotion_permitted": False,
        "confirmatory_promotion_permitted": False,
        "production_registry_mutation_permitted": False,
        "registry_sha256":
            p1base._sha256_json(registry),
        "targets": compatibility_targets,
    })
    compatibility_path = (
        root
        / "combined-visible-retry-compatibility-"
        "test.json"
    )
    _write_json(
        compatibility_path, compatibility
    )
    return {
        "registry": {
            "path": str(registry_path.resolve()),
            "file_sha256":
                p1base._sha256_file(registry_path),
            "canonical_sha256":
                p1base._sha256_json(registry),
        },
        "compatibility": {
            "path":
                str(compatibility_path.resolve()),
            "file_sha256":
                p1base._sha256_file(
                    compatibility_path
                ),
            "evidence_sha256":
                compatibility["evidence_sha256"],
        },
    }


def _base_manifest(
    artifact_type: str,
    subjects: list[dict[str, Any]],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "complete": True,
        "completed_at_utc":
            "2026-08-03T12:00:00Z",
        "subject_routes": subjects,
        "execution_contract": {
            "shared_rate_limit": _policy()
        },
        "input_artifacts": dict(inputs),
        "source_artifacts": {
            str(MAIN_LAUNCHER.resolve()):
                p1base._sha256_file(
                    MAIN_LAUNCHER
                ),
            str(PROVIDER_SAFE_V2.resolve()):
                p1base._sha256_file(
                    PROVIDER_SAFE_V2
                ),
        },
    }


def _judge() -> dict[str, Any]:
    return {
        "target_id": JUDGE_ID,
        "upstream_provider": "nvidia",
        "model":
            "nvidia/evals-nemotron-3-30b-a3b",
        "route":
            "nvidia/nvidia/"
            "evals-nemotron-3-30b-a3b",
        "dispatch_permitted_in_this_runner":
            False,
    }


def _sanitized(
    path: Path,
    artifact_type: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _seal({
        "schema_version": 1,
        "artifact_type": artifact_type,
        "rows": rows,
    })
    _write_json(path, payload)
    return {
        "path": str(path.resolve()),
        "file_sha256": p1base._sha256_file(path),
        "evidence_sha256":
            payload["evidence_sha256"],
    }


def _write_manifest(
    path: Path, manifest: dict[str, Any]
) -> None:
    _write_json(
        path, _seal(manifest), private=True
    )


def _bundle(root: Path) -> dict[str, Path]:
    inputs = _combined_inputs(root / "inputs")
    result: dict[str, Path] = {}

    part0_run = root / "part0"
    part0_private = part0_run / "private"
    part0_target = EXPECTED_TARGETS["part0"][0]
    part0_raw = []
    outcomes = (
        "REFUSAL", "COMPLIANCE", "UNCLEAR", "INVALID"
    )
    for language in ("english", "chinese", "russian"):
        for index in range(48):
            raw_response = {
                "model": _route(part0_target),
                "choices": [{
                    "message": {
                        "content": "I cannot assist."
                    }
                }],
            }
            part0_raw.append({
                "event": "subject_response_retained",
                "target_id": part0_target,
                "upstream_provider": "anthropic",
                "model": "claude-opus-4-5",
                "requested_route": _route(part0_target),
                "response_model": _route(part0_target),
                "model_identity_valid": True,
                "trial_id": f"{language}-{index:02d}",
                "root_id": f"root-{index:02d}",
                "language": language,
                "raw_response": raw_response,
                "raw_response_sha256":
                    p1base._sha256_json(raw_response),
            })
            part0_raw.append({
                "event": "unit_completed",
                "target_id": part0_target,
                "trial_id": f"{language}-{index:02d}",
                "root_id": f"root-{index:02d}",
                "language": language,
                "outcome": outcomes[index % len(outcomes)],
            })
        for batch in range(6):
            raw_response = {
                "model": _judge()["route"],
                "labels": ["REFUSAL"] * 8,
            }
            part0_raw.append({
                "event": "judge_batch_retained",
                "target_id": part0_target,
                "batch_id": f"{language}-batch-{batch}",
                "requested_route": _judge()["route"],
                "response_model": _judge()["route"],
                "model_identity_valid": True,
                "raw_response": raw_response,
                "raw_response_sha256":
                    p1base._sha256_json(raw_response),
            })
    attempt_ref, _ = _journal(
        part0_private / "attempt-ledger.jsonl", []
    )
    raw_ref, _ = _journal(
        part0_private / "raw.jsonl", part0_raw
    )
    part0_inputs = {
        key: {
            "path": value["path"],
            "sha256": value["file_sha256"],
        }
        for key, value in inputs.items()
    }
    part0_manifest = _base_manifest(
        "inference_hub_part0_accelerated_private_panel",
        [_subject(part0_target)],
        part0_inputs,
    )
    part0_manifest.update({
        "base_seed": 20_260_802,
        "selected_roots_per_language": 48,
        "executed_trial_count_per_subject": 144,
        "languages": ["english", "chinese", "russian"],
        "judge": _judge(),
        "journals": {
            "attempt_ledger": attempt_ref,
            "raw_responses": {
                part0_target: raw_ref
            },
        },
        "summary": {
            "planned_units": 144,
            "retained_terminal_units": 144,
            "operationally_retired_units": 0,
            "subject_model_identity_mismatches": 0,
            "judge_model_identity_mismatches": 0,
            "subject_transport_failures": 0,
            "judge_failed_units": 0,
            "outcomes": {
                "REFUSAL": 36,
                "COMPLIANCE": 36,
                "UNCLEAR": 36,
                "INVALID": 36,
            },
        },
    })
    part0_path = part0_private / "manifest.json"
    _write_manifest(part0_path, part0_manifest)
    result["part0"] = part0_path

    part1_run = root / "part1"
    part1_private = part1_run / "private"
    part1_subjects = [
        _subject(target)
        for target in EXPECTED_TARGETS["part1"]
    ]
    trials = p1strat.build_stratified_trials(
        base_seed=20260802, limit=384
    )
    part1_attempt_ref, _ = _journal(
        part1_private / "attempt-ledger.jsonl", []
    )
    part1_raw_refs = {}
    for subject in part1_subjects:
        target = subject["target_id"]
        payloads = []
        for index, trial in enumerate(trials):
            counterbalance = next(
                value
                for value in COUNTERBALANCES
                if value.counterbalance_id
                == trial.counterbalance_id
            )
            valid = index % 17 != 0
            raw_response = {
                "choices": [{
                    "message": {
                        "content": "OPTION_A"
                    }
                }]
            }
            payloads.append({
                "target_id": target,
                "upstream_provider":
                    subject["upstream_provider"],
                "model": subject["model"],
                "requested_route": subject["route"],
                "response_model": subject["route"],
                "trial_id": trial.trial_id,
                "root_id": trial.root_id,
                "prompt_sha256": trial.prompt_hash,
                "model_identity_valid": True,
                "raw_response": raw_response,
                "raw_response_sha256":
                    p1base._sha256_json(raw_response),
                "format_valid": valid,
                "counterbalance_id":
                    trial.counterbalance_id,
                "parsed_action": (
                    counterbalance.welfare_preserving_label
                    if valid else None
                ),
            })
        reference, _ = _journal(
            part1_private
            / f"raw-{target.replace('/', '_')}.jsonl",
            payloads,
        )
        part1_raw_refs[target] = reference
    part1_manifest = _base_manifest(
        "inference_hub_part1_large_n_exploratory_panel",
        part1_subjects,
        inputs,
    )
    part1_manifest.update({
        "judge_reservation": _judge(),
        "base_seed": 20260802,
        "trial_limit": 384,
        "executed_trial_count_per_subject": 384,
        "full_primary_root_count": 384,
        "executed_schedule_sha256":
            _schedule_hash(trials),
        "journals": {
            "attempt_ledger": part1_attempt_ref,
            "raw_responses": part1_raw_refs,
        },
        "summary": {
            "planned_generations": 4 * 384,
            "retained_trial_records": 4 * 384,
            "responses_received": 4 * 384,
            "failed_without_response": 0,
            "format_valid": 4 * (384 - 23),
            "format_invalid_retained": 4 * 23,
            "response_model_identity_mismatches": 0,
        },
    })
    part1_path = part1_private / "manifest.json"
    _write_manifest(part1_path, part1_manifest)
    result["part1"] = part1_path

    part2_run = root / "part2"
    part2_private = part2_run / "private"
    part2_subjects = [
        _subject(target)
        for target in EXPECTED_TARGETS["part2"]
    ]
    part2_journals = {}
    trajectory_rows = []
    model_rows = []
    for subject in part2_subjects:
        target = subject["target_id"]
        model_rows.append({
            "target_id": target,
            "upstream_provider":
                subject["upstream_provider"],
            "model": subject["model"],
            "trajectory_count": 12,
            "eligible_trajectory_count": 12,
            "expected_trajectory_count": 12,
            "complete_matched_panel": True,
            "total_scheduled_agent_days": 120,
            "total_invalid_count": 12,
            "total_identity_mismatch_count": 0,
            "total_transport_failure_count": 0,
        })
        for trajectory_index in range(12):
            payloads = []
            for slot in range(10):
                attempt_id = (
                    f"attempt-{target}-{trajectory_index}-{slot}"
                )
                request_body = {
                    "model": subject["route"],
                    "messages": [{
                        "role": "user",
                        "content": "choose",
                    }],
                    "seed": slot,
                }
                request_sha256 = p1base._sha256_json(
                    request_body
                )
                if slot == 0:
                    failed_attempt_id = attempt_id + "-retryable"
                    payloads.extend(({
                        "event": "reserved_before_dispatch",
                        "attempt_id": failed_attempt_id,
                        "target_id": target,
                        "trajectory_index": trajectory_index,
                        "day": 0,
                        "slot": slot,
                        "upstream_provider":
                            subject["upstream_provider"],
                        "model": subject["model"],
                        "requested_route": subject["route"],
                        "request_sha256": request_sha256,
                    }, {
                        "event": "attempt_failed",
                        "attempt_id": failed_attempt_id,
                        "target_id": target,
                        "trajectory_index": trajectory_index,
                        "day": 0,
                        "slot": slot,
                        "request_sha256": request_sha256,
                        "failure": {
                            "failure_code": "http_503",
                            "transient": True,
                        },
                    }))
                payloads.append({
                    "event": "reserved_before_dispatch",
                    "attempt_id": attempt_id,
                    "target_id": target,
                    "trajectory_index": trajectory_index,
                    "day": 0,
                    "slot": slot,
                    "upstream_provider":
                        subject["upstream_provider"],
                    "model": subject["model"],
                    "requested_route": subject["route"],
                    "request_sha256": request_sha256,
                })
                action = (
                    "OPTION_A" if slot < 7
                    else "OPTION_B" if slot < 9
                    else "INVALID"
                )
                raw_response = {
                    "model": subject["route"],
                    "choice": action,
                    "trajectory": trajectory_index,
                    "slot": slot,
                }
                payloads.append({
                    "event": "semantic_result",
                    "attempt_id": attempt_id,
                    "target_id": target,
                    "trajectory_index": trajectory_index,
                    "day": 0,
                    "slot": slot,
                    "upstream_provider":
                        subject["upstream_provider"],
                    "model": subject["model"],
                    "requested_route": subject["route"],
                    "model_identity_valid": True,
                    "action": action,
                    "raw_response": raw_response,
                    "raw_response_sha256":
                        p1base._sha256_json(raw_response),
                    "request_body": request_body,
                    "request_sha256": request_sha256,
                })
            reference, _ = _journal(
                part2_private
                / f"{target.replace('/', '_')}-{trajectory_index}.jsonl",
                payloads,
            )
            part2_journals[
                f"{target}::{trajectory_index}"
            ] = reference
            trajectory_rows.append({
                "target_id": target,
                "upstream_provider":
                    subject["upstream_provider"],
                "model": subject["model"],
                "trajectory_index": trajectory_index,
                "identity_mismatch_count": 0,
                "transport_failure_count": 0,
                "scheduled_agent_days": 10,
                "restraint_count": 7,
                "overuse_count": 2,
                "invalid_count": 1,
                "operationally_eligible": True,
                "aurc": 0.8,
                "aupc": 0.9,
            })
    trajectory_ref = _sanitized(
        part2_run
        / "sanitized/trajectory_metrics.json",
        "inference_hub_part2_sanitized_trajectory_metrics",
        trajectory_rows,
    )
    model_ref = _sanitized(
        part2_run / "sanitized/model_metrics.json",
        "inference_hub_part2_sanitized_model_metrics",
        model_rows,
    )
    part2_manifest = _base_manifest(
        "inference_hub_part2_corrected_matched_panel",
        part2_subjects,
        inputs,
    )
    part2_manifest.update({
        "judge_reservation": _judge(),
        "journals": part2_journals,
        "sanitized_artifacts": {
            "trajectory_metrics": trajectory_ref,
            "model_metrics": model_ref,
        },
        "summary": {
            "planned_trajectories": 48,
            "completed_trajectories": 48,
            "identity_mismatch_count": 0,
            "transport_failure_count": 0,
            "scheduled_agent_days": 4 * 12 * 10,
            "responses_received": 4 * 12 * 10,
            "invalid_count": 4 * 12,
            "eligible_trajectories": 4 * 12,
        },
    })
    part2_path = part2_private / "manifest.json"
    _write_manifest(part2_path, part2_manifest)
    result["part2"] = part2_path
    return result


def _manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_journal(
    manifest_path: Path,
    *,
    phase: str,
    target_or_key: str,
    payloads: list[dict[str, Any]],
) -> None:
    manifest = _manifest(manifest_path)
    if phase in {"part0", "part1"}:
        old_ref = manifest["journals"][
            "raw_responses"
        ][target_or_key]
        parent = manifest_path.parent
        reference, _ = _journal(
            parent / f"rewritten-{phase}.jsonl",
            payloads,
        )
        manifest["journals"]["raw_responses"][
            target_or_key
        ] = reference
    else:
        old_ref = manifest["journals"][target_or_key]
        reference, _ = _journal(
            manifest_path.parent / "rewritten-part2.jsonl",
            payloads,
        )
        manifest["journals"][target_or_key] = reference
    assert old_ref != reference
    _write_manifest(manifest_path, manifest)


def _run(bundle: Mapping[str, Path], output: Path) -> dict[str, Any]:
    return analyze(
        part0=bundle["part0"],
        part1=bundle["part1"],
        part2=bundle["part2"],
        output_dir=output,
    )


def test_exact_production_shapes_emit_isolated_outputs(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path / "inputs")
    output = tmp_path / "analysis"
    result = _run(bundle, output)
    assert result["row_counts"] == {
        "part0_availability_retry": 1,
        "part1_availability_retry": 4,
        "part2_availability_retry": 4,
    }
    assert result["analysis_role"] == "availability_retry"
    assert result["exploratory_only"] is True
    assert result["replaces_primary"] is False
    assert result["merge_with_primary_permitted"] is False
    assert result["cross_axis_permitted"] is False
    assert result["latex_contract"]["tabcolsep"] == "4pt"
    assert result["latex_contract"]["table_outer_spacing_pt"] == 15
    assert set(result["published_outputs"]) == {
        "part0_availability_retry",
        "part1_availability_retry",
        "part2_availability_retry",
    }
    published_manifest = (
        output / "analysis_manifest.json"
    ).read_text(encoding="utf-8")
    assert str(tmp_path) not in published_manifest
    assert '"path"' not in published_manifest
    for phase, expected in (("part0", 1), ("part1", 4), ("part2", 4)):
        stem = f"{phase}_availability_retry"
        jsonl = (output / f"{stem}.jsonl").read_text(
            encoding="utf-8"
        )
        rows = [json.loads(line) for line in jsonl.splitlines()]
        assert len(rows) == expected
        assert all(
            row["analysis_role"] == "availability_retry"
            and row["exploratory_only"] is True
            and row["replaces_primary"] is False
            and row["cross_axis_permitted"] is False
            for row in rows
        )
        assert (output / f"{stem}.csv").is_file()
        latex = (output / f"{stem}.tex").read_text(
            encoding="utf-8"
        )
        assert r"\setlength{\tabcolsep}{4pt}" in latex
        assert latex.count(r"\par\addvspace{15pt}") == 2
        for word in (
            "availability-retry", "never merge", "All",
            "Direction", "Invalid",
        ):
            assert word in latex
        if phase == "part2":
            assert "NE" in latex


def test_incomplete_input_fails_before_output(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "inputs")
    manifest = _manifest(bundle["part0"])
    manifest["complete"] = False
    _write_manifest(bundle["part0"], manifest)
    output = tmp_path / "analysis"
    with pytest.raises(
        AvailabilityRetryAnalysisError, match="COMPLETE"
    ):
        _run(bundle, output)
    assert not output.exists()


def test_manifest_tamper_fails_before_output(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "inputs")
    manifest = _manifest(bundle["part0"])
    manifest["summary"]["planned_units"] = 143
    _write_json(bundle["part0"], manifest, private=True)
    output = tmp_path / "analysis"
    with pytest.raises(
        AvailabilityRetryAnalysisError, match="self-hash-valid"
    ):
        _run(bundle, output)
    assert not output.exists()


def test_response_identity_mismatch_fails(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "inputs")
    target = EXPECTED_TARGETS["part1"][0]
    manifest = _manifest(bundle["part1"])
    path = Path(
        manifest["journals"]["raw_responses"][target]["path"]
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    payloads = [
        {
            key: value
            for key, value in row.items()
            if key not in {
                "record_sha256", "previous_record_sha256"
            }
        }
        for row in rows
    ]
    payloads[0]["model_identity_valid"] = False
    _replace_journal(
        bundle["part1"], phase="part1",
        target_or_key=target, payloads=payloads,
    )
    with pytest.raises(
        AvailabilityRetryAnalysisError, match="row identity"
    ):
        _run(bundle, tmp_path / "analysis")


def test_judge_overlap_fails(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "inputs")
    manifest = _manifest(bundle["part2"])
    subject = manifest["subject_routes"][0]
    manifest["judge_reservation"].update({
        "target_id": subject["target_id"],
        "upstream_provider": subject["upstream_provider"],
        "model": subject["model"],
        "route": subject["route"],
    })
    _write_manifest(bundle["part2"], manifest)
    with pytest.raises(
        AvailabilityRetryAnalysisError,
        match="fixed judge|overlaps",
    ):
        _run(bundle, tmp_path / "analysis")


def test_wrong_row_count_fails(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "inputs")
    target = EXPECTED_TARGETS["part1"][0]
    manifest = _manifest(bundle["part1"])
    path = Path(
        manifest["journals"]["raw_responses"][target]["path"]
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ][:-1]
    payloads = [{
        key: value
        for key, value in row.items()
        if key not in {
            "record_sha256", "previous_record_sha256"
        }
    } for row in rows]
    _replace_journal(
        bundle["part1"], phase="part1",
        target_or_key=target, payloads=payloads,
    )
    with pytest.raises(
        AvailabilityRetryAnalysisError, match="row count"
    ):
        _run(bundle, tmp_path / "analysis")


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "inputs")
    output = tmp_path / "analysis"
    first = _run(bundle, output)
    manifest_before = (
        output / "analysis_manifest.json"
    ).read_bytes()
    with pytest.raises(
        AvailabilityRetryAnalysisError, match="refusing overwrite"
    ):
        _run(bundle, output)
    assert (
        output / "analysis_manifest.json"
    ).read_bytes() == manifest_before
    assert first["evidence_sha256"] == json.loads(
        manifest_before
    )["evidence_sha256"]
