from __future__ import annotations

import json
import stat
import csv
from copy import deepcopy
from pathlib import Path

import pytest

from analysis import confirmatory_data_lock as data_lock
from analysis import confirmatory_judge_adapter as adapter
from analysis import judge_audit, part2_confirmatory, confirmatory_estimators
from experiments.misc.attempt_log import DurableAttemptLogger
from experiments.misc.run_metadata import sha256_file, stable_json_hash


def _write(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(str(value), encoding="utf-8")
    return sha256_file(path)


def _completed_attempt(started_at_utc: str, finished_at_utc: str) -> dict[str, object]:
    return {
        "attempt": 1,
        "started_at_utc": started_at_utc,
        "argv_sha256": "a" * 64,
        "resume": False,
        "finished_at_utc": finished_at_utc,
        "returncode": 0,
        "timed_out": False,
        "error": None,
    }


def _fake_adapter_source(
    tmp_path: Path, *, per_cell: int = 1
) -> tuple[list[dict[str, object]], dict[str, object], int]:
    files = {}
    for name in ("plan", "results", "attempts", "metadata", "registry"):
        path = tmp_path / f"{name}.source"
        digest = _write(path, f"{name}-bytes")
        files[name] = (path, digest)
    rows = []
    sequence = 0
    for language in adapter.LANGUAGES:
        for label in adapter.LABELS:
            for repetition in range(per_cell):
                sequence += 1
                payload = {
                "schema_version": adapter.SCHEMA_VERSION,
                "record_type": adapter.RECORD_TYPE,
                "item_id": f"ca_fixture_{sequence}",
                "language": language,
                "automated_label": label,
                "provider": "inference_hub",
                "model": "vendor/model",
                "target_id": "target-model",
                "arm": "harmful" if sequence % 2 else "control",
                "source": "harmbench" if sequence % 2 else "jbb",
                "category": f"category-{repetition % 2}",
                "semantic_cluster_id": f"cluster-{sequence}",
                "generation_block": ((sequence - 1) % 3) + 1,
                "request_original": f"request-{language}-{label}",
                "request_english": f"english-request-{language}-{label}",
                "response_original": f"visible-response-{language}-{label}",
                "response_english": "",
                "source_run": str(tmp_path / "run"),
                "source_row": sequence,
                "source_trial_id": f"trial-{sequence}",
                "source_record_sha256": stable_json_hash({"source": sequence}),
                }
                payload["record_sha256"] = stable_json_hash(payload)
                rows.append(payload)
    source = {
        "run_directory": str(tmp_path / "run"),
        "plan_path": str(files["plan"][0]),
        "plan_sha256": stable_json_hash({"plan": 1}),
        "plan_file_sha256": files["plan"][1],
        "results_path": str(files["results"][0]),
        "results_sha256": files["results"][1],
        "result_count": len(rows),
        "attempts_path": str(files["attempts"][0]),
        "attempts_sha256": files["attempts"][1],
        "metadata_path": str(files["metadata"][0]),
        "metadata_sha256": stable_json_hash({"metadata": 1}),
        "metadata_file_sha256": files["metadata"][1],
        "registry_path": str(files["registry"][0]),
        "registry_sha256": files["registry"][1],
        "provider": "inference_hub",
        "model": "vendor/model",
    }
    return rows, source, sum(row["automated_label"] == "unclear" for row in rows)


def test_adapter_bundle_is_hash_preserving_and_accepted_by_blinded_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, source, unclear = _fake_adapter_source(tmp_path)
    monkeypatch.setattr(adapter, "_native_run", lambda *args, **kwargs: (deepcopy(rows), deepcopy(source), unclear))
    registry = tmp_path / "registry.json"
    registry_hash = _write(registry, {})
    output = tmp_path / "adapted"
    manifest = adapter.build_confirmatory_audit_input(
        [tmp_path / "run"],
        registry_path=registry,
        registry_sha256=registry_hash,
        output_directory=output,
    )
    loaded, validated = adapter.load_confirmatory_audit_input(output / adapter.MANIFEST_NAME)
    assert len(loaded) == 9
    assert validated["manifest_sha256"] == manifest["manifest_sha256"]
    assert manifest["native_unclear_count"] == 3
    assert set(loaded[0]) == adapter._RECORD_KEYS
    assert all("reasoning" not in key for row in loaded for key in row)
    assert stat.S_IMODE((output / adapter.RECORDS_NAME).stat().st_mode) == 0o600
    audit_output = tmp_path / "audit"
    judge_audit.generate_audit(
        [output / adapter.MANIFEST_NAME],
        audit_output,
        per_stratum=1,
        duplicate_count=6,
        annotator_count=1,
    )
    assert (audit_output / "audit_key.csv").is_file()


def test_adapter_rejects_duplicate_runs_incomplete_strata_and_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, source, unclear = _fake_adapter_source(tmp_path)
    monkeypatch.setattr(adapter, "_native_run", lambda *args, **kwargs: (deepcopy(rows), deepcopy(source), unclear))
    registry = tmp_path / "registry.json"
    registry_hash = _write(registry, {})
    with pytest.raises(adapter.ConfirmatoryJudgeAdapterError, match="duplicates"):
        adapter.build_confirmatory_audit_input(
            [tmp_path / "run", tmp_path / "run"],
            registry_path=registry,
            registry_sha256=registry_hash,
            output_directory=tmp_path / "duplicate",
        )
    monkeypatch.setattr(
        adapter,
        "_native_run",
        lambda *args, **kwargs: (
            deepcopy(rows[:-1]),
            deepcopy(source),
            unclear - 1,
        ),
    )
    with pytest.raises(adapter.ConfirmatoryJudgeAdapterError, match="strata"):
        adapter.build_confirmatory_audit_input(
            [tmp_path / "run"], registry_path=registry, registry_sha256=registry_hash,
            output_directory=tmp_path / "incomplete",
        )
    monkeypatch.setattr(adapter, "_native_run", lambda *args, **kwargs: (deepcopy(rows), deepcopy(source), unclear))
    output = tmp_path / "valid"
    adapter.build_confirmatory_audit_input(
        [tmp_path / "run"], registry_path=registry, registry_sha256=registry_hash,
        output_directory=output,
    )
    records_path = output / adapter.RECORDS_NAME
    records_path.write_text(records_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(adapter.ConfirmatoryJudgeAdapterError, match="hash mismatch"):
        adapter.load_confirmatory_audit_input(output / adapter.MANIFEST_NAME)
    with pytest.raises(judge_audit.AuditError, match="integrity validation"):
        judge_audit.generate_audit(
            [output / adapter.MANIFEST_NAME], tmp_path / "never", per_stratum=1,
            duplicate_count=6, annotator_count=1,
        )


def test_confirmatory_key_replays_multifactor_allocation_and_rejects_rebalance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, source, unclear = _fake_adapter_source(tmp_path, per_cell=2)
    monkeypatch.setattr(
        adapter,
        "_native_run",
        lambda *args, **kwargs: (deepcopy(rows), deepcopy(source), unclear),
    )
    registry = tmp_path / "registry.json"
    registry_hash = _write(registry, {})
    adapted = tmp_path / "adapted-balanced"
    adapter.build_confirmatory_audit_input(
        [tmp_path / "run"], registry_path=registry,
        registry_sha256=registry_hash, output_directory=adapted,
    )
    audit = tmp_path / "audit-balanced"
    judge_audit.generate_audit(
        [adapted], audit, seed=17, per_stratum=1,
        duplicate_count=6, annotator_count=1,
    )
    with (audit / "audit_key.csv").open(newline="", encoding="utf-8") as handle:
        keys = [row for row in csv.DictReader(handle) if row["item_kind"] == "primary"]
    judge_audit._validate_confirmatory_key_sources(keys)
    selected = keys[0]
    alternate = next(
        row for row in rows
        if row["language"] == selected["language"]
        and row["automated_label"] == selected["automated_label"]
        and row["item_id"] != selected["item_id"]
    )
    selected.update(
        item_id=str(alternate["item_id"]),
        language=str(alternate["language"]),
        automated_label=str(alternate["automated_label"]),
        provider=str(alternate["provider"]),
        model=str(alternate["model"]),
        target_id=str(alternate["target_id"]),
        arm=str(alternate["arm"]),
        source=str(alternate["source"]),
        category=str(alternate["category"]),
        semantic_cluster_id=str(alternate["semantic_cluster_id"]),
        generation_block=str(alternate["generation_block"]),
        source_row=str(alternate["source_row"]),
        stimulus_sha256=judge_audit._stimulus_digest(
            {
                "language": str(alternate["language"]),
                "request_original": str(alternate["request_original"]),
                "request_english": str(alternate["request_english"]),
                "response_original": str(alternate["response_original"]),
                "response_english": str(alternate["response_english"]),
            }
        ),
    )
    with pytest.raises(judge_audit.AuditError, match="proportional blocked allocation"):
        judge_audit._validate_confirmatory_key_sources(keys)


@pytest.fixture()
def lock_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    jobs = []
    for experiment in ("part0", "part1"):
        for target_id in ("target-1", "target-2"):
            artifact_path = tmp_path / f"{experiment}-{target_id}.artifact"
            artifact_hash = _write(artifact_path, f"{experiment}-{target_id}-bytes")
            artifact = {
                "kind": "private_confirmatory_run",
                "output_dir": str(tmp_path / f"{experiment}-{target_id}"),
                "files": [{"path": str(artifact_path), "sha256": artifact_hash, "size_bytes": artifact_path.stat().st_size}],
            }
            jobs.append({
                "id": f"production-{experiment}-{target_id}",
                "target_id": target_id,
                "experiment": experiment,
                "stage": "production",
                "status": "complete",
                "attempts": [
                    _completed_attempt(
                        f"2026-08-02T{len(jobs) + 1:02d}:00:00Z",
                        f"2026-08-02T{len(jobs) + 1:02d}:30:00Z",
                    )
                ],
                "artifact": artifact,
            })
    meta = tmp_path / "part2_meta.json"
    csv = tmp_path / "part2.csv"
    _write(meta, {})
    _write(csv, "header\nvalue\n")
    part2_artifact = {"metadata_path": str(meta), "csv_path": str(csv), "rows": 1, "verified_at_utc": "2026-08-02T00:00:00+00:00"}
    jobs.append({
        "id": "part2-variance", "experiment": "part2",
        "stage": "part2_variance_pilot", "status": "complete",
        "attempts": [
            _completed_attempt(
                "2026-08-03T01:00:00Z", "2026-08-03T02:00:00Z"
            )
        ],
        "artifact": part2_artifact,
    })
    jobs.append({"id": "smoke-part2-variance", "experiment": "part2", "stage": "smoke", "status": "complete", "artifact": {**part2_artifact, "metadata_path": str(tmp_path / "smoke_meta.json"), "csv_path": str(tmp_path / "smoke.csv")}})
    _write(Path(jobs[-1]["artifact"]["metadata_path"]), {})
    _write(Path(jobs[-1]["artifact"]["csv_path"]), "header\nvalue\n")
    inputs = {}
    for name in ("part0_registry", "part1_bank", "endpoint_evidence"):
        path = tmp_path / f"{name}.json"
        digest = _write(path, {})
        inputs[name] = {"path": str(path), "sha256": digest}
    common = {
        "status": "complete",
        "created_at_utc": "2026-08-02T00:00:00Z",
        "cohorts": [{"id": "current_sota"}],
        "target_selection": {
            "mode": "complete_union",
            "selected_target_ids": ["target-1", "target-2"],
            "complete_union_target_count": 2,
        },
        "targets": [{"id": "target-1"}, {"id": "target-2"}],
        "roles": {"extractor_target_id": "target-1", "judge_target_id": "target-1"},
        "registry": {"version": "fixture"},
        "execution_freeze": {"git_commit": "a" * 40},
    }
    variance_campaign = {
        **deepcopy(common),
        "manifest_sha256": "b" * 64,
        "plan_sha256": "c" * 64,
        "jobs": jobs,
        "inputs": {**deepcopy(inputs), "variance_selection": None},
        "part2_design": {"scientific_stage": "part2_variance_pilot"},
    }
    variance_campaign_path = tmp_path / "variance-campaign.json"
    variance_campaign_file_hash = _write(variance_campaign_path, variance_campaign)
    selection = {
        "schema_version": 1,
        "selection_rule": "smallest_n_with_t95_half_width_at_most_0.05_capped_20_40",
        "identity_masked": True,
        "group_count": 1,
        "pilot_runs_per_group": 8,
        "common_environment_seed_count": 8,
        "common_environment_seeds_sha256": "1" * 64,
        "target_half_width": 0.05,
        "s_max": 0.01,
        "selected_common_run_count": 20,
        "t_critical": 2.0,
        "achieved_half_width": 0.01,
        "capped_at_maximum": False,
        "location_removed_masked_input_sha256": "2" * 64,
    }
    variance = part2_confirmatory._sealed_artifact(
        {
            "schema_version": 1,
            "artifact_type": "part2_identity_masked_variance_selection",
            "private_input_sha256": "3" * 64,
            "pilot_campaign_manifest_sha256": variance_campaign_file_hash,
            "selection": selection,
        }
    )
    variance_path = tmp_path / "variance.json"
    variance_hash = _write(variance_path, variance)
    inputs["variance_selection"] = {
        "path": str(variance_path),
        "sha256": variance_hash,
        "selected_n": 20,
        "artifact_sha256": variance["artifact_sha256"],
        "private_input_sha256": variance["private_input_sha256"],
        "pilot_campaign": {
            "path": str(variance_campaign_path),
            "sha256": variance_campaign_file_hash,
            "campaign_id": "fixture-variance",
            "plan_sha256": variance_campaign["plan_sha256"],
            "manifest_payload_sha256": variance_campaign["manifest_sha256"],
        },
    }
    baseline_meta = tmp_path / "baseline_meta.json"
    baseline_csv = tmp_path / "baseline.csv"
    baseline_smoke_meta = tmp_path / "baseline_smoke_meta.json"
    baseline_smoke_csv = tmp_path / "baseline_smoke.csv"
    for path, value in (
        (baseline_meta, {}), (baseline_csv, "header\nvalue\n"),
        (baseline_smoke_meta, {}), (baseline_smoke_csv, "header\nvalue\n"),
    ):
        _write(path, value)
    baseline_jobs = [
        {
            "id": "part2-baseline", "experiment": "part2",
            "stage": "part2_baseline_production", "status": "complete",
            "attempts": [
                _completed_attempt(
                    "2026-08-20T01:00:00Z", "2026-08-20T02:00:00Z"
                )
            ],
            "artifact": {"metadata_path": str(baseline_meta), "csv_path": str(baseline_csv), "rows": 1, "verified_at_utc": "2026-08-02T00:00:00+00:00"},
        },
        {
            "id": "smoke-part2-baseline", "experiment": "part2", "stage": "smoke",
            "status": "complete",
            "artifact": {"metadata_path": str(baseline_smoke_meta), "csv_path": str(baseline_smoke_csv), "rows": 1, "verified_at_utc": "2026-08-02T00:00:00+00:00"},
        },
    ]
    baseline_campaign = {
        **deepcopy(common),
        "manifest_sha256": "e" * 64,
        "plan_sha256": "f" * 64,
        "jobs": baseline_jobs,
        "inputs": deepcopy(inputs),
        "part2_design": {"scientific_stage": "part2_baseline_production"},
    }
    baseline_campaign_path = tmp_path / "baseline-campaign.json"
    _write(baseline_campaign_path, baseline_campaign)
    monkeypatch.setattr(data_lock.confirmatory_campaign, "validate_manifest", lambda value: None)
    monkeypatch.setattr(data_lock.confirmatory_campaign, "resolve_job_artifact", lambda job, before: deepcopy(job["artifact"]))
    monkeypatch.setattr(
        data_lock.confirmatory_campaign,
        "_validate_variance_gate",
        lambda *args, **kwargs: 20,
    )
    monkeypatch.setattr(
        data_lock,
        "_validate_cohort_estimand",
        lambda campaign: [
            {
                "system_id": target["id"],
                "cohort_id": "current_sota",
                "developer_id": f"fixture-developer-{index}",
            }
            for index, target in enumerate(campaign["targets"])
        ],
    )
    registry_source = tmp_path / "agent_config.registry.json"
    _write(registry_source, {})
    repo_root = tmp_path / "repo"
    for relative in (
        "docs/CONFIRMATORY_PROTOCOL.md",
        "analysis/judge_audit.py",
        "analysis/confirmatory_estimators.py",
        "analysis/confirmatory_data_lock.py",
        "analysis/confirmatory_judge_adapter.py",
    ):
        _write(repo_root / relative, relative)
    _write(repo_root / "agents/agent_config.registry.json", {})
    monkeypatch.setattr(data_lock.confirmatory_campaign, "REPO_ROOT", repo_root)
    def audit_ref(name: str) -> dict[str, object]:
        path = tmp_path / name
        _write(path, name)
        return {"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    judge = {
        "schema_version": 2,
        "workflow": "part0_multi_annotator_human_judge_validation",
        "annotator_ids": ["annotator_1", "annotator_2"],
        "primary_rows": 1800,
        "duplicate_rows_per_annotator": {"annotator_1": 360, "annotator_2": 360},
        "adjudication": {},
        "weighting": "fixture",
        "unclear_handling": "fixture",
        "overall": {},
        "per_language": {},
        "bootstrap_confidence_intervals": {},
        "inter_rater_reliability": {},
        "intra_rater_reliability": {},
        "input_integrity": {
            "audit_key": audit_ref("audit-key.csv"),
            "annotations": {
                "annotator_1": audit_ref("annotator-1.csv"),
                "annotator_2": audit_ref("annotator-2.csv"),
            },
            "duplicate_annotations": {
                "annotator_1": audit_ref("duplicate-1.csv"),
                "annotator_2": audit_ref("duplicate-2.csv"),
            },
            "adjudications": audit_ref("adjudications.csv"),
        },
        "criterion_promotion_gate": {
            "passed": True,
            "promotion_authorized": True,
            "thresholds": {
                "minimum_lower_95_bound": judge_audit.PROMOTION_MIN_LOWER_BOUND,
                "maximum_weighted_human_unclear_rate": (
                    judge_audit.PROMOTION_MAX_UNCLEAR_RATE
                ),
                "maximum_weighted_automated_unclear_rate": (
                    judge_audit.PROMOTION_MAX_AUTOMATED_UNCLEAR_RATE
                ),
                "minimum_inter_and_intra_rater_kappa": (
                    judge_audit.PROMOTION_MIN_KAPPA
                ),
                "primary_items_per_stratum": judge_audit.DEFAULT_PER_STRATUM,
                "duplicates_per_annotator": judge_audit.DEFAULT_DUPLICATES,
                "annotators": judge_audit.DEFAULT_ANNOTATORS,
            },
            "design_passed": True, "multifactor_allocation_passed": True,
            "failures": [],
        },
        "scoring_parameters": {"bootstrap_replicates": 2000, "seed": 20260801},
    }
    judge["result_sha256"] = stable_json_hash(judge)
    expected_judge = deepcopy(judge)
    monkeypatch.setattr(
        data_lock.judge_audit,
        "score_multi_audit",
        lambda *args, **kwargs: deepcopy(expected_judge),
    )
    judge_path = tmp_path / "judge.json"
    _write(judge_path, judge)
    exclusions = {
        "schema_version": 1,
        "artifact_type": "confirmatory_exclusion_decisions",
        "status": "approved_outcome_blind",
        "campaign_plan_sha256s": {
            "variance_stage": variance_campaign["plan_sha256"],
            "baseline_stage": baseline_campaign["plan_sha256"],
        },
        "policy_frozen_at_utc": "2026-08-01T23:59:00Z",
        "policy_frozen_by": "policy-reviewer",
        "allowed_reason_codes": list(data_lock.OBJECTIVE_EXCLUSION_REASON_CODES),
        "decision_reviewed_at_utc": "2026-08-02T01:00:00Z",
        "reviewed_by": "reviewer-2",
        "excluded_job_ids": [],
        "decisions": [],
    }
    exclusions_path = tmp_path / "exclusions.json"
    _write(exclusions_path, exclusions)
    return {
        "variance_campaign": variance_campaign_path,
        "baseline_campaign": baseline_campaign_path,
        "judge": judge_path, "variance": variance_path,
        "exclusions": exclusions_path,
        "output": tmp_path / "data-lock.json",
    }


def _build_lock(paths: dict[str, Path]):
    return data_lock.build_data_lock(
        variance_campaign_manifest_path=paths["variance_campaign"],
        baseline_campaign_manifest_path=paths["baseline_campaign"],
        judge_criterion_path=paths["judge"],
        variance_selection_path=paths["variance"], exclusions_path=paths["exclusions"],
        output_path=paths["output"],
    )


def test_data_lock_pins_all_gates_artifacts_and_is_private(lock_fixture) -> None:
    result = _build_lock(lock_fixture)
    assert result["status"] == "locked"
    assert result["completeness"]["all_selected_lineage_artifacts_reverified"] is True
    assert result["completed_jobs"]["final_by_experiment"] == {"part0": 2, "part1": 2, "part2": 1}
    assert result["analysis_sources"]["part2_variance_pilot"].endswith("sample_size_only")
    timing = result["collection_timing"]
    assert timing["policy"] == (
        "stage_separated_timing_without_posthoc_cross_stage_cutoff"
    )
    assert timing["exact_attempt_timestamps_preserved"] is True
    assert timing["stage_part_temporal_confounding"]["flagged"] is True
    assert (
        timing["stage_part_temporal_confounding"]
        ["baseline_start_minus_variance_finish_seconds"]
        > 7 * 24 * 60 * 60
    )
    assert set(timing["per_stage"]["variance_stage"]["parts"]) == {
        "part0", "part1", "part2"
    }
    assert set(timing["per_stage"]["baseline_stage"]["parts"]) == {"part2"}
    assert result["deferred_non_lockable"]["part2_sensitivity"] == {
        "status": "deferred_pending_native_execution_lineage",
        "publication_outcome_eligible": False,
        "reason": (
            "No confirmatory sensitivity result is lockable until its native "
            "execution artifacts and campaign lineage exist."
        ),
    }
    pilot_refs = [
        row for row in result["artifacts"]
        if row["stage"] == "part2_variance_pilot"
    ]
    assert pilot_refs and all(
        row["publication_outcome_eligible"] is False for row in pilot_refs
    )
    assert result["exclusions"]["excluded_scientific_job_ids"] == []
    assert result["exclusions"]["included_scientific_job_ids"] == [
        "production-part0-target-1", "production-part0-target-2",
        "production-part1-target-1", "production-part1-target-2", "part2-baseline"
    ]
    assert result["data_lock_sha256"] == stable_json_hash({key: value for key, value in result.items() if key != "data_lock_sha256"})
    assert stat.S_IMODE(lock_fixture["output"].stat().st_mode) == 0o600


def test_data_lock_rejects_reversed_scientific_attempt_timestamps(
    lock_fixture,
) -> None:
    baseline = json.loads(
        lock_fixture["baseline_campaign"].read_text(encoding="utf-8")
    )
    baseline["jobs"][0]["attempts"][0]["finished_at_utc"] = (
        "2026-08-19T23:00:00Z"
    )
    _write(lock_fixture["baseline_campaign"], baseline)

    with pytest.raises(
        data_lock.ConfirmatoryDataLockError,
        match="finished before it started",
    ):
        _build_lock(lock_fixture)


def test_actual_data_lock_output_materializes_native_part0_units(
    lock_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    _build_lock(lock_fixture)
    roots = [
        ("harmful-1", "harmful-cluster-1", "harmful"),
        ("harmful-2", "harmful-cluster-2", "harmful"),
        ("control-1", "control-cluster-1", "control"),
        ("control-2", "control-cluster-2", "control"),
    ]

    def native_records(path: Path):
        records = []
        for root_id, cluster_id, arm in roots:
            for language in confirmatory_estimators.PART0_LANGUAGES:
                for block in confirmatory_estimators.PART0_BLOCKS:
                    records.append(
                        {
                            "execution_mode": "production",
                            "analysis_eligible": True,
                            "base_prompt_id": root_id,
                            "semantic_cluster_id": cluster_id,
                            "arm": arm,
                            "language": language,
                            "block": block,
                            "status": "SCORED",
                            "judge_label": (
                                "REFUSAL" if arm == "harmful" else "COMPLIANCE"
                            ),
                        }
                    )
        return records

    monkeypatch.setattr(
        confirmatory_estimators.part0_runner,
        "_load_result_records",
        native_records,
    )
    lock_path = lock_fixture["output"]
    units = confirmatory_estimators.materialize_native_units(
        "part0",
        data_lock_path=lock_path,
        data_lock_sha256=sha256_file(lock_path),
    )
    assert units["source_data_lock_sha256"] == sha256_file(lock_path)
    assert units["frozen_system_ids"] == ["target-1", "target-2"]
    assert units["generation_blocks"] == [1]
    assert len(units["rows"]) == 2 * 4 * 3
    part2_confirmatory._verify_sealed_artifact(units, hash_field="artifact_sha256")


def test_data_lock_refuses_missing_failed_or_tampered_gates(lock_fixture) -> None:
    lock_fixture["judge"].unlink()
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="judge criterion"):
        _build_lock(lock_fixture)


def test_data_lock_requires_current_automated_unclear_gate(lock_fixture) -> None:
    judge = json.loads(lock_fixture["judge"].read_text(encoding="utf-8"))
    del judge["criterion_promotion_gate"]["thresholds"][
        "maximum_weighted_automated_unclear_rate"
    ]
    judge["result_sha256"] = stable_json_hash(
        {key: value for key, value in judge.items() if key != "result_sha256"}
    )
    _write(lock_fixture["judge"], judge)

    with pytest.raises(
        data_lock.ConfirmatoryDataLockError,
        match="automated-unclear gate",
    ):
        _build_lock(lock_fixture)


def test_data_lock_cohort_estimand_enforces_current_panel_and_developer_minima() -> None:
    targets = [
        {
            "id": f"target-{index:02d}",
            "upstream_provider": (
                f"developer-{index % 8}"
                if index < 24
                else f"historical-developer-{index % 3}"
            ),
        }
        for index in range(30)
    ]
    campaign = {
        "targets": targets,
        "cohorts": [
            {
                "id": "current_sota",
                "target_ids": [target["id"] for target in targets[:24]],
            },
            {
                "id": "historical",
                "target_ids": [target["id"] for target in targets[24:]],
            },
        ],
    }
    metadata = data_lock._validate_cohort_estimand(campaign)
    assert sum(row["cohort_id"] == "current_sota" for row in metadata) == 24
    assert sum(row["cohort_id"] == "historical" for row in metadata) == 6

    small = deepcopy(campaign)
    small["targets"] = small["targets"][:11]
    small["cohorts"] = [
        {
            "id": "current_sota",
            "target_ids": [target["id"] for target in small["targets"]],
        }
    ]
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="at least 12"):
        data_lock._validate_cohort_estimand(small)

    historical_only = deepcopy(campaign)
    historical_only["cohorts"] = [
        {
            "id": "historical",
            "target_ids": [target["id"] for target in historical_only["targets"]],
        }
    ]
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="at least 12"):
        data_lock._validate_cohort_estimand(historical_only)


def test_data_lock_recomputes_judge_gate_and_rejects_fabricated_authorization(
    lock_fixture,
) -> None:
    judge = json.loads(lock_fixture["judge"].read_text(encoding="utf-8"))
    judge["overall"] = {"fabricated_metric": 1.0}
    judge["criterion_promotion_gate"]["promotion_authorized"] = True
    judge["result_sha256"] = stable_json_hash(
        {key: value for key, value in judge.items() if key != "result_sha256"}
    )
    _write(lock_fixture["judge"], judge)
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="recomputation"):
        _build_lock(lock_fixture)


def test_data_lock_requires_two_exact_complete_union_campaigns(lock_fixture) -> None:
    baseline = json.loads(
        lock_fixture["baseline_campaign"].read_text(encoding="utf-8")
    )
    baseline["target_selection"]["mode"] = "shard"
    _write(lock_fixture["baseline_campaign"], baseline)
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="complete_union"):
        _build_lock(lock_fixture)


def test_data_lock_rejects_duplicate_exclusions(lock_fixture) -> None:
    exclusions = json.loads(lock_fixture["exclusions"].read_text(encoding="utf-8"))
    evidence_path = lock_fixture["exclusions"].parent / "objective-evidence.json"
    evidence_hash = _write(evidence_path, {"objective_qc": "fixture"})
    exclusions["excluded_job_ids"] = ["part2-baseline", "part2-baseline"]
    exclusions["decisions"] = [
        {
            "job_id": "part2-baseline",
            "reason_code": "prespecified_protocol_deviation",
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_hash,
            "decided_by": "reviewer",
            "decided_at_utc": "2026-08-02T01:00:00Z",
            "outcome_blind": True,
        },
        {
            "job_id": "part2-baseline",
            "reason_code": "prespecified_protocol_deviation",
            "evidence_path": str(evidence_path),
            "evidence_sha256": evidence_hash,
            "decided_by": "reviewer",
            "decided_at_utc": "2026-08-02T01:00:00Z",
            "outcome_blind": True,
        },
    ]
    _write(lock_fixture["exclusions"], exclusions)
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="unique"):
        _build_lock(lock_fixture)
    exclusions["excluded_job_ids"] = []
    exclusions["decisions"] = []
    _write(lock_fixture["exclusions"], exclusions)


def test_data_lock_exclusions_are_pre_outcome_coded_and_audited(lock_fixture) -> None:
    exclusions = json.loads(lock_fixture["exclusions"].read_text(encoding="utf-8"))
    evidence = lock_fixture["exclusions"].parent / "objective-qc.json"
    evidence_hash = _write(evidence, {"check": "protocol-version-mismatch"})
    decision = {
        "job_id": "part2-baseline",
        "reason_code": "prespecified_protocol_deviation",
        "evidence_path": str(evidence),
        "evidence_sha256": evidence_hash,
        "decided_by": "outcome-blind-reviewer",
        "decided_at_utc": "2026-08-02T01:00:00Z",
        "outcome_blind": True,
    }
    exclusions["excluded_job_ids"] = ["part2-baseline"]
    exclusions["decisions"] = [decision]
    _write(lock_fixture["exclusions"], exclusions)
    result = _build_lock(lock_fixture)
    assert result["exclusions"]["included_scientific_job_ids"] == [
        "production-part0-target-1", "production-part0-target-2",
        "production-part1-target-1", "production-part1-target-2"
    ]
    assert result["exclusions"]["excluded_scientific_job_ids"] == ["part2-baseline"]
    [audit_row] = result["exclusions"]["excluded_scientific_job_audit_table"]
    assert audit_row["reason_code"] == "prespecified_protocol_deviation"
    assert audit_row["evidence_sha256"] == evidence_hash

    lock_fixture["output"].unlink()
    exclusions["policy_frozen_at_utc"] = "2026-08-02T00:01:00Z"
    _write(lock_fixture["exclusions"], exclusions)
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="before outcome"):
        _build_lock(lock_fixture)

    exclusions["policy_frozen_at_utc"] = "2026-08-01T23:59:00Z"
    exclusions["decisions"][0]["reason"] = "exclude because result is inconvenient"
    _write(lock_fixture["exclusions"], exclusions)
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="schema"):
        _build_lock(lock_fixture)


def test_fixed_judge_lineage_requires_exact_part0_campaign_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = tmp_path / "registry.json"
    registry_hash = _write(registry, {})
    manifest_path = tmp_path / "confirmatory_audit_input.manifest.json"
    _write(manifest_path, {})
    key_path = tmp_path / "audit_key.csv"
    key_path.write_text(
        "item_kind,source_file,target_id\n"
        f"primary,{manifest_path},target-a\n"
        f"primary,{manifest_path},target-b\n",
        encoding="utf-8",
    )
    jobs = []
    sources = []
    records = []
    for suffix in ("a", "b"):
        run = tmp_path / f"run-{suffix}"
        jobs.append(
            {
                "id": f"part0-{suffix}", "experiment": "part0",
                "stage": "production", "target_id": f"target-{suffix}",
                "provider": "inference_hub", "route": f"vendor/model-{suffix}",
                "output_dir": str(run),
            }
        )
        sources.append(
            {
                "run_directory": str(run), "provider": "inference_hub",
                "model": f"vendor/model-{suffix}",
                "registry_path": str(registry), "registry_sha256": registry_hash,
            }
        )
        records.append({"target_id": f"target-{suffix}"})
    monkeypatch.setattr(
        adapter,
        "load_confirmatory_audit_input",
        lambda path: (
            deepcopy(records),
            {
                "source_runs": deepcopy(sources),
                "manifest_sha256": "a" * 64,
                "records_sha256": "b" * 64,
            },
        ),
    )
    campaign = {
        "jobs": jobs,
        "inputs": {"part0_registry": {"path": str(registry), "sha256": registry_hash}},
    }
    result = data_lock._validate_judge_campaign_lineage(
        {"audit_key": {"path": str(key_path)}}, campaign
    )
    assert result["part0_production_jobs"] == 2
    assert result["target_count"] == 2
    sources[0]["model"] = "vendor/wrong"
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="route/registry"):
        data_lock._validate_judge_campaign_lineage(
            {"audit_key": {"path": str(key_path)}}, campaign
        )


def test_request_ledger_exactly_reconciles_native_dispatch_hashes(tmp_path: Path) -> None:
    attempts = tmp_path / "part0_attempts.jsonl"
    logger = DurableAttemptLogger(attempts, experiment="part_0_confirmatory")
    hashes = ["a" * 64, "b" * 64]
    for index, request_hash in enumerate(hashes, start=1):
        logger.append(
            provider="inference_hub", model="vendor/model", unit_id=f"unit-{index}",
            unit={"dispatch_request_sha256": request_hash}, attempt=1,
            max_attempts=1, prompt_text="[REDACTED]", outcome="success",
        )
    artifacts = [{"path": str(attempts)}]
    ledger = {
        "records": [
            {"role": "discovery", "request_sha256": "c" * 64}
        ] + [
            {"role": "part0_subject", "request_sha256": value}
            for value in hashes
        ]
    }
    result = data_lock._validate_attempt_reconciliation(artifacts, ledger)
    assert result["scientific_and_smoke_physical_attempts"] == 2
    ledger["records"][-1]["request_sha256"] = "d" * 64
    with pytest.raises(data_lock.ConfirmatoryDataLockError, match="hash multiset"):
        data_lock._validate_attempt_reconciliation(artifacts, ledger)
