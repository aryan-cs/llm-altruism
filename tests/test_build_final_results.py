import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from analysis.build_final_results import (
    FinalResultsError,
    _combine_part1_overlay,
    _cross_axis,
    _overlay_contract,
    _part0,
    _part0_overlay,
    _part1_manifest,
    _part2,
    _part2_overlay,
    _part2_valid_trajectory_intervals,
    _parser,
    _preferred_part1,
    _reject_text_keys,
    _select_figure_rows,
    _self_hash,
    _sha256_file,
    _sha256_json,
    build_final_results,
)
from experiments.part1.confirmatory_design import DOMAINS, GAMES
from experiments.misc.inference_hub_part1_panel import _SOURCE_PATHS as PART1_SOURCE_PATHS


TARGET = "subject.alpha"
SUBJECT = {
    "target_id": TARGET, "upstream_provider": "alpha_&_lab",
    "model": "alpha_model%{v1}#", "route": "region/alpha-model",
}
SECOND_TARGET = "subject.beta"
SECOND_SUBJECT = {
    "target_id": SECOND_TARGET, "upstream_provider": "beta_lab",
    "model": "beta-model", "route": "region/beta-model",
}
JUDGE = {"target_id": "judge.route", "upstream_provider": "judge", "model": "judge-model", "route": "region/judge-model"}


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value["evidence_sha256"] = _sha256_json(value)
    return value


def _retirement_audit(
    *, target_id: str, migration_path: str, prior: str, resumed: str,
    retirement_id: str = "retire_test",
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "inference_hub_offline_target_retirement",
        "part": "part1",
        "target_id": target_id,
        "retirement_id": retirement_id,
        "provenance": "offline_target_bound_operational_retirement",
        "network_dispatch_performed_by_tool": False,
        "behavioral_outcomes_assigned_by_tool": False,
        "source_artifact_hash_migrations": [{
            "path": migration_path,
            "prior_sha256": prior,
            "resumed_runner_sha256": resumed,
        }],
    }
    audit["record_sha256"] = _sha256_json(audit)
    return audit


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def _mutate_manifest(path: Path, mutation: Any) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("evidence_sha256")
    mutation(value)
    _write_json(path, _seal(value))


def _write_journal(path: Path, payloads: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    rows = []
    for payload in payloads:
        row = {**payload, "previous_record_sha256": previous}
        row["record_sha256"] = _sha256_json(row)
        previous = row["record_sha256"]
        rows.append(row)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    os.chmod(path, 0o600)
    return {
        "path": str(path.resolve()), "record_count": len(rows),
        "tail_record_sha256": previous, "file_sha256": _sha256_file(path),
    }


def _raw(model: str, content: str | None = "X") -> dict[str, Any]:
    return {"id": "r", "model": model, "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def _part0_fixture(
    root: Path, *, human_validated: bool = False,
    subjects: tuple[dict[str, Any], ...] = (SUBJECT,), complete: bool = True,
    failed_targets: tuple[str, ...] = (), name: str = "part0",
    operational_failure_targets: tuple[str, ...] = (),
    semantic_invalid_targets: tuple[str, ...] = (),
    empty_visible_response_targets: tuple[str, ...] = (),
) -> Path:
    schedule = [
        {"trial_id": f"trial-{language}", "root_id": "root-1", "language": language}
        for language in ("english", "chinese", "russian")
    ]
    trial_ids = [row["trial_id"] for row in schedule]
    raw_refs: dict[str, Any] = {}
    for subject in subjects:
        target_id = subject["target_id"]
        if target_id in failed_targets:
            continue
        payloads: list[dict[str, Any]] = []
        if target_id in operational_failure_targets:
            for trial in schedule:
                payloads.append({
                    "schema_version": 1,
                    "artifact_type": "inference_hub_part0_private_record",
                    "event": "unit_completed", "target_id": target_id,
                    "trial_id": trial["trial_id"], "root_id": trial["root_id"],
                    "language": trial["language"], "outcome": "INVALID",
                    "judge_dispatched": False,
                    "operational_failure": {"failure_code": "transport_exhausted"},
                })
            raw_refs[target_id] = _write_journal(
                root / f"{name}/private/raw_responses/{target_id}.jsonl", payloads
            )
            continue
        for trial in schedule:
            raw = _raw(
                subject["route"],
                None if target_id in empty_visible_response_targets else "I cannot help.",
            )
            payloads.append({
                "schema_version": 1,
                "artifact_type": "inference_hub_part0_private_record",
                "event": "subject_response_retained", "target_id": target_id,
                "trial_id": trial["trial_id"], "root_id": trial["root_id"],
                "language": trial["language"], "requested_route": subject["route"],
                "response_model": subject["route"], "model_identity_valid": True,
                "raw_response": raw, "raw_response_sha256": _sha256_json(raw),
            })
        if target_id not in empty_visible_response_targets:
            judge_raw = _raw(JUDGE["route"], "labels")
            payloads.append({
                "schema_version": 1,
                "artifact_type": "inference_hub_part0_private_record",
                "event": "judge_batch_retained", "target_id": target_id,
                "batch_id": "batch-1", "trial_ids": trial_ids,
                "requested_route": JUDGE["route"], "response_model": JUDGE["route"],
                "model_identity_valid": True, "raw_response": judge_raw,
                "raw_response_sha256": _sha256_json(judge_raw),
            })
        for trial in schedule:
            empty_visible_response = target_id in empty_visible_response_targets
            payloads.append({
                "schema_version": 1,
                "artifact_type": "inference_hub_part0_private_record",
                "event": "unit_completed", "target_id": target_id,
                "trial_id": trial["trial_id"], "root_id": trial["root_id"],
                "language": trial["language"],
                "outcome": (
                    "INVALID"
                    if target_id in semantic_invalid_targets or empty_visible_response
                    else "REFUSAL"
                ),
                "judge_dispatched": not empty_visible_response,
                "judge_batch_id": None if empty_visible_response else "batch-1",
                "judge_label": None if empty_visible_response else "REFUSAL",
                "subject_visible_response_present": not empty_visible_response,
            })
        raw_refs[target_id] = _write_journal(
            root / f"{name}/private/raw_responses/{target_id}.jsonl", payloads
        )
    manifest = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part0_accelerated_private_panel",
        "complete": complete,
        "stimulus_reconstruction": "archived_english_source_prompt_crossed_with_explicit_response_language_instruction;legacy_translated_input_not_retained",
        "human_validation_complete": human_validated,
        "selected_roots_per_language": 1, "languages": ["english", "chinese", "russian"],
        "schedule": schedule, "subject_routes": list(subjects), "judge": JUDGE,
        "summary": {
            "subject_transport_failures": (
                len(failed_targets) + len(schedule) * len(operational_failure_targets)
            ),
            "judge_failed_units": 0,
            "subject_model_identity_mismatches": 0, "judge_model_identity_mismatches": 0,
        },
        "journals": {"raw_responses": raw_refs},
    })
    path = root / f"{name}/private/manifest.json"
    _write_json(path, manifest)
    return path


def _part1_fixture(
    root: Path, *, count: int, name: str, balanced: bool = True,
    subjects: tuple[dict[str, Any], ...] = (SUBJECT,), complete: bool = True,
    failed_targets: tuple[str, ...] = (),
    operational_failure_targets: tuple[str, ...] = (),
    semantic_invalid_targets: tuple[str, ...] = (),
    identity_failure_targets: tuple[str, ...] = (),
) -> Path:
    cells = [(game, domain) for game in GAMES for domain in DOMAINS]
    raw_refs: dict[str, Any] = {}
    for subject in subjects:
        target_id = subject["target_id"]
        if target_id in failed_targets:
            continue
        payloads = []
        for index in range(count):
            game, domain = cells[index % len(cells)] if balanced else cells[0]
            if target_id in operational_failure_targets:
                payloads.append({
                    "schema_version": 1,
                    "artifact_type": "inference_hub_part1_raw_response",
                    "target_id": target_id,
                    "upstream_provider": subject["upstream_provider"],
                    "model": subject["model"], "requested_route": subject["route"],
                    "response_model": None, "model_identity_valid": False,
                    "trial_id": f"trial-{index:03d}",
                    "root_id": f"root-{index:03d}", "game": game,
                    "domain": domain, "counterbalance_id": "CB_X_FIRST",
                    "parsed_action": None, "format_valid": False,
                    "raw_response": None, "raw_response_sha256": None,
                    "failure": {"failure_code": "transport_exhausted"},
                })
                continue
            response_route = (
                "region/unexpected-model"
                if target_id in identity_failure_targets else subject["route"]
            )
            raw = _raw(response_route, "X")
            parsed_action = (
                None if target_id in semantic_invalid_targets else "X"
            )
            payloads.append({
                "schema_version": 1, "artifact_type": "inference_hub_part1_raw_response",
                "target_id": target_id,
                "upstream_provider": subject["upstream_provider"],
                "model": subject["model"], "requested_route": subject["route"],
                "response_model": response_route,
                "model_identity_valid": target_id not in identity_failure_targets,
                "trial_id": f"trial-{index:03d}", "root_id": f"root-{index:03d}",
                "game": game, "domain": domain,
                "counterbalance_id": "CB_X_FIRST", "parsed_action": parsed_action,
                "format_valid": parsed_action == "X",
                "raw_response": raw, "raw_response_sha256": _sha256_json(raw),
            })
        raw_refs[target_id] = _write_journal(
            root / f"{name}/private/raw_responses/{target_id}.jsonl", payloads
        )
    full = count == 384
    manifest = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part1_large_n_exploratory_panel",
        "complete": complete, "executed_trial_count_per_subject": count,
        "trial_limit": None if full else count, "subject_routes": list(subjects),
        "summary": {
            "failed_without_response": (
                len(failed_targets) + count * len(operational_failure_targets)
            ),
            "response_model_identity_mismatches": (
                count * len(identity_failure_targets)
            ),
        },
        "journals": {"raw_responses": raw_refs},
    })
    path = root / f"{name}/private/manifest.json"
    _write_json(path, manifest)
    return path


def _part2_fixture(
    root: Path, *, tamper: bool = False,
    subjects: tuple[dict[str, Any], ...] = (SUBJECT,), complete: bool = True,
    failed_targets: tuple[str, ...] = (), name: str = "part2",
    operational_failures: dict[str, str] | None = None,
    semantic_invalid_targets: tuple[str, ...] = (),
) -> Path:
    operational_failures = operational_failures or {}
    trajectory_count = 8
    intervals = {
        "aurc": {"mean": 0.625, "lower": 0.5, "upper": 0.75, "n": trajectory_count, "method": "trajectory_t_95"},
        "aupc": {"mean": 0.875, "lower": 0.75, "upper": 1.0, "n": trajectory_count, "method": "trajectory_t_95"},
        "restraint_rate": {"mean": 0.375, "lower": 0.25, "upper": 0.5, "n": trajectory_count, "method": "trajectory_t_95"},
        "reserve_nondepletion": {
            "mean": 0.5, "lower": 0.2152, "upper": 0.7848, "n": trajectory_count,
            "successes": 4, "method": "trajectory_wilson_95",
        },
        "final_reserve": {"mean": 10.0, "lower": 8.0, "upper": 12.0, "n": trajectory_count, "method": "trajectory_t_95"},
        "population_retention": {"mean": 0.8, "lower": 0.7, "upper": 0.9, "n": trajectory_count, "method": "trajectory_t_95"},
        "cumulative_private_payoff": {"mean": 10.0, "lower": 8.0, "upper": 12.0, "n": trajectory_count, "method": "trajectory_t_95"},
        "cumulative_group_payoff": {"mean": 10.0, "lower": 8.0, "upper": 12.0, "n": trajectory_count, "method": "trajectory_t_95"},
    }
    model_rows = []
    trajectory_rows = []
    for subject in subjects:
        target_id = subject["target_id"]
        if target_id in failed_targets:
            continue
        failure_kind = operational_failures.get(target_id)
        identity_count = trajectory_count if failure_kind == "identity" else 0
        transport_count = trajectory_count if failure_kind == "transport" else 0
        eligible_count = 0 if failure_kind else trajectory_count
        semantic_invalid = target_id in semantic_invalid_targets
        model_rows.append({
            "target_id": target_id,
            "upstream_provider": subject["upstream_provider"],
            "model": subject["model"], "trajectory_count": trajectory_count,
            "eligible_trajectory_count": eligible_count,
            "complete_matched_panel": failure_kind is None,
            "total_scheduled_agent_days": 80,
            "total_invalid_count": trajectory_count if (failure_kind or semantic_invalid) else 0,
            "total_identity_mismatch_count": identity_count,
            "total_transport_failure_count": transport_count,
            "trajectory_level_95_percent_t_intervals": intervals,
        })
        trajectory_rows.extend({
            "target_id": target_id, "trajectory_index": index,
            "operationally_eligible": failure_kind is None,
            "identity_mismatch_count": 1 if failure_kind == "identity" else 0,
            "transport_failure_count": 1 if failure_kind == "transport" else 0,
            "invalid_count": 1 if (
                failure_kind or semantic_invalid
            ) else 0,
            "scheduled_agent_days": 10,
            "aurc": 0.625,
            "aupc": 0.875,
            "restraint_rate": 0.375,
            "reserve_nondepletion": index < 4,
            "final_reserve": 10.0,
            "population_retention": 0.8,
            "cumulative_private_payoff": 10.0,
            "cumulative_group_payoff": 10.0,
        } for index in range(trajectory_count))
    models = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part2_sanitized_model_metrics",
        "rows": model_rows,
    })
    trajectories = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part2_sanitized_trajectory_metrics",
        "rows": trajectory_rows,
    })
    model_path = root / f"{name}/sanitized/model_metrics.json"
    trajectory_path = root / f"{name}/sanitized/trajectory_metrics.json"
    _write_json(model_path, models)
    _write_json(trajectory_path, trajectories)
    manifest = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part2_corrected_matched_panel",
        "complete": complete, "subject_routes": list(subjects),
        "part2_contract": {
            "independent_trajectories": trajectory_count, "society_size": 5, "days": 12,
            "resource_capacity": 50, "option_a_private_gain": 1,
            "option_b_private_gain": 2, "option_b_reserve_cost": 2,
        },
        "summary": {
            "identity_mismatch_count": sum(
                trajectory_count for kind in operational_failures.values()
                if kind == "identity"
            ),
            "transport_failure_count": (
                len(failed_targets) + sum(
                    trajectory_count for kind in operational_failures.values()
                    if kind == "transport"
                )
            ),
        },
        "sanitized_artifacts": {
            "model_metrics": {"path": str(model_path.resolve()), "file_sha256": _sha256_file(model_path), "evidence_sha256": models["evidence_sha256"]},
            "trajectory_metrics": {"path": str(trajectory_path.resolve()), "file_sha256": _sha256_file(trajectory_path), "evidence_sha256": trajectories["evidence_sha256"]},
        },
    })
    manifest_path = root / f"{name}/private/manifest.json"
    _write_json(manifest_path, manifest)
    if tamper:
        model_path.write_text(model_path.read_text() + " ", encoding="utf-8")
    return manifest_path


def _panel(root: Path) -> Path:
    path = root / "panel.json"
    ids = [TARGET] + [f"subject.{index:02d}" for index in range(1, 24)]
    _write_json(path, {"schema_version": 1, "panel_id": "matched-24", "subject_target_ids": ids})
    return path


def test_builder_emits_sanitized_scoped_outputs_and_blocks_unready_cross_axis(tmp_path: Path) -> None:
    artifact = build_final_results(
        part0_manifest=_part0_fixture(tmp_path),
        part1_full_manifests=[_part1_fixture(tmp_path, count=384, name="part1-full")],
        part1_n96_manifests=[
            _part1_fixture(tmp_path, count=96, name="part1-n96"),
            _part1_fixture(tmp_path, count=12, name="part1-n12"),
        ],
        part2_manifest=_part2_fixture(tmp_path), panel_path=_panel(tmp_path),
        output_dir=tmp_path / "final", bootstrap_seed=7,
    )
    assert artifact["cross_axis"]["status"] == "not_emitted_fail_closed"
    assert not (tmp_path / "final/cross_axis_spearman.csv").exists()
    assert (tmp_path / "final/figures/model_axis_summary.pdf").is_file()
    expected_figure_paths = {
        "figures/matched_panel_axis_summary.pdf",
        "figures/matched_panel_axis_summary.png",
        "figures/model_axis_summary.pdf",
        "figures/model_axis_summary.png",
        "figures/part1_scope_facets.pdf",
        "figures/part1_scope_facets.png",
    }
    figure_references = {
        reference["path"]: reference for reference in artifact["outputs"]["figures"]
    }
    assert set(figure_references) == expected_figure_paths
    for relative, reference in figure_references.items():
        assert reference["file_sha256"] == _sha256_file(tmp_path / "final" / relative)
    for filename in (
        "part0_results_table.tex", "part1_results_table.tex", "part2_results_table.tex",
    ):
        assert (tmp_path / "final" / filename).is_file()
    part1_csv = (tmp_path / "final/part1_model_rates.csv").read_text()
    assert "full_384" in part1_csv and "balanced_partial" in part1_csv
    assert artifact["parameters"]["part1_balanced_partial_root_counts"] == [12, 96]
    preferred = [row for row in artifact["part1"] if row["preferred_for_descriptive_outputs"]]
    assert len(preferred) == 1
    assert preferred[0]["scope"] == "full_384"
    assert preferred[0]["root_count"] == 384
    part0_csv = (tmp_path / "final/part0_model_rates.csv").read_text()
    assert "reconstructed_response_language_condition_not_translated_prompt" in part0_csv
    public = (tmp_path / "final/final_results.json").read_text()
    assert "I cannot help" not in public and '"raw_response"' not in public
    assert artifact["evidence_sha256"] == _self_hash(artifact)


def test_publication_tables_are_exact_scoped_escaped_private_and_hash_bound(
    tmp_path: Path,
) -> None:
    output = tmp_path / "publication"
    artifact = build_final_results(
        part0_manifest=_part0_fixture(tmp_path),
        part1_full_manifests=[_part1_fixture(tmp_path, count=384, name="publication-full")],
        part1_n96_manifests=[
            _part1_fixture(tmp_path, count=96, name="publication-n96"),
            _part1_fixture(tmp_path, count=12, name="publication-n12"),
        ],
        part2_manifest=_part2_fixture(tmp_path), panel_path=_panel(tmp_path),
        output_dir=output, bootstrap_seed=19,
    )

    part0 = (output / "part0_results_table.tex").read_text(encoding="utf-8")
    assert "reconstructed response-language conditions" in part0
    assert "not translated-prompt" in part0
    assert "no benign controls" in part0
    assert "alpha\\_model\\%\\{v1\\}\\# & alpha\\_\\&\\_lab & 1" in part0
    assert "100.0\\% [100.0, 100.0]" in part0
    assert part0.count("100.0\\% [20.7, 100.0]") == 3
    assert "0/0 & Exploratory" in part0

    part1 = (output / "part1_results_table.tex").read_text(encoding="utf-8")
    assert "are never pooled" in part1
    assert part1.count("alpha\\_model\\%\\{v1\\}\\#") == 3
    assert "Balanced partial & 12 & 12/12 (1/stratum) & 100.0\\% [--]" in part1
    assert "Balanced partial & 96 & 12/12 (8/stratum) & 100.0\\% [100.0, 100.0]" in part1
    assert "Full & 384 & 12/12 (32/stratum) & 100.0\\% [100.0, 100.0]" in part1
    assert part1.count("384/384 (100.0\\%) & 0 & Exploratory") == 1

    part2 = (output / "part2_results_table.tex").read_text(encoding="utf-8")
    assert "95\\% $t$ intervals" in part2
    assert "0.625 [0.625, 0.625]" in part2
    assert "0.375 [0.375, 0.375]" in part2
    assert "4/8 [21.5, 78.5]\\%" in part2
    assert "0.875 [0.875, 0.875] & 0/80" in part2

    combined = part0 + part1 + part2
    for forbidden in (
        "I cannot help", "region/alpha-model", "raw_response", "prompt_text",
        "visible_response", "reasoning",
    ):
        assert forbidden not in combined
    for key, filename in (
        ("latex_part0_table", "part0_results_table.tex"),
        ("latex_part1_table", "part1_results_table.tex"),
        ("latex_part2_table", "part2_results_table.tex"),
    ):
        reference = artifact["outputs"][key]
        assert reference["path"] == filename
        assert reference["file_sha256"] == _sha256_file(output / filename)
    assert artifact["evidence_sha256"] == _self_hash(artifact)


def test_figure_selection_uses_p0_p2_intersection_and_keeps_all_part1_scopes() -> None:
    def identity(target_id: str) -> dict[str, Any]:
        return {
            "target_id": target_id, "upstream_provider": f"provider-{target_id}",
            "model": f"model-{target_id}",
        }

    part0 = [identity("p0-only"), identity("matched-b"), identity("matched-a")]
    part2 = [identity("p2-only"), identity("matched-a"), identity("matched-b")]
    part1 = [
        {**identity("matched-a"), "scope": "balanced_partial", "root_count": 96},
        {**identity("matched-b"), "scope": "balanced_partial", "root_count": 12},
        {**identity("matched-b"), "scope": "full_384", "root_count": 384},
        {**identity("part1-only"), "scope": "balanced_partial", "root_count": 96},
    ]

    matched, broad = _select_figure_rows(part0, part1, part2)
    assert {row["target_id"] for row in matched} == {"matched-a", "matched-b"}
    assert all(row["target_id"] not in {"p0-only", "p2-only"} for row in matched)
    selected_b = next(row for row in matched if row["target_id"] == "matched-b")
    assert selected_b["part1"]["scope"] == "full_384"
    assert selected_b["part1"]["root_count"] == 384
    assert [(row["target_id"], row["scope"], row["root_count"]) for row in broad] == [
        ("matched-b", "balanced_partial", 12),
        ("matched-a", "balanced_partial", 96),
        ("part1-only", "balanced_partial", 96),
        ("matched-b", "full_384", 384),
    ]


def test_part2_tampering_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FinalResultsError, match="file hash changed"):
        _part2(_part2_fixture(tmp_path, tamper=True))


def test_balanced_n12_partial_is_reportable_but_never_cross_axis_eligible(
    tmp_path: Path,
) -> None:
    partial_path = _part1_fixture(tmp_path, count=12, name="part1-slow-n12")
    rows, _identities, binding = _part1_manifest(
        partial_path, scope="balanced_partial", bootstrap_seed=11
    )
    assert binding["scope"] == "balanced_partial"
    assert rows[0]["root_count"] == 12
    assert rows[0]["roots_per_stratum"] == 1
    assert rows[0]["descriptive_reportable"] is True
    assert rows[0]["paper_eligible"] is False

    artifact = build_final_results(
        part0_manifest=_part0_fixture(tmp_path),
        part1_full_manifests=[], part1_n96_manifests=[partial_path],
        part2_manifest=_part2_fixture(tmp_path), panel_path=_panel(tmp_path),
        output_dir=tmp_path / "final-n12", bootstrap_seed=11,
    )
    assert artifact["parameters"]["part1_balanced_partial_root_counts"] == [12]
    assert artifact["cross_axis"]["status"] == "not_emitted_fail_closed"
    assert any(
        reason.startswith("part1_full_384_missing:")
        for reason in artifact["cross_axis"]["reasons"]
    )


def test_partial_root_count_must_be_balanced_multiple_of_twelve(tmp_path: Path) -> None:
    path = _part1_fixture(tmp_path, count=13, name="part1-bad-n13")
    with pytest.raises(FinalResultsError, match="wrong root count/scope"):
        _part1_manifest(path, scope="balanced_partial", bootstrap_seed=1)

    imbalanced = _part1_fixture(
        tmp_path, count=12, name="part1-bad-imbalanced-n12", balanced=False
    )
    with pytest.raises(FinalResultsError, match="all 12 game-domain strata equally"):
        _part1_manifest(imbalanced, scope="balanced_partial", bootstrap_seed=1)


def test_full_part1_is_preferred_over_partial_regardless_of_input_order() -> None:
    partial = {"target_id": TARGET, "scope": "balanced_partial", "root_count": 96}
    smaller = {"target_id": TARGET, "scope": "balanced_partial", "root_count": 12}
    full = {"target_id": TARGET, "scope": "full_384", "root_count": 384}
    assert _preferred_part1([partial, full, smaller])[TARGET] is full
    assert _preferred_part1([full, partial, smaller])[TARGET] is full
    assert _preferred_part1([smaller, partial])[TARGET] is partial


def test_legacy_n96_cli_flag_remains_an_alias_for_balanced_partials() -> None:
    common = [
        "--part0-manifest", "part0.json", "--part2-manifest", "part2.json",
        "--output-dir", "output",
    ]
    legacy = _parser().parse_args([*common, "--part1-n96-manifest", "n96.json"])
    current = _parser().parse_args([*common, "--part1-partial-manifest", "n12.json"])
    assert legacy.part1_n96_manifest == [Path("n96.json")]
    assert current.part1_n96_manifest == [Path("n12.json")]


def test_replacement_cli_flags_are_repeatable() -> None:
    args = _parser().parse_args([
        "--part0-manifest", "p0.json",
        "--part0-replacement-manifest", "p0-a.json",
        "--part0-replacement-manifest", "p0-b.json",
        "--part0-unavailable-target", "subject.p0",
        "--part1-partial-manifest", "p1.json",
        "--part1-replacement-manifest", "p1-r.json",
        "--part1-unavailable-target", "subject.p1",
        "--part2-manifest", "p2.json",
        "--part2-replacement-manifest", "p2-r.json",
        "--part2-unavailable-target", "subject.p2",
        "--output-dir", "output",
    ])
    assert args.part0_replacement_manifest == [Path("p0-a.json"), Path("p0-b.json")]
    assert args.part1_replacement_manifest == [Path("p1-r.json")]
    assert args.part2_replacement_manifest == [Path("p2-r.json")]
    assert args.part0_unavailable_target == ["subject.p0"]
    assert args.part1_unavailable_target == ["subject.p1"]
    assert args.part2_unavailable_target == ["subject.p2"]


def test_incomplete_primaries_accept_exact_complete_target_replacements(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    p0_primary = _part0_fixture(
        tmp_path, subjects=subjects, complete=False,
        failed_targets=(SECOND_TARGET,), name="overlay-p0-primary",
    )
    p0_replacement = _part0_fixture(
        tmp_path, subjects=(SECOND_SUBJECT,), name="overlay-p0-replacement",
    )
    p1_primary = _part1_fixture(
        tmp_path, count=12, subjects=subjects, complete=False,
        failed_targets=(SECOND_TARGET,), name="overlay-p1-primary",
    )
    p1_replacement = _part1_fixture(
        tmp_path, count=12, subjects=(SECOND_SUBJECT,),
        name="overlay-p1-replacement",
    )
    p2_primary = _part2_fixture(
        tmp_path, subjects=subjects, complete=False,
        failed_targets=(SECOND_TARGET,), name="overlay-p2-primary",
    )
    p2_replacement = _part2_fixture(
        tmp_path, subjects=(SECOND_SUBJECT,), name="overlay-p2-replacement",
    )
    artifact = build_final_results(
        part0_manifest=p0_primary,
        part0_replacement_manifests=[p0_replacement],
        part1_full_manifests=[], part1_n96_manifests=[p1_primary],
        part1_replacement_manifests=[p1_replacement],
        part2_manifest=p2_primary,
        part2_replacement_manifests=[p2_replacement],
        panel_path=_panel(tmp_path), output_dir=tmp_path / "overlay-final",
        bootstrap_seed=31,
    )
    for part in ("part0", "part1", "part2"):
        assert [row["target_id"] for row in artifact[part]] == [TARGET, SECOND_TARGET]
    for binding in (
        artifact["bindings"]["part0"], artifact["bindings"]["part1"][0],
        artifact["bindings"]["part2"],
    ):
        assert binding["overlay_schema_version"] == 1
        assert binding["primary"]["complete"] is False
        assert binding["primary"]["retained_target_ids"] == [TARGET]
        assert binding["replaced_target_ids"] == [SECOND_TARGET]
        assert binding["replacements"][0]["replacement_target_ids"] == [SECOND_TARGET]
        assert binding["primary"]["file_sha256"]
        assert binding["replacements"][0]["file_sha256"]
    public = (tmp_path / "overlay-final/final_results.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in public
    assert "region/alpha-model" not in public
    assert "region/beta-model" not in public
    assert artifact["evidence_sha256"] == _self_hash(artifact)


def test_part0_overlay_rejects_identity_contract_duplicates_and_incomplete_replacements(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    primary = _part0_fixture(
        tmp_path, subjects=subjects, complete=False,
        failed_targets=(SECOND_TARGET,), name="overlay-failure-primary",
    )
    replacement = _part0_fixture(
        tmp_path, subjects=(SECOND_SUBJECT,), name="overlay-good-replacement",
    )
    with pytest.raises(FinalResultsError, match="duplicated"):
        _part0_overlay(primary, [replacement, replacement])

    incomplete = _part0_fixture(
        tmp_path, subjects=(SECOND_SUBJECT,), complete=False,
        name="overlay-incomplete-replacement",
    )
    with pytest.raises(FinalResultsError, match="incomplete"):
        _part0_overlay(primary, [incomplete])

    mismatched_subject = {**SECOND_SUBJECT, "route": "region/wrong-beta-model"}
    identity_mismatch = _part0_fixture(
        tmp_path, subjects=(mismatched_subject,), name="overlay-identity-replacement",
    )
    with pytest.raises(FinalResultsError, match="identity differs"):
        _part0_overlay(primary, [identity_mismatch])

    contract_mismatch = _part0_fixture(
        tmp_path, subjects=(SECOND_SUBJECT,), name="overlay-contract-replacement",
    )
    _mutate_manifest(
        contract_mismatch,
        lambda value: value["schedule"][0].update({"root_id": "changed-root"}),
    )
    with pytest.raises(FinalResultsError, match="schedule or scientific contract differs"):
        _part0_overlay(primary, [contract_mismatch])


def test_overlay_cannot_hide_a_nonreplaced_primary_failure(tmp_path: Path) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    primary = _part0_fixture(
        tmp_path, subjects=subjects, complete=False,
        failed_targets=(SECOND_TARGET,), name="overlay-retained-failure-primary",
    )
    wrong_target_replacement = _part0_fixture(
        tmp_path, subjects=(SUBJECT,), name="overlay-wrong-target-replacement",
    )
    with pytest.raises(FinalResultsError, match="Raw journal binding is absent"):
        _part0_overlay(primary, [wrong_target_replacement])


def test_part1_and_part2_overlay_helpers_reject_complete_primary_replacement(
    tmp_path: Path,
) -> None:
    p1_primary = _part1_fixture(
        tmp_path, count=12, subjects=(SUBJECT, SECOND_SUBJECT),
        name="overlay-complete-p1-primary",
    )
    p1_replacement = _part1_fixture(
        tmp_path, count=12, subjects=(SECOND_SUBJECT,),
        name="overlay-complete-p1-replacement",
    )
    with pytest.raises(FinalResultsError, match="incomplete primary"):
        _combine_part1_overlay(
            [], [p1_primary], [p1_replacement], bootstrap_seed=1,
        )

    p2_primary = _part2_fixture(
        tmp_path, subjects=(SUBJECT, SECOND_SUBJECT),
        name="overlay-complete-p2-primary",
    )
    p2_replacement = _part2_fixture(
        tmp_path, subjects=(SECOND_SUBJECT,), name="overlay-complete-p2-replacement",
    )
    with pytest.raises(FinalResultsError, match="incomplete primary"):
        _part2_overlay(p2_primary, [p2_replacement])


def test_part1_overlay_matches_configured_contract_not_target_count_worker_cap(
    tmp_path: Path,
) -> None:
    primary = _part1_fixture(
        tmp_path, count=12, subjects=(SUBJECT, SECOND_SUBJECT), complete=False,
        failed_targets=(SECOND_TARGET,), name="overlay-worker-primary",
    )
    replacement = _part1_fixture(
        tmp_path, count=12, subjects=(SECOND_SUBJECT,),
        name="overlay-worker-replacement",
    )
    _mutate_manifest(primary, lambda value: value.update({
        "execution_contract": {
            "configured_max_workers": 8, "effective_max_workers": 8,
            "max_attempts_per_trial": 3,
            "initial_exponential_backoff_seconds": 1.0,
            "response_parser": "exact-final-token",
        },
    }))
    _mutate_manifest(replacement, lambda value: value.update({
        "execution_contract": {
            "configured_max_workers": 1, "effective_max_workers": 1,
            "max_attempts_per_trial": 6,
            "initial_exponential_backoff_seconds": 2.0,
            "response_parser": "exact-final-token",
        },
    }))
    rows, _identities, bindings = _combine_part1_overlay(
        [], [primary], [replacement], bootstrap_seed=5,
    )
    assert [row["target_id"] for row in rows] == [TARGET, SECOND_TARGET]
    assert bindings[0]["replaced_target_ids"] == [SECOND_TARGET]

    _mutate_manifest(replacement, lambda value: value["execution_contract"].update({
        "response_parser": "different-parser",
    }))
    with pytest.raises(FinalResultsError, match="must match exactly one primary"):
        _combine_part1_overlay(
            [], [primary], [replacement], bootstrap_seed=5,
        )


def test_explicit_operationally_unavailable_targets_are_omitted_with_sanitized_provenance(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    panel = _panel(tmp_path)
    panel_value = json.loads(panel.read_text(encoding="utf-8"))
    panel_value["subject_target_ids"][1] = SECOND_TARGET
    _write_json(panel, panel_value)
    artifact = build_final_results(
        part0_manifest=_part0_fixture(
            tmp_path, subjects=subjects, complete=False,
            operational_failure_targets=(SECOND_TARGET,),
            name="unavailable-p0-primary",
        ),
        part0_unavailable_targets=[SECOND_TARGET],
        part1_full_manifests=[],
        part1_n96_manifests=[_part1_fixture(
            tmp_path, count=12, subjects=subjects, complete=False,
            operational_failure_targets=(SECOND_TARGET,),
            name="unavailable-p1-primary",
        )],
        part1_unavailable_targets=[SECOND_TARGET],
        part2_manifest=_part2_fixture(
            tmp_path, subjects=subjects, complete=False,
            operational_failures={SECOND_TARGET: "transport"},
            name="unavailable-p2-primary",
        ),
        part2_unavailable_targets=[SECOND_TARGET],
        panel_path=panel, output_dir=tmp_path / "unavailable-final",
        bootstrap_seed=41,
    )
    for part in ("part0", "part1", "part2"):
        assert [row["target_id"] for row in artifact[part]] == [TARGET]
    for binding in (
        artifact["bindings"]["part0"], artifact["bindings"]["part1"][0],
        artifact["bindings"]["part2"],
    ):
        assert binding["unavailable_target_ids"] == [SECOND_TARGET]
        assert binding["replaced_target_ids"] == []
        failure = binding["unavailable_target_failures"][0]
        assert failure["target_id"] == SECOND_TARGET
        assert failure["total_failure_count"] > 0
        assert failure["provenance"] == "validated_target_bound_primary_evidence"
        assert binding["primary"]["retained_target_ids"] == [TARGET]
    assert artifact["cross_axis"]["status"] == "not_emitted_fail_closed"
    assert any(
        SECOND_TARGET in reason for reason in artifact["cross_axis"]["reasons"]
    )
    public = (tmp_path / "unavailable-final/final_results.json").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "failure_code", "transport_exhausted", '"raw_response":', '"response_text":',
        "region/beta-model",
    ):
        assert forbidden not in public


def test_semantic_or_format_invalid_outputs_cannot_be_declared_unavailable(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    p0 = _part0_fixture(
        tmp_path, subjects=subjects, complete=False,
        semantic_invalid_targets=(SECOND_TARGET,), name="semantic-p0",
    )
    with pytest.raises(FinalResultsError, match="no target-bound operational or identity"):
        _part0_overlay(p0, [], [SECOND_TARGET])

    p1 = _part1_fixture(
        tmp_path, count=12, subjects=subjects, complete=False,
        semantic_invalid_targets=(SECOND_TARGET,), name="semantic-p1",
    )
    with pytest.raises(FinalResultsError, match="no target-bound operational or identity"):
        _combine_part1_overlay(
            [], [p1], [], bootstrap_seed=1,
            unavailable_target_ids=[SECOND_TARGET],
        )

    p2 = _part2_fixture(
        tmp_path, subjects=subjects, complete=False,
        semantic_invalid_targets=(SECOND_TARGET,), name="semantic-p2",
    )
    with pytest.raises(FinalResultsError, match="no target-bound operational or identity"):
        _part2_overlay(p2, [], [SECOND_TARGET])


def test_part2_invalid_trajectories_are_not_rewarded_as_environmental_restraint() -> None:
    rows = [
        {
            "invalid_count": 60, "aurc": 1.0, "aupc": 1.0,
            "restraint_rate": 0.0, "reserve_nondepletion": True,
            "final_reserve": 50, "population_retention": 1.0,
            "cumulative_private_payoff": 0, "cumulative_group_payoff": 0,
        },
        {
            "invalid_count": 0, "aurc": 0.4, "aupc": 0.8,
            "restraint_rate": 0.25, "reserve_nondepletion": False,
            "final_reserve": 0, "population_retention": 0.6,
            "cumulative_private_payoff": 25, "cumulative_group_payoff": 0,
        },
    ]
    intervals = _part2_valid_trajectory_intervals(rows)
    assert intervals is not None
    assert intervals["aurc"]["mean"] == 0.4
    assert intervals["reserve_nondepletion"]["successes"] == 0
    assert intervals["aurc"]["n"] == 1
    assert _part2_valid_trajectory_intervals(rows[:1]) is None


def test_part0_retains_empty_visible_responses_as_invalid_without_judging(
    tmp_path: Path,
) -> None:
    manifest = _part0_fixture(
        tmp_path,
        complete=False,
        empty_visible_response_targets=(TARGET,),
        name="empty-visible-p0",
    )

    rows, identities, _binding = _part0(
        manifest,
        target_ids={TARGET},
        require_complete=False,
        require_summary=False,
    )

    assert set(identities) == {TARGET}
    assert len(rows) == 1
    assert rows[0]["overall_refusal"]["estimate"] == 0.0
    assert sum(condition["invalid_count"] for condition in rows[0]["conditions"]) == 3


def test_valid_target_cannot_be_excluded_because_another_target_failed(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    p0 = _part0_fixture(
        tmp_path, subjects=subjects, complete=False,
        operational_failure_targets=(TARGET,), name="other-failed-p0",
    )
    with pytest.raises(FinalResultsError, match="no target-bound operational or identity"):
        _part0_overlay(p0, [], [SECOND_TARGET])

    p1 = _part1_fixture(
        tmp_path, count=12, subjects=subjects, complete=False,
        operational_failure_targets=(TARGET,), name="other-failed-p1",
    )
    with pytest.raises(FinalResultsError, match="no target-bound operational or identity"):
        _combine_part1_overlay(
            [], [p1], [], bootstrap_seed=1,
            unavailable_target_ids=[SECOND_TARGET],
        )

    p2 = _part2_fixture(
        tmp_path, subjects=subjects, complete=False,
        operational_failures={TARGET: "transport"}, name="other-failed-p2",
    )
    with pytest.raises(FinalResultsError, match="no target-bound operational or identity"):
        _part2_overlay(p2, [], [SECOND_TARGET])


def test_unavailable_targets_reject_unknown_duplicates_complete_and_replacement_overlap(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    incomplete = _part0_fixture(
        tmp_path, subjects=subjects, complete=False,
        operational_failure_targets=(SECOND_TARGET,), name="unavailable-invalid-p0",
    )
    with pytest.raises(FinalResultsError, match="duplicated"):
        _part0_overlay(incomplete, [], [SECOND_TARGET, SECOND_TARGET])
    with pytest.raises(FinalResultsError, match="absent from the primary"):
        _part0_overlay(incomplete, [], ["subject.unknown"])

    complete = _part0_fixture(
        tmp_path, subjects=subjects, name="unavailable-complete-p0",
    )
    with pytest.raises(FinalResultsError, match="incomplete primary"):
        _part0_overlay(complete, [], [SECOND_TARGET])

    replacement = _part0_fixture(
        tmp_path, subjects=(SECOND_SUBJECT,), name="unavailable-overlap-p0",
    )
    with pytest.raises(FinalResultsError, match="both replaced and unavailable"):
        _part0_overlay(incomplete, [replacement], [SECOND_TARGET])


def test_part1_unavailable_target_must_resolve_to_exactly_one_primary(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    n12 = _part1_fixture(
        tmp_path, count=12, subjects=subjects, complete=False,
        operational_failure_targets=(SECOND_TARGET,), name="ambiguous-p1-n12",
    )
    n96 = _part1_fixture(
        tmp_path, count=96, subjects=subjects, complete=False,
        operational_failure_targets=(SECOND_TARGET,), name="ambiguous-p1-n96",
    )
    with pytest.raises(FinalResultsError, match="exactly one primary"):
        _combine_part1_overlay(
            [], [n12, n96], [], bootstrap_seed=2,
            unavailable_target_ids=[SECOND_TARGET],
        )


def test_part1_unavailable_filter_preserves_primary_bootstrap_subject_index(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    baseline = _part1_fixture(
        tmp_path, count=12, subjects=subjects, name="bootstrap-baseline-p1",
    )
    baseline_rows, _identities, _binding = _part1_manifest(
        baseline, scope="balanced_partial", bootstrap_seed=71,
    )
    expected = next(
        row for row in baseline_rows if row["target_id"] == SECOND_TARGET
    )["cooperation"]

    primary = _part1_fixture(
        tmp_path, count=12, subjects=subjects, complete=False,
        operational_failure_targets=(TARGET,), name="bootstrap-unavailable-p1",
    )
    rows, _identities, bindings = _combine_part1_overlay(
        [], [primary], [], bootstrap_seed=71,
        unavailable_target_ids=[TARGET],
    )
    assert [row["target_id"] for row in rows] == [SECOND_TARGET]
    assert rows[0]["cooperation"] == expected
    assert bindings[0]["unavailable_target_ids"] == [TARGET]


def test_identity_failures_are_valid_unavailable_evidence_and_overlap_still_fails(
    tmp_path: Path,
) -> None:
    subjects = (SUBJECT, SECOND_SUBJECT)
    p1_primary = _part1_fixture(
        tmp_path, count=12, subjects=subjects, complete=False,
        identity_failure_targets=(SECOND_TARGET,), name="identity-unavailable-p1",
    )
    rows, _identities, bindings = _combine_part1_overlay(
        [], [p1_primary], [], bootstrap_seed=3,
        unavailable_target_ids=[SECOND_TARGET],
    )
    assert [row["target_id"] for row in rows] == [TARGET]
    assert bindings[0]["unavailable_target_failures"][0]["failure_counts"] == {
        "response_model_identity_mismatch": 12,
    }
    p1_replacement = _part1_fixture(
        tmp_path, count=12, subjects=(SECOND_SUBJECT,),
        name="identity-overlap-p1-replacement",
    )
    with pytest.raises(FinalResultsError, match="both replaced and unavailable"):
        _combine_part1_overlay(
            [], [p1_primary], [p1_replacement], bootstrap_seed=3,
            unavailable_target_ids=[SECOND_TARGET],
        )

    p2_primary = _part2_fixture(
        tmp_path, subjects=subjects, complete=False,
        operational_failures={SECOND_TARGET: "identity"},
        name="identity-unavailable-p2",
    )
    rows, _identities, binding = _part2_overlay(
        p2_primary, [], [SECOND_TARGET],
    )
    assert [row["target_id"] for row in rows] == [TARGET]
    assert binding["unavailable_target_failures"][0]["failure_counts"] == {
        "response_model_identity_mismatch": 8,
    }

    p2_complete = _part2_fixture(
        tmp_path, subjects=subjects, name="complete-unavailable-p2",
    )
    with pytest.raises(FinalResultsError, match="incomplete primary"):
        _part2_overlay(p2_complete, [], [SECOND_TARGET])


def test_overlay_contract_ignores_only_operational_repair_fields() -> None:
    common_inputs = {"registry": {"sha256": "r"}, "compatibility": {"sha256": "c"}}
    p0_primary = {
        "execution_contract": {
            "max_workers": 32, "max_attempts_per_request": 3,
            "initial_exponential_backoff_seconds": 1.0,
            "judge_batch_parser": "exact",
        },
        "input_artifacts": {**common_inputs, "cross_axis_panel": {"sha256": "panel"}},
    }
    p0_repair = {
        "execution_contract": {
            "max_workers": 2, "max_attempts_per_request": 6,
            "initial_exponential_backoff_seconds": 2.0,
            "judge_batch_parser": "exact",
        },
        "input_artifacts": common_inputs,
    }
    assert _overlay_contract(p0_primary, part="part0") == _overlay_contract(
        p0_repair, part="part0"
    )
    p0_repair["execution_contract"]["judge_batch_parser"] = "changed"
    assert _overlay_contract(p0_primary, part="part0") != _overlay_contract(
        p0_repair, part="part0"
    )

    p2_primary = {
        "execution_contract": {
            "trajectory_workers": 24, "max_transport_attempts": 3,
            "initial_exponential_backoff_seconds": 1.0,
            "participant_workers": 5, "identity_check": "exact",
        },
    }
    p2_repair = {
        "execution_contract": {
            "trajectory_workers": 1, "max_transport_attempts": 10,
            "initial_exponential_backoff_seconds": 4.0,
            "participant_workers": 5, "identity_check": "exact",
        },
    }
    assert _overlay_contract(p2_primary, part="part2") == _overlay_contract(
        p2_repair, part="part2"
    )
    p2_repair["execution_contract"]["participant_workers"] = 1
    assert _overlay_contract(p2_primary, part="part2") != _overlay_contract(
        p2_repair, part="part2"
    )


def test_part1_overlay_normalizes_only_signed_retirement_runner_migration(
    tmp_path: Path,
) -> None:
    primary_path = _part1_fixture(
        tmp_path, count=12, name="retired-stratified-primary",
        subjects=(SUBJECT, SECOND_SUBJECT), complete=False,
        operational_failure_targets=(SECOND_TARGET,),
    )
    replacement_path = _part1_fixture(
        tmp_path, count=12, name="old-complete-deepseek-replacement",
        subjects=(SUBJECT,),
    )
    wrapper_path = (
        Path(__file__).parents[1]
        / "experiments/misc/inference_hub_part1_stratified_panel.py"
    ).resolve()
    runner_path = Path(PART1_SOURCE_PATHS[0]).resolve()
    current_sources = {
        str(path.resolve()): _sha256_file(path.resolve())
        for path in (*PART1_SOURCE_PATHS, wrapper_path)
    }
    prior_runner_hash = "a" * 64

    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    primary["source_artifacts"] = dict(current_sources)
    primary["target_retirements"] = [_retirement_audit(
        target_id=SECOND_TARGET, migration_path=str(runner_path),
        prior=prior_runner_hash, resumed=current_sources[str(runner_path)],
    )]
    primary.pop("evidence_sha256")
    _write_json(primary_path, _seal(primary))

    replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
    replacement["source_artifacts"] = {
        **current_sources, str(runner_path): prior_runner_hash,
    }
    replacement.pop("evidence_sha256")
    _write_json(replacement_path, _seal(replacement))

    rows, _identities, bindings = _combine_part1_overlay(
        [], [primary_path], [replacement_path], bootstrap_seed=3,
        unavailable_target_ids=[SECOND_TARGET],
    )
    assert [row["target_id"] for row in rows] == [TARGET]
    assert bindings[0]["replaced_target_ids"] == [TARGET]
    assert bindings[0]["unavailable_target_ids"] == [SECOND_TARGET]

    replacement = json.loads(replacement_path.read_text(encoding="utf-8"))
    unrelated_path = str(Path(PART1_SOURCE_PATHS[1]).resolve())
    replacement["source_artifacts"][unrelated_path] = "b" * 64
    replacement.pop("evidence_sha256")
    _write_json(replacement_path, _seal(replacement))
    with pytest.raises(FinalResultsError, match="must match exactly one primary"):
        _combine_part1_overlay(
            [], [primary_path], [replacement_path], bootstrap_seed=3,
            unavailable_target_ids=[SECOND_TARGET],
        )


def test_overlay_rejects_chained_or_inconsistent_retirement_migrations(
    tmp_path: Path,
) -> None:
    runner_path = Path(PART1_SOURCE_PATHS[0]).resolve()
    current_hash = _sha256_file(runner_path)
    manifest = {
        "complete": False,
        "source_artifacts": {
            str(path.resolve()): _sha256_file(path.resolve())
            for path in PART1_SOURCE_PATHS
        },
        "target_retirements": [
            _retirement_audit(
                target_id=TARGET, migration_path=str(runner_path),
                prior="a" * 64, resumed=current_hash, retirement_id="retire_one",
            ),
            _retirement_audit(
                target_id=SECOND_TARGET, migration_path=str(runner_path),
                prior="b" * 64, resumed=current_hash, retirement_id="retire_two",
            ),
        ],
    }
    with pytest.raises(FinalResultsError, match="chained or inconsistent"):
        _overlay_contract(manifest, part="part1")

    unrelated = Path(PART1_SOURCE_PATHS[1]).resolve()
    manifest["target_retirements"] = [_retirement_audit(
        target_id=TARGET, migration_path=str(unrelated),
        prior="a" * 64, resumed=_sha256_file(unrelated),
    )]
    with pytest.raises(FinalResultsError, match="chained or inconsistent"):
        _overlay_contract(manifest, part="part1")


def test_forbidden_text_fields_are_rejected_recursively() -> None:
    with pytest.raises(FinalResultsError, match="forbidden field"):
        _reject_text_keys({"safe": [{"prompt_text": "secret"}]})


def test_cross_axis_emits_only_exact_eligible_matched_24(tmp_path: Path) -> None:
    ids = [f"subject.{index:02d}" for index in range(24)]
    panel = tmp_path / "panel.json"
    _write_json(panel, {"schema_version": 1, "subject_target_ids": ids})
    identities = {target_id: ("provider", f"model-{index}", f"route-{index}") for index, target_id in enumerate(ids)}
    p0 = [{"target_id": target_id, "paper_eligible": True, "overall_refusal": {"estimate": index / 23}} for index, target_id in enumerate(ids)]
    p1 = [{"target_id": target_id, "paper_eligible": True, "scope": "full_384", "cooperation": {"estimate": index / 23}} for index, target_id in enumerate(ids)]
    p2 = [{"target_id": target_id, "paper_eligible": True, "trajectory_level_95_percent_t_intervals": {"restraint_rate": {"mean": index / 23}, "aurc": {"mean": (23 - index) / 23}}} for index, target_id in enumerate(ids)]
    result = _cross_axis(panel_path=panel, part0=p0, part1=p1, part2=p2, identities=(identities, identities, identities))
    assert result["status"] == "emitted_exact_matched_24"
    assert len(result["spearman_pairs"]) == 6
    assert result["spearman_pairs"][0]["rho"] == pytest.approx(1.0)
