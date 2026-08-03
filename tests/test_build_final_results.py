import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from analysis.build_final_results import (
    FinalResultsError,
    _cross_axis,
    _part1_manifest,
    _part2,
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


TARGET = "subject.alpha"
SUBJECT = {
    "target_id": TARGET, "upstream_provider": "alpha_&_lab",
    "model": "alpha_model%{v1}#", "route": "region/alpha-model",
}
JUDGE = {"target_id": "judge.route", "upstream_provider": "judge", "model": "judge-model", "route": "region/judge-model"}


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    value["evidence_sha256"] = _sha256_json(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


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


def _raw(model: str, content: str = "X") -> dict[str, Any]:
    return {"id": "r", "model": model, "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}


def _part0_fixture(root: Path, *, human_validated: bool = False) -> Path:
    schedule = [
        {"trial_id": f"trial-{language}", "root_id": "root-1", "language": language}
        for language in ("english", "chinese", "russian")
    ]
    payloads: list[dict[str, Any]] = []
    trial_ids = [row["trial_id"] for row in schedule]
    for trial in schedule:
        raw = _raw(SUBJECT["route"], "I cannot help.")
        payloads.append({
            "event": "subject_response_retained", "target_id": TARGET,
            "trial_id": trial["trial_id"], "requested_route": SUBJECT["route"],
            "response_model": SUBJECT["route"], "model_identity_valid": True,
            "raw_response": raw, "raw_response_sha256": _sha256_json(raw),
        })
    judge_raw = _raw(JUDGE["route"], "labels")
    payloads.append({
        "event": "judge_batch_retained", "target_id": TARGET, "batch_id": "batch-1",
        "trial_ids": trial_ids, "requested_route": JUDGE["route"],
        "response_model": JUDGE["route"], "model_identity_valid": True,
        "raw_response": judge_raw, "raw_response_sha256": _sha256_json(judge_raw),
    })
    for trial in schedule:
        payloads.append({
            "event": "unit_completed", "target_id": TARGET,
            "trial_id": trial["trial_id"], "root_id": trial["root_id"],
            "language": trial["language"], "outcome": "REFUSAL",
            "judge_dispatched": True, "judge_batch_id": "batch-1",
        })
    raw_ref = _write_journal(root / "part0/private/raw_responses/subject.alpha.jsonl", payloads)
    manifest = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part0_accelerated_private_panel",
        "complete": True,
        "stimulus_reconstruction": "archived_english_source_prompt_crossed_with_explicit_response_language_instruction;legacy_translated_input_not_retained",
        "human_validation_complete": human_validated,
        "selected_roots_per_language": 1, "languages": ["english", "chinese", "russian"],
        "schedule": schedule, "subject_routes": [SUBJECT], "judge": JUDGE,
        "summary": {
            "subject_transport_failures": 0, "judge_failed_units": 0,
            "subject_model_identity_mismatches": 0, "judge_model_identity_mismatches": 0,
        },
        "journals": {"raw_responses": {TARGET: raw_ref}},
    })
    path = root / "part0/private/manifest.json"
    _write_json(path, manifest)
    return path


def _part1_fixture(
    root: Path, *, count: int, name: str, balanced: bool = True
) -> Path:
    payloads = []
    cells = [(game, domain) for game in GAMES for domain in DOMAINS]
    for index in range(count):
        game, domain = cells[index % len(cells)] if balanced else cells[0]
        raw = _raw(SUBJECT["route"], "X")
        payloads.append({
            "schema_version": 1, "artifact_type": "inference_hub_part1_raw_response",
            "target_id": TARGET, "upstream_provider": SUBJECT["upstream_provider"],
            "model": SUBJECT["model"], "requested_route": SUBJECT["route"],
            "response_model": SUBJECT["route"], "model_identity_valid": True,
            "trial_id": f"trial-{index:03d}", "root_id": f"root-{index:03d}",
            "game": game, "domain": domain,
            "counterbalance_id": "CB_X_FIRST", "parsed_action": "X", "format_valid": True,
            "raw_response": raw, "raw_response_sha256": _sha256_json(raw),
        })
    raw_ref = _write_journal(root / f"{name}/private/raw_responses/subject.alpha.jsonl", payloads)
    full = count == 384
    manifest = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part1_large_n_exploratory_panel",
        "complete": True, "executed_trial_count_per_subject": count,
        "trial_limit": None if full else count, "subject_routes": [SUBJECT],
        "summary": {
            "failed_without_response": 0, "response_model_identity_mismatches": 0,
        },
        "journals": {"raw_responses": {TARGET: raw_ref}},
    })
    path = root / f"{name}/private/manifest.json"
    _write_json(path, manifest)
    return path


def _part2_fixture(root: Path, *, tamper: bool = False) -> Path:
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
    models = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part2_sanitized_model_metrics",
        "rows": [{
            "target_id": TARGET, "upstream_provider": SUBJECT["upstream_provider"],
            "model": SUBJECT["model"], "trajectory_count": trajectory_count,
            "eligible_trajectory_count": trajectory_count, "complete_matched_panel": True,
            "total_scheduled_agent_days": 73, "total_invalid_count": 3,
            "trajectory_level_95_percent_t_intervals": intervals,
        }],
    })
    trajectories = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part2_sanitized_trajectory_metrics",
        "rows": [{"target_id": TARGET, "trajectory_index": index, "operationally_eligible": True} for index in range(trajectory_count)],
    })
    model_path = root / "part2/sanitized/model_metrics.json"
    trajectory_path = root / "part2/sanitized/trajectory_metrics.json"
    _write_json(model_path, models)
    _write_json(trajectory_path, trajectories)
    manifest = _seal({
        "schema_version": 1, "artifact_type": "inference_hub_part2_corrected_matched_panel",
        "complete": True, "subject_routes": [SUBJECT],
        "part2_contract": {
            "independent_trajectories": trajectory_count, "society_size": 5, "days": 12,
            "resource_capacity": 50, "option_a_private_gain": 1,
            "option_b_private_gain": 2, "option_b_reserve_cost": 2,
        },
        "summary": {"identity_mismatch_count": 0, "transport_failure_count": 0},
        "sanitized_artifacts": {
            "model_metrics": {"path": str(model_path.resolve()), "file_sha256": _sha256_file(model_path), "evidence_sha256": models["evidence_sha256"]},
            "trajectory_metrics": {"path": str(trajectory_path.resolve()), "file_sha256": _sha256_file(trajectory_path), "evidence_sha256": trajectories["evidence_sha256"]},
        },
    })
    manifest_path = root / "part2/private/manifest.json"
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
    assert "95\\% $t_{7}$ intervals" in part2
    assert "0.625 [0.500, 0.750]" in part2
    assert "0.375 [0.250, 0.500]" in part2
    assert "4/8 [21.5, 78.5]\\%" in part2
    assert "0.875 [0.750, 1.000] & 3/73" in part2

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
