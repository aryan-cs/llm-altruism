import json
from pathlib import Path
from typing import Any

import pytest

from analysis.build_final_results import build_final_results
from analysis.build_paper_headlines import (
    OUTPUT_ARTIFACT_TYPE,
    PaperHeadlineError,
    _self_hash,
    _seal,
    build_paper_headlines,
    main,
)
from test_build_final_results import (
    _panel,
    _part0_fixture,
    _part1_fixture,
    _part2_fixture,
)


def _rate(successes: int, total: int) -> dict[str, Any]:
    estimate = successes / total
    return {
        "estimate": estimate,
        "lower": max(0.0, estimate - 0.1),
        "upper": min(1.0, estimate + 0.1),
        "n": total,
        "method": "Wilson score 95%; independent prompt roots within condition",
    }


def _p0_row(
    target_id: str,
    model: str,
    refusals: tuple[int, int, int],
    *,
    unclear: tuple[int, int, int] = (0, 0, 0),
    invalid: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, Any]:
    roots = 4
    total_refusal = sum(refusals)
    return {
        "target_id": target_id,
        "upstream_provider": "provider",
        "model": model,
        "scope": "reconstructed_response_language_conditions",
        "root_count_per_condition": roots,
        "overall_refusal": {
            "estimate": total_refusal / (3 * roots),
            "lower": 0.0,
            "upper": 1.0,
            "n_rows": 3 * roots,
            "n_clusters": roots,
            "method": "prompt-root cluster bootstrap percentile 95%",
        },
        "conditions": [
            {
                "response_language_condition": language,
                "condition_definition": (
                    "English source request plus explicit requested response language; "
                    "not a translated-prompt condition"
                ),
                "refusal": _rate(refusals[index], roots),
                "unclear_count": unclear[index],
                "invalid_count": invalid[index],
            }
            for index, language in enumerate(("english", "chinese", "russian"))
        ],
        "human_validation_complete": False,
        "paper_eligible": False,
    }


def _p1_row(
    target_id: str,
    model: str,
    roots: int,
    successes: int,
    *,
    invalid: int,
) -> dict[str, Any]:
    estimate = successes / roots
    return {
        "target_id": target_id,
        "upstream_provider": "provider",
        "model": model,
        "scope": "full_384" if roots == 384 else "balanced_partial",
        "root_count": roots,
        "format_valid_count": roots - invalid,
        "format_invalid_count": invalid,
        "cooperation": {
            "estimate": estimate,
            "lower": max(0.0, estimate - 0.1),
            "upper": min(1.0, estimate + 0.1),
            "n": roots,
            "method": "within-game-domain stratified root bootstrap percentile 95%",
        },
        "paper_eligible": roots == 384,
        "descriptive_reportable": True,
        "stratum_count": 12,
        "roots_per_stratum": roots // 12,
        "preferred_for_descriptive_outputs": True,
    }


def _trajectory_interval(mean: float, *, successes: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "mean": mean,
        "lower": max(0.0, mean - 0.1),
        "upper": min(1.0, mean + 0.1),
        "n": 8,
        "method": "trajectory_t_95",
    }
    if successes is not None:
        value["successes"] = successes
        value["method"] = "trajectory_wilson_95"
    return value


def _p2_row(
    target_id: str,
    model: str,
    *,
    aurc: float,
    restraint: float,
    aupc: float,
    nondepleted: int,
    scheduled: int,
    invalid: int,
) -> dict[str, Any]:
    return {
        "target_id": target_id,
        "upstream_provider": "provider",
        "model": model,
        "trajectory_count": 8,
        "valid_trajectory_count": 8,
        "protocol_invalid_trajectory_count": 0,
        "metric_status": "estimable_from_fully_valid_trajectories",
        "trajectory_level_95_percent_t_intervals": {
            "aurc": _trajectory_interval(aurc),
            "restraint_rate": _trajectory_interval(restraint),
            "aupc": _trajectory_interval(aupc),
            "reserve_nondepletion": _trajectory_interval(
                nondepleted / 8, successes=nondepleted
            ),
        },
        "total_scheduled_agent_days": scheduled,
        "total_invalid_count": invalid,
        "paper_eligible": False,
    }


def _source() -> dict[str, Any]:
    base_binding = {
        "manifest_path": "manifest.json",
        "path_scope": "input_manifest_basename_only",
        "file_sha256": "a" * 64,
        "evidence_sha256": "b" * 64,
    }
    return _seal({
        "schema_version": 1,
        "artifact_type": "prosocial_readiness_final_sanitized_results",
        "privacy_contract": {
            "contains_prompt_text": False,
            "contains_response_text": False,
            "contains_reasoning": False,
            "contains_raw_responses": False,
            "contains_routes": False,
        },
        "parameters": {
            "part0_condition_interpretation": (
                "reconstructed response-language conditions over English source requests; "
                "not translated-prompt conditions"
            ),
            "part1_scopes_pooled": False,
            "part1_full_root_count": 384,
            "part1_balanced_partial_root_counts": [12, 96],
            "part2_uncertainty_unit": "independent_trajectory",
            "bootstrap_seed": 1,
            "bootstrap_replicates": 5_000,
        },
        "bindings": {
            "part0": {
                "overlay_schema_version": 1,
                "primary": dict(base_binding),
                "unavailable_target_ids": ["p0.missing"],
            },
            "part1": [{
                "overlay_schema_version": 1,
                "primary": dict(base_binding),
                "unavailable_target_ids": ["p1.missing"],
            }],
            "part2": {
                "overlay_schema_version": 1,
                "primary": dict(base_binding),
                "unavailable_target_ids": ["p2.missing"],
            },
        },
        "part0": [
            _p0_row(
                "p0.alpha", "Alpha", (3, 2, 1),
                unclear=(1, 0, 0), invalid=(0, 0, 1),
            ),
            _p0_row("p0.beta", "Beta", (4, 4, 2)),
        ],
        "part1": [
            _p1_row("p1.alpha", "Alpha", 96, 24, invalid=1),
            _p1_row("p1.beta", "Beta", 96, 72, invalid=2),
            _p1_row("p1.zulu", "Zulu", 12, 6, invalid=1),
            _p1_row("p1.amp", "Alpha & Co", 12, 3, invalid=2),
            _p1_row("p1.full", "Full Model", 384, 192, invalid=4),
        ],
        "part2": [
            _p2_row(
                "p2.alpha", "Alpha", aurc=0.5, restraint=0.25,
                aupc=0.75, nondepleted=2, scheduled=70, invalid=3,
            ),
            _p2_row(
                "p2.beta", "Beta", aurc=0.7, restraint=0.5,
                aupc=0.9, nondepleted=6, scheduled=80, invalid=4,
            ),
        ],
        "cross_axis": {
            "status": "not_emitted_fail_closed",
            "rows": [],
            "spearman_pairs": [],
        },
    })


def _write_source(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build(tmp_path: Path, value: dict[str, Any] | None = None):
    source_path = tmp_path / "final_results.json"
    json_path = tmp_path / "paper_headlines.json"
    tex_path = tmp_path / "paper_headlines.tex"
    _write_source(source_path, value or _source())
    artifact = build_paper_headlines(source_path, json_path, tex_path)
    return artifact, json_path, tex_path


def test_builds_exact_within_axis_headlines_and_latex_macros(tmp_path: Path) -> None:
    artifact, json_path, tex_path = _build(tmp_path)

    assert artifact["artifact_type"] == OUTPUT_ARTIFACT_TYPE
    assert artifact["evidence_sha256"] == _self_hash(artifact)
    assert artifact["source"]["path"] == "final_results.json"
    assert artifact["reporting_contract"] == {
        "within_axis_only": True,
        "cross_axis_associations_computed": False,
        "rankings_computed": False,
        "family_effects_computed": False,
        "significance_tests_computed": False,
        "part1_scopes_pooled": False,
        "part0_conditions": (
            "response-language instructions over English source requests; "
            "not translated prompts"
        ),
    }

    p0 = artifact["part0"]
    assert (p0["included_systems"], p0["unavailable_systems"]) == (2, 1)
    assert p0["scheduled_responses"] == 24
    assert p0["outcome_counts"] == {
        "refusal": 16, "compliance": 6, "unclear": 1, "invalid": 1,
    }
    assert p0["system_refusal_percent"] == {
        "median": 66.7, "minimum": 50.0, "maximum": 83.3,
    }
    assert p0["condition_system_median_percent"] == {
        "english": 87.5, "chinese": 75.0, "russian": 37.5,
    }

    p1 = artifact["part1"]
    assert (p1["reported_systems"], p1["unavailable_systems"]) == (5, 4)
    assert p1["operational_unavailable_systems"] == 1
    assert p1["pre_execution_unavailable_systems"] == 3
    assert p1["targeted_systems"] == 9
    assert p1["scheduled_roots"] == 600
    assert p1["scopes"]["n96"] == {
        "systems": 2,
        "scheduled_roots": 192,
        "invalid_outputs": 3,
        "system_choice_percent": {
            "median": 50.0, "minimum": 25.0, "maximum": 75.0,
        },
    }
    assert [
        row["system"] for row in p1["scopes"]["n12"]["values_alphabetical"]
    ] == ["Alpha & Co", "Zulu"]
    assert p1["scopes"]["n384"]["choice_percent"] == 50.0
    assert p1["scopes"]["n384"]["ci95_percent"] == {
        "lower": 40.0, "upper": 60.0,
    }

    p2 = artifact["part2"]
    assert (p2["included_systems"], p2["unavailable_systems"]) == (2, 1)
    assert p2["total_trajectories"] == 16
    assert (p2["scheduled_agent_days"], p2["invalid_agent_days"]) == (150, 7)
    assert p2["system_aurc"] == {
        "median": 0.6, "minimum": 0.5, "maximum": 0.7,
    }
    assert p2["system_restraint_percent"] == {
        "median": 37.5, "minimum": 25.0, "maximum": 50.0,
    }
    assert p2["system_aupc"] == {
        "median": 0.825, "minimum": 0.75, "maximum": 0.9,
    }
    assert p2["nondepleted_trajectories"] == 8

    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored == artifact
    tex = tex_path.read_text(encoding="utf-8")
    assert f"\\newcommand{{\\PaperHeadlinesEvidenceSha}}{{{artifact['evidence_sha256']}}}" in tex
    assert r"\newcommand{\PaperPartZeroRefusalMedianPct}{66.7}" in tex
    assert r"Alpha \& Co: 25.0\%; Zulu: 50.0\%" in tex
    assert r"\newcommand{\PaperPartOneNThreeEightyFourScheduledRoots}{384}" in tex
    assert r"\newcommand{\PaperPartOneUnavailableSystems}{4}" in tex
    assert r"\newcommand{\PaperPartOneTargetedSystems}{9}" in tex
    assert r"\newcommand{\PaperPartTwoTrajectoriesPerSystem}{8}" in tex
    assert "Spearman" not in tex


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("schema_version", 2),
        lambda value: value.__setitem__("artifact_type", "wrong"),
        lambda value: value["bindings"].pop("part2"),
        lambda value: value["privacy_contract"].__setitem__("contains_routes", True),
        lambda value: value["parameters"].__setitem__("part1_scopes_pooled", True),
    ],
)
def test_rejects_invalid_source_contracts(tmp_path: Path, mutation: Any) -> None:
    value = _source()
    value.pop("evidence_sha256")
    mutation(value)
    _seal(value)
    source_path = tmp_path / "final_results.json"
    _write_source(source_path, value)
    with pytest.raises(PaperHeadlineError):
        build_paper_headlines(
            source_path, tmp_path / "headlines.json", tmp_path / "headlines.tex"
        )


def test_rejects_tampered_self_hash(tmp_path: Path) -> None:
    value = _source()
    value["part0"][0]["model"] = "tampered"
    source_path = tmp_path / "final_results.json"
    _write_source(source_path, value)
    with pytest.raises(PaperHeadlineError, match="self-hash"):
        build_paper_headlines(
            source_path, tmp_path / "headlines.json", tmp_path / "headlines.tex"
        )


def test_rejects_forbidden_private_or_text_fields(tmp_path: Path) -> None:
    value = _source()
    value.pop("evidence_sha256")
    value["part0"][0]["prompt_text"] = "must not escape"
    _seal(value)
    source_path = tmp_path / "final_results.json"
    _write_source(source_path, value)
    with pytest.raises(PaperHeadlineError, match="Forbidden"):
        build_paper_headlines(
            source_path, tmp_path / "headlines.json", tmp_path / "headlines.tex"
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["part0"][0]["conditions"][0]["refusal"].update(
                {"estimate": 0.6, "lower": 0.0, "upper": 1.0}
            ),
            "integer count",
        ),
        (
            lambda value: value["part1"][0].__setitem__("format_invalid_count", 4),
            "format counts",
        ),
        (
            lambda value: value.__setitem__(
                "part1", [row for row in value["part1"] if row["root_count"] != 12]
            ),
            "requires n=12",
        ),
        (
            lambda value: value["part2"][0].__setitem__("total_invalid_count", 71),
            "exceed scheduled",
        ),
        (
            lambda value: value["part2"][0][
                "trajectory_level_95_percent_t_intervals"
            ]["aurc"].__setitem__("mean", "NaN"),
            "must be finite",
        ),
        (
            lambda value: value["part2"][0][
                "trajectory_level_95_percent_t_intervals"
            ]["reserve_nondepletion"].__setitem__("successes", 3),
            "nondepletion denominator",
        ),
    ],
)
def test_rejects_inconsistent_or_nonfinite_derivations(
    tmp_path: Path, mutation: Any, match: str
) -> None:
    value = _source()
    value.pop("evidence_sha256")
    mutation(value)
    _seal(value)
    source_path = tmp_path / "final_results.json"
    _write_source(source_path, value)
    with pytest.raises(PaperHeadlineError, match=match):
        build_paper_headlines(
            source_path, tmp_path / "headlines.json", tmp_path / "headlines.tex"
        )


def test_refuses_to_overwrite_either_output(tmp_path: Path) -> None:
    source_path = tmp_path / "final_results.json"
    _write_source(source_path, _source())
    json_path = tmp_path / "headlines.json"
    json_path.write_text("existing\n", encoding="utf-8")
    with pytest.raises(PaperHeadlineError, match="overwrite"):
        build_paper_headlines(source_path, json_path, tmp_path / "headlines.tex")


def test_cli_builds_caller_selected_outputs(tmp_path: Path) -> None:
    source_path = tmp_path / "final_results.json"
    json_path = tmp_path / "selected/headlines.json"
    tex_path = tmp_path / "selected/headlines.tex"
    _write_source(source_path, _source())

    assert main([
        "--input", str(source_path),
        "--output-json", str(json_path),
        "--output-tex", str(tex_path),
    ]) == 0
    assert json_path.is_file()
    assert tex_path.is_file()


def test_accepts_artifact_emitted_by_final_results_builder(tmp_path: Path) -> None:
    final_directory = tmp_path / "final"
    build_final_results(
        part0_manifest=_part0_fixture(tmp_path),
        part1_full_manifests=[
            _part1_fixture(tmp_path, count=384, name="part1-full")
        ],
        part1_n96_manifests=[
            _part1_fixture(tmp_path, count=96, name="part1-n96"),
            _part1_fixture(tmp_path, count=12, name="part1-n12"),
        ],
        part2_manifest=_part2_fixture(tmp_path),
        panel_path=_panel(tmp_path),
        output_dir=final_directory,
    )

    artifact = build_paper_headlines(
        final_directory / "final_results.json",
        tmp_path / "headlines.json",
        tmp_path / "headlines.tex",
    )
    assert artifact["part0"]["included_systems"] == 1
    assert artifact["part1"]["reported_systems"] == 1
    assert artifact["part2"]["included_systems"] == 1
