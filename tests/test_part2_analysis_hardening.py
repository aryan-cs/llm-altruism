import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.part2_dynamics import (
    Part2StructuralCell,
    collapse_deaths,
    deterministic_policy_baseline,
)
from analysis.summarize_results import summarize_cross_part, summarize_part2
from analysis.validation import validate_part2_file
from experiments.part2.part_2 import RESULT_HEADERS
from experiments.part2 import part_2


def test_part2_analysis_modules_import_in_both_orders() -> None:
    for source in (
        "import analysis.part2_confirmatory; import analysis.summarize_results",
        "import analysis.summarize_results; import analysis.part2_confirmatory",
    ):
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_trajectory(
    path: Path,
    cell: Part2StructuralCell,
    restraint_counts: list[int],
    *,
    record_death_rate: bool = True,
    invalid_counts: list[int] | None = None,
    environment_seed: int | None = None,
    generation_seed: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    population = cell.society_size
    resource_units = cell.resource_capacity
    society_config = {
        "society_size": cell.society_size,
        "days": cell.horizon_days,
        "resource": cell.resource,
        "selfish_gain": cell.selfish_gain,
        "depletion_units": cell.depletion_units,
        "community_benefit": cell.community_benefit,
    }
    dynamics = part_2._part_2_dynamics_metadata()
    structural_cell_id = part_2._structural_cell_id(
        config=part_2.SocietyConfig(**society_config),
        resource_capacity=cell.resource_capacity,
        collapse_death_rate=cell.collapse_death_rate,
    )
    seed_material = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16)
    environment_seed = seed_material if environment_seed is None else environment_seed
    generation_seed = seed_material + 1 if generation_seed is None else generation_seed
    trajectory_id = part_2._trajectory_id(
        structural_cell_id=structural_cell_id,
        environment_seed=environment_seed,
    )
    run_id = f"p2run_test_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    living_slots = list(range(1, population + 1))
    rows: list[dict[str, object]] = []
    for day, restraint_count in enumerate(restraint_counts, start=1):
        if population <= 0:
            break
        if not 0 <= restraint_count <= population:
            raise ValueError("restraint count is outside the living population")
        invalid_count = invalid_counts[day - 1] if invalid_counts is not None else 0
        if not 0 <= invalid_count <= population - restraint_count:
            raise ValueError("invalid count is outside the remaining population")
        overuse_count = population - restraint_count - invalid_count
        resource_units = max(0, resource_units - overuse_count * cell.depletion_units)
        deaths = collapse_deaths(population, resource_units, cell.collapse_death_rate)
        population_end = population - deaths
        death_slots, attrition_seed = part_2._select_attrition_slots(
            living_slots=living_slots,
            deaths=deaths,
            environment_seed=environment_seed,
            trajectory_id=trajectory_id,
            day=day,
        )
        death_rank = {slot: rank for rank, slot in enumerate(death_slots, start=1)}
        for agent_index, anonymous_slot in enumerate(living_slots, start=1):
            invalid = restraint_count < agent_index <= restraint_count + invalid_count
            rows.append(
                {
                    "run_id": run_id,
                    "trajectory_id": trajectory_id,
                    "structural_cell_id": structural_cell_id,
                    "provider": cell.provider,
                    "model": cell.model,
                    "day": day,
                    "agent": f"society_{anonymous_slot}",
                    "anonymous_agent_slot": anonymous_slot,
                    "action": (
                        "RESTRAIN"
                        if agent_index <= restraint_count
                        else (
                            "INVALID"
                            if agent_index <= restraint_count + invalid_count
                            else "OVERUSE"
                        )
                    ),
                    "reasoning": "",
                    "invalid_reason": (
                        "ExtractorValidationError: malformed final answer"
                        if invalid
                        else ""
                    ),
                    "attempt_outcome": (
                        "invalid_response"
                        if invalid
                        else "success"
                    ),
                    "attempt_count": 1,
                    "environment_seed": environment_seed,
                    "generation_seed": generation_seed,
                    "call_seed": generation_seed + day + anonymous_slot,
                    "requested_model": cell.model,
                    "returned_model": cell.model,
                    "request_id": f"request-{day}-{anonymous_slot}",
                    "finish_reason": "stop",
                    "usage_json": "{}",
                    "raw_response_sha256": "0" * 64,
                    "population_start": population,
                    "population_end": population_end,
                    "restrain_count": restraint_count,
                    "overuse_count": overuse_count,
                    "invalid_count": invalid_count,
                    "resource_units_remaining": resource_units,
                    "resource_capacity": cell.resource_capacity,
                    "deaths": deaths,
                    "died_today": str(anonymous_slot in death_rank).lower(),
                    "attrition_rank": death_rank.get(anonymous_slot, ""),
                    "attrition_seed": attrition_seed if attrition_seed is not None else "",
                    "death_selected_slots_json": json.dumps(
                        death_slots, separators=(",", ":")
                    ),
                    "resource": cell.resource,
                    "selfish_gain": cell.selfish_gain,
                    "depletion_units": cell.depletion_units,
                    "community_benefit": cell.community_benefit,
                }
            )
        living_slots = [slot for slot in living_slots if slot not in set(death_slots)]
        population = population_end

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    parameters: dict[str, object] = {
        "part_2_schema_version": part_2.PART_2_SCHEMA_VERSION,
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "structural_cell_id": structural_cell_id,
        "society_config": society_config,
        "resource_capacity": cell.resource_capacity,
        "environment_seed": environment_seed,
        "generation_seed": generation_seed,
        "dynamics": dynamics,
    }
    if record_death_rate:
        parameters["collapse_death_rate"] = cell.collapse_death_rate
    path.with_name(f"{path.stem}_meta.json").write_text(
        json.dumps(
            {
                "provider": cell.provider,
                "model": cell.model,
                "parameters": parameters,
            }
        ),
        encoding="utf-8",
    )


def _cell(*, death_rate: float = 1.0, capacity: int = 2) -> Part2StructuralCell:
    return Part2StructuralCell(
        provider="provider",
        model="model",
        society_size=2,
        horizon_days=4,
        resource="water",
        resource_capacity=capacity,
        selfish_gain=2,
        depletion_units=1,
        community_benefit=5,
        collapse_death_rate=death_rate,
    )


def test_mixed_structural_cells_are_separate_and_uncertainty_is_run_level(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    part_dir = raw_dir / "part_2"
    baseline = _cell(death_rate=1.0)
    sensitivity = _cell(death_rate=0.5)
    _write_trajectory(
        part_dir / "part2__provider__model__n2__d4__water__s1__20260101_000000.csv",
        baseline,
        [1, 1],
    )
    _write_trajectory(
        part_dir / "part2__provider__model__n2__d4__water__s2__20260101_000001.csv",
        baseline,
        [2, 2, 2, 2],
    )
    _write_trajectory(
        part_dir / "part2__provider__model__n2__d4__water__dr0p5__s3__20260101_000002.csv",
        sensitivity,
        [2, 2, 2, 2],
    )

    output_dir = tmp_path / "tables"
    summarize_part2(raw_dir, output_dir)
    summaries = _read_rows(output_dir / "part2_model_summary.csv")
    run_rows = _read_rows(output_dir / "part2_run_summary.csv")

    assert len(run_rows) == 3
    assert len(summaries) == 2
    assert len({row["structural_cell_key"] for row in summaries}) == 2
    assert {float(row["collapse_death_rate"]) for row in summaries} == {0.5, 1.0}
    baseline_summary = next(row for row in summaries if row["collapse_death_rate"] == "1.0")
    sensitivity_summary = next(row for row in summaries if row["collapse_death_rate"] == "0.5")

    assert baseline_summary["run_count"] == "2"
    assert baseline_summary["uncertainty_unit"] == "run"
    assert baseline_summary["interval_method"] == "between_run_t_95"
    assert baseline_summary["restraint_ci_low"]
    assert baseline_summary["restraint_ci_high"]
    assert baseline_summary["bca_interval_method"] == "between_run_bca_bootstrap_95"
    assert baseline_summary["bca_bootstrap_replicates"] == "2000"
    for metric in (
        "restraint_rate",
        "normalized_aurc",
        "normalized_aupc",
        "restricted_mean_time_to_depletion",
        "survived_through_horizon",
    ):
        assert baseline_summary[f"{metric}_t_ci_low"]
        assert baseline_summary[f"{metric}_t_ci_high"]
        assert baseline_summary[f"{metric}_bca_ci_low"]
        assert baseline_summary[f"{metric}_bca_ci_high"]
    assert sensitivity_summary["run_count"] == "1"
    assert sensitivity_summary["interval_method"] == "not_estimable_single_run"
    assert sensitivity_summary["bca_interval_method"] == "not_estimable_single_run"
    assert sensitivity_summary["restraint_ci_low"] == ""
    assert sensitivity_summary["restraint_ci_high"] == ""
    assert sensitivity_summary["normalized_aurc_bca_ci_low"] == ""
    assert not any("wilson" in field.lower() for field in summaries[0])


def test_invalid_decisions_are_retained_in_primary_denominator(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    path = raw_dir / "part_2" / "part2__provider__model__invalid.csv"
    _write_trajectory(
        path,
        _cell(death_rate=1.0, capacity=8),
        [1],
        invalid_counts=[1],
    )

    report = validate_part2_file(path)
    assert report.status == "pass"
    assert report.metrics["invalid_decisions"] == 1

    output = tmp_path / "analysis"
    summarize_part2(raw_dir, output)
    row = _read_rows(output / "part2_run_summary.csv")[0]
    assert row["invalid_decisions"] == "1"
    assert float(row["restraint_rate"]) == 0.5
    assert float(row["restraint_rate_scorable"]) == 1.0
    assert float(row["invalid_rate"]) == 0.5


def test_cross_part_analysis_rejects_ambiguous_multiple_part2_cells(tmp_path: Path) -> None:
    _write_rows(
        tmp_path / "part0_model_summary.csv",
        [{"provider": "provider", "model": "model", "safety_refusal_rate": 0.5}],
    )
    _write_rows(
        tmp_path / "part1_model_summary.csv",
        [
            {
                "provider": "provider",
                "model": "model",
                "cooperation_rate": 0.5,
                "all_frames_cooperation_rate": 0.5,
            }
        ],
    )
    _write_rows(
        tmp_path / "part2_model_summary.csv",
        [
            {"provider": "provider", "model": "model", "structural_cell_key": "cell-a"},
            {"provider": "provider", "model": "model", "structural_cell_key": "cell-b"},
        ],
    )

    with pytest.raises(ValueError, match="select one explicit structural cell"):
        summarize_cross_part(tmp_path)


def test_undepleted_trajectory_is_right_censored_in_depletion_estimands(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    part_dir = raw_dir / "part_2"
    cell = _cell(death_rate=1.0)
    _write_trajectory(
        part_dir / "part2__provider__model__n2__d4__water__s1__20260101_000000.csv",
        cell,
        [1, 1],
    )
    _write_trajectory(
        part_dir / "part2__provider__model__n2__d4__water__s2__20260101_000001.csv",
        cell,
        [2, 2, 2, 2],
    )

    output_dir = tmp_path / "tables"
    summarize_part2(raw_dir, output_dir)
    [summary] = _read_rows(output_dir / "part2_model_summary.csv")
    runs = _read_rows(output_dir / "part2_run_summary.csv")

    assert summary["runs_depleted"] == "1"
    assert summary["runs_censored"] == "1"
    assert float(summary["restricted_mean_time_to_depletion"]) == 3.0
    assert float(summary["survival_through_horizon"]) == 0.5
    assert float(summary["mean_depletion_day_among_depleted"]) == 2.0
    assert {row["depletion_observed"] for row in runs} == {"0", "1"}
    censored = next(row for row in runs if row["depletion_observed"] == "0")
    assert censored["censoring_day"] == "4"
    assert censored["survived_through_horizon"] == "1"


def test_validator_uses_nondefault_recorded_death_rate(tmp_path: Path) -> None:
    cell = Part2StructuralCell(
        provider="provider",
        model="model",
        society_size=10,
        horizon_days=1,
        resource="water",
        resource_capacity=1,
        selfish_gain=2,
        depletion_units=1,
        community_benefit=5,
        collapse_death_rate=0.3,
    )
    path = tmp_path / "part2__provider__model__n10__d1__water__dr0p3__20260101_000000.csv"
    _write_trajectory(path, cell, [0])

    report = validate_part2_file(path)

    assert report.status == "pass"
    assert report.metrics["recorded_collapse_death_rate"] == 0.3
    assert report.metrics["transition_errors"] == 0


def test_validator_fails_closed_when_death_rate_metadata_is_missing(tmp_path: Path) -> None:
    cell = _cell(death_rate=0.5)
    path = tmp_path / "part2__provider__model__n2__d4__water__20260101_000000.csv"
    _write_trajectory(path, cell, [2, 2, 2, 2], record_death_rate=False)

    report = validate_part2_file(path)

    assert report.status == "fail"
    assert report.metrics["structural_metadata_valid"] is False
    assert any("collapse_death_rate" in error for error in report.errors)


def test_exact_always_policy_baselines_match_recorded_dynamics() -> None:
    cell = _cell(death_rate=0.5, capacity=4)

    restrain = deterministic_policy_baseline(cell, "always-restrain")
    overuse = deterministic_policy_baseline(cell, "always-overuse")

    assert restrain.depletion_day is None
    assert restrain.final_population == 2
    assert restrain.final_resource_units == 4
    assert restrain.normalized_aurc == 1.0
    assert restrain.normalized_aupc == 1.0
    assert overuse.depletion_day == 2
    assert overuse.days_completed == 3
    assert overuse.final_population == 0
    assert overuse.final_resource_units == 0
    assert overuse.total_deaths == 2
    assert overuse.normalized_aurc == 0.125
    assert overuse.normalized_aupc == 0.375


def test_strict_validator_rejects_tampered_attrition_and_summarizer_fails_closed(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    path = raw_dir / "part_2" / "part2__provider__model__attrition.csv"
    _write_trajectory(path, _cell(death_rate=1.0), [1, 1])
    rows = _read_rows(path)
    collapsed = next(row for row in rows if row["deaths"] != "0")
    collapsed["died_today"] = (
        "false" if collapsed["died_today"] == "true" else "true"
    )
    _write_rows(path, rows)

    report = validate_part2_file(path)

    assert report.status == "fail"
    assert report.metrics["attrition_errors"] > 0
    with pytest.raises(ValueError, match="Refusing to summarize invalid Part 2 artifact"):
        summarize_part2(raw_dir, tmp_path / "tables")


def test_strict_validator_rejects_incomplete_day_as_fatal(tmp_path: Path) -> None:
    path = tmp_path / "part2__provider__model__incomplete.csv"
    _write_trajectory(path, _cell(death_rate=1.0, capacity=8), [1])
    rows = _read_rows(path)
    _write_rows(path, rows[:-1])

    report = validate_part2_file(path)

    assert report.status == "fail"
    assert report.metrics["incomplete_days"] == 1
    assert any("fewer rows" in error for error in report.errors)


def test_structural_cell_id_binds_full_recorded_dynamics(tmp_path: Path) -> None:
    path = tmp_path / "part2__provider__model__dynamics.csv"
    _write_trajectory(path, _cell(), [2, 2, 2, 2])
    metadata_path = path.with_name(f"{path.stem}_meta.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["parameters"]["dynamics"]["attrition_policy"]["id"] = "tampered"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_part2_file(path)

    assert report.status == "fail"
    assert any("complete dynamics contract" in error for error in report.errors)


def test_duplicate_environment_trajectory_is_not_an_independent_replicate(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    part_dir = raw_dir / "part_2"
    cell = _cell(capacity=8)
    _write_trajectory(part_dir / "run-one.csv", cell, [2, 2, 2, 2], environment_seed=7)
    _write_trajectory(part_dir / "run-two.csv", cell, [2, 2, 2, 2], environment_seed=7)

    with pytest.raises(ValueError, match="Duplicate Part 2 trajectory_id"):
        summarize_part2(raw_dir, tmp_path / "tables")
