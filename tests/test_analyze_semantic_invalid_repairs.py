from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.analyze_semantic_invalid_repairs import (
    SemanticRepairAnalysisError,
    _self_hash,
    analyze,
)


def _write_run(
    root: Path,
    *,
    role: bool,
    max_rounds: int = 3,
) -> Path:
    private = root / "private"
    sanitized = root / "sanitized"
    private.mkdir(parents=True)
    sanitized.mkdir()
    target = "anthropic/claude-opus-5"
    rows = []
    for index, valid in enumerate((True, False, True)):
        row = {
            "target_id": target,
            "upstream_provider": "anthropic",
            "model": "claude-opus-5",
            "trial_id": f"trial-{index}",
            "original_format_valid": False,
            "rounds_reserved": 1 if valid else max_rounds,
            "rounds_with_retained_response": 1 if valid else 2,
            "repair_status": "repaired_valid_separate" if valid else "unrepaired_after_bounded_rounds",
            "repaired_format_valid": valid,
            "primary_record_mutated": False,
            "primary_denominator_changed": False,
        }
        if role:
            row.update({"frame_id": "advice", "frame_pooled": False, "model_pooled": False})
        rows.append(row)
    payload = {
        "schema_version": 1,
        "artifact_type": (
            "inference_hub_part1_role_semantic_invalid_repair_outcomes_v1"
            if role else "inference_hub_part1_semantic_invalid_repair_outcomes_v1"
        ),
        "raw_text_included": False,
        "original_records_mutated": False,
        "primary_denominators_changed": False,
        "repaired_estimates_separate_only": True,
        "promotion_permitted": False,
        "rows": rows,
    }
    payload["evidence_sha256"] = _self_hash(payload)
    outcome = sanitized / "repair_outcomes.json"
    outcome.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    import hashlib

    summary = {
        ("source_scheduled_draw_count" if role else "source_scheduled_unit_count"): 6912 if role else 28800,
        ("source_first_response_invalid_count" if role else "source_first_attempt_invalid_count"): 3,
        "repaired_valid_separate_count": 2,
        "unrepaired_after_bounded_rounds_count": 1,
        "primary_denominator": 6912 if role else 28800,
        "primary_denominator_changed": False,
    }
    manifest = {
        "schema_version": 1,
        "artifact_type": (
            "inference_hub_part1_role_semantic_invalid_repair_v1"
            if role else "inference_hub_part1_semantic_invalid_repair_v1"
        ),
        "complete": True,
        "completed_at_utc": "2026-08-03T13:30:00Z",
        "max_semantic_rounds": max_rounds,
        "primary_records_mutated": False,
        "primary_denominators_changed": False,
        "promotion_permitted": False,
        "summary": summary,
        "sanitized_artifact": {
            "path": str(outcome.resolve()),
            "file_sha256": hashlib.sha256(outcome.read_bytes()).hexdigest(),
            "evidence_sha256": payload["evidence_sha256"],
        },
    }
    manifest["evidence_sha256"] = _self_hash(manifest)
    (private / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return root


def test_full_repair_analysis_publishes_separate_reconciled_tables(tmp_path: Path) -> None:
    part1 = _write_run(tmp_path / "part1", role=False)
    role = _write_run(tmp_path / "role", role=True)
    output = tmp_path / "analysis"
    result = analyze(part1=part1, role=role, output_dir=output)
    assert result["primary_records_mutated"] is False
    assert result["primary_denominators_changed"] is False
    assert result["repaired_estimates_separate_only"] is True
    assert result["promotion_permitted"] is False
    assert result["table_outer_spacing_pt"] == 15
    for stem in ("part1_semantic_repair", "part1_role_semantic_repair"):
        rows = [json.loads(line) for line in (output / f"{stem}.jsonl").read_text().splitlines()]
        assert rows == [{
            **rows[0],
            "source_invalid_count": 3,
            "units_retried_count": 3,
            "repair_attempt_count": 2 + 3,
            "repaired_valid_count": 2,
            "still_invalid_count": 1,
            "repair_rate_among_source_invalid": 2 / 3,
            "primary_records_mutated": False,
            "primary_denominator_changed": False,
        }]
        latex = (output / f"{stem}.tex").read_text()
        assert latex.count(r"\par\addvspace{15pt}") == 2
        assert "Each row is one exact target route" in latex
        assert "Source invalid" in latex and "Still invalid" in latex and "Repair rate" in latex
        assert "general safety" in latex
        expected_columns = "llllcccccc" if "role" in stem else "lllcccccc"
        assert f"\\begin{{tabular}}{{{expected_columns}}}" in latex
    assert (output / "analysis_manifest.json").is_file()


def test_eight_round_campaign_is_supported(tmp_path: Path) -> None:
    part1 = _write_run(
        tmp_path / "part1",
        role=False,
        max_rounds=8,
    )
    role = _write_run(
        tmp_path / "role",
        role=True,
        max_rounds=8,
    )
    output = tmp_path / "analysis"
    result = analyze(part1=part1, role=role, output_dir=output)
    assert result["max_semantic_rounds"] == {
        "part1": 8,
        "role": 8,
    }
    assert "within at most 8 rounds" in (
        output / "part1_semantic_repair.tex"
    ).read_text()


def test_round_budget_above_eight_fails(tmp_path: Path) -> None:
    part1 = _write_run(
        tmp_path / "part1",
        role=False,
        max_rounds=9,
    )
    role = _write_run(tmp_path / "role", role=True)
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        SemanticRepairAnalysisError,
        match="round budget",
    ):
        analyze(part1=part1, role=role, output_dir=output)
    assert not output.exists()


def test_tampered_repair_payload_fails_before_output(tmp_path: Path) -> None:
    part1 = _write_run(tmp_path / "part1", role=False)
    role = _write_run(tmp_path / "role", role=True)
    payload = part1 / "sanitized/repair_outcomes.json"
    value = json.loads(payload.read_text())
    value["rows"][0]["primary_record_mutated"] = True
    payload.write_text(json.dumps(value) + "\n", encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(SemanticRepairAnalysisError):
        analyze(part1=part1, role=role, output_dir=output)
    assert not output.exists()
