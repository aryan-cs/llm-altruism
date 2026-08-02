from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from analysis.analyze_local_hf_part1_panel import (
    BOOTSTRAP_REPLICATES,
    LocalHFPart1AnalysisError,
    _self_hash,
    analyze_panel,
    main,
)
from experiments.misc.local_hf_part1_panel import run_panel
from experiments.part1.confirmatory_design import (
    COUNTERBALANCE_BY_ID,
    WELFARE_PRESERVING,
)


_SNAPSHOT = {
    "asset_count": 3,
    "total_size_bytes": 11,
    "snapshot_tree_sha256": "a" * 64,
}
_RUNTIME = {
    "python": "3.test",
    "torch": "test-torch",
    "transformers": "test-transformers",
    "device": "cpu",
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest.pop("evidence_sha256", None)
    manifest["evidence_sha256"] = _canonical_sha256(manifest)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    path.chmod(0o600)


def _registry(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registry_version": "analyzer-test-v1",
                "analysis_role": "exploratory_local_scale_controls",
                "frontier_route_substitution_permitted": False,
                "paper_result_promotion_permitted": False,
                "runtime_contract": {
                    "backend": "huggingface_transformers_offline",
                    "local_files_only": True,
                    "remote_code_permitted": False,
                    "generation_mode": "greedy",
                },
                "models": [
                    {
                        "id": "hf.alpha-135m",
                        "model_id": "example/alpha-135m",
                        "revision": "1" * 40,
                        "cache_repository_dir": "models--example--alpha-135m",
                        "parameter_scale": "135M",
                        "verification_status": "smoke_pending",
                    },
                    {
                        "id": "hf.beta-360m",
                        "model_id": "example/beta-360m",
                        "revision": "2" * 40,
                        "cache_repository_dir": "models--example--beta-360m",
                        "parameter_scale": "360M",
                        "verification_status": "smoke_pending",
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _response_for(index: int) -> str:
    if index % 10 == 0:
        return "not-an-action"
    return "X" if index % 2 == 0 else "brief rationale\nY"


def _write_model_result(
    *, model: Mapping[str, str], trials: Sequence[Any], output_path: Path
) -> dict[str, Any]:
    runtime_sha256 = _canonical_sha256(_RUNTIME)
    rows = []
    valid = 0
    for index, trial in enumerate(trials):
        response = _response_for(index)
        action = None if index % 10 == 0 else ("X" if index % 2 == 0 else "Y")
        valid += action is not None
        rows.append(
            {
                "schema_version": 2,
                "model_id": model["id"],
                "upstream_model_id": model["model_id"],
                "revision": model["revision"],
                "trial_id": trial.trial_id,
                "root_id": trial.root_id,
                "game": trial.game,
                "domain": trial.domain,
                "counterbalance_id": trial.counterbalance_id,
                "prompt_text": trial.prompt_text,
                "prompt_sha256": trial.prompt_hash,
                "response_text": response,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "parsed_action": action,
                "format_valid": action is not None,
                "snapshot_tree_sha256": _SNAPSHOT["snapshot_tree_sha256"],
                "runtime_sha256": runtime_sha256,
                "finished_at_utc": "2026-08-02T12:00:00Z",
            }
        )
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_path.chmod(0o600)
    return {
        "status": "passed",
        "started_at_utc": "2026-08-02T11:00:00Z",
        "finished_at_utc": "2026-08-02T12:00:00Z",
        "completed_trials": len(trials),
        "valid_actions": valid,
        "invalid_actions": len(trials) - valid,
        "output_file": output_path.name,
        "output_sha256": _file_sha256(output_path),
        "resumed_from_trial_count": 0,
        "snapshot": dict(_SNAPSHOT),
        "runtime": dict(_RUNTIME),
    }


def _full_panel(tmp_path: Path) -> tuple[Path, Path, list[Any]]:
    registry = _registry(tmp_path / "registry.json")
    panel_dir = tmp_path / "private" / "local-panel"
    panel_dir.parent.mkdir(mode=0o700)
    if os.name == "posix":
        panel_dir.parent.chmod(0o700)
    captured_trials: list[Any] = []

    def runner(*, model, trials, output_path, **_kwargs):
        if not captured_trials:
            captured_trials.extend(trials)
        return _write_model_result(model=model, trials=trials, output_path=output_path)

    run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=panel_dir,
        max_workers=2,
        model_runner=runner,
    )
    return panel_dir / "manifest.json", registry, captured_trials


def _all_keys(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        result.update(str(key) for key in value)
        for item in value.values():
            result.update(_all_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.update(_all_keys(item))
    return result


def test_full_panel_is_bound_and_aggregated_without_private_text(tmp_path: Path) -> None:
    manifest_path, registry_path, trials = _full_panel(tmp_path)

    artifact = analyze_panel(manifest_path, registry_path)

    assert artifact["evidence_sha256"] == _self_hash(artifact)
    assert artifact["confirmatory_or_paper_promotion_permitted"] is False
    assert artifact["privacy_contract"] == {
        "contains_prompt_text": False,
        "contains_raw_response_or_response_text": False,
        "contains_reasoning": False,
        "contains_only_identifiers_hash_bindings_and_derived_aggregates": True,
    }
    assert not {"prompt_text", "response_text", "parsed_action", "reasoning"} & _all_keys(
        artifact
    )
    assert artifact["parameters"]["bootstrap_replicates"] == BOOTSTRAP_REPLICATES
    assert artifact["coverage"] == {
        "model_count": 2,
        "roots_per_model": 384,
        "retained_model_trial_count": 768,
        "game_domain_strata": 12,
        "roots_per_stratum_per_model": 32,
        "all_selected_models_complete": True,
    }
    assert len(artifact["bindings"]["jsonl_files"]) == 2
    for binding in artifact["bindings"]["jsonl_files"]:
        evidence_path = Path(binding["path"])
        assert binding["record_count"] == 384
        assert binding["file_sha256"] == _file_sha256(evidence_path)
        assert binding["file_sha256"] == binding["manifest_output_sha256"]

    expected_valid = sum(index % 10 != 0 for index in range(384))
    expected_x = sum(index % 10 != 0 and index % 2 == 0 for index in range(384))
    expected_cooperation = 0
    for index, trial in enumerate(trials):
        action = None if index % 10 == 0 else ("X" if index % 2 == 0 else "Y")
        label = COUNTERBALANCE_BY_ID[trial.counterbalance_id].label_for(
            WELFARE_PRESERVING
        )
        expected_cooperation += action == label
    for summary in artifact["models"]:
        counts = summary["counts"]
        assert counts["format_valid_count"] == expected_valid
        assert counts["format_invalid_count"] == 384 - expected_valid
        assert counts["action_x_count"] == expected_x
        assert counts[
            "primary_action_x_rate_format_invalid_retained_as_non_x"
        ] == pytest.approx(expected_x / 384)
        assert counts[
            "primary_cooperation_rate_format_invalid_retained_as_noncooperation"
        ] == pytest.approx(expected_cooperation / 384)
        assert len(summary["per_game_domain"]) == 12
        assert all(row["planned_and_retained_count"] == 32 for row in summary["per_game_domain"])
    assert artifact["overall_equal_model"]["primary_action_x_rate_95_ci"][
        "estimate"
    ] == pytest.approx(expected_x / 384)
    assert artifact["schedule_binding"]["counterbalance_counts"] == {
        "CB_X_FIRST": 96,
        "CB_X_SECOND": 96,
        "CB_Y_FIRST": 96,
        "CB_Y_SECOND": 96,
    }


def test_rejects_jsonl_tampering_even_if_outer_hashes_are_resealed(tmp_path: Path) -> None:
    manifest_path, registry_path, _ = _full_panel(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_path = manifest_path.parent / manifest["models"]["hf.alpha-135m"]["output_file"]
    rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    rows[0]["response_text"] = "X"
    evidence_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    evidence_path.chmod(0o600)
    manifest["models"]["hf.alpha-135m"]["output_sha256"] = _file_sha256(evidence_path)
    _seal_manifest(manifest_path, manifest)

    with pytest.raises(LocalHFPart1AnalysisError, match="frozen schedule/model/content"):
        analyze_panel(manifest_path, registry_path)


def test_rejects_manifest_self_hash_and_registry_changes(tmp_path: Path) -> None:
    manifest_path, registry_path, _ = _full_panel(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.chmod(0o600)
    with pytest.raises(LocalHFPart1AnalysisError, match="self-hash"):
        analyze_panel(manifest_path, registry_path)

    manifest_path, registry_path, _ = _full_panel(tmp_path / "other")
    registry_path.write_text(registry_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(LocalHFPart1AnalysisError, match="Registry file/version binding"):
        analyze_panel(manifest_path, registry_path)


def test_rejects_partial_schedule_and_nonstandard_bootstrap(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    panel_dir = tmp_path / "private" / "partial-panel"
    panel_dir.parent.mkdir(mode=0o700)
    panel_dir.parent.chmod(0o700)
    run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=panel_dir,
        selected_ids=["hf.alpha-135m"],
        limit=12,
        model_runner=lambda **kwargs: _write_model_result(
            model=kwargs["model"],
            trials=kwargs["trials"],
            output_path=kwargs["output_path"],
        ),
    )
    with pytest.raises(LocalHFPart1AnalysisError, match="schedule hash/root binding"):
        analyze_panel(panel_dir / "manifest.json", registry)
    with pytest.raises(LocalHFPart1AnalysisError, match="exactly 5,000"):
        analyze_panel(
            panel_dir / "manifest.json", registry, bootstrap_replicates=100
        )


def test_cli_writes_private_aggregate_only_artifact(tmp_path: Path) -> None:
    manifest_path, registry_path, _ = _full_panel(tmp_path)
    output = tmp_path / "aggregate.json"

    assert (
        main(
            [
                "--manifest",
                str(manifest_path),
                "--registry",
                str(registry_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["evidence_sha256"] == _self_hash(written)
    assert "prompt_text" not in _all_keys(written)
    assert "response_text" not in _all_keys(written)
    if os.name == "posix":
        assert stat.S_IMODE(output.stat().st_mode) == 0o600
