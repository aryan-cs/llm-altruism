from __future__ import annotations

import json
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from experiments.misc.local_hf_part1_panel import (
    LocalHFPanelError,
    _parse_final_action,
    build_draft_trials,
    run_panel,
)


def _registry(path: Path) -> Path:
    models = []
    for index in range(2):
        models.append(
            {
                "id": f"hf.test-{index}",
                "model_id": f"example/test-{index}",
                "revision": str(index + 1) * 40,
                "cache_repository_dir": f"models--example--test-{index}",
                "parameter_scale": "tiny",
                "verification_status": "smoke_pending",
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registry_version": "test-v1",
                "analysis_role": "exploratory_local_scale_controls",
                "frontier_route_substitution_permitted": False,
                "paper_result_promotion_permitted": False,
                "runtime_contract": {
                    "backend": "huggingface_transformers_offline",
                    "local_files_only": True,
                    "remote_code_permitted": False,
                    "generation_mode": "greedy",
                },
                "models": models,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_draft_schedule_is_384_roots_balanced_over_twelve_cells() -> None:
    trials = build_draft_trials(base_seed=20260802)
    assert len(trials) == 384
    assert len({row.root_id for row in trials}) == 384
    assert set(Counter((row.game, row.domain) for row in trials).values()) == {32}
    assert len({row.trial_id for row in trials}) == 384


@pytest.mark.parametrize(
    ("text", "expected"),
    [("X", "X"), ("reason\nY", "Y"), ("X\nextra", None), ("choice: X", None)],
)
def test_parse_final_action_requires_exact_final_line(text: str, expected: str | None) -> None:
    assert _parse_final_action(text) == expected


def test_panel_parallelizes_models_and_retains_deterministic_manifest(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def runner(*, model, trials, output_path, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.05)
        output_path.write_text(
            "\n".join(json.dumps({"trial_id": row.trial_id}) for row in trials) + "\n",
            encoding="utf-8",
        )
        with lock:
            active -= 1
        return {
            "status": "passed",
            "completed_trials": len(trials),
            "valid_actions": len(trials),
            "invalid_actions": 0,
            "output_file": output_path.name,
            "output_sha256": "a" * 64,
        }

    manifest = run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=tmp_path / "panel",
        limit=12,
        max_workers=2,
        model_runner=runner,
    )

    assert maximum_active == 2
    assert manifest["trial_count_per_model"] == 12
    assert manifest["total_planned_generations"] == 24
    assert manifest["complete"] is True
    assert manifest["confirmatory_or_paper_promotion_permitted"] is False
    assert manifest["generation_contract"]["max_workers"] == 2
    assert json.loads((tmp_path / "panel" / "manifest.json").read_text()) == manifest


def test_panel_refuses_to_overwrite_evidence_directory(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    output = tmp_path / "prior"
    output.mkdir()
    with pytest.raises(LocalHFPanelError, match="already exists"):
        run_panel(
            registry_path=registry,
            cache_root=tmp_path,
            output_dir=output,
            limit=1,
            model_runner=lambda **_kwargs: {},
        )


def test_panel_resume_is_contract_bound_and_retains_failed_attempt(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "registry.json")
    output = tmp_path / "resume-panel"

    def fail(**_kwargs):
        raise RuntimeError("first attempt failed")

    first = run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=output,
        selected_ids=["hf.test-0"],
        limit=3,
        model_runner=fail,
    )
    assert first["complete"] is False

    def pass_run(*, trials, output_path, **_kwargs):
        output_path.write_text(
            "\n".join(json.dumps({"trial_id": row.trial_id}) for row in trials) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "passed",
            "completed_trials": len(trials),
            "valid_actions": len(trials),
            "invalid_actions": 0,
            "output_file": output_path.name,
            "output_sha256": "b" * 64,
        }

    resumed = run_panel(
        registry_path=registry,
        cache_root=tmp_path,
        output_dir=output,
        selected_ids=["hf.test-0"],
        limit=3,
        resume=True,
        model_runner=pass_run,
    )
    assert resumed["complete"] is True
    assert resumed["resume_count"] == 1
    assert resumed["model_attempt_history"]["hf.test-0"][0]["status"] == "failed"

    with pytest.raises(LocalHFPanelError, match="does not match"):
        run_panel(
            registry_path=registry,
            cache_root=tmp_path,
            output_dir=output,
            selected_ids=["hf.test-0"],
            limit=4,
            resume=True,
            model_runner=pass_run,
        )
