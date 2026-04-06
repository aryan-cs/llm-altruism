"""Tests for the paper batch runner helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from src.experiments.config import ExperimentSettings, ModelSpec

ROOT = Path(__file__).resolve().parents[1]


def _load_paper_batch_module():
    """Import the paper batch CLI module for direct helper testing."""
    module_path = ROOT / "scripts" / "run_paper_batch.py"
    spec = importlib.util.spec_from_file_location("run_paper_batch_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_find_existing_result_returns_latest_json(tmp_path: Path):
    """Resume should pick the newest completed JSON for a config name."""
    module = _load_paper_batch_module()
    (tmp_path / "paper-baseline-prisoners_dilemma-20260406T010000Z.json").write_text("{}", encoding="utf-8")
    latest = tmp_path / "paper-baseline-prisoners_dilemma-20260406T020000Z.json"
    latest.write_text("{}", encoding="utf-8")

    result = module.find_existing_result(tmp_path, "paper-baseline-prisoners_dilemma")

    assert result == latest


def test_manifest_entry_from_result_uses_existing_json_payload(tmp_path: Path):
    """Resume entries should preserve the completed result metadata."""
    module = _load_paper_batch_module()
    result_path = tmp_path / "paper-baseline-prisoners_dilemma-20260406T020000Z.json"
    result_path.write_text(
        json.dumps(
            {
                "experiment_id": "paper-baseline-prisoners_dilemma-20260406T020000Z",
                "aggregate_summary": {"cooperation_rate_a": 0.5},
                "skipped_models": [],
                "skipped_trials": [],
            }
        ),
        encoding="utf-8",
    )
    config = ExperimentSettings(
        name="paper-baseline-prisoners_dilemma",
        part=1,
        game="prisoners_dilemma",
        pairings=[(
            ModelSpec(model="llama3.1-8b", provider="cerebras"),
            ModelSpec(model="llama3.1-8b", provider="cerebras"),
        )],
    )

    entry = module.manifest_entry_from_result(
        result_path=result_path,
        config=config,
        track="baseline",
    )

    assert entry["experiment_id"] == "paper-baseline-prisoners_dilemma-20260406T020000Z"
    assert entry["name"] == "paper-baseline-prisoners_dilemma"
    assert entry["track"] == "baseline"
    assert entry["path"] == str(result_path)
    assert entry["aggregate_summary"] == {"cooperation_rate_a": 0.5}
