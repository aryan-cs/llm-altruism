"""Tests for canonical ecology suite orchestration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _load_suite_module():
    module_path = ROOT / "scripts" / "run_canonical_ecology_suite.py"
    spec = importlib.util.spec_from_file_location("run_canonical_ecology_suite_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suite_runs_cover_baseline_reputation_and_event_stress():
    module = _load_suite_module()

    runs = module.suite_runs("results/canonical")

    assert [run.name for run in runs] == ["baseline", "reputation", "event-stress"]
    assert runs[0].config_path == "configs/part2/society_baseline.yaml"
    assert runs[1].config_path == "configs/part3/society_reputation.yaml"
    assert runs[2].config_path == "configs/part2/society_event_stress.yaml"


def test_selected_runs_can_resume_from_reputation():
    module = _load_suite_module()

    runs = module.selected_runs(module.suite_runs("results/canonical"), "reputation")

    assert [run.name for run in runs] == ["reputation", "event-stress"]


def test_build_command_includes_models_results_dir_and_dry_run():
    module = _load_suite_module()
    run = module.SuiteRun(
        name="baseline",
        config_path="configs/part2/society_baseline.yaml",
        results_dir="results/canonical/baseline",
    )

    command = module.build_command(
        run,
        models=["cerebras:llama3.1-8b", "nvidia:deepseek-ai/deepseek-v3.2"],
        dry_run=True,
    )

    assert command[:5] == [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        "configs/part2/society_baseline.yaml",
        "--results-dir",
    ]
    assert "results/canonical/baseline" in command
    assert command.count("--model") == 2
    assert "--dry-run" in command


def test_build_command_can_forward_resume_log():
    module = _load_suite_module()
    run = module.SuiteRun(
        name="baseline",
        config_path="configs/part2/society_baseline.yaml",
        results_dir="results/canonical/baseline",
    )

    command = module.build_command(
        run,
        models=["cerebras:llama3.1-8b"],
        dry_run=False,
        resume_log="results/live_ecology_20260408/society-baseline-20260408T171454Z.jsonl",
    )

    assert "--resume-log" in command
    assert "results/live_ecology_20260408/society-baseline-20260408T171454Z.jsonl" in command
