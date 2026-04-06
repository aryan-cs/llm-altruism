"""End-to-end dry-run tests for config loading and runners."""

from __future__ import annotations

import asyncio
from pathlib import Path

from src.experiments import load_experiment_config
from src.experiments.runner import run_experiment_from_path


def test_can_load_sample_part1_config():
    config = load_experiment_config("configs/part1/prisoners_dilemma_baseline.yaml")
    assert config.part == 1
    assert config.game == "prisoners_dilemma"
    assert config.pairings


def test_part1_dry_run_produces_trials(tmp_path: Path):
    result = asyncio.run(
        run_experiment_from_path(
            "configs/part1/prisoners_dilemma_baseline.yaml",
            dry_run=True,
            results_dir=str(tmp_path),
        )
    )
    assert result["trials"]
    assert result["aggregate_summary"]["total_payoff_a"] >= 0


def test_part2_dry_run_produces_society_metrics(tmp_path: Path):
    result = asyncio.run(
        run_experiment_from_path(
            "configs/part2/society_baseline.yaml",
            dry_run=True,
            results_dir=str(tmp_path),
        )
    )
    assert result["trials"][0]["summary"]["average_gini"] >= 0


def test_part3_dry_run_logs_reputation(tmp_path: Path):
    result = asyncio.run(
        run_experiment_from_path(
            "configs/part3/society_reputation.yaml",
            dry_run=True,
            results_dir=str(tmp_path),
        )
    )
    first_round = result["trials"][0]["rounds"][0]
    assert "ratings" in first_round


def test_real_run_skips_unconfigured_models_instead_of_crashing(monkeypatch, tmp_path: Path):
    for env_var in [
        "CEREBRAS_API_KEY",
        "CEREBRAS_BASE_URL",
        "NVIDIA_API_KEY",
        "NVIDIA_BASE_URL",
        "OLLAMA_BASE_URL",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    result = asyncio.run(
        run_experiment_from_path(
            "configs/part1/prisoners_dilemma_baseline.yaml",
            dry_run=False,
            results_dir=str(tmp_path),
        )
    )
    assert result["trials"] == []
    assert result["skipped_trials"]
    assert result["skipped_models"]
