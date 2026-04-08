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


def test_apply_name_suffix_adds_clean_suffix():
    """Experiment names should get a predictable suffix when requested."""
    module = _load_paper_batch_module()

    assert module.apply_name_suffix("paper-society-prompts", "") == "paper-society-prompts"
    assert module.apply_name_suffix("paper-society-prompts", "replicated-r4") == (
        "paper-society-prompts-replicated-r4"
    )
    assert module.apply_name_suffix("paper-society-prompts", "-replicated-r4-") == (
        "paper-society-prompts-replicated-r4"
    )


def test_build_experiment_plan_applies_multiagent_overrides():
    """Society and reputation configs should honor repetition/concurrency overrides."""
    module = _load_paper_batch_module()
    models = [
        ModelSpec(model="llama3.1-8b", provider="cerebras"),
        ModelSpec(model="deepseek-ai/deepseek-v3.2", provider="nvidia"),
    ]

    plan = module.build_experiment_plan(
        models,
        fast=True,
        multiagent_repetitions=4,
        multiagent_concurrency=3,
        name_suffix="replicated-r4",
    )

    society = next(config for track, config, _ in plan if track == "society")
    reputation = next(config for track, config, _ in plan if track == "reputation")

    assert society.name == "paper-society-prompts-replicated-r4"
    assert reputation.name == "paper-reputation-prompts-replicated-r4"
    assert society.repetitions == 4
    assert reputation.repetitions == 4
    assert society.parameters.concurrency == 3
    assert reputation.parameters.concurrency == 3


def test_full_paper_batch_uses_long_horizon_society_defaults():
    """Default paper batches should use the new long-horizon ecology settings."""
    module = _load_paper_batch_module()
    models = [
        ModelSpec(model="llama3.1-8b", provider="cerebras"),
        ModelSpec(model="deepseek-ai/deepseek-v3.2", provider="nvidia"),
        ModelSpec(model="moonshotai/kimi-k2-instruct-0905", provider="nvidia"),
    ]

    plan = module.build_experiment_plan(
        models,
        fast=False,
        multiagent_repetitions=1,
        multiagent_concurrency=5,
        name_suffix="",
    )

    society = next(config for track, config, _meta in plan if track == "society")
    reputation = next(config for track, config, _meta in plan if track == "reputation")

    assert society.rounds == 120
    assert reputation.rounds == 120
    assert [variant.name for variant in society.prompt_variants] == [
        "task-only",
        "cooperative",
        "competitive",
    ]
    assert sum(agent.count for agent in society.agents) == 24
    assert sum(agent.count for agent in reputation.agents) == 24
    assert society.world.initial_public_food == 120
    assert society.world.initial_public_water == 160
    assert society.world.max_agents == 60
    assert society.society.allow_unmonitored_agents is False
    assert society.society.trade_offer_ttl == 4


def test_build_society_agents_balances_population_to_target_total():
    """Society populations should scale up to the target total size."""
    module = _load_paper_batch_module()
    models = [
        ModelSpec(model="llama3.1-8b", provider="cerebras"),
        ModelSpec(model="deepseek-ai/deepseek-v3.2", provider="nvidia"),
        ModelSpec(model="moonshotai/kimi-k2-instruct-0905", provider="nvidia"),
    ]

    agents = module.build_society_agents(models, target_total_agents=24)

    assert [agent.count for agent in agents] == [8, 8, 8]


def test_paper_batch_configs_use_finite_rate_limit_retries():
    """Paper batch configs should eventually skip a persistently congested model."""
    module = _load_paper_batch_module()
    models = [
        ModelSpec(model="llama3.1-8b", provider="cerebras"),
        ModelSpec(model="deepseek-ai/deepseek-v3.2", provider="nvidia"),
    ]

    plan = module.build_experiment_plan(
        models,
        fast=True,
        multiagent_repetitions=1,
        multiagent_concurrency=2,
        name_suffix="",
    )

    for _track, config, _meta in plan:
        assert config.parameters.max_rate_limit_retries == (
            module.DEFAULT_PAPER_BATCH_MAX_RATE_LIMIT_RETRIES
        )


def test_fast_paper_batch_uses_six_round_part1_protocol():
    """Fast paper batches should still use the six-round audited Part 1 protocol."""
    module = _load_paper_batch_module()
    models = [
        ModelSpec(model="llama3.1-8b", provider="cerebras"),
        ModelSpec(model="deepseek-ai/deepseek-v3.2", provider="nvidia"),
    ]

    plan = module.build_experiment_plan(
        models,
        fast=True,
        multiagent_repetitions=1,
        multiagent_concurrency=2,
        name_suffix="",
    )

    part1_rounds = {
        config.rounds
        for track, config, _meta in plan
        if track in {"baseline", "benchmark", "susceptibility"}
    }

    assert part1_rounds == {6}


def test_benchmark_variant_options_support_fiction_presentations():
    """Benchmark helpers should expose fiction narratives as indirect prompts."""
    module = _load_paper_batch_module()

    pd_options = module.benchmark_variant_options("prisoners_dilemma", "fiction")
    chicken_options = module.benchmark_variant_options("chicken", "fiction")
    stag_options = module.benchmark_variant_options("stag_hunt", "fiction")

    assert pd_options["prompt_overrides"]["description"].endswith("prisoners_dilemma/fiction_description.txt")
    assert chicken_options["action_aliases"] == {"swerve": "veer", "straight": "charge"}
    assert stag_options["action_aliases"] == {"stag": "unite", "hare": "scavenge"}
    assert module.prompting_approach("canonical") == "explicit"
    assert module.prompting_approach("fiction") == "indirect"
