"""Tests for experiment template and model selection helpers."""

from __future__ import annotations

from src.experiments import (
    apply_runtime_overrides,
    apply_model_selection,
    estimate_trial_conditions,
    list_experiment_templates,
    load_experiment_config,
    parse_model_selectors,
    template_description,
    wrap_picker_description,
)


def test_parse_model_selectors_supports_models_with_colons():
    selectors = parse_model_selectors(
        [
            "openrouter:openai/gpt-oss-20b:free",
            "ollama:llama3.2:3b",
        ]
    )
    assert selectors[0].provider == "openrouter"
    assert selectors[0].model == "openai/gpt-oss-20b:free"
    assert selectors[1].provider == "ollama"
    assert selectors[1].model == "llama3.2:3b"


def test_apply_model_selection_builds_pairwise_part1_matchups():
    config = load_experiment_config("configs/part1/prisoners_dilemma_baseline.yaml")
    selected = parse_model_selectors(
        [
            "cerebras:llama3.1-8b",
            "nvidia:deepseek-ai/deepseek-v3.2",
            "openrouter:openai/gpt-oss-20b:free",
        ]
    )

    updated = apply_model_selection(config, selected)

    assert len(updated.pairings) == 3
    assert updated.pairings[0][0].model == "llama3.1-8b"
    assert updated.pairings[0][1].model == "deepseek-ai/deepseek-v3.2"
    assert updated.pairings[2][0].model == "deepseek-ai/deepseek-v3.2"
    assert updated.pairings[2][1].model == "openai/gpt-oss-20b:free"


def test_apply_model_selection_uses_self_play_for_single_part1_model():
    config = load_experiment_config("configs/part1/prisoners_dilemma_baseline.yaml")
    selected = parse_model_selectors(["cerebras:llama3.1-8b"])

    updated = apply_model_selection(config, selected)

    assert len(updated.pairings) == 1
    left, right = updated.pairings[0]
    assert left.model == "llama3.1-8b"
    assert right.model == "llama3.1-8b"


def test_apply_model_selection_rebuilds_part2_population_evenly():
    config = load_experiment_config("configs/part2/society_baseline.yaml")
    selected = parse_model_selectors(
        [
            "cerebras:llama3.1-8b",
            "nvidia:deepseek-ai/deepseek-v3.2",
            "openrouter:openai/gpt-oss-20b:free",
        ]
    )

    updated = apply_model_selection(config, selected)

    assert [agent.count for agent in updated.agents] == [3, 3, 2]
    assert [agent.provider for agent in updated.agents] == ["cerebras", "nvidia", "openrouter"]


def test_apply_runtime_overrides_updates_core_run_settings():
    config = load_experiment_config("configs/part2/society_baseline.yaml")

    updated = apply_runtime_overrides(
        config,
        rounds=20,
        repetitions=3,
        temperatures=[0.0, 0.7],
        concurrency=9,
    )

    assert updated.rounds == 20
    assert updated.repetitions == 3
    assert updated.parameters.temperature == [0.0, 0.7]
    assert updated.parameters.concurrency == 9


def test_estimate_trial_conditions_matches_part1_grid():
    config = load_experiment_config("configs/part1/prisoners_dilemma_baseline.yaml")
    assert estimate_trial_conditions(config) == (
        len(config.pairings)
        * len(config.prompt_variants)
        * len(config.parameters.temperature)
        * config.repetitions
    )


def test_experiment_templates_are_discoverable():
    templates = list_experiment_templates()
    assert templates


def test_template_description_explains_setup():
    description = template_description("configs/part2/society_baseline.yaml")
    assert "scarce-resource society" in description
    assert "12 rounds" in description
    assert "private messages" in description


def test_wrap_picker_description_inserts_newlines_for_narrow_width():
    wrapped = wrap_picker_description(
        "Coordination experiment in Stag Hunt that compares persona effects.",
        columns=40,
    )
    assert "\n" in wrapped
    assert "Coordination experiment" in wrapped
