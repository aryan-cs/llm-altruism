import pytest

from experiments.wizard import (
    DEFAULT_COMMUNITY_BENEFIT,
    DEFAULT_DEPLETION_UNITS,
    DEFAULT_RESOURCE,
    DEFAULT_SOCIETY_DAYS,
    DEFAULT_SELFISH_GAIN,
    DEFAULT_SOCIETY_SIZE,
    choose_benchmark_models,
    choose_prompt_style,
    choose_provider_and_model,
    choose_society_config,
    parse_alignment_args,
)


def test_choose_provider_and_model_skips_wizard_when_values_provided() -> None:
    provider, model = choose_provider_and_model(
        "Test Experiment",
        provider="openai",
        model="gpt-4.1-mini",
    )

    assert provider == "openai"
    assert model == "gpt-4.1-mini"


def test_choose_provider_and_model_rejects_model_without_provider() -> None:
    with pytest.raises(ValueError, match="model cannot be provided without also providing a provider"):
        choose_provider_and_model("Test Experiment", model="gpt-4.1-mini")


def test_choose_provider_and_model_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        choose_provider_and_model("Test Experiment", provider="unknown")


def test_choose_benchmark_models_accepts_multiple_providers() -> None:
    models = choose_benchmark_models(
        "Part 0",
        benchmarks=[
            "openai:gpt-4.1-mini",
            "anthropic:claude-sonnet-4-5",
            "ollama:gpt-oss:20b",
        ],
    )

    assert models == {
        "openai": ["gpt-4.1-mini"],
        "anthropic": ["claude-sonnet-4-5"],
        "ollama": ["gpt-oss:20b"],
    }


def test_choose_benchmark_models_rejects_mixed_inputs() -> None:
    with pytest.raises(ValueError, match="Use either benchmark entries or provider/model"):
        choose_benchmark_models(
            "Part 0",
            benchmarks=["openai:gpt-4.1-mini"],
            provider="openai",
            model="gpt-4.1-mini",
        )


def test_parse_alignment_args_supports_repeatable_benchmarks() -> None:
    cli_args = parse_alignment_args(
        [
            "--benchmark",
            "openai:gpt-4.1-mini",
            "--benchmark",
            "anthropic:claude-sonnet-4-5",
        ]
    )

    assert cli_args.benchmark == [
        "openai:gpt-4.1-mini",
        "anthropic:claude-sonnet-4-5",
    ]
    assert cli_args.provider is None
    assert cli_args.model is None


def test_choose_prompt_style_skips_wizard_when_value_provided() -> None:
    prompt_style = choose_prompt_style("Test Experiment", prompt_style="direct")

    assert prompt_style == "direct"


def test_choose_prompt_style_rejects_unknown_style() -> None:
    with pytest.raises(ValueError, match="Unsupported prompt style"):
        choose_prompt_style("Test Experiment", prompt_style="unknown")


def test_choose_society_config_skips_wizard_when_values_provided() -> None:
    config = choose_society_config(
        "Test Experiment",
        society_size=25,
        days=12,
        resource="grain",
        selfish_gain=4,
        depletion_units=3,
        community_benefit=7,
    )

    assert config.society_size == 25
    assert config.days == 12
    assert config.resource == "grain"
    assert config.selfish_gain == 4
    assert config.depletion_units == 3
    assert config.community_benefit == 7


def test_choose_society_config_defaults_match_readme_starter_values() -> None:
    config = choose_society_config(
        "Test Experiment",
        society_size=DEFAULT_SOCIETY_SIZE,
        days=DEFAULT_SOCIETY_DAYS,
        resource=DEFAULT_RESOURCE,
        selfish_gain=DEFAULT_SELFISH_GAIN,
        depletion_units=DEFAULT_DEPLETION_UNITS,
        community_benefit=DEFAULT_COMMUNITY_BENEFIT,
    )

    assert config.society_size == 50
    assert config.days == 100
    assert config.resource == "water"
    assert config.selfish_gain == 2
    assert config.depletion_units == 2
    assert config.community_benefit == 5


def test_choose_society_config_allows_open_ended_runs() -> None:
    config = choose_society_config(
        "Test Experiment",
        society_size=DEFAULT_SOCIETY_SIZE,
        days=0,
        resource=DEFAULT_RESOURCE,
        selfish_gain=DEFAULT_SELFISH_GAIN,
        depletion_units=DEFAULT_DEPLETION_UNITS,
        community_benefit=DEFAULT_COMMUNITY_BENEFIT,
    )

    assert config.days == 0
