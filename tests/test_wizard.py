import io

import pytest
from rich.console import Console

from experiments.misc.wizard import (
    _format_prompt_count_preview,
    DEFAULT_COMMUNITY_BENEFIT,
    DEFAULT_DEPLETION_UNITS,
    DEFAULT_RESOURCE,
    DEFAULT_SOCIETY_DAYS,
    DEFAULT_SELFISH_GAIN,
    DEFAULT_SOCIETY_SIZE,
    choose_benchmark_models,
    choose_languages,
    choose_part_1_matrix,
    choose_prompt_count,
    choose_provider_and_model,
    choose_society_config,
    parse_alignment_args,
    parse_game_theory_args,
    parse_society_args,
)


def test_choose_provider_and_model_skips_wizard_when_values_provided() -> None:
    provider, model = choose_provider_and_model(
        "Test Experiment",
        experiment_key="part_1",
        provider="ollama",
        model="gpt-oss:20b",
    )

    assert provider == "ollama"
    assert model == "gpt-oss:20b"


def test_choose_provider_and_model_rejects_model_without_provider() -> None:
    with pytest.raises(ValueError, match="model cannot be provided without also providing a provider"):
        choose_provider_and_model(
            "Test Experiment",
            experiment_key="part_1",
            model="gpt-4.1-mini",
        )


def test_choose_provider_and_model_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported provider"):
        choose_provider_and_model(
            "Test Experiment",
            experiment_key="part_1",
            provider="unknown",
        )


def test_choose_benchmark_models_accepts_multiple_providers() -> None:
    models = choose_benchmark_models(
        "Part 0",
        experiment_key="part_0",
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
            experiment_key="part_0",
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
            "--language",
            "english",
            "--language",
            "spanish",
        ]
    )

    assert cli_args.benchmark == [
        "openai:gpt-4.1-mini",
        "anthropic:claude-sonnet-4-5",
    ]
    assert cli_args.language == ["english", "spanish"]
    assert cli_args.provider is None
    assert cli_args.model is None


def test_parse_alignment_args_supports_resume_flags() -> None:
    cli_args = parse_alignment_args(["--resume"])
    assert cli_args.resume is True

    alias_args = parse_alignment_args(["--pick-up-where-we-left-off"])
    assert alias_args.resume is True


def test_parse_society_args_supports_resume_and_headless_flags() -> None:
    cli_args = parse_society_args(["--resume", "--headless"])

    assert cli_args.resume is True
    assert cli_args.headless is True


def test_parse_alignment_args_supports_judge_after_flag() -> None:
    cli_args = parse_alignment_args(["--judge-after"])

    assert cli_args.judge_after is True


def test_parse_alignment_args_supports_prompt_count() -> None:
    cli_args = parse_alignment_args(["--prompt-count", "25"])

    assert cli_args.prompt_count == 25


def test_choose_languages_skips_wizard_when_values_provided() -> None:
    languages = choose_languages(
        "Part 0",
        available_languages=["english", "spanish", "french"],
        languages=["spanish", "english", "spanish"],
    )

    assert languages == ["spanish", "english"]


def test_choose_languages_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="Unsupported language"):
        choose_languages(
            "Part 0",
            available_languages=["english", "spanish"],
            languages=["english", "klingon"],
        )


def test_choose_prompt_count_skips_wizard_when_value_provided() -> None:
    prompt_count = choose_prompt_count(
        "Part 0",
        total_prompts=500,
        total_languages=10,
        total_models=3,
        prompt_count=25,
    )

    assert prompt_count == 25


def test_choose_prompt_count_rejects_values_above_available_prompts() -> None:
    with pytest.raises(ValueError, match="less than or equal to the number of available prompts"):
        choose_prompt_count(
            "Part 0",
            total_prompts=500,
            total_languages=10,
            total_models=3,
            prompt_count=501,
        )


def test_format_prompt_count_preview_uses_dynamic_prompt_range() -> None:
    preview = _format_prompt_count_preview(
        "25",
        total_prompts=484,
        total_languages=10,
        total_models=3,
    )

    assert preview == "25 prompts × 10 languages × 3 models = 750 total runs"

    empty_preview = _format_prompt_count_preview(
        "",
        total_prompts=484,
        total_languages=10,
        total_models=3,
    )
    assert "1-484" in empty_preview
    assert "10 languages × 3 models" in empty_preview


def test_parse_game_theory_args_supports_repeatable_matrix_filters() -> None:
    cli_args = parse_game_theory_args(
        [
            "--game",
            "prisoners_dilemma",
            "--frame",
            "advice",
            "--domain",
            "sports",
            "--presentation",
            "structured",
            "--limit",
            "4",
            "--headless",
            "--resume",
        ]
    )

    assert cli_args.game == ["prisoners_dilemma"]
    assert cli_args.frame == ["advice"]
    assert cli_args.domain == ["sports"]
    assert cli_args.presentation == ["structured"]
    assert cli_args.limit == 4
    assert cli_args.headless is True
    assert cli_args.resume is True


def test_choose_part_1_matrix_skips_wizard_when_values_provided() -> None:
    selection = choose_part_1_matrix(
        "Test Experiment",
        available_games=["prisoners_dilemma", "temptation_or_commons"],
        available_frames=["self_direct", "advice"],
        available_domains=["crime", "sports"],
        available_presentations=["narrative", "structured"],
        games=["prisoners_dilemma"],
        frames=["advice"],
        domains=["sports"],
        presentations=["structured"],
        limit=3,
    )

    assert selection.games == ["prisoners_dilemma"]
    assert selection.frames == ["advice"]
    assert selection.domains == ["sports"]
    assert selection.presentations == ["structured"]
    assert selection.limit == 3


def test_choose_part_1_matrix_reports_prompt_variants_with_scenario_expansion(
    monkeypatch,
) -> None:
    buffer = io.StringIO()
    test_console = Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=200,
    )
    monkeypatch.setattr("experiments.misc.wizard.console", test_console)

    choose_part_1_matrix(
        "Test Experiment",
        available_games=["prisoners_dilemma", "temptation_or_commons"],
        available_frames=["self_direct", "advice"],
        available_domains=["crime", "sports"],
        available_presentations=["narrative", "structured"],
        games=["prisoners_dilemma"],
        frames=["advice"],
        domains=["sports"],
        presentations=["structured"],
    )

    output = buffer.getvalue()
    assert "Prompt variants: 4" in output


def test_choose_part_1_matrix_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unsupported game"):
        choose_part_1_matrix(
            "Test Experiment",
            available_games=["prisoners_dilemma"],
            available_frames=["self_direct"],
            available_domains=["crime"],
            available_presentations=["narrative"],
            games=["unknown"],
            frames=["self_direct"],
            domains=["crime"],
            presentations=["narrative"],
        )


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
