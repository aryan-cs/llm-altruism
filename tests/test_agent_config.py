from agents.agent_config import (
    load_agent_config,
    load_all_model_options,
    load_experiment_model_options,
)


def test_load_agent_config_supports_comment_lines() -> None:
    load_agent_config.cache_clear()
    config = load_agent_config()

    assert "part_0" in config
    assert "part_1" in config
    assert "part_2" in config


def test_load_experiment_model_options_returns_enabled_models() -> None:
    load_agent_config.cache_clear()
    models = load_experiment_model_options("part_0")

    assert "ollama" in models
    assert "gpt-oss-safeguard:20b" in models["ollama"]
    assert "gpt-oss-safeguard:120b" in models["ollama"]


def test_load_all_model_options_combines_experiment_catalogs() -> None:
    load_agent_config.cache_clear()
    models = load_all_model_options()

    assert "openai" in models
    assert "gpt-4.1-mini" in models["openai"]


def test_part_1_model_catalog_matches_the_recorded_part_0_run() -> None:
    load_agent_config.cache_clear()
    part_1_models = load_experiment_model_options("part_1")

    assert part_1_models == {
        "ollama": [
            "gpt-oss:20b",
            "gpt-oss-safeguard:20b",
            "gurubot/gpt-oss-derestricted:20b",
            "llama2",
            "llama2-uncensored",
            "qwen2.5:7b",
            "qwen2.5:7b-instruct",
            "huihui_ai/qwen2.5-abliterate:7b",
            "huihui_ai/qwen2.5-abliterate:7b-instruct",
            "qwen3.5",
            "sorc/qwen3.5-instruct",
            "aratan/qwen3.5-uncensored:9b",
            "sorc/qwen3.5-instruct-uncensored",
        ]
    }


def test_part_2_still_matches_part_0_catalog() -> None:
    load_agent_config.cache_clear()
    part_0_models = load_experiment_model_options("part_0")
    part_2_models = load_experiment_model_options("part_2")

    assert part_2_models == part_0_models
