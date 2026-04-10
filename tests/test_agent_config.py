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
