from agents.agent_config import (
    load_agent_config,
    load_all_model_options,
    load_experiment_model_options,
    load_model_cohort,
    load_model_registry,
    model_registry_hash,
    resolve_model_registry_entry,
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

    assert part_1_models["ollama"] == [
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
    assert part_1_models["inference_hub"] == [
        target["model"] for target in load_model_cohort("current_sota")["targets"]
    ]


def test_part_2_still_matches_part_0_catalog() -> None:
    load_agent_config.cache_clear()
    part_0_models = load_experiment_model_options("part_0")
    part_2_models = load_experiment_model_options("part_2")

    assert part_2_models == part_0_models


def test_versioned_current_sota_cohort_contains_exact_inference_hub_roster() -> None:
    load_model_registry.cache_clear()
    cohort = load_model_cohort("current_sota")

    assert cohort["registry_version"] == "2026-08-01.1"
    assert cohort["version"] == "2026-08-01.1"
    assert [target["model"] for target in cohort["targets"]] == [
        "gpt-5.6-sol",
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "gemini-3.1-pro-preview",
        "gemini-3.6-flash",
        "google/gemma-4-31b-it",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "deepseek-ai/deepseek-v4-pro",
        "qwen/qwen3-next-80b-a3b-thinking",
        "moonshotai/kimi-k2-thinking",
        "z-ai/glm-5.2",
        "mistralai/mistral-nemotron",
    ]
    assert {target["provider"] for target in cohort["targets"]} == {
        "inference_hub"
    }


def test_historical_cohort_and_route_metadata_are_pinned() -> None:
    load_model_registry.cache_clear()
    cohort = load_model_cohort("historical")

    assert [target["model"] for target in cohort["targets"]] == [
        "gpt-3.5-turbo-0125",
        "gpt-4.1-2025-04-14",
        "gpt-5-2025-08-07",
        "gemini-2.5-pro",
        "google/gemma-3-27b-it",
        "openai/gpt-oss-120b",
    ]
    entry = resolve_model_registry_entry(
        "inference-hub",
        "openai/gpt-oss-120b",
    )
    assert entry is not None
    assert entry["upstream_provider"] == "openai"
    assert entry["route"] == "openai/gpt-oss-120b"
    assert entry["cohorts"] == ["historical"]
    assert entry["endpoint"]["credential_env"] == "NVIDIA_API_KEY"
    assert len(model_registry_hash()) == 64
